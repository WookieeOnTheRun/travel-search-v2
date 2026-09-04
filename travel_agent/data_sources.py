from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

import httpx

from config import settings
from .destination_insights import check_state_department_advisory
from .http_utils import describe_error, duffel_headers, get_with_retry, post_with_retry, rapidapi_headers
from .location_resolution import (
    resolve_airport_codes,
    resolve_cruise_destination_id,
    resolve_origin_airport_codes,
)
from .schemas import DestinationCandidate, GroundingPacket, LocationCodes

logger = logging.getLogger(__name__)


# Fetches a 5-day weather forecast summary for a specific coordinate pair
async def _fetch_weather_summary(client: httpx.AsyncClient, latitude: float, longitude: float) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 5,
        "timezone": "auto",
    }

    response = await get_with_retry(client, url, params=params)
    data = response.json()

    daily = data.get("daily", {})
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precipitation = daily.get("precipitation_probability_max", [])

    if not max_temps:
        return "Weather data unavailable for selected destination."

    avg_high = round(sum(max_temps) / len(max_temps), 1)
    avg_low = round(sum(min_temps) / len(min_temps), 1)
    rain_risk = round(sum(precipitation) / len(precipitation), 1) if precipitation else 0

    return (
        f"5-day outlook: average highs around {avg_high}°C, lows around {avg_low}°C, "
        f"and average precipitation probability near {rain_risk}%."
    )


# Retrieves the live U.S. Department of State travel advisory for the destination's country
# (see destination_insights.check_state_department_advisory for source details/verification).
async def _fetch_safety_summary(client: httpx.AsyncClient, destination: DestinationCandidate) -> tuple[str, str, str]:
    check = await check_state_department_advisory(client, destination)

    if check.error:
        return (
            f"U.S. Department of State advisory feed unavailable ({check.error}). "
            "Manually verify current travel advisories at travel.state.gov.",
            "Unknown",
            check.source_url,
        )
    if not check.found:
        return (
            "No matching U.S. Department of State advisory entry was found for this destination's "
            "country. Manually verify current guidance at travel.state.gov before booking.",
            "Unknown",
            check.source_url,
        )

    level_label = f"Level {check.level}" if check.level is not None else "Unknown level"
    recency_note = (
        f"last updated {check.updated.isoformat()}" if check.updated else "update date unavailable"
    )
    warning_note = (
        " ACTIVE WARNING: this is a Level 3/4 advisory updated within the last six months."
        if check.active_warning
        else ""
    )
    summary = (
        f"U.S. Department of State advisory: {check.title or level_label} ({recency_note}).{warning_note} "
        "Always cross-check with the latest travel.state.gov guidance before final bookings."
    )
    return summary, level_label, check.link or check.source_url


# Gathers generic snippets from a list of configured external travel API endpoints (optional extension point)
async def _fetch_external_api_snippets(
    client: httpx.AsyncClient, request_text: str, destination: str
) -> list[str]:
    if not settings.travel_api_endpoints:
        return []

    headers = {"Authorization": f"Bearer {settings.travel_api_key}"} if settings.travel_api_key else {}

    async def _fetch_one(endpoint: str) -> str:
        try:
            response = await get_with_retry(
                client,
                endpoint,
                params={
                    "query": request_text,
                    "destination": destination,
                    "currency": settings.default_currency,
                    "locale": settings.default_locale,
                },
                headers=headers,
            )
            payload = response.json()
            return f"{endpoint}: {str(payload)[:900]}"
        except Exception as ex:
            logger.warning("External API endpoint %s failed: %s", endpoint, describe_error(ex))
            return f"{endpoint}: API unavailable ({describe_error(ex)})"

    return list(await asyncio.gather(*(_fetch_one(endpoint) for endpoint in settings.travel_api_endpoints)))


async def _skip(message: str) -> str:
    return message


def _build_local_transport_notes(destination_name: str) -> str:
    return (
        f"For {destination_name}, compare premium rental cars for flexibility, airport taxis/Uber for short stays, "
        "and metro/rail passes for city-center access. For late-night arrivals, prefer pre-booked transfers "
        "from reputable operators with published safety standards."
    )


# ---------------------------------------------------------------------------
# Provider-specific id/code resolution for the user's confirmed destination.
# Flight and cruise lookups only run when they're actually relevant to this
# request (an origin was given / cruise interest was detected) so a request
# with no flight or cruise angle doesn't pay for those calls.
# ---------------------------------------------------------------------------


async def resolve_location_codes(
    client: httpx.AsyncClient,
    destination: DestinationCandidate,
    *,
    origin_city: str | None,
    wants_flights: bool,
    wants_cruise: bool,
) -> LocationCodes:
    # No hotel/accommodation location id needs resolving here: Duffel Stays searches by
    # geographic coordinates directly (see _fetch_duffel_stays_results below), which the
    # destination candidate already carries from geocoding.
    dest_airport_task = (
        asyncio.create_task(resolve_airport_codes(client, destination.name, country_hint=destination.country_code))
        if wants_flights
        else None
    )
    origin_airport_task = (
        asyncio.create_task(resolve_origin_airport_codes(client, origin_city))
        if (wants_flights and origin_city)
        else None
    )
    cruise_task = asyncio.create_task(resolve_cruise_destination_id(client, destination)) if wants_cruise else None

    pending = [t for t in (dest_airport_task, origin_airport_task, cruise_task) if t]
    await asyncio.gather(*pending)

    return LocationCodes(
        destination_airport_codes=dest_airport_task.result() if dest_airport_task else [],
        origin_airport_codes=origin_airport_task.result() if origin_airport_task else [],
        cruise_destination_id=cruise_task.result() if cruise_task else None,
    )


# ---------------------------------------------------------------------------
# Search calls -- each one uses the real, verified parameter contract for its
# provider and degrades to a clearly-labeled "unavailable" string on failure
# rather than raising, so one provider going down doesn't abort the itinerary.
# ---------------------------------------------------------------------------


# Kilometre search radius passed to Duffel Stays' location-based search (POST
# /stays/search, `data.location.radius` in km per
# https://duffel.com/docs/guides/getting-started-with-stays). This is a tuning parameter
# (how wide a net to cast around the destination's geocoded center), not a verified fact,
# and matches the value used in Duffel's own official example.
_STAYS_SEARCH_RADIUS_KM = 2


async def _fetch_duffel_stays_results(
    client: httpx.AsyncClient,
    destination: DestinationCandidate,
    check_in: date,
    check_out: date,
    traveler_count: int,
) -> str:
    if not settings.duffel_api_key:
        return "Duffel Stays: DUFFEL_API_KEY missing."

    url = f"{settings.duffel_api_base_url}/stays/search"
    payload = {
        "data": {
            "location": {
                "radius": _STAYS_SEARCH_RADIUS_KM,
                "geographic_coordinates": {
                    "latitude": destination.latitude,
                    "longitude": destination.longitude,
                },
            },
            "check_in_date": check_in.isoformat(),
            "check_out_date": check_out.isoformat(),
            "rooms": 1,
            "guests": [{"type": "adult"} for _ in range(traveler_count)],
        }
    }
    try:
        response = await post_with_retry(client, url, json=payload, headers=duffel_headers())
        return f"Duffel Stays: {str(response.json())[:2000]}"
    except Exception as ex:
        logger.warning("Duffel Stays search failed: %s", describe_error(ex))
        return f"Duffel Stays: unavailable ({describe_error(ex)})"


async def _fetch_duffel_flight_results(
    client: httpx.AsyncClient,
    origin_codes: list[str],
    destination_codes: list[str],
    depart_date: date,
    return_date: date | None,
    traveler_count: int,
) -> str:
    if not origin_codes or not destination_codes:
        return "Duffel Flights: unavailable (could not resolve origin/destination airport codes)."
    if not settings.duffel_api_key:
        return "Duffel Flights: DUFFEL_API_KEY missing."

    url = f"{settings.duffel_api_base_url}/air/offer_requests"
    slices: list[dict[str, Any]] = [
        {
            "origin": origin_codes[0],
            "destination": destination_codes[0],
            "departure_date": depart_date.isoformat(),
        }
    ]
    if return_date:
        slices.append(
            {
                "origin": destination_codes[0],
                "destination": origin_codes[0],
                "departure_date": return_date.isoformat(),
            }
        )

    payload = {
        "data": {
            "slices": slices,
            "passengers": [{"type": "adult"} for _ in range(traveler_count)],
            "cabin_class": "economy",
        }
    }
    try:
        # return_offers defaults to true, so the offer_request response already carries
        # the matching offers -- no separate GET /air/offers call is needed.
        response = await post_with_retry(client, url, json=payload, headers=duffel_headers())
        return f"Duffel Flights ({origin_codes[0]} -> {destination_codes[0]}): {str(response.json())[:2000]}"
    except Exception as ex:
        logger.warning("Duffel Flights search failed: %s", describe_error(ex))
        return f"Duffel Flights: unavailable ({describe_error(ex)})"


async def _fetch_visa_requirements(
    client: httpx.AsyncClient, citizenship_country_code: str, destination_country_code: str
) -> str:
    if not destination_country_code:
        return "Visa Requirements: unavailable (destination country code not resolved)."
    if not settings.rapidapi_key:
        return "Visa Requirements: RAPIDAPI_KEY missing."

    host = settings.rapidapi_visa_host
    url = f"https://{host}/v2/visa/check"
    try:
        response = await post_with_retry(
            client,
            url,
            json={"passport": citizenship_country_code, "destination": destination_country_code},
            headers={**rapidapi_headers(host), "Content-Type": "application/json"},
        )
        return f"Visa Requirements: {str(response.json())[:1200]}"
    except Exception as ex:
        logger.warning("Visa requirements lookup failed: %s", describe_error(ex))
        return f"Visa Requirements: unavailable ({describe_error(ex)})"


async def _fetch_cruise_search_results(
    client: httpx.AsyncClient, cruise_destination_id: str | None, destination_name: str
) -> str:
    if not cruise_destination_id:
        return (
            "Tripadvisor Cruises: unavailable. This endpoint has been unreliable on the "
            "current RapidAPI subscription -- check RapidAPI dashboard status if this persists."
        )
    if not settings.rapidapi_key:
        return "Tripadvisor Cruises: RAPIDAPI_KEY missing."

    host = settings.rapidapi_tripadvisor_host
    url = f"https://{host}/api/v1/cruises/searchCruises"
    params = {"destinationId": cruise_destination_id, "currencyCode": settings.default_currency}
    try:
        response = await get_with_retry(client, url, params=params, headers=rapidapi_headers(host))
        return f"Tripadvisor Cruises ({destination_name}): {str(response.json())[:2000]}"
    except Exception as ex:
        logger.warning("Cruise search for %s failed: %s", destination_name, describe_error(ex))
        return f"Tripadvisor Cruises: unavailable ({describe_error(ex)})"


async def gather_grounding_packet(
    destination: DestinationCandidate,
    *,
    request_text: str,
    origin_city: str | None,
    traveler_count: int,
    check_in: date,
    check_out: date,
    wants_flights: bool,
    wants_cruise: bool,
    citizenship_country_code: str | None = None,
) -> GroundingPacket:
    citizenship = citizenship_country_code or settings.default_citizenship_country_code

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        weather_task = asyncio.create_task(_fetch_weather_summary(client, destination.latitude, destination.longitude))
        safety_task = (
            asyncio.create_task(_fetch_safety_summary(client, destination))
            if destination.country_code
            else None
        )
        codes_task = asyncio.create_task(
            resolve_location_codes(
                client,
                destination,
                origin_city=origin_city,
                wants_flights=wants_flights,
                wants_cruise=wants_cruise,
            )
        )

        weather_summary = "Weather data unavailable."
        try:
            weather_summary = await weather_task
        except Exception as ex:
            logger.warning("Weather fetch failed for %s: %s", destination.name, ex)
            weather_summary = "Weather feed temporarily unavailable."

        safety_summary = "Advisory feed unavailable. Manually verify current travel advisories at travel.state.gov."
        advisory_level = "Unknown"
        advisory_source_url = "https://travel.state.gov/content/travel.html"
        if safety_task is not None:
            try:
                safety_summary, advisory_level, advisory_source_url = await safety_task
            except Exception as ex:
                logger.warning("Safety advisory fetch failed for %s: %s", destination.name, ex)

        codes = await codes_task

        flight_coro = (
            _fetch_duffel_flight_results(
                client, codes.origin_airport_codes, codes.destination_airport_codes, check_in, check_out, traveler_count
            )
            if wants_flights
            else _skip("Duffel Flights: skipped (no origin city provided, so a route can't be formed).")
        )
        cruise_coro = (
            _fetch_cruise_search_results(client, codes.cruise_destination_id, destination.name)
            if wants_cruise
            else _skip("Tripadvisor Cruises: skipped (no cruise interest detected in the request).")
        )

        search_snippets = list(
            await asyncio.gather(
                _fetch_duffel_stays_results(client, destination, check_in, check_out, traveler_count),
                _fetch_visa_requirements(client, citizenship, destination.country_code),
                flight_coro,
                cruise_coro,
            )
        )

        if settings.travel_api_endpoints:
            search_snippets.extend(await _fetch_external_api_snippets(client, request_text, destination.name))

    return GroundingPacket(
        destination_hint=destination.label,
        destination_country_code=destination.country_code,
        weather_summary=weather_summary,
        safety_summary=safety_summary,
        advisory_level=advisory_level,
        advisory_source_url=advisory_source_url,
        local_transport_notes=_build_local_transport_notes(destination.name),
        external_api_snippets=search_snippets,
    )
