from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

import httpx

from config import settings
from .destination_insights import check_state_department_advisory
from .http_utils import describe_error, get_with_retry, post_with_retry, rapidapi_headers
from .location_resolution import (
    resolve_booking_hotel_destination,
    resolve_destination_flight_location,
    resolve_origin_flight_location,
)
from .schemas import DestinationCandidate, GroundingPacket, LocationCodes
from .units import format_temp_c

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
        f"5-day outlook: average highs around {format_temp_c(avg_high)}, lows around "
        f"{format_temp_c(avg_low)}, and average precipitation probability near {rain_risk}%."
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
# Flight lookups only run when they're actually relevant to this request (an
# origin was given) so a request with no flight angle doesn't pay for those calls.
# ---------------------------------------------------------------------------


async def resolve_location_codes(
    client: httpx.AsyncClient,
    destination: DestinationCandidate,
    *,
    origin_city: str | None,
    wants_flights: bool,
) -> LocationCodes:
    # Hotel/accommodation search always runs (see gather_grounding_packet below), so its
    # destination id is always resolved, unlike the flight ids which are gated behind
    # request relevance.
    hotel_destination_task = asyncio.create_task(resolve_booking_hotel_destination(client, destination))
    dest_flight_task = (
        asyncio.create_task(resolve_destination_flight_location(client, destination))
        if wants_flights
        else None
    )
    origin_flight_task = (
        asyncio.create_task(resolve_origin_flight_location(client, origin_city))
        if (wants_flights and origin_city)
        else None
    )

    pending = [t for t in (hotel_destination_task, dest_flight_task, origin_flight_task) if t]
    await asyncio.gather(*pending)

    dest_codes, dest_flight_location_id = dest_flight_task.result() if dest_flight_task else ([], None)
    origin_codes, origin_flight_location_id = origin_flight_task.result() if origin_flight_task else ([], None)
    hotel_dest_id, hotel_search_type = hotel_destination_task.result()

    return LocationCodes(
        destination_airport_codes=dest_codes,
        origin_airport_codes=origin_codes,
        destination_flight_location_id=dest_flight_location_id,
        origin_flight_location_id=origin_flight_location_id,
        hotel_dest_id=hotel_dest_id,
        hotel_search_type=hotel_search_type,
    )


# ---------------------------------------------------------------------------
# Search calls -- each one uses the real, verified parameter contract for its
# provider and degrades to a clearly-labeled "unavailable" string on failure
# rather than raising, so one provider going down doesn't abort the itinerary.
# ---------------------------------------------------------------------------


# Hotel/accommodation search: GET /api/v1/hotels/searchHotels on Booking.com's RapidAPI
# product, keyed by the `dest_id`/`search_type` pair resolved via
# resolve_booking_hotel_destination (hotels/searchDestination). Both the request contract
# (query params) and the response shape (`data.hotels[]`, each entry carrying `property.name`,
# `property.reviewScore`/`reviewScoreWord`/`reviewCount`, `property.accuratePropertyClass`
# (star rating), and `property.priceBreakdown.grossPrice.amountRounded`) were verified against
# a real, live call on 2026-09-10 -- this endpoint returns genuine hotel results end-to-end.
# `priceBreakdown.grossPrice` is already the total price for the exact arrival_date ->
# departure_date span requested (confirmed live on 2026-09-10: for the same hotel and arrival
# date, querying 1/2/3/4/5/6 nights returned $123/$323/$472/$635/$797/$954 -- a running total
# that grows with each added night, not a flat nightly rate repeated; `property.checkinDate`/
# `checkoutDate` in every response also matched the full requested span, not a subset of it).
# So this is a display clarification, not a multiplication: the night count and an average
# nightly rate are surfaced explicitly alongside the already-correct total so nothing
# downstream (a reader, or a specialist agent summarizing this text) mistakes the total for a
# single night's rate -- multiplying the total by nights again would double-count it.
def _format_hotel_price(price_breakdown: dict[str, Any] | None, nights: int) -> str:
    gross = (price_breakdown or {}).get("grossPrice") or {}
    value = gross.get("value")
    currency = gross.get("currency", "")
    amount_rounded = gross.get("amountRounded")
    total_label = str(amount_rounded) if amount_rounded else (f"{value:,.2f} {currency}".strip() if value is not None else None)
    if total_label is None:
        return "price unavailable"

    night_label = f"{nights} night{'s' if nights != 1 else ''}" if nights > 0 else "the stay"
    if value is not None and nights > 0:
        per_night = value / nights
        return f"{total_label} total for {night_label} (avg {per_night:,.2f} {currency}/night)"
    return f"{total_label} total for {night_label}"


def _format_hotel_result(hotel: dict[str, Any], nights: int) -> str:
    property_ = hotel.get("property") or {}
    name = property_.get("name") or "Unknown property"
    stars = property_.get("accuratePropertyClass") or property_.get("propertyClass")
    star_label = f"{stars}-star" if stars else "unrated"
    review_score = property_.get("reviewScore")
    review_word = property_.get("reviewScoreWord")
    review_count = property_.get("reviewCount")
    review_label = (
        f"{review_score}/10 {review_word} ({review_count} reviews)"
        if review_score is not None
        else "no reviews yet"
    )
    price = _format_hotel_price(property_.get("priceBreakdown"), nights)
    return f"{name} ({star_label}, {review_label}): {price}"


async def _fetch_booking_hotel_results(
    client: httpx.AsyncClient,
    hotel_dest_id: str | None,
    hotel_search_type: str | None,
    destination_name: str,
    check_in: date,
    check_out: date,
    traveler_count: int,
) -> str:
    if not hotel_dest_id or not hotel_search_type:
        return "Booking.com Hotels: unavailable (could not resolve a hotel destination id for this destination)."
    if not settings.rapidapi_key:
        return "Booking.com Hotels: RAPIDAPI_KEY missing."

    host = settings.rapidapi_booking_host
    url = f"https://{host}/api/v1/hotels/searchHotels"
    params: dict[str, Any] = {
        "dest_id": hotel_dest_id,
        "search_type": hotel_search_type,
        "arrival_date": check_in.isoformat(),
        "departure_date": check_out.isoformat(),
        "adults": traveler_count,
        "room_qty": 1,
        "page_number": 1,
        "units": "metric",
        "temperature_unit": "c",
        "languagecode": settings.default_locale.lower(),
        "currency_code": settings.default_currency,
    }
    try:
        response = await get_with_retry(client, url, params=params, headers=rapidapi_headers(host))
        payload = response.json()
    except Exception as ex:
        logger.warning("Booking.com Hotels search failed: %s", describe_error(ex))
        return f"Booking.com Hotels: unavailable ({describe_error(ex)})"

    if payload.get("status") is False:
        raw_message = payload.get("message")
        message = "; ".join(str(m) for m in raw_message) if isinstance(raw_message, list) else str(raw_message or "unknown error")
        logger.warning("Booking.com Hotels search returned an error payload: %s", message)
        return f"Booking.com Hotels: unavailable (API returned an error: {message})"

    hotels = (payload.get("data") or {}).get("hotels") or []
    if not hotels:
        return f"Booking.com Hotels ({destination_name}): no properties returned."

    nights = (check_out - check_in).days
    top_hotels = hotels[:5]
    hotel_summaries = "; ".join(_format_hotel_result(hotel, nights) for hotel in top_hotels)
    return (
        f"Booking.com Hotels ({destination_name}), {len(hotels)} properties found for "
        f"{nights} night{'s' if nights != 1 else ''} ({check_in.isoformat()} to {check_out.isoformat()}), "
        f"showing top {len(top_hotels)}: {hotel_summaries}"
    )


# Flight search: GET /api/v1/flights/searchFlights on Booking.com's RapidAPI product
# (booking-com15.p.rapidapi.com), replacing the Tripadvisor flight endpoints -- those never
# returned usable results on this subscription (searchAirport: empty `data` for every query;
# searchFlights: HTTP 200 with a `{"status": false, ...}` error body for every valid request).
# The endpoint path, required query params (fromId/toId as Booking.com location ids like
# "JFK.AIRPORT", departDate/returnDate in ISO YYYY-MM-DD, adults, cabinClass, sort,
# currency_code, pageNo), and response shape (`data.flightOffers[]`, each with `segments[]` ->
# `legs[]` carrying departure/arrival airports+times+carriersData, and `priceBreakdown.total` as
# a {currencyCode, units, nanos} money value) were verified against real, live calls on
# 2026-09-10 -- this endpoint returns genuine flight offers end-to-end, one-way and round-trip
# alike. An error payload comes back as `{"status": false, "message": [...]}` (a list of
# per-field errors, not a single string like Tripadvisor's) on HTTP 200, so `status` is checked
# explicitly here too rather than relying on raise_for_status().
def _format_money(money: dict[str, Any] | None) -> str:
    if not money:
        return "price unavailable"
    units = money.get("units", 0)
    nanos = money.get("nanos", 0)
    amount = units + nanos / 1_000_000_000
    currency = money.get("currencyCode", "")
    return f"{amount:,.2f} {currency}".strip()


_MAX_FLIGHT_OFFERS = 3


def _offer_primary_carrier(offer: dict[str, Any]) -> str:
    """The airline used for per-offer diversity in _select_flight_offers: the operating
    carrier of the first leg of the outbound segment (i.e. what a user would call "the
    airline" for that offer), not every carrier across a multi-carrier itinerary.
    """
    segments = offer.get("segments") or []
    legs = segments[0].get("legs") if segments else None
    carriers = legs[0].get("carriersData") if legs else None
    return (carriers[0].get("name") if carriers else None) or "unknown carrier"


def _select_flight_offers(offers: list[dict[str, Any]], *, max_offers: int = _MAX_FLIGHT_OFFERS) -> list[dict[str, Any]]:
    """Picks up to `max_offers` offers from the BEST-sorted `offers` list, preferring one
    per distinct airline. This is a soft preference, not a hard requirement: if there
    aren't enough distinct airlines among the results, remaining slots are filled with the
    next-best offers regardless of airline, so a route dominated by one carrier still
    returns `max_offers` options rather than fewer. Original BEST-sort order is preserved.
    """
    chosen_indices: list[int] = []
    seen_carriers: set[str] = set()

    for index, offer in enumerate(offers):
        if len(chosen_indices) >= max_offers:
            break
        carrier = _offer_primary_carrier(offer)
        if carrier not in seen_carriers:
            chosen_indices.append(index)
            seen_carriers.add(carrier)

    if len(chosen_indices) < max_offers:
        already_chosen = set(chosen_indices)
        for index in range(len(offers)):
            if len(chosen_indices) >= max_offers:
                break
            if index not in already_chosen:
                chosen_indices.append(index)
                already_chosen.add(index)

    return [offers[index] for index in sorted(chosen_indices)]


def _format_flight_offer(offer: dict[str, Any]) -> str:
    price = _format_money(offer.get("priceBreakdown", {}).get("total"))

    leg_summaries = []
    for segment in offer.get("segments") or []:
        legs = segment.get("legs") or []
        stops = max(len(legs) - 1, 0)
        stop_label = "nonstop" if stops == 0 else f"{stops} stop{'s' if stops != 1 else ''}"
        dep_code = segment.get("departureAirport", {}).get("code", "?")
        arr_code = segment.get("arrivalAirport", {}).get("code", "?")
        dep_time = segment.get("departureTime", "?")
        arr_time = segment.get("arrivalTime", "?")
        carriers = {
            carrier.get("name")
            for leg in legs
            for carrier in (leg.get("carriersData") or [])
            if carrier.get("name")
        }
        carrier_label = "/".join(sorted(carriers)) or "unknown carrier"
        leg_summaries.append(f"{dep_code}->{arr_code} {dep_time} to {arr_time} ({stop_label}, {carrier_label})")

    return f"{price} total: " + " | ".join(leg_summaries)


async def _fetch_booking_flight_results(
    client: httpx.AsyncClient,
    origin_location_id: str | None,
    destination_location_id: str | None,
    depart_date: date,
    return_date: date | None,
    traveler_count: int,
) -> str:
    if not origin_location_id or not destination_location_id:
        return "Booking.com Flights: unavailable (could not resolve origin/destination location ids)."
    if not settings.rapidapi_key:
        return "Booking.com Flights: RAPIDAPI_KEY missing."

    host = settings.rapidapi_booking_host
    url = f"https://{host}/api/v1/flights/searchFlights"
    params: dict[str, Any] = {
        "fromId": origin_location_id,
        "toId": destination_location_id,
        "departDate": depart_date.isoformat(),
        "pageNo": 1,
        "adults": traveler_count,
        "sort": "BEST",
        "cabinClass": "ECONOMY",
        "currency_code": settings.default_currency,
    }
    if return_date:
        params["returnDate"] = return_date.isoformat()

    try:
        response = await get_with_retry(client, url, params=params, headers=rapidapi_headers(host))
        payload = response.json()
    except Exception as ex:
        logger.warning("Booking.com Flights search failed: %s", describe_error(ex))
        return f"Booking.com Flights: unavailable ({describe_error(ex)})"

    if payload.get("status") is False:
        raw_message = payload.get("message")
        message = "; ".join(str(m) for m in raw_message) if isinstance(raw_message, list) else str(raw_message or "unknown error")
        logger.warning("Booking.com Flights search returned an error payload: %s", message)
        return f"Booking.com Flights: unavailable (API returned an error: {message})"

    offers = (payload.get("data") or {}).get("flightOffers") or []
    if not offers:
        return f"Booking.com Flights ({origin_location_id} -> {destination_location_id}): no offers returned."

    top_offers = _select_flight_offers(offers)
    offer_summaries = "; ".join(_format_flight_offer(offer) for offer in top_offers)
    return (
        f"Booking.com Flights ({origin_location_id} -> {destination_location_id}), "
        f"{len(offers)} offers found, showing {len(top_offers)} by BEST sort "
        f"(preferring distinct airlines): {offer_summaries}"
    )


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


async def gather_grounding_packet(
    destination: DestinationCandidate,
    *,
    request_text: str,
    origin_city: str | None,
    traveler_count: int,
    check_in: date,
    check_out: date,
    wants_flights: bool,
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
            _fetch_booking_flight_results(
                client,
                codes.origin_flight_location_id,
                codes.destination_flight_location_id,
                check_in,
                check_out,
                traveler_count,
            )
            if wants_flights
            else _skip("Booking.com Flights: skipped (no origin city provided, so a route can't be formed).")
        )

        search_snippets = list(
            await asyncio.gather(
                _fetch_booking_hotel_results(
                    client, codes.hotel_dest_id, codes.hotel_search_type, destination.name, check_in, check_out, traveler_count
                ),
                _fetch_visa_requirements(client, citizenship, destination.country_code),
                flight_coro,
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
