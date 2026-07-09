from __future__ import annotations

import re
from typing import Any

import httpx

from config import settings
from .schemas import GroundingPacket


# Extracts a destination name from the user's request using common linguistic patterns
def extract_destination_hint(request_text: str) -> str:
    patterns = [
        r"\bto\s+([A-Z][a-zA-Z\s\-]+)",
        r"\bin\s+([A-Z][a-zA-Z\s\-]+)",
        r"\bvisit\s+([A-Z][a-zA-Z\s\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, request_text)
        if match:
            return match.group(1).strip()
    return "Requested destination"


# Converts a destination name into geographic coordinates and country codes using Open-Meteo
async def _geocode_destination(destination: str) -> dict[str, Any] | None:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": destination, "count": 1, "language": "en", "format": "json"}

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    results = payload.get("results") or []
    return results[0] if results else None


# Fetches a 5-day weather forecast summary for a specific coordinate pair
async def _fetch_weather_summary(latitude: float, longitude: float) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 5,
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
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


# Retrieves official travel safety advisories based on a country code
async def _fetch_safety_summary(country_code: str) -> tuple[str, str, str]:
    url = "https://www.travel-advisory.info/api"

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.get(url, params={"countrycode": country_code})
        response.raise_for_status()
        data = response.json()

    advisories = data.get("data", {})
    country_data = advisories.get(country_code.upper())
    if not country_data:
        return (
            "No advisory details returned. Verify current guidance from travel.state.gov before booking.",
            "Unknown",
            "https://travel.state.gov/content/travel.html",
        )

    advisory = country_data.get("advisory", {})
    score = advisory.get("score", "Unknown")
    level = advisory.get("message", "Unknown advisory level")
    source = advisory.get("source", "https://travel.state.gov/content/travel.html")

    summary = (
        f"Safety advisory indicator: {level} (score={score}). "
        "Always cross-check with U.S. Department of State updates before final bookings."
    )
    return summary, str(level), source


# Gathers generic snippets from a list of configured external travel API endpoints
async def _fetch_external_api_snippets(request_text: str, destination: str) -> list[str]:
    if not settings.travel_api_endpoints:
        return []

    snippets: list[str] = []
    headers = {"Authorization": f"Bearer {settings.travel_api_key}"} if settings.travel_api_key else {}

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        for endpoint in settings.travel_api_endpoints:
            try:
                response = await client.get(
                    endpoint,
                    params={
                        "query": request_text,
                        "destination": destination,
                        "currency": settings.default_currency,
                        "locale": settings.default_locale,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                snippets.append(f"{endpoint}: {str(payload)[:900]}")
            except Exception as ex:
                snippets.append(f"{endpoint}: API unavailable ({ex})")

    return snippets


def _rapidapi_headers(host: str) -> dict[str, str]:
    return {
        "x-rapidapi-key": settings.rapidapi_key,
        "x-rapidapi-host": host,
    }


async def _fetch_rapidapi_source_snippet(
    source_name: str,
    source_url: str,
    source_host: str,
    params: dict[str, Any],
) -> str:
    if not source_url:
        return f"{source_name}: URL not configured."
    if not source_host:
        return f"{source_name}: Host not configured."
    if not settings.rapidapi_key:
        return f"{source_name}: RAPIDAPI_KEY missing."

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(
                source_url,
                params=params,
                headers=_rapidapi_headers(source_host),
            )
            response.raise_for_status()
            payload = response.json()
        return f"{source_name}: {str(payload)[:1200]}"
    except Exception as ex:
        return f"{source_name}: API unavailable ({ex})"


async def _fetch_hotel_search_results(
    destination: str,
    hotel_params: dict[str, Any] | None = None,
) -> str:
    """
    Implements the two-step hotel search:
    1. searchLocation -> get documentId
    2. searchHotels -> get hotel details using geoId (documentId)
    """
    if not settings.rapidapi_key:
        return "Hotel search unavailable: RAPIDAPI_KEY missing."

    host = "tripadvisor16.p.rapidapi.com"
    location_url = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchLocation"
    search_url = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchHotels"

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            # Step 1: Search for location code
            loc_response = await client.get(
                location_url,
                params={"query": destination},
                headers=_rapidapi_headers(host),
            )
            loc_response.raise_for_status()
            loc_data = loc_response.json()

            # Extract documentId (assuming it's in the first result)
            # The API response structure usually has a list of locations
            results = loc_data.get("data", [])
            if not results:
                return f"Hotel search failed: No location found for {destination}."

            document_id = results[0].get("documentId")
            if not document_id:
                return f"Hotel search failed: No documentId found for {destination}."

            # Step 2: Search for hotels using geoId
            hotel_params_to_use = hotel_params or {}
            hotel_params_to_use["geoId"] = document_id
            if "currencyCode" not in hotel_params_to_use:
                hotel_params_to_use["currencyCode"] = settings.default_currency

            hotel_response = await client.get(
                search_url,
                params=hotel_params_to_use,
                headers=_rapidapi_headers(host),
            )
            hotel_response.raise_for_status()
            hotel_data = hotel_response.json()

            return f"Tripadvisor Hotels ({destination}): {str(hotel_data)[:2000]}"

    except Exception as ex:
        return f"Hotel search unavailable: {ex}"


async def _fetch_cruise_search_results(
    destination: str,
    cruise_params: dict[str, Any] | None = None,
) -> str:
    """
    Implements the two-step cruise search:
    1. getLocation -> get destinationId
    2. searchCruises -> get cruise details using destinationId
    """
    if not settings.rapidapi_key:
        return "Cruise search unavailable: RAPIDAPI_KEY missing."

    host = "tripadvisor16.p.rapidapi.com"
    location_url = "https://tripadvisor16.p.rapidapi.com/api/v1/cruises/getLocation"
    search_url = "https://tripadvisor16.p.rapidapi.com/api/v1/cruises/searchCruises"

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            # Step 1: Search for cruise location
            loc_response = await client.get(
                location_url,
                params={"query": destination},
                headers=_rapidapi_headers(host),
            )
            loc_response.raise_for_status()
            loc_data = loc_response.json()

            # Extract destinationId from the "data" list
            results = loc_data.get("data", [])
            if not results:
                return f"Cruise search failed: No cruise location found for {destination}."

            # The user specified that the location is under "name" in "data"
            # We need to find the one that matches the destination or just take the first relevant one
            # For simplicity, we take the first result's ID if available
            destination_id = results[0].get("documentId") or results[0].get("id")
            if not destination_id:
                return f"Cruise search failed: No destinationId found for {destination}."

            # Step 2: Search for cruises using destinationId
            cruise_params_to_use = cruise_params or {}
            cruise_params_to_use["destinationId"] = destination_id
            if "currencyCode" not in cruise_params_to_use:
                cruise_params_to_use["currencyCode"] = settings.default_currency

            cruise_response = await client.get(
                search_url,
                params=cruise_params_to_use,
                headers=_rapidapi_headers(host),
            )
            cruise_response.raise_for_status()
            cruise_data = cruise_response.json()

            return f"Tripadvisor Cruises ({destination}): {str(cruise_data)[:2000]}"

    except Exception as ex:
        return f"Cruise search unavailable: {ex}"


async def _fetch_rapidapi_snippets(
    request_text: str,
    destination: str,
    country_code: str | None,
    flight_params: dict[str, Any] | None = None,
    hotel_params: dict[str, Any] | None = None,
    cruise_params: dict[str, Any] | None = None,
) -> list[str]:
    common_params = {
        "query": request_text,
        "destination": destination,
        "currency": settings.default_currency,
        "locale": settings.default_locale,
    }

    providers = settings.rapidapi_sources

    tripadvisor = providers.get("Tripadvisor", {})
    booking = providers.get("Booking.com", {})
    flights_sky = providers.get("Flights Scraper Sky", {})
    google_flights = providers.get("Google Flights", {})
    visa = providers.get("Visa Requirements", {})

    visa_params = {
        "citizenship": settings.default_citizenship_country_code,
        "destinationCountry": country_code or "",
        "destination": destination,
    }

    # Use flight_params if provided, otherwise fallback to common_params
    flight_params_to_use = flight_params if flight_params else common_params

    hotel_results = await _fetch_hotel_search_results(destination, hotel_params)
    cruise_results = await _fetch_cruise_search_results(destination, cruise_params) if cruise_params is not None else "Cruise search disabled."

    results = [
        await _fetch_rapidapi_source_snippet(
            "Tripadvisor",
            tripadvisor.get("url", ""),
            tripadvisor.get("host", ""),
            common_params,
        ),
        await _fetch_rapidapi_source_snippet(
            "Booking.com",
            booking.get("url", ""),
            booking.get("host", ""),
            common_params,
        ),
        await _fetch_rapidapi_source_snippet(
            "Flights Scraper Sky",
            flights_sky.get("url", ""),
            flights_sky.get("host", ""),
            flight_params_to_use,
        ),
        await _fetch_rapidapi_source_snippet(
            "Google Flights",
            google_flights.get("url", ""),
            google_flights.get("host", ""),
            flight_params_to_use,
        ),
        await _fetch_rapidapi_source_snippet(
            "Visa Requirements",
            visa.get("url", ""),
            visa.get("host", ""),
            visa_params,
        ),
    ]

    results.append(hotel_results)
    results.append(cruise_results)

    return results


def _build_local_transport_notes(destination: str) -> str:
    return (
        f"For {destination}, compare premium rental cars for flexibility, airport taxis/Uber for short stays, "
        "and metro/rail passes for city-center access. For late-night arrivals, prefer pre-booked transfers "
        "from reputable operators with published safety standards."
    )


async def gather_grounding_packet(
    request_text: str, flight_params: dict[str, Any] | None = None, hotel_params: dict[str, Any] | None = None, cruise_params: dict[str, Any] | None = None
) -> GroundingPacket:
    destination = extract_destination_hint(request_text)

    weather_summary = "Weather data unavailable."
    safety_summary = "Safety data unavailable."
    advisory_level = "Unknown"
    advisory_source_url = "https://travel.state.gov/content/travel.html"

    geocode = await _geocode_destination(destination)
    country_code = None

    if geocode:
        latitude = geocode.get("latitude")
        longitude = geocode.get("longitude")
        country_code = geocode.get("country_code", "")

        if latitude is not None and longitude is not None:
            try:
                weather_summary = await _fetch_weather_summary(float(latitude), float(longitude))
            except Exception:
                weather_summary = "Weather feed temporarily unavailable."

        if country_code:
            try:
                safety_summary, advisory_level, advisory_source_url = await _fetch_safety_summary(country_code)
            except Exception:
                safety_summary = (
                    "Advisory feed unavailable. Manually verify current travel advisories at travel.state.gov."
                )

    external_snippets = await _fetch_rapidapi_snippets(
        request_text=request_text,
        destination=destination,
        country_code=country_code,
        flight_params=flight_params,
        hotel_params=hotel_params,
        cruise_params=cruise_params,
    )

    if settings.travel_api_endpoints:
        external_snippets.extend(await _fetch_external_api_snippets(request_text, destination))

    return GroundingPacket(
        destination_hint=destination,
        weather_summary=weather_summary,
        safety_summary=safety_summary,
        advisory_level=advisory_level,
        advisory_source_url=advisory_source_url,
        local_transport_notes=_build_local_transport_notes(destination),
        external_api_snippets=external_snippets,
    )
