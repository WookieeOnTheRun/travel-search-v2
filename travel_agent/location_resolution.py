from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
from datetime import date, datetime

import httpx
from dateutil import parser as dateutil_parser

from config import settings
from .geo import haversine_km
from .http_utils import describe_error, get_with_retry, rapidapi_headers
from .schemas import DestinationCandidate, NearbyAirport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Destination candidates (natural language -> disambiguated place options)
# ---------------------------------------------------------------------------

_DESTINATION_HINT_PATTERNS = [
    re.compile(r"\bto\s+(?:the\s+)?([A-Z][a-zA-Z\-]*(?:\s+[A-Z][a-zA-Z\-]*)*)"),
    re.compile(r"\bin\s+(?:the\s+)?([A-Z][a-zA-Z\-]*(?:\s+[A-Z][a-zA-Z\-]*)*)"),
    re.compile(r"\bvisit(?:ing)?\s+(?:the\s+)?([A-Z][a-zA-Z\-]*(?:\s+[A-Z][a-zA-Z\-]*)*)"),
]


def extract_destination_phrases(request_text: str, *, max_phrases: int = 3) -> list[str]:
    """Pulls candidate place-name phrases out of free text for geocoding.

    This is a heuristic, not a resolver -- it only has to be good enough to
    produce *candidates* for the user to confirm, unlike the old single-shot
    regex extractor that picked one destination and used it directly. Multiple
    patterns are tried and merged so one missed phrasing doesn't lose the
    destination entirely.
    """
    phrases: list[str] = []
    for pattern in _DESTINATION_HINT_PATTERNS:
        for match in pattern.finditer(request_text):
            phrase = match.group(1).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases[:max_phrases]


async def resolve_destination_candidates(
    client: httpx.AsyncClient, request_text: str, *, max_candidates: int = 6
) -> list[DestinationCandidate]:
    """End-to-end: pull place-name phrases out of free text, geocode each one,
    and merge into a deduplicated candidate list for the user to pick from.
    """
    phrases = extract_destination_phrases(request_text)
    if not phrases:
        return []

    results_lists = await asyncio.gather(*(geocode_candidates(client, phrase, count=3) for phrase in phrases))

    merged: list[DestinationCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidates in results_lists:
        for candidate in candidates:
            key = (candidate.name.lower(), candidate.country_code)
            if key not in seen:
                seen.add(key)
                merged.append(candidate)

    return merged[:max_candidates]


async def geocode_candidates(
    client: httpx.AsyncClient, destination_text: str, *, count: int = 5
) -> list[DestinationCandidate]:
    """Resolves free-text destination wording into ranked, disambiguated place
    options via Open-Meteo's public geocoding API (no key required).

    Returns multiple candidates on purpose: a name like "Paris" or "Springfield"
    is genuinely ambiguous, and picking one silently (as a regex-based extractor
    would) is exactly the reliability gap this function replaces. Callers are
    expected to have the user confirm one candidate before it's used to resolve
    any provider-specific code.
    """
    if not destination_text.strip():
        return []

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": destination_text.strip(), "count": count, "language": "en", "format": "json"}

    try:
        response = await get_with_retry(client, url, params=params)
        results = response.json().get("results") or []
    except Exception as ex:
        logger.warning("Geocoding failed for %r: %s", destination_text, describe_error(ex))
        return []

    candidates: list[DestinationCandidate] = []
    for entry in results:
        name = entry.get("name")
        country = entry.get("country")
        latitude = entry.get("latitude")
        longitude = entry.get("longitude")
        if not name or not country or latitude is None or longitude is None:
            continue
        candidates.append(
            DestinationCandidate(
                name=name,
                country=country,
                country_code=entry.get("country_code", ""),
                admin1=entry.get("admin1"),
                latitude=float(latitude),
                longitude=float(longitude),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Travel date parsing
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

# Only attempt a parse when the text contains something that actually looks like
# a calendar date (a month name next to a day number, or an explicit numeric
# date). Running dateutil's fuzzy parser over arbitrary prose without this guard
# readily misfires on unrelated numbers in the request (a "$7000 budget", "2
# travelers", "7-day trip") and silently produces a wrong travel date.
_DATE_HINT_PATTERN = re.compile(
    rf"\b{_MONTH_NAMES}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAMES}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE,
)


def parse_travel_start_date(request_text: str, *, today: date | None = None) -> date | None:
    """Best-effort extraction of an explicit travel start date from free text.

    Returns None (rather than guessing) whenever no clear date-shaped substring
    is present, or when parsing the matched substring fails -- callers must
    treat None as "ask the user", not as "default to some date".
    """
    today = today or date.today()
    match = _DATE_HINT_PATTERN.search(request_text)
    if not match:
        return None

    try:
        parsed = dateutil_parser.parse(
            match.group(0), fuzzy=True, default=datetime(today.year, today.month, today.day)
        )
    except (ValueError, OverflowError):
        return None

    parsed_date = parsed.date()
    if parsed_date < today:
        # A bare "March 5" mentioned when today is in November almost always means
        # next year's March 5, not a date in the past.
        try:
            parsed_date = parsed_date.replace(year=parsed_date.year + 1)
        except ValueError:
            return None

    return parsed_date


# ---------------------------------------------------------------------------
# Airport code resolution (OurAirports public dataset, by great-circle distance)
#
# Tripadvisor's RapidAPI flight airport-search endpoint (GET /api/v1/flights/searchAirport
# on tripadvisor16.p.rapidapi.com) was used here previously, but is confirmed (live,
# 2026-09-08) to return an empty `data` array for every query tried -- city names, IATA
# codes, and single letters alike -- on this app's subscription. Rather than build a
# "nearest airports" feature on top of a source that has never once returned a result, this
# resolves airports directly from OurAirports.com's public-domain airport dataset (a
# well-known, widely-used open aviation dataset -- see https://ourairports.com/data/ and
# https://github.com/davidmegginson/ourairports-data; confirmed live and cross-checked
# against OurAirports' own data-dictionary docs on 2026-09-09) by great-circle distance from
# the destination/origin coordinates already resolved via geocoding. No API key required,
# and every code returned is a real row from that dataset -- never an LLM-guessed code.
# ---------------------------------------------------------------------------

_OURAIRPORTS_CSV_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
# large_airport/medium_airport with a populated iata_code covers essentially every airport
# with real scheduled commercial service (small_airport rows are overwhelmingly private
# strips/heliports without commercial flights) -- verified by inspecting a live pull of the
# dataset on 2026-09-09 (~4,500 of ~86,000 rows match this filter).
_RELEVANT_AIRPORT_TYPES = {"large_airport", "medium_airport"}

# Cached for the lifetime of the process: the filtered dataset is ~4,500 rows (cheap to hold
# in memory) but the raw CSV is ~13MB (too expensive to refetch on every airport lookup). Not
# lock-protected -- see the loop-binding note on `_host_locks` in http_utils.py for why an
# asyncio.Lock reused across Streamlit's per-rerun event loops would be actively dangerous
# here; the worst case of two concurrent callers both missing a cold cache is one redundant
# fetch, not corrupted data.
_airports_cache: list[dict[str, str]] | None = None


async def _load_airports_dataset(client: httpx.AsyncClient) -> list[dict[str, str]]:
    global _airports_cache
    if _airports_cache is not None:
        return _airports_cache

    response = await get_with_retry(client, _OURAIRPORTS_CSV_URL)
    reader = csv.DictReader(io.StringIO(response.text))
    airports = [row for row in reader if row.get("iata_code") and row.get("type") in _RELEVANT_AIRPORT_TYPES]
    _airports_cache = airports
    return airports


async def find_nearest_airports(
    client: httpx.AsyncClient, latitude: float, longitude: float, *, max_results: int = 5
) -> list[NearbyAirport]:
    """Finds the closest real airports (by great-circle distance) to a coordinate. Returns
    real IATA codes and distances from the OurAirports dataset; never a guess. Callers must
    treat an empty result as "unavailable" (e.g. the dataset fetch failed), not "no airports
    near here" -- every inhabited region on Earth is within range of at least a few rows in
    this dataset.
    """
    try:
        airports = await _load_airports_dataset(client)
    except Exception as ex:
        logger.warning("Airport dataset fetch failed: %s", describe_error(ex))
        return []

    scored: list[tuple[float, dict[str, str]]] = []
    for row in airports:
        try:
            lat = float(row["latitude_deg"])
            lon = float(row["longitude_deg"])
        except (KeyError, ValueError):
            continue
        scored.append((haversine_km(latitude, longitude, lat, lon), row))

    scored.sort(key=lambda item: item[0])

    return [
        NearbyAirport(
            iata_code=row["iata_code"],
            name=row.get("name") or row["iata_code"],
            municipality=row.get("municipality") or None,
            country_code=row.get("iso_country") or "",
            distance_km=round(distance, 1),
        )
        for distance, row in scored[:max_results]
    ]


async def resolve_destination_airport_codes(
    client: httpx.AsyncClient, destination: DestinationCandidate, *, max_codes: int = 3
) -> list[str]:
    """Just the IATA codes (for search API params) of the nearest airports to a confirmed
    destination -- see find_nearest_airports for the full, richer result used for display.
    """
    airports = await find_nearest_airports(client, destination.latitude, destination.longitude, max_results=max_codes)
    return [a.iata_code for a in airports]


async def resolve_origin_airport_codes(client: httpx.AsyncClient, origin_city_text: str, *, max_codes: int = 3) -> list[str]:
    """Same airport resolution, but for a free-typed origin city with no prior
    disambiguation step. Geocodes it first (one cheap public API call) to get coordinates.
    """
    if not origin_city_text.strip():
        return []
    origin_candidates = await geocode_candidates(client, origin_city_text, count=1)
    if not origin_candidates:
        return []
    origin = origin_candidates[0]
    return await resolve_destination_airport_codes(client, origin, max_codes=max_codes)


# ---------------------------------------------------------------------------
# Flight location id (Booking.com RapidAPI, GET /api/v1/flights/searchDestination) -- this
# is the id flights/searchFlights' fromId/toId params require (e.g. "JFK.AIRPORT"), not a bare
# IATA code. Verified live on 2026-09-10: querying by IATA code (e.g. "JFK") reliably returns
# an AIRPORT entry whose `code` matches the query and whose `id` is the exact value
# searchFlights needs -- confirmed by feeding that id straight into searchFlights and getting
# real flight offers back. Querying by the already-resolved IATA code (rather than a free-typed
# city name) is what keeps this an exact, non-ambiguous match instead of a guess.
# ---------------------------------------------------------------------------


async def resolve_booking_flight_location_id(client: httpx.AsyncClient, iata_code: str) -> str | None:
    if not settings.rapidapi_key or not iata_code:
        return None

    host = settings.rapidapi_booking_host
    url = f"https://{host}/api/v1/flights/searchDestination"

    try:
        response = await get_with_retry(client, url, params={"query": iata_code}, headers=rapidapi_headers(host))
        payload = response.json()
    except Exception as ex:
        logger.warning("Booking.com flight location search for %r failed: %s", iata_code, describe_error(ex))
        return None

    if payload.get("status") is False:
        logger.warning(
            "Booking.com flight location search for %r returned an error payload: %s",
            iata_code,
            payload.get("message"),
        )
        return None

    results = payload.get("data") or []
    match = next(
        (r for r in results if r.get("type") == "AIRPORT" and str(r.get("code") or "").upper() == iata_code.upper()),
        None,
    )
    return match.get("id") if match else None


async def _resolve_flight_location(client: httpx.AsyncClient, iata_codes: list[str]) -> str | None:
    """Tries each candidate IATA code in order (nearest airport first) until one resolves to a
    Booking.com location id, since a small/regional airport occasionally has no matching
    AIRPORT entry on this product even though a nearby larger one does.
    """
    for code in iata_codes:
        location_id = await resolve_booking_flight_location_id(client, code)
        if location_id:
            return location_id
    return None


async def resolve_destination_flight_location(
    client: httpx.AsyncClient, destination: DestinationCandidate
) -> tuple[list[str], str | None]:
    """Combines airport-code resolution (OurAirports, by distance) with Booking.com location-id
    resolution for a confirmed destination. Returns the raw IATA codes alongside the resolved
    Booking.com id so callers can still report/log the underlying codes.
    """
    codes = await resolve_destination_airport_codes(client, destination)
    location_id = await _resolve_flight_location(client, codes) if codes else None
    return codes, location_id


async def resolve_origin_flight_location(
    client: httpx.AsyncClient, origin_city_text: str
) -> tuple[list[str], str | None]:
    """Same as resolve_destination_flight_location, but for a free-typed origin city."""
    codes = await resolve_origin_airport_codes(client, origin_city_text)
    location_id = await _resolve_flight_location(client, codes) if codes else None
    return codes, location_id


# ---------------------------------------------------------------------------
# Hotel/accommodation destination id (Booking.com RapidAPI, GET
# /api/v1/hotels/searchDestination) -- verified live on 2026-09-10: a query like "Paris" or
# "man" returns multiple candidate places, each carrying its own `dest_id` + `search_type`
# pair (e.g. {"dest_id": "-1456928", "search_type": "city", ...}, alongside "district",
# "landmark", "airport", "region" entries for the same query). That exact `dest_id`/
# `search_type` pair -- not a bare id -- is what hotels/searchHotels requires (see
# _fetch_booking_hotel_results in data_sources.py). A "city" entry is preferred and matched
# against both the resolved place name and country code (e.g. Manchester, NH vs Manchester,
# UK both come back as "city" entries for query "man"), falling back to any "city" entry and
# then the first result overall if no confident match is found.
# ---------------------------------------------------------------------------


async def resolve_booking_hotel_destination(
    client: httpx.AsyncClient, candidate: DestinationCandidate
) -> tuple[str, str] | tuple[None, None]:
    if not settings.rapidapi_key:
        return None, None

    host = settings.rapidapi_booking_host
    url = f"https://{host}/api/v1/hotels/searchDestination"

    try:
        response = await get_with_retry(client, url, params={"query": candidate.name}, headers=rapidapi_headers(host))
        payload = response.json()
    except Exception as ex:
        logger.warning("Hotel destination search for %r failed: %s", candidate.name, describe_error(ex))
        return None, None

    if payload.get("status") is False:
        logger.warning(
            "Hotel destination search for %r returned an error payload: %s", candidate.name, payload.get("message")
        )
        return None, None

    results = payload.get("data") or []
    if not results:
        return None, None

    def _place_label(entry: dict) -> str:
        return str(entry.get("label") or entry.get("name") or entry.get("city_name") or "")

    def _is_city_entry(entry: dict) -> bool:
        return str(entry.get("search_type") or "").lower() == "city"

    def _is_confident_city_match(entry: dict) -> bool:
        if not _is_city_entry(entry):
            return False
        name_matches = candidate.name.lower() in _place_label(entry).lower()
        country_matches = not candidate.country_code or str(entry.get("cc1") or "").lower() == candidate.country_code.lower()
        return name_matches and country_matches

    chosen = (
        next((r for r in results if _is_confident_city_match(r)), None)
        or next((r for r in results if _is_city_entry(r)), None)
        or results[0]
    )

    dest_id = chosen.get("dest_id")
    search_type = chosen.get("search_type")
    if dest_id is None or not search_type:
        return None, None
    return str(dest_id), str(search_type)
