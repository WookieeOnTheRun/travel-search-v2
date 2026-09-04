from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime

import httpx
from dateutil import parser as dateutil_parser

from config import settings
from .http_utils import describe_error, duffel_headers, get_with_retry, rapidapi_headers
from .schemas import DestinationCandidate

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
# Airport code resolution (Duffel Places suggestions API)
# ---------------------------------------------------------------------------


async def resolve_airport_codes(
    client: httpx.AsyncClient,
    query: str,
    *,
    country_hint: str | None = None,
    max_codes: int = 3,
) -> list[str]:
    """Resolves a place name to real IATA airport codes via Duffel's Places suggestions
    endpoint (GET /places/suggestions?query=... -- verified against
    https://duffel.com/docs/api/places/get-place-suggestions on 2026-09-04) -- never via
    an LLM guessing codes from its own memory. Only entries where the Place `type` is
    "airport" are collected, since that's the code shape Duffel's own flight-search
    examples (offer_requests slices) use for origin/destination.

    `country_hint`, when given, is expected to be an ISO 3166-1 alpha-2 country code
    (DestinationCandidate.country_code) to match against the Place's `iata_country_code`.
    """
    if not settings.duffel_api_key or not query.strip():
        return []

    url = f"{settings.duffel_api_base_url}/places/suggestions"

    try:
        response = await get_with_retry(client, url, params={"query": query}, headers=duffel_headers())
        results = response.json().get("data") or []
    except Exception as ex:
        logger.warning("Duffel place search for %r failed: %s", query, describe_error(ex))
        return []

    airports = [entry for entry in results if entry.get("type") == "airport" and entry.get("iata_code")]
    if not airports:
        return []

    def _matches_country(entry: dict) -> bool:
        if not country_hint:
            return True
        return str(entry.get("iata_country_code") or "").upper() == country_hint.upper()

    matched = [entry for entry in airports if _matches_country(entry)]
    if not matched:
        logger.warning(
            "No Duffel airport result for %r matched country hint %r; falling back to top result %r",
            query,
            country_hint,
            airports[0].get("name"),
        )
        matched = airports[:1]

    codes: list[str] = []
    for entry in matched:
        code = entry["iata_code"]
        if code not in codes:
            codes.append(code)
        if len(codes) >= max_codes:
            break

    return codes[:max_codes]


async def resolve_origin_airport_codes(client: httpx.AsyncClient, origin_city_text: str, *, max_codes: int = 3) -> list[str]:
    """Same airport resolution, but for a free-typed origin city with no prior
    disambiguation step. Geocodes it first (one cheap public API call) so the
    same country-aware matching used for the destination applies to the origin.
    """
    if not origin_city_text.strip():
        return []
    origin_candidates = await geocode_candidates(client, origin_city_text, count=1)
    country_hint = origin_candidates[0].country_code if origin_candidates else None
    return await resolve_airport_codes(client, origin_city_text, country_hint=country_hint, max_codes=max_codes)


# ---------------------------------------------------------------------------
# Cruise destination id (Tripadvisor) -- endpoint verified to exist but is
# currently failing server-side for every query tested; kept so it self-heals
# and callers must treat None as "unavailable", not "no cruises found".
# ---------------------------------------------------------------------------


async def resolve_cruise_destination_id(
    client: httpx.AsyncClient, candidate: DestinationCandidate
) -> str | None:
    if not settings.rapidapi_key:
        return None

    host = settings.rapidapi_tripadvisor_host
    url = f"https://{host}/api/v1/cruises/getLocation"

    try:
        response = await get_with_retry(client, url, params={"query": candidate.name}, headers=rapidapi_headers(host))
        payload = response.json()
    except Exception as ex:
        logger.warning("Cruise location search for %r failed: %s", candidate.name, describe_error(ex))
        return None

    if payload.get("status") is False:
        logger.warning("Cruise location search for %r returned an error payload: %s", candidate.name, payload.get("message"))
        return None

    results = payload.get("data") or []
    if not results:
        return None

    chosen = next(
        (r for r in results if candidate.name.lower() in str(r.get("name") or r.get("title") or "").lower()),
        results[0],
    )
    return chosen.get("documentId") or chosen.get("id")
