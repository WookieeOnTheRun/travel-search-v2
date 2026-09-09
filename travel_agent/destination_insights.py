from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timedelta

import httpx

from .http_utils import describe_error, get_with_retry, post_with_retry
from .schemas import (
    ActivitySearchResult,
    ActivitySuggestion,
    AdvisoryCheck,
    DestinationCandidate,
    WeatherOutlook,
)
from .units import format_temp_c

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature 1: U.S. Department of State travel advisory check.
#
# cadataapi.state.gov/api/TravelAdvisories is the Bureau of Consular Affairs'
# own JSON feed behind travel.state.gov's published advisories -- verified live
# on 2026-09-03 to return one entry per country with a "Title" of the form
# "<Country> - Level <1-4>: <headline>", a "Category" list of country codes
# (mostly ISO 3166-1 alpha-2, though a handful of countries -- e.g. Iraq as
# "IZ" -- use the State Department's legacy code instead of the ISO code), and
# "Published"/"Updated" ISO-8601 timestamps. No API key is required.
# ---------------------------------------------------------------------------

_STATE_DEPT_ADVISORIES_URL = "https://cadataapi.state.gov/api/TravelAdvisories"
_STATE_DEPT_ADVISORIES_PAGE = "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html"
_ADVISORY_WARNING_WINDOW_DAYS = 180  # "within the last six months"
# The feed covers essentially every country/territory the State Department tracks --
# live checks on 2026-09-03 consistently returned 216-224 entries. A response far
# shorter than that is more likely a truncated/partial fetch than a real drop to a
# fraction of the world's countries, so treat it as unavailable rather than risk
# reading a missing entry as "no advisory found" for its country.
_MIN_PLAUSIBLE_ADVISORY_COUNT = 100
_LEVEL_PATTERN = re.compile(r"Level\s+(\d)", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def _strip_html(text: str, *, max_chars: int = 500) -> str:
    plain = _HTML_TAG_PATTERN.sub(" ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:max_chars]


def _parse_advisory_timestamp(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


async def check_state_department_advisory(
    client: httpx.AsyncClient, destination: DestinationCandidate, *, today: date | None = None
) -> AdvisoryCheck:
    """Checks the live State Department advisory feed for the destination's country and
    flags whether a Level 3 ("Reconsider Travel") or Level 4 ("Do Not Travel") advisory was
    published/updated for it within the last six months. Returns found=False (not a guess)
    when no matching entry can be identified, rather than assuming "no warning".
    """
    today = today or date.today()

    # get_with_retry only retries transport errors and retryable HTTP status codes -- it has
    # no way to know that a 200 response carrying an implausibly short/empty JSON array is
    # itself a failure (a partial/truncated fetch, observed live). Retry that case here too,
    # rather than giving up on the first empty response.
    entries: list | None = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = await get_with_retry(client, _STATE_DEPT_ADVISORIES_URL)
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("unexpected response shape from State Department advisories feed")
            if len(payload) < _MIN_PLAUSIBLE_ADVISORY_COUNT:
                raise ValueError(f"advisories feed returned only {len(payload)} entries, likely a partial fetch")
            entries = payload
            break
        except Exception as ex:
            last_error = ex
            if attempt < 3:
                await asyncio.sleep(2 ** (attempt - 1))

    if entries is None:
        logger.warning("State Department advisory fetch failed: %s", describe_error(last_error))
        return AdvisoryCheck(error=f"unavailable ({describe_error(last_error)})")

    country_code = (destination.country_code or "").strip().upper()
    country_name = destination.country.strip().lower()

    def _matches(entry: dict) -> bool:
        categories = [str(c).strip().upper() for c in (entry.get("Category") or [])]
        if country_code and country_code in categories:
            return True
        title = str(entry.get("Title") or "").strip().lower()
        return bool(country_name) and title.startswith(country_name)

    matched = next((entry for entry in entries if _matches(entry)), None)
    if matched is None:
        return AdvisoryCheck(found=False)

    title = str(matched.get("Title") or "").strip() or None
    level_match = _LEVEL_PATTERN.search(title or "")
    level = int(level_match.group(1)) if level_match else None
    updated = _parse_advisory_timestamp(matched.get("Updated")) or _parse_advisory_timestamp(matched.get("Published"))
    summary_snippet = _strip_html(str(matched.get("Summary") or "")) or None
    link = str(matched.get("Link") or matched.get("id") or "") or None

    is_recent = updated is not None and (today - updated).days <= _ADVISORY_WARNING_WINDOW_DAYS
    active_warning = bool(level is not None and level >= 3 and is_recent)

    return AdvisoryCheck(
        found=True,
        active_warning=active_warning,
        level=level,
        title=title,
        updated=updated,
        link=link,
        summary_snippet=summary_snippet,
        source_url=_STATE_DEPT_ADVISORIES_PAGE,
    )


# ---------------------------------------------------------------------------
# Feature 2: weather outlook for the requested travel window.
#
# Historical averages come from Open-Meteo's Historical Weather API
# (archive-api.open-meteo.com/v1/archive, ERA5 reanalysis -- no key required),
# averaged over the same calendar window in each of the last 3 fully-elapsed
# years. When the trip departs within 14 days, an actual forecast for that
# window is pulled from Open-Meteo's standard forecast API (up to 16 days
# ahead), both verified live on 2026-09-03.
# ---------------------------------------------------------------------------

_HISTORICAL_YEARS_BACK = 3
_NEAR_TERM_WINDOW_DAYS = 14
_FORECAST_HORIZON_DAYS = 16


def _shift_to_reference_year(base_date: date, day_offset: int, reference_year: int) -> date:
    try:
        anchor = base_date.replace(year=reference_year)
    except ValueError:
        # base_date is Feb 29 and reference_year isn't a leap year.
        anchor = base_date.replace(year=reference_year, day=28)
    return anchor + timedelta(days=day_offset)


async def _fetch_historical_window(
    client: httpx.AsyncClient, latitude: float, longitude: float, start: date, end: date
) -> dict | None:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }
    try:
        response = await get_with_retry(client, url, params=params)
        return response.json().get("daily")
    except Exception as ex:
        logger.warning("Historical weather fetch failed for %s..%s: %s", start, end, describe_error(ex))
        return None


async def _fetch_near_term_forecast(
    client: httpx.AsyncClient, latitude: float, longitude: float, start: date, end: date
) -> str | None:
    horizon_end = date.today() + timedelta(days=_FORECAST_HORIZON_DAYS)
    covered_end = min(end, horizon_end)
    if covered_end < start:
        return "Departure is within 14 days, but the requested window falls outside Open-Meteo's 16-day forecast horizon."

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": covered_end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
    }
    try:
        response = await get_with_retry(client, url, params=params)
        daily = response.json().get("daily", {})
    except Exception as ex:
        logger.warning("Near-term forecast fetch failed for %s..%s: %s", start, end, describe_error(ex))
        return f"Current forecast unavailable ({describe_error(ex)})."

    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    if not max_temps:
        return "Current forecast unavailable for this window."

    avg_high = round(sum(max_temps) / len(max_temps), 1)
    avg_low = round(sum(min_temps) / len(min_temps), 1)
    coverage_note = (
        f"covers {start.isoformat()} to {covered_end.isoformat()} of your {start.isoformat()} to "
        f"{end.isoformat()} trip (the rest is beyond Open-Meteo's 16-day forecast horizon)"
        if covered_end < end
        else f"covers your full {start.isoformat()} to {end.isoformat()} trip window"
    )
    return (
        f"Departure is within 14 days, so here is the actual current forecast (not a historical average): "
        f"average highs around {format_temp_c(avg_high)}, lows around {format_temp_c(avg_low)}; "
        f"forecast {coverage_note}."
    )


async def fetch_weather_outlook(
    client: httpx.AsyncClient, destination: DestinationCandidate, start_date: date, end_date: date
) -> WeatherOutlook:
    """Averages historical daily highs/lows over the requested calendar window across the
    last few fully-elapsed years, and separately checks live forecast coverage when travel
    starts within 14 days.
    """
    today = date.today()
    trip_span_days = (end_date - start_date).days

    reference_years = [today.year - offset for offset in range(1, _HISTORICAL_YEARS_BACK + 1)]
    historical_windows = [
        (
            _shift_to_reference_year(start_date, 0, year),
            _shift_to_reference_year(start_date, trip_span_days, year),
        )
        for year in reference_years
    ]

    is_near_term = 0 <= (start_date - today).days <= _NEAR_TERM_WINDOW_DAYS

    historical_results, near_term_summary = await asyncio.gather(
        asyncio.gather(
            *(
                _fetch_historical_window(client, destination.latitude, destination.longitude, w_start, w_end)
                for w_start, w_end in historical_windows
            )
        ),
        _fetch_near_term_forecast(client, destination.latitude, destination.longitude, start_date, end_date)
        if is_near_term
        else _noop(),
    )

    all_highs: list[float] = []
    all_lows: list[float] = []
    years_used: list[int] = []
    for year, daily in zip(reference_years, historical_results):
        if not daily:
            continue
        highs = daily.get("temperature_2m_max") or []
        lows = daily.get("temperature_2m_min") or []
        if not highs:
            continue
        all_highs.extend(v for v in highs if v is not None)
        all_lows.extend(v for v in lows if v is not None)
        years_used.append(year)

    if not all_highs:
        return WeatherOutlook(
            error="unavailable (no historical weather data returned for this location/window)",
            is_near_term=is_near_term,
            near_term_forecast_summary=near_term_summary if is_near_term else None,
        )

    return WeatherOutlook(
        historical_avg_high_c=round(sum(all_highs) / len(all_highs), 1),
        historical_avg_low_c=round(sum(all_lows) / len(all_lows), 1),
        historical_years_used=years_used,
        is_near_term=is_near_term,
        near_term_forecast_summary=near_term_summary if is_near_term else None,
    )


async def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Feature 3: activities / sightseeing search with timeframe-appropriateness
# comments.
#
# Sourced from OpenStreetMap's public Overpass API (overpass-api.de/api/interpreter,
# no key required) -- verified live on 2026-09-03 to return named tourism/historic
# points of interest with tags including opening_hours. Tripadvisor's RapidAPI
# product already used elsewhere in this app (tripadvisor16.p.rapidapi.com) was
# checked for an attractions/things-to-do endpoint first: over a dozen plausible
# paths (attractions/searchAttractions, attraction/searchLocation, poi/*, tours/*,
# thingstodo/*, etc.) were probed live against it and every one 404'd, and its
# sibling restaurant endpoints that DO exist are currently failing server-side --
# so it is not used here rather than guessing at an unverified path.
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_SEARCH_RADIUS_METERS = 8000
_MAX_ACTIVITIES = 12
_TOURISM_TAGS = ["attraction", "museum", "viewpoint", "gallery", "zoo", "theme_park", "artwork"]

_MONTH_ABBREVS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_INDEX = {abbr: i + 1 for i, abbr in enumerate(_MONTH_ABBREVS)}
_MONTH_RANGE_PATTERN = re.compile(
    r"\b(" + "|".join(_MONTH_ABBREVS) + r")(?:-(" + "|".join(_MONTH_ABBREVS) + r"))?\b"
)


def _months_in_range(start_month: int, end_month: int) -> set[int]:
    if start_month <= end_month:
        return set(range(start_month, end_month + 1))
    return set(range(start_month, 13)) | set(range(1, end_month + 1))


def _assess_seasonal_appropriateness(opening_hours: str | None, start_date: date, end_date: date) -> str:
    """Best-effort read of an OpenStreetMap opening_hours string for month-scoped rules
    (e.g. "Apr-Oct: Mo-Su 10:00-18:00; Nov-Mar: off"). Not a full opening_hours-spec parser --
    OSM's syntax is broader than this -- so results are phrased as guidance to verify, not a
    guarantee.
    """
    if not opening_hours:
        return (
            "No seasonal restriction listed in OpenStreetMap data, so it appears open year-round -- "
            "should be available during your travel dates, but confirm current hours locally."
        )

    travel_months = _months_in_range(start_date.month, end_date.month)

    segments = [seg.strip() for seg in opening_hours.split(";") if seg.strip()]
    month_scoped_segments = []
    for segment in segments:
        match = _MONTH_RANGE_PATTERN.match(segment)
        if not match:
            continue
        start_month = _MONTH_INDEX[match.group(1)]
        end_month = _MONTH_INDEX[match.group(2)] if match.group(2) else start_month
        covered_months = _months_in_range(start_month, end_month)
        is_closed = bool(re.search(r"\boff\b|\bclosed\b", segment, re.IGNORECASE))
        month_scoped_segments.append((covered_months, is_closed, segment))

    if not month_scoped_segments:
        return (
            "Listed hours in OpenStreetMap don't call out seasonal closures, so it looks open "
            f"year-round (hours: {opening_hours}) -- should be appropriate for your travel dates, "
            "but confirm current hours locally."
        )

    overlapping_open = [seg for months, closed, seg in month_scoped_segments if not closed and months & travel_months]
    overlapping_closed = [seg for months, closed, seg in month_scoped_segments if closed and months & travel_months]

    if overlapping_open and not overlapping_closed:
        return f"Likely open during your travel window based on OpenStreetMap hours ({'; '.join(overlapping_open)}) -- verify locally before planning around it."
    if overlapping_closed and not overlapping_open:
        return f"May be closed for part or all of your travel window per OpenStreetMap hours ({'; '.join(overlapping_closed)}) -- worth double-checking before including it."
    if overlapping_open and overlapping_closed:
        return (
            "Seasonal hours in OpenStreetMap are mixed for your travel window "
            f"(open: {'; '.join(overlapping_open)}; closed: {'; '.join(overlapping_closed)}) -- check exact dates before planning around it."
        )
    return (
        f"OpenStreetMap lists seasonal hours ({opening_hours}) that don't clearly cover your travel months -- "
        "verify locally before planning around it."
    )


def _osm_link(element: dict) -> str | None:
    element_type = element.get("type")
    element_id = element.get("id")
    if not element_type or not element_id:
        return None
    return f"https://www.openstreetmap.org/{element_type}/{element_id}"


async def fetch_seasonal_activities(
    client: httpx.AsyncClient, destination: DestinationCandidate, start_date: date, end_date: date
) -> ActivitySearchResult:
    tag_filter = "|".join(_TOURISM_TAGS)
    query = (
        f"[out:json][timeout:20];"
        f"("
        f'node(around:{_SEARCH_RADIUS_METERS},{destination.latitude},{destination.longitude})[tourism~"^({tag_filter})$"];'
        f'way(around:{_SEARCH_RADIUS_METERS},{destination.latitude},{destination.longitude})[tourism~"^({tag_filter})$"];'
        f'node(around:{_SEARCH_RADIUS_METERS},{destination.latitude},{destination.longitude})[historic];'
        f");"
        f"out body {_MAX_ACTIVITIES * 3};"
    )

    try:
        response = await post_with_retry(
            client,
            _OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "agentic-travel-concierge/1.0 (destination activity search)"},
        )
        elements = response.json().get("elements", [])
    except Exception as ex:
        logger.warning("Overpass activity search failed for %s: %s", destination.name, describe_error(ex))
        return ActivitySearchResult(error=f"unavailable ({describe_error(ex)})")

    activities: list[ActivitySuggestion] = []
    seen_names: set[str] = set()
    for element in elements:
        tags = element.get("tags") or {}
        name = tags.get("name")
        if not name or name in seen_names:
            continue
        category = tags.get("tourism") or tags.get("historic") or "point of interest"
        opening_hours = tags.get("opening_hours")
        activities.append(
            ActivitySuggestion(
                name=name,
                category=str(category),
                appropriateness_comment=_assess_seasonal_appropriateness(opening_hours, start_date, end_date),
                raw_opening_hours=opening_hours,
                map_link=_osm_link(element),
            )
        )
        seen_names.add(name)
        if len(activities) >= _MAX_ACTIVITIES:
            break

    return ActivitySearchResult(activities=activities)
