from __future__ import annotations

import asyncio
import functools
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from .geoapify import osm_link, osm_tags, search_places, wikidata_id
from .http_utils import describe_error, get_with_retry
from .location_resolution import fold_text
from .schemas import (
    ActivitySearchResult,
    ActivitySuggestion,
    AdvisoryCheck,
    DestinationCandidate,
    WeatherOutlook,
)
from .units import format_temp_c
from .wikidata import fetch_sitelink_counts

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


# Advisory summaries arrive as HTML: strip tags, collapse whitespace, and truncate.
def _strip_html(text: str, *, max_chars: int = 500) -> str:
    plain = _HTML_TAG_PATTERN.sub(" ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:max_chars]


# The date part of an advisory's ISO-8601 timestamp, or None if missing or malformed.
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


# Maps a trip day (`base_date` + `day_offset`) onto the same calendar day in a past year, for
# looking up historical weather (Feb 29 falls back to Feb 28 in non-leap years).
def _shift_to_reference_year(base_date: date, day_offset: int, reference_year: int) -> date:
    try:
        anchor = base_date.replace(year=reference_year)
    except ValueError:
        # base_date is Feb 29 and reference_year isn't a leap year.
        anchor = base_date.replace(year=reference_year, day=28)
    return anchor + timedelta(days=day_offset)


# Daily max/min temperatures for one past date window from Open-Meteo's archive API (the
# response's "daily" block), or None if the call failed.
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


# A short text summary of Open-Meteo's live forecast (average high/low, rain chance) for the
# part of the trip within the 16-day forecast horizon, or a note explaining why there isn't one.
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


# Placeholder awaited instead of the forecast call when departure isn't within 14 days.
async def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Feature 3: activities / sightseeing search with timeframe-appropriateness
# comments.
#
# Returned as two lists -- most notable, and lesser-known but worth considering -- ranked toward
# the sightseeing interests named in the traveler's request (see fetch_seasonal_activities).
#
# Sourced from OpenStreetMap data via the Geoapify Places API (see geoapify.py for why it replaced
# the public Overpass API). Each place carries its original OSM tags, including opening_hours
# where mapped (verified 2026-10-19: 14 of 20 Lisbon museums had it). Tripadvisor's RapidAPI
# product (tripadvisor16.p.rapidapi.com, no longer used anywhere in this app -- flight and hotel
# search have since both moved to Booking.com) was checked for an attractions/things-to-do
# endpoint first: over a dozen plausible paths were probed live against it and every one 404'd.
# ---------------------------------------------------------------------------

_SEARCH_RADIUS_METERS = 8000
_MAX_NOTABLE = 8
_MAX_LESSER_KNOWN = 8
# Geoapify equivalents of the OSM tags searched before (tourism=attraction/museum/viewpoint/
# gallery/zoo/theme_park/artwork, historic=*); attraction covers viewpoints and artwork, and
# sights covers historic places.
_SIGHTSEEING_CATEGORIES = (
    "tourism.attraction,tourism.sights,entertainment.museum,entertainment.culture.gallery,"
    "entertainment.zoo,entertainment.theme_park"
)
# Fetched nearest-first and named-only (see geoapify.search_places). 200 places = 10 credits;
# verified 2026-10-19 that 100 isn't enough -- around central Rome the 100 nearest stop short
# of the Colosseum (~1.4 km away) and the 200 nearest include it. Measured ~2-3 s per request.
_SIGHTSEEING_FETCH_LIMIT = 200
# One extra, targeted request per interest named in the request (1 credit each), so places
# matching it are in the pool even where the general search is crowded out by statues and
# plaques. Capped so a request listing every topic stays within a few seconds.
_INTEREST_FETCH_LIMIT = 20
_MAX_INTEREST_SEARCHES = 4
# Wikidata sitelinks at or above this mark a place as "most notable". Verified 2026-10-19 on the
# 200-place pools: Lisbon -> Castelo de São Jorge 31, Elevador de Santa Justa 26, Rossio 18,
# Restauradores 15, Palácio da Bemposta 9; Denver -> Art Museum 25, Union Station 13; while
# minor statues/memorials sit at 0-4.
_NOTABLE_MIN_SITELINKS = 8


# A sightseeing interest the traveler can name in their request. `phrases` match the folded
# request text; `categories` is the Geoapify query for the targeted search (each name checked
# live 2026-10-19 -- unknown ones get HTTP 400); a place matches when one of its Geoapify
# categories starts with a `place_categories` entry or it carries one of `osm_keys`.
@dataclass(frozen=True)
class _Interest:
    label: str
    phrases: re.Pattern[str]
    categories: str
    place_categories: tuple[str, ...]
    osm_keys: tuple[str, ...] = ()


_INTERESTS = (
    _Interest(
        "history",
        re.compile(
            r"\bhistor(?:y|ic|ical)\b|\bancient\b|\bruins?\b|\barchaeolog\w*|\bmedieval\b|\bheritage\b"
            r"|\bcastles?\b|\bpalaces?\b|\bmonuments?\b|\broman\b"
        ),
        "tourism.sights.castle,tourism.sights.archaeological_site,tourism.sights.ruines,tourism.sights.memorial,heritage",
        ("heritage", "building.historic", "tourism.sights.castle", "tourism.sights.archaeological_site",
         "tourism.sights.ruines", "tourism.sights.memorial"),
        ("historic",),
    ),
    _Interest(
        "art",
        re.compile(r"\barts?\b|\bartists?\b|\bartwork\b|\bgaller(?:y|ies)\b|\bpaintings?\b|\bsculptures?\b|\bmurals?\b"),
        "entertainment.culture.gallery,tourism.attraction.artwork",
        ("entertainment.culture.gallery", "tourism.attraction.artwork"),
    ),
    _Interest("museums", re.compile(r"\bmuseums?\b"), "entertainment.museum", ("entertainment.museum",)),
    _Interest(
        "religious sites",
        re.compile(
            r"\bchurch(?:es)?\b|\bcathedrals?\b|\bbasilicas?\b|\btemples?\b|\bshrines?\b|\bmosques?\b"
            r"|\bsynagogues?\b|\bmonaster(?:y|ies)\b|\bspiritual\b|\breligious\b|\bsacred\b"
        ),
        "tourism.sights.place_of_worship,tourism.sights.monastery",
        ("tourism.sights.place_of_worship", "religion.place_of_worship", "tourism.sights.monastery"),
    ),
    _Interest(
        "parks & nature",
        re.compile(r"(?<!theme )(?<!water )\bparks?\b|\bgardens?\b|\bnature\b|\boutdoors?\b|\bgreen\s+spaces?\b|\bbotanical\b"),
        "leisure.park,natural",
        ("leisure.park", "natural"),
    ),
    _Interest(
        "views",
        re.compile(r"\bviews?\b|\bviewpoints?\b|\bpanoram\w*|\bskylines?\b|\bsunsets?\b|\blookouts?\b|\bscenic\b"),
        "tourism.attraction.viewpoint",
        ("tourism.attraction.viewpoint",),
    ),
    _Interest(
        "family & kids",
        re.compile(r"\bkids?\b|\bchildren\b|\bfamily\b|\bzoos?\b|\baquariums?\b|\btheme\s+parks?\b|\bamusement\b"),
        "entertainment.zoo,entertainment.aquarium,entertainment.theme_park",
        ("entertainment.zoo", "entertainment.aquarium", "entertainment.theme_park"),
    ),
    _Interest(
        "theater & performing arts",
        re.compile(r"\btheat(?:er|re)s?\b|\bopera\b|\bballet\b"),
        "entertainment.culture.theatre",
        ("entertainment.culture.theatre",),
    ),
    _Interest(
        "architecture",
        re.compile(r"\barchitect\w*|\bbridges?\b|\btowers?\b"),
        "tourism.sights.bridge,tourism.sights.tower",
        ("tourism.sights.bridge", "tourism.sights.tower"),
    ),
)

# A phrase is skipped when its own clause negates it ("no museums", "not into churches"), so
# "skip the museums, but I love art" yields only art.
_NEGATION_PATTERN = re.compile(r"\b(?:no|not|avoid|skip|without|hate|never|dislike|don'?t|nothing)\b")
_CLAUSE_BREAK_PATTERN = re.compile(r"[,.;:!?()]|\bbut\b|\band\b")

_MONTH_ABBREVS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_INDEX = {abbr: i + 1 for i, abbr in enumerate(_MONTH_ABBREVS)}
_MONTH_RANGE_PATTERN = re.compile(
    r"\b(" + "|".join(_MONTH_ABBREVS) + r")(?:-(" + "|".join(_MONTH_ABBREVS) + r"))?\b"
)


# Month numbers in an inclusive range, wrapping past December ("Nov-Feb" -> {11, 12, 1, 2}).
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


def extract_sightseeing_interests(request_text: str) -> list[str]:
    """Sightseeing interests the request actually names, in a fixed order -- rules only, so a
    topic is never attributed to the traveler unless a phrase in their own text says it."""
    text = fold_text(request_text or "")
    found = []
    for interest in _INTERESTS:
        for match in interest.phrases.finditer(text):
            clause = _CLAUSE_BREAK_PATTERN.split(text[max(0, match.start() - 40) : match.start()])[-1]
            if not _NEGATION_PATTERN.search(clause):
                found.append(interest.label)
                break
    return found


# Labels of the requested interests a place matches, by Geoapify category or OSM tag.
def _place_interests(place: dict, interests: list[_Interest]) -> list[str]:
    categories = place.get("categories") or []
    tags = osm_tags(place)
    return [
        i.label
        for i in interests
        if any(c.startswith(prefix) for c in categories for prefix in i.place_categories)
        or any(tags.get(key) for key in i.osm_keys)
    ]


def _documentation_signals(place: dict) -> list[str]:
    """Concrete, data-backed signs a less-documented place is still worth a visit."""
    tags = osm_tags(place)
    categories = place.get("categories") or []
    signals = []
    if wikidata_id(place):
        signals.append("has a Wikidata entry")
    if tags.get("wikipedia"):
        signals.append("has a Wikipedia article")
    if any(c.startswith("heritage") for c in categories):
        signals.append("listed as a heritage site")
    if any(c.startswith(("entertainment.museum", "entertainment.culture.gallery")) for c in categories):
        signals.append("a museum or gallery")
    if place.get("website") or tags.get("website"):
        signals.append("has its own website")
    if tags.get("opening_hours"):
        signals.append("lists opening hours")
    return signals


# "matches your interest in X and Y" -- the reason shown for an interest match.
def _interest_phrase(matched: list[str]) -> str:
    return f"matches your interest in {' and '.join(matched)}"


async def _gather_place_pool(
    client: httpx.AsyncClient, destination: DestinationCandidate, interests: list[_Interest]
) -> list[dict]:
    """The general sightseeing pool plus one targeted search per requested interest. The general
    search failing fails the whole lookup; an interest search failing only loses its extras."""
    search = functools.partial(
        search_places,
        client,
        latitude=destination.latitude,
        longitude=destination.longitude,
        radius_m=_SEARCH_RADIUS_METERS,
        nearest_first=True,
        named_only=True,
    )
    pool = await search(_SIGHTSEEING_CATEGORIES, limit=_SIGHTSEEING_FETCH_LIMIT)
    for interest in interests[:_MAX_INTEREST_SEARCHES]:
        try:
            pool += await search(interest.categories, limit=_INTEREST_FETCH_LIMIT)
        except Exception as ex:
            logger.warning("Geoapify %s search failed for %s: %s", interest.label, destination.name, describe_error(ex))

    unique: list[dict] = []
    seen: set[str] = set()
    for place in pool:
        name = place.get("name")
        if not name:
            continue
        keys = {f"name:{fold_text(name)}"} | ({f"wd:{wikidata_id(place)}"} if wikidata_id(place) else set())
        if keys & seen:
            continue
        seen |= keys
        unique.append(place)
    return unique


# Builds the ActivitySuggestion shown on the confirm screen for one place, including its
# opening-hours check against the travel dates.
def _to_suggestion(
    place: dict, start_date: date, end_date: date, sitelinks: int | None, matched: list[str], why: str
) -> ActivitySuggestion:
    tags = osm_tags(place)
    categories = place.get("categories") or []
    category = tags.get("tourism") or tags.get("historic") or (categories[-1].split(".")[-1] if categories else "point of interest")
    opening_hours = tags.get("opening_hours")
    return ActivitySuggestion(
        name=place["name"],
        category=str(category),
        appropriateness_comment=_assess_seasonal_appropriateness(opening_hours, start_date, end_date),
        raw_opening_hours=opening_hours,
        map_link=osm_link(place),
        sitelink_count=sitelinks,
        matched_interests=matched,
        why_listed=why[:1].upper() + why[1:],
    )


async def fetch_seasonal_activities(
    client: httpx.AsyncClient,
    destination: DestinationCandidate,
    start_date: date,
    end_date: date,
    request_text: str = "",
) -> ActivitySearchResult:
    """Nearby sightseeing in two ranked lists -- most notable, and lesser-known but worth a look
    -- each ordered toward the sightseeing interests named in `request_text`.

    Notability is Wikidata's sitelink count (how many Wikipedia language editions cover the
    place). If Wikidata is unreachable, having a Wikipedia article stands in for it, and the
    result says so.
    """
    interest_labels = extract_sightseeing_interests(request_text)
    interests = [i for i in _INTERESTS if i.label in interest_labels]
    try:
        places = await _gather_place_pool(client, destination, interests)
    except Exception as ex:
        logger.warning("Geoapify activity search failed for %s: %s", destination.name, describe_error(ex))
        return ActivitySearchResult(interests=interest_labels, error=f"unavailable ({describe_error(ex)})")

    notability_note = None
    try:
        sitelinks: dict[str, int] | None = await fetch_sitelink_counts(
            client, [wid for p in places if (wid := wikidata_id(p))]
        )
    except Exception as ex:
        logger.warning("Wikidata notability lookup failed for %s: %s", destination.name, describe_error(ex))
        sitelinks = None
        notability_note = (
            "Notability ranking is approximate: Wikidata was unavailable, so places with a Wikipedia "
            "article are treated as the most notable."
        )

    notable, lesser_known = [], []
    for place in places:
        count = sitelinks.get(wikidata_id(place) or "") if sitelinks is not None else None
        matched = _place_interests(place, interests)
        is_notable = (count or 0) >= _NOTABLE_MIN_SITELINKS if sitelinks is not None else bool(osm_tags(place).get("wikipedia"))
        if is_notable:
            why = f"covered by {count} Wikipedia language editions" if count is not None else "has a Wikipedia article"
            if matched:
                why += f"; {_interest_phrase(matched)}"
            notable.append((len(matched), count or 0, place, count, matched, why))
            continue
        signals = _documentation_signals(place)
        if not matched and not signals:
            continue  # nothing beyond a name and a category -- e.g. unremarkable bridges or plaques
        if count:
            signals = [f"covered by {count} Wikipedia language edition{'s' if count != 1 else ''}"] + [
                s for s in signals if s != "has a Wikidata entry"
            ]
        why = "; ".join(([_interest_phrase(matched)] if matched else []) + signals[:2])
        lesser_known.append((len(matched), count or 0, len(signals), place, count, matched, why))

    # Python's sort is stable, so ties keep Geoapify's nearest-first order.
    notable.sort(key=lambda row: (row[0], row[1]), reverse=True)
    lesser_known.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return ActivitySearchResult(
        notable=[_to_suggestion(p, start_date, end_date, c, m, w) for _, _, p, c, m, w in notable[:_MAX_NOTABLE]],
        lesser_known=[
            _to_suggestion(p, start_date, end_date, c, m, w) for _, _, _, p, c, m, w in lesser_known[:_MAX_LESSER_KNOWN]
        ],
        interests=interest_labels,
        notability_note=notability_note,
    )
