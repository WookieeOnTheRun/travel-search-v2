from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from textwrap import dedent

import httpx
from agent_framework import Agent

from .destination_insights import fetch_weather_outlook
from .geo import haversine_km
from .http_utils import describe_error, get_with_retry
from .itinerary_agent import build_ollama_client
from .geoapify import search_places
from .location_resolution import geocode_candidates, parse_travel_month
from .schemas import (
    DestinationCandidate,
    DestinationDiscoveryResult,
    DiscoveredDestination,
    TripConstraints,
)
from .units import format_temp_c

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: parse nebulous free text into structured constraints.
#
# Deterministic regex parsing, same philosophy as location_resolution.py's date
# parsing -- returns None/False for anything not clearly stated rather than
# guessing a value, so callers know what was actually understood versus
# defaulted.
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")"


# Converts a digit string or a number word ("three") to a float.
def _to_number(token: str) -> float:
    token = token.lower()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    return float(token)


# The number captured by `pattern`'s first group, or None if the pattern doesn't match.
def _extract_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _to_number(match.group(1))


# Each amount must start with a digit: a bare "[\d,]+" also matches a lone comma (e.g. "$,"),
# which then fails float() and aborts discovery for the whole request.
# An optional "k" suffix multiplies by 1,000 ("$3k").
_BUDGET_PATTERNS = [
    r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?(k)?\b",
    r"(\d[\d,]*(?:\.\d+)?)\s?(k)?\s*(?:usd|dollars)\b",
]
# Trip length is used as the check-out offset from the start date, so "5 nights" and "5 days"
# both mean 5. Tried in order; the first match wins.
_TRIP_LENGTH_PATTERNS = [
    (rf"\b({_NUMBER_PATTERN})[\s-]*(?:day|night)s?\b", 1),
    (rf"\b({_NUMBER_PATTERN})[\s-]*weeks?\b", 7),
    (r"\b(a)\s+fortnight\b", 14),
    # "a week", "beach week" -- but not "next week"/"per week", which are dates/rates.
    (r"(?<!\bnext )(?<!\bthis )(?<!\blast )(?<!\bper )(?<!\beach )\b(week)\b", 7),
    (r"\b(weekend)\b", 2),
]
_TRAVELER_PATTERNS = [
    rf"\b({_NUMBER_PATTERN})\s+(?:travell?ers?|people|persons|adults|guests|pax)\b",
    rf"\bfamily\s+of\s+({_NUMBER_PATTERN})\b",
]
_SOLO_PATTERN = re.compile(r"\bsolo\b|\bjust\s+me\b|\bby\s+myself\b|\btravell?ing\s+alone\b", re.IGNORECASE)
_FLIGHT_HOURS_PATTERN = rf"\b({_NUMBER_PATTERN})[\s-]*hour[s]?\s*flight"
_DRIVE_HOURS_PATTERN = rf"\b({_NUMBER_PATTERN})[\s-]*hour[s]?\s*drive"

_WARM_PATTERN = re.compile(r"\bwarm\b|\bhot\b|\btropical\b|\bsunny\b", re.IGNORECASE)
_BEACH_PATTERN = re.compile(r"\bbeach(?:es)?\b", re.IGNORECASE)
_POOL_RESORT_PATTERN = re.compile(r"\bpool\b|\bswim[\s-]?up\b|\bresort\b", re.IGNORECASE)
_FOOD_PATTERN = re.compile(
    r"\bfood(?:s|ie|ies)?\b|\bculinary\b|\bcuisine\b|\brestaurants?\b|\bdining\b|\beat(?:ing)?\b",
    re.IGNORECASE,
)
_HIKING_PATTERN = re.compile(r"\bhik(?:e|es|ing|er|ers)\b|\btrek(?:s|king)?\b|\btrails?\b", re.IGNORECASE)
_LOW_COST_PATTERN = re.compile(
    r"\bcheap(?:er|est|ly)?\b|\baffordable\b|\binexpensive\b|\blow[\s-]cost\b|\bbudget[\s-]friendly\b"
    r"|\bon\s+a\s+(?:tight\s+|shoestring\s+)?budget\b|\bshoestring\b",
    re.IGNORECASE,
)
_ENTERTAINMENT_PATTERN = re.compile(
    r"\bentertainment\b|\bnightlife\b|night\s*life|\bbars?\b|\bclubs?\b|live music", re.IGNORECASE
)
_AVOID_CULTURAL_PATTERN = re.compile(
    r"not\s+(?:really\s+)?interested\s+in\s+cultural|"
    r"no\s+interest\s+in\s+cultural|"
    r"not\s+into\s+(?:cultural|sightseeing|museums)|"
    r"skip\s+(?:the\s+)?(?:museums|sightseeing|cultural)|"
    r"avoid\s+(?:cultural|sightseeing)|"
    r"not\s+.{0,25}\bsightseeing\b",
    re.IGNORECASE,
)


# Total budget in dollars from the first "$N"/"N USD" mention, or None if none is stated.
def extract_budget(text: str) -> float | None:
    for pattern in _BUDGET_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1).replace(",", ""))
            return amount * 1000 if match.group(2) else amount
    return None


# Trip length in days ("5 nights", "two weeks", "a weekend"), or None if none is stated.
def extract_trip_length(text: str) -> int | None:
    for pattern, multiplier in _TRIP_LENGTH_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            token = match.group(1).lower()
            count = _to_number(token) if token in _NUMBER_WORDS or token[0].isdigit() else 1
            return int(count * multiplier)
    return None


# Number of travelers ("4 adults", "family of five"; 1 for "solo"), or None if not stated.
def extract_traveler_count(text: str) -> int | None:
    for pattern in _TRAVELER_PATTERNS:
        count = _extract_number(pattern, text)
        if count is not None:
            return int(count)
    return 1 if _SOLO_PATTERN.search(text) else None


def parse_trip_constraints(request_text: str) -> TripConstraints:
    """Deterministically extracts budget/duration/travel-time/vibe constraints from free
    text. Every field is None/False unless the text actually said so -- this never invents
    a number or preference the traveler didn't state.
    """
    budget = extract_budget(request_text)
    trip_length = extract_trip_length(request_text)
    flight_hours = _extract_number(_FLIGHT_HOURS_PATTERN, request_text)
    drive_hours = _extract_number(_DRIVE_HOURS_PATTERN, request_text)

    return TripConstraints(
        raw_text=request_text,
        budget_usd=budget,
        trip_length_days=trip_length,
        max_flight_hours=flight_hours,
        max_drive_hours=drive_hours,
        wants_warm=bool(_WARM_PATTERN.search(request_text)),
        wants_beach=bool(_BEACH_PATTERN.search(request_text)),
        wants_pool_resort=bool(_POOL_RESORT_PATTERN.search(request_text)),
        wants_food_scene=bool(_FOOD_PATTERN.search(request_text)),
        wants_entertainment=bool(_ENTERTAINMENT_PATTERN.search(request_text)),
        wants_hiking=bool(_HIKING_PATTERN.search(request_text)),
        wants_low_cost=bool(_LOW_COST_PATTERN.search(request_text)),
        avoid_cultural=bool(_AVOID_CULTURAL_PATTERN.search(request_text)),
        travel_month=parse_travel_month(request_text),
        traveler_count=extract_traveler_count(request_text),
    )


# ---------------------------------------------------------------------------
# Step 2: convert stated travel-time budgets into a search radius.
#
# These speed constants are estimates, not measurements -- documented and
# cited so the assumption is visible rather than hidden inside a "guessed"
# number:
#   - Flight: sources on commercial airliner speed (flyingmag.com,
#     executiveflyers.com, aerotime.aero -- see chat for citations) put cruise
#     speed at 550-600 mph, but the average speed across a whole flight (block
#     speed, including taxi/climb/descent) is meaningfully lower -- commonly
#     cited around 400-500 mph for typical routes, with shorter flights pulled
#     down more by the fixed climb/descent overhead. We use 500 mph (~805
#     km/h), the upper end of that band, since a "3 hour flight" traveler is
#     describing a short-to-medium-haul trip where cruise dominates more than
#     on very short hops.
#   - Driving: road-trip planning guidance (theroadtripexpert.com and others)
#     converges on ~55-65 mph as a realistic long-distance highway average
#     once stops/traffic/speed changes are included. We use 60 mph (~97 km/h).
# Both are used only to size the initial search radius; every surviving
# candidate's actual flight/drive time is re-derived from real geocoded
# coordinates (haversine for flight, live OSRM routing for driving) rather
# than trusted at this rough radius-sizing stage.
# ---------------------------------------------------------------------------

AVG_FLIGHT_SPEED_KMH = 805.0
AVG_DRIVE_SPEED_KMH = 97.0
# Estimates (great-circle distance / assumed average speed) are systematically
# optimistic versus real routes -- real flight paths and roads aren't straight
# lines. This tolerance keeps the initial candidate net wide enough that a
# genuinely-in-range destination isn't dropped due to estimate error; final
# feasibility for display uses the more precise per-candidate numbers.
_RADIUS_TOLERANCE = 1.25

# Our app's own working definition of "warm" for filtering purposes -- a design
# choice, not an external authority's definition. Documented so it's visible
# and adjustable, not a hidden assumption.
WARM_THRESHOLD_C = 24.0  # ~75°F


# ---------------------------------------------------------------------------
# Step 3: brainstorm candidate destination NAMES.
#
# The LLM's only job here is to propose real place names worth checking -- its
# instructions explicitly forbid asserting any fact (weather, distance,
# amenities) about them, since nothing it says is trusted directly. Every name
# it returns is independently geocoded and fact-checked in the verification
# step below before anything about it reaches the user; names that don't
# survive verification are dropped. If the model call fails (e.g. no local
# model endpoint configured/running), a small fallback list of real,
# well-known warm/coastal destination names is used instead -- it carries the
# same "just names, nothing asserted" property and goes through the identical
# verification pipeline.
# ---------------------------------------------------------------------------

_DESTINATION_IDEA_AGENT_INSTRUCTIONS = dedent(
    """
    You are a destination brainstorming specialist for a travel search tool.
    Given a traveler's described preferences and a departure point, propose a diverse list
    of REAL, well-known destinations that plausibly match the vibe described.

    Rules:
    - Output ONLY a plain list, one destination per line, formatted as "City, Country"
      (or "City, State, Country" for well-known US destinations).
    - Do not include any other text, headings, numbering, or explanation.
    - Do not state or imply any fact about weather, distance, flight time, prices, or
      amenities for any place you list -- a separate system independently verifies all of
      that. Your only job is to propose plausible names worth checking.
    - Propose at least 15 and up to 25 distinct real places. Prefer geographic diversity
      over repeating similar nearby towns.
    - Only propose places that really exist. Never invent a fictional or uncertain name.
    """
).strip()

# Real, well-known warm/coastal leisure destinations spanning multiple regions, used only
# if the brainstorming agent call fails. These are place NAMES only -- no claim about
# weather, amenities, or distance from any particular origin is made here; every name
# still goes through the same independent verification as LLM-suggested names.
_FALLBACK_CANDIDATE_NAMES = [
    "Cancun, Mexico", "Playa del Carmen, Mexico", "Tulum, Mexico", "Cabo San Lucas, Mexico",
    "Puerto Vallarta, Mexico", "Nassau, Bahamas", "Punta Cana, Dominican Republic",
    "San Juan, Puerto Rico", "Aruba", "Barbados", "Montego Bay, Jamaica", "George Town, Cayman Islands",
    "Miami, Florida, United States", "Fort Lauderdale, Florida, United States",
    "Key West, Florida, United States", "Honolulu, Hawaii, United States", "Lahaina, Hawaii, United States",
    "San Diego, California, United States", "Myrtle Beach, South Carolina, United States",
    "New Orleans, Louisiana, United States", "Las Vegas, Nevada, United States",
    "Panama City, Panama", "San Jose, Costa Rica", "Cartagena, Colombia",
    "Nice, France", "Palma de Mallorca, Spain", "Ibiza, Spain", "Santorini, Greece",
    "Mykonos, Greece", "Dubrovnik, Croatia", "Malta", "Marbella, Spain",
    "Phuket, Thailand", "Bali, Indonesia", "Da Nang, Vietnam", "Boracay, Philippines",
    "Dubai, United Arab Emirates", "Doha, Qatar", "Marrakesh, Morocco",
]


# Matches the "up to 25" the brainstorming prompt asks for. Model output isn't trusted to obey
# that, and every kept name costs a geocoding call, so the list is hard-capped here.
_MAX_BRAINSTORMED_NAMES = 25


# Turns the model's one-name-per-line reply into a deduplicated list, stripping bullets and
# numbering and dropping implausibly long lines.
def _parse_name_list(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*• ").strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned).strip()
        if not cleaned or len(cleaned) > 80:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(cleaned)
        if len(names) >= _MAX_BRAINSTORMED_NAMES:
            break
    return names


# Month name for a 1-12 month number, or None.
def _month_label(month: int | None) -> str | None:
    return date(2000, month, 1).strftime("%B") if month else None


# Step 2: asks the model for candidate destination names (names only -- nothing it says is
# trusted as fact). Falls back to _FALLBACK_CANDIDATE_NAMES if the model fails or returns
# nothing usable. Returns the names and a note on where they came from.
async def _brainstorm_candidate_names(constraints: TripConstraints, origin_label: str) -> tuple[list[str], str]:
    prompt = dedent(
        f"""
        Traveler's departure point: {origin_label}
        Traveler's request (verbatim): {constraints.raw_text}
        Parsed preferences: warm={constraints.wants_warm}, beach={constraints.wants_beach},
        pool/resort={constraints.wants_pool_resort}, food scene={constraints.wants_food_scene},
        entertainment/nightlife={constraints.wants_entertainment}, hiking={constraints.wants_hiking},
        low cost={constraints.wants_low_cost}, total budget USD={constraints.budget_usd},
        travel month={_month_label(constraints.travel_month)}, trip length days={constraints.trip_length_days},
        avoid cultural/sightseeing={constraints.avoid_cultural}

        List candidate destination names now, one per line, "City, Country" format only.
        """
    ).strip()

    try:
        client = build_ollama_client()
        agent = Agent(name="DestinationIdeaAgent", client=client, instructions=_DESTINATION_IDEA_AGENT_INSTRUCTIONS)
        response = await agent.run(prompt)
        names = _parse_name_list(str(response))
        if names:
            return names, "brainstormed by the configured travel-idea model, then independently verified"
        logger.warning("Destination idea agent returned no parseable names; using fallback list")
    except Exception as ex:
        logger.warning("Destination idea agent unavailable, using fallback list: %s", describe_error(ex))

    return list(_FALLBACK_CANDIDATE_NAMES), "drawn from a static list of well-known warm/coastal destinations, then independently verified"


# ---------------------------------------------------------------------------
# Step 4: independently verify every candidate against real data.
# ---------------------------------------------------------------------------

# Search radii: a beach within 6 km counts as "nearby"; restaurants/nightlife are counted within 3 km.
_BEACH_CHECK_RADIUS_M = 6000
_POI_DENSITY_RADIUS_M = 3000
# Result caps match where _score stops rising (20 restaurants, 10 bars), so each check costs one
# Geoapify credit (1 credit per 20 places) and a count at the cap is shown as "N+".
_RESTAURANT_CAP = 20
_NIGHTLIFE_CAP = 10
_NIGHTLIFE_CATEGORIES = "catering.bar,catering.pub,adult.nightclub"
# After this many candidates in a row whose place checks all failed, the rest of the shortlist
# is marked "unavailable" without calling the service. (Added after an Overpass outage where
# paced calls for 10 candidates took ~143 s and every one failed anyway.)
_MAX_CONSECUTIVE_PLACE_FAILURES = 2
# HTTPS, as in the demo server's own documented example query
# (https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server; verified live 2026-10-09) --
# plain HTTP would send the traveler's origin/destination coordinates unencrypted.
_OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
# The public OSRM demo server's usage policy caps this at 1 request/second and forbids
# heavy/scraped use -- real drive times are only fetched sequentially, with this pacing,
# and only for the bounded shortlist that survives the cheap distance pre-filter below.
_OSRM_MIN_INTERVAL_SECONDS = 1.05
_OSRM_MAX_DISTANCE_KM = 3500  # beyond this, driving isn't realistic; skip the routing call


# Real routed driving time in hours from OSRM, or None if no route was found or the call failed.
async def _fetch_real_drive_hours(client: httpx.AsyncClient, origin: DestinationCandidate, dest: DestinationCandidate) -> float | None:
    url = f"{_OSRM_URL}/{origin.longitude},{origin.latitude};{dest.longitude},{dest.latitude}"
    try:
        # max_attempts=2 (not the default 3): this is a best-effort enrichment signal with a
        # graceful straight-line-estimate fallback already in place, on a shared public demo
        # server -- worth bounding worst-case latency over squeezing out one more retry.
        response = await get_with_retry(client, url, params={"overview": "false"}, max_attempts=2)
        payload = response.json()
        routes = payload.get("routes") or []
        if not routes:
            return None
        return float(routes[0]["duration"]) / 3600.0
    except Exception as ex:
        logger.warning("OSRM drive-time lookup failed for %s -> %s: %s", origin.name, dest.name, describe_error(ex))
        return None


# Number of Geoapify places in `categories` within `radius_m` of the destination (at most
# `cap`), or None if the lookup failed -- so "zero found" and "couldn't check" stay distinct.
async def _count_places(
    client: httpx.AsyncClient, dest: DestinationCandidate, categories: str, radius_m: int, cap: int
) -> int | None:
    try:
        places = await search_places(
            client, categories, latitude=dest.latitude, longitude=dest.longitude, radius_m=radius_m, limit=cap
        )
        return len(places)
    except Exception as ex:
        logger.warning("Geoapify place check (%s) failed for %s: %s", categories, dest.name, describe_error(ex))
        return None


async def _check_beach_and_poi_density(
    client: httpx.AsyncClient, dest: DestinationCandidate
) -> tuple[bool | None, int | None, int | None]:
    """Beach presence, restaurant count and nightlife count from Geoapify (OpenStreetMap data),
    as three capped requests -- 3 credits per candidate. Verified live 2026-10-19: beaches found
    near Lisbon and Cancun, none near inland Oaxaca City, Denver or Madrid.
    """
    beach = await _count_places(client, dest, "beach", _BEACH_CHECK_RADIUS_M, 1)
    restaurants = await _count_places(client, dest, "catering.restaurant", _POI_DENSITY_RADIUS_M, _RESTAURANT_CAP)
    nightlife = await _count_places(client, dest, _NIGHTLIFE_CATEGORIES, _POI_DENSITY_RADIUS_M, _NIGHTLIFE_CAP)
    return (beach > 0 if beach is not None else None), restaurants, nightlife


# Display label for a capped count: "20+" when the cap was reached.
def _count_label(count: int | None, cap: int) -> str:
    return f"{count}+" if count is not None and count >= cap else str(count or 0)


# Deterministic ranking score: 1 base point, plus up to 3 for warmth above the threshold
# (1 per 2 °C), 2 for a beach (when beach/pool was wanted), up to 2 for restaurants (1 per 10)
# and up to 2 for nightlife (1 per 5). Only preferences the traveler stated count.
def _score(discovered: DiscoveredDestination, constraints: TripConstraints) -> float:
    score = 1.0
    if constraints.wants_warm and discovered.historical_avg_high_c is not None:
        score += min(3.0, max(0.0, (discovered.historical_avg_high_c - WARM_THRESHOLD_C) / 2.0))
    if (constraints.wants_beach or constraints.wants_pool_resort) and discovered.has_beach_nearby:
        score += 2.0
    if constraints.wants_food_scene and discovered.restaurant_count_nearby:
        score += min(2.0, discovered.restaurant_count_nearby / 10.0)
    if constraints.wants_entertainment and discovered.nightlife_count_nearby:
        score += min(2.0, discovered.nightlife_count_nearby / 5.0)
    return round(score, 2)


# Geocodes one brainstormed name and keeps it only if it's within the flight or drive radius
# (with tolerance). Returns the surviving candidate with distance and estimated travel times,
# or None if it can't be geocoded or is too far.
async def _verify_candidate(
    client: httpx.AsyncClient,
    name: str,
    origin: DestinationCandidate,
    constraints: TripConstraints,
    max_flight_radius_km: float | None,
    max_drive_radius_km: float | None,
) -> DiscoveredDestination | None:
    matches = await geocode_candidates(client, name, count=1)
    if not matches:
        return None
    destination = matches[0]

    distance_km = haversine_km(origin.latitude, origin.longitude, destination.latitude, destination.longitude)
    estimated_flight_hours = distance_km / AVG_FLIGHT_SPEED_KMH

    if max_flight_radius_km is None and max_drive_radius_km is None:
        passes_distance = True
    else:
        within_flight = max_flight_radius_km is not None and distance_km <= max_flight_radius_km * _RADIUS_TOLERANCE
        within_drive = max_drive_radius_km is not None and distance_km <= max_drive_radius_km * _RADIUS_TOLERANCE
        passes_distance = within_flight or within_drive
    if not passes_distance:
        return None

    # A straight-line/avg-speed drive estimate, kept only as a fallback -- overwritten with a
    # real routed value by _attach_real_drive_times when OSRM can reach it. Left unset (None)
    # when a drive constraint isn't even part of this search, so the UI doesn't show a
    # meaningless number for a query that never asked about driving.
    estimated_drive_hours = distance_km / AVG_DRIVE_SPEED_KMH if max_drive_radius_km is not None else None

    return DiscoveredDestination(
        destination=destination,
        distance_km=round(distance_km, 1),
        estimated_flight_hours=round(estimated_flight_hours, 2),
        drive_hours=round(estimated_drive_hours, 2) if estimated_drive_hours is not None else None,
        drive_hours_is_estimate=True,
    )


async def _apply_weather(
    client: httpx.AsyncClient, discovered: DiscoveredDestination, start_date: date, end_date: date
) -> DiscoveredDestination:
    """Weather checks all land on a couple of Open-Meteo hosts, which tolerated the app's
    existing concurrent usage fine (see destination_insights.py) -- kept concurrent across
    candidates via the caller's asyncio.gather.
    """
    weather = await fetch_weather_outlook(client, discovered.destination, start_date, end_date)

    discovered.historical_avg_high_c = weather.historical_avg_high_c
    discovered.historical_avg_low_c = weather.historical_avg_low_c
    discovered.weather_years_used = weather.historical_years_used

    discovered.rationale.append(
        f"~{discovered.distance_km:,.0f} km from your departure point "
        f"(estimated flight time ~{discovered.estimated_flight_hours:.1f}h at typical cruise speed)."
    )
    if weather.error:
        discovered.caveats.append(f"Historical weather unavailable ({weather.error}) -- warmth not verified.")
    else:
        years_label = ", ".join(str(y) for y in weather.historical_years_used) or "recent years"
        discovered.rationale.append(
            f"Historical average for your dates (based on {years_label}): "
            f"highs around {format_temp_c(weather.historical_avg_high_c)}, "
            f"lows around {format_temp_c(weather.historical_avg_low_c)}."
        )
    return discovered


async def _attach_beach_and_poi(
    client: httpx.AsyncClient, constraints: TripConstraints, candidates: list[DiscoveredDestination]
) -> None:
    """Sequential place checks per candidate (http_utils serializes requests to one host anyway),
    with a stop after repeated failures so an outage doesn't stall the whole shortlist.
    """
    consecutive_failures = 0
    for discovered in candidates:
        if consecutive_failures >= _MAX_CONSECUTIVE_PLACE_FAILURES:
            has_beach = restaurant_count = nightlife_count = None
        else:
            has_beach, restaurant_count, nightlife_count = await _check_beach_and_poi_density(
                client, discovered.destination
            )
            all_failed = has_beach is None and restaurant_count is None and nightlife_count is None
            consecutive_failures = consecutive_failures + 1 if all_failed else 0
        discovered.has_beach_nearby = has_beach
        discovered.restaurant_count_nearby = restaurant_count
        discovered.nightlife_count_nearby = nightlife_count

        if has_beach is None:
            discovered.caveats.append("Could not check for nearby beaches (place data unavailable).")
        elif has_beach:
            discovered.rationale.append("A beach is mapped within 6 km (OpenStreetMap data via Geoapify).")
        elif constraints.wants_beach or constraints.wants_pool_resort:
            discovered.caveats.append(
                "No beach mapped within 6 km in OpenStreetMap data -- it may still have one; "
                "OSM coverage varies by region, so this isn't a confirmed absence."
            )

        if restaurant_count is None and nightlife_count is None:
            discovered.caveats.append("Could not check for nearby restaurants/nightlife (place data unavailable).")
        else:
            discovered.rationale.append(
                f"{_count_label(restaurant_count, _RESTAURANT_CAP)} restaurants and "
                f"{_count_label(nightlife_count, _NIGHTLIFE_CAP)} bars/pubs/nightclubs mapped within 3 km "
                "(OpenStreetMap data via Geoapify)."
            )
        if constraints.wants_hiking:
            # No hiking-route data in Geoapify (no such category; a nature-area count was tested
            # 2026-10-19 and scored flat Amsterdam the same as Innsbruck), so this isn't scored.
            discovered.caveats.append("Hiking trails aren't checked automatically -- verify trail access yourself.")

        discovered.score = _score(discovered, constraints)


# Replaces each candidate's estimated drive time with a real OSRM route time, sequentially and
# paced per OSRM's usage policy; candidates too far to drive keep the estimate, with a caveat.
async def _attach_real_drive_times(
    client: httpx.AsyncClient, origin: DestinationCandidate, candidates: list[DiscoveredDestination]
) -> None:
    for discovered in candidates:
        if discovered.distance_km > _OSRM_MAX_DISTANCE_KM:
            discovered.caveats.append(
                f"Drive time is a straight-line estimate ({discovered.drive_hours:.1f}h) -- too far to "
                "attempt real driving directions."
            )
            continue
        drive_hours = await _fetch_real_drive_hours(client, origin, discovered.destination)
        if drive_hours is not None:
            discovered.drive_hours = round(drive_hours, 2)
            discovered.drive_hours_is_estimate = False
            discovered.rationale.append(f"Real routed drive time (via OSRM): ~{drive_hours:.1f}h.")
        else:
            discovered.caveats.append(
                f"Drive time is a straight-line estimate ({discovered.drive_hours:.1f}h) -- no drivable "
                "route could be confirmed (e.g. it may require a ferry/flight leg)."
            )
        await asyncio.sleep(_OSRM_MIN_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_MAX_CANDIDATES_TO_ENRICH = 10
_MAX_RESULTS = 6


async def discover_destinations(
    client: httpx.AsyncClient,
    request_text: str,
    origin_city: str,
    start_date: date,
    end_date: date,
    *,
    max_results: int = _MAX_RESULTS,
    constraints: TripConstraints | None = None,
) -> DestinationDiscoveryResult:
    """`constraints` lets the caller pass what intake already understood (including
    LLM-recovered, verified values -- see request_understanding.py); parsed from the text
    when omitted."""
    constraints = constraints or parse_trip_constraints(request_text)

    origin_matches = await geocode_candidates(client, origin_city, count=1)
    if not origin_matches:
        return DestinationDiscoveryResult(
            constraints=constraints,
            error=f"Could not resolve departure point {origin_city!r} to a real location.",
        )
    origin = origin_matches[0]

    max_flight_radius_km = (
        constraints.max_flight_hours * AVG_FLIGHT_SPEED_KMH if constraints.max_flight_hours else None
    )
    max_drive_radius_km = (
        constraints.max_drive_hours * AVG_DRIVE_SPEED_KMH if constraints.max_drive_hours else None
    )

    candidate_names, candidate_source = await _brainstorm_candidate_names(constraints, origin.label)

    verified = await asyncio.gather(
        *(
            _verify_candidate(client, name, origin, constraints, max_flight_radius_km, max_drive_radius_km)
            for name in candidate_names
        )
    )
    survivors = [c for c in verified if c is not None]
    rejected_distance = len(candidate_names) - len(survivors)

    survivors.sort(key=lambda c: c.distance_km)
    survivors = survivors[:_MAX_CANDIDATES_TO_ENRICH]

    enriched = list(
        await asyncio.gather(*(_apply_weather(client, c, start_date, end_date) for c in survivors))
    )
    await _attach_beach_and_poi(client, constraints, enriched)

    if max_drive_radius_km is not None:
        await _attach_real_drive_times(client, origin, enriched)
        for discovered in enriched:
            if not discovered.drive_hours_is_estimate and discovered.drive_hours is not None:
                within_drive = discovered.drive_hours <= constraints.max_drive_hours
                within_flight = (
                    constraints.max_flight_hours is not None
                    and discovered.estimated_flight_hours is not None
                    and discovered.estimated_flight_hours <= constraints.max_flight_hours * _RADIUS_TOLERANCE
                )
                if not within_drive and not within_flight:
                    discovered.caveats.append(
                        f"Real routed drive time (~{discovered.drive_hours:.1f}h) exceeds your "
                        f"{constraints.max_drive_hours:g}h budget, and estimated flight time also exceeds "
                        "your flight budget -- kept in results but flagged as a stretch."
                    )

    kept: list[DiscoveredDestination] = []
    rejected_weather = 0
    for discovered in enriched:
        if constraints.wants_warm and discovered.historical_avg_high_c is not None:
            if discovered.historical_avg_high_c < WARM_THRESHOLD_C:
                rejected_weather += 1
                continue
        kept.append(discovered)

    kept.sort(key=lambda c: c.score, reverse=True)

    return DestinationDiscoveryResult(
        constraints=constraints,
        origin_label=origin.label,
        candidate_source=candidate_source,
        candidates=kept[:max_results],
        candidates_considered=len(candidate_names),
        candidates_rejected_distance=rejected_distance,
        candidates_rejected_weather=rejected_weather,
    )
