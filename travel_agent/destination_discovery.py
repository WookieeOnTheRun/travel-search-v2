from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import date
from textwrap import dedent

import httpx
from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

from .destination_insights import fetch_weather_outlook
from .http_utils import describe_error, get_with_retry, post_with_retry
from .itinerary_agent import configure_model_env
from .location_resolution import geocode_candidates
from .schemas import (
    DestinationCandidate,
    DestinationDiscoveryResult,
    DiscoveredDestination,
    TripConstraints,
)

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


def _to_number(token: str) -> float:
    token = token.lower()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    return float(token)


def _extract_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _to_number(match.group(1))


_BUDGET_PATTERNS = [
    r"\$\s?([\d,]+(?:\.\d+)?)",
    r"([\d,]+(?:\.\d+)?)\s*(?:usd|dollars)\b",
]
_TRIP_LENGTH_PATTERN = rf"\b({_NUMBER_PATTERN})[\s-]*day"
_FLIGHT_HOURS_PATTERN = rf"\b({_NUMBER_PATTERN})[\s-]*hour[s]?\s*flight"
_DRIVE_HOURS_PATTERN = rf"\b({_NUMBER_PATTERN})[\s-]*hour[s]?\s*drive"

_WARM_PATTERN = re.compile(r"\bwarm\b|\bhot\b|\btropical\b|\bsunny\b", re.IGNORECASE)
_BEACH_PATTERN = re.compile(r"\bbeach(?:es)?\b", re.IGNORECASE)
_POOL_RESORT_PATTERN = re.compile(r"\bpool\b|\bswim[\s-]?up\b|\bresort\b", re.IGNORECASE)
_FOOD_PATTERN = re.compile(r"food scene|\bculinary\b|\bcuisine\b|\bfoodie\b|great food", re.IGNORECASE)
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


def parse_trip_constraints(request_text: str) -> TripConstraints:
    """Deterministically extracts budget/duration/travel-time/vibe constraints from free
    text. Every field is None/False unless the text actually said so -- this never invents
    a number or preference the traveler didn't state.
    """
    budget = None
    for pattern in _BUDGET_PATTERNS:
        match = re.search(pattern, request_text, re.IGNORECASE)
        if match:
            budget = float(match.group(1).replace(",", ""))
            break

    trip_length = _extract_number(_TRIP_LENGTH_PATTERN, request_text)
    flight_hours = _extract_number(_FLIGHT_HOURS_PATTERN, request_text)
    drive_hours = _extract_number(_DRIVE_HOURS_PATTERN, request_text)

    return TripConstraints(
        raw_text=request_text,
        budget_usd=budget,
        trip_length_days=int(trip_length) if trip_length is not None else None,
        max_flight_hours=flight_hours,
        max_drive_hours=drive_hours,
        wants_warm=bool(_WARM_PATTERN.search(request_text)),
        wants_beach=bool(_BEACH_PATTERN.search(request_text)),
        wants_pool_resort=bool(_POOL_RESORT_PATTERN.search(request_text)),
        wants_food_scene=bool(_FOOD_PATTERN.search(request_text)),
        wants_entertainment=bool(_ENTERTAINMENT_PATTERN.search(request_text)),
        avoid_cultural=bool(_AVOID_CULTURAL_PATTERN.search(request_text)),
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

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


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
    return names


async def _brainstorm_candidate_names(constraints: TripConstraints, origin_label: str) -> tuple[list[str], str]:
    prompt = dedent(
        f"""
        Traveler's departure point: {origin_label}
        Traveler's request (verbatim): {constraints.raw_text}
        Parsed preferences: warm={constraints.wants_warm}, beach={constraints.wants_beach},
        pool/resort={constraints.wants_pool_resort}, food scene={constraints.wants_food_scene},
        entertainment/nightlife={constraints.wants_entertainment},
        avoid cultural/sightseeing={constraints.avoid_cultural}

        List candidate destination names now, one per line, "City, Country" format only.
        """
    ).strip()

    try:
        configure_model_env()
        client = OllamaChatClient()
        agent = Agent(name="DestinationIdeaAgent", client=client, instructions=_DESTINATION_IDEA_AGENT_INSTRUCTIONS)
        response = await agent.run(prompt)
        names = _parse_name_list(str(response))
        if names:
            return names, "brainstormed by the configured travel-idea model, then independently verified"
        logger.warning("Destination idea agent returned no parseable names; using fallback list")
    except Exception as ex:
        logger.warning("Destination idea agent unavailable, using fallback list: %s", ex)

    return list(_FALLBACK_CANDIDATE_NAMES), "drawn from a static list of well-known warm/coastal destinations, then independently verified"


# ---------------------------------------------------------------------------
# Step 4: independently verify every candidate against real data.
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_BEACH_CHECK_RADIUS_M = 6000
_POI_DENSITY_RADIUS_M = 3000
# The public Overpass instance's usage policy asks for non-parallel, non-heavy use. Live
# testing (2026-09-03) showed that even fully serialized-but-rapid-fire requests from this
# app tripped its rate limiting; this pacing (used in _attach_beach_and_poi) fixed it.
_OVERPASS_MIN_INTERVAL_SECONDS = 1.1
_OSRM_URL = "http://router.project-osrm.org/route/v1/driving"
# The public OSRM demo server's usage policy caps this at 1 request/second and forbids
# heavy/scraped use -- real drive times are only fetched sequentially, with this pacing,
# and only for the bounded shortlist that survives the cheap distance pre-filter below.
_OSRM_MIN_INTERVAL_SECONDS = 1.05
_OSRM_MAX_DISTANCE_KM = 3500  # beyond this, driving isn't realistic; skip the routing call


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


async def _check_beach_and_poi_density(
    client: httpx.AsyncClient, dest: DestinationCandidate
) -> tuple[bool | None, int | None, int | None]:
    """One combined Overpass query per candidate (beach presence + restaurant count +
    nightlife count) instead of separate calls -- halves the request volume against the
    shared public Overpass instance, which matters given its stated no-heavy-usage policy.
    """
    query = (
        f"[out:json][timeout:20];"
        f"("
        f"node(around:{_BEACH_CHECK_RADIUS_M},{dest.latitude},{dest.longitude})[natural=beach];"
        f"way(around:{_BEACH_CHECK_RADIUS_M},{dest.latitude},{dest.longitude})[natural=beach];"
        f");"
        f"out count;"
        f'(node(around:{_POI_DENSITY_RADIUS_M},{dest.latitude},{dest.longitude})[amenity=restaurant];);'
        f"out count;"
        f'(node(around:{_POI_DENSITY_RADIUS_M},{dest.latitude},{dest.longitude})[amenity~"^(bar|nightclub|pub)$"];);'
        f"out count;"
    )
    try:
        # max_attempts=2: same reasoning as the OSRM call above -- best-effort signal, bound
        # the worst case rather than retry a third time against a shared public server.
        response = await post_with_retry(
            client, _OVERPASS_URL, data={"data": query},
            headers={"User-Agent": "agentic-travel-concierge/1.0 (destination discovery)"},
            max_attempts=2,
        )
        elements = response.json().get("elements", [])
        counts = [int(e.get("tags", {}).get("total", 0)) for e in elements if e.get("type") == "count"]
        has_beach = counts[0] > 0 if len(counts) > 0 else None
        restaurant_count = counts[1] if len(counts) > 1 else None
        nightlife_count = counts[2] if len(counts) > 2 else None
        return has_beach, restaurant_count, nightlife_count
    except Exception as ex:
        logger.warning("Overpass beach/POI check failed for %s: %s", dest.name, describe_error(ex))
        return None, None, None


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
            f"highs around {weather.historical_avg_high_c}°C, lows around {weather.historical_avg_low_c}°C."
        )
    return discovered


async def _attach_beach_and_poi(
    client: httpx.AsyncClient, constraints: TripConstraints, candidates: list[DiscoveredDestination]
) -> None:
    """Sequential, paced Overpass calls (one combined query per candidate) rather than firing
    the whole shortlist concurrently -- live testing showed even serialized-but-rapid-fire
    requests trip the shared public Overpass instance's rate limiting, whose usage policy
    explicitly asks for non-parallel, non-heavy use.
    """
    for discovered in candidates:
        has_beach, restaurant_count, nightlife_count = await _check_beach_and_poi_density(client, discovered.destination)
        discovered.has_beach_nearby = has_beach
        discovered.restaurant_count_nearby = restaurant_count
        discovered.nightlife_count_nearby = nightlife_count

        if has_beach is None:
            discovered.caveats.append("Could not check OpenStreetMap for nearby beaches (data source unavailable).")
        elif has_beach:
            discovered.rationale.append("A beach is tagged within 6 km in OpenStreetMap data.")
        elif constraints.wants_beach or constraints.wants_pool_resort:
            discovered.caveats.append(
                "No beach tagged within 6 km in OpenStreetMap data -- it may still have one; "
                "OSM coverage varies by region, so this isn't a confirmed absence."
            )

        if restaurant_count is None and nightlife_count is None:
            discovered.caveats.append(
                "Could not check OpenStreetMap for nearby restaurants/nightlife (data source unavailable)."
            )
        else:
            discovered.rationale.append(
                f"{restaurant_count or 0} restaurants and {nightlife_count or 0} bars/nightlife venues "
                f"found within 3 km in OpenStreetMap data."
            )

        discovered.score = _score(discovered, constraints)
        await asyncio.sleep(_OVERPASS_MIN_INTERVAL_SECONDS)


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
) -> DestinationDiscoveryResult:
    constraints = parse_trip_constraints(request_text)

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
