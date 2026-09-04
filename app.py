from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Any

import httpx
import streamlit as st

from config import ENV_FILE_IN_USE, settings
from travel_agent import (
    DEFAULT_TRIP_LENGTH_DAYS,
    DestinationCandidate,
    UserTripRequest,
    build_itineraries,
    check_state_department_advisory,
    discover_destinations,
    fetch_seasonal_activities,
    fetch_weather_outlook,
    parse_travel_start_date,
    resolve_destination_candidates,
)
from travel_agent.destination_discovery import parse_trip_constraints
from travel_agent.schemas import (
    ActivitySearchResult,
    AdvisoryCheck,
    DestinationDiscoveryResult,
    WeatherOutlook,
)

logger = logging.getLogger(__name__)


# Helper to parse a single markdown table row into a list of cells
def _parse_markdown_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


# Extracts the 'Deterministic Scoring Table' from the agent's markdown response
# and converts it into a list of structured dictionaries for display/analysis.
def _extract_scoring_table(markdown_text: str) -> list[dict[str, Any]]:
    lines = markdown_text.splitlines()
    heading_index = -1

    for index, line in enumerate(lines):
        if re.search(r"deterministic\s+scoring\s+table", line, flags=re.IGNORECASE):
            heading_index = index
            break

    if heading_index == -1:
        return []

    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.strip().startswith("|"):
            table_lines.append(line)
            continue
        if table_lines:
            break

    if len(table_lines) < 2:
        return []

    headers = _parse_markdown_row(table_lines[0])
    if not headers:
        return []

    records: list[dict[str, Any]] = []
    numeric_candidates = {
        "rating_quality",
        "review_confidence",
        "safety",
        "budget_fit",
        "final_score",
    }

    for row in table_lines[1:]:
        if re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$", row.strip()):
            continue

        values = _parse_markdown_row(row)
        if len(values) != len(headers):
            continue

        item: dict[str, Any] = {}
        for header, value in zip(headers, values):
            key = header.strip().lower().replace(" ", "_")
            cleaned_value = value.strip()

            if key in numeric_candidates:
                number_match = re.search(r"-?\d+(?:\.\d+)?", cleaned_value)
                item[header] = float(number_match.group()) if number_match else cleaned_value
            else:
                item[header] = cleaned_value

        records.append(item)

    return records


async def _resolve_intake(request_text: str) -> tuple[list[DestinationCandidate], date | None]:
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        candidates = await resolve_destination_candidates(client, request_text)
    parsed_date = parse_travel_start_date(request_text)
    return candidates, parsed_date


# Used when no specific destination could be named in the request but an origin was given --
# suggests real, independently-verified candidate destinations instead of a dead-end warning.
#
# A short, explicit *connect* timeout matters a lot here specifically: discovery fans out to
# public, best-effort services (Overpass, OSRM) that can occasionally become slow/unreachable.
# A flat 45s timeout (connect+read combined, as a bare `timeout=45.0` sets) turns one bad
# connection attempt into a 45s stall, repeated across the retry budget and every candidate in
# the shortlist -- verified live to blow past 20 minutes total. Failing a connection attempt
# fast lets the retry/backoff and per-candidate pacing already in destination_discovery.py do
# their job without compounding into a multi-minute hang.
_DISCOVERY_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


async def _run_discovery(request_text: str, origin_city: str, start_date: date, end_date: date):
    async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT) as client:
        return await discover_destinations(client, request_text, origin_city, start_date, end_date)


# Runs the three destination-selection background checks (State Department advisory,
# weather outlook, seasonal activities) concurrently as soon as a destination + travel
# window are confirmed -- ahead of, and independent from, full itinerary generation.
#
# Uses the same short-connect-timeout as _DISCOVERY_TIMEOUT rather than a flat
# settings.request_timeout_seconds: fetch_seasonal_activities calls the same public
# Overpass API discovery does, and a flat timeout has the identical multi-minute-hang
# failure mode (verified live) when that service is slow/unreachable.
async def _gather_destination_insights(
    destination: DestinationCandidate, start_date: date, end_date: date
) -> tuple[AdvisoryCheck, WeatherOutlook, ActivitySearchResult]:
    async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT) as client:
        return await asyncio.gather(
            check_state_department_advisory(client, destination),
            fetch_weather_outlook(client, destination, start_date, end_date),
            fetch_seasonal_activities(client, destination, start_date, end_date),
        )


def _reset() -> None:
    for key in (
        "candidates",
        "parsed_date",
        "pending",
        "result",
        "insight_key",
        "insight_advisory",
        "insight_weather",
        "insight_activities",
        "discovery_result",
        "discovery_start_date",
        "discovery_end_date",
    ):
        st.session_state.pop(key, None)
    st.session_state.stage = "intake"


# Renders the parsed constraints + ranked candidates from a destination-discovery run
# (triggered when the traveler didn't name a specific place), so the user can see exactly
# what the app understood before picking one to carry into the normal confirm/itinerary flow.
def _render_discovery_result(result: DestinationDiscoveryResult) -> None:
    c = result.constraints
    with st.expander("What we understood from your description", expanded=False):
        st.write(
            f"Budget: {f'${c.budget_usd:,.0f}' if c.budget_usd is not None else 'not specified'} "
            "(checked later against real flight/hotel pricing once you pick a destination -- "
            "not used to filter suggestions below) | "
            f"Trip length: {f'{c.trip_length_days} days' if c.trip_length_days is not None else 'not specified'} | "
            f"Max flight time: {f'{c.max_flight_hours:g}h' if c.max_flight_hours is not None else 'not specified'} | "
            f"Max drive time: {f'{c.max_drive_hours:g}h' if c.max_drive_hours is not None else 'not specified'}"
        )
        st.write(
            f"Wants warm weather: {c.wants_warm} | Wants beach: {c.wants_beach} | "
            f"Wants pool/resort: {c.wants_pool_resort} | Wants food scene: {c.wants_food_scene} | "
            f"Wants entertainment/nightlife: {c.wants_entertainment} | "
            f"Wants to skip cultural/sightseeing: {c.avoid_cultural}"
        )
        st.caption(
            f"Departure point resolved to: {result.origin_label}. Candidate names were "
            f"{result.candidate_source}. Considered {result.candidates_considered} candidates -- "
            f"{result.candidates_rejected_distance} were outside your travel-time budget, "
            f"{result.candidates_rejected_weather} didn't meet your warmth preference."
        )

    if not result.candidates:
        st.warning(
            "No destinations survived both the travel-time and warmth checks. Try loosening your "
            "flight/drive time budget, dropping the warm-weather requirement, or naming a region "
            "you'd consider."
        )
        return

    st.write("Here are destinations that matched, ranked by how well they fit what you described:")
    labels = []
    for d in result.candidates:
        drive_note = (
            f", ~{d.drive_hours:.1f}h drive ({'real route' if not d.drive_hours_is_estimate else 'estimated'})"
            if d.drive_hours is not None
            else ""
        )
        labels.append(f"{d.destination.label} -- ~{d.estimated_flight_hours:.1f}h flight{drive_note}")

    selected_label = st.radio(
        "Suggested destinations", labels, index=0, label_visibility="collapsed", key="discovery_pick"
    )
    selected = result.candidates[labels.index(selected_label)]

    for reason in selected.rationale:
        st.write(f"- {reason}")
    for caveat in selected.caveats:
        st.caption(f"Note: {caveat}")

    if st.button("Continue with this destination", type="primary"):
        st.session_state.candidates = [selected.destination]
        st.session_state.parsed_date = None
        st.session_state.stage = "confirm"
        st.rerun()


# Renders the three destination-selection background checks (safety advisory, weather
# outlook, seasonal activities) directly on the confirm screen, as soon as a destination
# and travel window are chosen -- independent of the "Generate itinerary options" button.
def _render_destination_insights(
    destination: DestinationCandidate, advisory: AdvisoryCheck, weather: WeatherOutlook, activities: ActivitySearchResult
) -> None:
    st.subheader("Automatic destination checks")

    if advisory.error:
        st.caption(f"U.S. Department of State advisory check unavailable ({advisory.error}).")
    elif advisory.active_warning:
        st.error(
            f"**Active U.S. Department of State advisory for {destination.country}:** "
            f"{advisory.title} (updated {advisory.updated.isoformat() if advisory.updated else 'recently'}, "
            f"within the last 6 months).\n\n{advisory.summary_snippet or ''}\n\n"
            f"Full advisory: {advisory.link or advisory.source_url}"
        )
    elif advisory.found:
        st.success(
            f"No Level 3/4 U.S. Department of State advisory has been issued or updated for "
            f"{destination.country} in the last 6 months. Current listing: **{advisory.title}** "
            f"(updated {advisory.updated.isoformat() if advisory.updated else 'unknown date'})."
        )
    else:
        st.caption(
            f"No matching U.S. Department of State advisory entry was found for {destination.country}. "
            f"Verify current guidance at {advisory.source_url}."
        )

    if weather.error:
        st.caption(f"Weather outlook unavailable ({weather.error}).")
    else:
        years_label = ", ".join(str(y) for y in weather.historical_years_used) or "recent years"
        st.info(
            f"Historical average for your travel window (based on {years_label}): "
            f"highs around {weather.historical_avg_high_c}°C, lows around {weather.historical_avg_low_c}°C."
        )
        if weather.is_near_term and weather.near_term_forecast_summary:
            st.info(weather.near_term_forecast_summary)

    if activities.error:
        st.caption(f"Activity/sightseeing search unavailable ({activities.error}).")
    elif activities.activities:
        st.write("Nearby activities & sightseeing (auto-checked against your travel dates):")
        st.dataframe(
            [
                {
                    "Name": a.name,
                    "Category": a.category,
                    "Appropriate for your dates?": a.appropriateness_comment,
                    "Listed hours": a.raw_opening_hours or "Not listed",
                }
                for a in activities.activities
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(activities.source_note)
    else:
        st.caption("No named attractions were found nearby in OpenStreetMap data for this destination.")


# UI Configuration and Page Setup
st.set_page_config(page_title="Agentic Travel Concierge", page_icon="🧳", layout="wide")

st.title("🧳 Agentic Travel Concierge")
st.write(
    "Describe your ideal trip (budget, climate, activities, pace, dates, and preferences). "
    "The app identifies matching destinations for you to confirm, then generates multiple "
    "luxury-within-budget itinerary options with safety-aware guidance."
)
if ENV_FILE_IN_USE:
    st.caption(f"ENV file in use: {ENV_FILE_IN_USE}")
else:
    st.caption("ENV file in use: default process environment (no env file found)")

if "stage" not in st.session_state:
    st.session_state.stage = "intake"

# --- Stage 1: free-text intake -----------------------------------------------------
if st.session_state.stage == "intake":
    with st.form("trip_form"):
        request_text = st.text_area(
            "Travel request",
            height=180,
            placeholder=(
                "Example: Plan a 7-day honeymoon in Japan from Seattle for 2 travelers with a total budget under $7000, "
                "moderate weather, food tours, ryokan stay, and low-risk neighborhoods."
            ),
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            origin_city = st.text_input(
                "Origin city (optional for flight search; required if you want us to suggest "
                "destinations instead of naming one)"
            )
        with col2:
            trip_length_days = st.number_input("Trip length (days, optional)", min_value=0, max_value=60, value=0)
        with col3:
            traveler_count = st.number_input("Number of travelers", min_value=1, max_value=20, value=1)

        submitted = st.form_submit_button("Find destinations")

    if submitted:
        if len(request_text.strip()) < 10:
            st.warning("Please provide a more detailed request.")
        else:
            candidates: list[DestinationCandidate] = []
            parsed_date: date | None = None
            resolution_failed = False
            with st.spinner("Identifying destination options..."):
                try:
                    candidates, parsed_date = asyncio.run(_resolve_intake(request_text.strip()))
                except Exception:
                    logger.exception(
                        "Destination resolution failed (request length=%d chars)", len(request_text)
                    )
                    resolution_failed = True

            if resolution_failed:
                st.error(
                    "We couldn't look up destinations for that request right now (an upstream service may be "
                    "unavailable or rate-limited). Please try again in a moment."
                )
            elif not candidates and not origin_city.strip():
                st.warning(
                    "We couldn't identify a specific destination in that request. Either name a city, region, "
                    "or country explicitly (e.g. \"...trip to Lisbon...\"), or fill in a departure city above "
                    "so we can suggest destinations based on travel time from there."
                )
            elif not candidates:
                # No place was named, but we have a departure point -- try to suggest destinations
                # that match the described vibe/constraints instead of a dead end.
                text_constraints = parse_trip_constraints(request_text.strip())
                trip_length_days = (
                    trip_length_days if trip_length_days > 0
                    else text_constraints.trip_length_days or DEFAULT_TRIP_LENGTH_DAYS
                )
                discovery_start = parsed_date or date.today()
                discovery_end = discovery_start + timedelta(days=trip_length_days)

                discovery_result = None
                discovery_failed = False
                with st.spinner(
                    "No specific destination was named -- searching for destinations that match what "
                    "you described (this checks real distance, weather, and points-of-interest data, "
                    "so it can take a little while)..."
                ):
                    try:
                        discovery_result = asyncio.run(
                            _run_discovery(request_text.strip(), origin_city.strip(), discovery_start, discovery_end)
                        )
                    except Exception:
                        logger.exception(
                            "Destination discovery failed (request length=%d chars)", len(request_text)
                        )
                        discovery_failed = True

                if discovery_failed or discovery_result is None:
                    st.error(
                        "We couldn't search for matching destinations right now (an upstream service may be "
                        "unavailable or rate-limited). Please try again in a moment."
                    )
                elif discovery_result.error:
                    st.error(discovery_result.error)
                else:
                    st.session_state.stage = "discover"
                    st.session_state.discovery_result = discovery_result
                    st.session_state.pending = {
                        "request_text": request_text.strip(),
                        "origin_city": origin_city.strip() or None,
                        "trip_length_days": trip_length_days,
                        "traveler_count": traveler_count,
                    }
                    st.rerun()
            else:
                st.session_state.stage = "confirm"
                st.session_state.candidates = candidates
                st.session_state.parsed_date = parsed_date
                st.session_state.pending = {
                    "request_text": request_text.strip(),
                    "origin_city": origin_city.strip() or None,
                    "trip_length_days": trip_length_days if trip_length_days > 0 else None,
                    "traveler_count": traveler_count,
                }
                st.rerun()

# --- Stage 1b: no destination named -- suggest some based on described constraints -----
elif st.session_state.stage == "discover":
    st.subheader("Suggested destinations")
    st.caption(
        "You didn't name a specific place, so we searched for destinations matching your description -- "
        "every fact shown below (distance, weather, beach/POI presence) comes from a live, independent check, "
        "not a guess."
    )
    _render_discovery_result(st.session_state.discovery_result)

    if st.button("Back"):
        _reset()
        st.rerun()

# --- Stage 2: confirm destination + date -------------------------------------------
elif st.session_state.stage == "confirm":
    candidates: list[DestinationCandidate] = st.session_state.candidates
    parsed_date: date | None = st.session_state.parsed_date

    st.subheader("Which destination did you mean?")
    st.caption("Picked from your request -- confirm one before we search flights and hotels for it.")
    selected_label = st.radio("Destination", [c.label for c in candidates], index=0, label_visibility="collapsed")
    selected_destination = next(c for c in candidates if c.label == selected_label)

    st.subheader("When are you traveling?")
    if parsed_date is not None:
        st.caption(f"Detected travel start date from your request: {parsed_date.isoformat()}")
        use_detected_date = st.checkbox("Use this date", value=True)
        travel_start_date = parsed_date if use_detected_date else st.date_input(
            "Travel start date", min_value=date.today()
        )
    else:
        st.info("We couldn't find a clear travel start date in your request -- please choose one.")
        travel_start_date = st.date_input("Travel start date", min_value=date.today())

    trip_length_days = st.session_state.pending.get("trip_length_days") or DEFAULT_TRIP_LENGTH_DAYS
    insight_end_date = travel_start_date + timedelta(days=trip_length_days)
    insight_key = (selected_destination.label, travel_start_date.isoformat(), insight_end_date.isoformat())

    if st.session_state.get("insight_key") != insight_key:
        with st.spinner("Checking travel advisories, weather outlook, and nearby activities..."):
            try:
                advisory, weather, activities = asyncio.run(
                    _gather_destination_insights(selected_destination, travel_start_date, insight_end_date)
                )
            except Exception:
                logger.exception("Destination insight checks failed for a selected destination")
                advisory, weather, activities = (
                    AdvisoryCheck(error="an unexpected error occurred while checking"),
                    WeatherOutlook(error="an unexpected error occurred while checking"),
                    ActivitySearchResult(error="an unexpected error occurred while checking"),
                )
        st.session_state.insight_key = insight_key
        st.session_state.insight_advisory = advisory
        st.session_state.insight_weather = weather
        st.session_state.insight_activities = activities

    _render_destination_insights(
        selected_destination,
        st.session_state.insight_advisory,
        st.session_state.insight_weather,
        st.session_state.insight_activities,
    )

    back_col, generate_col = st.columns([1, 3])
    with back_col:
        if st.button("Back"):
            _reset()
            st.rerun()
    with generate_col:
        generate = st.button("Generate itinerary options", type="primary")

    if generate:
        pending = st.session_state.pending
        request = UserTripRequest(
            request_text=pending["request_text"],
            origin_city=pending["origin_city"],
            trip_length_days=pending["trip_length_days"],
            traveler_count=pending["traveler_count"],
            travel_start_date=travel_start_date,
        )
        result = None
        with st.spinner("Planning your trip with the agentic workflow..."):
            try:
                result = asyncio.run(build_itineraries(request, selected_destination))
            except Exception:
                # Don't log the raw request text: it's free-form user input that
                # commonly includes names, travel dates, and budget — logging it
                # verbatim at exception level puts PII into whatever aggregates
                # server logs. Length is enough to correlate/debug from.
                logger.exception(
                    "build_itineraries failed (request length=%d chars)", len(pending["request_text"])
                )
                st.error(
                    "We couldn't generate itinerary options this time (an upstream service may be "
                    "unavailable or rate-limited). Please try again in a moment."
                )

        if result is not None:
            st.session_state.result = result
            st.session_state.stage = "results"
            st.rerun()

# --- Stage 3: results ----------------------------------------------------------------
elif st.session_state.stage == "results":
    result = st.session_state.result

    if st.button("Start over"):
        _reset()
        st.rerun()

    st.subheader(f"Destination focus: {result['destination']}")
    st.markdown(result["itinerary_markdown"])

    score_rows = _extract_scoring_table(result["itinerary_markdown"])
    if score_rows:
        st.subheader("Deterministic Score Breakdown")
        st.dataframe(score_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Scoring table not available for this run.")

    st.divider()
    st.caption("Safety grounding")
    st.write(result["safety_summary"])
    st.write(f"Primary advisory source: {result['advisory_source_url']}")
