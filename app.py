from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import streamlit as st

from config import settings
from travel_agent import (
    DEFAULT_TRIP_LENGTH_DAYS,
    DestinationCandidate,
    NearbyAirport,
    UserTripRequest,
    build_itineraries,
    check_state_department_advisory,
    discover_destinations,
    fetch_seasonal_activities,
    fetch_weather_outlook,
    find_nearest_airports,
    month_window_start,
    understand_trip_request,
)
from travel_agent.geoapify import ATTRIBUTION as GEOAPIFY_ATTRIBUTION
from travel_agent.http_utils import DEFAULT_HTTP_TIMEOUT, describe_error
from travel_agent.persistence import (
    DuplicateItineraryNameError,
    Itinerary,
    ItineraryNotFoundError,
    MAX_ITINERARY_NAME_LENGTH,
    SearchRun,
    SearchRunNotFoundError,
    SessionStore,
    is_valid_session_id,
    new_session_id,
)
from travel_agent.schemas import (
    ActivitySearchResult,
    AdvisoryCheck,
    DestinationDiscoveryResult,
    ParsedTripRequest,
    TripConstraints,
    WeatherOutlook,
)
from travel_agent.units import format_temp_c

logger = logging.getLogger(__name__)

# How many nearby airports the confirm screen lists for a destination.
_NEAREST_AIRPORTS_COUNT = 5

# Input size limits. The request text is sent verbatim to every LLM agent (several prompts per
# search) and into upstream API query params, so its length directly drives cost and latency.
# Streamlit enforces max_chars in the browser; the same limits are re-checked server-side on
# submit, since the browser is not a trust boundary.
_MAX_REQUEST_CHARS = 2000
_MAX_ORIGIN_CHARS = 100

# URL query parameter carrying this browser's anonymous session id. Streamlit can't set
# cookies, and its own session ends on every page refresh, so the id rides in the URL to
# survive refreshes. Anyone holding the full URL can see that session's itineraries -- fine
# for testing, but don't share the link.
_SESSION_QUERY_PARAM = "sid"
# Save-form choice meaning "create a new itinerary with the typed name".
_NEW_ITINERARY_OPTION = "➕ New itinerary…"


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


# Parses the intake request into structured trip details and geocoded destination candidates.
async def _resolve_intake(request_text: str) -> ParsedTripRequest:
    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
        return await understand_trip_request(client, request_text)


# The start date to pre-fill from the request, and how it was derived: an explicit date wins;
# a month-only mention ("a week in March") falls back to the 1st of that month.
def _start_date_from_request(parsed: ParsedTripRequest) -> tuple[date | None, str | None]:
    if parsed.start_date is not None:
        return parsed.start_date, f"Detected travel start date from your request: {parsed.start_date.isoformat()}"
    if parsed.constraints.travel_month is not None:
        start = month_window_start(parsed.constraints.travel_month)
        return start, (
            f"Your request names {start.strftime('%B')} but no specific day -- using {start.isoformat()} "
            "as the start date; change it if you'd like."
        )
    return None, None


# Used when no specific destination could be named in the request but an origin was given --
# suggests real, independently-verified candidate destinations instead of a dead-end warning.
#
# A short, explicit *connect* timeout matters a lot here specifically: discovery fans out to
# public, best-effort services (OSRM, Geoapify) that can occasionally become slow/unreachable.
# A flat 45s timeout (connect+read combined, as a bare `timeout=45.0` sets) turns one bad
# connection attempt into a 45s stall, repeated across the retry budget and every candidate in
# the shortlist -- verified live to blow past 20 minutes total. Failing a connection attempt
# fast lets the retry/backoff and per-candidate pacing already in destination_discovery.py do
# their job without compounding into a multi-minute hang.
_DISCOVERY_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


# Runs destination discovery (see the timeout note above) for a request that named no place.
async def _run_discovery(
    request_text: str, origin_city: str, start_date: date, end_date: date, constraints: TripConstraints
):
    async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT) as client:
        return await discover_destinations(
            client, request_text, origin_city, start_date, end_date, constraints=constraints
        )


# Runs a batch of named coroutines concurrently and calls on_progress(label) as each one
# completes (completion order, not the order given) -- this is what drives the live
# per-check status bar instead of one opaque spinner for the whole concurrent batch. Each
# task's own exception (if it somehow raises past its internal error handling) is isolated
# to that task rather than failing the whole batch, so one broken check doesn't blank out
# checks that already succeeded.
async def _gather_tracked(jobs: dict[str, Any], on_progress) -> dict[str, Any]:
    async def _tag(label: str, coro) -> tuple[str, Any]:
        try:
            return label, await coro
        except Exception as ex:
            logger.exception("Destination check %r failed", label)
            return label, ex

    results: dict[str, Any] = {}
    for finished in asyncio.as_completed([_tag(label, coro) for label, coro in jobs.items()]):
        label, result = await finished
        results[label] = result
        if on_progress:
            on_progress(label)
    return results


# Runs the destination-selection background checks (State Department advisory, weather
# outlook, nearest airports, and -- only when requested -- seasonal activities/sightseeing)
# concurrently as soon as a destination + travel window are confirmed -- ahead of, and
# independent from, full itinerary generation. Nearest-airport resolution runs unconditionally
# here specifically because it needs to happen as early as possible after a destination is
# selected, per the "closest airports" requirement -- not deferred until flight search.
#
# Uses the same short-connect-timeout as _DISCOVERY_TIMEOUT rather than a flat
# settings.request_timeout_seconds: fetch_seasonal_activities calls the same public
# place-data API discovery does, and a flat timeout has the identical multi-minute-hang
# failure mode (verified live) when that service is slow/unreachable.
async def _gather_destination_insights(
    destination: DestinationCandidate,
    start_date: date,
    end_date: date,
    *,
    include_activities: bool,
    request_text: str = "",
    on_progress=None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT) as client:
        jobs: dict[str, Any] = {
            "advisory": check_state_department_advisory(client, destination),
            "weather": fetch_weather_outlook(client, destination, start_date, end_date),
            "airports": find_nearest_airports(client, destination.latitude, destination.longitude, max_results=_NEAREST_AIRPORTS_COUNT),
        }
        if include_activities:
            jobs["activities"] = fetch_seasonal_activities(client, destination, start_date, end_date, request_text)

        results = await _gather_tracked(jobs, on_progress)

    return {
        # describe_error, not str(): these messages are shown on screen, and a raw HTTP error
        # embeds the full request URL including its query string.
        "advisory": results["advisory"] if not isinstance(results["advisory"], Exception) else AdvisoryCheck(error=describe_error(results["advisory"])),
        "weather": results["weather"] if not isinstance(results["weather"], Exception) else WeatherOutlook(error=describe_error(results["weather"])),
        "airports": results["airports"] if not isinstance(results["airports"], Exception) else [],
        "activities": (
            (results["activities"] if not isinstance(results["activities"], Exception) else ActivitySearchResult(error=describe_error(results["activities"])))
            if include_activities
            else ActivitySearchResult()
        ),
    }


# Session-state keys that outlive a search. Everything else belongs to the in-progress search
# (request, parsed constraints, candidates, insights, result, widget selections, itinerary-view
# return state) and is discarded by _reset, so a new search can't inherit any of it -- including
# keys added later that nobody remembers to list here. Saved itineraries live in the
# SessionStore, not session state, so they're unaffected.
_SESSION_KEYS_KEPT_ON_RESET = (
    "session_id",
    # Keeps increasing across searches so each new result gets a save form it has never had.
    "result_seq",
    "flash",
)


# Starts a brand-new search: drops all context from the previous one.
def _reset() -> None:
    for key in list(st.session_state.keys()):
        if key not in _SESSION_KEYS_KEPT_ON_RESET:
            del st.session_state[key]
    st.session_state.stage = "intake"


# Starts a new search pre-filled from a saved search, scoped to the itinerary it was saved in:
# the intake form opens with the saved request's values, and the save form defaults to that
# itinerary, so a refined result lands back where the original lives. Like "Start over", it
# first discards whatever search was in progress.
def _start_refine(itinerary: Itinerary, context: dict[str, Any]) -> None:
    _reset()
    st.session_state.refine = {
        "itinerary_id": itinerary.itinerary_id,
        "itinerary_name": itinerary.name,
        "request": context.get("request") or {},
        "show_activities": bool(context.get("show_activities", False)),
    }


# The saved search's start date, if it's still usable (today or later), for a refined search
# whose edited request text no longer names a date.
def _refine_start_date(refine: dict[str, Any] | None) -> date | None:
    raw = ((refine or {}).get("request") or {}).get("travel_start_date")
    try:
        saved = date.fromisoformat(raw) if raw else None
    except ValueError:
        return None
    return saved if saved and saved >= date.today() else None


# Shows which trip details were understood from the request and where each came from
# (deterministic parsing or a quote-verified LLM extraction), plus any parsing notes.
def _render_request_understanding(parsed: ParsedTripRequest | None) -> None:
    if parsed is None:
        return
    with st.expander("How we read your request", expanded=False):
        sources = ", ".join(f"{field} ({source})" for field, source in parsed.field_sources.items()) or "nothing"
        st.write(f"Understood: {sources}.")
        st.caption(
            f"AI-assisted parsing: {parsed.llm_status}. Values marked (llm) were only accepted after "
            "being matched to a verbatim quote from your request (and, for places, a real geocoding match)."
        )
        for note in parsed.notes:
            st.caption(f"- {note}")


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
            f"Max drive time: {f'{c.max_drive_hours:g}h' if c.max_drive_hours is not None else 'not specified'} | "
            f"Travel month: {date(2000, c.travel_month, 1).strftime('%B') if c.travel_month else 'not specified'}"
        )
        st.write(
            f"Wants warm weather: {c.wants_warm} | Wants beach: {c.wants_beach} | "
            f"Wants pool/resort: {c.wants_pool_resort} | Wants food scene: {c.wants_food_scene} | "
            f"Wants entertainment/nightlife: {c.wants_entertainment} | Wants hiking: {c.wants_hiking} | "
            f"Wants low cost: {c.wants_low_cost} (not verified against prices at this stage) | "
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
    # Required by Geoapify's free plan wherever its place data is shown.
    st.caption(GEOAPIFY_ATTRIBUTION)

    if st.button("Continue with this destination", type="primary"):
        st.session_state.candidates = [selected.destination]
        # Carry the discovery window's start (e.g. the month the traveler named) into confirm.
        st.session_state.parsed_date = st.session_state.get("discovery_start_date")
        st.session_state.stage = "confirm"
        st.rerun()


# Renders the destination-selection background checks (safety advisory, weather outlook,
# nearest airports, and -- only when the "show sightseeing" checkbox is on -- seasonal
# activities) directly on the confirm screen, as soon as a destination and travel window are
# chosen -- independent of the "Generate itinerary options" button.
def _render_destination_insights(
    destination: DestinationCandidate,
    advisory: AdvisoryCheck,
    weather: WeatherOutlook,
    airports: list[NearbyAirport],
    activities: ActivitySearchResult,
    *,
    show_activities: bool,
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
            f"highs around {format_temp_c(weather.historical_avg_high_c)}, "
            f"lows around {format_temp_c(weather.historical_avg_low_c)}."
        )
        if weather.is_near_term and weather.near_term_forecast_summary:
            st.info(weather.near_term_forecast_summary)

    if airports:
        st.write("Closest airports:")
        st.dataframe(
            [
                {
                    "Code": a.iata_code,
                    "Airport": a.name,
                    "City": a.municipality or "—",
                    "Country": a.country_code,
                    "Distance": f"{a.distance_km:,.0f} km",
                }
                for a in airports
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Source: OurAirports public dataset (ourairports.com), matched by great-circle "
            "distance from the destination -- these are the real IATA codes flight search uses."
        )
    else:
        st.caption("Nearest-airport lookup unavailable right now.")

    if not show_activities:
        st.caption("Sightseeing & points of interest are hidden -- enable the checkbox on the search page to show them.")
    elif activities.error:
        st.caption(f"Activity/sightseeing search unavailable ({activities.error}).")
    elif activities.notable or activities.lesser_known:
        st.write("Nearby activities & sightseeing (auto-checked against your travel dates):")
        if activities.interests:
            st.caption(f"Ranked toward the interests in your request: {', '.join(activities.interests)}.")
        else:
            st.caption("No specific sightseeing interests found in your request -- ranked by notability and distance.")
        for heading, suggestions, empty_note in (
            ("Most notable", activities.notable, "No widely documented landmarks were found near this destination."),
            ("Lesser-known, worth considering", activities.lesser_known, "No further documented places were found nearby."),
        ):
            st.markdown(f"**{heading}**")
            if suggestions:
                st.dataframe(
                    [
                        {
                            "Name": a.name,
                            "Category": a.category,
                            "Why it's listed": a.why_listed,
                            "Appropriate for your dates?": a.appropriateness_comment,
                            "Listed hours": a.raw_opening_hours or "Not listed",
                        }
                        for a in suggestions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(empty_note)
        if activities.notability_note:
            st.caption(activities.notability_note)
        st.caption(activities.source_note)
    else:
        st.caption("No named attractions were found nearby in OpenStreetMap data for this destination.")


# One SessionStore per server process, shared by every browser session (each call opens its
# own SQLite connection, so sharing is safe).
@st.cache_resource
def _get_session_store() -> SessionStore:
    return SessionStore(settings.session_db_path)


# Resolves this browser session's anonymous id: reuses a well-formed id from the URL (so a
# refresh keeps the same itineraries), otherwise mints a new one. Expired rows are purged
# once per Streamlit session, here, rather than on a schedule -- the store already hides
# expired rows at read time, so the purge only reclaims space.
def _ensure_session_id(store: SessionStore) -> str:
    if "session_id" not in st.session_state:
        candidate = st.query_params.get(_SESSION_QUERY_PARAM)
        session_id = candidate if is_valid_session_id(candidate) else new_session_id()
        try:
            store.purge_expired()
            store.touch_session(session_id)
        except Exception:
            logger.exception("Session store initialization failed")
        st.session_state.session_id = session_id
    if st.query_params.get(_SESSION_QUERY_PARAM) != st.session_state.session_id:
        st.query_params[_SESSION_QUERY_PARAM] = st.session_state.session_id
    return st.session_state.session_id


# Shows (once) a confirmation queued before an st.rerun(), so the sidebar and page re-render
# with the change already applied instead of showing stale itinerary data.
def _flash(message: str) -> None:
    st.session_state.flash = message


# Displays and clears the message queued by _flash, if any.
def _show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


# "N min ago" / "N h ago" label for a past UTC timestamp.
def _hours_ago(moment: datetime) -> str:
    hours = (datetime.now(timezone.utc) - moment).total_seconds() / 3600
    if hours < 1:
        return f"{max(int(hours * 60), 0)} min ago"
    return f"{int(hours)} h ago"


# Whole hours until a future UTC timestamp (0 once it has passed).
def _hours_left(moment: datetime) -> int:
    return max(int((moment - datetime.now(timezone.utc)).total_seconds() // 3600), 0)


# Logs a failed itinerary action. Expected failures (a blank/duplicate name, an expired
# itinerary) are logged without a traceback; anything else gets one. Itinerary names are
# free-form user input, so only ids and the exception type are logged, never the name.
def _log_itinerary_failure(action: str, exc: Exception, **ids: str | None) -> None:
    # The session id is the access token for that session's itineraries (it rides in the URL),
    # so only a short prefix is logged -- enough to correlate entries, not enough to reuse.
    context = " ".join(
        f"{key}={value[:8] + '…' if key == 'session_id' else value}" for key, value in ids.items() if value
    )
    expected = isinstance(exc, (ValueError, LookupError))
    logger.error(
        "Itinerary action failed: %s (%s) %s", action, type(exc).__name__, context, exc_info=not expected
    )


# Stages that show saved itineraries rather than a step of the search flow.
_ITINERARY_VIEWS = ("itinerary", "all_itineraries")


# Switches to an itinerary view, remembering which stage to return to so browsing
# itineraries mid-search doesn't throw the in-progress search away.
def _open_view(stage: str) -> None:
    if st.session_state.stage not in _ITINERARY_VIEWS:
        st.session_state.return_stage = st.session_state.stage
    st.session_state.stage = stage


# Opens the single-itinerary view for `itinerary_id`.
def _open_itinerary(itinerary_id: str) -> None:
    st.session_state.viewing_itinerary_id = itinerary_id
    _open_view("itinerary")


# Sidebar listing this session's live itineraries (newest change first), a link to the
# all-itineraries view, and a form to create an empty itinerary.
def _render_itinerary_sidebar(store: SessionStore, session_id: str) -> None:
    with st.sidebar:
        st.header("Itineraries")
        st.caption("Itineraries changed in the last 72 hours. Older ones are removed automatically.")

        try:
            itineraries = store.list_recent_itineraries(session_id)
        except Exception as exc:
            _log_itinerary_failure("list itineraries", exc, session_id=session_id)
            st.error("Itineraries are unavailable right now.")
            return

        if st.button(
            "📚 All itineraries",
            key="open_all_itineraries",
            use_container_width=True,
            type="primary" if st.session_state.stage == "all_itineraries" else "secondary",
        ):
            _open_view("all_itineraries")
            st.rerun()

        if not itineraries:
            st.caption("No itineraries yet -- save a search result to start one.")
        for itinerary in itineraries:
            if st.button(
                itinerary.name,
                key=f"open_itinerary_{itinerary.itinerary_id}",
                help=(
                    f"{itinerary.run_count} saved search(es) · changed {_hours_ago(itinerary.updated_at)} · "
                    f"expires in ~{_hours_left(itinerary.expires_at)} h"
                ),
                use_container_width=True,
                type="primary" if itinerary.itinerary_id == st.session_state.get("viewing_itinerary_id")
                and st.session_state.stage == "itinerary" else "secondary",
            ):
                _open_itinerary(itinerary.itinerary_id)
                st.rerun()

        with st.form("sidebar_new_itinerary", clear_on_submit=True):
            name = st.text_input("New itinerary name", max_chars=MAX_ITINERARY_NAME_LENGTH)
            if st.form_submit_button("Create itinerary"):
                try:
                    itinerary_id = store.create_itinerary(session_id, name)
                except (DuplicateItineraryNameError, ValueError) as exc:
                    _log_itinerary_failure("create itinerary", exc, session_id=session_id)
                    st.error(str(exc))
                except Exception as exc:
                    _log_itinerary_failure("create itinerary", exc, session_id=session_id)
                    st.error("The itinerary couldn't be created. Please try again.")
                else:
                    _flash(f'Created itinerary "{" ".join(name.split())}".')
                    _open_itinerary(itinerary_id)
                    st.rerun()


# Lets the user save the current search result into an existing itinerary or a newly named
# one. Nothing is stored until this form is submitted; the first save stores the result and
# later saves file that same stored run into further itineraries.
def _render_save_to_itinerary(
    store: SessionStore,
    session_id: str,
    result: dict[str, Any],
    travel_start_date: date | None,
    context: dict[str, Any] | None,
) -> None:
    try:
        itineraries = store.list_recent_itineraries(session_id)
    except Exception as exc:
        _log_itinerary_failure("list itineraries", exc, session_id=session_id)
        st.error("Itineraries are unavailable right now, so this search can't be saved.")
        return

    run_id = st.session_state.get("result_run_id")
    if run_id is None:
        st.caption("This search isn't saved yet -- it's discarded on \"Start over\" unless saved to an itinerary.")
    # Keyed by a per-search counter (bumped each time a new result is generated) so every new
    # search gets a fresh form instead of inheriting the previous one's widget state.
    form_key = f"save_result_{st.session_state.get('result_seq', 0)}"
    options = [_NEW_ITINERARY_OPTION] + [i.name for i in itineraries]
    # A refined search defaults to the itinerary it came from (if that itinerary still exists).
    refine_id = (st.session_state.get("refine") or {}).get("itinerary_id")
    default_index = next(
        (index + 1 for index, i in enumerate(itineraries) if i.itinerary_id == refine_id), 0
    )
    with st.form(form_key, clear_on_submit=True):
        st.markdown("**Save this search to an itinerary**")
        choice = st.selectbox("Itinerary", options, index=default_index, key=f"{form_key}_choice")
        new_name = st.text_input(
            "New itinerary name (used when \"New itinerary\" is selected)",
            max_chars=MAX_ITINERARY_NAME_LENGTH,
            key=f"{form_key}_name",
        )
        if not st.form_submit_button("Save to itinerary"):
            return

    itinerary_id = None
    try:
        if choice == _NEW_ITINERARY_OPTION:
            itinerary_id = store.create_itinerary(session_id, new_name)
            target_name = " ".join(new_name.split())
        elif new_name.strip():
            # A typed name with an existing itinerary selected is ambiguous -- refuse rather than
            # silently ignore the name and file the search somewhere the user didn't intend.
            raise ValueError(
                'Choose "New itinerary" to use the name you typed, or clear the name to save to '
                f'"{choice}".'
            )
        else:
            target = next(i for i in itineraries if i.name == choice)
            itinerary_id, target_name = target.itinerary_id, target.name
        if run_id is None:
            st.session_state.result_run_id = store.save_search_to_itinerary(
                session_id, itinerary_id, result, travel_start_date, context
            )
        else:
            store.add_run_to_itinerary(session_id, itinerary_id, run_id)
    except (DuplicateItineraryNameError, ValueError) as exc:
        _log_itinerary_failure("save search", exc, session_id=session_id, itinerary_id=itinerary_id)
        st.error(str(exc))
    except (ItineraryNotFoundError, SearchRunNotFoundError) as exc:
        _log_itinerary_failure("save search", exc, session_id=session_id, itinerary_id=itinerary_id, run_id=run_id)
        st.error("That itinerary or search has expired. Please refresh and try again.")
    except Exception as exc:
        _log_itinerary_failure("save search", exc, session_id=session_id, itinerary_id=itinerary_id, run_id=run_id)
        st.error("This search couldn't be saved. Please try again.")
    else:
        _flash(f'Saved to itinerary "{target_name}".')
        st.rerun()


# Renders one generated result (the dict build_itineraries returns): the itinerary markdown,
# its scoring table, and the safety summary it was grounded on.
def _render_search_result(result: dict[str, Any]) -> None:
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


# Shows the search a saved run came from, so it can be reviewed or refined later.
def _render_search_context(context: dict[str, Any]) -> None:
    request = context.get("request") or {}
    if not request:
        return
    days = request.get("trip_length_days")
    budget = request.get("budget_usd")
    details = [
        f"From: {request.get('origin_city') or 'not specified'}",
        f"Trip length: {f'{days} days' if days else 'not specified'}",
        f"Travelers: {request.get('traveler_count', 1)}",
        f"Budget: {f'${budget:,.0f}' if budget is not None else 'not specified'}",
    ]
    st.markdown("**Search details**")
    # Free-form user input (request text, origin): shown as plain text, never rendered as Markdown.
    st.text(" · ".join(details))
    st.text(request.get("request_text", ""))


# Renders one saved search inside an itinerary view. Searches saved with their context get a
# "Refine this search" button that reopens the request for editing, scoped to `itinerary`.
def _render_stored_run(run: SearchRun, itinerary: Itinerary) -> None:
    st.caption(
        f"Searched {_hours_ago(run.created_at)}. Prices and availability are a snapshot from that "
        "time, not live quotes."
    )
    # Outside the try below: st.rerun() works by raising, which that handler must not swallow.
    if run.context.get("request") and st.button(
        "Refine this search",
        key=f"refine_{itinerary.itinerary_id}_{run.run_id}",
        help=f'Start a new search from this one; its result can be saved back to "{itinerary.name}".',
    ):
        _start_refine(itinerary, run.context)
        st.rerun()
    try:
        _render_search_context(run.context)
        _render_search_result(run.result)
    except Exception as exc:
        _log_itinerary_failure("render saved search", exc, run_id=run.run_id)
        st.error("This saved search couldn't be displayed.")


# Lists an itinerary's saved searches, or returns None (after logging and showing an error)
# if they can't be loaded.
def _load_itinerary_runs(store: SessionStore, session_id: str, itinerary_id: str) -> list[SearchRun] | None:
    try:
        return store.list_itinerary_runs(session_id, itinerary_id)
    except Exception as exc:
        _log_itinerary_failure("list saved searches", exc, session_id=session_id, itinerary_id=itinerary_id)
        st.error("This itinerary's saved searches are unavailable right now.")
        return None


# UI Configuration and Page Setup
st.set_page_config(page_title="Agentic Travel Concierge", page_icon="🧳", layout="wide")

st.title("🧳 Agentic Travel Concierge")
st.write(
    "Describe your ideal trip (budget, climate, activities, pace, dates, and preferences). "
    "The app identifies matching destinations for you to confirm, then generates multiple "
    "luxury-within-budget itinerary options with safety-aware guidance."
)

# Which screen to show: "intake" -> ("discover") -> "confirm" -> "results", or one of the
# _ITINERARY_VIEWS. Each Streamlit rerun executes this script top to bottom and renders the
# branch for the current stage.
if "stage" not in st.session_state:
    st.session_state.stage = "intake"

session_store = _get_session_store()
session_id = _ensure_session_id(session_store)
_render_itinerary_sidebar(session_store, session_id)
_show_flash()

# --- Stage 1: free-text intake -----------------------------------------------------
if st.session_state.stage == "intake":
    # Set by "Refine this search": pre-fills the form below from the saved search.
    refine = st.session_state.get("refine")
    prefill = (refine or {}).get("request") or {}
    if refine:
        st.info(
            f'Refining a saved search from itinerary "{refine["itinerary_name"]}". Edit the request '
            "below; the new result can be saved back to that itinerary."
        )
        if st.button("Cancel refining"):
            _reset()
            st.rerun()

    with st.form("trip_form"):
        request_text = st.text_area(
            "Travel request",
            value=prefill.get("request_text") or "",
            height=180,
            max_chars=_MAX_REQUEST_CHARS,
            placeholder=(
                "Example: Plan a 7-day honeymoon in Japan from Seattle for 2 travelers with a total budget under $7000, "
                "moderate weather, food tours, ryokan stay, and low-risk neighborhoods."
            ),
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            origin_city = st.text_input(
                "Origin city (optional for flight search; required if you want us to suggest "
                "destinations instead of naming one)",
                value=prefill.get("origin_city") or "",
                max_chars=_MAX_ORIGIN_CHARS,
            )
        # Saved values are clamped to the inputs' ranges, which a widget's initial value must respect.
        with col2:
            trip_length_days = st.number_input(
                "Trip length (days, optional)", min_value=0, max_value=60,
                value=min(max(int(prefill.get("trip_length_days") or 0), 0), 60),
            )
        with col3:
            traveler_count = st.number_input(
                "Number of travelers", min_value=1, max_value=20,
                value=min(max(int(prefill.get("traveler_count") or 1), 1), 20),
            )

        show_activities = st.checkbox(
            "Show sightseeing opportunities & places of interest",
            value=bool((refine or {}).get("show_activities", False)),
            help=(
                "When checked, the confirm screen will search OpenStreetMap for nearby "
                "attractions/points of interest and display them. Off by default."
            ),
        )

        submitted = st.form_submit_button("Find destinations")

    if submitted:
        if len(request_text.strip()) < 10:
            st.warning("Please provide a more detailed request.")
        elif len(request_text) > _MAX_REQUEST_CHARS or len(origin_city) > _MAX_ORIGIN_CHARS:
            st.warning(
                f"Please keep the request under {_MAX_REQUEST_CHARS} characters and the origin city "
                f"under {_MAX_ORIGIN_CHARS}."
            )
        else:
            candidates: list[DestinationCandidate] = []
            parsed_date: date | None = None
            parsed_date_note: str | None = None
            parsed: ParsedTripRequest | None = None
            resolution_failed = False
            with st.status("Identifying destination options...", expanded=True) as status:
                try:
                    parsed = asyncio.run(_resolve_intake(request_text.strip()))
                    candidates = parsed.candidates
                    parsed_date, parsed_date_note = _start_date_from_request(parsed)
                    if parsed_date is None and (saved_start := _refine_start_date(refine)):
                        parsed_date = saved_start
                        parsed_date_note = f"Using the start date from the saved search: {saved_start.isoformat()}"
                    if candidates:
                        label = f"Found {len(candidates)} destination option(s)"
                    elif parsed.open_ended:
                        label = "No specific destination requested -- we'll suggest some"
                    else:
                        label = "No destination found in request text"
                    status.update(label=label, state="complete")
                except Exception:
                    logger.exception(
                        "Destination resolution failed (request length=%d chars)", len(request_text)
                    )
                    status.update(label="Destination lookup failed", state="error")
                    resolution_failed = True

            if parsed is not None:
                # Values stated in the text fill form fields left at their defaults (0 days / 1 traveler).
                trip_length_days = trip_length_days or parsed.constraints.trip_length_days or 0
                if traveler_count == 1 and parsed.constraints.traveler_count:
                    traveler_count = parsed.constraints.traveler_count
                st.session_state.parsed_request = parsed
                st.session_state.parsed_date_note = parsed_date_note

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
                trip_length_days = trip_length_days or DEFAULT_TRIP_LENGTH_DAYS
                discovery_start = parsed_date or date.today()
                st.session_state.discovery_start_date = parsed_date
                discovery_end = discovery_start + timedelta(days=trip_length_days)

                discovery_result = None
                discovery_failed = False
                with st.status(
                    "No specific destination was named -- searching for destinations that match what "
                    "you described (this checks real distance, weather, and points-of-interest data, "
                    "so it can take a little while)...",
                    expanded=True,
                ) as status:
                    try:
                        discovery_result = asyncio.run(
                            _run_discovery(
                                request_text.strip(), origin_city.strip(), discovery_start, discovery_end,
                                parsed.constraints,
                            )
                        )
                        status.update(label="Destination search complete", state="complete")
                    except Exception:
                        logger.exception(
                            "Destination discovery failed (request length=%d chars)", len(request_text)
                        )
                        status.update(label="Destination search failed", state="error")
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
                        "budget_usd": parsed.constraints.budget_usd,
                        "show_activities": show_activities,
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
                    "budget_usd": parsed.constraints.budget_usd,
                    "show_activities": show_activities,
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
    _render_request_understanding(st.session_state.get("parsed_request"))
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
    _render_request_understanding(st.session_state.get("parsed_request"))

    st.subheader("When are you traveling?")
    if parsed_date is not None:
        st.caption(
            st.session_state.get("parsed_date_note")
            or f"Detected travel start date from your request: {parsed_date.isoformat()}"
        )
        use_detected_date = st.checkbox("Use this date", value=True)
        travel_start_date = parsed_date if use_detected_date else st.date_input(
            "Travel start date", min_value=date.today()
        )
    else:
        st.info("We couldn't find a clear travel start date in your request -- please choose one.")
        travel_start_date = st.date_input("Travel start date", min_value=date.today())

    trip_length_days = st.session_state.pending.get("trip_length_days") or DEFAULT_TRIP_LENGTH_DAYS
    insight_end_date = travel_start_date + timedelta(days=trip_length_days)
    show_activities = bool(st.session_state.pending.get("show_activities", False))
    insight_key = (selected_destination.label, travel_start_date.isoformat(), insight_end_date.isoformat(), show_activities)

    _INSIGHT_STEP_LABELS = {
        "advisory": "U.S. State Department travel advisory",
        "weather": "Weather outlook",
        "airports": "Closest airports",
        "activities": "Sightseeing & points of interest",
    }

    if st.session_state.get("insight_key") != insight_key:
        with st.status("Running destination checks...", expanded=True) as status:
            def _on_insight_progress(step_key: str) -> None:
                status.write(f"✅ {_INSIGHT_STEP_LABELS.get(step_key, step_key)}")

            try:
                insights = asyncio.run(
                    _gather_destination_insights(
                        selected_destination,
                        travel_start_date,
                        insight_end_date,
                        include_activities=show_activities,
                        request_text=st.session_state.pending.get("request_text", ""),
                        on_progress=_on_insight_progress,
                    )
                )
                status.update(label="Destination checks complete", state="complete")
            except Exception:
                logger.exception("Destination insight checks failed for a selected destination")
                status.update(label="Destination checks failed", state="error")
                insights = {
                    "advisory": AdvisoryCheck(error="an unexpected error occurred while checking"),
                    "weather": WeatherOutlook(error="an unexpected error occurred while checking"),
                    "airports": [],
                    "activities": ActivitySearchResult(error="an unexpected error occurred while checking"),
                }
        st.session_state.insight_key = insight_key
        st.session_state.insight_advisory = insights["advisory"]
        st.session_state.insight_weather = insights["weather"]
        st.session_state.insight_airports = insights["airports"]
        st.session_state.insight_activities = insights["activities"]

    _render_destination_insights(
        selected_destination,
        st.session_state.insight_advisory,
        st.session_state.insight_weather,
        st.session_state.insight_airports,
        st.session_state.insight_activities,
        show_activities=show_activities,
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
            budget_usd=pending.get("budget_usd"),
            travel_start_date=travel_start_date,
        )
        result = None
        with st.status("Planning your trip with the agentic workflow...", expanded=True) as status:
            def _on_build_progress(message: str) -> None:
                status.write(message)

            try:
                result = asyncio.run(build_itineraries(request, selected_destination, on_progress=_on_build_progress))
                status.update(label="Itinerary generation complete", state="complete")
            except Exception:
                # Don't log the raw request text: it's free-form user input that
                # commonly includes names, travel dates, and budget — logging it
                # verbatim at exception level puts PII into whatever aggregates
                # server logs. Length is enough to correlate/debug from.
                logger.exception(
                    "build_itineraries failed (request length=%d chars)", len(pending["request_text"])
                )
                status.update(label="Itinerary generation failed", state="error")
                st.error(
                    "We couldn't generate itinerary options this time (an upstream service may be "
                    "unavailable or rate-limited). Please try again in a moment."
                )

        if result is not None:
            # The result is only held in this browser session; it's stored when (and only if)
            # the user saves it to an itinerary from the results screen.
            st.session_state.pop("result_run_id", None)
            st.session_state.result_seq = st.session_state.get("result_seq", 0) + 1
            st.session_state.result_travel_start_date = travel_start_date
            st.session_state.result = result
            # Snapshot of the search behind this result, saved alongside it if the user files it
            # into an itinerary -- the live search state is discarded on "Start over".
            st.session_state.result_context = {
                "request": request.model_dump(mode="json"),
                "destination": selected_destination.model_dump(mode="json"),
                "show_activities": show_activities,
            }
            st.session_state.stage = "results"
            st.rerun()

# --- Stage 3: results ----------------------------------------------------------------
elif st.session_state.stage == "results":
    result = st.session_state.result

    if st.button("Start over"):
        _reset()
        st.rerun()

    if refine := st.session_state.get("refine"):
        st.caption(f'Refined from a search in itinerary "{refine["itinerary_name"]}".')
    _render_save_to_itinerary(
        session_store,
        session_id,
        result,
        st.session_state.get("result_travel_start_date"),
        st.session_state.get("result_context"),
    )

    _render_search_result(result)

# --- Itinerary views (opened from the sidebar) ------------------------------------------
elif st.session_state.stage in _ITINERARY_VIEWS:
    if st.button("Back"):
        st.session_state.stage = st.session_state.pop("return_stage", "intake")
        st.session_state.pop("viewing_itinerary_id", None)
        st.rerun()

    if st.session_state.stage == "all_itineraries":
        st.subheader("All itineraries")
        try:
            all_itineraries = session_store.list_recent_itineraries(session_id)
        except Exception as exc:
            _log_itinerary_failure("list itineraries", exc, session_id=session_id)
            st.error("Itineraries are unavailable right now.")
            all_itineraries = []
        else:
            if not all_itineraries:
                st.info("No itineraries yet -- save a search result to start one.")
        for itinerary in all_itineraries:
            with st.container(border=True):
                name_col, open_col = st.columns([4, 1])
                with name_col:
                    st.markdown(f"#### {itinerary.name}")
                    st.caption(
                        f"{itinerary.run_count} saved search(es) · changed {_hours_ago(itinerary.updated_at)} · "
                        f"removed automatically in ~{_hours_left(itinerary.expires_at)} h unless changed again."
                    )
                with open_col:
                    if st.button("Open", key=f"all_open_{itinerary.itinerary_id}", use_container_width=True):
                        _open_itinerary(itinerary.itinerary_id)
                        st.rerun()
                saved_runs = _load_itinerary_runs(session_store, session_id, itinerary.itinerary_id)
                if saved_runs == []:
                    st.caption("No searches saved to this itinerary yet.")
                for run in saved_runs or []:
                    with st.expander(run.label):
                        _render_stored_run(run, itinerary)
    else:
        itinerary_id = st.session_state.get("viewing_itinerary_id")
        try:
            itinerary = session_store.get_itinerary(session_id, itinerary_id) if itinerary_id else None
        except Exception as exc:
            _log_itinerary_failure("open itinerary", exc, session_id=session_id, itinerary_id=itinerary_id)
            st.error("This itinerary is unavailable right now.")
        else:
            if itinerary is None:
                _log_itinerary_failure(
                    "open itinerary", ItineraryNotFoundError(itinerary_id), session_id=session_id,
                    itinerary_id=itinerary_id,
                )
                st.warning(
                    "This itinerary is no longer available -- it may have expired after 72 hours without changes."
                )
            else:
                st.subheader(f"Itinerary: {itinerary.name}")
                st.caption(
                    f"Last changed {_hours_ago(itinerary.updated_at)} · removed automatically in "
                    f"~{_hours_left(itinerary.expires_at)} h unless changed again."
                )

                saved_runs = _load_itinerary_runs(session_store, session_id, itinerary.itinerary_id)
                if saved_runs == []:
                    st.info("No searches saved to this itinerary yet.")
                for run in saved_runs or []:
                    with st.expander(run.label):
                        _render_stored_run(run, itinerary)
