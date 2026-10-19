from __future__ import annotations

import asyncio
import logging

import httpx
from datetime import timedelta
from textwrap import dedent
from typing import Awaitable, Callable

from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient
from ollama import AsyncClient as OllamaAsyncClient

from config import settings
from .data_sources import gather_grounding_packet
from .http_utils import describe_error
from .schemas import DEFAULT_TRIP_LENGTH_DAYS, DestinationCandidate, UserTripRequest

logger = logging.getLogger(__name__)

# Receives human-readable progress messages (shown live in the UI's status panel).
ProgressCallback = Callable[[str], None]


# Sends a progress message to the UI's callback, if one was given.
def _report(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


# Runs a batch of named coroutines concurrently and reports each one to `on_progress` as it
# completes (in completion order, not the order given) rather than only after the whole batch
# finishes -- this is what lets the UI show a live per-search status bar instead of one opaque
# spinner for the entire concurrent batch.
async def _gather_with_progress(
    jobs: dict[str, Awaitable[str]], on_progress: ProgressCallback | None
) -> dict[str, str]:
    async def _tag(label: str, coro: Awaitable[str]) -> tuple[str, str]:
        return label, await coro

    results: dict[str, str] = {}
    for finished in asyncio.as_completed([_tag(label, coro) for label, coro in jobs.items()]):
        label, result = await finished
        results[label] = result
        _report(on_progress, f"{label}: done")
    return results


# Builds the OllamaChatClient used by every agent in this app. Public (also used by
# destination_discovery.py's brainstorming agent) since both modules need the same client,
# configured against this app's Ollama Cloud account.
#
# agent_framework_ollama.OllamaChatClient has no parameter of its own for auth headers, so
# the Authorization: Bearer <OLLAMA_API_KEY> header Ollama Cloud requires (verified against
# https://docs.ollama.com/api/authentication) is attached by building the underlying
# ollama.AsyncClient ourselves and handing it in via the `client=` kwarg.
#
# The ollama SDK's own default is no timeout at all (verified: its httpx client reports
# Timeout(timeout=None)), so a stalled model call would block the user's session forever. The
# slowest call in a measured end-to-end build (2026-10-19) took 21.7 s; 120 s leaves 5x margin.
# A timeout surfaces as an exception that every caller already handles (specialist fallback
# text, discovery's fallback name list, intake's rule-only parse).
_OLLAMA_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


# Raises RuntimeError when OLLAMA_API_KEY is unset, so callers fall back instead of sending
# unauthenticated requests.
def build_ollama_client() -> OllamaChatClient:
    if not settings.ollama_api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. An Ollama Cloud API key is required to reach "
            f"{settings.ollama_host} -- create one at https://ollama.com/settings/keys and set "
            "it in your environment or .env file (never paste the key itself into chat)."
        )
    ollama_client = OllamaAsyncClient(
        host=settings.ollama_host,
        headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        timeout=_OLLAMA_TIMEOUT,
    )
    return OllamaChatClient(client=ollama_client, model=settings.ollama_model)


# Defines the strict scoring logic used by the ConciergeSynthesizer to rank itinerary options
def _deterministic_ranking_rubric() -> str:
    return dedent(
        """
        Deterministic ranking rubric (0-100):
        - Rating quality score (0-10) -> weight 40%
            Use normalized rating values from Booking.com Hotels/other provided APIs.
            If multiple ratings exist, compute arithmetic mean.
        - Review confidence score (0-10) -> weight 20%
            Combine review volume, recency, and consistency.
            High volume + recent + stable sentiment => higher score.
        - Safety score (0-10) -> weight 25%
            Derive from advisory severity, known local risks, and safe-area availability.
        - Budget-fit score (0-10) -> weight 15%
            Compare total itinerary estimate against user budget and value density.

        Formula:
        final_score = round(
            rating_quality*4.0 + review_confidence*2.0 + safety*2.5 + budget_fit*1.5,
            2
        )

        Deterministic rules:
        1) Always rank by final_score descending.
        2) If scores tie, use higher safety score.
        3) If still tied, use higher review confidence.
        4) If still tied, use lower total estimated cost.
        5) If still tied, keep lexical order by itinerary title.

        Missing-data fallback:
        - If a component has insufficient data, assign neutral 5.0 for that component.
        - Explicitly note which component used fallback.
        """
    ).strip()


# Shared by the logistics and synthesis agents: every option carries at least one real
# accommodation from the Booking.com data (never an invented or generic one).
_ACCOMMODATION_RULE = (
    "Every option must recommend at least one accommodation taken from the Booking.com Hotels data, "
    "using the property's exact name, star class, review score, and total stay price exactly as given. "
    "Match each option's accommodation to that option's budget, and use a different property for each "
    "option when the data has enough. Never recommend an accommodation that is not in the Booking.com "
    "Hotels data; if that data is unavailable, say so in each option instead."
)


# Prompt text telling the agents how to treat budget: stay within the stated total, or, with no
# budget, make the three options low/medium/high budget tiers.
def _budget_instruction(budget_usd: float | None) -> str:
    if budget_usd is None:
        return (
            "No budget was given. Make the 3 options three budget tiers, in this order: 'Low budget', "
            "'Medium budget', 'High budget'. Put the tier name in each option's heading, give each a target "
            "total cost range that rises from tier to tier, and score budget_fit against each option's own tier."
        )
    return (
        f"The traveler's total budget is ${budget_usd:,.0f}. Keep every option within it; if the data makes "
        "that impossible, say which option exceeds it and by how much."
    )


# Runs a specialist agent and converts a hard failure (LLM endpoint down, timeout,
# malformed response) into a clearly-labeled fallback string instead of aborting the
# whole itinerary build. Callers still see that a section degraded.
async def _run_specialist(agent: Agent, prompt: str) -> str:
    try:
        return str(await agent.run(prompt))
    except Exception as ex:
        logger.exception("Specialist agent %s failed", getattr(agent, "name", agent.__class__.__name__))
        # describe_error, not str(ex): this text reaches the final prompt and the user's screen,
        # and a raw HTTP error message embeds the full request URL.
        return f"{getattr(agent, 'name', 'Specialist agent')} unavailable: {describe_error(ex)}"


# Builds the itinerary options: two specialist calls in parallel (logistics, experiences), then
# one synthesis call.
# `destination` must already be a user-confirmed candidate (see travel_agent.location_resolution
# .geocode_candidates) -- this function resolves real provider codes for it and searches, it does
# not itself guess which place the user meant.
async def build_itineraries(
    user_request: UserTripRequest,
    destination: DestinationCandidate,
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, str]:
    if user_request.travel_start_date is None:
        raise ValueError("build_itineraries requires a resolved travel_start_date; ask the user for one first.")

    check_in = user_request.travel_start_date
    trip_length_days = user_request.trip_length_days or DEFAULT_TRIP_LENGTH_DAYS
    check_out = check_in + timedelta(days=trip_length_days)

    wants_flights = bool(user_request.origin_city and user_request.origin_city.strip())

    _report(on_progress, "Gathering hotel, flight, visa, and safety/weather grounding data...")
    grounding = await gather_grounding_packet(
        destination,
        request_text=user_request.request_text,
        origin_city=user_request.origin_city,
        traveler_count=user_request.traveler_count,
        check_in=check_in,
        check_out=check_out,
        wants_flights=wants_flights,
    )
    _report(on_progress, "Grounding data gathered.")

    client = build_ollama_client()
    budget_instruction = _budget_instruction(user_request.budget_usd)

    # Three model calls, not eight. A per-source summary stage (hotel, flight, visa, airport
    # agents) and a separate safety agent used to run first, but logistics and the synthesizer
    # already receive the same raw API data and advisory text. An A/B test on identical grounding
    # (2026-10-19, 3 runs each) gave the same structural results -- scoring table parsed 3/3,
    # safety/evidence/sources sections, API hotels named, no non-API hotels -- at a mean of
    # 17.9 s vs 37.7 s. The removed agents' no-invention rules now live in the logistics
    # instructions below.
    logistics_agent = Agent(
        name="LogisticsAgent",
        client=client,
        instructions=dedent(
            """
            You are a luxury-on-budget travel planner.
            Build 3 itinerary options that are luxurious, budget-aware, and realistic.
            Include flights when relevant, hotels that are 3-star or better, and local transport guidance.
            Use all relevant provided API evidence: Booking.com Hotels, Booking.com Flights,
            and Visa Requirements.
            Prioritize higher-rated options using both source ratings and customer review quality signals
            (review score, review volume, recency, consistency).
            Prefer accommodation options with strong Booking.com Hotels ratings and review confidence.
            Booking.com Hotels gives each property's star class, review score/count, and the total price
            for the whole stay; Booking.com Flights gives total price, route, times, stops and carriers.
            If the API data says a hotel or flight search was skipped or unavailable, say so plainly --
            never invent hotel names, ratings, prices, flight routes, or airport codes. Use only airport
            codes that appear in the API data.
            Return practical, bookable-seeming plans with transparent cost ranges and clear tradeoffs.
            """
        ).strip()
        + "\n"
        + _ACCOMMODATION_RULE,
    )

    experiences_agent = Agent(
        name="ExperiencesAgent",
        client=client,
        instructions=dedent(
            """
            You are a destination curator.
            Recommend culturally meaningful, destination-unique experiences and dining/activities with safety in mind.
            Avoid generic suggestions.
            Prioritize authentic, high-value experiences backed by reliable public signals
            (Booking.com reviews/ratings, official tourism boards, and other reputable public sources).
            Favor activities with consistently high ratings and positive recent sentiment.
            """
        ).strip(),
    )

    # Independent of each other (both read only the grounding packet), so they run concurrently;
    # _gather_with_progress reports each to the status bar as it completes.
    specialist_results = await _gather_with_progress(
        {
            "Logistics plan": _run_specialist(
                logistics_agent,
                dedent(
                    f"""
                    Traveler request: {user_request.request_text}
                    Origin city: {user_request.origin_city or 'Not provided'}
                    Trip dates: {check_in.isoformat()} to {check_out.isoformat()} ({trip_length_days} days)
                    Number of travelers: {user_request.traveler_count}
                    Budget: {budget_instruction}
                    Weather grounding: {grounding.weather_summary}
                    Transport grounding: {grounding.local_transport_notes}
                    External API data: {grounding.external_api_snippets or 'No external API snippets provided'}
                    """
                ).strip(),
            ),
            "Experience plan": _run_specialist(
                experiences_agent,
                dedent(
                    f"""
                    Traveler request: {user_request.request_text}
                    Destination hint: {grounding.destination_hint}
                    Climate/weather grounding: {grounding.weather_summary}
                    """
                ).strip(),
            ),
        },
        on_progress,
    )
    logistics_plan = specialist_results["Logistics plan"]
    experiences_plan = specialist_results["Experience plan"]

    final_agent = Agent(
        name="ConciergeSynthesizer",
        client=client,
        instructions=dedent(
            """
            You are an executive travel concierge.
            Create polished itinerary recommendations in markdown.
            Requirements:
            - Provide exactly 3 itinerary options.
            - Each option must include: target budget range, flight plan (if relevant), hotel plan (3-star+), unique experiences, and local transport recommendation.
            - Use all relevant provided API evidence in each option: Booking.com Hotels,
              Booking.com Flights, and Visa Requirements.
            - Apply the deterministic ranking rubric exactly as provided in the user context.
            - For hotels and activities, explicitly reference rating confidence based on score + review volume.
            - If an API source is missing/unavailable, state that clearly and continue with available evidence.
            - Include a dedicated safety section with explicit caution areas and behavior recommendations.
            - Keep a luxury feel within each option's budget, following the Budget instruction in the user context.
            - Include a section titled 'Deterministic Scoring Table' that shows, for each itinerary:
              rating_quality, review_confidence, safety, budget_fit, final_score, and tie-break explanation.
            - End with two sections:
              1) 'API Evidence Coverage' listing what was used from each provided API
              2) 'Grounding Sources' list containing URLs.
            """
        ).strip()
        + "\n- "
        + _ACCOMMODATION_RULE,
    )

    _report(on_progress, "Synthesizing final itinerary options...")
    final_markdown = await final_agent.run(
        dedent(
            f"""
            Traveler request:
            {user_request.request_text}

            Budget: {budget_instruction}

            Inputs from specialist agents:
            Logistics plan:
            {logistics_plan}

            Experience plan:
            {experiences_plan}

            Hard grounding to preserve:
            - Destination: {grounding.destination_hint}
            - Weather: {grounding.weather_summary}
            - Safety summary: {grounding.safety_summary}
            - Advisory level: {grounding.advisory_level}
            - Advisory source URL: {grounding.advisory_source_url}
            - Local transport notes: {grounding.local_transport_notes}
            - External APIs: {grounding.external_api_snippets or 'None'}
                        - APIs expected to be considered when available:
                            Booking.com Hotels, Booking.com Flights, Visa Requirements
                        - Deterministic rubric to apply exactly:
                            {_deterministic_ranking_rubric()}
            - Source URL that must be included: https://travel.state.gov/content/travel.html
            """
        ).strip()
    )
    _report(on_progress, "Final itinerary ready.")

    return {
        "destination": grounding.destination_hint,
        "itinerary_markdown": str(final_markdown),
        "safety_summary": grounding.safety_summary,
        "advisory_source_url": grounding.advisory_source_url,
    }
