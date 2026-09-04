from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import timedelta
from textwrap import dedent

from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

from config import settings
from .data_sources import gather_grounding_packet
from .schemas import DEFAULT_TRIP_LENGTH_DAYS, DestinationCandidate, UserTripRequest
from .search_agents import HotelSearchAgent, FlightSearchAgent, VisaSearchAgent, AirportSearchAgent, CruiseSearchAgent

logger = logging.getLogger(__name__)

_CRUISE_INTEREST_PATTERN = re.compile(r"\bcruis", re.IGNORECASE)


# Sets the necessary Ollama environment variables from the application settings. Public
# (also used by destination_discovery.py's brainstorming agent) since both modules need
# the same OllamaChatClient configured against this app's OLLAMA_* settings.
def configure_model_env() -> None:
    for key, value in settings.ollama_env.items():
        os.environ[key] = value


# Defines the strict scoring logic used by the ConciergeSynthesizer to rank itinerary options
def _deterministic_ranking_rubric() -> str:
    return dedent(
        """
        Deterministic ranking rubric (0-100):
        - Rating quality score (0-10) -> weight 40%
            Use normalized rating values from Duffel Stays/other provided APIs.
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


# Runs a specialist agent and converts a hard failure (LLM endpoint down, timeout,
# malformed response) into a clearly-labeled fallback string instead of aborting the
# whole itinerary build. Callers still see that a section degraded.
async def _run_specialist(agent: Agent, prompt: str) -> str:
    try:
        return str(await agent.run(prompt))
    except Exception as ex:
        logger.exception("Specialist agent %s failed", getattr(agent, "name", agent.__class__.__name__))
        return f"{getattr(agent, 'name', 'Specialist agent')} unavailable: {ex}"


# Main orchestration function that coordinates multiple specialist agents to build a complete trip plan.
# `destination` must already be a user-confirmed candidate (see travel_agent.location_resolution
# .geocode_candidates) -- this function resolves real provider codes for it and searches, it does
# not itself guess which place the user meant.
async def build_itineraries(user_request: UserTripRequest, destination: DestinationCandidate) -> dict[str, str]:
    configure_model_env()

    if user_request.travel_start_date is None:
        raise ValueError("build_itineraries requires a resolved travel_start_date; ask the user for one first.")

    check_in = user_request.travel_start_date
    trip_length_days = user_request.trip_length_days or DEFAULT_TRIP_LENGTH_DAYS
    check_out = check_in + timedelta(days=trip_length_days)

    wants_flights = bool(user_request.origin_city and user_request.origin_city.strip())
    wants_cruise = bool(_CRUISE_INTEREST_PATTERN.search(user_request.request_text))

    grounding = await gather_grounding_packet(
        destination,
        request_text=user_request.request_text,
        origin_city=user_request.origin_city,
        traveler_count=user_request.traveler_count,
        check_in=check_in,
        check_out=check_out,
        wants_flights=wants_flights,
        wants_cruise=wants_cruise,
    )

    client = OllamaChatClient()

    hotel_agent = HotelSearchAgent(client)
    flight_agent = FlightSearchAgent(client)
    visa_agent = VisaSearchAgent(client)
    airport_agent = AirportSearchAgent(client)
    cruise_agent = CruiseSearchAgent(client)

    safety_agent = Agent(
        name="SafetyAgent",
        client=client,
        instructions=dedent(
            """
            You are a travel safety specialist.
            Prioritize traveler safety and legal compliance.
            Use U.S. Department of State-aligned caution language and avoid recommending risky districts.
            Output concise safety constraints and risk mitigations.
            """
        ).strip(),
    )

    logistics_agent = Agent(
        name="LogisticsAgent",
        client=client,
        instructions=dedent(
            """
            You are a luxury-on-budget travel planner.
            Build 3 itinerary options that are luxurious, budget-aware, and realistic.
            Include flights when relevant, hotels that are 3-star or better, and local transport guidance.
            Use all relevant provided API evidence: Duffel Stays, Duffel Flights,
            Visa Requirements, and Tripadvisor Cruises (when present).
            Prioritize higher-rated options using both source ratings and customer review quality signals
            (review score, review volume, recency, consistency).
            Prefer accommodation options with strong Duffel Stays ratings and review confidence.
            Return practical, bookable-seeming plans with transparent cost ranges and clear tradeoffs.
            """
        ).strip(),
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
            (Tripadvisor reviews/ratings, official tourism boards, and other reputable public sources).
            Favor activities with consistently high ratings and positive recent sentiment.
            """
        ).strip(),
    )

    # These five specialist calls are mutually independent (each only reads the
    # shared grounding packet) — run them concurrently rather than paying five
    # sequential LLM round-trips.
    (
        hotel_search_results,
        flight_search_results,
        visa_search_results,
        airport_search_results,
        cruise_search_results,
    ) = await asyncio.gather(
        _run_specialist(
            hotel_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Destination: {grounding.destination_hint}
                API Data: {grounding.external_api_snippets or 'No data'}

                Hotel Search Requirements:
                - Use the accommodation's 'name', 'rating', 'review_score', and 'review_count' for the summary.
                - Use 'cheapest_rate_total_amount'/'cheapest_rate_currency' for cost, and 'location.address' for
                  the property's location.
                - Reference 'photos' entries when present.

                Find the best hotel/accommodation options.
                """
            ).strip(),
        ),
        _run_specialist(
            flight_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Origin: {user_request.origin_city or 'Not provided'}
                Destination: {grounding.destination_hint}
                Requested dates: {check_in.isoformat()} to {check_out.isoformat()}
                API Data: {grounding.external_api_snippets or 'No data'}

                {"Flights were not searched because no origin city was provided." if not wants_flights else ""}
                Find the best flight options from the Duffel Flights API data above.
                """
            ).strip(),
        ),
        _run_specialist(
            visa_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Destination: {grounding.destination_hint}
                Destination country code: {grounding.destination_country_code or 'Unknown'}
                API Data: {grounding.external_api_snippets or 'No data'}
                Find visa and entry requirements.
                """
            ).strip(),
        ),
        _run_specialist(
            airport_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Destination: {grounding.destination_hint}
                API Data: {grounding.external_api_snippets or 'No data'}
                Summarize the airports already resolved in the API data above (do not invent codes
                that aren't present in that data).
                """
            ).strip(),
        ),
        _run_specialist(
            cruise_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Destination: {grounding.destination_hint}
                API Data: {grounding.external_api_snippets or 'No data'}

                {"" if wants_cruise else "No cruise interest was detected in the traveler request, so cruise search was skipped -- say so briefly and move on."}
                Find the best cruise options. Translate technical logistics into user-friendly summaries.
                """
            ).strip(),
        ),
    )

    # safety_brief and experiences_plan only need the grounding packet; logistics_plan
    # needs the specialist results above but not the other two. None of the three
    # depend on each other, so run them concurrently too.
    safety_brief, logistics_plan, experiences_plan = await asyncio.gather(
        _run_specialist(
            safety_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Destination hint: {grounding.destination_hint}
                Safety grounding: {grounding.safety_summary}
                Advisory level: {grounding.advisory_level}
                Advisory source: {grounding.advisory_source_url}
                """
            ).strip(),
        ),
        _run_specialist(
            logistics_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Origin city: {user_request.origin_city or 'Not provided'}
                Trip dates: {check_in.isoformat()} to {check_out.isoformat()} ({trip_length_days} days)
                Number of travelers: {user_request.traveler_count}
                Weather grounding: {grounding.weather_summary}
                Transport grounding: {grounding.local_transport_notes}
                External API data: {grounding.external_api_snippets or 'No external API snippets provided'}

                Hotel Search Results:
                {hotel_search_results}

                Flight Search Results:
                {flight_search_results}

                Visa Requirements:
                {visa_search_results}

                Airport Recommendations:
                {airport_search_results}

                Cruise Search Results:
                {cruise_search_results}
                """
            ).strip(),
        ),
        _run_specialist(
            experiences_agent,
            dedent(
                f"""
                Traveler request: {user_request.request_text}
                Destination hint: {grounding.destination_hint}
                Climate/weather grounding: {grounding.weather_summary}
                """
            ).strip(),
        ),
    )

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
            - Use all relevant provided API evidence in each option: Duffel Stays,
              Duffel Flights, Visa Requirements, and Tripadvisor Cruises.
            - Apply the deterministic ranking rubric exactly as provided in the user context.
            - For hotels and activities, explicitly reference rating confidence based on score + review volume.
            - If an API source is missing/unavailable, state that clearly and continue with available evidence.
            - Include a dedicated safety section with explicit caution areas and behavior recommendations.
            - Keep luxury feel while honoring budget constraints.
            - Include a section titled 'Deterministic Scoring Table' that shows, for each itinerary:
              rating_quality, review_confidence, safety, budget_fit, final_score, and tie-break explanation.
            - End with two sections:
              1) 'API Evidence Coverage' listing what was used from each provided API
              2) 'Grounding Sources' list containing URLs.
            """
        ).strip(),
    )

    final_markdown = await final_agent.run(
        dedent(
            f"""
            Traveler request:
            {user_request.request_text}

            Inputs from specialist agents:
            Safety brief:
            {safety_brief}

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
                            Duffel Stays, Duffel Flights, Visa Requirements, Tripadvisor Cruises
                        - Deterministic rubric to apply exactly:
                            {_deterministic_ranking_rubric()}
            - Source URL that must be included: https://travel.state.gov/content/travel.html
            """
        ).strip()
    )

    return {
        "destination": grounding.destination_hint,
        "itinerary_markdown": str(final_markdown),
        "safety_summary": grounding.safety_summary,
        "advisory_source_url": grounding.advisory_source_url,
    }
