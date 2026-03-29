from __future__ import annotations

import os
from textwrap import dedent

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from config import settings
from .data_sources import gather_grounding_packet
from .schemas import UserTripRequest


def _configure_model_env() -> None:
    for key, value in settings.openai_env.items():
        os.environ[key] = value


def _deterministic_ranking_rubric() -> str:
        return dedent(
                """
                Deterministic ranking rubric (0-100):
                - Rating quality score (0-10) -> weight 40%
                    Use normalized rating values from Booking.com/Tripadvisor/other provided APIs.
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


async def build_itineraries(user_request: UserTripRequest) -> dict[str, str]:
    _configure_model_env()
    grounding = await gather_grounding_packet(user_request.request_text)

    client = OpenAIChatClient()

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
            Use all relevant provided APIs when available: Booking.com, Tripadvisor, Flights Scraper Sky,
            Google Flights, and Visa Requirements.
            Prioritize higher-rated options using both source ratings and customer review quality signals
            (review score, review volume, recency, consistency).
            Prefer options with strong ratings in both Booking.com and Tripadvisor when both are available.
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

    safety_brief = await safety_agent.run(
        dedent(
            f"""
            Traveler request: {user_request.request_text}
            Destination hint: {grounding.destination_hint}
            Safety grounding: {grounding.safety_summary}
            Advisory level: {grounding.advisory_level}
            Advisory source: {grounding.advisory_source_url}
            """
        ).strip()
    )

    logistics_plan = await logistics_agent.run(
        dedent(
            f"""
            Traveler request: {user_request.request_text}
            Origin city: {user_request.origin_city or 'Not provided'}
            Trip length (days): {user_request.trip_length_days or 'Not provided'}
            Number of travelers: {user_request.traveler_count}
            Weather grounding: {grounding.weather_summary}
            Transport grounding: {grounding.local_transport_notes}
            External API data: {grounding.external_api_snippets or 'No external API snippets provided'}
            """
        ).strip()
    )

    experiences_plan = await experiences_agent.run(
        dedent(
            f"""
            Traveler request: {user_request.request_text}
            Destination hint: {grounding.destination_hint}
            Climate/weather grounding: {grounding.weather_summary}
            """
        ).strip()
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
                        - Use all relevant provided API evidence in each option: Booking.com, Tripadvisor,
                            Flights Scraper Sky, Google Flights, and Visa Requirements.
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
                            Booking.com, Tripadvisor, Flights Scraper Sky, Google Flights, Visa Requirements
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
