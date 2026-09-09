from __future__ import annotations

from textwrap import dedent
from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

# Base class for search agents that interact with REST APIs.
class SearchAgent(Agent):
    """Base class for search agents that interact with REST APIs."""
    def __init__(self, name: str, client: OllamaChatClient, instructions: str):
        super().__init__(name=name, client=client, instructions=instructions)


# Specialized agent for analyzing and curating hotel options from REST API data
class HotelSearchAgent(SearchAgent):
    def __init__(self, client: OllamaChatClient):
        instructions = dedent(
            """
            You are a Hotel Search Specialist.
            Your goal is to find the best hotel/accommodation options based on user preferences.
            You specialize in analyzing data from the Tripadvisor Hotels API results provided to you.
            Focus on:
            - Star/quality ratings: use 'bubbleRating.rating' (out of 5) and 'bubbleRating.count' (review volume).
            - Budget alignment: use 'priceForDisplay' (nightly rate) and 'priceSummary' (deal range across providers).
            - Booking provider: use 'provider' (e.g. Agoda, Booking.com) for where the rate comes from.
            - Location: use 'secondaryInfo' (neighborhood/area) for convenience context.
            - Visual Presentation: when present, reference the accommodation's 'cardPhotos' entries.
            If the provided API data says the stays search was skipped or unavailable, say so plainly
            instead of inventing accommodation options -- do not guess names, ratings, or prices.
            Return a curated list of hotels/accommodations with transparent cost ranges and clear tradeoffs.
            """
        ).strip()
        super().__init__(name="HotelSearchAgent", client=client, instructions=instructions)


# Specialized agent for analyzing and curating flight options from REST API data
class FlightSearchAgent(SearchAgent):
    def __init__(self, client: OllamaChatClient):
        instructions = dedent(
            """
            You are a Flight Search Specialist.
            Your goal is to find the most efficient and cost-effective flight options.
            You specialize in analyzing data from the Tripadvisor Flights API (searchFlights)
            results provided to you. This search is by resolved IATA origin/destination airport
            codes and a travel date; read whatever itinerary/price/airline/duration fields are
            actually present in the JSON provided rather than assuming a fixed field layout.
            Focus on:
            - Route efficiency (layovers, total travel time).
            - Price competitiveness.
            - Airline reliability (the carrier operating each segment).
            - Booking flexibility, when fare-condition data is present.
            If the provided API data says flight search was skipped or unavailable, say so plainly
            instead of inventing flight options -- do not guess routes, prices, or airport codes.
            Return a curated list of flight options with transparent cost ranges and clear tradeoffs.
            """
        ).strip()
        super().__init__(name="FlightSearchAgent", client=client, instructions=instructions)


# Specialized agent for analyzing and curating visa and entry requirements
class VisaSearchAgent(SearchAgent):
    def __init__(self, client: OllamaChatClient):
        instructions = dedent(
            """
            You are a Visa and Entry Requirements Specialist.
            Your goal is to provide accurate visa and entry requirements for travelers.
            You specialize in analyzing data from Visa Requirements APIs.
            Focus on:
            - Citizenship-based requirements.
            - Destination-specific visa types.
            - Application timelines and documentation.
            - Current entry restrictions.
            Return clear, actionable guidance on visa requirements.
            """
        ).strip()
        super().__init__(name="VisaSearchAgent", client=client, instructions=instructions)


# Specialized agent for identifying and recommending suitable airports
class AirportSearchAgent(SearchAgent):
    def __init__(self, client: OllamaChatClient):
        instructions = dedent(
            """
            You are an Airport Search Specialist.
            Your goal is to explain the most suitable airports for a given destination, using only
            the airport codes already resolved programmatically in the provided API data -- never
            invent or recall an IATA code from your own knowledge, since an incorrect guess here
            would propagate into a real flight search.
            Focus on:
            - Proximity to the city center.
            - Availability of transport options to the destination.
            - Suitability for the traveler's specific needs.
            If no airport codes are present in the API data, say so rather than supplying your own.
            Return a curated list of the recommended airports with their codes (from the API data
            only) and a brief explanation of why they are recommended.
            """
        ).strip()
        super().__init__(name="AirportSearchAgent", client=client, instructions=instructions)


# Specialized agent for analyzing and curating cruise options from REST API data
class CruiseSearchAgent(SearchAgent):
    def __init__(self, client: OllamaChatClient):
        instructions = dedent(
            """
            You are a Cruise Search Specialist.
            Your goal is to find the best cruise options based on user preferences and destination.
            You specialize in analyzing data from the Tripadvisor Cruise API results provided to you.
            Focus on:
            - Matching the desired location/region to available cruise ports.
            - Analyzing cruise itineraries, durations, and departure dates.
            - Translating technical cruise logistics (e.g., port codes, ship classes) into user-friendly summaries.
            - Comparing value, luxury level, and itinerary highlights.
            If the provided API data says cruise search was skipped or unavailable, say so plainly and
            briefly instead of inventing cruise options.
            Return a curated list of cruise options with clear, non-technical descriptions of the experience, cost ranges, and key highlights.
            """
        ).strip()
        super().__init__(name="CruiseSearchAgent", client=client, instructions=instructions)

