from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# Used whenever a trip length wasn't given (and none was parsed from the user's prompt) --
# long enough to plan a real itinerary/weather-and-activity window around, short enough to
# be a neutral default rather than an assumption about the traveler's intent. Shared so the
# itinerary build and the destination-selection insight checks never disagree on trip length.
DEFAULT_TRIP_LENGTH_DAYS = 5


# Represents the initial request from the user, including their natural language prompt and key constraints
class UserTripRequest(BaseModel):
    request_text: str = Field(min_length=10)
    origin_city: str | None = None
    trip_length_days: int | None = None
    traveler_count: int = 1
    travel_start_date: date | None = None
    # Total budget understood from the request (see request_understanding.py); None = not given,
    # which makes the itinerary options low/medium/high budget tiers.
    budget_usd: float | None = None


# One candidate destination resolved from the user's free-text prompt via geocoding.
# Carries enough structure (country, coordinates) to disambiguate same-named places
# (e.g. Paris, France vs. Paris, Texas) when resolving airport/hotel codes later.
class DestinationCandidate(BaseModel):
    name: str
    country: str
    country_code: str
    admin1: str | None = None
    latitude: float
    longitude: float
    # GeoNames feature code (e.g. "PPLC" = capital, "PCLI" = country) and population, as
    # returned by the geocoder -- used to judge whether a match is a plausible destination.
    feature_code: str | None = None
    population: int | None = None

    # "Name, Region, Country" -- the unambiguous display name.
    @property
    def label(self) -> str:
        region = f", {self.admin1}" if self.admin1 else ""
        return f"{self.name}{region}, {self.country}"


# One real airport near a destination or origin, resolved by great-circle distance from the
# OurAirports public dataset (see location_resolution.find_nearest_airports) -- never an
# LLM-guessed code.
class NearbyAirport(BaseModel):
    iata_code: str
    name: str
    municipality: str | None = None
    country_code: str = ""
    distance_km: float


# Provider-specific ids/codes resolved for the user's confirmed destination selection.
# These are what actually get passed to search APIs -- never LLM-guessed. All are Booking.com
# ids (see location_resolution.py): `destination_flight_location_id`/`origin_flight_location_id`
# come from flights/searchDestination (e.g. "JFK.AIRPORT") and are required by
# flights/searchFlights. `hotel_dest_id`/`hotel_search_type` come as a matched pair from
# hotels/searchDestination and are both required, together, by hotels/searchHotels.
class LocationCodes(BaseModel):
    destination_airport_codes: list[str] = Field(default_factory=list)
    origin_airport_codes: list[str] = Field(default_factory=list)
    destination_flight_location_id: str | None = None
    origin_flight_location_id: str | None = None
    hotel_dest_id: str | None = None
    hotel_search_type: str | None = None


# A consolidated data packet containing all external grounding information used by the agents to build itineraries
class GroundingPacket(BaseModel):
    destination_hint: str
    destination_country_code: str = ""
    weather_summary: str
    safety_summary: str
    advisory_level: str
    advisory_source_url: str
    local_transport_notes: str
    external_api_snippets: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Destination-selection insight checks (run as soon as the user confirms a
# destination + travel date, ahead of full itinerary generation).
# ---------------------------------------------------------------------------


# Result of checking the live U.S. Department of State travel advisories feed
# (cadataapi.state.gov) for the confirmed destination's country.
class AdvisoryCheck(BaseModel):
    found: bool = False
    active_warning: bool = False
    level: int | None = None
    title: str | None = None
    updated: date | None = None
    link: str | None = None
    summary_snippet: str | None = None
    source_url: str = "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html"
    error: str | None = None


# Result of the weather timeframe check: a multi-year historical average for the
# requested travel window, plus (when travel starts within 14 days) actual forecast
# coverage for that near-term window.
class WeatherOutlook(BaseModel):
    historical_avg_high_c: float | None = None
    historical_avg_low_c: float | None = None
    historical_years_used: list[int] = Field(default_factory=list)
    is_near_term: bool = False
    near_term_forecast_summary: str | None = None
    error: str | None = None


# One nearby attraction/sightseeing point sourced from OpenStreetMap, with a
# best-effort comment on whether it looks appropriate (i.e. open) during the
# traveler's requested timeframe.
class ActivitySuggestion(BaseModel):
    name: str
    category: str
    appropriateness_comment: str
    raw_opening_hours: str | None = None
    map_link: str | None = None
    # Number of Wikipedia/Wikimedia language editions covering the place (Wikidata sitelinks);
    # None when the place has no Wikidata item or the lookup was unavailable.
    sitelink_count: int | None = None
    # Sightseeing interests from the traveler's request that this place matches (e.g. "art").
    matched_interests: list[str] = Field(default_factory=list)
    # Plain-language reason it made the list, built only from the signals above.
    why_listed: str = ""


# Sightseeing split two ways: `notable` holds widely documented landmarks, `lesser_known` holds
# less-documented places that still show a concrete sign of being worth a visit. Both lists are
# ranked toward `interests` -- sightseeing topics found in the traveler's own request.
class ActivitySearchResult(BaseModel):
    notable: list[ActivitySuggestion] = Field(default_factory=list)
    lesser_known: list[ActivitySuggestion] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    # Set when notability couldn't be measured and a coarser signal was used instead.
    notability_note: str | None = None
    source_note: str = (
        "Source: OpenStreetMap contributors via Geoapify (Powered by Geoapify), with notability from "
        "Wikidata (number of Wikipedia language editions covering each place) -- community-maintained "
        "data, not an official attractions directory. Confirm current hours/status before visiting."
    )
    error: str | None = None


# ---------------------------------------------------------------------------
# "I don't know where I want to go" destination discovery: the user describes
# a vibe/budget/travel-time envelope instead of naming a place, and the app
# proposes real, independently-verified candidates.
# ---------------------------------------------------------------------------


# Structured constraints parsed (deterministically, via regex -- not an LLM guess)
# out of a free-text request that doesn't name a specific destination.
class TripConstraints(BaseModel):
    raw_text: str
    budget_usd: float | None = None
    trip_length_days: int | None = None
    max_flight_hours: float | None = None
    max_drive_hours: float | None = None
    wants_warm: bool = False
    wants_beach: bool = False
    wants_pool_resort: bool = False
    wants_food_scene: bool = False
    wants_entertainment: bool = False
    wants_hiking: bool = False
    wants_low_cost: bool = False
    avoid_cultural: bool = False
    # Month (1-12) named without a specific day, e.g. "a week in March".
    travel_month: int | None = None
    traveler_count: int | None = None


# One destination proposed during discovery, with every displayed fact independently
# verified against a real source (never taken on the LLM's word) -- see
# destination_discovery.py. `rationale` lists verified reasons it was kept; `caveats`
# lists anything that could NOT be verified or was estimated rather than measured.
class DiscoveredDestination(BaseModel):
    destination: DestinationCandidate
    distance_km: float
    estimated_flight_hours: float | None = None
    drive_hours: float | None = None
    drive_hours_is_estimate: bool = True
    historical_avg_high_c: float | None = None
    historical_avg_low_c: float | None = None
    weather_years_used: list[int] = Field(default_factory=list)
    has_beach_nearby: bool | None = None
    restaurant_count_nearby: int | None = None
    nightlife_count_nearby: int | None = None
    score: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# Outcome of one discovery run: the constraints used, the ranked surviving candidates, and
# counts of how many were considered and why the rest were dropped. `error` is set (and
# `candidates` empty) when discovery couldn't run at all, e.g. the origin couldn't be geocoded.
class DestinationDiscoveryResult(BaseModel):
    constraints: TripConstraints
    origin_label: str | None = None
    candidate_source: str = ""
    candidates: list[DiscoveredDestination] = Field(default_factory=list)
    candidates_considered: int = 0
    candidates_rejected_distance: int = 0
    candidates_rejected_weather: int = 0
    error: str | None = None


# Everything understood from a free-text request at intake (see request_understanding.py):
# rule-based parsing first, then -- only when the rules are unsure -- an LLM extraction whose
# every value had to be backed by a verbatim quote from the request (and, for destinations, a
# real geocoding match) before being accepted.
class ParsedTripRequest(BaseModel):
    constraints: TripConstraints
    candidates: list[DestinationCandidate] = Field(default_factory=list)
    start_date: date | None = None
    open_ended: bool = False
    llm_status: str = "not needed"
    # field name -> "rules" | "llm", for every field that ended up with a value.
    field_sources: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
