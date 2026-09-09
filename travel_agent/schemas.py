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
# These are what actually get passed to search APIs -- never LLM-guessed. `hotel_location_id`
# is the Tripadvisor `geoId` from hotels/searchLocation (see location_resolution.py), required
# by hotels/searchHotels for accommodation search.
class LocationCodes(BaseModel):
    destination_airport_codes: list[str] = Field(default_factory=list)
    origin_airport_codes: list[str] = Field(default_factory=list)
    hotel_location_id: str | None = None
    cruise_destination_id: str | None = None


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
    historical_avg_precipitation_probability: float | None = None
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


class ActivitySearchResult(BaseModel):
    activities: list[ActivitySuggestion] = Field(default_factory=list)
    source_note: str = (
        "Source: OpenStreetMap contributors via the Overpass API -- community-maintained data, "
        "not an official attractions directory. Confirm current hours/status before visiting."
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
    avoid_cultural: bool = False


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


class DestinationDiscoveryResult(BaseModel):
    constraints: TripConstraints
    origin_label: str | None = None
    candidate_source: str = ""
    candidates: list[DiscoveredDestination] = Field(default_factory=list)
    candidates_considered: int = 0
    candidates_rejected_distance: int = 0
    candidates_rejected_weather: int = 0
    error: str | None = None
