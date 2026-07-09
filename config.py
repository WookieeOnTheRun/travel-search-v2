from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv


# Environment loading logic to find and load .env files from multiple candidate paths
def _load_environment() -> Path | None:
    project_root = Path(__file__).resolve().parent
    explicit_env_path = os.getenv("TRAVEL_AGENT_ENV_FILE", "").strip()

    candidates = []
    if explicit_env_path:
        candidates.append(Path(explicit_env_path))

    candidates.extend(
        [
            project_root / "travel_agent" / "main.env",
            project_root / ".env",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=True)
            return candidate

    load_dotenv(override=True)
    return None



ENV_FILE_IN_USE = _load_environment()


# Standard keys used across different travel API providers for consistent lookup
LOOKUP_ROUTE_KEYS: List[str] = [
    "AIRPORT_LOOKUP",
    "FLIGHT_LOOKUP",
    "FLIGHT_DETAILS",
    "HOTEL_LOOKUP",
    "HOTEL_DETAILS",
    "HOTEL_REVIEW_SCORES",
    "CAR_RENTAL_LOCATION",
    "CAR_RENTAL_LOOKUP",
    "CAR_RENTAL_DETAILS",
    "ATTRACTION_LOOKUP",
    "ATTRACTION_DETAILS",
]


# Helper to ensure all required lookup keys are present in a provider's configuration
def _build_lookup_dictionary(defaults: Dict[str, str]) -> Dict[str, str]:
    return {key: defaults.get(key, "") for key in LOOKUP_ROUTE_KEYS}



TRIPADVISOR_API_LOOKUP_PATHS: Dict[str, str] = _build_lookup_dictionary(
    {
        "AIRPORT_LOOKUP": "flights/searchAirport?query=",
        "FLIGHT_LOOKUP": "flights/searchFlights?sourceAirportCode={xxx}&destinationAirportCode={yyy}&itineraryType=ROUND_TRIP",
        "HOTEL_LOOKUP": "hotels/searchHotels?query=",
        "HOTEL_DETAILS": "hotels/getHotelDetails",
        "HOTEL_REVIEW_SCORES": "hotels/getHotelReviews",
        "CAR_RENTAL_LOCATION": "cars/searchLocation",
        "CAR_RENTAL_LOOKUP": "cars/searchCars",
        "CAR_RENTAL_DETAILS": "cars/getCarDetails",
        "ATTRACTION_LOOKUP": "attractions/searchAttractions",
        "ATTRACTION_DETAILS": "attractions/getAttractionDetails",
    }
)


BOOKING_COM_API_LOOKUP_PATHS: Dict[str, str] = _build_lookup_dictionary(
    {
        "AIRPORT_LOOKUP": "flights/searchDestination",
        "FLIGHT_LOOKUP": "flights/searchFlights",
        "FLIGHT_DETAILS": "flights/getFlightDetails",
        "HOTEL_LOOKUP": "hotels/searchDestination",
        "HOTEL_DETAILS": "hotels/getHotelDetails",
        "HOTEL_REVIEW_SCORES": "hotels/getHotelReviews",
        "CAR_RENTAL_LOCATION": "cars/searchLocation",
        "CAR_RENTAL_LOOKUP": "cars/searchCarRentals",
        "CAR_RENTAL_DETAILS": "cars/getCarRentalDetails",
        "ATTRACTION_LOOKUP": "attractions/searchAttractions",
        "ATTRACTION_DETAILS": "attractions/getAttractionDetails",
    }
)


FLIGHTS_SCRAPER_SKY_API_LOOKUP_PATHS: Dict[str, str] = _build_lookup_dictionary(
    {
        "AIRPORT_LOOKUP": "v1/flights/searchAirport",
        "FLIGHT_LOOKUP": "v1/flights/searchFlights",
        "FLIGHT_DETAILS": "v1/flights/getFlightDetails",
        "HOTEL_LOOKUP": "v1/hotels/searchHotels",
        "HOTEL_DETAILS": "v1/hotels/getHotelDetails",
        "HOTEL_REVIEW_SCORES": "v1/hotels/getHotelReviews",
        "CAR_RENTAL_LOCATION": "v1/cars/searchLocation",
        "CAR_RENTAL_LOOKUP": "v1/cars/searchCars",
        "CAR_RENTAL_DETAILS": "v1/cars/getCarDetails",
        "ATTRACTION_LOOKUP": "v1/attractions/searchAttractions",
        "ATTRACTION_DETAILS": "v1/attractions/getAttractionDetails",
    }
)


GOOGLE_FLIGHTS_API_LOOKUP_PATHS: Dict[str, str] = _build_lookup_dictionary(
    {
        "AIRPORT_LOOKUP": "lookup/airport",
        "FLIGHT_LOOKUP": "searchFlights",
        "FLIGHT_DETAILS": "flightDetails",
        "HOTEL_LOOKUP": "searchHotels",
        "HOTEL_DETAILS": "hotelDetails",
        "HOTEL_REVIEW_SCORES": "hotelReviews",
        "CAR_RENTAL_LOCATION": "carRentalLocations",
        "CAR_RENTAL_LOOKUP": "searchCarRentals",
        "CAR_RENTAL_DETAILS": "carRentalDetails",
        "ATTRACTION_LOOKUP": "searchAttractions",
        "ATTRACTION_DETAILS": "attractionDetails",
    }
)


VISA_REQUIREMENTS_API_LOOKUP_PATHS: Dict[str, str] = _build_lookup_dictionary(
    {
        "AIRPORT_LOOKUP": "lookupAirport",
        "FLIGHT_LOOKUP": "lookupFlightRoute",
        "FLIGHT_DETAILS": "lookupFlightEntryRules",
        "HOTEL_LOOKUP": "lookupStayRules",
        "HOTEL_DETAILS": "stayRuleDetails",
        "HOTEL_REVIEW_SCORES": "hospitalityRiskScore",
        "CAR_RENTAL_LOCATION": "lookupRoadEntryPoint",
        "CAR_RENTAL_LOOKUP": "lookupRoadTravelRules",
        "CAR_RENTAL_DETAILS": "roadTravelRuleDetails",
        "ATTRACTION_LOOKUP": "lookupVisitPurpose",
        "ATTRACTION_DETAILS": "visitPurposeDetails",
    }
)


@dataclass(slots=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "phi-4")
    model_endpoint: str = os.getenv("MODEL_ENDPOINT", "http://localhost:5273/v1")
    model_api_key: str = os.getenv("MODEL_API_KEY", "local")

    travel_api_endpoints: List[str] = field(
        default_factory=lambda: [
            endpoint.strip()
            for endpoint in os.getenv("TRAVEL_API_ENDPOINTS", "").split(",")
            if endpoint.strip()
        ]
    )
    travel_api_key: str = os.getenv("TRAVEL_API_KEY", "")
    rapidapi_key: str = os.getenv("RAPIDAPI_KEY", os.getenv("TRAVEL_API_KEY", ""))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))

    default_currency: str = os.getenv("DEFAULT_CURRENCY", "USD")
    default_locale: str = os.getenv("DEFAULT_LOCALE", "en-US")
    default_citizenship_country_code: str = os.getenv("DEFAULT_CITIZENSHIP_COUNTRY_CODE", "US")

    rapidapi_tripadvisor_url: str = os.getenv("RAPIDAPI_TRIPADVISOR_URL", "")
    rapidapi_tripadvisor_host: str = os.getenv("RAPIDAPI_TRIPADVISOR_HOST", "")

    rapidapi_booking_url: str = os.getenv("RAPIDAPI_BOOKING_URL", "")
    rapidapi_booking_host: str = os.getenv("RAPIDAPI_BOOKING_HOST", "")

    rapidapi_flights_scraper_sky_url: str = os.getenv("RAPIDAPI_FLIGHTS_SCRAPER_SKY_URL", "")
    rapidapi_flights_scraper_sky_host: str = os.getenv("RAPIDAPI_FLIGHTS_SCRAPER_SKY_HOST", "")

    rapidapi_google_flights_url: str = os.getenv("RAPIDAPI_GOOGLE_FLIGHTS_URL", "")
    rapidapi_google_flights_host: str = os.getenv("RAPIDAPI_GOOGLE_FLIGHTS_HOST", "")

    rapidapi_visa_requirements_url: str = os.getenv("RAPIDAPI_VISA_REQUIREMENTS_URL", "")
    rapidapi_visa_requirements_host: str = os.getenv("RAPIDAPI_VISA_REQUIREMENTS_HOST", "")

    @property
    def openai_env(self) -> Dict[str, str]:
        return {
            "OPENAI_API_KEY": self.model_api_key,
            "OPENAI_BASE_URL": self.model_endpoint,
            "OPENAI_CHAT_MODEL_ID": self.model_name,
            "OPENAI_RESPONSES_MODEL_ID": self.model_name,
        }

    @staticmethod
    def _host_from_url(url: str) -> str:
        if not url:
            return ""
        return urlparse(url).netloc

    @property
    def rapidapi_sources(self) -> Dict[str, Dict[str, str]]:
        return {
            "Tripadvisor": {
                "url": self.rapidapi_tripadvisor_url,
                "host": self.rapidapi_tripadvisor_host or self._host_from_url(self.rapidapi_tripadvisor_url),
            },
            "Booking.com": {
                "url": self.rapidapi_booking_url,
                "host": self.rapidapi_booking_host or self._host_from_url(self.rapidapi_booking_url),
            },
            "Flights Scraper Sky": {
                "url": self.rapidapi_flights_scraper_sky_url,
                "host": self.rapidapi_flights_scraper_sky_host
                or self._host_from_url(self.rapidapi_flights_scraper_sky_url),
            },
            "Google Flights": {
                "url": self.rapidapi_google_flights_url,
                "host": self.rapidapi_google_flights_host
                or self._host_from_url(self.rapidapi_google_flights_url),
            },
            "Visa Requirements": {
                "url": self.rapidapi_visa_requirements_url,
                "host": self.rapidapi_visa_requirements_host
                or self._host_from_url(self.rapidapi_visa_requirements_url),
            },
        }

    @property
    def rapidapi_lookup_paths(self) -> Dict[str, Dict[str, str]]:
        return {
            "Tripadvisor": dict(TRIPADVISOR_API_LOOKUP_PATHS),
            "Booking.com": dict(BOOKING_COM_API_LOOKUP_PATHS),
            "Flights Scraper Sky": dict(FLIGHTS_SCRAPER_SKY_API_LOOKUP_PATHS),
            "Google Flights": dict(GOOGLE_FLIGHTS_API_LOOKUP_PATHS),
            "Visa Requirements": dict(VISA_REQUIREMENTS_API_LOOKUP_PATHS),
        }


settings = Settings()
