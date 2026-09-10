from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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


@dataclass(slots=True)
class Settings:
    # Ollama's cloud-hosted API (verified against https://docs.ollama.com/api/authentication
    # on 2026-09-08). `ollama_host` is the *root* host handed to the ollama-python SDK's
    # AsyncClient(host=...) -- that client's chat()/generate() calls already prepend
    # "/api/<endpoint>" themselves (confirmed by reading ollama/_client.py and by building a
    # real httpx.Request against both forms), so this must be "https://ollama.com" and NOT
    # "https://ollama.com/api" -- the latter produces a broken double "/api/api/..." path.
    # The documented raw-HTTP base URL ("https://ollama.com/api/...", as in the official curl
    # example) only applies to direct HTTP calls, not to this SDK's host parameter.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
    ollama_host: str = os.getenv("OLLAMA_HOST", "https://ollama.com")
    # Ollama Cloud authenticates every request with `Authorization: Bearer <OLLAMA_API_KEY>`
    # (see https://docs.ollama.com/api/authentication). Create a key at
    # https://ollama.com/settings/keys. Server-side only: never log it, never render it in
    # the Streamlit UI, and never paste a real value into chat, a PR, or an issue.
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")

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

    # Each of these hosts + the fixed endpoint paths in travel_agent/location_resolution.py
    # and travel_agent/data_sources.py were verified against the live RapidAPI products on
    # 2026-09-03 (see session notes). Only the host is configurable: a RapidAPI "host" maps
    # 1:1 to one product's fixed route structure, so letting the *path* be configured separately
    # (as an older version of this file did) just reintroduces the base-URL-with-no-path bug --
    # a mismatched host is the only thing swapping the underlying product actually requires.
    #
    # Flight and hotel/accommodation search both run against the Booking.com RapidAPI product
    # (`booking-com15.p.rapidapi.com`), verified live end-to-end on 2026-09-10: flights via
    # flights/searchDestination -> flights/searchFlights, hotels via a matching two-step flow,
    # hotels/searchDestination (returns a `dest_id`/`search_type` pair per candidate place) ->
    # hotels/searchHotels (takes that exact `dest_id`/`search_type` pair). This product replaced
    # Tripadvisor for both: Tripadvisor's flight endpoints (searchAirport, searchFlights) never
    # returned usable results on this subscription (searchAirport: empty `data` for every query;
    # searchFlights: HTTP 200 with a `{"status": false, ...}` error body for every valid
    # request), and Tripadvisor is no longer used for hotels either so the app depends on a
    # single flight+hotel provider instead of two.
    rapidapi_booking_host: str = os.getenv("RAPIDAPI_BOOKING_HOST", "booking-com15.p.rapidapi.com")
    rapidapi_visa_host: str = os.getenv("RAPIDAPI_VISA_HOST", "visa-requirement.p.rapidapi.com")


settings = Settings()
