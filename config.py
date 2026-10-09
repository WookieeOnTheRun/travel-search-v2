from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


# Loads the first env file found (TRAVEL_AGENT_ENV_FILE, then travel_agent/main.env, then
# .env). override=False (python-dotenv's default) means a variable already set in the real
# process environment wins over the file -- so on a deployed host, secrets/config set by the
# platform are never silently replaced by a stale env file shipped alongside the code.
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
            load_dotenv(dotenv_path=candidate, override=False)
            return candidate

    load_dotenv(override=False)
    return None


# One-time logging setup for the whole process, done here because config is the first project
# module every other one imports -- so it runs before any log line is emitted. Python's root
# logger has no handler by default: without this, only WARNING+ would reach stderr (via the
# last-resort handler, with no timestamps) and INFO lines would never print. basicConfig is a
# no-op once the root logger has a handler, so Streamlit's reruns can't add duplicates. httpx
# is pinned to WARNING: at INFO it logs every request's full URL, and for this app the query
# string carries the traveler's free-text request.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

ENV_FILE_IN_USE = _load_environment()
# Logged once per process at import, server-side only -- the env file's path is deliberately
# not shown in the UI, since it discloses the server's filesystem layout to every visitor.
logging.getLogger(__name__).info(
    "Env file in use: %s", ENV_FILE_IN_USE or "none (process environment only)"
)


# A `KEY=` line in an env file sets the variable to "" (python-dotenv behaviour, verified), and
# os.getenv(key, default) returns that "" rather than the default -- so every setting with a
# default goes through this, and a blank value in a copied .env.example can't produce a broken
# "https:///..." URL or a float("") crash at import.
def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


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
    ollama_model: str = _env("OLLAMA_MODEL", "gemma4:cloud")
    ollama_host: str = _env("OLLAMA_HOST", "https://ollama.com")
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
    request_timeout_seconds: float = float(_env("REQUEST_TIMEOUT_SECONDS", "20"))

    default_currency: str = _env("DEFAULT_CURRENCY", "USD")
    default_locale: str = _env("DEFAULT_LOCALE", "en-US")
    default_citizenship_country_code: str = _env("DEFAULT_CITIZENSHIP_COUNTRY_CODE", "US")

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
    rapidapi_booking_host: str = _env("RAPIDAPI_BOOKING_HOST", "booking-com15.p.rapidapi.com")
    rapidapi_visa_host: str = _env("RAPIDAPI_VISA_HOST", "visa-requirement.p.rapidapi.com")

    # SQLite file holding per-session search runs and itineraries (see
    # travel_agent/persistence.py). Nothing here is meant to outlive the host: on a container
    # or host without a persistent disk, a restart wipes every session's itineraries.
    session_db_path: str = _env(
        "SESSION_DB_PATH", str(Path(__file__).resolve().parent / "data" / "travel_sessions.db")
    )


settings = Settings()
