from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:31b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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

    # Flights (offer_requests/offers, places/suggestions for airport resolution) and Stays
    # (accommodation search) both run against Duffel's REST API -- verified against the
    # official docs at https://duffel.com/docs/api on 2026-09-04. The same base URL and
    # Duffel-Version header are used for both test (`duffel_test_...`) and live tokens; only
    # the token itself selects the environment. See travel_agent/http_utils.py's
    # duffel_headers() and travel_agent/location_resolution.py / data_sources.py for the
    # exact endpoints.
    duffel_api_key: str = os.getenv("DUFFEL_API_KEY", "")
    duffel_api_base_url: str = os.getenv("DUFFEL_API_BASE_URL", "https://api.duffel.com")
    duffel_api_version: str = os.getenv("DUFFEL_API_VERSION", "v2")

    # Each of these hosts + the fixed endpoint paths in travel_agent/location_resolution.py
    # and travel_agent/data_sources.py were verified against the live RapidAPI products on
    # 2026-09-03 (see session notes). Only the host is configurable: a RapidAPI "host" maps
    # 1:1 to one product's fixed route structure, so letting the *path* be configured separately
    # (as an older version of this file did) just reintroduces the base-URL-with-no-path bug --
    # a mismatched host is the only thing swapping the underlying product actually requires.
    #
    # Flights and hotel/accommodation search moved to the Duffel API above -- these two RapidAPI
    # hosts remain only for cruise search (Tripadvisor) and visa requirements, neither of which
    # Duffel covers.
    rapidapi_tripadvisor_host: str = os.getenv("RAPIDAPI_TRIPADVISOR_HOST", "tripadvisor16.p.rapidapi.com")
    rapidapi_visa_host: str = os.getenv("RAPIDAPI_VISA_HOST", "visa-requirement.p.rapidapi.com")

    @property
    def ollama_env(self) -> Dict[str, str]:
        # Field names match agent_framework_ollama.OllamaChatClient's env-based settings
        # loader, which reads env_prefix "OLLAMA_" + the field name uppercased (i.e.
        # OLLAMA_HOST / OLLAMA_MODEL) -- verified against the installed
        # agent-framework-ollama package on 2026-09-03.
        return {
            "OLLAMA_HOST": self.ollama_host,
            "OLLAMA_MODEL": self.ollama_model,
        }


settings = Settings()
