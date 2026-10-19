from __future__ import annotations

import re
from typing import Any

import httpx

from config import settings
from .http_utils import get_with_retry

# ---------------------------------------------------------------------------
# Geoapify Places API (https://apidocs.geoapify.com/docs/places/) -- OpenStreetMap-based place
# search, replacing the public Overpass API. Overpass's public instance was verified
# (2026-10-19) to be at capacity for automated use: 502 "at capacity" or 35-70 s queues on
# every call, and its operators propose ~50 requests/day per application as a fair share.
#
# Free plan (https://www.geoapify.com/pricing, checked 2026-10-19): 3,000 credits/day, up to
# 5 requests/s, commercial use allowed with "Powered by Geoapify" + OpenStreetMap attribution.
# A Places request costs 1 credit per 20 places returned, so callers ask only for as many
# results as they use. Requests to one host are already serialized by http_utils, and each
# takes ~0.4-1.3 s (measured), which keeps this well under 5 requests/s without extra pacing.
#
# Category names below were each checked live against the API (an unknown category is
# rejected with HTTP 400).
# ---------------------------------------------------------------------------

_PLACES_URL = "https://api.geoapify.com/v2/places"

ATTRIBUTION = "Powered by Geoapify (geoapify.com) · Map data © OpenStreetMap contributors"

_WIKIDATA_ID = re.compile(r"Q[1-9]\d*")

# Geoapify's one-letter OSM element types -> the path segment openstreetmap.org links use.
_OSM_TYPES = {"n": "node", "w": "way", "r": "relation"}


# Raised when Geoapify can't be used at all (no API key configured).
class GeoapifyUnavailable(RuntimeError):
    pass


async def search_places(
    client: httpx.AsyncClient,
    categories: str,
    *,
    latitude: float,
    longitude: float,
    radius_m: int,
    limit: int,
    nearest_first: bool = False,
    named_only: bool = False,
) -> list[dict[str, Any]]:
    """Returns the `properties` of each place within `radius_m` of the point.

    Without `nearest_first` the API returns places in no useful order (verified 2026-10-19: the
    first 100 sightseeing results within 8 km of central Rome held none of the Colosseum,
    Pantheon or Trevi Fountain). `nearest_first` adds `bias=proximity` so a capped request
    keeps the places closest to the point; `named_only` adds `conditions=named` so unnamed
    places don't use up the limit (and credits). Both verified live the same day.

    The API key travels as the `apiKey` query parameter (the documented form). It never
    reaches logs: describe_error and the retry warnings in http_utils both strip query
    strings, and the httpx logger is pinned to WARNING in config.py.
    """
    if not settings.geoapify_api_key:
        raise GeoapifyUnavailable("GEOAPIFY_API_KEY missing")
    params: dict[str, Any] = {
        "categories": categories,
        "filter": f"circle:{longitude},{latitude},{radius_m}",
        "limit": limit,
        "apiKey": settings.geoapify_api_key,
    }
    if nearest_first:
        params["bias"] = f"proximity:{longitude},{latitude}"
    if named_only:
        params["conditions"] = "named"
    response = await get_with_retry(
        client,
        _PLACES_URL,
        params=params,
        # Best-effort enrichment with a labelled "unavailable" fallback at every call site:
        # bound the worst case rather than retry a third time.
        max_attempts=2,
    )
    return [feature.get("properties") or {} for feature in response.json().get("features", [])]


def osm_tags(place: dict[str, Any]) -> dict[str, Any]:
    """The place's original OpenStreetMap tags (e.g. opening_hours, tourism, historic)."""
    return (place.get("datasource") or {}).get("raw") or {}


def wikidata_id(place: dict[str, Any]) -> str | None:
    """The place's Wikidata item id ("Q10285"), from its OSM tags or Geoapify's wiki_and_media
    block. Anything that isn't a single well-formed id (OSM sometimes holds "Q1;Q2") is ignored."""
    value = osm_tags(place).get("wikidata") or (place.get("wiki_and_media") or {}).get("wikidata")
    return value if isinstance(value, str) and _WIKIDATA_ID.fullmatch(value) else None


# openstreetmap.org link for a place's underlying OSM element, or None if it has no OSM id.
def osm_link(place: dict[str, Any]) -> str | None:
    tags = osm_tags(place)
    element_type = _OSM_TYPES.get(str(tags.get("osm_type") or ""))
    element_id = tags.get("osm_id")
    if not element_type or not element_id:
        return None
    return f"https://www.openstreetmap.org/{element_type}/{element_id}"
