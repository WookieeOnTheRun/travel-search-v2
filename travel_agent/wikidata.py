from __future__ import annotations

import httpx

from .http_utils import get_with_retry

# ---------------------------------------------------------------------------
# Wikidata (https://www.wikidata.org/w/api.php, action=wbgetentities) -- used as a measured
# notability signal for places: the number of sitelinks an item has is the number of wikis
# (mostly language editions of Wikipedia) with an article about it. Verified live 2026-10-19
# on Geoapify sightseeing results: Colosseo 148, Castelo de São Jorge 31, Denver Art Museum 25,
# while small local memorials and statues sit at 0-4. No API key; each call takes ~0.1-0.3 s.
#
# Wikimedia's robot policy (https://w.wiki/4wJS) rejects default library User-Agents with a 403
# (verified: httpx's default UA is refused, curl's is not); a UA naming the client, a contact
# URL and the library is accepted.
# ---------------------------------------------------------------------------

_API_URL = "https://www.wikidata.org/w/api.php"
_USER_AGENT = f"travel-search-v2/0.1 (https://github.com/WookieeOnTheRun/travel-search-v2) httpx/{httpx.__version__}"
# wbgetentities' per-request cap on ids for clients without bot rights.
_MAX_IDS_PER_REQUEST = 50


async def fetch_sitelink_counts(client: httpx.AsyncClient, item_ids: list[str]) -> dict[str, int]:
    """Sitelink count per Wikidata item id. Ids Wikidata doesn't know (deleted items) are left
    out. Raises on any other failure -- callers fall back to a coarser signal."""
    unique_ids = list(dict.fromkeys(item_ids))
    counts: dict[str, int] = {}
    for start in range(0, len(unique_ids), _MAX_IDS_PER_REQUEST):
        batch = unique_ids[start : start + _MAX_IDS_PER_REQUEST]
        while batch:
            payload = (await _get_entities(client, batch)).json()
            error = payload.get("error") or {}
            # One unknown id fails the whole request (HTTP 200 with an error body, verified
            # live), and OSM's wikidata tags can point at since-deleted items: drop it, retry.
            if error.get("code") == "no-such-entity" and error.get("id") in batch:
                batch.remove(error["id"])
                continue
            if error:
                raise RuntimeError(f"Wikidata error: {error.get('code') or 'unknown'}")
            for item_id, entity in (payload.get("entities") or {}).items():
                if "missing" not in entity:
                    counts[item_id] = len(entity.get("sitelinks") or {})
            break
    return counts


# One wbgetentities call returning only the sitelinks of up to 50 items.
async def _get_entities(client: httpx.AsyncClient, item_ids: list[str]) -> httpx.Response:
    return await get_with_retry(
        client,
        _API_URL,
        params={"action": "wbgetentities", "ids": "|".join(item_ids), "props": "sitelinks", "format": "json"},
        headers={"User-Agent": _USER_AGENT},
        max_attempts=2,
    )
