from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Serializes requests per host, since _host_lock_for keys purely on hostname. Verified live
# (2026-09-03) against the RapidAPI hosts used here (Tripadvisor, visa requirements): firing two concurrent
# requests at the same host reliably triggers 429s that exhaust the retry budget and come
# back as empty/incomplete data rather than raising -- silently degrading result quality
# instead of failing loudly. A different host is unaffected, so this only serializes calls
# that would actually contend for the same upstream rate limit.
#
# Keyed by (running event loop id, host) rather than just host: callers here (e.g.
# Streamlit's app.py) call asyncio.run() fresh on every script rerun, which creates
# a brand-new event loop each time. An asyncio.Lock is bound to the loop it's first
# used on, so a plain host-keyed dict -- which outlives any single asyncio.run() call
# -- would hand a later rerun a lock object still bound to a previous, now-closed
# loop and crash with "bound to a different event loop" the moment two concurrent
# calls to the same host contend for it (verified live -- this is exactly what
# happened to the concurrent historical-weather calls in destination_insights.py
# once Streamlit was actually run through more than one rerun).
_host_locks: dict[tuple[int, str], asyncio.Lock] = {}


async def _host_lock_for(url: str) -> asyncio.Lock:
    host = httpx.URL(url).host
    key = (id(asyncio.get_running_loop()), host)
    lock = _host_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _host_locks[key] = lock
    return lock


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_attempts: int = 3,
) -> httpx.Response:
    """GET with bounded retries. Honors Retry-After on 429 and backs off on 5xx/transport errors.

    Non-retryable failures (4xx other than 429) raise immediately.
    """
    async with await _host_lock_for(url):
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.TransportError:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))
                continue

            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response

            if attempt >= max_attempts:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt - 1)
            logger.warning(
                "Retryable status %s from %s (attempt %s/%s), backing off %.1fs",
                response.status_code,
                url,
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_attempts: int = 3,
) -> httpx.Response:
    """POST with the same bounded-retry and per-host-serialization semantics as get_with_retry.

    Pass exactly one of `json` (JSON body) or `data` (form-encoded body) depending on what the
    target API expects.
    """
    async with await _host_lock_for(url):
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await client.post(url, json=json, data=data, headers=headers)
            except httpx.TransportError:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))
                continue

            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response

            if attempt >= max_attempts:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt - 1)
            logger.warning(
                "Retryable status %s from %s (attempt %s/%s), backing off %.1fs",
                response.status_code,
                url,
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)


def describe_error(ex: Exception) -> str:
    """Renders an exception for logs/prompts without its query string.

    httpx.HTTPStatusError and httpx.RequestError both embed the full request
    URL -- including query params -- in str(ex). Our RapidAPI/travel-API calls
    put the user's raw free-text request or destination in query params, so an
    un-redacted error message written to logs (or echoed back into an LLM
    prompt) would carry that text into wherever those logs end up. Strip the
    query string, keep the path.
    """
    request = getattr(ex, "request", None)
    if request is not None:
        try:
            safe_url = f"{request.url.scheme}://{request.url.host}{request.url.path}"
        except Exception:
            safe_url = "<unknown url>"
        if isinstance(ex, httpx.HTTPStatusError):
            return f"HTTP {ex.response.status_code} from {safe_url}"
        return f"{type(ex).__name__} contacting {safe_url}"
    return f"{type(ex).__name__}: {ex}"


def rapidapi_headers(host: str) -> dict[str, str]:
    return {
        "x-rapidapi-key": settings.rapidapi_key,
        "x-rapidapi-host": host,
    }
