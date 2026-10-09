from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Upper bound on how long a single backoff may sleep. A server-sent Retry-After is honored up
# to this limit only: the wait happens while holding the per-host lock, so an unbounded value
# would stall every other request to that host for as long as the server asked.
_MAX_BACKOFF_SECONDS = 30.0

# Serializes requests per host. Verified live (2026-09-03) against the RapidAPI hosts used
# here: firing two concurrent requests at the same host reliably triggers 429s that exhaust
# the retry budget and come back as empty/incomplete data rather than raising -- silently
# degrading result quality instead of failing loudly. A different host is unaffected, so this
# only serializes calls that would actually contend for the same upstream rate limit.
#
# Locks are kept per event loop: Streamlit's app.py calls asyncio.run() on every script rerun,
# which creates a brand-new loop each time, and an asyncio.Lock that has been contended on one
# loop raises "bound to a different event loop" if used on another. Keying on the loop object
# itself (not id(loop), which Python only guarantees unique among objects alive at the same
# time, so a new loop could reuse a closed loop's id and be handed its stale lock) rules that
# out. A WeakKeyDictionary is not enough on its own here: a used asyncio.Lock holds a strong
# reference to its loop, so the entries would never be freed. Instead, closed loops' entries
# are pruned on each lookup. Streamlit runs each browser session's script on its own thread,
# so the shared map is guarded by a threading.Lock.
_host_locks: dict[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = {}
_host_locks_guard = threading.Lock()


def _host_lock_for(url: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    host = httpx.URL(url).host
    with _host_locks_guard:
        for stale_loop in [known for known in _host_locks if known.is_closed()]:
            del _host_locks[stale_loop]
        loop_locks = _host_locks.setdefault(loop, {})
        if host not in loop_locks:
            loop_locks[host] = asyncio.Lock()
        return loop_locks[host]


# Shared retry loop for GET and POST: bounded attempts, backoff on transport errors and
# retryable status codes (honoring Retry-After, capped), immediate raise on any other 4xx/5xx.
async def _request_with_retry(
    client: httpx.AsyncClient, method: str, url: str, *, max_attempts: int, **request_kwargs: Any
) -> httpx.Response:
    async with _host_lock_for(url):
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await client.request(method, url, **request_kwargs)
            except httpx.TransportError:
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))
                continue

            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt >= max_attempts:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt - 1)
            delay = min(delay, _MAX_BACKOFF_SECONDS)
            logger.warning(
                "Retryable status %s from %s (attempt %s/%s), backing off %.1fs",
                response.status_code,
                url,
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)


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
    return await _request_with_retry(
        client, "GET", url, max_attempts=max_attempts, params=params, headers=headers
    )


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
    return await _request_with_retry(
        client, "POST", url, max_attempts=max_attempts, json=json, data=data, headers=headers
    )


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
