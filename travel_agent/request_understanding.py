from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from textwrap import dedent

import httpx
from agent_framework import Agent

from .destination_discovery import parse_trip_constraints
from .http_utils import describe_error
from .itinerary_agent import build_ollama_client
from .location_resolution import (
    extract_destination_phrases,
    fold_text,
    is_non_place_word,
    parse_travel_start_date,
    resolve_destination_phrases,
)
from .schemas import ParsedTripRequest, TripConstraints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intake: turn a free-text request into verified structure.
#
# 1. Deterministic rules (location_resolution / destination_discovery parsers) run first.
# 2. Only when they're unsure -- no plausible destination found, or the request is
#    open-ended ("somewhere warm...") -- an LLM extracts the same fields as JSON.
# 3. Nothing the LLM says is trusted directly. Every value must come with a verbatim quote
#    from the request that actually contains it; destinations must also geocode to a real
#    place with that exact name, and a month/weekday can never be a destination. Rule values
#    always win; the LLM only fills gaps. Dates and interests stay rules-only.
# ---------------------------------------------------------------------------

_OPEN_ENDED_PATTERN = re.compile(
    r"\b(?:somewhere|anywhere|any\s+place|no\s+idea\s+where|(?:not\s+sure|don'?t\s+know|undecided)\s+where"
    r"|where\s+should\s+(?:i|we)\s+go|suggest\s+(?:a\s+)?(?:destination|place)s?)\b",
    re.IGNORECASE,
)

# A cloud model call is a few seconds normally; past this, fall back to the rule result
# rather than hold up the intake screen.
_LLM_TIMEOUT_SECONDS = 45.0
_MAX_CANDIDATES = 6

_INTEREST_FIELDS = {
    "warm": "wants_warm",
    "beach": "wants_beach",
    "pool_resort": "wants_pool_resort",
    "food": "wants_food_scene",
    "nightlife": "wants_entertainment",
    "hiking": "wants_hiking",
    "low_cost": "wants_low_cost",
    "avoid_cultural": "avoid_cultural",
}
_NUMERIC_FIELDS = ("trip_length_days", "budget_usd", "traveler_count", "travel_month")

_REQUEST_PARSING_AGENT_INSTRUCTIONS = dedent(
    """
    You extract structured facts from a traveler's trip request. Reply with ONE JSON object
    and nothing else:
    {
      "destinations": [{"name": "City, Country", "evidence": "<exact quote>"}],
      "open_ended": true | false,
      "trip_length_days": {"value": <int>, "evidence": "<exact quote>"} | null,
      "budget_usd": {"value": <number>, "evidence": "<exact quote>"} | null,
      "traveler_count": {"value": <int>, "evidence": "<exact quote>"} | null,
      "travel_month": {"value": <1-12>, "evidence": "<exact quote>"} | null,
      "interests": [{"value": "<one of: warm, beach, pool_resort, food, nightlife, hiking, low_cost, avoid_cultural>",
                     "evidence": "<exact quote>"}]
    }
    Rules:
    - "evidence" must be copied character-for-character from the request.
    - destinations: only places the traveler explicitly wants to go to. Never the departure
      city, never a month, weekday or holiday, and never a place you are suggesting yourself.
      If no place is named, return [] and set open_ended to true.
    - trip_length_days counts nights (e.g. "5 nights" -> 5, "a week" -> 7).
    - Use null / [] for anything not stated. Do not guess.
    """
).strip()


# The outermost {...} object in the model's reply (which may wrap it in prose or code fences).
def _parse_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model reply is not a JSON object")
    return payload


# Asks the parsing agent for the request's fields as JSON, each with a supporting quote. Raises
# on a timeout, a model error, or an unparseable reply; nothing returned is trusted until verified.
async def _llm_extract(request_text: str) -> dict:
    agent = Agent(
        name="RequestParsingAgent", client=build_ollama_client(), instructions=_REQUEST_PARSING_AGENT_INSTRUCTIONS
    )
    response = await asyncio.wait_for(agent.run(f"Trip request:\n{request_text}"), timeout=_LLM_TIMEOUT_SECONDS)
    return _parse_json_object(str(response))


# --- verification -----------------------------------------------------------

# Words a quote may use to state a number ("two travelers", "solo", "an extra night").
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "a": 1, "an": 1,
    "solo": 1, "alone": 1, "myself": 1,
}
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")


# Whether the model's quote really appears in the request (ignoring case and accents).
def _quote_in_request(evidence: object, request_text: str) -> bool:
    return isinstance(evidence, str) and bool(evidence.strip()) and fold_text(evidence) in fold_text(request_text)


def _numbers_stated_in(evidence: str) -> set[float]:
    """Every number the quote itself states: digits ("1,500", "3k"), number words, and the
    length words a request uses for trips ("week", "weekend", "fortnight")."""
    folded = fold_text(evidence)
    numbers = {
        float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s?(k)?\b", folded)
    }
    numbers |= {float(_WORD_NUMBERS[w]) for w in re.findall(r"[a-z]+", folded) if w in _WORD_NUMBERS}
    counts = numbers or {1.0}
    if re.search(r"\bweeks?\b", folded):
        numbers |= {n * 7 for n in counts}
    if "weekend" in folded:
        numbers.add(2.0)
    if "fortnight" in folded:
        numbers.add(14.0)
    return numbers


# Accepts an LLM-extracted number only if its quote is in the request and states that value
# (or, for travel_month, names that month). Returns (value or None, note for the parsing log).
def _verify_number(field: str, item: object, request_text: str) -> tuple[float | None, str | None]:
    if item is None:
        return None, None
    if not isinstance(item, dict) or not isinstance(item.get("value"), (int, float)):
        return None, f"LLM {field} rejected: malformed value."
    value, evidence = float(item["value"]), item.get("evidence")
    if not _quote_in_request(evidence, request_text):
        return None, f"LLM {field}={value:g} rejected: its quote {evidence!r} isn't in the request."
    if field == "travel_month":
        stated = 1 <= value <= 12 and re.search(rf"\b{_MONTHS[int(value) - 1][:3]}", fold_text(evidence))
    else:
        stated = value in _numbers_stated_in(evidence)
    if not stated:
        return None, f"LLM {field}={value:g} rejected: the quote {evidence!r} doesn't state that value."
    return value, f"LLM {field}={value:g} accepted (quote: {evidence!r})."


def _destination_quote_supports(name: str, evidence: str) -> bool:
    """The quoted text has to actually be that place: a shared word ("Cabo" -> "Cabo San
    Lucas"), or an all-caps abbreviation ("NYC"). Blocks e.g. "somewhere warm" -> "Cancun"."""
    if re.fullmatch(r"[A-Z]{2,4}", evidence.strip()):
        return True
    name_words = set(re.findall(r"\w+", fold_text(name.partition(",")[0])))
    evidence_words = set(re.findall(r"\w+", fold_text(evidence)))
    return bool(name_words & evidence_words)


# Keeps LLM-named destinations whose quote is in the request and actually names that place,
# then geocodes them. Rejections and geocoding outcomes are appended to `notes`.
async def _verify_destinations(client: httpx.AsyncClient, items: object, request_text: str, notes: list[str]):
    accepted_names: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        name, evidence = item["name"].strip(), item.get("evidence")
        head = name.partition(",")[0].strip()
        if not _quote_in_request(evidence, request_text):
            notes.append(f"LLM destination {name!r} rejected: its quote {evidence!r} isn't in the request.")
        elif is_non_place_word(head) or all(is_non_place_word(w) for w in str(evidence).split()):
            notes.append(f"LLM destination {name!r} rejected: a month/weekday/filler word is never a destination.")
        elif not _destination_quote_supports(name, evidence):
            notes.append(f"LLM destination {name!r} rejected: the quote {evidence!r} doesn't name that place.")
        else:
            accepted_names.append(name)
    if not accepted_names:
        return []
    candidates, geo_notes = await resolve_destination_phrases(client, accepted_names, strict=False)
    notes.extend(f"LLM destination check: {n}" for n in geo_notes)
    notes.extend(f"LLM destination accepted after geocoding: {c.label}." for c in candidates)
    return candidates


# Marks every field the rule-based parse filled in as sourced from "rules".
def _set_fields(constraints: TripConstraints) -> dict[str, str]:
    fields = {f: "rules" for f in (*_NUMERIC_FIELDS, "max_flight_hours", "max_drive_hours") if getattr(constraints, f) is not None}
    fields |= {f: "rules" for f in _INTEREST_FIELDS.values() if getattr(constraints, f)}
    return fields


# Entry point for intake (see the module notes above): rule-based parsing first; the LLM is
# consulted only when the rules found no destination or the request is open-ended, and only
# fills fields the rules left empty, after verification. Every accepted value records its
# source ("rules" / "llm") in `field_sources`.
async def understand_trip_request(
    client: httpx.AsyncClient,
    request_text: str,
    *,
    today: date | None = None,
    use_llm: bool = True,
) -> ParsedTripRequest:
    constraints = parse_trip_constraints(request_text)
    start_date = parse_travel_start_date(request_text, today=today)
    open_ended = bool(_OPEN_ENDED_PATTERN.search(request_text))

    candidates, notes = await resolve_destination_phrases(client, extract_destination_phrases(request_text))
    sources = _set_fields(constraints) | ({"start_date": "rules"} if start_date else {})
    if candidates:
        sources["destinations"] = "rules"

    result = ParsedTripRequest(
        constraints=constraints, candidates=candidates, start_date=start_date, open_ended=open_ended, notes=notes
    )
    if candidates and not open_ended:
        result.field_sources = sources
        return result
    if not use_llm:
        result.llm_status = "skipped (disabled)"
        result.field_sources = sources
        return result

    try:
        extracted = await _llm_extract(request_text)
    except Exception as ex:
        # Don't log the request text itself -- it's free-form user input (see app.py).
        logger.warning("Request-parsing LLM step unavailable, using rule-based parse only: %s", describe_error(ex))
        result.llm_status = f"unavailable ({describe_error(ex)}) -- rule-based parse only"
        result.field_sources = sources
        return result

    result.llm_status = "used (rules were unsure)"
    updates: dict[str, object] = {}
    for field in _NUMERIC_FIELDS:
        value, note = _verify_number(field, extracted.get(field), request_text)
        if note:
            notes.append(note)
        if value is not None and getattr(constraints, field) is None:
            updates[field] = value if field == "budget_usd" else int(value)
            sources[field] = "llm"

    # Interests stay rules-only: a quote can prove the words are in the request, but not that
    # they mean that interest (a test answer tagged "food" as nightlife and passed the quote
    # check). Suggestions are kept in the notes for review instead of being applied.
    for item in extracted.get("interests") or []:
        field = _INTEREST_FIELDS.get(item.get("value")) if isinstance(item, dict) else None
        if field is not None and not getattr(constraints, field):
            notes.append(
                f"LLM suggested interest {item['value']!r} (quote: {item.get('evidence')!r}) -- not applied; "
                "an interest's meaning can't be verified from a quote."
            )

    seen = {(fold_text(c.name), c.country_code) for c in candidates}
    for candidate in await _verify_destinations(client, extracted.get("destinations"), request_text, notes):
        if (fold_text(candidate.name), candidate.country_code) not in seen:
            seen.add((fold_text(candidate.name), candidate.country_code))
            candidates.append(candidate)
            sources["destinations"] = sources.get("destinations", "llm")

    result.constraints = constraints.model_copy(update=updates)
    result.notes = notes
    result.candidates = candidates[:_MAX_CANDIDATES]
    result.open_ended = open_ended or (extracted.get("open_ended") is True and not result.candidates)
    result.field_sources = sources
    return result
