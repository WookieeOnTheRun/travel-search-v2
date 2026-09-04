# Agentic Travel App (Python + Microsoft Agent Framework + Streamlit)

This project builds a travel concierge workflow that:
- accepts free-form user travel preferences,
- resolves them into disambiguated destination candidates for the user to confirm
  (e.g. distinguishing "Paris, France" from "Paris, Texas") before anything is searched,
- when no specific destination is named -- e.g. "somewhere warm within a 3-hour flight or
  7-hour drive with a great beach and food scene, not interested in sightseeing" -- proposes
  real, independently-verified candidate destinations instead of a dead end (see "Destination
  discovery" below),
- programmatically resolves the confirmed destination into the real provider codes
  search APIs need (IATA airport codes, hotel location ids, country codes) --
  never by having an LLM guess a code from memory,
- gathers grounding context from reputable/public travel sources,
- as soon as a destination + date are confirmed, automatically checks for active U.S.
  Department of State advisories, a weather outlook for the travel window, and nearby
  activities annotated for seasonal appropriateness (see "Destination-selection checks" below),
- uses a multi-agent workflow to generate 3 itinerary options,
- includes flights (if an origin city was given), 3-star+ hotel recommendations, unique activities,
- recommends local transportation options,
- prioritizes luxury experience within budget and traveler safety.

Flight search and hotel/accommodation search both run against the **Duffel API**
(`https://api.duffel.com`, `Duffel-Version: v2` -- verified against the official docs at
https://duffel.com/docs/api on 2026-09-04; see `travel_agent/location_resolution.py` and
`travel_agent/data_sources.py` for the exact endpoints/parameters):

| Source | Used for |
|---|---|
| Duffel Places (`GET /places/suggestions`) | Airport code resolution (origin + destination), matched by ISO country code for real disambiguation |
| Duffel Flights (`POST /air/offer_requests`) | Flight search (one-way or round-trip, by resolved IATA airport codes) |
| Duffel Stays (`POST /stays/search`) | Hotel/accommodation search, by the destination's geocoded coordinates + a radius |

A remaining couple of sources stay on RapidAPI, since Duffel doesn't cover them:

| Source | Used for |
|---|---|
| Tripadvisor (`tripadvisor16.p.rapidapi.com`) | Cruise search only (currently failing server-side on this product -- see note below) |
| Visa Requirements (`visa-requirement.p.rapidapi.com`) | Visa/entry requirement lookup |

**Tripadvisor cruise search (`cruises/getLocation`) is currently unreliable.** The path
is registered (not a 404) but the backend returns a generic error for every query tested,
including real port cities. Cruise search is still wired up -- gated so it's only called
when the request mentions cruising -- and degrades to a clear "unavailable" message rather
than fabricating results; it will start working automatically if the upstream issue clears.

## Destination-selection checks (`travel_agent/destination_insights.py`)

As soon as a destination + travel date are confirmed on the "confirm" screen -- before
itinerary generation -- three checks run automatically and render directly on that screen,
each from a source verified live (2026-09-03), no API key required for any of them:

| Check | Source | Behavior |
|---|---|---|
| Safety advisory | U.S. Dept. of State (`cadataapi.state.gov/api/TravelAdvisories`) | An obvious red banner appears if the destination's country has a Level 3 ("Reconsider Travel") or Level 4 ("Do Not Travel") advisory published/updated in the last 6 months. |
| Weather outlook | Open-Meteo Historical Weather API (`archive-api.open-meteo.com`) + standard Forecast API (`api.open-meteo.com`) | Average highs/lows for the trip's calendar window, averaged across the last 3 fully-elapsed years. If departure is within 14 days, an actual forecast for that window is also shown (forecast API covers up to 16 days out). |
| Activities & sightseeing | OpenStreetMap Overpass API (`overpass-api.de/api/interpreter`) | Nearby named tourism/historic points of interest, each annotated with a best-effort comment on whether its listed `opening_hours` cover the trip's travel months. |

The same State Department check also now grounds the final itinerary's safety section in
`data_sources.py` (replacing the previous third-party `travel-advisory.info` aggregator),
so the confirm-screen banner and the generated itinerary never cite conflicting sources.

**Tripadvisor was evaluated for the activities check first and ruled out, not guessed
around.** Since `tripadvisor16.p.rapidapi.com` is already used elsewhere in this app, it was
checked live for an attractions/things-to-do endpoint before reaching for OpenStreetMap: over
a dozen plausible paths (`attractions/searchAttractions`, `attraction/searchLocation`,
`poi/searchLocation`, `tours/searchLocation`, `thingstodo/searchLocation`, etc.) were probed
against the live API and every one 404'd, and its sibling `restaurant/*` endpoints that do
exist are currently failing server-side too (same class of issue as the cruise endpoint
above). OpenStreetMap's Overpass API needs no key, is well-documented, and was verified live
to return real, named points of interest with usable tags.

## Destination discovery (`travel_agent/destination_discovery.py`)

If the intake form's regex-based destination extraction can't find a named place in the
request, but an origin city was given, the app switches to discovery mode instead of a dead
end: it proposes real destinations matching the described vibe, budget/time envelope, and
climate. Example input: *"I have three days and a $5000 budget. I want to go somewhere warm
that is within a three hour flight or a seven hour drive of my departure point. I want a
destination that has either a great beach or an amazing pool with a swim up bar. I want a
great food scene and lots of entertainment, and am not interested in cultural activities or
sightseeing."*

The pipeline is deliberately split so an LLM only ever proposes *names*, never *facts*:

1. **Parse constraints deterministically.** `parse_trip_constraints` regexes budget, trip
   length, max flight/drive hours (digits or number-words, e.g. "three hours"), and
   warm/beach/pool/food/entertainment/avoid-cultural flags straight out of the text -- the
   same "return None rather than guess" philosophy as `location_resolution.py`'s date parser.
2. **Brainstorm candidate names.** A `DestinationIdeaAgent` (using this app's configured
   model) is instructed to propose 15-25 real place names only -- explicitly forbidden from
   asserting any fact about them, since nothing it says is trusted directly. If the model call
   fails (e.g. no local model endpoint running), a small static list of well-known
   warm/coastal destinations is used instead; either way, only names come out of this step.
3. **Independently verify every candidate against real data**, dropping any that fail a hard
   check and annotating the rest with `rationale` (verified facts kept) and `caveats`
   (anything estimated or unavailable):
   - **Distance/travel time**: geocoded via the same Open-Meteo geocoding already used
     elsewhere, filtered by great-circle distance converted from the stated flight/drive-hour
     budget. Flight-time and driving-speed conversion constants are cited estimates (aviation
     speed references put average block speed around 400-500mph; road-trip planning guidance
     converges on ~55-65mph for long-distance driving) used only to size the initial net;
     surviving candidates get a **real routed drive time from OSRM**
     (`router.project-osrm.org`, free/keyless) rather than trusting the estimate.
   - **Warmth**: reuses `destination_insights.fetch_weather_outlook`'s real historical
     averages for the requested dates; candidates below the app's documented warm-weather
     threshold (24°C/75°F average high -- a stated design choice, not an external authority)
     are dropped when warmth was requested.
   - **Beach / food scene / nightlife**: a single combined OpenStreetMap Overpass query per
     candidate counts nearby beaches, restaurants, and bars/nightclubs -- real counts, used
     both as rationale text and as scoring signal, never asserted without a successful check.
4. Surviving candidates are ranked by a deterministic, documented scoring formula (warmth
   margin + beach match + food/nightlife density) and the top few are shown for the user to
   pick from; picking one carries it into the normal confirm -> itinerary flow.

**Public-API courtesy and latency.** OSRM's demo server and Overpass's public instance both
publish no-heavy-usage policies (OSRM: 1 request/second; Overpass: no parallel querying).
Both are called sequentially with explicit pacing for the bounded post-filter shortlist,
never fired concurrently across candidates. Just as important: the `httpx` client used for
discovery sets a short, explicit *connect* timeout (5s) separate from the read timeout --
live testing during development showed that a flat single timeout value turns one slow/dead
connection attempt, multiplied across the retry budget and every candidate, into a
20+ minute hang; a fast-failing connect timeout keeps the worst case (a fully unreachable
public service) bounded to a couple of minutes while leaving normal operation unaffected.

## Stack

- `agent-framework-core` for agent orchestration
- `agent-framework-ollama` for a locally-hosted Ollama model as the chat client
- `streamlit` for web UI
- `httpx` for external data grounding calls
- `python-dateutil` for parsing a travel date out of free text
- `python-dotenv` + `pydantic` for configuration and schema management

## Project Layout

- `config.py` model + endpoint configuration
- `app.py` Streamlit entry point (multi-step: free text -> [discover destinations if none
  named] -> confirm destination/date -> results)
- `travel_agent/location_resolution.py` destination candidate geocoding, travel-date parsing,
  and provider-code resolution (Duffel-resolved airport codes, Tripadvisor cruise location id)
- `travel_agent/data_sources.py` grounding + search API calls (Duffel Flights, Duffel Stays,
  Tripadvisor cruises, visa requirements), gated by relevance (flights only called if an origin
  was given; cruise only if the request mentions cruising)
- `travel_agent/destination_insights.py` the three destination-selection checks (State
  Department advisory, weather outlook, seasonal activities) -- see above
- `travel_agent/destination_discovery.py` "suggest a destination" search for requests that
  don't name a place -- see "Destination discovery" above
- `travel_agent/http_utils.py` shared retry/backoff HTTP helpers, including per-host request
  serialization (see "Rate limits" below)
- `travel_agent/itinerary_agent.py` multi-agent itinerary workflow
- `travel_agent/schemas.py` request/grounding/candidate schemas
- `travel_agent/main.env` local secrets (gitignored -- see "Configure" below)

## Configure

1. Copy `travel_agent/main.env` (or create `.env`) and set:

```env
OLLAMA_MODEL=gemma4:31b
OLLAMA_HOST=http://localhost:11434

DUFFEL_API_KEY=
DUFFEL_API_BASE_URL=https://api.duffel.com
DUFFEL_API_VERSION=v2

RAPIDAPI_KEY=

RAPIDAPI_TRIPADVISOR_HOST=tripadvisor16.p.rapidapi.com
RAPIDAPI_VISA_HOST=visa-requirement.p.rapidapi.com

DEFAULT_CITIZENSHIP_COUNTRY_CODE=US
```

`DUFFEL_API_KEY` is a Duffel access token from your Duffel dashboard -- use a test token
(prefixed `duffel_test_`) for development; test-mode flight search reliably returns
results from Duffel's own "Duffel Airways" (`ZZ`) sandbox airline rather than needing a
real airline sandbox. See https://duffel.com/docs/api/overview/test-mode.

Only the RapidAPI **host** is configurable per RapidAPI provider -- endpoint paths are
fixed, verified constants in code. A RapidAPI host maps 1:1 to one product's route
structure, so an older design that also let the base *path* be configured separately just
reintroduced bugs (several of those configured URLs were missing their endpoint path
entirely and could never have returned real results). Swapping to a different underlying
product for the same capability is the only thing that actually requires changing this
config, and that's exactly what changing the host does.

The LLM itself is a locally-hosted [Ollama](https://ollama.com) model, reached via
`agent-framework-ollama`'s `OllamaChatClient`. Before running the app, make sure Ollama is
installed and running locally and that the configured model has been pulled, e.g.:

```
ollama pull gemma4:31b
```

`OLLAMA_MODEL`/`OLLAMA_HOST` fall back to `gemma4:31b`/`http://localhost:11434` (Ollama's
default port) if left unset.

2. **Never commit `main.env` or `.env`, and never paste a real key into chat, a PR, or an
   issue.** Both are gitignored. If a key is ever exposed (committed, pasted, logged),
   rotate it on Duffel (or RapidAPI, whichever key was exposed) immediately -- treat it as
   compromised even if the exposure was private, since it lives on in history until rotated.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How search works

1. You describe your trip in free text. The app extracts candidate place names and
   geocodes each one (via Open-Meteo's public geocoding API) into a short list of
   disambiguated destinations -- same-named places (Paris/France vs. Paris/Texas,
   Springfield/Illinois vs. Springfield/Missouri) show up as distinct options.
2. You confirm one destination and a travel start date (parsed from your text if it was
   explicit there, e.g. "October 15" or "2026-11-03"; otherwise you're asked for one).
3. Only then does the app resolve real provider codes for that confirmed destination
   (Duffel-resolved IATA airport codes for flights) and run search:
   - Hotel/accommodation search always runs (Duffel Stays, searched by the destination's
     geocoded coordinates + a radius, using real `check_in_date`/`check_out_date` values).
   - Flight search only runs if you gave an origin city (otherwise there's no route to search);
     it resolves airport codes via Duffel Places, then searches via Duffel Flights
     (`offer_requests`).
   - Cruise search only runs if your request mentions cruising.
   - Visa search always runs, using the destination's actual country code (resolved during
     geocoding, not a guess).

This relevance gating -- plus resolving each destination's codes once and reusing them,
rather than re-deriving them per search call -- is what keeps the number of outbound API
calls proportional to what the request actually needs.

## Rate limits

Live testing surfaced that firing concurrent requests at the *same* host (e.g. resolving
origin and destination airport codes at the same time, both against Duffel Places) can
trip that host's rate limit and come back with empty/incomplete data rather than a loud
error. `travel_agent/http_utils.py` serializes requests per host (a lock keyed by
hostname -- this applies equally to `api.duffel.com` and the remaining RapidAPI hosts) to
avoid self-inflicted rate limiting -- calls to *different* hosts still run fully in
parallel. If you see empty search results in bursts, check your Duffel/RapidAPI dashboard
for per-product plan/quota/rate limits before assuming the code is wrong.

The per-host lock is scoped to the *currently running* event loop, not just the hostname.
Streamlit's `app.py` calls `asyncio.run(...)` fresh on every script rerun (every widget
interaction), which creates a brand-new event loop each time; a lock object created in one
run is bound to that run's loop and crashes with "bound to a different event loop" if a
later run tries to reuse it. This was caught live once the destination-selection weather
check (which fires three concurrent requests at the same Open-Meteo host) was actually run
through more than one Streamlit rerun -- see `_host_lock_for` in `http_utils.py`.

## Safety Grounding Notes

The app uses a public travel-advisory feed and explicitly includes U.S. Department of State as a required source in generated output:
- https://travel.state.gov/content/travel.html

Always validate final travel decisions against the latest official advisories and local regulations.
