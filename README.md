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

Flight search, hotel/accommodation search, and cruise search all run against the
**Tripadvisor API via RapidAPI** (`tripadvisor16.p.rapidapi.com` -- endpoint paths and
parameter contracts verified against the live API on 2026-09-08; see
`travel_agent/location_resolution.py` and `travel_agent/data_sources.py` for the exact
endpoints/parameters):

| Source | Used for |
|---|---|
| Tripadvisor Hotels (`GET /api/v1/hotels/searchLocation` + `GET /api/v1/hotels/searchHotels`) | Hotel/accommodation search, by a resolved Tripadvisor `geoId` for the destination -- verified working end-to-end |
| Tripadvisor Flights (`GET /api/v1/flights/searchAirport` + `GET /api/v1/flights/searchFlights`) | Airport code resolution and flight search, by resolved IATA airport codes -- endpoint paths and parameter names are verified, but see the reliability note below |
| Tripadvisor Cruises (`GET /api/v1/cruises/getLocation` + `GET /api/v1/cruises/searchCruises`) | Cruise search (currently failing server-side on this product -- see note below) |

A separate RapidAPI host handles the one thing Tripadvisor doesn't cover:

| Source | Used for |
|---|---|
| Visa Requirements (`visa-requirement.p.rapidapi.com`) | Visa/entry requirement lookup |

**Tripadvisor's flight endpoints and cruise search (`cruises/getLocation`) are currently
unreliable.** All three paths are registered (not 404s) and their request parameters are
verified against the live API's own validation layer, but `flights/searchAirport` returns an
empty result for every query tried, `flights/searchFlights` returns a generic server-side
error for every syntactically valid request tried, and `cruises/getLocation` returns a
generic error for every query tested, including real port cities. All three are still wired
up per their real, verified contracts -- flight search is gated so it's only called when an
origin city is given, cruise search only when the request mentions cruising -- and each
degrades to a clear "unavailable" message rather than fabricating results; they'll start
working automatically if the upstream issue clears. Hotel/accommodation search has no such
issue: it's verified working end-to-end with real results.

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
- `agent-framework-ollama` (plus the `ollama` SDK directly, for Cloud auth headers) as the chat client, pointed at Ollama's cloud-hosted API
- `streamlit` for web UI
- `httpx` for external data grounding calls
- `python-dateutil` for parsing a travel date out of free text
- `python-dotenv` + `pydantic` for configuration and schema management

## Project Layout

- `config.py` model + endpoint configuration
- `app.py` Streamlit entry point (multi-step: free text -> [discover destinations if none
  named] -> confirm destination/date -> results)
- `travel_agent/location_resolution.py` destination candidate geocoding, travel-date parsing,
  and provider-code resolution (Tripadvisor-resolved airport codes, hotel location id, and
  cruise location id)
- `travel_agent/data_sources.py` grounding + search API calls (Tripadvisor Flights, Tripadvisor
  Hotels, Tripadvisor Cruises, visa requirements), gated by relevance (flights only called if an
  origin was given; cruise only if the request mentions cruising)
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
OLLAMA_MODEL=gemma4:cloud
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=

RAPIDAPI_KEY=

RAPIDAPI_TRIPADVISOR_HOST=tripadvisor16.p.rapidapi.com
RAPIDAPI_VISA_HOST=visa-requirement.p.rapidapi.com

DEFAULT_CITIZENSHIP_COUNTRY_CODE=US
```

`RAPIDAPI_KEY` is a RapidAPI application key subscribed to both the Tripadvisor
(`tripadvisor16.p.rapidapi.com`) and Visa Requirements products -- get one from your RapidAPI
dashboard.

Only the RapidAPI **host** is configurable per RapidAPI provider -- endpoint paths are
fixed, verified constants in code. A RapidAPI host maps 1:1 to one product's route
structure, so an older design that also let the base *path* be configured separately just
reintroduced bugs (several of those configured URLs were missing their endpoint path
entirely and could never have returned real results). Swapping to a different underlying
product for the same capability is the only thing that actually requires changing this
config, and that's exactly what changing the host does.

The LLM itself is [Ollama's cloud-hosted API](https://ollama.com) (`https://ollama.com/api`),
reached via `agent-framework-ollama`'s `OllamaChatClient`, wired up in
`travel_agent/itinerary_agent.py`'s `build_ollama_client()`. No local Ollama install, running
daemon, or `ollama pull` is required -- this keeps the app's LLM dependency available
regardless of where it's hosted (a laptop, a container, or a cloud deployment all just need
outbound HTTPS access and a valid `OLLAMA_API_KEY`).

Authentication is via an `Authorization: Bearer <OLLAMA_API_KEY>` header (verified against
https://docs.ollama.com/api/authentication on 2026-09-08) -- `agent_framework_ollama`'s
`OllamaChatClient` has no built-in way to attach that header itself, so `build_ollama_client()`
constructs the underlying `ollama.AsyncClient` directly with it and passes that in.
`OLLAMA_HOST` is the SDK's host *root* (`https://ollama.com`), not the `/api`-suffixed form
shown in Ollama's raw HTTP/curl docs -- the ollama-python client already appends
`/api/<endpoint>` to every call itself, so including `/api` in the host produces a broken
`/api/api/...` path (confirmed by inspecting `ollama/_client.py` and building a real request
against both forms).

Create an API key at https://ollama.com/settings/keys and set it as `OLLAMA_API_KEY`.
`OLLAMA_MODEL`/`OLLAMA_HOST` fall back to `gemma4:cloud`/`https://ollama.com` if left unset,
but `OLLAMA_API_KEY` has no fallback -- `build_ollama_client()` raises a clear error if it's
missing rather than silently failing partway through a request.

2. **Never commit `main.env` or `.env`, and never paste a real key into chat, a PR, or an
   issue.** Both are gitignored. `OLLAMA_API_KEY` is read from that file/environment only --
   it is never logged, and never rendered anywhere in the Streamlit UI, so it stays
   inaccessible to users of the running app. If a key is ever exposed (committed, pasted,
   logged), rotate it immediately on Ollama (https://ollama.com/settings/keys) or RapidAPI
   (whichever key was exposed) -- treat it as compromised even if the exposure was private,
   since it lives on in history until rotated.

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
   (Tripadvisor-resolved IATA airport codes for flights, a Tripadvisor `geoId` for hotels)
   and run search:
   - Hotel/accommodation search always runs: the destination name is resolved to a Tripadvisor
     `geoId` (`hotels/searchLocation`), then searched with real `checkIn`/`checkOut` dates
     (`hotels/searchHotels`).
   - Flight search only runs if you gave an origin city (otherwise there's no route to search);
     it resolves airport codes via `flights/searchAirport`, then searches via
     `flights/searchFlights`.
   - Cruise search only runs if your request mentions cruising.
   - Visa search always runs, using the destination's actual country code (resolved during
     geocoding, not a guess).

This relevance gating -- plus resolving each destination's codes once and reusing them,
rather than re-deriving them per search call -- is what keeps the number of outbound API
calls proportional to what the request actually needs.

## Rate limits

Live testing surfaced that firing concurrent requests at the *same* host (e.g. resolving
origin and destination airport codes at the same time, both against Tripadvisor's flight
airport-search endpoint) can trip that host's rate limit and come back with empty/incomplete
data rather than a loud error. `travel_agent/http_utils.py` serializes requests per host (a
lock keyed by hostname) to avoid self-inflicted rate limiting -- calls to *different* hosts
still run fully in parallel. If you see empty search results in bursts, check your RapidAPI
dashboard for per-product plan/quota/rate limits before assuming the code is wrong.

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
