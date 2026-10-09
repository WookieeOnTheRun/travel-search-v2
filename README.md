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
- prioritizes luxury experience within budget and traveler safety,
- lets the user save generated results into named itineraries and browse them all in one
  place (see "Saved itineraries" below).

Flight and hotel/accommodation search both run against the **Booking.com API via RapidAPI**
(`booking-com15.p.rapidapi.com` -- endpoint paths and parameter contracts verified against the
live API on 2026-09-10); see `travel_agent/location_resolution.py` and
`travel_agent/data_sources.py` for the exact endpoints/parameters:

| Source | Used for |
|---|---|
| Booking.com Flights (`GET /api/v1/flights/searchDestination` + `GET /api/v1/flights/searchFlights`) | Airport-code-to-location-id resolution and flight search, by resolved IATA airport codes -- verified working end-to-end, including round-trip, with real priced offers |
| Booking.com Hotels (`GET /api/v1/hotels/searchDestination` + `GET /api/v1/hotels/searchHotels`) | Hotel/accommodation search: `searchDestination` resolves the destination name to a matched `dest_id`/`search_type` pair, which `searchHotels` requires together -- verified working end-to-end |

A separate RapidAPI host handles the one thing neither of the above covers:

| Source | Used for |
|---|---|
| Visa Requirements (`visa-requirement.p.rapidapi.com`) | Visa/entry requirement lookup |

**Both flight and hotel search moved off Tripadvisor to Booking.com because Tripadvisor's
endpoints never returned usable results on this subscription** -- `flights/searchAirport`
returned an empty result for every query tried, `flights/searchFlights` returned a generic
server-side error (HTTP 200 with `{"status": false, ...}`) for every syntactically valid
request tried, and its cruise endpoints (`cruises/getLocation`/`cruises/searchCruises`) never
returned usable results either. Booking.com's flight and hotel endpoints were verified live on
2026-09-10 to return real data end-to-end: for flights, querying `searchDestination` by an
already-resolved IATA code (from the OurAirports dataset -- see the airport-resolution note
below) reliably returns the exact Booking.com location id (`fromId`/`toId`, e.g.
`"JFK.AIRPORT"`) that `searchFlights` requires; for hotels, querying `searchDestination` by the
confirmed destination name returns a `dest_id`/`search_type` pair (matched against a "city"
entry, disambiguated by country code) that `searchHotels` requires together, and both
`searchFlights` and `searchHotels` return real, priced results end-to-end. There is no cruise
search feature in this app -- it depended entirely on Tripadvisor's unreliable cruise
endpoints and was removed along with the rest of the Tripadvisor integration rather than kept
around with no working data source.

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
around.** Back when `tripadvisor16.p.rapidapi.com` was still used elsewhere in this app (before
flight and hotel search both moved to Booking.com -- see above), it was checked live for an
attractions/things-to-do endpoint before reaching for OpenStreetMap: over a dozen plausible
paths (`attractions/searchAttractions`, `attraction/searchLocation`, `poi/searchLocation`,
`tours/searchLocation`, `thingstodo/searchLocation`, etc.) were probed against the live API
and every one 404'd, and its sibling `restaurant/*` endpoints that do exist were failing
server-side too. OpenStreetMap's Overpass API needs no key, is well-documented, and was
verified live to return real, named points of interest with usable tags.

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
     (`https://router.project-osrm.org`, free/keyless, called over HTTPS) rather than
     trusting the estimate.
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
  named] -> confirm destination/date -> results), plus the itinerary sidebar, single-itinerary
  view, and "All itineraries" view
- `travel_agent/persistence.py` SQLite store for per-session itineraries and their saved
  searches (see "Saved itineraries" below)
- `.streamlit/config.toml` beta-deployment hardening for the Streamlit UI (see "Beta
  deployment notes" below)
- `travel_agent/location_resolution.py` destination candidate geocoding, travel-date parsing,
  and provider-code resolution (OurAirports-resolved IATA airport codes turned into Booking.com
  flight location ids, and Booking.com hotel `dest_id`/`search_type` pairs)
- `travel_agent/data_sources.py` grounding + search API calls (Booking.com Flights, Booking.com
  Hotels, visa requirements), gated by relevance (flights only called if an origin was given)
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

RAPIDAPI_BOOKING_HOST=booking-com15.p.rapidapi.com
RAPIDAPI_VISA_HOST=visa-requirement.p.rapidapi.com

DEFAULT_CITIZENSHIP_COUNTRY_CODE=US

# Optional: where the itinerary database lives (default: data/travel_sessions.db)
# SESSION_DB_PATH=
```

The first env file found is loaded, in this order: the path in `TRAVEL_AGENT_ENV_FILE`,
`travel_agent/main.env`, then `.env`. **A variable already set in the real process
environment always wins over the file** (python-dotenv's `override=False`), so on a deployed
host, values set through the platform's own secret/config settings are never replaced by a
stale env file shipped with the code. Which env file was loaded is logged server-side at
startup (it is not shown in the UI). A blank value (`KEY=`) in an env file counts as unset
and falls back to the built-in default, so `.env.example` can be copied as-is.

`RAPIDAPI_KEY` is a RapidAPI application key subscribed to the Booking.com
(`booking-com15.p.rapidapi.com`) and Visa Requirements products -- get one from your RapidAPI
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

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS (including Codespaces):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run `streamlit run` from the project root so `.streamlit/config.toml` is picked up --
Streamlit reads that file from the directory it is launched from.

## How search works

1. You describe your trip in free text. The app extracts candidate place names and
   geocodes each one (via Open-Meteo's public geocoding API) into a short list of
   disambiguated destinations -- same-named places (Paris/France vs. Paris/Texas,
   Springfield/Illinois vs. Springfield/Missouri) show up as distinct options.
2. You confirm one destination and a travel start date (parsed from your text if it was
   explicit there, e.g. "October 15" or "2026-11-03"; otherwise you're asked for one).
3. Only then does the app resolve real provider codes for that confirmed destination
   (OurAirports-resolved IATA airport codes turned into Booking.com location ids for flights, a
   Booking.com `dest_id`/`search_type` pair for hotels) and run search:
   - Hotel/accommodation search always runs, as a two-step Booking.com flow: the destination
     name is resolved to a matched `dest_id`/`search_type` pair (`hotels/searchDestination`,
     preferring a "city" entry disambiguated by country code), then both are passed together
     to `hotels/searchHotels` along with real `arrival_date`/`departure_date` dates.
   - Flight search only runs if you gave an origin city (otherwise there's no route to search);
     it resolves the nearest real IATA airport codes (OurAirports dataset, by great-circle
     distance), converts the nearest one to a Booking.com location id via
     `flights/searchDestination`, then searches via `flights/searchFlights`.
   - Visa search always runs, using the destination's actual country code (resolved during
     geocoding, not a guess).

This relevance gating -- plus resolving each destination's codes once and reusing them,
rather than re-deriving them per search call -- is what keeps the number of outbound API
calls proportional to what the request actually needs.

## Saved itineraries

Generated results can be filed into named itineraries, managed from the **Itineraries**
sidebar:

- **Saving.** A generated result lives only in the browser session until you use **Save this
  search to an itinerary** on the results screen -- pick an existing itinerary, or choose
  "New itinerary" and give it a name (unique per session, case-insensitive, up to 80
  characters; letters, numbers, spaces and `- . , ' & ( ) / : ? +` only -- Streamlit renders
  names as Markdown, and this rule is what keeps a name from carrying a link or an image). Nothing is written to the database for a search that is never saved; "Start
  over" discards it. The same result can be saved to more than one itinerary. A search is
  only ever stored together with its itinerary link, in a single transaction.
- **Viewing.** Each itinerary has its own button in the sidebar (opens that itinerary and its
  saved searches). **All itineraries** lists every itinerary in the session together, each
  with its saved searches and an Open button. Saved searches are snapshots: prices and
  availability are as of when the search ran, not live quotes.
- **Creating an empty itinerary.** The sidebar's "New itinerary name" form creates one with no
  searches yet.
- **Retention.** An itinerary is kept for 72 hours after its last change (creation or the most
  recent search added to it), then hidden immediately and deleted on the next cleanup, which
  runs when any new browser session starts. Searches no longer in any itinerary are deleted in
  the same cleanup.
- **Sessions.** There are no accounts. Each browser session gets a random id (a UUID4) carried
  in the page URL as `?sid=...`, so a refresh keeps the same itineraries. Opening the app
  without that parameter starts a new, empty session; bookmark the full URL to come back.
  **Anyone holding the full URL can see and add to that session's itineraries -- don't share
  it.**
- **Storage.** A local SQLite file (`SESSION_DB_PATH`, default `data/travel_sessions.db`,
  gitignored). It is only as durable as the disk it sits on: a host or container without a
  persistent disk loses every itinerary on restart, and it is not shared across multiple app
  instances.
- **Error logging.** Every failed itinerary action (listing, creating, saving, opening,
  displaying) logs one line to the server console -- `Itinerary action failed: <action>
  (<error type>) session_id=<first 8 chars>… itinerary_id=...`. Only a prefix of the
  session id is logged, since the full id is what grants access to that session's
  itineraries. Expected failures (blank/duplicate name,
  expired itinerary) are logged without a traceback; anything unexpected includes one.
  Itinerary names are never logged, since they are free-form user input.

## Beta deployment notes

- **Error display.** `.streamlit/config.toml` sets `client.showErrorDetails = "none"`, so
  an uncaught exception shows testers only a generic message; the full traceback still prints
  to the server console. It also sets `client.toolbarMode = "viewer"` (hides developer menu
  items) and `browser.gatherUsageStats = false`. Values were checked against
  `streamlit config show` for the pinned Streamlit 1.43.2.
- **Input limits.** The travel request is capped at 2,000 characters and the origin city at
  100, enforced in the browser and re-checked on the server. The request text is sent to every
  LLM agent, so this bounds per-search cost and latency.
- **LLM output.** The final itinerary is model-generated markdown rendered with
  `st.markdown`, whose default (`unsafe_allow_html=False`) escapes any HTML in it.
  Upstream API data (hotels, flights, visa results) is passed into prompts as text, so the
  model's output should still be read as suggestions to verify, not trusted facts.
- **Logging.** `app.py` configures logging once at INFO level with timestamps (stderr), so
  the "which env file loaded" line, upstream-service warnings and itinerary errors all show
  in the server console. The `httpx` logger is deliberately held at WARNING: at INFO it logs
  every request URL including its query string, which for this app carries travelers'
  free-text requests and destinations. Keep it that way if you change the logging setup.
- **Secrets.** See "Configure" above -- keys come from the environment/env file only and are
  never rendered in the UI.

## Rate limits

Live testing surfaced that firing concurrent requests at the *same* host (e.g. resolving
origin and destination provider ids at the same time, both against the same RapidAPI product)
can trip that host's rate limit and come back with empty/incomplete data rather than a loud
error -- confirmed live against Tripadvisor's hosts on 2026-09-03; the same serialization
below applies uniformly to every host, including Booking.com's, as a precaution.
`travel_agent/http_utils.py` serializes requests per host (a
lock keyed by hostname) to avoid self-inflicted rate limiting -- calls to *different* hosts
still run fully in parallel. If you see empty search results in bursts, check your RapidAPI
dashboard for per-product plan/quota/rate limits before assuming the code is wrong.

The per-host lock is scoped to the *currently running* event loop, not just the hostname.
Streamlit's `app.py` calls `asyncio.run(...)` fresh on every script rerun (every widget
interaction), which creates a brand-new event loop each time; a lock object created in one
run is bound to that run's loop and crashes with "bound to a different event loop" if a
later run tries to reuse it. This was caught live once the destination-selection weather
check (which fires three concurrent requests at the same Open-Meteo host) was actually run
through more than one Streamlit rerun. Locks are keyed by the loop object itself (not its
`id()`, which Python may reuse once a loop is gone), entries for closed loops are pruned on
each lookup so they don't accumulate, and the shared map is guarded by a thread lock because
Streamlit runs each browser session on its own thread -- see `_host_lock_for` in
`http_utils.py`.

Retries back off exponentially and honor a server's `Retry-After` header, capped at 30
seconds per wait (the wait holds that host's lock, so an uncapped value would stall every
other request to the same host).

## Safety Grounding Notes

The app uses a public travel-advisory feed and explicitly includes U.S. Department of State as a required source in generated output:
- https://travel.state.gov/content/travel.html

Always validate final travel decisions against the latest official advisories and local regulations.
