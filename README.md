# Agentic Travel App (Python + Microsoft Agent Framework + Streamlit)

This project builds a travel concierge workflow that:
- accepts free-form user travel preferences,
- gathers grounding context from reputable/public travel sources,
- uses a multi-agent workflow to generate 3 itinerary options,
- includes flights (if relevant), 3-star+ hotel recommendations, unique activities,
- recommends local transportation options,
- prioritizes luxury experience within budget and traveler safety.

It is configured to use these RapidAPI-backed sources for itinerary grounding:
- Tripadvisor
- Booking.com
- Flights Scraper Sky
- Google Flights
- Visa Requirements

## Stack

- `agent-framework-core` for agent orchestration
- `agent-framework-azure-ai` for compatible Azure AI integrations
- `streamlit` for web UI
- `httpx` for external data grounding calls
- `python-dotenv` + `pydantic` for configuration and schema management

## Project Layout

- `config.py` model + endpoint configuration (as requested)
- `app.py` Streamlit entry point
- `travel_agent/data_sources.py` grounding data collectors
- `travel_agent/itinerary_agent.py` multi-agent itinerary workflow
- `travel_agent/schemas.py` request/grounding schemas
- `.env.example` environment variable template

## Configure

1. Copy `.env.example` to `.env`
2. Adjust values as needed:

```env
MODEL_NAME=phi-4
MODEL_ENDPOINT=http://localhost:5273/v1
MODEL_API_KEY=local

RAPIDAPI_KEY=

RAPIDAPI_TRIPADVISOR_URL=
RAPIDAPI_TRIPADVISOR_HOST=
RAPIDAPI_BOOKING_URL=
RAPIDAPI_BOOKING_HOST=
RAPIDAPI_FLIGHTS_SCRAPER_SKY_URL=
RAPIDAPI_FLIGHTS_SCRAPER_SKY_HOST=
RAPIDAPI_GOOGLE_FLIGHTS_URL=
RAPIDAPI_GOOGLE_FLIGHTS_HOST=
RAPIDAPI_VISA_REQUIREMENTS_URL=
RAPIDAPI_VISA_REQUIREMENTS_HOST=

DEFAULT_CITIZENSHIP_COUNTRY_CODE=US
```

Use the exact endpoint URL + host shown in each RapidAPI product page. The app sends standard query parameters
(`query`, `destination`, `currency`, `locale`) and visa parameters (`citizenship`, `destinationCountry`, `destination`).

`TRAVEL_API_ENDPOINTS` still supports optional extra non-RapidAPI sources.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Safety Grounding Notes

The app uses a public travel-advisory feed and explicitly includes U.S. Department of State as a required source in generated output:
- https://travel.state.gov/content/travel.html

Always validate final travel decisions against the latest official advisories and local regulations.
