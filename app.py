from __future__ import annotations

import asyncio
import re
from typing import Any

import streamlit as st

from config import ENV_FILE_IN_USE
from travel_agent import build_itineraries
from travel_agent.schemas import UserTripRequest


# Helper to parse a single markdown table row into a list of cells
def _parse_markdown_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


# Extracts the 'Deterministic Scoring Table' from the agent's markdown response
# and converts it into a list of structured dictionaries for display/analysis.
def _extract_scoring_table(markdown_text: str) -> list[dict[str, Any]]:
    lines = markdown_text.splitlines()
    heading_index = -1


    for index, line in enumerate(lines):
        if re.search(r"deterministic\s+scoring\s+table", line, flags=re.IGNORECASE):
            heading_index = index
            break

    if heading_index == -1:
        return []

    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.strip().startswith("|"):
            table_lines.append(line)
            continue
        if table_lines:
            break

    if len(table_lines) < 2:
        return []

    headers = _parse_markdown_row(table_lines[0])
    if not headers:
        return []

    records: list[dict[str, Any]] = []
    numeric_candidates = {
        "rating_quality",
        "review_confidence",
        "safety",
        "budget_fit",
        "final_score",
    }

    for row in table_lines[1:]:
        if re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$", row.strip()):
            continue

        values = _parse_markdown_row(row)
        if len(values) != len(headers):
            continue

        item: dict[str, Any] = {}
        for header, value in zip(headers, values):
            key = header.strip().lower().replace(" ", "_")
            cleaned_value = value.strip()

            if key in numeric_candidates:
                number_match = re.search(r"-?\d+(?:\.\d+)?", cleaned_value)
                item[header] = float(number_match.group()) if number_match else cleaned_value
            else:
                item[header] = cleaned_value

        records.append(item)

    return records

# UI Configuration and Page Setup
st.set_page_config(page_title="Agentic Travel Concierge", page_icon="🧳", layout="wide")

st.title("🧳 Agentic Travel Concierge")
st.write(
    "Describe your ideal trip (budget, climate, activities, pace, dates, and preferences). "
    "The app generates multiple luxury-within-budget itinerary options with safety-aware guidance."
)
if ENV_FILE_IN_USE:
    st.caption(f"ENV file in use: {ENV_FILE_IN_USE}")
else:
    st.caption("ENV file in use: default process environment (no env file found)")

# User Input Form for Trip Requests
with st.form("trip_form"):
    request_text = st.text_area(
        "Travel request",
        height=180,
        placeholder=(
            "Example: Plan a 7-day honeymoon in Japan from Seattle for 2 travelers with a total budget under $7000, "
            "moderate weather, food tours, ryokan stay, and low-risk neighborhoods."
        ),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        origin_city = st.text_input("Origin city (optional)")
    with col2:
        trip_length_days = st.number_input("Trip length (days, optional)", min_value=0, max_value=60, value=0)
    with col3:
        traveler_count = st.number_input("Number of travelers", min_value=1, max_value=20, value=1)

    submitted = st.form_submit_button("Generate itinerary options")

if submitted:
    if len(request_text.strip()) < 10:
        st.warning("Please provide a more detailed request.")
    else:
        with st.spinner("Planning your trip with the agentic workflow..."):
            request = UserTripRequest(
                request_text=request_text.strip(),
                origin_city=origin_city.strip() or None,
                trip_length_days=trip_length_days if trip_length_days > 0 else None,
                traveler_count=traveler_count,
            )
            result = asyncio.run(build_itineraries(request))

        st.subheader(f"Destination focus: {result['destination']}")
        st.markdown(result["itinerary_markdown"])

        score_rows = _extract_scoring_table(result["itinerary_markdown"])
        if score_rows:
            st.subheader("Deterministic Score Breakdown")
            st.dataframe(score_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Scoring table not available for this run.")

        st.divider()
        st.caption("Safety grounding")
        st.write(result["safety_summary"])
        st.write(f"Primary advisory source: {result['advisory_source_url']}")
