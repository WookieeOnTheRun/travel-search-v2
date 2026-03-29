from __future__ import annotations

from pydantic import BaseModel, Field


class UserTripRequest(BaseModel):
    request_text: str = Field(min_length=10)
    origin_city: str | None = None
    trip_length_days: int | None = None
    traveler_count: int = 1


class GroundingPacket(BaseModel):
    destination_hint: str
    weather_summary: str
    safety_summary: str
    advisory_level: str
    advisory_source_url: str
    local_transport_notes: str
    external_api_snippets: list[str] = Field(default_factory=list)
