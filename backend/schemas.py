from datetime import datetime
from pydantic import BaseModel


class DetectionOut(BaseModel):
    id: int
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    behavior: str | None = None
    behavior_confidence: float | None = None
    in_restricted_zone: bool
    direction: str | None = None
    velocity: float | None = None
    acceleration: float | None = None
    timestamp: datetime

    class Config:
        from_attributes = True


class ZoneEventOut(BaseModel):
    id: int
    track_id: int
    event_type: str
    timestamp: datetime

    class Config:
        from_attributes = True


class AnomalyEventOut(BaseModel):
    id: int
    track_id: int
    reasons: list[str]
    incident_summary: str | None = None
    timestamp: datetime

    class Config:
        from_attributes = True


class IdentityOut(BaseModel):
    id: int
    person_id: str
    display_name: str
    consented_at: datetime

    class Config:
        from_attributes = True


class IdentityEnrollIn(BaseModel):
    person_id: str
    display_name: str


class PlateReadOut(BaseModel):
    id: int
    track_id: int
    plate_text: str
    confidence: float
    timestamp: datetime

    class Config:
        from_attributes = True


class SummaryRequest(BaseModel):
    anomaly_event_id: int
