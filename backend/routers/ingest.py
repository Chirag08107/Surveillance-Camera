"""
pipeline.py (the CV process) POSTs its per-frame results here rather than
writing to Postgres directly. That keeps the CV process and the API
process decoupled (you can run pipeline.py on a machine with a GPU and
point it at a backend running elsewhere) and lets this single endpoint
fan results out to both the database and any connected dashboard
websocket clients.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models
from backend.routers.stream import broadcaster

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


class DetectionIn(BaseModel):
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
    in_restricted_zone: bool = False
    direction: str | None = None
    velocity: float | None = None
    acceleration: float | None = None


class ZoneEventIn(BaseModel):
    track_id: int
    event_type: str  # "enter" | "exit"


class AnomalyEventIn(BaseModel):
    track_id: int
    reasons: list[str]


class FrameIngest(BaseModel):
    detections: list[DetectionIn] = []
    zone_events: list[ZoneEventIn] = []
    anomaly_events: list[AnomalyEventIn] = []


@router.post("/frame")
async def ingest_frame(payload: FrameIngest, db: Session = Depends(get_db)):
    for d in payload.detections:
        db.add(models.Detection(**d.model_dump()))

    for e in payload.zone_events:
        db.add(models.ZoneEvent(**e.model_dump()))

    for a in payload.anomaly_events:
        db.add(models.AnomalyEvent(**a.model_dump()))

    db.commit()

    await broadcaster.broadcast(
        {
            "type": "frame",
            "timestamp": datetime.utcnow().isoformat(),
            "detections": [d.model_dump() for d in payload.detections],
            "zone_events": [e.model_dump() for e in payload.zone_events],
            "anomaly_events": [a.model_dump() for a in payload.anomaly_events],
        }
    )

    return {"status": "ok", "ingested": len(payload.detections)}
