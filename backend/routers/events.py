from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models, schemas
from backend.llm_summary import summarize_incident

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("/zone", response_model=list[schemas.ZoneEventOut])
def list_zone_events(limit: int = 100, db: Session = Depends(get_db)):
    return (
        db.query(models.ZoneEvent)
        .order_by(models.ZoneEvent.timestamp.desc())
        .limit(limit)
        .all()
    )


@router.get("/anomalies", response_model=list[schemas.AnomalyEventOut])
def list_anomalies(limit: int = 100, db: Session = Depends(get_db)):
    return (
        db.query(models.AnomalyEvent)
        .order_by(models.AnomalyEvent.timestamp.desc())
        .limit(limit)
        .all()
    )


@router.post("/anomalies/{event_id}/summarize", response_model=schemas.AnomalyEventOut)
def summarize_anomaly(event_id: int, db: Session = Depends(get_db)):
    event = db.query(models.AnomalyEvent).filter(models.AnomalyEvent.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Anomaly event not found")

    summary = summarize_incident(
        track_id=event.track_id,
        reasons=event.reasons,
        context={"timestamp": event.timestamp.isoformat()},
    )

    event.incident_summary = summary
    db.commit()
    db.refresh(event)
    return event
