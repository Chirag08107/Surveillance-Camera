from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models, schemas

router = APIRouter(prefix="/api/detections", tags=["detections"])


@router.get("/", response_model=list[schemas.DetectionOut])
def list_detections(
    track_id: int | None = None,
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
):
    query = db.query(models.Detection)
    if track_id is not None:
        query = query.filter(models.Detection.track_id == track_id)
    return query.order_by(models.Detection.timestamp.desc()).limit(limit).all()


@router.get("/latest", response_model=list[schemas.DetectionOut])
def latest_frame(db: Session = Depends(get_db)):
    """Most recent detection per active track - what the dashboard's live view polls."""
    latest = (
        db.query(models.Detection)
        .order_by(models.Detection.track_id, models.Detection.timestamp.desc())
        .distinct(models.Detection.track_id)
        .all()
    )
    return latest
