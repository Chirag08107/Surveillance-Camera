"""
STEP 14 - ORM MODELS

Stores exactly what the pipeline (main.py's logic, extended in
pipeline.py) already produces per frame: detections, zone events,
behavior classifications, anomaly flags, enrolled identities, and
plate reads.
"""

from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    class_name = Column(String)
    confidence = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    center_x = Column(Float)
    center_y = Column(Float)
    behavior = Column(String, nullable=True)
    behavior_confidence = Column(Float, nullable=True)
    in_restricted_zone = Column(Boolean, default=False)
    direction = Column(String, nullable=True)
    velocity = Column(Float, nullable=True)
    acceleration = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ZoneEvent(Base):
    __tablename__ = "zone_events"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    event_type = Column(String)  # "enter" | "exit"
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    reasons = Column(JSON)  # list[str], e.g. ["motion_outlier", "loitering_in_restricted_zone"]
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    incident_summary = Column(String, nullable=True)  # filled in by LLM summary step


class Identity(Base):
    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    consented_at = Column(DateTime, default=datetime.utcnow)


class PlateRead(Base):
    __tablename__ = "plate_reads"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    plate_text = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
