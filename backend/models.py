"""
ORM MODELS

Stores detections, zone events, anomaly events, identities,
and license plate reads produced by the surveillance system.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    JSON,
    Index,
)

from backend.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    # Monitoring session that produced this detection.
    # Nullable for compatibility with older records.
    session_id = Column(String(36), nullable=True, index=True)

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

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )


class ZoneEvent(Base):
    __tablename__ = "zone_events"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String(36), nullable=True, index=True)

    track_id = Column(Integer, index=True)
    event_type = Column(String)  # "enter" or "exit"

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String(36), nullable=True, index=True)

    track_id = Column(Integer, index=True)

    # Example: ["classifier:Anomaly"]
    reasons = Column(JSON)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )

    incident_summary = Column(String, nullable=True)


class Identity(Base):
    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, index=True)

    person_id = Column(String, unique=True, index=True)
    display_name = Column(String)

    consented_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


class PlateRead(Base):
    __tablename__ = "plate_reads"

    id = Column(Integer, primary_key=True, index=True)

    track_id = Column(Integer, index=True)
    plate_text = Column(String, index=True)
    confidence = Column(Float)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )