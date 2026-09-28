"""
STEP 14 - DATABASE CONNECTION

Reads DATABASE_URL from the environment (see .env.example). Defaults to
a local Postgres instance matching docker-compose.yml so `docker compose
up` works out of the box.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://surveillance:surveillance@localhost:5432/surveillance_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables. Called once on backend startup."""
    from backend import models  # noqa: F401 - ensures models are registered on Base
    Base.metadata.create_all(bind=engine)
