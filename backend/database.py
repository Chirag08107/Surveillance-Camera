"""
Database connection for the surveillance backend.

The connection string is read from DATABASE_URL in the .env file.

Example:
DATABASE_URL=postgresql://surveillance_user:password@127.0.0.1:5433/surveillance_db
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Create a .env file in the project root with your PostgreSQL connection string."
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


Base = declarative_base()


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all database tables defined by the SQLAlchemy models.
    Called when the FastAPI application starts.
    """

    from backend import models  # noqa: F401

    Base.metadata.create_all(bind=engine)