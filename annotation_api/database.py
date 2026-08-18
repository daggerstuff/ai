"""Database initialization and session management for the annotation queue."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

# Use SQLite database in the data directory
DATABASE_URL = os.getenv(
    "ANNOTATION_DATABASE_URL",
    "sqlite:///data/annotation_queue.db",
)


def get_engine():
    """Get the SQLAlchemy engine."""
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def get_session_factory():
    """Get a session factory."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the database by creating all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session]:
    """Get a database session as a context manager."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
