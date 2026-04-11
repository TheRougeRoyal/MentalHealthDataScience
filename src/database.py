"""Database configuration and session management.

Provides a SQLAlchemy 2.0 engine and session factory with automatic
backend detection: PostgreSQL when credentials are available, SQLite
otherwise.  All module-level state is intentionally lazy-safe so that
imports never trigger network I/O.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------

_SQLITE_FALLBACK = "sqlite:///./mhras.db"


def _build_database_url() -> str:
    """Build the database URL.

    Resolution order:
      1. ``DATABASE_URL`` environment variable (honours Heroku / Railway
         convention).
      2. Assembled from ``settings.database`` fields when *host*, *user*,
         and *name* are all non-empty.
      3. Local SQLite file as a development fallback.
    """
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        # Heroku-style postgres:// → postgresql://
        if explicit_url.startswith("postgres://"):
            explicit_url = explicit_url.replace("postgres://", "postgresql://", 1)
        return explicit_url

    db = settings.database
    if db.password and db.host and db.user and db.name:
        password = quote_plus(db.password)
        return (
            f"postgresql+psycopg2://{db.user}:{password}"
            f"@{db.host}:{db.port}/{db.name}"
        )

    logger.warning("No PostgreSQL credentials found – falling back to SQLite")
    return _SQLITE_FALLBACK


DATABASE_URL: str = _build_database_url()
IS_SQLITE: bool = DATABASE_URL.startswith("sqlite")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _create_engine():
    """Create the SQLAlchemy engine with backend-specific tuning."""
    kwargs: dict = {"future": True, "echo": False}

    if IS_SQLITE:
        kwargs["connect_args"] = {"check_same_thread": False}
        if DATABASE_URL in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        logger.info("Using SQLite backend")
    else:
        db = settings.database
        kwargs.update(
            {
                "pool_pre_ping": True,
                "pool_recycle": 3600,
                "pool_size": db.pool_size,
                "max_overflow": 10,
            }
        )
        logger.info("Using PostgreSQL backend (%s:%s/%s)", db.host, db.port, db.name)

    return create_engine(DATABASE_URL, **kwargs)


engine = _create_engine()


# Enable foreign-key enforcement and WAL journal mode for SQLite.
if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a transactional database session.

    Usage::

        @router.get("/screenings")
        def list_screenings(db: Session = Depends(get_db)):
            ...
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context-manager variant for use outside of FastAPI.

    Usage::

        with get_session() as session:
            session.add(obj)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables registered on :pyattr:`Base.metadata`.

    Safe to call multiple times – existing tables are not recreated.
    In production, prefer Alembic migrations.
    """
    logger.info("Initializing database tables …")
    # Import models so they register with Base.metadata before create_all.
    from src import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")


def check_health() -> bool:
    """Return ``True`` if the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database health-check failed")
        return False
