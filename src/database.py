"""Database configuration and session management using SQLAlchemy."""

import logging
import os
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import settings

logger = logging.getLogger(__name__)


def _build_database_url() -> str:
    """Resolve the database URL with PostgreSQL preferred and SQLite fallback."""
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    db_config = settings.database
    if db_config.host and db_config.user and db_config.name:
        password = quote_plus(db_config.password or "")
        return (
            f"postgresql+psycopg2://{db_config.user}:{password}"
            f"@{db_config.host}:{db_config.port}/{db_config.name}"
        )

    return "sqlite:///./mhras.db"


DATABASE_URL = _build_database_url()
is_sqlite = DATABASE_URL.startswith("sqlite")

engine_kwargs = {
    "future": True,
}

if is_sqlite:
    engine_kwargs.update(
        {
            "connect_args": {"check_same_thread": False},
        }
    )
    if DATABASE_URL in {"sqlite://", "sqlite:///:memory:"}:
        # Needed for in-memory SQLite so all sessions share one connection.
        engine_kwargs["poolclass"] = StaticPool
    logger.info("Configured SQLite database")
else:
    db_config = settings.database
    engine_kwargs.update(
        {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": db_config.pool_size,
            "max_overflow": 10,
        }
    )
    logger.info(
        "Configured PostgreSQL database: %s:%s/%s",
        db_config.host,
        db_config.port,
        db_config.name,
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)

if is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        """Enable SQLite foreign key constraints."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function that provides a database session.

    Yields:
        SQLAlchemy Session object

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Call this during application startup to ensure tables exist.
    For production, use Alembic migrations instead.
    """
    logger.info("Initializing database tables...")
    try:
        # Import all models to ensure they're registered with Base
        from src import models  # noqa: F401

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_db_connection() -> Session:
    """
    Get a database connection for legacy code compatibility.

    Returns:
        SQLAlchemy Session object

    Note: Prefer using get_db() dependency injection in new code.
    """
    return SessionLocal()
