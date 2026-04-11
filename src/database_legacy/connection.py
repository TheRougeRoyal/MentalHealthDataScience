"""Database connection management with connection pooling."""

import logging
from contextlib import contextmanager
from typing import Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from src.config import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connection pool."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize database connection pool.

        Args:
            database_url: PostgreSQL connection string (defaults to config)
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: Optional[pool.SimpleConnectionPool] = None

    def initialize(self):
        """Initialize the connection pool."""
        if self._pool is not None:
            logger.warning("Connection pool already initialized")
            return

        try:
            self._pool = psycopg2.pool.SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                self.database_url
            )
            logger.info(
                f"Database connection pool initialized "
                f"(min={self.min_connections}, max={self.max_connections})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def close(self):
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed")

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Yields:
            Database connection with automatic return to pool
        """
        if self._pool is None:
            raise RuntimeError("Connection pool not initialized")

        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """
        Get a cursor from a pooled connection.

        Args:
            dict_cursor: If True, return RealDictCursor for dict-like results

        Yields:
            Database cursor
        """
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()


# Global connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """
    Get the global database connection instance.

    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
        _db_connection.initialize()
    return _db_connection
