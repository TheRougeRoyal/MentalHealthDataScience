"""Database migration runner for MHRAS."""

import os
import logging
from pathlib import Path
from typing import List
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)


class MigrationRunner:
    """Runs database migrations from SQL files."""

    def __init__(self, database_url: str):
        """
        Initialize migration runner.

        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent / "migrations"

    def _get_migration_files(self) -> List[Path]:
        """Get sorted list of migration files."""
        if not self.migrations_dir.exists():
            return []

        migration_files = [
            f for f in self.migrations_dir.glob("*.sql")
            if f.name != "README.md"
        ]
        return sorted(migration_files)

    def _create_migrations_table(self, conn):
        """Create migrations tracking table if it doesn't exist."""
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    migration_name VARCHAR(255) NOT NULL UNIQUE,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()

    def _is_migration_applied(self, conn, migration_name: str) -> bool:
        """Check if migration has already been applied."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE migration_name = %s",
                (migration_name,)
            )
            count = cur.fetchone()[0]
            return count > 0

    def _mark_migration_applied(self, conn, migration_name: str):
        """Mark migration as applied."""
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schema_migrations (migration_name) VALUES (%s)",
                (migration_name,)
            )
        conn.commit()

    def run_migrations(self) -> int:
        """
        Run all pending migrations.

        Returns:
            Number of migrations applied
        """
        migration_files = self._get_migration_files()
        if not migration_files:
            logger.info("No migration files found")
            return 0

        conn = psycopg2.connect(self.database_url)
        try:
            # Create migrations tracking table
            self._create_migrations_table(conn)

            applied_count = 0
            for migration_file in migration_files:
                migration_name = migration_file.name

                if self._is_migration_applied(conn, migration_name):
                    logger.info(f"Migration {migration_name} already applied, skipping")
                    continue

                logger.info(f"Applying migration: {migration_name}")

                # Read and execute migration SQL
                with open(migration_file, 'r') as f:
                    sql = f.read()

                with conn.cursor() as cur:
                    cur.execute(sql)

                # Mark as applied
                self._mark_migration_applied(conn, migration_name)
                applied_count += 1

                logger.info(f"Successfully applied migration: {migration_name}")

            logger.info(f"Applied {applied_count} migrations")
            return applied_count

        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL connection string"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    runner = MigrationRunner(args.database_url)
    runner.run_migrations()
