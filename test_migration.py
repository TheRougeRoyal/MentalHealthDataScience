#!/usr/bin/env python3
"""
Test script to verify database migrations work correctly.
This script creates a temporary test database, runs migrations, and verifies the schema.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.migration_runner import MigrationRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_database(admin_url: str, test_db_name: str) -> str:
    """Create a temporary test database."""
    logger.info(f"Creating test database: {test_db_name}")
    
    # Connect to default postgres database
    conn = psycopg2.connect(admin_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        with conn.cursor() as cur:
            # Drop if exists
            cur.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
            # Create new database
            cur.execute(f"CREATE DATABASE {test_db_name}")
        logger.info(f"Test database {test_db_name} created successfully")
    finally:
        conn.close()
    
    # Return connection string for test database
    return admin_url.rsplit('/', 1)[0] + f'/{test_db_name}'


def drop_test_database(admin_url: str, test_db_name: str):
    """Drop the test database."""
    logger.info(f"Dropping test database: {test_db_name}")
    
    conn = psycopg2.connect(admin_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
        logger.info(f"Test database {test_db_name} dropped successfully")
    finally:
        conn.close()


def verify_tables(database_url: str) -> bool:
    """Verify that all expected tables exist."""
    expected_tables = [
        # Experiment tracking
        'experiments',
        'runs',
        'metrics',
        'artifacts',
        # Data versioning
        'dataset_versions',
        'data_lineage',
        'drift_reports',
        # Feature store
        'features',
        'feature_values',
        # Migration tracking
        'schema_migrations'
    ]
    
    logger.info("Verifying database schema...")
    conn = psycopg2.connect(database_url)
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            
            logger.info(f"Found tables: {existing_tables}")
            
            missing_tables = [t for t in expected_tables if t not in existing_tables]
            
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False
            
            logger.info("✓ All expected tables exist")
            
            # Verify views
            cur.execute("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = 'public'
            """)
            views = [row[0] for row in cur.fetchall()]
            logger.info(f"Found views: {views}")
            
            expected_views = ['latest_runs', 'latest_dataset_versions']
            missing_views = [v for v in expected_views if v not in views]
            
            if missing_views:
                logger.error(f"Missing views: {missing_views}")
                return False
            
            logger.info("✓ All expected views exist")
            
            # Verify indexes
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            indexes = [row[0] for row in cur.fetchall()]
            logger.info(f"Found {len(indexes)} indexes")
            
            return True
            
    finally:
        conn.close()


def main():
    """Main test function."""
    # Get database URL from environment or use default
    admin_url = os.getenv(
        'DATABASE_URL',
        'postgresql://mhras_user:changeme@localhost:5432/postgres'
    )
    
    test_db_name = 'mhras_migration_test'
    
    try:
        # Create test database
        test_db_url = create_test_database(admin_url, test_db_name)
        
        # Run migrations
        logger.info("Running migrations...")
        runner = MigrationRunner(test_db_url)
        applied_count = runner.run_migrations()
        logger.info(f"✓ Applied {applied_count} migrations")
        
        # Verify schema
        if verify_tables(test_db_url):
            logger.info("✓ Migration test PASSED")
            return 0
        else:
            logger.error("✗ Migration test FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"✗ Migration test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        try:
            drop_test_database(admin_url, test_db_name)
        except Exception as e:
            logger.warning(f"Failed to drop test database: {e}")


if __name__ == '__main__':
    sys.exit(main())
