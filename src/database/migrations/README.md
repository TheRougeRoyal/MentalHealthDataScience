# Database Migrations

This directory contains SQL migration scripts for the MHRAS database schema.

## Migration Files

- `001_initial_schema.sql`: Initial schema creation for predictions, audit_log, consent, and human_review_queue tables
- `002_resources_schema.sql`: Schema for recommendation resources and resource catalog
- `003_data_science_schema.sql`: Data science enhancements including experiment tracking, data versioning, and feature store

## Running Migrations

Migrations can be run using the migration runner:

```python
from src.database.migration_runner import MigrationRunner

runner = MigrationRunner(database_url)
runner.run_migrations()
```

Or using the CLI:

```bash
python -m src.database.migration_runner --database-url "postgresql://user:pass@host:port/dbname"
```

## Migration Naming Convention

Migrations should be named with a sequential number prefix followed by a descriptive name:
- `001_initial_schema.sql`
- `002_resources_schema.sql`
- `003_data_science_schema.sql`
- etc.

## Testing Migrations

To test migrations on a clean database, use the test script:

```bash
# Set database connection (optional, defaults to localhost)
export DATABASE_URL="postgresql://mhras_user:changeme@localhost:5432/postgres"

# Run migration test
python test_migration.py
```

This will:
1. Create a temporary test database
2. Run all migrations
3. Verify all tables, indexes, and views exist
4. Clean up the test database

## Schema Overview

### predictions
Stores risk prediction results with model metadata and lineage information.

### audit_log
Comprehensive audit trail for all system activities and decisions.

### consent
Tracks consent status for data processing with granular data type permissions.

### human_review_queue
Manages high-risk cases requiring human review with priority and escalation tracking.

### experiments
Stores experiment metadata for ML experiment tracking.

### runs
Tracks individual experiment runs with parameters, status, and version information.

### metrics
Stores time-series metrics logged during experiment runs.

### artifacts
Tracks artifacts (models, plots, data files) generated during experiments.

### dataset_versions
Versions datasets with metadata, statistics, and lineage information.

### data_lineage
Tracks data transformation lineage between dataset versions.

### drift_reports
Stores data drift analysis reports comparing dataset versions.

### features
Defines feature engineering transformations in the feature store.

### feature_values
Stores computed feature values for entities.
