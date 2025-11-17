# Database Layer Usage Guide

This guide explains how to use the database layer for the Mental Health Risk Assessment System (MHRAS).

## Overview

The database layer provides:
- **Migration scripts** for schema management
- **Connection pooling** for efficient database access
- **Repository pattern** for clean data access
- **Transaction management** for data consistency

## Database Schema

The system uses four main tables:

### predictions
Stores risk prediction results with model metadata.

### audit_log
Comprehensive audit trail for all system activities.

### consent
Tracks consent status for data processing.

### human_review_queue
Manages high-risk cases requiring human review.

## Setup

### 1. Configure Database Connection

Set environment variables in `.env`:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mhras
DB_USER=mhras_user
DB_PASSWORD=your_secure_password
DB_POOL_SIZE=10
```

### 2. Run Migrations

```python
from src.database.migration_runner import MigrationRunner

# Initialize and run migrations
runner = MigrationRunner(database_url="postgresql://user:pass@host:port/dbname")
runner.run_migrations()
```

Or use the CLI:

```bash
python -m src.database.migration_runner --database-url "postgresql://user:pass@host:port/dbname"
```

### 3. Initialize Connection Pool

```python
from src.database import get_db_connection

# Get global connection instance
db = get_db_connection()
```

## Using Repositories

### PredictionRepository

```python
from src.database import get_db_connection, PredictionRepository
from src.database.models import Prediction

# Initialize repository
db = get_db_connection()
pred_repo = PredictionRepository(db)

# Create a prediction
prediction = Prediction(
    anonymized_id="abc123",
    risk_score=75.5,
    risk_level="HIGH",
    confidence=0.85,
    model_version="v1.0.0",
    features_hash="hash123",
    contributing_factors={"sleep": 0.4, "mood": 0.3}
)
prediction_id = pred_repo.create(prediction)

# Get prediction by ID
prediction = pred_repo.get_by_id(prediction_id)

# Get predictions for an individual
predictions = pred_repo.get_by_anonymized_id("abc123", limit=10)

# Get recent high-risk predictions
high_risk = pred_repo.get_recent_predictions(hours=24, risk_level="HIGH")
```

### AuditLogRepository

```python
from src.database import AuditLogRepository
from src.database.models import AuditLog

audit_repo = AuditLogRepository(db)

# Create audit log entry
log = AuditLog(
    event_type="SCREENING_REQUEST",
    anonymized_id="abc123",
    user_id="clinician_456",
    details={
        "request_id": "req_789",
        "risk_score": 75.5,
        "model_version": "v1.0.0"
    }
)
log_id = audit_repo.create(log)

# Get logs by event type
logs = audit_repo.get_by_event_type(
    "SCREENING_REQUEST",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)

# Get logs for an individual
logs = audit_repo.get_by_anonymized_id("abc123", limit=100)

# Get summary of events
summary = audit_repo.get_summary(
    start_date=datetime(2025, 11, 1),
    end_date=datetime(2025, 11, 30)
)
# Returns: {"SCREENING_REQUEST": 150, "PREDICTION": 150, ...}
```

### ConsentRepository

```python
from src.database import ConsentRepository
from src.database.models import Consent
from datetime import datetime, timedelta

consent_repo = ConsentRepository(db)

# Create consent record
consent = Consent(
    anonymized_id="abc123",
    data_types=["survey", "wearable", "emr"],
    expires_at=datetime.utcnow() + timedelta(days=365)
)
consent_repo.create(consent)

# Get consent record
consent = consent_repo.get_by_anonymized_id("abc123")

# Check if consent is valid
is_valid = consent_repo.is_valid("abc123", data_types=["survey", "wearable"])

# Revoke consent
consent_repo.revoke("abc123")
```

### HumanReviewQueueRepository

```python
from src.database import HumanReviewQueueRepository
from src.database.models import HumanReviewCase

review_repo = HumanReviewQueueRepository(db)

# Create review case
case = HumanReviewCase(
    anonymized_id="abc123",
    risk_score=85.0,
    prediction_id=prediction_id,
    status="PENDING",
    priority="HIGH"
)
case_id = review_repo.create(case)

# Get pending cases
pending = review_repo.get_pending_cases(limit=50)

# Assign case to reviewer
review_repo.assign_case(case_id, reviewer_id="clinician_456")

# Submit review decision
review_repo.submit_review(
    case_id,
    decision="CONFIRMED",
    decision_notes="Risk assessment confirmed after clinical review"
)

# Escalate overdue cases
overdue = review_repo.get_overdue_cases(hours=4)
for case in overdue:
    review_repo.escalate_case(case.case_id)
```

## Transaction Management

Repositories automatically handle transactions through the connection manager:

```python
from src.database import get_db_connection

db = get_db_connection()

# Transactions are automatically committed on success
with db.get_cursor() as cur:
    cur.execute("INSERT INTO ...")
    # Automatically committed when context exits

# Transactions are automatically rolled back on error
try:
    with db.get_cursor() as cur:
        cur.execute("INSERT INTO ...")
        raise Exception("Something went wrong")
except Exception:
    # Automatically rolled back
    pass
```

## Connection Pooling

The connection pool is managed automatically:

```python
from src.database import DatabaseConnection

# Initialize with custom pool size
db = DatabaseConnection(
    database_url="postgresql://...",
    min_connections=2,
    max_connections=20
)
db.initialize()

# Use connections
with db.get_connection() as conn:
    # Connection automatically returned to pool
    pass

# Close pool when shutting down
db.close()
```

## Best Practices

1. **Use repositories** instead of raw SQL queries for consistency
2. **Initialize connection pool** once at application startup
3. **Close connection pool** during graceful shutdown
4. **Use context managers** for automatic resource cleanup
5. **Log all database operations** for audit trail
6. **Handle exceptions** appropriately and rollback on errors
7. **Use connection pooling** to avoid connection overhead
8. **Index frequently queried columns** for performance

## Error Handling

```python
from src.database import get_db_connection, PredictionRepository
from src.exceptions import DatabaseError

db = get_db_connection()
pred_repo = PredictionRepository(db)

try:
    prediction_id = pred_repo.create(prediction)
except Exception as e:
    logger.error(f"Failed to create prediction: {e}")
    raise DatabaseError(f"Database operation failed: {e}")
```

## Performance Considerations

- **Connection pooling** reduces connection overhead
- **Indexes** are created on frequently queried columns
- **Batch operations** should be used for bulk inserts
- **Query optimization** through proper index usage
- **Connection limits** prevent resource exhaustion

## Monitoring

Monitor these metrics:
- Connection pool utilization
- Query execution times
- Transaction rollback rates
- Database connection errors
- Table sizes and growth rates

## Migration Management

Migrations are tracked in the `schema_migrations` table:

```sql
SELECT * FROM schema_migrations ORDER BY applied_at DESC;
```

To add a new migration:
1. Create a new SQL file in `src/database/migrations/`
2. Name it with sequential number: `002_add_feature.sql`
3. Run the migration runner
4. Migration is automatically tracked and won't run again

## Security

- **Use parameterized queries** to prevent SQL injection
- **Store credentials** in environment variables, not code
- **Encrypt connections** using SSL/TLS in production
- **Limit database permissions** to minimum required
- **Audit all data access** through audit_log table
- **Anonymize identifiers** before storing in database
