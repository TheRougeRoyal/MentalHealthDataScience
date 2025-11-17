# Data Science Enhancements Deployment Guide

This guide covers the deployment of data science enhancements including experiment tracking, data versioning, and feature store capabilities.

## Prerequisites

- PostgreSQL 12+ database
- Redis 6+ (for feature store caching)
- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Kubernetes cluster (for production deployment)

## Configuration Changes

### 1. Python Dependencies

The following new dependencies have been added to `requirements.txt`:

- `plotly>=5.17.0` - Interactive visualizations
- `redis>=5.0.0` - Feature store caching

Existing dependencies that support data science features:
- `seaborn>=0.13.0` - Statistical visualizations
- `optuna>=3.4.0` - Hyperparameter optimization
- `scipy>=1.10.0` - Statistical tests

### 2. Environment Variables

Add the following environment variables to your deployment:

```bash
# Experiment Tracking
EXPERIMENT_STORAGE_BACKEND=filesystem  # or 's3' for production
EXPERIMENT_ARTIFACTS_PATH=/app/experiments/artifacts
EXPERIMENT_DB_TABLE_PREFIX=exp_

# Data Versioning
DATA_VERSION_STORAGE_PATH=/app/data/versions
DATA_VERSION_COMPRESSION=gzip
DATA_VERSION_DEDUPLICATION=true

# Feature Store
FEATURE_STORE_CACHE_BACKEND=redis
FEATURE_STORE_REDIS_URL=redis://redis:6379
FEATURE_STORE_CACHE_TTL=3600

# EDA Module
EDA_MAX_DATASET_SIZE=1000000
EDA_VISUALIZATION_DPI=300
EDA_REPORT_TEMPLATE=default

# Model Cards
MODEL_CARD_TEMPLATE=default
MODEL_CARD_INCLUDE_SHAP=true
MODEL_CARD_INCLUDE_FAIRNESS=true
```

## Deployment Options

### Option 1: Docker Compose (Development/Testing)

1. **Start all services:**

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database
- Redis cache
- MHRAS API service

2. **Run database migrations:**

```bash
docker-compose exec api python3 -m src.database.migration_runner \
  --database-url "${DATABASE_URL}"
```

Or use the convenience script:

```bash
docker-compose exec api ./run_migrations.sh
```

3. **Verify services:**

```bash
# Check API health
curl http://localhost:8000/health

# Check Redis
docker-compose exec redis redis-cli ping

# Check PostgreSQL
docker-compose exec postgres psql -U mhras_user -d mhras -c "\dt"
```

### Option 2: Kubernetes (Production)

1. **Deploy Redis:**

```bash
kubectl apply -f k8s/redis-deployment.yaml
```

This creates:
- Redis deployment with persistent storage
- Redis service for internal communication
- PersistentVolumeClaim for Redis data

2. **Update ConfigMap:**

```bash
kubectl apply -f k8s/configmap.yaml
```

The ConfigMap now includes all data science configuration variables.

3. **Deploy API with updated configuration:**

```bash
kubectl apply -f k8s/deployment.yaml
```

This creates:
- Updated API deployment with new environment variables
- PersistentVolumeClaims for experiments and datasets
- Volume mounts for artifact storage

4. **Run database migrations:**

```bash
# Get a pod name
POD=$(kubectl get pods -l app=mhras,component=api -o jsonpath='{.items[0].metadata.name}')

# Run migrations
kubectl exec -it $POD -- python3 -m src.database.migration_runner \
  --database-url "${DATABASE_URL}"
```

Or use the convenience script:

```bash
kubectl exec -it $POD -- ./run_migrations.sh
```

## Database Migrations

### Migration 003: Data Science Schema

The migration creates the following tables:

**Experiment Tracking:**
- `experiments` - Experiment metadata
- `runs` - Individual experiment runs
- `metrics` - Time-series metrics
- `artifacts` - Stored artifacts

**Data Versioning:**
- `dataset_versions` - Versioned datasets
- `data_lineage` - Transformation lineage
- `drift_reports` - Data drift analysis

**Feature Store:**
- `features` - Feature definitions
- `feature_values` - Computed feature values

### Running Migrations

**Manual execution:**

```bash
python3 -m src.database.migration_runner \
  --database-url "postgresql://user:pass@host:port/dbname"
```

**Using the script:**

```bash
export DATABASE_URL="postgresql://user:pass@host:port/dbname"
./run_migrations.sh
```

**Testing migrations:**

```bash
# Test on a temporary database
python3 test_migration.py
```

## Storage Requirements

### Persistent Volumes

The deployment requires the following persistent storage:

1. **Experiments** (50GB recommended)
   - Stores experiment artifacts, models, plots
   - Path: `/app/experiments`

2. **Datasets** (100GB recommended)
   - Stores versioned datasets
   - Path: `/app/data/versions`

3. **Models** (10GB existing)
   - Stores trained models
   - Path: `/app/models`

4. **Redis** (5GB recommended)
   - Feature store cache
   - Path: `/data`

### Docker Compose Volumes

Volumes are automatically created and managed by Docker Compose:

```bash
# List volumes
docker volume ls | grep mhras

# Inspect a volume
docker volume inspect mhras_experiments_data
```

### Kubernetes PersistentVolumeClaims

PVCs are defined in the deployment manifests:

```bash
# Check PVC status
kubectl get pvc

# Check PVC details
kubectl describe pvc mhras-experiments-pvc
```

## Redis Configuration

### Docker Compose

Redis is configured with:
- Append-only file (AOF) persistence
- 512MB max memory
- LRU eviction policy

### Kubernetes

Redis deployment includes:
- Persistent storage (5GB)
- Resource limits (512Mi memory, 500m CPU)
- Health checks (liveness and readiness probes)

### Connecting to Redis

**From application:**
```python
import redis
r = redis.from_url(os.getenv('FEATURE_STORE_REDIS_URL'))
```

**Testing connection:**
```bash
# Docker Compose
docker-compose exec redis redis-cli ping

# Kubernetes
kubectl exec -it <redis-pod> -- redis-cli ping
```

## Verification

### 1. Verify Database Schema

```bash
# Connect to database
psql $DATABASE_URL

# List tables
\dt

# Check experiment tracking tables
SELECT COUNT(*) FROM experiments;
SELECT COUNT(*) FROM runs;

# Check data versioning tables
SELECT COUNT(*) FROM dataset_versions;

# Check feature store tables
SELECT COUNT(*) FROM features;
```

### 2. Verify Redis Connection

```bash
# Test Redis from API pod
kubectl exec -it <api-pod> -- python3 -c "
import redis
import os
r = redis.from_url(os.getenv('FEATURE_STORE_REDIS_URL'))
print('Redis ping:', r.ping())
"
```

### 3. Verify API Endpoints

```bash
# Test experiment tracking endpoint
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "test", "description": "Test experiment"}'

# Test feature store endpoint
curl http://localhost:8000/api/v1/features
```

## Monitoring

### Metrics to Monitor

1. **Redis:**
   - Memory usage
   - Cache hit rate
   - Connection count

2. **Storage:**
   - Disk usage for experiments volume
   - Disk usage for datasets volume
   - Growth rate

3. **Database:**
   - Table sizes (experiments, runs, metrics)
   - Query performance
   - Connection pool usage

### Prometheus Metrics

The API exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics | grep -E "experiment|feature|dataset"
```

## Troubleshooting

### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis  # Docker Compose
kubectl get pods -l component=redis  # Kubernetes

# Check Redis logs
docker-compose logs redis  # Docker Compose
kubectl logs -l component=redis  # Kubernetes

# Test connection
redis-cli -h <host> -p 6379 ping
```

### Migration Failures

```bash
# Check migration status
psql $DATABASE_URL -c "SELECT * FROM schema_migrations ORDER BY applied_at DESC;"

# Manually rollback if needed (use with caution)
psql $DATABASE_URL -c "DELETE FROM schema_migrations WHERE migration_name = '003_data_science_schema.sql';"

# Re-run migrations
./run_migrations.sh
```

### Storage Issues

```bash
# Check disk usage (Docker)
docker system df -v

# Check PVC usage (Kubernetes)
kubectl exec -it <api-pod> -- df -h /app/experiments
kubectl exec -it <api-pod> -- df -h /app/data/versions

# Clean up old artifacts if needed
kubectl exec -it <api-pod> -- find /app/experiments -type f -mtime +90 -delete
```

## Rollback Procedure

If you need to rollback the data science enhancements:

1. **Stop using new features** in application code

2. **Remove Redis deployment** (Kubernetes):
```bash
kubectl delete -f k8s/redis-deployment.yaml
```

3. **Revert ConfigMap** to previous version:
```bash
kubectl apply -f k8s/configmap.yaml.backup
```

4. **Revert deployment** to previous version:
```bash
kubectl rollout undo deployment/mhras-api
```

5. **Optionally drop new tables** (if needed):
```sql
-- Connect to database
psql $DATABASE_URL

-- Drop data science tables
DROP TABLE IF EXISTS feature_values CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS drift_reports CASCADE;
DROP TABLE IF EXISTS data_lineage CASCADE;
DROP TABLE IF EXISTS dataset_versions CASCADE;
DROP TABLE IF EXISTS artifacts CASCADE;
DROP TABLE IF EXISTS metrics CASCADE;
DROP TABLE IF EXISTS runs CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;

-- Remove migration record
DELETE FROM schema_migrations WHERE migration_name = '003_data_science_schema.sql';
```

## Security Considerations

1. **Redis Security:**
   - Use password authentication in production
   - Enable TLS for Redis connections
   - Restrict network access to Redis

2. **Storage Security:**
   - Encrypt volumes at rest
   - Use appropriate access controls on PVCs
   - Regular backups of experiment data

3. **Database Security:**
   - Use strong passwords
   - Enable SSL for database connections
   - Regular backups

## Next Steps

After deployment:

1. Review the [Experiment Tracking Usage Guide](docs/experiment_tracking_usage.md)
2. Review the [Data Versioning Usage Guide](docs/data_versioning_usage.md)
3. Review the [Feature Store Usage Guide](docs/feature_store_usage.md)
4. Review the [EDA Usage Guide](docs/eda_usage.md)
5. Explore example notebooks in `examples/`

## Support

For issues or questions:
- Check the troubleshooting section above
- Review application logs
- Consult the main [Deployment Guide](docs/deployment_guide.md)
