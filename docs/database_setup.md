# Database Setup Guide

This guide provides comprehensive instructions for setting up the MHRAS PostgreSQL database with consistent credentials across all environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Default Credentials](#default-credentials)
- [Setup Methods](#setup-methods)
- [Environment-Specific Setup](#environment-specific-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Automated Setup (Recommended)

The easiest way to set up the database is using the automated setup script:

```bash
# Run the setup script
./setup_database.sh
```

This script will:
1. Check if PostgreSQL is installed and running
2. Create the database and user with consistent credentials
3. Run all migrations
4. Create/update your `.env` file
5. Verify the setup

### Manual Setup

If you prefer manual setup, see the [Manual Setup](#manual-setup) section below.

---

## Default Credentials

All MHRAS configurations use these consistent default credentials for development:

| Parameter | Value |
|-----------|-------|
| **Host** | `localhost` |
| **Port** | `5432` |
| **Database** | `mhras` |
| **User** | `mhras_user` |
| **Password** | `mhras_dev_password_2024` |

**Connection String:**
```
postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras
```

> ⚠️ **IMPORTANT**: These are development credentials. **Always change them in production!**

---

## Setup Methods

### Method 1: Automated Script (Recommended)

```bash
# Make script executable (if not already)
chmod +x setup_database.sh

# Run setup
./setup_database.sh
```

The script will guide you through the process and handle:
- PostgreSQL installation check
- Database and user creation
- Migration execution
- Environment file configuration
- Setup verification

### Method 2: Docker Compose

```bash
# Start PostgreSQL with Docker Compose
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready

# Run migrations
./run_migrations.sh
```

### Method 3: Manual Setup

#### Step 1: Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**RHEL/CentOS:**
```bash
sudo yum install postgresql postgresql-server
sudo postgresql-setup initdb
sudo systemctl start postgresql
```

#### Step 2: Create Database and User

```bash
# Connect as postgres superuser
sudo -u postgres psql

# In psql, run:
CREATE USER mhras_user WITH PASSWORD 'mhras_dev_password_2024';
CREATE DATABASE mhras OWNER mhras_user;
GRANT ALL PRIVILEGES ON DATABASE mhras TO mhras_user;
ALTER USER mhras_user CREATEDB;
\q
```

#### Step 3: Configure Environment

```bash
# Copy example configuration
cp config/.env.example .env

# The .env file already has the correct default credentials
# Edit if you need to change them
nano .env
```

#### Step 4: Run Migrations

```bash
# Using the migration script
./run_migrations.sh

# Or directly with Python
python3 -m src.database.migration_runner \
  --database-url "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras"
```

---

## Environment-Specific Setup

### Development Environment

Use the default credentials as-is:

```bash
# .env file
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mhras
DB_USER=mhras_user
DB_PASSWORD=mhras_dev_password_2024
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f postgres

# Run migrations (if not auto-run)
docker-compose exec api python -m src.database.migration_runner \
  --database-url "postgresql://mhras_user:mhras_dev_password_2024@postgres:5432/mhras"
```

### Kubernetes/Production

1. **Update the secret:**

```bash
# Edit k8s/secret.yaml
kubectl create secret generic mhras-secrets \
  --from-literal=db_host=your-db-host \
  --from-literal=db_port=5432 \
  --from-literal=db_name=mhras \
  --from-literal=db_user=mhras_user \
  --from-literal=db_password=YOUR_SECURE_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -
```

2. **Apply configurations:**

```bash
kubectl apply -f k8s/
```

3. **Run migrations:**

```bash
# Create a migration job
kubectl run mhras-migrate --rm -it --restart=Never \
  --image=mhras-api:latest \
  --env="DB_HOST=postgres-service" \
  --env="DB_PORT=5432" \
  --env="DB_NAME=mhras" \
  --env="DB_USER=mhras_user" \
  --env="DB_PASSWORD=YOUR_SECURE_PASSWORD" \
  -- python -m src.database.migration_runner
```

---

## Verification

### Check Database Connection

```bash
# Using psql
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c '\dt'

# Using Python
python3 -c "
from src.database.connection import get_db_connection
db = get_db_connection()
print('✓ Database connection successful')
"
```

### Verify Tables

```bash
# List all tables
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c "
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema='public' 
ORDER BY table_name;
"
```

Expected tables:
- `predictions`
- `audit_log`
- `consent`
- `human_review_queue`
- `resources`
- `resource_recommendations`
- `experiments`
- `runs`
- `metrics`
- `artifacts`
- `dataset_versions`
- `data_lineage`
- `drift_reports`
- `features`
- `feature_values`
- `schema_migrations`

### Check Migration Status

```bash
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c "
SELECT migration_name, applied_at 
FROM schema_migrations 
ORDER BY applied_at;
"
```

Expected migrations:
- `001_initial_schema.sql`
- `002_resources_schema.sql`
- `003_data_science_schema.sql`

### Run Integration Test

```bash
# Test database integration
python3 verify_integration.py

# Or run specific database tests
pytest tests/test_database.py -v
```

---

## Troubleshooting

### PostgreSQL Not Running

**Error:** `could not connect to server: Connection refused`

**Solution:**
```bash
# Check status
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Enable auto-start
sudo systemctl enable postgresql
```

### Authentication Failed

**Error:** `FATAL: password authentication failed for user "mhras_user"`

**Solution:**
1. Check your `.env` file has the correct password
2. Verify the user exists:
   ```bash
   sudo -u postgres psql -c "\du"
   ```
3. Reset the password if needed:
   ```bash
   sudo -u postgres psql -c "ALTER USER mhras_user WITH PASSWORD 'mhras_dev_password_2024';"
   ```

### Database Does Not Exist

**Error:** `FATAL: database "mhras" does not exist`

**Solution:**
```bash
# Create the database
sudo -u postgres psql -c "CREATE DATABASE mhras OWNER mhras_user;"

# Or run the setup script
./setup_database.sh
```

### Permission Denied

**Error:** `ERROR: permission denied for schema public`

**Solution:**
```bash
sudo -u postgres psql -d mhras -c "
GRANT ALL PRIVILEGES ON DATABASE mhras TO mhras_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mhras_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mhras_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mhras_user;
"
```

### Migration Already Applied

**Error:** Migration appears to run but tables don't exist

**Solution:**
```bash
# Check migration status
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c "
SELECT * FROM schema_migrations;
"

# If needed, reset migrations (WARNING: This drops all data!)
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c "
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO mhras_user;
"

# Then re-run migrations
./run_migrations.sh
```

### Docker Connection Issues

**Error:** Cannot connect to PostgreSQL in Docker

**Solution:**
```bash
# Check if container is running
docker-compose ps

# Check container logs
docker-compose logs postgres

# Restart containers
docker-compose restart postgres

# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready -U mhras_user -d mhras
```

### Port Already in Use

**Error:** `bind: address already in use`

**Solution:**
```bash
# Find process using port 5432
sudo lsof -i :5432

# Stop the process or use a different port
# Update DB_PORT in .env and docker-compose.yml
```

---

## Configuration Files Reference

All database credentials are configured in these files:

### Development
- **`.env`** - Local environment variables (created from `.env.example`)
- **`config/.env.example`** - Template with default credentials
- **`src/config.py`** - Configuration management (reads from `.env`)

### Docker
- **`docker-compose.yml`** - Docker Compose configuration
- **`Dockerfile`** - Container build configuration

### Kubernetes
- **`k8s/secret.yaml`** - Kubernetes secrets (base64 encoded)
- **`k8s/configmap.yaml`** - Non-sensitive configuration
- **`k8s/deployment.yaml`** - Deployment with environment variables

### Scripts
- **`setup_database.sh`** - Automated database setup
- **`run_migrations.sh`** - Migration runner script
- **`src/database/migration_runner.py`** - Python migration runner

---

## Security Best Practices

### Development
- ✅ Use the default credentials for local development
- ✅ Never commit `.env` file to version control
- ✅ Keep `.env.example` updated with non-sensitive defaults

### Production
- ⚠️ **ALWAYS** change default passwords
- ⚠️ Use strong, randomly generated passwords (32+ characters)
- ⚠️ Store credentials in secure secret management systems:
  - Kubernetes Secrets
  - HashiCorp Vault
  - AWS Secrets Manager
  - Azure Key Vault
  - Google Secret Manager
- ⚠️ Enable SSL/TLS for database connections
- ⚠️ Restrict database access by IP/network
- ⚠️ Use separate credentials for different environments
- ⚠️ Rotate credentials regularly
- ⚠️ Enable database audit logging
- ⚠️ Use read-only users for reporting/analytics

### Generate Secure Passwords

```bash
# Generate a secure password
openssl rand -base64 32

# Or use Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Database Migration README](../src/database/migrations/README.md)
- [Configuration Guide](../config/README.md)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)

---

## Support

If you encounter issues not covered in this guide:

1. Check the [main README](../README.md) for general setup
2. Review [SETUP.md](../SETUP.md) for installation steps
3. Run the verification script: `python3 verify_integration.py`
4. Check application logs for detailed error messages
5. Consult the troubleshooting section above
