# Configuration

This directory contains configuration files for the Mental Health Risk Assessment System (MHRAS).

## Environment Configuration

The application uses environment variables for configuration management. Configuration is handled through Pydantic Settings, which supports:

- Environment variables
- `.env` files
- Default values

### Setup

#### Quick Setup (Recommended)

Use the automated database setup script:
```bash
./setup_database.sh
```

This will create the database, run migrations, and configure your `.env` file automatically.

#### Manual Setup

1. Copy the example environment file:
   ```bash
   cp config/.env.example .env
   ```

2. The `.env` file comes with consistent default credentials for development:
   - Database: `mhras`
   - User: `mhras_user`
   - Password: `mhras_dev_password_2024`

3. For production, update the `.env` file with secure values:
   - Generate strong passwords: `openssl rand -base64 32`
   - Update `DB_PASSWORD`
   - Update `SECURITY_JWT_SECRET`
   - Update `SECURITY_ANONYMIZATION_SALT`

4. **Important**: Never commit `.env` files with sensitive data to version control

**Database Setup:**
- Quick: [DATABASE_QUICKSTART.md](../DATABASE_QUICKSTART.md)
- Detailed: [docs/database_setup.md](../docs/database_setup.md)
- Security: [CREDENTIALS.md](../CREDENTIALS.md)

## Configuration Sections

### Application
- `ENVIRONMENT`: Environment name (development, staging, production)
- `DEBUG`: Enable debug mode (true/false)

### Database
- `DB_HOST`: Database host (default: `localhost`)
- `DB_PORT`: Database port (default: `5432`)
- `DB_NAME`: Database name (default: `mhras`)
- `DB_USER`: Database user (default: `mhras_user`)
- `DB_PASSWORD`: Database password (default: `mhras_dev_password_2024` - **CHANGE IN PRODUCTION!**)
- `DB_POOL_SIZE`: Connection pool size (default: `10`)

**Default Connection String:**
```
postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras
```

> ⚠️ **Security Note**: The default password is for development only. Always use strong, unique passwords in production environments.

### API
- `API_HOST`: API server host
- `API_PORT`: API server port
- `API_WORKERS`: Number of worker processes
- `API_TIMEOUT`: Request timeout in seconds
- `API_MAX_REQUEST_SIZE`: Maximum request size in bytes

### Machine Learning
- `ML_MODEL_STORAGE_PATH`: Path to model storage
- `ML_INFERENCE_TIMEOUT`: Inference timeout in seconds
- `ML_ENSEMBLE_WEIGHTS`: JSON string of ensemble weights
- `ML_RISK_THRESHOLD_HIGH`: High risk threshold (51-75)
- `ML_RISK_THRESHOLD_CRITICAL`: Critical risk threshold (>75)

### Logging
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FORMAT`: Log format (json or text)
- `LOG_OUTPUT`: Log output (stdout or file path)

### Security
- `SECURITY_JWT_SECRET`: Secret key for JWT tokens (change in production!)
- `SECURITY_JWT_ALGORITHM`: JWT algorithm (HS256)
- `SECURITY_JWT_EXPIRY_HOURS`: JWT expiry time in hours
- `SECURITY_ANONYMIZATION_SALT`: Salt for data anonymization (change in production!)

### Governance
- `GOVERNANCE_AUDIT_LOG_RETENTION_DAYS`: Audit log retention period
- `GOVERNANCE_HUMAN_REVIEW_THRESHOLD`: Risk score threshold for human review
- `GOVERNANCE_REVIEW_ESCALATION_HOURS`: Hours before review escalation
- `GOVERNANCE_DRIFT_THRESHOLD`: Drift detection threshold

## Usage in Code

```python
from src.config import settings

# Access configuration
db_host = settings.database.host
api_port = settings.api.port
log_level = settings.logging.level
```

## Environment-Specific Configurations

For different environments, you can:

1. Use different `.env` files (e.g., `.env.development`, `.env.production`)
2. Override with environment variables in deployment
3. Use configuration management tools (Kubernetes ConfigMaps, AWS Parameter Store, etc.)
