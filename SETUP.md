# MHRAS Setup Guide

This guide will help you set up the Mental Health Risk Assessment System (MHRAS) development environment.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- PostgreSQL 12 or higher (for production use)

## Installation Steps

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Database

**Option A: Automated Setup (Recommended)**

```bash
# Run the automated database setup script
./setup_database.sh
```

This script will:
- Check PostgreSQL installation
- Create database and user with consistent credentials
- Run all migrations
- Create/update your `.env` file
- Verify the setup

**Option B: Manual Setup**

```bash
# Copy the example configuration
cp config/.env.example .env

# The .env file already has default development credentials:
# - Database: mhras
# - User: mhras_user
# - Password: mhras_dev_password_2024

# Create PostgreSQL database and user
sudo -u postgres psql <<EOF
CREATE USER mhras_user WITH PASSWORD 'mhras_dev_password_2024';
CREATE DATABASE mhras OWNER mhras_user;
GRANT ALL PRIVILEGES ON DATABASE mhras TO mhras_user;
ALTER USER mhras_user CREATEDB;
EOF

# Run migrations
./run_migrations.sh
```

**For Production**: Update the following values in `.env`:
- `DB_PASSWORD`: Use a strong random password (generate with `openssl rand -base64 32`)
- `SECURITY_JWT_SECRET`: Use a strong random string
- `SECURITY_ANONYMIZATION_SALT`: Use a strong random string

**Database Documentation:**
- [DATABASE_QUICKSTART.md](DATABASE_QUICKSTART.md) - Quick reference
- [docs/database_setup.md](docs/database_setup.md) - Detailed guide
- [CREDENTIALS.md](CREDENTIALS.md) - Security & credentials

### 4. Verify Installation

```bash
# Test imports
python3 -c "from src.config import settings; print('Configuration loaded successfully')"

# Run tests
pytest tests/ -v

# Test logging
python3 src/main.py
```

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── .env.example       # Example environment configuration
│   └── README.md          # Configuration documentation
├── src/                   # Source code
│   ├── api/              # API layer components
│   ├── governance/       # Governance and compliance
│   ├── ingestion/        # Data ingestion
│   ├── ml/               # Machine learning components
│   ├── processing/       # ETL and data processing
│   ├── config.py         # Configuration management
│   ├── exceptions.py     # Exception classes
│   ├── logging_config.py # Logging setup
│   └── main.py           # Application entry point
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Format code (if using black)
black src/ tests/

# Lint code (if using flake8)
flake8 src/ tests/

# Type checking (if using mypy)
mypy src/
```

### Running the Application

```bash
# Initialize the application
python3 src/main.py

# Run API server (once implemented)
uvicorn src.api.main:app --reload
```

## Next Steps

After completing the setup:

1. Review the requirements document: `.kiro/specs/mental-health-risk-assessment/requirements.md`
2. Review the design document: `.kiro/specs/mental-health-risk-assessment/design.md`
3. Check the task list: `.kiro/specs/mental-health-risk-assessment/tasks.md`
4. Begin implementing the next task in the implementation plan

## Troubleshooting

### Import Errors

If you encounter import errors:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Configuration Errors

If configuration fails to load:
```bash
# Verify .env file exists
ls -la .env

# Check for syntax errors in .env
cat .env
```

### Database Connection Issues

If database connection fails:
```bash
# Verify PostgreSQL is running
sudo systemctl status postgresql

# Check database credentials in .env
cat .env | grep DB_

# Test connection with default credentials
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c '\dt'

# Re-run setup script if needed
./setup_database.sh
```

**See:** [docs/database_setup.md](docs/database_setup.md#troubleshooting) for more help

## Support

For issues or questions:
1. Check the design document for architecture details
2. Review the requirements document for specifications
3. Consult the task list for implementation guidance
