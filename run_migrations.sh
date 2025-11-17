#!/bin/bash
# Script to run database migrations for MHRAS

set -e

# Default database URL (can be overridden by environment variable)
DATABASE_URL=${DATABASE_URL:-"postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras"}

echo "Running database migrations..."
echo "Database URL: ${DATABASE_URL}"
echo ""

# Run migrations using Python
python3 -m src.database.migration_runner --database-url "${DATABASE_URL}"

echo ""
echo "✓ Migrations completed successfully"
