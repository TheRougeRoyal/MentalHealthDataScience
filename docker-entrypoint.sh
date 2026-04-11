#!/bin/bash
set -e

echo "=========================================="
echo "MHRAS Docker Entry Point"
echo "=========================================="

echo "Starting deployment checks..."

# Simple delay to give database time to start up
if [[ "$DATABASE_URL" == *"postgres"* ]]; then
    echo "Wait up to 10s for database..."
    sleep 5
fi

echo "=========================================="
echo "Startup complete! Starting MHRAS API..."
echo "=========================================="

# Execute the main command
exec "$@"
