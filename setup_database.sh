#!/bin/bash
# MHRAS Database Setup Script
# This script sets up the PostgreSQL database with consistent credentials

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default database credentials (consistent across all configs)
DEFAULT_DB_HOST="${DB_HOST:-localhost}"
DEFAULT_DB_PORT="${DB_PORT:-5432}"
DEFAULT_DB_NAME="${DB_NAME:-mhras}"
DEFAULT_DB_USER="${DB_USER:-mhras_user}"
DEFAULT_DB_PASSWORD="${DB_PASSWORD:-mhras_dev_password_2024}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MHRAS Database Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if PostgreSQL is installed
check_postgres() {
    if ! command -v psql &> /dev/null; then
        print_error "PostgreSQL client (psql) is not installed"
        echo ""
        echo "Please install PostgreSQL:"
        echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
        echo "  macOS: brew install postgresql"
        echo "  RHEL/CentOS: sudo yum install postgresql postgresql-server"
        exit 1
    fi
    print_success "PostgreSQL client found"
}

# Function to check if PostgreSQL server is running
check_postgres_running() {
    if ! pg_isready -h "$DEFAULT_DB_HOST" -p "$DEFAULT_DB_PORT" &> /dev/null; then
        print_warning "PostgreSQL server is not running on $DEFAULT_DB_HOST:$DEFAULT_DB_PORT"
        echo ""
        echo "Please start PostgreSQL:"
        echo "  Ubuntu/Debian: sudo systemctl start postgresql"
        echo "  macOS: brew services start postgresql"
        echo "  Docker: docker-compose up -d postgres"
        echo ""
        read -p "Press Enter after starting PostgreSQL, or Ctrl+C to exit..."
    fi
    print_success "PostgreSQL server is running"
}

# Function to create database and user
setup_database() {
    print_info "Setting up database and user..."
    
    # Check if we can connect as postgres superuser
    if psql -h "$DEFAULT_DB_HOST" -p "$DEFAULT_DB_PORT" -U postgres -c '\q' 2>/dev/null; then
        POSTGRES_USER="postgres"
    else
        print_warning "Cannot connect as 'postgres' user"
        echo "Please enter PostgreSQL superuser credentials:"
        read -p "Superuser name [postgres]: " POSTGRES_USER
        POSTGRES_USER=${POSTGRES_USER:-postgres}
    fi
    
    # Create user if not exists
    print_info "Creating database user: $DEFAULT_DB_USER"
    psql -h "$DEFAULT_DB_HOST" -p "$DEFAULT_DB_PORT" -U "$POSTGRES_USER" -tc \
        "SELECT 1 FROM pg_roles WHERE rolname='$DEFAULT_DB_USER'" | grep -q 1 || \
        psql -h "$DEFAULT_DB_HOST" -p "$DEFAULT_DB_PORT" -U "$POSTGRES_USER" <<EOF
CREATE USER $DEFAULT_DB_USER WITH PASSWORD '$DEFAULT_DB_PASSWORD';
ALTER USER $DEFAULT_DB_USER CREATEDB;
EOF
    print_success "Database user created/verified"
    
    # Create database if not exists
    print_info "Creating database: $DEFAULT_DB_NAME"
    psql -h "$DEFAULT_DB_HOST" -p "$DEFAULT_DB_PORT" -U "$POSTGRES_USER" -tc \
        "SELECT 1 FROM pg_database WHERE datname='$DEFAULT_DB_NAME'" | grep -q 1 || \
        psql -h "$DEFAULT_DB_HOST" -p "$DEFAULT_DB_PORT" -U "$POSTGRES_USER" <<EOF
CREATE DATABASE $DEFAULT_DB_NAME OWNER $DEFAULT_DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DEFAULT_DB_NAME TO $DEFAULT_DB_USER;
EOF
    print_success "Database created/verified"
}

# Function to run migrations
run_migrations() {
    print_info "Running database migrations..."
    
    DATABASE_URL="postgresql://$DEFAULT_DB_USER:$DEFAULT_DB_PASSWORD@$DEFAULT_DB_HOST:$DEFAULT_DB_PORT/$DEFAULT_DB_NAME"
    
    if [ -f "run_migrations.sh" ]; then
        DATABASE_URL="$DATABASE_URL" bash run_migrations.sh
    else
        python3 -m src.database.migration_runner --database-url "$DATABASE_URL"
    fi
    
    print_success "Migrations completed"
}

# Function to verify database setup
verify_setup() {
    print_info "Verifying database setup..."
    
    DATABASE_URL="postgresql://$DEFAULT_DB_USER:$DEFAULT_DB_PASSWORD@$DEFAULT_DB_HOST:$DEFAULT_DB_PORT/$DEFAULT_DB_NAME"
    
    # Check if we can connect
    if psql "$DATABASE_URL" -c '\q' 2>/dev/null; then
        print_success "Database connection successful"
    else
        print_error "Failed to connect to database"
        exit 1
    fi
    
    # Check if tables exist
    TABLE_COUNT=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
    print_info "Found $TABLE_COUNT tables in database"
    
    # List tables
    print_info "Database tables:"
    psql "$DATABASE_URL" -c "\dt"
}

# Function to create .env file
create_env_file() {
    if [ -f ".env" ]; then
        print_warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping .env file creation"
            return
        fi
    fi
    
    print_info "Creating .env file with database credentials..."
    cp config/.env.example .env
    
    # Update database credentials in .env
    sed -i.bak "s/DB_HOST=.*/DB_HOST=$DEFAULT_DB_HOST/" .env
    sed -i.bak "s/DB_PORT=.*/DB_PORT=$DEFAULT_DB_PORT/" .env
    sed -i.bak "s/DB_NAME=.*/DB_NAME=$DEFAULT_DB_NAME/" .env
    sed -i.bak "s/DB_USER=.*/DB_USER=$DEFAULT_DB_USER/" .env
    sed -i.bak "s/DB_PASSWORD=.*/DB_PASSWORD=$DEFAULT_DB_PASSWORD/" .env
    rm -f .env.bak
    
    print_success ".env file created"
}

# Function to display connection info
display_connection_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Database Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Database Credentials:"
    echo "  Host:     $DEFAULT_DB_HOST"
    echo "  Port:     $DEFAULT_DB_PORT"
    echo "  Database: $DEFAULT_DB_NAME"
    echo "  User:     $DEFAULT_DB_USER"
    echo "  Password: $DEFAULT_DB_PASSWORD"
    echo ""
    echo "Connection String:"
    echo "  postgresql://$DEFAULT_DB_USER:$DEFAULT_DB_PASSWORD@$DEFAULT_DB_HOST:$DEFAULT_DB_PORT/$DEFAULT_DB_NAME"
    echo ""
    echo "Next Steps:"
    echo "  1. Review and update .env file if needed"
    echo "  2. Start the API server: python run_api.py"
    echo "  3. Access API docs: http://localhost:8000/docs"
    echo ""
    print_warning "IMPORTANT: Change the default password in production!"
    echo ""
}

# Main execution
main() {
    echo "This script will set up the MHRAS database with the following credentials:"
    echo "  Host:     $DEFAULT_DB_HOST"
    echo "  Port:     $DEFAULT_DB_PORT"
    echo "  Database: $DEFAULT_DB_NAME"
    echo "  User:     $DEFAULT_DB_USER"
    echo "  Password: $DEFAULT_DB_PASSWORD"
    echo ""
    
    read -p "Continue with these settings? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi
    
    check_postgres
    check_postgres_running
    setup_database
    run_migrations
    verify_setup
    create_env_file
    display_connection_info
}

# Run main function
main
