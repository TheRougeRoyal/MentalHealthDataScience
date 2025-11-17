# Database Quick Start

> **TL;DR**: Run `./setup_database.sh` and you're done! ✨

## Quick Setup

```bash
# 1. Run automated setup
./setup_database.sh

# 2. Start the API
python run_api.py

# 3. Access API docs
open http://localhost:8000/docs
```

## Default Credentials (Development)

**Connection String:**
```
postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras
```

**Individual Parameters:**
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mhras
DB_USER=mhras_user
DB_PASSWORD=mhras_dev_password_2024
```

---

## Quick Commands

### Setup Database
```bash
./setup_database.sh
```

### Test Connection
```bash
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c '\dt'
```

### Run Migrations
```bash
./run_migrations.sh
```

### Start with Docker
```bash
docker-compose up -d
```

### Check Status
```bash
python run_cli.py db check
```

---

## Troubleshooting

### PostgreSQL Not Running
```bash
sudo systemctl start postgresql
```

### Database Doesn't Exist
```bash
./setup_database.sh
```

### Connection Refused
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check credentials
cat .env | grep DB_
```

### Reset Everything
```bash
# Drop and recreate (WARNING: Deletes all data!)
sudo -u postgres psql -c "DROP DATABASE IF EXISTS mhras;"
./setup_database.sh
```

---

## Production Deployment

⚠️ **NEVER use default credentials in production!**

```bash
# Generate secure password
openssl rand -base64 32

# Update Kubernetes secret
kubectl create secret generic mhras-secrets \
  --from-literal=db_password="YOUR_SECURE_PASSWORD" \
  --namespace=mhras
```

**See:** [CREDENTIALS.md](CREDENTIALS.md) for complete production setup guide.

---

## Documentation

- **[CREDENTIALS.md](CREDENTIALS.md)** - Complete credential reference & security guide
- **[docs/database_setup.md](docs/database_setup.md)** - Detailed setup & troubleshooting
- **[README.md](README.md)** - Main project documentation
