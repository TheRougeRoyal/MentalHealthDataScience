# MHRAS Credentials Reference

This document provides a comprehensive reference for all default credentials used across the MHRAS system. These credentials are consistent across all configuration files and deployment methods.

## ⚠️ Security Warning

**The credentials listed here are for DEVELOPMENT ONLY.**

**NEVER use these default credentials in production environments!**

For production deployments:
1. Generate strong, unique passwords
2. Use secure secret management systems
3. Rotate credentials regularly
4. Follow the security guidelines in this document

---

## Default Development Credentials

### Database (PostgreSQL)

| Parameter | Value | Environment Variable |
|-----------|-------|---------------------|
| Host | `localhost` | `DB_HOST` |
| Port | `5432` | `DB_PORT` |
| Database Name | `mhras` | `DB_NAME` |
| Username | `mhras_user` | `DB_USER` |
| Password | `mhras_dev_password_2024` | `DB_PASSWORD` |
| Pool Size | `10` | `DB_POOL_SIZE` |

**Connection String:**
```
postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras
```

### Redis (Feature Store Cache)

| Parameter | Value | Environment Variable |
|-----------|-------|---------------------|
| Host | `localhost` | `FEATURE_STORE_REDIS_URL` |
| Port | `6379` | (included in URL) |
| URL | `redis://localhost:6379` | `FEATURE_STORE_REDIS_URL` |

### API Security

| Parameter | Default Value | Environment Variable |
|-----------|--------------|---------------------|
| JWT Secret | `change-me-in-production-use-strong-random-string` | `SECURITY_JWT_SECRET` |
| JWT Algorithm | `HS256` | `SECURITY_JWT_ALGORITHM` |
| JWT Expiry | `24` hours | `SECURITY_JWT_EXPIRY_HOURS` |
| Anonymization Salt | `change-me-in-production-use-strong-random-string` | `SECURITY_ANONYMIZATION_SALT` |

---

## Configuration Files

All credentials are configured consistently across these files:

### 1. Environment Configuration

**File:** `.env` (created from `config/.env.example`)

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mhras
DB_USER=mhras_user
DB_PASSWORD=mhras_dev_password_2024
DB_POOL_SIZE=10

# Security
SECURITY_JWT_SECRET=change-me-in-production-use-strong-random-string
SECURITY_JWT_ALGORITHM=HS256
SECURITY_JWT_EXPIRY_HOURS=24
SECURITY_ANONYMIZATION_SALT=change-me-in-production-use-strong-random-string

# Feature Store
FEATURE_STORE_REDIS_URL=redis://localhost:6379
FEATURE_STORE_CACHE_BACKEND=redis
FEATURE_STORE_CACHE_TTL=3600
```

### 2. Docker Compose

**File:** `docker-compose.yml`

PostgreSQL service:
```yaml
environment:
  POSTGRES_DB: mhras
  POSTGRES_USER: mhras_user
  POSTGRES_PASSWORD: ${DB_PASSWORD:-mhras_dev_password_2024}
```

API service:
```yaml
environment:
  DB_HOST: postgres
  DB_PORT: "5432"
  DB_NAME: mhras
  DB_USER: mhras_user
  DB_PASSWORD: ${DB_PASSWORD:-mhras_dev_password_2024}
```

### 3. Kubernetes Secrets

**File:** `k8s/secret.yaml`

```yaml
stringData:
  db_host: "postgres-service"
  db_port: "5432"
  db_name: "mhras"
  db_user: "mhras_user"
  db_password: "CHANGE_ME_IN_PRODUCTION"
  jwt_secret_key: "CHANGE_ME_GENERATE_RANDOM_SECRET"
  anonymization_salt: "CHANGE_ME_GENERATE_RANDOM_SALT"
```

### 4. Migration Scripts

**File:** `run_migrations.sh`

```bash
DATABASE_URL=${DATABASE_URL:-"postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras"}
```

---

## Deployment-Specific Credentials

### Local Development

Use the default credentials as-is. They are pre-configured in all files.

**Setup:**
```bash
# Automated setup
./setup_database.sh

# Manual setup
cp config/.env.example .env
# No changes needed for development
```

### Docker Development

Use environment variables to override defaults if needed:

```bash
# Start with default credentials
docker-compose up -d

# Or override password
DB_PASSWORD=my_custom_password docker-compose up -d
```

### Kubernetes/Production

**IMPORTANT:** Always use secure, unique credentials in production.

#### Step 1: Generate Secure Credentials

```bash
# Generate database password
DB_PASSWORD=$(openssl rand -base64 32)

# Generate JWT secret
JWT_SECRET=$(openssl rand -base64 32)

# Generate anonymization salt
ANON_SALT=$(openssl rand -base64 32)
```

#### Step 2: Create Kubernetes Secret

```bash
kubectl create secret generic mhras-secrets \
  --from-literal=db_host=your-postgres-host \
  --from-literal=db_port=5432 \
  --from-literal=db_name=mhras \
  --from-literal=db_user=mhras_user \
  --from-literal=db_password="$DB_PASSWORD" \
  --from-literal=jwt_secret_key="$JWT_SECRET" \
  --from-literal=jwt_algorithm=HS256 \
  --from-literal=jwt_expiry_hours=24 \
  --from-literal=anonymization_salt="$ANON_SALT" \
  --namespace=mhras
```

#### Step 3: Use External Secret Management (Recommended)

For production, use a dedicated secret management system:

**AWS Secrets Manager:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mhras-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: mhras-secrets
  data:
  - secretKey: db_password
    remoteRef:
      key: mhras/database/password
  - secretKey: jwt_secret_key
    remoteRef:
      key: mhras/api/jwt-secret
```

**HashiCorp Vault:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mhras-secrets
spec:
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: mhras-secrets
  data:
  - secretKey: db_password
    remoteRef:
      key: secret/mhras/database
      property: password
```

---

## Credential Rotation

### Database Password Rotation

1. **Create new password:**
   ```bash
   NEW_PASSWORD=$(openssl rand -base64 32)
   ```

2. **Update PostgreSQL:**
   ```sql
   ALTER USER mhras_user WITH PASSWORD 'new_password_here';
   ```

3. **Update configurations:**
   - Update `.env` file
   - Update Kubernetes secrets
   - Update Docker Compose environment
   - Restart services

4. **Verify:**
   ```bash
   psql "postgresql://mhras_user:new_password@localhost:5432/mhras" -c '\dt'
   ```

### JWT Secret Rotation

1. **Generate new secret:**
   ```bash
   NEW_JWT_SECRET=$(openssl rand -base64 32)
   ```

2. **Update configurations:**
   - Update `.env` file: `SECURITY_JWT_SECRET=new_secret`
   - Update Kubernetes secrets
   - Restart API services

3. **Note:** All existing JWT tokens will be invalidated

### Anonymization Salt Rotation

⚠️ **WARNING:** Rotating the anonymization salt will make previously anonymized IDs incompatible.

Only rotate if:
- Salt has been compromised
- Implementing a planned migration strategy

---

## Security Best Practices

### Password Requirements

For production, use passwords that:
- Are at least 32 characters long
- Contain uppercase, lowercase, numbers, and special characters
- Are randomly generated (not dictionary words)
- Are unique per environment
- Are never committed to version control

### Secret Management

✅ **DO:**
- Use environment variables for secrets
- Use dedicated secret management systems (Vault, AWS Secrets Manager, etc.)
- Rotate credentials regularly (every 90 days minimum)
- Use different credentials for each environment
- Enable audit logging for secret access
- Encrypt secrets at rest and in transit
- Use least-privilege access principles

❌ **DON'T:**
- Commit secrets to version control
- Share secrets via email or chat
- Use the same password across environments
- Store secrets in plain text files
- Use weak or predictable passwords
- Share database credentials with application users

### Access Control

1. **Database Access:**
   - Use separate users for different services
   - Grant minimum required permissions
   - Use read-only users for reporting
   - Enable SSL/TLS connections
   - Restrict access by IP/network

2. **API Access:**
   - Implement JWT token expiration
   - Use refresh tokens for long-lived sessions
   - Implement rate limiting
   - Log all authentication attempts
   - Use HTTPS in production

3. **Secret Access:**
   - Limit who can view/modify secrets
   - Use role-based access control (RBAC)
   - Enable audit logging
   - Implement approval workflows for changes

---

## Verification

### Check Current Credentials

```bash
# View environment variables (sanitized)
env | grep -E "DB_|SECURITY_" | sed 's/=.*/=***/'

# Test database connection
psql "postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras" -c '\conninfo'

# Verify Kubernetes secrets (base64 decoded)
kubectl get secret mhras-secrets -o jsonpath='{.data.db_user}' | base64 -d
```

### Test Connections

```bash
# Database
python3 -c "
from src.database.connection import get_db_connection
db = get_db_connection()
print('✓ Database connection successful')
"

# Redis
redis-cli -h localhost -p 6379 ping

# API
curl http://localhost:8000/health
```

---

## Troubleshooting

### Authentication Failed

**Problem:** Cannot connect to database

**Solutions:**
1. Verify credentials in `.env` match database
2. Check PostgreSQL pg_hba.conf allows password authentication
3. Ensure user exists: `sudo -u postgres psql -c "\du"`
4. Reset password: `ALTER USER mhras_user WITH PASSWORD 'new_password';`

### JWT Token Invalid

**Problem:** API returns 401 Unauthorized

**Solutions:**
1. Verify `SECURITY_JWT_SECRET` is set correctly
2. Check token hasn't expired
3. Ensure secret hasn't changed since token was issued
4. Verify JWT algorithm matches (`HS256`)

### Kubernetes Secret Not Found

**Problem:** Pods fail to start with secret errors

**Solutions:**
1. Verify secret exists: `kubectl get secrets -n mhras`
2. Check secret has required keys: `kubectl describe secret mhras-secrets -n mhras`
3. Recreate secret if needed
4. Verify deployment references correct secret name

---

## Quick Reference Commands

### Generate Secure Passwords

```bash
# 32-character password
openssl rand -base64 32

# URL-safe password
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Hex password
openssl rand -hex 32
```

### Update Environment File

```bash
# Update database password
sed -i 's/DB_PASSWORD=.*/DB_PASSWORD=new_password/' .env

# Update JWT secret
sed -i 's/SECURITY_JWT_SECRET=.*/SECURITY_JWT_SECRET=new_secret/' .env
```

### Update Kubernetes Secret

```bash
# Update single value
kubectl patch secret mhras-secrets -n mhras \
  -p '{"stringData":{"db_password":"new_password"}}'

# Replace entire secret
kubectl delete secret mhras-secrets -n mhras
kubectl create secret generic mhras-secrets \
  --from-literal=db_password=new_password \
  --namespace=mhras
```

---

## Additional Resources

- **[Database Setup Guide](docs/database_setup.md)** - Detailed setup & troubleshooting
- **[Configuration Guide](config/README.md)** - Environment configuration
- **[Main README](README.md)** - Project documentation
- [PostgreSQL Security](https://www.postgresql.org/docs/current/auth-password.html)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)

---

## Quick Reference

**Development Credentials:**
```
postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras
```

**Quick Setup:**
```bash
./setup_database.sh
```

**Production Checklist:**
- ✅ Generate strong passwords: `openssl rand -base64 32`
- ✅ Use secret management systems
- ✅ Enable SSL/TLS
- ✅ Rotate credentials regularly
- ✅ Implement access controls
- ✅ Enable audit logging

**Need Help?** See [docs/database_setup.md](docs/database_setup.md) for detailed instructions.
