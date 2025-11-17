# MHRAS Deployment Guide

This comprehensive guide covers the complete deployment process for the Mental Health Risk Assessment System (MHRAS), from infrastructure requirements to production deployment and ongoing operations.

## Table of Contents

1. [Overview](#overview)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Deployment Procedures](#deployment-procedures)
5. [Configuration Management](#configuration-management)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Hardening](#security-hardening)
8. [Scaling and Performance](#scaling-and-performance)
9. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
10. [Troubleshooting](#troubleshooting)
11. [Production Checklist](#production-checklist)

## Overview

The MHRAS deployment architecture includes:

- **Docker containerization** with multi-stage builds for optimized images
- **Kubernetes orchestration** for high availability and auto-scaling
- **PostgreSQL database** for persistent data storage
- **Monitoring stack** with Prometheus and Grafana
- **Alerting system** for operational awareness
- **Network policies** for security isolation
- **TLS/HTTPS** for encrypted communications

**Deployment Targets:**
- Development: Local Docker or Minikube
- Staging: Kubernetes cluster (3+ nodes)
- Production: Kubernetes cluster (5+ nodes) with HA database

## Infrastructure Requirements

### Minimum Requirements

**Development Environment:**
- 1 node with 4 CPU cores, 8GB RAM
- Docker 20.10+
- Kubernetes 1.24+ (Minikube or kind)
- 50GB storage

**Staging Environment:**
- 3 Kubernetes nodes: 4 CPU cores, 16GB RAM each
- PostgreSQL 14+ (managed or self-hosted)
- 200GB storage (SSD recommended)
- Load balancer (cloud provider or MetalLB)

**Production Environment:**
- 5+ Kubernetes nodes: 8 CPU cores, 32GB RAM each
- PostgreSQL 14+ with HA (primary + replica)
- 500GB+ storage (SSD required)
- Cloud load balancer with TLS termination
- Backup storage (S3-compatible)
- Log aggregation system (ELK, Loki, or cloud service)

### Software Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| Kubernetes | 1.24+ | Container orchestration |
| Docker | 20.10+ | Container runtime |
| PostgreSQL | 14+ | Database |
| Python | 3.11+ | Application runtime |
| Prometheus | 2.40+ | Metrics collection |
| Grafana | 9.0+ | Visualization |
| kubectl | 1.24+ | Kubernetes CLI |
| Helm | 3.10+ (optional) | Package management |

### Network Requirements

**Ports:**
- 80/443: HTTP/HTTPS ingress
- 8000: API service (internal)
- 5432: PostgreSQL (internal)
- 9090: Prometheus (internal)
- 3000: Grafana (internal)

**Bandwidth:**
- Minimum: 100 Mbps
- Recommended: 1 Gbps
- Expected traffic: ~1000 requests/hour (adjust based on usage)

**DNS:**
- API endpoint: `api.mhras.example.com`
- Monitoring: `monitoring.mhras.example.com`
- Grafana: `grafana.mhras.example.com`

### Storage Requirements

**Application:**
- Model artifacts: 5-10GB
- Logs: 10GB/day (with rotation)
- Temporary data: 20GB

**Database:**
- Initial: 10GB
- Growth: ~1GB/month (varies by usage)
- Backups: 3x database size

**Monitoring:**
- Prometheus data: 50GB (30-day retention)
- Grafana: 5GB

## Pre-Deployment Checklist

Before deploying MHRAS, ensure the following prerequisites are met:

### Infrastructure Setup

- [ ] Kubernetes cluster provisioned and accessible
- [ ] kubectl configured with cluster credentials
- [ ] Namespaces created (`mhras`, `monitoring`)
- [ ] Storage classes configured
- [ ] Load balancer provisioned
- [ ] DNS records configured
- [ ] TLS certificates obtained

### Database Setup

- [ ] PostgreSQL instance provisioned
- [ ] Database `mhras` created
- [ ] Database user created with appropriate permissions
- [ ] Connection tested from Kubernetes cluster
- [ ] Backup strategy configured
- [ ] Monitoring enabled

### Security Setup

- [ ] Kubernetes RBAC configured
- [ ] Network policies reviewed
- [ ] Secrets management solution chosen (Kubernetes Secrets, Vault, etc.)
- [ ] JWT secret key generated (256-bit minimum)
- [ ] Anonymization salt generated (256-bit minimum)
- [ ] TLS certificates installed
- [ ] Security scanning tools configured

### Monitoring Setup

- [ ] Prometheus installed
- [ ] Grafana installed
- [ ] Alert manager configured
- [ ] Notification channels set up (email, Slack, PagerDuty)
- [ ] Log aggregation configured
- [ ] Dashboards imported

### Application Setup

- [ ] Docker registry accessible
- [ ] Model artifacts uploaded to storage
- [ ] Configuration files prepared
- [ ] Environment variables documented
- [ ] Deployment manifests reviewed
- [ ] Rollback plan documented

## Deployment Procedures

### Step 1: Prepare Environment

```bash
# Set environment variables
export MHRAS_VERSION=v1.0.0
export DOCKER_REGISTRY=your-registry.example.com
export KUBE_NAMESPACE=mhras
export MONITORING_NAMESPACE=monitoring

# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Create namespaces
kubectl create namespace ${KUBE_NAMESPACE}
kubectl create namespace ${MONITORING_NAMESPACE}

# Label namespaces
kubectl label namespace ${KUBE_NAMESPACE} app=mhras
kubectl label namespace ${MONITORING_NAMESPACE} app=monitoring
```

### Step 2: Build and Push Docker Image

```bash
# Navigate to project root
cd /path/to/mhras

# Build Docker image
docker build -t mhras-api:${MHRAS_VERSION} .

# Tag for registry
docker tag mhras-api:${MHRAS_VERSION} \
  ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}
docker tag mhras-api:${MHRAS_VERSION} \
  ${DOCKER_REGISTRY}/mhras-api:latest

# Push to registry
docker push ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}
docker push ${DOCKER_REGISTRY}/mhras-api:latest

# Verify image
docker pull ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}
docker inspect ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}
```

**Multi-Architecture Build (Optional):**
```bash
# Build for multiple architectures
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION} \
  --push .
```

### Step 3: Configure Secrets

```bash
# Generate secrets
JWT_SECRET=$(openssl rand -base64 32)
ANON_SALT=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -base64 24)

# Create Kubernetes secret
kubectl create secret generic mhras-secrets \
  --from-literal=database_url="postgresql://mhras:${DB_PASSWORD}@postgres:5432/mhras" \
  --from-literal=jwt_secret_key="${JWT_SECRET}" \
  --from-literal=anonymization_salt="${ANON_SALT}" \
  --namespace=${KUBE_NAMESPACE}

# Verify secret
kubectl get secret mhras-secrets -n ${KUBE_NAMESPACE}

# For production, use external secret management
# Example with Sealed Secrets:
# kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
# kubectl apply -f sealed-secret.yaml
```

### Step 4: Deploy Database

**Option A: Managed Database (Recommended for Production)**

```bash
# Use cloud provider's managed PostgreSQL
# Configure connection in secrets (Step 3)
# Example: AWS RDS, Google Cloud SQL, Azure Database
```

**Option B: Self-Hosted PostgreSQL**

```bash
# Deploy PostgreSQL using Helm
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --namespace ${KUBE_NAMESPACE} \
  --set auth.username=mhras \
  --set auth.password=${DB_PASSWORD} \
  --set auth.database=mhras \
  --set primary.persistence.size=100Gi \
  --set metrics.enabled=true

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=postgresql \
  -n ${KUBE_NAMESPACE} \
  --timeout=300s

# Run database migrations
kubectl run migration-job \
  --image=${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION} \
  --restart=Never \
  --namespace=${KUBE_NAMESPACE} \
  --env="DATABASE_URL=postgresql://mhras:${DB_PASSWORD}@postgres:5432/mhras" \
  --command -- python -m src.database.migration_runner

# Check migration logs
kubectl logs migration-job -n ${KUBE_NAMESPACE}
```

### Step 5: Deploy Application

```bash
# Update image version in deployment manifest
sed -i "s|image:.*|image: ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}|" \
  k8s/deployment.yaml

# Apply ConfigMap
kubectl apply -f k8s/configmap.yaml -n ${KUBE_NAMESPACE}

# Apply Deployment
kubectl apply -f k8s/deployment.yaml -n ${KUBE_NAMESPACE}

# Apply Service
kubectl apply -f k8s/service.yaml -n ${KUBE_NAMESPACE}

# Apply HPA (Horizontal Pod Autoscaler)
kubectl apply -f k8s/hpa.yaml -n ${KUBE_NAMESPACE}

# Apply Network Policy
kubectl apply -f k8s/network-policy.yaml -n ${KUBE_NAMESPACE}

# Wait for deployment to be ready
kubectl rollout status deployment/mhras-api -n ${KUBE_NAMESPACE}

# Verify pods are running
kubectl get pods -n ${KUBE_NAMESPACE} -l app=mhras-api
```

### Step 6: Deploy Ingress

```bash
# Install ingress controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Create TLS secret from certificates
kubectl create secret tls mhras-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=${KUBE_NAMESPACE}

# Apply Ingress
kubectl apply -f k8s/ingress.yaml -n ${KUBE_NAMESPACE}

# Get ingress IP/hostname
kubectl get ingress -n ${KUBE_NAMESPACE}
```

### Step 7: Deploy Monitoring Stack

```bash
# Deploy Prometheus
kubectl apply -f monitoring/k8s/prometheus-deployment.yaml \
  -n ${MONITORING_NAMESPACE}

# Deploy Grafana
kubectl apply -f monitoring/k8s/grafana-deployment.yaml \
  -n ${MONITORING_NAMESPACE}

# Wait for monitoring pods
kubectl wait --for=condition=ready pod \
  -l app=prometheus \
  -n ${MONITORING_NAMESPACE} \
  --timeout=300s

kubectl wait --for=condition=ready pod \
  -l app=grafana \
  -n ${MONITORING_NAMESPACE} \
  --timeout=300s

# Import Grafana dashboards
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n ${MONITORING_NAMESPACE}

# Configure Prometheus data source in Grafana
# Access Grafana UI and add Prometheus as data source
# URL: http://prometheus:9090
```

### Step 8: Verify Deployment

```bash
# Check all resources
kubectl get all -n ${KUBE_NAMESPACE}
kubectl get all -n ${MONITORING_NAMESPACE}

# Test health endpoint
kubectl port-forward svc/mhras-api 8000:80 -n ${KUBE_NAMESPACE} &
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": 1700220600.123}

# Test API endpoint (requires authentication)
# Generate token first
TOKEN=$(python -c "from src.api.auth import authenticator; print(authenticator.generate_token('test_user', 'clinician'))")

curl -H "Authorization: Bearer ${TOKEN}" \
  http://localhost:8000/

# Check logs
kubectl logs -f deployment/mhras-api -n ${KUBE_NAMESPACE}

# Check metrics
curl http://localhost:8000/metrics

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n ${MONITORING_NAMESPACE}
# Visit http://localhost:3000 (default: admin/admin)
```

### Step 9: Configure Alerts

```bash
# Apply Prometheus alert rules
kubectl apply -f monitoring/prometheus/alerts/mhras-alerts.yaml \
  -n ${MONITORING_NAMESPACE}

# Verify alerts are loaded
kubectl port-forward svc/prometheus 9090:9090 -n ${MONITORING_NAMESPACE} &
# Visit http://localhost:9090/alerts

# Configure alert notifications (edit prometheus.yaml)
# Add Alertmanager configuration for email, Slack, PagerDuty, etc.
```

### Step 10: Production Smoke Tests

```bash
# Run smoke tests
python tests/smoke_tests.py --env=production --api-url=https://api.mhras.example.com

# Test screening endpoint
curl -X POST https://api.mhras.example.com/screen \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_screening_request.json

# Monitor response times
kubectl top pods -n ${KUBE_NAMESPACE}

# Check for errors in logs
kubectl logs -f deployment/mhras-api -n ${KUBE_NAMESPACE} | grep ERROR
```

## Configuration Management

### Environment Variables

**Application Configuration (ConfigMap):**

| Variable | Default | Description |
|----------|---------|-------------|
| `MHRAS_ENV` | `production` | Environment (development, staging, production) |
| `MHRAS_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MHRAS_LOG_FORMAT` | `json` | Log format (json, text) |
| `RISK_THRESHOLD_LOW` | `25` | Low risk threshold (0-25) |
| `RISK_THRESHOLD_MODERATE` | `50` | Moderate risk threshold (26-50) |
| `RISK_THRESHOLD_HIGH` | `75` | High risk threshold (51-75) |
| `HUMAN_REVIEW_THRESHOLD` | `75` | Score triggering human review |
| `DRIFT_THRESHOLD` | `0.3` | Drift detection threshold |
| `MODEL_CACHE_SIZE` | `5` | Number of models to cache |
| `REQUEST_TIMEOUT` | `10` | Request timeout in seconds |
| `SCREENING_TIMEOUT` | `5` | Screening endpoint timeout |

**Secrets (Kubernetes Secret):**

| Variable | Description | Generation |
|----------|-------------|------------|
| `DATABASE_URL` | PostgreSQL connection string | Provided by DBA |
| `JWT_SECRET_KEY` | JWT signing key | `openssl rand -base64 32` |
| `ANONYMIZATION_SALT` | PII hashing salt | `openssl rand -base64 32` |
| `MODEL_STORAGE_KEY` | S3/storage access key | Cloud provider |
| `ALERT_WEBHOOK_URL` | Alert notification webhook | Slack/PagerDuty |

**Editing Configuration:**

```bash
# Edit ConfigMap
kubectl edit configmap mhras-config -n ${KUBE_NAMESPACE}

# Or update from file
kubectl apply -f k8s/configmap.yaml -n ${KUBE_NAMESPACE}

# Restart pods to pick up changes
kubectl rollout restart deployment/mhras-api -n ${KUBE_NAMESPACE}

# Edit Secrets (use external secret management in production)
kubectl edit secret mhras-secrets -n ${KUBE_NAMESPACE}
```

### Model Management

**Model Artifacts Storage:**

```bash
# Models stored in S3-compatible storage
# Structure:
# s3://mhras-models/
#   ├── baseline/
#   │   ├── logistic_regression_v1.2.3.pkl
#   │   └── lightgbm_v1.2.3.pkl
#   ├── temporal/
#   │   ├── rnn_v1.2.3.pt
#   │   └── tft_v1.2.3.pt
#   └── anomaly/
#       └── isolation_forest_v1.2.3.pkl

# Upload models
aws s3 sync models/ s3://mhras-models/ --exclude "*.pyc"

# Configure model storage in deployment
# Set MODEL_STORAGE_URL environment variable
```

**Model Registry Configuration:**

```python
# In src/ml/model_registry.py
# Models are loaded from storage on startup
# Registry tracks active models and versions
# Supports A/B testing and gradual rollout
```

### Feature Flags

**Enabling/Disabling Features:**

```yaml
# In k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mhras-config
data:
  # Feature flags
  ENABLE_SHAP_EXPLANATIONS: "true"
  ENABLE_COUNTERFACTUALS: "true"
  ENABLE_RULE_EXTRACTION: "false"  # Disabled for performance
  ENABLE_DRIFT_MONITORING: "true"
  ENABLE_HUMAN_REVIEW_QUEUE: "true"
  ENABLE_CRISIS_OVERRIDE: "true"
```

## Monitoring Setup

### Prometheus Configuration

**Metrics Collection:**

Prometheus scrapes metrics from:
- MHRAS API pods (`/metrics` endpoint)
- Kubernetes metrics
- PostgreSQL exporter
- Node exporter

**Configuration File:** `monitoring/prometheus/prometheus.yaml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mhras-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - mhras
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: mhras-api
```

**Key Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `mhras_http_requests_total` | Counter | Total HTTP requests by endpoint, method, status |
| `mhras_http_request_duration_seconds` | Histogram | Request duration distribution |
| `mhras_screenings_total` | Counter | Total screening requests |
| `mhras_risk_scores` | Histogram | Distribution of risk scores |
| `mhras_alerts_triggered_total` | Counter | Total alerts triggered |
| `mhras_human_reviews_queued_total` | Counter | Cases queued for human review |
| `mhras_model_inference_duration_seconds` | Histogram | Model inference time |
| `mhras_drift_score` | Gauge | Current drift score |
| `mhras_consent_verifications_total` | Counter | Consent verification attempts |
| `mhras_consent_failures_total` | Counter | Consent verification failures |

**Accessing Prometheus:**

```bash
# Port forward to Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Visit http://localhost:9090
# Query examples:
# - rate(mhras_http_requests_total[5m])
# - histogram_quantile(0.95, mhras_http_request_duration_seconds_bucket)
# - mhras_human_reviews_queued_total
```

### Grafana Dashboards

**Pre-Built Dashboards:**

1. **Operations Dashboard** (`operations-dashboard.json`)
   - Request rate and latency
   - Error rates by endpoint
   - Pod resource usage
   - API availability

2. **ML Dashboard** (`ml-dashboard.json`)
   - Model inference times
   - Prediction distribution
   - Drift scores over time
   - Model performance metrics

3. **Clinical Dashboard** (`clinical-dashboard.json`)
   - Risk score distribution
   - Alert frequency
   - Human review queue size
   - Crisis override usage

4. **Compliance Dashboard** (`compliance-dashboard.json`)
   - Audit log volume
   - Consent verification rates
   - Data lineage tracking
   - Access patterns

**Importing Dashboards:**

```bash
# Dashboards are in monitoring/grafana/dashboards/

# Option 1: Import via UI
# 1. Access Grafana: kubectl port-forward svc/grafana 3000:3000 -n monitoring
# 2. Navigate to Dashboards > Import
# 3. Upload JSON files

# Option 2: Provision via ConfigMap
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n monitoring

# Update Grafana deployment to mount ConfigMap
# See monitoring/grafana/provisioning/dashboards.yaml
```

**Grafana Configuration:**

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Default credentials: admin/admin (change immediately)

# Add Prometheus data source:
# 1. Configuration > Data Sources > Add data source
# 2. Select Prometheus
# 3. URL: http://prometheus:9090
# 4. Save & Test
```

### Alert Configuration

**Alert Rules:** `monitoring/prometheus/alerts/mhras-alerts.yaml`

**Critical Alerts:**

| Alert | Condition | Action |
|-------|-----------|--------|
| `APIDown` | No healthy pods for 5 minutes | Page on-call engineer |
| `HighErrorRate` | Error rate > 5% for 5 minutes | Page on-call engineer |
| `HighLatency` | P95 latency > 10s for 5 minutes | Page on-call engineer |
| `DatabaseDown` | Cannot connect to database | Page on-call engineer |
| `CriticalRiskBacklog` | >50 critical cases in review queue | Page clinical team |

**Warning Alerts:**

| Alert | Condition | Action |
|-------|-----------|--------|
| `ModerateErrorRate` | Error rate > 2% for 10 minutes | Notify team |
| `SlowRequests` | P95 latency > 7s for 10 minutes | Notify team |
| `HighDrift` | Drift score > 0.5 | Notify ML team |
| `LowModelConfidence` | Avg confidence < 0.6 for 1 hour | Notify ML team |
| `HighMemoryUsage` | Memory usage > 80% | Notify ops team |

**Info Alerts:**

| Alert | Condition | Action |
|-------|-----------|--------|
| `ModelRetrained` | New model version deployed | Log for review |
| `ConsentExpiring` | Consents expiring in 7 days | Log for review |
| `HighReviewQueueSize` | >20 cases in review queue | Log for review |

**Notification Channels:**

```yaml
# Configure in Alertmanager
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<pagerduty-key>'
        severity: 'critical'
  
  - name: 'slack'
    slack_configs:
      - api_url: '<slack-webhook-url>'
        channel: '#mhras-alerts'
        title: 'MHRAS Alert'
  
  - name: 'email'
    email_configs:
      - to: 'ops-team@example.com'
        from: 'alerts@mhras.example.com'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'
```

### Log Aggregation

**Structured Logging:**

All logs are in JSON format for easy parsing:

```json
{
  "timestamp": "2025-11-17T10:30:00.123Z",
  "level": "INFO",
  "logger": "src.api.endpoints",
  "message": "Screening request completed",
  "request_id": "req_abc123",
  "user_id": "clinician_001",
  "anonymized_id": "a1b2c3d4e5f6",
  "risk_score": 68.5,
  "response_time": 3.456
}
```

**Log Collection Options:**

**Option 1: ELK Stack (Elasticsearch, Logstash, Kibana)**

```bash
# Deploy ELK stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch -n monitoring
helm install kibana elastic/kibana -n monitoring
helm install filebeat elastic/filebeat -n monitoring

# Configure Filebeat to collect logs from pods
# See monitoring/elk/filebeat-config.yaml
```

**Option 2: Loki + Grafana**

```bash
# Deploy Loki
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack -n monitoring

# Loki integrates with Grafana for log viewing
# Access via Grafana > Explore > Select Loki data source
```

**Option 3: Cloud Provider Logging**

```bash
# AWS CloudWatch
# Enable CloudWatch Container Insights
eksctl utils install-cloudwatch-logs --cluster=mhras-cluster

# Google Cloud Logging
# Automatically enabled for GKE clusters

# Azure Monitor
# Enable Container Insights in Azure Portal
```

**Viewing Logs:**

```bash
# View logs from all pods
kubectl logs -f deployment/mhras-api -n mhras

# View logs from specific pod
kubectl logs -f <pod-name> -n mhras

# View logs with grep
kubectl logs deployment/mhras-api -n mhras | grep ERROR

# View logs from previous pod (after crash)
kubectl logs <pod-name> -n mhras --previous
```

### Health Checks and Probes

**Liveness Probe:**
- Endpoint: `/health`
- Interval: 10s
- Timeout: 5s
- Failure threshold: 3

**Readiness Probe:**
- Endpoint: `/health`
- Interval: 5s
- Timeout: 3s
- Failure threshold: 2

**Startup Probe:**
- Endpoint: `/health`
- Interval: 5s
- Timeout: 3s
- Failure threshold: 30 (allows 150s for startup)

```yaml
# In k8s/deployment.yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2

startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 30
```

## Security Hardening

### Container Security

**Non-Root User:**
```dockerfile
# In Dockerfile
RUN useradd -m -u 1000 mhras
USER mhras
```

**Security Scanning:**
```bash
# Scan image for vulnerabilities
docker scan ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}

# Or use Trivy
trivy image ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}

# Fail build on high/critical vulnerabilities
trivy image --severity HIGH,CRITICAL --exit-code 1 \
  ${DOCKER_REGISTRY}/mhras-api:${MHRAS_VERSION}
```

**Security Context:**
```yaml
# In k8s/deployment.yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

### Network Security

**Network Policies:**

```yaml
# Restrict ingress to API pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mhras-api-policy
spec:
  podSelector:
    matchLabels:
      app: mhras-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 53  # DNS
```

**TLS/HTTPS Configuration:**

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=api.mhras.example.com"

# Create TLS secret
kubectl create secret tls mhras-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n mhras

# Production: Use Let's Encrypt with cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
EOF
```

**Rate Limiting:**

```yaml
# In Ingress annotations
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mhras-ingress
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"  # requests per minute
    nginx.ingress.kubernetes.io/limit-rps: "10"    # requests per second
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "2"
```

### Data Security

**PII Anonymization:**
- All identifiers hashed with SHA-256 + salt
- Salt stored securely in Kubernetes Secret
- Anonymization applied before any processing
- No PII in logs, metrics, or error messages

**Encryption:**

```bash
# Encrypt secrets at rest (Kubernetes)
# Enable encryption provider in kube-apiserver
# See: https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/

# Database encryption
# Enable TLS for PostgreSQL connections
# Use encrypted storage volumes (cloud provider)

# Backup encryption
# Encrypt backups before storing
gpg --encrypt --recipient admin@example.com backup.sql
```

**Audit Logging:**

All data access is logged:
- User ID and role
- Anonymized individual ID
- Data types accessed
- Timestamp
- Action performed
- Result (success/failure)

```python
# Audit log example
{
  "event_type": "screening_request",
  "user_id": "clinician_001",
  "user_role": "clinician",
  "anonymized_id": "a1b2c3d4e5f6",
  "data_types": ["survey", "wearable", "emr"],
  "action": "generate_risk_score",
  "result": "success",
  "risk_score": 68.5,
  "timestamp": "2025-11-17T10:30:00Z"
}
```

### RBAC Configuration

**Kubernetes RBAC:**

```yaml
# Service account for MHRAS pods
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mhras-api
  namespace: mhras

---
# Role for accessing ConfigMaps and Secrets
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mhras-api-role
  namespace: mhras
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list"]

---
# Bind role to service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mhras-api-binding
  namespace: mhras
subjects:
  - kind: ServiceAccount
    name: mhras-api
    namespace: mhras
roleRef:
  kind: Role
  name: mhras-api-role
  apiGroup: rbac.authorization.k8s.io
```

**Application RBAC:**

User roles and permissions:

| Role | Permissions |
|------|-------------|
| `admin` | Full access to all endpoints and configuration |
| `clinician` | Screen individuals, view risk scores, access review queue |
| `data_scientist` | View predictions, access explanations, monitor drift |
| `auditor` | Read-only access to audit logs and compliance reports |

### Compliance

**HIPAA Compliance:**
- PHI encrypted at rest and in transit
- Access controls and audit logging
- Business Associate Agreements (BAA) with vendors
- Regular security risk assessments
- Incident response plan

**GDPR Compliance:**
- Right to access: API for data retrieval
- Right to erasure: Data deletion procedures
- Data portability: Export functionality
- Consent management: Granular consent tracking
- Data breach notification: Incident response plan

**Security Audits:**
- Quarterly penetration testing
- Annual security audits
- Continuous vulnerability scanning
- Code security reviews

## Scaling and Performance

### Horizontal Pod Autoscaling (HPA)

**Configuration:**
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mhras-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mhras-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 30
        - type: Pods
          value: 2
          periodSeconds: 30
      selectPolicy: Max
```

**Manual Scaling:**
```bash
# Scale to specific number of replicas
kubectl scale deployment mhras-api --replicas=5 -n mhras

# Check HPA status
kubectl get hpa -n mhras

# Describe HPA for details
kubectl describe hpa mhras-api-hpa -n mhras
```

### Vertical Scaling

**Resource Requests and Limits:**

| Environment | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-------------|-------------|-----------|----------------|--------------|
| Development | 500m | 1000m | 1Gi | 2Gi |
| Staging | 1000m | 2000m | 2Gi | 4Gi |
| Production | 2000m | 4000m | 4Gi | 8Gi |

**Updating Resources:**
```bash
# Edit deployment
kubectl edit deployment mhras-api -n mhras

# Or patch deployment
kubectl patch deployment mhras-api -n mhras -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"mhras-api","resources":{"requests":{"memory":"4Gi","cpu":"2000m"},"limits":{"memory":"8Gi","cpu":"4000m"}}}]}}}}'

# Rollout restart to apply changes
kubectl rollout restart deployment/mhras-api -n mhras
```

### Database Scaling

**Read Replicas:**
```bash
# Add read replica for PostgreSQL
# Configure connection pooling to distribute reads

# Example with PgBouncer
helm install pgbouncer bitnami/pgbouncer \
  --set postgresql.host=postgres-primary \
  --set postgresql.port=5432 \
  --set replicaCount=3
```

**Connection Pooling:**
```python
# In src/database/connection.py
# SQLAlchemy connection pool configuration
engine = create_engine(
    database_url,
    pool_size=20,          # Number of connections to maintain
    max_overflow=10,       # Additional connections when pool is full
    pool_timeout=30,       # Timeout for getting connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True     # Verify connections before use
)
```

### Performance Optimization

**Caching:**
```python
# Model caching (already implemented)
# Consent caching (already implemented)
# Feature caching for repeated requests

# Add Redis for distributed caching (optional)
helm install redis bitnami/redis -n mhras
```

**Database Indexing:**
```sql
-- Ensure indexes exist on frequently queried columns
CREATE INDEX idx_predictions_anonymized_id ON predictions(anonymized_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX idx_consent_expires_at ON consent(expires_at);
```

**Query Optimization:**
```bash
# Monitor slow queries
kubectl exec -it postgres-0 -n mhras -- psql -U mhras -c \
  "SELECT query, mean_exec_time, calls FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Enable query logging
# Set log_min_duration_statement = 1000 (log queries > 1s)
```

### Load Testing

**Using Locust:**
```python
# tests/load_test.py
from locust import HttpUser, task, between

class MHRASUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Generate auth token
        self.token = generate_token()
    
    @task
    def screen_individual(self):
        self.client.post(
            "/screen",
            json={
                "anonymized_id": "test_user",
                "survey_data": {"phq9_score": 15},
                "consent_verified": True
            },
            headers={"Authorization": f"Bearer {self.token}"}
        )

# Run load test
locust -f tests/load_test.py --host=https://api.mhras.example.com
```

**Performance Targets:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Screening latency (p50) | < 3s | 50th percentile |
| Screening latency (p95) | < 5s | 95th percentile |
| Screening latency (p99) | < 7s | 99th percentile |
| Throughput | > 100 req/s | Sustained load |
| Error rate | < 0.1% | Under normal load |
| Availability | > 99.9% | Monthly uptime |

## Backup and Disaster Recovery

### Database Backups

**Automated Backups:**
```bash
# Using pg_dump
kubectl run backup-job \
  --image=postgres:14 \
  --restart=Never \
  --namespace=mhras \
  --env="PGPASSWORD=${DB_PASSWORD}" \
  --command -- bash -c \
  "pg_dump -h postgres -U mhras -d mhras -F c -f /backup/mhras_$(date +%Y%m%d_%H%M%S).dump"

# Schedule with CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: mhras
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:14
              env:
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: mhras-secrets
                      key: db_password
              command:
                - /bin/bash
                - -c
                - |
                  pg_dump -h postgres -U mhras -d mhras -F c | \
                  aws s3 cp - s3://mhras-backups/postgres/backup_\$(date +%Y%m%d_%H%M%S).dump
          restartPolicy: OnFailure
EOF
```

**Backup Retention:**
- Daily backups: 7 days
- Weekly backups: 4 weeks
- Monthly backups: 12 months

**Backup Verification:**
```bash
# Test restore on staging
pg_restore -h staging-postgres -U mhras -d mhras_test backup.dump

# Verify data integrity
psql -h staging-postgres -U mhras -d mhras_test -c "SELECT COUNT(*) FROM predictions;"
```

### Model Artifacts Backup

```bash
# Backup models to S3
aws s3 sync /models s3://mhras-backups/models/ --exclude "*.pyc"

# Versioned backups
aws s3 cp /models s3://mhras-backups/models/$(date +%Y%m%d)/ --recursive
```

### Configuration Backup

```bash
# Backup Kubernetes manifests
kubectl get all,configmap,secret,ingress,hpa,networkpolicy \
  -n mhras -o yaml > mhras-backup-$(date +%Y%m%d).yaml

# Store in version control
git add k8s/
git commit -m "Backup Kubernetes configuration"
git push
```

### Disaster Recovery Plan

**Recovery Time Objective (RTO):** 4 hours  
**Recovery Point Objective (RPO):** 24 hours

**Recovery Steps:**

1. **Assess Damage**
   ```bash
   # Check cluster status
   kubectl get nodes
   kubectl get pods --all-namespaces
   
   # Check database status
   kubectl exec -it postgres-0 -n mhras -- psql -U mhras -c "SELECT 1;"
   ```

2. **Restore Database**
   ```bash
   # Download latest backup
   aws s3 cp s3://mhras-backups/postgres/latest.dump /tmp/backup.dump
   
   # Restore database
   kubectl run restore-job \
     --image=postgres:14 \
     --restart=Never \
     --namespace=mhras \
     --command -- pg_restore -h postgres -U mhras -d mhras /tmp/backup.dump
   ```

3. **Redeploy Application**
   ```bash
   # Apply all manifests
   kubectl apply -f k8s/ -n mhras
   
   # Wait for rollout
   kubectl rollout status deployment/mhras-api -n mhras
   ```

4. **Verify Recovery**
   ```bash
   # Test health endpoint
   curl https://api.mhras.example.com/health
   
   # Test screening endpoint
   curl -X POST https://api.mhras.example.com/screen \
     -H "Authorization: Bearer ${TOKEN}" \
     -d @test_request.json
   
   # Check metrics
   kubectl port-forward svc/prometheus 9090:9090 -n monitoring
   ```

5. **Post-Recovery**
   - Review incident timeline
   - Update runbooks
   - Conduct post-mortem
   - Implement preventive measures

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

**Symptoms:**
- Pods in `Pending`, `CrashLoopBackOff`, or `ImagePullBackOff` state

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n mhras

# Describe pod for events
kubectl describe pod <pod-name> -n mhras

# Check logs
kubectl logs <pod-name> -n mhras

# Check previous logs (if crashed)
kubectl logs <pod-name> -n mhras --previous
```

**Common Causes:**
- Insufficient resources: Check `kubectl top nodes`
- Image pull errors: Verify registry credentials
- Configuration errors: Check ConfigMap and Secrets
- Failed health checks: Review probe configuration

**Solutions:**
```bash
# Increase resources
kubectl patch deployment mhras-api -n mhras -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"mhras-api","resources":{"requests":{"memory":"4Gi"}}}]}}}}'

# Fix image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n mhras

# Restart deployment
kubectl rollout restart deployment/mhras-api -n mhras
```

#### 2. High Latency

**Symptoms:**
- API responses taking > 5 seconds
- Timeout errors (504)

**Diagnosis:**
```bash
# Check HPA status
kubectl get hpa -n mhras

# Check resource usage
kubectl top pods -n mhras

# Check Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Query: histogram_quantile(0.95, mhras_http_request_duration_seconds_bucket)

# Check database performance
kubectl exec -it postgres-0 -n mhras -- psql -U mhras -c \
  "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

**Common Causes:**
- Insufficient replicas: HPA not scaling fast enough
- Database slow queries: Missing indexes
- Model loading delays: Cold start issues
- Network latency: Cross-region calls

**Solutions:**
```bash
# Increase min replicas
kubectl patch hpa mhras-api-hpa -n mhras -p '{"spec":{"minReplicas":5}}'

# Add database indexes (see Performance Optimization section)

# Warm up model cache
kubectl exec -it <pod-name> -n mhras -- python -c \
  "from src.ml.model_registry import ModelRegistry; ModelRegistry().load_all_models()"
```

#### 3. Database Connection Issues

**Symptoms:**
- `OperationalError: could not connect to server`
- Pods failing health checks

**Diagnosis:**
```bash
# Verify secret
kubectl get secret mhras-secrets -n mhras -o jsonpath='{.data.database_url}' | base64 -d

# Test connection from pod
kubectl exec -it <pod-name> -n mhras -- bash
# Inside pod:
python -c "from src.database.connection import get_engine; get_engine().connect()"

# Check PostgreSQL status
kubectl get pods -l app=postgres -n mhras
kubectl logs postgres-0 -n mhras
```

**Common Causes:**
- Incorrect connection string
- Database not ready
- Network policy blocking traffic
- Connection pool exhausted

**Solutions:**
```bash
# Update database URL
kubectl edit secret mhras-secrets -n mhras

# Restart PostgreSQL
kubectl rollout restart statefulset/postgres -n mhras

# Check network policy
kubectl get networkpolicy -n mhras
kubectl describe networkpolicy mhras-api-policy -n mhras

# Increase connection pool
# Edit src/database/connection.py and redeploy
```

#### 4. High Error Rate

**Symptoms:**
- Error rate > 2% in Grafana
- 500/503 errors in logs

**Diagnosis:**
```bash
# Check error logs
kubectl logs -f deployment/mhras-api -n mhras | grep ERROR

# Check Prometheus
# Query: rate(mhras_http_requests_total{status=~"5.."}[5m])

# Check model status
kubectl exec -it <pod-name> -n mhras -- python -c \
  "from src.ml.model_registry import ModelRegistry; print(ModelRegistry().get_active_models())"
```

**Common Causes:**
- Model loading failures
- Validation errors
- Consent verification failures
- Feature engineering errors

**Solutions:**
```bash
# Check model files
kubectl exec -it <pod-name> -n mhras -- ls -la /models

# Review validation schemas
kubectl exec -it <pod-name> -n mhras -- cat /app/src/ingestion/schemas/survey_schema.json

# Check consent database
kubectl exec -it postgres-0 -n mhras -- psql -U mhras -c "SELECT COUNT(*) FROM consent;"
```

#### 5. Memory Leaks

**Symptoms:**
- Memory usage increasing over time
- OOMKilled pods

**Diagnosis:**
```bash
# Monitor memory usage
kubectl top pods -n mhras --watch

# Check pod events
kubectl get events -n mhras --sort-by='.lastTimestamp'

# Profile memory usage
kubectl exec -it <pod-name> -n mhras -- python -m memory_profiler src/main.py
```

**Solutions:**
```bash
# Increase memory limits
kubectl patch deployment mhras-api -n mhras -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"mhras-api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'

# Enable periodic restarts
kubectl patch deployment mhras-api -n mhras -p \
  '{"spec":{"template":{"metadata":{"annotations":{"restart-policy":"periodic"}}}}}'

# Review code for memory leaks
# Check model caching, feature caching, connection pooling
```

### Rollback Procedures

**View Deployment History:**
```bash
# View rollout history
kubectl rollout history deployment/mhras-api -n mhras

# View specific revision
kubectl rollout history deployment/mhras-api -n mhras --revision=3
```

**Rollback to Previous Version:**
```bash
# Rollback to previous revision
kubectl rollout undo deployment/mhras-api -n mhras

# Rollback to specific revision
kubectl rollout undo deployment/mhras-api -n mhras --to-revision=2

# Monitor rollback
kubectl rollout status deployment/mhras-api -n mhras
```

**Rollback Database Migration:**
```bash
# Run down migration
kubectl run migration-rollback \
  --image=${DOCKER_REGISTRY}/mhras-api:previous-version \
  --restart=Never \
  --namespace=mhras \
  --command -- python -m src.database.migration_runner --down

# Or restore from backup
kubectl run restore-job \
  --image=postgres:14 \
  --restart=Never \
  --namespace=mhras \
  --command -- pg_restore -h postgres -U mhras -d mhras /backup/previous.dump
```

### Emergency Procedures

**Complete System Outage:**
```bash
# 1. Check cluster health
kubectl get nodes
kubectl get pods --all-namespaces

# 2. Check critical services
kubectl get pods -n mhras
kubectl get pods -n monitoring

# 3. Restart all services
kubectl rollout restart deployment/mhras-api -n mhras
kubectl rollout restart deployment/prometheus -n monitoring
kubectl rollout restart deployment/grafana -n monitoring

# 4. Verify recovery
curl https://api.mhras.example.com/health

# 5. Notify stakeholders
# Send status update via communication channels
```

**Data Breach Response:**
```bash
# 1. Isolate affected systems
kubectl scale deployment mhras-api --replicas=0 -n mhras

# 2. Preserve evidence
kubectl logs deployment/mhras-api -n mhras > incident-logs.txt
kubectl get events -n mhras > incident-events.txt

# 3. Investigate
# Review audit logs, access logs, database queries

# 4. Remediate
# Rotate secrets, patch vulnerabilities, update access controls

# 5. Restore service
kubectl scale deployment mhras-api --replicas=3 -n mhras

# 6. Post-incident
# Conduct post-mortem, update security procedures, notify affected parties
```

## Production Checklist

### Pre-Deployment

**Infrastructure:**
- [ ] Kubernetes cluster provisioned (5+ nodes for production)
- [ ] kubectl configured with cluster credentials
- [ ] Namespaces created (`mhras`, `monitoring`)
- [ ] Storage classes configured
- [ ] Load balancer provisioned
- [ ] DNS records configured
- [ ] TLS certificates obtained (Let's Encrypt or commercial)

**Database:**
- [ ] PostgreSQL instance provisioned (managed service recommended)
- [ ] Database `mhras` created
- [ ] Database user created with appropriate permissions
- [ ] Connection tested from Kubernetes cluster
- [ ] Backup strategy configured (daily backups, 30-day retention)
- [ ] Monitoring enabled (pg_stat_statements, slow query log)
- [ ] Read replicas configured (if needed)

**Security:**
- [ ] Kubernetes RBAC configured
- [ ] Network policies reviewed and applied
- [ ] Secrets management solution configured (Vault, Sealed Secrets, or cloud KMS)
- [ ] JWT secret key generated (256-bit minimum)
- [ ] Anonymization salt generated (256-bit minimum)
- [ ] TLS certificates installed
- [ ] Security scanning tools configured (Trivy, Snyk)
- [ ] Vulnerability scanning scheduled

**Application:**
- [ ] Docker registry accessible
- [ ] Model artifacts uploaded to storage (S3, GCS, Azure Blob)
- [ ] Configuration files prepared (ConfigMap, Secrets)
- [ ] Environment variables documented
- [ ] Deployment manifests reviewed
- [ ] Resource limits configured appropriately
- [ ] Health check endpoints tested

**Monitoring:**
- [ ] Prometheus installed and configured
- [ ] Grafana installed with dashboards imported
- [ ] Alert manager configured
- [ ] Notification channels set up (email, Slack, PagerDuty)
- [ ] Log aggregation configured (ELK, Loki, or cloud service)
- [ ] Metrics retention configured (30 days minimum)
- [ ] On-call rotation established

### Deployment

- [ ] Docker image built and pushed to registry
- [ ] Image scanned for vulnerabilities (no high/critical issues)
- [ ] Database migrations applied
- [ ] ConfigMap applied
- [ ] Secrets applied (all `CHANGE_ME` values updated)
- [ ] Deployment applied
- [ ] Service applied
- [ ] HPA applied
- [ ] Network policies applied
- [ ] Ingress applied with TLS
- [ ] Monitoring stack deployed
- [ ] Alert rules applied

### Post-Deployment

**Verification:**
- [ ] All pods running and healthy
- [ ] Health endpoint responding
- [ ] API endpoints accessible via ingress
- [ ] Authentication working
- [ ] Database connections successful
- [ ] Model loading successful
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Alerts configured and firing (test alerts)

**Testing:**
- [ ] Smoke tests passed
- [ ] End-to-end screening test successful
- [ ] Load testing completed (meets performance targets)
- [ ] Failover testing completed
- [ ] Backup and restore tested
- [ ] Rollback procedure tested

**Documentation:**
- [ ] Deployment runbook updated
- [ ] Troubleshooting guide updated
- [ ] On-call playbook created
- [ ] Architecture diagram updated
- [ ] API documentation published
- [ ] User guides updated

**Training:**
- [ ] Operations team trained on deployment procedures
- [ ] Clinical team trained on system usage
- [ ] On-call team trained on incident response
- [ ] Stakeholders briefed on system capabilities

**Compliance:**
- [ ] Security audit completed
- [ ] HIPAA compliance verified
- [ ] GDPR compliance verified (if applicable)
- [ ] Data retention policies configured
- [ ] Audit logging enabled and tested
- [ ] Incident response plan documented
- [ ] Business continuity plan documented

### Ongoing Operations

**Daily:**
- [ ] Check dashboard for anomalies
- [ ] Review error logs
- [ ] Monitor alert notifications
- [ ] Check backup completion

**Weekly:**
- [ ] Review performance metrics
- [ ] Check resource utilization
- [ ] Review security scan results
- [ ] Update dependencies (if needed)

**Monthly:**
- [ ] Review and update documentation
- [ ] Conduct disaster recovery drill
- [ ] Review and optimize costs
- [ ] Security patch updates

**Quarterly:**
- [ ] Penetration testing
- [ ] Capacity planning review
- [ ] Incident post-mortem review
- [ ] Update disaster recovery plan

## Additional Resources

### Documentation

- [Kubernetes Deployment Guide](../k8s/README.md) - Detailed Kubernetes configuration
- [Monitoring Setup](../monitoring/README.md) - Prometheus and Grafana setup
- [API Documentation](./api_usage.md) - Complete API reference
- [Governance Documentation](./governance_usage.md) - Compliance and governance
- [Database Usage](./database_usage.md) - Database schema and operations
- [Model Training](./model_training_usage.md) - ML model training and deployment

### External Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

### Tools

- **kubectl**: Kubernetes CLI
- **helm**: Kubernetes package manager
- **k9s**: Terminal UI for Kubernetes
- **stern**: Multi-pod log tailing
- **kubectx/kubens**: Context and namespace switching
- **kustomize**: Kubernetes configuration management

### Support Contacts

**Technical Support:**
- Email: support@mhras.example.com
- Slack: #mhras-support
- On-call: PagerDuty rotation

**Security Issues:**
- Email: security@mhras.example.com
- PGP Key: Available on website

**Escalation:**
- Level 1: On-call engineer
- Level 2: DevOps team lead
- Level 3: CTO

## Appendix

### Useful Commands

```bash
# Quick health check
kubectl get pods -n mhras && curl https://api.mhras.example.com/health

# View logs from all pods
kubectl logs -f deployment/mhras-api -n mhras --all-containers=true

# Get resource usage
kubectl top nodes && kubectl top pods -n mhras

# Check HPA status
kubectl get hpa -n mhras -w

# Port forward to services
kubectl port-forward svc/mhras-api 8000:80 -n mhras
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Restart deployment
kubectl rollout restart deployment/mhras-api -n mhras

# Scale deployment
kubectl scale deployment mhras-api --replicas=5 -n mhras

# View deployment history
kubectl rollout history deployment/mhras-api -n mhras

# Rollback deployment
kubectl rollout undo deployment/mhras-api -n mhras
```

### Environment Variables Reference

See [Configuration Management](#configuration-management) section for complete list.

### Troubleshooting Decision Tree

```
Issue: API not responding
├─ Check pods: kubectl get pods -n mhras
│  ├─ Pods not running → Check events: kubectl describe pod
│  │  ├─ ImagePullBackOff → Check registry credentials
│  │  ├─ CrashLoopBackOff → Check logs: kubectl logs
│  │  └─ Pending → Check resources: kubectl top nodes
│  └─ Pods running → Check service: kubectl get svc -n mhras
│     ├─ Service not found → Apply service manifest
│     └─ Service exists → Check ingress: kubectl get ingress -n mhras
│        ├─ Ingress not found → Apply ingress manifest
│        └─ Ingress exists → Check TLS certificate
│
Issue: High latency
├─ Check HPA: kubectl get hpa -n mhras
│  ├─ Not scaling → Check metrics server
│  └─ Scaling → Check resource limits
├─ Check database: kubectl exec -it postgres-0 -n mhras -- psql
│  ├─ Slow queries → Add indexes
│  └─ Connection pool exhausted → Increase pool size
└─ Check model loading: kubectl logs deployment/mhras-api -n mhras | grep "model"
   ├─ Models not cached → Warm up cache
   └─ Models loading slowly → Check storage performance
```

---

**Document Version:** 1.0.0  
**Last Updated:** 2025-11-17  
**Maintained By:** DevOps Team
