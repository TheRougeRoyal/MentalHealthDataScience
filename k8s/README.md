# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Mental Health Risk Assessment System (MHRAS).

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured to access your cluster
- Container registry for storing Docker images
- PostgreSQL database (can be deployed in-cluster or external)
- Persistent storage provisioner

## Files Overview

- `deployment.yaml` - Main API deployment with 3 replicas, health checks, and resource limits
- `service.yaml` - ClusterIP service for load balancing
- `configmap.yaml` - Application configuration (non-sensitive)
- `secret.yaml` - Sensitive configuration (credentials, keys)
- `hpa.yaml` - Horizontal Pod Autoscaler for auto-scaling
- `ingress.yaml` - Ingress resource for external access
- `network-policy.yaml` - Network policies for security

## Deployment Steps

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t your-registry/mhras-api:v1.0.0 .

# Push to your container registry
docker push your-registry/mhras-api:v1.0.0
```

### 2. Update Configuration

Edit `secret.yaml` and replace all `CHANGE_ME` values with actual secrets:

```bash
# Generate JWT secret
openssl rand -hex 32

# Generate anonymization salt
openssl rand -hex 32
```

**Important**: In production, use a proper secret management solution like:
- Kubernetes External Secrets Operator
- HashiCorp Vault
- Cloud provider secret managers (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)

### 3. Update Image Reference

Edit `deployment.yaml` and update the image reference:

```yaml
image: your-registry/mhras-api:v1.0.0
```

### 4. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace mhras

# Apply configurations
kubectl apply -f k8s/configmap.yaml -n mhras
kubectl apply -f k8s/secret.yaml -n mhras
kubectl apply -f k8s/deployment.yaml -n mhras
kubectl apply -f k8s/service.yaml -n mhras
kubectl apply -f k8s/hpa.yaml -n mhras

# Optional: Apply ingress if you need external access
kubectl apply -f k8s/ingress.yaml -n mhras

# Optional: Apply network policies for security
kubectl apply -f k8s/network-policy.yaml -n mhras
```

### 5. Verify Deployment

```bash
# Check pod status
kubectl get pods -n mhras

# Check service
kubectl get svc -n mhras

# Check logs
kubectl logs -f deployment/mhras-api -n mhras

# Check health endpoint
kubectl port-forward svc/mhras-api 8000:80 -n mhras
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

Key environment variables are defined in `configmap.yaml` and `secret.yaml`:

**ConfigMap (non-sensitive)**:
- `log_level` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `risk_threshold_*` - Risk classification thresholds
- `human_review_threshold` - Score threshold for human review
- `drift_threshold` - Data drift detection threshold

**Secrets (sensitive)**:
- `database_url` - PostgreSQL connection string
- `jwt_secret_key` - JWT signing key
- `anonymization_salt` - Salt for PII hashing

### Resource Limits

Default resource allocation per pod:
- **Requests**: 1 CPU, 2GB RAM
- **Limits**: 2 CPU, 4GB RAM

Adjust based on your workload in `deployment.yaml`.

### Scaling

**Manual scaling**:
```bash
kubectl scale deployment mhras-api --replicas=5 -n mhras
```

**Auto-scaling**: Configured via `hpa.yaml`
- Min replicas: 3
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%

## Health Checks

The deployment includes:

**Liveness Probe**:
- Endpoint: `/health`
- Initial delay: 30s
- Period: 10s
- Failure threshold: 3

**Readiness Probe**:
- Endpoint: `/health`
- Initial delay: 10s
- Period: 5s
- Failure threshold: 3

## Storage

The deployment uses a PersistentVolumeClaim for model storage:
- Name: `mhras-models-pvc`
- Size: 10GB
- Access mode: ReadOnlyMany

Models should be pre-loaded to this volume or pulled from a model registry on startup.

## Security

### Network Policies

`network-policy.yaml` restricts traffic to:
- Ingress: Only from ingress controller and Prometheus
- Egress: DNS, database, and external HTTPS

### Pod Security

- Runs as non-root user (UID 1000)
- Read-only root filesystem (for model volume)
- Security context enforced

### TLS/HTTPS

Configure TLS in `ingress.yaml`:
- Uses cert-manager for automatic certificate management
- Forces HTTPS redirect

## Monitoring

Prometheus metrics are exposed at `/metrics` endpoint:
- Annotations configured in deployment for auto-discovery
- Scrape interval: 30s (default)

## Troubleshooting

### Pods not starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n mhras

# Check logs
kubectl logs <pod-name> -n mhras

# Check resource constraints
kubectl top pods -n mhras
```

### Database connection issues

```bash
# Verify secret is correct
kubectl get secret mhras-secrets -n mhras -o yaml

# Test database connectivity from pod
kubectl exec -it <pod-name> -n mhras -- bash
# Inside pod: test connection
```

### High latency

```bash
# Check HPA status
kubectl get hpa -n mhras

# Check resource usage
kubectl top pods -n mhras

# Check if pods are being throttled
kubectl describe pod <pod-name> -n mhras | grep -i throttl
```

## Rollback

```bash
# View deployment history
kubectl rollout history deployment/mhras-api -n mhras

# Rollback to previous version
kubectl rollout undo deployment/mhras-api -n mhras

# Rollback to specific revision
kubectl rollout undo deployment/mhras-api --to-revision=2 -n mhras
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/ -n mhras

# Delete namespace
kubectl delete namespace mhras
```

## Production Considerations

1. **High Availability**:
   - Deploy across multiple availability zones
   - Use pod anti-affinity rules (already configured)
   - Set up database replication

2. **Backup**:
   - Regular database backups
   - Model artifact versioning
   - Configuration backups

3. **Monitoring**:
   - Set up Prometheus and Grafana (see monitoring section)
   - Configure alerting rules
   - Set up log aggregation (ELK, Loki, etc.)

4. **Security**:
   - Use external secret management
   - Enable pod security policies/standards
   - Regular security scanning of images
   - Network policies enforcement

5. **Performance**:
   - Tune resource limits based on actual usage
   - Configure connection pooling
   - Enable caching where appropriate
   - Consider using a CDN for static assets
