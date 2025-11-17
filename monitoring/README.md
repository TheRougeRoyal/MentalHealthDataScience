# MHRAS Monitoring and Alerting

This directory contains monitoring and alerting configuration for the Mental Health Risk Assessment System using Prometheus and Grafana.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification

## Directory Structure

```
monitoring/
├── prometheus/
│   ├── prometheus.yaml          # Prometheus configuration
│   └── alerts/
│       └── mhras-alerts.yaml    # Alerting rules
├── grafana/
│   ├── dashboards/              # Pre-built dashboards
│   │   ├── operations-dashboard.json
│   │   ├── ml-dashboard.json
│   │   ├── clinical-dashboard.json
│   │   └── compliance-dashboard.json
│   └── provisioning/            # Grafana provisioning configs
│       ├── datasources.yaml
│       └── dashboards.yaml
└── k8s/                         # Kubernetes manifests
    ├── prometheus-deployment.yaml
    └── grafana-deployment.yaml
```

## Metrics Exposed

### Application Metrics

**HTTP Metrics**:
- `mhras_http_requests_total` - Total HTTP requests by method, endpoint, status
- `mhras_http_request_duration_seconds` - Request duration histogram

**ML Metrics**:
- `mhras_screenings_total` - Total screenings by risk level
- `mhras_prediction_duration_seconds` - Prediction generation time
- `mhras_model_inference_duration_seconds` - Model inference time by type
- `mhras_drift_score` - Data drift scores by feature

**Governance Metrics**:
- `mhras_human_review_queue_size` - Current queue size
- `mhras_errors_total` - Total errors by type

## Dashboards

### 1. Operations Dashboard
Monitors system health and performance:
- Request rate and latency
- Error rates
- CPU and memory usage
- Active pods

### 2. ML Dashboard
Tracks machine learning performance:
- Prediction duration
- Model inference times
- Drift scores
- Risk level distribution

### 3. Clinical Dashboard
Clinical workflow monitoring:
- Human review queue size
- Risk score distribution
- Critical and high-risk cases
- Screening trends

### 4. Compliance Dashboard
Audit and compliance tracking:
- Total screenings
- Human reviews completed
- Crisis overrides
- Consent violations
- Audit events

## Alerting Rules

### Critical Alerts
- **HighErrorRate**: Error rate > 5% for 5 minutes
- **HighResponseLatency**: p95 latency > 10s for 5 minutes
- **ServiceDown**: Service unavailable for 2 minutes

### Warning Alerts
- **FeatureDriftDetected**: Drift score > 0.3 for 10 minutes
- **LargeHumanReviewQueue**: Queue size > 50 for 15 minutes
- **HighMemoryUsage**: Memory usage > 85% for 10 minutes
- **HighCPUUsage**: CPU usage > 1.7 cores for 10 minutes
- **SlowModelInference**: p95 inference time > 2s for 10 minutes

### Info Alerts
- **HighCriticalRiskRate**: Critical risk rate > 10% for 30 minutes
- **ModelPerformanceDegradation**: Screening rate drops 50% vs 24h ago

## Deployment

### Prerequisites

```bash
# Create monitoring namespace
kubectl create namespace monitoring
```

### Deploy Prometheus

```bash
# Apply Prometheus configuration
kubectl apply -f monitoring/k8s/prometheus-deployment.yaml
```

### Deploy Grafana

```bash
# Set admin password
kubectl create secret generic grafana-secrets \
  --from-literal=admin_password='YOUR_SECURE_PASSWORD' \
  -n monitoring

# Apply Grafana configuration
kubectl apply -f monitoring/k8s/grafana-deployment.yaml

# Import dashboards
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n monitoring
```

### Access Dashboards

```bash
# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Access at http://localhost:3000
# Default credentials: admin / YOUR_SECURE_PASSWORD
```

## Configuration

### Adding Custom Metrics

To add custom metrics to your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metric
my_metric = Counter('mhras_my_metric_total', 'Description', ['label1'])

# Increment metric
my_metric.labels(label1='value').inc()
```

### Adding Alert Rules

Add new rules to `prometheus/alerts/mhras-alerts.yaml`:

```yaml
- alert: MyNewAlert
  expr: my_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Alert summary"
    description: "Alert description"
```

### Creating Custom Dashboards

1. Create dashboard in Grafana UI
2. Export as JSON
3. Save to `grafana/dashboards/`
4. Update ConfigMap and redeploy

## Alertmanager Configuration

Configure alert routing in Alertmanager:

```yaml
route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty'
  - match:
      severity: warning
    receiver: 'slack'

receivers:
- name: 'default'
  email_configs:
  - to: 'team@example.com'
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: 'YOUR_KEY'
- name: 'slack'
  slack_configs:
  - api_url: 'YOUR_WEBHOOK_URL'
    channel: '#alerts'
```

## Troubleshooting

### Prometheus not scraping metrics

```bash
# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Visit http://localhost:9090/targets

# Check pod annotations
kubectl get pod <pod-name> -n mhras -o yaml | grep prometheus
```

### Grafana dashboards not loading

```bash
# Check ConfigMap
kubectl get configmap grafana-dashboards -n monitoring

# Check Grafana logs
kubectl logs deployment/grafana -n monitoring
```

### Alerts not firing

```bash
# Check alert rules in Prometheus UI
# Visit http://localhost:9090/alerts

# Check Alertmanager
kubectl logs deployment/alertmanager -n monitoring
```

## Best Practices

1. **Metric Naming**: Follow Prometheus naming conventions
   - Use base unit (seconds, bytes)
   - Suffix with unit (_seconds, _bytes, _total)

2. **Label Cardinality**: Keep label cardinality low
   - Avoid high-cardinality labels (user IDs, timestamps)
   - Use aggregation for high-cardinality data

3. **Alert Fatigue**: Design alerts carefully
   - Set appropriate thresholds
   - Use proper severity levels
   - Group related alerts

4. **Dashboard Design**: Create focused dashboards
   - One purpose per dashboard
   - Use consistent time ranges
   - Include context and documentation

5. **Retention**: Configure appropriate retention
   - Prometheus: 30 days default
   - Long-term storage: Use Thanos or Cortex

## Production Considerations

1. **High Availability**:
   - Run multiple Prometheus replicas
   - Use Thanos for long-term storage
   - Deploy Alertmanager cluster

2. **Security**:
   - Enable authentication
   - Use TLS for communication
   - Restrict access with RBAC

3. **Performance**:
   - Tune scrape intervals
   - Use recording rules for expensive queries
   - Monitor Prometheus resource usage

4. **Backup**:
   - Backup Prometheus data
   - Version control dashboards
   - Document alert configurations
