# Mental Health Risk Assessment System (MHRAS)

A production-grade ML platform for mental health risk assessment that ingests multi-modal health data, generates interpretable risk predictions, and provides personalized recommendations while maintaining strict governance and ethical standards.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [CLI Tool](#cli-tool)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Development](#development)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Security & Compliance](#security--compliance)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Mental Health Risk Assessment System (MHRAS) is a comprehensive platform that:

- **Screens individuals** using data from surveys, wearable devices, and electronic medical records
- **Generates risk predictions** using ensemble machine learning models
- **Provides interpretable explanations** using SHAP and counterfactual analysis
- **Recommends personalized resources** based on risk level and individual profile
- **Maintains governance** through audit logging, human review queues, and drift monitoring
- **Ensures compliance** with HIPAA-ready features and consent management

### Key Capabilities

✅ **Multi-Modal Data Processing**
- Survey responses (PHQ-9, GAD-7, mood ratings)
- Wearable device metrics (heart rate, HRV, sleep, activity)
- Electronic medical records (diagnoses, medications, therapy history)

✅ **Advanced ML Pipeline**
- Multiple model types (Logistic Regression, Random Forest, LightGBM, LSTM, Transformer)
- Ensemble predictions with confidence scoring
- Real-time inference (<5 seconds)
- Automated model versioning and registry

✅ **Interpretability & Explainability**
- SHAP value explanations
- Feature importance rankings
- Counterfactual scenarios
- Clinical interpretations

✅ **Personalized Recommendations**
- Risk-based resource matching
- Profile-aware scoring
- Multiple resource types (therapy, medication, support groups, crisis hotlines)

✅ **Governance & Compliance**
- Complete audit trail
- Human review queue for high-risk cases
- Data drift monitoring
- Consent verification
- PII anonymization

---

## Features

### Core Features

1. **End-to-End Screening Workflow**
   - Data validation with JSON schemas
   - Consent verification
   - ETL processing (cleaning, imputation, normalization)
   - Feature engineering (behavioral, physiological, NLP)
   - ML inference with ensemble models
   - Interpretability generation
   - Personalized recommendations
   - Audit logging and review routing

2. **Machine Learning Pipeline**
   - Model registry with versioning
   - Multiple model types support
   - Ensemble predictions with voting
   - Confidence scoring
   - Alert triggering for high-risk cases
   - Automated retraining support

3. **Governance Layer**
   - Audit logging for all operations
   - Human review queue with escalation
   - Data and prediction drift monitoring
   - Consent management
   - Anonymization and de-identification

4. **API & Integration**
   - RESTful API with OpenAPI documentation
   - JWT authentication
   - Request validation
   - Error handling
   - Prometheus metrics
   - CLI tool for operations

### Data Science Platform Features

5. **Experiment Tracking**
   - Track ML experiments with parameters, metrics, and artifacts
   - Compare runs and identify best models
   - Automatic git commit tracking
   - Integration with model registry
   - Search and filter experiments

6. **Data Version Control**
   - Version datasets with automatic statistics
   - Track data lineage through transformations
   - Detect statistical drift between versions
   - Reproducible data snapshots
   - Compression and deduplication

7. **Exploratory Data Analysis (EDA)**
   - Automated statistical summaries
   - Data quality issue detection
   - Correlation analysis
   - Visualization generation
   - Export reports in HTML/PDF

8. **Model Cards**
   - Automated model documentation
   - Performance metrics and fairness analysis
   - Feature importance and SHAP visualizations
   - Baseline comparisons
   - Export in multiple formats

9. **Feature Store**
   - Centralized feature repository
   - Versioned feature definitions
   - Online and batch serving
   - Feature lineage tracking
   - Schema validation

10. **Notebook Integration**
    - Jupyter notebook tracking
    - IPython magic commands
    - Environment capture
    - Output stripping for version control

11. **Hyperparameter Optimization**
    - Bayesian, grid, and random search
    - Parallel trial execution
    - Early stopping
    - Parameter importance analysis
    - Visualization of optimization progress

12. **Performance Monitoring**
    - Continuous model performance tracking
    - Automated alerting on degradation
    - Drift detection
    - Weekly summary reports
    - Retraining recommendations

### Performance Targets

- **Screening Time**: <5 seconds per individual
- **Throughput**: 100+ screenings per minute
- **Availability**: 99.9% uptime
- **Accuracy**: >85% on validation set
- **Feature Serving**: <100ms latency (online mode)
- **Experiment Query**: <2 seconds for 10K runs

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  (FastAPI endpoints with authentication & middleware)        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Integration Layer                          │
│         (MHRASIntegration - Component orchestration)         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│  ML Pipeline   │  │   Governance    │  │  Data Layer     │
│                │  │                 │  │                 │
│ • Models       │  │ • Audit         │  │ • Database      │
│ • Inference    │  │ • Review Queue  │  │ • Repositories  │
│ • Features     │  │ • Drift Monitor │  │ • Migrations    │
│ • Ensemble     │  │ • Consent       │  │                 │
│ • Interpret    │  │ • Anonymization │  │                 │
└────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Screening Service                           │
│        (End-to-end orchestration of all components)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Data Science Platform                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Experiment  │  │    Data      │  │   Feature    │      │
│  │   Tracker    │  │  Versioning  │  │    Store     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼───────┐     │
│  │      Metadata & Artifact Storage (PostgreSQL)      │     │
│  └──────┬──────────────────┬──────────────────┬───────┘     │
│         │                  │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │     EDA      │  │   Model      │  │  Notebook    │      │
│  │   Module     │  │   Cards      │  │ Integration  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Hyperparam   │  │ Performance  │                        │
│  │  Optimizer   │  │  Monitor     │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Request → Validation → Consent → ETL → Features → ML Inference
    → Explanations → Recommendations → Governance → Response
```

### Component Layers

1. **Ingestion Layer** (`src/ingestion/`)
   - Data validation with JSON schemas
   - Multi-source data support
   - Schema versioning

2. **Processing Layer** (`src/processing/`)
   - ETL pipeline orchestration
   - Data cleaning and outlier detection
   - Missing value imputation
   - Feature normalization and encoding

3. **ML Layer** (`src/ml/`)
   - Model registry and versioning
   - Inference engine
   - Ensemble predictor
   - Interpretability engine
   - Feature engineering pipeline
   - Multiple model types (baseline, temporal, anomaly detection)

4. **Governance Layer** (`src/governance/`)
   - Audit logger
   - Human review queue
   - Drift monitor
   - Consent verifier
   - Anonymizer

5. **Recommendation Layer** (`src/recommendations/`)
   - Recommendation engine
   - Resource catalog
   - Profile-based matching

6. **API Layer** (`src/api/`)
   - REST endpoints
   - Authentication and authorization
   - Request/response validation
   - Middleware (logging, error handling, CORS)
   - Prometheus metrics

7. **Database Layer** (`src/database/`)
   - Connection management
   - Repository pattern
   - Schema migrations
   - Models (predictions, audit logs, consent, review queue, resources)

---

## Quick Start

> **🚀 New to the system?** See [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide!

### Prerequisites

- Python 3.9+
- PostgreSQL 12+ (optional, for full features)
- 4GB RAM minimum
- 10GB disk space

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mhras

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config/.env.example .env
# Edit .env with your settings (default works for development!)
```

### Minimum Configuration

The `.env` file comes pre-configured with development defaults:
```bash
# Database (default development credentials)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mhras
DB_USER=mhras_user
DB_PASSWORD=mhras_dev_password_2024

# Security (change in production!)
SECURITY_JWT_SECRET=change-me-in-production-use-strong-random-string
SECURITY_ANONYMIZATION_SALT=change-me-in-production-use-strong-random-string
```

**For production**, generate secure credentials:
```bash
# Generate secure password
openssl rand -base64 32

# Update .env with secure values
```

**See:** [CREDENTIALS.md](CREDENTIALS.md) for security best practices.

### Initialize Database

**Quick Setup (Recommended):**
```bash
# Run automated database setup
./setup_database.sh
```

This will create the database, user, run migrations, and configure your `.env` file.

**Manual Setup:**
```bash
# Run migrations
./run_migrations.sh

# Or use CLI
python run_cli.py db migrate

# Verify connection
python run_cli.py db check
```

**Default Development Credentials:**
- Database: `mhras`
- User: `mhras_user`
- Password: `mhras_dev_password_2024`
- Connection: `postgresql://mhras_user:mhras_dev_password_2024@localhost:5432/mhras`

> ⚠️ **Important**: Change the default password in production! See [CREDENTIALS.md](CREDENTIALS.md) for details.

**Documentation:**
- Quick reference: [DATABASE_QUICKSTART.md](DATABASE_QUICKSTART.md)
- Detailed guide: [docs/database_setup.md](docs/database_setup.md)
- Security: [CREDENTIALS.md](CREDENTIALS.md)

### Start the System

#### Option 1: API Server

```bash
python run_api.py
```

Access:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

#### Option 2: CLI Tool

```bash
# Check system health
python run_cli.py system health

# View statistics
python run_cli.py system stats

# List models
python run_cli.py models list
```

#### Option 3: Python Integration

```python
from src.integration import get_integration
from src.screening_service import ScreeningRequest
import asyncio

# Initialize
integration = get_integration()
service = integration.get_screening_service()

# Create request
request = ScreeningRequest(
    anonymized_id="patient_001",
    survey_data={
        "phq9_score": 15,
        "gad7_score": 12,
        "mood_rating": 3
    }
)

# Run screening
response = asyncio.run(service.screen_individual(request))

print(f"Risk Score: {response.risk_score}")
print(f"Risk Level: {response.risk_level}")
print(f"Recommendations: {len(response.recommendations)}")
```

### Quick Start: Data Science Features

#### Experiment Tracking

```python
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.storage import FileSystemStorage
from src.database.connection import get_db_connection

# Initialize
storage = FileSystemStorage(base_path="experiments/artifacts")
db = get_db_connection()
tracker = ExperimentTracker(storage_backend=storage, db_connection=db)

# Start a run
run = tracker.start_run(
    experiment_name="risk_prediction",
    run_name="baseline_v1",
    tags={"model": "random_forest"}
)

# Log parameters and metrics
tracker.log_params({"n_estimators": 100, "max_depth": 10})
tracker.log_metrics({"accuracy": 0.92, "f1_score": 0.89})
tracker.log_artifact("model.pkl", artifact_type="model")

# End run
tracker.end_run(status="FINISHED")

# Search and compare runs
runs = tracker.search_runs(experiment_name="risk_prediction")
comparison = tracker.compare_runs(run_ids=[run1.run_id, run2.run_id])
```

#### Data Versioning

```python
from src.ds.data_versioning import DataVersionControl
import pandas as pd

# Initialize
dvc = DataVersionControl(storage_backend=storage, db_connection=db)

# Register dataset
df = pd.read_csv("patient_data.csv")
version = dvc.register_dataset(
    dataset=df,
    dataset_name="patient_assessments",
    source="emr_system",
    metadata={"collection_date": "2024-01-15"}
)

# Retrieve specific version
df, version = dvc.get_dataset("patient_assessments", version="v1.0")

# Detect drift
drift_report = dvc.detect_drift(version_id1, version_id2)
print(f"Drift detected: {drift_report.drift_detected}")
```

#### Feature Store

```python
from src.ds.feature_store import FeatureStore, FeatureDefinition

# Initialize
feature_store = FeatureStore(db_connection=db)

# Register feature
feature_def = FeatureDefinition(
    feature_name="age_normalized",
    feature_type="numeric",
    description="Normalized age (0-1 scale)",
    transformation_code="def transform(df): return (df['age'] - 18) / 82",
    input_schema={"age": "int"},
    output_schema={"age_normalized": "float"},
    version="v1.0"
)
feature_store.register_feature("age_normalized", feature_def)

# Compute features
features = feature_store.compute_features(
    feature_names=["age_normalized", "severity_score"],
    input_data=df
)
```

#### Exploratory Data Analysis

```python
from src.ds.eda import EDAModule

# Initialize
eda = EDAModule()

# Run analysis
report = eda.analyze_dataset(df, target_column="risk_level")

# Export report
eda.export_report(report, format="html", output_path="eda_report.html")

# Check quality issues
for issue in report.quality_issues:
    print(f"[{issue.severity}] {issue.description}")
```

See the [examples/](examples/) directory for complete Jupyter notebooks demonstrating each feature.

---

## Usage

### Performing a Screening

#### Via API

```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "anonymized_id": "patient_001",
    "consent_verified": true,
    "survey_data": {
      "phq9_score": 15,
      "gad7_score": 12,
      "mood_rating": 3,
      "stress_level": 7,
      "sleep_quality": 4
    },
    "wearable_data": {
      "avg_heart_rate": 75,
      "avg_hrv": 45,
      "sleep_duration": 6.5,
      "activity_minutes": 30
    }
  }'
```

#### Via CLI

```bash
# Create data file
cat > patient_data.json << EOF
{
  "phq9_score": 15,
  "gad7_score": 12,
  "mood_rating": 3
}
EOF

# Run screening
python run_cli.py screen \
  --anonymized-id patient_001 \
  --survey-data patient_data.json
```

### Understanding Results

#### Risk Levels

- **LOW** (0-25): Minimal risk, routine monitoring
- **MODERATE** (26-50): Some concern, consider intervention
- **HIGH** (51-75): Significant risk, intervention recommended
- **CRITICAL** (76-100): Severe risk, immediate action required

#### Response Structure

```json
{
  "risk_score": {
    "anonymized_id": "patient_001",
    "score": 68.5,
    "risk_level": "HIGH",
    "confidence": 0.85,
    "contributing_factors": ["phq9_score", "sleep_quality", "stress_level"],
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "recommendations": [
    {
      "resource_type": "THERAPY",
      "name": "Cognitive Behavioral Therapy",
      "description": "Evidence-based therapy for depression and anxiety",
      "urgency": "HIGH",
      "contact_info": "..."
    }
  ],
  "explanations": {
    "top_features": [["phq9_score", 0.35], ["sleep_quality", 0.22]],
    "counterfactual": "Improving sleep quality could reduce risk",
    "clinical_interpretation": "..."
  },
  "requires_human_review": true,
  "alert_triggered": false
}
```

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Authentication

Most endpoints require JWT authentication:
```
Authorization: Bearer <token>
```

### Core Endpoints

#### POST /screen
Perform risk screening for an individual.

**Request Body:**
```json
{
  "anonymized_id": "string",
  "consent_verified": true,
  "survey_data": {},
  "wearable_data": {},
  "emr_data": {}
}
```

**Response:** Risk score, recommendations, explanations

#### GET /risk-score/{anonymized_id}
Retrieve the most recent risk score.

#### POST /explain
Generate detailed explanation for a prediction.

#### GET /review-queue
Get pending cases in human review queue.

**Query Parameters:**
- `limit` (optional): Maximum number of cases

#### GET /statistics
Get system statistics and metrics.

#### POST /drift-check
Check for data and prediction drift.

#### GET /health
Health check endpoint.

#### GET /metrics
Prometheus metrics endpoint.

### Data Science Endpoints

#### Experiment Tracking

- `POST /api/v1/experiments` - Create experiment
- `GET /api/v1/experiments` - List experiments
- `GET /api/v1/experiments/{id}` - Get experiment details
- `POST /api/v1/experiments/{id}/runs` - Create run
- `GET /api/v1/runs/{id}` - Get run details
- `POST /api/v1/runs/{id}/metrics` - Log metrics
- `POST /api/v1/runs/{id}/params` - Log parameters
- `POST /api/v1/runs/{id}/artifacts` - Log artifacts
- `GET /api/v1/runs/search` - Search runs
- `GET /api/v1/runs/compare` - Compare runs

#### Data Versioning

- `POST /api/v1/datasets` - Register dataset
- `GET /api/v1/datasets/{name}/versions` - List versions
- `GET /api/v1/datasets/{name}/versions/{v}` - Get specific version
- `GET /api/v1/datasets/{id}/lineage` - Get lineage
- `POST /api/v1/datasets/drift` - Check drift

#### Feature Store

- `POST /api/v1/features` - Register feature
- `GET /api/v1/features` - List features
- `GET /api/v1/features/{name}` - Get feature details
- `POST /api/v1/features/compute` - Compute features
- `GET /api/v1/features/serve` - Serve features (online)

#### EDA & Reporting

- `POST /api/v1/eda/analyze` - Run EDA analysis
- `GET /api/v1/eda/reports/{id}` - Get EDA report
- `POST /api/v1/model-cards/generate` - Generate model card
- `GET /api/v1/model-cards/{id}` - Get model card

### Interactive Documentation

Full API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

For detailed data science API documentation, see [docs/ds_api_endpoints.md](docs/ds_api_endpoints.md)

---

## CLI Tool

### Database Management

```bash
# Run migrations
python run_cli.py db migrate

# Check connection
python run_cli.py db check
```

### Model Management

```bash
# List all models
python run_cli.py models list

# Activate a model
python run_cli.py models activate <model_id>

# Deactivate a model
python run_cli.py models deactivate <model_id>
```

### Review Queue Operations

```bash
# Show queue status
python run_cli.py review queue

# Check and escalate overdue cases
python run_cli.py review escalate
```

### Audit Reports

```bash
# Generate report for last 7 days
python run_cli.py audit report --days 7 --output report.json
```

### System Monitoring

```bash
# Check system health
python run_cli.py system health

# View statistics
python run_cli.py system stats

# Clear caches
python run_cli.py system clear-cache
```

### Direct Screening

```bash
python run_cli.py screen \
  --anonymized-id ID123 \
  --survey-data data.json \
  --wearable-data wearable.json
```

### Data Science Operations

#### Experiment Management

```bash
# List experiments
python run_cli.py experiments list

# Show experiment details
python run_cli.py experiments show <experiment_id>

# List runs
python run_cli.py runs list <experiment_id>

# Compare runs
python run_cli.py runs compare <run_id1> <run_id2>
```

#### Data Versioning

```bash
# List datasets
python run_cli.py datasets list

# Show dataset versions
python run_cli.py datasets versions <dataset_name>

# Show lineage
python run_cli.py datasets lineage <version_id>

# Check drift
python run_cli.py datasets drift <version_id1> <version_id2>
```

#### Feature Store

```bash
# List features
python run_cli.py features list

# Show feature details
python run_cli.py features show <feature_name>

# Register feature
python run_cli.py features register --config feature_config.json

# Compute features
python run_cli.py features compute \
  --features age_normalized,severity_score \
  --input data.csv \
  --output features.csv
```

#### EDA & Model Cards

```bash
# Run EDA analysis
python run_cli.py eda analyze patient_data.csv \
  --target risk_level \
  --output eda_report.html

# Generate model card
python run_cli.py model-card generate <model_id> \
  --output model_card.html

# Run hyperparameter optimization
python run_cli.py optimize --config optimization_config.json
```

---

## Configuration

### Environment Variables

```bash
# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=mhras
DATABASE_USER=postgres
DATABASE_PASSWORD=secret

# API
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key

# ML
MODEL_REGISTRY_PATH=models/
ALERT_THRESHOLD=75.0
HUMAN_REVIEW_THRESHOLD=75.0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Configuration Files

- `.env` - Environment variables (created from `.env.example`)
- `config/.env.example` - Example configuration with defaults
- `config/README.md` - Configuration documentation

### Database Documentation

- `DATABASE_QUICKSTART.md` - Quick database setup reference
- `CREDENTIALS.md` - Complete credential & security guide
- `docs/database_setup.md` - Detailed setup & troubleshooting

---

## Deployment

### Docker

```bash
# Build image
docker build -t mhras:latest .

# Run container
docker run -p 8000:8000 \
  -e API_SECRET_KEY=your-secret \
  -e DATABASE_URL=postgresql://... \
  mhras:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace mhras

# Apply configurations
kubectl apply -f k8s/ -n mhras

# Check status
kubectl get pods -n mhras

# Access service
kubectl port-forward svc/mhras-service 8000:8000 -n mhras
```

### Production Considerations

1. **Security**
   - Change default `API_SECRET_KEY`
   - Use strong database passwords
   - Enable HTTPS/TLS
   - Configure firewall rules
   - Set up proper authentication

2. **Performance**
   - Enable caching
   - Configure connection pooling
   - Use batch processing
   - Monitor and optimize bottlenecks

3. **Monitoring**
   - Set up Prometheus and Grafana
   - Configure alerts
   - Monitor metrics
   - Review audit logs

4. **Backup & Recovery**
   - Regular database backups
   - Model versioning
   - Disaster recovery plan

---

## Development

### Project Structure

```
mhras/
├── config/                 # Configuration files
├── src/
│   ├── api/               # REST API (7 files)
│   │   ├── app.py        # FastAPI application
│   │   ├── endpoints.py  # API endpoints
│   │   ├── ds_endpoints.py  # Data science endpoints
│   │   ├── auth.py       # Authentication
│   │   ├── middleware.py # Middleware
│   │   ├── models.py     # Pydantic models
│   │   └── ds_models.py  # DS Pydantic models
│   ├── ml/                # ML pipeline (13 files)
│   │   ├── model_registry.py
│   │   ├── inference_engine.py
│   │   ├── ensemble_predictor.py
│   │   ├── interpretability.py
│   │   ├── feature_pipeline.py
│   │   ├── baseline_models.py
│   │   ├── temporal_models.py
│   │   └── ...
│   ├── ds/                # Data science platform (11 files)
│   │   ├── experiment_tracker.py
│   │   ├── data_versioning.py
│   │   ├── feature_store.py
│   │   ├── eda.py
│   │   ├── model_cards.py
│   │   ├── notebook_integration.py
│   │   ├── ipython_magic.py
│   │   ├── hyperparameter_optimizer.py
│   │   ├── performance_monitor.py
│   │   ├── storage.py
│   │   ├── models.py
│   │   └── repositories.py
│   ├── governance/        # Governance (6 files)
│   │   ├── audit_logger.py
│   │   ├── human_review_queue.py
│   │   ├── drift_monitor.py
│   │   ├── consent.py
│   │   ├── anonymization.py
│   │   └── crisis_override.py
│   ├── processing/        # Data processing (5 files)
│   │   ├── etl_pipeline.py
│   │   ├── cleaning.py
│   │   ├── imputation.py
│   │   ├── normalization.py
│   │   └── encoding.py
│   ├── recommendations/   # Recommendations (4 files)
│   │   ├── recommendation_engine.py
│   │   ├── resource_catalog.py
│   │   └── populate_resources.py
│   ├── database/          # Database layer (7 files)
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── repositories.py
│   │   ├── migration_runner.py
│   │   └── migrations/
│   │       ├── 001_initial_schema.sql
│   │       ├── 002_resources_schema.sql
│   │       └── 003_data_science_schema.sql
│   ├── ingestion/         # Data ingestion (2 files)
│   │   ├── validation.py
│   │   └── schemas/
│   ├── integration.py     # Central integration
│   ├── screening_service.py  # End-to-end service
│   ├── cli.py             # CLI tool
│   ├── main.py            # Application entry
│   ├── config.py          # Configuration
│   ├── exceptions.py      # Custom exceptions
│   └── logging_config.py  # Logging setup
├── tests/                 # Test suite (30+ files)
│   ├── test_experiment_tracker.py
│   ├── test_data_versioning.py
│   ├── test_feature_store.py
│   ├── test_eda.py
│   ├── test_model_cards.py
│   ├── test_hyperparameter_optimizer.py
│   ├── test_performance_monitor.py
│   ├── test_ds_api.py
│   └── ...
├── docs/                  # Documentation (20+ files)
│   ├── experiment_tracking_usage.md
│   ├── data_versioning_usage.md
│   ├── feature_store_usage.md
│   ├── eda_usage.md
│   ├── ds_api_endpoints.md
│   └── ...
├── examples/              # Usage examples
│   ├── experiment_tracking_example.ipynb
│   ├── data_versioning_example.ipynb
│   ├── feature_engineering_example.ipynb
│   ├── hyperparameter_tuning_example.ipynb
│   └── screening_example.py
├── k8s/                   # Kubernetes manifests (8 files)
├── monitoring/            # Prometheus & Grafana configs
├── run_api.py             # API server launcher
├── run_cli.py             # CLI launcher
├── verify_integration.py  # Integration verification
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Adding New Components

1. Create component in appropriate directory
2. Add to `MHRASIntegration` initialization in `src/integration.py`
3. Update API endpoints if needed in `src/api/endpoints.py`
4. Add tests in `tests/`
5. Update documentation

### Code Style

- Follow PEP 8
- Use type hints
- Document with docstrings
- Add logging statements
- Write tests for new features

---

## Testing

### Run All Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_integration_complete.py -v

# Integration tests only
pytest tests/test_integration_complete.py -v
```

### Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - Component interaction testing
- **API Tests** - Endpoint testing
- **Performance Tests** - Load and stress testing

### Verification Script

```bash
# Verify complete integration
python verify_integration.py
```

---

## Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `mhras_http_requests_total` - Total HTTP requests
- `mhras_http_request_duration_seconds` - Request duration
- `mhras_screenings_total` - Total screenings by risk level
- `mhras_prediction_duration_seconds` - Prediction time
- `mhras_human_review_queue_size` - Review queue size
- `mhras_drift_score` - Drift scores by feature
- `mhras_errors_total` - Error counts by type

### Grafana Dashboards

Pre-configured dashboards in `monitoring/grafana/dashboards/`:
- `operations-dashboard.json` - System operations
- `ml-dashboard.json` - ML performance
- `clinical-dashboard.json` - Clinical metrics
- `compliance-dashboard.json` - Audit and compliance

### Setting Up Monitoring

1. Deploy Prometheus: `kubectl apply -f monitoring/k8s/prometheus-deployment.yaml`
2. Deploy Grafana: `kubectl apply -f monitoring/k8s/grafana-deployment.yaml`
3. Import dashboards from `monitoring/grafana/dashboards/`
4. Configure alerts in `monitoring/prometheus/alerts/`

---

## Security & Compliance

### Authentication & Authorization

- JWT-based authentication
- Token expiration and refresh
- Role-based access control (RBAC)
- API key support

### Data Protection

- PII anonymization
- Consent verification before data access
- Audit logging for all operations
- Encryption at rest and in transit

### Compliance Features

- **HIPAA-Ready**
  - Audit trails
  - Access controls
  - Data encryption
  - Consent management

- **GDPR Support**
  - Right to access
  - Right to erasure
  - Data portability
  - Consent tracking

### Security Best Practices

1. Change default `API_SECRET_KEY`
2. Use strong database passwords
3. Enable HTTPS/TLS in production
4. Configure firewall rules
5. Regular security audits
6. Keep dependencies updated
7. Monitor audit logs

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed

```bash
# Check environment variables
echo $DATABASE_URL

# Verify database is running
python run_cli.py db check

# Check logs
tail -f logs/app.log
```

#### 2. No Active Models

```bash
# List models
python run_cli.py models list

# Models should be auto-registered on first run
# If not, check logs for errors
```

#### 3. API Won't Start

```bash
# Check configuration
python -c "from src.config import settings; print(settings)"

# Verify port is available
lsof -i :8000

# Check logs
tail -f logs/app.log
```

#### 4. High Memory Usage

```bash
# Clear caches
python run_cli.py system clear-cache

# Restart service
```

#### 5. High Review Queue

```bash
# Check queue status
python run_cli.py review queue

# Escalate overdue cases
python run_cli.py review escalate
```

#### 6. Drift Detected

- Review drift report
- Retrain models with recent data
- Update reference distributions

### Getting Help

- Check documentation in `docs/` directory
- Review examples in `examples/` directory
- Check test cases for usage patterns
- Consult API documentation at `/docs`

### Data Science Platform Documentation

Comprehensive guides for each data science feature:

- **[Experiment Tracking Usage](docs/experiment_tracking_usage.md)** - Track ML experiments
- **[Data Versioning Usage](docs/data_versioning_usage.md)** - Version datasets and track lineage
- **[Feature Store Usage](docs/feature_store_usage.md)** - Centralized feature management
- **[EDA Usage](docs/eda_usage.md)** - Automated exploratory data analysis
- **[Data Science API Endpoints](docs/ds_api_endpoints.md)** - Complete API reference

Example notebooks in `examples/`:
- `experiment_tracking_example.ipynb` - Complete experiment tracking workflow
- `data_versioning_example.ipynb` - Dataset versioning and drift detection
- `feature_engineering_example.ipynb` - Feature store usage
- `hyperparameter_tuning_example.ipynb` - Automated hyperparameter optimization

---

## Technology Stack

### Backend

- **Python 3.9+** - Core language
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **structlog** - Structured logging

### Machine Learning

- **scikit-learn** - Classical ML algorithms
- **LightGBM** - Gradient boosting
- **PyTorch** - Deep learning
- **SHAP** - Model interpretability
- **imbalanced-learn** - Handling class imbalance
- **Optuna** - Hyperparameter optimization

### Data Processing

- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **plotly** - Interactive visualizations

### NLP

- **transformers** - Pre-trained models
- **vaderSentiment** - Sentiment analysis

### Database

- **PostgreSQL** - Primary database
- **psycopg2** - PostgreSQL adapter
- **SQLAlchemy** - ORM
- **Redis** - Feature store caching (optional)

### Monitoring

- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **prometheus-client** - Python client

### Security

- **python-jose** - JWT handling
- **passlib** - Password hashing
- **cryptography** - Encryption

### Development

- **pytest** - Testing framework
- **pytest-asyncio** - Async testing
- **pytest-cov** - Coverage reporting
- **click** - CLI framework

---

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

### Code Standards

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions/classes
- Maintain test coverage >80%
- Add logging for important operations

### Testing Requirements

- All new features must have tests
- Integration tests for component interactions
- API tests for new endpoints
- Performance tests for critical paths

### Documentation

- Update README for major changes
- Add docstrings to new code
- Update API documentation
- Add examples for new features

---

## License

[Add your license here]

---

## Acknowledgments

This system was designed and implemented following best practices for:
- Machine learning in healthcare
- Privacy-preserving data processing
- Interpretable AI systems
- Production ML systems
- Clinical decision support

---

## Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check documentation in `docs/`
- Review examples in `examples/`
- Consult API docs at `/docs` endpoint

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Status:** Production Ready ✅
