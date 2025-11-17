# Data Science API Endpoints

This document describes the API endpoints for data science features in the Mental Health Risk Assessment System.

## Overview

The data science API endpoints provide programmatic access to:
- Experiment tracking
- Data versioning
- Feature store
- Exploratory data analysis (EDA)
- Model card generation

All endpoints require authentication via Bearer token.

## Base URL

```
/api/v1
```

## Experiment Tracking Endpoints

### Create Experiment
```
POST /api/v1/experiments
```

Create a new experiment for tracking ML runs.

**Request Body:**
```json
{
  "experiment_name": "depression_risk_model_v2",
  "description": "Testing new feature engineering approach",
  "tags": {"team": "ml", "priority": "high"}
}
```

**Response:** `201 Created`
```json
{
  "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
  "experiment_name": "depression_risk_model_v2",
  "description": "Testing new feature engineering approach",
  "tags": {"team": "ml", "priority": "high"},
  "created_at": "2025-11-17T10:30:00Z"
}
```

### List Experiments
```
GET /api/v1/experiments?limit=100
```

List all experiments.

**Response:** `200 OK`
```json
[
  {
    "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
    "experiment_name": "depression_risk_model_v2",
    "description": "Testing new feature engineering approach",
    "tags": {"team": "ml"},
    "created_at": "2025-11-17T10:30:00Z"
  }
]
```

### Get Experiment
```
GET /api/v1/experiments/{experiment_id}
```

Get experiment details by ID.

### Create Run
```
POST /api/v1/experiments/{experiment_id}/runs
```

Create a new run within an experiment.

**Request Body:**
```json
{
  "run_name": "baseline_model",
  "tags": {"model_type": "random_forest", "version": "1.0"}
}
```

**Response:** `201 Created`
```json
{
  "run_id": "660e8400-e29b-41d4-a716-446655440000",
  "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
  "run_name": "baseline_model",
  "status": "RUNNING",
  "start_time": "2025-11-17T10:30:00Z",
  "end_time": null,
  "params": {},
  "tags": {"model_type": "random_forest"},
  "git_commit": "abc123def456"
}
```

### Log Metrics
```
POST /api/v1/runs/{run_id}/metrics
```

Log metrics for a specific run.

**Request Body:**
```json
{
  "metrics": {"accuracy": 0.85, "f1_score": 0.82},
  "step": 100
}
```

**Response:** `200 OK`
```json
{
  "message": "Metrics logged successfully",
  "run_id": "660e8400-e29b-41d4-a716-446655440000"
}
```

### Log Parameters
```
POST /api/v1/runs/{run_id}/params
```

Log parameters for a specific run.

**Request Body:**
```json
{
  "params": {
    "learning_rate": 0.01,
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

### Log Artifact
```
POST /api/v1/runs/{run_id}/artifacts
```

Log an artifact for a specific run.

**Request Body:**
```json
{
  "artifact_type": "model",
  "artifact_path": "/tmp/model.pkl",
  "metadata": {"format": "pickle", "size_mb": 15.2}
}
```

### Search Runs
```
POST /api/v1/runs/search
```

Search and filter runs.

**Request Body:**
```json
{
  "experiment_name": "depression_risk_model_v2",
  "filter_string": "metrics.accuracy > 0.8",
  "order_by": ["metrics.accuracy DESC"],
  "limit": 50
}
```

### Compare Runs
```
POST /api/v1/runs/compare
```

Compare metrics across multiple runs.

**Request Body:**
```json
{
  "run_ids": [
    "660e8400-e29b-41d4-a716-446655440000",
    "770e8400-e29b-41d4-a716-446655440000"
  ],
  "metric_names": ["accuracy", "f1_score", "auc"]
}
```

## Data Versioning Endpoints

### Register Dataset
```
POST /api/v1/datasets
```

Register a new dataset version.

**Request Body:**
```json
{
  "dataset_name": "training_data_v1",
  "source": "emr_system",
  "data": {
    "columns": ["feature1", "feature2", "target"],
    "values": [[1.0, 2.0, 0], [1.5, 2.5, 1]]
  },
  "metadata": {"description": "Initial training dataset"}
}
```

**Response:** `201 Created`
```json
{
  "version_id": "880e8400-e29b-41d4-a716-446655440000",
  "dataset_name": "training_data_v1",
  "version": "v1.0.0",
  "source": "emr_system",
  "num_rows": 10000,
  "num_columns": 25,
  "schema": {"feature1": "float64", "feature2": "float64"},
  "created_at": "2025-11-17T10:30:00Z"
}
```

### List Dataset Versions
```
GET /api/v1/datasets/{dataset_name}/versions
```

List all versions of a dataset.

### Get Dataset Lineage
```
GET /api/v1/datasets/{version_id}/lineage?direction=upstream
```

Get dataset lineage (upstream or downstream).

**Query Parameters:**
- `direction`: Either "upstream" or "downstream"

### Check Data Drift
```
POST /api/v1/datasets/drift
```

Check for data drift between two dataset versions.

**Request Body:**
```json
{
  "dataset_version_id1": "880e8400-e29b-41d4-a716-446655440000",
  "dataset_version_id2": "990e8400-e29b-41d4-a716-446655440000"
}
```

**Response:** `200 OK`
```json
{
  "version_id1": "880e8400-e29b-41d4-a716-446655440000",
  "version_id2": "990e8400-e29b-41d4-a716-446655440000",
  "drift_detected": true,
  "drift_score": 0.35,
  "feature_drifts": {"feature1": 0.45, "feature2": 0.25},
  "recommendations": ["Consider retraining the model"]
}
```

## Feature Store Endpoints

### Register Feature
```
POST /api/v1/features
```

Register a new feature in the feature store.

**Request Body:**
```json
{
  "feature_name": "phq9_severity",
  "feature_type": "categorical",
  "description": "PHQ-9 severity category",
  "transformation_code": "lambda x: 'severe' if x > 20 else 'moderate' if x > 10 else 'mild'",
  "input_schema": {"phq9_score": "int"},
  "output_schema": {"phq9_severity": "str"},
  "dependencies": []
}
```

**Response:** `201 Created`
```json
{
  "feature_id": "aa0e8400-e29b-41d4-a716-446655440000",
  "feature_name": "phq9_severity",
  "feature_type": "categorical",
  "description": "PHQ-9 severity category",
  "version": "1.0",
  "created_at": "2025-11-17T10:30:00Z"
}
```

### List Features
```
GET /api/v1/features?limit=100
```

List all features in the feature store.

### Get Feature
```
GET /api/v1/features/{feature_name}
```

Get feature details by name.

### Compute Features
```
POST /api/v1/features/compute
```

Compute features from input data.

**Request Body:**
```json
{
  "feature_names": ["phq9_severity", "sleep_quality_score"],
  "input_data": {
    "columns": ["phq9_score", "sleep_hours"],
    "values": [[15, 6.5], [22, 4.0]]
  }
}
```

### Serve Features
```
GET /api/v1/features/serve?feature_names=phq9_severity,sleep_quality_score&entity_ids=patient_001,patient_002
```

Serve features for online inference.

**Query Parameters:**
- `feature_names`: Comma-separated list of feature names
- `entity_ids`: Comma-separated list of entity IDs

## EDA and Reporting Endpoints

### Analyze Dataset
```
POST /api/v1/eda/analyze
```

Run exploratory data analysis on a dataset.

**Request Body:**
```json
{
  "data": {
    "columns": ["feature1", "feature2", "target"],
    "values": [[1.0, 2.0, 0], [1.5, 2.5, 1]]
  },
  "target_column": "target",
  "dataset_name": "training_data"
}
```

**Response:** `200 OK`
```json
{
  "report_id": "bb0e8400-e29b-41d4-a716-446655440000",
  "dataset_name": "training_data",
  "num_rows": 10000,
  "num_columns": 25,
  "summary_statistics": {"feature1": {"mean": 1.5, "std": 0.5}},
  "quality_issues": [
    {
      "issue_type": "missing",
      "severity": "medium",
      "description": "10% missing values in feature2"
    }
  ],
  "recommendations": ["Impute missing values in feature2"],
  "generated_at": "2025-11-17T10:30:00Z"
}
```

### Get EDA Report
```
GET /api/v1/eda/reports/{report_id}
```

Get a previously generated EDA report.

**Note:** Reports are not persisted in the current version.

### Generate Model Card
```
POST /api/v1/model-cards/generate
```

Generate a model card for a trained model.

**Request Body:**
```json
{
  "model_id": "model_depression_v2",
  "run_id": "660e8400-e29b-41d4-a716-446655440000",
  "include_fairness": true
}
```

**Response:** `200 OK`
```json
{
  "card_id": "cc0e8400-e29b-41d4-a716-446655440000",
  "model_id": "model_depression_v2",
  "model_name": "Depression Risk Predictor",
  "model_type": "RandomForest",
  "version": "2.0",
  "metrics": {"accuracy": 0.85, "f1_score": 0.82},
  "fairness_metrics": {"demographic_parity": 0.95},
  "feature_importance": [["phq9_score", 0.35], ["sleep_hours", 0.25]],
  "generated_at": "2025-11-17T10:30:00Z"
}
```

### Get Model Card
```
GET /api/v1/model-cards/{card_id}
```

Get a previously generated model card.

**Note:** Model cards are not persisted in the current version.

## Authentication

All endpoints require authentication using a Bearer token:

```
Authorization: Bearer <token>
```

## Error Responses

All endpoints may return the following error responses:

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service not initialized

Error response format:
```json
{
  "error": "ValidationError",
  "message": "Invalid request data",
  "details": {},
  "timestamp": "2025-11-17T10:30:00Z"
}
```

## Testing

API integration tests are available in `tests/test_ds_api.py`.

Run tests with:
```bash
pytest tests/test_ds_api.py -v
```
