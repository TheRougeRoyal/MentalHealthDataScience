# Experiment Tracking Usage Guide

## Overview

The Experiment Tracker provides a comprehensive system for logging, tracking, and comparing machine learning experiments. It captures hyperparameters, metrics, artifacts, and code versions to ensure reproducibility and enable systematic model development.

## Quick Start

```python
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.storage import FileSystemStorage
from src.database.connection import get_db_connection

# Initialize the tracker
storage = FileSystemStorage(base_path="experiments/artifacts")
db = get_db_connection()
tracker = ExperimentTracker(storage_backend=storage, db_connection=db)

# Start a new run
run = tracker.start_run(
    experiment_name="mental_health_risk_model",
    run_name="baseline_v1",
    tags={"model_type": "random_forest", "dataset": "v1.2"}
)

# Log hyperparameters
tracker.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.01
})

# Log metrics during training
for epoch in range(10):
    accuracy = train_epoch()
    tracker.log_metrics({"accuracy": accuracy, "loss": loss}, step=epoch)

# Log artifacts
tracker.log_artifact("model.pkl", artifact_type="model")
tracker.log_artifact("confusion_matrix.png", artifact_type="plot")

# End the run
tracker.end_run(status="FINISHED")
```

## Core Concepts

### Experiments

An experiment is a collection of related runs that explore a specific hypothesis or model architecture.

```python
# Experiments are created automatically when you start a run
# All runs with the same experiment_name are grouped together
run1 = tracker.start_run(experiment_name="risk_prediction")
run2 = tracker.start_run(experiment_name="risk_prediction")
```

### Runs

A run represents a single execution of your training pipeline with specific parameters.

```python
# Start a run with metadata
run = tracker.start_run(
    experiment_name="risk_prediction",
    run_name="xgboost_tuned",
    tags={
        "model": "xgboost",
        "feature_set": "v2",
        "author": "data_scientist_1"
    }
)

print(f"Run ID: {run.run_id}")
print(f"Status: {run.status}")
```

### Parameters

Log hyperparameters and configuration values:

```python
# Log individual parameters
tracker.log_params({
    "model_type": "xgboost",
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8
})

# Parameters are immutable once logged
```

### Metrics

Track performance metrics over time:

```python
# Log metrics at specific steps
tracker.log_metrics({"train_accuracy": 0.85}, step=0)
tracker.log_metrics({"train_accuracy": 0.89}, step=1)
tracker.log_metrics({"train_accuracy": 0.92}, step=2)

# Log multiple metrics at once
tracker.log_metrics({
    "train_accuracy": 0.92,
    "val_accuracy": 0.88,
    "train_loss": 0.15,
    "val_loss": 0.22
}, step=3)

# Log final metrics without step
tracker.log_metrics({
    "final_accuracy": 0.91,
    "final_f1": 0.89,
    "final_auc": 0.94
})
```

### Artifacts

Store files associated with your run:

```python
# Log model files
tracker.log_artifact("models/model.pkl", artifact_type="model")

# Log visualizations
tracker.log_artifact("plots/roc_curve.png", artifact_type="plot")
tracker.log_artifact("plots/feature_importance.png", artifact_type="plot")

# Log data files
tracker.log_artifact("data/predictions.csv", artifact_type="data")

# Log reports
tracker.log_artifact("reports/model_card.html", artifact_type="report")
```

## Querying Experiments

### Get a Specific Run

```python
# Retrieve run by ID
run = tracker.get_run(run_id="abc-123-def")

print(f"Run Name: {run.run_name}")
print(f"Parameters: {run.params}")
print(f"Metrics: {run.metrics}")
print(f"Artifacts: {run.artifacts}")
```

### Search Runs

```python
# Search all runs in an experiment
runs = tracker.search_runs(experiment_name="risk_prediction")

# Filter by metrics
runs = tracker.search_runs(
    experiment_name="risk_prediction",
    filter_string="metrics.accuracy > 0.9"
)

# Order by metric
runs = tracker.search_runs(
    experiment_name="risk_prediction",
    order_by=["metrics.accuracy DESC"]
)

# Complex filtering
runs = tracker.search_runs(
    experiment_name="risk_prediction",
    filter_string="params.model_type = 'xgboost' AND metrics.accuracy > 0.85",
    order_by=["metrics.f1_score DESC", "start_time DESC"]
)
```

### Compare Runs

```python
# Compare multiple runs
comparison = tracker.compare_runs(
    run_ids=["run1", "run2", "run3"],
    metric_names=["accuracy", "f1_score", "auc"]
)

print(comparison)
# Output: DataFrame with runs as rows and metrics as columns
```

## Integration with Model Registry

Link experiments to deployed models:

```python
from src.ml.model_registry import ModelRegistry

# Train and track experiment
run = tracker.start_run(experiment_name="production_model")
tracker.log_params({"n_estimators": 150})
model = train_model()
tracker.log_metrics({"accuracy": 0.93})
tracker.log_artifact("model.pkl", artifact_type="model")
tracker.end_run()

# Register model with experiment link
registry = ModelRegistry(db_connection=db)
registry.register_model(
    model_name="risk_predictor_v2",
    model_path="model.pkl",
    metadata={
        "experiment_id": run.experiment_id,
        "run_id": run.run_id,
        "accuracy": 0.93
    }
)
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
run = tracker.start_run(
    experiment_name="suicide_risk_lstm_baseline",
    run_name="lstm_128_units_dropout_0.3"
)

# Avoid
run = tracker.start_run(
    experiment_name="exp1",
    run_name="run1"
)
```

### 2. Tag Your Runs

```python
tracker.start_run(
    experiment_name="risk_prediction",
    tags={
        "model_family": "tree_based",
        "feature_version": "v2.1",
        "data_split": "temporal",
        "purpose": "hyperparameter_tuning"
    }
)
```

### 3. Log Incrementally

```python
# Log metrics as they become available
for epoch in range(num_epochs):
    train_metrics = train_epoch()
    tracker.log_metrics(train_metrics, step=epoch)
    
    if epoch % 5 == 0:
        val_metrics = validate()
        tracker.log_metrics(val_metrics, step=epoch)
```

### 4. Handle Failures Gracefully

```python
try:
    run = tracker.start_run(experiment_name="risk_model")
    tracker.log_params(params)
    
    model = train_model()
    tracker.log_metrics({"accuracy": accuracy})
    tracker.log_artifact("model.pkl")
    
    tracker.end_run(status="FINISHED")
except Exception as e:
    tracker.end_run(status="FAILED")
    raise
```

### 5. Use Context Managers

```python
from contextlib import contextmanager

@contextmanager
def tracked_run(tracker, experiment_name, **kwargs):
    run = tracker.start_run(experiment_name, **kwargs)
    try:
        yield tracker
        tracker.end_run(status="FINISHED")
    except Exception as e:
        tracker.end_run(status="FAILED")
        raise

# Usage
with tracked_run(tracker, "risk_model", run_name="test") as t:
    t.log_params({"lr": 0.01})
    model = train_model()
    t.log_metrics({"accuracy": 0.9})
```

## CLI Commands

### List Experiments

```bash
python run_cli.py experiments list
```

### Show Experiment Details

```bash
python run_cli.py experiments show <experiment_id>
```

### List Runs

```bash
python run_cli.py runs list <experiment_id>
```

### Compare Runs

```bash
python run_cli.py runs compare <run_id1> <run_id2>
```

## API Endpoints

### Create Experiment

```bash
curl -X POST http://localhost:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "risk_prediction",
    "description": "Suicide risk prediction models",
    "tags": {"project": "mental_health"}
  }'
```

### Create Run

```bash
curl -X POST http://localhost:8000/api/v1/experiments/{experiment_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "baseline_v1",
    "tags": {"model": "random_forest"}
  }'
```

### Log Metrics

```bash
curl -X POST http://localhost:8000/api/v1/runs/{run_id}/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "accuracy",
    "metric_value": 0.92,
    "step": 10
  }'
```

### Search Runs

```bash
curl -X GET "http://localhost:8000/api/v1/runs/search?experiment_name=risk_prediction&filter=metrics.accuracy>0.9"
```

## Troubleshooting

### Issue: Run not found

```python
# Ensure you're using the correct run_id
run = tracker.get_run(run_id="correct-uuid-here")
```

### Issue: Artifact storage fails

```python
# Check storage backend configuration
storage = FileSystemStorage(base_path="experiments/artifacts")
# Ensure directory exists and has write permissions
```

### Issue: Metrics not appearing

```python
# Ensure you're logging metrics before ending the run
tracker.log_metrics({"accuracy": 0.9})
tracker.end_run()  # Metrics are committed here
```

## Advanced Usage

### Custom Storage Backend

```python
from src.ds.storage import StorageBackend

class S3Storage(StorageBackend):
    def save_artifact(self, artifact: bytes, path: str) -> str:
        # Upload to S3
        s3_uri = upload_to_s3(artifact, path)
        return s3_uri
    
    def load_artifact(self, uri: str) -> bytes:
        # Download from S3
        return download_from_s3(uri)

# Use custom storage
tracker = ExperimentTracker(
    storage_backend=S3Storage(bucket="my-experiments"),
    db_connection=db
)
```

### Automatic Git Tracking

The tracker automatically captures git commit information:

```python
run = tracker.start_run(experiment_name="risk_model")
# run.git_commit contains the current git commit hash
# run.code_version contains version information
```

## See Also

- [Data Versioning Usage](data_versioning_usage.md)
- [Feature Store Usage](feature_store_usage.md)
- [Model Cards Usage](model_cards_usage.md)
- [Hyperparameter Optimization](hyperparameter_optimization_usage.md)
