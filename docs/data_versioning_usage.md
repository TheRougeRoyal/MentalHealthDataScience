# Data Versioning Usage Guide

## Overview

The Data Version Control System provides versioning, lineage tracking, and drift detection for datasets used in machine learning pipelines. It ensures reproducibility by maintaining snapshots of data at different stages and tracking transformations.

## Quick Start

```python
from src.ds.data_versioning import DataVersionControl
from src.ds.storage import FileSystemStorage
from src.database.connection import get_db_connection
import pandas as pd

# Initialize the system
storage = FileSystemStorage(base_path="data/versions")
db = get_db_connection()
dvc = DataVersionControl(storage_backend=storage, db_connection=db)

# Register a dataset
df = pd.read_csv("raw_data.csv")
version = dvc.register_dataset(
    dataset=df,
    dataset_name="patient_assessments",
    source="emr_system",
    metadata={"collection_date": "2024-01-15"}
)

print(f"Registered version: {version.version}")
print(f"Version ID: {version.version_id}")
```

## Core Concepts

### Dataset Versions

Each dataset registration creates an immutable version with metadata:

```python
# Register initial dataset
v1 = dvc.register_dataset(
    dataset=raw_df,
    dataset_name="assessments",
    source="database_export",
    metadata={"export_date": "2024-01-01"}
)

# Register processed version
processed_df = clean_data(raw_df)
v2 = dvc.register_dataset(
    dataset=processed_df,
    dataset_name="assessments",
    source="cleaning_pipeline",
    metadata={"parent_version": v1.version_id}
)
```

### Version Information

Each version contains comprehensive metadata:

```python
version = dvc.get_dataset("assessments", version="v1.0")[1]

print(f"Version ID: {version.version_id}")
print(f"Dataset Name: {version.dataset_name}")
print(f"Version: {version.version}")
print(f"Source: {version.source}")
print(f"Rows: {version.num_rows}")
print(f"Columns: {version.num_columns}")
print(f"Schema: {version.schema}")
print(f"Statistics: {version.statistics}")
print(f"Created: {version.created_at}")
```

## Registering Datasets

### Basic Registration

```python
# Register with automatic versioning
version = dvc.register_dataset(
    dataset=df,
    dataset_name="patient_data",
    source="emr_export"
)
```

### With Metadata

```python
# Include custom metadata
version = dvc.register_dataset(
    dataset=df,
    dataset_name="patient_data",
    source="emr_export",
    metadata={
        "collection_period": "2024-Q1",
        "num_patients": 1500,
        "data_quality_score": 0.95,
        "preprocessing_steps": ["deduplication", "imputation"]
    }
)
```

### Automatic Statistics

The system automatically computes statistics:

```python
version = dvc.register_dataset(df, "assessments", "export")

# Access computed statistics
stats = version.statistics
print(f"Numerical features: {stats['numerical_features']}")
print(f"Categorical features: {stats['categorical_features']}")
print(f"Missing values: {stats['missing_values']}")
print(f"Duplicates: {stats['duplicate_rows']}")
```

## Retrieving Datasets

### Get Latest Version

```python
# Retrieve the most recent version
df, version = dvc.get_dataset("patient_data")
print(f"Retrieved version: {version.version}")
```

### Get Specific Version

```python
# Retrieve by version string
df, version = dvc.get_dataset("patient_data", version="v1.2")

# Retrieve by version ID
df, version = dvc.get_dataset("patient_data", version=version_id)
```

### List All Versions

```python
# Get all versions of a dataset
versions = dvc.list_versions("patient_data")

for v in versions:
    print(f"{v.version} - {v.created_at} - {v.num_rows} rows")
```

## Data Lineage Tracking

### Track Transformations

```python
# Register source dataset
raw_version = dvc.register_dataset(
    dataset=raw_df,
    dataset_name="assessments",
    source="database"
)

# Apply transformation
cleaned_df = clean_missing_values(raw_df)

# Register transformed dataset
cleaned_version = dvc.register_dataset(
    dataset=cleaned_df,
    dataset_name="assessments_cleaned",
    source="cleaning_pipeline"
)

# Track the transformation
dvc.track_transformation(
    input_version_id=raw_version.version_id,
    output_version_id=cleaned_version.version_id,
    transformation_code="""
    def clean_missing_values(df):
        df = df.dropna(subset=['patient_id'])
        df['age'].fillna(df['age'].median(), inplace=True)
        return df
    """,
    transformation_type="cleaning"
)
```

### Query Lineage

```python
# Get upstream lineage (what data was this derived from?)
upstream = dvc.get_lineage(
    dataset_version_id=cleaned_version.version_id,
    direction="upstream"
)

for ancestor in upstream:
    print(f"Derived from: {ancestor.dataset_name} v{ancestor.version}")

# Get downstream lineage (what was derived from this data?)
downstream = dvc.get_lineage(
    dataset_version_id=raw_version.version_id,
    direction="downstream"
)

for descendant in downstream:
    print(f"Used to create: {descendant.dataset_name} v{descendant.version}")
```

### Lineage Visualization

```python
# Get full lineage graph
lineage_graph = dvc.get_lineage_graph(dataset_version_id)

# Visualize (requires graphviz)
from src.ds.data_versioning import visualize_lineage
visualize_lineage(lineage_graph, output_path="lineage.png")
```

## Drift Detection

### Detect Drift Between Versions

```python
# Compare two versions
drift_report = dvc.detect_drift(
    dataset_version_id1=v1.version_id,
    dataset_version_id2=v2.version_id
)

print(f"Drift detected: {drift_report.drift_detected}")
print(f"Overall drift score: {drift_report.drift_score}")
print(f"Feature drifts: {drift_report.feature_drifts}")
```

### Interpret Drift Results

```python
drift_report = dvc.detect_drift(v1.version_id, v2.version_id)

# Check overall drift
if drift_report.drift_detected:
    print(f"⚠️ Significant drift detected (score: {drift_report.drift_score:.3f})")
else:
    print("✓ No significant drift detected")

# Check feature-level drift
for feature, score in drift_report.feature_drifts.items():
    if score > 0.1:  # Threshold for significant drift
        print(f"  - {feature}: {score:.3f} (HIGH)")
    else:
        print(f"  - {feature}: {score:.3f} (low)")

# View statistical tests
for feature, test_result in drift_report.statistical_tests.items():
    print(f"{feature}:")
    print(f"  Test: {test_result['test_name']}")
    print(f"  Statistic: {test_result['statistic']:.4f}")
    print(f"  P-value: {test_result['p_value']:.4f}")

# Get recommendations
for rec in drift_report.recommendations:
    print(f"💡 {rec}")
```

### Continuous Drift Monitoring

```python
# Monitor drift against production baseline
baseline_version = dvc.get_dataset("production_data", version="v1.0")[1]

# Check new data periodically
new_df = fetch_recent_data()
new_version = dvc.register_dataset(new_df, "production_data", "live")

drift_report = dvc.detect_drift(
    baseline_version.version_id,
    new_version.version_id
)

if drift_report.drift_detected:
    # Trigger retraining or alert
    send_alert(f"Data drift detected: {drift_report.drift_score}")
```

## Integration with Experiment Tracking

```python
from src.ds.experiment_tracker import ExperimentTracker

# Register dataset version
version = dvc.register_dataset(df, "training_data", "pipeline")

# Track in experiment
tracker = ExperimentTracker(storage, db)
run = tracker.start_run(experiment_name="risk_model")

# Log dataset version
tracker.log_params({
    "dataset_name": version.dataset_name,
    "dataset_version": version.version,
    "dataset_version_id": str(version.version_id),
    "num_samples": version.num_rows
})

# Train model...
tracker.end_run()
```

## Best Practices

### 1. Version at Key Pipeline Stages

```python
# Raw data
raw_v = dvc.register_dataset(raw_df, "assessments", "database_export")

# After cleaning
clean_v = dvc.register_dataset(clean_df, "assessments", "cleaning")
dvc.track_transformation(raw_v.version_id, clean_v.version_id, 
                        cleaning_code, "cleaning")

# After feature engineering
features_v = dvc.register_dataset(features_df, "assessments", "features")
dvc.track_transformation(clean_v.version_id, features_v.version_id,
                        feature_code, "feature_engineering")

# Training/test splits
train_v = dvc.register_dataset(train_df, "assessments", "train_split")
test_v = dvc.register_dataset(test_df, "assessments", "test_split")
```

### 2. Use Descriptive Metadata

```python
version = dvc.register_dataset(
    dataset=df,
    dataset_name="patient_assessments",
    source="emr_system",
    metadata={
        "collection_start": "2024-01-01",
        "collection_end": "2024-03-31",
        "num_facilities": 15,
        "inclusion_criteria": "adults_18+",
        "exclusion_criteria": "incomplete_assessments",
        "data_quality_checks": ["deduplication", "validation"],
        "responsible_team": "data_engineering"
    }
)
```

### 3. Track Transformations

```python
# Always track how data was transformed
dvc.track_transformation(
    input_version_id=input_v.version_id,
    output_version_id=output_v.version_id,
    transformation_code=inspect.getsource(transform_function),
    transformation_type="feature_engineering"
)
```

### 4. Monitor Drift Regularly

```python
# Set up periodic drift checks
def check_drift_weekly():
    baseline = dvc.get_dataset("production_data", version="baseline")[1]
    current = dvc.get_dataset("production_data")[1]  # Latest
    
    drift = dvc.detect_drift(baseline.version_id, current.version_id)
    
    if drift.drift_score > 0.15:
        alert_team(f"High drift detected: {drift.drift_score}")
        recommend_retraining()
```

### 5. Use Semantic Versioning

```python
# Use meaningful version numbers
# v1.0 - Initial production dataset
# v1.1 - Minor data additions
# v2.0 - Major schema changes or reprocessing
```

## CLI Commands

### List Datasets

```bash
python run_cli.py datasets list
```

### Show Dataset Versions

```bash
python run_cli.py datasets versions patient_assessments
```

### Show Lineage

```bash
python run_cli.py datasets lineage <version_id>
```

### Check Drift

```bash
python run_cli.py datasets drift <version_id_1> <version_id_2>
```

## API Endpoints

### Register Dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "patient_assessments",
    "source": "emr_export",
    "data": {...},
    "metadata": {"collection_date": "2024-01-15"}
  }'
```

### Get Dataset Versions

```bash
curl -X GET http://localhost:8000/api/v1/datasets/patient_assessments/versions
```

### Get Specific Version

```bash
curl -X GET http://localhost:8000/api/v1/datasets/patient_assessments/versions/v1.2
```

### Get Lineage

```bash
curl -X GET http://localhost:8000/api/v1/datasets/{version_id}/lineage?direction=upstream
```

### Check Drift

```bash
curl -X POST http://localhost:8000/api/v1/datasets/drift \
  -H "Content-Type: application/json" \
  -d '{
    "version_id1": "uuid-1",
    "version_id2": "uuid-2"
  }'
```

## Troubleshooting

### Issue: Large datasets slow to register

```python
# Use compression
dvc = DataVersionControl(
    storage_backend=storage,
    db_connection=db,
    compression="gzip"
)
```

### Issue: Storage space concerns

```python
# Enable deduplication
dvc = DataVersionControl(
    storage_backend=storage,
    db_connection=db,
    deduplication=True
)

# Archive old versions
dvc.archive_versions(dataset_name="old_data", keep_recent=5)
```

### Issue: Drift false positives

```python
# Adjust drift threshold
drift_report = dvc.detect_drift(v1, v2, threshold=0.2)  # Less sensitive

# Focus on specific features
drift_report = dvc.detect_drift(
    v1, v2,
    features=["age", "severity_score", "diagnosis"]
)
```

## Advanced Usage

### Custom Hash Functions

```python
# Use custom hashing for deduplication
def custom_hash(df):
    return hashlib.sha256(df.to_json().encode()).hexdigest()

dvc = DataVersionControl(
    storage_backend=storage,
    db_connection=db,
    hash_function=custom_hash
)
```

### Parallel Processing

```python
# Process large datasets in chunks
for chunk in pd.read_csv("large_file.csv", chunksize=10000):
    version = dvc.register_dataset(
        dataset=chunk,
        dataset_name="large_dataset",
        source="streaming"
    )
```

## See Also

- [Experiment Tracking Usage](experiment_tracking_usage.md)
- [Feature Store Usage](feature_store_usage.md)
- [EDA Usage](eda_usage.md)
