# Feature Store Usage Guide

## Overview

The Feature Store provides a centralized repository for storing, versioning, and serving feature engineering transformations. It ensures consistency between training and inference, enables feature reuse across projects, and supports both online and batch serving.

## Quick Start

```python
from src.ds.feature_store import FeatureStore, FeatureDefinition
from src.database.connection import get_db_connection
import pandas as pd

# Initialize the feature store
db = get_db_connection()
feature_store = FeatureStore(
    db_connection=db,
    cache_backend="redis",  # Optional: for online serving
    cache_url="redis://localhost:6379"
)

# Register a feature
feature_def = FeatureDefinition(
    feature_name="age_risk_score",
    feature_type="numeric",
    description="Age-based risk score for mental health assessment",
    transformation_code="""
def transform(df):
    return (df['age'] - 18) / 82 * 10
""",
    input_schema={"age": "int"},
    output_schema={"age_risk_score": "float"},
    version="v1.0",
    dependencies=[],
    owner="data_science_team"
)

feature_store.register_feature("age_risk_score", feature_def)

# Compute features
input_data = pd.DataFrame({"age": [25, 45, 67]})
features = feature_store.compute_features(
    feature_names=["age_risk_score"],
    input_data=input_data
)
print(features)
```

## Core Concepts

### Feature Definitions

A feature definition specifies how to compute a feature from raw data:

```python
feature_def = FeatureDefinition(
    feature_name="severity_score",
    feature_type="numeric",
    description="Composite severity score from multiple assessments",
    transformation_code="""
def transform(df):
    weights = {'phq9': 0.4, 'gad7': 0.3, 'pcl5': 0.3}
    score = (
        df['phq9_score'] * weights['phq9'] +
        df['gad7_score'] * weights['gad7'] +
        df['pcl5_score'] * weights['pcl5']
    )
    return score / 27 * 100  # Normalize to 0-100
""",
    input_schema={
        "phq9_score": "int",
        "gad7_score": "int",
        "pcl5_score": "int"
    },
    output_schema={"severity_score": "float"},
    version="v1.0",
    dependencies=[],
    owner="clinical_team"
)
```

### Feature Types

Supported feature types:

- **numeric**: Continuous numerical values
- **categorical**: Discrete categories
- **embedding**: Dense vector representations
- **boolean**: Binary flags
- **text**: Text features

```python
# Categorical feature
category_feature = FeatureDefinition(
    feature_name="risk_category",
    feature_type="categorical",
    transformation_code="""
def transform(df):
    return pd.cut(df['risk_score'], 
                  bins=[0, 30, 60, 100],
                  labels=['low', 'medium', 'high'])
""",
    input_schema={"risk_score": "float"},
    output_schema={"risk_category": "str"}
)

# Embedding feature
embedding_feature = FeatureDefinition(
    feature_name="text_embedding",
    feature_type="embedding",
    transformation_code="""
def transform(df):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(df['text'].tolist())
""",
    input_schema={"text": "str"},
    output_schema={"text_embedding": "array"}
)
```

## Registering Features

### Simple Feature

```python
# Register a basic transformation
feature_store.register_feature(
    feature_name="age_normalized",
    feature_definition=FeatureDefinition(
        feature_name="age_normalized",
        feature_type="numeric",
        description="Normalized age (0-1 scale)",
        transformation_code="""
def transform(df):
    return (df['age'] - 18) / (100 - 18)
""",
        input_schema={"age": "int"},
        output_schema={"age_normalized": "float"},
        version="v1.0"
    )
)
```

### Feature with Dependencies

```python
# Register base feature
feature_store.register_feature("age_normalized", age_norm_def)

# Register dependent feature
feature_store.register_feature(
    feature_name="age_risk_interaction",
    feature_definition=FeatureDefinition(
        feature_name="age_risk_interaction",
        feature_type="numeric",
        description="Interaction between age and risk score",
        transformation_code="""
def transform(df):
    return df['age_normalized'] * df['risk_score']
""",
        input_schema={
            "age_normalized": "float",
            "risk_score": "float"
        },
        output_schema={"age_risk_interaction": "float"},
        version="v1.0",
        dependencies=["age_normalized"]  # Will be computed first
    )
)
```

### Versioned Features

```python
# Register v1.0
feature_store.register_feature("severity_score", severity_v1)

# Update to v2.0 with improved logic
severity_v2 = FeatureDefinition(
    feature_name="severity_score",
    feature_type="numeric",
    description="Improved severity score with additional factors",
    transformation_code="""
def transform(df):
    # Enhanced calculation
    base_score = (df['phq9'] * 0.3 + df['gad7'] * 0.3 + 
                  df['pcl5'] * 0.2 + df['sleep_quality'] * 0.2)
    return base_score / 30 * 100
""",
    input_schema={
        "phq9": "int", "gad7": "int", 
        "pcl5": "int", "sleep_quality": "int"
    },
    output_schema={"severity_score": "float"},
    version="v2.0"
)

feature_store.register_feature("severity_score", severity_v2)
```

## Computing Features

### Batch Computation

```python
# Prepare input data
input_df = pd.DataFrame({
    'patient_id': [1, 2, 3],
    'age': [25, 45, 67],
    'phq9_score': [15, 8, 22],
    'gad7_score': [12, 5, 18]
})

# Compute multiple features
features = feature_store.compute_features(
    feature_names=["age_normalized", "severity_score", "risk_category"],
    input_data=input_df
)

print(features)
# Output: DataFrame with computed feature columns
```

### Online Serving

```python
# Get features for a single entity (with caching)
features = feature_store.get_features(
    feature_names=["age_normalized", "severity_score"],
    entity_ids=["patient_123"],
    mode="online"
)

# Features are cached for fast retrieval
# Subsequent calls return cached values
```

### Batch Serving for Training

```python
# Get features for multiple entities
entity_ids = ["patient_1", "patient_2", "patient_3"]
features = feature_store.get_features(
    feature_names=["age_normalized", "severity_score", "risk_category"],
    entity_ids=entity_ids,
    mode="batch"
)

# Use for model training
X = features[["age_normalized", "severity_score"]]
y = labels
model.fit(X, y)
```

## Materialization

Pre-compute and store features for faster access:

```python
from src.ds.data_versioning import DataVersionControl

# Register dataset version
dvc = DataVersionControl(storage, db)
dataset_version = dvc.register_dataset(
    dataset=raw_df,
    dataset_name="patient_data",
    source="database"
)

# Materialize features for this dataset
materialized_path = feature_store.materialize_features(
    feature_names=["age_normalized", "severity_score", "risk_category"],
    dataset_version_id=str(dataset_version.version_id)
)

print(f"Features materialized to: {materialized_path}")

# Load materialized features
materialized_df = pd.read_parquet(materialized_path)
```

## Feature Validation

### Schema Validation

```python
# Feature store automatically validates schemas
try:
    features = feature_store.compute_features(
        feature_names=["age_normalized"],
        input_data=pd.DataFrame({"wrong_column": [1, 2, 3]})
    )
except ValueError as e:
    print(f"Schema validation failed: {e}")
    # Output: Missing required column 'age'
```

### Type Validation

```python
# Validates output types match definition
feature_def = FeatureDefinition(
    feature_name="age_category",
    feature_type="categorical",
    transformation_code="""
def transform(df):
    return pd.cut(df['age'], bins=[0, 18, 65, 100], 
                  labels=['youth', 'adult', 'senior'])
""",
    input_schema={"age": "int"},
    output_schema={"age_category": "str"}
)

feature_store.register_feature("age_category", feature_def)
```

## Integration with ML Pipeline

### Training Pipeline

```python
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.data_versioning import DataVersionControl

# Start experiment
tracker = ExperimentTracker(storage, db)
run = tracker.start_run(experiment_name="risk_prediction")

# Version input data
dvc = DataVersionControl(storage, db)
data_version = dvc.register_dataset(raw_df, "training_data", "pipeline")

# Compute features
features = feature_store.compute_features(
    feature_names=["age_normalized", "severity_score", "risk_category"],
    input_data=raw_df
)

# Log feature versions
tracker.log_params({
    "features": ["age_normalized", "severity_score", "risk_category"],
    "feature_versions": ["v1.0", "v2.0", "v1.0"],
    "data_version": str(data_version.version_id)
})

# Train model
model = train_model(features, labels)
tracker.log_metrics({"accuracy": 0.92})
tracker.end_run()
```

### Inference Pipeline

```python
# Use same features for inference
def predict(patient_data):
    # Compute features using feature store
    features = feature_store.compute_features(
        feature_names=["age_normalized", "severity_score", "risk_category"],
        input_data=patient_data
    )
    
    # Make prediction
    prediction = model.predict(features)
    return prediction

# Ensures consistency between training and inference
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
feature_store.register_feature("patient_age_normalized_0_1", age_def)
feature_store.register_feature("phq9_severity_weighted_score", severity_def)

# Avoid
feature_store.register_feature("feat1", age_def)
feature_store.register_feature("x", severity_def)
```

### 2. Document Transformations

```python
feature_def = FeatureDefinition(
    feature_name="crisis_risk_score",
    feature_type="numeric",
    description="""
    Composite crisis risk score combining:
    - Recent assessment scores (40%)
    - Historical trend (30%)
    - Demographic risk factors (30%)
    
    Range: 0-100 (higher = greater risk)
    Threshold: >70 triggers immediate review
    """,
    transformation_code=transform_code,
    input_schema=input_schema,
    output_schema=output_schema
)
```

### 3. Version Features Appropriately

```python
# Use semantic versioning
# v1.0 - Initial implementation
# v1.1 - Bug fix or minor improvement
# v2.0 - Breaking change or major algorithm update

# Track changes
feature_v2 = FeatureDefinition(
    feature_name="severity_score",
    version="v2.0",
    description="v2.0: Added sleep quality factor (20% weight)",
    # ... rest of definition
)
```

### 4. Test Features Before Registration

```python
# Test transformation locally
def test_feature():
    test_df = pd.DataFrame({
        'age': [25, 45, 67],
        'phq9_score': [15, 8, 22]
    })
    
    # Test transformation
    result = transform_function(test_df)
    
    # Validate output
    assert result.shape[0] == test_df.shape[0]
    assert result.min() >= 0 and result.max() <= 100
    
    print("✓ Feature test passed")

test_feature()

# Then register
feature_store.register_feature(feature_name, feature_def)
```

### 5. Monitor Feature Quality

```python
# Track feature statistics
def monitor_feature_quality(feature_name, computed_features):
    stats = {
        'mean': computed_features[feature_name].mean(),
        'std': computed_features[feature_name].std(),
        'null_count': computed_features[feature_name].isnull().sum(),
        'unique_count': computed_features[feature_name].nunique()
    }
    
    # Log to monitoring system
    log_feature_stats(feature_name, stats)
    
    # Alert on anomalies
    if stats['null_count'] > 0:
        alert(f"Null values detected in {feature_name}")
```

## CLI Commands

### List Features

```bash
python run_cli.py features list
```

### Show Feature Details

```bash
python run_cli.py features show age_normalized
```

### Register Feature

```bash
python run_cli.py features register --config feature_config.json
```

### Compute Features

```bash
python run_cli.py features compute \
  --features age_normalized,severity_score \
  --input data.csv \
  --output features.csv
```

## API Endpoints

### Register Feature

```bash
curl -X POST http://localhost:8000/api/v1/features \
  -H "Content-Type: application/json" \
  -d '{
    "feature_name": "age_normalized",
    "feature_type": "numeric",
    "description": "Normalized age",
    "transformation_code": "def transform(df): return df[\"age\"] / 100",
    "input_schema": {"age": "int"},
    "output_schema": {"age_normalized": "float"}
  }'
```

### List Features

```bash
curl -X GET http://localhost:8000/api/v1/features
```

### Get Feature

```bash
curl -X GET http://localhost:8000/api/v1/features/age_normalized
```

### Compute Features

```bash
curl -X POST http://localhost:8000/api/v1/features/compute \
  -H "Content-Type: application/json" \
  -d '{
    "feature_names": ["age_normalized", "severity_score"],
    "input_data": [
      {"age": 25, "phq9_score": 15},
      {"age": 45, "phq9_score": 8}
    ]
  }'
```

### Serve Features (Online)

```bash
curl -X GET "http://localhost:8000/api/v1/features/serve?entity_ids=patient_123&features=age_normalized,severity_score"
```

## Troubleshooting

### Issue: Feature computation fails

```python
# Check transformation code syntax
try:
    exec(feature_def.transformation_code)
except SyntaxError as e:
    print(f"Syntax error in transformation: {e}")

# Validate with sample data
test_df = pd.DataFrame({"age": [25]})
result = transform(test_df)
```

### Issue: Slow online serving

```python
# Enable caching
feature_store = FeatureStore(
    db_connection=db,
    cache_backend="redis",
    cache_url="redis://localhost:6379",
    cache_ttl=3600  # 1 hour
)

# Pre-materialize frequently accessed features
feature_store.materialize_features(
    feature_names=["commonly_used_feature"],
    dataset_version_id=version_id
)
```

### Issue: Schema mismatch

```python
# Verify input schema matches data
input_df = pd.DataFrame({"age": [25], "name": ["John"]})
feature_def = FeatureDefinition(
    input_schema={"age": "int"},  # Only requires 'age'
    # ...
)

# Extra columns are ignored, missing columns raise error
```

## Advanced Usage

### Custom Transformation Functions

```python
# Use external libraries
feature_def = FeatureDefinition(
    feature_name="sentiment_score",
    transformation_code="""
def transform(df):
    from textblob import TextBlob
    return df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
""",
    input_schema={"text": "str"},
    output_schema={"sentiment_score": "float"}
)
```

### Feature Groups

```python
# Define related features together
demographic_features = [
    "age_normalized",
    "gender_encoded",
    "location_risk_score"
]

clinical_features = [
    "phq9_score",
    "gad7_score",
    "severity_score"
]

# Compute by group
demo_feats = feature_store.compute_features(demographic_features, input_df)
clin_feats = feature_store.compute_features(clinical_features, input_df)
```

## See Also

- [Experiment Tracking Usage](experiment_tracking_usage.md)
- [Data Versioning Usage](data_versioning_usage.md)
- [Feature Engineering Usage](feature_engineering_usage.md)
