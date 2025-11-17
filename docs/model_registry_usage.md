# Model Registry and Inference Usage Guide

This guide demonstrates how to use the ModelRegistry, InferenceEngine, and EnsemblePredictor components for managing models and generating predictions.

## Overview

The model registry and inference system consists of three main components:

1. **ModelRegistry**: Manages model versions, metadata, and lifecycle
2. **InferenceEngine**: Generates predictions from individual models
3. **EnsemblePredictor**: Combines predictions from multiple models with confidence scoring and risk classification

## Basic Usage

### 1. Initialize Components

```python
from src.ml import ModelRegistry, InferenceEngine, EnsemblePredictor

# Initialize registry
registry = ModelRegistry(registry_dir="models/registry")

# Initialize inference engine
inference_engine = InferenceEngine(
    model_registry=registry,
    batch_size=32
)

# Initialize ensemble predictor
ensemble_predictor = EnsemblePredictor(
    model_registry=registry,
    inference_engine=inference_engine,
    alert_threshold=75.0
)
```

### 2. Register Models

```python
from src.ml import BaselineModelTrainer
import pandas as pd

# Train a model
trainer = BaselineModelTrainer(model_dir="models")
model, metadata = trainer.train_logistic_regression(X_train, y_train, X_val, y_val)

# Register the model
model_id = registry.register_model(
    model=model,
    model_type='logistic_regression',
    metadata=metadata,
    artifacts={'scaler': trainer.scaler},  # Include preprocessing artifacts
    set_active=True
)

print(f"Model registered with ID: {model_id}")
```

### 3. Load Models

```python
# Load a specific model
model_bundle, metadata = registry.load_model(model_id)
model = model_bundle['model']
scaler = model_bundle['artifacts']['scaler']

# Get active models
active_models = registry.get_active_models()
print(f"Found {len(active_models)} active models")
```

### 4. Generate Predictions

#### Baseline Models (Logistic Regression, LightGBM)

```python
# Prepare features
features = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0],
    'feature2': [0.5, 1.5, 2.5],
    # ... more features
})

# Generate predictions
predictions = inference_engine.predict_baseline(
    model_id=model_id,
    features=features,
    return_proba=True
)

print(f"Predictions: {predictions}")
```

#### Temporal Models (RNN, TFT)

```python
import numpy as np

# Prepare sequences (n_samples, seq_length, n_features)
sequences = np.random.rand(10, 30, 20)

# Generate predictions
predictions = inference_engine.predict_temporal(
    model_id=rnn_model_id,
    sequences=sequences,
    return_proba=True
)

print(f"Temporal predictions: {predictions}")
```

#### Anomaly Detection (Isolation Forest)

```python
# Detect anomalies
anomaly_scores = inference_engine.detect_anomalies(
    model_id=anomaly_model_id,
    features=features,
    return_scores=True,
    calibrate=True  # Calibrate to 0-100 scale
)

print(f"Anomaly scores: {anomaly_scores}")
```

### 5. Ensemble Predictions

```python
# Generate ensemble predictions with automatic model selection
results = ensemble_predictor.predict_with_ensemble(
    features=features,
    individual_ids=['patient_001', 'patient_002', 'patient_003']
)

# Access results
risk_scores = results['risk_scores']
confidence = results['confidence']
risk_levels = results['risk_levels']
alerts = results['alerts_triggered']

print(f"Risk scores: {risk_scores}")
print(f"Confidence: {confidence}")
print(f"Risk levels: {risk_levels}")
print(f"Alerts triggered: {sum(alerts)}")
```

### 6. Manual Ensemble with Custom Weights

```python
# Specify models and weights
model_ids = ['model_1', 'model_2', 'model_3']
weights = [0.5, 0.3, 0.2]  # Must sum to 1.0

results = ensemble_predictor.predict_with_ensemble(
    features=features,
    model_ids=model_ids,
    weights=weights
)
```

### 7. Risk Classification

```python
from src.ml import RiskLevel

# Classify individual risk scores
risk_score = 82.5
risk_level = ensemble_predictor.classify_risk_level(risk_score)

print(f"Risk score {risk_score} classified as: {risk_level.value}")
# Output: Risk score 82.5 classified as: critical
```

### 8. Model Lifecycle Management

```python
# List all models
all_models = registry.list_models()

# List models by type
lgbm_models = registry.list_models(model_type='lightgbm')

# List active models only
active_models = registry.list_models(status='active')

# Get model metadata
metadata = registry.get_model_metadata(model_id)
print(f"Model trained at: {metadata['trained_at']}")
print(f"Validation score: {metadata['val_score']}")

# Retire a model
registry.retire_model(
    model_id='old_model_id',
    reason='Replaced by better performing model'
)

# Clear cache
registry.clear_cache()
```

## Advanced Usage

### Custom Alert Callback

```python
def alert_handler(alert_data):
    """Custom alert handler for high-risk cases."""
    print(f"ALERT: Individual {alert_data['individual_id']}")
    print(f"Risk score: {alert_data['risk_score']}")
    print(f"Risk level: {alert_data['risk_level']}")
    
    # Send notification, log to database, etc.
    # ...

# Initialize with custom callback
ensemble_predictor = EnsemblePredictor(
    model_registry=registry,
    inference_engine=inference_engine,
    alert_threshold=75.0,
    alert_callback=alert_handler
)
```

### Confidence Calculation Methods

```python
# Different confidence calculation methods
predictions_list = [pred1, pred2, pred3]

# Agreement-based (default)
confidence_agreement = ensemble_predictor.calculate_confidence(
    predictions_list,
    method='agreement'
)

# Variance-based
confidence_variance = ensemble_predictor.calculate_confidence(
    predictions_list,
    method='variance'
)

# Entropy-based
confidence_entropy = ensemble_predictor.calculate_confidence(
    predictions_list,
    method='entropy'
)
```

### Ensemble Methods

```python
# Different ensemble methods
predictions_list = [pred1, pred2, pred3]

# Weighted average (default)
ensemble_weighted = ensemble_predictor.ensemble_predictions(
    predictions_list,
    weights=[0.5, 0.3, 0.2],
    method='weighted_average'
)

# Simple average
ensemble_avg = ensemble_predictor.ensemble_predictions(
    predictions_list,
    method='simple_average'
)

# Maximum
ensemble_max = ensemble_predictor.ensemble_predictions(
    predictions_list,
    method='max'
)

# Median
ensemble_median = ensemble_predictor.ensemble_predictions(
    predictions_list,
    method='median'
)
```

## Performance Considerations

### Caching

The ModelRegistry automatically caches loaded models for improved performance:

```python
# First load - reads from disk
model_bundle, metadata = registry.load_model(model_id, use_cache=True)

# Subsequent loads - uses cache
model_bundle, metadata = registry.load_model(model_id, use_cache=True)

# Force reload from disk
model_bundle, metadata = registry.load_model(model_id, use_cache=False)
```

### Batch Processing

The InferenceEngine processes predictions in batches for efficiency:

```python
# Configure batch size
inference_engine = InferenceEngine(
    model_registry=registry,
    batch_size=64  # Larger batches for better throughput
)
```

### GPU Acceleration

For PyTorch models (RNN, TFT), GPU acceleration is automatically used if available:

```python
# Force CPU
inference_engine = InferenceEngine(
    model_registry=registry,
    device='cpu'
)

# Force GPU
inference_engine = InferenceEngine(
    model_registry=registry,
    device='cuda'
)
```

## Error Handling

```python
from src.exceptions import (
    ModelNotFoundError,
    ModelRegistrationError,
    InferenceError,
    EnsembleError
)

try:
    # Load model
    model_bundle, metadata = registry.load_model('nonexistent_model')
except ModelNotFoundError as e:
    print(f"Model not found: {e}")

try:
    # Generate predictions
    predictions = inference_engine.predict_baseline(model_id, features)
except InferenceError as e:
    print(f"Inference failed: {e}")

try:
    # Ensemble predictions
    results = ensemble_predictor.predict_with_ensemble(features)
except EnsembleError as e:
    print(f"Ensemble failed: {e}")
```

## Best Practices

1. **Always register models with metadata**: Include validation scores, feature names, and training parameters
2. **Use artifacts for preprocessing**: Store scalers, encoders, and other preprocessing objects
3. **Set active models explicitly**: Mark the best-performing models as active for ensemble selection
4. **Monitor confidence scores**: Low confidence indicates model disagreement and may require human review
5. **Retire old models**: Keep the registry clean by retiring deprecated models
6. **Clear cache periodically**: Prevent memory issues in long-running processes
7. **Use appropriate ensemble methods**: Weighted average works well when models have different performance levels
8. **Set alert thresholds carefully**: Balance sensitivity and specificity based on clinical requirements

## Integration with Full Pipeline

```python
from src.processing import ETLPipeline
from src.ml import FeatureEngineeringPipeline

# Complete prediction pipeline
def predict_risk(raw_data):
    # 1. ETL
    etl = ETLPipeline()
    cleaned_data = etl.process(raw_data)
    
    # 2. Feature engineering
    feature_pipeline = FeatureEngineeringPipeline()
    features = feature_pipeline.extract_all_features(cleaned_data)
    
    # 3. Ensemble prediction
    results = ensemble_predictor.predict_with_ensemble(
        features=features,
        individual_ids=cleaned_data['anonymized_id'].tolist()
    )
    
    return results

# Use the pipeline
results = predict_risk(raw_data)
```

## Monitoring and Logging

All components include comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Components will log:
# - Model registration and loading
# - Inference timing and throughput
# - Ensemble composition and weights
# - Alert triggers
# - Errors and warnings
```

## Next Steps

- See `model_training_usage.md` for training models
- See `feature_engineering_usage.md` for feature preparation
- See API documentation for REST endpoint integration
