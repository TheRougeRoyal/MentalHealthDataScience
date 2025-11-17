# Interpretability Engine Usage Guide

This guide demonstrates how to use the InterpretabilityEngine to generate comprehensive model explanations.

## Overview

The InterpretabilityEngine provides three types of explanations:
1. **SHAP Values**: Feature importance and contribution analysis
2. **Counterfactual Explanations**: Minimal changes needed to alter predictions
3. **Rule Extraction**: Simple decision rules that approximate model behavior

## Basic Usage

### Initialize the Engine

```python
from src.ml.interpretability import InterpretabilityEngine
from src.ml.model_registry import ModelRegistry

# Initialize model registry
model_registry = ModelRegistry(storage_path='models/')

# Initialize interpretability engine with clinical mappings
clinical_mappings = {
    'sleep_duration': 'Sleep Duration (hours)',
    'hrv_rmssd': 'Heart Rate Variability (RMSSD)',
    'activity_level': 'Physical Activity Level',
    'sentiment_score': 'Emotional Sentiment Score'
}

engine = InterpretabilityEngine(
    model_registry=model_registry,
    clinical_mappings=clinical_mappings
)
```

## Comprehensive Explanation Generation

### Generate All Components

The `generate_explanation()` method orchestrates all interpretability components and ensures they complete within the specified timeout (default: 3 seconds).

```python
import pandas as pd

# Prepare input features
features = pd.DataFrame({
    'sleep_duration': [6.5],
    'hrv_rmssd': [45.0],
    'activity_level': [3.2],
    'sentiment_score': [0.6]
})

# Prepare training data for rule extraction (optional)
training_data = pd.DataFrame({
    'sleep_duration': [5.5, 7.2, 6.8, 5.0, 7.5],
    'hrv_rmssd': [38.0, 52.0, 48.0, 35.0, 55.0],
    'activity_level': [2.5, 4.1, 3.8, 2.0, 4.5],
    'sentiment_score': [0.4, 0.7, 0.65, 0.3, 0.75]
})
training_labels = pd.Series([1, 0, 0, 1, 0])  # 1 = high risk, 0 = low risk

# Generate comprehensive explanation
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    training_data=training_data,
    training_labels=training_labels,
    target_class='low',
    top_k_features=10,
    max_counterfactual_changes=5,
    max_rule_depth=3,
    timeout_seconds=3.0
)

# Print clinical summary
print(explanation['clinical_summary'])
```

### Output Structure

```python
{
    'model_id': 'lgbm_v1.0.0',
    'n_instances': 1,
    'timestamp': 1700000000.0,
    'components': {
        'shap': {
            'base_value': 0.45,
            'top_features': [
                {
                    'feature': 'sleep_duration',
                    'clinical_name': 'Sleep Duration (hours)',
                    'importance': 0.82,
                    'mean_shap_value': -0.15
                },
                # ... more features
            ],
            'feature_names': ['sleep_duration', 'hrv_rmssd', ...]
        },
        'counterfactuals': [
            {
                'instance_id': 0,
                'current_risk_level': 'high',
                'current_score': 68.5,
                'target_risk_level': 'low',
                'changes_needed': [
                    {
                        'feature': 'sleep_duration',
                        'clinical_name': 'Sleep Duration (hours)',
                        'original_value': 5.5,
                        'proposed_value': 7.2,
                        'change_magnitude': 1.7,
                        'importance': 0.82
                    },
                    # ... more changes
                ],
                'counterfactual_score': 22.3,
                'counterfactual_risk_level': 'low',
                'achieved_target': True,
                'description': 'To reach low risk level...'
            }
        ],
        'rules': {
            'rules': [
                {
                    'conditions': ['Sleep Duration (hours) <= 6.0', 'Heart Rate Variability (RMSSD) <= 40.0'],
                    'prediction': 'high_risk',
                    'confidence': 0.89,
                    'n_samples': 45,
                    'rule_text': 'Sleep Duration (hours) <= 6.0 AND Heart Rate Variability (RMSSD) <= 40.0'
                },
                # ... more rules
            ],
            'max_depth': 3,
            'n_rules': 8,
            'fidelity': 0.87,
            'feature_names': ['sleep_duration', 'hrv_rmssd', ...]
        }
    },
    'generation_time': {
        'shap': 0.45,
        'counterfactuals': 0.82,
        'rules': 0.38
    },
    'total_generation_time': 1.65,
    'timeout_exceeded': False,
    'errors': [],
    'clinical_summary': '=== Model Explanation Summary ===\n...'
}
```

## Selective Component Generation

You can generate only specific components to optimize performance:

### SHAP Values Only

```python
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    include_shap=True,
    include_counterfactuals=False,
    include_rules=False,
    timeout_seconds=1.0
)

# Access SHAP results
shap_data = explanation['components']['shap']
for feature in shap_data['top_features'][:5]:
    print(f"{feature['clinical_name']}: {feature['importance']:.3f}")
```

### Counterfactuals Only

```python
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    target_class='low',
    include_shap=False,
    include_counterfactuals=True,
    include_rules=False,
    timeout_seconds=1.0
)

# Access counterfactual results
cf_data = explanation['components']['counterfactuals'][0]
print(cf_data['description'])
```

### Rules Only

```python
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    training_data=training_data,
    training_labels=training_labels,
    include_shap=False,
    include_counterfactuals=False,
    include_rules=True,
    timeout_seconds=1.0
)

# Access rule results
rules_data = explanation['components']['rules']
print(f"Model Fidelity: {rules_data['fidelity']:.1%}")
for rule in rules_data['rules'][:3]:
    print(f"IF {rule['rule_text']} THEN {rule['prediction']} (confidence: {rule['confidence']:.1%})")
```

## Individual Component Methods

You can also use individual methods for more control:

### SHAP Values

```python
shap_result = engine.compute_shap_values(
    model_id='lgbm_v1.0.0',
    features=features,
    top_k=10,
    background_data=training_data  # Optional
)

# Access top features
for feature in shap_result['top_features']:
    print(f"{feature['clinical_name']}: {feature['importance']:.3f}")
```

### Counterfactual Explanations

```python
counterfactuals = engine.generate_counterfactuals(
    model_id='lgbm_v1.0.0',
    features=features,
    target_class='low',
    max_changes=5,
    feature_ranges={
        'sleep_duration': (4.0, 10.0),
        'hrv_rmssd': (20.0, 100.0)
    }
)

# Access counterfactual for first instance
cf = counterfactuals[0]
print(cf['description'])
print(f"Changes needed: {len(cf['changes_needed'])}")
```

### Rule Extraction

```python
rules = engine.extract_rule_set(
    model_id='lgbm_v1.0.0',
    training_data=training_data,
    training_labels=training_labels,
    max_depth=3,
    min_samples_leaf=50
)

# Access rules
print(f"Extracted {rules['n_rules']} rules with fidelity: {rules['fidelity']:.1%}")
for rule in rules['rules']:
    print(f"IF {rule['rule_text']} THEN {rule['prediction']}")
```

## Performance Considerations

### Timeout Management

The `generate_explanation()` method enforces a timeout to ensure explanations are generated within the required 3 seconds:

```python
# Set custom timeout
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    timeout_seconds=2.0  # Stricter timeout
)

# Check if timeout was exceeded
if explanation['timeout_exceeded']:
    print("Warning: Explanation generation exceeded timeout")
    print(f"Total time: {explanation['total_generation_time']:.3f}s")
```

### Component Timing

Monitor individual component performance:

```python
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    training_data=training_data,
    training_labels=training_labels
)

# Print timing breakdown
for component, duration in explanation['generation_time'].items():
    print(f"{component}: {duration:.3f}s")
print(f"Total: {explanation['total_generation_time']:.3f}s")
```

### Error Handling

The engine gracefully handles component failures:

```python
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features
)

# Check for errors
if explanation['errors']:
    print("Some components failed:")
    for error in explanation['errors']:
        print(f"  - {error['component']}: {error['error']}")

# Components that succeeded are still available
if explanation['components']['shap']:
    print("SHAP values generated successfully")
if explanation['components']['counterfactuals']:
    print("Counterfactuals generated successfully")
```

## Clinical Summary

The engine automatically generates a human-readable clinical summary:

```python
explanation = engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    training_data=training_data,
    training_labels=training_labels
)

# Print the clinical summary
print(explanation['clinical_summary'])
```

Example output:
```
=== Model Explanation Summary ===

Key Contributing Factors:
  1. Sleep Duration (hours) (importance: 0.820)
  2. Heart Rate Variability (RMSSD) (importance: 0.650)
  3. Physical Activity Level (importance: 0.420)
  4. Emotional Sentiment Score (importance: 0.380)

Actionable Changes:
  Current Risk: high
  Target Risk: low
  Changes Needed: 2
    1. Increase Sleep Duration (hours) from 5.50 to 7.20
    2. Increase Heart Rate Variability (RMSSD) from 38.00 to 48.00

Decision Rules:
  Total Rules: 8
  Model Fidelity: 87.0%

  High Risk Indicators:
    1. IF Sleep Duration (hours) <= 6.0 AND Heart Rate Variability (RMSSD) <= 40.0
       THEN high risk (confidence: 89.0%)

Generation Performance:
  shap: 0.450s
  counterfactuals: 0.820s
  rules: 0.380s
  Total: 1.650s
```

## Best Practices

1. **Use Clinical Mappings**: Always provide clinical terminology mappings for better interpretability
2. **Set Appropriate Timeouts**: Default 3 seconds is usually sufficient, but adjust based on your needs
3. **Selective Generation**: Only generate components you need to optimize performance
4. **Cache Training Data**: Reuse training data for rule extraction across multiple predictions
5. **Monitor Performance**: Track generation times to identify bottlenecks
6. **Handle Errors Gracefully**: Check for errors and use partial results when available

## Integration with Prediction Pipeline

```python
from src.ml.inference_engine import InferenceEngine
from src.ml.ensemble_predictor import EnsemblePredictor

# Generate prediction
inference_engine = InferenceEngine(model_registry)
predictions = inference_engine.predict_baseline('lgbm_v1.0.0', features)

# Generate explanation
explanation = interpretability_engine.generate_explanation(
    model_id='lgbm_v1.0.0',
    features=features,
    training_data=training_data,
    training_labels=training_labels,
    timeout_seconds=3.0
)

# Combine prediction and explanation
result = {
    'risk_score': predictions['risk_scores'][0],
    'risk_level': predictions['risk_levels'][0],
    'explanation': explanation['clinical_summary'],
    'top_factors': explanation['components']['shap']['top_features'][:5],
    'actionable_changes': explanation['components']['counterfactuals'][0]['changes_needed']
}
```

## Requirements Compliance

This implementation satisfies Requirement 8.5:
- ✅ Generates SHAP values showing top 10 features (8.1)
- ✅ Provides counterfactual explanations (8.2)
- ✅ Extracts rule sets with max depth 3 (8.3)
- ✅ Presents explanations in clinical terminology (8.4)
- ✅ Generates all interpretability outputs within 3 seconds (8.5)
