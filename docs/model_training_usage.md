# Model Training Infrastructure Usage Guide

This guide demonstrates how to use the model training infrastructure for the Mental Health Risk Assessment System.

## Overview

The model training infrastructure provides:
- **Data Splitting**: Chronological splits with class balancing
- **Baseline Models**: Logistic Regression and LightGBM
- **Temporal Models**: RNN (LSTM/GRU) and Temporal Fusion Transformer
- **Anomaly Detection**: Isolation Forest for unusual patterns
- **Model Evaluation**: Comprehensive metrics including AUROC, PR-AUC, calibration, decision curves, and fairness audits

## 1. Data Splitting

### Basic Chronological Split

```python
from src.ml import DataSplitter
import pandas as pd

# Initialize splitter
splitter = DataSplitter(test_size=0.2, random_state=42)

# Prepare your data with timestamp column
df = pd.DataFrame({
    'timestamp': [...],
    'feature1': [...],
    'feature2': [...],
    'target': [...]
})

# Split chronologically (most recent 20% as test set)
X_train, X_test, y_train, y_test = splitter.chronological_split(
    df,
    timestamp_col='timestamp',
    target_col='target'
)
```

### Stratified K-Fold Cross-Validation

```python
# Create stratified k-fold splits
splits = splitter.stratified_kfold_split(X_train, y_train, n_splits=5)

# Use splits for cross-validation
for fold_idx, (train_idx, val_idx) in enumerate(splits):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # Train model on this fold
    # ...
```

### Class Balancing

```python
# Balance classes if minority class < 20%
X_balanced, y_balanced = splitter.balance_classes(
    X_train,
    y_train,
    method='smote',  # Options: 'smote', 'undersample', 'smoteenn'
    minority_threshold=0.2
)
```

### Complete Train/Val/Test Split

```python
# Get train/val/test split with automatic balancing
X_train, X_val, X_test, y_train, y_val, y_test = splitter.get_train_val_test_split(
    df,
    timestamp_col='timestamp',
    target_col='target',
    val_size=0.2,
    balance_train=True,
    balance_method='smote'
)
```

## 2. Baseline Model Training

### Logistic Regression

```python
from src.ml import BaselineModelTrainer

# Initialize trainer
trainer = BaselineModelTrainer(model_dir="models", random_state=42)

# Train logistic regression with hyperparameter tuning
model, metadata = trainer.train_logistic_regression(
    X_train,
    y_train,
    X_val=X_val,
    y_val=y_val,
    cv=5,
    search_method='grid'  # or 'random'
)

# Model and metadata are automatically saved to disk
print(f"Best CV score: {metadata['cv_score']:.4f}")
print(f"Validation score: {metadata['val_score']:.4f}")

# Load saved model
model, scaler, metadata = trainer.load_logistic_regression()
```

### LightGBM

```python
# Train LightGBM with early stopping
model, metadata = trainer.train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    early_stopping_rounds=50,
    cv=5
)

# View feature importance
for feature_info in metadata['feature_importance'][:10]:
    print(f"{feature_info['feature']}: {feature_info['importance']:.2f}")

# Load saved model
model, metadata = trainer.load_lgbm()
```

## 3. Temporal Model Training

### RNN (LSTM/GRU)

```python
from src.ml import TemporalModelTrainer

# Initialize trainer
trainer = TemporalModelTrainer(model_dir="models", random_state=42)

# Prepare sequences from time-series data
sequences_train, labels_train, feature_names = trainer.prepare_sequences(
    df_train,
    id_col='anonymized_id',
    timestamp_col='timestamp',
    target_col='target',
    seq_length=30,
    padding_value=0.0
)

sequences_val, labels_val, _ = trainer.prepare_sequences(
    df_val,
    id_col='anonymized_id',
    timestamp_col='timestamp',
    target_col='target',
    seq_length=30
)

# Train LSTM model
model, metadata = trainer.train_rnn(
    sequences_train,
    labels_train,
    sequences_val,
    labels_val,
    feature_names,
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    rnn_type='lstm',  # or 'gru'
    bidirectional=False,
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    early_stopping_patience=10
)

print(f"Best validation loss: {metadata['best_val_loss']:.4f}")
print(f"Validation AUROC: {metadata['val_auroc']:.4f}")
```

### Temporal Fusion Transformer

```python
# Train Temporal Fusion Transformer
model, metadata = trainer.train_temporal_fusion(
    df,
    time_idx='time_idx',
    target='target',
    group_ids=['anonymized_id'],
    static_categoricals=['gender', 'age_group'],
    static_reals=['baseline_score'],
    time_varying_known_categoricals=['day_of_week'],
    time_varying_known_reals=['time_idx'],
    time_varying_unknown_reals=['feature1', 'feature2'],
    max_encoder_length=30,
    max_prediction_length=1,
    batch_size=64,
    max_epochs=50
)
```

## 4. Anomaly Detection

### Isolation Forest

```python
from src.ml import AnomalyDetectionTrainer

# Initialize trainer
trainer = AnomalyDetectionTrainer(model_dir="models", random_state=42)

# Train Isolation Forest
model, metadata = trainer.train_isolation_forest(
    X_train,
    y_train=y_train,  # Optional, for evaluation
    X_val=X_val,
    y_val=y_val,
    contamination=0.1,  # Expected proportion of anomalies
    n_estimators=100,
    calibrate_scores=True
)

# View anomaly statistics
print(f"Training anomaly rate: {metadata['training_stats']['anomaly_rate']:.2%}")
print(f"Validation AUROC: {metadata['validation_metrics']['auroc']:.4f}")

# Calibrate a single score
calibration_params = metadata['calibration_params']
raw_score = -0.15
calibrated_score = trainer.calibrate_score(raw_score, calibration_params)
print(f"Calibrated risk score: {calibrated_score:.1f}/100")

# Load saved model
model, scaler, metadata = trainer.load_isolation_forest()
```

## 5. Model Evaluation

### Comprehensive Evaluation

```python
from src.ml import ModelEvaluator
import numpy as np

# Initialize evaluator
evaluator = ModelEvaluator(output_dir="evaluation_results")

# Get predictions from your model
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Run comprehensive evaluation
results = evaluator.comprehensive_evaluation(
    y_test,
    y_pred_proba,
    sensitive_features=sensitive_features_test,  # Optional
    model_name="lgbm_v1"
)

# Access individual metrics
print(f"AUROC: {results['auroc_metrics']['auroc']:.4f}")
print(f"PR-AUC: {results['pr_auc_metrics']['pr_auc']:.4f}")
print(f"Brier Score: {results['calibration_metrics']['brier_score']:.4f}")
print(f"Max Net Benefit: {results['decision_curve_metrics']['max_benefit']:.4f}")
```

### Individual Metric Calculations

```python
# AUROC
auroc_results = evaluator.calculate_auroc(y_test, y_pred_proba, "my_model")
print(f"AUROC: {auroc_results['auroc']:.4f}")
print(f"Optimal threshold: {auroc_results['optimal_threshold']:.4f}")

# PR-AUC
pr_auc_results = evaluator.calculate_pr_auc(y_test, y_pred_proba, "my_model")
print(f"PR-AUC: {pr_auc_results['pr_auc']:.4f}")
print(f"Optimal F1: {pr_auc_results['optimal_f1']:.4f}")

# Calibration
calibration_results = evaluator.generate_calibration_curve(
    y_test, y_pred_proba, "my_model", n_bins=10
)
print(f"Expected Calibration Error: {calibration_results['expected_calibration_error']:.4f}")

# Decision Curve Analysis
dca_results = evaluator.decision_curve_analysis(y_test, y_pred_proba, "my_model")
print(f"Max benefit at threshold: {dca_results['max_benefit_threshold']:.4f}")
```

### Fairness Audit

```python
# Prepare sensitive features
sensitive_features = pd.DataFrame({
    'gender': test_df['gender'],
    'age_group': test_df['age_group'],
    'ethnicity': test_df['ethnicity']
})

# Run fairness audit
fairness_results = evaluator.fairness_audit(
    y_test,
    y_pred_proba,
    sensitive_features,
    model_name="my_model",
    threshold=0.5,
    disparity_threshold=0.1  # 10% maximum disparity
)

# Check for violations
if fairness_results['flags']:
    print(f"Fairness violations detected: {len(fairness_results['flags'])}")
    for flag in fairness_results['flags']:
        print(f"  {flag['feature']} - {flag['metric']}: {flag['disparity']:.2%}")
else:
    print("No fairness violations detected")

# View group-specific metrics
for feature, groups in fairness_results['groups'].items():
    print(f"\n{feature}:")
    for group, metrics in groups.items():
        print(f"  {group}: AUROC={metrics['auroc']:.4f}, TPR={metrics['tpr']:.4f}")
```

## Complete Training Pipeline Example

```python
from src.ml import (
    DataSplitter,
    BaselineModelTrainer,
    TemporalModelTrainer,
    AnomalyDetectionTrainer,
    ModelEvaluator
)
import pandas as pd

# 1. Load and prepare data
df = pd.read_csv("mental_health_data.csv")

# 2. Split data chronologically
splitter = DataSplitter(test_size=0.2, random_state=42)
X_train, X_val, X_test, y_train, y_val, y_test = splitter.get_train_val_test_split(
    df,
    timestamp_col='timestamp',
    target_col='risk_label',
    val_size=0.2,
    balance_train=True,
    balance_method='smote'
)

# 3. Train baseline models
baseline_trainer = BaselineModelTrainer(model_dir="models/baseline")

# Logistic Regression
lr_model, lr_metadata = baseline_trainer.train_logistic_regression(
    X_train, y_train, X_val, y_val
)

# LightGBM
lgbm_model, lgbm_metadata = baseline_trainer.train_lgbm(
    X_train, y_train, X_val, y_val
)

# 4. Train anomaly detection
anomaly_trainer = AnomalyDetectionTrainer(model_dir="models/anomaly")
anomaly_model, anomaly_metadata = anomaly_trainer.train_isolation_forest(
    X_train, y_train, X_val, y_val, contamination=0.1
)

# 5. Evaluate all models
evaluator = ModelEvaluator(output_dir="evaluation_results")

# Prepare sensitive features for fairness audit
sensitive_features = df.loc[X_test.index, ['gender', 'age_group', 'ethnicity']]

# Evaluate Logistic Regression
lr_model_loaded, lr_scaler, _ = baseline_trainer.load_logistic_regression()
X_test_scaled = lr_scaler.transform(X_test)
y_pred_lr = lr_model_loaded.predict_proba(X_test_scaled)[:, 1]

lr_results = evaluator.comprehensive_evaluation(
    y_test, y_pred_lr, sensitive_features, "logistic_regression"
)

# Evaluate LightGBM
lgbm_model_loaded, _ = baseline_trainer.load_lgbm()
y_pred_lgbm = lgbm_model_loaded.predict(X_test)

lgbm_results = evaluator.comprehensive_evaluation(
    y_test, y_pred_lgbm, sensitive_features, "lightgbm"
)

# 6. Compare models
print("\nModel Comparison:")
print(f"Logistic Regression AUROC: {lr_results['auroc_metrics']['auroc']:.4f}")
print(f"LightGBM AUROC: {lgbm_results['auroc_metrics']['auroc']:.4f}")

# 7. Check fairness
if lr_results['fairness_metrics']['flags']:
    print("\nLogistic Regression has fairness violations")
if lgbm_results['fairness_metrics']['flags']:
    print("LightGBM has fairness violations")
```

## Best Practices

1. **Always use chronological splits** for time-series data to prevent data leakage
2. **Balance training data** when minority class < 20% of samples
3. **Use cross-validation** for hyperparameter tuning to avoid overfitting
4. **Monitor calibration** to ensure predicted probabilities are reliable
5. **Perform fairness audits** to detect and mitigate bias across demographic groups
6. **Save all metadata** for model versioning and reproducibility
7. **Use early stopping** for neural networks to prevent overfitting
8. **Evaluate on held-out test set** only after all model selection is complete

## Notes

- All models are automatically saved with metadata for reproducibility
- Evaluation plots are saved to the output directory
- Fairness audits flag models with >10% performance disparity between groups
- Calibration is critical for risk assessment - monitor ECE and Brier score
- Decision curve analysis helps evaluate clinical utility at different thresholds
