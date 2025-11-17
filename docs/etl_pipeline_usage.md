# ETL Pipeline Usage Guide

## Overview

The ETL Pipeline orchestrates data cleaning, imputation, encoding, and normalization for the Mental Health Risk Assessment System. It processes data in a consistent, reproducible manner for both training and inference.

## Quick Start

```python
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
import pandas as pd

# Configure the pipeline
config = ETLPipelineConfig(
    outlier_method="iqr",
    iqr_multiplier=1.5,
    imputation_strategies={
        "heart_rate": "median",
        "sleep_hours": "median",
        "mood_score": "forward_fill",
    },
    standardize_columns=["heart_rate", "sleep_hours", "mood_score"],
    group_by_column="anonymized_id",
)

# Create pipeline
pipeline = ETLPipeline(config)

# Fit and transform training data
df_train_transformed = pipeline.fit_transform(df_train)

# Transform test data using fitted parameters
df_test_transformed = pipeline.transform(df_test)
```

## Pipeline Stages

The ETL pipeline executes four stages in sequence:

### 1. Data Cleaning
- Removes duplicate records based on anonymized ID and timestamp
- Detects outliers using IQR method or domain-specific rules
- Handles invalid values by replacing with NaN

### 2. Imputation
- Fills missing values using configurable strategies:
  - `median`: Use median value (good for physiological metrics)
  - `mean`: Use mean value
  - `mode`: Use most frequent value
  - `forward_fill`: Propagate last valid value forward
  - `backward_fill`: Propagate next valid value backward
  - `constant`: Fill with constant value (default 0)

### 3. Encoding
- **Target Encoding**: Encode categorical variables using target mean with smoothing
- **One-Hot Encoding**: Create binary columns for low-cardinality categoricals
- **Ordinal Encoding**: Map ordered categories to integers

### 4. Normalization
- **Standardization**: Scale to zero mean and unit variance
- **Time-series Normalization**: Per-individual standardization for temporal data
- **Min-Max Scaling**: Scale to specified range (default [0, 1])

## Configuration Options

### ETLPipelineConfig Parameters

```python
config = ETLPipelineConfig(
    # Cleaning configuration
    outlier_method="iqr",  # or "domain"
    iqr_multiplier=1.5,
    domain_rules={
        "heart_rate": (40, 200),  # (min, max) valid range
        "sleep_hours": (0, 24),
    },
    
    # Imputation configuration
    imputation_strategies={
        "column_name": "strategy",  # median, mean, mode, forward_fill, etc.
    },
    
    # Encoding configuration
    target_column="target",  # For target encoding
    target_encode_columns=["category1", "category2"],
    one_hot_encode_columns=["low_cardinality_cat"],
    ordinal_encode_columns=["ordered_cat"],
    ordinal_ordering={
        "ordered_cat": ["low", "medium", "high"]
    },
    
    # Normalization configuration
    standardize_columns=["feature1", "feature2"],
    normalize_timeseries_columns=["temporal_feature"],
    min_max_columns=["bounded_feature"],
    group_by_column="anonymized_id",
)
```

## Batch Processing

For large datasets, use batch processing:

```python
# Process in batches of 1000 records
df_transformed = pipeline.process_batch(
    df,
    batch_size=1000,
    fit=False  # Set to True for training
)
```

## Parameter Persistence

Save and load fitted pipeline parameters:

```python
# Save pipeline
pipeline.save_pipeline("pipeline_config.json")

# Load pipeline
loaded_pipeline = ETLPipeline.load_pipeline("pipeline_config.json")

# Or manually get/set parameters
params = pipeline.get_fitted_params()
new_pipeline.set_fitted_params(params)
```

## Pipeline Statistics

Access detailed statistics from each stage:

```python
stats = pipeline.get_pipeline_stats()

# Example output:
# {
#     "cleaning": {
#         "duplicates_removed": 5,
#         "outliers_detected": {"heart_rate": 12},
#         "invalid_values_replaced": {"mood_score": 3}
#     },
#     "imputation": {
#         "imputed_values": {"heart_rate": 50, "sleep_hours": 30}
#     },
#     "encoding": {...},
#     "normalization": {...},
#     "fit_transform_time": 2.34,
#     "input_records": 1000,
#     "output_records": 995
# }
```

## Error Handling

The pipeline raises specific exceptions:

- `ValidationError`: Pipeline not fitted before transform
- `DataProcessingError`: Processing failure in any stage

```python
from src.exceptions import ValidationError, DataProcessingError

try:
    df_transformed = pipeline.transform(df)
except ValidationError as e:
    print(f"Pipeline not fitted: {e}")
except DataProcessingError as e:
    print(f"Processing failed: {e}")
```

## Performance

The pipeline is designed to process 1000 records within 60 seconds:

- Efficient batch processing
- Vectorized operations using pandas/numpy
- Minimal memory overhead
- Comprehensive logging for monitoring

## Best Practices

1. **Always fit on training data first**: Use `fit_transform()` on training data, then `transform()` on test/inference data

2. **Configure domain rules**: Provide domain-specific outlier detection rules for better data quality

3. **Choose appropriate imputation strategies**: Use median for physiological metrics, forward-fill for time-series

4. **Use batch processing for large datasets**: Process in batches to manage memory usage

5. **Save fitted parameters**: Persist pipeline parameters for consistent inference

6. **Monitor statistics**: Review pipeline statistics to identify data quality issues

## Example: Complete Workflow

```python
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
import pandas as pd

# 1. Load data
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# 2. Configure pipeline
config = ETLPipelineConfig(
    outlier_method="iqr",
    domain_rules={"heart_rate": (40, 200)},
    imputation_strategies={
        "heart_rate": "median",
        "sleep_hours": "median",
    },
    standardize_columns=["heart_rate", "sleep_hours"],
)

# 3. Create and fit pipeline
pipeline = ETLPipeline(config)
df_train_clean = pipeline.fit_transform(df_train)

# 4. Save pipeline
pipeline.save_pipeline("models/etl_pipeline.json")

# 5. Transform test data
df_test_clean = pipeline.transform(df_test)

# 6. Review statistics
stats = pipeline.get_pipeline_stats()
print(f"Processing time: {stats['fit_transform_time']:.2f}s")
print(f"Records processed: {stats['input_records']} -> {stats['output_records']}")
```

## Integration with ML Pipeline

The ETL pipeline integrates seamlessly with the ML pipeline:

```python
# ETL stage
etl_pipeline = ETLPipeline(etl_config)
df_features = etl_pipeline.fit_transform(df_raw)

# Feature engineering stage
feature_extractor = FeatureEngineeringPipeline(feature_config)
df_engineered = feature_extractor.extract_features(df_features)

# Model training stage
model = train_model(df_engineered)
```

## Troubleshooting

### Issue: Pipeline takes too long
- Use batch processing with smaller batch sizes
- Reduce the number of columns being processed
- Check for very large datasets that need chunking

### Issue: Missing values after imputation
- Check imputation strategy configuration
- Verify column names match exactly
- Review imputation statistics for details

### Issue: Outliers not detected
- Adjust IQR multiplier (lower = more sensitive)
- Provide domain-specific rules for better detection
- Check that columns are numeric type

### Issue: Transform fails after loading
- Ensure all fitted parameters are loaded
- Verify pipeline configuration matches training
- Check that test data has same columns as training
