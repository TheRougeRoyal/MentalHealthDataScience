# Exploratory Data Analysis (EDA) Usage Guide

## Overview

The EDA Module provides automated exploratory data analysis capabilities including statistical summaries, data quality detection, correlation analysis, and visualization generation. It helps data scientists quickly understand datasets and identify potential issues.

## Quick Start

```python
from src.ds.eda import EDAModule
import pandas as pd

# Initialize EDA module
eda = EDAModule()

# Load your data
df = pd.read_csv("patient_assessments.csv")

# Run comprehensive analysis
report = eda.analyze_dataset(
    data=df,
    target_column="risk_level"  # Optional: for supervised learning
)

# Export report
eda.export_report(
    report=report,
    format="html",
    output_path="eda_report.html"
)

print(f"Report generated: eda_report.html")
```

## Core Features

### 1. Summary Statistics

Generate comprehensive statistical summaries:

```python
# Get summary statistics
stats = eda.generate_summary_statistics(df)

# Numerical features
print("Numerical Features:")
for feature, feature_stats in stats['numerical_features'].items():
    print(f"\n{feature}:")
    print(f"  Mean: {feature_stats['mean']:.2f}")
    print(f"  Median: {feature_stats['median']:.2f}")
    print(f"  Std Dev: {feature_stats['std']:.2f}")
    print(f"  Min: {feature_stats['min']:.2f}")
    print(f"  Max: {feature_stats['max']:.2f}")
    print(f"  Q1: {feature_stats['q1']:.2f}")
    print(f"  Q3: {feature_stats['q3']:.2f}")

# Categorical features
print("\nCategorical Features:")
for feature, feature_stats in stats['categorical_features'].items():
    print(f"\n{feature}:")
    print(f"  Unique values: {feature_stats['unique_count']}")
    print(f"  Most common: {feature_stats['mode']}")
    print(f"  Value counts: {feature_stats['value_counts']}")
```

### 2. Data Quality Detection

Automatically detect data quality issues:

```python
# Detect issues
issues = eda.detect_data_quality_issues(df)

# Review issues by severity
for issue in issues:
    print(f"\n[{issue.severity.upper()}] {issue.issue_type}")
    print(f"  Column: {issue.column}")
    print(f"  Description: {issue.description}")
    print(f"  Affected rows: {issue.affected_rows}")
    print(f"  Recommendation: {issue.recommendation}")
```

Common issue types detected:
- **Missing values**: Null or NaN values
- **Outliers**: Statistical outliers using IQR method
- **Duplicates**: Duplicate rows
- **Inconsistent**: Data type inconsistencies
- **High cardinality**: Categorical features with too many unique values
- **Imbalanced**: Target variable imbalance

### 3. Correlation Analysis

Analyze feature correlations:

```python
# Compute correlation matrix
corr_matrix = eda.analyze_correlations(
    data=df,
    method="pearson"  # or "spearman", "kendall"
)

print(corr_matrix)

# Find highly correlated features
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print("\nHighly correlated features (|r| > 0.8):")
for feat1, feat2, corr in high_corr:
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")
```

### 4. Visualization Generation

Generate comprehensive visualizations:

```python
# Generate all visualizations
viz_paths = eda.generate_visualizations(
    data=df,
    output_dir="eda_plots"
)

print("Generated visualizations:")
for path in viz_paths:
    print(f"  - {path}")
```

Generated visualizations include:
- Distribution plots (histograms with KDE)
- Box plots for outlier detection
- Correlation heatmaps
- Missing value patterns
- Feature importance (if target provided)
- Pairwise scatter plots (for key features)

## Comprehensive Analysis

### Full EDA Report

```python
# Run complete analysis
report = eda.analyze_dataset(
    data=df,
    target_column="suicide_risk"
)

# Access report components
print(f"Dataset: {report.dataset_name}")
print(f"Rows: {report.num_rows}")
print(f"Columns: {report.num_columns}")
print(f"Generated: {report.generated_at}")

# Summary statistics
print("\nSummary Statistics:")
print(report.summary_statistics)

# Missing values
print("\nMissing Values:")
for col, count in report.missing_values.items():
    if count > 0:
        pct = (count / report.num_rows) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

# Data quality issues
print(f"\nData Quality Issues: {len(report.quality_issues)}")
for issue in report.quality_issues:
    print(f"  [{issue.severity}] {issue.description}")

# Recommendations
print("\nRecommendations:")
for rec in report.recommendations:
    print(f"  - {rec}")
```

### Export Reports

Export in multiple formats:

```python
# HTML report (interactive)
eda.export_report(
    report=report,
    format="html",
    output_path="reports/eda_report.html"
)

# PDF report
eda.export_report(
    report=report,
    format="pdf",
    output_path="reports/eda_report.pdf"
)

# JSON report (for programmatic access)
eda.export_report(
    report=report,
    format="json",
    output_path="reports/eda_report.json"
)
```

## Use Cases

### 1. Initial Data Exploration

```python
# Quick exploration of new dataset
df = pd.read_csv("new_patient_data.csv")

report = eda.analyze_dataset(df)

# Check for immediate issues
critical_issues = [i for i in report.quality_issues if i.severity == "critical"]
if critical_issues:
    print("⚠️ Critical issues found:")
    for issue in critical_issues:
        print(f"  - {issue.description}")
else:
    print("✓ No critical issues detected")
```

### 2. Pre-Processing Validation

```python
# Before processing
raw_df = load_raw_data()
raw_report = eda.analyze_dataset(raw_df)

# After processing
clean_df = clean_data(raw_df)
clean_report = eda.analyze_dataset(clean_df)

# Compare
print(f"Missing values before: {sum(raw_report.missing_values.values())}")
print(f"Missing values after: {sum(clean_report.missing_values.values())}")
print(f"Issues before: {len(raw_report.quality_issues)}")
print(f"Issues after: {len(clean_report.quality_issues)}")
```

### 3. Feature Selection

```python
# Analyze with target variable
report = eda.analyze_dataset(
    data=df,
    target_column="risk_score"
)

# Check correlations with target
target_corr = report.correlations["risk_score"].sort_values(ascending=False)
print("Features most correlated with target:")
print(target_corr.head(10))

# Identify low-correlation features for removal
low_corr = target_corr[abs(target_corr) < 0.1]
print(f"\nLow correlation features (|r| < 0.1): {len(low_corr)}")
```

### 4. Data Quality Monitoring

```python
# Regular data quality checks
def monitor_data_quality(df, baseline_report):
    current_report = eda.analyze_dataset(df)
    
    # Compare missing values
    missing_increase = (
        sum(current_report.missing_values.values()) -
        sum(baseline_report.missing_values.values())
    )
    
    if missing_increase > 100:
        alert(f"Missing values increased by {missing_increase}")
    
    # Compare quality issues
    critical_issues = [
        i for i in current_report.quality_issues 
        if i.severity == "critical"
    ]
    
    if critical_issues:
        alert(f"Critical data quality issues: {len(critical_issues)}")
    
    return current_report

# Run periodically
baseline = eda.analyze_dataset(baseline_df)
current = monitor_data_quality(new_df, baseline)
```

## Advanced Features

### Custom Analysis

```python
# Analyze specific columns
subset_df = df[["age", "phq9_score", "gad7_score", "risk_level"]]
report = eda.analyze_dataset(subset_df, target_column="risk_level")

# Focus on specific issue types
issues = eda.detect_data_quality_issues(df)
outlier_issues = [i for i in issues if i.issue_type == "outlier"]
missing_issues = [i for i in issues if i.issue_type == "missing"]
```

### Statistical Tests

```python
# The EDA module includes statistical tests in reports
report = eda.analyze_dataset(df, target_column="outcome")

# Normality tests for numerical features
# Chi-square tests for categorical features
# ANOVA for group comparisons
# Results included in report.statistical_tests
```

### Outlier Detection

```python
# Detect outliers using IQR method
issues = eda.detect_data_quality_issues(df)
outlier_issues = [i for i in issues if i.issue_type == "outlier"]

for issue in outlier_issues:
    print(f"Outliers in {issue.column}:")
    print(f"  Count: {issue.affected_rows}")
    print(f"  Recommendation: {issue.recommendation}")

# Access outlier indices
outliers = report.outliers
for column, indices in outliers.items():
    print(f"{column}: {len(indices)} outliers at indices {indices[:5]}...")
```

## Integration with Other Components

### With Data Versioning

```python
from src.ds.data_versioning import DataVersionControl

# Version dataset
dvc = DataVersionControl(storage, db)
version = dvc.register_dataset(df, "patient_data", "database")

# Run EDA
report = eda.analyze_dataset(df)

# Store EDA report with version
metadata = {
    "eda_summary": {
        "num_rows": report.num_rows,
        "num_columns": report.num_columns,
        "quality_issues": len(report.quality_issues),
        "missing_values": sum(report.missing_values.values())
    }
}

# Update version metadata
version.metadata.update(metadata)
```

### With Experiment Tracking

```python
from src.ds.experiment_tracker import ExperimentTracker

# Start experiment
tracker = ExperimentTracker(storage, db)
run = tracker.start_run(experiment_name="data_exploration")

# Run EDA
report = eda.analyze_dataset(df)

# Log EDA metrics
tracker.log_metrics({
    "num_features": report.num_columns,
    "num_samples": report.num_rows,
    "missing_value_pct": sum(report.missing_values.values()) / (report.num_rows * report.num_columns),
    "quality_issues": len(report.quality_issues)
})

# Log EDA report as artifact
eda.export_report(report, format="html", output_path="eda_report.html")
tracker.log_artifact("eda_report.html", artifact_type="report")

tracker.end_run()
```

## Best Practices

### 1. Run EDA Early

```python
# Always run EDA before modeling
df = load_data()
report = eda.analyze_dataset(df, target_column="outcome")

# Review issues before proceeding
if report.quality_issues:
    print("Address these issues before modeling:")
    for issue in report.quality_issues:
        print(f"  - {issue.description}")
```

### 2. Document Findings

```python
# Export comprehensive report
eda.export_report(
    report=report,
    format="html",
    output_path=f"reports/eda_{dataset_name}_{date}.html"
)

# Share with team
# Include in project documentation
```

### 3. Iterate on Data Quality

```python
# Initial EDA
report_v1 = eda.analyze_dataset(raw_df)
print(f"Issues found: {len(report_v1.quality_issues)}")

# Clean data based on recommendations
clean_df = apply_cleaning(raw_df, report_v1.recommendations)

# Re-run EDA
report_v2 = eda.analyze_dataset(clean_df)
print(f"Issues remaining: {len(report_v2.quality_issues)}")

# Iterate until acceptable
```

### 4. Monitor Over Time

```python
# Track data quality metrics over time
def track_quality_metrics(df, date):
    report = eda.analyze_dataset(df)
    
    metrics = {
        "date": date,
        "num_rows": report.num_rows,
        "missing_pct": sum(report.missing_values.values()) / (report.num_rows * report.num_columns),
        "num_issues": len(report.quality_issues),
        "critical_issues": len([i for i in report.quality_issues if i.severity == "critical"])
    }
    
    log_metrics(metrics)
    return metrics

# Run weekly
weekly_metrics = track_quality_metrics(df, "2024-01-15")
```

## CLI Commands

### Run EDA Analysis

```bash
python run_cli.py eda analyze patient_data.csv \
  --target risk_level \
  --output eda_report.html
```

### Generate Specific Visualizations

```bash
python run_cli.py eda visualize patient_data.csv \
  --output-dir plots/ \
  --types distribution,correlation,outliers
```

## API Endpoints

### Run EDA Analysis

```bash
curl -X POST http://localhost:8000/api/v1/eda/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "patient_assessments",
    "target_column": "risk_level"
  }'
```

### Get EDA Report

```bash
curl -X GET http://localhost:8000/api/v1/eda/reports/{report_id}
```

### Download Report

```bash
curl -X GET http://localhost:8000/api/v1/eda/reports/{report_id}/download?format=html \
  -o eda_report.html
```

## Troubleshooting

### Issue: Large dataset takes too long

```python
# Sample large datasets
if len(df) > 100000:
    sample_df = df.sample(n=100000, random_state=42)
    report = eda.analyze_dataset(sample_df)
else:
    report = eda.analyze_dataset(df)
```

### Issue: Memory errors with visualizations

```python
# Limit visualization generation
viz_paths = eda.generate_visualizations(
    data=df,
    output_dir="plots",
    max_features=20  # Only visualize top 20 features
)
```

### Issue: Too many quality issues

```python
# Filter by severity
critical = [i for i in report.quality_issues if i.severity == "critical"]
high = [i for i in report.quality_issues if i.severity == "high"]

# Address critical and high first
priority_issues = critical + high
```

## Performance Tips

1. **Sample large datasets**: Use `df.sample()` for initial exploration
2. **Limit features**: Focus on relevant columns
3. **Disable visualizations**: Skip viz generation for quick analysis
4. **Use appropriate methods**: Spearman for non-linear relationships

```python
# Fast EDA for large datasets
report = eda.analyze_dataset(
    data=df.sample(n=50000),
    target_column="outcome",
    generate_visualizations=False  # Skip viz for speed
)
```

## See Also

- [Data Versioning Usage](data_versioning_usage.md)
- [Experiment Tracking Usage](experiment_tracking_usage.md)
- [Feature Engineering Usage](feature_engineering_usage.md)
