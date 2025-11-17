# Feature Engineering Pipeline Usage Guide

This document describes how to use the feature engineering components in the Mental Health Risk Assessment System (MHRAS).

## Overview

The feature engineering pipeline extracts behavioral, sentiment, physiological, and adherence features from multi-modal health data. It supports parallel processing and handles missing data gracefully.

## Components

### 1. BehavioralFeatureExtractor

Extracts features from activity and social interaction data.

**Features extracted:**
- Activity counts and frequencies (7-day and 30-day windows)
- Routine consistency scores (based on entropy measures)
- Social interaction metrics (frequency and duration)

**Example usage:**

```python
from src.ml import BehavioralFeatureExtractor
import pandas as pd

extractor = BehavioralFeatureExtractor()

# Prepare activity data
activity_df = pd.DataFrame({
    'anonymized_id': ['user1', 'user1', 'user2'],
    'activity_type': ['exercise', 'social', 'work'],
    'timestamp': pd.date_range('2025-01-01', periods=3, freq='D')
})

# Extract activity features
activity_features = extractor.extract_activity_features(activity_df)

# Compute routine consistency
routine_features = extractor.compute_routine_consistency(activity_df)

# Calculate social interaction metrics
social_features = extractor.calculate_social_interaction_metrics(activity_df)
```

### 2. SentimentAnalyzer

Analyzes sentiment from text responses and detects crisis keywords.

**Features extracted:**
- Valence (pleasure/displeasure)
- Arousal (activation level)
- Dominance (sense of control)
- Crisis keyword flags

**Example usage:**

```python
from src.ml import SentimentAnalyzer

# Initialize with lexicon-based approach (faster)
analyzer = SentimentAnalyzer(use_transformers=False)

# Or use transformer-based approach (more accurate but slower)
# analyzer = SentimentAnalyzer(use_transformers=True)

# Analyze single text
text = "I feel anxious and worried today"
sentiment = analyzer.analyze_sentiment(text)
print(f"Valence: {sentiment.valence}, Arousal: {sentiment.arousal}")

# Detect crisis keywords
keywords = analyzer.detect_crisis_keywords(text)
if keywords:
    print(f"Crisis keywords detected: {keywords}")

# Batch analysis
text_df = pd.DataFrame({
    'anonymized_id': ['user1', 'user2'],
    'text_response': ['I feel great', 'I feel hopeless']
})
result = analyzer.analyze_batch(text_df)
```

### 3. PhysiologicalFeatureExtractor

Extracts features from wearable device data (sleep, HRV, activity).

**Features extracted:**
- Sleep duration, efficiency, and interruptions
- Heart rate variability (RMSSD, SDNN, pNN50)
- Activity intensity zones and step counts

**Example usage:**

```python
from src.ml import PhysiologicalFeatureExtractor

extractor = PhysiologicalFeatureExtractor()

# Sleep features
sleep_df = pd.DataFrame({
    'anonymized_id': ['user1', 'user1'],
    'sleep_start': pd.to_datetime(['2025-01-01 22:00', '2025-01-02 22:30']),
    'sleep_end': pd.to_datetime(['2025-01-02 06:00', '2025-01-03 06:30']),
    'interruptions': [2, 1]
})
sleep_features = extractor.extract_sleep_features(sleep_df)

# HRV features
hrv_df = pd.DataFrame({
    'anonymized_id': ['user1'] * 10,
    'heart_rate': [70, 72, 68, 71, 69, 73, 70, 72, 71, 69],
    'timestamp': pd.date_range('2025-01-01', periods=10, freq='min')
})
hrv_features = extractor.compute_hrv_metrics(hrv_df)

# Activity intensity
activity_df = pd.DataFrame({
    'anonymized_id': ['user1'] * 5,
    'heart_rate': [120, 130, 125, 135, 128],
    'steps': [5000, 6000, 5500, 7000, 6500],
    'timestamp': pd.date_range('2025-01-01', periods=5, freq='H')
})
activity_features = extractor.calculate_activity_intensity(activity_df)

# Extract all physiological features at once
all_features = extractor.extract_all_physiological_features(
    sleep_df=sleep_df,
    hrv_df=hrv_df,
    activity_df=activity_df
)
```

### 4. AdherenceTracker

Tracks therapy adherence and engagement.

**Features extracted:**
- Adherence rates (overall and by intervention type)
- Missed session patterns
- Engagement scores from interaction logs

**Example usage:**

```python
from src.ml import AdherenceTracker

tracker = AdherenceTracker()

# Adherence rates
adherence_df = pd.DataFrame({
    'anonymized_id': ['user1', 'user2'],
    'scheduled_sessions': [10, 8],
    'completed_sessions': [8, 6],
    'intervention_type': ['therapy', 'therapy']
})
adherence_features = tracker.calculate_adherence_rate(adherence_df)

# Missed sessions
sessions_df = pd.DataFrame({
    'anonymized_id': ['user1'] * 5,
    'session_date': pd.date_range('2025-01-01', periods=5, freq='D'),
    'status': ['completed', 'missed', 'completed', 'missed', 'completed']
})
missed_features = tracker.flag_missed_sessions(sessions_df)

# Engagement scores
engagement_df = pd.DataFrame({
    'anonymized_id': ['user1'] * 10,
    'interaction_count': [5, 3, 7, 4, 6, 5, 8, 4, 6, 5],
    'session_duration_minutes': [20, 15, 30, 25, 20, 22, 28, 18, 24, 21],
    'timestamp': pd.date_range('2025-01-01', periods=10, freq='D')
})
engagement_features = tracker.compute_engagement_score(engagement_df)
```

### 5. FeatureEngineeringPipeline

Unified pipeline that orchestrates all feature extractors.

**Example usage:**

```python
from src.ml import FeatureEngineeringPipeline

# Initialize pipeline with parallel processing
pipeline = FeatureEngineeringPipeline(
    use_parallel=True,
    max_workers=4,
    use_transformers_sentiment=False
)

# Extract all features at once
features = pipeline.extract_features(
    behavioral_df=behavioral_df,
    text_df=text_df,
    sleep_df=sleep_df,
    hrv_df=hrv_df,
    activity_df=activity_df,
    adherence_df=adherence_df,
    sessions_df=sessions_df,
    engagement_df=engagement_df,
    validate=True
)

# Get pipeline statistics
stats = pipeline.get_pipeline_stats()
print(f"Processing time: {stats['total_time_seconds']:.2f}s")
print(f"Time per individual: {stats['time_per_individual']:.3f}s")

# Get feature summary
summary = pipeline.get_feature_summary(features)
print(f"Total features: {summary['num_features']}")
print(f"Behavioral features: {summary['behavioral_features']}")
print(f"Sentiment features: {summary['sentiment_features']}")
print(f"Physiological features: {summary['physiological_features']}")
print(f"Adherence features: {summary['adherence_features']}")
```

## Performance Considerations

### Target Performance
- Feature extraction should complete within **2 seconds per individual**
- Sentiment analysis should complete within **500ms per text response**

### Optimization Tips

1. **Use parallel processing** for large datasets:
   ```python
   pipeline = FeatureEngineeringPipeline(use_parallel=True, max_workers=4)
   ```

2. **Use lexicon-based sentiment analysis** for faster processing:
   ```python
   pipeline = FeatureEngineeringPipeline(use_transformers_sentiment=False)
   ```

3. **Batch process** multiple individuals together rather than one at a time

4. **Handle missing data gracefully** - all extractors skip missing data sources without errors

## Data Requirements

### Required Columns

**Behavioral Data:**
- `anonymized_id`: Individual identifier
- `activity_type`: Type of activity
- `timestamp`: Activity timestamp

**Text Data:**
- `anonymized_id`: Individual identifier
- `text_response`: Text to analyze

**Sleep Data:**
- `anonymized_id`: Individual identifier
- `sleep_start`: Sleep start time
- `sleep_end`: Sleep end time
- `interruptions` (optional): Number of interruptions

**HRV Data:**
- `anonymized_id`: Individual identifier
- `heart_rate` or `rr_intervals`: Heart rate metrics
- `timestamp`: Measurement timestamp

**Activity Data:**
- `anonymized_id`: Individual identifier
- `heart_rate`: Heart rate during activity
- `steps` (optional): Step count
- `timestamp`: Measurement timestamp

**Adherence Data:**
- `anonymized_id`: Individual identifier
- `scheduled_sessions`: Number of scheduled sessions
- `completed_sessions`: Number of completed sessions

**Sessions Data:**
- `anonymized_id`: Individual identifier
- `session_date`: Session date
- `status`: Session status (e.g., 'completed', 'missed')

**Engagement Data:**
- `anonymized_id`: Individual identifier
- `interaction_count` (optional): Number of interactions
- `session_duration_minutes` (optional): Session duration
- `timestamp`: Interaction timestamp

## Error Handling

All feature extractors handle errors gracefully:

- **Empty DataFrames**: Return empty results without errors
- **Missing columns**: Skip features that require missing columns
- **Invalid data**: Log warnings and continue processing
- **Missing data sources**: Process available sources only

Example:
```python
# Pipeline handles missing data sources gracefully
features = pipeline.extract_features(
    behavioral_df=behavioral_df,  # Provided
    text_df=None,                 # Not available
    sleep_df=None,                # Not available
    # ... other sources
)
# Returns features from behavioral data only
```

## Testing

Run tests with:
```bash
pytest tests/test_feature_engineering.py -v
```

## Dependencies

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- vaderSentiment >= 3.3.2 (for lexicon-based sentiment)
- transformers >= 4.30.0 (for transformer-based sentiment, optional)

Install with:
```bash
pip install -r requirements.txt
```
