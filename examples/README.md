# MHRAS Examples

This directory contains example scripts demonstrating how to use the Mental Health Risk Assessment System (MHRAS).

## Available Examples

### screening_example.py

Demonstrates the end-to-end screening workflow including:

1. **Single Individual Screening**
   - Creating a screening request with survey and wearable data
   - Performing the screening
   - Accessing risk scores, recommendations, and explanations
   - Displaying results in a formatted manner

2. **Batch Screening**
   - Processing multiple individuals in parallel
   - Using batch processing for improved performance
   - Analyzing batch results and statistics
   - Identifying high-risk cases

3. **Service Statistics**
   - Accessing model registry information
   - Monitoring human review queue
   - Tracking service performance

## Running the Examples

### Prerequisites

Ensure you have:
- Python 3.8+
- All dependencies installed: `pip install -r requirements.txt`
- Database configured (if using real database)
- Models registered in the ModelRegistry

### Basic Usage

```bash
# Run the screening example
python examples/screening_example.py
```

### Expected Output

The example will display:
- Risk assessment results (score, level, confidence)
- Contributing factors to the risk score
- Personalized resource recommendations
- Interpretability explanations
- Batch processing statistics
- Service monitoring information

## Customizing Examples

### Modify Data

Edit the `survey_data` and `wearable_data` dictionaries in the examples to test different scenarios:

```python
survey_data={
    "responses": {
        "mood": "low",  # Try: "low", "moderate", "high"
        "sleep_quality": "poor",  # Try: "poor", "fair", "good"
        "anxiety_level": "high",  # Try: "low", "moderate", "high"
        "suicidal_ideation": False,
        "self_harm": False
    }
}
```

### Adjust Thresholds

Modify alert and human review thresholds:

```python
service = ScreeningServiceSync(
    alert_threshold=80.0,  # Trigger alerts at 80+
    human_review_threshold=70.0  # Route to review at 70+
)
```

### Enable/Disable Caching

```python
service = ScreeningServiceSync(use_cache=True)  # Enable caching
service = ScreeningServiceSync(use_cache=False)  # Disable caching
```

## Testing Different Scenarios

### Low Risk Scenario
```python
survey_data={
    "responses": {
        "mood": "high",
        "sleep_quality": "good",
        "anxiety_level": "low",
        "suicidal_ideation": False,
        "self_harm": False
    }
}
wearable_data={
    "metrics": {
        "sleep_duration_hours": 7.5,
        "heart_rate_avg": 65,
        "hrv_rmssd": 45
    }
}
```

### High Risk Scenario
```python
survey_data={
    "responses": {
        "mood": "very_low",
        "sleep_quality": "very_poor",
        "anxiety_level": "severe",
        "suicidal_ideation": True,
        "self_harm": True
    }
}
wearable_data={
    "metrics": {
        "sleep_duration_hours": 3.0,
        "heart_rate_avg": 95,
        "hrv_rmssd": 15
    }
}
```

## Performance Testing

To test performance with larger batches:

```python
# Create 100 screening requests
requests = [
    ScreeningRequest(
        anonymized_id=f"perf_test_{i:04d}",
        survey_data={...}
    )
    for i in range(100)
]

# Process with different batch sizes
for batch_size in [5, 10, 20, 50]:
    start = time.time()
    responses = service.screen_batch(requests, batch_size=batch_size)
    elapsed = time.time() - start
    print(f"Batch size {batch_size}: {elapsed:.2f}s ({len(requests)/elapsed:.1f} screenings/sec)")
```

## Troubleshooting

### No Models Available
If you see "No active models found", ensure models are registered:
```python
from src.ml.model_registry import ModelRegistry
registry = ModelRegistry()
print(registry.list_models())
```

### Consent Errors
If consent verification fails, ensure consent records exist in the database or use mock mode.

### Slow Performance
- Enable caching: `use_cache=True`
- Preload models: `service.preload_models()`
- Increase batch size for batch processing
- Ensure database indexes are created

## Additional Resources

- **Usage Guide**: `docs/screening_service_usage.md`
- **Implementation Summary**: `docs/task_13_implementation_summary.md`
- **API Documentation**: `docs/api_usage.md`
- **Tests**: `tests/test_screening_service.py`

## Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the test cases in `tests/`
3. Examine the source code in `src/screening_service.py`
