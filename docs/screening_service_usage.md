# Screening Service Usage Guide

## Overview

The `ScreeningService` is the end-to-end orchestration layer for the Mental Health Risk Assessment System (MHRAS). It integrates all components including data validation, consent verification, anonymization, ETL processing, feature engineering, ML inference, interpretability, recommendations, and governance.

## Features

- **Async Processing**: Built with async/await for high performance
- **Comprehensive Error Handling**: Graceful error handling with detailed logging
- **Caching**: Optional caching for frequently accessed data
- **Batch Processing**: Process multiple screenings in parallel
- **Audit Logging**: Complete audit trail for compliance
- **Human Review Queue**: Automatic routing of high-risk cases
- **Performance Optimization**: Sub-5-second response time target

## Basic Usage

### Synchronous API

```python
from src.screening_service import ScreeningServiceSync, ScreeningRequest
from datetime import datetime

# Initialize service
service = ScreeningServiceSync()

# Create screening request
request = ScreeningRequest(
    anonymized_id="abc123def456",
    survey_data={
        "responses": {
            "mood": "low",
            "sleep_quality": "poor",
            "suicidal_ideation": False
        },
        "timestamp": datetime.utcnow().isoformat()
    },
    wearable_data={
        "metrics": {
            "sleep_duration_hours": 4.5,
            "heart_rate_avg": 85,
            "hrv_rmssd": 25
        },
        "data_quality": {
            "completeness_percent": 95,
            "wear_time_minutes": 1200
        }
    },
    user_id="clinician_001"
)

# Perform screening
response = service.screen_individual(request)

# Access results
print(f"Risk Score: {response.risk_score}")
print(f"Risk Level: {response.risk_level}")
print(f"Confidence: {response.confidence}")
print(f"Requires Human Review: {response.requires_human_review}")
print(f"Alert Triggered: {response.alert_triggered}")
print(f"Processing Time: {response.processing_time_seconds}s")

# Access recommendations
for rec in response.recommendations:
    print(f"- {rec['name']}: {rec['description']}")

# Access explanations
print(f"Top Contributing Factors: {response.contributing_factors}")
print(f"Counterfactual: {response.explanations['counterfactual']}")
```

### Asynchronous API

```python
import asyncio
from src.screening_service import ScreeningService, ScreeningRequest

async def main():
    # Initialize service
    service = ScreeningService()
    
    # Create request
    request = ScreeningRequest(
        anonymized_id="abc123def456",
        survey_data={...},
        wearable_data={...}
    )
    
    # Perform screening
    response = await service.screen_individual(request)
    
    print(f"Risk Score: {response.risk_score}")

# Run async function
asyncio.run(main())
```

## Batch Processing

Process multiple individuals efficiently:

```python
from src.screening_service import ScreeningServiceSync, ScreeningRequest

service = ScreeningServiceSync()

# Create multiple requests
requests = [
    ScreeningRequest(anonymized_id=f"user_{i}", survey_data={...})
    for i in range(100)
]

# Process in batches
responses = service.screen_batch(requests, batch_size=10)

# Analyze results
high_risk_count = sum(1 for r in responses if r.risk_level in ["high", "critical"])
print(f"High risk individuals: {high_risk_count}/{len(responses)}")
```

## Performance Optimization

### Preload Models

Preload models into memory for faster inference:

```python
service = ScreeningServiceSync()

# Preload all active models
service.preload_models()

# Now screening will be faster
response = service.screen_individual(request)
```

### Enable Caching

```python
# Enable caching (default is True)
service = ScreeningServiceSync(use_cache=True)

# Clear cache when needed
service.clear_cache()
```

### Database Optimization

Ensure database indexes are created via migrations:

```bash
python -m src.database.migration_runner
```

## Configuration

### Custom ETL Configuration

```python
from src.processing.etl_pipeline import ETLPipelineConfig
from src.screening_service import ScreeningServiceSync

# Create custom ETL config
etl_config = ETLPipelineConfig(
    outlier_method="iqr",
    iqr_multiplier=1.5,
    imputation_strategies={
        "forward_fill": ["sleep_duration", "heart_rate"],
        "median": ["activity_count"],
        "mode": ["mood_category"]
    },
    standardize_columns=["hrv_rmssd", "sleep_efficiency"],
    group_by_column="anonymized_id"
)

# Initialize service with custom config
service = ScreeningServiceSync(etl_config=etl_config)
```

### Custom Thresholds

```python
# Set custom alert and human review thresholds
service = ScreeningServiceSync(
    alert_threshold=80.0,  # Trigger alerts at 80+
    human_review_threshold=70.0  # Route to human review at 70+
)
```

## Monitoring and Statistics

### Get Service Statistics

```python
stats = service.get_screening_statistics()

print(f"Active Models: {stats['model_registry']['active_models']}")
print(f"Pending Reviews: {stats['review_queue']['pending_count']}")
print(f"Overdue Reviews: {stats['review_queue']['overdue_count']}")
```

### Access Human Review Queue

```python
from src.governance.human_review_queue import HumanReviewQueue

# Access the queue directly
queue = service.service.human_review_queue

# Get pending cases
pending_cases = queue.get_pending_cases(limit=10)

for case in pending_cases:
    print(f"Case {case.case_id}: Risk {case.risk_score}, Priority {case.priority}")

# Get queue statistics
queue_stats = queue.get_queue_statistics()
print(f"Total cases: {queue_stats['total_cases']}")
print(f"Average review time: {queue_stats['average_review_time_hours']}h")
```

## Error Handling

The service raises specific exceptions for different error types:

```python
from src.exceptions import (
    ValidationError,
    ConsentError,
    DataProcessingError,
    InferenceError,
    ScreeningError
)

try:
    response = service.screen_individual(request)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Details: {e.details}")
except ConsentError as e:
    print(f"Consent verification failed: {e.message}")
except DataProcessingError as e:
    print(f"Data processing failed: {e.message}")
except InferenceError as e:
    print(f"Model inference failed: {e.message}")
except ScreeningError as e:
    print(f"Screening failed: {e.message}")
```

## Audit Trail

All screening activities are automatically logged:

```python
from src.governance.audit_logger import AuditLogger
from datetime import datetime, timedelta

# Access audit logger
audit_logger = service.service.audit_logger

# Get audit trail for an individual
trail = audit_logger.get_audit_trail(
    anonymized_id="abc123def456",
    start_date=datetime.utcnow() - timedelta(days=30)
)

for entry in trail:
    print(f"{entry['timestamp']}: {entry['event_type']}")

# Generate audit report
report = audit_logger.generate_audit_report(
    start_date=datetime.utcnow() - timedelta(days=7),
    end_date=datetime.utcnow()
)

print(f"Total screenings: {report['screening_statistics']['total_screenings']}")
print(f"Alerts triggered: {report['screening_statistics']['alerts_triggered']}")
```

## Performance Requirements

The screening service is designed to meet these performance targets:

- **End-to-end screening**: < 5 seconds per individual
- **Validation**: < 100ms per record
- **Feature engineering**: < 2 seconds per individual
- **Model inference**: < 2 seconds for ensemble
- **Explanation generation**: < 3 seconds

Monitor performance with:

```python
response = service.screen_individual(request)
print(f"Processing time: {response.processing_time_seconds}s")

if response.processing_time_seconds > 5.0:
    print("WARNING: Exceeded 5s target")
```

## Best Practices

1. **Use batch processing** for multiple screenings to improve throughput
2. **Preload models** at service startup for faster inference
3. **Enable caching** for production deployments
4. **Monitor queue statistics** to ensure timely human reviews
5. **Review audit logs** regularly for compliance
6. **Set appropriate thresholds** based on your clinical workflow
7. **Handle errors gracefully** with proper exception handling
8. **Clear cache periodically** to prevent memory growth

## Integration with API

The screening service integrates with the FastAPI endpoints:

```python
# In src/api/endpoints.py
from src.screening_service import ScreeningServiceSync, ScreeningRequest

service = ScreeningServiceSync()

@app.post("/screen")
async def screen_endpoint(request: ScreeningRequestModel):
    screening_request = ScreeningRequest(
        anonymized_id=request.anonymized_id,
        survey_data=request.survey_data,
        wearable_data=request.wearable_data,
        emr_data=request.emr_data,
        user_id=request.user_id
    )
    
    response = service.screen_individual(screening_request)
    
    return response.to_dict()
```

## Troubleshooting

### Slow Performance

- Check if models are preloaded: `service.preload_models()`
- Verify database indexes are created
- Enable caching: `use_cache=True`
- Reduce batch size if memory constrained

### High Memory Usage

- Clear cache periodically: `service.clear_cache()`
- Reduce cache size limits in the code
- Process smaller batches

### Missing Predictions

- Verify models are registered in ModelRegistry
- Check that active models exist: `model_registry.get_active_models()`
- Review error logs for inference failures

### Consent Errors

- Verify consent records exist in database
- Check consent expiration dates
- Ensure data types match consent grants
