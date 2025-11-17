# Resource Recommendation Engine Usage

## Overview

The Resource Recommendation Engine provides personalized mental health resource recommendations based on individual risk levels and profile characteristics. It matches resources to individuals using a scoring algorithm that considers multiple factors.

## Components

### ResourceCatalog

Manages a catalog of mental health resources with filtering and search capabilities.

```python
from src.recommendations import ResourceCatalog, ResourceType

# Initialize catalog with default resources
catalog = ResourceCatalog()

# Get all resources
all_resources = catalog.get_all_resources()

# Filter by risk level
high_risk_resources = catalog.filter_by_risk_level("high")

# Filter by type
therapy_resources = catalog.filter_by_type(ResourceType.THERAPY)

# Search with multiple filters
resources = catalog.search(
    risk_level="moderate",
    resource_type=ResourceType.SUPPORT_GROUP,
    tags=["depression"]
)
```

### RecommendationEngine

Generates personalized recommendations based on individual profiles.

```python
from src.recommendations import RecommendationEngine, IndividualProfile

# Initialize engine
engine = RecommendationEngine()

# Create individual profile
profile = IndividualProfile(
    anonymized_id="abc123",
    risk_level="high",
    contributing_factors=["depression", "anxiety"],
    age_group="adult",
    has_therapy_history=True,
    prefers_online=True,
    specific_conditions=["depression"]
)

# Get personalized recommendations
recommendations = engine.get_recommendations(profile, max_recommendations=5)

# Simplified interface (risk level only)
recommendations = engine.get_recommendations_by_risk_level(
    risk_level="moderate",
    contributing_factors=["sleep issues"],
    max_recommendations=5
)

# Get as dictionaries for API responses
recommendations_dict = engine.get_recommendations_dict(profile)
```

## Resource Types

- **CRISIS_LINE**: Crisis hotlines and text lines
- **EMERGENCY**: Emergency services (911, etc.)
- **THERAPY**: Therapeutic interventions (CBT, DBT, etc.)
- **SUPPORT_GROUP**: Peer support groups
- **WELLNESS**: Wellness and prevention programs
- **MEDICATION**: Medication management services
- **SELF_HELP**: Self-help resources (apps, online tools)
- **COMMUNITY**: Community mental health centers

## Urgency Levels

- **IMMEDIATE**: Requires immediate attention (crisis situations)
- **SOON**: Should be addressed within days/weeks
- **ROUTINE**: Can be addressed on a routine basis

## Personalization Features

The recommendation engine personalizes recommendations based on:

1. **Risk Level**: Primary factor for filtering resources
2. **Age Group**: Youth, adult, or senior-specific resources
3. **Therapy History**: Adjusts for familiarity with therapy
4. **Medication History**: Prioritizes medication management if relevant
5. **Support System**: Emphasizes support groups if lacking support
6. **Online Preference**: Prioritizes virtual/online resources
7. **Group Preference**: Adjusts for individual vs. group preference
8. **Specific Conditions**: Matches resources to diagnosed conditions
9. **Contributing Factors**: Matches resources to identified risk factors

## Database Integration

### Populating Resources

```python
from src.database.connection import DatabaseConnection
from src.recommendations.populate_resources import populate_resources

# Populate database with default resources
db = DatabaseConnection()
count = populate_resources(db)
print(f"Populated {count} resources")
```

Or run the script directly:
```bash
python -m src.recommendations.populate_resources
```

### Using ResourceRepository

```python
from src.database.connection import DatabaseConnection
from src.database.repositories import ResourceRepository

db = DatabaseConnection()
repo = ResourceRepository(db)

# Get resource by ID
resource = repo.get_by_id("crisis_line_988")

# Get resources by risk level
resources = repo.get_by_risk_level("critical")

# Search with filters
resources = repo.search(
    risk_level="moderate",
    resource_type="therapy",
    tags=["depression"]
)

# Get all active resources
all_resources = repo.get_all_active()
```

### Tracking Recommendations

```python
from src.database.repositories import ResourceRecommendationRepository

rec_repo = ResourceRecommendationRepository(db)

# Create recommendation record
rec_id = rec_repo.create(
    anonymized_id="abc123",
    resource_id="cbt_therapy",
    prediction_id=prediction_id,
    relevance_score=85.5
)

# Get recommendations for individual
recommendations = rec_repo.get_by_anonymized_id("abc123")

# Mark as accessed
rec_repo.mark_accessed(rec_id)

# Get statistics
stats = rec_repo.get_recommendation_stats("cbt_therapy")
print(f"Total: {stats['total_recommendations']}")
print(f"Accessed: {stats['accessed_count']}")
print(f"Avg Score: {stats['avg_relevance_score']}")
```

## API Integration

The recommendation engine integrates with the screening API:

```python
from src.recommendations import RecommendationEngine, IndividualProfile
from src.api.models import ResourceRecommendation

engine = RecommendationEngine()

# Create profile from screening data
profile = IndividualProfile(
    anonymized_id=request.anonymized_id,
    risk_level=risk_score.risk_level.lower(),
    contributing_factors=risk_score.contributing_factors,
    # Add other profile features from request
)

# Get recommendations
resources = engine.get_recommendations(profile)

# Convert to API models
recommendations = [
    ResourceRecommendation(
        resource_type=r.resource_type.value,
        name=r.name,
        description=r.description,
        contact_info=r.contact_info,
        urgency=r.urgency.value,
        eligibility_criteria=r.eligibility_criteria
    )
    for r in resources
]
```

## Adding Custom Resources

### In-Memory (Catalog)

```python
from src.recommendations import ResourceCatalog, Resource, ResourceType, Urgency

catalog = ResourceCatalog()

# Add custom resource
custom_resource = Resource(
    id="custom_resource_1",
    resource_type=ResourceType.THERAPY,
    name="Custom Therapy Program",
    description="Specialized therapy program",
    contact_info="555-1234",
    urgency=Urgency.SOON,
    eligibility_criteria=["Specific criteria"],
    risk_levels=["moderate", "high"],
    tags=["custom", "specialized"],
    priority=75
)

catalog.add_resource(custom_resource)
```

### In Database

```python
from src.database.repositories import ResourceRepository

repo = ResourceRepository(db)

resource_data = {
    'id': 'custom_resource_1',
    'resource_type': 'therapy',
    'name': 'Custom Therapy Program',
    'description': 'Specialized therapy program',
    'contact_info': '555-1234',
    'urgency': 'soon',
    'eligibility_criteria': '["Specific criteria"]',
    'risk_levels': '["moderate", "high"]',
    'tags': '["custom", "specialized"]',
    'priority': 75
}

repo.create(resource_data)
```

## Best Practices

1. **Keep Resources Updated**: Regularly review and update resource information
2. **Monitor Usage**: Track which resources are recommended and accessed
3. **Personalize When Possible**: Provide profile information for better recommendations
4. **Prioritize Crisis Resources**: Ensure immediate resources are always available for critical cases
5. **Validate Contact Info**: Ensure all contact information is current and accurate
6. **Consider Accessibility**: Include resources with various access methods (phone, text, online)
7. **Track Outcomes**: Monitor recommendation effectiveness and adjust priorities

## Example: Complete Workflow

```python
from src.database.connection import DatabaseConnection
from src.database.repositories import (
    ResourceRepository,
    ResourceRecommendationRepository,
    PredictionRepository
)
from src.recommendations import RecommendationEngine, IndividualProfile

# Initialize
db = DatabaseConnection()
engine = RecommendationEngine()
resource_repo = ResourceRepository(db)
rec_repo = ResourceRecommendationRepository(db)

# Create profile
profile = IndividualProfile(
    anonymized_id="user123",
    risk_level="high",
    contributing_factors=["depression", "sleep issues"],
    age_group="adult",
    has_therapy_history=False,
    prefers_online=True,
    specific_conditions=["depression"]
)

# Get recommendations
recommendations = engine.get_recommendations(profile, max_recommendations=5)

# Store recommendation records
for resource in recommendations:
    rec_repo.create(
        anonymized_id=profile.anonymized_id,
        resource_id=resource.id,
        prediction_id=prediction_id,  # From screening
        relevance_score=None  # Could calculate and store
    )

# Later: track access
# rec_repo.mark_accessed(recommendation_id)
```

## Configuration

Resources can be configured through:

1. **Default Catalog**: Built-in resources in `ResourceCatalog`
2. **Database**: Persistent storage with CRUD operations
3. **Environment Variables**: Configure database connection
4. **Custom Initialization**: Load resources from external sources

## Troubleshooting

### No Recommendations Returned

- Check that resources exist for the specified risk level
- Verify catalog is initialized properly
- Check database connection if using persistent storage

### Unexpected Recommendations

- Review relevance scoring algorithm
- Check profile features are set correctly
- Verify resource tags and metadata

### Database Errors

- Ensure migrations have been run
- Check database connection configuration
- Verify JSON fields are properly formatted
