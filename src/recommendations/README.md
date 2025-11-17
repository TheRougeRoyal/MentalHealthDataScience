# Resource Recommendation Engine

## Overview

The Resource Recommendation Engine provides personalized mental health resource recommendations based on individual risk levels and profile characteristics. It implements Requirement 1.4 from the MHRAS requirements document.

## Components

### 1. ResourceCatalog (`resource_catalog.py`)

Manages a catalog of 15+ default mental health resources with filtering and search capabilities.

**Features:**
- Pre-populated with crisis lines, therapy options, support groups, wellness programs, and more
- Filter by risk level, resource type, tags, and eligibility criteria
- Search with multiple filters
- Priority-based sorting

**Resource Types:**
- CRISIS_LINE: Crisis hotlines and text lines
- EMERGENCY: Emergency services (911)
- THERAPY: Various therapeutic interventions (CBT, DBT, IOP, etc.)
- SUPPORT_GROUP: Peer support groups
- WELLNESS: Wellness and prevention programs
- MEDICATION: Medication management services
- SELF_HELP: Self-help resources (apps, online tools)
- COMMUNITY: Community mental health centers

### 2. RecommendationEngine (`recommendation_engine.py`)

Generates personalized recommendations using a relevance scoring algorithm.

**Personalization Factors:**
- Risk level (primary filter)
- Age group (youth, adult, senior)
- Therapy history
- Medication history
- Support system availability
- Online preference
- Group vs. individual preference
- Specific diagnosed conditions
- Contributing risk factors

**Scoring Algorithm:**
- Base priority score from resource
- +20 for exact risk level match
- +15 for age-appropriate resources
- +10-25 for condition/factor matching
- +10-20 for preference alignment
- +15 for support system needs

### 3. Database Integration

**Tables:**
- `resources`: Persistent storage of resources
- `resource_recommendations`: Tracking of recommendations made

**Repositories:**
- `ResourceRepository`: CRUD operations for resources
- `ResourceRecommendationRepository`: Track recommendations and access

**Migration:** `002_resources_schema.sql`

### 4. API Integration

Updated `src/api/endpoints.py` to use the recommendation engine in the screening workflow.

**Changes:**
- Replaced hardcoded recommendations with engine-based recommendations
- Added support for profile-based personalization
- Integrated with existing screening endpoint

## Usage

### Basic Usage

```python
from src.recommendations import RecommendationEngine, IndividualProfile

# Initialize engine
engine = RecommendationEngine()

# Create profile
profile = IndividualProfile(
    anonymized_id="user123",
    risk_level="high",
    contributing_factors=["depression", "anxiety"],
    prefers_online=True,
    specific_conditions=["depression"]
)

# Get recommendations
recommendations = engine.get_recommendations(profile, max_recommendations=5)
```

### Simplified Interface

```python
# Get recommendations by risk level only
recommendations = engine.get_recommendations_by_risk_level(
    risk_level="moderate",
    contributing_factors=["sleep issues"],
    max_recommendations=5
)
```

### Database Population

```bash
# Populate database with default resources
python -m src.recommendations.populate_resources
```

## Testing

Comprehensive test suite in `tests/test_recommendations.py`:
- Catalog initialization and filtering
- Recommendation generation for all risk levels
- Personalization features
- Relevance scoring
- API integration

## Documentation

See `docs/recommendation_engine_usage.md` for detailed usage examples and best practices.

## Files Created

1. `src/recommendations/__init__.py` - Module exports
2. `src/recommendations/resource_catalog.py` - Resource catalog and data structures
3. `src/recommendations/recommendation_engine.py` - Recommendation engine with scoring
4. `src/recommendations/populate_resources.py` - Database population utility
5. `src/recommendations/README.md` - This file
6. `src/database/migrations/002_resources_schema.sql` - Database schema
7. `src/database/repositories.py` - Added ResourceRepository and ResourceRecommendationRepository
8. `docs/recommendation_engine_usage.md` - Comprehensive usage documentation
9. `tests/test_recommendations.py` - Test suite

## Requirements Satisfied

✅ **Requirement 1.4**: "THE MHRAS SHALL provide resource recommendations matched to the calculated risk level and individual profile"

The implementation:
- Matches resources to risk levels (low, moderate, high, critical)
- Personalizes based on individual profile features
- Provides 5+ relevant recommendations per screening
- Includes contact information and eligibility criteria
- Prioritizes resources by relevance and urgency
