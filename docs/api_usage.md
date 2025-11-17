# MHRAS API Documentation

## Overview

The Mental Health Risk Assessment System (MHRAS) API provides RESTful endpoints for mental health risk screening, prediction generation, and interpretable explanations. The API is built with FastAPI and follows OpenAPI 3.0 specifications.

**Version:** 1.0.0  
**Base URL (Development):** `http://localhost:8000`  
**Base URL (Production):** `https://api.mhras.example.com`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Access interactive documentation
open http://localhost:8000/docs
```

## Authentication

All protected endpoints require JWT token authentication using the Bearer token scheme.

### Authentication Requirements

- **Token Type:** JWT (JSON Web Token)
- **Header:** `Authorization: Bearer <token>`
- **Token Expiry:** 60 minutes (default)
- **Supported Roles:** `admin`, `clinician`, `data_scientist`, `auditor`

### Generate Authentication Token

```python
from src.api.auth import authenticator

# Generate token for a user
token = authenticator.generate_token(
    user_id="user123",
    role="clinician",
    expiry_minutes=60
)

print(f"Token: {token}")
```

### Using Tokens in Requests

**cURL Example:**
```bash
curl -X POST http://localhost:8000/screen \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d @request.json
```

**Python Example:**
```python
import requests

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/screen",
    json=request_data,
    headers=headers
)
```

### Token Validation

Tokens are validated on each request:
- Signature verification using secret key
- Expiration time check
- Revocation status check (if applicable)

### Token Revocation

```python
# Revoke a token
authenticator.revoke_token(token)
```

## API Endpoints

### Root Endpoint

#### GET `/`

Get API service information.

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "service": "Mental Health Risk Assessment System",
  "version": "1.0.0",
  "status": "operational"
}
```

---

### Health Check

#### GET `/health`

Check API health status. Used for monitoring and load balancer health checks.

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": 1700220600.123
}
```

---

### Screen Individual

#### POST `/screen`

Screen an individual and generate a comprehensive risk assessment including risk score, resource recommendations, and interpretable explanations.

**Authentication:** Required (Bearer token)

**Performance:** < 5 seconds (requirement)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `anonymized_id` | string | Yes | Anonymized identifier (1-64 chars) |
| `survey_data` | object | No | Survey response data (e.g., PHQ-9, GAD-7) |
| `wearable_data` | object | No | Wearable device metrics (e.g., sleep, HRV) |
| `emr_data` | object | No | Electronic medical records data |
| `consent_verified` | boolean | Yes | Must be `true` to process data |
| `timestamp` | string | No | ISO 8601 timestamp (defaults to current time) |

**Request Example:**
```json
{
  "anonymized_id": "a1b2c3d4e5f6789012345678",
  "survey_data": {
    "phq9_score": 15,
    "phq9_items": [2, 2, 2, 1, 2, 1, 2, 1, 2],
    "gad7_score": 12,
    "gad7_items": [2, 1, 2, 2, 1, 2, 2],
    "recent_stressors": ["work", "relationships"],
    "support_system_rating": 3
  },
  "wearable_data": {
    "avg_heart_rate": 75,
    "resting_heart_rate": 62,
    "hrv_rmssd": 35.2,
    "sleep_hours": 6.5,
    "sleep_efficiency": 0.82,
    "sleep_interruptions": 3,
    "steps_per_day": 5200,
    "active_minutes": 25
  },
  "emr_data": {
    "diagnosis_codes": ["F32.1", "F41.1"],
    "medications": ["sertraline 50mg"],
    "recent_visits": 2,
    "therapy_sessions_attended": 4,
    "therapy_sessions_scheduled": 6
  },
  "consent_verified": true,
  "timestamp": "2025-11-17T10:30:00Z"
}
```

**Response (200 OK):**
```json
{
  "risk_score": {
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "score": 68.5,
    "risk_level": "high",
    "confidence": 0.85,
    "contributing_factors": [
      "Elevated PHQ-9 score indicating moderate-severe depression",
      "Poor sleep quality with reduced duration and efficiency",
      "Decreased physical activity levels",
      "Suboptimal therapy adherence",
      "Limited social support system"
    ],
    "timestamp": "2025-11-17T10:30:05Z"
  },
  "recommendations": [
    {
      "resource_type": "crisis_line",
      "name": "National Suicide Prevention Lifeline",
      "description": "24/7 crisis support and suicide prevention services",
      "contact_info": "Call or text 988",
      "urgency": "immediate",
      "eligibility_criteria": ["Available to all individuals in crisis"]
    },
    {
      "resource_type": "therapy",
      "name": "Cognitive Behavioral Therapy (CBT)",
      "description": "Evidence-based therapy for depression and anxiety",
      "contact_info": "Contact your healthcare provider for referral",
      "urgency": "soon",
      "eligibility_criteria": [
        "Diagnosed depression or anxiety",
        "Insurance coverage or self-pay"
      ]
    },
    {
      "resource_type": "support_group",
      "name": "Depression and Bipolar Support Alliance (DBSA)",
      "description": "Peer-led support groups for mood disorders",
      "contact_info": "Visit dbsalliance.org to find local groups",
      "urgency": "routine",
      "eligibility_criteria": ["Open to all with mood disorders"]
    }
  ],
  "explanations": {
    "top_features": [
      ["phq9_score", 0.25],
      ["sleep_quality_index", -0.18],
      ["therapy_adherence_rate", -0.15],
      ["social_interaction_frequency", -0.12],
      ["hrv_rmssd", -0.10]
    ],
    "counterfactual": "If sleep quality improved by 20% (7.8 hours with 90% efficiency) and therapy adherence increased to 100%, the predicted risk level would decrease to moderate (score: 45-50).",
    "rule_approximation": "IF phq9_score > 15 AND sleep_hours < 7 AND therapy_adherence < 0.7 THEN risk = high",
    "clinical_interpretation": "The elevated risk is primarily driven by depressive symptoms (PHQ-9: 15, moderate-severe range) combined with sleep disturbance and suboptimal engagement with treatment. The individual shows physiological markers of stress (reduced HRV) and behavioral indicators of withdrawal (decreased activity, limited social interaction)."
  },
  "requires_human_review": true,
  "alert_triggered": false
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `risk_score` | object | Complete risk assessment |
| `risk_score.score` | float | Risk score 0-100 |
| `risk_score.risk_level` | string | `low`, `moderate`, `high`, or `critical` |
| `risk_score.confidence` | float | Model confidence 0-1 |
| `recommendations` | array | Personalized resource recommendations |
| `explanations` | object | Interpretable model explanations |
| `requires_human_review` | boolean | Whether case needs clinician review |
| `alert_triggered` | boolean | Whether critical alert was sent |

**Status Codes:**

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Screening completed successfully |
| 400 | Validation Error | Invalid data format or missing required fields |
| 401 | Authentication Error | Invalid or expired token |
| 403 | Consent Error | Consent not verified or expired |
| 500 | Processing Error | Internal error during ETL or feature engineering |
| 503 | Service Unavailable | Model inference failed |
| 504 | Timeout | Request exceeded 5-second limit |

**Error Response Example (400):**
```json
{
  "error": "ValidationError",
  "message": "Survey data validation failed",
  "details": {
    "validation_errors": [
      {
        "field": "survey_data.phq9_score",
        "message": "Value must be between 0 and 27",
        "type": "value_error",
        "input": 35
      }
    ]
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

**Error Response Example (403):**
```json
{
  "error": "ConsentError",
  "message": "Consent verification failed: Consent expired",
  "details": {
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "consent_status": "expired",
    "expired_at": "2025-10-15T00:00:00Z"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

---

### Get Risk Score

#### GET `/risk-score/{anonymized_id}`

Retrieve the most recent risk score for an individual from the database.

**Authentication:** Required (Bearer token)

**Performance:** < 1 second

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `anonymized_id` | string | Yes | Anonymized identifier |

**Request Example:**
```bash
GET /risk-score/a1b2c3d4e5f6789012345678
Authorization: Bearer YOUR_TOKEN
```

**Response (200 OK):**
```json
{
  "risk_score": {
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "score": 68.5,
    "risk_level": "high",
    "confidence": 0.85,
    "contributing_factors": [
      "Elevated PHQ-9 score indicating moderate-severe depression",
      "Poor sleep quality with reduced duration and efficiency"
    ],
    "timestamp": "2025-11-17T10:30:05Z"
  },
  "found": true
}
```

**Response (404 Not Found):**
```json
{
  "error": "NotFoundError",
  "message": "No risk score found for a1b2c3d4e5f6789012345678",
  "details": {
    "anonymized_id": "a1b2c3d4e5f6789012345678"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success - Risk score found |
| 401 | Authentication Error |
| 404 | Risk score not found |
| 500 | Internal Server Error |

---

### Explain Prediction

#### POST `/explain`

Generate detailed interpretable explanations for a specific prediction.

**Authentication:** Required (Bearer token)

**Performance:** < 3 seconds

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `anonymized_id` | string | Yes | Anonymized identifier |
| `prediction_id` | string | No | Specific prediction ID (uses most recent if omitted) |

**Request Example:**
```json
{
  "anonymized_id": "a1b2c3d4e5f6789012345678",
  "prediction_id": "pred_20251117_103005_abc123"
}
```

**Response (200 OK):**
```json
{
  "anonymized_id": "a1b2c3d4e5f6789012345678",
  "explanations": {
    "top_features": [
      ["phq9_score", 0.25],
      ["sleep_quality_index", -0.18],
      ["therapy_adherence_rate", -0.15],
      ["social_interaction_frequency", -0.12],
      ["hrv_rmssd", -0.10],
      ["activity_level_7day", -0.08],
      ["gad7_score", 0.07],
      ["medication_adherence", -0.06],
      ["recent_stressor_count", 0.05],
      ["support_system_rating", -0.04]
    ],
    "counterfactual": "If sleep quality improved by 20% (7.8 hours with 90% efficiency) and therapy adherence increased to 100%, the predicted risk level would decrease to moderate (score: 45-50). Additionally, increasing social interactions by 2-3 per week would further reduce risk.",
    "rule_approximation": "IF phq9_score > 15 AND sleep_hours < 7 AND therapy_adherence < 0.7 THEN risk = high\nELSE IF phq9_score > 10 AND (sleep_hours < 6 OR therapy_adherence < 0.5) THEN risk = high\nELSE IF phq9_score > 15 THEN risk = moderate",
    "clinical_interpretation": "The elevated risk is primarily driven by depressive symptoms (PHQ-9: 15, moderate-severe range) combined with sleep disturbance and suboptimal engagement with treatment. The individual shows physiological markers of stress (reduced HRV: 35.2 ms, below healthy range of 50+ ms) and behavioral indicators of withdrawal (decreased activity: 5,200 steps vs. recommended 10,000; limited social interaction). The combination of these factors suggests a need for immediate clinical attention and potential treatment adjustment."
  },
  "risk_score": {
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "score": 68.5,
    "risk_level": "high",
    "confidence": 0.85,
    "contributing_factors": [
      "Elevated PHQ-9 score indicating moderate-severe depression",
      "Poor sleep quality with reduced duration and efficiency"
    ],
    "timestamp": "2025-11-17T10:30:05Z"
  }
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Validation Error |
| 401 | Authentication Error |
| 404 | Prediction not found |
| 500 | Explanation generation failed |

---

## Error Handling

### Error Response Format

All API errors follow a consistent structure:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "additional": "context-specific information"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Status | Description | When It Occurs |
|------|--------|-------------|----------------|
| 200 | OK | Success | Request completed successfully |
| 400 | Bad Request | Validation Error | Invalid input data, schema violations |
| 401 | Unauthorized | Authentication Error | Missing, invalid, or expired token |
| 403 | Forbidden | Consent Error | Missing or expired consent |
| 404 | Not Found | Resource Not Found | Requested resource doesn't exist |
| 500 | Internal Server Error | Processing Error | ETL, feature engineering, or internal failures |
| 503 | Service Unavailable | Model Error | Model inference failures |
| 504 | Gateway Timeout | Timeout Error | Request exceeded time limit (5s for screening) |

### Error Types and Examples

#### 400 - Validation Error

**Scenario:** Invalid request data or schema violations

```json
{
  "error": "ValidationError",
  "message": "Survey data validation failed",
  "details": {
    "validation_errors": [
      {
        "field": "survey_data.phq9_score",
        "message": "Value must be between 0 and 27",
        "type": "value_error",
        "input": 35
      },
      {
        "field": "anonymized_id",
        "message": "String should have at least 1 character",
        "type": "string_too_short",
        "input": ""
      }
    ]
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

#### 401 - Authentication Error

**Scenario:** Invalid or expired authentication token

```json
{
  "error": "AuthenticationError",
  "message": "Invalid authentication credentials",
  "details": {
    "reason": "Token signature verification failed"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

**Common Causes:**
- Token expired (> 60 minutes old)
- Invalid token signature
- Token revoked
- Missing Authorization header

#### 403 - Consent Error

**Scenario:** Missing or expired consent for data processing

```json
{
  "error": "ConsentError",
  "message": "Consent verification failed: Consent expired",
  "details": {
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "consent_status": "expired",
    "expired_at": "2025-10-15T00:00:00Z",
    "data_types_requested": ["survey", "wearable", "emr"]
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

**Common Causes:**
- Consent not granted for requested data types
- Consent expired
- Consent revoked by individual
- `consent_verified` field set to `false`

#### 404 - Not Found Error

**Scenario:** Requested resource doesn't exist

```json
{
  "error": "NotFoundError",
  "message": "No risk score found for a1b2c3d4e5f6789012345678",
  "details": {
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "resource_type": "risk_score"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

#### 500 - Processing Error

**Scenario:** Internal processing failures (ETL, feature engineering)

```json
{
  "error": "ProcessingError",
  "message": "Feature engineering error: Missing required wearable data fields",
  "details": {
    "component": "PhysiologicalFeatureExtractor",
    "missing_fields": ["heart_rate_data", "sleep_data"],
    "request_id": "req_abc123"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

#### 503 - Service Unavailable

**Scenario:** Model inference failures

```json
{
  "error": "InferenceError",
  "message": "Model inference error: Failed to load model version 1.2.3",
  "details": {
    "model_id": "lgbm_v1.2.3",
    "reason": "Model file not found",
    "fallback_attempted": true,
    "request_id": "req_abc123"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

#### 504 - Gateway Timeout

**Scenario:** Request exceeded time limit

```json
{
  "error": "TimeoutError",
  "message": "Screening request exceeded 5s timeout",
  "details": {
    "elapsed_time": 5.234,
    "timeout_limit": 5.0,
    "component": "FeatureEngineeringPipeline",
    "request_id": "req_abc123"
  },
  "timestamp": "2025-11-17T10:30:00Z"
}
```

### Error Recovery Strategies

**Client-Side Recommendations:**

| Error Code | Recommended Action |
|------------|-------------------|
| 400 | Fix validation errors and retry |
| 401 | Refresh authentication token and retry |
| 403 | Verify consent status, do not retry without consent |
| 404 | Verify resource identifier, do not retry |
| 500 | Retry with exponential backoff (max 3 attempts) |
| 503 | Retry with exponential backoff (max 3 attempts) |
| 504 | Retry with longer timeout or simplified request |

**Retry Logic Example:**
```python
import time
import requests

def call_api_with_retry(url, data, headers, max_retries=3):
    """Call API with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            
            # Success
            if response.status_code == 200:
                return response.json()
            
            # Don't retry on client errors (4xx except 429)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise Exception(f"Client error: {response.json()}")
            
            # Retry on server errors (5xx) or rate limiting (429)
            if response.status_code >= 500 or response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Max retries exceeded: {response.json()}")
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Timeout, retry {attempt + 1}/{max_retries} after {wait_time}s")
                time.sleep(wait_time)
                continue
            else:
                raise Exception("Request timeout after max retries")
    
    raise Exception("Unexpected error in retry logic")
```

## Client Examples

### Python Client

**Complete Example with Error Handling:**

```python
import requests
import time
from typing import Dict, Optional

class MHRASClient:
    """Python client for MHRAS API"""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def screen_individual(
        self,
        anonymized_id: str,
        survey_data: Optional[Dict] = None,
        wearable_data: Optional[Dict] = None,
        emr_data: Optional[Dict] = None,
        consent_verified: bool = True
    ) -> Dict:
        """
        Screen an individual and get risk assessment.
        
        Args:
            anonymized_id: Anonymized identifier
            survey_data: Survey responses (PHQ-9, GAD-7, etc.)
            wearable_data: Wearable device metrics
            emr_data: Electronic medical records
            consent_verified: Consent verification status
        
        Returns:
            Screening response with risk score and recommendations
        
        Raises:
            Exception: If request fails
        """
        request_data = {
            "anonymized_id": anonymized_id,
            "consent_verified": consent_verified
        }
        
        if survey_data:
            request_data["survey_data"] = survey_data
        if wearable_data:
            request_data["wearable_data"] = wearable_data
        if emr_data:
            request_data["emr_data"] = emr_data
        
        response = requests.post(
            f"{self.base_url}/screen",
            json=request_data,
            headers=self.headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            raise Exception(
                f"Screening failed ({response.status_code}): "
                f"{error_data.get('message', 'Unknown error')}"
            )
    
    def get_risk_score(self, anonymized_id: str) -> Dict:
        """
        Get the most recent risk score for an individual.
        
        Args:
            anonymized_id: Anonymized identifier
        
        Returns:
            Risk score response
        
        Raises:
            Exception: If request fails
        """
        response = requests.get(
            f"{self.base_url}/risk-score/{anonymized_id}",
            headers=self.headers,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"found": False, "risk_score": None}
        else:
            error_data = response.json()
            raise Exception(
                f"Risk score retrieval failed ({response.status_code}): "
                f"{error_data.get('message', 'Unknown error')}"
            )
    
    def explain_prediction(
        self,
        anonymized_id: str,
        prediction_id: Optional[str] = None
    ) -> Dict:
        """
        Get explanation for a prediction.
        
        Args:
            anonymized_id: Anonymized identifier
            prediction_id: Specific prediction ID (optional)
        
        Returns:
            Explanation response
        
        Raises:
            Exception: If request fails
        """
        request_data = {"anonymized_id": anonymized_id}
        if prediction_id:
            request_data["prediction_id"] = prediction_id
        
        response = requests.post(
            f"{self.base_url}/explain",
            json=request_data,
            headers=self.headers,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            raise Exception(
                f"Explanation failed ({response.status_code}): "
                f"{error_data.get('message', 'Unknown error')}"
            )


# Usage Example
if __name__ == "__main__":
    # Initialize client
    from src.api.auth import authenticator
    
    token = authenticator.generate_token(
        user_id="clinician_001",
        role="clinician",
        expiry_minutes=60
    )
    
    client = MHRASClient(
        base_url="http://localhost:8000",
        token=token
    )
    
    # Screen an individual
    try:
        result = client.screen_individual(
            anonymized_id="a1b2c3d4e5f6789012345678",
            survey_data={
                "phq9_score": 15,
                "gad7_score": 12
            },
            wearable_data={
                "avg_heart_rate": 75,
                "sleep_hours": 6.5,
                "hrv_rmssd": 35.2
            },
            emr_data={
                "diagnosis_codes": ["F32.1"],
                "medications": ["sertraline 50mg"]
            }
        )
        
        # Extract results
        risk_score = result["risk_score"]
        print(f"Risk Score: {risk_score['score']:.1f}")
        print(f"Risk Level: {risk_score['risk_level']}")
        print(f"Confidence: {risk_score['confidence']:.2f}")
        print(f"Alert Triggered: {result['alert_triggered']}")
        print(f"Requires Review: {result['requires_human_review']}")
        
        # Print recommendations
        print("\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  - {rec['name']} ({rec['urgency']})")
        
        # Print top contributing factors
        print("\nContributing Factors:")
        for factor in risk_score["contributing_factors"]:
            print(f"  - {factor}")
            
    except Exception as e:
        print(f"Error: {e}")
```

### cURL Examples

**Screen Individual:**
```bash
curl -X POST http://localhost:8000/screen \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "anonymized_id": "a1b2c3d4e5f6789012345678",
    "survey_data": {
      "phq9_score": 15,
      "gad7_score": 12
    },
    "wearable_data": {
      "avg_heart_rate": 75,
      "sleep_hours": 6.5
    },
    "consent_verified": true
  }'
```

**Get Risk Score:**
```bash
curl -X GET http://localhost:8000/risk-score/a1b2c3d4e5f6789012345678 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Explain Prediction:**
```bash
curl -X POST http://localhost:8000/explain \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "anonymized_id": "a1b2c3d4e5f6789012345678"
  }'
```

### JavaScript/TypeScript Client

```typescript
interface ScreeningRequest {
  anonymized_id: string;
  survey_data?: Record<string, any>;
  wearable_data?: Record<string, any>;
  emr_data?: Record<string, any>;
  consent_verified: boolean;
  timestamp?: string;
}

interface RiskScore {
  anonymized_id: string;
  score: number;
  risk_level: 'low' | 'moderate' | 'high' | 'critical';
  confidence: number;
  contributing_factors: string[];
  timestamp: string;
}

interface ScreeningResponse {
  risk_score: RiskScore;
  recommendations: any[];
  explanations: any;
  requires_human_review: boolean;
  alert_triggered: boolean;
}

class MHRASClient {
  private baseUrl: string;
  private token: string;

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.token = token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        `API Error (${response.status}): ${error.message || 'Unknown error'}`
      );
    }

    return response.json();
  }

  async screenIndividual(
    request: ScreeningRequest
  ): Promise<ScreeningResponse> {
    return this.request<ScreeningResponse>('/screen', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getRiskScore(anonymizedId: string): Promise<{ risk_score: RiskScore; found: boolean }> {
    return this.request(`/risk-score/${anonymizedId}`, {
      method: 'GET',
    });
  }

  async explainPrediction(
    anonymizedId: string,
    predictionId?: string
  ): Promise<any> {
    return this.request('/explain', {
      method: 'POST',
      body: JSON.stringify({
        anonymized_id: anonymizedId,
        prediction_id: predictionId,
      }),
    });
  }
}

// Usage
const client = new MHRASClient('http://localhost:8000', 'YOUR_TOKEN');

const result = await client.screenIndividual({
  anonymized_id: 'a1b2c3d4e5f6789012345678',
  survey_data: { phq9_score: 15, gad7_score: 12 },
  wearable_data: { avg_heart_rate: 75, sleep_hours: 6.5 },
  consent_verified: true,
});

console.log(`Risk Level: ${result.risk_score.risk_level}`);
console.log(`Score: ${result.risk_score.score}`);
```

## Running the API

### Development Mode

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MHRAS_ENV=development
export MHRAS_LOG_LEVEL=DEBUG
export MHRAS_DB_URL=postgresql://user:pass@localhost:5432/mhras

# Run with uvicorn (auto-reload enabled)
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Or use the main entry point
python -m src.main
```

### Production Mode

```bash
# Set production environment variables
export MHRAS_ENV=production
export MHRAS_LOG_LEVEL=INFO
export MHRAS_DB_URL=postgresql://user:pass@db-host:5432/mhras
export MHRAS_JWT_SECRET=your-secret-key

# Run with multiple workers (4 workers recommended for production)
uvicorn src.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log \
  --proxy-headers

# Or use gunicorn with uvicorn workers
gunicorn src.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 30 \
  --access-logfile - \
  --error-logfile -
```

### Docker Deployment

```bash
# Build Docker image
docker build -t mhras-api:latest .

# Run container
docker run -d \
  --name mhras-api \
  -p 8000:8000 \
  -e MHRAS_ENV=production \
  -e MHRAS_DB_URL=postgresql://user:pass@db:5432/mhras \
  mhras-api:latest
```

### Kubernetes Deployment

See `k8s/deployment.yaml` for complete Kubernetes configuration.

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -l app=mhras-api
kubectl logs -f deployment/mhras-api
```

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
  - Interactive API explorer
  - Try out endpoints directly in browser
  - View request/response schemas
  
- **ReDoc**: http://localhost:8000/redoc
  - Clean, readable documentation
  - Better for sharing with stakeholders
  - Printable format

- **OpenAPI JSON**: http://localhost:8000/openapi.json
  - Machine-readable API specification
  - Use for code generation
  - Import into API testing tools

## API Middleware

The API includes the following middleware layers (applied in order):

### 1. RequestLoggingMiddleware
- Logs all incoming requests and outgoing responses
- Includes request ID, method, path, status code, response time
- Structured JSON logging for easy parsing

### 2. AuthenticationMiddleware
- Validates JWT tokens on protected endpoints
- Extracts user information from tokens
- Handles token expiration and revocation

### 3. ErrorHandlingMiddleware
- Centralized error handling and formatting
- Converts exceptions to standardized error responses
- Prevents sensitive information leakage

### 4. TimeoutMiddleware
- Enforces request timeouts (10s default, 5s for /screen)
- Prevents resource exhaustion from slow requests
- Returns 504 Gateway Timeout on exceeded limits

### 5. CORSMiddleware
- Handles Cross-Origin Resource Sharing (CORS)
- Configurable allowed origins, methods, headers
- Required for browser-based clients

**Middleware Configuration:**
```python
# In src/api/app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

## Performance Requirements

| Endpoint | Target Latency | Notes |
|----------|---------------|-------|
| `/screen` | < 5 seconds | End-to-end screening (requirement 1.1) |
| `/risk-score/{id}` | < 1 second | Database query only |
| `/explain` | < 3 seconds | Explanation generation (requirement 8.5) |
| `/health` | < 100ms | Health check |

**Performance Monitoring:**
- All endpoints include `X-Response-Time` header
- Prometheus metrics exported at `/metrics`
- Grafana dashboards for visualization
- Alerts on p95 latency exceeding targets

## Security

### Authentication & Authorization

- **JWT Token-Based Authentication**
  - Tokens signed with HS256 algorithm
  - Secret key stored in environment variable
  - Token expiration: 60 minutes (default)
  - Token revocation support via blacklist

- **Role-Based Access Control (RBAC)**
  - Roles: `admin`, `clinician`, `data_scientist`, `auditor`
  - Permissions enforced at endpoint level
  - Future: Fine-grained permissions per resource

### Data Protection

- **HTTPS Required in Production**
  - TLS 1.3 minimum
  - Valid SSL certificates
  - HSTS headers enabled

- **Input Validation**
  - Pydantic models validate all inputs
  - Type checking and constraint enforcement
  - Prevents injection attacks

- **PII Protection**
  - All identifiers anonymized before processing
  - No PII in logs or error messages
  - Audit trail for all data access

### Security Headers

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

### Rate Limiting

**To Be Implemented:**
- 100 requests per minute per user
- 1000 requests per hour per user
- Burst allowance: 20 requests
- Returns 429 Too Many Requests when exceeded

### Vulnerability Protection

- **SQL Injection**: Parameterized queries, ORM usage
- **XSS**: Input sanitization, output encoding
- **CSRF**: Token-based protection for state-changing operations
- **DoS**: Request timeouts, rate limiting, resource limits

## Monitoring & Observability

### Request Tracking

All requests include custom headers:

| Header | Description | Example |
|--------|-------------|---------|
| `X-Request-ID` | Unique request identifier | `req_abc123def456` |
| `X-Response-Time` | Response time in seconds | `2.345` |
| `X-Model-Version` | Model version used (screening only) | `ensemble_v1.2.3` |

**Usage:**
```bash
curl -v http://localhost:8000/screen \
  -H "Authorization: Bearer TOKEN" \
  -d @request.json

# Response headers:
# X-Request-ID: req_abc123def456
# X-Response-Time: 3.456
# X-Model-Version: ensemble_v1.2.3
```

### Logging

**Structured JSON Logging:**
```json
{
  "timestamp": "2025-11-17T10:30:00.123Z",
  "level": "INFO",
  "logger": "src.api.endpoints",
  "message": "Screening request completed",
  "request_id": "req_abc123",
  "user_id": "clinician_001",
  "anonymized_id": "a1b2c3d4e5f6",
  "risk_score": 68.5,
  "risk_level": "high",
  "response_time": 3.456,
  "alert_triggered": false
}
```

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (e.g., slow requests)
- `ERROR`: Error messages (e.g., processing failures)
- `CRITICAL`: Critical errors requiring immediate attention

### Metrics

**Prometheus Metrics Exported:**

- `mhras_requests_total`: Total requests by endpoint, method, status
- `mhras_request_duration_seconds`: Request duration histogram
- `mhras_screening_risk_scores`: Risk score distribution
- `mhras_model_inference_duration_seconds`: Model inference time
- `mhras_alerts_triggered_total`: Total alerts triggered
- `mhras_human_reviews_queued_total`: Cases queued for review

**Access Metrics:**
```bash
curl http://localhost:8000/metrics
```

### Dashboards

See `monitoring/grafana/dashboards/` for pre-built Grafana dashboards:

1. **Operations Dashboard**: Request rates, latencies, errors
2. **ML Dashboard**: Model performance, drift, predictions
3. **Clinical Dashboard**: Risk distributions, alerts, reviews
4. **Compliance Dashboard**: Audit logs, consent status

## Troubleshooting

### Common Issues

**Issue: 401 Authentication Error**
```
Solution: Check token validity and expiration
- Verify token is not expired (< 60 minutes old)
- Ensure Authorization header format: "Bearer <token>"
- Check token hasn't been revoked
```

**Issue: 403 Consent Error**
```
Solution: Verify consent status
- Check consent exists in database for anonymized_id
- Verify consent covers requested data types
- Ensure consent hasn't expired or been revoked
```

**Issue: 504 Timeout Error**
```
Solution: Optimize request or increase timeout
- Reduce amount of data in request
- Check for slow database queries
- Verify model loading is cached
- Consider async processing for large requests
```

**Issue: 503 Service Unavailable**
```
Solution: Check model availability
- Verify models are loaded in registry
- Check model files exist in storage
- Review model loading logs
- Restart API service if needed
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export MHRAS_LOG_LEVEL=DEBUG
uvicorn src.api.app:app --reload --log-level debug
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check with detailed status (admin only)
curl -H "Authorization: Bearer ADMIN_TOKEN" \
  http://localhost:8000/health/detailed
```

## Support

For API support and questions:
- **Documentation**: http://localhost:8000/docs
- **Issues**: Submit via project issue tracker
- **Email**: support@mhras.example.com
