"""Pydantic models for API request/response validation"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ScreeningRequest(BaseModel):
    """Request model for screening endpoint"""
    anonymized_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Anonymized identifier for the individual"
    )
    survey_data: Optional[Dict] = Field(
        None,
        description="Survey response data"
    )
    wearable_data: Optional[Dict] = Field(
        None,
        description="Wearable device metrics"
    )
    emr_data: Optional[Dict] = Field(
        None,
        description="Electronic medical records data"
    )
    consent_verified: bool = Field(
        ...,
        description="Whether consent has been verified"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request timestamp"
    )

    @field_validator('anonymized_id')
    @classmethod
    def validate_anonymized_id(cls, v: str) -> str:
        """Validate anonymized ID format"""
        if not v or not v.strip():
            raise ValueError("anonymized_id cannot be empty")
        return v.strip()

    @field_validator('consent_verified')
    @classmethod
    def validate_consent(cls, v: bool) -> bool:
        """Ensure consent is verified"""
        if not v:
            raise ValueError("consent_verified must be True to process data")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "anonymized_id": "a1b2c3d4e5f6",
                "survey_data": {
                    "phq9_score": 15,
                    "gad7_score": 12
                },
                "wearable_data": {
                    "avg_heart_rate": 75,
                    "sleep_hours": 6.5
                },
                "emr_data": {
                    "diagnosis_codes": ["F32.1"],
                    "medications": ["sertraline"]
                },
                "consent_verified": True,
                "timestamp": "2025-11-17T10:30:00Z"
            }
        }


class RiskScore(BaseModel):
    """Risk score model"""
    anonymized_id: str = Field(
        ...,
        description="Anonymized identifier"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk score between 0 and 100"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk level classification"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence between 0 and 1"
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Key factors contributing to the risk score"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "anonymized_id": "a1b2c3d4e5f6",
                "score": 68.5,
                "risk_level": "high",
                "confidence": 0.85,
                "contributing_factors": [
                    "Elevated PHQ-9 score",
                    "Poor sleep quality",
                    "Decreased social interaction"
                ],
                "timestamp": "2025-11-17T10:30:05Z"
            }
        }


class ExplanationSummary(BaseModel):
    """Model explanation summary"""
    top_features: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Top features with SHAP values"
    )
    counterfactual: str = Field(
        default="",
        description="Human-readable counterfactual explanation"
    )
    rule_approximation: str = Field(
        default="",
        description="Simple if-then rules approximating the model"
    )
    clinical_interpretation: str = Field(
        default="",
        description="Clinical interpretation of the prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "top_features": [
                    ("phq9_score", 0.25),
                    ("sleep_quality", -0.18),
                    ("social_interaction_frequency", -0.15)
                ],
                "counterfactual": "If sleep quality improved by 20% and social interactions increased by 2 per week, risk level would decrease to moderate.",
                "rule_approximation": "IF phq9_score > 15 AND sleep_hours < 6 THEN risk = high",
                "clinical_interpretation": "The elevated risk is primarily driven by depressive symptoms and sleep disturbance."
            }
        }


class ResourceRecommendation(BaseModel):
    """Resource recommendation model"""
    resource_type: str = Field(
        ...,
        description="Type of resource (e.g., therapy, crisis_line, support_group)"
    )
    name: str = Field(
        ...,
        description="Name of the resource"
    )
    description: str = Field(
        ...,
        description="Description of the resource"
    )
    contact_info: Optional[str] = Field(
        None,
        description="Contact information"
    )
    urgency: str = Field(
        ...,
        description="Urgency level (immediate, soon, routine)"
    )
    eligibility_criteria: Optional[List[str]] = Field(
        None,
        description="Eligibility criteria for the resource"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "resource_type": "crisis_line",
                "name": "National Suicide Prevention Lifeline",
                "description": "24/7 crisis support and suicide prevention",
                "contact_info": "988",
                "urgency": "immediate",
                "eligibility_criteria": ["Available to all"]
            }
        }


class ScreeningResponse(BaseModel):
    """Response model for screening endpoint"""
    risk_score: RiskScore = Field(
        ...,
        description="Calculated risk score"
    )
    recommendations: List[ResourceRecommendation] = Field(
        default_factory=list,
        description="Recommended resources"
    )
    explanations: ExplanationSummary = Field(
        ...,
        description="Model explanations"
    )
    requires_human_review: bool = Field(
        ...,
        description="Whether case requires human review"
    )
    alert_triggered: bool = Field(
        ...,
        description="Whether an alert was triggered"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": {
                    "anonymized_id": "a1b2c3d4e5f6",
                    "score": 68.5,
                    "risk_level": "high",
                    "confidence": 0.85,
                    "contributing_factors": ["Elevated PHQ-9 score"],
                    "timestamp": "2025-11-17T10:30:05Z"
                },
                "recommendations": [
                    {
                        "resource_type": "therapy",
                        "name": "Cognitive Behavioral Therapy",
                        "description": "Evidence-based therapy for depression",
                        "contact_info": "Contact your provider",
                        "urgency": "soon",
                        "eligibility_criteria": ["Diagnosed depression"]
                    }
                ],
                "explanations": {
                    "top_features": [("phq9_score", 0.25)],
                    "counterfactual": "If sleep improved...",
                    "rule_approximation": "IF phq9_score > 15...",
                    "clinical_interpretation": "Elevated risk..."
                },
                "requires_human_review": True,
                "alert_triggered": False
            }
        }


class RiskScoreResponse(BaseModel):
    """Response model for risk score retrieval"""
    risk_score: RiskScore = Field(
        ...,
        description="Retrieved risk score"
    )
    found: bool = Field(
        default=True,
        description="Whether the risk score was found"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": {
                    "anonymized_id": "a1b2c3d4e5f6",
                    "score": 68.5,
                    "risk_level": "high",
                    "confidence": 0.85,
                    "contributing_factors": ["Elevated PHQ-9 score"],
                    "timestamp": "2025-11-17T10:30:05Z"
                },
                "found": True
            }
        }


class ExplanationRequest(BaseModel):
    """Request model for explanation endpoint"""
    anonymized_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Anonymized identifier"
    )
    prediction_id: Optional[str] = Field(
        None,
        description="Specific prediction ID to explain"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "anonymized_id": "a1b2c3d4e5f6",
                "prediction_id": "pred_123456"
            }
        }


class ExplanationResponse(BaseModel):
    """Response model for explanation endpoint"""
    anonymized_id: str = Field(
        ...,
        description="Anonymized identifier"
    )
    explanations: ExplanationSummary = Field(
        ...,
        description="Model explanations"
    )
    risk_score: Optional[RiskScore] = Field(
        None,
        description="Associated risk score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "anonymized_id": "a1b2c3d4e5f6",
                "explanations": {
                    "top_features": [("phq9_score", 0.25)],
                    "counterfactual": "If sleep improved...",
                    "rule_approximation": "IF phq9_score > 15...",
                    "clinical_interpretation": "Elevated risk..."
                },
                "risk_score": {
                    "anonymized_id": "a1b2c3d4e5f6",
                    "score": 68.5,
                    "risk_level": "high",
                    "confidence": 0.85,
                    "contributing_factors": ["Elevated PHQ-9 score"],
                    "timestamp": "2025-11-17T10:30:05Z"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[Dict] = Field(
        None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request data",
                "details": {
                    "field": "anonymized_id",
                    "issue": "Field required"
                },
                "timestamp": "2025-11-17T10:30:00Z"
            }
        }
