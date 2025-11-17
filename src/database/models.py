"""Database models for MHRAS."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Prediction record model."""

    id: Optional[UUID] = None
    anonymized_id: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: str = Field(pattern="^(LOW|MODERATE|HIGH|CRITICAL)$")
    confidence: float = Field(ge=0, le=1)
    model_version: str
    features_hash: str
    contributing_factors: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class AuditLog(BaseModel):
    """Audit log record model."""

    id: Optional[UUID] = None
    event_type: str
    anonymized_id: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any]
    created_at: Optional[datetime] = None


class Consent(BaseModel):
    """Consent record model."""

    anonymized_id: str
    data_types: List[str]
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class HumanReviewCase(BaseModel):
    """Human review queue case model."""

    case_id: Optional[UUID] = None
    anonymized_id: str
    risk_score: float = Field(ge=0, le=100)
    prediction_id: Optional[UUID] = None
    assigned_to: Optional[str] = None
    status: str = Field(pattern="^(PENDING|IN_REVIEW|COMPLETED|ESCALATED)$")
    decision: Optional[str] = Field(None, pattern="^(CONFIRMED|MODIFIED|OVERRIDDEN)$")
    decision_notes: Optional[str] = None
    priority: str = Field(default="NORMAL", pattern="^(LOW|NORMAL|HIGH|CRITICAL)$")
    created_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
