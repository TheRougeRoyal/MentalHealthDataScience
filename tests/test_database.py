"""Tests for database layer."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.database.models import Prediction, AuditLog, Consent, HumanReviewCase


class TestDatabaseModels:
    """Test database model validation."""

    def test_prediction_model_valid(self):
        """Test valid prediction model creation."""
        prediction = Prediction(
            anonymized_id="test_id_123",
            risk_score=75.5,
            risk_level="HIGH",
            confidence=0.85,
            model_version="v1.0.0",
            features_hash="abc123def456",
            contributing_factors={"factor1": 0.5, "factor2": 0.3}
        )
        assert prediction.anonymized_id == "test_id_123"
        assert prediction.risk_score == 75.5
        assert prediction.risk_level == "HIGH"

    def test_prediction_model_invalid_risk_score(self):
        """Test prediction model with invalid risk score."""
        with pytest.raises(ValueError):
            Prediction(
                anonymized_id="test_id_123",
                risk_score=150.0,  # Invalid: > 100
                risk_level="HIGH",
                confidence=0.85,
                model_version="v1.0.0",
                features_hash="abc123def456"
            )

    def test_prediction_model_invalid_risk_level(self):
        """Test prediction model with invalid risk level."""
        with pytest.raises(ValueError):
            Prediction(
                anonymized_id="test_id_123",
                risk_score=75.5,
                risk_level="INVALID",  # Invalid risk level
                confidence=0.85,
                model_version="v1.0.0",
                features_hash="abc123def456"
            )

    def test_audit_log_model_valid(self):
        """Test valid audit log model creation."""
        audit_log = AuditLog(
            event_type="SCREENING_REQUEST",
            anonymized_id="test_id_123",
            user_id="user_456",
            details={"request": "data", "response": "result"}
        )
        assert audit_log.event_type == "SCREENING_REQUEST"
        assert audit_log.details["request"] == "data"

    def test_consent_model_valid(self):
        """Test valid consent model creation."""
        consent = Consent(
            anonymized_id="test_id_123",
            data_types=["survey", "wearable", "emr"],
            expires_at=datetime.utcnow() + timedelta(days=365)
        )
        assert len(consent.data_types) == 3
        assert "survey" in consent.data_types

    def test_human_review_case_model_valid(self):
        """Test valid human review case model creation."""
        case = HumanReviewCase(
            anonymized_id="test_id_123",
            risk_score=85.0,
            prediction_id=uuid4(),
            status="PENDING",
            priority="HIGH"
        )
        assert case.risk_score == 85.0
        assert case.status == "PENDING"
        assert case.priority == "HIGH"

    def test_human_review_case_invalid_status(self):
        """Test human review case with invalid status."""
        with pytest.raises(ValueError):
            HumanReviewCase(
                anonymized_id="test_id_123",
                risk_score=85.0,
                status="INVALID_STATUS",  # Invalid status
                priority="HIGH"
            )
