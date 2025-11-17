"""
Tests for the ScreeningService end-to-end workflow.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.screening_service import (
    ScreeningService,
    ScreeningServiceSync,
    ScreeningRequest,
    ScreeningResponse
)
from src.exceptions import ValidationError, ConsentError, ScreeningError


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    return None


@pytest.fixture
def sample_screening_request():
    """Create a sample screening request."""
    return ScreeningRequest(
        anonymized_id="test_user_123",
        survey_data={
            "responses": {
                "mood": "low",
                "sleep_quality": "poor",
                "suicidal_ideation": False,
                "self_harm": False
            },
            "timestamp": datetime.utcnow().isoformat()
        },
        wearable_data={
            "metrics": {
                "sleep_duration_hours": 5.0,
                "heart_rate_avg": 75,
                "hrv_rmssd": 30
            },
            "data_quality": {
                "completeness_percent": 90,
                "wear_time_minutes": 1200
            }
        },
        user_id="clinician_001"
    )


class TestScreeningRequest:
    """Test ScreeningRequest class."""
    
    def test_screening_request_creation(self):
        """Test creating a screening request."""
        request = ScreeningRequest(
            anonymized_id="test_123",
            survey_data={"test": "data"}
        )
        
        assert request.anonymized_id == "test_123"
        assert request.survey_data == {"test": "data"}
        assert request.wearable_data is None
        assert request.emr_data is None
        assert isinstance(request.timestamp, datetime)
    
    def test_screening_request_to_dict(self, sample_screening_request):
        """Test converting request to dictionary."""
        request_dict = sample_screening_request.to_dict()
        
        assert request_dict["anonymized_id"] == "test_user_123"
        assert "survey_data" in request_dict
        assert "wearable_data" in request_dict
        assert "timestamp" in request_dict


class TestScreeningResponse:
    """Test ScreeningResponse class."""
    
    def test_screening_response_creation(self):
        """Test creating a screening response."""
        response = ScreeningResponse(
            anonymized_id="test_123",
            risk_score=65.5,
            risk_level="high",
            confidence=0.85,
            contributing_factors=["sleep", "mood"],
            recommendations=[{"name": "Test Resource"}],
            explanations={"top_features": ["sleep", "mood"]},
            requires_human_review=True,
            alert_triggered=False,
            processing_time_seconds=2.5
        )
        
        assert response.anonymized_id == "test_123"
        assert response.risk_score == 65.5
        assert response.risk_level == "high"
        assert response.confidence == 0.85
        assert response.requires_human_review is True
    
    def test_screening_response_to_dict(self):
        """Test converting response to dictionary."""
        response = ScreeningResponse(
            anonymized_id="test_123",
            risk_score=65.5,
            risk_level="high",
            confidence=0.85,
            contributing_factors=["sleep"],
            recommendations=[],
            explanations={},
            requires_human_review=False,
            alert_triggered=False,
            processing_time_seconds=2.0
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["anonymized_id"] == "test_123"
        assert response_dict["risk_score"]["score"] == 65.5
        assert response_dict["risk_score"]["risk_level"] == "high"
        assert "recommendations" in response_dict
        assert "explanations" in response_dict


class TestScreeningService:
    """Test ScreeningService class."""
    
    def test_service_initialization(self, mock_db_connection):
        """Test service initialization."""
        service = ScreeningService(db_connection=mock_db_connection)
        
        assert service.validator is not None
        assert service.consent_verifier is not None
        assert service.anonymizer is not None
        assert service.etl_pipeline is not None
        assert service.feature_pipeline is not None
        assert service.model_registry is not None
        assert service.inference_engine is not None
        assert service.ensemble_predictor is not None
        assert service.recommendation_engine is not None
        assert service.audit_logger is not None
        assert service.human_review_queue is not None
    
    def test_service_with_custom_thresholds(self):
        """Test service with custom thresholds."""
        service = ScreeningService(
            alert_threshold=80.0,
            human_review_threshold=70.0
        )
        
        assert service.alert_threshold == 80.0
        assert service.human_review_threshold == 70.0
    
    def test_default_etl_config(self):
        """Test default ETL configuration."""
        service = ScreeningService()
        config = service._default_etl_config()
        
        assert config.outlier_method == "iqr"
        assert config.iqr_multiplier == 1.5
        assert config.group_by_column == "anonymized_id"
    
    def test_clear_cache(self):
        """Test cache clearing."""
        service = ScreeningService()
        
        # Add some cache entries
        service._cache["test_key"] = "test_value"
        service._model_cache["model_1"] = "model_data"
        
        # Clear cache
        service.clear_cache()
        
        assert len(service._cache) == 0
        assert len(service._model_cache) == 0
    
    def test_get_screening_statistics(self):
        """Test getting screening statistics."""
        service = ScreeningService()
        stats = service.get_screening_statistics()
        
        assert "review_queue" in stats
        assert "model_registry" in stats
        assert "active_models" in stats["model_registry"]


class TestScreeningServiceSync:
    """Test synchronous wrapper."""
    
    def test_sync_service_initialization(self):
        """Test synchronous service initialization."""
        service = ScreeningServiceSync()
        
        assert service.service is not None
        assert isinstance(service.service, ScreeningService)
    
    def test_sync_service_methods(self):
        """Test synchronous service methods exist."""
        service = ScreeningServiceSync()
        
        assert hasattr(service, 'screen_individual')
        assert hasattr(service, 'screen_batch')
        assert hasattr(service, 'get_screening_statistics')
        assert hasattr(service, 'clear_cache')
        assert hasattr(service, 'preload_models')


# Integration test would require full setup
@pytest.mark.skip(reason="Requires full system setup with models and database")
class TestScreeningServiceIntegration:
    """Integration tests for screening service."""
    
    @pytest.mark.asyncio
    async def test_full_screening_workflow(self, sample_screening_request):
        """Test complete screening workflow."""
        service = ScreeningService()
        
        response = await service.screen_individual(sample_screening_request)
        
        assert response.anonymized_id == sample_screening_request.anonymized_id
        assert 0 <= response.risk_score <= 100
        assert response.risk_level in ["low", "moderate", "high", "critical"]
        assert 0 <= response.confidence <= 1
        assert response.processing_time_seconds > 0
