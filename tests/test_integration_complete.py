"""
Complete integration tests for MHRAS system.

Tests the full end-to-end workflow with all components integrated.
"""

import pytest
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np

from src.integration import MHRASIntegration, get_integration, reset_integration
from src.screening_service import ScreeningRequest


class TestCompleteIntegration:
    """Test complete system integration"""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance"""
        integration = MHRASIntegration()
        yield integration
        integration.shutdown()
    
    def test_integration_initialization(self, integration):
        """Test that all components are initialized"""
        assert integration.model_registry is not None
        assert integration.inference_engine is not None
        assert integration.ensemble_predictor is not None
        assert integration.interpretability_engine is not None
        assert integration.feature_pipeline is not None
        assert integration.etl_pipeline is not None
        assert integration.data_validator is not None
        assert integration.anonymizer is not None
        assert integration.consent_verifier is not None
        assert integration.audit_logger is not None
        assert integration.human_review_queue is not None
        assert integration.drift_monitor is not None
        assert integration.recommendation_engine is not None
        assert integration.screening_service is not None
    
    def test_health_check(self, integration):
        """Test system health check"""
        health = integration.health_check()
        
        assert "overall_status" in health
        assert "components" in health
        assert "timestamp" in health
        
        # Check that key components are reported
        assert "model_registry" in health["components"]
        assert "feature_pipeline" in health["components"]
        assert "recommendation_engine" in health["components"]
    
    def test_get_statistics(self, integration):
        """Test system statistics"""
        stats = integration.get_statistics()
        
        assert "timestamp" in stats
        assert "screening_service" in stats
        assert "review_queue" in stats
        assert "models" in stats
    
    @pytest.mark.asyncio
    async def test_end_to_end_screening(self, integration):
        """Test complete end-to-end screening workflow"""
        # Create screening request
        request = ScreeningRequest(
            anonymized_id="test_individual_001",
            survey_data={
                "phq9_score": 15,
                "gad7_score": 12,
                "mood_rating": 3,
                "stress_level": 7,
                "sleep_quality": 4
            },
            wearable_data={
                "avg_heart_rate": 75,
                "avg_hrv": 45,
                "sleep_duration": 6.5,
                "activity_minutes": 30
            },
            emr_data={
                "diagnosis_history": ["depression", "anxiety"],
                "medication_adherence": 0.8,
                "therapy_sessions": 5
            }
        )
        
        # Run screening
        service = integration.get_screening_service()
        response = await service.screen_individual(request)
        
        # Verify response
        assert response is not None
        assert response.anonymized_id == "test_individual_001"
        assert 0 <= response.risk_score <= 100
        assert response.risk_level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert 0 <= response.confidence <= 1
        assert isinstance(response.recommendations, list)
        assert isinstance(response.explanations, dict)
        assert isinstance(response.requires_human_review, bool)
        assert isinstance(response.alert_triggered, bool)
        assert response.processing_time_seconds > 0
    
    def test_model_registry_integration(self, integration):
        """Test model registry integration"""
        registry = integration.get_model_registry()
        
        # List models
        all_models = registry.list_models()
        assert isinstance(all_models, list)
        
        # Get active models
        active_models = registry.get_active_models()
        assert isinstance(active_models, list)
    
    def test_recommendation_engine_integration(self, integration):
        """Test recommendation engine integration"""
        from src.recommendations.recommendation_engine import IndividualProfile
        
        engine = integration.get_recommendation_engine()
        
        # Create profile
        profile = IndividualProfile(
            anonymized_id="test_001",
            risk_level="HIGH",
            contributing_factors=["depression", "anxiety"]
        )
        
        # Get recommendations
        recommendations = engine.get_recommendations(profile, max_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    def test_audit_logger_integration(self, integration):
        """Test audit logger integration"""
        audit_logger = integration.get_audit_logger()
        
        # Log a test event
        audit_logger.log_screening_request(
            request={"anonymized_id": "test_001"},
            response={"risk_score": {"score": 50, "risk_level": "MODERATE"}},
            anonymized_id="test_001",
            user_id="test_user"
        )
        
        # Generate report
        from datetime import timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        report = audit_logger.generate_audit_report(start_date, end_date)
        
        assert "total_events" in report
        assert "event_counts" in report
    
    def test_human_review_queue_integration(self, integration):
        """Test human review queue integration"""
        queue = integration.get_human_review_queue()
        
        # Enqueue a case
        case_id = queue.enqueue_case(
            anonymized_id="test_001",
            risk_score=85.0,
            risk_level="HIGH",
            prediction_data={"score": 85.0},
            features={"feature1": 1.0}
        )
        
        assert case_id is not None
        
        # Get queue statistics
        stats = queue.get_queue_statistics()
        
        assert "total_cases" in stats
        assert "pending_count" in stats
        assert stats["total_cases"] > 0
    
    def test_drift_monitor_integration(self, integration):
        """Test drift monitor integration"""
        drift_monitor = integration.get_drift_monitor()
        
        # Create reference data
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        drift_monitor.set_reference_data(reference_data)
        
        # Create current data (similar distribution)
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        # Detect drift
        report = drift_monitor.detect_feature_drift(current_data)
        
        assert report is not None
        assert hasattr(report, 'drift_detected')
        assert hasattr(report, 'drift_score')
    
    def test_singleton_pattern(self):
        """Test that get_integration returns singleton"""
        integration1 = get_integration()
        integration2 = get_integration()
        
        assert integration1 is integration2
        
        # Reset and verify new instance
        reset_integration()
        integration3 = get_integration()
        
        assert integration3 is not integration1
    
    @pytest.mark.asyncio
    async def test_batch_screening(self, integration):
        """Test batch screening functionality"""
        # Create multiple requests
        requests = []
        for i in range(3):
            request = ScreeningRequest(
                anonymized_id=f"test_batch_{i}",
                survey_data={
                    "phq9_score": 10 + i * 5,
                    "gad7_score": 8 + i * 3,
                    "mood_rating": 5 - i,
                    "stress_level": 5 + i,
                    "sleep_quality": 6 - i
                }
            )
            requests.append(request)
        
        # Run batch screening
        service = integration.get_screening_service()
        responses = await service.screen_batch(requests, batch_size=2)
        
        # Verify responses
        assert len(responses) == 3
        for response in responses:
            assert response is not None
            assert hasattr(response, 'risk_score')
            assert hasattr(response, 'risk_level')
    
    def test_component_getters(self, integration):
        """Test all component getter methods"""
        assert integration.get_screening_service() is not None
        assert integration.get_model_registry() is not None
        assert integration.get_recommendation_engine() is not None
        assert integration.get_audit_logger() is not None
        assert integration.get_human_review_queue() is not None
        assert integration.get_drift_monitor() is not None


class TestAPIIntegration:
    """Test API integration with components"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from src.api.app import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestCLIIntegration:
    """Test CLI integration"""
    
    def test_cli_import(self):
        """Test that CLI can be imported"""
        from src.cli import cli
        assert cli is not None
    
    def test_cli_commands(self):
        """Test CLI command structure"""
        from src.cli import cli, db, models, review, audit, system
        
        assert db is not None
        assert models is not None
        assert review is not None
        assert audit is not None
        assert system is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
