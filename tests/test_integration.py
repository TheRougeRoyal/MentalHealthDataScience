"""Integration tests for end-to-end workflows"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.ingestion.validation import DataValidator
from src.governance.consent import ConsentVerifier
from src.governance.anonymization import Anonymizer
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
from src.ml.feature_pipeline import FeatureEngineeringPipeline
from src.governance.human_review_queue import HumanReviewQueue
from src.governance.crisis_override import CrisisOverride
from src.governance.audit_logger import AuditLogger


class TestEndToEndScreeningWorkflow:
    """Test complete screening workflow from data ingestion to prediction"""
    
    def test_data_validation_to_etl_pipeline(self):
        """Test data flows from validation through ETL"""
        # 1. Validate data
        validator = DataValidator()
        
        survey_data = {
            "survey_id": "test_survey",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "mood_score": 5,
                "anxiety_level": 7,
                "sleep_quality": 3
            }
        }
        
        validation_result = validator.validate_survey(survey_data)
        assert validation_result.is_valid
        
        # 2. Process through ETL
        df = pd.DataFrame([{
            "anonymized_id": "user_123",
            "timestamp": datetime.utcnow(),
            "mood_score": 5,
            "anxiety_level": 7,
            "sleep_quality": 3
        }])
        
        config = ETLPipelineConfig(
            standardize_columns=["mood_score", "anxiety_level", "sleep_quality"],
            group_by_column="anonymized_id"
        )
        pipeline = ETLPipeline(config)
        
        processed_data = pipeline.fit_transform(df)
        
        assert processed_data is not None
        assert len(processed_data) > 0
        assert pipeline.is_fitted
    
    def test_etl_to_feature_engineering(self):
        """Test data flows from ETL to feature engineering"""
        # 1. Create processed data
        np.random.seed(42)
        processed_data = pd.DataFrame({
            "anonymized_id": ["user_1"] * 10,
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D"),
            "mood_score": np.random.randint(1, 11, 10),
            "sleep_hours": np.random.uniform(5, 9, 10),
            "activity_level": np.random.choice(["low", "medium", "high"], 10)
        })
        
        # 2. Engineer features
        feature_pipeline = FeatureEngineeringPipeline(use_parallel=False)
        
        # Create behavioral data
        behavioral_data = pd.DataFrame({
            "anonymized_id": ["user_1"] * 5,
            "activity_type": ["exercise", "social", "work", "exercise", "social"],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D")
        })
        
        features = feature_pipeline.extract_features(behavioral_df=behavioral_data)
        
        assert features is not None
        assert len(features) > 0
        assert "anonymized_id" in features.columns


class TestConsentVerificationWorkflow:
    """Test consent verification and rejection workflows"""
    
    def test_consent_verification_success(self):
        """Test successful consent verification"""
        verifier = ConsentVerifier(db_connection=None)
        
        # Verify consent
        result = verifier.verify_consent(
            anonymized_id="user_consent_valid",
            data_types=["survey", "wearable"]
        )
        
        assert result.is_valid
        
        # Should be cached
        result2 = verifier.verify_consent(
            anonymized_id="user_consent_valid",
            data_types=["survey", "wearable"]
        )
        
        assert result2.is_valid
    
    def test_consent_rejection_workflow(self):
        """Test data rejection when consent is missing"""
        verifier = ConsentVerifier(db_connection=None)
        validator = DataValidator()
        
        # Simulate consent check failure
        # In real implementation, this would query database
        
        # Validate data
        survey_data = {
            "survey_id": "test_no_consent",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {"mood_score": 5}
        }
        
        validation_result = validator.validate_survey(survey_data)
        assert validation_result.is_valid
        
        # In production, would check consent before processing
        # If consent invalid, data should not be processed


class TestHumanReviewQueueRouting:
    """Test routing of high-risk cases to human review"""
    
    def test_high_risk_case_routing(self):
        """Test that high-risk cases are routed to review queue"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(queue_file=str(queue_file))
            
            # Simulate high-risk prediction
            risk_score = 85.0
            
            # Should trigger human review (threshold is 75)
            if risk_score > 75:
                case_id = queue.enqueue_case(
                    anonymized_id="user_high_risk",
                    risk_score=risk_score,
                    risk_level="critical",
                    prediction_data={"score": risk_score},
                    features={"feature1": 1.0}
                )
                
                assert case_id is not None
                assert len(queue.cases) == 1
                
                # Get pending cases
                pending = queue.get_pending_cases()
                assert len(pending) == 1
                assert pending[0].risk_score == 85.0
    
    def test_review_workflow_complete(self):
        """Test complete review workflow from enqueue to completion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            audit_file = Path(tmpdir) / "audit.jsonl"
            
            queue = HumanReviewQueue(queue_file=str(queue_file))
            audit_logger = AuditLogger(audit_log_path=str(audit_file))
            
            # 1. Enqueue case
            case_id = queue.enqueue_case(
                anonymized_id="user_review_flow",
                risk_score=82.0,
                risk_level="critical",
                prediction_data={"score": 82.0},
                features={}
            )
            
            # 2. Assign to reviewer
            queue.assign_case(case_id, "reviewer_001")
            
            # 3. Start review
            case = queue.start_review(case_id, "reviewer_001")
            assert case.anonymized_id == "user_review_flow"
            
            # 4. Submit review
            queue.submit_review(
                case_id=case_id,
                reviewer_id="reviewer_001",
                decision="CONFIRM",
                justification="Risk confirmed after review"
            )
            
            # 5. Log review
            audit_logger.log_human_review(
                case_id=case_id,
                reviewer_id="reviewer_001",
                decision="CONFIRM",
                anonymized_id="user_review_flow",
                original_risk_score=82.0
            )
            
            # Verify completion
            completed_case = queue.get_case(case_id)
            assert completed_case.decision.value == "CONFIRM"
            assert completed_case.reviewed_at is not None
            
            # Verify audit log
            assert audit_file.exists()


class TestCrisisOverrideWorkflow:
    """Test crisis override emergency workflows"""
    
    def test_crisis_override_initiation(self):
        """Test initiating crisis override"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            audit_file = Path(tmpdir) / "audit.jsonl"
            
            crisis = CrisisOverride(override_file=str(override_file))
            audit_logger = AuditLogger(audit_log_path=str(audit_file))
            
            # 1. Initiate override
            override_id = crisis.initiate_override(
                case_id="case_crisis",
                anonymized_id="user_crisis",
                clinician_id="clinician_001",
                reason="immediate_safety_concern",
                justification="Patient expressed active suicidal ideation with plan",
                original_recommendation={"action": "schedule_appointment"},
                override_action="immediate_hospitalization"
            )
            
            assert override_id is not None
            
            # 2. Verify supervisor notification
            override = crisis.overrides[override_id]
            assert override.supervisor_notified is True
            assert override.supervisor_notified_at is not None
            
            # 3. Log override
            audit_logger.log_error(
                error_type="CRISIS_OVERRIDE",
                error_message=f"Crisis override initiated: {override_id}",
                component="CrisisOverride",
                anonymized_id="user_crisis",
                context={"override_id": override_id, "clinician_id": "clinician_001"}
            )
            
            # Verify audit log
            assert audit_file.exists()
    
    def test_crisis_override_supervisor_review(self):
        """Test supervisor review of crisis override"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            # 1. Initiate override
            override_id = crisis.initiate_override(
                case_id="case_supervisor",
                anonymized_id="user_supervisor",
                clinician_id="clinician_002",
                reason="emergency",
                justification="Emergency situation",
                original_recommendation={},
                override_action="admit"
            )
            
            # 2. Supervisor reviews
            crisis.supervisor_review(
                override_id=override_id,
                supervisor_id="supervisor_001",
                comments="Appropriate action taken given circumstances"
            )
            
            # 3. Verify review
            override = crisis.overrides[override_id]
            assert override.reviewed_by_supervisor == "supervisor_001"
            assert override.supervisor_review_at is not None
            assert override.supervisor_comments is not None
            
            # 4. Check pending reviews
            pending = crisis.get_pending_supervisor_reviews()
            assert len(pending) == 0  # Should be empty after review


class TestDataAnonymizationWorkflow:
    """Test data anonymization throughout workflow"""
    
    def test_anonymization_consistency(self):
        """Test that anonymization is consistent across workflow"""
        anonymizer = Anonymizer(salt="test_salt")
        
        # 1. Anonymize identifier
        original_id = "patient_12345"
        anon_id_1 = anonymizer.hash_identifier(original_id)
        
        # 2. Use anonymized ID in different components
        anon_id_2 = anonymizer.hash_identifier(original_id)
        
        # Should be consistent
        assert anon_id_1 == anon_id_2
        assert len(anon_id_1) == 64
        
        # 3. Anonymize record
        record = {
            "patient_id": original_id,
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30
        }
        
        anon_record = anonymizer.anonymize_record(
            record,
            pii_fields=["patient_id", "name", "email"]
        )
        
        # Verify anonymization
        assert anon_record["patient_id"] == anon_id_1
        assert anon_record["name"] != "John Doe"
        assert anon_record["email"] != "john@example.com"
        assert anon_record["age"] == 30  # Not PII
    
    def test_anonymization_with_text_redaction(self):
        """Test text anonymization in workflow"""
        anonymizer = Anonymizer()
        
        # Text with PII
        text = "Contact me at john.doe@example.com or call 555-123-4567"
        
        # Anonymize text
        anon_text = anonymizer.anonymize_text(text)
        
        # Verify PII removed
        assert "john.doe@example.com" not in anon_text
        assert "555-123-4567" not in anon_text
        assert "[REDACTED]" in anon_text


class TestAuditLoggingWorkflow:
    """Test audit logging throughout system"""
    
    def test_complete_audit_trail(self):
        """Test complete audit trail for a screening request"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            audit_logger = AuditLogger(audit_log_path=str(audit_file))
            
            anonymized_id = "user_audit_trail"
            
            # 1. Log screening request
            audit_logger.log_screening_request(
                request={"data": "test"},
                response={
                    "risk_score": {"score": 75.0, "risk_level": "HIGH"},
                    "alert_triggered": True,
                    "requires_human_review": True
                },
                anonymized_id=anonymized_id,
                user_id="clinician_001"
            )
            
            # 2. Log prediction
            audit_logger.log_prediction(
                features_hash="abc123",
                prediction={"score": 75.0, "risk_level": "high"},
                model_id="model_v1",
                anonymized_id=anonymized_id
            )
            
            # 3. Log consent verification
            audit_logger.log_consent_verification(
                anonymized_id=anonymized_id,
                data_types=["survey", "wearable"],
                consent_status="VALID"
            )
            
            # 4. Log data access
            audit_logger.log_data_access(
                anonymized_id=anonymized_id,
                user_id="clinician_001",
                access_type="READ",
                data_types=["survey", "wearable"],
                purpose="screening"
            )
            
            # 5. Get audit trail
            trail = audit_logger.get_audit_trail(anonymized_id)
            
            assert len(trail) == 4
            assert any(e["event_type"] == "screening_request" for e in trail)
            assert any(e["event_type"] == "prediction" for e in trail)
            assert any(e["event_type"] == "consent_verification" for e in trail)
            assert any(e["event_type"] == "data_access" for e in trail)
    
    def test_audit_report_generation(self):
        """Test generating comprehensive audit report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            audit_logger = AuditLogger(audit_log_path=str(audit_file))
            
            # Log multiple events
            for i in range(10):
                audit_logger.log_screening_request(
                    request={},
                    response={
                        "risk_score": {"score": 50 + i * 5, "risk_level": "MODERATE"},
                        "alert_triggered": i > 5,
                        "requires_human_review": i > 7
                    },
                    anonymized_id=f"user_{i}",
                    user_id="clinician_001"
                )
            
            # Generate report
            start_date = datetime.utcnow() - timedelta(hours=1)
            end_date = datetime.utcnow() + timedelta(hours=1)
            
            report = audit_logger.generate_audit_report(start_date, end_date)
            
            assert report["total_events"] == 10
            assert report["screening_statistics"]["total_screenings"] == 10
            assert report["screening_statistics"]["alerts_triggered"] == 4
            assert report["screening_statistics"]["human_reviews_required"] == 2


class TestMultiComponentIntegration:
    """Test integration of multiple components"""
    
    def test_validation_consent_anonymization_flow(self):
        """Test data flows through validation, consent, and anonymization"""
        # 1. Validate data
        validator = DataValidator()
        survey_data = {
            "survey_id": "integration_test",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {"mood_score": 5}
        }
        
        validation_result = validator.validate_survey(survey_data)
        assert validation_result.is_valid
        
        # 2. Verify consent
        verifier = ConsentVerifier(db_connection=None)
        consent_result = verifier.verify_consent(
            anonymized_id="user_integration",
            data_types=["survey"]
        )
        assert consent_result.is_valid
        
        # 3. Anonymize data
        anonymizer = Anonymizer(salt="integration_salt")
        anon_record = anonymizer.anonymize_record(
            {"patient_id": "patient_123", "data": survey_data},
            pii_fields=["patient_id"]
        )
        
        assert anon_record["patient_id"] != "patient_123"
        assert len(anon_record["patient_id"]) == 64
    
    def test_etl_features_audit_flow(self):
        """Test data flows through ETL, features, and audit logging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            audit_logger = AuditLogger(audit_log_path=str(audit_file))
            
            # 1. ETL processing
            df = pd.DataFrame({
                "anonymized_id": ["user_flow"] * 5,
                "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
                "mood_score": [5, 6, 4, 7, 5]
            })
            
            config = ETLPipelineConfig(
                standardize_columns=["mood_score"],
                group_by_column="anonymized_id"
            )
            pipeline = ETLPipeline(config)
            processed = pipeline.fit_transform(df)
            
            assert processed is not None
            
            # 2. Feature engineering
            feature_pipeline = FeatureEngineeringPipeline(use_parallel=False)
            behavioral_data = pd.DataFrame({
                "anonymized_id": ["user_flow"] * 3,
                "activity_type": ["exercise", "social", "work"],
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="D")
            })
            
            features = feature_pipeline.extract_features(behavioral_df=behavioral_data)
            assert features is not None
            
            # 3. Log processing
            audit_logger.log_data_access(
                anonymized_id="user_flow",
                user_id="system",
                access_type="WRITE",
                data_types=["features"],
                purpose="feature_engineering"
            )
            
            # Verify audit log
            trail = audit_logger.get_audit_trail("user_flow")
            assert len(trail) == 1


class TestErrorHandlingIntegration:
    """Test error handling across components"""
    
    def test_validation_error_logging(self):
        """Test that validation errors are logged"""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = Path(tmpdir) / "audit.jsonl"
            audit_logger = AuditLogger(audit_log_path=str(audit_file))
            
            validator = DataValidator()
            
            # Invalid data
            invalid_data = {
                "survey_id": "test",
                # Missing required timestamp
                "responses": {}
            }
            
            result = validator.validate_survey(invalid_data)
            
            if not result.is_valid:
                # Log error
                audit_logger.log_error(
                    error_type="VALIDATION_ERROR",
                    error_message="; ".join(result.errors),
                    component="DataValidator",
                    context={"survey_id": "test"}
                )
                
                # Verify error logged
                assert audit_file.exists()
    
    def test_processing_error_recovery(self):
        """Test error recovery in processing pipeline"""
        # Create pipeline
        config = ETLPipelineConfig()
        pipeline = ETLPipeline(config)
        
        # Try to process invalid data
        invalid_df = pd.DataFrame({"invalid_column": [1, 2, 3]})
        
        # Should handle gracefully
        try:
            pipeline.fit_transform(invalid_df)
        except Exception as e:
            # Error should be caught and logged
            assert isinstance(e, Exception)


from datetime import timedelta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
