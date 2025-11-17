"""Tests for governance components"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.governance.audit_logger import AuditLogger
from src.governance.drift_monitor import DriftMonitor, DriftReport
from src.governance.human_review_queue import HumanReviewQueue, ReviewCase, ReviewStatus, ReviewDecision
from src.governance.crisis_override import CrisisOverride, Override
import pandas as pd
import numpy as np


class TestAuditLogger:
    """Test AuditLogger class"""
    
    def test_initialization(self):
        """Test audit logger initializes correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(audit_log_path=str(log_path))
            
            assert logger.audit_log_path == log_path
            assert log_path.parent.exists()
    
    def test_log_screening_request(self):
        """Test logging screening request"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(audit_log_path=str(log_path))
            
            request = {"data": "test_request"}
            response = {
                "risk_score": {"score": 75.5, "risk_level": "HIGH"},
                "alert_triggered": True,
                "requires_human_review": True
            }
            
            logger.log_screening_request(
                request=request,
                response=response,
                anonymized_id="test_user_123",
                user_id="clinician_001"
            )
            
            # Verify log file was created and contains entry
            assert log_path.exists()
            with open(log_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1
                entry = json.loads(lines[0])
                assert entry["event_type"] == "screening_request"
                assert entry["anonymized_id"] == "test_user_123"
                assert entry["risk_level"] == "HIGH"
    
    def test_log_prediction(self):
        """Test logging prediction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(audit_log_path=str(log_path))
            
            prediction = {
                "score": 65.0,
                "risk_level": "high",
                "confidence": 0.85
            }
            
            logger.log_prediction(
                features_hash="abc123",
                prediction=prediction,
                model_id="model_v1",
                anonymized_id="test_user_456",
                model_version="1.0.0"
            )
            
            with open(log_path, "r") as f:
                entry = json.loads(f.readline())
                assert entry["event_type"] == "prediction"
                assert entry["model_id"] == "model_v1"
                assert entry["risk_score"] == 65.0
    
    def test_log_human_review(self):
        """Test logging human review"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(audit_log_path=str(log_path))
            
            logger.log_human_review(
                case_id="case_123",
                reviewer_id="reviewer_001",
                decision="CONFIRM",
                anonymized_id="test_user_789",
                original_risk_score=75.0,
                justification="Confirmed high risk",
                review_duration_seconds=300.0
            )
            
            with open(log_path, "r") as f:
                entry = json.loads(f.readline())
                assert entry["event_type"] == "human_review"
                assert entry["decision"] == "CONFIRM"
                assert entry["review_duration_seconds"] == 300.0
    
    def test_generate_audit_report(self):
        """Test generating audit report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(audit_log_path=str(log_path))
            
            # Log multiple events
            for i in range(5):
                logger.log_screening_request(
                    request={"test": f"request_{i}"},
                    response={
                        "risk_score": {"score": 50 + i * 10, "risk_level": "MODERATE"},
                        "alert_triggered": False,
                        "requires_human_review": False
                    },
                    anonymized_id=f"user_{i}",
                    user_id="clinician_001"
                )
            
            # Generate report
            start_date = datetime.utcnow() - timedelta(hours=1)
            end_date = datetime.utcnow() + timedelta(hours=1)
            
            report = logger.generate_audit_report(start_date, end_date)
            
            assert report["total_events"] == 5
            assert "screening_request" in report["event_counts"]
            assert report["event_counts"]["screening_request"] == 5
            assert report["unique_individuals"] == 5


class TestDriftMonitor:
    """Test DriftMonitor class"""
    
    def test_initialization(self):
        """Test drift monitor initializes correctly"""
        monitor = DriftMonitor(
            feature_drift_threshold=0.3,
            prediction_drift_threshold=0.2
        )
        
        assert monitor.feature_drift_threshold == 0.3
        assert monitor.prediction_drift_threshold == 0.2
        assert monitor.reference_features is None
        assert monitor.reference_predictions is None
    
    def test_set_reference_data(self):
        """Test setting reference data"""
        monitor = DriftMonitor()
        
        features = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1]
        })
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        monitor.set_reference_data(features, predictions)
        
        assert monitor.reference_features is not None
        assert len(monitor.reference_features) == 5
        assert monitor.reference_predictions is not None
        assert len(monitor.reference_predictions) == 5
    
    def test_detect_feature_drift_no_drift(self):
        """Test feature drift detection with no drift"""
        monitor = DriftMonitor(feature_drift_threshold=0.5)
        
        # Set reference data
        np.random.seed(42)
        reference = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100)
        })
        monitor.set_reference_data(reference)
        
        # Current data from same distribution
        current = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100)
        })
        
        report = monitor.detect_feature_drift(current, method="ks")
        
        assert isinstance(report, DriftReport)
        assert report.drift_detected is False
        assert report.drift_score < 0.5
    
    def test_detect_feature_drift_with_drift(self):
        """Test feature drift detection with drift"""
        monitor = DriftMonitor(feature_drift_threshold=0.2)
        
        # Set reference data
        np.random.seed(42)
        reference = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100)
        })
        monitor.set_reference_data(reference)
        
        # Current data from different distribution
        current = pd.DataFrame({
            "feature1": np.random.normal(5, 1, 100)  # Shifted mean
        })
        
        report = monitor.detect_feature_drift(current, method="ks")
        
        assert report.drift_detected is True
        assert report.drift_score > 0.2
    
    def test_detect_prediction_drift(self):
        """Test prediction drift detection"""
        monitor = DriftMonitor(prediction_drift_threshold=0.3)
        
        # Set reference predictions
        np.random.seed(42)
        reference_preds = np.random.uniform(0, 1, 100)
        monitor.set_reference_data(pd.DataFrame(), reference_preds)
        
        # Current predictions from similar distribution
        current_preds = np.random.uniform(0, 1, 100)
        
        report = monitor.detect_prediction_drift(current_preds, method="ks")
        
        assert isinstance(report, DriftReport)
        assert report.drift_score >= 0


class TestHumanReviewQueue:
    """Test HumanReviewQueue class"""
    
    def test_initialization(self):
        """Test review queue initializes correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(
                queue_file=str(queue_file),
                escalation_threshold_hours=4
            )
            
            assert queue.queue_file == queue_file
            assert queue.escalation_threshold == timedelta(hours=4)
            assert len(queue.cases) == 0
    
    def test_enqueue_case(self):
        """Test enqueueing a case"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(queue_file=str(queue_file))
            
            case_id = queue.enqueue_case(
                anonymized_id="user_123",
                risk_score=85.0,
                risk_level="critical",
                prediction_data={"score": 85.0},
                features={"feature1": 1.0},
                priority=2
            )
            
            assert case_id is not None
            assert len(queue.cases) == 1
            assert queue.cases[case_id].risk_score == 85.0
            assert queue.cases[case_id].priority == 2
    
    def test_get_pending_cases(self):
        """Test getting pending cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(queue_file=str(queue_file))
            
            # Enqueue multiple cases
            for i in range(3):
                queue.enqueue_case(
                    anonymized_id=f"user_{i}",
                    risk_score=75.0 + i * 5,
                    risk_level="high",
                    prediction_data={},
                    features={},
                    priority=i
                )
            
            pending = queue.get_pending_cases()
            
            assert len(pending) == 3
            # Should be sorted by priority descending
            assert pending[0].priority >= pending[1].priority
    
    def test_assign_case(self):
        """Test assigning a case to reviewer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(queue_file=str(queue_file))
            
            case_id = queue.enqueue_case(
                anonymized_id="user_assign",
                risk_score=80.0,
                risk_level="high",
                prediction_data={},
                features={}
            )
            
            queue.assign_case(case_id, "reviewer_001")
            
            case = queue.get_case(case_id)
            assert case.assigned_to == "reviewer_001"
            assert case.status == ReviewStatus.ASSIGNED
    
    def test_submit_review(self):
        """Test submitting a review"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(queue_file=str(queue_file))
            
            case_id = queue.enqueue_case(
                anonymized_id="user_review",
                risk_score=78.0,
                risk_level="high",
                prediction_data={},
                features={}
            )
            
            queue.assign_case(case_id, "reviewer_001")
            queue.start_review(case_id, "reviewer_001")
            
            queue.submit_review(
                case_id=case_id,
                reviewer_id="reviewer_001",
                decision="CONFIRM",
                justification="Risk confirmed"
            )
            
            case = queue.get_case(case_id)
            assert case.status == ReviewStatus.COMPLETED
            assert case.decision == ReviewDecision.CONFIRM
            assert case.reviewed_at is not None
    
    def test_check_escalations(self):
        """Test checking for escalations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(
                queue_file=str(queue_file),
                escalation_threshold_hours=0  # Immediate escalation for testing
            )
            
            case_id = queue.enqueue_case(
                anonymized_id="user_escalate",
                risk_score=90.0,
                risk_level="critical",
                prediction_data={},
                features={}
            )
            
            # Manually set created_at to past
            queue.cases[case_id].created_at = datetime.utcnow() - timedelta(hours=5)
            
            escalated = queue.check_escalations()
            
            assert len(escalated) == 1
            assert escalated[0].case_id == case_id
            assert escalated[0].status == ReviewStatus.ESCALATED
    
    def test_get_queue_statistics(self):
        """Test getting queue statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_file = Path(tmpdir) / "queue.json"
            queue = HumanReviewQueue(queue_file=str(queue_file))
            
            # Enqueue cases
            for i in range(5):
                queue.enqueue_case(
                    anonymized_id=f"user_{i}",
                    risk_score=75.0 + i * 5,
                    risk_level="high",
                    prediction_data={},
                    features={}
                )
            
            stats = queue.get_queue_statistics()
            
            assert stats["total_cases"] == 5
            assert stats["pending_count"] == 5
            assert "by_status" in stats
            assert "by_priority" in stats


class TestCrisisOverride:
    """Test CrisisOverride class"""
    
    def test_initialization(self):
        """Test crisis override initializes correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            assert crisis.override_file == override_file
            assert len(crisis.overrides) == 0
    
    def test_initiate_override(self):
        """Test initiating a crisis override"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            override_id = crisis.initiate_override(
                case_id="case_123",
                anonymized_id="user_override",
                clinician_id="clinician_001",
                reason="immediate_safety_concern",
                justification="Patient expressed suicidal ideation",
                original_recommendation={"action": "schedule_appointment"},
                override_action="immediate_hospitalization"
            )
            
            assert override_id is not None
            assert len(crisis.overrides) == 1
            
            override = crisis.overrides[override_id]
            assert override.clinician_id == "clinician_001"
            assert override.reason == "immediate_safety_concern"
            assert override.supervisor_notified is True
    
    def test_get_override_history(self):
        """Test getting override history"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            # Create multiple overrides
            for i in range(3):
                crisis.initiate_override(
                    case_id=f"case_{i}",
                    anonymized_id=f"user_{i}",
                    clinician_id="clinician_001",
                    reason="safety_concern",
                    justification=f"Override {i}",
                    original_recommendation={},
                    override_action="escalate"
                )
            
            history = crisis.get_override_history()
            
            assert len(history) == 3
    
    def test_get_override_history_filtered(self):
        """Test getting filtered override history"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            # Create overrides for different clinicians
            crisis.initiate_override(
                case_id="case_1",
                anonymized_id="user_1",
                clinician_id="clinician_001",
                reason="safety",
                justification="Test",
                original_recommendation={},
                override_action="escalate"
            )
            
            crisis.initiate_override(
                case_id="case_2",
                anonymized_id="user_2",
                clinician_id="clinician_002",
                reason="safety",
                justification="Test",
                original_recommendation={},
                override_action="escalate"
            )
            
            # Filter by clinician
            history = crisis.get_override_history(clinician_id="clinician_001")
            
            assert len(history) == 1
            assert history[0].clinician_id == "clinician_001"
    
    def test_supervisor_review(self):
        """Test supervisor review of override"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            override_id = crisis.initiate_override(
                case_id="case_review",
                anonymized_id="user_review",
                clinician_id="clinician_001",
                reason="safety",
                justification="Test",
                original_recommendation={},
                override_action="escalate"
            )
            
            crisis.supervisor_review(
                override_id=override_id,
                supervisor_id="supervisor_001",
                comments="Appropriate action taken"
            )
            
            override = crisis.overrides[override_id]
            assert override.reviewed_by_supervisor == "supervisor_001"
            assert override.supervisor_review_at is not None
            assert override.supervisor_comments == "Appropriate action taken"
    
    def test_get_override_statistics(self):
        """Test getting override statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            # Create multiple overrides
            for i in range(5):
                crisis.initiate_override(
                    case_id=f"case_{i}",
                    anonymized_id=f"user_{i}",
                    clinician_id=f"clinician_{i % 2}",
                    reason="safety_concern",
                    justification=f"Override {i}",
                    original_recommendation={},
                    override_action="escalate"
                )
            
            stats = crisis.get_override_statistics()
            
            assert stats["total_overrides"] == 5
            assert stats["unique_clinicians"] == 2
            assert stats["unique_cases"] == 5
            assert "by_reason" in stats
            assert "by_clinician" in stats
    
    def test_get_pending_supervisor_reviews(self):
        """Test getting pending supervisor reviews"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            # Create override
            override_id = crisis.initiate_override(
                case_id="case_pending",
                anonymized_id="user_pending",
                clinician_id="clinician_001",
                reason="safety",
                justification="Test",
                original_recommendation={},
                override_action="escalate"
            )
            
            pending = crisis.get_pending_supervisor_reviews()
            
            assert len(pending) == 1
            assert pending[0].override_id == override_id
            
            # Review the override
            crisis.supervisor_review(override_id, "supervisor_001")
            
            pending = crisis.get_pending_supervisor_reviews()
            assert len(pending) == 0
    
    def test_get_clinician_override_count(self):
        """Test getting clinician override count"""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_file = Path(tmpdir) / "overrides.json"
            crisis = CrisisOverride(override_file=str(override_file))
            
            # Create overrides for specific clinician
            for i in range(3):
                crisis.initiate_override(
                    case_id=f"case_{i}",
                    anonymized_id=f"user_{i}",
                    clinician_id="clinician_001",
                    reason="safety",
                    justification=f"Override {i}",
                    original_recommendation={},
                    override_action="escalate"
                )
            
            count = crisis.get_clinician_override_count("clinician_001", days=30)
            
            assert count == 3


class TestDriftReport:
    """Test DriftReport class"""
    
    def test_drift_report_creation(self):
        """Test creating drift report"""
        report = DriftReport(
            drift_detected=True,
            drift_score=0.45,
            threshold=0.3,
            feature_drifts={"feature1": 0.45, "feature2": 0.2}
        )
        
        assert report.drift_detected is True
        assert report.drift_score == 0.45
        assert report.threshold == 0.3
        assert len(report.feature_drifts) == 2
    
    def test_drift_report_to_dict(self):
        """Test converting drift report to dictionary"""
        report = DriftReport(
            drift_detected=False,
            drift_score=0.15,
            threshold=0.3
        )
        
        report_dict = report.to_dict()
        
        assert report_dict["drift_detected"] is False
        assert report_dict["drift_score"] == 0.15
        assert report_dict["threshold"] == 0.3
        assert "timestamp" in report_dict


class TestReviewCase:
    """Test ReviewCase class"""
    
    def test_review_case_creation(self):
        """Test creating review case"""
        case = ReviewCase(
            case_id="case_123",
            anonymized_id="user_123",
            risk_score=85.0,
            risk_level="critical",
            prediction_data={"score": 85.0},
            features={"feature1": 1.0},
            priority=2
        )
        
        assert case.case_id == "case_123"
        assert case.risk_score == 85.0
        assert case.status == ReviewStatus.PENDING
        assert case.priority == 2
    
    def test_review_case_to_dict(self):
        """Test converting review case to dictionary"""
        case = ReviewCase(
            case_id="case_456",
            anonymized_id="user_456",
            risk_score=78.0,
            risk_level="high",
            prediction_data={},
            features={}
        )
        
        case_dict = case.to_dict()
        
        assert case_dict["case_id"] == "case_456"
        assert case_dict["risk_score"] == 78.0
        assert case_dict["status"] == "PENDING"
    
    def test_review_case_from_dict(self):
        """Test creating review case from dictionary"""
        case_dict = {
            "case_id": "case_789",
            "anonymized_id": "user_789",
            "risk_score": 90.0,
            "risk_level": "critical",
            "prediction_data": {},
            "features": {},
            "created_at": datetime.utcnow().isoformat(),
            "priority": 3,
            "status": "PENDING",
            "assigned_to": None,
            "reviewed_at": None,
            "decision": None,
            "modified_risk_score": None,
            "justification": None,
            "escalated_at": None
        }
        
        case = ReviewCase.from_dict(case_dict)
        
        assert case.case_id == "case_789"
        assert case.risk_score == 90.0
        assert case.priority == 3


class TestOverride:
    """Test Override class"""
    
    def test_override_creation(self):
        """Test creating override"""
        override = Override(
            override_id="override_123",
            case_id="case_123",
            anonymized_id="user_123",
            clinician_id="clinician_001",
            reason="safety_concern",
            justification="Immediate risk",
            original_recommendation={"action": "schedule"},
            override_action="hospitalize"
        )
        
        assert override.override_id == "override_123"
        assert override.clinician_id == "clinician_001"
        assert override.supervisor_notified is False
    
    def test_override_to_dict(self):
        """Test converting override to dictionary"""
        override = Override(
            override_id="override_456",
            case_id="case_456",
            anonymized_id="user_456",
            clinician_id="clinician_002",
            reason="emergency",
            justification="Test",
            original_recommendation={},
            override_action="escalate"
        )
        
        override_dict = override.to_dict()
        
        assert override_dict["override_id"] == "override_456"
        assert override_dict["clinician_id"] == "clinician_002"
        assert "created_at" in override_dict
    
    def test_override_from_dict(self):
        """Test creating override from dictionary"""
        override_dict = {
            "override_id": "override_789",
            "case_id": "case_789",
            "anonymized_id": "user_789",
            "clinician_id": "clinician_003",
            "reason": "crisis",
            "justification": "Test",
            "original_recommendation": {},
            "override_action": "admit",
            "created_at": datetime.utcnow().isoformat(),
            "supervisor_notified": True,
            "supervisor_notified_at": datetime.utcnow().isoformat(),
            "reviewed_by_supervisor": "supervisor_001",
            "supervisor_review_at": datetime.utcnow().isoformat(),
            "supervisor_comments": "Appropriate"
        }
        
        override = Override.from_dict(override_dict)
        
        assert override.override_id == "override_789"
        assert override.supervisor_notified is True
        assert override.reviewed_by_supervisor == "supervisor_001"
