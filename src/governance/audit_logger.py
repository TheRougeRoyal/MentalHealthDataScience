"""
Audit logging for compliance tracking and governance.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from src.logging_config import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """
    Logs all system activities for compliance and audit purposes.
    
    Tracks screening requests, predictions, human reviews, and generates
    audit reports for regulatory compliance.
    """
    
    def __init__(self, audit_log_path: str = "logs/audit.jsonl"):
        """
        Initialize the AuditLogger.
        
        Args:
            audit_log_path: Path to the audit log file
        """
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AuditLogger initialized with log path: {audit_log_path}")
    
    def _write_audit_entry(self, entry: Dict[str, Any]) -> None:
        """
        Write an audit entry to the log file.
        
        Args:
            entry: Audit entry dictionary
        """
        # Add timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat()
        
        # Write to file in JSON Lines format
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Also log to structured logger
        logger.info("audit_entry", extra={"audit_data": entry})
    
    def _hash_data(self, data: Any) -> str:
        """
        Create a hash of data for audit trail.
        
        Args:
            data: Data to hash
            
        Returns:
            SHA-256 hash of the data
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_screening_request(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        anonymized_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log a screening request with request and response details.
        
        Args:
            request: Screening request data
            response: Screening response data
            anonymized_id: Anonymized identifier for the individual
            user_id: ID of the user making the request
        """
        # Create hashes instead of storing full data
        request_hash = self._hash_data(request)
        response_hash = self._hash_data(response)
        
        entry = {
            "event_type": "screening_request",
            "anonymized_id": anonymized_id,
            "user_id": user_id,
            "request_hash": request_hash,
            "response_hash": response_hash,
            "risk_score": response.get("risk_score", {}).get("score"),
            "risk_level": response.get("risk_score", {}).get("risk_level"),
            "alert_triggered": response.get("alert_triggered", False),
            "requires_human_review": response.get("requires_human_review", False),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._write_audit_entry(entry)
        logger.info(
            f"Logged screening request for {anonymized_id}, "
            f"risk_level={entry['risk_level']}"
        )
    
    def log_prediction(
        self,
        features_hash: str,
        prediction: Dict[str, Any],
        model_id: str,
        anonymized_id: str,
        model_version: Optional[str] = None
    ) -> None:
        """
        Log a prediction including features hash and model version.
        
        Args:
            features_hash: Hash of the input features
            prediction: Prediction results
            model_id: ID of the model used
            anonymized_id: Anonymized identifier for the individual
            model_version: Version of the model
        """
        entry = {
            "event_type": "prediction",
            "anonymized_id": anonymized_id,
            "features_hash": features_hash,
            "model_id": model_id,
            "model_version": model_version,
            "risk_score": prediction.get("score"),
            "risk_level": prediction.get("risk_level"),
            "confidence": prediction.get("confidence"),
            "ensemble_models": prediction.get("ensemble_models", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._write_audit_entry(entry)
        logger.info(
            f"Logged prediction for {anonymized_id}, "
            f"model={model_id}, score={entry['risk_score']}"
        )
    
    def log_human_review(
        self,
        case_id: str,
        reviewer_id: str,
        decision: str,
        anonymized_id: str,
        original_risk_score: float,
        modified_risk_score: Optional[float] = None,
        justification: Optional[str] = None,
        review_duration_seconds: Optional[float] = None
    ) -> None:
        """
        Log a human review decision.
        
        Args:
            case_id: ID of the case being reviewed
            reviewer_id: ID of the reviewer
            decision: Review decision (CONFIRM, MODIFY, OVERRIDE)
            anonymized_id: Anonymized identifier for the individual
            original_risk_score: Original risk score from the system
            modified_risk_score: Modified risk score if changed
            justification: Justification for the decision
            review_duration_seconds: Time taken for review
        """
        entry = {
            "event_type": "human_review",
            "case_id": case_id,
            "anonymized_id": anonymized_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "original_risk_score": original_risk_score,
            "modified_risk_score": modified_risk_score,
            "justification": justification,
            "review_duration_seconds": review_duration_seconds,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._write_audit_entry(entry)
        logger.info(
            f"Logged human review for case {case_id}, "
            f"decision={decision}, reviewer={reviewer_id}"
        )
    
    def log_consent_verification(
        self,
        anonymized_id: str,
        data_types: List[str],
        consent_status: str,
        consent_expiry: Optional[datetime] = None
    ) -> None:
        """
        Log consent verification events.
        
        Args:
            anonymized_id: Anonymized identifier for the individual
            data_types: Types of data being accessed
            consent_status: Status of consent (VALID, EXPIRED, REVOKED, MISSING)
            consent_expiry: Expiry date of consent
        """
        entry = {
            "event_type": "consent_verification",
            "anonymized_id": anonymized_id,
            "data_types": data_types,
            "consent_status": consent_status,
            "consent_expiry": consent_expiry.isoformat() if consent_expiry else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._write_audit_entry(entry)
    
    def log_data_access(
        self,
        anonymized_id: str,
        user_id: str,
        access_type: str,
        data_types: List[str],
        purpose: Optional[str] = None
    ) -> None:
        """
        Log data access events.
        
        Args:
            anonymized_id: Anonymized identifier for the individual
            user_id: ID of the user accessing data
            access_type: Type of access (READ, WRITE, DELETE)
            data_types: Types of data accessed
            purpose: Purpose of data access
        """
        entry = {
            "event_type": "data_access",
            "anonymized_id": anonymized_id,
            "user_id": user_id,
            "access_type": access_type,
            "data_types": data_types,
            "purpose": purpose,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._write_audit_entry(entry)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        anonymized_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log system errors.
        
        Args:
            error_type: Type of error
            error_message: Error message
            component: Component where error occurred
            anonymized_id: Anonymized identifier if applicable
            context: Additional context
        """
        entry = {
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "component": component,
            "anonymized_id": anonymized_id,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._write_audit_entry(entry)
    
    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an audit report for a date range.
        
        Args:
            start_date: Start date for the report
            end_date: End date for the report
            event_types: Filter by specific event types
            
        Returns:
            Audit report with statistics and summaries
        """
        logger.info(
            f"Generating audit report from {start_date.isoformat()} "
            f"to {end_date.isoformat()}"
        )
        
        # Read audit log entries
        entries = []
        if self.audit_log_path.exists():
            with open(self.audit_log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        
                        # Filter by date range
                        if start_date <= entry_time <= end_date:
                            # Filter by event type if specified
                            if event_types is None or entry.get("event_type") in event_types:
                                entries.append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse audit log entry: {e}")
        
        # Generate statistics
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_events": len(entries),
            "event_counts": {},
            "screening_statistics": {},
            "human_review_statistics": {},
            "consent_statistics": {},
            "error_statistics": {},
            "unique_individuals": set(),
            "unique_reviewers": set()
        }
        
        # Count events by type
        for entry in entries:
            event_type = entry.get("event_type", "unknown")
            report["event_counts"][event_type] = report["event_counts"].get(event_type, 0) + 1
            
            # Track unique individuals
            if "anonymized_id" in entry:
                report["unique_individuals"].add(entry["anonymized_id"])
            
            # Track unique reviewers
            if "reviewer_id" in entry:
                report["unique_reviewers"].add(entry["reviewer_id"])
        
        # Screening statistics
        screening_entries = [e for e in entries if e.get("event_type") == "screening_request"]
        if screening_entries:
            risk_levels = [e.get("risk_level") for e in screening_entries if e.get("risk_level")]
            report["screening_statistics"] = {
                "total_screenings": len(screening_entries),
                "alerts_triggered": sum(1 for e in screening_entries if e.get("alert_triggered")),
                "human_reviews_required": sum(1 for e in screening_entries if e.get("requires_human_review")),
                "risk_level_distribution": {
                    "LOW": risk_levels.count("LOW"),
                    "MODERATE": risk_levels.count("MODERATE"),
                    "HIGH": risk_levels.count("HIGH"),
                    "CRITICAL": risk_levels.count("CRITICAL")
                }
            }
        
        # Human review statistics
        review_entries = [e for e in entries if e.get("event_type") == "human_review"]
        if review_entries:
            decisions = [e.get("decision") for e in review_entries if e.get("decision")]
            durations = [e.get("review_duration_seconds") for e in review_entries if e.get("review_duration_seconds")]
            
            report["human_review_statistics"] = {
                "total_reviews": len(review_entries),
                "decision_distribution": {
                    "CONFIRM": decisions.count("CONFIRM"),
                    "MODIFY": decisions.count("MODIFY"),
                    "OVERRIDE": decisions.count("OVERRIDE")
                },
                "average_review_duration_seconds": sum(durations) / len(durations) if durations else None,
                "unique_reviewers": len(report["unique_reviewers"])
            }
        
        # Consent statistics
        consent_entries = [e for e in entries if e.get("event_type") == "consent_verification"]
        if consent_entries:
            statuses = [e.get("consent_status") for e in consent_entries if e.get("consent_status")]
            report["consent_statistics"] = {
                "total_verifications": len(consent_entries),
                "status_distribution": {
                    "VALID": statuses.count("VALID"),
                    "EXPIRED": statuses.count("EXPIRED"),
                    "REVOKED": statuses.count("REVOKED"),
                    "MISSING": statuses.count("MISSING")
                }
            }
        
        # Error statistics
        error_entries = [e for e in entries if e.get("event_type") == "error"]
        if error_entries:
            error_types = [e.get("error_type") for e in error_entries if e.get("error_type")]
            components = [e.get("component") for e in error_entries if e.get("component")]
            
            report["error_statistics"] = {
                "total_errors": len(error_entries),
                "error_type_distribution": {et: error_types.count(et) for et in set(error_types)},
                "component_distribution": {c: components.count(c) for c in set(components)}
            }
        
        # Convert sets to counts
        report["unique_individuals"] = len(report["unique_individuals"])
        report["unique_reviewers"] = len(report["unique_reviewers"])
        
        logger.info(
            f"Generated audit report: {report['total_events']} events, "
            f"{report['unique_individuals']} unique individuals"
        )
        
        return report
    
    def get_audit_trail(
        self,
        anonymized_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail for a specific individual.
        
        Args:
            anonymized_id: Anonymized identifier for the individual
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of audit entries for the individual
        """
        entries = []
        
        if self.audit_log_path.exists():
            with open(self.audit_log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Filter by anonymized_id
                        if entry.get("anonymized_id") == anonymized_id:
                            # Filter by date range if specified
                            if start_date or end_date:
                                entry_time = datetime.fromisoformat(entry["timestamp"])
                                if start_date and entry_time < start_date:
                                    continue
                                if end_date and entry_time > end_date:
                                    continue
                            
                            entries.append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse audit log entry: {e}")
        
        return entries
