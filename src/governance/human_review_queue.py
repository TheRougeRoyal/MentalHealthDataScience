"""
Human review queue for high-risk cases requiring clinician oversight.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path
import logging

from src.logging_config import get_logger

logger = get_logger(__name__)


class ReviewStatus(Enum):
    """Status of a review case."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_REVIEW = "IN_REVIEW"
    COMPLETED = "COMPLETED"
    ESCALATED = "ESCALATED"


class ReviewDecision(Enum):
    """Decision made by reviewer."""
    CONFIRM = "CONFIRM"
    MODIFY = "MODIFY"
    OVERRIDE = "OVERRIDE"


class ReviewCase:
    """Represents a case requiring human review."""
    
    def __init__(
        self,
        case_id: str,
        anonymized_id: str,
        risk_score: float,
        risk_level: str,
        prediction_data: Dict[str, Any],
        features: Dict[str, Any],
        created_at: Optional[datetime] = None,
        priority: int = 1
    ):
        self.case_id = case_id
        self.anonymized_id = anonymized_id
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.prediction_data = prediction_data
        self.features = features
        self.created_at = created_at or datetime.utcnow()
        self.priority = priority
        self.status = ReviewStatus.PENDING
        self.assigned_to: Optional[str] = None
        self.reviewed_at: Optional[datetime] = None
        self.decision: Optional[ReviewDecision] = None
        self.modified_risk_score: Optional[float] = None
        self.justification: Optional[str] = None
        self.escalated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert case to dictionary."""
        return {
            "case_id": self.case_id,
            "anonymized_id": self.anonymized_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "prediction_data": self.prediction_data,
            "features": self.features,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "decision": self.decision.value if self.decision else None,
            "modified_risk_score": self.modified_risk_score,
            "justification": self.justification,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewCase':
        """Create case from dictionary."""
        case = cls(
            case_id=data["case_id"],
            anonymized_id=data["anonymized_id"],
            risk_score=data["risk_score"],
            risk_level=data["risk_level"],
            prediction_data=data["prediction_data"],
            features=data["features"],
            created_at=datetime.fromisoformat(data["created_at"]),
            priority=data.get("priority", 1)
        )
        case.status = ReviewStatus(data["status"])
        case.assigned_to = data.get("assigned_to")
        if data.get("reviewed_at"):
            case.reviewed_at = datetime.fromisoformat(data["reviewed_at"])
        if data.get("decision"):
            case.decision = ReviewDecision(data["decision"])
        case.modified_risk_score = data.get("modified_risk_score")
        case.justification = data.get("justification")
        if data.get("escalated_at"):
            case.escalated_at = datetime.fromisoformat(data["escalated_at"])
        return case



class HumanReviewQueue:
    """
    Manages queue of cases requiring human review.
    
    Routes high-risk cases to clinicians, tracks review status,
    and escalates overdue cases.
    """
    
    def __init__(
        self,
        queue_file: str = "data/review_queue.json",
        escalation_threshold_hours: int = 4,
        escalation_callback: Optional[callable] = None
    ):
        """
        Initialize the HumanReviewQueue.
        
        Args:
            queue_file: Path to queue persistence file
            escalation_threshold_hours: Hours before escalation
            escalation_callback: Optional callback for escalations
        """
        self.queue_file = Path(queue_file)
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.escalation_threshold = timedelta(hours=escalation_threshold_hours)
        self.escalation_callback = escalation_callback
        
        # In-memory queue
        self.cases: Dict[str, ReviewCase] = {}
        
        # Load existing queue
        self._load_queue()
        
        logger.info(
            f"HumanReviewQueue initialized with {len(self.cases)} cases, "
            f"escalation_threshold={escalation_threshold_hours}h"
        )
    
    def _load_queue(self) -> None:
        """Load queue from file."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, "r") as f:
                    data = json.load(f)
                    for case_data in data.get("cases", []):
                        case = ReviewCase.from_dict(case_data)
                        self.cases[case.case_id] = case
                logger.info(f"Loaded {len(self.cases)} cases from queue file")
            except Exception as e:
                logger.error(f"Error loading queue file: {e}")
    
    def _save_queue(self) -> None:
        """Save queue to file."""
        try:
            data = {
                "cases": [case.to_dict() for case in self.cases.values()],
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(self.queue_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving queue file: {e}")

    
    def enqueue_case(
        self,
        anonymized_id: str,
        risk_score: float,
        risk_level: str,
        prediction_data: Dict[str, Any],
        features: Dict[str, Any],
        priority: Optional[int] = None
    ) -> str:
        """
        Add a case to the review queue for risk scores > 75.
        
        Args:
            anonymized_id: Anonymized identifier for the individual
            risk_score: Risk score from the model
            risk_level: Risk level classification
            prediction_data: Full prediction data
            features: Feature data used for prediction
            priority: Priority level (higher = more urgent)
            
        Returns:
            Case ID
        """
        # Determine priority based on risk score if not provided
        if priority is None:
            if risk_score >= 90:
                priority = 3  # Critical
            elif risk_score >= 80:
                priority = 2  # High
            else:
                priority = 1  # Standard
        
        case_id = str(uuid.uuid4())
        
        case = ReviewCase(
            case_id=case_id,
            anonymized_id=anonymized_id,
            risk_score=risk_score,
            risk_level=risk_level,
            prediction_data=prediction_data,
            features=features,
            priority=priority
        )
        
        self.cases[case_id] = case
        self._save_queue()
        
        logger.info(
            f"Enqueued case {case_id} for {anonymized_id}, "
            f"risk_score={risk_score}, priority={priority}"
        )
        
        return case_id
    
    def get_pending_cases(
        self,
        reviewer_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ReviewCase]:
        """
        Get pending cases for reviewer assignment.
        
        Args:
            reviewer_id: Filter by assigned reviewer
            limit: Maximum number of cases to return
            
        Returns:
            List of pending review cases
        """
        # Filter cases
        pending = []
        for case in self.cases.values():
            # Filter by status
            if case.status not in [ReviewStatus.PENDING, ReviewStatus.ASSIGNED, ReviewStatus.IN_REVIEW]:
                continue
            
            # Filter by reviewer if specified
            if reviewer_id is not None:
                if case.assigned_to != reviewer_id:
                    continue
            
            pending.append(case)
        
        # Sort by priority (descending) and created_at (ascending)
        pending.sort(key=lambda c: (-c.priority, c.created_at))
        
        # Apply limit
        if limit is not None:
            pending = pending[:limit]
        
        logger.info(
            f"Retrieved {len(pending)} pending cases"
            + (f" for reviewer {reviewer_id}" if reviewer_id else "")
        )
        
        return pending

    
    def assign_case(
        self,
        case_id: str,
        reviewer_id: str
    ) -> None:
        """
        Assign a case to a reviewer.
        
        Args:
            case_id: ID of the case
            reviewer_id: ID of the reviewer
        """
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        case.assigned_to = reviewer_id
        case.status = ReviewStatus.ASSIGNED
        
        self._save_queue()
        
        logger.info(f"Assigned case {case_id} to reviewer {reviewer_id}")
    
    def start_review(
        self,
        case_id: str,
        reviewer_id: str
    ) -> ReviewCase:
        """
        Mark a case as in review.
        
        Args:
            case_id: ID of the case
            reviewer_id: ID of the reviewer
            
        Returns:
            ReviewCase object
        """
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        
        # Verify reviewer
        if case.assigned_to and case.assigned_to != reviewer_id:
            logger.warning(
                f"Case {case_id} assigned to {case.assigned_to}, "
                f"but {reviewer_id} is starting review"
            )
        
        case.assigned_to = reviewer_id
        case.status = ReviewStatus.IN_REVIEW
        
        self._save_queue()
        
        logger.info(f"Started review of case {case_id} by {reviewer_id}")
        
        return case
    
    def submit_review(
        self,
        case_id: str,
        reviewer_id: str,
        decision: str,
        modified_risk_score: Optional[float] = None,
        justification: Optional[str] = None
    ) -> None:
        """
        Submit a review decision for a case.
        
        Args:
            case_id: ID of the case
            reviewer_id: ID of the reviewer
            decision: Review decision (CONFIRM, MODIFY, OVERRIDE)
            modified_risk_score: Modified risk score if changed
            justification: Justification for the decision
        """
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        case = self.cases[case_id]
        
        # Verify reviewer
        if case.assigned_to != reviewer_id:
            raise ValueError(
                f"Case {case_id} is assigned to {case.assigned_to}, "
                f"not {reviewer_id}"
            )
        
        # Update case
        case.decision = ReviewDecision(decision)
        case.modified_risk_score = modified_risk_score
        case.justification = justification
        case.reviewed_at = datetime.utcnow()
        case.status = ReviewStatus.COMPLETED
        
        self._save_queue()
        
        logger.info(
            f"Review submitted for case {case_id} by {reviewer_id}, "
            f"decision={decision}"
        )

    
    def check_escalations(self) -> List[ReviewCase]:
        """
        Check for cases that need escalation (not reviewed within threshold).
        
        Returns:
            List of cases that were escalated
        """
        now = datetime.utcnow()
        escalated_cases = []
        
        for case in self.cases.values():
            # Skip completed or already escalated cases
            if case.status in [ReviewStatus.COMPLETED, ReviewStatus.ESCALATED]:
                continue
            
            # Check if case is overdue
            time_in_queue = now - case.created_at
            if time_in_queue > self.escalation_threshold:
                # Escalate case
                case.status = ReviewStatus.ESCALATED
                case.escalated_at = now
                escalated_cases.append(case)
                
                logger.warning(
                    f"Escalating case {case.case_id} - in queue for "
                    f"{time_in_queue.total_seconds() / 3600:.1f} hours"
                )
                
                # Call escalation callback if provided
                if self.escalation_callback:
                    try:
                        self.escalation_callback(case)
                    except Exception as e:
                        logger.error(f"Error in escalation callback: {e}")
        
        if escalated_cases:
            self._save_queue()
            logger.warning(f"Escalated {len(escalated_cases)} cases")
        
        return escalated_cases
    
    def get_case(self, case_id: str) -> Optional[ReviewCase]:
        """
        Get a specific case by ID.
        
        Args:
            case_id: ID of the case
            
        Returns:
            ReviewCase or None if not found
        """
        return self.cases.get(case_id)
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the review queue.
        
        Returns:
            Dictionary with queue statistics
        """
        stats = {
            "total_cases": len(self.cases),
            "by_status": {},
            "by_priority": {},
            "pending_count": 0,
            "overdue_count": 0,
            "average_review_time_hours": None,
            "oldest_pending_hours": None
        }
        
        # Count by status
        for status in ReviewStatus:
            stats["by_status"][status.value] = 0
        
        for case in self.cases.values():
            stats["by_status"][case.status.value] += 1
        
        # Count by priority
        for case in self.cases.values():
            priority = case.priority
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
        
        # Pending and overdue counts
        now = datetime.utcnow()
        pending_cases = []
        completed_review_times = []
        
        for case in self.cases.values():
            if case.status in [ReviewStatus.PENDING, ReviewStatus.ASSIGNED, ReviewStatus.IN_REVIEW]:
                stats["pending_count"] += 1
                pending_cases.append(case)
                
                time_in_queue = now - case.created_at
                if time_in_queue > self.escalation_threshold:
                    stats["overdue_count"] += 1
            
            if case.status == ReviewStatus.COMPLETED and case.reviewed_at:
                review_time = (case.reviewed_at - case.created_at).total_seconds() / 3600
                completed_review_times.append(review_time)
        
        # Average review time
        if completed_review_times:
            stats["average_review_time_hours"] = sum(completed_review_times) / len(completed_review_times)
        
        # Oldest pending case
        if pending_cases:
            oldest = min(pending_cases, key=lambda c: c.created_at)
            stats["oldest_pending_hours"] = (now - oldest.created_at).total_seconds() / 3600
        
        return stats
