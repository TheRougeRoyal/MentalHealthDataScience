"""
Crisis override system for emergency workflows.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

from src.logging_config import get_logger

logger = get_logger(__name__)


class Override:
    """Represents a crisis override event."""
    
    def __init__(
        self,
        override_id: str,
        case_id: str,
        anonymized_id: str,
        clinician_id: str,
        reason: str,
        justification: str,
        original_recommendation: Dict[str, Any],
        override_action: str,
        created_at: Optional[datetime] = None
    ):
        self.override_id = override_id
        self.case_id = case_id
        self.anonymized_id = anonymized_id
        self.clinician_id = clinician_id
        self.reason = reason
        self.justification = justification
        self.original_recommendation = original_recommendation
        self.override_action = override_action
        self.created_at = created_at or datetime.utcnow()
        self.supervisor_notified = False
        self.supervisor_notified_at: Optional[datetime] = None
        self.reviewed_by_supervisor: Optional[str] = None
        self.supervisor_review_at: Optional[datetime] = None
        self.supervisor_comments: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert override to dictionary."""
        return {
            "override_id": self.override_id,
            "case_id": self.case_id,
            "anonymized_id": self.anonymized_id,
            "clinician_id": self.clinician_id,
            "reason": self.reason,
            "justification": self.justification,
            "original_recommendation": self.original_recommendation,
            "override_action": self.override_action,
            "created_at": self.created_at.isoformat(),
            "supervisor_notified": self.supervisor_notified,
            "supervisor_notified_at": self.supervisor_notified_at.isoformat() if self.supervisor_notified_at else None,
            "reviewed_by_supervisor": self.reviewed_by_supervisor,
            "supervisor_review_at": self.supervisor_review_at.isoformat() if self.supervisor_review_at else None,
            "supervisor_comments": self.supervisor_comments
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Override':
        """Create override from dictionary."""
        override = cls(
            override_id=data["override_id"],
            case_id=data["case_id"],
            anonymized_id=data["anonymized_id"],
            clinician_id=data["clinician_id"],
            reason=data["reason"],
            justification=data["justification"],
            original_recommendation=data["original_recommendation"],
            override_action=data["override_action"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
        override.supervisor_notified = data.get("supervisor_notified", False)
        if data.get("supervisor_notified_at"):
            override.supervisor_notified_at = datetime.fromisoformat(data["supervisor_notified_at"])
        override.reviewed_by_supervisor = data.get("reviewed_by_supervisor")
        if data.get("supervisor_review_at"):
            override.supervisor_review_at = datetime.fromisoformat(data["supervisor_review_at"])
        override.supervisor_comments = data.get("supervisor_comments")
        return override


class CrisisOverride:
    """
    Manages crisis override workflows for emergency situations.
    
    Allows clinicians to bypass automated recommendations in emergencies,
    with full audit trail and supervisor notification.
    """
    
    def __init__(
        self,
        override_file: str = "data/crisis_overrides.json",
        notification_callback: Optional[callable] = None
    ):
        """
        Initialize the CrisisOverride system.
        
        Args:
            override_file: Path to override persistence file
            notification_callback: Optional callback for supervisor notifications
        """
        self.override_file = Path(override_file)
        self.override_file.parent.mkdir(parents=True, exist_ok=True)
        self.notification_callback = notification_callback
        
        # In-memory storage
        self.overrides: Dict[str, Override] = {}
        
        # Load existing overrides
        self._load_overrides()
        
        logger.info(
            f"CrisisOverride initialized with {len(self.overrides)} historical overrides"
        )
    
    def _load_overrides(self) -> None:
        """Load overrides from file."""
        if self.override_file.exists():
            try:
                with open(self.override_file, "r") as f:
                    data = json.load(f)
                    for override_data in data.get("overrides", []):
                        override = Override.from_dict(override_data)
                        self.overrides[override.override_id] = override
                logger.info(f"Loaded {len(self.overrides)} overrides from file")
            except Exception as e:
                logger.error(f"Error loading override file: {e}")
    
    def _save_overrides(self) -> None:
        """Save overrides to file."""
        try:
            data = {
                "overrides": [override.to_dict() for override in self.overrides.values()],
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(self.override_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving override file: {e}")

    
    def initiate_override(
        self,
        case_id: str,
        anonymized_id: str,
        clinician_id: str,
        reason: str,
        justification: str,
        original_recommendation: Dict[str, Any],
        override_action: str
    ) -> str:
        """
        Initiate a crisis override with clinician ID and justification.
        
        Args:
            case_id: ID of the case being overridden
            anonymized_id: Anonymized identifier for the individual
            clinician_id: ID of the clinician initiating override
            reason: Reason for override (e.g., "immediate_safety_concern")
            justification: Detailed justification for the override
            original_recommendation: Original system recommendation
            override_action: Action taken by clinician
            
        Returns:
            Override ID
        """
        override_id = str(uuid.uuid4())
        
        override = Override(
            override_id=override_id,
            case_id=case_id,
            anonymized_id=anonymized_id,
            clinician_id=clinician_id,
            reason=reason,
            justification=justification,
            original_recommendation=original_recommendation,
            override_action=override_action
        )
        
        self.overrides[override_id] = override
        self._save_overrides()
        
        logger.warning(
            f"Crisis override initiated: override_id={override_id}, "
            f"case_id={case_id}, clinician={clinician_id}, reason={reason}"
        )
        
        # Notify supervisor immediately
        self._notify_supervisor(override)
        
        return override_id
    
    def _notify_supervisor(self, override: Override) -> None:
        """
        Notify supervisor of override.
        
        Args:
            override: Override object
        """
        override.supervisor_notified = True
        override.supervisor_notified_at = datetime.utcnow()
        
        notification_message = (
            f"CRISIS OVERRIDE ALERT: Clinician {override.clinician_id} "
            f"initiated override for case {override.case_id}. "
            f"Reason: {override.reason}. "
            f"Justification: {override.justification}"
        )
        
        logger.warning(notification_message)
        
        # Call custom notification callback if provided
        if self.notification_callback:
            try:
                self.notification_callback(override)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
        
        self._save_overrides()
    
    def get_override_history(
        self,
        case_id: Optional[str] = None,
        anonymized_id: Optional[str] = None,
        clinician_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Override]:
        """
        Get override history with optional filters.
        
        Args:
            case_id: Filter by case ID
            anonymized_id: Filter by anonymized ID
            clinician_id: Filter by clinician ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of Override objects
        """
        overrides = list(self.overrides.values())
        
        # Apply filters
        if case_id:
            overrides = [o for o in overrides if o.case_id == case_id]
        
        if anonymized_id:
            overrides = [o for o in overrides if o.anonymized_id == anonymized_id]
        
        if clinician_id:
            overrides = [o for o in overrides if o.clinician_id == clinician_id]
        
        if start_date:
            overrides = [o for o in overrides if o.created_at >= start_date]
        
        if end_date:
            overrides = [o for o in overrides if o.created_at <= end_date]
        
        # Sort by created_at descending
        overrides.sort(key=lambda o: o.created_at, reverse=True)
        
        return overrides

    
    def supervisor_review(
        self,
        override_id: str,
        supervisor_id: str,
        comments: Optional[str] = None
    ) -> None:
        """
        Record supervisor review of an override.
        
        Args:
            override_id: ID of the override
            supervisor_id: ID of the supervisor
            comments: Supervisor's comments
        """
        if override_id not in self.overrides:
            raise ValueError(f"Override {override_id} not found")
        
        override = self.overrides[override_id]
        override.reviewed_by_supervisor = supervisor_id
        override.supervisor_review_at = datetime.utcnow()
        override.supervisor_comments = comments
        
        self._save_overrides()
        
        logger.info(
            f"Supervisor {supervisor_id} reviewed override {override_id}"
        )
    
    def get_override_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about overrides.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary with override statistics
        """
        overrides = self.get_override_history(
            start_date=start_date,
            end_date=end_date
        )
        
        stats = {
            "total_overrides": len(overrides),
            "by_reason": {},
            "by_clinician": {},
            "supervisor_reviewed_count": 0,
            "pending_supervisor_review": 0,
            "unique_clinicians": set(),
            "unique_cases": set(),
            "unique_individuals": set()
        }
        
        for override in overrides:
            # Count by reason
            reason = override.reason
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
            
            # Count by clinician
            clinician = override.clinician_id
            stats["by_clinician"][clinician] = stats["by_clinician"].get(clinician, 0) + 1
            
            # Supervisor review status
            if override.reviewed_by_supervisor:
                stats["supervisor_reviewed_count"] += 1
            else:
                stats["pending_supervisor_review"] += 1
            
            # Track unique values
            stats["unique_clinicians"].add(override.clinician_id)
            stats["unique_cases"].add(override.case_id)
            stats["unique_individuals"].add(override.anonymized_id)
        
        # Convert sets to counts
        stats["unique_clinicians"] = len(stats["unique_clinicians"])
        stats["unique_cases"] = len(stats["unique_cases"])
        stats["unique_individuals"] = len(stats["unique_individuals"])
        
        return stats
    
    def get_pending_supervisor_reviews(self) -> List[Override]:
        """
        Get overrides pending supervisor review.
        
        Returns:
            List of Override objects
        """
        pending = [
            override for override in self.overrides.values()
            if not override.reviewed_by_supervisor
        ]
        
        # Sort by created_at ascending (oldest first)
        pending.sort(key=lambda o: o.created_at)
        
        return pending
    
    def get_clinician_override_count(
        self,
        clinician_id: str,
        days: int = 30
    ) -> int:
        """
        Get number of overrides by a clinician in recent days.
        
        Args:
            clinician_id: ID of the clinician
            days: Number of days to look back
            
        Returns:
            Count of overrides
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        overrides = self.get_override_history(
            clinician_id=clinician_id,
            start_date=start_date
        )
        return len(overrides)
