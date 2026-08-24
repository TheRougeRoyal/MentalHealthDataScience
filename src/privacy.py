"""Privacy controls for persisted screening data."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from cryptography.fernet import Fernet
from firebase_admin import firestore
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PERSISTED_FEATURES = frozenset({
    "phq9_score", "gad7_score", "sleep_hours", "avg_heart_rate",
    "diagnosis_codes", "medications",
})


def minimize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only features required by the configured risk model."""
    return {key: data[key] for key in _PERSISTED_FEATURES if key in data}


def _fernet() -> Fernet:
    key = os.environ.get("SECURITY_DATA_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("SECURITY_DATA_ENCRYPTION_KEY is required for Firestore persistence")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("SECURITY_DATA_ENCRYPTION_KEY must be a valid Fernet key") from exc


def encrypt_input(data: Dict[str, Any]) -> str:
    """Encrypt minimized clinical inputs before they leave the process."""
    payload = json.dumps(minimize_input(data), separators=(",", ":"), sort_keys=True)
    return _fernet().encrypt(payload.encode("utf-8")).decode("ascii")


def write_audit_event(db: Any, *, action: str, user_id: str, screening_id: str) -> None:
    """Record access metadata without copying sensitive screening content."""
    db.collection("audit_logs").document().set({
        "action": action,
        "actor_user_id": user_id,
        "resource_type": "screening",
        "resource_id": screening_id,
        "created_at": firestore.SERVER_TIMESTAMP,
    })


def delete_expired_screenings(db: Any, retention_days: int) -> int:
    """Delete expired screening records and their linked explanations/reviews."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0
    for document in db.collection("screenings").where("created_at", "<", cutoff).stream():
        screening_id = document.id
        db.collection("screenings").document(screening_id).delete()
        db.collection("explanations").document(screening_id).delete()
        db.collection("reviews").document(screening_id).delete()
        deleted += 1
    logger.info("Deleted %d expired screening records", deleted)
    return deleted


def delete_expired_audit_logs(db: Any, retention_days: int) -> int:
    """Delete audit metadata past its configured retention period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0
    for document in db.collection("audit_logs").where("created_at", "<", cutoff).stream():
        document.reference.delete()
        deleted += 1
    logger.info("Deleted %d expired audit events", deleted)
    return deleted