"""Delete screening records past the configured governance retention period."""

from src.config import settings
from src.firebase_admin import get_firestore_client
from src.privacy import delete_expired_audit_logs, delete_expired_screenings


def main() -> None:
    db = get_firestore_client()
    if db is None:
        raise RuntimeError("Firestore is required for retention cleanup")
    delete_expired_screenings(db, settings.governance.screening_retention_days)
    delete_expired_audit_logs(db, settings.governance.audit_log_retention_days)


if __name__ == "__main__":
    main()