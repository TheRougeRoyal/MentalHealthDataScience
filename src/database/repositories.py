"""Repository classes for database access."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from src.database.connection import DatabaseConnection
from src.database.models import Prediction, AuditLog, Consent, HumanReviewCase

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations."""

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize repository.

        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection


class PredictionRepository(BaseRepository):
    """Repository for predictions table."""

    def create(self, prediction: Prediction) -> UUID:
        """
        Create a new prediction record.

        Args:
            prediction: Prediction model instance

        Returns:
            UUID of created prediction
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    anonymized_id, risk_score, risk_level, confidence,
                    model_version, features_hash, contributing_factors
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    prediction.anonymized_id,
                    prediction.risk_score,
                    prediction.risk_level,
                    prediction.confidence,
                    prediction.model_version,
                    prediction.features_hash,
                    prediction.contributing_factors,
                )
            )
            result = cur.fetchone()
            prediction_id = result["id"]
            logger.info(f"Created prediction {prediction_id} for {prediction.anonymized_id}")
            return prediction_id

    def get_by_id(self, prediction_id: UUID) -> Optional[Prediction]:
        """
        Get prediction by ID.

        Args:
            prediction_id: Prediction UUID

        Returns:
            Prediction instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM predictions WHERE id = %s",
                (str(prediction_id),)
            )
            row = cur.fetchone()
            return Prediction(**row) if row else None

    def get_by_anonymized_id(
        self,
        anonymized_id: str,
        limit: int = 10
    ) -> List[Prediction]:
        """
        Get predictions for an anonymized ID.

        Args:
            anonymized_id: Anonymized identifier
            limit: Maximum number of results

        Returns:
            List of Prediction instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM predictions
                WHERE anonymized_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (anonymized_id, limit)
            )
            rows = cur.fetchall()
            return [Prediction(**row) for row in rows]

    def get_recent_predictions(
        self,
        hours: int = 24,
        risk_level: Optional[str] = None
    ) -> List[Prediction]:
        """
        Get recent predictions within specified hours.

        Args:
            hours: Number of hours to look back
            risk_level: Optional filter by risk level

        Returns:
            List of Prediction instances
        """
        with self.db.get_cursor() as cur:
            if risk_level:
                cur.execute(
                    """
                    SELECT * FROM predictions
                    WHERE created_at >= NOW() - INTERVAL '%s hours'
                    AND risk_level = %s
                    ORDER BY created_at DESC
                    """,
                    (hours, risk_level)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM predictions
                    WHERE created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    """,
                    (hours,)
                )
            rows = cur.fetchall()
            return [Prediction(**row) for row in rows]


class AuditLogRepository(BaseRepository):
    """Repository for audit_log table."""

    def create(self, audit_log: AuditLog) -> UUID:
        """
        Create a new audit log entry.

        Args:
            audit_log: AuditLog model instance

        Returns:
            UUID of created audit log entry
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log (
                    event_type, anonymized_id, user_id, details
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    audit_log.event_type,
                    audit_log.anonymized_id,
                    audit_log.user_id,
                    audit_log.details,
                )
            )
            result = cur.fetchone()
            log_id = result["id"]
            logger.debug(f"Created audit log {log_id} for event {audit_log.event_type}")
            return log_id

    def get_by_event_type(
        self,
        event_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit logs by event type.

        Args:
            event_type: Type of event to filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results

        Returns:
            List of AuditLog instances
        """
        with self.db.get_cursor() as cur:
            query = "SELECT * FROM audit_log WHERE event_type = %s"
            params = [event_type]

            if start_date:
                query += " AND created_at >= %s"
                params.append(start_date)

            if end_date:
                query += " AND created_at <= %s"
                params.append(end_date)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            return [AuditLog(**row) for row in rows]

    def get_by_anonymized_id(
        self,
        anonymized_id: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit logs for an anonymized ID.

        Args:
            anonymized_id: Anonymized identifier
            limit: Maximum number of results

        Returns:
            List of AuditLog instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM audit_log
                WHERE anonymized_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (anonymized_id, limit)
            )
            rows = cur.fetchall()
            return [AuditLog(**row) for row in rows]

    def get_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, int]:
        """
        Get summary of audit events by type for a date range.

        Args:
            start_date: Start date for summary
            end_date: End date for summary

        Returns:
            Dictionary mapping event types to counts
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM audit_log
                WHERE created_at >= %s AND created_at <= %s
                GROUP BY event_type
                ORDER BY count DESC
                """,
                (start_date, end_date)
            )
            rows = cur.fetchall()
            return {row["event_type"]: row["count"] for row in rows}


class ConsentRepository(BaseRepository):
    """Repository for consent table."""

    def create(self, consent: Consent) -> str:
        """
        Create a new consent record.

        Args:
            consent: Consent model instance

        Returns:
            Anonymized ID of created consent
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO consent (
                    anonymized_id, data_types, granted_at, expires_at, metadata
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (anonymized_id) DO UPDATE
                SET data_types = EXCLUDED.data_types,
                    granted_at = EXCLUDED.granted_at,
                    expires_at = EXCLUDED.expires_at,
                    metadata = EXCLUDED.metadata,
                    revoked_at = NULL
                RETURNING anonymized_id
                """,
                (
                    consent.anonymized_id,
                    consent.data_types,
                    consent.granted_at or datetime.utcnow(),
                    consent.expires_at,
                    consent.metadata,
                )
            )
            result = cur.fetchone()
            logger.info(f"Created/updated consent for {consent.anonymized_id}")
            return result["anonymized_id"]

    def get_by_anonymized_id(self, anonymized_id: str) -> Optional[Consent]:
        """
        Get consent record by anonymized ID.

        Args:
            anonymized_id: Anonymized identifier

        Returns:
            Consent instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM consent WHERE anonymized_id = %s",
                (anonymized_id,)
            )
            row = cur.fetchone()
            return Consent(**row) if row else None

    def revoke(self, anonymized_id: str) -> bool:
        """
        Revoke consent for an anonymized ID.

        Args:
            anonymized_id: Anonymized identifier

        Returns:
            True if consent was revoked, False if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE consent
                SET revoked_at = %s
                WHERE anonymized_id = %s AND revoked_at IS NULL
                RETURNING anonymized_id
                """,
                (datetime.utcnow(), anonymized_id)
            )
            result = cur.fetchone()
            if result:
                logger.info(f"Revoked consent for {anonymized_id}")
                return True
            return False

    def is_valid(
        self,
        anonymized_id: str,
        data_types: Optional[List[str]] = None
    ) -> bool:
        """
        Check if consent is valid for given data types.

        Args:
            anonymized_id: Anonymized identifier
            data_types: Optional list of data types to check

        Returns:
            True if consent is valid, False otherwise
        """
        consent = self.get_by_anonymized_id(anonymized_id)
        if not consent:
            return False

        # Check if revoked
        if consent.revoked_at:
            return False

        # Check if expired
        if consent.expires_at and consent.expires_at < datetime.utcnow():
            return False

        # Check data types if specified
        if data_types:
            for dt in data_types:
                if dt not in consent.data_types:
                    return False

        return True


class HumanReviewQueueRepository(BaseRepository):
    """Repository for human_review_queue table."""

    def create(self, case: HumanReviewCase) -> UUID:
        """
        Create a new review case.

        Args:
            case: HumanReviewCase model instance

        Returns:
            UUID of created case
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO human_review_queue (
                    anonymized_id, risk_score, prediction_id, assigned_to,
                    status, priority
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING case_id
                """,
                (
                    case.anonymized_id,
                    case.risk_score,
                    str(case.prediction_id) if case.prediction_id else None,
                    case.assigned_to,
                    case.status,
                    case.priority,
                )
            )
            result = cur.fetchone()
            case_id = result["case_id"]
            logger.info(f"Created review case {case_id} for {case.anonymized_id}")
            return case_id

    def get_by_case_id(self, case_id: UUID) -> Optional[HumanReviewCase]:
        """
        Get review case by ID.

        Args:
            case_id: Case UUID

        Returns:
            HumanReviewCase instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM human_review_queue WHERE case_id = %s",
                (str(case_id),)
            )
            row = cur.fetchone()
            return HumanReviewCase(**row) if row else None

    def get_pending_cases(
        self,
        assigned_to: Optional[str] = None,
        limit: int = 50
    ) -> List[HumanReviewCase]:
        """
        Get pending review cases.

        Args:
            assigned_to: Optional filter by assigned reviewer
            limit: Maximum number of results

        Returns:
            List of HumanReviewCase instances ordered by priority
        """
        with self.db.get_cursor() as cur:
            if assigned_to:
                cur.execute(
                    """
                    SELECT * FROM human_review_queue
                    WHERE status IN ('PENDING', 'IN_REVIEW')
                    AND assigned_to = %s
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                    """,
                    (assigned_to, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM human_review_queue
                    WHERE status IN ('PENDING', 'IN_REVIEW')
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                    """,
                    (limit,)
                )
            rows = cur.fetchall()
            return [HumanReviewCase(**row) for row in rows]

    def assign_case(self, case_id: UUID, reviewer_id: str) -> bool:
        """
        Assign a case to a reviewer.

        Args:
            case_id: Case UUID
            reviewer_id: Reviewer identifier

        Returns:
            True if assignment successful, False otherwise
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE human_review_queue
                SET assigned_to = %s, status = 'IN_REVIEW'
                WHERE case_id = %s AND status = 'PENDING'
                RETURNING case_id
                """,
                (reviewer_id, str(case_id))
            )
            result = cur.fetchone()
            if result:
                logger.info(f"Assigned case {case_id} to {reviewer_id}")
                return True
            return False

    def submit_review(
        self,
        case_id: UUID,
        decision: str,
        decision_notes: Optional[str] = None
    ) -> bool:
        """
        Submit review decision for a case.

        Args:
            case_id: Case UUID
            decision: Review decision (CONFIRMED, MODIFIED, OVERRIDDEN)
            decision_notes: Optional notes about the decision

        Returns:
            True if submission successful, False otherwise
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE human_review_queue
                SET status = 'COMPLETED',
                    decision = %s,
                    decision_notes = %s,
                    reviewed_at = %s
                WHERE case_id = %s AND status = 'IN_REVIEW'
                RETURNING case_id
                """,
                (decision, decision_notes, datetime.utcnow(), str(case_id))
            )
            result = cur.fetchone()
            if result:
                logger.info(f"Submitted review for case {case_id}: {decision}")
                return True
            return False

    def escalate_case(self, case_id: UUID) -> bool:
        """
        Escalate a case for supervisor review.

        Args:
            case_id: Case UUID

        Returns:
            True if escalation successful, False otherwise
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE human_review_queue
                SET status = 'ESCALATED',
                    escalated_at = %s,
                    priority = 'CRITICAL'
                WHERE case_id = %s
                RETURNING case_id
                """,
                (datetime.utcnow(), str(case_id))
            )
            result = cur.fetchone()
            if result:
                logger.info(f"Escalated case {case_id}")
                return True
            return False

    def get_overdue_cases(self, hours: int = 4) -> List[HumanReviewCase]:
        """
        Get cases that have not been reviewed within specified hours.

        Args:
            hours: Number of hours to consider overdue

        Returns:
            List of overdue HumanReviewCase instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM human_review_queue
                WHERE status IN ('PENDING', 'IN_REVIEW')
                AND created_at < NOW() - INTERVAL '%s hours'
                ORDER BY created_at ASC
                """,
                (hours,)
            )
            rows = cur.fetchall()
            return [HumanReviewCase(**row) for row in rows]



class ResourceRepository(BaseRepository):
    """Repository for resources table."""

    def create(self, resource_data: Dict[str, Any]) -> str:
        """
        Create a new resource record.

        Args:
            resource_data: Dictionary with resource fields

        Returns:
            ID of created resource
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO resources (
                    id, resource_type, name, description, contact_info,
                    urgency, eligibility_criteria, risk_levels, tags, priority
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    resource_data['id'],
                    resource_data['resource_type'],
                    resource_data['name'],
                    resource_data['description'],
                    resource_data['contact_info'],
                    resource_data['urgency'],
                    resource_data['eligibility_criteria'],
                    resource_data['risk_levels'],
                    resource_data.get('tags', []),
                    resource_data.get('priority', 0),
                )
            )
            result = cur.fetchone()
            resource_id = result["id"]
            logger.info(f"Created resource {resource_id}")
            return resource_id

    def get_by_id(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get resource by ID.

        Args:
            resource_id: Resource ID

        Returns:
            Resource dictionary or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM resources WHERE id = %s AND active = TRUE",
                (resource_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_by_risk_level(self, risk_level: str) -> List[Dict[str, Any]]:
        """
        Get resources for a specific risk level.

        Args:
            risk_level: Risk level (low, moderate, high, critical)

        Returns:
            List of resource dictionaries
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM resources
                WHERE active = TRUE
                AND risk_levels @> %s::jsonb
                ORDER BY priority DESC, name ASC
                """,
                (f'["{risk_level}"]',)
            )
            return [dict(row) for row in cur.fetchall()]

    def get_by_type(self, resource_type: str) -> List[Dict[str, Any]]:
        """
        Get resources by type.

        Args:
            resource_type: Resource type

        Returns:
            List of resource dictionaries
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM resources
                WHERE active = TRUE
                AND resource_type = %s
                ORDER BY priority DESC, name ASC
                """,
                (resource_type,)
            )
            return [dict(row) for row in cur.fetchall()]

    def search(
        self,
        risk_level: Optional[str] = None,
        resource_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        urgency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search resources with multiple filters.

        Args:
            risk_level: Filter by risk level
            resource_type: Filter by resource type
            tags: Filter by tags (any match)
            urgency: Filter by urgency level

        Returns:
            List of matching resource dictionaries
        """
        conditions = ["active = TRUE"]
        params = []

        if risk_level:
            conditions.append("risk_levels @> %s::jsonb")
            params.append(f'["{risk_level}"]')

        if resource_type:
            conditions.append("resource_type = %s")
            params.append(resource_type)

        if tags:
            # Match any of the provided tags
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags @> %s::jsonb")
                params.append(f'["{tag}"]')
            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        if urgency:
            conditions.append("urgency = %s")
            params.append(urgency)

        where_clause = " AND ".join(conditions)

        with self.db.get_cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM resources
                WHERE {where_clause}
                ORDER BY priority DESC, name ASC
                """,
                tuple(params)
            )
            return [dict(row) for row in cur.fetchall()]

    def get_all_active(self) -> List[Dict[str, Any]]:
        """
        Get all active resources.

        Returns:
            List of all active resource dictionaries
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM resources
                WHERE active = TRUE
                ORDER BY priority DESC, name ASC
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def update(self, resource_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a resource.

        Args:
            resource_id: Resource ID
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found
        """
        if not updates:
            return False

        set_clauses = []
        params = []

        for key, value in updates.items():
            if key != 'id':  # Don't allow ID updates
                set_clauses.append(f"{key} = %s")
                params.append(value)

        if not set_clauses:
            return False

        params.append(resource_id)

        with self.db.get_cursor() as cur:
            cur.execute(
                f"""
                UPDATE resources
                SET {', '.join(set_clauses)}
                WHERE id = %s
                RETURNING id
                """,
                tuple(params)
            )
            result = cur.fetchone()
            if result:
                logger.info(f"Updated resource {resource_id}")
                return True
            return False

    def deactivate(self, resource_id: str) -> bool:
        """
        Deactivate a resource (soft delete).

        Args:
            resource_id: Resource ID

        Returns:
            True if deactivated, False if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE resources
                SET active = FALSE
                WHERE id = %s
                RETURNING id
                """,
                (resource_id,)
            )
            result = cur.fetchone()
            if result:
                logger.info(f"Deactivated resource {resource_id}")
                return True
            return False


class ResourceRecommendationRepository(BaseRepository):
    """Repository for resource_recommendations table."""

    def create(
        self,
        anonymized_id: str,
        resource_id: str,
        prediction_id: Optional[UUID] = None,
        relevance_score: Optional[float] = None
    ) -> UUID:
        """
        Create a resource recommendation record.

        Args:
            anonymized_id: Anonymized identifier
            resource_id: Resource ID
            prediction_id: Optional prediction ID
            relevance_score: Optional relevance score

        Returns:
            UUID of created recommendation
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO resource_recommendations (
                    anonymized_id, resource_id, prediction_id, relevance_score
                )
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (anonymized_id, resource_id, str(prediction_id) if prediction_id else None, relevance_score)
            )
            result = cur.fetchone()
            rec_id = result["id"]
            logger.info(f"Created resource recommendation {rec_id}")
            return rec_id

    def get_by_anonymized_id(
        self,
        anonymized_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get resource recommendations for an individual.

        Args:
            anonymized_id: Anonymized identifier
            limit: Maximum number of results

        Returns:
            List of recommendation dictionaries with resource details
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT 
                    rr.*,
                    r.resource_type,
                    r.name,
                    r.description,
                    r.contact_info,
                    r.urgency,
                    r.eligibility_criteria
                FROM resource_recommendations rr
                JOIN resources r ON rr.resource_id = r.id
                WHERE rr.anonymized_id = %s
                ORDER BY rr.recommended_at DESC
                LIMIT %s
                """,
                (anonymized_id, limit)
            )
            return [dict(row) for row in cur.fetchall()]

    def mark_accessed(self, recommendation_id: UUID) -> bool:
        """
        Mark a recommendation as accessed.

        Args:
            recommendation_id: Recommendation UUID

        Returns:
            True if updated, False if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE resource_recommendations
                SET accessed = TRUE, accessed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id
                """,
                (str(recommendation_id),)
            )
            result = cur.fetchone()
            return result is not None

    def get_recommendation_stats(
        self,
        resource_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a resource's recommendations.

        Args:
            resource_id: Resource ID
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary with recommendation statistics
        """
        conditions = ["resource_id = %s"]
        params = [resource_id]

        if start_date:
            conditions.append("recommended_at >= %s")
            params.append(start_date)

        if end_date:
            conditions.append("recommended_at <= %s")
            params.append(end_date)

        where_clause = " AND ".join(conditions)

        with self.db.get_cursor() as cur:
            cur.execute(
                f"""
                SELECT 
                    COUNT(*) as total_recommendations,
                    COUNT(*) FILTER (WHERE accessed = TRUE) as accessed_count,
                    AVG(relevance_score) as avg_relevance_score
                FROM resource_recommendations
                WHERE {where_clause}
                """,
                tuple(params)
            )
            row = cur.fetchone()
            return dict(row) if row else {}
