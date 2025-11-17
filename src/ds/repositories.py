"""Repository classes for data science database access."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from src.database.connection import DatabaseConnection
from src.ds.models import Experiment, Run, Metric, Artifact

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


class ExperimentRepository(BaseRepository):
    """Repository for experiment tracking tables."""

    def create_experiment(self, experiment: Experiment) -> UUID:
        """
        Create a new experiment.

        Args:
            experiment: Experiment model instance

        Returns:
            UUID of created experiment
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiments (
                    experiment_name, description, tags
                )
                VALUES (%s, %s, %s)
                RETURNING experiment_id
                """,
                (
                    experiment.experiment_name,
                    experiment.description,
                    experiment.tags,
                )
            )
            result = cur.fetchone()
            experiment_id = result["experiment_id"]
            logger.info(f"Created experiment {experiment_id}: {experiment.experiment_name}")
            return experiment_id

    def get_experiment_by_id(self, experiment_id: UUID) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment UUID

        Returns:
            Experiment instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM experiments WHERE experiment_id = %s",
                (str(experiment_id),)
            )
            row = cur.fetchone()
            return Experiment(**row) if row else None

    def get_experiment_by_name(self, experiment_name: str) -> Optional[Experiment]:
        """
        Get experiment by name.

        Args:
            experiment_name: Experiment name

        Returns:
            Experiment instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM experiments WHERE experiment_name = %s",
                (experiment_name,)
            )
            row = cur.fetchone()
            return Experiment(**row) if row else None

    def list_experiments(self, limit: int = 100) -> List[Experiment]:
        """
        List all experiments.

        Args:
            limit: Maximum number of results

        Returns:
            List of Experiment instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM experiments
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
            return [Experiment(**row) for row in rows]

    def create_run(self, run: Run) -> UUID:
        """
        Create a new run.

        Args:
            run: Run model instance

        Returns:
            UUID of created run
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    experiment_id, run_name, status, start_time,
                    end_time, params, tags, git_commit, code_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    str(run.experiment_id),
                    run.run_name,
                    run.status,
                    run.start_time,
                    run.end_time,
                    run.params,
                    run.tags,
                    run.git_commit,
                    run.code_version,
                )
            )
            result = cur.fetchone()
            run_id = result["run_id"]
            logger.info(f"Created run {run_id} for experiment {run.experiment_id}")
            return run_id

    def get_run_by_id(self, run_id: UUID) -> Optional[Run]:
        """
        Get run by ID.

        Args:
            run_id: Run UUID

        Returns:
            Run instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM runs WHERE run_id = %s",
                (str(run_id),)
            )
            row = cur.fetchone()
            return Run(**row) if row else None

    def update_run(
        self,
        run_id: UUID,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update run fields.

        Args:
            run_id: Run UUID
            status: Optional new status
            end_time: Optional end time
            params: Optional parameters to merge

        Returns:
            True if updated, False if not found
        """
        updates = []
        values = []

        if status is not None:
            updates.append("status = %s")
            values.append(status)

        if end_time is not None:
            updates.append("end_time = %s")
            values.append(end_time)

        if params is not None:
            updates.append("params = params || %s::jsonb")
            values.append(params)

        if not updates:
            return False

        values.append(str(run_id))

        with self.db.get_cursor() as cur:
            cur.execute(
                f"""
                UPDATE runs
                SET {', '.join(updates)}
                WHERE run_id = %s
                RETURNING run_id
                """,
                tuple(values)
            )
            result = cur.fetchone()
            return result is not None

    def search_runs(
        self,
        experiment_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Run]:
        """
        Search runs with filters.

        Args:
            experiment_id: Optional experiment ID filter
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of Run instances
        """
        conditions = []
        params = []

        if experiment_id is not None:
            conditions.append("experiment_id = %s")
            params.append(str(experiment_id))

        if status is not None:
            conditions.append("status = %s")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        with self.db.get_cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM runs
                WHERE {where_clause}
                ORDER BY start_time DESC
                LIMIT %s
                """,
                tuple(params)
            )
            rows = cur.fetchall()
            return [Run(**row) for row in rows]

    def log_metric(self, metric: Metric) -> UUID:
        """
        Log a metric for a run.

        Args:
            metric: Metric model instance

        Returns:
            UUID of created metric
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO metrics (
                    run_id, metric_name, metric_value, step
                )
                VALUES (%s, %s, %s, %s)
                RETURNING metric_id
                """,
                (
                    str(metric.run_id),
                    metric.metric_name,
                    metric.metric_value,
                    metric.step,
                )
            )
            result = cur.fetchone()
            metric_id = result["metric_id"]
            logger.debug(f"Logged metric {metric.metric_name}={metric.metric_value} for run {metric.run_id}")
            return metric_id

    def get_metrics_for_run(
        self,
        run_id: UUID,
        metric_name: Optional[str] = None
    ) -> List[Metric]:
        """
        Get metrics for a run.

        Args:
            run_id: Run UUID
            metric_name: Optional metric name filter

        Returns:
            List of Metric instances
        """
        with self.db.get_cursor() as cur:
            if metric_name:
                cur.execute(
                    """
                    SELECT * FROM metrics
                    WHERE run_id = %s AND metric_name = %s
                    ORDER BY step ASC, timestamp ASC
                    """,
                    (str(run_id), metric_name)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM metrics
                    WHERE run_id = %s
                    ORDER BY metric_name ASC, step ASC, timestamp ASC
                    """,
                    (str(run_id),)
                )
            rows = cur.fetchall()
            return [Metric(**row) for row in rows]

    def log_artifact(self, artifact: Artifact) -> UUID:
        """
        Log an artifact for a run.

        Args:
            artifact: Artifact model instance

        Returns:
            UUID of created artifact
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts (
                    run_id, artifact_type, artifact_path, size_bytes, metadata
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING artifact_id
                """,
                (
                    str(artifact.run_id),
                    artifact.artifact_type,
                    artifact.artifact_path,
                    artifact.size_bytes,
                    artifact.metadata,
                )
            )
            result = cur.fetchone()
            artifact_id = result["artifact_id"]
            logger.info(f"Logged artifact {artifact_id} for run {artifact.run_id}")
            return artifact_id

    def get_artifacts_for_run(
        self,
        run_id: UUID,
        artifact_type: Optional[str] = None
    ) -> List[Artifact]:
        """
        Get artifacts for a run.

        Args:
            run_id: Run UUID
            artifact_type: Optional artifact type filter

        Returns:
            List of Artifact instances
        """
        with self.db.get_cursor() as cur:
            if artifact_type:
                cur.execute(
                    """
                    SELECT * FROM artifacts
                    WHERE run_id = %s AND artifact_type = %s
                    ORDER BY created_at DESC
                    """,
                    (str(run_id), artifact_type)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM artifacts
                    WHERE run_id = %s
                    ORDER BY created_at DESC
                    """,
                    (str(run_id),)
                )
            rows = cur.fetchall()
            return [Artifact(**row) for row in rows]

    def get_artifact_by_id(self, artifact_id: UUID) -> Optional[Artifact]:
        """
        Get artifact by ID.

        Args:
            artifact_id: Artifact UUID

        Returns:
            Artifact instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM artifacts WHERE artifact_id = %s",
                (str(artifact_id),)
            )
            row = cur.fetchone()
            return Artifact(**row) if row else None
