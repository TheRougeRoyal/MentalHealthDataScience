"""Experiment tracking system for ML experiments."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import UUID
import pandas as pd

from src.database.connection import DatabaseConnection
from src.ds.storage import StorageBackend
from src.ds.repositories import ExperimentRepository
from src.ds.models import Experiment, Run, Metric, Artifact

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Central experiment tracking system."""

    def __init__(
        self,
        storage_backend: StorageBackend,
        db_connection: DatabaseConnection
    ):
        """
        Initialize ExperimentTracker.

        Args:
            storage_backend: Storage backend for artifacts
            db_connection: Database connection instance
        """
        self.storage = storage_backend
        self.db = db_connection
        self.repository = ExperimentRepository(db_connection)
        self.active_run: Optional[Run] = None
        self.active_run_id: Optional[UUID] = None

        logger.info("ExperimentTracker initialized")

    def create_experiment(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            experiment_name: Name of the experiment
            description: Optional description
            tags: Optional tags

        Returns:
            Experiment instance
        """
        experiment = Experiment(
            experiment_name=experiment_name,
            description=description,
            tags=tags
        )
        experiment_id = self.repository.create_experiment(experiment)
        experiment.experiment_id = experiment_id

        logger.info(f"Created experiment '{experiment_name}' with ID {experiment_id}")

        return experiment

    def list_experiments(self, limit: int = 100) -> List[Experiment]:
        """
        List all experiments.

        Args:
            limit: Maximum number of experiments to return

        Returns:
            List of Experiment instances
        """
        return self.repository.list_experiments(limit=limit)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID (UUID string)

        Returns:
            Experiment instance or None if not found
        """
        try:
            exp_uuid = UUID(experiment_id)
            return self.repository.get_experiment_by_id(exp_uuid)
        except (ValueError, AttributeError):
            return None

    def start_run(
        self,
        experiment_name: str = None,
        experiment_id: str = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Run:
        """
        Start a new experiment run.

        Args:
            experiment_name: Name of the experiment (deprecated, use experiment_id)
            experiment_id: ID of the experiment
            run_name: Optional name for this run
            tags: Optional tags for the run

        Returns:
            Run instance

        Raises:
            RuntimeError: If a run is already active
        """
        if self.active_run is not None:
            raise RuntimeError(
                f"Run {self.active_run_id} is already active. "
                "End the current run before starting a new one."
            )

        # Get or create experiment
        if experiment_id:
            experiment = self.repository.get_experiment_by_id(UUID(experiment_id))
            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} not found")
        elif experiment_name:
            experiment = self.repository.get_experiment_by_name(experiment_name)
            if experiment is None:
                exp_id = self.repository.create_experiment(
                    Experiment(experiment_name=experiment_name)
                )
                experiment = self.repository.get_experiment_by_id(exp_id)
        else:
            raise ValueError("Either experiment_name or experiment_id must be provided")

        # Capture git commit and code version
        git_commit = self._get_git_commit()
        code_version = self._get_code_version()

        # Create run
        run = Run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
            status="RUNNING",
            start_time=datetime.utcnow(),
            tags=tags,
            git_commit=git_commit,
            code_version=code_version,
            params={}
        )

        run_id = self.repository.create_run(run)
        run.run_id = run_id

        self.active_run = run
        self.active_run_id = run_id

        logger.info(f"Started run {run_id} for experiment '{experiment_name}'")

        return run

    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None) -> None:
        """
        Log hyperparameters for a run.

        Args:
            params: Dictionary of parameter names to values
            run_id: Optional run ID (uses active run if not provided)

        Raises:
            RuntimeError: If no run is active and run_id not provided
        """
        if run_id:
            target_run_id = UUID(run_id)
        elif self.active_run is not None:
            target_run_id = self.active_run_id
        else:
            raise RuntimeError("No active run. Call start_run() first or provide run_id.")

        # Update run params
        self.repository.update_run(
            target_run_id,
            params=params
        )

        # Update local cache if this is the active run
        if self.active_run and target_run_id == self.active_run_id:
            if self.active_run.params is None:
                self.active_run.params = {}
            self.active_run.params.update(params)

        logger.debug(f"Logged {len(params)} parameters for run {target_run_id}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log metrics for a run.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (e.g., epoch, iteration)
            run_id: Optional run ID (uses active run if not provided)

        Raises:
            RuntimeError: If no run is active and run_id not provided
        """
        if run_id:
            target_run_id = UUID(run_id)
        elif self.active_run is not None:
            target_run_id = self.active_run_id
        else:
            raise RuntimeError("No active run. Call start_run() first or provide run_id.")

        # Log each metric
        for metric_name, metric_value in metrics.items():
            metric = Metric(
                run_id=target_run_id,
                metric_name=metric_name,
                metric_value=metric_value,
                step=step
            )
            self.repository.log_metric(metric)

        logger.debug(
            f"Logged {len(metrics)} metrics for run {target_run_id} "
            f"at step {step}"
        )

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "file",
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an artifact for a run.

        Args:
            artifact_path: Local path to the artifact
            artifact_type: Type of artifact (file, model, plot, data, report)
            run_id: Optional run ID (uses active run if not provided)
            metadata: Optional metadata for the artifact

        Returns:
            URI of stored artifact

        Raises:
            RuntimeError: If no run is active and run_id not provided
            FileNotFoundError: If artifact path doesn't exist
        """
        if run_id:
            target_run_id = UUID(run_id)
        elif self.active_run is not None:
            target_run_id = self.active_run_id
        else:
            raise RuntimeError("No active run. Call start_run() first or provide run_id.")

        artifact_file = Path(artifact_path)
        if not artifact_file.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Read artifact
        with open(artifact_file, 'rb') as f:
            artifact_data = f.read()

        # Generate storage path
        storage_path = f"runs/{target_run_id}/{artifact_file.name}"

        # Save to storage backend
        artifact_uri = self.storage.save_artifact(artifact_data, storage_path)

        # Merge metadata
        artifact_metadata = {"original_path": str(artifact_path)}
        if metadata:
            artifact_metadata.update(metadata)

        # Log to database
        artifact = Artifact(
            run_id=target_run_id,
            artifact_type=artifact_type,
            artifact_path=artifact_uri,
            size_bytes=len(artifact_data),
            metadata=artifact_metadata
        )
        artifact_id = self.repository.log_artifact(artifact)

        logger.info(
            f"Logged artifact {artifact_id} ({artifact_type}) "
            f"for run {target_run_id}"
        )

        return artifact_uri

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the active run.

        Args:
            status: Final status (FINISHED or FAILED)

        Raises:
            RuntimeError: If no run is active
        """
        if self.active_run is None:
            raise RuntimeError("No active run to end.")

        # Update run status
        self.repository.update_run(
            self.active_run_id,
            status=status,
            end_time=datetime.utcnow()
        )

        logger.info(f"Ended run {self.active_run_id} with status {status}")

        self.active_run = None
        self.active_run_id = None

    def get_run(self, run_id: UUID) -> Optional[Run]:
        """
        Retrieve a specific run.

        Args:
            run_id: Run UUID

        Returns:
            Run instance or None if not found
        """
        return self.repository.get_run_by_id(run_id)

    def search_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Run]:
        """
        Search and filter runs.

        Args:
            experiment_name: Optional experiment name filter
            filter_string: Optional filter string (not implemented yet)
            order_by: Optional ordering (not implemented yet)
            limit: Maximum number of results

        Returns:
            List of Run instances
        """
        experiment_id = None
        if experiment_name:
            experiment = self.repository.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id

        runs = self.repository.search_runs(experiment_id=experiment_id, limit=limit)

        logger.info(f"Found {len(runs)} runs")

        return runs

    def compare_runs(
        self,
        run_ids: List[str],
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run ID strings to compare
            metric_names: Optional list of metric names to include

        Returns:
            DataFrame with run comparison
        """
        comparison_data = []

        for run_id_str in run_ids:
            run_id = UUID(run_id_str)
            run = self.repository.get_run_by_id(run_id)
            if run is None:
                logger.warning(f"Run {run_id} not found, skipping")
                continue

            # Get metrics for this run
            metrics = self.repository.get_metrics_for_run(run_id)

            # Build row data
            row_data = {
                'run_id': str(run_id),
                'run_name': run.run_name,
                'status': run.status,
                'start_time': run.start_time,
                'end_time': run.end_time,
            }

            # Add parameters
            if run.params:
                for param_name, param_value in run.params.items():
                    row_data[f'param_{param_name}'] = param_value

            # Add metrics (latest value for each metric)
            metric_dict = {}
            for metric in metrics:
                if metric_names is None or metric.metric_name in metric_names:
                    # Keep latest value
                    if metric.metric_name not in metric_dict:
                        metric_dict[metric.metric_name] = metric.metric_value
                    else:
                        # Update if this is a later step
                        if metric.step is not None:
                            metric_dict[metric.metric_name] = metric.metric_value

            for metric_name, metric_value in metric_dict.items():
                row_data[f'metric_{metric_name}'] = metric_value

            comparison_data.append(row_data)

        df = pd.DataFrame(comparison_data)

        logger.info(f"Compared {len(comparison_data)} runs")

        return df

    def _get_git_commit(self) -> Optional[str]:
        """
        Get current git commit hash.

        Returns:
            Git commit hash or None if not in a git repo
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git commit: {e}")

        return None

    def _get_code_version(self) -> Optional[str]:
        """
        Get code version (e.g., from package version).

        Returns:
            Code version string or None
        """
        try:
            # Try to get version from git tag
            result = subprocess.run(
                ['git', 'describe', '--tags', '--always'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get code version: {e}")

        return None
