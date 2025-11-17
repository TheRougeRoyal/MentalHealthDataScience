"""Tests for experiment tracking system."""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from pathlib import Path
import tempfile
import shutil

from src.ds.models import Experiment, Run, Metric, Artifact
from src.ds.repositories import ExperimentRepository
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.storage import FileSystemStorage
from src.database.connection import DatabaseConnection


class TestExperimentModels:
    """Test experiment tracking models."""

    def test_experiment_model_valid(self):
        """Test valid experiment model creation."""
        experiment = Experiment(
            experiment_name="test_experiment",
            description="Test description",
            tags={"project": "test"}
        )
        assert experiment.experiment_name == "test_experiment"
        assert experiment.description == "Test description"
        assert experiment.tags == {"project": "test"}

    def test_run_model_valid(self):
        """Test valid run model creation."""
        run = Run(
            experiment_id=uuid4(),
            run_name="test_run",
            status="RUNNING",
            start_time=datetime.utcnow(),
            params={"learning_rate": 0.01},
            tags={"env": "test"}
        )
        assert run.run_name == "test_run"
        assert run.status == "RUNNING"
        assert run.params == {"learning_rate": 0.01}

    def test_run_model_invalid_status(self):
        """Test run model with invalid status."""
        with pytest.raises(ValueError):
            Run(
                experiment_id=uuid4(),
                status="INVALID_STATUS",
                start_time=datetime.utcnow()
            )

    def test_metric_model_valid(self):
        """Test valid metric model creation."""
        metric = Metric(
            run_id=uuid4(),
            metric_name="accuracy",
            metric_value=0.95,
            step=10
        )
        assert metric.metric_name == "accuracy"
        assert metric.metric_value == 0.95
        assert metric.step == 10

    def test_artifact_model_valid(self):
        """Test valid artifact model creation."""
        artifact = Artifact(
            run_id=uuid4(),
            artifact_type="model",
            artifact_path="/path/to/model.pkl",
            size_bytes=1024,
            metadata={"format": "pickle"}
        )
        assert artifact.artifact_type == "model"
        assert artifact.artifact_path == "/path/to/model.pkl"
        assert artifact.size_bytes == 1024


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage_backend(temp_storage_dir):
    """Create storage backend for testing."""
    return FileSystemStorage(base_path=temp_storage_dir)


@pytest.fixture
def mock_db_connection(mocker):
    """Create mock database connection."""
    mock_conn = mocker.Mock(spec=DatabaseConnection)
    mock_cursor = mocker.MagicMock()
    mock_conn.get_cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor


@pytest.fixture
def experiment_tracker(storage_backend, mock_db_connection):
    """Create experiment tracker for testing."""
    db_conn, _ = mock_db_connection
    return ExperimentTracker(storage_backend, db_conn)


class TestExperimentTracker:
    """Test ExperimentTracker functionality."""

    def test_start_run_creates_experiment(self, experiment_tracker, mock_db_connection):
        """Test starting a run creates experiment if needed."""
        _, mock_cursor = mock_db_connection
        
        # Mock experiment doesn't exist
        mock_cursor.fetchone.side_effect = [
            None,  # get_experiment_by_name returns None
            {"experiment_id": uuid4()},  # create_experiment returns ID
            {"experiment_id": uuid4()},  # get_experiment_by_id returns experiment
            {"run_id": uuid4()}  # create_run returns ID
        ]
        
        run = experiment_tracker.start_run("test_experiment")
        
        assert run is not None
        assert run.status == "RUNNING"
        assert experiment_tracker.active_run is not None

    def test_start_run_with_existing_experiment(self, experiment_tracker, mock_db_connection):
        """Test starting a run with existing experiment."""
        _, mock_cursor = mock_db_connection
        
        experiment_id = uuid4()
        mock_cursor.fetchone.side_effect = [
            {"experiment_id": experiment_id, "experiment_name": "test", "description": None, "created_at": datetime.utcnow(), "tags": None},
            {"run_id": uuid4()}
        ]
        
        run = experiment_tracker.start_run("test_experiment", run_name="run_1")
        
        assert run.run_name == "run_1"
        assert run.experiment_id == experiment_id

    def test_start_run_fails_if_run_active(self, experiment_tracker, mock_db_connection):
        """Test starting a run fails if another run is active."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.side_effect = [
            None,
            {"experiment_id": uuid4()},
            {"experiment_id": uuid4()},
            {"run_id": uuid4()}
        ]
        
        experiment_tracker.start_run("test_experiment")
        
        with pytest.raises(RuntimeError, match="already active"):
            experiment_tracker.start_run("test_experiment")

    def test_log_params(self, experiment_tracker, mock_db_connection):
        """Test logging parameters."""
        _, mock_cursor = mock_db_connection
        
        # Start a run first
        mock_cursor.fetchone.side_effect = [
            None,
            {"experiment_id": uuid4()},
            {"experiment_id": uuid4()},
            {"run_id": uuid4()},
            {"run_id": uuid4()}  # update_run
        ]
        
        experiment_tracker.start_run("test_experiment")
        experiment_tracker.log_params({"learning_rate": 0.01, "batch_size": 32})
        
        assert experiment_tracker.active_run.params["learning_rate"] == 0.01
        assert experiment_tracker.active_run.params["batch_size"] == 32

    def test_log_params_fails_without_active_run(self, experiment_tracker):
        """Test logging params fails without active run."""
        with pytest.raises(RuntimeError, match="No active run"):
            experiment_tracker.log_params({"learning_rate": 0.01})

    def test_log_metrics(self, experiment_tracker, mock_db_connection):
        """Test logging metrics."""
        _, mock_cursor = mock_db_connection
        
        # Start a run first
        mock_cursor.fetchone.side_effect = [
            None,
            {"experiment_id": uuid4()},
            {"experiment_id": uuid4()},
            {"run_id": uuid4()},
            {"metric_id": uuid4()},  # log_metric for accuracy
            {"metric_id": uuid4()}   # log_metric for loss
        ]
        
        experiment_tracker.start_run("test_experiment")
        experiment_tracker.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=1)
        
        # Verify metrics were logged
        assert mock_cursor.execute.call_count >= 2

    def test_log_metrics_fails_without_active_run(self, experiment_tracker):
        """Test logging metrics fails without active run."""
        with pytest.raises(RuntimeError, match="No active run"):
            experiment_tracker.log_metrics({"accuracy": 0.95})

    def test_log_artifact(self, experiment_tracker, mock_db_connection, temp_storage_dir):
        """Test logging artifacts."""
        _, mock_cursor = mock_db_connection
        
        # Create a test artifact file
        test_file = Path(temp_storage_dir) / "test_artifact.txt"
        test_file.write_text("test content")
        
        # Start a run first
        run_id = uuid4()
        mock_cursor.fetchone.side_effect = [
            None,
            {"experiment_id": uuid4()},
            {"experiment_id": uuid4()},
            {"run_id": run_id},
            {"artifact_id": uuid4()}
        ]
        
        experiment_tracker.start_run("test_experiment")
        artifact_uri = experiment_tracker.log_artifact(str(test_file), artifact_type="data")
        
        assert artifact_uri is not None
        assert "test_artifact.txt" in artifact_uri

    def test_log_artifact_fails_without_active_run(self, experiment_tracker):
        """Test logging artifact fails without active run."""
        with pytest.raises(RuntimeError, match="No active run"):
            experiment_tracker.log_artifact("/path/to/file.txt")

    def test_log_artifact_fails_with_nonexistent_file(self, experiment_tracker, mock_db_connection):
        """Test logging artifact fails with nonexistent file."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.side_effect = [
            None,
            {"experiment_id": uuid4()},
            {"experiment_id": uuid4()},
            {"run_id": uuid4()}
        ]
        
        experiment_tracker.start_run("test_experiment")
        
        with pytest.raises(FileNotFoundError):
            experiment_tracker.log_artifact("/nonexistent/file.txt")

    def test_end_run(self, experiment_tracker, mock_db_connection):
        """Test ending a run."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.side_effect = [
            None,
            {"experiment_id": uuid4()},
            {"experiment_id": uuid4()},
            {"run_id": uuid4()},
            {"run_id": uuid4()}  # update_run
        ]
        
        experiment_tracker.start_run("test_experiment")
        experiment_tracker.end_run(status="FINISHED")
        
        assert experiment_tracker.active_run is None
        assert experiment_tracker.active_run_id is None

    def test_end_run_fails_without_active_run(self, experiment_tracker):
        """Test ending run fails without active run."""
        with pytest.raises(RuntimeError, match="No active run"):
            experiment_tracker.end_run()

    def test_get_run(self, experiment_tracker, mock_db_connection):
        """Test retrieving a run."""
        _, mock_cursor = mock_db_connection
        
        run_id = uuid4()
        experiment_id = uuid4()
        mock_cursor.fetchone.return_value = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "run_name": "test_run",
            "status": "FINISHED",
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
            "params": {"lr": 0.01},
            "tags": None,
            "git_commit": None,
            "code_version": None,
            "created_at": datetime.utcnow()
        }
        
        run = experiment_tracker.get_run(run_id)
        
        assert run is not None
        assert run.run_id == run_id
        assert run.run_name == "test_run"

    def test_search_runs(self, experiment_tracker, mock_db_connection):
        """Test searching runs."""
        _, mock_cursor = mock_db_connection
        
        experiment_id = uuid4()
        mock_cursor.fetchone.return_value = {
            "experiment_id": experiment_id,
            "experiment_name": "test",
            "description": None,
            "created_at": datetime.utcnow(),
            "tags": None
        }
        
        mock_cursor.fetchall.return_value = [
            {
                "run_id": uuid4(),
                "experiment_id": experiment_id,
                "run_name": "run_1",
                "status": "FINISHED",
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow(),
                "params": {},
                "tags": None,
                "git_commit": None,
                "code_version": None,
                "created_at": datetime.utcnow()
            }
        ]
        
        runs = experiment_tracker.search_runs(experiment_name="test")
        
        assert len(runs) == 1
        assert runs[0].run_name == "run_1"

    def test_compare_runs(self, experiment_tracker, mock_db_connection):
        """Test comparing multiple runs."""
        _, mock_cursor = mock_db_connection
        
        run_id_1 = uuid4()
        run_id_2 = uuid4()
        experiment_id = uuid4()
        
        # Mock get_run_by_id calls
        mock_cursor.fetchone.side_effect = [
            {
                "run_id": run_id_1,
                "experiment_id": experiment_id,
                "run_name": "run_1",
                "status": "FINISHED",
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow(),
                "params": {"lr": 0.01},
                "tags": None,
                "git_commit": None,
                "code_version": None,
                "created_at": datetime.utcnow()
            },
            {
                "run_id": run_id_2,
                "experiment_id": experiment_id,
                "run_name": "run_2",
                "status": "FINISHED",
                "start_time": datetime.utcnow(),
                "end_time": datetime.utcnow(),
                "params": {"lr": 0.001},
                "tags": None,
                "git_commit": None,
                "code_version": None,
                "created_at": datetime.utcnow()
            }
        ]
        
        # Mock get_metrics_for_run calls
        mock_cursor.fetchall.side_effect = [
            [
                {
                    "metric_id": uuid4(),
                    "run_id": run_id_1,
                    "metric_name": "accuracy",
                    "metric_value": 0.95,
                    "step": 10,
                    "timestamp": datetime.utcnow()
                }
            ],
            [
                {
                    "metric_id": uuid4(),
                    "run_id": run_id_2,
                    "metric_name": "accuracy",
                    "metric_value": 0.92,
                    "step": 10,
                    "timestamp": datetime.utcnow()
                }
            ]
        ]
        
        df = experiment_tracker.compare_runs([run_id_1, run_id_2])
        
        assert len(df) == 2
        assert "param_lr" in df.columns
        assert "metric_accuracy" in df.columns


class TestModelRegistryIntegration:
    """Test integration with model registry."""

    def test_register_model_with_run_id(self, mocker):
        """Test registering model with run ID."""
        from src.ml.model_registry import ModelRegistry
        
        # Create temporary registry
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(registry_dir=temp_dir)
            
            # Create a simple mock model
            mock_model = mocker.Mock()
            
            run_id = str(uuid4())
            experiment_id = str(uuid4())
            
            model_id = registry.register_model(
                model=mock_model,
                model_type="test_model",
                metadata={"accuracy": 0.95},
                run_id=run_id,
                experiment_id=experiment_id
            )
            
            # Verify experiment tracking fields are stored
            metadata = registry.get_model_metadata(model_id)
            assert metadata["run_id"] == run_id
            assert metadata["experiment_id"] == experiment_id

    def test_get_models_by_experiment(self, mocker):
        """Test retrieving models by experiment ID."""
        from src.ml.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(registry_dir=temp_dir)
            
            mock_model = mocker.Mock()
            experiment_id = str(uuid4())
            
            # Register multiple models for same experiment
            model_id_1 = registry.register_model(
                model=mock_model,
                model_type="test_model",
                metadata={"accuracy": 0.95},
                experiment_id=experiment_id
            )
            
            model_id_2 = registry.register_model(
                model=mock_model,
                model_type="test_model",
                metadata={"accuracy": 0.96},
                experiment_id=experiment_id
            )
            
            models = registry.get_models_by_experiment(experiment_id)
            
            assert len(models) == 2
            assert any(m["model_id"] == model_id_1 for m in models)
            assert any(m["model_id"] == model_id_2 for m in models)

    def test_get_models_by_run(self, mocker):
        """Test retrieving models by run ID."""
        from src.ml.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(registry_dir=temp_dir)
            
            mock_model = mocker.Mock()
            run_id = str(uuid4())
            
            model_id = registry.register_model(
                model=mock_model,
                model_type="test_model",
                metadata={"accuracy": 0.95},
                run_id=run_id
            )
            
            models = registry.get_models_by_run(run_id)
            
            assert len(models) == 1
            assert models[0]["model_id"] == model_id
            assert models[0]["run_id"] == run_id
