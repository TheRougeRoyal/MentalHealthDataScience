"""Tests for hyperparameter optimization system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4

from src.ds.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    OptimizationResult
)
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.storage import FileSystemStorage
from src.database.connection import DatabaseConnection


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


class TestOptimizationResult:
    """Test OptimizationResult model."""

    def test_optimization_result_valid(self):
        """Test valid optimization result creation."""
        result = OptimizationResult(
            best_params={"learning_rate": 0.01},
            best_score=0.95,
            all_trials=[{"number": 0, "params": {"learning_rate": 0.01}, "value": 0.95, "state": "COMPLETE"}],
            optimization_history=[0.95],
            n_trials=1,
            optimization_time=10.5,
            strategy="bayesian"
        )
        
        assert result.best_params == {"learning_rate": 0.01}
        assert result.best_score == 0.95
        assert result.n_trials == 1
        assert result.strategy == "bayesian"


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer functionality."""

    def test_optimizer_initialization_bayesian(self, experiment_tracker):
        """Test optimizer initialization with Bayesian strategy."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="bayesian"
        )
        
        assert optimizer.strategy == "bayesian"
        assert optimizer.tracker == experiment_tracker

    def test_optimizer_initialization_random(self, experiment_tracker):
        """Test optimizer initialization with random strategy."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        assert optimizer.strategy == "random"

    def test_optimizer_initialization_grid(self, experiment_tracker):
        """Test optimizer initialization with grid strategy."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="grid"
        )
        
        assert optimizer.strategy == "grid"

    def test_optimizer_initialization_invalid_strategy(self, experiment_tracker):
        """Test optimizer initialization with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            HyperparameterOptimizer(
                experiment_tracker=experiment_tracker,
                strategy="invalid"
            )

    def test_optimize_random_search(self, experiment_tracker, mock_db_connection):
        """Test random search optimization."""
        _, mock_cursor = mock_db_connection
        
        # Mock database responses for multiple runs
        mock_responses = []
        for i in range(5):
            mock_responses.extend([
                None,  # get_experiment_by_name
                {"experiment_id": uuid4()},  # create_experiment
                {"experiment_id": uuid4()},  # get_experiment_by_id
                {"run_id": uuid4()},  # create_run
                {"run_id": uuid4()},  # update_run (log_params)
                {"metric_id": uuid4()},  # log_metric
                {"run_id": uuid4()}  # update_run (end_run)
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        # Simple objective function
        def objective(params):
            return params["x"] ** 2
        
        param_space = {
            "x": {"type": "float", "low": -10.0, "high": 10.0}
        }
        
        result = optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=5,
            direction="minimize"
        )
        
        assert result is not None
        assert result.n_trials == 5
        assert result.strategy == "random"
        assert result.best_params is not None
        assert result.best_score is not None
        assert len(result.all_trials) == 5
        assert len(result.optimization_history) == 5

    def test_optimize_grid_search(self, experiment_tracker, mock_db_connection):
        """Test grid search optimization."""
        _, mock_cursor = mock_db_connection
        
        # Mock database responses for 4 runs (2x2 grid)
        mock_responses = []
        for i in range(4):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="grid"
        )
        
        def objective(params):
            return params["x"] + params["y"]
        
        param_space = {
            "x": {"type": "float", "values": [1.0, 2.0]},
            "y": {"type": "float", "values": [3.0, 4.0]}
        }
        
        result = optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            direction="maximize"
        )
        
        assert result.n_trials == 4
        assert result.strategy == "grid"
        assert result.best_score == 6.0  # max(1+3, 1+4, 2+3, 2+4) = 6

    def test_optimize_with_categorical_params(self, experiment_tracker, mock_db_connection):
        """Test optimization with categorical parameters."""
        _, mock_cursor = mock_db_connection
        
        mock_responses = []
        for i in range(3):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        def objective(params):
            multiplier = {"a": 1, "b": 2, "c": 3}[params["choice"]]
            return params["x"] * multiplier
        
        param_space = {
            "x": {"type": "float", "low": 1.0, "high": 10.0},
            "choice": {"type": "categorical", "choices": ["a", "b", "c"]}
        }
        
        result = optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=3,
            direction="maximize"
        )
        
        assert result.n_trials == 3
        assert "choice" in result.best_params
        assert result.best_params["choice"] in ["a", "b", "c"]

    def test_optimize_with_int_params(self, experiment_tracker, mock_db_connection):
        """Test optimization with integer parameters."""
        _, mock_cursor = mock_db_connection
        
        mock_responses = []
        for i in range(3):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        def objective(params):
            return params["n"] * 2
        
        param_space = {
            "n": {"type": "int", "low": 1, "high": 10}
        }
        
        result = optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=3,
            direction="maximize"
        )
        
        assert result.n_trials == 3
        assert isinstance(result.best_params["n"], int)
        assert 1 <= result.best_params["n"] <= 10

    def test_get_best_params(self, experiment_tracker, mock_db_connection):
        """Test getting best parameters."""
        _, mock_cursor = mock_db_connection
        
        mock_responses = []
        for i in range(3):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        def objective(params):
            return params["x"] ** 2
        
        param_space = {
            "x": {"type": "float", "low": -5.0, "high": 5.0}
        }
        
        optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=3,
            direction="minimize"
        )
        
        best_params = optimizer.get_best_params()
        
        assert "best_params" in best_params
        assert "best_score" in best_params
        assert "n_trials" in best_params
        assert "strategy" in best_params
        assert best_params["strategy"] == "random"

    def test_get_best_params_without_optimization(self, experiment_tracker):
        """Test getting best params without running optimization."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        with pytest.raises(RuntimeError, match="No optimization has been run"):
            optimizer.get_best_params()

    def test_visualize_optimization(self, experiment_tracker, mock_db_connection, temp_storage_dir):
        """Test optimization visualization."""
        _, mock_cursor = mock_db_connection
        
        mock_responses = []
        for i in range(5):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        def objective(params):
            return params["x"] ** 2
        
        param_space = {
            "x": {"type": "float", "low": -5.0, "high": 5.0}
        }
        
        optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=5,
            direction="minimize"
        )
        
        output_path = Path(temp_storage_dir) / "optimization_report.png"
        result_path = optimizer.visualize_optimization(str(output_path))
        
        assert Path(result_path).exists()

    def test_visualize_optimization_without_optimization(self, experiment_tracker):
        """Test visualization without running optimization."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        with pytest.raises(RuntimeError, match="No optimization has been run"):
            optimizer.visualize_optimization()

    def test_generate_summary_report(self, experiment_tracker, mock_db_connection):
        """Test generating summary report."""
        _, mock_cursor = mock_db_connection
        
        mock_responses = []
        for i in range(5):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        def objective(params):
            return params["x"] ** 2
        
        param_space = {
            "x": {"type": "float", "low": -5.0, "high": 5.0}
        }
        
        optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=5,
            direction="minimize"
        )
        
        summary = optimizer.generate_summary_report()
        
        assert "strategy" in summary
        assert "n_trials" in summary
        assert "best_score" in summary
        assert "best_params" in summary
        assert "score_statistics" in summary
        assert "recommendations" in summary
        assert summary["n_trials"] == 5

    def test_generate_summary_report_without_optimization(self, experiment_tracker):
        """Test summary report without running optimization."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        with pytest.raises(RuntimeError, match="No optimization has been run"):
            optimizer.generate_summary_report()

    def test_grid_search_missing_values(self, experiment_tracker):
        """Test grid search with missing values key."""
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="grid"
        )
        
        def objective(params):
            return params["x"]
        
        param_space = {
            "x": {"type": "float", "low": 1.0, "high": 10.0}  # Missing 'values' key
        }
        
        with pytest.raises(ValueError, match="requires 'values' key"):
            optimizer.optimize(
                objective_function=objective,
                param_space=param_space,
                direction="maximize"
            )

    def test_failed_trial_handling(self, experiment_tracker, mock_db_connection):
        """Test handling of failed trials."""
        _, mock_cursor = mock_db_connection
        
        mock_responses = []
        for i in range(3):
            mock_responses.extend([
                None,
                {"experiment_id": uuid4()},
                {"experiment_id": uuid4()},
                {"run_id": uuid4()},
                {"run_id": uuid4()},
                {"metric_id": uuid4()},
                {"run_id": uuid4()}
            ])
        
        mock_cursor.fetchone.side_effect = mock_responses
        
        optimizer = HyperparameterOptimizer(
            experiment_tracker=experiment_tracker,
            strategy="random"
        )
        
        trial_count = [0]
        
        def objective(params):
            trial_count[0] += 1
            if trial_count[0] == 2:
                raise ValueError("Simulated failure")
            return params["x"]
        
        param_space = {
            "x": {"type": "float", "low": 1.0, "high": 10.0}
        }
        
        # Should handle the failure and continue
        result = optimizer.optimize(
            objective_function=objective,
            param_space=param_space,
            n_trials=3,
            direction="maximize"
        )
        
        # Check that failed trial is recorded
        failed_trials = [t for t in result.all_trials if t["state"] == "FAILED"]
        assert len(failed_trials) == 1
