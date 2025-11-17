"""Tests for notebook integration functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from uuid import uuid4

from src.ds.notebook_integration import (
    NotebookTracker,
    strip_notebook_outputs,
    detect_unversioned_dependencies
)
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.models import Run


class TestNotebookTracker:
    """Test notebook tracking functionality."""

    @pytest.fixture
    def mock_experiment_tracker(self):
        """Create a mock experiment tracker."""
        tracker = Mock(spec=ExperimentTracker)
        tracker.active_run = None
        tracker.active_run_id = None
        return tracker

    @pytest.fixture
    def notebook_tracker(self, mock_experiment_tracker):
        """Create a notebook tracker instance."""
        return NotebookTracker(experiment_tracker=mock_experiment_tracker)

    def test_initialization(self, notebook_tracker):
        """Test notebook tracker initialization."""
        assert notebook_tracker.tracker is not None
        assert notebook_tracker.notebook_path is None
        assert notebook_tracker.cell_executions == []
        assert notebook_tracker.environment is None

    def test_start_notebook_run(self, notebook_tracker, mock_experiment_tracker):
        """Test starting a notebook run."""
        # Setup mock
        run_id = uuid4()
        mock_run = Run(
            run_id=run_id,
            experiment_id=uuid4(),
            run_name="test_notebook",
            status="RUNNING",
            start_time=datetime.utcnow()
        )
        mock_experiment_tracker.start_run.return_value = mock_run

        # Start notebook run
        result = notebook_tracker.start_notebook_run(
            notebook_path="test.ipynb",
            experiment_name="test_experiment",
            run_name="test_run"
        )

        # Verify
        assert result == str(run_id)
        assert notebook_tracker.notebook_path == "test.ipynb"
        assert notebook_tracker.environment is not None
        mock_experiment_tracker.start_run.assert_called_once()
        mock_experiment_tracker.log_params.assert_called_once()

    def test_start_notebook_run_with_active_run(self, notebook_tracker, mock_experiment_tracker):
        """Test starting a notebook run when one is already active."""
        # Setup mock with active run
        mock_experiment_tracker.active_run = Mock()

        # Should raise error
        with pytest.raises(RuntimeError, match="already active"):
            notebook_tracker.start_notebook_run(
                notebook_path="test.ipynb",
                experiment_name="test_experiment"
            )

    def test_log_cell_execution(self, notebook_tracker, mock_experiment_tracker):
        """Test logging cell execution."""
        # Setup active run
        mock_experiment_tracker.active_run = Mock()
        mock_experiment_tracker.active_run_id = uuid4()

        # Log cell execution
        notebook_tracker.log_cell_execution(
            cell_id="cell_1",
            cell_code="print('hello')",
            outputs=["hello"],
            execution_count=1
        )

        # Verify
        assert len(notebook_tracker.cell_executions) == 1
        cell_exec = notebook_tracker.cell_executions[0]
        assert cell_exec['cell_id'] == "cell_1"
        assert cell_exec['execution_count'] == 1
        assert cell_exec['has_outputs'] is True

    def test_log_cell_execution_no_active_run(self, notebook_tracker):
        """Test logging cell execution without active run."""
        # Should not raise error, just log warning
        notebook_tracker.log_cell_execution(
            cell_id="cell_1",
            cell_code="print('hello')"
        )
        # No cells should be logged
        assert len(notebook_tracker.cell_executions) == 0

    @patch('subprocess.run')
    def test_capture_environment(self, mock_subprocess, notebook_tracker):
        """Test capturing Python environment."""
        # Mock pip list output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps([
            {"name": "pandas", "version": "2.0.0"},
            {"name": "numpy", "version": "1.24.0"}
        ])
        mock_subprocess.return_value = mock_result

        # Capture environment
        env = notebook_tracker.capture_environment()

        # Verify
        assert 'python_version' in env
        assert 'python_executable' in env
        assert 'platform' in env
        assert 'packages' in env
        assert 'package_count' in env
        assert env['package_count'] == '2'

    @patch('subprocess.run')
    def test_capture_environment_pip_failure(self, mock_subprocess, notebook_tracker):
        """Test capturing environment when pip fails."""
        # Mock pip failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        # Should still return basic environment
        env = notebook_tracker.capture_environment()

        assert 'python_version' in env
        assert env['packages'] == '{}'
        assert env['package_count'] == '0'

    def test_end_notebook_run(self, notebook_tracker, mock_experiment_tracker):
        """Test ending a notebook run."""
        # Setup active run
        mock_experiment_tracker.active_run = Mock()
        mock_experiment_tracker.active_run_id = uuid4()

        # Add some cell executions
        notebook_tracker.cell_executions = [
            {'cell_id': '1', 'has_outputs': True},
            {'cell_id': '2', 'has_outputs': False}
        ]
        notebook_tracker.environment = {'python_version': '3.9'}

        # End run
        notebook_tracker.end_notebook_run(status="FINISHED")

        # Verify
        mock_experiment_tracker.log_metrics.assert_called_once()
        mock_experiment_tracker.end_run.assert_called_once_with(status="FINISHED")
        assert notebook_tracker.notebook_path is None
        assert notebook_tracker.cell_executions == []
        assert notebook_tracker.environment is None

    def test_end_notebook_run_no_active_run(self, notebook_tracker):
        """Test ending notebook run without active run."""
        # Should not raise error
        notebook_tracker.end_notebook_run()


class TestStripNotebookOutputs:
    """Test notebook output stripping functionality."""

    @pytest.fixture
    def sample_notebook(self):
        """Create a sample notebook with outputs."""
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "source": ["print('hello')"],
                    "outputs": [
                        {
                            "output_type": "stream",
                            "text": ["hello\n"]
                        }
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": ["# Title"]
                },
                {
                    "cell_type": "code",
                    "execution_count": 2,
                    "source": ["x = 5"],
                    "outputs": []
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }

    def test_strip_notebook_outputs(self, sample_notebook):
        """Test stripping outputs from notebook."""
        # Create temporary notebook file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(sample_notebook, f)
            notebook_path = f.name

        try:
            # Strip outputs
            result_path = strip_notebook_outputs(notebook_path)

            # Read stripped notebook
            with open(result_path, 'r') as f:
                stripped = json.load(f)

            # Verify outputs are stripped
            for cell in stripped['cells']:
                if cell['cell_type'] == 'code':
                    assert cell['outputs'] == []
                    assert cell['execution_count'] is None

        finally:
            # Cleanup
            Path(notebook_path).unlink()

    def test_strip_notebook_outputs_to_different_file(self, sample_notebook):
        """Test stripping outputs to a different file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(sample_notebook, f)
            input_path = f.name

        output_path = input_path.replace('.ipynb', '_stripped.ipynb')

        try:
            # Strip to different file
            result_path = strip_notebook_outputs(input_path, output_path)

            assert result_path == output_path
            assert Path(output_path).exists()

            # Original should still have outputs
            with open(input_path, 'r') as f:
                original = json.load(f)
            assert len(original['cells'][0]['outputs']) > 0

            # Stripped should not have outputs
            with open(output_path, 'r') as f:
                stripped = json.load(f)
            assert stripped['cells'][0]['outputs'] == []

        finally:
            # Cleanup
            Path(input_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_strip_notebook_outputs_file_not_found(self):
        """Test stripping outputs from non-existent file."""
        with pytest.raises(FileNotFoundError):
            strip_notebook_outputs("nonexistent.ipynb")

    def test_strip_notebook_outputs_invalid_json(self):
        """Test stripping outputs from invalid notebook."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            f.write("invalid json")
            notebook_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid notebook format"):
                strip_notebook_outputs(notebook_path)
        finally:
            Path(notebook_path).unlink()

    def test_strip_notebook_outputs_missing_cells(self):
        """Test stripping outputs from notebook without cells field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump({"metadata": {}}, f)
            notebook_path = f.name

        try:
            with pytest.raises(ValueError, match="missing 'cells' field"):
                strip_notebook_outputs(notebook_path)
        finally:
            Path(notebook_path).unlink()


class TestDetectUnversionedDependencies:
    """Test detection of unversioned dependencies."""

    @pytest.fixture
    def sample_notebook_with_issues(self):
        """Create a notebook with various issues."""
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "import pandas as pd\n",
                        "df = pd.read_csv('/absolute/path/to/data.csv')\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "import pickle\n",
                        "model = pickle.load(open('model.pkl', 'rb'))\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "source": [
                        "data = pd.read_excel('C:\\\\Users\\\\data.xlsx')\n"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "source": ["# Just markdown"]
                }
            ]
        }

    def test_detect_unversioned_dependencies(self, sample_notebook_with_issues):
        """Test detecting unversioned dependencies."""
        # Create temporary notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(sample_notebook_with_issues, f)
            notebook_path = f.name

        try:
            # Detect issues
            issues = detect_unversioned_dependencies(notebook_path)

            # Verify issues detected
            assert len(issues) > 0

            # Check for specific issue types
            issue_types = [issue['type'] for issue in issues]
            assert 'hardcoded_path' in issue_types
            assert 'unversioned_data' in issue_types
            assert 'untracked_model' in issue_types

            # Verify issue structure
            for issue in issues:
                assert 'type' in issue
                assert 'severity' in issue
                assert 'cell_index' in issue
                assert 'message' in issue
                assert 'recommendation' in issue

        finally:
            Path(notebook_path).unlink()

    def test_detect_unversioned_dependencies_clean_notebook(self):
        """Test detecting dependencies in a clean notebook."""
        clean_notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd\n", "print('hello')\n"]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(clean_notebook, f)
            notebook_path = f.name

        try:
            issues = detect_unversioned_dependencies(notebook_path)
            # Should have no or minimal issues
            assert isinstance(issues, list)

        finally:
            Path(notebook_path).unlink()

    def test_detect_unversioned_dependencies_file_not_found(self):
        """Test detecting dependencies in non-existent file."""
        with pytest.raises(FileNotFoundError):
            detect_unversioned_dependencies("nonexistent.ipynb")
