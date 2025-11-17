"""Notebook integration for experiment tracking and reproducibility."""

import logging
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import UUID
import sys

from src.ds.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class NotebookTracker:
    """Track notebook executions for reproducibility."""

    def __init__(self, experiment_tracker: ExperimentTracker):
        """
        Initialize NotebookTracker.

        Args:
            experiment_tracker: ExperimentTracker instance
        """
        self.tracker = experiment_tracker
        self.notebook_path: Optional[str] = None
        self.cell_executions: List[Dict[str, Any]] = []
        self.environment: Optional[Dict[str, str]] = None

        logger.info("NotebookTracker initialized")

    def start_notebook_run(
        self,
        notebook_path: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start tracking a notebook execution.

        Args:
            notebook_path: Path to the notebook file
            experiment_name: Name of the experiment
            run_name: Optional name for this run
            tags: Optional tags for the run

        Returns:
            Run ID as string

        Raises:
            RuntimeError: If a run is already active
        """
        if self.tracker.active_run is not None:
            raise RuntimeError(
                "A run is already active. End the current run before starting a new one."
            )

        self.notebook_path = notebook_path
        self.cell_executions = []

        # Capture environment
        self.environment = self.capture_environment()

        # Add notebook metadata to tags
        notebook_tags = tags or {}
        notebook_tags.update({
            'notebook_path': notebook_path,
            'notebook_name': Path(notebook_path).name,
            'execution_type': 'notebook'
        })

        # Start run
        run = self.tracker.start_run(
            experiment_name=experiment_name,
            run_name=run_name or f"notebook_{Path(notebook_path).stem}",
            tags=notebook_tags
        )

        # Log environment as parameters
        self.tracker.log_params({
            'python_version': self.environment.get('python_version', 'unknown'),
            'notebook_path': notebook_path
        })

        logger.info(
            f"Started notebook run {run.run_id} for '{notebook_path}' "
            f"in experiment '{experiment_name}'"
        )

        return str(run.run_id)

    def log_cell_execution(
        self,
        cell_id: str,
        cell_code: str,
        outputs: Optional[List[Any]] = None,
        execution_count: Optional[int] = None
    ) -> None:
        """
        Log individual cell execution.

        Args:
            cell_id: Unique cell identifier
            cell_code: Source code of the cell
            outputs: Optional list of cell outputs
            execution_count: Optional execution count
        """
        if self.tracker.active_run is None:
            logger.warning("No active run. Cell execution not logged.")
            return

        cell_execution = {
            'cell_id': cell_id,
            'execution_count': execution_count,
            'code_length': len(cell_code),
            'has_outputs': outputs is not None and len(outputs) > 0,
            'timestamp': datetime.utcnow().isoformat()
        }

        self.cell_executions.append(cell_execution)

        logger.debug(
            f"Logged cell execution {cell_id} "
            f"(count: {execution_count}) for run {self.tracker.active_run_id}"
        )

    def capture_environment(self) -> Dict[str, str]:
        """
        Capture Python environment including package versions.

        Returns:
            Dictionary of environment information
        """
        environment = {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': sys.platform
        }

        # Capture installed packages
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)
                package_versions = {
                    pkg['name']: pkg['version']
                    for pkg in packages
                }
                environment['packages'] = json.dumps(package_versions)
                environment['package_count'] = str(len(packages))

        except Exception as e:
            logger.warning(f"Could not capture package list: {e}")
            environment['packages'] = '{}'
            environment['package_count'] = '0'

        logger.info(
            f"Captured environment with {environment.get('package_count', 0)} packages"
        )

        return environment

    def end_notebook_run(self, status: str = "FINISHED") -> None:
        """
        End the notebook run and log execution summary.

        Args:
            status: Final status (FINISHED or FAILED)
        """
        if self.tracker.active_run is None:
            logger.warning("No active run to end.")
            return

        # Log cell execution summary as metrics
        if self.cell_executions:
            self.tracker.log_metrics({
                'total_cells_executed': float(len(self.cell_executions)),
                'cells_with_outputs': float(
                    sum(1 for cell in self.cell_executions if cell['has_outputs'])
                )
            })

        # Save environment as artifact
        if self.environment:
            env_path = Path('/tmp') / f"environment_{self.tracker.active_run_id}.json"
            with open(env_path, 'w') as f:
                json.dump(self.environment, f, indent=2)

            try:
                self.tracker.log_artifact(str(env_path), artifact_type="environment")
            except Exception as e:
                logger.warning(f"Could not log environment artifact: {e}")
            finally:
                # Clean up temp file
                if env_path.exists():
                    env_path.unlink()

        # Save cell execution log
        if self.cell_executions:
            cells_path = Path('/tmp') / f"cells_{self.tracker.active_run_id}.json"
            with open(cells_path, 'w') as f:
                json.dump(self.cell_executions, f, indent=2)

            try:
                self.tracker.log_artifact(str(cells_path), artifact_type="cell_log")
            except Exception as e:
                logger.warning(f"Could not log cell execution artifact: {e}")
            finally:
                # Clean up temp file
                if cells_path.exists():
                    cells_path.unlink()

        # End the run
        self.tracker.end_run(status=status)

        logger.info(f"Ended notebook run with status {status}")

        # Reset state
        self.notebook_path = None
        self.cell_executions = []
        self.environment = None


def strip_notebook_outputs(notebook_path: str, output_path: Optional[str] = None) -> str:
    """
    Strip output cells from a Jupyter notebook for version control.

    Args:
        notebook_path: Path to the notebook file
        output_path: Optional output path (defaults to overwriting input)

    Returns:
        Path to the stripped notebook

    Raises:
        FileNotFoundError: If notebook file doesn't exist
        ValueError: If notebook format is invalid
    """
    notebook_file = Path(notebook_path)
    if not notebook_file.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Read notebook
    with open(notebook_file, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid notebook format: {e}")

    # Validate notebook structure
    if 'cells' not in notebook:
        raise ValueError("Invalid notebook: missing 'cells' field")

    # Strip outputs and execution counts
    cells_stripped = 0
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'code':
            # Clear outputs
            if 'outputs' in cell and cell['outputs']:
                cell['outputs'] = []
                cells_stripped += 1

            # Clear execution count
            if 'execution_count' in cell:
                cell['execution_count'] = None

    # Determine output path
    if output_path is None:
        output_path = notebook_path
    output_file = Path(output_path)

    # Write stripped notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')  # Add trailing newline

    logger.info(
        f"Stripped outputs from {cells_stripped} cells in '{notebook_path}'"
    )

    return str(output_file)


def detect_unversioned_dependencies(notebook_path: str) -> List[Dict[str, Any]]:
    """
    Detect unversioned data or dependencies in a notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        List of detected issues with details

    Raises:
        FileNotFoundError: If notebook file doesn't exist
    """
    notebook_file = Path(notebook_path)
    if not notebook_file.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Read notebook
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    issues = []

    # Patterns to detect
    patterns = {
        'hardcoded_path': re.compile(r'["\'](?:/[^"\']+|[A-Za-z]:\\[^"\']+)["\']'),
        'read_csv': re.compile(r'\.read_csv\s*\('),
        'read_excel': re.compile(r'\.read_excel\s*\('),
        'pickle_load': re.compile(r'pickle\.load\s*\('),
        'joblib_load': re.compile(r'joblib\.load\s*\('),
    }

    # Check each code cell
    for idx, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # Check for hardcoded paths
        if patterns['hardcoded_path'].search(source):
            issues.append({
                'type': 'hardcoded_path',
                'severity': 'medium',
                'cell_index': idx,
                'message': 'Hardcoded file path detected',
                'recommendation': 'Use relative paths or configuration'
            })

        # Check for data loading without versioning
        if patterns['read_csv'].search(source) or patterns['read_excel'].search(source):
            issues.append({
                'type': 'unversioned_data',
                'severity': 'high',
                'cell_index': idx,
                'message': 'Data loading without version tracking detected',
                'recommendation': 'Use DataVersionControl to track dataset versions'
            })

        # Check for model loading without tracking
        if patterns['pickle_load'].search(source) or patterns['joblib_load'].search(source):
            issues.append({
                'type': 'untracked_model',
                'severity': 'medium',
                'cell_index': idx,
                'message': 'Model loading without tracking detected',
                'recommendation': 'Use ModelRegistry to load tracked models'
            })

    logger.info(
        f"Detected {len(issues)} potential issues in '{notebook_path}'"
    )

    return issues
