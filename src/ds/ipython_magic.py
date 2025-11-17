"""IPython magic commands for MHRAS data science features."""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import (
    argument, magic_arguments, parse_argstring
)
from IPython import get_ipython

from src.database.connection import DatabaseConnection
from src.ds.storage import FileSystemStorage
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.notebook_integration import NotebookTracker
from src.ds.data_versioning import DataVersionControl
from src.ds.eda import EDAModule
from src.config import Config

logger = logging.getLogger(__name__)


@magics_class
class MHRASMagics(Magics):
    """IPython magic commands for MHRAS data science platform."""

    def __init__(self, shell):
        """
        Initialize magic commands.

        Args:
            shell: IPython shell instance
        """
        super().__init__(shell)

        # Initialize components (lazy loading)
        self._db_connection: Optional[DatabaseConnection] = None
        self._experiment_tracker: Optional[ExperimentTracker] = None
        self._notebook_tracker: Optional[NotebookTracker] = None
        self._data_version_control: Optional[DataVersionControl] = None
        self._eda_module: Optional[EDAModule] = None
        self._config: Optional[Config] = None

        logger.info("MHRAS magic commands loaded")

    @property
    def config(self) -> Config:
        """Get or create config instance."""
        if self._config is None:
            self._config = Config()
        return self._config

    @property
    def db_connection(self) -> DatabaseConnection:
        """Get or create database connection."""
        if self._db_connection is None:
            self._db_connection = DatabaseConnection(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
        return self._db_connection

    @property
    def experiment_tracker(self) -> ExperimentTracker:
        """Get or create experiment tracker."""
        if self._experiment_tracker is None:
            storage = FileSystemStorage(
                base_path=self.config.experiment_artifacts_path
            )
            self._experiment_tracker = ExperimentTracker(
                storage_backend=storage,
                db_connection=self.db_connection
            )
        return self._experiment_tracker

    @property
    def notebook_tracker(self) -> NotebookTracker:
        """Get or create notebook tracker."""
        if self._notebook_tracker is None:
            self._notebook_tracker = NotebookTracker(
                experiment_tracker=self.experiment_tracker
            )
        return self._notebook_tracker

    @property
    def data_version_control(self) -> DataVersionControl:
        """Get or create data version control."""
        if self._data_version_control is None:
            storage = FileSystemStorage(
                base_path=self.config.data_version_storage_path
            )
            self._data_version_control = DataVersionControl(
                storage_backend=storage,
                db_connection=self.db_connection
            )
        return self._data_version_control

    @property
    def eda_module(self) -> EDAModule:
        """Get or create EDA module."""
        if self._eda_module is None:
            self._eda_module = EDAModule()
        return self._eda_module

    @line_magic
    @magic_arguments()
    @argument('experiment_name', type=str, help='Name of the experiment')
    @argument('--run-name', type=str, default=None, help='Optional run name')
    @argument('--notebook-path', type=str, default='notebook.ipynb', help='Notebook path')
    def start_run(self, line):
        """
        Start an experiment run.

        Usage:
            %start_run my_experiment --run-name "test_run" --notebook-path "analysis.ipynb"
        """
        args = parse_argstring(self.start_run, line)

        try:
            run_id = self.notebook_tracker.start_notebook_run(
                notebook_path=args.notebook_path,
                experiment_name=args.experiment_name,
                run_name=args.run_name
            )
            print(f"✓ Started run: {run_id}")
            print(f"  Experiment: {args.experiment_name}")
            if args.run_name:
                print(f"  Run name: {args.run_name}")

        except Exception as e:
            print(f"✗ Error starting run: {e}")
            logger.error(f"Error starting run: {e}", exc_info=True)

    @line_magic
    def log_param(self, line):
        """
        Log a parameter to the active run.

        Usage:
            %log_param learning_rate=0.01
            %log_param batch_size=32 epochs=100
        """
        try:
            # Parse key=value pairs
            params = {}
            for pair in line.split():
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Try to convert to appropriate type
                    try:
                        # Try int
                        params[key] = int(value)
                    except ValueError:
                        try:
                            # Try float
                            params[key] = float(value)
                        except ValueError:
                            # Keep as string
                            params[key] = value

            if params:
                self.experiment_tracker.log_params(params)
                print(f"✓ Logged {len(params)} parameter(s)")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            else:
                print("✗ No parameters provided. Usage: %log_param key=value")

        except Exception as e:
            print(f"✗ Error logging parameters: {e}")
            logger.error(f"Error logging parameters: {e}", exc_info=True)

    @line_magic
    def log_metric(self, line):
        """
        Log a metric to the active run.

        Usage:
            %log_metric accuracy=0.95
            %log_metric loss=0.05 accuracy=0.95 --step 10
        """
        try:
            # Parse arguments
            parts = line.split()
            metrics = {}
            step = None

            i = 0
            while i < len(parts):
                part = parts[i]
                if part == '--step' and i + 1 < len(parts):
                    step = int(parts[i + 1])
                    i += 2
                elif '=' in part:
                    key, value = part.split('=', 1)
                    metrics[key] = float(value)
                    i += 1
                else:
                    i += 1

            if metrics:
                self.experiment_tracker.log_metrics(metrics, step=step)
                print(f"✓ Logged {len(metrics)} metric(s)" + (f" at step {step}" if step else ""))
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            else:
                print("✗ No metrics provided. Usage: %log_metric key=value")

        except Exception as e:
            print(f"✗ Error logging metrics: {e}")
            logger.error(f"Error logging metrics: {e}", exc_info=True)

    @line_magic
    @magic_arguments()
    @argument('artifact_path', type=str, help='Path to artifact file')
    @argument('--type', type=str, default='file', help='Artifact type')
    def log_artifact(self, line):
        """
        Log an artifact to the active run.

        Usage:
            %log_artifact model.pkl --type model
            %log_artifact plot.png --type plot
        """
        args = parse_argstring(self.log_artifact, line)

        try:
            artifact_uri = self.experiment_tracker.log_artifact(
                artifact_path=args.artifact_path,
                artifact_type=args.type
            )
            print(f"✓ Logged artifact: {args.artifact_path}")
            print(f"  Type: {args.type}")
            print(f"  URI: {artifact_uri}")

        except Exception as e:
            print(f"✗ Error logging artifact: {e}")
            logger.error(f"Error logging artifact: {e}", exc_info=True)

    @line_magic
    @magic_arguments()
    @argument('--status', type=str, default='FINISHED', help='Run status')
    def end_run(self, line):
        """
        End the active run.

        Usage:
            %end_run
            %end_run --status FAILED
        """
        args = parse_argstring(self.end_run, line)

        try:
            self.notebook_tracker.end_notebook_run(status=args.status)
            print(f"✓ Ended run with status: {args.status}")

        except Exception as e:
            print(f"✗ Error ending run: {e}")
            logger.error(f"Error ending run: {e}", exc_info=True)

    @line_magic
    @magic_arguments()
    @argument('dataset_var', type=str, help='Variable name containing the dataset')
    @argument('dataset_name', type=str, help='Name for the dataset')
    @argument('--source', type=str, default='notebook', help='Data source')
    def track_data(self, line):
        """
        Track a dataset version from a notebook variable.

        Usage:
            %track_data df my_dataset --source "preprocessing"
        """
        args = parse_argstring(self.track_data, line)

        try:
            # Get the dataset from the user namespace
            user_ns = self.shell.user_ns
            if args.dataset_var not in user_ns:
                print(f"✗ Variable '{args.dataset_var}' not found in namespace")
                return

            dataset = user_ns[args.dataset_var]

            if not isinstance(dataset, pd.DataFrame):
                print(f"✗ Variable '{args.dataset_var}' is not a pandas DataFrame")
                return

            # Register the dataset
            dataset_version = self.data_version_control.register_dataset(
                dataset=dataset,
                dataset_name=args.dataset_name,
                source=args.source
            )

            print(f"✓ Tracked dataset: {args.dataset_name}")
            print(f"  Version: {dataset_version.version}")
            print(f"  Rows: {dataset_version.num_rows}")
            print(f"  Columns: {dataset_version.num_columns}")
            print(f"  Version ID: {dataset_version.version_id}")

        except Exception as e:
            print(f"✗ Error tracking dataset: {e}")
            logger.error(f"Error tracking dataset: {e}", exc_info=True)

    @cell_magic
    @magic_arguments()
    @argument('dataset_var', type=str, help='Variable name containing the dataset')
    @argument('--output-dir', type=str, default='eda_output', help='Output directory for visualizations')
    @argument('--target', type=str, default=None, help='Target column name')
    def eda(self, line, cell=None):
        """
        Run exploratory data analysis on a dataset.

        Usage:
            %%eda df --output-dir ./plots --target outcome
        """
        args = parse_argstring(self.eda, line)

        try:
            # Get the dataset from the user namespace
            user_ns = self.shell.user_ns
            if args.dataset_var not in user_ns:
                print(f"✗ Variable '{args.dataset_var}' not found in namespace")
                return

            dataset = user_ns[args.dataset_var]

            if not isinstance(dataset, pd.DataFrame):
                print(f"✗ Variable '{args.dataset_var}' is not a pandas DataFrame")
                return

            print(f"Running EDA on '{args.dataset_var}'...")
            print(f"  Shape: {dataset.shape}")

            # Run EDA
            report = self.eda_module.analyze_dataset(
                data=dataset,
                target_column=args.target
            )

            # Display summary
            print(f"\n✓ EDA Complete")
            print(f"  Dataset: {report.dataset_name}")
            print(f"  Rows: {report.num_rows:,}")
            print(f"  Columns: {report.num_columns}")
            print(f"  Quality Issues: {len(report.quality_issues)}")

            # Show quality issues
            if report.quality_issues:
                print(f"\n  Quality Issues:")
                for issue in report.quality_issues[:5]:  # Show first 5
                    print(f"    [{issue.severity.upper()}] {issue.description}")
                if len(report.quality_issues) > 5:
                    print(f"    ... and {len(report.quality_issues) - 5} more")

            # Generate visualizations
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            viz_paths = self.eda_module.generate_visualizations(
                data=dataset,
                output_dir=str(output_path)
            )

            if viz_paths:
                print(f"\n  Generated {len(viz_paths)} visualization(s) in '{args.output_dir}'")

            # Store report in namespace for further inspection
            user_ns['eda_report'] = report
            print(f"\n  Report stored in variable 'eda_report'")

        except Exception as e:
            print(f"✗ Error running EDA: {e}")
            logger.error(f"Error running EDA: {e}", exc_info=True)

    @line_magic
    def mhras_status(self, line):
        """
        Show status of active run and connections.

        Usage:
            %mhras_status
        """
        print("MHRAS Data Science Platform Status")
        print("=" * 50)

        # Check active run
        if self._experiment_tracker and self._experiment_tracker.active_run:
            run = self._experiment_tracker.active_run
            print(f"✓ Active Run: {run.run_id}")
            print(f"  Experiment: {run.experiment_id}")
            print(f"  Status: {run.status}")
            print(f"  Started: {run.start_time}")
        else:
            print("○ No active run")

        # Check database connection
        try:
            if self._db_connection:
                print("✓ Database: Connected")
            else:
                print("○ Database: Not initialized")
        except Exception as e:
            print(f"✗ Database: Error - {e}")

        print("=" * 50)


def load_ipython_extension(ipython):
    """
    Load the IPython extension.

    This function is called by IPython when the extension is loaded.

    Args:
        ipython: IPython shell instance
    """
    ipython.register_magics(MHRASMagics)
    print("MHRAS magic commands loaded successfully!")
    print("Available commands:")
    print("  %start_run <experiment_name> - Start an experiment run")
    print("  %log_param key=value - Log parameters")
    print("  %log_metric key=value - Log metrics")
    print("  %log_artifact <path> - Log artifacts")
    print("  %end_run - End the active run")
    print("  %track_data <var> <name> - Track dataset version")
    print("  %%eda <var> - Run exploratory data analysis")
    print("  %mhras_status - Show platform status")
    print("\nFor help on any command, use: %command_name?")


def unload_ipython_extension(ipython):
    """
    Unload the IPython extension.

    Args:
        ipython: IPython shell instance
    """
    print("MHRAS magic commands unloaded")
