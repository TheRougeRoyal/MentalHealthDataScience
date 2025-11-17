"""
Command-line interface for MHRAS management and operations.
"""

import click
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import json

from src.integration import get_integration, reset_integration
from src.database.connection import DatabaseConnection
from src.database.migration_runner import MigrationRunner
from src.screening_service import ScreeningRequest
from src.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


@click.group()
def cli():
    """MHRAS Command Line Interface"""
    pass


@cli.group()
def db():
    """Database management commands"""
    pass


@db.command()
@click.option('--host', default='localhost', help='Database host')
@click.option('--port', default=5432, help='Database port')
@click.option('--database', default='mhras', help='Database name')
@click.option('--user', default='postgres', help='Database user')
@click.option('--password', prompt=True, hide_input=True, help='Database password')
def migrate(host, port, database, user, password):
    """Run database migrations"""
    click.echo("Running database migrations...")
    
    try:
        db_conn = DatabaseConnection(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        runner = MigrationRunner(db_conn)
        runner.run_migrations()
        
        click.echo("✓ Migrations completed successfully")
    except Exception as e:
        click.echo(f"✗ Migration failed: {str(e)}", err=True)
        raise click.Abort()


@db.command()
@click.option('--host', default='localhost', help='Database host')
@click.option('--port', default=5432, help='Database port')
@click.option('--database', default='mhras', help='Database name')
@click.option('--user', default='postgres', help='Database user')
@click.option('--password', prompt=True, hide_input=True, help='Database password')
def check(host, port, database, user, password):
    """Check database connection"""
    click.echo("Checking database connection...")
    
    try:
        db_conn = DatabaseConnection(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        db_conn.health_check()
        click.echo("✓ Database connection successful")
    except Exception as e:
        click.echo(f"✗ Database connection failed: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def models():
    """Model management commands"""
    pass


@models.command()
def list():
    """List all registered models"""
    integration = get_integration()
    registry = integration.get_model_registry()
    
    all_models = registry.list_models()
    active_models = registry.get_active_models()
    
    click.echo(f"\nTotal models: {len(all_models)}")
    click.echo(f"Active models: {len(active_models)}\n")
    
    for model in all_models:
        status = "✓ ACTIVE" if model in active_models else "  INACTIVE"
        click.echo(f"{status} {model['model_id']} ({model['model_type']})")
        click.echo(f"         Version: {model.get('version', 'N/A')}")
        click.echo(f"         Registered: {model.get('registered_at', 'N/A')}\n")


@models.command()
@click.argument('model_id')
def activate(model_id):
    """Activate a model"""
    integration = get_integration()
    registry = integration.get_model_registry()
    
    try:
        registry.set_model_active(model_id, True)
        click.echo(f"✓ Model {model_id} activated")
    except Exception as e:
        click.echo(f"✗ Failed to activate model: {str(e)}", err=True)
        raise click.Abort()


@models.command()
@click.argument('model_id')
def deactivate(model_id):
    """Deactivate a model"""
    integration = get_integration()
    registry = integration.get_model_registry()
    
    try:
        registry.set_model_active(model_id, False)
        click.echo(f"✓ Model {model_id} deactivated")
    except Exception as e:
        click.echo(f"✗ Failed to deactivate model: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def review():
    """Human review queue commands"""
    pass


@review.command()
def queue():
    """Show review queue status"""
    integration = get_integration()
    queue = integration.get_human_review_queue()
    
    stats = queue.get_queue_statistics()
    
    click.echo("\n=== Review Queue Status ===\n")
    click.echo(f"Total cases: {stats['total_cases']}")
    click.echo(f"Pending: {stats['pending_count']}")
    click.echo(f"Overdue: {stats['overdue_count']}")
    
    if stats['oldest_pending_hours']:
        click.echo(f"Oldest pending: {stats['oldest_pending_hours']:.1f} hours")
    
    if stats['average_review_time_hours']:
        click.echo(f"Avg review time: {stats['average_review_time_hours']:.1f} hours")
    
    click.echo("\nBy Status:")
    for status, count in stats['by_status'].items():
        click.echo(f"  {status}: {count}")
    
    click.echo("\nBy Priority:")
    for priority, count in stats['by_priority'].items():
        click.echo(f"  Priority {priority}: {count}")


@review.command()
def escalate():
    """Check and escalate overdue cases"""
    integration = get_integration()
    queue = integration.get_human_review_queue()
    
    escalated = queue.check_escalations()
    
    if escalated:
        click.echo(f"✓ Escalated {len(escalated)} overdue cases")
        for case in escalated:
            click.echo(f"  - Case {case.case_id} (Risk: {case.risk_score})")
    else:
        click.echo("No cases require escalation")


@cli.group()
def audit():
    """Audit and compliance commands"""
    pass


@audit.command()
@click.option('--days', default=7, help='Number of days to include in report')
@click.option('--output', type=click.Path(), help='Output file path')
def report(days, output):
    """Generate audit report"""
    integration = get_integration()
    audit_logger = integration.get_audit_logger()
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    click.echo(f"Generating audit report for last {days} days...")
    
    report_data = audit_logger.generate_audit_report(start_date, end_date)
    
    # Display summary
    click.echo(f"\n=== Audit Report ===")
    click.echo(f"Period: {start_date.date()} to {end_date.date()}")
    click.echo(f"Total events: {report_data['total_events']}")
    click.echo(f"Unique individuals: {report_data['unique_individuals']}")
    
    if report_data['screening_statistics']:
        stats = report_data['screening_statistics']
        click.echo(f"\nScreenings: {stats['total_screenings']}")
        click.echo(f"Alerts triggered: {stats['alerts_triggered']}")
        click.echo(f"Human reviews required: {stats['human_reviews_required']}")
    
    # Save to file if requested
    if output:
        with open(output, 'w') as f:
            json.dump(report_data, f, indent=2)
        click.echo(f"\n✓ Report saved to {output}")


@cli.group()
def system():
    """System management commands"""
    pass


@system.command()
def health():
    """Check system health"""
    integration = get_integration()
    
    health_status = integration.health_check()
    
    click.echo("\n=== System Health Check ===\n")
    click.echo(f"Overall Status: {health_status['overall_status'].upper()}")
    click.echo(f"Timestamp: {health_status['timestamp']}\n")
    
    click.echo("Components:")
    for component, status in health_status['components'].items():
        if isinstance(status, dict):
            status_str = status.get('status', 'unknown')
            click.echo(f"  {component}: {status_str}")
        else:
            click.echo(f"  {component}: {status}")


@system.command()
def stats():
    """Show system statistics"""
    integration = get_integration()
    
    stats = integration.get_statistics()
    
    click.echo("\n=== System Statistics ===\n")
    click.echo(f"Timestamp: {stats['timestamp']}\n")
    
    if 'models' in stats:
        click.echo(f"Models:")
        click.echo(f"  Active: {stats['models']['active_count']}")
        click.echo(f"  Total: {stats['models']['total_count']}\n")
    
    if 'review_queue' in stats:
        click.echo(f"Review Queue:")
        click.echo(f"  Total cases: {stats['review_queue']['total_cases']}")
        click.echo(f"  Pending: {stats['review_queue']['pending_count']}")
        click.echo(f"  Overdue: {stats['review_queue']['overdue_count']}\n")


@system.command()
def clear_cache():
    """Clear system caches"""
    integration = get_integration()
    
    integration.screening_service.clear_cache()
    
    click.echo("✓ System caches cleared")


@cli.group()
def experiments():
    """Experiment tracking commands"""
    pass


@experiments.command('list')
def experiments_list():
    """List all experiments"""
    from src.ds.experiment_tracker import ExperimentTracker
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    tracker = ExperimentTracker(storage, db_conn)
    
    try:
        all_experiments = tracker.list_experiments()
        
        if not all_experiments:
            click.echo("No experiments found")
            return
        
        click.echo(f"\nTotal experiments: {len(all_experiments)}\n")
        
        for exp in all_experiments:
            click.echo(f"ID: {exp['experiment_id']}")
            click.echo(f"Name: {exp['experiment_name']}")
            if exp.get('description'):
                click.echo(f"Description: {exp['description']}")
            click.echo(f"Created: {exp['created_at']}")
            if exp.get('tags'):
                click.echo(f"Tags: {exp['tags']}")
            click.echo()
        
    except Exception as e:
        click.echo(f"✗ Failed to list experiments: {str(e)}", err=True)
        raise click.Abort()


@experiments.command('show')
@click.argument('experiment_id')
def experiments_show(experiment_id):
    """Display experiment details"""
    from src.ds.experiment_tracker import ExperimentTracker
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    tracker = ExperimentTracker(storage, db_conn)
    
    try:
        experiment = tracker.get_experiment(experiment_id)
        
        if not experiment:
            click.echo(f"✗ Experiment {experiment_id} not found", err=True)
            raise click.Abort()
        
        click.echo(f"\n=== Experiment Details ===\n")
        click.echo(f"ID: {experiment['experiment_id']}")
        click.echo(f"Name: {experiment['experiment_name']}")
        if experiment.get('description'):
            click.echo(f"Description: {experiment['description']}")
        click.echo(f"Created: {experiment['created_at']}")
        if experiment.get('tags'):
            click.echo(f"Tags: {json.dumps(experiment['tags'], indent=2)}")
        
        # Get runs for this experiment
        runs = tracker.search_runs(experiment_name=experiment['experiment_name'])
        click.echo(f"\nTotal runs: {len(runs)}")
        
        if runs:
            click.echo("\nRecent runs:")
            for run in runs[:5]:
                click.echo(f"  - {run.run_id} ({run.status}) - {run.start_time}")
        
    except Exception as e:
        click.echo(f"✗ Failed to show experiment: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def runs():
    """Experiment run commands"""
    pass


@runs.command('list')
@click.argument('experiment_id')
@click.option('--status', help='Filter by status (RUNNING, FINISHED, FAILED)')
@click.option('--limit', default=20, help='Maximum number of runs to display')
def runs_list(experiment_id, status, limit):
    """List runs for an experiment"""
    from src.ds.experiment_tracker import ExperimentTracker
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    tracker = ExperimentTracker(storage, db_conn)
    
    try:
        # Get experiment to validate it exists
        experiment = tracker.get_experiment(experiment_id)
        if not experiment:
            click.echo(f"✗ Experiment {experiment_id} not found", err=True)
            raise click.Abort()
        
        # Search runs
        filter_string = f"status = '{status}'" if status else None
        runs_list = tracker.search_runs(
            experiment_name=experiment['experiment_name'],
            filter_string=filter_string
        )
        
        if not runs_list:
            click.echo("No runs found")
            return
        
        click.echo(f"\nExperiment: {experiment['experiment_name']}")
        click.echo(f"Total runs: {len(runs_list)}\n")
        
        for run in runs_list[:limit]:
            click.echo(f"Run ID: {run.run_id}")
            click.echo(f"  Name: {run.run_name or 'N/A'}")
            click.echo(f"  Status: {run.status}")
            click.echo(f"  Start: {run.start_time}")
            if run.end_time:
                click.echo(f"  End: {run.end_time}")
            if run.metrics:
                click.echo(f"  Metrics: {', '.join(run.metrics.keys())}")
            click.echo()
        
        if len(runs_list) > limit:
            click.echo(f"... and {len(runs_list) - limit} more runs")
        
    except Exception as e:
        click.echo(f"✗ Failed to list runs: {str(e)}", err=True)
        raise click.Abort()


@runs.command('compare')
@click.argument('run_id1')
@click.argument('run_id2')
@click.option('--metrics', help='Comma-separated list of metrics to compare')
def runs_compare(run_id1, run_id2, metrics):
    """Compare two experiment runs"""
    from src.ds.experiment_tracker import ExperimentTracker
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    tracker = ExperimentTracker(storage, db_conn)
    
    try:
        metric_names = metrics.split(',') if metrics else None
        comparison = tracker.compare_runs([run_id1, run_id2], metric_names)
        
        if comparison.empty:
            click.echo("No comparison data available")
            return
        
        click.echo(f"\n=== Run Comparison ===\n")
        click.echo(comparison.to_string())
        click.echo()
        
    except Exception as e:
        click.echo(f"✗ Failed to compare runs: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def datasets():
    """Data versioning commands"""
    pass


@datasets.command('list')
def datasets_list():
    """List all datasets"""
    from src.ds.data_versioning import DataVersionControl
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    dvc = DataVersionControl(storage, db_conn)
    
    try:
        all_datasets = dvc.list_datasets()
        
        if not all_datasets:
            click.echo("No datasets found")
            return
        
        click.echo(f"\nTotal datasets: {len(all_datasets)}\n")
        
        for dataset_name, versions in all_datasets.items():
            click.echo(f"Dataset: {dataset_name}")
            click.echo(f"  Versions: {len(versions)}")
            if versions:
                latest = versions[0]
                click.echo(f"  Latest: {latest['version']} ({latest['created_at']})")
                click.echo(f"  Rows: {latest.get('num_rows', 'N/A')}, Columns: {latest.get('num_columns', 'N/A')}")
            click.echo()
        
    except Exception as e:
        click.echo(f"✗ Failed to list datasets: {str(e)}", err=True)
        raise click.Abort()


@datasets.command('versions')
@click.argument('dataset_name')
def datasets_versions(dataset_name):
    """Show versions of a dataset"""
    from src.ds.data_versioning import DataVersionControl
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    dvc = DataVersionControl(storage, db_conn)
    
    try:
        versions = dvc.list_versions(dataset_name)
        
        if not versions:
            click.echo(f"No versions found for dataset '{dataset_name}'")
            return
        
        click.echo(f"\nDataset: {dataset_name}")
        click.echo(f"Total versions: {len(versions)}\n")
        
        for version in versions:
            click.echo(f"Version: {version.version}")
            click.echo(f"  ID: {version.version_id}")
            click.echo(f"  Source: {version.source}")
            click.echo(f"  Rows: {version.num_rows}, Columns: {version.num_columns}")
            click.echo(f"  Created: {version.created_at}")
            if version.parent_version_id:
                click.echo(f"  Parent: {version.parent_version_id}")
            click.echo()
        
    except Exception as e:
        click.echo(f"✗ Failed to list versions: {str(e)}", err=True)
        raise click.Abort()


@datasets.command('lineage')
@click.argument('version_id')
@click.option('--direction', type=click.Choice(['upstream', 'downstream']), default='upstream', help='Lineage direction')
def datasets_lineage(version_id, direction):
    """Display dataset lineage"""
    from src.ds.data_versioning import DataVersionControl
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    dvc = DataVersionControl(storage, db_conn)
    
    try:
        lineage = dvc.get_lineage(version_id, direction)
        
        if not lineage:
            click.echo(f"No {direction} lineage found for version {version_id}")
            return
        
        click.echo(f"\n=== {direction.capitalize()} Lineage ===\n")
        click.echo(f"Starting from version: {version_id}\n")
        
        for i, version in enumerate(lineage, 1):
            click.echo(f"{i}. {version.dataset_name} (v{version.version})")
            click.echo(f"   ID: {version.version_id}")
            click.echo(f"   Source: {version.source}")
            click.echo(f"   Created: {version.created_at}")
            click.echo()
        
    except Exception as e:
        click.echo(f"✗ Failed to get lineage: {str(e)}", err=True)
        raise click.Abort()


@datasets.command('drift')
@click.argument('version_id1')
@click.argument('version_id2')
def datasets_drift(version_id1, version_id2):
    """Check drift between two dataset versions"""
    from src.ds.data_versioning import DataVersionControl
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    dvc = DataVersionControl(storage, db_conn)
    
    try:
        drift_report = dvc.detect_drift(version_id1, version_id2)
        
        click.echo(f"\n=== Drift Analysis ===\n")
        click.echo(f"Version 1: {version_id1}")
        click.echo(f"Version 2: {version_id2}")
        click.echo(f"\nDrift Detected: {'Yes' if drift_report.drift_detected else 'No'}")
        click.echo(f"Overall Drift Score: {drift_report.drift_score:.4f}")
        
        if drift_report.feature_drifts:
            click.echo(f"\nFeature-level Drift Scores:")
            sorted_drifts = sorted(
                drift_report.feature_drifts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, score in sorted_drifts[:10]:
                click.echo(f"  {feature}: {score:.4f}")
            
            if len(sorted_drifts) > 10:
                click.echo(f"  ... and {len(sorted_drifts) - 10} more features")
        
        if drift_report.recommendations:
            click.echo(f"\nRecommendations:")
            for rec in drift_report.recommendations:
                click.echo(f"  - {rec}")
        
    except Exception as e:
        click.echo(f"✗ Failed to check drift: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def features():
    """Feature store commands"""
    pass


@features.command('list')
def features_list():
    """List all features"""
    from src.ds.feature_store import FeatureStore
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    feature_store = FeatureStore(db_conn)
    
    try:
        all_features = feature_store.list_features()
        
        if not all_features:
            click.echo("No features found")
            return
        
        click.echo(f"\nTotal features: {len(all_features)}\n")
        
        for feature in all_features:
            click.echo(f"Feature: {feature.feature_name}")
            click.echo(f"  Type: {feature.feature_type}")
            click.echo(f"  Version: {feature.version}")
            if feature.description:
                click.echo(f"  Description: {feature.description}")
            if feature.dependencies:
                click.echo(f"  Dependencies: {', '.join(feature.dependencies)}")
            click.echo(f"  Owner: {feature.owner}")
            click.echo(f"  Created: {feature.created_at}")
            click.echo()
        
    except Exception as e:
        click.echo(f"✗ Failed to list features: {str(e)}", err=True)
        raise click.Abort()


@features.command('register')
@click.option('--name', required=True, help='Feature name')
@click.option('--type', 'feature_type', required=True, type=click.Choice(['numeric', 'categorical', 'embedding']), help='Feature type')
@click.option('--description', help='Feature description')
@click.option('--code', required=True, help='Transformation code (Python expression or function)')
@click.option('--input-schema', help='Input schema as JSON string')
@click.option('--output-schema', help='Output schema as JSON string')
@click.option('--version', default='1.0.0', help='Feature version')
@click.option('--owner', default='system', help='Feature owner')
@click.option('--dependencies', help='Comma-separated list of dependent features')
def features_register(name, feature_type, description, code, input_schema, output_schema, version, owner, dependencies):
    """Register a new feature"""
    from src.ds.feature_store import FeatureStore, FeatureDefinition
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    feature_store = FeatureStore(db_conn)
    
    try:
        # Parse schemas
        input_schema_dict = json.loads(input_schema) if input_schema else {}
        output_schema_dict = json.loads(output_schema) if output_schema else {name: feature_type}
        
        # Parse dependencies
        deps = [d.strip() for d in dependencies.split(',')] if dependencies else []
        
        # Create feature definition
        feature_def = FeatureDefinition(
            feature_name=name,
            feature_type=feature_type,
            description=description,
            transformation_code=code,
            input_schema=input_schema_dict,
            output_schema=output_schema_dict,
            version=version,
            dependencies=deps,
            owner=owner
        )
        
        feature_store.register_feature(name, feature_def)
        
        click.echo(f"✓ Feature '{name}' registered successfully")
        
    except Exception as e:
        click.echo(f"✗ Failed to register feature: {str(e)}", err=True)
        raise click.Abort()


@features.command('compute')
@click.option('--features', required=True, help='Comma-separated list of feature names')
@click.option('--input-data', required=True, type=click.Path(exists=True), help='Input data CSV file')
@click.option('--output', type=click.Path(), help='Output file path (CSV)')
def features_compute(features, input_data, output):
    """Compute features for batch data"""
    from src.ds.feature_store import FeatureStore
    import pandas as pd
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    feature_store = FeatureStore(db_conn)
    
    try:
        # Parse feature names
        feature_names = [f.strip() for f in features.split(',')]
        
        # Load input data
        click.echo(f"Loading input data from {input_data}...")
        input_df = pd.read_csv(input_data)
        click.echo(f"Loaded {len(input_df)} rows")
        
        # Compute features
        click.echo(f"Computing features: {', '.join(feature_names)}...")
        result_df = feature_store.compute_features(feature_names, input_df)
        
        click.echo(f"✓ Computed {len(result_df.columns)} features for {len(result_df)} rows")
        
        # Save output
        if output:
            result_df.to_csv(output, index=False)
            click.echo(f"✓ Results saved to {output}")
        else:
            # Display first few rows
            click.echo("\nFirst 5 rows:")
            click.echo(result_df.head().to_string())
        
    except Exception as e:
        click.echo(f"✗ Failed to compute features: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def eda():
    """Exploratory data analysis commands"""
    pass


@eda.command('analyze')
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--target', help='Target column name for supervised analysis')
@click.option('--output', type=click.Path(), help='Output directory for reports and visualizations')
@click.option('--format', type=click.Choice(['html', 'pdf']), default='html', help='Report format')
def eda_analyze(dataset, target, output, format):
    """Run exploratory data analysis on a dataset"""
    from src.ds.eda import EDAModule
    import pandas as pd
    from pathlib import Path
    
    try:
        # Load dataset
        click.echo(f"Loading dataset from {dataset}...")
        if dataset.endswith('.csv'):
            data = pd.read_csv(dataset)
        elif dataset.endswith('.json'):
            data = pd.read_json(dataset)
        else:
            click.echo("✗ Unsupported file format. Use CSV or JSON.", err=True)
            raise click.Abort()
        
        click.echo(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Run EDA
        click.echo("Running exploratory data analysis...")
        eda_module = EDAModule()
        report = eda_module.analyze_dataset(data, target_column=target)
        
        # Display summary
        click.echo(f"\n=== EDA Summary ===\n")
        click.echo(f"Dataset: {Path(dataset).name}")
        click.echo(f"Rows: {report.num_rows}")
        click.echo(f"Columns: {report.num_columns}")
        
        if report.missing_values:
            total_missing = sum(report.missing_values.values())
            click.echo(f"Missing values: {total_missing}")
        
        if report.quality_issues:
            click.echo(f"Quality issues detected: {len(report.quality_issues)}")
            for issue in report.quality_issues[:5]:
                click.echo(f"  - [{issue.severity.upper()}] {issue.description}")
        
        # Export report
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            click.echo("\nGenerating visualizations...")
            viz_paths = eda_module.generate_visualizations(data, str(output_dir))
            click.echo(f"✓ Generated {len(viz_paths)} visualizations")
            
            # Export report
            report_path = output_dir / f"eda_report.{format}"
            eda_module.export_report(report, format=format, output_path=str(report_path))
            click.echo(f"✓ Report saved to {report_path}")
        else:
            click.echo("\nUse --output to save detailed report and visualizations")
        
    except Exception as e:
        click.echo(f"✗ EDA failed: {str(e)}", err=True)
        raise click.Abort()


@cli.group()
def model_card():
    """Model card generation commands"""
    pass


@model_card.command('generate')
@click.argument('model_id')
@click.option('--run-id', help='Experiment run ID to link')
@click.option('--output', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['html', 'pdf', 'markdown']), default='html', help='Output format')
@click.option('--include-fairness/--no-fairness', default=True, help='Include fairness analysis')
def model_card_generate(model_id, run_id, output, format, include_fairness):
    """Generate model card for a trained model"""
    from src.ds.model_cards import ModelCardGenerator
    from src.ds.experiment_tracker import ExperimentTracker
    from src.ds.storage import FileSystemStorage
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    model_registry = integration.get_model_registry()
    
    storage = FileSystemStorage()
    experiment_tracker = ExperimentTracker(storage, db_conn)
    
    try:
        click.echo(f"Generating model card for model {model_id}...")
        
        generator = ModelCardGenerator(model_registry, experiment_tracker)
        model_card = generator.generate_model_card(
            model_id,
            run_id=run_id,
            include_fairness=include_fairness
        )
        
        # Display summary
        click.echo(f"\n=== Model Card Summary ===\n")
        click.echo(f"Model: {model_card.model_name}")
        click.echo(f"Type: {model_card.model_type}")
        click.echo(f"Version: {model_card.version}")
        click.echo(f"Owner: {model_card.owner}")
        
        if model_card.metrics:
            click.echo(f"\nPerformance Metrics:")
            for metric, value in list(model_card.metrics.items())[:5]:
                click.echo(f"  {metric}: {value:.4f}")
        
        if model_card.known_limitations:
            click.echo(f"\nKnown Limitations: {len(model_card.known_limitations)}")
        
        # Export model card
        if output:
            generator.export_card(model_card, format=format, output_path=output)
            click.echo(f"\n✓ Model card saved to {output}")
        else:
            # Default output path
            default_output = f"model_card_{model_id}.{format}"
            generator.export_card(model_card, format=format, output_path=default_output)
            click.echo(f"\n✓ Model card saved to {default_output}")
        
    except Exception as e:
        click.echo(f"✗ Failed to generate model card: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--n-trials', default=100, help='Number of optimization trials')
@click.option('--n-jobs', default=-1, help='Number of parallel jobs (-1 for all cores)')
@click.option('--output', type=click.Path(), help='Output directory for results')
def optimize(config_file, n_trials, n_jobs, output):
    """Run hyperparameter optimization"""
    from src.ds.hyperparameter_optimizer import HyperparameterOptimizer
    from src.ds.experiment_tracker import ExperimentTracker
    from src.ds.storage import FileSystemStorage
    from pathlib import Path
    import yaml
    
    integration = get_integration()
    db_conn = integration.get_database_connection()
    storage = FileSystemStorage()
    tracker = ExperimentTracker(storage, db_conn)
    
    try:
        # Load config
        click.echo(f"Loading configuration from {config_file}...")
        with open(config_file) as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                click.echo("✗ Config file must be JSON or YAML", err=True)
                raise click.Abort()
        
        # Extract configuration
        experiment_name = config.get('experiment_name', 'hyperparameter_optimization')
        strategy = config.get('strategy', 'bayesian')
        param_space = config.get('param_space', {})
        
        if not param_space:
            click.echo("✗ No param_space defined in config", err=True)
            raise click.Abort()
        
        click.echo(f"Experiment: {experiment_name}")
        click.echo(f"Strategy: {strategy}")
        click.echo(f"Parameters: {', '.join(param_space.keys())}")
        click.echo(f"Trials: {n_trials}")
        
        # Note: This is a simplified version. In practice, you'd need to define
        # the objective function based on the config
        click.echo("\n⚠ Note: Full optimization requires defining an objective function")
        click.echo("This command shows the structure. Implement objective function in code.")
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(tracker, strategy=strategy)
        
        click.echo(f"\n✓ Optimizer initialized with {strategy} strategy")
        click.echo("Define your objective function and call optimizer.optimize()")
        
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Results will be saved to {output_dir}")
        
    except Exception as e:
        click.echo(f"✗ Optimization setup failed: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--anonymized-id', required=True, help='Anonymized ID')
@click.option('--survey-data', type=click.Path(exists=True), help='Survey data JSON file')
@click.option('--wearable-data', type=click.Path(exists=True), help='Wearable data JSON file')
@click.option('--emr-data', type=click.Path(exists=True), help='EMR data JSON file')
def screen(anonymized_id, survey_data, wearable_data, emr_data):
    """Perform screening for an individual"""
    integration = get_integration()
    service = integration.get_screening_service()
    
    # Load data from files
    survey = None
    wearable = None
    emr = None
    
    if survey_data:
        with open(survey_data) as f:
            survey = json.load(f)
    
    if wearable_data:
        with open(wearable_data) as f:
            wearable = json.load(f)
    
    if emr_data:
        with open(emr_data) as f:
            emr = json.load(f)
    
    # Create screening request
    request = ScreeningRequest(
        anonymized_id=anonymized_id,
        survey_data=survey,
        wearable_data=wearable,
        emr_data=emr
    )
    
    click.echo(f"Screening individual {anonymized_id}...")
    
    try:
        # Run screening
        response = asyncio.run(service.screen_individual(request))
        
        # Display results
        click.echo(f"\n=== Screening Results ===\n")
        click.echo(f"Risk Score: {response.risk_score:.2f}")
        click.echo(f"Risk Level: {response.risk_level}")
        click.echo(f"Confidence: {response.confidence:.2f}")
        click.echo(f"Alert Triggered: {response.alert_triggered}")
        click.echo(f"Requires Review: {response.requires_human_review}")
        click.echo(f"Processing Time: {response.processing_time_seconds:.3f}s")
        
        if response.contributing_factors:
            click.echo(f"\nTop Contributing Factors:")
            for factor in response.contributing_factors[:5]:
                click.echo(f"  - {factor}")
        
        if response.recommendations:
            click.echo(f"\nRecommendations: {len(response.recommendations)}")
            for i, rec in enumerate(response.recommendations[:3], 1):
                click.echo(f"  {i}. {rec['name']} ({rec['resource_type']})")
        
        click.echo(f"\n✓ Screening completed successfully")
        
    except Exception as e:
        click.echo(f"✗ Screening failed: {str(e)}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()
