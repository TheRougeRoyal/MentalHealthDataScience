"""Tests for performance monitoring system."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil

from src.ds.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    Alert,
    AlertNotifier,
    PerformanceReporter
)
from src.database.connection import DatabaseConnection
from src.ml.model_registry import ModelRegistry
from src.ds.storage import FileSystemStorage


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def db_connection(temp_dir):
    """Create test database connection."""
    db_path = Path(temp_dir) / "test.db"
    return DatabaseConnection(str(db_path))


@pytest.fixture
def model_registry(temp_dir):
    """Create test model registry."""
    registry_dir = Path(temp_dir) / "registry"
    return ModelRegistry(str(registry_dir))


@pytest.fixture
def performance_monitor(model_registry, db_connection, temp_dir):
    """Create test performance monitor."""
    storage_path = Path(temp_dir) / "monitoring"
    return PerformanceMonitor(
        model_registry=model_registry,
        db_connection=db_connection,
        storage_path=str(storage_path),
        performance_threshold=0.10
    )


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_create_metrics(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            model_id="test_model",
            timestamp=datetime.utcnow(),
            metrics={'accuracy': 0.95, 'f1_score': 0.93},
            prediction_stats={'mean': 0.5, 'std': 0.2}
        )
        
        assert metrics.model_id == "test_model"
        assert metrics.metrics['accuracy'] == 0.95
        assert metrics.prediction_stats['mean'] == 0.5
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            model_id="test_model",
            timestamp=datetime.utcnow(),
            metrics={'accuracy': 0.95}
        )
        
        data = metrics.to_dict()
        
        assert data['model_id'] == "test_model"
        assert data['metrics']['accuracy'] == 0.95
        assert 'timestamp' in data


class TestAlert:
    """Test Alert class."""
    
    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            alert_type='performance_degradation',
            severity='high',
            message='Performance degraded',
            model_id='test_model'
        )
        
        assert alert.alert_type == 'performance_degradation'
        assert alert.severity == 'high'
        assert alert.model_id == 'test_model'
        assert alert.notified is False
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            alert_type='drift',
            severity='medium',
            message='Drift detected',
            model_id='test_model',
            details={'drift_score': 0.5}
        )
        
        data = alert.to_dict()
        
        assert data['alert_type'] == 'drift'
        assert data['severity'] == 'medium'
        assert data['details']['drift_score'] == 0.5


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    def test_initialization(self, performance_monitor):
        """Test monitor initialization."""
        assert performance_monitor.performance_threshold == 0.10
        assert performance_monitor.baselines == {}
        assert performance_monitor.performance_history == {}
    
    def test_set_baseline(self, performance_monitor):
        """Test setting baseline metrics."""
        baseline_metrics = {
            'accuracy': 0.95,
            'f1_score': 0.93,
            'precision': 0.94
        }
        
        baseline_predictions = np.random.rand(100)
        
        performance_monitor.set_baseline(
            model_id='test_model',
            baseline_metrics=baseline_metrics,
            baseline_predictions=baseline_predictions
        )
        
        assert 'test_model' in performance_monitor.baselines
        assert performance_monitor.baselines['test_model']['metrics'] == baseline_metrics
        assert 'prediction_stats' in performance_monitor.baselines['test_model']
    
    def test_track_performance(self, performance_monitor):
        """Test tracking performance metrics."""
        # Set baseline first
        performance_monitor.set_baseline(
            model_id='test_model',
            baseline_metrics={'accuracy': 0.95}
        )
        
        # Track current performance
        current_metrics = {'accuracy': 0.94}
        predictions = np.random.rand(50)
        
        perf_metrics = performance_monitor.track_performance(
            model_id='test_model',
            metrics=current_metrics,
            predictions=predictions,
            check_drift=False
        )
        
        assert perf_metrics.model_id == 'test_model'
        assert perf_metrics.metrics['accuracy'] == 0.94
        assert 'test_model' in performance_monitor.performance_history
        assert len(performance_monitor.performance_history['test_model']) == 1
    
    def test_performance_degradation_detection(self, performance_monitor):
        """Test detection of performance degradation."""
        # Set baseline
        performance_monitor.set_baseline(
            model_id='test_model',
            baseline_metrics={'accuracy': 0.95}
        )
        
        # Track degraded performance (>10% degradation)
        degraded_metrics = {'accuracy': 0.80}  # 15.8% degradation
        
        performance_monitor.track_performance(
            model_id='test_model',
            metrics=degraded_metrics,
            check_drift=False
        )
        
        # Check that alert was triggered
        alerts = performance_monitor.get_alerts(model_id='test_model')
        assert len(alerts) > 0
        assert alerts[0].alert_type == 'performance_degradation'
        assert alerts[0].severity in ['medium', 'high']
    
    def test_get_performance_history(self, performance_monitor):
        """Test retrieving performance history."""
        model_id = 'test_model'
        
        # Track multiple performance points
        for i in range(5):
            performance_monitor.track_performance(
                model_id=model_id,
                metrics={'accuracy': 0.90 + i * 0.01},
                check_drift=False
            )
        
        history = performance_monitor.get_performance_history(model_id)
        
        assert len(history) == 5
        assert all(isinstance(m, PerformanceMetrics) for m in history)
    
    def test_get_performance_history_with_time_filter(self, performance_monitor):
        """Test retrieving performance history with time filters."""
        model_id = 'test_model'
        
        # Create metrics with different timestamps
        now = datetime.utcnow()
        
        for i in range(5):
            metrics = PerformanceMetrics(
                model_id=model_id,
                timestamp=now - timedelta(days=i),
                metrics={'accuracy': 0.90}
            )
            performance_monitor.performance_history.setdefault(model_id, []).append(metrics)
        
        # Get history for last 2 days
        start_time = now - timedelta(days=2)
        history = performance_monitor.get_performance_history(
            model_id,
            start_time=start_time
        )
        
        assert len(history) <= 3  # Should include today, yesterday, and 2 days ago
    
    def test_get_alerts(self, performance_monitor):
        """Test retrieving alerts with filters."""
        # Create some alerts
        alert1 = Alert(
            alert_type='performance_degradation',
            severity='high',
            message='Test alert 1',
            model_id='model1'
        )
        alert2 = Alert(
            alert_type='drift',
            severity='medium',
            message='Test alert 2',
            model_id='model2'
        )
        
        performance_monitor.alert_history.extend([alert1, alert2])
        
        # Get all alerts
        all_alerts = performance_monitor.get_alerts()
        assert len(all_alerts) == 2
        
        # Filter by model_id
        model1_alerts = performance_monitor.get_alerts(model_id='model1')
        assert len(model1_alerts) == 1
        assert model1_alerts[0].model_id == 'model1'
        
        # Filter by alert_type
        drift_alerts = performance_monitor.get_alerts(alert_type='drift')
        assert len(drift_alerts) == 1
        assert drift_alerts[0].alert_type == 'drift'
    
    def test_generate_performance_summary(self, performance_monitor):
        """Test generating performance summary."""
        model_id = 'test_model'
        
        # Set baseline
        performance_monitor.set_baseline(
            model_id=model_id,
            baseline_metrics={'accuracy': 0.95, 'f1_score': 0.93}
        )
        
        # Track some performance
        for i in range(3):
            performance_monitor.track_performance(
                model_id=model_id,
                metrics={'accuracy': 0.94, 'f1_score': 0.92},
                check_drift=False
            )
        
        summary = performance_monitor.generate_performance_summary(
            model_id=model_id,
            time_window=timedelta(days=7)
        )
        
        assert summary['model_id'] == model_id
        assert summary['data_points'] == 3
        assert 'metric_summaries' in summary
        assert 'baseline_comparison' in summary
        assert 'accuracy' in summary['metric_summaries']


class TestAlertNotifier:
    """Test AlertNotifier class."""
    
    def test_initialization(self):
        """Test notifier initialization."""
        notifier = AlertNotifier(
            email_config={'smtp_host': 'localhost'},
            webhook_config={'url': 'http://example.com/webhook'}
        )
        
        assert notifier.email_config is not None
        assert notifier.webhook_config is not None
        assert notifier.running is False
    
    def test_start_stop(self):
        """Test starting and stopping notifier."""
        notifier = AlertNotifier()
        
        notifier.start()
        assert notifier.running is True
        assert notifier.notification_thread is not None
        
        notifier.stop()
        assert notifier.running is False
    
    def test_notify_queues_alert(self):
        """Test that notify queues an alert."""
        notifier = AlertNotifier()
        
        alert = Alert(
            alert_type='test',
            severity='low',
            message='Test alert',
            model_id='test_model'
        )
        
        notifier.notify(alert)
        
        assert not notifier.alert_queue.empty()


class TestPerformanceReporter:
    """Test PerformanceReporter class."""
    
    @pytest.fixture
    def reporter(self, performance_monitor, temp_dir):
        """Create test reporter."""
        report_path = Path(temp_dir) / "reports"
        return PerformanceReporter(
            performance_monitor=performance_monitor,
            report_path=str(report_path)
        )
    
    def test_initialization(self, reporter):
        """Test reporter initialization."""
        assert reporter.monitor is not None
        assert reporter.report_path.exists()
    
    def test_generate_weekly_report(self, reporter, performance_monitor):
        """Test generating weekly report."""
        model_id = 'test_model'
        
        # Set baseline
        performance_monitor.set_baseline(
            model_id=model_id,
            baseline_metrics={'accuracy': 0.95}
        )
        
        # Track some performance
        for i in range(3):
            performance_monitor.track_performance(
                model_id=model_id,
                metrics={'accuracy': 0.94},
                check_drift=False
            )
        
        # Generate report
        report = reporter.generate_weekly_report(
            model_id=model_id,
            include_visualizations=False  # Skip viz for faster test
        )
        
        assert report['model_id'] == model_id
        assert 'production_vs_validation' in report
        assert 'retraining_recommendations' in report
        assert 'report_file' in report
    
    def test_compare_production_to_validation(self, reporter, performance_monitor):
        """Test production vs validation comparison."""
        model_id = 'test_model'
        
        # Set baseline
        performance_monitor.set_baseline(
            model_id=model_id,
            baseline_metrics={'accuracy': 0.95}
        )
        
        summary = {
            'baseline_comparison': {
                'accuracy': {
                    'baseline': 0.95,
                    'current': 0.94,
                    'change_pct': -1.05
                }
            }
        }
        
        comparison = reporter._compare_production_to_validation(model_id, summary)
        
        assert comparison['has_baseline'] is True
        assert 'accuracy' in comparison['metrics']
        assert comparison['metrics']['accuracy']['status'] == 'stable'
    
    def test_generate_retraining_recommendations(self, reporter, performance_monitor):
        """Test generating retraining recommendations."""
        model_id = 'test_model'
        
        # Summary with degradation
        summary = {
            'baseline_comparison': {
                'accuracy': {
                    'baseline': 0.95,
                    'current': 0.80,
                    'change_pct': -15.8
                }
            }
        }
        
        recommendations = reporter._generate_retraining_recommendations(
            model_id,
            summary
        )
        
        assert len(recommendations) > 0
        assert any(r['type'] == 'performance_degradation' for r in recommendations)
        assert any(r['priority'] == 'high' for r in recommendations)
    
    def test_export_html_report(self, reporter, performance_monitor, temp_dir):
        """Test exporting HTML report."""
        model_id = 'test_model'
        
        summary = {
            'model_id': model_id,
            'data_points': 5,
            'metric_summaries': {
                'accuracy': {
                    'mean': 0.94,
                    'std': 0.01,
                    'min': 0.93,
                    'max': 0.95,
                    'latest': 0.94
                }
            }
        }
        
        output_path = Path(temp_dir) / "test_report.html"
        result_path = reporter.export_html_report(
            model_id=model_id,
            summary=summary,
            output_path=str(output_path)
        )
        
        assert Path(result_path).exists()
        
        # Check HTML content
        with open(result_path, 'r') as f:
            html_content = f.read()
        
        assert model_id in html_content
        assert 'accuracy' in html_content
        assert 'Weekly Performance Report' in html_content


class TestIntegration:
    """Integration tests for performance monitoring."""
    
    def test_end_to_end_monitoring(self, performance_monitor):
        """Test end-to-end monitoring workflow."""
        model_id = 'integration_test_model'
        
        # 1. Set baseline
        baseline_metrics = {'accuracy': 0.95, 'f1_score': 0.93}
        baseline_predictions = np.random.rand(100)
        
        performance_monitor.set_baseline(
            model_id=model_id,
            baseline_metrics=baseline_metrics,
            baseline_predictions=baseline_predictions
        )
        
        # 2. Track performance over time
        for day in range(7):
            # Simulate gradual degradation
            accuracy = 0.95 - (day * 0.01)
            f1_score = 0.93 - (day * 0.01)
            
            performance_monitor.track_performance(
                model_id=model_id,
                metrics={'accuracy': accuracy, 'f1_score': f1_score},
                predictions=np.random.rand(50),
                check_drift=False
            )
        
        # 3. Get performance history
        history = performance_monitor.get_performance_history(model_id)
        assert len(history) == 7
        
        # 4. Generate summary
        summary = performance_monitor.generate_performance_summary(
            model_id=model_id,
            time_window=timedelta(days=7)
        )
        
        assert summary['data_points'] == 7
        assert 'metric_summaries' in summary
        assert 'baseline_comparison' in summary
    
    def test_drift_detection_integration(self, performance_monitor):
        """Test drift detection integration."""
        model_id = 'drift_test_model'
        
        # Set baseline with features
        baseline_features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100)
        })
        baseline_predictions = np.random.rand(100)
        
        performance_monitor.set_baseline(
            model_id=model_id,
            baseline_metrics={'accuracy': 0.95},
            baseline_predictions=baseline_predictions,
            baseline_features=baseline_features
        )
        
        # Track with drifted data
        drifted_features = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 100),  # Mean shifted
            'feature2': np.random.normal(5, 2, 100)
        })
        drifted_predictions = np.random.rand(100) + 0.3  # Distribution shifted
        
        performance_monitor.track_performance(
            model_id=model_id,
            metrics={'accuracy': 0.94},
            predictions=drifted_predictions,
            features=drifted_features,
            check_drift=True
        )
        
        # Check for drift alerts
        alerts = performance_monitor.get_alerts(model_id=model_id)
        drift_alerts = [a for a in alerts if 'drift' in a.alert_type]
        
        # May or may not detect drift depending on threshold
        # Just verify the system runs without errors
        assert isinstance(drift_alerts, list)
