"""
Performance monitoring system for deployed models.

This module provides continuous monitoring of model performance metrics,
prediction distributions, and drift detection with alerting capabilities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID
from pathlib import Path
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import queue
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.database.connection import DatabaseConnection
from src.governance.drift_monitor import DriftMonitor, DriftReport
from src.ml.model_registry import ModelRegistry
from src.logging_config import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(
        self,
        model_id: str,
        timestamp: datetime,
        metrics: Dict[str, float],
        prediction_stats: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance metrics.
        
        Args:
            model_id: Model identifier
            timestamp: Timestamp of metrics
            metrics: Dictionary of metric names to values
            prediction_stats: Optional prediction distribution statistics
        """
        self.model_id = model_id
        self.timestamp = timestamp
        self.metrics = metrics
        self.prediction_stats = prediction_stats or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'prediction_stats': self.prediction_stats
        }


class Alert:
    """Performance alert."""
    
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        model_id: str,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
        alert_id: Optional[str] = None
    ):
        """
        Initialize alert.
        
        Args:
            alert_type: Type of alert (performance_degradation, drift, etc.)
            severity: Severity level (low, medium, high, critical)
            message: Alert message
            model_id: Model identifier
            timestamp: Alert timestamp
            details: Additional alert details
            alert_id: Optional unique alert identifier
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.model_id = model_id
        self.timestamp = timestamp or datetime.utcnow()
        self.details = details or {}
        self.alert_id = alert_id or f"{model_id}_{self.timestamp.timestamp()}"
        self.notified = False
        self.notification_attempts = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'notified': self.notified,
            'notification_attempts': self.notification_attempts
        }


class PerformanceMonitor:
    """
    Continuous performance monitoring for deployed models.
    
    Tracks model performance metrics, prediction distributions,
    and detects drift with automated alerting.
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        db_connection: DatabaseConnection,
        storage_path: str = "monitoring/performance",
        performance_threshold: float = 0.10,  # 10% degradation
        alert_callbacks: Optional[List[Callable]] = None,
        alert_notifier: Optional['AlertNotifier'] = None
    ):
        """
        Initialize PerformanceMonitor.
        
        Args:
            model_registry: Model registry instance
            db_connection: Database connection
            storage_path: Path for storing monitoring data
            performance_threshold: Threshold for performance degradation (0.10 = 10%)
            alert_callbacks: Optional list of alert callback functions
            alert_notifier: Optional AlertNotifier for sending notifications
        """
        self.model_registry = model_registry
        self.db = db_connection
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.performance_threshold = performance_threshold
        self.alert_callbacks = alert_callbacks or []
        self.alert_notifier = alert_notifier
        
        # Initialize drift monitor
        self.drift_monitor = DriftMonitor(
            feature_drift_threshold=0.3,
            prediction_drift_threshold=0.2,
            alert_callback=self._handle_drift_alert
        )
        
        # Storage for baseline metrics and predictions
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.alert_history: List[Alert] = []
        
        # Load existing baselines
        self._load_baselines()
        
        # Start alert notifier if provided
        if self.alert_notifier:
            self.alert_notifier.start()
        
        logger.info(
            f"PerformanceMonitor initialized with threshold={performance_threshold}"
        )
    
    def set_baseline(
        self,
        model_id: str,
        baseline_metrics: Dict[str, float],
        baseline_predictions: Optional[np.ndarray] = None,
        baseline_features: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Set baseline performance metrics for a model.
        
        Args:
            model_id: Model identifier
            baseline_metrics: Baseline performance metrics (e.g., accuracy, f1_score)
            baseline_predictions: Optional baseline prediction distribution
            baseline_features: Optional baseline feature distribution
        """
        logger.info(f"Setting baseline for model {model_id}")
        
        baseline_data = {
            'model_id': model_id,
            'metrics': baseline_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store baseline predictions
        if baseline_predictions is not None:
            baseline_data['prediction_stats'] = {
                'mean': float(np.mean(baseline_predictions)),
                'std': float(np.std(baseline_predictions)),
                'min': float(np.min(baseline_predictions)),
                'max': float(np.max(baseline_predictions)),
                'median': float(np.median(baseline_predictions))
            }
            
            # Set reference for drift monitor
            self.drift_monitor.set_reference_data(
                features=baseline_features if baseline_features is not None else pd.DataFrame(),
                predictions=baseline_predictions
            )
        
        self.baselines[model_id] = baseline_data
        self._save_baselines()
        
        logger.info(f"Baseline set for model {model_id}: {baseline_metrics}")
    
    def track_performance(
        self,
        model_id: str,
        metrics: Dict[str, float],
        predictions: Optional[np.ndarray] = None,
        features: Optional[pd.DataFrame] = None,
        check_drift: bool = True
    ) -> PerformanceMetrics:
        """
        Track current performance metrics for a model.
        
        Args:
            model_id: Model identifier
            metrics: Current performance metrics
            predictions: Optional current predictions for distribution tracking
            features: Optional current features for drift detection
            check_drift: Whether to check for drift
        
        Returns:
            PerformanceMetrics instance
        """
        logger.debug(f"Tracking performance for model {model_id}")
        
        # Calculate prediction statistics
        prediction_stats = None
        if predictions is not None:
            prediction_stats = {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions))
            }
        
        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.utcnow(),
            metrics=metrics,
            prediction_stats=prediction_stats
        )
        
        # Store in history
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        self.performance_history[model_id].append(perf_metrics)
        
        # Check for performance degradation
        if model_id in self.baselines:
            self._check_performance_degradation(model_id, perf_metrics)
        
        # Check for drift
        if check_drift:
            if predictions is not None and self.drift_monitor.reference_predictions is not None:
                drift_report = self.drift_monitor.detect_prediction_drift(predictions)
                if drift_report.drift_detected:
                    logger.warning(f"Prediction drift detected for model {model_id}")
            
            if features is not None and self.drift_monitor.reference_features is not None:
                drift_report = self.drift_monitor.detect_feature_drift(features)
                if drift_report.drift_detected:
                    logger.warning(f"Feature drift detected for model {model_id}")
        
        return perf_metrics
    
    def _check_performance_degradation(
        self,
        model_id: str,
        current_metrics: PerformanceMetrics
    ) -> None:
        """
        Check if performance has degraded beyond threshold.
        
        Args:
            model_id: Model identifier
            current_metrics: Current performance metrics
        """
        baseline = self.baselines[model_id]
        baseline_metrics = baseline['metrics']
        
        # Check each metric
        for metric_name, current_value in current_metrics.metrics.items():
            if metric_name not in baseline_metrics:
                continue
            
            baseline_value = baseline_metrics[metric_name]
            
            # Calculate relative change (assuming higher is better)
            if baseline_value > 0:
                relative_change = (current_value - baseline_value) / baseline_value
            else:
                relative_change = 0.0
            
            # Check if degradation exceeds threshold
            if relative_change < -self.performance_threshold:
                degradation_pct = abs(relative_change) * 100
                
                alert = Alert(
                    alert_type='performance_degradation',
                    severity='high' if degradation_pct > 20 else 'medium',
                    message=(
                        f"Performance degradation detected for model {model_id}: "
                        f"{metric_name} decreased by {degradation_pct:.1f}% "
                        f"(baseline: {baseline_value:.4f}, current: {current_value:.4f})"
                    ),
                    model_id=model_id,
                    details={
                        'metric_name': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'degradation_pct': degradation_pct
                    }
                )
                
                self._trigger_alert(alert)
    
    def _handle_drift_alert(
        self,
        drift_report: DriftReport,
        drift_type: str
    ) -> None:
        """
        Handle drift alerts from drift monitor.
        
        Args:
            drift_report: Drift report
            drift_type: Type of drift (feature or prediction)
        """
        # Extract model_id from context (simplified - in production would track this)
        model_id = "current_model"
        
        alert = Alert(
            alert_type=f'{drift_type}_drift',
            severity='medium',
            message=(
                f"{drift_type.capitalize()} drift detected: "
                f"score={drift_report.drift_score:.4f}, "
                f"threshold={drift_report.threshold:.4f}"
            ),
            model_id=model_id,
            details={
                'drift_type': drift_type,
                'drift_score': drift_report.drift_score,
                'threshold': drift_report.threshold,
                'feature_drifts': drift_report.feature_drifts
            }
        )
        
        self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Alert) -> None:
        """
        Trigger an alert and notify via callbacks.
        
        Ensures alerts are sent within 5 minutes of detection.
        
        Args:
            alert: Alert instance
        """
        logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Store alert
        self.alert_history.append(alert)
        self._save_alert(alert)
        
        # Send notification via AlertNotifier
        if self.alert_notifier:
            self.alert_notifier.notify(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_performance_history(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceMetrics]:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
        
        Returns:
            List of PerformanceMetrics
        """
        if model_id not in self.performance_history:
            return []
        
        history = self.performance_history[model_id]
        
        # Apply time filters
        if start_time or end_time:
            filtered = []
            for metrics in history:
                if start_time and metrics.timestamp < start_time:
                    continue
                if end_time and metrics.timestamp > end_time:
                    continue
                filtered.append(metrics)
            return filtered
        
        return history
    
    def get_alerts(
        self,
        model_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Get alerts with optional filters.
        
        Args:
            model_id: Optional model ID filter
            alert_type: Optional alert type filter
            severity: Optional severity filter
            start_time: Optional start time filter
        
        Returns:
            List of Alert instances
        """
        filtered_alerts = []
        
        for alert in self.alert_history:
            if model_id and alert.model_id != model_id:
                continue
            if alert_type and alert.alert_type != alert_type:
                continue
            if severity and alert.severity != severity:
                continue
            if start_time and alert.timestamp < start_time:
                continue
            
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def _load_baselines(self) -> None:
        """Load baselines from storage."""
        baseline_file = self.storage_path / "baselines.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    self.baselines = json.load(f)
                logger.info(f"Loaded {len(self.baselines)} baselines")
            except Exception as e:
                logger.error(f"Error loading baselines: {e}")
    
    def _save_baselines(self) -> None:
        """Save baselines to storage."""
        baseline_file = self.storage_path / "baselines.json"
        
        try:
            with open(baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
            logger.debug("Baselines saved")
        except Exception as e:
            logger.error(f"Error saving baselines: {e}")
    
    def _save_alert(self, alert: Alert) -> None:
        """
        Save alert to storage.
        
        Args:
            alert: Alert instance
        """
        alerts_file = self.storage_path / "alerts.jsonl"
        
        try:
            with open(alerts_file, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    def generate_performance_summary(
        self,
        model_id: str,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Generate performance summary for a model.
        
        Args:
            model_id: Model identifier
            time_window: Time window for summary
        
        Returns:
            Dictionary with performance summary
        """
        logger.info(f"Generating performance summary for model {model_id}")
        
        # Get recent performance history
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        history = self.get_performance_history(model_id, start_time, end_time)
        
        if not history:
            return {
                'model_id': model_id,
                'time_window': str(time_window),
                'data_points': 0,
                'message': 'No performance data available'
            }
        
        # Calculate summary statistics
        metric_names = set()
        for metrics in history:
            metric_names.update(metrics.metrics.keys())
        
        metric_summaries = {}
        for metric_name in metric_names:
            values = [
                m.metrics[metric_name]
                for m in history
                if metric_name in m.metrics
            ]
            
            if values:
                metric_summaries[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': values[-1]
                }
        
        # Get baseline comparison
        baseline_comparison = {}
        if model_id in self.baselines:
            baseline_metrics = self.baselines[model_id]['metrics']
            for metric_name, summary in metric_summaries.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    current_value = summary['latest']
                    change_pct = ((current_value - baseline_value) / baseline_value * 100
                                  if baseline_value > 0 else 0.0)
                    
                    baseline_comparison[metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': change_pct
                    }
        
        # Get recent alerts
        recent_alerts = self.get_alerts(
            model_id=model_id,
            start_time=start_time
        )
        
        summary = {
            'model_id': model_id,
            'time_window': str(time_window),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'data_points': len(history),
            'metric_summaries': metric_summaries,
            'baseline_comparison': baseline_comparison,
            'alert_count': len(recent_alerts),
            'alerts_by_type': self._count_alerts_by_type(recent_alerts),
            'alerts_by_severity': self._count_alerts_by_severity(recent_alerts)
        }
        
        return summary
    
    def _count_alerts_by_type(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by type."""
        counts = {}
        for alert in alerts:
            counts[alert.alert_type] = counts.get(alert.alert_type, 0) + 1
        return counts
    
    def _count_alerts_by_severity(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by severity."""
        counts = {}
        for alert in alerts:
            counts[alert.severity] = counts.get(alert.severity, 0) + 1
        return counts


class AlertNotifier:
    """
    Alert notification system with email and webhook support.
    
    Sends alerts within 5 minutes of detection with retry logic.
    """
    
    def __init__(
        self,
        email_config: Optional[Dict[str, Any]] = None,
        webhook_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: int = 60  # seconds
    ):
        """
        Initialize AlertNotifier.
        
        Args:
            email_config: Email configuration (smtp_host, smtp_port, username, password, recipients)
            webhook_config: Webhook configuration (url, headers, method)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.email_config = email_config
        self.webhook_config = webhook_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Alert queue for async processing
        self.alert_queue: queue.Queue = queue.Queue()
        self.notification_thread: Optional[threading.Thread] = None
        self.running = False
        
        logger.info("AlertNotifier initialized")
    
    def start(self) -> None:
        """Start the notification worker thread."""
        if self.running:
            logger.warning("AlertNotifier already running")
            return
        
        self.running = True
        self.notification_thread = threading.Thread(
            target=self._notification_worker,
            daemon=True
        )
        self.notification_thread.start()
        logger.info("AlertNotifier worker started")
    
    def stop(self) -> None:
        """Stop the notification worker thread."""
        self.running = False
        if self.notification_thread:
            self.notification_thread.join(timeout=10)
        logger.info("AlertNotifier worker stopped")
    
    def notify(self, alert: Alert) -> None:
        """
        Queue an alert for notification.
        
        Args:
            alert: Alert to notify
        """
        self.alert_queue.put(alert)
        logger.debug(f"Alert {alert.alert_id} queued for notification")
    
    def _notification_worker(self) -> None:
        """Worker thread that processes alert notifications."""
        logger.info("Notification worker thread started")
        
        while self.running:
            try:
                # Get alert from queue with timeout
                alert = self.alert_queue.get(timeout=1)
                
                # Process notification
                self._process_alert(alert)
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
    
    def _process_alert(self, alert: Alert) -> None:
        """
        Process and send alert notification.
        
        Args:
            alert: Alert to process
        """
        logger.info(f"Processing alert {alert.alert_id}")
        
        success = False
        attempts = 0
        
        while attempts < self.max_retries and not success:
            attempts += 1
            alert.notification_attempts = attempts
            
            try:
                # Send via email
                if self.email_config:
                    self._send_email(alert)
                
                # Send via webhook
                if self.webhook_config:
                    self._send_webhook(alert)
                
                success = True
                alert.notified = True
                logger.info(f"Alert {alert.alert_id} notified successfully")
                
            except Exception as e:
                logger.error(
                    f"Failed to send alert {alert.alert_id} "
                    f"(attempt {attempts}/{self.max_retries}): {e}"
                )
                
                if attempts < self.max_retries:
                    # Wait before retry
                    import time
                    time.sleep(self.retry_delay)
        
        if not success:
            logger.error(
                f"Failed to notify alert {alert.alert_id} "
                f"after {self.max_retries} attempts"
            )
    
    def _send_email(self, alert: Alert) -> None:
        """
        Send alert via email.
        
        Args:
            alert: Alert to send
        """
        if not self.email_config:
            return
        
        smtp_host = self.email_config.get('smtp_host')
        smtp_port = self.email_config.get('smtp_port', 587)
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        recipients = self.email_config.get('recipients', [])
        
        if not all([smtp_host, username, password, recipients]):
            logger.warning("Incomplete email configuration, skipping email notification")
            return
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[{alert.severity.upper()}] Model Alert: {alert.alert_type}"
        
        # Email body
        body = f"""
Model Performance Alert

Alert ID: {alert.alert_id}
Model ID: {alert.model_id}
Alert Type: {alert.alert_type}
Severity: {alert.severity}
Timestamp: {alert.timestamp.isoformat()}

Message:
{alert.message}

Details:
{json.dumps(alert.details, indent=2)}

---
This is an automated alert from the Model Performance Monitoring System.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        
        logger.info(f"Email sent for alert {alert.alert_id}")
    
    def _send_webhook(self, alert: Alert) -> None:
        """
        Send alert via webhook.
        
        Args:
            alert: Alert to send
        """
        if not self.webhook_config:
            return
        
        url = self.webhook_config.get('url')
        headers = self.webhook_config.get('headers', {})
        method = self.webhook_config.get('method', 'POST').upper()
        
        if not url:
            logger.warning("No webhook URL configured, skipping webhook notification")
            return
        
        # Prepare payload
        payload = alert.to_dict()
        
        # Send webhook
        if method == 'POST':
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )
        elif method == 'PUT':
            response = requests.put(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )
        else:
            raise ValueError(f"Unsupported webhook method: {method}")
        
        response.raise_for_status()
        
        logger.info(f"Webhook sent for alert {alert.alert_id}")
    
    def send_immediate(self, alert: Alert) -> bool:
        """
        Send alert immediately (synchronous).
        
        Args:
            alert: Alert to send
        
        Returns:
            True if notification succeeded, False otherwise
        """
        try:
            self._process_alert(alert)
            return alert.notified
        except Exception as e:
            logger.error(f"Failed to send immediate alert: {e}")
            return False


class PerformanceReporter:
    """
    Automated performance reporting system.
    
    Generates weekly performance summary reports with visualizations
    and retraining recommendations.
    """
    
    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        report_path: str = "monitoring/reports"
    ):
        """
        Initialize PerformanceReporter.
        
        Args:
            performance_monitor: PerformanceMonitor instance
            report_path: Path for storing reports
        """
        self.monitor = performance_monitor
        self.report_path = Path(report_path)
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("PerformanceReporter initialized")
    
    def generate_weekly_report(
        self,
        model_id: str,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate weekly performance summary report.
        
        Args:
            model_id: Model identifier
            include_visualizations: Whether to generate visualizations
        
        Returns:
            Dictionary with report data
        """
        logger.info(f"Generating weekly report for model {model_id}")
        
        # Get performance summary
        summary = self.monitor.generate_performance_summary(
            model_id=model_id,
            time_window=timedelta(days=7)
        )
        
        # Add production vs validation comparison
        comparison = self._compare_production_to_validation(model_id, summary)
        summary['production_vs_validation'] = comparison
        
        # Add retraining recommendations
        recommendations = self._generate_retraining_recommendations(
            model_id,
            summary
        )
        summary['retraining_recommendations'] = recommendations
        
        # Generate visualizations
        if include_visualizations:
            viz_paths = self._generate_visualizations(model_id, summary)
            summary['visualizations'] = viz_paths
        
        # Save report
        report_file = self._save_report(model_id, summary)
        summary['report_file'] = str(report_file)
        
        logger.info(f"Weekly report generated: {report_file}")
        
        return summary
    
    def _compare_production_to_validation(
        self,
        model_id: str,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare production metrics to validation baseline.
        
        Args:
            model_id: Model identifier
            summary: Performance summary
        
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'has_baseline': False,
            'metrics': {}
        }
        
        if model_id not in self.monitor.baselines:
            return comparison
        
        comparison['has_baseline'] = True
        baseline = self.monitor.baselines[model_id]
        
        # Compare each metric
        if 'baseline_comparison' in summary:
            for metric_name, comp_data in summary['baseline_comparison'].items():
                baseline_value = comp_data['baseline']
                current_value = comp_data['current']
                change_pct = comp_data['change_pct']
                
                # Determine status
                if abs(change_pct) < 5:
                    status = 'stable'
                elif change_pct > 0:
                    status = 'improved'
                else:
                    status = 'degraded'
                
                comparison['metrics'][metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change_pct,
                    'status': status
                }
        
        return comparison
    
    def _generate_retraining_recommendations(
        self,
        model_id: str,
        summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate retraining recommendations based on performance and drift.
        
        Args:
            model_id: Model identifier
            summary: Performance summary
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check for performance degradation
        if 'baseline_comparison' in summary:
            for metric_name, comp_data in summary['baseline_comparison'].items():
                change_pct = comp_data['change_pct']
                
                if change_pct < -10:  # More than 10% degradation
                    recommendations.append({
                        'type': 'performance_degradation',
                        'priority': 'high',
                        'reason': f"{metric_name} degraded by {abs(change_pct):.1f}%",
                        'action': 'Consider retraining with recent data'
                    })
        
        # Check for drift alerts
        recent_alerts = self.monitor.get_alerts(
            model_id=model_id,
            start_time=datetime.utcnow() - timedelta(days=7)
        )
        
        drift_alerts = [a for a in recent_alerts if 'drift' in a.alert_type]
        
        if drift_alerts:
            recommendations.append({
                'type': 'data_drift',
                'priority': 'medium',
                'reason': f"{len(drift_alerts)} drift alerts in past week",
                'action': 'Investigate data distribution changes and retrain if necessary'
            })
        
        # Check alert frequency
        if len(recent_alerts) > 10:
            recommendations.append({
                'type': 'high_alert_frequency',
                'priority': 'medium',
                'reason': f"{len(recent_alerts)} alerts in past week",
                'action': 'Review model stability and consider retraining'
            })
        
        # If no issues, recommend periodic retraining
        if not recommendations:
            recommendations.append({
                'type': 'periodic_maintenance',
                'priority': 'low',
                'reason': 'No critical issues detected',
                'action': 'Continue monitoring; consider retraining quarterly'
            })
        
        return recommendations
    
    def _generate_visualizations(
        self,
        model_id: str,
        summary: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate performance trend visualizations.
        
        Args:
            model_id: Model identifier
            summary: Performance summary
        
        Returns:
            Dictionary of visualization names to file paths
        """
        viz_dir = self.report_path / model_id / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        viz_paths = {}
        
        # Get performance history
        history = self.monitor.get_performance_history(
            model_id=model_id,
            start_time=datetime.utcnow() - timedelta(days=7)
        )
        
        if not history:
            logger.warning(f"No performance history for model {model_id}")
            return viz_paths
        
        # Extract time series data
        timestamps = [m.timestamp for m in history]
        
        # Plot metric trends
        if 'metric_summaries' in summary:
            for metric_name in summary['metric_summaries'].keys():
                values = [
                    m.metrics.get(metric_name)
                    for m in history
                    if metric_name in m.metrics
                ]
                
                if values:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(timestamps[:len(values)], values, marker='o', linewidth=2)
                    
                    # Add baseline if available
                    if model_id in self.monitor.baselines:
                        baseline_value = self.monitor.baselines[model_id]['metrics'].get(metric_name)
                        if baseline_value:
                            ax.axhline(
                                y=baseline_value,
                                color='r',
                                linestyle='--',
                                label='Baseline'
                            )
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel(metric_name)
                    ax.set_title(f'{metric_name} Over Time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    viz_path = viz_dir / f"{metric_name}_trend.png"
                    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    viz_paths[f"{metric_name}_trend"] = str(viz_path)
        
        # Plot prediction distribution if available
        prediction_means = [
            m.prediction_stats.get('mean')
            for m in history
            if m.prediction_stats and 'mean' in m.prediction_stats
        ]
        
        if prediction_means:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                timestamps[:len(prediction_means)],
                prediction_means,
                marker='o',
                linewidth=2,
                label='Mean Prediction'
            )
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Mean Prediction')
            ax.set_title('Prediction Distribution Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            viz_path = viz_dir / "prediction_distribution.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_paths['prediction_distribution'] = str(viz_path)
        
        # Plot alert frequency
        recent_alerts = self.monitor.get_alerts(
            model_id=model_id,
            start_time=datetime.utcnow() - timedelta(days=7)
        )
        
        if recent_alerts:
            # Count alerts by day
            alert_dates = [a.timestamp.date() for a in recent_alerts]
            alert_counts = pd.Series(alert_dates).value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            alert_counts.plot(kind='bar', ax=ax, color='orange')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Alert Count')
            ax.set_title('Alert Frequency')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            viz_path = viz_dir / "alert_frequency.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_paths['alert_frequency'] = str(viz_path)
        
        logger.info(f"Generated {len(viz_paths)} visualizations")
        
        return viz_paths
    
    def _save_report(
        self,
        model_id: str,
        summary: Dict[str, Any]
    ) -> Path:
        """
        Save report to file.
        
        Args:
            model_id: Model identifier
            summary: Report summary
        
        Returns:
            Path to saved report
        """
        report_dir = self.report_path / model_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"weekly_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return report_file
    
    def export_html_report(
        self,
        model_id: str,
        summary: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export report as HTML.
        
        Args:
            model_id: Model identifier
            summary: Report summary
            output_path: Optional output path
        
        Returns:
            Path to HTML report
        """
        if output_path is None:
            report_dir = self.report_path / model_id
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = str(report_dir / f"weekly_report_{timestamp}.html")
        
        # Generate HTML
        html = self._generate_html_report(model_id, summary)
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report exported to {output_path}")
        
        return output_path
    
    def _generate_html_report(
        self,
        model_id: str,
        summary: Dict[str, Any]
    ) -> str:
        """
        Generate HTML report content.
        
        Args:
            model_id: Model identifier
            summary: Report summary
        
        Returns:
            HTML string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Performance Report - {model_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metric {{
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .alert {{
            background-color: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
        }}
        .recommendation {{
            background-color: #d1ecf1;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
            border-radius: 4px;
        }}
        .status-improved {{ color: #28a745; }}
        .status-degraded {{ color: #dc3545; }}
        .status-stable {{ color: #6c757d; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        .viz {{
            margin: 20px 0;
            text-align: center;
        }}
        .viz img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Weekly Performance Report</h1>
        <p><strong>Model ID:</strong> {model_id}</p>
        <p><strong>Report Period:</strong> {summary.get('start_time', 'N/A')} to {summary.get('end_time', 'N/A')}</p>
        <p><strong>Data Points:</strong> {summary.get('data_points', 0)}</p>
        
        <h2>Performance Metrics</h2>
"""
        
        # Add metric summaries
        if 'metric_summaries' in summary:
            html += "<table><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Latest</th></tr>"
            for metric_name, stats in summary['metric_summaries'].items():
                html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{stats['mean']:.4f}</td>
                    <td>{stats['std']:.4f}</td>
                    <td>{stats['min']:.4f}</td>
                    <td>{stats['max']:.4f}</td>
                    <td>{stats['latest']:.4f}</td>
                </tr>
                """
            html += "</table>"
        
        # Add baseline comparison
        if 'production_vs_validation' in summary and summary['production_vs_validation'].get('has_baseline'):
            html += "<h2>Production vs Validation Baseline</h2>"
            for metric_name, comp in summary['production_vs_validation']['metrics'].items():
                status_class = f"status-{comp['status']}"
                html += f"""
                <div class="metric">
                    <strong>{metric_name}</strong>: 
                    <span class="{status_class}">{comp['status'].upper()}</span><br>
                    Baseline: {comp['baseline']:.4f} | Current: {comp['current']:.4f} | 
                    Change: {comp['change_pct']:.1f}%
                </div>
                """
        
        # Add alerts
        if summary.get('alert_count', 0) > 0:
            html += f"<h2>Alerts ({summary['alert_count']})</h2>"
            if 'alerts_by_type' in summary:
                html += "<div class='alert'>"
                html += "<strong>Alerts by Type:</strong><br>"
                for alert_type, count in summary['alerts_by_type'].items():
                    html += f"{alert_type}: {count}<br>"
                html += "</div>"
        
        # Add recommendations
        if 'retraining_recommendations' in summary:
            html += "<h2>Retraining Recommendations</h2>"
            for rec in summary['retraining_recommendations']:
                html += f"""
                <div class="recommendation">
                    <strong>[{rec['priority'].upper()}] {rec['type']}</strong><br>
                    Reason: {rec['reason']}<br>
                    Action: {rec['action']}
                </div>
                """
        
        # Add visualizations
        if 'visualizations' in summary and summary['visualizations']:
            html += "<h2>Performance Trends</h2>"
            for viz_name, viz_path in summary['visualizations'].items():
                html += f"""
                <div class="viz">
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    <img src="{viz_path}" alt="{viz_name}">
                </div>
                """
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
