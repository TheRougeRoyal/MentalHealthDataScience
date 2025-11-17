"""
Drift monitoring for detecting distribution changes in data and predictions.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from scipy.special import kl_div
import logging

from src.logging_config import get_logger

logger = get_logger(__name__)


class DriftReport:
    """Report containing drift detection results."""
    
    def __init__(
        self,
        drift_detected: bool,
        drift_score: float,
        threshold: float,
        feature_drifts: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.drift_detected = drift_detected
        self.drift_score = drift_score
        self.threshold = threshold
        self.feature_drifts = feature_drifts or {}
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "feature_drifts": self.feature_drifts,
            "timestamp": self.timestamp.isoformat()
        }


class DriftMonitor:
    """
    Monitors data and prediction distributions for drift.
    
    Detects feature drift using statistical tests (KS test, KL divergence)
    and prediction drift by comparing score distributions over time.
    """
    
    def __init__(
        self,
        feature_drift_threshold: float = 0.3,
        prediction_drift_threshold: float = 0.2,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize the DriftMonitor.
        
        Args:
            feature_drift_threshold: Threshold for feature drift detection
            prediction_drift_threshold: Threshold for prediction drift detection
            alert_callback: Optional callback function for alerts
        """
        self.feature_drift_threshold = feature_drift_threshold
        self.prediction_drift_threshold = prediction_drift_threshold
        self.alert_callback = alert_callback
        
        # Store reference distributions
        self.reference_features: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
        
        logger.info(
            f"DriftMonitor initialized with feature_threshold={feature_drift_threshold}, "
            f"prediction_threshold={prediction_drift_threshold}"
        )
    
    def set_reference_data(
        self,
        features: pd.DataFrame,
        predictions: Optional[np.ndarray] = None
    ) -> None:
        """
        Set reference data for drift comparison.
        
        Args:
            features: Reference feature data
            predictions: Reference prediction data
        """
        self.reference_features = features.copy()
        if predictions is not None:
            self.reference_predictions = predictions.copy()
        
        logger.info(
            f"Set reference data: {len(features)} samples, "
            f"{len(features.columns)} features"
        )

    
    def _compute_ks_statistic(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """
        Compute Kolmogorov-Smirnov statistic.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            
        Returns:
            KS statistic (0-1, higher means more drift)
        """
        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Compute KS test
        statistic, _ = stats.ks_2samp(reference, current)
        return statistic
    
    def _compute_kl_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute KL divergence between distributions.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram
            
        Returns:
            KL divergence (higher means more drift)
        """
        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        # Create histograms with same bins
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        curr_hist = curr_hist + epsilon
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # Compute KL divergence
        kl = np.sum(kl_div(curr_hist, ref_hist))
        
        # Handle inf/nan
        if np.isnan(kl) or np.isinf(kl):
            return 0.0
        
        return float(kl)

    
    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        method: str = "ks"
    ) -> DriftReport:
        """
        Detect feature drift using KL divergence or KS tests.
        
        Args:
            current_data: Current feature data
            method: Method to use ("ks" or "kl")
            
        Returns:
            DriftReport with drift detection results
        """
        if self.reference_features is None:
            raise ValueError("Reference features not set. Call set_reference_data() first.")
        
        logger.info(f"Detecting feature drift using {method} method")
        
        feature_drifts = {}
        drift_scores = []
        
        # Check each feature
        for column in self.reference_features.columns:
            if column not in current_data.columns:
                logger.warning(f"Feature {column} not found in current data")
                continue
            
            ref_values = self.reference_features[column].values
            curr_values = current_data[column].values
            
            # Skip non-numeric columns
            if not np.issubdtype(ref_values.dtype, np.number):
                continue
            
            # Compute drift score
            if method == "ks":
                drift_score = self._compute_ks_statistic(ref_values, curr_values)
            elif method == "kl":
                drift_score = self._compute_kl_divergence(ref_values, curr_values)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            feature_drifts[column] = drift_score
            drift_scores.append(drift_score)
        
        # Overall drift score (max or mean)
        overall_drift = max(drift_scores) if drift_scores else 0.0
        drift_detected = overall_drift > self.feature_drift_threshold
        
        report = DriftReport(
            drift_detected=drift_detected,
            drift_score=overall_drift,
            threshold=self.feature_drift_threshold,
            feature_drifts=feature_drifts
        )
        
        logger.info(
            f"Feature drift detection complete: drift_detected={drift_detected}, "
            f"score={overall_drift:.4f}, threshold={self.feature_drift_threshold}"
        )
        
        # Trigger alert if drift detected
        if drift_detected:
            self.alert_on_drift(report, "feature")
        
        return report

    
    def detect_prediction_drift(
        self,
        predictions: np.ndarray,
        method: str = "ks"
    ) -> DriftReport:
        """
        Detect prediction drift for score distribution shifts.
        
        Args:
            predictions: Current prediction scores
            method: Method to use ("ks" or "kl")
            
        Returns:
            DriftReport with drift detection results
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set. Call set_reference_data() first.")
        
        logger.info(f"Detecting prediction drift using {method} method")
        
        # Compute drift score
        if method == "ks":
            drift_score = self._compute_ks_statistic(
                self.reference_predictions,
                predictions
            )
        elif method == "kl":
            drift_score = self._compute_kl_divergence(
                self.reference_predictions,
                predictions
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        drift_detected = drift_score > self.prediction_drift_threshold
        
        report = DriftReport(
            drift_detected=drift_detected,
            drift_score=drift_score,
            threshold=self.prediction_drift_threshold
        )
        
        logger.info(
            f"Prediction drift detection complete: drift_detected={drift_detected}, "
            f"score={drift_score:.4f}, threshold={self.prediction_drift_threshold}"
        )
        
        # Trigger alert if drift detected
        if drift_detected:
            self.alert_on_drift(report, "prediction")
        
        return report
    
    def alert_on_drift(
        self,
        drift_report: DriftReport,
        drift_type: str
    ) -> None:
        """
        Trigger notifications when drift is detected.
        
        Args:
            drift_report: Drift detection report
            drift_type: Type of drift ("feature" or "prediction")
        """
        alert_message = (
            f"DRIFT ALERT: {drift_type.upper()} drift detected! "
            f"Score: {drift_report.drift_score:.4f}, "
            f"Threshold: {drift_report.threshold:.4f}"
        )
        
        logger.warning(alert_message)
        
        # Call custom alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(drift_report, drift_type)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Log detailed feature drifts if available
        if drift_report.feature_drifts:
            top_drifted = sorted(
                drift_report.feature_drifts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            logger.warning(
                f"Top drifted features: {', '.join([f'{k}={v:.4f}' for k, v in top_drifted])}"
            )
    
    def get_drift_summary(
        self,
        current_data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive drift summary.
        
        Args:
            current_data: Current feature data
            predictions: Current predictions
            
        Returns:
            Dictionary with drift summary
        """
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "feature_drift": None,
            "prediction_drift": None
        }
        
        # Check feature drift
        if self.reference_features is not None:
            feature_report = self.detect_feature_drift(current_data)
            summary["feature_drift"] = feature_report.to_dict()
        
        # Check prediction drift
        if predictions is not None and self.reference_predictions is not None:
            prediction_report = self.detect_prediction_drift(predictions)
            summary["prediction_drift"] = prediction_report.to_dict()
        
        return summary
