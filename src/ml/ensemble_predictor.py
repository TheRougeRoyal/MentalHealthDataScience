"""
Ensemble predictor for combining predictions from multiple models.

This module provides an EnsemblePredictor class for combining predictions
from multiple models using weighted averaging, calculating confidence scores,
and classifying risk levels with alerting.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from src.ml.model_registry import ModelRegistry
from src.ml.inference_engine import InferenceEngine
from src.exceptions import EnsembleError

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EnsemblePredictor:
    """Predictor for combining multiple model predictions."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        inference_engine: InferenceEngine,
        alert_threshold: float = 75.0,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize EnsemblePredictor.
        
        Args:
            model_registry: ModelRegistry instance
            inference_engine: InferenceEngine instance
            alert_threshold: Risk score threshold for triggering alerts
            alert_callback: Optional callback function for alerts
        """
        self.model_registry = model_registry
        self.inference_engine = inference_engine
        self.alert_threshold = alert_threshold
        self.alert_callback = alert_callback
        
        logger.info(
            f"EnsemblePredictor initialized with alert threshold: {alert_threshold}"
        )

    def ensemble_predictions(
        self,
        predictions: List[np.ndarray],
        weights: Optional[List[float]] = None,
        method: str = 'weighted_average'
    ) -> np.ndarray:
        """
        Combine predictions from multiple models using weighted averaging.
        
        Args:
            predictions: List of prediction arrays from different models
            weights: Optional list of weights for each model (must sum to 1.0)
            method: Ensemble method ('weighted_average', 'simple_average', 'max', 'median')
            
        Returns:
            Array of ensemble predictions
            
        Raises:
            EnsembleError: If ensemble fails
        """
        try:
            if not predictions:
                raise EnsembleError("No predictions provided for ensemble")
            
            # Validate all predictions have same length
            n_samples = len(predictions[0])
            for i, pred in enumerate(predictions):
                if len(pred) != n_samples:
                    raise EnsembleError(
                        f"Prediction {i} has length {len(pred)}, expected {n_samples}"
                    )
            
            # Convert to array for easier manipulation
            predictions_array = np.array(predictions)
            
            # Apply ensemble method
            if method == 'weighted_average':
                if weights is None:
                    # Equal weights if not provided
                    weights = [1.0 / len(predictions)] * len(predictions)
                
                # Validate weights
                if len(weights) != len(predictions):
                    raise EnsembleError(
                        f"Number of weights ({len(weights)}) must match "
                        f"number of predictions ({len(predictions)})"
                    )
                
                if not np.isclose(sum(weights), 1.0):
                    raise EnsembleError(f"Weights must sum to 1.0, got {sum(weights)}")
                
                # Weighted average
                ensemble = np.average(predictions_array, axis=0, weights=weights)
                
            elif method == 'simple_average':
                ensemble = np.mean(predictions_array, axis=0)
                
            elif method == 'max':
                ensemble = np.max(predictions_array, axis=0)
                
            elif method == 'median':
                ensemble = np.median(predictions_array, axis=0)
                
            else:
                raise EnsembleError(f"Unknown ensemble method: {method}")
            
            logger.info(
                f"Ensemble predictions generated using {method} "
                f"from {len(predictions)} models"
            )
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise EnsembleError(f"Ensemble prediction failed: {e}")

    def calculate_confidence(
        self,
        predictions: List[np.ndarray],
        method: str = 'agreement'
    ) -> np.ndarray:
        """
        Calculate prediction confidence from model agreement.
        
        Higher confidence when models agree, lower when they disagree.
        
        Args:
            predictions: List of prediction arrays from different models
            method: Confidence calculation method ('agreement', 'variance', 'entropy')
            
        Returns:
            Array of confidence scores (0-1)
            
        Raises:
            EnsembleError: If confidence calculation fails
        """
        try:
            if not predictions:
                raise EnsembleError("No predictions provided for confidence calculation")
            
            predictions_array = np.array(predictions)
            
            if method == 'agreement':
                # Confidence based on standard deviation (inverse)
                # Low std = high agreement = high confidence
                std = np.std(predictions_array, axis=0)
                max_std = 0.5  # Maximum possible std for binary predictions
                confidence = 1.0 - (std / max_std)
                confidence = np.clip(confidence, 0, 1)
                
            elif method == 'variance':
                # Similar to agreement but using variance
                var = np.var(predictions_array, axis=0)
                max_var = 0.25  # Maximum variance for binary predictions
                confidence = 1.0 - (var / max_var)
                confidence = np.clip(confidence, 0, 1)
                
            elif method == 'entropy':
                # Confidence based on entropy of predictions
                # Low entropy = high confidence
                mean_pred = np.mean(predictions_array, axis=0)
                # Binary entropy
                epsilon = 1e-10  # Avoid log(0)
                entropy = -(
                    mean_pred * np.log2(mean_pred + epsilon) +
                    (1 - mean_pred) * np.log2(1 - mean_pred + epsilon)
                )
                max_entropy = 1.0  # Maximum entropy for binary
                confidence = 1.0 - (entropy / max_entropy)
                confidence = np.clip(confidence, 0, 1)
                
            else:
                raise EnsembleError(f"Unknown confidence method: {method}")
            
            logger.info(
                f"Confidence calculated using {method} method. "
                f"Mean confidence: {confidence.mean():.3f}"
            )
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            raise EnsembleError(f"Confidence calculation failed: {e}")

    def classify_risk_level(
        self,
        risk_score: float
    ) -> RiskLevel:
        """
        Map continuous risk score to categorical risk level.
        
        Risk levels:
        - LOW: 0-25
        - MODERATE: 26-50
        - HIGH: 51-75
        - CRITICAL: 76-100
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            RiskLevel enum
        """
        if risk_score < 0 or risk_score > 100:
            logger.warning(f"Risk score {risk_score} outside valid range [0, 100]")
            risk_score = np.clip(risk_score, 0, 100)
        
        if risk_score <= 25:
            return RiskLevel.LOW
        elif risk_score <= 50:
            return RiskLevel.MODERATE
        elif risk_score <= 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def trigger_alert(
        self,
        risk_score: float,
        individual_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger alert for high-risk cases.
        
        Args:
            risk_score: Risk score
            individual_id: Individual identifier
            metadata: Optional additional metadata
            
        Returns:
            True if alert was triggered, False otherwise
        """
        if risk_score > self.alert_threshold:
            alert_data = {
                'individual_id': individual_id,
                'risk_score': risk_score,
                'risk_level': self.classify_risk_level(risk_score).value,
                'threshold': self.alert_threshold,
                'metadata': metadata or {}
            }
            
            logger.warning(
                f"ALERT: High risk detected for {individual_id}. "
                f"Risk score: {risk_score:.2f} (threshold: {self.alert_threshold})"
            )
            
            # Call alert callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            return True
        
        return False

    def predict_with_ensemble(
        self,
        features: pd.DataFrame,
        model_ids: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        individual_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate ensemble predictions with confidence and risk classification.
        
        Args:
            features: Input features
            model_ids: Optional list of model IDs to use (uses active models if None)
            weights: Optional weights for ensemble
            individual_ids: Optional list of individual IDs for alerting
            
        Returns:
            Dictionary with predictions, confidence, risk levels, and alerts
        """
        try:
            # Get active models if not specified
            if model_ids is None:
                active_models = self.model_registry.get_active_models()
                if not active_models:
                    raise EnsembleError("No active models found in registry")
                
                model_ids = [m['model_id'] for m in active_models]
                
                # Use validation scores as weights if available
                if weights is None:
                    val_scores = [
                        m.get('val_score') or m.get('cv_score', 0.5)
                        for m in active_models
                    ]
                    # Normalize to sum to 1
                    total_score = sum(val_scores)
                    if total_score > 0:
                        weights = [s / total_score for s in val_scores]
            
            logger.info(f"Running ensemble prediction with {len(model_ids)} models")
            
            # Generate predictions from each model
            predictions_list = []
            
            for model_id in model_ids:
                metadata = self.model_registry.get_model_metadata(model_id)
                model_type = metadata['model_type']
                
                # Determine model category
                if model_type in ['logistic_regression', 'lightgbm']:
                    pred = self.inference_engine.predict_baseline(model_id, features)
                elif model_type.startswith('rnn_') or model_type == 'temporal_fusion_transformer':
                    # For temporal models, assume features are already sequences
                    pred = self.inference_engine.predict_temporal(model_id, features)
                elif model_type == 'isolation_forest':
                    pred = self.inference_engine.detect_anomalies(
                        model_id, features, return_scores=True, calibrate=True
                    )
                else:
                    logger.warning(f"Unknown model type {model_type}, skipping")
                    continue
                
                # Convert to 0-100 scale if needed
                if pred.max() <= 1.0:
                    pred = pred * 100
                
                predictions_list.append(pred)
            
            if not predictions_list:
                raise EnsembleError("No valid predictions generated")
            
            # Ensemble predictions
            ensemble_scores = self.ensemble_predictions(
                predictions_list,
                weights=weights,
                method='weighted_average'
            )
            
            # Calculate confidence
            confidence_scores = self.calculate_confidence(
                predictions_list,
                method='agreement'
            )
            
            # Classify risk levels
            risk_levels = [
                self.classify_risk_level(score)
                for score in ensemble_scores
            ]
            
            # Trigger alerts for high-risk cases
            alerts_triggered = []
            if individual_ids is not None:
                for i, (score, ind_id) in enumerate(zip(ensemble_scores, individual_ids)):
                    alert_triggered = self.trigger_alert(
                        score,
                        ind_id,
                        metadata={
                            'confidence': float(confidence_scores[i]),
                            'risk_level': risk_levels[i].value
                        }
                    )
                    alerts_triggered.append(alert_triggered)
            else:
                # Check for alerts without individual IDs
                for score in ensemble_scores:
                    alerts_triggered.append(score > self.alert_threshold)
            
            # Compile results
            results = {
                'risk_scores': ensemble_scores,
                'confidence': confidence_scores,
                'risk_levels': [rl.value for rl in risk_levels],
                'alerts_triggered': alerts_triggered,
                'n_models': len(predictions_list),
                'model_ids': model_ids,
                'weights': weights
            }
            
            logger.info(
                f"Ensemble prediction completed. "
                f"Mean risk score: {ensemble_scores.mean():.2f}, "
                f"Mean confidence: {confidence_scores.mean():.3f}, "
                f"Alerts triggered: {sum(alerts_triggered)}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise EnsembleError(f"Ensemble prediction failed: {e}")
