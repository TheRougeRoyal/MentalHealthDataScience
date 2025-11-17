"""
Anomaly detection model training for mental health risk assessment.

This module provides training functions for Isolation Forest models
to detect unusual patterns in mental health data.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

logger = logging.getLogger(__name__)


class AnomalyDetectionTrainer:
    """Trains anomaly detection models using Isolation Forest."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """
        Initialize AnomalyDetectionTrainer.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
    
    def train_isolation_forest(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        calibrate_scores: bool = True
    ) -> Tuple[IsolationForest, Dict[str, Any]]:
        """
        Train Isolation Forest for unusual pattern detection.
        
        The model learns to identify anomalous patterns that may indicate
        mental health risk. Anomaly scores are calibrated to align with
        risk assessment scales.
        
        Args:
            X_train: Training features
            y_train: Training labels (optional, for evaluation)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            max_features: Number of features to draw for each tree
            calibrate_scores: Whether to calibrate anomaly scores
            
        Returns:
            Tuple of (trained_model, metadata_dict)
        """
        logger.info("Training Isolation Forest model...")
        logger.info(f"Contamination: {contamination}")
        logger.info(f"Number of estimators: {n_estimators}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize Isolation Forest
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit model
        logger.info("Fitting Isolation Forest...")
        model.fit(X_train_scaled)
        
        # Get anomaly scores (negative scores indicate anomalies)
        train_scores = model.score_samples(X_train_scaled)
        train_predictions = model.predict(X_train_scaled)
        
        # Count anomalies
        n_anomalies = np.sum(train_predictions == -1)
        anomaly_rate = n_anomalies / len(train_predictions)
        
        logger.info(f"Training anomalies detected: {n_anomalies} ({anomaly_rate:.2%})")
        logger.info(f"Anomaly score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
        
        # Calibrate scores if requested
        calibration_params = None
        if calibrate_scores:
            calibration_params = self._calibrate_anomaly_scores(
                train_scores,
                y_train if y_train is not None else None
            )
            logger.info(f"Calibration parameters: {calibration_params}")
        
        # Evaluate on validation set if provided
        val_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate_anomaly_detection(
                model,
                scaler,
                X_val,
                y_val,
                calibration_params
            )
            logger.info(f"Validation metrics: {val_metrics}")
        
        # Feature importance (based on path length)
        feature_importance = self._compute_feature_importance(
            model,
            X_train_scaled,
            X_train.columns
        )
        
        logger.info("Top 10 features for anomaly detection:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create metadata
        metadata = {
            'model_type': 'isolation_forest',
            'parameters': {
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'contamination': contamination,
                'max_features': max_features
            },
            'training_stats': {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'n_anomalies': int(n_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'score_min': float(train_scores.min()),
                'score_max': float(train_scores.max()),
                'score_mean': float(train_scores.mean()),
                'score_std': float(train_scores.std())
            },
            'calibration_params': calibration_params,
            'validation_metrics': val_metrics,
            'feature_names': list(X_train.columns),
            'feature_importance': feature_importance.to_dict('records'),
            'trained_at': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Save model, scaler, and metadata
        model_path = self.model_dir / "isolation_forest_model.pkl"
        scaler_path = self.model_dir / "isolation_forest_scaler.pkl"
        metadata_path = self.model_dir / "isolation_forest_metadata.json"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model, metadata
    
    def _calibrate_anomaly_scores(
        self,
        scores: np.ndarray,
        labels: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calibrate anomaly scores to 0-100 risk scale.
        
        Maps raw anomaly scores (typically negative values) to a 0-100 scale
        where higher values indicate higher risk/anomaly.
        
        Args:
            scores: Raw anomaly scores from Isolation Forest
            labels: Optional true labels for supervised calibration
            
        Returns:
            Dictionary with calibration parameters
        """
        # Basic calibration: map to 0-100 scale
        # Lower (more negative) scores = higher anomaly = higher risk
        
        score_min = scores.min()
        score_max = scores.max()
        
        # Invert and scale to 0-100
        # Most normal (highest score) -> 0
        # Most anomalous (lowest score) -> 100
        calibrated = 100 * (score_max - scores) / (score_max - score_min)
        
        calibration_params = {
            'method': 'minmax_invert',
            'score_min': float(score_min),
            'score_max': float(score_max),
            'calibrated_mean': float(calibrated.mean()),
            'calibrated_std': float(calibrated.std())
        }
        
        # If labels provided, compute optimal threshold
        if labels is not None:
            # Find threshold that maximizes F1 score
            from sklearn.metrics import f1_score
            
            thresholds = np.percentile(calibrated, [50, 60, 70, 75, 80, 85, 90, 95])
            best_f1 = 0
            best_threshold = 75
            
            for threshold in thresholds:
                predictions = (calibrated >= threshold).astype(int)
                f1 = f1_score(labels, predictions)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            calibration_params['optimal_threshold'] = float(best_threshold)
            calibration_params['optimal_f1'] = float(best_f1)
        
        return calibration_params
    
    def _evaluate_anomaly_detection(
        self,
        model: IsolationForest,
        scaler: StandardScaler,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        calibration_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate anomaly detection model on validation set.
        
        Args:
            model: Trained Isolation Forest model
            scaler: Fitted scaler
            X_val: Validation features
            y_val: Validation labels
            calibration_params: Calibration parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Scale validation data
        X_val_scaled = scaler.transform(X_val)
        
        # Get anomaly scores
        val_scores = model.score_samples(X_val_scaled)
        
        # Calibrate scores if parameters provided
        if calibration_params is not None:
            score_min = calibration_params['score_min']
            score_max = calibration_params['score_max']
            calibrated_scores = 100 * (score_max - val_scores) / (score_max - score_min)
        else:
            calibrated_scores = -val_scores  # Use negative scores as risk
        
        # Normalize to 0-1 for AUROC calculation
        normalized_scores = (calibrated_scores - calibrated_scores.min()) / \
                           (calibrated_scores.max() - calibrated_scores.min())
        
        # Calculate metrics
        try:
            auroc = roc_auc_score(y_val, normalized_scores)
        except Exception as e:
            logger.warning(f"Could not calculate AUROC: {e}")
            auroc = None
        
        # Get predictions
        val_predictions = model.predict(X_val_scaled)
        val_predictions_binary = (val_predictions == -1).astype(int)
        
        # Calculate precision and recall
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        try:
            precision = precision_score(y_val, val_predictions_binary)
            recall = recall_score(y_val, val_predictions_binary)
            f1 = f1_score(y_val, val_predictions_binary)
        except Exception as e:
            logger.warning(f"Could not calculate precision/recall: {e}")
            precision = recall = f1 = None
        
        metrics = {
            'auroc': float(auroc) if auroc is not None else None,
            'precision': float(precision) if precision is not None else None,
            'recall': float(recall) if recall is not None else None,
            'f1_score': float(f1) if f1 is not None else None,
            'n_anomalies': int(np.sum(val_predictions == -1)),
            'anomaly_rate': float(np.mean(val_predictions == -1))
        }
        
        return metrics
    
    def _compute_feature_importance(
        self,
        model: IsolationForest,
        X_scaled: np.ndarray,
        feature_names: pd.Index
    ) -> pd.DataFrame:
        """
        Compute feature importance for anomaly detection.
        
        Uses permutation importance to measure feature contribution
        to anomaly detection.
        
        Args:
            model: Trained Isolation Forest model
            X_scaled: Scaled feature array
            feature_names: Feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        # Get baseline scores
        baseline_scores = model.score_samples(X_scaled)
        baseline_mean = baseline_scores.mean()
        
        # Compute permutation importance
        importances = []
        
        for i in range(X_scaled.shape[1]):
            # Permute feature
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get scores with permuted feature
            permuted_scores = model.score_samples(X_permuted)
            permuted_mean = permuted_scores.mean()
            
            # Importance is change in mean score
            importance = abs(baseline_mean - permuted_mean)
            importances.append(importance)
        
        # Create dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def calibrate_score(
        self,
        raw_score: float,
        calibration_params: Dict[str, float]
    ) -> float:
        """
        Calibrate a single anomaly score to 0-100 scale.
        
        Args:
            raw_score: Raw anomaly score from model
            calibration_params: Calibration parameters
            
        Returns:
            Calibrated score (0-100)
        """
        score_min = calibration_params['score_min']
        score_max = calibration_params['score_max']
        
        # Invert and scale
        calibrated = 100 * (score_max - raw_score) / (score_max - score_min)
        
        # Clip to 0-100 range
        calibrated = np.clip(calibrated, 0, 100)
        
        return float(calibrated)
    
    def load_isolation_forest(
        self
    ) -> Tuple[IsolationForest, StandardScaler, Dict[str, Any]]:
        """
        Load saved Isolation Forest model.
        
        Returns:
            Tuple of (model, scaler, metadata)
        """
        model_path = self.model_dir / "isolation_forest_model.pkl"
        scaler_path = self.model_dir / "isolation_forest_scaler.pkl"
        metadata_path = self.model_dir / "isolation_forest_metadata.json"
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded Isolation Forest model from {model_path}")
        
        return model, scaler, metadata
