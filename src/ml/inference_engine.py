"""
Inference engine for generating predictions from trained models.

This module provides an InferenceEngine class for running predictions
on baseline, temporal, and anomaly detection models with batching support.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

from src.ml.model_registry import ModelRegistry
from src.exceptions import InferenceError, ModelNotFoundError

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Engine for generating predictions from registered models."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize InferenceEngine.
        
        Args:
            model_registry: ModelRegistry instance
            batch_size: Batch size for inference
            device: Device for PyTorch models ('cuda' or 'cpu')
        """
        self.model_registry = model_registry
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"InferenceEngine initialized with device: {self.device}")

    def predict_baseline(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Generate predictions from baseline models (Logistic Regression, LightGBM).
        
        Args:
            model_id: Model ID from registry
            features: Input features as DataFrame
            return_proba: Whether to return probabilities (True) or binary predictions (False)
            
        Returns:
            Array of predictions
            
        Raises:
            InferenceError: If inference fails
        """
        try:
            start_time = time.time()
            
            # Load model
            model_bundle, metadata = self.model_registry.load_model(model_id)
            model = model_bundle['model']
            artifacts = model_bundle.get('artifacts', {})
            model_type = metadata['model_type']
            
            logger.info(f"Running inference with {model_type} model {model_id}")
            logger.info(f"Input shape: {features.shape}")
            
            # Validate features
            expected_features = metadata.get('feature_names', [])
            if expected_features:
                missing_features = set(expected_features) - set(features.columns)
                if missing_features:
                    raise InferenceError(
                        f"Missing features: {missing_features}"
                    )
                
                # Reorder features to match training
                features = features[expected_features]
            
            # Apply preprocessing if scaler available
            if 'scaler' in artifacts:
                features_scaled = artifacts['scaler'].transform(features)
            else:
                features_scaled = features.values
            
            # Generate predictions based on model type
            if model_type == 'logistic_regression':
                if return_proba:
                    predictions = model.predict_proba(features_scaled)[:, 1]
                else:
                    predictions = model.predict(features_scaled)
                    
            elif model_type == 'lightgbm':
                predictions = model.predict(features_scaled)
                if not return_proba:
                    predictions = (predictions >= 0.5).astype(int)
            
            else:
                raise InferenceError(f"Unsupported baseline model type: {model_type}")
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Inference completed in {elapsed_time:.3f}s "
                f"({len(features) / elapsed_time:.1f} samples/sec)"
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise InferenceError(f"Baseline inference failed: {e}")

    def predict_temporal(
        self,
        model_id: str,
        sequences: np.ndarray,
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Generate predictions from temporal models (RNN, TFT).
        
        Args:
            model_id: Model ID from registry
            sequences: Input sequences as array (n_samples, seq_length, n_features)
            return_proba: Whether to return probabilities (True) or binary predictions (False)
            
        Returns:
            Array of predictions
            
        Raises:
            InferenceError: If inference fails
        """
        try:
            start_time = time.time()
            
            # Load model
            model_bundle, metadata = self.model_registry.load_model(model_id)
            model = model_bundle['model']
            model_type = metadata['model_type']
            
            logger.info(f"Running inference with {model_type} model {model_id}")
            logger.info(f"Input shape: {sequences.shape}")
            
            # Validate sequence shape
            expected_seq_length = metadata.get('sequence_length')
            expected_n_features = metadata.get('n_features')
            
            if expected_seq_length and sequences.shape[1] != expected_seq_length:
                raise InferenceError(
                    f"Expected sequence length {expected_seq_length}, "
                    f"got {sequences.shape[1]}"
                )
            
            if expected_n_features and sequences.shape[2] != expected_n_features:
                raise InferenceError(
                    f"Expected {expected_n_features} features, "
                    f"got {sequences.shape[2]}"
                )
            
            # Generate predictions based on model type
            if model_type.startswith('rnn_'):
                predictions = self._predict_rnn(model, sequences)
                
            elif model_type == 'temporal_fusion_transformer':
                predictions = self._predict_tft(model, sequences, metadata)
                
            else:
                raise InferenceError(f"Unsupported temporal model type: {model_type}")
            
            if not return_proba:
                predictions = (predictions >= 0.5).astype(int)
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Inference completed in {elapsed_time:.3f}s "
                f"({len(sequences) / elapsed_time:.1f} samples/sec)"
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Temporal inference failed: {e}")
            raise InferenceError(f"Temporal inference failed: {e}")
    
    def _predict_rnn(
        self,
        model: torch.nn.Module,
        sequences: np.ndarray
    ) -> np.ndarray:
        """
        Generate predictions from RNN model.
        
        Args:
            model: RNN model
            sequences: Input sequences
            
        Returns:
            Array of predictions
        """
        model.eval()
        model.to(self.device)
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def _predict_tft(
        self,
        model: Any,
        sequences: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate predictions from Temporal Fusion Transformer.
        
        Args:
            model: TFT model
            sequences: Input sequences
            metadata: Model metadata
            
        Returns:
            Array of predictions
        """
        # TFT requires specific data format
        # This is a simplified implementation
        # In practice, would need proper TimeSeriesDataSet
        
        model.eval()
        
        # For now, return placeholder
        # Full implementation would require TimeSeriesDataSet preparation
        logger.warning("TFT inference not fully implemented, returning placeholder")
        
        return np.random.rand(len(sequences))

    def detect_anomalies(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_scores: bool = True,
        calibrate: bool = True
    ) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            model_id: Model ID from registry
            features: Input features as DataFrame
            return_scores: Whether to return anomaly scores (True) or binary predictions (False)
            calibrate: Whether to calibrate scores to 0-100 scale
            
        Returns:
            Array of anomaly scores or predictions
            
        Raises:
            InferenceError: If inference fails
        """
        try:
            start_time = time.time()
            
            # Load model
            model_bundle, metadata = self.model_registry.load_model(model_id)
            model = model_bundle['model']
            artifacts = model_bundle.get('artifacts', {})
            
            logger.info(f"Running anomaly detection with model {model_id}")
            logger.info(f"Input shape: {features.shape}")
            
            # Validate features
            expected_features = metadata.get('feature_names', [])
            if expected_features:
                missing_features = set(expected_features) - set(features.columns)
                if missing_features:
                    raise InferenceError(
                        f"Missing features: {missing_features}"
                    )
                
                # Reorder features to match training
                features = features[expected_features]
            
            # Apply preprocessing if scaler available
            if 'scaler' in artifacts:
                features_scaled = artifacts['scaler'].transform(features)
            else:
                features_scaled = features.values
            
            # Get anomaly scores
            raw_scores = model.score_samples(features_scaled)
            
            if return_scores:
                if calibrate and 'calibration_params' in metadata:
                    # Calibrate scores to 0-100 scale
                    calibration_params = metadata['calibration_params']
                    score_min = calibration_params['score_min']
                    score_max = calibration_params['score_max']
                    
                    # Invert and scale (lower raw score = higher anomaly = higher risk)
                    scores = 100 * (score_max - raw_scores) / (score_max - score_min)
                    scores = np.clip(scores, 0, 100)
                else:
                    # Return raw scores (negative values)
                    scores = raw_scores
            else:
                # Return binary predictions (-1 for anomaly, 1 for normal)
                scores = model.predict(features_scaled)
                # Convert to 0/1 (1 for anomaly)
                scores = (scores == -1).astype(int)
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Anomaly detection completed in {elapsed_time:.3f}s "
                f"({len(features) / elapsed_time:.1f} samples/sec)"
            )
            
            return scores
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise InferenceError(f"Anomaly detection failed: {e}")
    
    def batch_predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        model_category: str = 'baseline'
    ) -> np.ndarray:
        """
        Run batch prediction with automatic model category detection.
        
        Args:
            model_id: Model ID from registry
            features: Input features
            model_category: Category of model ('baseline', 'temporal', 'anomaly')
            
        Returns:
            Array of predictions
        """
        if model_category == 'baseline':
            return self.predict_baseline(model_id, features)
        elif model_category == 'temporal':
            # Assume features are already in sequence format
            return self.predict_temporal(model_id, features)
        elif model_category == 'anomaly':
            return self.detect_anomalies(model_id, features)
        else:
            raise InferenceError(f"Unknown model category: {model_category}")
