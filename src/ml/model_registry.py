"""
Model registry for version management and model lifecycle.

This module provides a ModelRegistry class for storing, loading, and managing
machine learning models with metadata, versioning, and caching capabilities.
"""

import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import joblib
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.exceptions import ModelNotFoundError, ModelRegistrationError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing model versions and metadata."""
    
    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialize ModelRegistry.
        
        Args:
            registry_dir: Directory for storing models and metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded models
        self._model_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        
        # Registry index file
        self.index_file = self.registry_dir / "registry_index.json"
        self._load_or_create_index()
        
        logger.info(f"ModelRegistry initialized at {self.registry_dir}")
    
    def _load_or_create_index(self) -> None:
        """Load or create the registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
            logger.info(f"Loaded registry index with {len(self.index)} models")
        else:
            self.index = {}
            self._save_index()
            logger.info("Created new registry index")
    
    def _save_index(self) -> None:
        """Save the registry index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _generate_model_id(
        self,
        model_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate unique model ID based on type and metadata.
        
        Args:
            model_type: Type of model
            metadata: Model metadata
            
        Returns:
            Unique model ID
        """
        # Create hash from model type and key metadata
        hash_input = f"{model_type}_{datetime.now().isoformat()}"
        model_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        # Format: modeltype_timestamp_hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{timestamp}_{model_hash}"
        
        return model_id
    
    def register_model(
        self,
        model: Any,
        model_type: str,
        metadata: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
        set_active: bool = True,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> str:
        """
        Register a model with metadata and optional artifacts.
        
        Args:
            model: Trained model object
            model_type: Type of model (e.g., 'logistic_regression', 'lightgbm', 'rnn_lstm')
            metadata: Model metadata including performance metrics
            artifacts: Optional additional artifacts (e.g., scalers, encoders)
            set_active: Whether to set this model as active for its type
            run_id: Optional experiment run ID to link with
            experiment_id: Optional experiment ID to link with
            
        Returns:
            Model ID
            
        Raises:
            ModelRegistrationError: If registration fails
        """
        try:
            # Generate model ID
            model_id = self._generate_model_id(model_type, metadata)
            
            logger.info(f"Registering model {model_id} of type {model_type}")
            
            # Create model directory
            model_dir = self.registry_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model based on type
            model_path = self._save_model(model, model_type, model_dir)
            
            # Save artifacts if provided
            artifact_paths = {}
            if artifacts:
                artifact_paths = self._save_artifacts(artifacts, model_dir)
            
            # Enhance metadata
            full_metadata = {
                'model_id': model_id,
                'model_type': model_type,
                'model_path': str(model_path.relative_to(self.registry_dir)),
                'artifact_paths': artifact_paths,
                'registered_at': datetime.now().isoformat(),
                'is_active': set_active,
                'status': 'active',
                **metadata
            }
            
            # Add experiment tracking fields if provided
            if run_id is not None:
                full_metadata['run_id'] = run_id
            if experiment_id is not None:
                full_metadata['experiment_id'] = experiment_id
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update index
            self.index[model_id] = full_metadata
            
            # Set as active if requested
            if set_active:
                self._set_active_model(model_type, model_id)
            
            self._save_index()
            
            logger.info(f"Successfully registered model {model_id}")
            logger.info(f"Model path: {model_path}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise ModelRegistrationError(f"Model registration failed: {e}")
    
    def _save_model(
        self,
        model: Any,
        model_type: str,
        model_dir: Path
    ) -> Path:
        """
        Save model to disk based on type.
        
        Args:
            model: Model object
            model_type: Type of model
            model_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        if model_type in ['logistic_regression', 'isolation_forest']:
            # Scikit-learn models
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
        elif model_type == 'lightgbm':
            # LightGBM model
            model_path = model_dir / "model.txt"
            model.save_model(str(model_path))
            
        elif model_type.startswith('rnn_'):
            # PyTorch RNN models
            model_path = model_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
        elif model_type == 'temporal_fusion_transformer':
            # TFT model (PyTorch Lightning checkpoint)
            model_path = model_dir / "model.ckpt"
            # Assume model is already saved, just reference it
            
        else:
            # Generic pickle for unknown types
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
        
        return model_path
    
    def _save_artifacts(
        self,
        artifacts: Dict[str, Any],
        model_dir: Path
    ) -> Dict[str, str]:
        """
        Save model artifacts (scalers, encoders, etc.).
        
        Args:
            artifacts: Dictionary of artifact name to object
            model_dir: Directory to save artifacts
            
        Returns:
            Dictionary of artifact name to relative path
        """
        artifact_paths = {}
        
        for name, artifact in artifacts.items():
            artifact_path = model_dir / f"{name}.pkl"
            joblib.dump(artifact, artifact_path)
            artifact_paths[name] = str(artifact_path.relative_to(self.registry_dir))
        
        return artifact_paths
    
    def _set_active_model(self, model_type: str, model_id: str) -> None:
        """
        Set a model as active for its type.
        
        Args:
            model_type: Type of model
            model_id: Model ID to set as active
        """
        # Deactivate other models of same type
        for mid, meta in self.index.items():
            if meta['model_type'] == model_type and mid != model_id:
                meta['is_active'] = False
        
        # Activate the specified model
        if model_id in self.index:
            self.index[model_id]['is_active'] = True
    
    def load_model(
        self,
        model_id: str,
        use_cache: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model by ID with optional caching.
        
        Args:
            model_id: Model ID to load
            use_cache: Whether to use cached model if available
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            ModelNotFoundError: If model ID not found
        """
        # Check cache first
        if use_cache and model_id in self._model_cache:
            logger.debug(f"Loading model {model_id} from cache")
            return self._model_cache[model_id]
        
        # Check if model exists in index
        if model_id not in self.index:
            raise ModelNotFoundError(f"Model {model_id} not found in registry")
        
        metadata = self.index[model_id]
        model_type = metadata['model_type']
        
        logger.info(f"Loading model {model_id} of type {model_type}")
        
        # Load model based on type
        model_dir = self.registry_dir / model_id
        model = self._load_model_by_type(model_type, model_dir, metadata)
        
        # Load artifacts if present
        artifacts = {}
        if 'artifact_paths' in metadata and metadata['artifact_paths']:
            artifacts = self._load_artifacts(metadata['artifact_paths'])
        
        # Combine model and artifacts
        model_bundle = {
            'model': model,
            'artifacts': artifacts
        }
        
        # Cache the loaded model
        if use_cache:
            self._model_cache[model_id] = (model_bundle, metadata)
        
        logger.info(f"Successfully loaded model {model_id}")
        
        return model_bundle, metadata
    
    def _load_model_by_type(
        self,
        model_type: str,
        model_dir: Path,
        metadata: Dict[str, Any]
    ) -> Any:
        """
        Load model based on type.
        
        Args:
            model_type: Type of model
            model_dir: Directory containing model
            metadata: Model metadata
            
        Returns:
            Loaded model object
        """
        if model_type in ['logistic_regression', 'isolation_forest']:
            # Scikit-learn models
            model_path = model_dir / "model.pkl"
            model = joblib.load(model_path)
            
        elif model_type == 'lightgbm':
            # LightGBM model
            model_path = model_dir / "model.txt"
            model = lgb.Booster(model_file=str(model_path))
            
        elif model_type.startswith('rnn_'):
            # PyTorch RNN models - need to reconstruct architecture
            from src.ml.temporal_models import RNNModel
            
            model_path = model_dir / "model.pt"
            arch = metadata.get('architecture', {})
            
            model = RNNModel(
                input_size=arch.get('input_size', 10),
                hidden_size=arch.get('hidden_size', 64),
                num_layers=arch.get('num_layers', 2),
                dropout=arch.get('dropout', 0.3),
                rnn_type=model_type.split('_')[1],  # lstm or gru
                bidirectional=arch.get('bidirectional', False)
            )
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
        elif model_type == 'temporal_fusion_transformer':
            # TFT model
            from pytorch_forecasting import TemporalFusionTransformer
            import pytorch_lightning as pl
            
            model_path = model_dir / "model.ckpt"
            model = TemporalFusionTransformer.load_from_checkpoint(str(model_path))
            model.eval()
            
        else:
            # Generic pickle
            model_path = model_dir / "model.pkl"
            model = joblib.load(model_path)
        
        return model
    
    def _load_artifacts(
        self,
        artifact_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Load model artifacts.
        
        Args:
            artifact_paths: Dictionary of artifact name to relative path
            
        Returns:
            Dictionary of artifact name to loaded object
        """
        artifacts = {}
        
        for name, rel_path in artifact_paths.items():
            artifact_path = self.registry_dir / rel_path
            artifacts[name] = joblib.load(artifact_path)
        
        return artifacts
    
    def get_active_models(
        self,
        model_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active models for ensemble selection.
        
        Args:
            model_types: Optional list of model types to filter by
            
        Returns:
            List of active model metadata dictionaries
        """
        active_models = []
        
        for model_id, metadata in self.index.items():
            if metadata.get('is_active', False) and metadata.get('status') == 'active':
                # Filter by model type if specified
                if model_types is None or metadata['model_type'] in model_types:
                    active_models.append(metadata)
        
        logger.info(f"Found {len(active_models)} active models")
        
        return active_models
    
    def retire_model(
        self,
        model_id: str,
        reason: Optional[str] = None
    ) -> None:
        """
        Retire a model (mark as deprecated).
        
        Args:
            model_id: Model ID to retire
            reason: Optional reason for retirement
            
        Raises:
            ModelNotFoundError: If model ID not found
        """
        if model_id not in self.index:
            raise ModelNotFoundError(f"Model {model_id} not found in registry")
        
        logger.info(f"Retiring model {model_id}")
        
        # Update metadata
        self.index[model_id]['status'] = 'retired'
        self.index[model_id]['is_active'] = False
        self.index[model_id]['retired_at'] = datetime.now().isoformat()
        
        if reason:
            self.index[model_id]['retirement_reason'] = reason
        
        # Remove from cache
        if model_id in self._model_cache:
            del self._model_cache[model_id]
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Model {model_id} retired successfully")
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List models in registry with optional filters.
        
        Args:
            model_type: Optional model type filter
            status: Optional status filter ('active', 'retired')
            
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for model_id, metadata in self.index.items():
            # Apply filters
            if model_type and metadata['model_type'] != model_type:
                continue
            if status and metadata.get('status') != status:
                continue
            
            models.append(metadata)
        
        return models
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata dictionary
            
        Raises:
            ModelNotFoundError: If model ID not found
        """
        if model_id not in self.index:
            raise ModelNotFoundError(f"Model {model_id} not found in registry")
        
        return self.index[model_id]
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_models_by_experiment(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Get all models linked to an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for model_id, metadata in self.index.items():
            if metadata.get('experiment_id') == experiment_id:
                models.append(metadata)
        
        logger.info(f"Found {len(models)} models for experiment {experiment_id}")
        
        return models
    
    def get_models_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all models linked to a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for model_id, metadata in self.index.items():
            if metadata.get('run_id') == run_id:
                models.append(metadata)
        
        logger.info(f"Found {len(models)} models for run {run_id}")
        
        return models
