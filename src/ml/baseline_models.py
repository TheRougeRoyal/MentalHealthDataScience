"""
Baseline model training for mental health risk assessment.

This module provides training functions for logistic regression and LightGBM models
with hyperparameter tuning, cross-validation, and model serialization.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """Trains baseline models (Logistic Regression and LightGBM)."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """
        Initialize BaselineModelTrainer.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        param_grid: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        search_method: str = 'grid'
    ) -> Tuple[LogisticRegression, Dict[str, Any]]:
        """
        Train logistic regression model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            param_grid: Hyperparameter grid for search
            cv: Number of cross-validation folds
            search_method: 'grid' or 'random' search
            
        Returns:
            Tuple of (trained_model, metadata_dict)
        """
        logger.info("Training Logistic Regression model...")
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000],
                'class_weight': ['balanced', None]
            }
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Base model
        base_model = LogisticRegression(random_state=self.random_state)
        
        # Hyperparameter search
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        
        # Fit model
        search.fit(X_train_scaled, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score (AUROC): {best_score:.4f}")
        
        # Evaluate on validation set if provided
        val_score = None
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            val_score = best_model.score(X_val_scaled, y_val)
            logger.info(f"Validation score: {val_score:.4f}")
        
        # Create metadata
        metadata = {
            'model_type': 'logistic_regression',
            'best_params': best_params,
            'cv_score': float(best_score),
            'val_score': float(val_score) if val_score is not None else None,
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'training_samples': len(X_train),
            'trained_at': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Save model and scaler
        model_path = self.model_dir / "logistic_regression_model.pkl"
        scaler_path = self.model_dir / "logistic_regression_scaler.pkl"
        metadata_path = self.model_dir / "logistic_regression_metadata.json"
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return best_model, metadata
    
    def train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
        cv: int = 5
    ) -> Tuple[lgb.Booster, Dict[str, Any]]:
        """
        Train LightGBM model with early stopping and cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            param_grid: Hyperparameter grid for tuning
            early_stopping_rounds: Rounds for early stopping
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (trained_model, metadata_dict)
        """
        logger.info("Training LightGBM model...")
        
        # Default parameters
        if param_grid is None:
            param_grid = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Hyperparameter tuning with cross-validation
        logger.info("Performing hyperparameter tuning...")
        
        # Parameter search space for tuning
        tuning_params = {
            'num_leaves': [15, 31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 5, 10, 15],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        
        best_score = 0
        best_params = param_grid.copy()
        
        # Simple grid search over key parameters
        for num_leaves in tuning_params['num_leaves']:
            for learning_rate in tuning_params['learning_rate']:
                for max_depth in tuning_params['max_depth']:
                    test_params = param_grid.copy()
                    test_params.update({
                        'num_leaves': num_leaves,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth
                    })
                    
                    # Cross-validation
                    cv_results = lgb.cv(
                        test_params,
                        train_data,
                        num_boost_round=1000,
                        nfold=cv,
                        stratified=True,
                        shuffle=True,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=False,
                        seed=self.random_state
                    )
                    
                    # Get best score
                    score = max(cv_results['valid auc-mean'])
                    
                    if score > best_score:
                        best_score = score
                        best_params.update({
                            'num_leaves': num_leaves,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth
                        })
        
        logger.info(f"Best CV score (AUROC): {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # Get validation score
        y_val_pred = model.predict(X_val)
        from sklearn.metrics import roc_auc_score
        val_score = roc_auc_score(y_val, y_val_pred)
        
        logger.info(f"Final validation score (AUROC): {val_score:.4f}")
        logger.info(f"Best iteration: {model.best_iteration}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features by importance:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        # Create metadata
        metadata = {
            'model_type': 'lightgbm',
            'best_params': best_params,
            'cv_score': float(best_score),
            'val_score': float(val_score),
            'best_iteration': int(model.best_iteration),
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'feature_importance': feature_importance.to_dict('records'),
            'training_samples': len(X_train),
            'trained_at': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        # Save model and metadata
        model_path = self.model_dir / "lightgbm_model.txt"
        metadata_path = self.model_dir / "lightgbm_metadata.json"
        
        model.save_model(str(model_path))
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model, metadata
    
    def load_logistic_regression(self) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
        """
        Load saved logistic regression model.
        
        Returns:
            Tuple of (model, scaler, metadata)
        """
        model_path = self.model_dir / "logistic_regression_model.pkl"
        scaler_path = self.model_dir / "logistic_regression_scaler.pkl"
        metadata_path = self.model_dir / "logistic_regression_metadata.json"
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded logistic regression model from {model_path}")
        
        return model, scaler, metadata
    
    def load_lgbm(self) -> Tuple[lgb.Booster, Dict[str, Any]]:
        """
        Load saved LightGBM model.
        
        Returns:
            Tuple of (model, metadata)
        """
        model_path = self.model_dir / "lightgbm_model.txt"
        metadata_path = self.model_dir / "lightgbm_metadata.json"
        
        model = lgb.Booster(model_file=str(model_path))
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded LightGBM model from {model_path}")
        
        return model, metadata
