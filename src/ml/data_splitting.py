"""
Data splitting utilities for model training with chronological ordering.

This module provides functions for splitting data while maintaining temporal ordering
and handling class imbalance for mental health risk assessment models.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

logger = logging.getLogger(__name__)


class DataSplitter:
    """Handles data splitting with chronological ordering and class balancing."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataSplitter.
        
        Args:
            test_size: Proportion of data to reserve for test set (default 0.2)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
    def chronological_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        target_col: str = 'target'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data chronologically ensuring temporal ordering.
        
        Most recent data (last 20%) is reserved as held-out test set.
        Training data precedes test data in time.
        
        Args:
            df: Input dataframe with features and target
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ValueError: If timestamp column is missing or data is empty
        """
        if df.empty:
            raise ValueError("Input dataframe is empty")
            
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe")
            
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Sort by timestamp to ensure chronological ordering
        df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
        
        # Calculate split index
        n_samples = len(df_sorted)
        split_idx = int(n_samples * (1 - self.test_size))
        
        # Split chronologically
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, timestamp_col]]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        logger.info(
            f"Chronological split: train={len(X_train)} samples, "
            f"test={len(X_test)} samples"
        )
        logger.info(
            f"Train period: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}"
        )
        logger.info(
            f"Test period: {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}"
        )
        
        return X_train, X_test, y_train, y_test
    
    def stratified_kfold_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified k-fold splits maintaining class distribution.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_splits: Number of folds (default 5)
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
            
        Raises:
            ValueError: If data is empty or n_splits is invalid
        """
        if X.empty or len(y) == 0:
            raise ValueError("Input data is empty")
            
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
            
        if len(y) < n_splits:
            raise ValueError(f"Cannot split {len(y)} samples into {n_splits} folds")
        
        # Check class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Verify each class has at least n_splits samples
        min_class_count = class_counts.min()
        if min_class_count < n_splits:
            logger.warning(
                f"Smallest class has only {min_class_count} samples, "
                f"which is less than n_splits={n_splits}. "
                f"Reducing n_splits to {min_class_count}"
            )
            n_splits = min_class_count
        
        # Create stratified k-fold splitter
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Generate splits
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            splits.append((train_idx, val_idx))
            
            # Log fold statistics
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            logger.info(
                f"Fold {fold_idx + 1}: train={len(train_idx)} samples, "
                f"val={len(val_idx)} samples"
            )
            logger.info(
                f"  Train class distribution: {y_train_fold.value_counts().to_dict()}"
            )
            logger.info(
                f"  Val class distribution: {y_val_fold.value_counts().to_dict()}"
            )
        
        return splits
    
    def balance_classes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'smote',
        minority_threshold: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes when minority class is less than threshold.
        
        Args:
            X: Feature dataframe
            y: Target series
            method: Balancing method ('smote', 'undersample', 'smoteenn')
            minority_threshold: Apply balancing if minority class < this proportion
            
        Returns:
            Tuple of (X_balanced, y_balanced)
            
        Raises:
            ValueError: If method is invalid or data is empty
        """
        if X.empty or len(y) == 0:
            raise ValueError("Input data is empty")
            
        valid_methods = ['smote', 'undersample', 'smoteenn']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Calculate class proportions
        class_counts = y.value_counts()
        total_samples = len(y)
        class_proportions = class_counts / total_samples
        
        minority_proportion = class_proportions.min()
        
        logger.info(f"Class proportions: {class_proportions.to_dict()}")
        logger.info(f"Minority class proportion: {minority_proportion:.3f}")
        
        # Check if balancing is needed
        if minority_proportion >= minority_threshold:
            logger.info(
                f"Minority class proportion ({minority_proportion:.3f}) >= "
                f"threshold ({minority_threshold}). No balancing needed."
            )
            return X, y
        
        logger.info(
            f"Applying {method} balancing (minority proportion: {minority_proportion:.3f})"
        )
        
        # Convert to numpy for resampling
        X_array = X.values
        y_array = y.values
        
        # Apply balancing method
        if method == 'smote':
            # SMOTE for oversampling minority class
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'undersample':
            # Random undersampling of majority class
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:  # smoteenn
            # Combined SMOTE and Edited Nearest Neighbors
            sampler = SMOTEENN(random_state=self.random_state)
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_array, y_array)
        except Exception as e:
            logger.error(f"Balancing failed: {e}. Returning original data.")
            return X, y
        
        # Convert back to pandas
        X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        y_balanced = pd.Series(y_resampled, name=y.name)
        
        # Log results
        balanced_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"Balanced class distribution: {balanced_counts.to_dict()}")
        logger.info(
            f"Original samples: {len(y)}, Balanced samples: {len(y_balanced)}"
        )
        
        return X_balanced, y_balanced
    
    def get_train_val_test_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        target_col: str = 'target',
        val_size: float = 0.2,
        balance_train: bool = True,
        balance_method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Complete train/val/test split with chronological ordering and optional balancing.
        
        Process:
        1. Split chronologically: most recent 20% as test set
        2. Split remaining data: 80% train, 20% validation (chronologically)
        3. Apply class balancing to training set if needed
        
        Args:
            df: Input dataframe with features, target, and timestamp
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            val_size: Proportion of non-test data for validation
            balance_train: Whether to balance training set classes
            balance_method: Method for class balancing
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set (most recent 20%)
        X_trainval, X_test, y_trainval, y_test = self.chronological_split(
            df, timestamp_col, target_col
        )
        
        # Create temporary dataframe for second split
        trainval_df = X_trainval.copy()
        trainval_df[target_col] = y_trainval
        
        # Add timestamp back for chronological split
        df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - self.test_size))
        trainval_timestamps = df_sorted.iloc[:split_idx][timestamp_col]
        trainval_df[timestamp_col] = trainval_timestamps.values
        
        # Second split: separate validation set from training set
        temp_splitter = DataSplitter(test_size=val_size, random_state=self.random_state)
        X_train, X_val, y_train, y_val = temp_splitter.chronological_split(
            trainval_df, timestamp_col, target_col
        )
        
        # Apply class balancing to training set if requested
        if balance_train:
            X_train, y_train = self.balance_classes(
                X_train, y_train, method=balance_method
            )
        
        logger.info(
            f"Final split sizes - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
