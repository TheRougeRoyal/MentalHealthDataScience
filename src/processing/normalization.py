"""
Feature normalization module for scaling numerical features.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Handles feature scaling operations including standardization,
    min-max scaling, and per-individual time-series normalization.
    """

    def __init__(self):
        """Initialize Normalizer."""
        self.scaling_params: Dict = {}
        self.normalization_stats: Dict = {}

    def normalize_timeseries(
        self,
        df: DataFrame,
        columns: List[str],
        group_by: str = "anonymized_id",
        fit: bool = True,
    ) -> DataFrame:
        """
        Normalize time-series data with per-individual standardization.

        Each individual's time-series is standardized to zero mean and unit variance
        within their own history, preserving individual-specific patterns.

        Args:
            df: Input DataFrame
            columns: List of columns to normalize
            group_by: Column to group by for per-individual normalization
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with normalized time-series features
        """
        df_result = df.copy()
        normalization_counts = {}

        if group_by not in df_result.columns:
            logger.warning(
                f"Group column {group_by} not found. "
                "Falling back to global standardization."
            )
            return self.standardize_features(df_result, columns, fit)

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping normalization.")
                continue

            if not pd.api.types.is_numeric_dtype(df_result[col]):
                logger.warning(f"Column {col} is not numeric. Skipping normalization.")
                continue

            if fit:
                # Calculate per-individual mean and std
                group_stats = df_result.groupby(group_by)[col].agg(["mean", "std"])
                self.scaling_params[f"{col}_timeseries_mean"] = group_stats["mean"].to_dict()
                self.scaling_params[f"{col}_timeseries_std"] = group_stats["std"].to_dict()
            else:
                # Use fitted parameters
                mean_dict = self.scaling_params.get(f"{col}_timeseries_mean", {})
                std_dict = self.scaling_params.get(f"{col}_timeseries_std", {})

                if not mean_dict or not std_dict:
                    logger.warning(
                        f"No fitted parameters for {col}. Using current statistics."
                    )
                    group_stats = df_result.groupby(group_by)[col].agg(["mean", "std"])
                    mean_dict = group_stats["mean"].to_dict()
                    std_dict = group_stats["std"].to_dict()

            # Apply per-individual normalization
            def normalize_group(group):
                group_id = group.name
                mean_val = (
                    self.scaling_params.get(f"{col}_timeseries_mean", {}).get(group_id)
                    if not fit
                    else group[col].mean()
                )
                std_val = (
                    self.scaling_params.get(f"{col}_timeseries_std", {}).get(group_id)
                    if not fit
                    else group[col].std()
                )

                # Handle case where std is 0 or NaN
                if pd.isna(std_val) or std_val == 0:
                    std_val = 1.0

                return (group[col] - mean_val) / std_val

            df_result[col] = df_result.groupby(group_by, group_keys=False).apply(
                normalize_group
            )

            normalization_counts[col] = len(df_result[col].dropna())
            logger.info(
                f"Normalized {normalization_counts[col]} values in {col} "
                f"with per-individual standardization"
            )

        self.normalization_stats["timeseries_normalized"] = normalization_counts

        return df_result

    def standardize_features(
        self, df: DataFrame, columns: List[str], fit: bool = True
    ) -> DataFrame:
        """
        Standardize features to zero mean and unit variance.

        Args:
            df: Input DataFrame
            columns: List of columns to standardize
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with standardized features
        """
        df_result = df.copy()
        standardization_counts = {}

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping standardization.")
                continue

            if not pd.api.types.is_numeric_dtype(df_result[col]):
                logger.warning(f"Column {col} is not numeric. Skipping standardization.")
                continue

            if fit:
                # Calculate and store mean and std
                mean_val = df_result[col].mean()
                std_val = df_result[col].std()

                # Handle case where std is 0 or NaN
                if pd.isna(std_val) or std_val == 0:
                    std_val = 1.0
                    logger.warning(
                        f"Column {col} has zero or NaN std. Using std=1.0 to avoid division by zero."
                    )

                self.scaling_params[f"{col}_mean"] = mean_val
                self.scaling_params[f"{col}_std"] = std_val
            else:
                # Use fitted parameters
                mean_val = self.scaling_params.get(f"{col}_mean")
                std_val = self.scaling_params.get(f"{col}_std")

                if mean_val is None or std_val is None:
                    logger.warning(
                        f"No fitted parameters for {col}. Using current statistics."
                    )
                    mean_val = df_result[col].mean()
                    std_val = df_result[col].std()
                    if pd.isna(std_val) or std_val == 0:
                        std_val = 1.0

            # Apply standardization
            df_result[col] = (df_result[col] - mean_val) / std_val

            standardization_counts[col] = len(df_result[col].dropna())
            logger.info(
                f"Standardized {standardization_counts[col]} values in {col} "
                f"(mean={mean_val:.2f}, std={std_val:.2f})"
            )

        self.normalization_stats["standardized"] = standardization_counts

        return df_result

    def min_max_scale(
        self,
        df: DataFrame,
        columns: List[str],
        feature_range: Tuple[float, float] = (0, 1),
        fit: bool = True,
    ) -> DataFrame:
        """
        Scale features to a specified range using min-max normalization.

        Args:
            df: Input DataFrame
            columns: List of columns to scale
            feature_range: Tuple of (min, max) for the target range
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with min-max scaled features
        """
        df_result = df.copy()
        scaling_counts = {}
        min_target, max_target = feature_range

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping min-max scaling.")
                continue

            if not pd.api.types.is_numeric_dtype(df_result[col]):
                logger.warning(f"Column {col} is not numeric. Skipping min-max scaling.")
                continue

            if fit:
                # Calculate and store min and max
                min_val = df_result[col].min()
                max_val = df_result[col].max()

                # Handle case where min equals max
                if min_val == max_val:
                    logger.warning(
                        f"Column {col} has constant value. Setting to middle of range."
                    )
                    max_val = min_val + 1.0

                self.scaling_params[f"{col}_min"] = min_val
                self.scaling_params[f"{col}_max"] = max_val
            else:
                # Use fitted parameters
                min_val = self.scaling_params.get(f"{col}_min")
                max_val = self.scaling_params.get(f"{col}_max")

                if min_val is None or max_val is None:
                    logger.warning(
                        f"No fitted parameters for {col}. Using current min/max."
                    )
                    min_val = df_result[col].min()
                    max_val = df_result[col].max()
                    if min_val == max_val:
                        max_val = min_val + 1.0

            # Apply min-max scaling
            # Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max_target - min_target) + min_target
            df_result[col] = (
                (df_result[col] - min_val) / (max_val - min_val) * (max_target - min_target)
                + min_target
            )

            scaling_counts[col] = len(df_result[col].dropna())
            logger.info(
                f"Min-max scaled {scaling_counts[col]} values in {col} "
                f"from [{min_val:.2f}, {max_val:.2f}] to [{min_target}, {max_target}]"
            )

        self.normalization_stats["min_max_scaled"] = scaling_counts

        return df_result

    def get_scaling_params(self) -> Dict:
        """
        Get fitted scaling parameters for inference.

        Returns:
            Dictionary containing all fitted scaling parameters
        """
        return self.scaling_params.copy()

    def set_scaling_params(self, params: Dict) -> None:
        """
        Set scaling parameters for inference.

        Args:
            params: Dictionary of scaling parameters
        """
        self.scaling_params = params.copy()

    def get_normalization_stats(self) -> Dict:
        """
        Get statistics from the last normalization operation.

        Returns:
            Dictionary containing normalization statistics
        """
        return self.normalization_stats.copy()

    def reset_stats(self) -> None:
        """Reset normalization statistics."""
        self.normalization_stats = {}

    def save_params(self, filepath: str) -> None:
        """
        Save scaling parameters to a file for persistence.

        Args:
            filepath: Path to save the parameters (JSON format)
        """
        import json

        with open(filepath, "w") as f:
            # Convert any numpy types to native Python types for JSON serialization
            serializable_params = {}
            for key, value in self.scaling_params.items():
                if isinstance(value, dict):
                    serializable_params[key] = {
                        str(k): float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_params[key] = float(value)
                else:
                    serializable_params[key] = value

            json.dump(serializable_params, f, indent=2)

        logger.info(f"Saved scaling parameters to {filepath}")

    def load_params(self, filepath: str) -> None:
        """
        Load scaling parameters from a file.

        Args:
            filepath: Path to load the parameters from (JSON format)
        """
        import json

        with open(filepath, "r") as f:
            self.scaling_params = json.load(f)

        logger.info(f"Loaded scaling parameters from {filepath}")
