"""
Data imputation module for handling missing values with domain-specific strategies.
"""
import logging
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


ImputationStrategy = Literal["mean", "median", "mode", "forward_fill", "backward_fill", "constant"]


class ImputationConfig:
    """Configuration for imputation strategies per column."""

    def __init__(self, strategies: Optional[Dict[str, ImputationStrategy]] = None):
        """
        Initialize imputation configuration.

        Args:
            strategies: Dict mapping column names to imputation strategies
        """
        self.strategies = strategies or {}

    def set_strategy(self, column: str, strategy: ImputationStrategy) -> None:
        """Set imputation strategy for a column."""
        self.strategies[column] = strategy

    def get_strategy(self, column: str) -> Optional[ImputationStrategy]:
        """Get imputation strategy for a column."""
        return self.strategies.get(column)


class Imputer:
    """
    Handles missing value imputation using various domain-specific strategies.
    """

    def __init__(self):
        """Initialize Imputer."""
        self.imputation_stats: Dict = {}
        self.fitted_params: Dict = {}

    def impute_missing(
        self, df: DataFrame, config: ImputationConfig, fit: bool = True
    ) -> DataFrame:
        """
        Impute missing values using configured strategies.

        Args:
            df: Input DataFrame
            config: ImputationConfig specifying strategies per column
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with missing values imputed
        """
        df_result = df.copy()
        imputation_counts = {}

        for col in df_result.columns:
            missing_count = df_result[col].isna().sum()
            if missing_count == 0:
                continue

            strategy = config.get_strategy(col)
            if strategy is None:
                logger.debug(f"No imputation strategy specified for {col}. Skipping.")
                continue

            logger.info(
                f"Imputing {missing_count} missing values in {col} using {strategy} strategy"
            )

            if strategy == "mean":
                df_result[col] = self._impute_mean(df_result[col], col, fit)
            elif strategy == "median":
                df_result[col] = self._impute_median(df_result[col], col, fit)
            elif strategy == "mode":
                df_result[col] = self._impute_mode(df_result[col], col, fit)
            elif strategy == "forward_fill":
                df_result[col] = df_result[col].fillna(method="ffill")
            elif strategy == "backward_fill":
                df_result[col] = df_result[col].fillna(method="bfill")
            elif strategy == "constant":
                df_result[col] = df_result[col].fillna(0)
            else:
                logger.warning(f"Unknown strategy {strategy} for {col}. Skipping.")
                continue

            imputed_count = missing_count - df_result[col].isna().sum()
            imputation_counts[col] = imputed_count

        self.imputation_stats["imputed_values"] = imputation_counts

        return df_result

    def _impute_mean(self, series: pd.Series, col: str, fit: bool) -> pd.Series:
        """Impute using mean value."""
        if fit:
            mean_val = series.mean()
            self.fitted_params[f"{col}_mean"] = mean_val
        else:
            mean_val = self.fitted_params.get(f"{col}_mean")
            if mean_val is None:
                logger.warning(f"No fitted mean for {col}. Using current mean.")
                mean_val = series.mean()

        return series.fillna(mean_val)

    def _impute_median(self, series: pd.Series, col: str, fit: bool) -> pd.Series:
        """Impute using median value."""
        if fit:
            median_val = series.median()
            self.fitted_params[f"{col}_median"] = median_val
        else:
            median_val = self.fitted_params.get(f"{col}_median")
            if median_val is None:
                logger.warning(f"No fitted median for {col}. Using current median.")
                median_val = series.median()

        return series.fillna(median_val)

    def _impute_mode(self, series: pd.Series, col: str, fit: bool) -> pd.Series:
        """Impute using mode value."""
        if fit:
            mode_val = series.mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
            else:
                mode_val = None
            self.fitted_params[f"{col}_mode"] = mode_val
        else:
            mode_val = self.fitted_params.get(f"{col}_mode")
            if mode_val is None:
                logger.warning(f"No fitted mode for {col}. Using current mode.")
                mode_val = series.mode()
                if len(mode_val) > 0:
                    mode_val = mode_val[0]

        if mode_val is not None:
            return series.fillna(mode_val)
        return series

    def forward_fill_timeseries(
        self,
        df: DataFrame,
        columns: List[str],
        group_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> DataFrame:
        """
        Forward fill missing values in time-series data for temporal continuity.

        Args:
            df: Input DataFrame (should be sorted by time)
            columns: List of columns to forward fill
            group_by: Optional column to group by (e.g., anonymized_id)
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            DataFrame with forward-filled values
        """
        df_result = df.copy()

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping forward fill.")
                continue

            missing_before = df_result[col].isna().sum()

            if group_by and group_by in df_result.columns:
                # Forward fill within each group
                df_result[col] = df_result.groupby(group_by)[col].fillna(
                    method="ffill", limit=limit
                )
            else:
                # Forward fill across entire column
                df_result[col] = df_result[col].fillna(method="ffill", limit=limit)

            missing_after = df_result[col].isna().sum()
            filled_count = missing_before - missing_after

            logger.info(f"Forward filled {filled_count} values in {col}")

        return df_result

    def impute_with_median(
        self, df: DataFrame, columns: List[str], fit: bool = True
    ) -> DataFrame:
        """
        Impute missing values with median for physiological metrics.

        Args:
            df: Input DataFrame
            columns: List of columns to impute with median
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with median-imputed values
        """
        df_result = df.copy()
        imputation_counts = {}

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping median imputation.")
                continue

            missing_count = df_result[col].isna().sum()
            if missing_count == 0:
                continue

            if fit:
                median_val = df_result[col].median()
                self.fitted_params[f"{col}_median"] = median_val
            else:
                median_val = self.fitted_params.get(f"{col}_median")
                if median_val is None:
                    logger.warning(f"No fitted median for {col}. Using current median.")
                    median_val = df_result[col].median()

            df_result[col] = df_result[col].fillna(median_val)

            imputed_count = missing_count - df_result[col].isna().sum()
            imputation_counts[col] = imputed_count

            logger.info(f"Imputed {imputed_count} values in {col} with median {median_val:.2f}")

        self.imputation_stats["median_imputed"] = imputation_counts

        return df_result

    def get_imputation_stats(self) -> Dict:
        """
        Get statistics from the last imputation operation.

        Returns:
            Dictionary containing imputation statistics
        """
        return self.imputation_stats.copy()

    def get_fitted_params(self) -> Dict:
        """
        Get fitted parameters for inference.

        Returns:
            Dictionary containing fitted parameters
        """
        return self.fitted_params.copy()

    def set_fitted_params(self, params: Dict) -> None:
        """
        Set fitted parameters for inference.

        Args:
            params: Dictionary of fitted parameters
        """
        self.fitted_params = params.copy()

    def reset_stats(self) -> None:
        """Reset imputation statistics."""
        self.imputation_stats = {}
