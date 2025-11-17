"""
Categorical encoding module for transforming categorical variables.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class Encoder:
    """
    Handles categorical variable encoding using various strategies including
    target encoding, one-hot encoding, and ordinal encoding.
    """

    def __init__(self):
        """Initialize Encoder."""
        self.encoding_mappings: Dict = {}
        self.encoding_stats: Dict = {}

    def target_encode(
        self,
        df: DataFrame,
        columns: List[str],
        target: str,
        smoothing: float = 1.0,
        fit: bool = True,
    ) -> DataFrame:
        """
        Apply target encoding with smoothing to prevent overfitting.

        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            target: Target column name for computing means
            smoothing: Smoothing parameter to prevent overfitting
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with target-encoded features
        """
        df_result = df.copy()
        encoding_counts = {}

        if target not in df.columns:
            logger.error(f"Target column {target} not found in DataFrame.")
            return df_result

        global_mean = df[target].mean() if fit else self.encoding_mappings.get("global_mean", 0)

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping target encoding.")
                continue

            if fit:
                # Calculate target mean per category with smoothing
                category_stats = df.groupby(col)[target].agg(["mean", "count"])
                category_means = (
                    category_stats["count"] * category_stats["mean"]
                    + smoothing * global_mean
                ) / (category_stats["count"] + smoothing)

                self.encoding_mappings[f"{col}_target_encoding"] = category_means.to_dict()
                self.encoding_mappings["global_mean"] = global_mean
            else:
                # Use fitted mappings
                category_means = self.encoding_mappings.get(f"{col}_target_encoding", {})

            # Apply encoding
            encoded_col = f"{col}_encoded"
            df_result[encoded_col] = df_result[col].map(category_means).fillna(global_mean)

            encoding_counts[col] = len(df_result[encoded_col].dropna())
            logger.info(f"Target encoded {encoding_counts[col]} values in {col}")

        self.encoding_stats["target_encoded"] = encoding_counts

        return df_result

    def one_hot_encode(
        self, df: DataFrame, columns: List[str], drop_first: bool = True, fit: bool = True
    ) -> DataFrame:
        """
        Apply one-hot encoding for low-cardinality categorical variables.

        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            drop_first: Whether to drop first category to avoid multicollinearity
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with one-hot encoded features
        """
        df_result = df.copy()
        encoding_counts = {}

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping one-hot encoding.")
                continue

            if fit:
                # Get unique categories
                categories = df_result[col].unique().tolist()
                self.encoding_mappings[f"{col}_categories"] = categories
            else:
                # Use fitted categories
                categories = self.encoding_mappings.get(f"{col}_categories", [])

            # Apply one-hot encoding
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)

            # Ensure all expected columns exist (for inference)
            if not fit and categories:
                expected_cols = [f"{col}_{cat}" for cat in categories]
                if drop_first and expected_cols:
                    expected_cols = expected_cols[1:]

                for exp_col in expected_cols:
                    if exp_col not in dummies.columns:
                        dummies[exp_col] = 0

            df_result = pd.concat([df_result, dummies], axis=1)
            encoding_counts[col] = len(dummies.columns)

            logger.info(f"One-hot encoded {col} into {len(dummies.columns)} columns")

        self.encoding_stats["one_hot_encoded"] = encoding_counts

        return df_result

    def ordinal_encode(
        self, df: DataFrame, columns: List[str], ordering: Dict[str, List], fit: bool = True
    ) -> DataFrame:
        """
        Apply ordinal encoding for ordered categorical variables.

        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            ordering: Dict mapping column names to ordered category lists
            fit: Whether to fit parameters (True for training, False for inference)

        Returns:
            DataFrame with ordinal encoded features
        """
        df_result = df.copy()
        encoding_counts = {}

        for col in columns:
            if col not in df_result.columns:
                logger.warning(f"Column {col} not found. Skipping ordinal encoding.")
                continue

            if col not in ordering:
                logger.warning(f"No ordering specified for {col}. Skipping.")
                continue

            order_list = ordering[col]

            if fit:
                # Store ordering mapping
                order_mapping = {cat: idx for idx, cat in enumerate(order_list)}
                self.encoding_mappings[f"{col}_ordinal_mapping"] = order_mapping
            else:
                # Use fitted mapping
                order_mapping = self.encoding_mappings.get(f"{col}_ordinal_mapping", {})

            # Apply ordinal encoding
            encoded_col = f"{col}_ordinal"
            df_result[encoded_col] = df_result[col].map(order_mapping)

            # Handle unmapped categories
            unmapped_count = df_result[encoded_col].isna().sum()
            if unmapped_count > 0:
                logger.warning(
                    f"{unmapped_count} values in {col} not found in ordering. Setting to -1."
                )
                df_result[encoded_col] = df_result[encoded_col].fillna(-1)

            encoding_counts[col] = len(df_result[encoded_col].dropna())
            logger.info(f"Ordinal encoded {encoding_counts[col]} values in {col}")

        self.encoding_stats["ordinal_encoded"] = encoding_counts

        return df_result

    def get_encoding_mappings(self) -> Dict:
        """
        Get fitted encoding mappings for inference.

        Returns:
            Dictionary containing all encoding mappings
        """
        return self.encoding_mappings.copy()

    def set_encoding_mappings(self, mappings: Dict) -> None:
        """
        Set encoding mappings for inference.

        Args:
            mappings: Dictionary of encoding mappings
        """
        self.encoding_mappings = mappings.copy()

    def get_encoding_stats(self) -> Dict:
        """
        Get statistics from the last encoding operation.

        Returns:
            Dictionary containing encoding statistics
        """
        return self.encoding_stats.copy()

    def reset_stats(self) -> None:
        """Reset encoding statistics."""
        self.encoding_stats = {}
