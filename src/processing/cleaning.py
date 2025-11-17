"""
Data cleaning module for removing duplicates and handling data quality issues.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations including duplicate removal,
    outlier detection, and invalid value handling.
    """

    def __init__(self, outlier_method: str = "iqr", iqr_multiplier: float = 1.5):
        """
        Initialize DataCleaner with configuration.

        Args:
            outlier_method: Method for outlier detection ('iqr' or 'domain')
            iqr_multiplier: Multiplier for IQR method (default 1.5)
        """
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier
        self.cleaning_stats: Dict = {}

    def remove_duplicates(
        self, df: DataFrame, id_column: str = "anonymized_id", timestamp_column: str = "timestamp"
    ) -> DataFrame:
        """
        Remove duplicate records based on anonymized ID and timestamp.

        Args:
            df: Input DataFrame
            id_column: Column name for anonymized identifier
            timestamp_column: Column name for timestamp

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)

        # Check if required columns exist
        if id_column not in df.columns or timestamp_column not in df.columns:
            logger.warning(
                f"Required columns {id_column} or {timestamp_column} not found. "
                "Skipping duplicate removal."
            )
            return df

        # Remove exact duplicates based on ID and timestamp
        df_cleaned = df.drop_duplicates(subset=[id_column, timestamp_column], keep="first")

        duplicates_removed = initial_count - len(df_cleaned)
        self.cleaning_stats["duplicates_removed"] = duplicates_removed

        logger.info(
            f"Removed {duplicates_removed} duplicate records "
            f"({duplicates_removed / initial_count * 100:.2f}%)"
        )

        return df_cleaned.reset_index(drop=True)

    def detect_outliers(
        self,
        df: DataFrame,
        columns: List[str],
        domain_rules: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> DataFrame:
        """
        Detect and flag outliers using IQR method or domain-specific rules.

        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            domain_rules: Optional dict mapping column names to (min, max) valid ranges

        Returns:
            DataFrame with outlier flags added
        """
        df_result = df.copy()
        outlier_counts = {}

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame. Skipping.")
                continue

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric. Skipping outlier detection.")
                continue

            outlier_mask = pd.Series([False] * len(df), index=df.index)

            # Apply domain rules if provided
            if domain_rules and col in domain_rules:
                min_val, max_val = domain_rules[col]
                outlier_mask = (df[col] < min_val) | (df[col] > max_val)
                logger.info(
                    f"Applied domain rules for {col}: valid range [{min_val}, {max_val}]"
                )

            # Apply IQR method if no domain rules or as additional check
            elif self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                logger.info(
                    f"Applied IQR method for {col}: bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

            # Add outlier flag column
            flag_col = f"{col}_outlier"
            df_result[flag_col] = outlier_mask

            outlier_count = outlier_mask.sum()
            outlier_counts[col] = outlier_count

            if outlier_count > 0:
                logger.warning(
                    f"Detected {outlier_count} outliers in {col} "
                    f"({outlier_count / len(df) * 100:.2f}%)"
                )

        self.cleaning_stats["outliers_detected"] = outlier_counts

        return df_result

    def handle_invalid_values(
        self, df: DataFrame, invalid_values: Optional[List] = None
    ) -> DataFrame:
        """
        Replace invalid values with NaN for subsequent imputation.

        Args:
            df: Input DataFrame
            invalid_values: List of values to treat as invalid (default: common invalid markers)

        Returns:
            DataFrame with invalid values replaced by NaN
        """
        if invalid_values is None:
            # Common invalid value markers
            invalid_values = [
                "",
                " ",
                "NA",
                "N/A",
                "null",
                "NULL",
                "None",
                "none",
                "-",
                "?",
                "unknown",
                "Unknown",
                "UNKNOWN",
            ]

        df_result = df.copy()
        replacement_counts = {}

        for col in df_result.columns:
            initial_nulls = df_result[col].isna().sum()

            # Replace invalid string values
            if df_result[col].dtype == "object":
                df_result[col] = df_result[col].replace(invalid_values, np.nan)

            # Replace infinite values for numeric columns
            if pd.api.types.is_numeric_dtype(df_result[col]):
                df_result[col] = df_result[col].replace([np.inf, -np.inf], np.nan)

            final_nulls = df_result[col].isna().sum()
            replacements = final_nulls - initial_nulls

            if replacements > 0:
                replacement_counts[col] = replacements
                logger.info(f"Replaced {replacements} invalid values in {col} with NaN")

        self.cleaning_stats["invalid_values_replaced"] = replacement_counts

        return df_result

    def get_cleaning_stats(self) -> Dict:
        """
        Get statistics from the last cleaning operation.

        Returns:
            Dictionary containing cleaning statistics
        """
        return self.cleaning_stats.copy()

    def reset_stats(self) -> None:
        """Reset cleaning statistics."""
        self.cleaning_stats = {}
