"""
ETL Pipeline module for orchestrating data cleaning, imputation, encoding, and normalization.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame

from src.exceptions import DataProcessingError, ValidationError
from src.processing.cleaning import DataCleaner
from src.processing.encoding import Encoder
from src.processing.imputation import Imputer, ImputationConfig
from src.processing.normalization import Normalizer

logger = logging.getLogger(__name__)


class ETLPipelineConfig:
    """Configuration for ETL pipeline stages."""

    def __init__(
        self,
        # Cleaning config
        outlier_method: str = "iqr",
        iqr_multiplier: float = 1.5,
        domain_rules: Optional[Dict[str, Tuple[float, float]]] = None,
        # Imputation config
        imputation_strategies: Optional[Dict[str, str]] = None,
        # Encoding config
        target_column: Optional[str] = None,
        target_encode_columns: Optional[List[str]] = None,
        one_hot_encode_columns: Optional[List[str]] = None,
        ordinal_encode_columns: Optional[List[str]] = None,
        ordinal_ordering: Optional[Dict[str, List]] = None,
        # Normalization config
        standardize_columns: Optional[List[str]] = None,
        normalize_timeseries_columns: Optional[List[str]] = None,
        min_max_columns: Optional[List[str]] = None,
        group_by_column: str = "anonymized_id",
    ):
        """
        Initialize ETL pipeline configuration.

        Args:
            outlier_method: Method for outlier detection ('iqr' or 'domain')
            iqr_multiplier: Multiplier for IQR method
            domain_rules: Domain-specific rules for outlier detection
            imputation_strategies: Dict mapping columns to imputation strategies
            target_column: Target column for target encoding
            target_encode_columns: Columns to target encode
            one_hot_encode_columns: Columns to one-hot encode
            ordinal_encode_columns: Columns to ordinal encode
            ordinal_ordering: Ordering for ordinal encoding
            standardize_columns: Columns to standardize
            normalize_timeseries_columns: Columns to normalize as time-series
            min_max_columns: Columns to min-max scale
            group_by_column: Column for grouping in time-series normalization
        """
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier
        self.domain_rules = domain_rules or {}
        self.imputation_strategies = imputation_strategies or {}
        self.target_column = target_column
        self.target_encode_columns = target_encode_columns or []
        self.one_hot_encode_columns = one_hot_encode_columns or []
        self.ordinal_encode_columns = ordinal_encode_columns or []
        self.ordinal_ordering = ordinal_ordering or {}
        self.standardize_columns = standardize_columns or []
        self.normalize_timeseries_columns = normalize_timeseries_columns or []
        self.min_max_columns = min_max_columns or []
        self.group_by_column = group_by_column


class ETLPipeline:
    """
    Orchestrates the complete ETL pipeline including cleaning, imputation,
    encoding, and normalization stages.
    """

    def __init__(self, config: ETLPipelineConfig):
        """
        Initialize ETL pipeline with configuration.

        Args:
            config: ETLPipelineConfig object with pipeline settings
        """
        self.config = config
        self.cleaner = DataCleaner(
            outlier_method=config.outlier_method, iqr_multiplier=config.iqr_multiplier
        )
        self.imputer = Imputer()
        self.encoder = Encoder()
        self.normalizer = Normalizer()

        self.pipeline_stats: Dict[str, Any] = {}
        self.is_fitted = False

    def fit_transform(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        timestamp_column: str = "timestamp",
    ) -> DataFrame:
        """
        Fit the pipeline on training data and transform it.

        Args:
            df: Input DataFrame
            id_column: Column name for anonymized identifier
            timestamp_column: Column name for timestamp

        Returns:
            Transformed DataFrame

        Raises:
            DataProcessingError: If pipeline processing fails
        """
        start_time = time.time()

        try:
            logger.info(f"Starting ETL pipeline fit_transform on {len(df)} records")

            # Stage 1: Data Cleaning
            df_cleaned = self._clean_data(df, id_column, timestamp_column)

            # Stage 2: Imputation
            df_imputed = self._impute_data(df_cleaned, fit=True)

            # Stage 3: Encoding
            df_encoded = self._encode_data(df_imputed, fit=True)

            # Stage 4: Normalization
            df_normalized = self._normalize_data(df_encoded, fit=True)

            self.is_fitted = True

            elapsed_time = time.time() - start_time
            self.pipeline_stats["fit_transform_time"] = elapsed_time
            self.pipeline_stats["input_records"] = len(df)
            self.pipeline_stats["output_records"] = len(df_normalized)

            logger.info(
                f"ETL pipeline fit_transform completed in {elapsed_time:.2f}s. "
                f"Processed {len(df)} -> {len(df_normalized)} records"
            )

            return df_normalized

        except Exception as e:
            logger.error(f"ETL pipeline fit_transform failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"ETL pipeline fit_transform failed: {str(e)}") from e

    def transform(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        timestamp_column: str = "timestamp",
    ) -> DataFrame:
        """
        Transform data using fitted pipeline parameters.

        Args:
            df: Input DataFrame
            id_column: Column name for anonymized identifier
            timestamp_column: Column name for timestamp

        Returns:
            Transformed DataFrame

        Raises:
            ValidationError: If pipeline is not fitted
            DataProcessingError: If pipeline processing fails
        """
        if not self.is_fitted:
            raise ValidationError("Pipeline must be fitted before transform. Call fit_transform first.")

        start_time = time.time()

        try:
            logger.info(f"Starting ETL pipeline transform on {len(df)} records")

            # Stage 1: Data Cleaning
            df_cleaned = self._clean_data(df, id_column, timestamp_column)

            # Stage 2: Imputation
            df_imputed = self._impute_data(df_cleaned, fit=False)

            # Stage 3: Encoding
            df_encoded = self._encode_data(df_imputed, fit=False)

            # Stage 4: Normalization
            df_normalized = self._normalize_data(df_encoded, fit=False)

            elapsed_time = time.time() - start_time
            self.pipeline_stats["transform_time"] = elapsed_time
            self.pipeline_stats["input_records"] = len(df)
            self.pipeline_stats["output_records"] = len(df_normalized)

            logger.info(
                f"ETL pipeline transform completed in {elapsed_time:.2f}s. "
                f"Processed {len(df)} -> {len(df_normalized)} records"
            )

            return df_normalized

        except Exception as e:
            logger.error(f"ETL pipeline transform failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"ETL pipeline transform failed: {str(e)}") from e

    def process_batch(
        self,
        df: DataFrame,
        batch_size: int = 1000,
        id_column: str = "anonymized_id",
        timestamp_column: str = "timestamp",
        fit: bool = False,
    ) -> DataFrame:
        """
        Process data in batches for memory efficiency.

        Args:
            df: Input DataFrame
            batch_size: Number of records per batch
            id_column: Column name for anonymized identifier
            timestamp_column: Column name for timestamp
            fit: Whether to fit pipeline parameters

        Returns:
            Transformed DataFrame

        Raises:
            DataProcessingError: If batch processing fails
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting batch processing: {len(df)} records in batches of {batch_size}"
            )

            # If fitting, process all data at once to ensure consistent parameters
            if fit:
                logger.info("Fitting mode: processing all data together")
                return self.fit_transform(df, id_column, timestamp_column)

            # For transform, process in batches
            if not self.is_fitted:
                raise ValidationError("Pipeline must be fitted before batch transform.")

            batches = []
            num_batches = (len(df) + batch_size - 1) // batch_size

            for i in range(0, len(df), batch_size):
                batch_num = i // batch_size + 1
                batch_df = df.iloc[i : i + batch_size]

                logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch_df)} records)")

                transformed_batch = self.transform(batch_df, id_column, timestamp_column)
                batches.append(transformed_batch)

            # Combine all batches
            result_df = pd.concat(batches, ignore_index=True)

            elapsed_time = time.time() - start_time
            self.pipeline_stats["batch_processing_time"] = elapsed_time
            self.pipeline_stats["num_batches"] = num_batches
            self.pipeline_stats["batch_size"] = batch_size

            logger.info(
                f"Batch processing completed in {elapsed_time:.2f}s. "
                f"Processed {num_batches} batches, {len(result_df)} total records"
            )

            return result_df

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Batch processing failed: {str(e)}") from e

    def _clean_data(
        self, df: DataFrame, id_column: str, timestamp_column: str
    ) -> DataFrame:
        """
        Execute data cleaning stage.

        Args:
            df: Input DataFrame
            id_column: Column name for anonymized identifier
            timestamp_column: Column name for timestamp

        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("Stage 1: Data Cleaning")

            # Remove duplicates
            df_cleaned = self.cleaner.remove_duplicates(df, id_column, timestamp_column)

            # Detect outliers for numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                df_cleaned = self.cleaner.detect_outliers(
                    df_cleaned, numeric_cols, self.config.domain_rules
                )

            # Handle invalid values
            df_cleaned = self.cleaner.handle_invalid_values(df_cleaned)

            cleaning_stats = self.cleaner.get_cleaning_stats()
            self.pipeline_stats["cleaning"] = cleaning_stats

            logger.info(f"Cleaning completed: {cleaning_stats}")

            return df_cleaned

        except Exception as e:
            logger.error(f"Data cleaning stage failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Data cleaning failed: {str(e)}") from e

    def _impute_data(self, df: DataFrame, fit: bool) -> DataFrame:
        """
        Execute imputation stage.

        Args:
            df: Input DataFrame
            fit: Whether to fit imputation parameters

        Returns:
            Imputed DataFrame
        """
        try:
            logger.info(f"Stage 2: Imputation (fit={fit})")

            # Create imputation config
            imputation_config = ImputationConfig(self.config.imputation_strategies)

            # Apply imputation
            df_imputed = self.imputer.impute_missing(df, imputation_config, fit=fit)

            imputation_stats = self.imputer.get_imputation_stats()
            self.pipeline_stats["imputation"] = imputation_stats

            logger.info(f"Imputation completed: {imputation_stats}")

            return df_imputed

        except Exception as e:
            logger.error(f"Imputation stage failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Imputation failed: {str(e)}") from e

    def _encode_data(self, df: DataFrame, fit: bool) -> DataFrame:
        """
        Execute encoding stage.

        Args:
            df: Input DataFrame
            fit: Whether to fit encoding parameters

        Returns:
            Encoded DataFrame
        """
        try:
            logger.info(f"Stage 3: Encoding (fit={fit})")

            df_encoded = df.copy()

            # Target encoding
            if self.config.target_encode_columns and self.config.target_column:
                df_encoded = self.encoder.target_encode(
                    df_encoded,
                    self.config.target_encode_columns,
                    self.config.target_column,
                    fit=fit,
                )

            # One-hot encoding
            if self.config.one_hot_encode_columns:
                df_encoded = self.encoder.one_hot_encode(
                    df_encoded, self.config.one_hot_encode_columns, fit=fit
                )

            # Ordinal encoding
            if self.config.ordinal_encode_columns:
                df_encoded = self.encoder.ordinal_encode(
                    df_encoded,
                    self.config.ordinal_encode_columns,
                    self.config.ordinal_ordering,
                    fit=fit,
                )

            encoding_stats = self.encoder.get_encoding_stats()
            self.pipeline_stats["encoding"] = encoding_stats

            logger.info(f"Encoding completed: {encoding_stats}")

            return df_encoded

        except Exception as e:
            logger.error(f"Encoding stage failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Encoding failed: {str(e)}") from e

    def _normalize_data(self, df: DataFrame, fit: bool) -> DataFrame:
        """
        Execute normalization stage.

        Args:
            df: Input DataFrame
            fit: Whether to fit normalization parameters

        Returns:
            Normalized DataFrame
        """
        try:
            logger.info(f"Stage 4: Normalization (fit={fit})")

            df_normalized = df.copy()

            # Time-series normalization
            if self.config.normalize_timeseries_columns:
                df_normalized = self.normalizer.normalize_timeseries(
                    df_normalized,
                    self.config.normalize_timeseries_columns,
                    self.config.group_by_column,
                    fit=fit,
                )

            # Standardization
            if self.config.standardize_columns:
                df_normalized = self.normalizer.standardize_features(
                    df_normalized, self.config.standardize_columns, fit=fit
                )

            # Min-max scaling
            if self.config.min_max_columns:
                df_normalized = self.normalizer.min_max_scale(
                    df_normalized, self.config.min_max_columns, fit=fit
                )

            normalization_stats = self.normalizer.get_normalization_stats()
            self.pipeline_stats["normalization"] = normalization_stats

            logger.info(f"Normalization completed: {normalization_stats}")

            return df_normalized

        except Exception as e:
            logger.error(f"Normalization stage failed: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Normalization failed: {str(e)}") from e

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all pipeline stages.

        Returns:
            Dictionary containing statistics from all stages
        """
        return self.pipeline_stats.copy()

    def get_fitted_params(self) -> Dict[str, Any]:
        """
        Get all fitted parameters from pipeline components.

        Returns:
            Dictionary containing fitted parameters from all components
        """
        return {
            "imputer_params": self.imputer.get_fitted_params(),
            "encoder_mappings": self.encoder.get_encoding_mappings(),
            "normalizer_params": self.normalizer.get_scaling_params(),
        }

    def set_fitted_params(self, params: Dict[str, Any]) -> None:
        """
        Set fitted parameters for all pipeline components.

        Args:
            params: Dictionary containing fitted parameters
        """
        if "imputer_params" in params:
            self.imputer.set_fitted_params(params["imputer_params"])

        if "encoder_mappings" in params:
            self.encoder.set_encoding_mappings(params["encoder_mappings"])

        if "normalizer_params" in params:
            self.normalizer.set_scaling_params(params["normalizer_params"])

        self.is_fitted = True
        logger.info("Loaded fitted parameters into pipeline")

    def save_pipeline(self, filepath: str) -> None:
        """
        Save pipeline configuration and fitted parameters.

        Args:
            filepath: Path to save the pipeline (JSON format)
        """
        import json

        pipeline_data = {
            "config": {
                "outlier_method": self.config.outlier_method,
                "iqr_multiplier": self.config.iqr_multiplier,
                "domain_rules": self.config.domain_rules,
                "imputation_strategies": self.config.imputation_strategies,
                "target_column": self.config.target_column,
                "target_encode_columns": self.config.target_encode_columns,
                "one_hot_encode_columns": self.config.one_hot_encode_columns,
                "ordinal_encode_columns": self.config.ordinal_encode_columns,
                "ordinal_ordering": self.config.ordinal_ordering,
                "standardize_columns": self.config.standardize_columns,
                "normalize_timeseries_columns": self.config.normalize_timeseries_columns,
                "min_max_columns": self.config.min_max_columns,
                "group_by_column": self.config.group_by_column,
            },
            "fitted_params": self.get_fitted_params(),
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "w") as f:
            json.dump(pipeline_data, f, indent=2)

        logger.info(f"Saved pipeline to {filepath}")

    @classmethod
    def load_pipeline(cls, filepath: str) -> "ETLPipeline":
        """
        Load pipeline configuration and fitted parameters.

        Args:
            filepath: Path to load the pipeline from (JSON format)

        Returns:
            ETLPipeline instance with loaded configuration and parameters
        """
        import json

        with open(filepath, "r") as f:
            pipeline_data = json.load(f)

        # Reconstruct config
        config = ETLPipelineConfig(**pipeline_data["config"])

        # Create pipeline
        pipeline = cls(config)

        # Load fitted parameters
        if pipeline_data.get("is_fitted", False):
            pipeline.set_fitted_params(pipeline_data["fitted_params"])

        logger.info(f"Loaded pipeline from {filepath}")

        return pipeline
