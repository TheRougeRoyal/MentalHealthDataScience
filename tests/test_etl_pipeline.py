"""Tests for ETL pipeline module"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
from src.exceptions import DataProcessingError, ValidationError


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "anonymized_id": ["user1"] * 50 + ["user2"] * 50,
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="H"),
            "heart_rate": np.random.normal(70, 10, 100),
            "sleep_hours": np.random.normal(7, 1.5, 100),
            "activity_level": np.random.choice(["low", "medium", "high"], 100),
            "mood_score": np.random.randint(1, 11, 100),
            "target": np.random.choice([0, 1], 100),
        }
    )


@pytest.fixture
def sample_config():
    """Create sample ETL pipeline configuration"""
    return ETLPipelineConfig(
        outlier_method="iqr",
        iqr_multiplier=1.5,
        imputation_strategies={
            "heart_rate": "median",
            "sleep_hours": "median",
            "mood_score": "forward_fill",
        },
        standardize_columns=["heart_rate", "sleep_hours", "mood_score"],
        group_by_column="anonymized_id",
    )


@pytest.fixture
def pipeline(sample_config):
    """Create ETL pipeline instance"""
    return ETLPipeline(sample_config)


class TestETLPipelineInitialization:
    """Test suite for ETL pipeline initialization"""

    def test_initialization(self, pipeline, sample_config):
        """Test pipeline initializes correctly"""
        assert pipeline is not None
        assert pipeline.config == sample_config
        assert pipeline.cleaner is not None
        assert pipeline.imputer is not None
        assert pipeline.encoder is not None
        assert pipeline.normalizer is not None
        assert not pipeline.is_fitted

    def test_config_initialization(self):
        """Test pipeline config initialization with defaults"""
        config = ETLPipelineConfig()
        assert config.outlier_method == "iqr"
        assert config.iqr_multiplier == 1.5
        assert config.imputation_strategies == {}
        assert config.group_by_column == "anonymized_id"


class TestFitTransform:
    """Test suite for fit_transform method"""

    def test_fit_transform_basic(self, pipeline, sample_dataframe):
        """Test basic fit_transform operation"""
        df = sample_dataframe.copy()
        result = pipeline.fit_transform(df)

        assert result is not None
        assert len(result) > 0
        assert pipeline.is_fitted
        assert "fit_transform_time" in pipeline.pipeline_stats

    def test_fit_transform_with_duplicates(self, pipeline):
        """Test fit_transform removes duplicates"""
        df = pd.DataFrame(
            {
                "anonymized_id": ["user1", "user1", "user2"],
                "timestamp": ["2025-01-01", "2025-01-01", "2025-01-02"],
                "value": [1.0, 1.0, 2.0],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        result = pipeline.fit_transform(df)

        # Should remove one duplicate
        assert len(result) == 2

    def test_fit_transform_with_missing_values(self, pipeline):
        """Test fit_transform handles missing values"""
        df = pd.DataFrame(
            {
                "anonymized_id": ["user1"] * 5,
                "timestamp": pd.date_range("2025-01-01", periods=5),
                "heart_rate": [70.0, np.nan, 75.0, np.nan, 80.0],
                "sleep_hours": [7.0, 8.0, np.nan, 7.5, 8.5],
            }
        )

        result = pipeline.fit_transform(df)

        # Missing values should be imputed
        assert result["heart_rate"].isna().sum() < df["heart_rate"].isna().sum()

    def test_fit_transform_preserves_records(self, pipeline, sample_dataframe):
        """Test fit_transform preserves record count (after duplicate removal)"""
        df = sample_dataframe.copy()
        initial_count = len(df)

        result = pipeline.fit_transform(df)

        # Should have same or fewer records (due to duplicate removal)
        assert len(result) <= initial_count

    def test_fit_transform_stores_stats(self, pipeline, sample_dataframe):
        """Test fit_transform stores comprehensive statistics"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        stats = pipeline.get_pipeline_stats()
        assert "cleaning" in stats
        assert "imputation" in stats
        assert "normalization" in stats
        assert "input_records" in stats
        assert "output_records" in stats


class TestTransform:
    """Test suite for transform method"""

    def test_transform_requires_fitting(self, pipeline, sample_dataframe):
        """Test transform raises error if not fitted"""
        df = sample_dataframe.copy()

        with pytest.raises(ValidationError):
            pipeline.transform(df)

    def test_transform_after_fit(self, pipeline, sample_dataframe):
        """Test transform works after fitting"""
        df_train = sample_dataframe.copy()
        pipeline.fit_transform(df_train)

        df_test = sample_dataframe.iloc[:10].copy()
        result = pipeline.transform(df_test)

        assert result is not None
        assert len(result) == len(df_test)

    def test_transform_uses_fitted_params(self, pipeline, sample_dataframe):
        """Test transform uses fitted parameters"""
        df_train = sample_dataframe.copy()
        pipeline.fit_transform(df_train)

        # Get fitted params
        fitted_params = pipeline.get_fitted_params()
        assert len(fitted_params) > 0

        # Transform new data
        df_test = sample_dataframe.iloc[:10].copy()
        result = pipeline.transform(df_test)

        # Fitted params should remain unchanged
        assert pipeline.get_fitted_params() == fitted_params


class TestBatchProcessing:
    """Test suite for batch processing"""

    def test_process_batch_fit_mode(self, pipeline, sample_dataframe):
        """Test batch processing in fit mode"""
        df = sample_dataframe.copy()
        result = pipeline.process_batch(df, batch_size=50, fit=True)

        assert result is not None
        assert len(result) > 0
        assert pipeline.is_fitted

    def test_process_batch_transform_mode(self, pipeline, sample_dataframe):
        """Test batch processing in transform mode"""
        df = sample_dataframe.copy()

        # First fit
        pipeline.fit_transform(df)

        # Then batch transform
        result = pipeline.process_batch(df, batch_size=30, fit=False)

        assert result is not None
        assert len(result) > 0
        assert "batch_processing_time" in pipeline.pipeline_stats
        assert "num_batches" in pipeline.pipeline_stats

    def test_process_batch_requires_fitting_for_transform(self, pipeline, sample_dataframe):
        """Test batch transform requires fitting"""
        df = sample_dataframe.copy()

        with pytest.raises(ValidationError):
            pipeline.process_batch(df, batch_size=50, fit=False)

    def test_process_batch_handles_small_batches(self, pipeline, sample_dataframe):
        """Test batch processing with small batch size"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        result = pipeline.process_batch(df, batch_size=10, fit=False)

        assert len(result) == len(df)
        assert pipeline.pipeline_stats["num_batches"] > 1

    def test_process_batch_handles_large_batches(self, pipeline, sample_dataframe):
        """Test batch processing with large batch size"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        result = pipeline.process_batch(df, batch_size=1000, fit=False)

        assert len(result) == len(df)
        assert pipeline.pipeline_stats["num_batches"] == 1


class TestPipelineStages:
    """Test suite for individual pipeline stages"""

    def test_cleaning_stage(self, pipeline, sample_dataframe):
        """Test cleaning stage execution"""
        df = sample_dataframe.copy()
        result = pipeline._clean_data(df, "anonymized_id", "timestamp")

        assert result is not None
        assert "cleaning" in pipeline.pipeline_stats

    def test_imputation_stage(self, pipeline, sample_dataframe):
        """Test imputation stage execution"""
        df = sample_dataframe.copy()
        # Add some NaN values
        df.loc[0, "heart_rate"] = np.nan

        result = pipeline._impute_data(df, fit=True)

        assert result is not None
        assert "imputation" in pipeline.pipeline_stats

    def test_encoding_stage(self, pipeline, sample_dataframe):
        """Test encoding stage execution"""
        df = sample_dataframe.copy()
        result = pipeline._encode_data(df, fit=True)

        assert result is not None
        assert "encoding" in pipeline.pipeline_stats

    def test_normalization_stage(self, pipeline, sample_dataframe):
        """Test normalization stage execution"""
        df = sample_dataframe.copy()
        result = pipeline._normalize_data(df, fit=True)

        assert result is not None
        assert "normalization" in pipeline.pipeline_stats


class TestParameterPersistence:
    """Test suite for parameter saving and loading"""

    def test_get_fitted_params(self, pipeline, sample_dataframe):
        """Test getting fitted parameters"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        params = pipeline.get_fitted_params()
        assert "imputer_params" in params
        assert "encoder_mappings" in params
        assert "normalizer_params" in params

    def test_set_fitted_params(self, pipeline, sample_dataframe):
        """Test setting fitted parameters"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        params = pipeline.get_fitted_params()

        # Create new pipeline and set params
        new_pipeline = ETLPipeline(pipeline.config)
        new_pipeline.set_fitted_params(params)

        assert new_pipeline.is_fitted

    def test_save_and_load_pipeline(self, pipeline, sample_dataframe):
        """Test saving and loading complete pipeline"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save pipeline
            pipeline.save_pipeline(filepath)

            # Load pipeline
            loaded_pipeline = ETLPipeline.load_pipeline(filepath)

            assert loaded_pipeline.is_fitted
            assert loaded_pipeline.config.outlier_method == pipeline.config.outlier_method

            # Test that loaded pipeline can transform
            df_test = sample_dataframe.iloc[:10].copy()
            result = loaded_pipeline.transform(df_test)
            assert result is not None

        finally:
            Path(filepath).unlink()


class TestPerformance:
    """Test suite for performance requirements"""

    def test_batch_processing_performance(self, pipeline):
        """Test that batch processing meets performance requirements"""
        # Create 1000 records as per requirement
        df = pd.DataFrame(
            {
                "anonymized_id": [f"user{i % 10}" for i in range(1000)],
                "timestamp": pd.date_range("2025-01-01", periods=1000, freq="H"),
                "heart_rate": np.random.normal(70, 10, 1000),
                "sleep_hours": np.random.normal(7, 1.5, 1000),
                "mood_score": np.random.randint(1, 11, 1000),
            }
        )

        import time

        start_time = time.time()
        result = pipeline.fit_transform(df)
        elapsed_time = time.time() - start_time

        # Should complete within 60 seconds for 1000 records
        assert elapsed_time < 60.0
        assert len(result) > 0


class TestErrorHandling:
    """Test suite for error handling"""

    def test_fit_transform_handles_errors(self, pipeline):
        """Test fit_transform handles processing errors"""
        # Create invalid dataframe
        df = pd.DataFrame({"invalid": [1, 2, 3]})

        # Should raise DataProcessingError
        with pytest.raises(DataProcessingError):
            pipeline.fit_transform(df)

    def test_transform_handles_errors(self, pipeline, sample_dataframe):
        """Test transform handles processing errors"""
        df = sample_dataframe.copy()
        pipeline.fit_transform(df)

        # Create invalid test dataframe
        df_test = pd.DataFrame({"invalid": [1, 2, 3]})

        with pytest.raises(DataProcessingError):
            pipeline.transform(df_test)


class TestEdgeCases:
    """Test suite for edge cases"""

    def test_empty_dataframe(self, pipeline):
        """Test pipeline handles empty dataframe"""
        df = pd.DataFrame(
            {
                "anonymized_id": [],
                "timestamp": [],
                "heart_rate": [],
            }
        )

        result = pipeline.fit_transform(df)
        assert len(result) == 0

    def test_single_record(self, pipeline):
        """Test pipeline handles single record"""
        df = pd.DataFrame(
            {
                "anonymized_id": ["user1"],
                "timestamp": [pd.Timestamp("2025-01-01")],
                "heart_rate": [70.0],
            }
        )

        result = pipeline.fit_transform(df)
        assert len(result) == 1

    def test_all_missing_values(self, pipeline):
        """Test pipeline handles all missing values in a column"""
        df = pd.DataFrame(
            {
                "anonymized_id": ["user1"] * 5,
                "timestamp": pd.date_range("2025-01-01", periods=5),
                "heart_rate": [np.nan] * 5,
            }
        )

        result = pipeline.fit_transform(df)
        assert result is not None
