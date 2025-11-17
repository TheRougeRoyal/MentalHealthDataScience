"""Tests for feature normalization module"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.processing.normalization import Normalizer


@pytest.fixture
def normalizer():
    """Create Normalizer instance for testing"""
    return Normalizer()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        "anonymized_id": ["user1", "user1", "user1", "user2", "user2", "user2"],
        "feature1": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        "feature2": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        "feature3": [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],  # Constant per group
        "timestamp": pd.date_range("2025-01-01", periods=6, freq="D")
    })


class TestNormalizerInitialization:
    """Test suite for Normalizer initialization"""
    
    def test_initialization(self, normalizer):
        """Test normalizer initializes correctly"""
        assert normalizer is not None
        assert normalizer.scaling_params == {}
        assert normalizer.normalization_stats == {}


class TestStandardizeFeatures:
    """Test suite for standardize_features method"""
    
    def test_standardize_single_column(self, normalizer, sample_dataframe):
        """Test standardization of a single column"""
        df = sample_dataframe.copy()
        result = normalizer.standardize_features(df, ["feature1"], fit=True)
        
        # Check that mean is approximately 0 and std is approximately 1
        assert abs(result["feature1"].mean()) < 1e-10
        assert abs(result["feature1"].std() - 1.0) < 1e-10
        
        # Check that parameters were stored
        assert "feature1_mean" in normalizer.scaling_params
        assert "feature1_std" in normalizer.scaling_params
    
    def test_standardize_multiple_columns(self, normalizer, sample_dataframe):
        """Test standardization of multiple columns"""
        df = sample_dataframe.copy()
        result = normalizer.standardize_features(df, ["feature1", "feature2"], fit=True)
        
        # Check both columns are standardized
        assert abs(result["feature1"].mean()) < 1e-10
        assert abs(result["feature2"].mean()) < 1e-10
        assert abs(result["feature1"].std() - 1.0) < 1e-10
        assert abs(result["feature2"].std() - 1.0) < 1e-10
    
    def test_standardize_with_fitted_params(self, normalizer, sample_dataframe):
        """Test standardization using pre-fitted parameters"""
        df = sample_dataframe.copy()
        
        # Fit on training data
        normalizer.standardize_features(df, ["feature1"], fit=True)
        
        # Apply to new data without fitting
        new_df = pd.DataFrame({"feature1": [15.0, 25.0, 35.0]})
        result = normalizer.standardize_features(new_df, ["feature1"], fit=False)
        
        # Check that the same parameters were used
        mean_val = normalizer.scaling_params["feature1_mean"]
        std_val = normalizer.scaling_params["feature1_std"]
        expected = (new_df["feature1"] - mean_val) / std_val
        pd.testing.assert_series_equal(result["feature1"], expected, check_names=False)
    
    def test_standardize_constant_column(self, normalizer):
        """Test standardization handles constant values"""
        df = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0]})
        result = normalizer.standardize_features(df, ["constant"], fit=True)
        
        # Should handle zero std gracefully
        assert not result["constant"].isna().any()
        assert normalizer.scaling_params["constant_std"] == 1.0


class TestMinMaxScale:
    """Test suite for min_max_scale method"""
    
    def test_min_max_scale_default_range(self, normalizer, sample_dataframe):
        """Test min-max scaling to default [0, 1] range"""
        df = sample_dataframe.copy()
        result = normalizer.min_max_scale(df, ["feature1"], fit=True)
        
        # Check values are in [0, 1] range
        assert result["feature1"].min() >= 0.0
        assert result["feature1"].max() <= 1.0
        assert abs(result["feature1"].min()) < 1e-10
        assert abs(result["feature1"].max() - 1.0) < 1e-10
    
    def test_min_max_scale_custom_range(self, normalizer, sample_dataframe):
        """Test min-max scaling to custom range"""
        df = sample_dataframe.copy()
        result = normalizer.min_max_scale(
            df, ["feature1"], feature_range=(-1, 1), fit=True
        )
        
        # Check values are in [-1, 1] range
        assert result["feature1"].min() >= -1.0
        assert result["feature1"].max() <= 1.0
        assert abs(result["feature1"].min() - (-1.0)) < 1e-10
        assert abs(result["feature1"].max() - 1.0) < 1e-10
    
    def test_min_max_scale_with_fitted_params(self, normalizer, sample_dataframe):
        """Test min-max scaling using pre-fitted parameters"""
        df = sample_dataframe.copy()
        
        # Fit on training data
        normalizer.min_max_scale(df, ["feature1"], fit=True)
        
        # Apply to new data
        new_df = pd.DataFrame({"feature1": [15.0, 25.0, 35.0]})
        result = normalizer.min_max_scale(new_df, ["feature1"], fit=False)
        
        # Check that the same parameters were used
        min_val = normalizer.scaling_params["feature1_min"]
        max_val = normalizer.scaling_params["feature1_max"]
        expected = (new_df["feature1"] - min_val) / (max_val - min_val)
        pd.testing.assert_series_equal(result["feature1"], expected, check_names=False)
    
    def test_min_max_scale_constant_column(self, normalizer):
        """Test min-max scaling handles constant values"""
        df = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0]})
        result = normalizer.min_max_scale(df, ["constant"], fit=True)
        
        # Should handle constant values gracefully
        assert not result["constant"].isna().any()


class TestNormalizeTimeseries:
    """Test suite for normalize_timeseries method"""
    
    def test_normalize_timeseries_per_individual(self, normalizer, sample_dataframe):
        """Test per-individual time-series normalization"""
        df = sample_dataframe.copy()
        result = normalizer.normalize_timeseries(
            df, ["feature1"], group_by="anonymized_id", fit=True
        )
        
        # Check that each individual's data is standardized separately
        user1_data = result[result["anonymized_id"] == "user1"]["feature1"]
        user2_data = result[result["anonymized_id"] == "user2"]["feature1"]
        
        # Each group should have mean ~0 and std ~1
        assert abs(user1_data.mean()) < 1e-10
        assert abs(user2_data.mean()) < 1e-10
        assert abs(user1_data.std() - 1.0) < 1e-10
        assert abs(user2_data.std() - 1.0) < 1e-10
    
    def test_normalize_timeseries_preserves_individual_patterns(self, normalizer):
        """Test that normalization preserves relative patterns within individuals"""
        df = pd.DataFrame({
            "anonymized_id": ["user1", "user1", "user1"],
            "value": [10.0, 20.0, 30.0]  # Linear increase
        })
        
        result = normalizer.normalize_timeseries(
            df, ["value"], group_by="anonymized_id", fit=True
        )
        
        # Normalized values should still show increasing pattern
        values = result["value"].values
        assert values[0] < values[1] < values[2]
    
    def test_normalize_timeseries_with_fitted_params(self, normalizer, sample_dataframe):
        """Test time-series normalization using pre-fitted parameters"""
        df = sample_dataframe.copy()
        
        # Fit on training data
        normalizer.normalize_timeseries(
            df, ["feature1"], group_by="anonymized_id", fit=True
        )
        
        # Apply to new data for same individuals
        new_df = pd.DataFrame({
            "anonymized_id": ["user1", "user2"],
            "feature1": [25.0, 250.0]
        })
        result = normalizer.normalize_timeseries(
            new_df, ["feature1"], group_by="anonymized_id", fit=False
        )
        
        # Check that fitted parameters were used
        assert "feature1_timeseries_mean" in normalizer.scaling_params
        assert "feature1_timeseries_std" in normalizer.scaling_params
    
    def test_normalize_timeseries_missing_group_column(self, normalizer, sample_dataframe):
        """Test fallback when group column is missing"""
        df = sample_dataframe.copy()
        result = normalizer.normalize_timeseries(
            df, ["feature1"], group_by="nonexistent_column", fit=True
        )
        
        # Should fall back to global standardization
        assert abs(result["feature1"].mean()) < 1e-10
        assert abs(result["feature1"].std() - 1.0) < 1e-10


class TestParameterPersistence:
    """Test suite for parameter saving and loading"""
    
    def test_save_and_load_params(self, normalizer, sample_dataframe):
        """Test saving and loading scaling parameters"""
        df = sample_dataframe.copy()
        
        # Fit and save parameters
        normalizer.standardize_features(df, ["feature1", "feature2"], fit=True)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            normalizer.save_params(filepath)
            
            # Create new normalizer and load parameters
            new_normalizer = Normalizer()
            new_normalizer.load_params(filepath)
            
            # Check parameters match
            assert new_normalizer.scaling_params == normalizer.scaling_params
        finally:
            Path(filepath).unlink()
    
    def test_get_and_set_scaling_params(self, normalizer, sample_dataframe):
        """Test getting and setting scaling parameters"""
        df = sample_dataframe.copy()
        normalizer.standardize_features(df, ["feature1"], fit=True)
        
        # Get parameters
        params = normalizer.get_scaling_params()
        assert "feature1_mean" in params
        assert "feature1_std" in params
        
        # Set parameters on new normalizer
        new_normalizer = Normalizer()
        new_normalizer.set_scaling_params(params)
        
        # Apply normalization with loaded parameters
        result = new_normalizer.standardize_features(df, ["feature1"], fit=False)
        assert not result["feature1"].isna().any()


class TestStatistics:
    """Test suite for statistics tracking"""
    
    def test_get_normalization_stats(self, normalizer, sample_dataframe):
        """Test retrieval of normalization statistics"""
        df = sample_dataframe.copy()
        normalizer.standardize_features(df, ["feature1", "feature2"], fit=True)
        
        stats = normalizer.get_normalization_stats()
        assert "standardized" in stats
        assert "feature1" in stats["standardized"]
        assert "feature2" in stats["standardized"]
    
    def test_reset_stats(self, normalizer, sample_dataframe):
        """Test resetting statistics"""
        df = sample_dataframe.copy()
        normalizer.standardize_features(df, ["feature1"], fit=True)
        
        assert len(normalizer.normalization_stats) > 0
        
        normalizer.reset_stats()
        assert len(normalizer.normalization_stats) == 0


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_normalize_nonexistent_column(self, normalizer, sample_dataframe):
        """Test handling of nonexistent columns"""
        df = sample_dataframe.copy()
        result = normalizer.standardize_features(df, ["nonexistent"], fit=True)
        
        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, df)
    
    def test_normalize_non_numeric_column(self, normalizer):
        """Test handling of non-numeric columns"""
        df = pd.DataFrame({"text_col": ["a", "b", "c"]})
        result = normalizer.standardize_features(df, ["text_col"], fit=True)
        
        # Should skip non-numeric columns
        pd.testing.assert_frame_equal(result, df)
    
    def test_normalize_with_nan_values(self, normalizer):
        """Test normalization with NaN values"""
        df = pd.DataFrame({"feature": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = normalizer.standardize_features(df, ["feature"], fit=True)
        
        # NaN values should remain NaN
        assert result["feature"].isna().sum() == 1
        assert not result["feature"].dropna().isna().any()
    
    def test_normalize_empty_dataframe(self, normalizer):
        """Test normalization of empty dataframe"""
        df = pd.DataFrame({"feature": []})
        result = normalizer.standardize_features(df, ["feature"], fit=True)
        
        # Should handle empty dataframe gracefully
        assert len(result) == 0
