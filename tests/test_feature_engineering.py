"""
Tests for feature engineering components.
"""
import pandas as pd
import pytest

from src.ml import (
    AdherenceTracker,
    BehavioralFeatureExtractor,
    FeatureEngineeringPipeline,
    PhysiologicalFeatureExtractor,
    SentimentAnalyzer,
)


class TestBehavioralFeatureExtractor:
    """Tests for BehavioralFeatureExtractor."""

    def test_extract_activity_features(self):
        """Test activity feature extraction."""
        extractor = BehavioralFeatureExtractor()

        # Create sample data
        data = {
            "anonymized_id": ["user1", "user1", "user1", "user2", "user2"],
            "activity_type": ["exercise", "social", "exercise", "work", "social"],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
        df = pd.DataFrame(data)

        result = extractor.extract_activity_features(df)

        assert not result.empty
        assert "anonymized_id" in result.columns
        assert "activity_count_7d" in result.columns
        assert len(result) == 2  # Two unique users

    def test_compute_routine_consistency(self):
        """Test routine consistency computation."""
        extractor = BehavioralFeatureExtractor()

        data = {
            "anonymized_id": ["user1"] * 10,
            "activity_type": ["exercise"] * 10,
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="H"),
        }
        df = pd.DataFrame(data)

        result = extractor.compute_routine_consistency(df)

        assert not result.empty
        assert "overall_routine_consistency" in result.columns
        assert 0 <= result["overall_routine_consistency"].iloc[0] <= 1

    def test_calculate_social_interaction_metrics(self):
        """Test social interaction metrics calculation."""
        extractor = BehavioralFeatureExtractor()

        data = {
            "anonymized_id": ["user1"] * 5,
            "interaction_type": ["call", "message", "call", "video", "message"],
            "duration_minutes": [10, 5, 15, 30, 3],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
        df = pd.DataFrame(data)

        result = extractor.calculate_social_interaction_metrics(df)

        assert not result.empty
        assert "interaction_count" in result.columns
        assert "mean_interaction_duration" in result.columns


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        analyzer = SentimentAnalyzer(use_transformers=False)

        text = "I feel happy and hopeful today"
        result = analyzer.analyze_sentiment(text)

        assert result.valence is not None
        assert result.arousal is not None
        assert result.dominance is not None
        assert -1 <= result.valence <= 1

    def test_detect_crisis_keywords(self):
        """Test crisis keyword detection."""
        analyzer = SentimentAnalyzer(use_transformers=False)

        # Text with crisis keyword
        text = "I feel hopeless and want to end it all"
        keywords = analyzer.detect_crisis_keywords(text)

        assert len(keywords) > 0
        assert any("hopeless" in k or "end it all" in k for k in keywords)

        # Text without crisis keywords
        text = "I had a good day today"
        keywords = analyzer.detect_crisis_keywords(text)

        assert len(keywords) == 0

    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        analyzer = SentimentAnalyzer(use_transformers=False)

        data = {
            "anonymized_id": ["user1", "user2", "user3"],
            "text_response": [
                "I feel great today",
                "I am very anxious",
                "Everything is terrible",
            ],
        }
        df = pd.DataFrame(data)

        result = analyzer.analyze_batch(df)

        assert not result.empty
        assert "sentiment_valence" in result.columns
        assert "crisis_flag" in result.columns


class TestPhysiologicalFeatureExtractor:
    """Tests for PhysiologicalFeatureExtractor."""

    def test_extract_sleep_features(self):
        """Test sleep feature extraction."""
        extractor = PhysiologicalFeatureExtractor()

        data = {
            "anonymized_id": ["user1"] * 3,
            "sleep_start": pd.date_range("2025-01-01 22:00", periods=3, freq="D"),
            "sleep_end": pd.date_range("2025-01-02 06:00", periods=3, freq="D"),
            "interruptions": [2, 1, 3],
        }
        df = pd.DataFrame(data)

        result = extractor.extract_sleep_features(df)

        assert not result.empty
        assert "mean_sleep_duration" in result.columns
        assert "mean_interruptions" in result.columns

    def test_compute_hrv_metrics(self):
        """Test HRV metrics computation."""
        extractor = PhysiologicalFeatureExtractor()

        data = {
            "anonymized_id": ["user1"] * 10,
            "heart_rate": [70, 72, 68, 71, 69, 73, 70, 72, 71, 69],
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="min"),
        }
        df = pd.DataFrame(data)

        result = extractor.compute_hrv_metrics(df)

        assert not result.empty
        assert "mean_heart_rate" in result.columns

    def test_calculate_activity_intensity(self):
        """Test activity intensity calculation."""
        extractor = PhysiologicalFeatureExtractor()

        data = {
            "anonymized_id": ["user1"] * 5,
            "heart_rate": [120, 130, 125, 135, 128],
            "steps": [5000, 6000, 5500, 7000, 6500],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="H"),
        }
        df = pd.DataFrame(data)

        result = extractor.calculate_activity_intensity(df)

        assert not result.empty
        assert "activity_intensity_score" in result.columns
        assert "total_steps" in result.columns


class TestAdherenceTracker:
    """Tests for AdherenceTracker."""

    def test_calculate_adherence_rate(self):
        """Test adherence rate calculation."""
        tracker = AdherenceTracker()

        data = {
            "anonymized_id": ["user1", "user2"],
            "scheduled_sessions": [10, 8],
            "completed_sessions": [8, 6],
        }
        df = pd.DataFrame(data)

        result = tracker.calculate_adherence_rate(df)

        assert not result.empty
        assert "overall_adherence_rate" in result.columns
        assert result["overall_adherence_rate"].iloc[0] == 80.0

    def test_flag_missed_sessions(self):
        """Test missed session flagging."""
        tracker = AdherenceTracker()

        data = {
            "anonymized_id": ["user1"] * 5,
            "session_date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "status": ["completed", "missed", "completed", "missed", "missed"],
        }
        df = pd.DataFrame(data)

        result = tracker.flag_missed_sessions(df)

        assert not result.empty
        assert "missed_sessions_count" in result.columns
        assert "max_consecutive_missed" in result.columns

    def test_compute_engagement_score(self):
        """Test engagement score computation."""
        tracker = AdherenceTracker()

        data = {
            "anonymized_id": ["user1"] * 10,
            "interaction_count": [5, 3, 7, 4, 6, 5, 8, 4, 6, 5],
            "session_duration_minutes": [20, 15, 30, 25, 20, 22, 28, 18, 24, 21],
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="D"),
        }
        df = pd.DataFrame(data)

        result = tracker.compute_engagement_score(df)

        assert not result.empty
        assert "overall_engagement_score" in result.columns
        assert 0 <= result["overall_engagement_score"].iloc[0] <= 1


class TestFeatureEngineeringPipeline:
    """Tests for FeatureEngineeringPipeline."""

    def test_extract_features_sequential(self):
        """Test sequential feature extraction."""
        pipeline = FeatureEngineeringPipeline(use_parallel=False)

        # Create sample behavioral data
        behavioral_data = {
            "anonymized_id": ["user1"] * 5,
            "activity_type": ["exercise", "social", "work", "exercise", "social"],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
        behavioral_df = pd.DataFrame(behavioral_data)

        result = pipeline.extract_features(behavioral_df=behavioral_df)

        assert not result.empty
        assert "anonymized_id" in result.columns

    def test_extract_features_parallel(self):
        """Test parallel feature extraction."""
        pipeline = FeatureEngineeringPipeline(use_parallel=True, max_workers=2)

        # Create sample data
        behavioral_data = {
            "anonymized_id": ["user1"] * 5,
            "activity_type": ["exercise", "social", "work", "exercise", "social"],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
        behavioral_df = pd.DataFrame(behavioral_data)

        text_data = {
            "anonymized_id": ["user1"],
            "text_response": ["I feel good today"],
        }
        text_df = pd.DataFrame(text_data)

        result = pipeline.extract_features(behavioral_df=behavioral_df, text_df=text_df)

        assert not result.empty
        assert "anonymized_id" in result.columns

    def test_get_feature_summary(self):
        """Test feature summary generation."""
        pipeline = FeatureEngineeringPipeline(use_parallel=False)

        behavioral_data = {
            "anonymized_id": ["user1"] * 5,
            "activity_type": ["exercise", "social", "work", "exercise", "social"],
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
        behavioral_df = pd.DataFrame(behavioral_data)

        features = pipeline.extract_features(behavioral_df=behavioral_df)
        summary = pipeline.get_feature_summary(features)

        assert "num_individuals" in summary
        assert "num_features" in summary
        assert summary["num_individuals"] > 0

    def test_empty_input_handling(self):
        """Test handling of empty input DataFrames."""
        pipeline = FeatureEngineeringPipeline(use_parallel=False)

        result = pipeline.extract_features()

        # Should return empty DataFrame without errors
        assert isinstance(result, pd.DataFrame)
