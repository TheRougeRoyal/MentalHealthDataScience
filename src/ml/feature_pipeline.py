"""
Unified feature engineering pipeline orchestrating all feature extractors.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
from pandas import DataFrame

from src.exceptions import ProcessingError
from src.ml.adherence_tracking import AdherenceTracker
from src.ml.behavioral_features import BehavioralFeatureExtractor
from src.ml.physiological_features import PhysiologicalFeatureExtractor
from src.ml.sentiment_analysis import SentimentAnalyzer

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates all feature extractors into a unified pipeline with
    parallel processing and validation.
    """

    def __init__(
        self,
        use_parallel: bool = True,
        max_workers: int = 4,
        use_transformers_sentiment: bool = False,
    ):
        """
        Initialize FeatureEngineeringPipeline.

        Args:
            use_parallel: Whether to use parallel processing for independent feature groups
            max_workers: Maximum number of parallel workers
            use_transformers_sentiment: Whether to use transformer-based sentiment analysis
        """
        self.use_parallel = use_parallel
        self.max_workers = max_workers

        # Initialize feature extractors
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.sentiment_analyzer = SentimentAnalyzer(use_transformers=use_transformers_sentiment)
        self.physiological_extractor = PhysiologicalFeatureExtractor()
        self.adherence_tracker = AdherenceTracker()

        self.pipeline_stats: Dict = {}

    def extract_features(
        self,
        behavioral_df: Optional[DataFrame] = None,
        text_df: Optional[DataFrame] = None,
        sleep_df: Optional[DataFrame] = None,
        hrv_df: Optional[DataFrame] = None,
        activity_df: Optional[DataFrame] = None,
        adherence_df: Optional[DataFrame] = None,
        sessions_df: Optional[DataFrame] = None,
        engagement_df: Optional[DataFrame] = None,
        id_column: str = "anonymized_id",
        validate: bool = True,
    ) -> DataFrame:
        """
        Extract all features from provided data sources.

        Args:
            behavioral_df: Optional DataFrame with behavioral/activity data
            text_df: Optional DataFrame with text responses
            sleep_df: Optional DataFrame with sleep data
            hrv_df: Optional DataFrame with HRV data
            activity_df: Optional DataFrame with activity intensity data
            adherence_df: Optional DataFrame with adherence data
            sessions_df: Optional DataFrame with session data
            engagement_df: Optional DataFrame with engagement data
            id_column: Column containing anonymized identifiers
            validate: Whether to validate features after extraction

        Returns:
            DataFrame with all extracted features per individual

        Note:
            Pipeline should complete within 2 seconds per individual
        """
        start_time = time.time()

        logger.info("Starting feature engineering pipeline")

        if self.use_parallel:
            features_df = self._extract_features_parallel(
                behavioral_df=behavioral_df,
                text_df=text_df,
                sleep_df=sleep_df,
                hrv_df=hrv_df,
                activity_df=activity_df,
                adherence_df=adherence_df,
                sessions_df=sessions_df,
                engagement_df=engagement_df,
                id_column=id_column,
            )
        else:
            features_df = self._extract_features_sequential(
                behavioral_df=behavioral_df,
                text_df=text_df,
                sleep_df=sleep_df,
                hrv_df=hrv_df,
                activity_df=activity_df,
                adherence_df=adherence_df,
                sessions_df=sessions_df,
                engagement_df=engagement_df,
                id_column=id_column,
            )

        # Validate features if requested
        if validate and not features_df.empty:
            features_df = self._validate_features(features_df)

        elapsed_time = time.time() - start_time
        num_individuals = len(features_df) if not features_df.empty else 0
        time_per_individual = elapsed_time / num_individuals if num_individuals > 0 else 0

        self.pipeline_stats = {
            "total_time_seconds": elapsed_time,
            "num_individuals": num_individuals,
            "time_per_individual": time_per_individual,
            "num_features": len(features_df.columns) - 1 if not features_df.empty else 0,
        }

        logger.info(
            f"Feature engineering completed in {elapsed_time:.2f}s "
            f"({time_per_individual:.3f}s per individual, {num_individuals} individuals)"
        )

        if time_per_individual > 2.0:
            logger.warning(
                f"Feature engineering exceeded 2s per individual target: {time_per_individual:.3f}s"
            )

        return features_df

    def _extract_features_parallel(
        self,
        behavioral_df: Optional[DataFrame],
        text_df: Optional[DataFrame],
        sleep_df: Optional[DataFrame],
        hrv_df: Optional[DataFrame],
        activity_df: Optional[DataFrame],
        adherence_df: Optional[DataFrame],
        sessions_df: Optional[DataFrame],
        engagement_df: Optional[DataFrame],
        id_column: str,
    ) -> DataFrame:
        """Extract features using parallel processing."""
        feature_dfs = []
        futures = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit behavioral feature extraction
            if behavioral_df is not None and not behavioral_df.empty:
                futures.append(
                    executor.submit(
                        self._extract_behavioral_features, behavioral_df, id_column
                    )
                )

            # Submit sentiment analysis
            if text_df is not None and not text_df.empty:
                futures.append(
                    executor.submit(self._extract_sentiment_features, text_df, id_column)
                )

            # Submit physiological feature extraction
            if any(
                df is not None and not df.empty for df in [sleep_df, hrv_df, activity_df]
            ):
                futures.append(
                    executor.submit(
                        self._extract_physiological_features,
                        sleep_df,
                        hrv_df,
                        activity_df,
                        id_column,
                    )
                )

            # Submit adherence feature extraction
            if any(
                df is not None and not df.empty
                for df in [adherence_df, sessions_df, engagement_df]
            ):
                futures.append(
                    executor.submit(
                        self._extract_adherence_features,
                        adherence_df,
                        sessions_df,
                        engagement_df,
                        id_column,
                    )
                )

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        feature_dfs.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel feature extraction: {e}")

        # Merge all feature DataFrames
        if feature_dfs:
            return self._merge_features(feature_dfs, id_column)
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()

    def _extract_features_sequential(
        self,
        behavioral_df: Optional[DataFrame],
        text_df: Optional[DataFrame],
        sleep_df: Optional[DataFrame],
        hrv_df: Optional[DataFrame],
        activity_df: Optional[DataFrame],
        adherence_df: Optional[DataFrame],
        sessions_df: Optional[DataFrame],
        engagement_df: Optional[DataFrame],
        id_column: str,
    ) -> DataFrame:
        """Extract features sequentially."""
        feature_dfs = []

        # Extract behavioral features
        if behavioral_df is not None and not behavioral_df.empty:
            try:
                behavioral_features = self._extract_behavioral_features(
                    behavioral_df, id_column
                )
                if not behavioral_features.empty:
                    feature_dfs.append(behavioral_features)
            except Exception as e:
                logger.error(f"Error extracting behavioral features: {e}")

        # Extract sentiment features
        if text_df is not None and not text_df.empty:
            try:
                sentiment_features = self._extract_sentiment_features(text_df, id_column)
                if not sentiment_features.empty:
                    feature_dfs.append(sentiment_features)
            except Exception as e:
                logger.error(f"Error extracting sentiment features: {e}")

        # Extract physiological features
        if any(df is not None and not df.empty for df in [sleep_df, hrv_df, activity_df]):
            try:
                physiological_features = self._extract_physiological_features(
                    sleep_df, hrv_df, activity_df, id_column
                )
                if not physiological_features.empty:
                    feature_dfs.append(physiological_features)
            except Exception as e:
                logger.error(f"Error extracting physiological features: {e}")

        # Extract adherence features
        if any(
            df is not None and not df.empty
            for df in [adherence_df, sessions_df, engagement_df]
        ):
            try:
                adherence_features = self._extract_adherence_features(
                    adherence_df, sessions_df, engagement_df, id_column
                )
                if not adherence_features.empty:
                    feature_dfs.append(adherence_features)
            except Exception as e:
                logger.error(f"Error extracting adherence features: {e}")

        # Merge all feature DataFrames
        if feature_dfs:
            return self._merge_features(feature_dfs, id_column)
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()

    def _extract_behavioral_features(
        self, behavioral_df: DataFrame, id_column: str
    ) -> DataFrame:
        """Extract behavioral features."""
        logger.info("Extracting behavioral features")

        # Extract activity features
        activity_features = self.behavioral_extractor.extract_activity_features(
            behavioral_df, id_column=id_column
        )

        # Compute routine consistency
        routine_features = self.behavioral_extractor.compute_routine_consistency(
            behavioral_df, id_column=id_column
        )

        # Calculate social interaction metrics
        social_features = self.behavioral_extractor.calculate_social_interaction_metrics(
            behavioral_df, id_column=id_column
        )

        # Merge behavioral features
        features = [activity_features, routine_features, social_features]
        features = [f for f in features if not f.empty]

        if features:
            result = features[0]
            for df in features[1:]:
                result = result.merge(df, on=id_column, how="outer")
            return result
        else:
            return pd.DataFrame()

    def _extract_sentiment_features(self, text_df: DataFrame, id_column: str) -> DataFrame:
        """Extract sentiment features."""
        logger.info("Extracting sentiment features")

        # Analyze sentiment for batch
        sentiment_df = self.sentiment_analyzer.analyze_batch(text_df, id_column=id_column)

        # Aggregate sentiment features per individual
        if not sentiment_df.empty and id_column in sentiment_df.columns:
            agg_dict = {
                "sentiment_valence": ["mean", "std", "min", "max"],
                "sentiment_arousal": ["mean", "std"],
                "sentiment_dominance": ["mean", "std"],
                "sentiment_compound": ["mean", "std"],
                "crisis_flag": "sum",
            }

            # Only aggregate columns that exist
            agg_dict = {k: v for k, v in agg_dict.items() if k in sentiment_df.columns}

            if agg_dict:
                sentiment_features = sentiment_df.groupby(id_column).agg(agg_dict)
                sentiment_features.columns = [
                    f"{col}_{stat}" for col, stat in sentiment_features.columns
                ]
                sentiment_features = sentiment_features.reset_index()
                return sentiment_features

        return pd.DataFrame()

    def _extract_physiological_features(
        self,
        sleep_df: Optional[DataFrame],
        hrv_df: Optional[DataFrame],
        activity_df: Optional[DataFrame],
        id_column: str,
    ) -> DataFrame:
        """Extract physiological features."""
        logger.info("Extracting physiological features")

        return self.physiological_extractor.extract_all_physiological_features(
            sleep_df=sleep_df,
            hrv_df=hrv_df,
            activity_df=activity_df,
            id_column=id_column,
        )

    def _extract_adherence_features(
        self,
        adherence_df: Optional[DataFrame],
        sessions_df: Optional[DataFrame],
        engagement_df: Optional[DataFrame],
        id_column: str,
    ) -> DataFrame:
        """Extract adherence features."""
        logger.info("Extracting adherence features")

        return self.adherence_tracker.extract_all_adherence_features(
            adherence_df=adherence_df,
            sessions_df=sessions_df,
            engagement_df=engagement_df,
            id_column=id_column,
        )

    def _merge_features(self, feature_dfs: List[DataFrame], id_column: str) -> DataFrame:
        """Merge multiple feature DataFrames."""
        if not feature_dfs:
            return pd.DataFrame()

        result = feature_dfs[0]
        for df in feature_dfs[1:]:
            if not df.empty and id_column in df.columns:
                result = result.merge(df, on=id_column, how="outer")

        logger.info(f"Merged features: {len(result)} individuals, {len(result.columns)} features")

        return result

    def _validate_features(self, features_df: DataFrame) -> DataFrame:
        """
        Validate features and handle quality issues.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Validated DataFrame
        """
        logger.info("Validating features")

        initial_rows = len(features_df)
        initial_cols = len(features_df.columns)

        # Check for infinite values
        numeric_cols = features_df.select_dtypes(include=["number"]).columns
        inf_mask = features_df[numeric_cols].isin([float("inf"), float("-inf")]).any(axis=1)
        if inf_mask.sum() > 0:
            logger.warning(f"Found {inf_mask.sum()} rows with infinite values. Replacing with NaN.")
            features_df[numeric_cols] = features_df[numeric_cols].replace(
                [float("inf"), float("-inf")], float("nan")
            )

        # Check for excessive missing values per row
        missing_pct = features_df.isnull().sum(axis=1) / len(features_df.columns)
        high_missing_mask = missing_pct > 0.8

        if high_missing_mask.sum() > 0:
            logger.warning(
                f"Found {high_missing_mask.sum()} individuals with >80% missing features"
            )

        # Check for constant columns (no variance)
        constant_cols = []
        for col in numeric_cols:
            if features_df[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            logger.warning(f"Found {len(constant_cols)} constant columns: {constant_cols[:5]}")

        # Log validation summary
        final_rows = len(features_df)
        final_cols = len(features_df.columns)

        logger.info(
            f"Validation complete: {final_rows} individuals, {final_cols} features "
            f"(removed {initial_rows - final_rows} rows, {initial_cols - final_cols} columns)"
        )

        return features_df

    def get_pipeline_stats(self) -> Dict:
        """
        Get statistics from the last pipeline execution.

        Returns:
            Dictionary containing pipeline statistics
        """
        return self.pipeline_stats.copy()

    def get_feature_names(self, features_df: DataFrame) -> List[str]:
        """
        Get list of feature names (excluding ID column).

        Args:
            features_df: DataFrame with features

        Returns:
            List of feature names
        """
        if features_df.empty:
            return []

        # Exclude ID column
        id_columns = ["anonymized_id", "id"]
        feature_names = [col for col in features_df.columns if col not in id_columns]

        return feature_names

    def get_feature_summary(self, features_df: DataFrame) -> Dict:
        """
        Get summary statistics for extracted features.

        Args:
            features_df: DataFrame with features

        Returns:
            Dictionary with feature summary statistics
        """
        if features_df.empty:
            return {}

        numeric_cols = features_df.select_dtypes(include=["number"]).columns

        summary = {
            "num_individuals": len(features_df),
            "num_features": len(features_df.columns) - 1,  # Exclude ID
            "num_numeric_features": len(numeric_cols),
            "missing_values_pct": (features_df.isnull().sum().sum() / features_df.size) * 100,
        }

        # Feature group counts
        summary["behavioral_features"] = len(
            [c for c in features_df.columns if "activity" in c or "routine" in c or "social" in c]
        )
        summary["sentiment_features"] = len(
            [c for c in features_df.columns if "sentiment" in c or "crisis" in c]
        )
        summary["physiological_features"] = len(
            [c for c in features_df.columns if "sleep" in c or "hrv" in c or "heart" in c]
        )
        summary["adherence_features"] = len(
            [c for c in features_df.columns if "adherence" in c or "engagement" in c or "missed" in c]
        )

        return summary
