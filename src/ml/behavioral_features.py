"""
Behavioral feature extraction module for activity patterns and social interactions.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class BehavioralFeatureExtractor:
    """
    Extracts behavioral features from activity data including activity patterns,
    routine consistency, and social interaction metrics.
    """

    def __init__(self):
        """Initialize BehavioralFeatureExtractor."""
        self.feature_stats: Dict = {}

    def extract_activity_features(
        self,
        df: DataFrame,
        activity_column: str = "activity_type",
        timestamp_column: str = "timestamp",
        id_column: str = "anonymized_id",
        windows: List[int] = [7, 30],
    ) -> DataFrame:
        """
        Extract activity features with configurable time windows.

        Args:
            df: Input DataFrame with activity data
            activity_column: Column containing activity types
            timestamp_column: Column containing timestamps
            id_column: Column containing anonymized identifiers
            windows: List of window sizes in days (default: [7, 30])

        Returns:
            DataFrame with activity features aggregated per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        features_list = []

        # Group by individual
        for individual_id, group in df.groupby(id_column):
            individual_features = {id_column: individual_id}

            # Sort by timestamp
            group = group.sort_values(timestamp_column)

            # Extract features for each window
            for window_days in windows:
                # Get most recent data within window
                if timestamp_column in group.columns:
                    cutoff_date = group[timestamp_column].max() - pd.Timedelta(days=window_days)
                    window_data = group[group[timestamp_column] >= cutoff_date]
                else:
                    window_data = group

                # Activity count
                activity_count = len(window_data)
                individual_features[f"activity_count_{window_days}d"] = activity_count

                # Activity frequency (per day)
                if activity_count > 0 and timestamp_column in window_data.columns:
                    date_range = (
                        window_data[timestamp_column].max()
                        - window_data[timestamp_column].min()
                    ).days + 1
                    activity_frequency = activity_count / max(date_range, 1)
                else:
                    activity_frequency = 0
                individual_features[f"activity_frequency_{window_days}d"] = activity_frequency

                # Unique activity types
                if activity_column in window_data.columns:
                    unique_activities = window_data[activity_column].nunique()
                    individual_features[f"unique_activities_{window_days}d"] = unique_activities

                    # Most common activity
                    if not window_data[activity_column].empty:
                        most_common = window_data[activity_column].mode()
                        if len(most_common) > 0:
                            individual_features[f"most_common_activity_{window_days}d"] = (
                                most_common.iloc[0]
                            )

                # Activity diversity (entropy of activity distribution)
                if activity_column in window_data.columns and not window_data.empty:
                    activity_counts = window_data[activity_column].value_counts()
                    activity_probs = activity_counts / activity_counts.sum()
                    activity_entropy = entropy(activity_probs)
                    individual_features[f"activity_diversity_{window_days}d"] = activity_entropy

            features_list.append(individual_features)

        result_df = pd.DataFrame(features_list)

        logger.info(
            f"Extracted activity features for {len(result_df)} individuals "
            f"with windows: {windows} days"
        )

        return result_df

    def compute_routine_consistency(
        self,
        df: DataFrame,
        timestamp_column: str = "timestamp",
        activity_column: str = "activity_type",
        id_column: str = "anonymized_id",
    ) -> DataFrame:
        """
        Compute routine consistency using entropy measures of activity timing.

        Args:
            df: Input DataFrame with activity data
            timestamp_column: Column containing timestamps
            activity_column: Column containing activity types
            id_column: Column containing anonymized identifiers

        Returns:
            DataFrame with routine consistency scores per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        df = df.copy()
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            df["hour_of_day"] = df[timestamp_column].dt.hour
            df["day_of_week"] = df[timestamp_column].dt.dayofweek

        consistency_list = []

        for individual_id, group in df.groupby(id_column):
            consistency_features = {id_column: individual_id}

            # Hour of day consistency (lower entropy = more consistent)
            if "hour_of_day" in group.columns and not group.empty:
                hour_counts = group["hour_of_day"].value_counts()
                hour_probs = hour_counts / hour_counts.sum()
                hour_entropy = entropy(hour_probs)
                # Normalize by max possible entropy (log(24))
                max_hour_entropy = np.log(24)
                consistency_features["hour_consistency"] = 1 - (hour_entropy / max_hour_entropy)
            else:
                consistency_features["hour_consistency"] = 0.0

            # Day of week consistency
            if "day_of_week" in group.columns and not group.empty:
                day_counts = group["day_of_week"].value_counts()
                day_probs = day_counts / day_counts.sum()
                day_entropy = entropy(day_probs)
                # Normalize by max possible entropy (log(7))
                max_day_entropy = np.log(7)
                consistency_features["day_consistency"] = 1 - (day_entropy / max_day_entropy)
            else:
                consistency_features["day_consistency"] = 0.0

            # Activity type consistency
            if activity_column in group.columns and not group.empty:
                activity_counts = group[activity_column].value_counts()
                activity_probs = activity_counts / activity_counts.sum()
                activity_entropy = entropy(activity_probs)
                # Normalize by number of unique activities
                num_unique = len(activity_counts)
                max_activity_entropy = np.log(num_unique) if num_unique > 1 else 1
                consistency_features["activity_type_consistency"] = 1 - (
                    activity_entropy / max_activity_entropy
                )
            else:
                consistency_features["activity_type_consistency"] = 0.0

            # Overall routine consistency (average of components)
            consistency_features["overall_routine_consistency"] = np.mean(
                [
                    consistency_features["hour_consistency"],
                    consistency_features["day_consistency"],
                    consistency_features["activity_type_consistency"],
                ]
            )

            consistency_list.append(consistency_features)

        result_df = pd.DataFrame(consistency_list)

        logger.info(f"Computed routine consistency for {len(result_df)} individuals")

        return result_df

    def calculate_social_interaction_metrics(
        self,
        df: DataFrame,
        interaction_column: str = "interaction_type",
        duration_column: Optional[str] = "duration_minutes",
        timestamp_column: str = "timestamp",
        id_column: str = "anonymized_id",
        window_days: int = 30,
    ) -> DataFrame:
        """
        Calculate social interaction frequency and duration metrics.

        Args:
            df: Input DataFrame with social interaction data
            interaction_column: Column containing interaction types
            duration_column: Optional column containing interaction duration
            timestamp_column: Column containing timestamps
            id_column: Column containing anonymized identifiers
            window_days: Time window in days for aggregation

        Returns:
            DataFrame with social interaction metrics per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        df = df.copy()
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        interaction_list = []

        for individual_id, group in df.groupby(id_column):
            interaction_features = {id_column: individual_id}

            # Filter to window
            if timestamp_column in group.columns:
                cutoff_date = group[timestamp_column].max() - pd.Timedelta(days=window_days)
                window_data = group[group[timestamp_column] >= cutoff_date]
            else:
                window_data = group

            # Interaction frequency
            interaction_count = len(window_data)
            interaction_features["interaction_count"] = interaction_count
            interaction_features["interaction_frequency_per_day"] = interaction_count / window_days

            # Unique interaction types
            if interaction_column in window_data.columns:
                unique_interactions = window_data[interaction_column].nunique()
                interaction_features["unique_interaction_types"] = unique_interactions

            # Duration metrics
            if duration_column and duration_column in window_data.columns:
                durations = window_data[duration_column].dropna()
                if not durations.empty:
                    interaction_features["total_interaction_duration"] = durations.sum()
                    interaction_features["mean_interaction_duration"] = durations.mean()
                    interaction_features["median_interaction_duration"] = durations.median()
                    interaction_features["max_interaction_duration"] = durations.max()
                else:
                    interaction_features["total_interaction_duration"] = 0
                    interaction_features["mean_interaction_duration"] = 0
                    interaction_features["median_interaction_duration"] = 0
                    interaction_features["max_interaction_duration"] = 0

            # Interaction regularity (coefficient of variation of daily counts)
            if timestamp_column in window_data.columns and not window_data.empty:
                daily_counts = (
                    window_data.groupby(window_data[timestamp_column].dt.date).size()
                )
                if len(daily_counts) > 1:
                    cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else 0
                    interaction_features["interaction_regularity"] = 1 / (1 + cv)
                else:
                    interaction_features["interaction_regularity"] = 0.0

            interaction_list.append(interaction_features)

        result_df = pd.DataFrame(interaction_list)

        logger.info(f"Calculated social interaction metrics for {len(result_df)} individuals")

        return result_df

    def get_feature_stats(self) -> Dict:
        """
        Get statistics from the last feature extraction operation.

        Returns:
            Dictionary containing feature extraction statistics
        """
        return self.feature_stats.copy()
