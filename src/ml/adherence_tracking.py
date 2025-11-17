"""
Adherence tracking module for therapy engagement and intervention completion.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class AdherenceTracker:
    """
    Tracks therapy adherence including intervention completion rates,
    missed sessions, and engagement scores.
    """

    def __init__(self):
        """Initialize AdherenceTracker."""
        self.adherence_stats: Dict = {}

    def calculate_adherence_rate(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        scheduled_column: str = "scheduled_sessions",
        completed_column: str = "completed_sessions",
        intervention_type_column: Optional[str] = "intervention_type",
    ) -> DataFrame:
        """
        Calculate adherence rate for intervention completion.

        Args:
            df: Input DataFrame with intervention data
            id_column: Column containing anonymized identifiers
            scheduled_column: Column containing number of scheduled sessions
            completed_column: Column containing number of completed sessions
            intervention_type_column: Optional column for intervention type

        Returns:
            DataFrame with adherence rates per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        adherence_list = []

        for individual_id, group in df.groupby(id_column):
            adherence_features = {id_column: individual_id}

            # Overall adherence rate
            if scheduled_column in group.columns and completed_column in group.columns:
                total_scheduled = group[scheduled_column].sum()
                total_completed = group[completed_column].sum()

                if total_scheduled > 0:
                    adherence_rate = (total_completed / total_scheduled) * 100
                    adherence_features["overall_adherence_rate"] = adherence_rate
                    adherence_features["total_scheduled"] = total_scheduled
                    adherence_features["total_completed"] = total_completed
                    adherence_features["total_missed"] = total_scheduled - total_completed
                else:
                    adherence_features["overall_adherence_rate"] = 0.0
                    adherence_features["total_scheduled"] = 0
                    adherence_features["total_completed"] = 0
                    adherence_features["total_missed"] = 0

                # Adherence category
                if adherence_rate >= 80:
                    adherence_features["adherence_category"] = "high"
                elif adherence_rate >= 50:
                    adherence_features["adherence_category"] = "moderate"
                else:
                    adherence_features["adherence_category"] = "low"

            # Adherence by intervention type
            if (
                intervention_type_column
                and intervention_type_column in group.columns
                and scheduled_column in group.columns
                and completed_column in group.columns
            ):
                for intervention_type, type_group in group.groupby(intervention_type_column):
                    type_scheduled = type_group[scheduled_column].sum()
                    type_completed = type_group[completed_column].sum()

                    if type_scheduled > 0:
                        type_adherence = (type_completed / type_scheduled) * 100
                        adherence_features[f"adherence_{intervention_type}"] = type_adherence

            # Recent adherence trend (last 30 days if timestamp available)
            if "timestamp" in group.columns:
                group_sorted = group.copy()
                group_sorted["timestamp"] = pd.to_datetime(group_sorted["timestamp"])
                cutoff_date = group_sorted["timestamp"].max() - pd.Timedelta(days=30)
                recent_data = group_sorted[group_sorted["timestamp"] >= cutoff_date]

                if not recent_data.empty:
                    recent_scheduled = recent_data[scheduled_column].sum()
                    recent_completed = recent_data[completed_column].sum()

                    if recent_scheduled > 0:
                        recent_adherence = (recent_completed / recent_scheduled) * 100
                        adherence_features["recent_adherence_rate_30d"] = recent_adherence

            adherence_list.append(adherence_features)

        result_df = pd.DataFrame(adherence_list)

        logger.info(f"Calculated adherence rates for {len(result_df)} individuals")

        return result_df

    def flag_missed_sessions(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        session_date_column: str = "session_date",
        status_column: str = "status",
        missed_status: str = "missed",
        window_days: int = 30,
    ) -> DataFrame:
        """
        Flag patterns of missed appointments and sessions.

        Args:
            df: Input DataFrame with session data
            id_column: Column containing anonymized identifiers
            session_date_column: Column containing session dates
            status_column: Column containing session status
            missed_status: Value indicating a missed session
            window_days: Time window for analysis

        Returns:
            DataFrame with missed session flags per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        df = df.copy()
        if session_date_column in df.columns:
            df[session_date_column] = pd.to_datetime(df[session_date_column])

        missed_session_list = []

        for individual_id, group in df.groupby(id_column):
            missed_features = {id_column: individual_id}

            # Filter to window
            if session_date_column in group.columns:
                cutoff_date = group[session_date_column].max() - pd.Timedelta(days=window_days)
                window_data = group[group[session_date_column] >= cutoff_date]
            else:
                window_data = group

            if status_column in window_data.columns:
                # Count missed sessions
                missed_count = (window_data[status_column] == missed_status).sum()
                total_sessions = len(window_data)

                missed_features["missed_sessions_count"] = missed_count
                missed_features["total_sessions"] = total_sessions

                if total_sessions > 0:
                    missed_rate = (missed_count / total_sessions) * 100
                    missed_features["missed_sessions_rate"] = missed_rate

                    # Flag high miss rate
                    missed_features["high_miss_rate_flag"] = missed_rate > 30
                else:
                    missed_features["missed_sessions_rate"] = 0.0
                    missed_features["high_miss_rate_flag"] = False

                # Consecutive missed sessions
                if session_date_column in window_data.columns:
                    window_sorted = window_data.sort_values(session_date_column)
                    missed_mask = window_sorted[status_column] == missed_status

                    # Find longest streak of consecutive misses
                    max_consecutive = 0
                    current_consecutive = 0

                    for is_missed in missed_mask:
                        if is_missed:
                            current_consecutive += 1
                            max_consecutive = max(max_consecutive, current_consecutive)
                        else:
                            current_consecutive = 0

                    missed_features["max_consecutive_missed"] = max_consecutive
                    missed_features["consecutive_miss_flag"] = max_consecutive >= 3

                # Recent missed sessions (last 7 days)
                if session_date_column in window_data.columns:
                    recent_cutoff = window_data[session_date_column].max() - pd.Timedelta(days=7)
                    recent_data = window_data[window_data[session_date_column] >= recent_cutoff]

                    if not recent_data.empty:
                        recent_missed = (recent_data[status_column] == missed_status).sum()
                        missed_features["recent_missed_7d"] = recent_missed
                        missed_features["recent_miss_flag"] = recent_missed > 0

            missed_session_list.append(missed_features)

        result_df = pd.DataFrame(missed_session_list)

        logger.info(f"Flagged missed sessions for {len(result_df)} individuals")

        return result_df

    def compute_engagement_score(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        interaction_column: Optional[str] = "interaction_count",
        duration_column: Optional[str] = "session_duration_minutes",
        completion_column: Optional[str] = "completed_activities",
        timestamp_column: str = "timestamp",
        window_days: int = 30,
    ) -> DataFrame:
        """
        Compute engagement score from interaction logs.

        Args:
            df: Input DataFrame with interaction data
            id_column: Column containing anonymized identifiers
            interaction_column: Optional column containing interaction counts
            duration_column: Optional column containing session durations
            completion_column: Optional column containing completed activities
            timestamp_column: Column containing timestamps
            window_days: Time window for analysis

        Returns:
            DataFrame with engagement scores per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        df = df.copy()
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        engagement_list = []

        for individual_id, group in df.groupby(id_column):
            engagement_features = {id_column: individual_id}

            # Filter to window
            if timestamp_column in group.columns:
                cutoff_date = group[timestamp_column].max() - pd.Timedelta(days=window_days)
                window_data = group[group[timestamp_column] >= cutoff_date]
            else:
                window_data = group

            engagement_components = []

            # Interaction frequency component
            if interaction_column and interaction_column in window_data.columns:
                total_interactions = window_data[interaction_column].sum()
                interaction_frequency = total_interactions / window_days

                # Normalize to 0-1 scale (assuming 5+ interactions per day is high)
                interaction_score = min(interaction_frequency / 5, 1.0)
                engagement_components.append(interaction_score)

                engagement_features["interaction_frequency"] = interaction_frequency
                engagement_features["interaction_score"] = interaction_score

            # Duration component
            if duration_column and duration_column in window_data.columns:
                durations = window_data[duration_column].dropna()

                if not durations.empty:
                    mean_duration = durations.mean()
                    total_duration = durations.sum()

                    # Normalize to 0-1 scale (assuming 30+ min per session is high)
                    duration_score = min(mean_duration / 30, 1.0)
                    engagement_components.append(duration_score)

                    engagement_features["mean_session_duration"] = mean_duration
                    engagement_features["total_session_duration"] = total_duration
                    engagement_features["duration_score"] = duration_score

            # Completion component
            if completion_column and completion_column in window_data.columns:
                completions = window_data[completion_column].dropna()

                if not completions.empty:
                    total_completions = completions.sum()
                    completion_rate = total_completions / len(window_data)

                    # Normalize to 0-1 scale (assuming 80%+ completion is high)
                    completion_score = min(completion_rate / 0.8, 1.0)
                    engagement_components.append(completion_score)

                    engagement_features["total_completions"] = total_completions
                    engagement_features["completion_rate"] = completion_rate
                    engagement_features["completion_score"] = completion_score

            # Consistency component (regularity of engagement)
            if timestamp_column in window_data.columns and not window_data.empty:
                daily_engagement = window_data.groupby(
                    window_data[timestamp_column].dt.date
                ).size()

                if len(daily_engagement) > 1:
                    # Coefficient of variation (lower is more consistent)
                    cv = daily_engagement.std() / daily_engagement.mean() if daily_engagement.mean() > 0 else 0
                    consistency_score = 1 / (1 + cv)
                    engagement_components.append(consistency_score)

                    engagement_features["engagement_consistency"] = consistency_score
                    engagement_features["active_days"] = len(daily_engagement)
                    engagement_features["active_days_pct"] = (len(daily_engagement) / window_days) * 100

            # Overall engagement score (average of components)
            if engagement_components:
                overall_engagement = np.mean(engagement_components)
                engagement_features["overall_engagement_score"] = overall_engagement

                # Engagement level category
                if overall_engagement >= 0.7:
                    engagement_features["engagement_level"] = "high"
                elif overall_engagement >= 0.4:
                    engagement_features["engagement_level"] = "moderate"
                else:
                    engagement_features["engagement_level"] = "low"
            else:
                engagement_features["overall_engagement_score"] = 0.0
                engagement_features["engagement_level"] = "none"

            engagement_list.append(engagement_features)

        result_df = pd.DataFrame(engagement_list)

        logger.info(f"Computed engagement scores for {len(result_df)} individuals")

        return result_df

    def extract_all_adherence_features(
        self,
        adherence_df: Optional[DataFrame] = None,
        sessions_df: Optional[DataFrame] = None,
        engagement_df: Optional[DataFrame] = None,
        id_column: str = "anonymized_id",
    ) -> DataFrame:
        """
        Extract all adherence features and merge into single DataFrame.

        Args:
            adherence_df: Optional DataFrame with adherence data
            sessions_df: Optional DataFrame with session data
            engagement_df: Optional DataFrame with engagement data
            id_column: Column containing anonymized identifiers

        Returns:
            DataFrame with all adherence features per individual
        """
        result_dfs = []

        # Calculate adherence rates
        if adherence_df is not None and not adherence_df.empty:
            try:
                adherence_features = self.calculate_adherence_rate(
                    adherence_df, id_column=id_column
                )
                if not adherence_features.empty:
                    result_dfs.append(adherence_features)
            except Exception as e:
                logger.warning(f"Failed to calculate adherence rates: {e}")

        # Flag missed sessions
        if sessions_df is not None and not sessions_df.empty:
            try:
                missed_features = self.flag_missed_sessions(sessions_df, id_column=id_column)
                if not missed_features.empty:
                    result_dfs.append(missed_features)
            except Exception as e:
                logger.warning(f"Failed to flag missed sessions: {e}")

        # Compute engagement scores
        if engagement_df is not None and not engagement_df.empty:
            try:
                engagement_features = self.compute_engagement_score(
                    engagement_df, id_column=id_column
                )
                if not engagement_features.empty:
                    result_dfs.append(engagement_features)
            except Exception as e:
                logger.warning(f"Failed to compute engagement scores: {e}")

        # Merge all features
        if result_dfs:
            result = result_dfs[0]
            for df in result_dfs[1:]:
                result = result.merge(df, on=id_column, how="outer")

            logger.info(f"Extracted all adherence features for {len(result)} individuals")
            return result
        else:
            logger.warning("No adherence features extracted. Returning empty DataFrame.")
            return pd.DataFrame()

    def get_adherence_stats(self) -> Dict:
        """
        Get statistics from the last adherence tracking operation.

        Returns:
            Dictionary containing adherence statistics
        """
        return self.adherence_stats.copy()
