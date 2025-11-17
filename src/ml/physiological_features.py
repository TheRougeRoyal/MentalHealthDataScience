"""
Physiological feature extraction module for wearable data including sleep and HRV metrics.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class PhysiologicalFeatureExtractor:
    """
    Extracts physiological features from wearable data including sleep quality,
    heart rate variability (HRV), and activity intensity metrics.
    """

    def __init__(self):
        """Initialize PhysiologicalFeatureExtractor."""
        self.feature_stats: Dict = {}

    def extract_sleep_features(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        sleep_start_column: str = "sleep_start",
        sleep_end_column: str = "sleep_end",
        interruptions_column: Optional[str] = "interruptions",
        sleep_stages_column: Optional[str] = "sleep_stages",
    ) -> DataFrame:
        """
        Extract sleep quality metrics including duration, efficiency, and interruptions.

        Args:
            df: Input DataFrame with sleep data
            id_column: Column containing anonymized identifiers
            sleep_start_column: Column containing sleep start time
            sleep_end_column: Column containing sleep end time
            interruptions_column: Optional column containing interruption count
            sleep_stages_column: Optional column containing sleep stage data

        Returns:
            DataFrame with sleep features per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        df = df.copy()

        # Ensure datetime columns
        if sleep_start_column in df.columns:
            df[sleep_start_column] = pd.to_datetime(df[sleep_start_column])
        if sleep_end_column in df.columns:
            df[sleep_end_column] = pd.to_datetime(df[sleep_end_column])

        sleep_features_list = []

        for individual_id, group in df.groupby(id_column):
            sleep_features = {id_column: individual_id}

            # Calculate sleep duration
            if sleep_start_column in group.columns and sleep_end_column in group.columns:
                group["sleep_duration_hours"] = (
                    group[sleep_end_column] - group[sleep_start_column]
                ).dt.total_seconds() / 3600

                # Filter out invalid durations (negative or > 24 hours)
                valid_durations = group["sleep_duration_hours"][
                    (group["sleep_duration_hours"] > 0) & (group["sleep_duration_hours"] <= 24)
                ]

                if not valid_durations.empty:
                    sleep_features["mean_sleep_duration"] = valid_durations.mean()
                    sleep_features["median_sleep_duration"] = valid_durations.median()
                    sleep_features["std_sleep_duration"] = valid_durations.std()
                    sleep_features["min_sleep_duration"] = valid_durations.min()
                    sleep_features["max_sleep_duration"] = valid_durations.max()

                    # Sleep regularity (coefficient of variation)
                    if valid_durations.mean() > 0:
                        cv = valid_durations.std() / valid_durations.mean()
                        sleep_features["sleep_regularity"] = 1 / (1 + cv)
                    else:
                        sleep_features["sleep_regularity"] = 0.0
                else:
                    sleep_features["mean_sleep_duration"] = 0.0
                    sleep_features["median_sleep_duration"] = 0.0
                    sleep_features["std_sleep_duration"] = 0.0
                    sleep_features["min_sleep_duration"] = 0.0
                    sleep_features["max_sleep_duration"] = 0.0
                    sleep_features["sleep_regularity"] = 0.0

            # Sleep interruptions
            if interruptions_column and interruptions_column in group.columns:
                interruptions = group[interruptions_column].dropna()
                if not interruptions.empty:
                    sleep_features["mean_interruptions"] = interruptions.mean()
                    sleep_features["median_interruptions"] = interruptions.median()
                    sleep_features["max_interruptions"] = interruptions.max()
                else:
                    sleep_features["mean_interruptions"] = 0.0
                    sleep_features["median_interruptions"] = 0.0
                    sleep_features["max_interruptions"] = 0.0

            # Sleep efficiency (if available)
            if "sleep_efficiency" in group.columns:
                efficiency = group["sleep_efficiency"].dropna()
                if not efficiency.empty:
                    sleep_features["mean_sleep_efficiency"] = efficiency.mean()
                    sleep_features["median_sleep_efficiency"] = efficiency.median()
                else:
                    sleep_features["mean_sleep_efficiency"] = 0.0
                    sleep_features["median_sleep_efficiency"] = 0.0
            elif sleep_start_column in group.columns and sleep_end_column in group.columns:
                # Estimate efficiency from duration (assuming 7-9 hours is optimal)
                if "sleep_duration_hours" in group.columns:
                    durations = group["sleep_duration_hours"].dropna()
                    if not durations.empty:
                        # Efficiency based on proximity to 8 hours
                        efficiency = 1 - np.abs(durations - 8) / 8
                        efficiency = efficiency.clip(0, 1)
                        sleep_features["mean_sleep_efficiency"] = efficiency.mean()
                        sleep_features["median_sleep_efficiency"] = efficiency.median()

            sleep_features_list.append(sleep_features)

        result_df = pd.DataFrame(sleep_features_list)

        logger.info(f"Extracted sleep features for {len(result_df)} individuals")

        return result_df

    def compute_hrv_metrics(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        rr_intervals_column: Optional[str] = "rr_intervals",
        heart_rate_column: Optional[str] = "heart_rate",
        timestamp_column: str = "timestamp",
    ) -> DataFrame:
        """
        Compute heart rate variability (HRV) metrics including RMSSD, SDNN, and LF/HF ratio.

        Args:
            df: Input DataFrame with heart rate data
            id_column: Column containing anonymized identifiers
            rr_intervals_column: Optional column containing RR intervals (ms)
            heart_rate_column: Optional column containing heart rate (bpm)
            timestamp_column: Column containing timestamps

        Returns:
            DataFrame with HRV metrics per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        hrv_features_list = []

        for individual_id, group in df.groupby(id_column):
            hrv_features = {id_column: individual_id}

            # Compute HRV from RR intervals if available
            if rr_intervals_column and rr_intervals_column in group.columns:
                rr_intervals = group[rr_intervals_column].dropna()

                if len(rr_intervals) > 1:
                    # RMSSD (Root Mean Square of Successive Differences)
                    successive_diffs = np.diff(rr_intervals)
                    rmssd = np.sqrt(np.mean(successive_diffs**2))
                    hrv_features["rmssd"] = rmssd

                    # SDNN (Standard Deviation of NN intervals)
                    sdnn = np.std(rr_intervals, ddof=1)
                    hrv_features["sdnn"] = sdnn

                    # pNN50 (percentage of successive RR intervals that differ by more than 50ms)
                    nn50 = np.sum(np.abs(successive_diffs) > 50)
                    pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
                    hrv_features["pnn50"] = pnn50

                    # Frequency domain features (simplified)
                    # Note: Full frequency domain analysis requires proper signal processing
                    # This is a simplified estimation
                    hrv_features["mean_rr"] = np.mean(rr_intervals)
                    hrv_features["std_rr"] = np.std(rr_intervals)

                    # Estimate LF/HF ratio (simplified)
                    # In practice, this requires FFT and proper frequency band analysis
                    # Here we use a proxy based on variability patterns
                    if sdnn > 0:
                        lf_hf_ratio = rmssd / sdnn
                        hrv_features["lf_hf_ratio_proxy"] = lf_hf_ratio
                    else:
                        hrv_features["lf_hf_ratio_proxy"] = 0.0
                else:
                    hrv_features["rmssd"] = 0.0
                    hrv_features["sdnn"] = 0.0
                    hrv_features["pnn50"] = 0.0
                    hrv_features["mean_rr"] = 0.0
                    hrv_features["std_rr"] = 0.0
                    hrv_features["lf_hf_ratio_proxy"] = 0.0

            # Compute basic HR statistics if RR intervals not available
            elif heart_rate_column and heart_rate_column in group.columns:
                heart_rates = group[heart_rate_column].dropna()

                if not heart_rates.empty:
                    hrv_features["mean_heart_rate"] = heart_rates.mean()
                    hrv_features["std_heart_rate"] = heart_rates.std()
                    hrv_features["min_heart_rate"] = heart_rates.min()
                    hrv_features["max_heart_rate"] = heart_rates.max()
                    hrv_features["heart_rate_range"] = (
                        heart_rates.max() - heart_rates.min()
                    )

                    # Estimate HRV from HR variability (less accurate)
                    if len(heart_rates) > 1:
                        hr_diffs = np.diff(heart_rates)
                        hrv_features["hr_variability"] = np.std(hr_diffs)
                else:
                    hrv_features["mean_heart_rate"] = 0.0
                    hrv_features["std_heart_rate"] = 0.0
                    hrv_features["min_heart_rate"] = 0.0
                    hrv_features["max_heart_rate"] = 0.0
                    hrv_features["heart_rate_range"] = 0.0
                    hrv_features["hr_variability"] = 0.0

            hrv_features_list.append(hrv_features)

        result_df = pd.DataFrame(hrv_features_list)

        logger.info(f"Computed HRV metrics for {len(result_df)} individuals")

        return result_df

    def calculate_activity_intensity(
        self,
        df: DataFrame,
        id_column: str = "anonymized_id",
        heart_rate_column: str = "heart_rate",
        steps_column: Optional[str] = "steps",
        calories_column: Optional[str] = "calories",
        timestamp_column: str = "timestamp",
    ) -> DataFrame:
        """
        Calculate activity intensity from heart rate zones and activity metrics.

        Args:
            df: Input DataFrame with activity data
            id_column: Column containing anonymized identifiers
            heart_rate_column: Column containing heart rate (bpm)
            steps_column: Optional column containing step count
            calories_column: Optional column containing calories burned
            timestamp_column: Column containing timestamps

        Returns:
            DataFrame with activity intensity features per individual
        """
        if df.empty:
            logger.warning("Empty DataFrame provided. Returning empty result.")
            return pd.DataFrame()

        activity_features_list = []

        for individual_id, group in df.groupby(id_column):
            activity_features = {id_column: individual_id}

            # Heart rate zones (using standard zones)
            # Zone 1: 50-60% max HR (very light)
            # Zone 2: 60-70% max HR (light)
            # Zone 3: 70-80% max HR (moderate)
            # Zone 4: 80-90% max HR (hard)
            # Zone 5: 90-100% max HR (maximum)
            # Assuming max HR = 220 - age, we'll use a default or estimate

            if heart_rate_column in group.columns:
                heart_rates = group[heart_rate_column].dropna()

                if not heart_rates.empty:
                    # Estimate max HR (using 220 - 30 as default, assuming average age 30)
                    estimated_max_hr = 190

                    # Calculate time in each zone
                    zone1 = ((heart_rates >= 0.5 * estimated_max_hr) & 
                            (heart_rates < 0.6 * estimated_max_hr)).sum()
                    zone2 = ((heart_rates >= 0.6 * estimated_max_hr) & 
                            (heart_rates < 0.7 * estimated_max_hr)).sum()
                    zone3 = ((heart_rates >= 0.7 * estimated_max_hr) & 
                            (heart_rates < 0.8 * estimated_max_hr)).sum()
                    zone4 = ((heart_rates >= 0.8 * estimated_max_hr) & 
                            (heart_rates < 0.9 * estimated_max_hr)).sum()
                    zone5 = (heart_rates >= 0.9 * estimated_max_hr).sum()

                    total_readings = len(heart_rates)
                    activity_features["zone1_pct"] = (zone1 / total_readings) * 100
                    activity_features["zone2_pct"] = (zone2 / total_readings) * 100
                    activity_features["zone3_pct"] = (zone3 / total_readings) * 100
                    activity_features["zone4_pct"] = (zone4 / total_readings) * 100
                    activity_features["zone5_pct"] = (zone5 / total_readings) * 100

                    # Overall activity intensity score (weighted by zone)
                    intensity_score = (
                        zone1 * 1 + zone2 * 2 + zone3 * 3 + zone4 * 4 + zone5 * 5
                    ) / total_readings
                    activity_features["activity_intensity_score"] = intensity_score

                    # Active time (zones 3-5)
                    active_time_pct = ((zone3 + zone4 + zone5) / total_readings) * 100
                    activity_features["active_time_pct"] = active_time_pct

            # Steps metrics
            if steps_column and steps_column in group.columns:
                steps = group[steps_column].dropna()

                if not steps.empty:
                    activity_features["total_steps"] = steps.sum()
                    activity_features["mean_daily_steps"] = steps.mean()
                    activity_features["median_daily_steps"] = steps.median()
                    activity_features["max_daily_steps"] = steps.max()

                    # Step consistency
                    if steps.mean() > 0:
                        cv = steps.std() / steps.mean()
                        activity_features["step_consistency"] = 1 / (1 + cv)
                    else:
                        activity_features["step_consistency"] = 0.0

            # Calories metrics
            if calories_column and calories_column in group.columns:
                calories = group[calories_column].dropna()

                if not calories.empty:
                    activity_features["total_calories"] = calories.sum()
                    activity_features["mean_daily_calories"] = calories.mean()
                    activity_features["median_daily_calories"] = calories.median()

            activity_features_list.append(activity_features)

        result_df = pd.DataFrame(activity_features_list)

        logger.info(f"Calculated activity intensity for {len(result_df)} individuals")

        return result_df

    def extract_all_physiological_features(
        self,
        sleep_df: Optional[DataFrame] = None,
        hrv_df: Optional[DataFrame] = None,
        activity_df: Optional[DataFrame] = None,
        id_column: str = "anonymized_id",
    ) -> DataFrame:
        """
        Extract all physiological features and merge into single DataFrame.
        Handles missing wearable data gracefully.

        Args:
            sleep_df: Optional DataFrame with sleep data
            hrv_df: Optional DataFrame with HRV data
            activity_df: Optional DataFrame with activity data
            id_column: Column containing anonymized identifiers

        Returns:
            DataFrame with all physiological features per individual
        """
        result_dfs = []

        # Extract sleep features
        if sleep_df is not None and not sleep_df.empty:
            try:
                sleep_features = self.extract_sleep_features(sleep_df, id_column=id_column)
                if not sleep_features.empty:
                    result_dfs.append(sleep_features)
            except Exception as e:
                logger.warning(f"Failed to extract sleep features: {e}")

        # Extract HRV features
        if hrv_df is not None and not hrv_df.empty:
            try:
                hrv_features = self.compute_hrv_metrics(hrv_df, id_column=id_column)
                if not hrv_features.empty:
                    result_dfs.append(hrv_features)
            except Exception as e:
                logger.warning(f"Failed to extract HRV features: {e}")

        # Extract activity features
        if activity_df is not None and not activity_df.empty:
            try:
                activity_features = self.calculate_activity_intensity(
                    activity_df, id_column=id_column
                )
                if not activity_features.empty:
                    result_dfs.append(activity_features)
            except Exception as e:
                logger.warning(f"Failed to extract activity features: {e}")

        # Merge all features
        if result_dfs:
            result = result_dfs[0]
            for df in result_dfs[1:]:
                result = result.merge(df, on=id_column, how="outer")

            logger.info(
                f"Extracted all physiological features for {len(result)} individuals"
            )
            return result
        else:
            logger.warning("No physiological features extracted. Returning empty DataFrame.")
            return pd.DataFrame()

    def get_feature_stats(self) -> Dict:
        """
        Get statistics from the last feature extraction operation.

        Returns:
            Dictionary containing feature extraction statistics
        """
        return self.feature_stats.copy()
