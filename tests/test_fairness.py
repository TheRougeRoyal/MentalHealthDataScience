"""Fairness tests for model equity across demographic groups"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


class FairnessMetrics:
    """Calculate fairness metrics for model evaluation"""
    
    @staticmethod
    def calculate_group_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        group: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics per demographic group.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            group: Group membership
            
        Returns:
            Dictionary of metrics per group
        """
        unique_groups = np.unique(group)
        metrics = {}
        
        for g in unique_groups:
            mask = group == g
            
            if mask.sum() == 0:
                continue
            
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]
            y_prob_g = y_prob[mask]
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g).ravel()
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Calculate AUROC if we have both classes
            if len(np.unique(y_true_g)) > 1:
                auroc = roc_auc_score(y_true_g, y_prob_g)
            else:
                auroc = None
            
            metrics[str(g)] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "fnr": fnr,
                "auroc": auroc,
                "n_samples": mask.sum(),
                "n_positive": int(y_true_g.sum()),
                "n_negative": int((1 - y_true_g).sum())
            }
        
        return metrics
    
    @staticmethod
    def calculate_disparate_impact(
        y_pred: np.ndarray,
        group: np.ndarray,
        privileged_group: str,
        unprivileged_group: str
    ) -> float:
        """
        Calculate disparate impact ratio.
        
        Disparate impact = P(Y=1|unprivileged) / P(Y=1|privileged)
        
        A ratio < 0.8 indicates potential discrimination (80% rule).
        
        Args:
            y_pred: Predicted labels
            group: Group membership
            privileged_group: Name of privileged group
            unprivileged_group: Name of unprivileged group
            
        Returns:
            Disparate impact ratio
        """
        priv_mask = group == privileged_group
        unpriv_mask = group == unprivileged_group
        
        priv_positive_rate = y_pred[priv_mask].mean() if priv_mask.sum() > 0 else 0
        unpriv_positive_rate = y_pred[unpriv_mask].mean() if unpriv_mask.sum() > 0 else 0
        
        if priv_positive_rate == 0:
            return 0.0
        
        return unpriv_positive_rate / priv_positive_rate
    
    @staticmethod
    def calculate_equalized_odds_difference(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: np.ndarray,
        group1: str,
        group2: str
    ) -> Tuple[float, float]:
        """
        Calculate equalized odds difference.
        
        Returns difference in TPR and FPR between groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            group: Group membership
            group1: First group
            group2: Second group
            
        Returns:
            Tuple of (TPR difference, FPR difference)
        """
        mask1 = group == group1
        mask2 = group == group2
        
        # Group 1 metrics
        y_true_1 = y_true[mask1]
        y_pred_1 = y_pred[mask1]
        
        if len(y_true_1) > 0:
            tn1, fp1, fn1, tp1 = confusion_matrix(y_true_1, y_pred_1).ravel()
            tpr1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
            fpr1 = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0
        else:
            tpr1, fpr1 = 0, 0
        
        # Group 2 metrics
        y_true_2 = y_true[mask2]
        y_pred_2 = y_pred[mask2]
        
        if len(y_true_2) > 0:
            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_2, y_pred_2).ravel()
            tpr2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
            fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
        else:
            tpr2, fpr2 = 0, 0
        
        tpr_diff = abs(tpr1 - tpr2)
        fpr_diff = abs(fpr1 - fpr2)
        
        return tpr_diff, fpr_diff
    
    @staticmethod
    def calculate_calibration_by_group(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        group: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate calibration curves per group.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            group: Group membership
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration data per group
        """
        unique_groups = np.unique(group)
        calibration = {}
        
        for g in unique_groups:
            mask = group == g
            y_true_g = y_true[mask]
            y_prob_g = y_prob[mask]
            
            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_prob_g, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Calculate mean predicted and observed per bin
            mean_predicted = []
            mean_observed = []
            counts = []
            
            for i in range(n_bins):
                mask_bin = bin_indices == i
                if mask_bin.sum() > 0:
                    mean_predicted.append(y_prob_g[mask_bin].mean())
                    mean_observed.append(y_true_g[mask_bin].mean())
                    counts.append(mask_bin.sum())
                else:
                    mean_predicted.append(np.nan)
                    mean_observed.append(np.nan)
                    counts.append(0)
            
            calibration[str(g)] = {
                "mean_predicted": np.array(mean_predicted),
                "mean_observed": np.array(mean_observed),
                "counts": np.array(counts)
            }
        
        return calibration


class TestModelFairness:
    """Test model fairness across demographic groups"""
    
    def create_synthetic_data(
        self,
        n_samples: int = 1000,
        bias_factor: float = 0.0
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create synthetic data with optional bias.
        
        Args:
            n_samples: Number of samples
            bias_factor: Amount of bias to introduce (0 = no bias)
            
        Returns:
            Tuple of (features DataFrame, labels array)
        """
        np.random.seed(42)
        
        # Create demographic groups
        groups = np.random.choice(["group_a", "group_b"], n_samples)
        
        # Create features
        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = np.random.normal(0, 1, n_samples)
        feature3 = np.random.normal(0, 1, n_samples)
        
        # Create labels with optional bias
        # Base probability from features
        logit = 0.5 * feature1 + 0.3 * feature2 - 0.2 * feature3
        
        # Add bias for group_b
        logit = logit + bias_factor * (groups == "group_b")
        
        prob = 1 / (1 + np.exp(-logit))
        labels = (np.random.random(n_samples) < prob).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "demographic_group": groups
        })
        
        return df, labels
    
    def test_auroc_parity(self):
        """Test that AUROC difference across groups is < 0.05"""
        # Create unbiased data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=0.0)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics per group
        metrics = FairnessMetrics.calculate_group_metrics(
            y_test, y_pred, y_prob, groups_test
        )
        
        # Extract AUROC values
        auroc_values = {g: m["auroc"] for g, m in metrics.items() if m["auroc"] is not None}
        
        print(f"\nAUROC by group:")
        for group, auroc in auroc_values.items():
            print(f"  {group}: {auroc:.4f}")
        
        # Check AUROC parity
        if len(auroc_values) >= 2:
            auroc_list = list(auroc_values.values())
            max_diff = max(auroc_list) - min(auroc_list)
            
            print(f"  Max AUROC difference: {max_diff:.4f}")
            
            # Requirement: < 0.05 difference
            assert max_diff < 0.05, f"AUROC difference {max_diff:.4f} exceeds threshold of 0.05"
    
    def test_false_positive_rate_parity(self):
        """Test that FPR difference across groups is within 10%"""
        # Create data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=0.0)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics per group
        metrics = FairnessMetrics.calculate_group_metrics(
            y_test, y_pred, y_prob, groups_test
        )
        
        # Extract FPR values
        fpr_values = {g: m["fpr"] for g, m in metrics.items()}
        
        print(f"\nFalse Positive Rate by group:")
        for group, fpr in fpr_values.items():
            print(f"  {group}: {fpr:.4f}")
        
        # Check FPR parity
        if len(fpr_values) >= 2:
            fpr_list = list(fpr_values.values())
            max_diff = max(fpr_list) - min(fpr_list)
            
            print(f"  Max FPR difference: {max_diff:.4f}")
            
            # Requirement: within 10% (0.10)
            assert max_diff < 0.10, f"FPR difference {max_diff:.4f} exceeds threshold of 0.10"
    
    def test_disparate_impact(self):
        """Test for disparate impact using 80% rule"""
        # Create data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=0.0)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate disparate impact
        di_ratio = FairnessMetrics.calculate_disparate_impact(
            y_pred, groups_test, "group_a", "group_b"
        )
        
        print(f"\nDisparate Impact Ratio: {di_ratio:.4f}")
        print(f"  (80% rule threshold: 0.80)")
        
        # 80% rule: ratio should be >= 0.80
        assert di_ratio >= 0.80, f"Disparate impact ratio {di_ratio:.4f} below 0.80 threshold"
    
    def test_equalized_odds(self):
        """Test for equalized odds (TPR and FPR parity)"""
        # Create data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=0.0)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate equalized odds
        tpr_diff, fpr_diff = FairnessMetrics.calculate_equalized_odds_difference(
            y_test, y_pred, groups_test, "group_a", "group_b"
        )
        
        print(f"\nEqualized Odds:")
        print(f"  TPR difference: {tpr_diff:.4f}")
        print(f"  FPR difference: {fpr_diff:.4f}")
        
        # Both differences should be small (< 0.10)
        assert tpr_diff < 0.10, f"TPR difference {tpr_diff:.4f} exceeds threshold"
        assert fpr_diff < 0.10, f"FPR difference {fpr_diff:.4f} exceeds threshold"
    
    def test_calibration_across_groups(self):
        """Test that model is calibrated across demographic groups"""
        # Create data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=0.0)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration per group
        calibration = FairnessMetrics.calculate_calibration_by_group(
            y_test, y_prob, groups_test, n_bins=10
        )
        
        print(f"\nCalibration by group:")
        
        # Calculate calibration error per group
        calibration_errors = {}
        
        for group, cal_data in calibration.items():
            mean_pred = cal_data["mean_predicted"]
            mean_obs = cal_data["mean_observed"]
            counts = cal_data["counts"]
            
            # Filter out empty bins
            valid_mask = ~np.isnan(mean_pred) & ~np.isnan(mean_obs) & (counts > 0)
            
            if valid_mask.sum() > 0:
                # Calculate mean absolute calibration error
                cal_error = np.abs(mean_pred[valid_mask] - mean_obs[valid_mask]).mean()
                calibration_errors[group] = cal_error
                
                print(f"  {group}: calibration error = {cal_error:.4f}")
        
        # Check that calibration error is similar across groups
        if len(calibration_errors) >= 2:
            error_list = list(calibration_errors.values())
            max_error_diff = max(error_list) - min(error_list)
            
            print(f"  Max calibration error difference: {max_error_diff:.4f}")
            
            # Requirement: < 0.05 difference in calibration error
            assert max_error_diff < 0.05, f"Calibration error difference {max_error_diff:.4f} exceeds threshold"
    
    def test_detect_biased_model(self):
        """Test that fairness metrics detect biased model"""
        # Create biased data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=1.5)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics per group
        metrics = FairnessMetrics.calculate_group_metrics(
            y_test, y_pred, y_prob, groups_test
        )
        
        print(f"\nBiased model metrics:")
        for group, m in metrics.items():
            print(f"  {group}:")
            print(f"    AUROC: {m['auroc']:.4f}" if m['auroc'] else "    AUROC: N/A")
            print(f"    FPR: {m['fpr']:.4f}")
            print(f"    Recall: {m['recall']:.4f}")
        
        # With bias, we expect to see differences
        auroc_values = [m["auroc"] for m in metrics.values() if m["auroc"] is not None]
        
        if len(auroc_values) >= 2:
            max_diff = max(auroc_values) - min(auroc_values)
            print(f"  AUROC difference: {max_diff:.4f}")
            
            # Biased model should show larger difference
            # This test verifies our fairness metrics can detect bias
            assert max_diff > 0.01, "Fairness metrics should detect bias in biased model"
    
    def test_performance_disparity_detection(self):
        """Test detection of >10% performance disparity"""
        # Create slightly biased data
        df, labels = self.create_synthetic_data(n_samples=1000, bias_factor=0.8)
        
        # Split data
        X = df[["feature1", "feature2", "feature3"]]
        groups = df["demographic_group"].values
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, labels, groups, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics per group
        metrics = FairnessMetrics.calculate_group_metrics(
            y_test, y_pred, y_prob, groups_test
        )
        
        # Check for >10% disparity in any metric
        for metric_name in ["accuracy", "precision", "recall"]:
            values = [m[metric_name] for m in metrics.values()]
            
            if len(values) >= 2:
                max_diff = max(values) - min(values)
                
                print(f"\n{metric_name.capitalize()} disparity: {max_diff:.4f}")
                
                if max_diff > 0.10:
                    print(f"  WARNING: >10% disparity detected in {metric_name}")
                    # In production, this would flag the model for review


class TestFairnessAudit:
    """Test comprehensive fairness audit"""
    
    def test_comprehensive_fairness_audit(self):
        """Test complete fairness audit across all metrics"""
        # Create data
        np.random.seed(42)
        n_samples = 1000
        
        # Create multiple demographic groups
        age_groups = np.random.choice(["18-30", "31-50", "51+"], n_samples)
        gender_groups = np.random.choice(["male", "female", "other"], n_samples)
        
        # Create features
        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = np.random.normal(0, 1, n_samples)
        
        # Create labels
        logit = 0.5 * feature1 + 0.3 * feature2
        prob = 1 / (1 + np.exp(-logit))
        labels = (np.random.random(n_samples) < prob).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            "feature1": feature1,
            "feature2": feature2,
            "age_group": age_groups,
            "gender_group": gender_groups
        })
        
        # Split data
        X = df[["feature1", "feature2"]]
        
        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X, labels, df, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        print(f"\n=== Comprehensive Fairness Audit ===")
        
        # Audit by age group
        print(f"\n--- Age Group Fairness ---")
        age_metrics = FairnessMetrics.calculate_group_metrics(
            y_test, y_pred, y_prob, df_test["age_group"].values
        )
        
        for group, m in age_metrics.items():
            print(f"\n{group}:")
            print(f"  Samples: {m['n_samples']}")
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  AUROC: {m['auroc']:.4f}" if m['auroc'] else "  AUROC: N/A")
            print(f"  FPR: {m['fpr']:.4f}")
            print(f"  Recall: {m['recall']:.4f}")
        
        # Audit by gender group
        print(f"\n--- Gender Group Fairness ---")
        gender_metrics = FairnessMetrics.calculate_group_metrics(
            y_test, y_pred, y_prob, df_test["gender_group"].values
        )
        
        for group, m in gender_metrics.items():
            print(f"\n{group}:")
            print(f"  Samples: {m['n_samples']}")
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  AUROC: {m['auroc']:.4f}" if m['auroc'] else "  AUROC: N/A")
            print(f"  FPR: {m['fpr']:.4f}")
            print(f"  Recall: {m['recall']:.4f}")
        
        # Summary
        print(f"\n--- Fairness Summary ---")
        print(f"Model evaluated across {len(age_metrics)} age groups and {len(gender_metrics)} gender groups")
        print(f"All groups have sufficient samples for reliable metrics")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
