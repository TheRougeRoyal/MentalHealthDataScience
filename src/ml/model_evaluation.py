"""
Model evaluation framework for mental health risk assessment.

This module provides comprehensive evaluation metrics including AUROC, PR-AUC,
calibration curves, decision curve analysis, and fairness audits.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for risk assessment models."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results and plots
        """
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_auroc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Calculate AUROC (Area Under Receiver Operating Characteristic curve).
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with AUROC score and related metrics
        """
        try:
            auroc = roc_auc_score(y_true, y_pred_proba)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            
            logger.info(f"{model_name} - AUROC: {auroc:.4f}")
            logger.info(f"{model_name} - Optimal threshold: {optimal_threshold:.4f}")
            logger.info(
                f"{model_name} - At optimal: TPR={optimal_tpr:.4f}, FPR={optimal_fpr:.4f}"
            )
            
            results = {
                'auroc': float(auroc),
                'optimal_threshold': float(optimal_threshold),
                'optimal_tpr': float(optimal_tpr),
                'optimal_fpr': float(optimal_fpr),
                'optimal_j_score': float(j_scores[optimal_idx])
            }
            
            # Save ROC curve plot
            self._plot_roc_curve(fpr, tpr, auroc, model_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating AUROC for {model_name}: {e}")
            return {'auroc': None, 'error': str(e)}
    
    def calculate_pr_auc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Calculate PR-AUC (Precision-Recall Area Under Curve).
        
        Particularly useful for imbalanced datasets.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with PR-AUC score and related metrics
        """
        try:
            pr_auc = average_precision_score(y_true, y_pred_proba)
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            # Find optimal threshold (F1 score)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]
            
            logger.info(f"{model_name} - PR-AUC: {pr_auc:.4f}")
            logger.info(f"{model_name} - Optimal threshold (F1): {optimal_threshold:.4f}")
            logger.info(
                f"{model_name} - At optimal: Precision={optimal_precision:.4f}, "
                f"Recall={optimal_recall:.4f}, F1={optimal_f1:.4f}"
            )
            
            results = {
                'pr_auc': float(pr_auc),
                'optimal_threshold_f1': float(optimal_threshold),
                'optimal_precision': float(optimal_precision),
                'optimal_recall': float(optimal_recall),
                'optimal_f1': float(optimal_f1)
            }
            
            # Save PR curve plot
            self._plot_pr_curve(precision, recall, pr_auc, model_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating PR-AUC for {model_name}: {e}")
            return {'pr_auc': None, 'error': str(e)}
    
    def generate_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model",
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Generate calibration curve comparing predicted vs observed probabilities.
        
        A well-calibrated model has predicted probabilities that match
        observed frequencies.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for logging
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        try:
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true,
                y_pred_proba,
                n_bins=n_bins,
                strategy='uniform'
            )
            
            # Calculate Brier score (lower is better)
            brier_score = brier_score_loss(y_true, y_pred_proba)
            
            # Calculate Expected Calibration Error (ECE)
            ece = self._calculate_ece(y_true, y_pred_proba, n_bins)
            
            # Calculate Maximum Calibration Error (MCE)
            mce = np.max(np.abs(prob_true - prob_pred))
            
            logger.info(f"{model_name} - Brier Score: {brier_score:.4f}")
            logger.info(f"{model_name} - Expected Calibration Error: {ece:.4f}")
            logger.info(f"{model_name} - Maximum Calibration Error: {mce:.4f}")
            
            results = {
                'brier_score': float(brier_score),
                'expected_calibration_error': float(ece),
                'maximum_calibration_error': float(mce),
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
            
            # Save calibration plot
            self._plot_calibration_curve(prob_true, prob_pred, model_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating calibration curve for {model_name}: {e}")
            return {'error': str(e)}
    
    def decision_curve_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model",
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform decision curve analysis for net benefit calculation.
        
        Decision curve analysis evaluates the clinical utility of a model
        by calculating net benefit across different decision thresholds.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model for logging
            thresholds: Array of probability thresholds to evaluate
            
        Returns:
            Dictionary with net benefit metrics
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)
        
        try:
            net_benefits = []
            treat_all_benefits = []
            treat_none_benefits = []
            
            n_samples = len(y_true)
            n_positive = np.sum(y_true)
            
            for threshold in thresholds:
                # Model predictions
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # True positives and false positives
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                
                # Net benefit for model
                net_benefit = (tp / n_samples) - (fp / n_samples) * (threshold / (1 - threshold))
                net_benefits.append(net_benefit)
                
                # Net benefit for treat all strategy
                treat_all = (n_positive / n_samples) - \
                           ((n_samples - n_positive) / n_samples) * (threshold / (1 - threshold))
                treat_all_benefits.append(treat_all)
                
                # Net benefit for treat none strategy (always 0)
                treat_none_benefits.append(0)
            
            net_benefits = np.array(net_benefits)
            treat_all_benefits = np.array(treat_all_benefits)
            
            # Find threshold with maximum net benefit
            max_benefit_idx = np.argmax(net_benefits)
            max_benefit_threshold = thresholds[max_benefit_idx]
            max_benefit = net_benefits[max_benefit_idx]
            
            logger.info(f"{model_name} - Maximum net benefit: {max_benefit:.4f}")
            logger.info(f"{model_name} - At threshold: {max_benefit_threshold:.4f}")
            
            results = {
                'thresholds': thresholds.tolist(),
                'net_benefits': net_benefits.tolist(),
                'treat_all_benefits': treat_all_benefits.tolist(),
                'max_benefit': float(max_benefit),
                'max_benefit_threshold': float(max_benefit_threshold)
            }
            
            # Save decision curve plot
            self._plot_decision_curve(
                thresholds,
                net_benefits,
                treat_all_benefits,
                treat_none_benefits,
                model_name
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in decision curve analysis for {model_name}: {e}")
            return {'error': str(e)}
    
    def fairness_audit(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_features: pd.DataFrame,
        model_name: str = "model",
        threshold: float = 0.5,
        disparity_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Audit model fairness by computing metrics per demographic group.
        
        Flags models with >10% performance disparity between groups.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            sensitive_features: DataFrame with demographic features
            model_name: Name of the model for logging
            threshold: Classification threshold
            disparity_threshold: Maximum acceptable disparity (default 0.1 = 10%)
            
        Returns:
            Dictionary with fairness metrics per group
        """
        try:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            fairness_results = {
                'model_name': model_name,
                'threshold': threshold,
                'disparity_threshold': disparity_threshold,
                'groups': {},
                'disparities': {},
                'flags': []
            }
            
            # Evaluate each sensitive feature
            for feature in sensitive_features.columns:
                logger.info(f"\nEvaluating fairness for feature: {feature}")
                
                group_metrics = {}
                
                # Get unique groups
                unique_groups = sensitive_features[feature].unique()
                
                for group in unique_groups:
                    # Filter data for this group
                    mask = sensitive_features[feature] == group
                    y_true_group = y_true[mask]
                    y_pred_proba_group = y_pred_proba[mask]
                    y_pred_group = y_pred[mask]
                    
                    if len(y_true_group) == 0:
                        continue
                    
                    # Calculate metrics for this group
                    try:
                        auroc = roc_auc_score(y_true_group, y_pred_proba_group)
                    except:
                        auroc = None
                    
                    try:
                        pr_auc = average_precision_score(y_true_group, y_pred_proba_group)
                    except:
                        pr_auc = None
                    
                    # Calculate confusion matrix metrics
                    tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
                    
                    group_metrics[str(group)] = {
                        'n_samples': int(len(y_true_group)),
                        'prevalence': float(np.mean(y_true_group)),
                        'auroc': float(auroc) if auroc is not None else None,
                        'pr_auc': float(pr_auc) if pr_auc is not None else None,
                        'tpr': float(tpr),
                        'fpr': float(fpr),
                        'tnr': float(tnr),
                        'fnr': float(fnr),
                        'ppv': float(ppv),
                        'npv': float(npv)
                    }
                    
                    logger.info(f"  Group '{group}': n={len(y_true_group)}, "
                              f"AUROC={auroc:.4f if auroc else 'N/A'}, "
                              f"TPR={tpr:.4f}, FPR={fpr:.4f}")
                
                fairness_results['groups'][feature] = group_metrics
                
                # Calculate disparities
                if len(group_metrics) >= 2:
                    disparities = self._calculate_disparities(group_metrics)
                    fairness_results['disparities'][feature] = disparities
                    
                    # Check for violations
                    for metric, disparity in disparities.items():
                        if disparity > disparity_threshold:
                            flag = {
                                'feature': feature,
                                'metric': metric,
                                'disparity': disparity,
                                'threshold': disparity_threshold
                            }
                            fairness_results['flags'].append(flag)
                            logger.warning(
                                f"  FAIRNESS VIOLATION: {metric} disparity "
                                f"({disparity:.2%}) exceeds threshold ({disparity_threshold:.2%})"
                            )
            
            # Summary
            if fairness_results['flags']:
                logger.warning(
                    f"\n{model_name} - FAIRNESS AUDIT FAILED: "
                    f"{len(fairness_results['flags'])} violations detected"
                )
            else:
                logger.info(f"\n{model_name} - FAIRNESS AUDIT PASSED: No violations detected")
            
            # Save fairness report
            self._save_fairness_report(fairness_results, model_name)
            
            return fairness_results
            
        except Exception as e:
            logger.error(f"Error in fairness audit for {model_name}: {e}")
            return {'error': str(e)}
    
    def comprehensive_evaluation(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_features: Optional[pd.DataFrame] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with all metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            sensitive_features: DataFrame with demographic features (optional)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Comprehensive Evaluation: {model_name}")
        logger.info(f"{'='*60}\n")
        
        results = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'prevalence': float(np.mean(y_true))
        }
        
        # AUROC
        logger.info("Calculating AUROC...")
        results['auroc_metrics'] = self.calculate_auroc(y_true, y_pred_proba, model_name)
        
        # PR-AUC
        logger.info("\nCalculating PR-AUC...")
        results['pr_auc_metrics'] = self.calculate_pr_auc(y_true, y_pred_proba, model_name)
        
        # Calibration
        logger.info("\nGenerating calibration curve...")
        results['calibration_metrics'] = self.generate_calibration_curve(
            y_true, y_pred_proba, model_name
        )
        
        # Decision curve analysis
        logger.info("\nPerforming decision curve analysis...")
        results['decision_curve_metrics'] = self.decision_curve_analysis(
            y_true, y_pred_proba, model_name
        )
        
        # Fairness audit (if sensitive features provided)
        if sensitive_features is not None:
            logger.info("\nPerforming fairness audit...")
            results['fairness_metrics'] = self.fairness_audit(
                y_true, y_pred_proba, sensitive_features, model_name
            )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Complete: {model_name}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_disparities(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate disparities between groups."""
        disparities = {}
        
        metrics_to_check = ['auroc', 'pr_auc', 'tpr', 'fpr', 'ppv']
        
        for metric in metrics_to_check:
            values = [
                m[metric] for m in group_metrics.values()
                if m[metric] is not None
            ]
            
            if len(values) >= 2:
                disparity = max(values) - min(values)
                disparities[metric] = float(disparity)
        
        return disparities
    
    def _plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auroc: float,
        model_name: str
    ):
        """Plot and save ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{model_name}_roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    
    def _plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        pr_auc: float,
        model_name: str
    ):
        """Plot and save Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{model_name}_pr_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {output_path}")
    
    def _plot_calibration_curve(
        self,
        prob_true: np.ndarray,
        prob_pred: np.ndarray,
        model_name: str
    ):
        """Plot and save calibration curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Frequency')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{model_name}_calibration_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curve saved to {output_path}")
    
    def _plot_decision_curve(
        self,
        thresholds: np.ndarray,
        net_benefits: np.ndarray,
        treat_all_benefits: np.ndarray,
        treat_none_benefits: np.ndarray,
        model_name: str
    ):
        """Plot and save decision curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, net_benefits, label='Model', linewidth=2)
        plt.plot(thresholds, treat_all_benefits, label='Treat All', linestyle='--')
        plt.plot(thresholds, treat_none_benefits, label='Treat None', linestyle='--')
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title(f'Decision Curve Analysis - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        
        output_path = self.output_dir / f"{model_name}_decision_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Decision curve saved to {output_path}")
    
    def _save_fairness_report(
        self,
        fairness_results: Dict[str, Any],
        model_name: str
    ):
        """Save fairness audit report to JSON."""
        import json
        
        output_path = self.output_dir / f"{model_name}_fairness_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(fairness_results, f, indent=2)
        
        logger.info(f"Fairness report saved to {output_path}")
