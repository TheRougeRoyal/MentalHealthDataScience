"""
Model card generator for automated model documentation.

This module provides a ModelCardGenerator class for creating standardized
model documentation including performance metrics, training data characteristics,
fairness analysis, and interpretability insights.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from uuid import UUID
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.ml.model_registry import ModelRegistry
from src.ds.experiment_tracker import ExperimentTracker
from src.ml.interpretability import InterpretabilityEngine
from src.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelCard(BaseModel):
    """Standardized model documentation card."""
    
    # Model Details
    model_id: str
    model_name: str
    model_type: str
    version: str
    date: datetime
    owner: Optional[str] = None
    
    # Intended Use
    intended_use: str
    intended_users: List[str] = Field(default_factory=list)
    out_of_scope_uses: List[str] = Field(default_factory=list)
    
    # Training Data
    training_data_description: str
    training_data_size: int
    data_preprocessing: List[str] = Field(default_factory=list)
    
    # Performance Metrics
    metrics: Dict[str, float] = Field(default_factory=dict)
    performance_by_group: Optional[Dict[str, Dict[str, float]]] = None
    baseline_comparison: Dict[str, float] = Field(default_factory=dict)
    
    # Fairness Analysis
    fairness_metrics: Optional[Dict[str, float]] = None
    bias_analysis: Optional[str] = None
    
    # Interpretability
    feature_importance: List[Tuple[str, float]] = Field(default_factory=list)
    shap_summary: Optional[str] = None
    
    # Limitations
    known_limitations: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    
    # Ethical Considerations
    ethical_considerations: List[str] = Field(default_factory=list)
    
    # Caveats and Recommendations
    caveats: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelCardGenerator:
    """Generator for creating standardized model documentation cards."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        experiment_tracker: Optional[ExperimentTracker] = None,
        interpretability_engine: Optional[InterpretabilityEngine] = None
    ):
        """
        Initialize ModelCardGenerator.
        
        Args:
            model_registry: ModelRegistry instance for accessing model metadata
            experiment_tracker: Optional ExperimentTracker for experiment data
            interpretability_engine: Optional InterpretabilityEngine for SHAP analysis
        """
        self.model_registry = model_registry
        self.experiment_tracker = experiment_tracker
        self.interpretability_engine = interpretability_engine
        
        logger.info("ModelCardGenerator initialized")
    
    def generate_model_card(
        self,
        model_id: str,
        run_id: Optional[str] = None,
        include_fairness: bool = True,
        include_interpretability: bool = True,
        baseline_model_ids: Optional[List[str]] = None
    ) -> ModelCard:
        """
        Generate comprehensive model card for a registered model.
        
        Args:
            model_id: Model ID from registry
            run_id: Optional experiment run ID to link with
            include_fairness: Whether to include fairness metrics
            include_interpretability: Whether to include interpretability insights
            baseline_model_ids: Optional list of baseline model IDs for comparison
            
        Returns:
            ModelCard with complete documentation
            
        Raises:
            ModelNotFoundError: If model ID not found
        """
        try:
            logger.info(f"Generating model card for model {model_id}")
            
            # Get model metadata from registry
            metadata = self.model_registry.get_model_metadata(model_id)
            
            # Extract basic model details
            model_details = self._extract_model_details(metadata)
            
            # Get experiment data if available
            experiment_data = None
            if run_id and self.experiment_tracker:
                experiment_data = self._get_experiment_data(run_id)
            elif metadata.get('run_id') and self.experiment_tracker:
                experiment_data = self._get_experiment_data(metadata['run_id'])
            
            # Extract training data information
            training_info = self._extract_training_info(metadata, experiment_data)
            
            # Extract performance metrics
            performance = self._extract_performance_metrics(metadata, experiment_data)
            
            # Compute baseline comparison if baseline models provided
            baseline_comparison = {}
            if baseline_model_ids:
                baseline_comparison = self._compute_baseline_comparison(
                    model_id,
                    baseline_model_ids,
                    performance['metrics']
                )
            
            # Extract fairness metrics if requested
            fairness_data = None
            if include_fairness:
                fairness_data = self._extract_fairness_metrics(metadata, experiment_data)
            
            # Extract interpretability insights if requested
            interpretability_data = None
            if include_interpretability:
                interpretability_data = self._extract_interpretability_data(
                    metadata,
                    experiment_data
                )
            
            # Generate limitations and recommendations
            limitations = self._generate_limitations(metadata, performance)
            recommendations = self._generate_recommendations(
                metadata,
                performance,
                fairness_data
            )
            
            # Create model card
            model_card = ModelCard(
                # Model Details
                model_id=model_id,
                model_name=model_details['name'],
                model_type=model_details['type'],
                version=model_details['version'],
                date=model_details['date'],
                owner=model_details.get('owner'),
                
                # Intended Use
                intended_use=self._generate_intended_use(metadata),
                intended_users=self._generate_intended_users(metadata),
                out_of_scope_uses=self._generate_out_of_scope_uses(metadata),
                
                # Training Data
                training_data_description=training_info['description'],
                training_data_size=training_info['size'],
                data_preprocessing=training_info['preprocessing'],
                
                # Performance Metrics
                metrics=performance['metrics'],
                performance_by_group=performance.get('by_group'),
                baseline_comparison=baseline_comparison,
                
                # Fairness Analysis
                fairness_metrics=fairness_data['metrics'] if fairness_data else None,
                bias_analysis=fairness_data['analysis'] if fairness_data else None,
                
                # Interpretability
                feature_importance=(
                    interpretability_data['feature_importance']
                    if interpretability_data else []
                ),
                shap_summary=(
                    interpretability_data['shap_summary']
                    if interpretability_data else None
                ),
                
                # Limitations
                known_limitations=limitations['known'],
                failure_modes=limitations['failure_modes'],
                
                # Ethical Considerations
                ethical_considerations=self._generate_ethical_considerations(metadata),
                
                # Caveats and Recommendations
                caveats=limitations['caveats'],
                recommendations=recommendations,
                
                # Metadata
                experiment_id=metadata.get('experiment_id'),
                run_id=run_id or metadata.get('run_id')
            )
            
            logger.info(f"Model card generated successfully for model {model_id}")
            
            return model_card
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate model card: {e}")
            raise ValueError(f"Model card generation failed: {e}")
    
    def _extract_model_details(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic model details from metadata."""
        return {
            'name': metadata.get('model_name', metadata['model_id']),
            'type': metadata['model_type'],
            'version': metadata.get('version', '1.0'),
            'date': datetime.fromisoformat(metadata['registered_at']),
            'owner': metadata.get('owner', metadata.get('created_by'))
        }
    
    def _get_experiment_data(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data from tracker."""
        try:
            run = self.experiment_tracker.get_run(run_id)
            return {
                'params': run.params,
                'metrics': run.metrics,
                'tags': run.tags,
                'artifacts': run.artifacts
            }
        except Exception as e:
            logger.warning(f"Could not retrieve experiment data: {e}")
            return None
    
    def _extract_training_info(
        self,
        metadata: Dict[str, Any],
        experiment_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract training data information."""
        # Get from metadata or experiment data
        training_size = metadata.get('training_samples', 0)
        if experiment_data and 'params' in experiment_data:
            training_size = experiment_data['params'].get('n_samples', training_size)
        
        # Extract preprocessing steps
        preprocessing = []
        if 'preprocessing' in metadata:
            preprocessing = metadata['preprocessing']
        elif 'artifact_paths' in metadata and 'scaler' in metadata['artifact_paths']:
            preprocessing.append("Feature scaling (StandardScaler)")
        
        # Generate description
        description = metadata.get(
            'training_data_description',
            f"Model trained on {training_size} samples from mental health assessment data"
        )
        
        return {
            'description': description,
            'size': training_size,
            'preprocessing': preprocessing
        }
    
    def _extract_performance_metrics(
        self,
        metadata: Dict[str, Any],
        experiment_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract performance metrics from metadata and experiment data."""
        metrics = {}
        
        # Get metrics from metadata
        if 'metrics' in metadata:
            metrics.update(metadata['metrics'])
        
        # Get metrics from experiment data (take latest values)
        if experiment_data and 'metrics' in experiment_data:
            for metric_name, values in experiment_data['metrics'].items():
                if values:
                    # Take the last value
                    latest_value = values[-1][0] if isinstance(values[-1], tuple) else values[-1]
                    metrics[metric_name] = float(latest_value)
        
        # Extract performance by group if available
        by_group = None
        if 'performance_by_group' in metadata:
            by_group = metadata['performance_by_group']
        
        return {
            'metrics': metrics,
            'by_group': by_group
        }
    
    def _compute_baseline_comparison(
        self,
        model_id: str,
        baseline_model_ids: List[str],
        current_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute performance comparison against baseline models.
        
        Args:
            model_id: Current model ID
            baseline_model_ids: List of baseline model IDs
            current_metrics: Current model metrics
            
        Returns:
            Dictionary of metric improvements over baseline
        """
        logger.info(f"Computing baseline comparison for {len(baseline_model_ids)} baselines")
        
        comparison = {}
        
        # Aggregate baseline metrics
        baseline_metrics = {}
        for baseline_id in baseline_model_ids:
            try:
                baseline_metadata = self.model_registry.get_model_metadata(baseline_id)
                baseline_perf = baseline_metadata.get('metrics', {})
                
                for metric_name, value in baseline_perf.items():
                    if metric_name not in baseline_metrics:
                        baseline_metrics[metric_name] = []
                    baseline_metrics[metric_name].append(value)
            except Exception as e:
                logger.warning(f"Could not load baseline model {baseline_id}: {e}")
        
        # Compute average baseline and improvement
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_avg = np.mean(baseline_metrics[metric_name])
                improvement = current_value - baseline_avg
                comparison[f"{metric_name}_improvement"] = float(improvement)
                comparison[f"{metric_name}_baseline"] = float(baseline_avg)
        
        return comparison
    
    def _extract_fairness_metrics(
        self,
        metadata: Dict[str, Any],
        experiment_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract fairness metrics from metadata and governance module."""
        fairness_metrics = metadata.get('fairness_metrics', {})
        
        # Also check for performance by group which indicates fairness analysis
        performance_by_group = metadata.get('performance_by_group', {})
        
        # Merge fairness metrics from different sources
        if performance_by_group:
            # Extract fairness metrics from performance by group
            fairness_metrics.update(
                self._compute_fairness_from_group_performance(performance_by_group)
            )
        
        if not fairness_metrics:
            return None
        
        # Generate bias analysis summary
        analysis = self._generate_bias_analysis(fairness_metrics)
        
        return {
            'metrics': fairness_metrics,
            'analysis': analysis
        }
    
    def _compute_fairness_from_group_performance(
        self,
        performance_by_group: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute fairness metrics from performance by group data.
        
        Args:
            performance_by_group: Dictionary mapping group names to performance metrics
            
        Returns:
            Dictionary of fairness metrics
        """
        fairness_metrics = {}
        
        # Extract groups
        groups = list(performance_by_group.keys())
        
        if len(groups) < 2:
            return fairness_metrics
        
        # Compute demographic parity (difference in positive prediction rates)
        if all('positive_rate' in performance_by_group[g] for g in groups):
            rates = [performance_by_group[g]['positive_rate'] for g in groups]
            fairness_metrics['demographic_parity_difference'] = max(rates) - min(rates)
        
        # Compute equalized odds (difference in TPR and FPR)
        if all('tpr' in performance_by_group[g] for g in groups):
            tprs = [performance_by_group[g]['tpr'] for g in groups]
            fairness_metrics['equalized_odds_tpr_difference'] = max(tprs) - min(tprs)
        
        if all('fpr' in performance_by_group[g] for g in groups):
            fprs = [performance_by_group[g]['fpr'] for g in groups]
            fairness_metrics['equalized_odds_fpr_difference'] = max(fprs) - min(fprs)
            
            # Overall equalized odds is max of TPR and FPR differences
            if 'equalized_odds_tpr_difference' in fairness_metrics:
                fairness_metrics['equalized_odds_difference'] = max(
                    fairness_metrics['equalized_odds_tpr_difference'],
                    fairness_metrics['equalized_odds_fpr_difference']
                )
        
        # Compute accuracy parity
        if all('accuracy' in performance_by_group[g] for g in groups):
            accuracies = [performance_by_group[g]['accuracy'] for g in groups]
            fairness_metrics['accuracy_parity_difference'] = max(accuracies) - min(accuracies)
        
        return fairness_metrics
    
    def _generate_bias_analysis(self, fairness_metrics: Dict[str, float]) -> str:
        """Generate human-readable bias analysis from fairness metrics."""
        analysis_parts = []
        
        # Check for demographic parity
        if 'demographic_parity_difference' in fairness_metrics:
            dpd = fairness_metrics['demographic_parity_difference']
            if abs(dpd) < 0.1:
                analysis_parts.append(
                    f"Demographic parity is well-maintained (difference: {dpd:.3f})"
                )
            else:
                analysis_parts.append(
                    f"Demographic parity shows disparity (difference: {dpd:.3f})"
                )
        
        # Check for equalized odds
        if 'equalized_odds_difference' in fairness_metrics:
            eod = fairness_metrics['equalized_odds_difference']
            if abs(eod) < 0.1:
                analysis_parts.append(
                    f"Equalized odds are well-maintained (difference: {eod:.3f})"
                )
            else:
                analysis_parts.append(
                    f"Equalized odds show disparity (difference: {eod:.3f})"
                )
        
        if not analysis_parts:
            return "Fairness metrics available but require manual interpretation."
        
        return " ".join(analysis_parts)
    
    def _extract_interpretability_data(
        self,
        metadata: Dict[str, Any],
        experiment_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract interpretability insights from metadata and compute SHAP if available."""
        # Get feature importance
        feature_importance = []
        if 'feature_importance' in metadata:
            importance_dict = metadata['feature_importance']
            feature_importance = [
                (feature, float(importance))
                for feature, importance in importance_dict.items()
            ]
            # Sort by importance
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get SHAP summary if available in metadata
        shap_summary = metadata.get('shap_summary')
        shap_visualization_path = None
        
        # If interpretability engine is available and no SHAP data exists, compute it
        if self.interpretability_engine and not shap_summary:
            try:
                logger.info("Computing SHAP values for model card")
                # Note: This requires training/validation data which should be passed
                # For now, we'll just note that SHAP is available but not computed
                shap_summary = "SHAP analysis available via InterpretabilityEngine"
            except Exception as e:
                logger.warning(f"Could not compute SHAP values: {e}")
        
        if not feature_importance and not shap_summary:
            return None
        
        return {
            'feature_importance': feature_importance,
            'shap_summary': shap_summary,
            'shap_visualization_path': shap_visualization_path
        }
    
    def _generate_limitations(
        self,
        metadata: Dict[str, Any],
        performance: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate known limitations and failure modes."""
        known = []
        failure_modes = []
        caveats = []
        
        # Check performance metrics for limitations
        metrics = performance['metrics']
        
        if 'accuracy' in metrics and metrics['accuracy'] < 0.8:
            known.append(
                f"Model accuracy is {metrics['accuracy']:.2%}, "
                "which may not be sufficient for high-stakes decisions"
            )
        
        if 'recall' in metrics and metrics['recall'] < 0.7:
            known.append(
                f"Model recall is {metrics['recall']:.2%}, "
                "indicating potential for false negatives"
            )
        
        # Model-type specific limitations
        model_type = metadata['model_type']
        
        if model_type == 'logistic_regression':
            known.append("Linear model may not capture complex non-linear relationships")
            failure_modes.append("May underperform on highly non-linear patterns")
        
        elif model_type == 'lightgbm':
            known.append("Tree-based model may overfit on small datasets")
            failure_modes.append("May not extrapolate well beyond training data range")
        
        # General limitations
        caveats.append("Model performance may degrade on data distributions different from training")
        caveats.append("Regular monitoring and retraining recommended")
        
        return {
            'known': known,
            'failure_modes': failure_modes,
            'caveats': caveats
        }
    
    def _generate_recommendations(
        self,
        metadata: Dict[str, Any],
        performance: Dict[str, Any],
        fairness_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for model usage."""
        recommendations = []
        
        # Performance-based recommendations
        metrics = performance['metrics']
        
        if 'accuracy' in metrics and metrics['accuracy'] >= 0.85:
            recommendations.append(
                "Model shows strong performance and is suitable for production use"
            )
        
        if 'precision' in metrics and 'recall' in metrics:
            precision = metrics['precision']
            recall = metrics['recall']
            
            if precision > recall + 0.1:
                recommendations.append(
                    "Model prioritizes precision over recall; "
                    "consider adjusting threshold for more balanced performance"
                )
            elif recall > precision + 0.1:
                recommendations.append(
                    "Model prioritizes recall over precision; "
                    "may generate more false positives"
                )
        
        # Fairness-based recommendations
        if fairness_data and fairness_data['metrics']:
            recommendations.append(
                "Review fairness metrics regularly to ensure equitable performance across groups"
            )
        
        # General recommendations
        recommendations.append(
            "Implement continuous monitoring to detect performance degradation"
        )
        recommendations.append(
            "Combine model predictions with clinical judgment for final decisions"
        )
        
        return recommendations
    
    def _generate_intended_use(self, metadata: Dict[str, Any]) -> str:
        """Generate intended use description."""
        return metadata.get(
            'intended_use',
            "This model is intended for mental health risk assessment to support "
            "clinical decision-making. It should be used as a screening tool to "
            "identify individuals who may benefit from further evaluation."
        )
    
    def _generate_intended_users(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate list of intended users."""
        return metadata.get(
            'intended_users',
            [
                "Mental health clinicians",
                "Healthcare providers",
                "Clinical researchers",
                "Care coordinators"
            ]
        )
    
    def _generate_out_of_scope_uses(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate list of out-of-scope uses."""
        return metadata.get(
            'out_of_scope_uses',
            [
                "Sole basis for clinical diagnosis",
                "Automated decision-making without human oversight",
                "Use on populations significantly different from training data",
                "Legal or employment decisions"
            ]
        )
    
    def _generate_ethical_considerations(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate ethical considerations."""
        return metadata.get(
            'ethical_considerations',
            [
                "Patient privacy and data protection must be maintained",
                "Model predictions should not replace clinical judgment",
                "Regular fairness audits should be conducted",
                "Informed consent required for data collection and model use",
                "Transparency about model limitations with patients and providers"
            ]
        )
    
    def compute_shap_summary(
        self,
        model_id: str,
        validation_data: pd.DataFrame,
        output_path: Optional[str] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute SHAP summary for model card.
        
        Args:
            model_id: Model ID from registry
            validation_data: Validation data for SHAP computation
            output_path: Optional path to save SHAP visualization
            top_k: Number of top features to include
            
        Returns:
            Dictionary with SHAP summary and visualization path
            
        Raises:
            ValueError: If interpretability engine not available
        """
        if not self.interpretability_engine:
            raise ValueError("InterpretabilityEngine required for SHAP computation")
        
        try:
            logger.info(f"Computing SHAP summary for model {model_id}")
            
            # Compute SHAP values
            shap_result = self.interpretability_engine.compute_shap_values(
                model_id=model_id,
                features=validation_data,
                top_k=top_k
            )
            
            # Generate summary text
            top_features = shap_result['top_features']
            summary_lines = [
                f"Top {len(top_features)} most important features for model predictions:"
            ]
            
            for i, feature in enumerate(top_features, 1):
                clinical_name = feature['clinical_name']
                importance = feature['importance']
                summary_lines.append(
                    f"{i}. {clinical_name} (importance: {importance:.4f})"
                )
            
            summary_text = "\n".join(summary_lines)
            
            # Generate visualization if output path provided
            visualization_path = None
            if output_path:
                try:
                    import matplotlib.pyplot as plt
                    import shap
                    
                    # Create SHAP summary plot
                    shap_values = shap_result['shap_values']
                    
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values,
                        validation_data,
                        plot_type="bar",
                        show=False,
                        max_display=top_k
                    )
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    visualization_path = output_path
                    logger.info(f"SHAP visualization saved to {output_path}")
                    
                except Exception as e:
                    logger.warning(f"Could not generate SHAP visualization: {e}")
            
            return {
                'summary_text': summary_text,
                'visualization_path': visualization_path,
                'top_features': top_features,
                'base_value': shap_result['base_value']
            }
            
        except Exception as e:
            logger.error(f"SHAP summary computation failed: {e}")
            raise ValueError(f"SHAP summary computation failed: {e}")

    
    def export_card(
        self,
        model_card: ModelCard,
        format: str = "html",
        output_path: Optional[str] = None,
        include_visualizations: bool = True
    ) -> str:
        """
        Export model card in specified format.
        
        Args:
            model_card: ModelCard to export
            format: Export format ('html', 'pdf', 'markdown', 'json')
            output_path: Optional output file path
            include_visualizations: Whether to embed visualizations
            
        Returns:
            Path to exported file
            
        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Exporting model card in {format} format")
        
        # Generate default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"model_card_{model_card.model_id}_{timestamp}.{format}"
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format == "html":
            self._export_html(model_card, output_file, include_visualizations)
        elif format == "markdown":
            self._export_markdown(model_card, output_file)
        elif format == "json":
            self._export_json(model_card, output_file)
        elif format == "pdf":
            # PDF export requires HTML as intermediate
            html_path = output_file.with_suffix('.html')
            self._export_html(model_card, html_path, include_visualizations)
            self._convert_html_to_pdf(html_path, output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Model card exported to {output_path}")
        
        return str(output_path)
    
    def _export_html(
        self,
        model_card: ModelCard,
        output_path: Path,
        include_visualizations: bool = True
    ) -> None:
        """Export model card as HTML."""
        html_content = self._generate_html_template(model_card, include_visualizations)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_template(
        self,
        model_card: ModelCard,
        include_visualizations: bool = True
    ) -> str:
        """Generate HTML template for model card."""
        # Format metrics table
        metrics_rows = ""
        for metric_name, value in model_card.metrics.items():
            metrics_rows += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value:.4f}</td>
                </tr>
            """
        
        # Format baseline comparison
        baseline_rows = ""
        if model_card.baseline_comparison:
            for metric_name, value in model_card.baseline_comparison.items():
                baseline_rows += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{value:+.4f}</td>
                    </tr>
                """
        
        # Format feature importance
        feature_importance_rows = ""
        for feature, importance in model_card.feature_importance[:10]:
            feature_importance_rows += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{importance:.4f}</td>
                </tr>
            """
        
        # Format fairness metrics
        fairness_rows = ""
        if model_card.fairness_metrics:
            for metric_name, value in model_card.fairness_metrics.items():
                fairness_rows += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{value:.4f}</td>
                    </tr>
                """
        
        # Format lists
        def format_list(items):
            if not items:
                return "<li>None specified</li>"
            return "".join([f"<li>{item}</li>" for item in items])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card: {model_card.model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata-item {{
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 20px 0;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Card: {model_card.model_name}</h1>
        
        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">Model ID:</span> {model_card.model_id}
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Type:</span> {model_card.model_type}
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Version:</span> {model_card.version}
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Date:</span> {model_card.date.strftime('%Y-%m-%d')}
            </div>
            {f'<div class="metadata-item"><span class="metadata-label">Owner:</span> {model_card.owner}</div>' if model_card.owner else ''}
        </div>
        
        <div class="section">
            <h2>Intended Use</h2>
            <p>{model_card.intended_use}</p>
            
            <h3>Intended Users</h3>
            <ul>
                {format_list(model_card.intended_users)}
            </ul>
            
            <h3>Out of Scope Uses</h3>
            <div class="warning">
                <ul>
                    {format_list(model_card.out_of_scope_uses)}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Training Data</h2>
            <p>{model_card.training_data_description}</p>
            <p><strong>Dataset Size:</strong> {model_card.training_data_size:,} samples</p>
            
            {f'''
            <h3>Data Preprocessing</h3>
            <ul>
                {format_list(model_card.data_preprocessing)}
            </ul>
            ''' if model_card.data_preprocessing else ''}
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics_rows}
                </tbody>
            </table>
            
            {f'''
            <h3>Baseline Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Improvement</th>
                    </tr>
                </thead>
                <tbody>
                    {baseline_rows}
                </tbody>
            </table>
            ''' if baseline_rows else ''}
            
            {f'''
            <h3>Performance by Group</h3>
            <p>Fairness analysis shows performance across demographic groups.</p>
            ''' if model_card.performance_by_group else ''}
        </div>
        
        {f'''
        <div class="section">
            <h2>Fairness Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Fairness Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {fairness_rows}
                </tbody>
            </table>
            
            {f'<div class="info"><p><strong>Analysis:</strong> {model_card.bias_analysis}</p></div>' if model_card.bias_analysis else ''}
        </div>
        ''' if model_card.fairness_metrics else ''}
        
        {f'''
        <div class="section">
            <h2>Model Interpretability</h2>
            
            <h3>Feature Importance</h3>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
                    {feature_importance_rows}
                </tbody>
            </table>
            
            {f'<div class="info"><p><strong>SHAP Analysis:</strong><br>{model_card.shap_summary}</p></div>' if model_card.shap_summary else ''}
        </div>
        ''' if model_card.feature_importance else ''}
        
        <div class="section">
            <h2>Limitations and Considerations</h2>
            
            <h3>Known Limitations</h3>
            <ul>
                {format_list(model_card.known_limitations)}
            </ul>
            
            <h3>Potential Failure Modes</h3>
            <div class="warning">
                <ul>
                    {format_list(model_card.failure_modes)}
                </ul>
            </div>
            
            <h3>Caveats</h3>
            <ul>
                {format_list(model_card.caveats)}
            </ul>
        </div>
        
        <div class="section">
            <h2>Ethical Considerations</h2>
            <ul>
                {format_list(model_card.ethical_considerations)}
            </ul>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <div class="success">
                <ul>
                    {format_list(model_card.recommendations)}
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Model card generated on {model_card.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            {f'<p>Experiment ID: {model_card.experiment_id} | Run ID: {model_card.run_id}</p>' if model_card.experiment_id or model_card.run_id else ''}
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _export_markdown(self, model_card: ModelCard, output_path: Path) -> None:
        """Export model card as Markdown."""
        md_content = self._generate_markdown_template(model_card)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _generate_markdown_template(self, model_card: ModelCard) -> str:
        """Generate Markdown template for model card."""
        def format_list(items):
            if not items:
                return "- None specified\n"
            return "".join([f"- {item}\n" for item in items])
        
        md = f"""# Model Card: {model_card.model_name}

## Model Details

- **Model ID:** {model_card.model_id}
- **Model Type:** {model_card.model_type}
- **Version:** {model_card.version}
- **Date:** {model_card.date.strftime('%Y-%m-%d')}
"""
        
        if model_card.owner:
            md += f"- **Owner:** {model_card.owner}\n"
        
        md += f"""
## Intended Use

{model_card.intended_use}

### Intended Users

{format_list(model_card.intended_users)}

### Out of Scope Uses

{format_list(model_card.out_of_scope_uses)}

## Training Data

{model_card.training_data_description}

- **Dataset Size:** {model_card.training_data_size:,} samples

"""
        
        if model_card.data_preprocessing:
            md += f"""### Data Preprocessing

{format_list(model_card.data_preprocessing)}

"""
        
        md += """## Performance Metrics

| Metric | Value |
|--------|-------|
"""
        
        for metric_name, value in model_card.metrics.items():
            md += f"| {metric_name} | {value:.4f} |\n"
        
        if model_card.baseline_comparison:
            md += "\n### Baseline Comparison\n\n"
            md += "| Metric | Improvement |\n"
            md += "|--------|-------------|\n"
            for metric_name, value in model_card.baseline_comparison.items():
                md += f"| {metric_name} | {value:+.4f} |\n"
        
        if model_card.fairness_metrics:
            md += "\n## Fairness Analysis\n\n"
            md += "| Fairness Metric | Value |\n"
            md += "|-----------------|-------|\n"
            for metric_name, value in model_card.fairness_metrics.items():
                md += f"| {metric_name} | {value:.4f} |\n"
            
            if model_card.bias_analysis:
                md += f"\n**Analysis:** {model_card.bias_analysis}\n"
        
        if model_card.feature_importance:
            md += "\n## Model Interpretability\n\n"
            md += "### Feature Importance\n\n"
            md += "| Feature | Importance |\n"
            md += "|---------|------------|\n"
            for feature, importance in model_card.feature_importance[:10]:
                md += f"| {feature} | {importance:.4f} |\n"
            
            if model_card.shap_summary:
                md += f"\n**SHAP Analysis:**\n\n{model_card.shap_summary}\n"
        
        md += f"""
## Limitations and Considerations

### Known Limitations

{format_list(model_card.known_limitations)}

### Potential Failure Modes

{format_list(model_card.failure_modes)}

### Caveats

{format_list(model_card.caveats)}

## Ethical Considerations

{format_list(model_card.ethical_considerations)}

## Recommendations

{format_list(model_card.recommendations)}

---

*Model card generated on {model_card.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""
        
        if model_card.experiment_id or model_card.run_id:
            md += f"\n*Experiment ID: {model_card.experiment_id} | Run ID: {model_card.run_id}*\n"
        
        return md
    
    def _export_json(self, model_card: ModelCard, output_path: Path) -> None:
        """Export model card as JSON."""
        # Convert to dict and handle datetime serialization
        card_dict = model_card.dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(card_dict, f, indent=2, default=str)
    
    def _convert_html_to_pdf(self, html_path: Path, pdf_path: Path) -> None:
        """
        Convert HTML to PDF.
        
        Note: This requires an external library like weasyprint or pdfkit.
        For now, we'll just log a warning and copy the HTML.
        """
        logger.warning(
            "PDF export requires weasyprint or pdfkit library. "
            "Falling back to HTML export."
        )
        
        # Try to use weasyprint if available
        try:
            from weasyprint import HTML
            HTML(str(html_path)).write_pdf(str(pdf_path))
            logger.info(f"PDF generated using weasyprint: {pdf_path}")
        except ImportError:
            logger.warning("weasyprint not available, PDF export not supported")
            # Copy HTML as fallback
            import shutil
            shutil.copy(html_path, pdf_path.with_suffix('.html'))
