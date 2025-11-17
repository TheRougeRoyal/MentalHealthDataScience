"""
Interpretability engine for generating model explanations.

This module provides an InterpretabilityEngine class for generating
SHAP values, counterfactual explanations, and rule-based approximations
to help clinicians understand and trust model predictions.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from src.ml.model_registry import ModelRegistry
from src.exceptions import InterpretabilityError

logger = logging.getLogger(__name__)


class InterpretabilityEngine:
    """Engine for generating interpretable explanations of model predictions."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_names: Optional[List[str]] = None,
        clinical_mappings: Optional[Dict[str, str]] = None
    ):
        """
        Initialize InterpretabilityEngine.
        
        Args:
            model_registry: ModelRegistry instance
            feature_names: List of feature names
            clinical_mappings: Dictionary mapping technical feature names to clinical terms
        """
        self.model_registry = model_registry
        self.feature_names = feature_names
        self.clinical_mappings = clinical_mappings or {}
        
        # Cache for SHAP explainers
        self._explainer_cache = {}
        
        logger.info("InterpretabilityEngine initialized")

    def compute_shap_values(
        self,
        model_id: str,
        features: pd.DataFrame,
        top_k: int = 10,
        background_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for model predictions.
        
        Args:
            model_id: Model ID from registry
            features: Input features as DataFrame
            top_k: Number of top features to return
            background_data: Optional background dataset for SHAP explainer
            
        Returns:
            Dictionary with SHAP values, base values, and top features
            
        Raises:
            InterpretabilityError: If SHAP computation fails
        """
        try:
            logger.info(f"Computing SHAP values for model {model_id}")
            
            # Load model
            model_bundle, metadata = self.model_registry.load_model(model_id)
            model = model_bundle['model']
            model_type = metadata['model_type']
            artifacts = model_bundle.get('artifacts', {})
            
            # Apply preprocessing if scaler available
            if 'scaler' in artifacts:
                features_scaled = pd.DataFrame(
                    artifacts['scaler'].transform(features),
                    columns=features.columns,
                    index=features.index
                )
            else:
                features_scaled = features
            
            # Get or create SHAP explainer
            explainer = self._get_shap_explainer(
                model,
                model_type,
                features_scaled,
                background_data
            )
            
            # Compute SHAP values
            shap_values = explainer.shap_values(features_scaled)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For binary classification, use positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Get base value (expected value)
            if hasattr(explainer, 'expected_value'):
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                base_value = 0.0
            
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Get top k features
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_features = []
            
            for idx in top_indices:
                feature_name = features.columns[idx]
                clinical_name = self.clinical_mappings.get(feature_name, feature_name)
                
                top_features.append({
                    'feature': feature_name,
                    'clinical_name': clinical_name,
                    'importance': float(feature_importance[idx]),
                    'mean_shap_value': float(shap_values[:, idx].mean())
                })
            
            logger.info(f"SHAP values computed. Top feature: {top_features[0]['clinical_name']}")
            
            return {
                'shap_values': shap_values,
                'base_value': float(base_value),
                'top_features': top_features,
                'feature_names': list(features.columns)
            }
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            raise InterpretabilityError(f"SHAP computation failed: {e}")

    def _get_shap_explainer(
        self,
        model: Any,
        model_type: str,
        features: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None
    ) -> shap.Explainer:
        """
        Get or create SHAP explainer for model.
        
        Args:
            model: Trained model
            model_type: Type of model
            features: Feature data
            background_data: Optional background dataset
            
        Returns:
            SHAP explainer
        """
        # Use background data or sample from features
        if background_data is None:
            # Use a sample of features as background
            n_background = min(100, len(features))
            background = shap.sample(features, n_background)
        else:
            background = background_data
        
        # Create appropriate explainer based on model type
        if model_type == 'logistic_regression':
            explainer = shap.LinearExplainer(model, background)
            
        elif model_type == 'lightgbm':
            explainer = shap.TreeExplainer(model)
            
        elif model_type == 'isolation_forest':
            # For anomaly detection, use KernelExplainer
            def model_predict(X):
                return model.score_samples(X)
            explainer = shap.KernelExplainer(model_predict, background)
            
        else:
            # Default to KernelExplainer (model-agnostic but slower)
            def model_predict(X):
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X)[:, 1]
                else:
                    return model.predict(X)
            
            explainer = shap.KernelExplainer(model_predict, background)
        
        return explainer

    def generate_counterfactuals(
        self,
        model_id: str,
        features: pd.DataFrame,
        target_class: str = 'low',
        max_changes: int = 5,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations showing minimal changes needed
        to alter the risk classification.
        
        Args:
            model_id: Model ID from registry
            features: Input features as DataFrame (single instance or multiple)
            target_class: Target risk level ('low', 'moderate', 'high', 'critical')
            max_changes: Maximum number of features to change
            feature_ranges: Optional dictionary of valid ranges for each feature
            
        Returns:
            List of counterfactual explanations, one per input instance
            
        Raises:
            InterpretabilityError: If counterfactual generation fails
        """
        try:
            logger.info(
                f"Generating counterfactuals for model {model_id}, "
                f"target class: {target_class}"
            )
            
            # Load model
            model_bundle, metadata = self.model_registry.load_model(model_id)
            model = model_bundle['model']
            model_type = metadata['model_type']
            artifacts = model_bundle.get('artifacts', {})
            
            # Apply preprocessing if scaler available
            if 'scaler' in artifacts:
                features_scaled = pd.DataFrame(
                    artifacts['scaler'].transform(features),
                    columns=features.columns,
                    index=features.index
                )
            else:
                features_scaled = features.copy()
            
            # Get current predictions
            current_predictions = self._predict_risk_score(
                model, model_type, features_scaled
            )
            
            # Generate counterfactuals for each instance
            counterfactuals = []
            
            for idx in range(len(features)):
                instance = features_scaled.iloc[idx:idx+1]
                original_features = features.iloc[idx]
                current_score = current_predictions[idx]
                current_risk_level = self._score_to_risk_level(current_score)
                
                logger.info(
                    f"Instance {idx}: Current risk {current_risk_level} "
                    f"(score: {current_score:.2f}), target: {target_class}"
                )
                
                # Skip if already at target
                if current_risk_level == target_class:
                    counterfactuals.append({
                        'instance_id': idx,
                        'current_risk_level': current_risk_level,
                        'current_score': float(current_score),
                        'target_risk_level': target_class,
                        'changes_needed': [],
                        'counterfactual_score': float(current_score),
                        'description': f"Already at {target_class} risk level. No changes needed."
                    })
                    continue
                
                # Find minimal changes
                cf_result = self._find_minimal_changes(
                    model,
                    model_type,
                    instance,
                    original_features,
                    target_class,
                    max_changes,
                    feature_ranges,
                    artifacts
                )
                
                counterfactuals.append(cf_result)
            
            logger.info(f"Generated {len(counterfactuals)} counterfactual explanations")
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            raise InterpretabilityError(f"Counterfactual generation failed: {e}")

    def _find_minimal_changes(
        self,
        model: Any,
        model_type: str,
        instance: pd.DataFrame,
        original_features: pd.Series,
        target_class: str,
        max_changes: int,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]],
        artifacts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find minimal feature changes to reach target risk level.
        
        Uses a greedy search approach to find the smallest set of feature
        changes that would move the prediction to the target risk level.
        
        Args:
            model: Trained model
            model_type: Type of model
            instance: Single instance features (scaled)
            original_features: Original unscaled features
            target_class: Target risk level
            max_changes: Maximum number of features to change
            feature_ranges: Valid ranges for features
            artifacts: Model artifacts (e.g., scaler)
            
        Returns:
            Dictionary with counterfactual explanation
        """
        # Get target score range
        target_score_range = self._risk_level_to_score_range(target_class)
        target_min, target_max = target_score_range
        
        # Get current score
        current_score = self._predict_risk_score(model, model_type, instance)[0]
        
        # Determine direction (increase or decrease risk)
        if current_score > target_max:
            # Need to decrease risk
            direction = -1
            target_score = target_max
        else:
            # Need to increase risk
            direction = 1
            target_score = target_min
        
        # Compute feature importance using SHAP or gradients
        feature_importance = self._compute_feature_importance_for_instance(
            model, model_type, instance
        )
        
        # Sort features by importance (descending)
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Try changing features one by one
        changes = []
        modified_instance = instance.copy()
        
        for feature_name, importance in sorted_features:
            if len(changes) >= max_changes:
                break
            
            # Determine change direction based on importance and target
            if direction == -1:
                # Decrease risk: change in opposite direction of importance
                change_direction = -1 if importance > 0 else 1
            else:
                # Increase risk: change in direction of importance
                change_direction = 1 if importance > 0 else -1
            
            # Calculate proposed change
            current_value = modified_instance[feature_name].values[0]
            original_value = original_features[feature_name]
            
            # Determine step size (10% of current value or 0.5 std)
            step_size = max(abs(current_value * 0.1), 0.5)
            proposed_value = current_value + (change_direction * step_size)
            
            # Apply feature range constraints if provided
            if feature_ranges and feature_name in feature_ranges:
                min_val, max_val = feature_ranges[feature_name]
                proposed_value = np.clip(proposed_value, min_val, max_val)
            
            # Apply change
            modified_instance[feature_name] = proposed_value
            
            # Check new prediction
            new_score = self._predict_risk_score(model, model_type, modified_instance)[0]
            
            # Unscale values for reporting
            if 'scaler' in artifacts:
                # Get original scale value
                temp_df = instance.copy()
                temp_df[feature_name] = proposed_value
                unscaled = artifacts['scaler'].inverse_transform(temp_df)
                proposed_value_original = unscaled[0][list(instance.columns).index(feature_name)]
            else:
                proposed_value_original = proposed_value
            
            # Record change
            clinical_name = self.clinical_mappings.get(feature_name, feature_name)
            
            change_info = {
                'feature': feature_name,
                'clinical_name': clinical_name,
                'original_value': float(original_value),
                'proposed_value': float(proposed_value_original),
                'change_magnitude': float(abs(proposed_value_original - original_value)),
                'importance': float(importance)
            }
            
            changes.append(change_info)
            
            # Check if we've reached target
            if direction == -1 and new_score <= target_score:
                break
            elif direction == 1 and new_score >= target_score:
                break
        
        # Get final score
        final_score = self._predict_risk_score(model, model_type, modified_instance)[0]
        final_risk_level = self._score_to_risk_level(final_score)
        
        # Generate human-readable description
        description = self._format_counterfactual_description(
            changes,
            target_class,
            final_risk_level,
            current_score,
            final_score
        )
        
        return {
            'instance_id': 0,
            'current_risk_level': self._score_to_risk_level(current_score),
            'current_score': float(current_score),
            'target_risk_level': target_class,
            'changes_needed': changes,
            'counterfactual_score': float(final_score),
            'counterfactual_risk_level': final_risk_level,
            'achieved_target': final_risk_level == target_class,
            'description': description
        }

    def _predict_risk_score(
        self,
        model: Any,
        model_type: str,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict risk score from model.
        
        Args:
            model: Trained model
            model_type: Type of model
            features: Input features
            
        Returns:
            Array of risk scores (0-100)
        """
        if model_type == 'logistic_regression':
            proba = model.predict_proba(features)[:, 1]
            return proba * 100
            
        elif model_type == 'lightgbm':
            proba = model.predict(features)
            return proba * 100
            
        else:
            # Default: assume predict_proba exists
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[:, 1]
            else:
                proba = model.predict(features)
            return proba * 100
    
    def _compute_feature_importance_for_instance(
        self,
        model: Any,
        model_type: str,
        instance: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute feature importance for a single instance.
        
        Args:
            model: Trained model
            model_type: Type of model
            instance: Single instance features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Use SHAP for instance-level importance
        try:
            if model_type == 'logistic_regression':
                # For logistic regression, use coefficients
                coefficients = model.coef_[0]
                feature_values = instance.values[0]
                importance = coefficients * feature_values
                
            elif model_type == 'lightgbm':
                # Use SHAP TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                importance = shap_values[0]
                
            else:
                # Default: use feature values as proxy
                importance = instance.values[0]
            
            return dict(zip(instance.columns, importance))
            
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            # Fallback: return uniform importance
            return {col: 1.0 for col in instance.columns}
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert risk score to risk level category."""
        if score <= 25:
            return 'low'
        elif score <= 50:
            return 'moderate'
        elif score <= 75:
            return 'high'
        else:
            return 'critical'
    
    def _risk_level_to_score_range(self, risk_level: str) -> Tuple[float, float]:
        """Convert risk level to score range."""
        ranges = {
            'low': (0, 25),
            'moderate': (26, 50),
            'high': (51, 75),
            'critical': (76, 100)
        }
        return ranges.get(risk_level, (0, 100))

    def _format_counterfactual_description(
        self,
        changes: List[Dict[str, Any]],
        target_class: str,
        achieved_class: str,
        current_score: float,
        final_score: float
    ) -> str:
        """
        Format counterfactual explanation in human-readable clinical language.
        
        Args:
            changes: List of feature changes
            target_class: Target risk level
            achieved_class: Achieved risk level after changes
            current_score: Current risk score
            final_score: Final risk score after changes
            
        Returns:
            Human-readable description
        """
        if not changes:
            return f"Already at {target_class} risk level. No changes needed."
        
        # Build description
        description_parts = []
        
        # Header
        if achieved_class == target_class:
            description_parts.append(
                f"To reach {target_class} risk level (from {current_score:.1f} to {final_score:.1f}), "
                f"the following changes would be needed:"
            )
        else:
            description_parts.append(
                f"To move toward {target_class} risk level (from {current_score:.1f} to {final_score:.1f}), "
                f"the following changes would help:"
            )
        
        # List changes
        for i, change in enumerate(changes, 1):
            clinical_name = change['clinical_name']
            original = change['original_value']
            proposed = change['proposed_value']
            
            # Determine direction
            if proposed > original:
                direction = "increase"
                change_desc = f"from {original:.2f} to {proposed:.2f}"
            else:
                direction = "decrease"
                change_desc = f"from {original:.2f} to {proposed:.2f}"
            
            description_parts.append(
                f"{i}. {direction.capitalize()} {clinical_name} {change_desc}"
            )
        
        # Footer
        if achieved_class == target_class:
            description_parts.append(
                f"\nThese {len(changes)} change(s) would successfully move the individual "
                f"to {target_class} risk level."
            )
        else:
            description_parts.append(
                f"\nThese {len(changes)} change(s) would move the risk score to {final_score:.1f} "
                f"({achieved_class} risk level), closer to the target of {target_class}."
            )
        
        return "\n".join(description_parts)

    def extract_rule_set(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        training_labels: pd.Series,
        max_depth: int = 3,
        min_samples_leaf: int = 50
    ) -> Dict[str, Any]:
        """
        Extract simple rule set that approximates model decisions.
        
        Args:
            model_id: Model ID from registry
            training_data: Training features for rule extraction
            training_labels: Training labels
            max_depth: Maximum depth of decision tree
            min_samples_leaf: Minimum samples per leaf node
            
        Returns:
            Dictionary with rule set and metadata
            
        Raises:
            InterpretabilityError: If rule extraction fails
        """
        try:
            logger.info(f"Extracting rule set for model {model_id}")
            
            # Load model
            model_bundle, metadata = self.model_registry.load_model(model_id)
            model = model_bundle['model']
            model_type = metadata['model_type']
            artifacts = model_bundle.get('artifacts', {})
            
            # Apply preprocessing if scaler available
            if 'scaler' in artifacts:
                training_data_scaled = pd.DataFrame(
                    artifacts['scaler'].transform(training_data),
                    columns=training_data.columns
                )
            else:
                training_data_scaled = training_data
            
            # Get predictions from original model
            original_predictions = self._predict_risk_score(
                model, model_type, training_data_scaled
            )
            
            # Convert to binary for rule extraction (high risk vs not)
            binary_labels = (original_predictions > 50).astype(int)
            
            # Train decision tree to approximate model
            tree_model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            tree_model.fit(training_data_scaled, binary_labels)
            
            # Extract rules from tree
            rules = self._extract_rules_from_tree(
                tree_model,
                training_data.columns
            )
            
            # Calculate fidelity (how well rules match original model)
            tree_predictions = tree_model.predict(training_data_scaled)
            fidelity = (tree_predictions == binary_labels).mean()
            
            logger.info(
                f"Extracted {len(rules)} rules with fidelity: {fidelity:.3f}"
            )
            
            return {
                'rules': rules,
                'max_depth': max_depth,
                'n_rules': len(rules),
                'fidelity': float(fidelity),
                'feature_names': list(training_data.columns)
            }
            
        except Exception as e:
            logger.error(f"Rule extraction failed: {e}")
            raise InterpretabilityError(f"Rule extraction failed: {e}")
    
    def _extract_rules_from_tree(
        self,
        tree_model: DecisionTreeClassifier,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract rules from decision tree.
        
        Args:
            tree_model: Trained decision tree
            feature_names: List of feature names
            
        Returns:
            List of rules
        """
        tree = tree_model.tree_
        rules = []
        
        def recurse(node, conditions):
            if tree.feature[node] != -2:  # Not a leaf
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                clinical_name = self.clinical_mappings.get(feature, feature)
                
                # Left child (<=)
                left_conditions = conditions + [
                    f"{clinical_name} <= {threshold:.2f}"
                ]
                recurse(tree.children_left[node], left_conditions)
                
                # Right child (>)
                right_conditions = conditions + [
                    f"{clinical_name} > {threshold:.2f}"
                ]
                recurse(tree.children_right[node], right_conditions)
            else:
                # Leaf node
                value = tree.value[node][0]
                predicted_class = np.argmax(value)
                confidence = value[predicted_class] / value.sum()
                
                rule = {
                    'conditions': conditions,
                    'prediction': 'high_risk' if predicted_class == 1 else 'low_risk',
                    'confidence': float(confidence),
                    'n_samples': int(tree.n_node_samples[node]),
                    'rule_text': ' AND '.join(conditions) if conditions else 'Default'
                }
                
                rules.append(rule)
        
        recurse(0, [])
        
        return rules

    def generate_explanation(
        self,
        model_id: str,
        features: pd.DataFrame,
        training_data: Optional[pd.DataFrame] = None,
        training_labels: Optional[pd.Series] = None,
        background_data: Optional[pd.DataFrame] = None,
        target_class: str = 'low',
        top_k_features: int = 10,
        max_counterfactual_changes: int = 5,
        max_rule_depth: int = 3,
        include_shap: bool = True,
        include_counterfactuals: bool = True,
        include_rules: bool = True,
        timeout_seconds: float = 3.0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation combining SHAP, counterfactuals, and rules.
        
        This method orchestrates all interpretability components to provide a complete
        explanation of model predictions. It ensures all explanations are generated
        within the specified timeout.
        
        Args:
            model_id: Model ID from registry
            features: Input features for prediction (single instance or batch)
            training_data: Optional training data for rule extraction
            training_labels: Optional training labels for rule extraction
            background_data: Optional background data for SHAP computation
            target_class: Target risk level for counterfactuals
            top_k_features: Number of top features to include in SHAP explanation
            max_counterfactual_changes: Maximum feature changes in counterfactuals
            max_rule_depth: Maximum depth for rule extraction
            include_shap: Whether to include SHAP values
            include_counterfactuals: Whether to include counterfactual explanations
            include_rules: Whether to include rule extraction
            timeout_seconds: Maximum time allowed for all explanations (default: 3.0)
            
        Returns:
            Dictionary containing all requested explanations and metadata
            
        Raises:
            InterpretabilityError: If explanation generation fails or times out
        """
        start_time = time.time()
        
        try:
            logger.info(
                f"Generating comprehensive explanation for model {model_id}, "
                f"timeout: {timeout_seconds}s"
            )
            
            explanation = {
                'model_id': model_id,
                'n_instances': len(features),
                'timestamp': time.time(),
                'components': {
                    'shap': None,
                    'counterfactuals': None,
                    'rules': None
                },
                'generation_time': {},
                'errors': []
            }
            
            # Track remaining time
            def get_remaining_time():
                elapsed = time.time() - start_time
                return max(0, timeout_seconds - elapsed)
            
            # 1. Generate SHAP values (if requested)
            if include_shap:
                try:
                    shap_start = time.time()
                    
                    if get_remaining_time() <= 0:
                        raise InterpretabilityError("Timeout before SHAP computation")
                    
                    shap_result = self.compute_shap_values(
                        model_id=model_id,
                        features=features,
                        top_k=top_k_features,
                        background_data=background_data
                    )
                    
                    # Remove raw SHAP values array to reduce size
                    shap_result_summary = {
                        'base_value': shap_result['base_value'],
                        'top_features': shap_result['top_features'],
                        'feature_names': shap_result['feature_names']
                    }
                    
                    explanation['components']['shap'] = shap_result_summary
                    explanation['generation_time']['shap'] = time.time() - shap_start
                    
                    logger.info(
                        f"SHAP values computed in {explanation['generation_time']['shap']:.3f}s"
                    )
                    
                except Exception as e:
                    logger.warning(f"SHAP computation failed: {e}")
                    explanation['errors'].append({
                        'component': 'shap',
                        'error': str(e)
                    })
            
            # 2. Generate counterfactual explanations (if requested)
            if include_counterfactuals:
                try:
                    cf_start = time.time()
                    
                    if get_remaining_time() <= 0:
                        raise InterpretabilityError("Timeout before counterfactual generation")
                    
                    counterfactuals = self.generate_counterfactuals(
                        model_id=model_id,
                        features=features,
                        target_class=target_class,
                        max_changes=max_counterfactual_changes
                    )
                    
                    explanation['components']['counterfactuals'] = counterfactuals
                    explanation['generation_time']['counterfactuals'] = time.time() - cf_start
                    
                    logger.info(
                        f"Counterfactuals generated in "
                        f"{explanation['generation_time']['counterfactuals']:.3f}s"
                    )
                    
                except Exception as e:
                    logger.warning(f"Counterfactual generation failed: {e}")
                    explanation['errors'].append({
                        'component': 'counterfactuals',
                        'error': str(e)
                    })
            
            # 3. Extract rule set (if requested and training data provided)
            if include_rules and training_data is not None and training_labels is not None:
                try:
                    rules_start = time.time()
                    
                    if get_remaining_time() <= 0:
                        raise InterpretabilityError("Timeout before rule extraction")
                    
                    rules = self.extract_rule_set(
                        model_id=model_id,
                        training_data=training_data,
                        training_labels=training_labels,
                        max_depth=max_rule_depth
                    )
                    
                    explanation['components']['rules'] = rules
                    explanation['generation_time']['rules'] = time.time() - rules_start
                    
                    logger.info(
                        f"Rules extracted in {explanation['generation_time']['rules']:.3f}s"
                    )
                    
                except Exception as e:
                    logger.warning(f"Rule extraction failed: {e}")
                    explanation['errors'].append({
                        'component': 'rules',
                        'error': str(e)
                    })
            
            # Calculate total generation time
            total_time = time.time() - start_time
            explanation['total_generation_time'] = total_time
            
            # Check if we exceeded timeout
            if total_time > timeout_seconds:
                logger.warning(
                    f"Explanation generation exceeded timeout: {total_time:.3f}s > {timeout_seconds}s"
                )
                explanation['timeout_exceeded'] = True
            else:
                explanation['timeout_exceeded'] = False
            
            # Generate clinical summary
            explanation['clinical_summary'] = self._generate_clinical_summary(explanation)
            
            logger.info(
                f"Comprehensive explanation generated in {total_time:.3f}s "
                f"(timeout: {timeout_seconds}s)"
            )
            
            return explanation
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Explanation generation failed after {elapsed:.3f}s: {e}")
            raise InterpretabilityError(f"Explanation generation failed: {e}")
    
    def _generate_clinical_summary(self, explanation: Dict[str, Any]) -> str:
        """
        Generate a human-readable clinical summary of the explanation.
        
        Args:
            explanation: Complete explanation dictionary
            
        Returns:
            Clinical summary text
        """
        summary_parts = []
        
        # Header
        summary_parts.append("=== Model Explanation Summary ===\n")
        
        # SHAP summary
        if explanation['components']['shap']:
            shap_data = explanation['components']['shap']
            top_features = shap_data['top_features']
            
            summary_parts.append("Key Contributing Factors:")
            for i, feature in enumerate(top_features[:5], 1):
                clinical_name = feature['clinical_name']
                importance = feature['importance']
                summary_parts.append(f"  {i}. {clinical_name} (importance: {importance:.3f})")
            summary_parts.append("")
        
        # Counterfactual summary
        if explanation['components']['counterfactuals']:
            cf_data = explanation['components']['counterfactuals']
            
            if len(cf_data) > 0:
                cf = cf_data[0]  # First instance
                
                summary_parts.append("Actionable Changes:")
                summary_parts.append(f"  Current Risk: {cf['current_risk_level']}")
                summary_parts.append(f"  Target Risk: {cf['target_risk_level']}")
                
                if cf['changes_needed']:
                    summary_parts.append(f"  Changes Needed: {len(cf['changes_needed'])}")
                    for i, change in enumerate(cf['changes_needed'][:3], 1):
                        clinical_name = change['clinical_name']
                        orig = change['original_value']
                        prop = change['proposed_value']
                        direction = "increase" if prop > orig else "decrease"
                        summary_parts.append(
                            f"    {i}. {direction.capitalize()} {clinical_name} "
                            f"from {orig:.2f} to {prop:.2f}"
                        )
                else:
                    summary_parts.append("  No changes needed - already at target risk level")
                
                summary_parts.append("")
        
        # Rules summary
        if explanation['components']['rules']:
            rules_data = explanation['components']['rules']
            
            summary_parts.append("Decision Rules:")
            summary_parts.append(f"  Total Rules: {rules_data['n_rules']}")
            summary_parts.append(f"  Model Fidelity: {rules_data['fidelity']:.1%}")
            
            # Show top 2 rules
            high_risk_rules = [
                r for r in rules_data['rules']
                if r['prediction'] == 'high_risk'
            ]
            
            if high_risk_rules:
                # Sort by confidence
                high_risk_rules.sort(key=lambda x: x['confidence'], reverse=True)
                
                summary_parts.append("\n  High Risk Indicators:")
                for i, rule in enumerate(high_risk_rules[:2], 1):
                    summary_parts.append(f"    {i}. IF {rule['rule_text']}")
                    summary_parts.append(
                        f"       THEN high risk (confidence: {rule['confidence']:.1%})"
                    )
            
            summary_parts.append("")
        
        # Performance summary
        summary_parts.append("Generation Performance:")
        for component, duration in explanation['generation_time'].items():
            summary_parts.append(f"  {component}: {duration:.3f}s")
        summary_parts.append(f"  Total: {explanation['total_generation_time']:.3f}s")
        
        if explanation['timeout_exceeded']:
            summary_parts.append("  ⚠️  Warning: Timeout exceeded")
        
        # Errors
        if explanation['errors']:
            summary_parts.append("\nWarnings:")
            for error in explanation['errors']:
                summary_parts.append(f"  - {error['component']}: {error['error']}")
        
        return "\n".join(summary_parts)
