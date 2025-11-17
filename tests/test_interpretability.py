"""
Tests for interpretability engine.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from sklearn.linear_model import LogisticRegression

from src.ml.interpretability import InterpretabilityEngine
from src.ml.model_registry import ModelRegistry
from src.exceptions import InterpretabilityError


@pytest.fixture
def mock_model_registry():
    """Create mock model registry."""
    registry = Mock(spec=ModelRegistry)
    return registry


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    return pd.DataFrame({
        'sleep_duration': [6.5, 7.2, 5.8],
        'hrv_rmssd': [45.0, 52.0, 38.0],
        'activity_level': [3.2, 4.1, 2.5],
        'sentiment_score': [0.6, 0.7, 0.4]
    })


@pytest.fixture
def mock_logistic_model():
    """Create mock logistic regression model."""
    model = Mock(spec=LogisticRegression)
    model.coef_ = np.array([[0.5, -0.3, 0.2, 0.4]])
    model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]))
    return model


class TestInterpretabilityEngine:
    """Tests for InterpretabilityEngine class."""
    
    def test_initialization(self, mock_model_registry):
        """Test engine initialization."""
        engine = InterpretabilityEngine(
            model_registry=mock_model_registry,
            feature_names=['sleep_duration', 'hrv_rmssd'],
            clinical_mappings={'sleep_duration': 'Sleep Duration (hours)'}
        )
        
        assert engine.model_registry == mock_model_registry
        assert 'sleep_duration' in engine.clinical_mappings
        assert engine.clinical_mappings['sleep_duration'] == 'Sleep Duration (hours)'
    
    def test_score_to_risk_level(self, mock_model_registry):
        """Test risk score to level conversion."""
        engine = InterpretabilityEngine(mock_model_registry)
        
        assert engine._score_to_risk_level(15) == 'low'
        assert engine._score_to_risk_level(35) == 'moderate'
        assert engine._score_to_risk_level(60) == 'high'
        assert engine._score_to_risk_level(85) == 'critical'
    
    def test_risk_level_to_score_range(self, mock_model_registry):
        """Test risk level to score range conversion."""
        engine = InterpretabilityEngine(mock_model_registry)
        
        assert engine._risk_level_to_score_range('low') == (0, 25)
        assert engine._risk_level_to_score_range('moderate') == (26, 50)
        assert engine._risk_level_to_score_range('high') == (51, 75)
        assert engine._risk_level_to_score_range('critical') == (76, 100)
    
    def test_predict_risk_score_logistic(self, mock_model_registry, sample_features):
        """Test risk score prediction for logistic regression."""
        engine = InterpretabilityEngine(mock_model_registry)
        
        model = Mock()
        model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]))
        
        scores = engine._predict_risk_score(model, 'logistic_regression', sample_features)
        
        assert len(scores) == 3
        assert scores[0] == 70.0  # 0.7 * 100
        assert scores[1] == 40.0  # 0.4 * 100
        assert scores[2] == 80.0  # 0.8 * 100
    
    def test_format_counterfactual_description_success(self, mock_model_registry):
        """Test counterfactual description formatting when target is achieved."""
        engine = InterpretabilityEngine(
            mock_model_registry,
            clinical_mappings={'sleep_duration': 'Sleep Duration (hours)'}
        )
        
        changes = [
            {
                'feature': 'sleep_duration',
                'clinical_name': 'Sleep Duration (hours)',
                'original_value': 5.5,
                'proposed_value': 7.5,
                'change_magnitude': 2.0,
                'importance': 0.8
            }
        ]
        
        description = engine._format_counterfactual_description(
            changes, 'low', 'low', 65.0, 20.0
        )
        
        assert 'To reach low risk level' in description
        assert 'from 65.0 to 20.0' in description
        assert 'Sleep Duration (hours)' in description
        assert 'from 5.50 to 7.50' in description
        assert 'successfully move' in description
    
    def test_format_counterfactual_description_partial(self, mock_model_registry):
        """Test counterfactual description when target is not fully achieved."""
        engine = InterpretabilityEngine(mock_model_registry)
        
        changes = [
            {
                'feature': 'hrv_rmssd',
                'clinical_name': 'hrv_rmssd',
                'original_value': 35.0,
                'proposed_value': 45.0,
                'change_magnitude': 10.0,
                'importance': 0.6
            }
        ]
        
        description = engine._format_counterfactual_description(
            changes, 'low', 'moderate', 65.0, 35.0
        )
        
        assert 'To move toward low risk level' in description
        assert 'from 65.0 to 35.0' in description
        assert 'closer to the target' in description
    
    def test_format_counterfactual_description_no_changes(self, mock_model_registry):
        """Test counterfactual description when no changes needed."""
        engine = InterpretabilityEngine(mock_model_registry)
        
        description = engine._format_counterfactual_description(
            [], 'low', 'low', 15.0, 15.0
        )
        
        assert 'Already at low risk level' in description
        assert 'No changes needed' in description
    
    @patch('src.ml.interpretability.shap')
    def test_compute_shap_values(self, mock_shap, mock_model_registry, sample_features, mock_logistic_model):
        """Test SHAP value computation."""
        # Setup mocks
        mock_model_registry.load_model.return_value = (
            {'model': mock_logistic_model, 'artifacts': {}},
            {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
        )
        
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.array([
            [0.1, -0.2, 0.3, 0.15],
            [0.2, -0.1, 0.25, 0.1],
            [0.15, -0.25, 0.35, 0.2]
        ])
        mock_explainer.expected_value = 0.5
        
        mock_shap.LinearExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = sample_features
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        result = engine.compute_shap_values('model_1', sample_features, top_k=3)
        
        assert 'shap_values' in result
        assert 'base_value' in result
        assert 'top_features' in result
        assert len(result['top_features']) == 3
        assert result['base_value'] == 0.5
    
    def test_generate_counterfactuals_already_at_target(self, mock_model_registry, sample_features):
        """Test counterfactual generation when already at target."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.85, 0.15]]))  # Low risk (15%)
        
        mock_model_registry.load_model.return_value = (
            {'model': mock_model, 'artifacts': {}},
            {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
        )
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        # Test with single instance
        single_instance = sample_features.iloc[0:1]
        counterfactuals = engine.generate_counterfactuals(
            'model_1',
            single_instance,
            target_class='low'
        )
        
        assert len(counterfactuals) == 1
        assert counterfactuals[0]['current_risk_level'] == 'low'
        assert counterfactuals[0]['target_risk_level'] == 'low'
        assert len(counterfactuals[0]['changes_needed']) == 0
        assert 'No changes needed' in counterfactuals[0]['description']
    
    def test_generate_counterfactuals_with_changes(self, mock_model_registry, sample_features):
        """Test counterfactual generation requiring changes."""
        # Setup mock that returns high risk initially
        mock_model = Mock()
        
        # First call returns high risk, subsequent calls return lower risk
        call_count = [0]
        def predict_side_effect(X):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([[0.2, 0.8]])  # High risk (80%)
            else:
                # Gradually decrease risk with each feature change
                return np.array([[0.7, 0.3]])  # Low risk (30%)
        
        mock_model.predict_proba = Mock(side_effect=predict_side_effect)
        mock_model.coef_ = np.array([[0.5, -0.3, 0.2, 0.4]])
        
        mock_model_registry.load_model.return_value = (
            {'model': mock_model, 'artifacts': {}},
            {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
        )
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        # Test with single instance
        single_instance = sample_features.iloc[0:1]
        counterfactuals = engine.generate_counterfactuals(
            'model_1',
            single_instance,
            target_class='low',
            max_changes=3
        )
        
        assert len(counterfactuals) == 1
        cf = counterfactuals[0]
        
        assert cf['current_risk_level'] == 'critical'
        assert cf['target_risk_level'] == 'low'
        assert len(cf['changes_needed']) > 0
        assert 'changes_needed' in cf
        assert 'description' in cf
        
        # Check that changes have required fields
        for change in cf['changes_needed']:
            assert 'feature' in change
            assert 'clinical_name' in change
            assert 'original_value' in change
            assert 'proposed_value' in change
            assert 'change_magnitude' in change
    
    def test_extract_rule_set(self, mock_model_registry, sample_features):
        """Test rule extraction from model."""
        # Create larger training dataset
        np.random.seed(42)
        n_samples = 200
        training_data = pd.DataFrame({
            'sleep_duration': np.random.uniform(4, 9, n_samples),
            'hrv_rmssd': np.random.uniform(20, 80, n_samples),
            'activity_level': np.random.uniform(1, 5, n_samples),
            'sentiment_score': np.random.uniform(0, 1, n_samples)
        })
        
        # Create labels based on simple rules
        training_labels = pd.Series([
            1 if (row['sleep_duration'] < 6 or row['hrv_rmssd'] < 40) else 0
            for _, row in training_data.iterrows()
        ])
        
        # Setup mock model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.column_stack([
            1 - training_labels.values,
            training_labels.values
        ]))
        
        mock_model_registry.load_model.return_value = (
            {'model': mock_model, 'artifacts': {}},
            {'model_type': 'logistic_regression', 'feature_names': list(training_data.columns)}
        )
        
        engine = InterpretabilityEngine(
            mock_model_registry,
            clinical_mappings={
                'sleep_duration': 'Sleep Duration (hours)',
                'hrv_rmssd': 'Heart Rate Variability (RMSSD)'
            }
        )
        
        result = engine.extract_rule_set(
            'model_1',
            training_data,
            training_labels,
            max_depth=3,
            min_samples_leaf=20
        )
        
        # Verify result structure
        assert 'rules' in result
        assert 'max_depth' in result
        assert 'n_rules' in result
        assert 'fidelity' in result
        assert 'feature_names' in result
        
        # Verify max depth constraint
        assert result['max_depth'] == 3
        
        # Verify rules structure
        assert len(result['rules']) > 0
        assert result['n_rules'] == len(result['rules'])
        
        for rule in result['rules']:
            assert 'conditions' in rule
            assert 'prediction' in rule
            assert 'confidence' in rule
            assert 'n_samples' in rule
            assert 'rule_text' in rule
            assert rule['prediction'] in ['high_risk', 'low_risk']
            assert 0 <= rule['confidence'] <= 1
            assert rule['n_samples'] > 0
        
        # Verify fidelity is reasonable
        assert 0 <= result['fidelity'] <= 1
        
        # Verify clinical terminology is used in rules
        rule_texts = [rule['rule_text'] for rule in result['rules']]
        combined_text = ' '.join(rule_texts)
        
        # At least some rules should use clinical names
        has_clinical_terms = any(
            'Sleep Duration (hours)' in text or 'Heart Rate Variability (RMSSD)' in text
            for text in rule_texts
        )
        assert has_clinical_terms or len(result['rules']) == 1  # Allow for single default rule
    
    def test_extract_rule_set_with_error(self, mock_model_registry, sample_features):
        """Test rule extraction error handling."""
        # Setup mock to raise error
        mock_model_registry.load_model.side_effect = Exception("Model load failed")
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        with pytest.raises(InterpretabilityError) as exc_info:
            engine.extract_rule_set(
                'model_1',
                sample_features,
                pd.Series([0, 1, 0]),
                max_depth=3
            )
        
        assert "Rule extraction failed" in str(exc_info.value)
    
    @patch('src.ml.interpretability.shap')
    def test_generate_explanation_all_components(
        self, mock_shap, mock_model_registry, sample_features
    ):
        """Test comprehensive explanation generation with all components."""
        # Create training data
        np.random.seed(42)
        n_samples = 100
        training_data = pd.DataFrame({
            'sleep_duration': np.random.uniform(4, 9, n_samples),
            'hrv_rmssd': np.random.uniform(20, 80, n_samples),
            'activity_level': np.random.uniform(1, 5, n_samples),
            'sentiment_score': np.random.uniform(0, 1, n_samples)
        })
        training_labels = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Setup mock model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        mock_model.coef_ = np.array([[0.5, -0.3, 0.2, 0.4]])
        
        mock_model_registry.load_model.return_value = (
            {'model': mock_model, 'artifacts': {}},
            {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
        )
        
        # Setup SHAP mock
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, 0.15]])
        mock_explainer.expected_value = 0.5
        mock_shap.LinearExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = sample_features
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        # Generate explanation with single instance
        single_instance = sample_features.iloc[0:1]
        result = engine.generate_explanation(
            model_id='model_1',
            features=single_instance,
            training_data=training_data,
            training_labels=training_labels,
            target_class='low',
            timeout_seconds=5.0
        )
        
        # Verify structure
        assert 'model_id' in result
        assert 'n_instances' in result
        assert 'components' in result
        assert 'generation_time' in result
        assert 'total_generation_time' in result
        assert 'timeout_exceeded' in result
        assert 'clinical_summary' in result
        
        # Verify components
        assert result['components']['shap'] is not None
        assert result['components']['counterfactuals'] is not None
        assert result['components']['rules'] is not None
        
        # Verify timing
        assert result['total_generation_time'] < 5.0
        assert result['timeout_exceeded'] is False
        
        # Verify SHAP component
        shap_data = result['components']['shap']
        assert 'base_value' in shap_data
        assert 'top_features' in shap_data
        assert len(shap_data['top_features']) > 0
        
        # Verify counterfactuals component
        cf_data = result['components']['counterfactuals']
        assert len(cf_data) == 1
        assert 'current_risk_level' in cf_data[0]
        assert 'target_risk_level' in cf_data[0]
        
        # Verify rules component
        rules_data = result['components']['rules']
        assert 'rules' in rules_data
        assert 'fidelity' in rules_data
        assert 'n_rules' in rules_data
        
        # Verify clinical summary
        assert isinstance(result['clinical_summary'], str)
        assert len(result['clinical_summary']) > 0
    
    @patch('src.ml.interpretability.shap')
    def test_generate_explanation_selective_components(
        self, mock_shap, mock_model_registry, sample_features
    ):
        """Test explanation generation with selective components."""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        mock_model_registry.load_model.return_value = (
            {'model': mock_model, 'artifacts': {}},
            {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
        )
        
        # Setup SHAP mock
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, 0.15]])
        mock_explainer.expected_value = 0.5
        mock_shap.LinearExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = sample_features
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        # Generate explanation with only SHAP and counterfactuals
        single_instance = sample_features.iloc[0:1]
        result = engine.generate_explanation(
            model_id='model_1',
            features=single_instance,
            include_shap=True,
            include_counterfactuals=True,
            include_rules=False,
            timeout_seconds=3.0
        )
        
        # Verify only requested components are present
        assert result['components']['shap'] is not None
        assert result['components']['counterfactuals'] is not None
        assert result['components']['rules'] is None
        
        # Verify timing
        assert result['total_generation_time'] < 3.0
    
    def test_generate_explanation_timeout(self, mock_model_registry, sample_features):
        """Test explanation generation with timeout."""
        # Setup mock that takes too long
        def slow_load_model(*args, **kwargs):
            import time
            time.sleep(0.5)
            mock_model = Mock()
            mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
            return (
                {'model': mock_model, 'artifacts': {}},
                {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
            )
        
        mock_model_registry.load_model = slow_load_model
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        # Generate explanation with very short timeout
        single_instance = sample_features.iloc[0:1]
        
        # Should complete but may exceed timeout
        result = engine.generate_explanation(
            model_id='model_1',
            features=single_instance,
            include_shap=True,
            include_counterfactuals=True,
            include_rules=False,
            timeout_seconds=0.1
        )
        
        # Should have attempted generation
        assert 'total_generation_time' in result
        assert 'timeout_exceeded' in result
    
    def test_generate_explanation_with_errors(self, mock_model_registry, sample_features):
        """Test explanation generation with component errors."""
        # Setup mock that fails for SHAP but succeeds for counterfactuals
        call_count = [0]
        
        def load_model_with_errors(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (SHAP) fails
                raise Exception("SHAP computation failed")
            else:
                # Subsequent calls succeed
                mock_model = Mock()
                mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
                mock_model.coef_ = np.array([[0.5, -0.3, 0.2, 0.4]])
                return (
                    {'model': mock_model, 'artifacts': {}},
                    {'model_type': 'logistic_regression', 'feature_names': list(sample_features.columns)}
                )
        
        mock_model_registry.load_model = load_model_with_errors
        
        engine = InterpretabilityEngine(mock_model_registry)
        
        # Generate explanation
        single_instance = sample_features.iloc[0:1]
        result = engine.generate_explanation(
            model_id='model_1',
            features=single_instance,
            include_shap=True,
            include_counterfactuals=True,
            include_rules=False,
            timeout_seconds=3.0
        )
        
        # Should have errors recorded
        assert 'errors' in result
        assert len(result['errors']) > 0
        
        # SHAP should have failed
        assert result['components']['shap'] is None
        
        # Counterfactuals should have succeeded
        assert result['components']['counterfactuals'] is not None
    
    def test_clinical_summary_generation(self, mock_model_registry):
        """Test clinical summary generation."""
        engine = InterpretabilityEngine(
            mock_model_registry,
            clinical_mappings={'sleep_duration': 'Sleep Duration (hours)'}
        )
        
        # Create mock explanation
        explanation = {
            'model_id': 'model_1',
            'n_instances': 1,
            'components': {
                'shap': {
                    'base_value': 0.5,
                    'top_features': [
                        {
                            'feature': 'sleep_duration',
                            'clinical_name': 'Sleep Duration (hours)',
                            'importance': 0.8,
                            'mean_shap_value': 0.3
                        }
                    ],
                    'feature_names': ['sleep_duration']
                },
                'counterfactuals': [
                    {
                        'current_risk_level': 'high',
                        'target_risk_level': 'low',
                        'changes_needed': [
                            {
                                'clinical_name': 'Sleep Duration (hours)',
                                'original_value': 5.5,
                                'proposed_value': 7.5
                            }
                        ]
                    }
                ],
                'rules': {
                    'n_rules': 5,
                    'fidelity': 0.85,
                    'rules': [
                        {
                            'prediction': 'high_risk',
                            'confidence': 0.9,
                            'rule_text': 'Sleep Duration (hours) <= 6.0'
                        }
                    ]
                }
            },
            'generation_time': {
                'shap': 0.5,
                'counterfactuals': 0.8,
                'rules': 0.3
            },
            'total_generation_time': 1.6,
            'timeout_exceeded': False,
            'errors': []
        }
        
        summary = engine._generate_clinical_summary(explanation)
        
        # Verify summary content
        assert isinstance(summary, str)
        assert 'Model Explanation Summary' in summary
        assert 'Key Contributing Factors' in summary
        assert 'Sleep Duration (hours)' in summary
        assert 'Actionable Changes' in summary
        assert 'Decision Rules' in summary
        assert 'Generation Performance' in summary
        assert '1.6' in summary  # Total time
