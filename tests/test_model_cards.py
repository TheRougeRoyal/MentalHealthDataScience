"""Tests for model card generator."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import json

from src.ds.model_cards import ModelCard, ModelCardGenerator
from src.ml.model_registry import ModelRegistry
from src.ds.experiment_tracker import ExperimentTracker
from src.ml.interpretability import InterpretabilityEngine


class TestModelCard:
    """Test ModelCard model."""
    
    def test_model_card_creation(self):
        """Test creating a valid model card."""
        card = ModelCard(
            model_id="test_model_123",
            model_name="Test Model",
            model_type="logistic_regression",
            version="1.0",
            date=datetime.utcnow(),
            intended_use="Testing purposes",
            training_data_description="Test dataset",
            training_data_size=1000,
            metrics={"accuracy": 0.85, "precision": 0.82}
        )
        
        assert card.model_id == "test_model_123"
        assert card.model_name == "Test Model"
        assert card.model_type == "logistic_regression"
        assert card.metrics["accuracy"] == 0.85
    
    def test_model_card_with_fairness(self):
        """Test model card with fairness metrics."""
        card = ModelCard(
            model_id="test_model_123",
            model_name="Test Model",
            model_type="logistic_regression",
            version="1.0",
            date=datetime.utcnow(),
            intended_use="Testing purposes",
            training_data_description="Test dataset",
            training_data_size=1000,
            metrics={"accuracy": 0.85},
            fairness_metrics={
                "demographic_parity_difference": 0.05,
                "equalized_odds_difference": 0.08
            },
            bias_analysis="Model shows acceptable fairness metrics"
        )
        
        assert card.fairness_metrics is not None
        assert "demographic_parity_difference" in card.fairness_metrics
        assert card.bias_analysis is not None
    
    def test_model_card_with_interpretability(self):
        """Test model card with interpretability data."""
        card = ModelCard(
            model_id="test_model_123",
            model_name="Test Model",
            model_type="logistic_regression",
            version="1.0",
            date=datetime.utcnow(),
            intended_use="Testing purposes",
            training_data_description="Test dataset",
            training_data_size=1000,
            metrics={"accuracy": 0.85},
            feature_importance=[
                ("feature1", 0.45),
                ("feature2", 0.30),
                ("feature3", 0.25)
            ],
            shap_summary="Top features contribute significantly to predictions"
        )
        
        assert len(card.feature_importance) == 3
        assert card.feature_importance[0][0] == "feature1"
        assert card.shap_summary is not None


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model_registry(temp_dir):
    """Create mock model registry."""
    registry = ModelRegistry(registry_dir=temp_dir)
    
    # Register a test model
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    model = LogisticRegression()
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    metadata = {
        "model_name": "Test Logistic Regression",
        "version": "1.0",
        "owner": "test_user",
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        },
        "training_samples": 1000,
        "preprocessing": ["Feature scaling"],
        "fairness_metrics": {
            "demographic_parity_difference": 0.05
        },
        "feature_importance": {
            "feature1": 0.45,
            "feature2": 0.30,
            "feature3": 0.15,
            "feature4": 0.07,
            "feature5": 0.03
        }
    }
    
    model_id = registry.register_model(
        model=model,
        model_type="logistic_regression",
        metadata=metadata
    )
    
    return registry, model_id


@pytest.fixture
def model_card_generator(mock_model_registry):
    """Create model card generator."""
    registry, model_id = mock_model_registry
    generator = ModelCardGenerator(model_registry=registry)
    return generator, model_id


class TestModelCardGenerator:
    """Test ModelCardGenerator class."""
    
    def test_generator_initialization(self, mock_model_registry):
        """Test generator initialization."""
        registry, _ = mock_model_registry
        generator = ModelCardGenerator(model_registry=registry)
        
        assert generator.model_registry is not None
        assert generator.experiment_tracker is None
        assert generator.interpretability_engine is None
    
    def test_generate_basic_model_card(self, model_card_generator):
        """Test generating a basic model card."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(
            model_id=model_id,
            include_fairness=True,
            include_interpretability=True
        )
        
        assert card.model_id == model_id
        assert card.model_name == "Test Logistic Regression"
        assert card.model_type == "logistic_regression"
        assert card.version == "1.0"
        assert card.owner == "test_user"
        assert "accuracy" in card.metrics
        assert card.metrics["accuracy"] == 0.85
    
    def test_generate_card_with_baseline_comparison(self, model_card_generator):
        """Test generating card with baseline comparison."""
        generator, model_id = model_card_generator
        registry = generator.model_registry
        
        # Register a baseline model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        baseline_model = LogisticRegression()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        baseline_model.fit(X, y)
        
        baseline_metadata = {
            "model_name": "Baseline Model",
            "metrics": {
                "accuracy": 0.75,
                "precision": 0.72
            }
        }
        
        baseline_id = registry.register_model(
            model=baseline_model,
            model_type="logistic_regression",
            metadata=baseline_metadata,
            set_active=False
        )
        
        # Generate card with baseline comparison
        card = generator.generate_model_card(
            model_id=model_id,
            baseline_model_ids=[baseline_id]
        )
        
        assert card.baseline_comparison is not None
        assert "accuracy_improvement" in card.baseline_comparison
        assert card.baseline_comparison["accuracy_improvement"] == pytest.approx(0.10, abs=0.01)
    
    def test_extract_fairness_metrics(self, model_card_generator):
        """Test extracting fairness metrics."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(
            model_id=model_id,
            include_fairness=True
        )
        
        assert card.fairness_metrics is not None
        assert "demographic_parity_difference" in card.fairness_metrics
        assert card.bias_analysis is not None
    
    def test_extract_interpretability_data(self, model_card_generator):
        """Test extracting interpretability data."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(
            model_id=model_id,
            include_interpretability=True
        )
        
        assert len(card.feature_importance) > 0
        assert card.feature_importance[0][0] == "feature1"
        assert card.feature_importance[0][1] == 0.45
    
    def test_generate_limitations(self, model_card_generator):
        """Test generating limitations."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        assert len(card.known_limitations) > 0
        assert len(card.caveats) > 0
    
    def test_generate_recommendations(self, model_card_generator):
        """Test generating recommendations."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        assert len(card.recommendations) > 0
    
    def test_compute_fairness_from_group_performance(self, model_card_generator):
        """Test computing fairness metrics from group performance."""
        generator, _ = model_card_generator
        
        performance_by_group = {
            "group_a": {
                "accuracy": 0.85,
                "tpr": 0.80,
                "fpr": 0.15,
                "positive_rate": 0.40
            },
            "group_b": {
                "accuracy": 0.82,
                "tpr": 0.75,
                "fpr": 0.20,
                "positive_rate": 0.35
            }
        }
        
        fairness_metrics = generator._compute_fairness_from_group_performance(
            performance_by_group
        )
        
        assert "demographic_parity_difference" in fairness_metrics
        assert fairness_metrics["demographic_parity_difference"] == pytest.approx(0.05)
        assert "equalized_odds_difference" in fairness_metrics
        assert "accuracy_parity_difference" in fairness_metrics


class TestModelCardExport:
    """Test model card export functionality."""
    
    def test_export_html(self, model_card_generator, temp_dir):
        """Test exporting model card as HTML."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        output_path = Path(temp_dir) / "model_card.html"
        exported_path = generator.export_card(
            model_card=card,
            format="html",
            output_path=str(output_path)
        )
        
        assert Path(exported_path).exists()
        
        # Check HTML content
        with open(exported_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test Logistic Regression" in content
            assert "Model Card" in content
            assert "accuracy" in content
    
    def test_export_markdown(self, model_card_generator, temp_dir):
        """Test exporting model card as Markdown."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        output_path = Path(temp_dir) / "model_card.md"
        exported_path = generator.export_card(
            model_card=card,
            format="markdown",
            output_path=str(output_path)
        )
        
        assert Path(exported_path).exists()
        
        # Check Markdown content
        with open(exported_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Model Card" in content
            assert "Test Logistic Regression" in content
            assert "## Performance Metrics" in content
    
    def test_export_json(self, model_card_generator, temp_dir):
        """Test exporting model card as JSON."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        output_path = Path(temp_dir) / "model_card.json"
        exported_path = generator.export_card(
            model_card=card,
            format="json",
            output_path=str(output_path)
        )
        
        assert Path(exported_path).exists()
        
        # Check JSON content
        with open(exported_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["model_id"] == model_id
            assert data["model_name"] == "Test Logistic Regression"
            assert "metrics" in data
    
    def test_export_default_path(self, model_card_generator, temp_dir):
        """Test exporting with default path generation."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        # Export without specifying path
        exported_path = generator.export_card(
            model_card=card,
            format="html"
        )
        
        assert Path(exported_path).exists()
        assert "model_card_" in exported_path
        assert exported_path.endswith(".html")
        
        # Clean up
        Path(exported_path).unlink()
    
    def test_export_unsupported_format(self, model_card_generator):
        """Test exporting with unsupported format."""
        generator, model_id = model_card_generator
        
        card = generator.generate_model_card(model_id=model_id)
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            generator.export_card(
                model_card=card,
                format="xml"
            )
