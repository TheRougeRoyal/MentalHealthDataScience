"""Tests for Data Science API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from uuid import uuid4

from src.api.app import app
from src.ds.models import Experiment, Run, DatasetVersion, DriftReport, EDAReport
from src.ds.feature_store import FeatureDefinition


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch('src.api.ds_endpoints.verify_authentication') as mock:
        mock.return_value = Mock(user_id="test_user", authenticated=True)
        yield mock


@pytest.fixture
def mock_experiment_tracker():
    """Mock experiment tracker."""
    with patch('src.api.ds_endpoints._experiment_tracker') as mock:
        yield mock


@pytest.fixture
def mock_data_version_control():
    """Mock data version control."""
    with patch('src.api.ds_endpoints._data_version_control') as mock:
        yield mock


@pytest.fixture
def mock_feature_store():
    """Mock feature store."""
    with patch('src.api.ds_endpoints._feature_store') as mock:
        yield mock


@pytest.fixture
def mock_eda_module():
    """Mock EDA module."""
    with patch('src.api.ds_endpoints._eda_module') as mock:
        yield mock


@pytest.fixture
def mock_model_card_generator():
    """Mock model card generator."""
    with patch('src.api.ds_endpoints._model_card_generator') as mock:
        yield mock


# ============================================================================
# Experiment Tracking Tests
# ============================================================================

def test_create_experiment(client, mock_auth, mock_experiment_tracker):
    """Test creating an experiment."""
    experiment_id = str(uuid4())
    mock_experiment = Mock(
        experiment_id=experiment_id,
        experiment_name="test_experiment",
        description="Test description",
        tags={"team": "ml"},
        created_at=datetime.utcnow()
    )
    mock_experiment_tracker.create_experiment.return_value = mock_experiment
    
    response = client.post(
        "/api/v1/experiments",
        json={
            "experiment_name": "test_experiment",
            "description": "Test description",
            "tags": {"team": "ml"}
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["experiment_name"] == "test_experiment"
    assert data["description"] == "Test description"


def test_list_experiments(client, mock_auth, mock_experiment_tracker):
    """Test listing experiments."""
    experiments = [
        Mock(
            experiment_id=str(uuid4()),
            experiment_name=f"experiment_{i}",
            description=None,
            tags=None,
            created_at=datetime.utcnow()
        )
        for i in range(3)
    ]
    mock_experiment_tracker.list_experiments.return_value = experiments
    
    response = client.get(
        "/api/v1/experiments",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_create_run(client, mock_auth, mock_experiment_tracker):
    """Test creating a run."""
    experiment_id = str(uuid4())
    run_id = str(uuid4())
    mock_run = Mock(
        run_id=run_id,
        experiment_id=experiment_id,
        run_name="test_run",
        status="RUNNING",
        start_time=datetime.utcnow(),
        end_time=None,
        params={},
        tags={"version": "1.0"},
        git_commit="abc123"
    )
    mock_experiment_tracker.start_run.return_value = mock_run
    
    response = client.post(
        f"/api/v1/experiments/{experiment_id}/runs",
        json={
            "run_name": "test_run",
            "tags": {"version": "1.0"}
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["run_name"] == "test_run"
    assert data["status"] == "RUNNING"


def test_log_metrics(client, mock_auth, mock_experiment_tracker):
    """Test logging metrics."""
    run_id = str(uuid4())
    
    response = client.post(
        f"/api/v1/runs/{run_id}/metrics",
        json={
            "metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "step": 100
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Metrics logged successfully"


def test_search_runs(client, mock_auth, mock_experiment_tracker):
    """Test searching runs."""
    runs = [
        Mock(
            run_id=str(uuid4()),
            experiment_id=str(uuid4()),
            run_name=f"run_{i}",
            status="FINISHED",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            params={"lr": 0.01},
            tags=None,
            git_commit="abc123"
        )
        for i in range(2)
    ]
    mock_experiment_tracker.search_runs.return_value = runs
    
    response = client.post(
        "/api/v1/runs/search",
        json={
            "experiment_name": "test_experiment",
            "limit": 50
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


# ============================================================================
# Data Versioning Tests
# ============================================================================

def test_register_dataset(client, mock_auth, mock_data_version_control):
    """Test registering a dataset."""
    version_id = str(uuid4())
    mock_version = Mock(
        version_id=version_id,
        dataset_name="test_dataset",
        version="v1.0.0",
        source="test_source",
        num_rows=100,
        num_columns=5,
        schema={"col1": "float64", "col2": "int64"},
        created_at=datetime.utcnow()
    )
    mock_data_version_control.register_dataset.return_value = mock_version
    
    response = client.post(
        "/api/v1/datasets",
        json={
            "dataset_name": "test_dataset",
            "source": "test_source",
            "data": {
                "columns": ["col1", "col2"],
                "values": [[1.0, 2], [3.0, 4]]
            }
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["dataset_name"] == "test_dataset"
    assert data["num_rows"] == 100


def test_list_dataset_versions(client, mock_auth, mock_data_version_control):
    """Test listing dataset versions."""
    versions = [
        Mock(
            version_id=str(uuid4()),
            dataset_name="test_dataset",
            version=f"v1.{i}.0",
            source="test_source",
            num_rows=100,
            num_columns=5,
            schema={},
            created_at=datetime.utcnow()
        )
        for i in range(3)
    ]
    mock_data_version_control.list_versions.return_value = versions
    
    response = client.get(
        "/api/v1/datasets/test_dataset/versions",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_check_drift(client, mock_auth, mock_data_version_control):
    """Test checking data drift."""
    version_id1 = str(uuid4())
    version_id2 = str(uuid4())
    mock_drift_report = Mock(
        version_id1=version_id1,
        version_id2=version_id2,
        drift_detected=True,
        drift_score=0.35,
        feature_drifts={"feature1": 0.45, "feature2": 0.25},
        recommendations=["Consider retraining"]
    )
    mock_data_version_control.detect_drift.return_value = mock_drift_report
    
    response = client.post(
        "/api/v1/datasets/drift",
        json={
            "dataset_version_id1": version_id1,
            "dataset_version_id2": version_id2
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is True
    assert data["drift_score"] == 0.35


# ============================================================================
# Feature Store Tests
# ============================================================================

def test_register_feature(client, mock_auth, mock_feature_store):
    """Test registering a feature."""
    feature_id = str(uuid4())
    mock_feature = Mock(
        feature_id=feature_id,
        feature_name="test_feature",
        feature_type="numeric",
        description="Test feature",
        version="1.0",
        created_at=datetime.utcnow()
    )
    mock_feature_store.get_feature.return_value = mock_feature
    
    response = client.post(
        "/api/v1/features",
        json={
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": "Test feature",
            "transformation_code": "lambda x: x * 2",
            "input_schema": {"input": "float"},
            "output_schema": {"output": "float"}
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["feature_name"] == "test_feature"


def test_list_features(client, mock_auth, mock_feature_store):
    """Test listing features."""
    features = [
        Mock(
            feature_id=str(uuid4()),
            feature_name=f"feature_{i}",
            feature_type="numeric",
            description="Test",
            version="1.0",
            created_at=datetime.utcnow()
        )
        for i in range(3)
    ]
    mock_feature_store.list_features.return_value = features
    
    response = client.get(
        "/api/v1/features",
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_compute_features(client, mock_auth, mock_feature_store):
    """Test computing features."""
    import pandas as pd
    mock_df = pd.DataFrame({"feature_1": [1.0, 2.0], "feature_2": [3.0, 4.0]})
    mock_feature_store.compute_features.return_value = mock_df
    
    response = client.post(
        "/api/v1/features/compute",
        json={
            "feature_names": ["feature_1", "feature_2"],
            "input_data": {
                "columns": ["input"],
                "values": [[1.0], [2.0]]
            }
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "features" in data


# ============================================================================
# EDA and Reporting Tests
# ============================================================================

def test_analyze_dataset(client, mock_auth, mock_eda_module):
    """Test EDA analysis."""
    from src.ds.eda import DataQualityIssue
    
    mock_report = Mock(
        num_rows=100,
        num_columns=5,
        summary_statistics={"col1": {"mean": 1.5}},
        quality_issues=[
            DataQualityIssue(
                issue_type="missing",
                severity="medium",
                column="col1",
                description="10% missing",
                affected_rows=10,
                recommendation="Impute"
            )
        ],
        recommendations=["Impute missing values"],
        generated_at=datetime.utcnow()
    )
    mock_eda_module.analyze_dataset.return_value = mock_report
    
    response = client.post(
        "/api/v1/eda/analyze",
        json={
            "data": {
                "columns": ["col1", "col2"],
                "values": [[1.0, 2.0], [3.0, 4.0]]
            },
            "dataset_name": "test_dataset"
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["num_rows"] == 100
    assert len(data["quality_issues"]) == 1


def test_generate_model_card(client, mock_auth, mock_model_card_generator):
    """Test generating model card."""
    from src.ds.model_cards import ModelCard
    
    mock_card = Mock(
        model_id="test_model",
        model_name="Test Model",
        model_type="RandomForest",
        version="1.0",
        metrics={"accuracy": 0.85},
        fairness_metrics={"demographic_parity": 0.95},
        feature_importance=[("feature1", 0.5), ("feature2", 0.3)],
        date=datetime.utcnow()
    )
    mock_model_card_generator.generate_model_card.return_value = mock_card
    
    response = client.post(
        "/api/v1/model-cards/generate",
        json={
            "model_id": "test_model",
            "include_fairness": True
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Test Model"
    assert data["metrics"]["accuracy"] == 0.85
