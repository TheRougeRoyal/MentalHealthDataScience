"""Pydantic models for Data Science API endpoints"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID


# Experiment Tracking Models

class ExperimentCreate(BaseModel):
    """Request model for creating an experiment"""
    experiment_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the experiment"
    )
    description: Optional[str] = Field(
        None,
        description="Description of the experiment"
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Tags for the experiment"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "depression_risk_model_v2",
                "description": "Testing new feature engineering approach",
                "tags": {"team": "ml", "priority": "high"}
            }
        }


class ExperimentResponse(BaseModel):
    """Response model for experiment"""
    experiment_id: str = Field(..., description="Experiment ID")
    experiment_name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Description")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
                "experiment_name": "depression_risk_model_v2",
                "description": "Testing new feature engineering approach",
                "tags": {"team": "ml", "priority": "high"},
                "created_at": "2025-11-17T10:30:00Z"
            }
        }


class RunCreate(BaseModel):
    """Request model for creating a run"""
    run_name: Optional[str] = Field(
        None,
        description="Name of the run"
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Tags for the run"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "run_name": "baseline_model",
                "tags": {"model_type": "random_forest", "version": "1.0"}
            }
        }


class RunResponse(BaseModel):
    """Response model for run"""
    run_id: str = Field(..., description="Run ID")
    experiment_id: str = Field(..., description="Experiment ID")
    run_name: Optional[str] = Field(None, description="Run name")
    status: str = Field(..., description="Run status")
    start_time: datetime = Field(..., description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags")
    git_commit: Optional[str] = Field(None, description="Git commit hash")

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "660e8400-e29b-41d4-a716-446655440000",
                "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
                "run_name": "baseline_model",
                "status": "RUNNING",
                "start_time": "2025-11-17T10:30:00Z",
                "end_time": None,
                "params": {"learning_rate": 0.01, "n_estimators": 100},
                "tags": {"model_type": "random_forest"},
                "git_commit": "abc123def456"
            }
        }


class MetricLog(BaseModel):
    """Request model for logging metrics"""
    metrics: Dict[str, float] = Field(
        ...,
        description="Metrics to log"
    )
    step: Optional[int] = Field(
        None,
        description="Step number for the metrics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {"accuracy": 0.85, "f1_score": 0.82},
                "step": 100
            }
        }


class ParamLog(BaseModel):
    """Request model for logging parameters"""
    params: Dict[str, Any] = Field(
        ...,
        description="Parameters to log"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "params": {
                    "learning_rate": 0.01,
                    "n_estimators": 100,
                    "max_depth": 10
                }
            }
        }


class ArtifactLog(BaseModel):
    """Request model for logging artifacts"""
    artifact_type: str = Field(
        ...,
        description="Type of artifact (model, plot, data, report)"
    )
    artifact_path: str = Field(
        ...,
        description="Path to the artifact file"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "artifact_type": "model",
                "artifact_path": "/tmp/model.pkl",
                "metadata": {"format": "pickle", "size_mb": 15.2}
            }
        }


class RunSearchRequest(BaseModel):
    """Request model for searching runs"""
    experiment_name: Optional[str] = Field(
        None,
        description="Filter by experiment name"
    )
    filter_string: Optional[str] = Field(
        None,
        description="Filter expression (e.g., 'metrics.accuracy > 0.8')"
    )
    order_by: Optional[List[str]] = Field(
        None,
        description="Order by fields (e.g., ['metrics.accuracy DESC'])"
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "depression_risk_model_v2",
                "filter_string": "metrics.accuracy > 0.8",
                "order_by": ["metrics.accuracy DESC"],
                "limit": 50
            }
        }


class RunCompareRequest(BaseModel):
    """Request model for comparing runs"""
    run_ids: List[str] = Field(
        ...,
        min_length=2,
        description="List of run IDs to compare"
    )
    metric_names: Optional[List[str]] = Field(
        None,
        description="Specific metrics to compare"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "run_ids": [
                    "660e8400-e29b-41d4-a716-446655440000",
                    "770e8400-e29b-41d4-a716-446655440000"
                ],
                "metric_names": ["accuracy", "f1_score", "auc"]
            }
        }


# Data Versioning Models

class DatasetRegister(BaseModel):
    """Request model for registering a dataset"""
    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the dataset"
    )
    source: str = Field(
        ...,
        description="Source of the dataset"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Dataset data (will be converted to DataFrame)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "training_data_v1",
                "source": "emr_system",
                "data": {
                    "columns": ["feature1", "feature2", "target"],
                    "values": [[1.0, 2.0, 0], [1.5, 2.5, 1]]
                },
                "metadata": {"description": "Initial training dataset"}
            }
        }


class DatasetVersionResponse(BaseModel):
    """Response model for dataset version"""
    version_id: str = Field(..., description="Version ID")
    dataset_name: str = Field(..., description="Dataset name")
    version: str = Field(..., description="Version string")
    source: str = Field(..., description="Source")
    num_rows: int = Field(..., description="Number of rows")
    num_columns: int = Field(..., description="Number of columns")
    schema: Dict[str, str] = Field(..., description="Column schema")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "version_id": "880e8400-e29b-41d4-a716-446655440000",
                "dataset_name": "training_data_v1",
                "version": "v1.0.0",
                "source": "emr_system",
                "num_rows": 10000,
                "num_columns": 25,
                "schema": {"feature1": "float64", "feature2": "float64"},
                "created_at": "2025-11-17T10:30:00Z"
            }
        }


class DriftCheckRequest(BaseModel):
    """Request model for drift checking"""
    dataset_version_id1: str = Field(
        ...,
        description="First dataset version ID"
    )
    dataset_version_id2: str = Field(
        ...,
        description="Second dataset version ID"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_version_id1": "880e8400-e29b-41d4-a716-446655440000",
                "dataset_version_id2": "990e8400-e29b-41d4-a716-446655440000"
            }
        }


class DriftReportResponse(BaseModel):
    """Response model for drift report"""
    version_id1: str = Field(..., description="First version ID")
    version_id2: str = Field(..., description="Second version ID")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Overall drift score")
    feature_drifts: Dict[str, float] = Field(..., description="Per-feature drift scores")
    recommendations: List[str] = Field(..., description="Recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "version_id1": "880e8400-e29b-41d4-a716-446655440000",
                "version_id2": "990e8400-e29b-41d4-a716-446655440000",
                "drift_detected": True,
                "drift_score": 0.35,
                "feature_drifts": {"feature1": 0.45, "feature2": 0.25},
                "recommendations": ["Consider retraining the model"]
            }
        }


# Feature Store Models

class FeatureRegister(BaseModel):
    """Request model for registering a feature"""
    feature_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the feature"
    )
    feature_type: str = Field(
        ...,
        description="Type of feature (numeric, categorical, embedding)"
    )
    description: str = Field(
        ...,
        description="Description of the feature"
    )
    transformation_code: str = Field(
        ...,
        description="Python code for feature transformation"
    )
    input_schema: Dict[str, str] = Field(
        ...,
        description="Input schema"
    )
    output_schema: Dict[str, str] = Field(
        ...,
        description="Output schema"
    )
    dependencies: Optional[List[str]] = Field(
        None,
        description="List of dependent features"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_name": "phq9_severity",
                "feature_type": "categorical",
                "description": "PHQ-9 severity category",
                "transformation_code": "lambda x: 'severe' if x > 20 else 'moderate' if x > 10 else 'mild'",
                "input_schema": {"phq9_score": "int"},
                "output_schema": {"phq9_severity": "str"},
                "dependencies": []
            }
        }


class FeatureResponse(BaseModel):
    """Response model for feature"""
    feature_id: str = Field(..., description="Feature ID")
    feature_name: str = Field(..., description="Feature name")
    feature_type: str = Field(..., description="Feature type")
    description: str = Field(..., description="Description")
    version: str = Field(..., description="Version")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "feature_id": "aa0e8400-e29b-41d4-a716-446655440000",
                "feature_name": "phq9_severity",
                "feature_type": "categorical",
                "description": "PHQ-9 severity category",
                "version": "1.0",
                "created_at": "2025-11-17T10:30:00Z"
            }
        }


class FeatureComputeRequest(BaseModel):
    """Request model for computing features"""
    feature_names: List[str] = Field(
        ...,
        min_length=1,
        description="List of features to compute"
    )
    input_data: Dict[str, Any] = Field(
        ...,
        description="Input data for feature computation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_names": ["phq9_severity", "sleep_quality_score"],
                "input_data": {
                    "columns": ["phq9_score", "sleep_hours"],
                    "values": [[15, 6.5], [22, 4.0]]
                }
            }
        }


class FeatureServeRequest(BaseModel):
    """Request model for serving features"""
    feature_names: List[str] = Field(
        ...,
        min_length=1,
        description="List of features to serve"
    )
    entity_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of entity IDs"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "feature_names": ["phq9_severity", "sleep_quality_score"],
                "entity_ids": ["patient_001", "patient_002"]
            }
        }


# EDA and Reporting Models

class EDAAnalyzeRequest(BaseModel):
    """Request model for EDA analysis"""
    data: Dict[str, Any] = Field(
        ...,
        description="Dataset to analyze"
    )
    target_column: Optional[str] = Field(
        None,
        description="Target column for supervised analysis"
    )
    dataset_name: Optional[str] = Field(
        None,
        description="Name for the dataset"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "columns": ["feature1", "feature2", "target"],
                    "values": [[1.0, 2.0, 0], [1.5, 2.5, 1]]
                },
                "target_column": "target",
                "dataset_name": "training_data"
            }
        }


class EDAReportResponse(BaseModel):
    """Response model for EDA report"""
    report_id: str = Field(..., description="Report ID")
    dataset_name: str = Field(..., description="Dataset name")
    num_rows: int = Field(..., description="Number of rows")
    num_columns: int = Field(..., description="Number of columns")
    summary_statistics: Dict[str, Any] = Field(..., description="Summary statistics")
    quality_issues: List[Dict[str, Any]] = Field(..., description="Data quality issues")
    recommendations: List[str] = Field(..., description="Recommendations")
    generated_at: datetime = Field(..., description="Generation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "bb0e8400-e29b-41d4-a716-446655440000",
                "dataset_name": "training_data",
                "num_rows": 10000,
                "num_columns": 25,
                "summary_statistics": {"feature1": {"mean": 1.5, "std": 0.5}},
                "quality_issues": [
                    {"issue_type": "missing", "severity": "medium", "description": "10% missing values in feature2"}
                ],
                "recommendations": ["Impute missing values in feature2"],
                "generated_at": "2025-11-17T10:30:00Z"
            }
        }


class ModelCardGenerateRequest(BaseModel):
    """Request model for generating model card"""
    model_id: str = Field(
        ...,
        description="Model ID from model registry"
    )
    run_id: Optional[str] = Field(
        None,
        description="Associated experiment run ID"
    )
    include_fairness: bool = Field(
        default=True,
        description="Include fairness analysis"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_depression_v2",
                "run_id": "660e8400-e29b-41d4-a716-446655440000",
                "include_fairness": True
            }
        }


class ModelCardResponse(BaseModel):
    """Response model for model card"""
    card_id: str = Field(..., description="Model card ID")
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    fairness_metrics: Optional[Dict[str, float]] = Field(None, description="Fairness metrics")
    feature_importance: List[tuple] = Field(..., description="Feature importance")
    generated_at: datetime = Field(..., description="Generation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "card_id": "cc0e8400-e29b-41d4-a716-446655440000",
                "model_id": "model_depression_v2",
                "model_name": "Depression Risk Predictor",
                "model_type": "RandomForest",
                "version": "2.0",
                "metrics": {"accuracy": 0.85, "f1_score": 0.82},
                "fairness_metrics": {"demographic_parity": 0.95},
                "feature_importance": [("phq9_score", 0.35), ("sleep_hours", 0.25)],
                "generated_at": "2025-11-17T10:30:00Z"
            }
        }
