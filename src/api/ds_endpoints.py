"""API endpoints for Data Science features"""

import logging
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from uuid import uuid4

from src.api.ds_models import (
    # Experiment Tracking
    ExperimentCreate, ExperimentResponse,
    RunCreate, RunResponse,
    MetricLog, ParamLog, ArtifactLog,
    RunSearchRequest, RunCompareRequest,
    # Data Versioning
    DatasetRegister, DatasetVersionResponse,
    DriftCheckRequest, DriftReportResponse,
    # Feature Store
    FeatureRegister, FeatureResponse,
    FeatureComputeRequest, FeatureServeRequest,
    # EDA and Reporting
    EDAAnalyzeRequest, EDAReportResponse,
    ModelCardGenerateRequest, ModelCardResponse
)
from src.api.auth import AuthResult, authenticator
from src.api.endpoints import verify_authentication
from src.ds.experiment_tracker import ExperimentTracker
from src.ds.data_versioning import DataVersionControl
from src.ds.feature_store import FeatureStore
from src.ds.eda import EDAModule
from src.ds.model_cards import ModelCardGenerator
from src.ds.storage import FileSystemStorage
from src.ml.model_registry import ModelRegistry
from src.database.connection import get_db_connection

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Data Science"])

# Global components (initialized on startup)
_experiment_tracker: Optional[ExperimentTracker] = None
_data_version_control: Optional[DataVersionControl] = None
_feature_store: Optional[FeatureStore] = None
_eda_module: Optional[EDAModule] = None
_model_card_generator: Optional[ModelCardGenerator] = None


def initialize_ds_components():
    """Initialize data science components"""
    global _experiment_tracker, _data_version_control, _feature_store
    global _eda_module, _model_card_generator
    
    logger.info("Initializing data science API components...")
    
    try:
        # Initialize storage backend
        storage = FileSystemStorage()
        
        # Get database connection
        db_conn = get_db_connection()
        
        # Initialize components
        _experiment_tracker = ExperimentTracker(storage, db_conn)
        _data_version_control = DataVersionControl(storage, db_conn)
        _feature_store = FeatureStore(db_conn)
        _eda_module = EDAModule()
        
        # Initialize model card generator with dependencies
        model_registry = ModelRegistry()
        _model_card_generator = ModelCardGenerator(model_registry, _experiment_tracker)
        
        logger.info("Data science API components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data science components: {e}")
        raise


# ============================================================================
# Experiment Tracking Endpoints
# ============================================================================

@router.post(
    "/experiments",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new experiment"
)
async def create_experiment(
    request: ExperimentCreate,
    auth: AuthResult = Depends(verify_authentication)
) -> ExperimentResponse:
    """
    Create a new experiment for tracking ML runs.
    
    Args:
        request: Experiment creation request
        auth: Authentication result
    
    Returns:
        Created experiment details
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        logger.info(f"Creating experiment '{request.experiment_name}' by user {auth.user_id}")
        
        # Create experiment in tracker
        experiment = _experiment_tracker.create_experiment(
            experiment_name=request.experiment_name,
            description=request.description,
            tags=request.tags
        )
        
        return ExperimentResponse(
            experiment_id=str(experiment.experiment_id),
            experiment_name=experiment.experiment_name,
            description=experiment.description,
            tags=experiment.tags,
            created_at=experiment.created_at
        )
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}"
        )


@router.get(
    "/experiments",
    response_model=List[ExperimentResponse],
    summary="List all experiments"
)
async def list_experiments(
    auth: AuthResult = Depends(verify_authentication),
    limit: int = 100
) -> List[ExperimentResponse]:
    """
    List all experiments.
    
    Args:
        auth: Authentication result
        limit: Maximum number of experiments to return
    
    Returns:
        List of experiments
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        experiments = _experiment_tracker.list_experiments(limit=limit)
        
        return [
            ExperimentResponse(
                experiment_id=str(exp.experiment_id),
                experiment_name=exp.experiment_name,
                description=exp.description,
                tags=exp.tags,
                created_at=exp.created_at
            )
            for exp in experiments
        ]
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}"
        )


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment by ID"
)
async def get_experiment(
    experiment_id: str,
    auth: AuthResult = Depends(verify_authentication)
) -> ExperimentResponse:
    """
    Get experiment details by ID.
    
    Args:
        experiment_id: Experiment ID
        auth: Authentication result
    
    Returns:
        Experiment details
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        experiment = _experiment_tracker.get_experiment(experiment_id)
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        return ExperimentResponse(
            experiment_id=str(experiment.experiment_id),
            experiment_name=experiment.experiment_name,
            description=experiment.description,
            tags=experiment.tags,
            created_at=experiment.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment: {str(e)}"
        )


@router.post(
    "/experiments/{experiment_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new run in an experiment"
)
async def create_run(
    experiment_id: str,
    request: RunCreate,
    auth: AuthResult = Depends(verify_authentication)
) -> RunResponse:
    """
    Create a new run within an experiment.
    
    Args:
        experiment_id: Experiment ID
        request: Run creation request
        auth: Authentication result
    
    Returns:
        Created run details
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        logger.info(f"Creating run in experiment {experiment_id} by user {auth.user_id}")
        
        # Start run
        run = _experiment_tracker.start_run(
            experiment_id=experiment_id,
            run_name=request.run_name,
            tags=request.tags
        )
        
        return RunResponse(
            run_id=str(run.run_id),
            experiment_id=str(run.experiment_id),
            run_name=run.run_name,
            status=run.status,
            start_time=run.start_time,
            end_time=run.end_time,
            params=run.params,
            tags=run.tags,
            git_commit=run.git_commit
        )
    except Exception as e:
        logger.error(f"Error creating run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create run: {str(e)}"
        )


@router.post(
    "/runs/{run_id}/metrics",
    status_code=status.HTTP_200_OK,
    summary="Log metrics for a run"
)
async def log_metrics(
    run_id: str,
    request: MetricLog,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Log metrics for a specific run.
    
    Args:
        run_id: Run ID
        request: Metrics to log
        auth: Authentication result
    
    Returns:
        Success message
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        _experiment_tracker.log_metrics(
            run_id=run_id,
            metrics=request.metrics,
            step=request.step
        )
        
        return {"message": "Metrics logged successfully", "run_id": run_id}
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log metrics: {str(e)}"
        )


@router.post(
    "/runs/{run_id}/params",
    status_code=status.HTTP_200_OK,
    summary="Log parameters for a run"
)
async def log_params(
    run_id: str,
    request: ParamLog,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Log parameters for a specific run.
    
    Args:
        run_id: Run ID
        request: Parameters to log
        auth: Authentication result
    
    Returns:
        Success message
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        _experiment_tracker.log_params(
            run_id=run_id,
            params=request.params
        )
        
        return {"message": "Parameters logged successfully", "run_id": run_id}
    except Exception as e:
        logger.error(f"Error logging parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log parameters: {str(e)}"
        )


@router.post(
    "/runs/{run_id}/artifacts",
    status_code=status.HTTP_200_OK,
    summary="Log artifact for a run"
)
async def log_artifact(
    run_id: str,
    request: ArtifactLog,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Log an artifact for a specific run.
    
    Args:
        run_id: Run ID
        request: Artifact to log
        auth: Authentication result
    
    Returns:
        Success message with artifact URI
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        artifact_uri = _experiment_tracker.log_artifact(
            run_id=run_id,
            artifact_path=request.artifact_path,
            artifact_type=request.artifact_type,
            metadata=request.metadata
        )
        
        return {
            "message": "Artifact logged successfully",
            "run_id": run_id,
            "artifact_uri": artifact_uri
        }
    except Exception as e:
        logger.error(f"Error logging artifact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log artifact: {str(e)}"
        )


@router.post(
    "/runs/search",
    response_model=List[RunResponse],
    summary="Search runs"
)
async def search_runs(
    request: RunSearchRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> List[RunResponse]:
    """
    Search and filter runs.
    
    Args:
        request: Search criteria
        auth: Authentication result
    
    Returns:
        List of matching runs
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        runs = _experiment_tracker.search_runs(
            experiment_name=request.experiment_name,
            filter_string=request.filter_string,
            order_by=request.order_by,
            limit=request.limit
        )
        
        return [
            RunResponse(
                run_id=str(run.run_id),
                experiment_id=str(run.experiment_id),
                run_name=run.run_name,
                status=run.status,
                start_time=run.start_time,
                end_time=run.end_time,
                params=run.params,
                tags=run.tags,
                git_commit=run.git_commit
            )
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Error searching runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search runs: {str(e)}"
        )


@router.post(
    "/runs/compare",
    summary="Compare multiple runs"
)
async def compare_runs(
    request: RunCompareRequest,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Compare metrics across multiple runs.
    
    Args:
        request: Comparison request
        auth: Authentication result
    
    Returns:
        Comparison results
    """
    try:
        if not _experiment_tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment tracker not initialized"
            )
        
        comparison_df = _experiment_tracker.compare_runs(
            run_ids=request.run_ids,
            metric_names=request.metric_names
        )
        
        # Convert DataFrame to dict for JSON response
        comparison_data = comparison_df.to_dict(orient='records')
        
        return {
            "runs": comparison_data,
            "metric_names": request.metric_names or list(comparison_df.columns)
        }
    except Exception as e:
        logger.error(f"Error comparing runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare runs: {str(e)}"
        )



# ============================================================================
# Data Versioning Endpoints
# ============================================================================

@router.post(
    "/datasets",
    response_model=DatasetVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new dataset"
)
async def register_dataset(
    request: DatasetRegister,
    auth: AuthResult = Depends(verify_authentication)
) -> DatasetVersionResponse:
    """
    Register a new dataset version.
    
    Args:
        request: Dataset registration request
        auth: Authentication result
    
    Returns:
        Dataset version details
    """
    try:
        if not _data_version_control:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data version control not initialized"
            )
        
        logger.info(f"Registering dataset '{request.dataset_name}' by user {auth.user_id}")
        
        # Convert data dict to DataFrame
        df = pd.DataFrame(request.data)
        
        # Register dataset
        dataset_version = _data_version_control.register_dataset(
            dataset=df,
            dataset_name=request.dataset_name,
            source=request.source,
            metadata=request.metadata
        )
        
        return DatasetVersionResponse(
            version_id=str(dataset_version.version_id),
            dataset_name=dataset_version.dataset_name,
            version=dataset_version.version,
            source=dataset_version.source,
            num_rows=dataset_version.num_rows,
            num_columns=dataset_version.num_columns,
            schema=dataset_version.schema,
            created_at=dataset_version.created_at
        )
    except Exception as e:
        logger.error(f"Error registering dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register dataset: {str(e)}"
        )


@router.get(
    "/datasets/{dataset_name}/versions",
    response_model=List[DatasetVersionResponse],
    summary="List dataset versions"
)
async def list_dataset_versions(
    dataset_name: str,
    auth: AuthResult = Depends(verify_authentication)
) -> List[DatasetVersionResponse]:
    """
    List all versions of a dataset.
    
    Args:
        dataset_name: Name of the dataset
        auth: Authentication result
    
    Returns:
        List of dataset versions
    """
    try:
        if not _data_version_control:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data version control not initialized"
            )
        
        versions = _data_version_control.list_versions(dataset_name)
        
        return [
            DatasetVersionResponse(
                version_id=str(v.version_id),
                dataset_name=v.dataset_name,
                version=v.version,
                source=v.source,
                num_rows=v.num_rows,
                num_columns=v.num_columns,
                schema=v.schema,
                created_at=v.created_at
            )
            for v in versions
        ]
    except Exception as e:
        logger.error(f"Error listing dataset versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list dataset versions: {str(e)}"
        )


@router.get(
    "/datasets/{version_id}/lineage",
    summary="Get dataset lineage"
)
async def get_dataset_lineage(
    version_id: str,
    direction: str = "upstream",
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Get dataset lineage (upstream or downstream).
    
    Args:
        version_id: Dataset version ID
        direction: Lineage direction ('upstream' or 'downstream')
        auth: Authentication result
    
    Returns:
        Dataset lineage information
    """
    try:
        if not _data_version_control:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data version control not initialized"
            )
        
        if direction not in ["upstream", "downstream"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direction must be 'upstream' or 'downstream'"
            )
        
        lineage = _data_version_control.get_lineage(version_id, direction)
        
        return {
            "version_id": version_id,
            "direction": direction,
            "lineage": [
                {
                    "version_id": str(v.version_id),
                    "dataset_name": v.dataset_name,
                    "version": v.version,
                    "created_at": v.created_at.isoformat()
                }
                for v in lineage
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset lineage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset lineage: {str(e)}"
        )


@router.post(
    "/datasets/drift",
    response_model=DriftReportResponse,
    summary="Check for data drift"
)
async def check_drift(
    request: DriftCheckRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> DriftReportResponse:
    """
    Check for data drift between two dataset versions.
    
    Args:
        request: Drift check request
        auth: Authentication result
    
    Returns:
        Drift report
    """
    try:
        if not _data_version_control:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data version control not initialized"
            )
        
        logger.info(
            f"Checking drift between {request.dataset_version_id1} and "
            f"{request.dataset_version_id2} by user {auth.user_id}"
        )
        
        drift_report = _data_version_control.detect_drift(
            dataset_version_id1=request.dataset_version_id1,
            dataset_version_id2=request.dataset_version_id2
        )
        
        return DriftReportResponse(
            version_id1=str(drift_report.version_id1),
            version_id2=str(drift_report.version_id2),
            drift_detected=drift_report.drift_detected,
            drift_score=drift_report.drift_score,
            feature_drifts=drift_report.feature_drifts,
            recommendations=drift_report.recommendations
        )
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check drift: {str(e)}"
        )


# ============================================================================
# Feature Store Endpoints
# ============================================================================

@router.post(
    "/features",
    response_model=FeatureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new feature"
)
async def register_feature(
    request: FeatureRegister,
    auth: AuthResult = Depends(verify_authentication)
) -> FeatureResponse:
    """
    Register a new feature in the feature store.
    
    Args:
        request: Feature registration request
        auth: Authentication result
    
    Returns:
        Feature details
    """
    try:
        if not _feature_store:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feature store not initialized"
            )
        
        logger.info(f"Registering feature '{request.feature_name}' by user {auth.user_id}")
        
        from src.ds.feature_store import FeatureDefinition
        
        # Create feature definition
        feature_def = FeatureDefinition(
            feature_name=request.feature_name,
            feature_type=request.feature_type,
            description=request.description,
            transformation_code=request.transformation_code,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            dependencies=request.dependencies or []
        )
        
        # Register feature
        _feature_store.register_feature(
            feature_name=request.feature_name,
            feature_definition=feature_def
        )
        
        # Get the registered feature
        registered_feature = _feature_store.get_feature(request.feature_name)
        
        return FeatureResponse(
            feature_id=str(registered_feature.feature_id),
            feature_name=registered_feature.feature_name,
            feature_type=registered_feature.feature_type,
            description=registered_feature.description,
            version=registered_feature.version,
            created_at=registered_feature.created_at
        )
    except Exception as e:
        logger.error(f"Error registering feature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register feature: {str(e)}"
        )


@router.get(
    "/features",
    response_model=List[FeatureResponse],
    summary="List all features"
)
async def list_features(
    auth: AuthResult = Depends(verify_authentication),
    limit: int = 100
) -> List[FeatureResponse]:
    """
    List all features in the feature store.
    
    Args:
        auth: Authentication result
        limit: Maximum number of features to return
    
    Returns:
        List of features
    """
    try:
        if not _feature_store:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feature store not initialized"
            )
        
        features = _feature_store.list_features(limit=limit)
        
        return [
            FeatureResponse(
                feature_id=str(f.feature_id),
                feature_name=f.feature_name,
                feature_type=f.feature_type,
                description=f.description,
                version=f.version,
                created_at=f.created_at
            )
            for f in features
        ]
    except Exception as e:
        logger.error(f"Error listing features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list features: {str(e)}"
        )


@router.get(
    "/features/{feature_name}",
    response_model=FeatureResponse,
    summary="Get feature by name"
)
async def get_feature(
    feature_name: str,
    auth: AuthResult = Depends(verify_authentication)
) -> FeatureResponse:
    """
    Get feature details by name.
    
    Args:
        feature_name: Name of the feature
        auth: Authentication result
    
    Returns:
        Feature details
    """
    try:
        if not _feature_store:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feature store not initialized"
            )
        
        feature = _feature_store.get_feature(feature_name)
        
        if not feature:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature '{feature_name}' not found"
            )
        
        return FeatureResponse(
            feature_id=str(feature.feature_id),
            feature_name=feature.feature_name,
            feature_type=feature.feature_type,
            description=feature.description,
            version=feature.version,
            created_at=feature.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature: {str(e)}"
        )


@router.post(
    "/features/compute",
    summary="Compute features"
)
async def compute_features(
    request: FeatureComputeRequest,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Compute features from input data.
    
    Args:
        request: Feature computation request
        auth: Authentication result
    
    Returns:
        Computed features
    """
    try:
        if not _feature_store:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feature store not initialized"
            )
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame(request.input_data)
        
        # Compute features
        features_df = _feature_store.compute_features(
            feature_names=request.feature_names,
            input_data=input_df
        )
        
        # Convert to dict for JSON response
        return {
            "features": features_df.to_dict(orient='records'),
            "feature_names": request.feature_names
        }
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute features: {str(e)}"
        )


@router.get(
    "/features/serve",
    summary="Serve features for online inference"
)
async def serve_features(
    feature_names: str,
    entity_ids: str,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Serve features for online inference.
    
    Args:
        feature_names: Comma-separated list of feature names
        entity_ids: Comma-separated list of entity IDs
        auth: Authentication result
    
    Returns:
        Feature values
    """
    try:
        if not _feature_store:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feature store not initialized"
            )
        
        # Parse comma-separated strings
        feature_list = [f.strip() for f in feature_names.split(',')]
        entity_list = [e.strip() for e in entity_ids.split(',')]
        
        # Get features
        features_df = _feature_store.get_features(
            feature_names=feature_list,
            entity_ids=entity_list,
            mode="online"
        )
        
        # Convert to dict for JSON response
        return {
            "features": features_df.to_dict(orient='records'),
            "feature_names": feature_list,
            "entity_ids": entity_list
        }
    except Exception as e:
        logger.error(f"Error serving features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve features: {str(e)}"
        )


# ============================================================================
# EDA and Reporting Endpoints
# ============================================================================

@router.post(
    "/eda/analyze",
    response_model=EDAReportResponse,
    summary="Run EDA analysis"
)
async def analyze_dataset(
    request: EDAAnalyzeRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> EDAReportResponse:
    """
    Run exploratory data analysis on a dataset.
    
    Args:
        request: EDA analysis request
        auth: Authentication result
    
    Returns:
        EDA report
    """
    try:
        if not _eda_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="EDA module not initialized"
            )
        
        logger.info(f"Running EDA analysis by user {auth.user_id}")
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Run EDA
        eda_report = _eda_module.analyze_dataset(
            data=df,
            target_column=request.target_column
        )
        
        # Generate report ID
        report_id = str(uuid4())
        
        return EDAReportResponse(
            report_id=report_id,
            dataset_name=request.dataset_name or "unnamed_dataset",
            num_rows=eda_report.num_rows,
            num_columns=eda_report.num_columns,
            summary_statistics=eda_report.summary_statistics,
            quality_issues=[
                {
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "affected_rows": issue.affected_rows
                }
                for issue in eda_report.quality_issues
            ],
            recommendations=eda_report.recommendations,
            generated_at=eda_report.generated_at
        )
    except Exception as e:
        logger.error(f"Error running EDA analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run EDA analysis: {str(e)}"
        )


@router.get(
    "/eda/reports/{report_id}",
    summary="Get EDA report"
)
async def get_eda_report(
    report_id: str,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Get a previously generated EDA report.
    
    Args:
        report_id: Report ID
        auth: Authentication result
    
    Returns:
        EDA report
    """
    try:
        # In a production system, this would retrieve from storage
        # For now, return a placeholder
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found. Reports are not persisted in this version."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting EDA report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get EDA report: {str(e)}"
        )


@router.post(
    "/model-cards/generate",
    response_model=ModelCardResponse,
    summary="Generate model card"
)
async def generate_model_card(
    request: ModelCardGenerateRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> ModelCardResponse:
    """
    Generate a model card for a trained model.
    
    Args:
        request: Model card generation request
        auth: Authentication result
    
    Returns:
        Model card
    """
    try:
        if not _model_card_generator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model card generator not initialized"
            )
        
        logger.info(f"Generating model card for {request.model_id} by user {auth.user_id}")
        
        # Generate model card
        model_card = _model_card_generator.generate_model_card(
            model_id=request.model_id,
            run_id=request.run_id,
            include_fairness=request.include_fairness
        )
        
        # Generate card ID
        card_id = str(uuid4())
        
        return ModelCardResponse(
            card_id=card_id,
            model_id=model_card.model_id,
            model_name=model_card.model_name,
            model_type=model_card.model_type,
            version=model_card.version,
            metrics=model_card.metrics,
            fairness_metrics=model_card.fairness_metrics,
            feature_importance=model_card.feature_importance,
            generated_at=model_card.date
        )
    except Exception as e:
        logger.error(f"Error generating model card: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate model card: {str(e)}"
        )


@router.get(
    "/model-cards/{card_id}",
    summary="Get model card"
)
async def get_model_card(
    card_id: str,
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Get a previously generated model card.
    
    Args:
        card_id: Model card ID
        auth: Authentication result
    
    Returns:
        Model card
    """
    try:
        # In a production system, this would retrieve from storage
        # For now, return a placeholder
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model card {card_id} not found. Cards are not persisted in this version."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model card: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model card: {str(e)}"
        )
