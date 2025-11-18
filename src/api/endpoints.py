"""Core API endpoints for MHRAS"""

import logging
import time
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import pandas as pd

from src.api.models import (
    ScreeningRequest,
    ScreeningResponse,
    RiskScore,
    RiskScoreResponse,
    ExplanationRequest,
    ExplanationResponse,
    ExplanationSummary,
    ResourceRecommendation,
    RiskLevel,
    ErrorResponse
)
from src.api.auth import Authenticator, AuthResult, authenticator
from src.ml.model_registry import ModelRegistry
from src.ml.inference_engine import InferenceEngine
from src.ml.ensemble_predictor import EnsemblePredictor
from src.ml.interpretability import InterpretabilityEngine
from src.ml.feature_pipeline import FeatureEngineeringPipeline
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
from src.ingestion.validation import DataValidator
from src.governance.consent import ConsentVerifier
from src.governance.anonymization import Anonymizer
from src.recommendations.recommendation_engine import RecommendationEngine, IndividualProfile
from src.exceptions import (
    ValidationError,
    ConsentError,
    InferenceError,
    InterpretabilityError
)

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Initialize app
app = FastAPI(
    title="Mental Health Risk Assessment System API",
    description="API for mental health risk screening and prediction",
    version="1.0.0"
)


# Dependency for authentication (optional - disabled for development)
async def verify_authentication(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> AuthResult:
    """
    Verify JWT token authentication (optional for development).
    
    Args:
        credentials: HTTP authorization credentials (optional)
    
    Returns:
        AuthResult with user information
    """
    # For development: allow requests without authentication
    if credentials is None:
        logger.debug("No authentication provided - using development mode")
        return AuthResult(
            authenticated=True,
            user_id="dev_user",
            role="admin"
        )
    
    # If token is provided, verify it
    token = credentials.credentials
    auth_result = authenticator.verify_token(token)
    
    if not auth_result.authenticated:
        # In development mode, log warning but allow access
        logger.warning(f"Token verification failed: {auth_result.error}")
        return AuthResult(
            authenticated=True,
            user_id="dev_user",
            role="admin"
        )
    
    return auth_result


# Global components (in production, use dependency injection)
_model_registry: Optional[ModelRegistry] = None
_inference_engine: Optional[InferenceEngine] = None
_ensemble_predictor: Optional[EnsemblePredictor] = None
_interpretability_engine: Optional[InterpretabilityEngine] = None
_feature_pipeline: Optional[FeatureEngineeringPipeline] = None
_etl_pipeline: Optional[ETLPipeline] = None
_data_validator: Optional[DataValidator] = None
_consent_verifier: Optional[ConsentVerifier] = None
_anonymizer: Optional[Anonymizer] = None
_recommendation_engine: Optional[RecommendationEngine] = None
_audit_logger: Optional[Any] = None
_human_review_queue: Optional[Any] = None
_drift_monitor: Optional[Any] = None


def initialize_components():
    """Initialize all ML and processing components"""
    global _model_registry, _inference_engine, _ensemble_predictor
    global _interpretability_engine, _feature_pipeline, _etl_pipeline
    global _data_validator, _consent_verifier, _anonymizer
    global _recommendation_engine, _audit_logger, _human_review_queue, _drift_monitor
    
    logger.info("Initializing API components...")
    
    # Import governance components
    from src.governance.audit_logger import AuditLogger
    from src.governance.human_review_queue import HumanReviewQueue
    from src.governance.drift_monitor import DriftMonitor
    
    # Initialize ML and processing components
    _model_registry = ModelRegistry()
    _inference_engine = InferenceEngine(_model_registry)
    _ensemble_predictor = EnsemblePredictor(_model_registry, _inference_engine)
    _interpretability_engine = InterpretabilityEngine(_model_registry)
    _feature_pipeline = FeatureEngineeringPipeline()
    etl_config = ETLPipelineConfig()  # Use default configuration
    _etl_pipeline = ETLPipeline(etl_config)
    _data_validator = DataValidator()
    _consent_verifier = ConsentVerifier()
    _anonymizer = Anonymizer()
    
    # Initialize recommendation and governance components
    _recommendation_engine = RecommendationEngine()
    _audit_logger = AuditLogger()
    _human_review_queue = HumanReviewQueue()
    _drift_monitor = DriftMonitor()
    
    logger.info("API components initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Mental Health Risk Assessment System",
        "version": "1.0.0",
        "status": "operational",
        "authentication": "optional (development mode)",
        "docs": "/docs"
    }


@app.post("/auth/token")
async def generate_token(user_id: str, role: str = "user"):
    """
    Generate a JWT token for testing/development.
    
    Args:
        user_id: User identifier
        role: User role (default: user)
    
    Returns:
        JWT token
    
    Note: In production, this should be protected and use proper authentication
    """
    token = authenticator.generate_token(user_id=user_id, role=role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "role": role
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.post(
    "/screen",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        403: {"model": ErrorResponse, "description": "Consent error"},
        500: {"model": ErrorResponse, "description": "Processing error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        504: {"model": ErrorResponse, "description": "Timeout"}
    }
)
async def screen_individual(
    request: ScreeningRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> ScreeningResponse:
    """
    Screen an individual and generate risk score with recommendations.
    
    This endpoint:
    1. Validates input data
    2. Verifies consent
    3. Processes data through ETL pipeline
    4. Engineers features
    5. Generates ensemble predictions
    6. Provides interpretable explanations
    7. Recommends resources
    8. Triggers alerts if needed
    
    Args:
        request: Screening request with individual data
        auth: Authentication result
    
    Returns:
        ScreeningResponse with risk score, recommendations, and explanations
    
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Screening request received for {request.anonymized_id} "
            f"by user {auth.user_id}"
        )
        
        # 1. Verify consent
        if not request.consent_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Consent not verified"
            )
        
        # Verify consent in database
        try:
            data_types = []
            if request.survey_data:
                data_types.append("survey")
            if request.wearable_data:
                data_types.append("wearable")
            if request.emr_data:
                data_types.append("emr")
            
            consent_status = _consent_verifier.verify_consent(
                request.anonymized_id,
                data_types
            )
            
            if not consent_status.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Consent verification failed: {consent_status.reason}"
                )
        except ConsentError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Consent error: {str(e)}"
            )
        
        # 2. Validate data
        try:
            if request.survey_data:
                _data_validator.validate_survey(request.survey_data)
            if request.wearable_data:
                _data_validator.validate_wearable(request.wearable_data)
            if request.emr_data:
                _data_validator.validate_emr(request.emr_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation error: {str(e)}"
            )
        
        # 3. Combine data into DataFrame
        # In production, this would be more sophisticated
        combined_data = {}
        if request.survey_data:
            combined_data.update(request.survey_data)
        if request.wearable_data:
            combined_data.update(request.wearable_data)
        if request.emr_data:
            combined_data.update(request.emr_data)
        
        df = pd.DataFrame([combined_data])
        
        # 4. Process through ETL pipeline
        try:
            processed_data = _etl_pipeline.process(df)
        except Exception as e:
            logger.error(f"ETL processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data processing error: {str(e)}"
            )
        
        # 5. Engineer features
        try:
            features = _feature_pipeline.engineer_features(processed_data)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature engineering error: {str(e)}"
            )
        
        # 6. Generate ensemble predictions
        try:
            prediction_result = _ensemble_predictor.predict_with_ensemble(
                features=features,
                individual_ids=[request.anonymized_id]
            )
            
            risk_score_value = float(prediction_result['risk_scores'][0])
            confidence = float(prediction_result['confidence'][0])
            risk_level_str = prediction_result['risk_levels'][0]
            alert_triggered = prediction_result['alerts_triggered'][0]
            
        except InferenceError as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model inference error: {str(e)}"
            )
        
        # 7. Generate explanations
        try:
            explanation_result = _interpretability_engine.generate_explanation(
                model_id=prediction_result['model_ids'][0],  # Use first model for explanation
                features=features,
                include_shap=True,
                include_counterfactuals=True,
                include_rules=False,  # Skip rules for speed
                timeout_seconds=3.0
            )
            
            # Extract top features for contributing factors
            contributing_factors = []
            if explanation_result['components']['shap']:
                shap_data = explanation_result['components']['shap']
                for feature in shap_data['top_features'][:5]:
                    contributing_factors.append(feature['clinical_name'])
            
            # Format counterfactual
            counterfactual_text = ""
            if explanation_result['components']['counterfactuals']:
                cf_data = explanation_result['components']['counterfactuals'][0]
                counterfactual_text = cf_data.get('description', '')
            
            explanations = ExplanationSummary(
                top_features=[(f['feature'], f['mean_shap_value']) 
                             for f in explanation_result['components']['shap']['top_features']]
                             if explanation_result['components']['shap'] else [],
                counterfactual=counterfactual_text,
                rule_approximation="",  # Not generated for speed
                clinical_interpretation=explanation_result.get('clinical_summary', '')
            )
            
        except InterpretabilityError as e:
            logger.warning(f"Interpretability failed: {e}")
            # Continue with empty explanations
            contributing_factors = []
            explanations = ExplanationSummary(
                top_features=[],
                counterfactual="Explanation generation failed",
                rule_approximation="",
                clinical_interpretation=""
            )
        
        # 8. Create risk score object
        risk_score = RiskScore(
            anonymized_id=request.anonymized_id,
            score=risk_score_value,
            risk_level=RiskLevel(risk_level_str),
            confidence=confidence,
            contributing_factors=contributing_factors,
            timestamp=request.timestamp
        )
        
        # 9. Generate resource recommendations
        recommendations = _generate_recommendations(
            risk_level_str,
            contributing_factors,
            anonymized_id=request.anonymized_id
        )
        
        # 10. Determine if human review is required
        requires_human_review = risk_score_value > 75 or confidence < 0.6
        
        # 11. Enqueue for human review if needed
        if requires_human_review and _human_review_queue:
            try:
                case_id = _human_review_queue.enqueue_case(
                    anonymized_id=request.anonymized_id,
                    risk_score=risk_score_value,
                    risk_level=risk_level_str,
                    prediction_data=prediction_result,
                    features=features.to_dict()
                )
                logger.info(f"Enqueued case {case_id} for human review")
            except Exception as e:
                logger.error(f"Failed to enqueue for human review: {e}")
        
        # 12. Log audit trail
        if _audit_logger:
            try:
                _audit_logger.log_screening_request(
                    request=request.dict(),
                    response={
                        "risk_score": {"score": risk_score_value, "risk_level": risk_level_str},
                        "alert_triggered": alert_triggered,
                        "requires_human_review": requires_human_review
                    },
                    anonymized_id=request.anonymized_id,
                    user_id=auth.user_id
                )
            except Exception as e:
                logger.error(f"Failed to log audit trail: {e}")
        
        # Check timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > 5.0:
            logger.warning(
                f"Screening request exceeded 5s timeout: {elapsed_time:.3f}s"
            )
        
        logger.info(
            f"Screening completed in {elapsed_time:.3f}s. "
            f"Risk: {risk_level_str} ({risk_score_value:.2f}), "
            f"Alert: {alert_triggered}, Review: {requires_human_review}"
        )
        
        # 13. Return response
        return ScreeningResponse(
            risk_score=risk_score,
            recommendations=recommendations,
            explanations=explanations,
            requires_human_review=requires_human_review,
            alert_triggered=alert_triggered
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in screening: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/risk-score/{anonymized_id}",
    response_model=RiskScoreResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"},
        404: {"model": ErrorResponse, "description": "Risk score not found"}
    }
)
async def get_risk_score(
    anonymized_id: str,
    auth: AuthResult = Depends(verify_authentication)
) -> RiskScoreResponse:
    """
    Retrieve the most recent risk score for an individual.
    
    Args:
        anonymized_id: Anonymized identifier
        auth: Authentication result
    
    Returns:
        RiskScoreResponse with risk score if found
    
    Raises:
        HTTPException: If risk score not found
    """
    try:
        logger.info(
            f"Risk score retrieval requested for {anonymized_id} "
            f"by user {auth.user_id}"
        )
        
        # In production, this would query a database
        # For now, return a placeholder response
        
        # Simulate database lookup
        # TODO: Implement actual database query
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No risk score found for {anonymized_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving risk score: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        404: {"model": ErrorResponse, "description": "Prediction not found"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def explain_prediction(
    request: ExplanationRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> ExplanationResponse:
    """
    Generate explanation for a prediction.
    
    Args:
        request: Explanation request
        auth: Authentication result
    
    Returns:
        ExplanationResponse with model explanations
    
    Raises:
        HTTPException: If prediction not found or explanation fails
    """
    try:
        logger.info(
            f"Explanation requested for {request.anonymized_id} "
            f"by user {auth.user_id}"
        )
        
        # In production, this would:
        # 1. Retrieve the prediction and features from database
        # 2. Generate explanations using InterpretabilityEngine
        # 3. Return formatted explanations
        
        # For now, return a placeholder
        # TODO: Implement actual explanation retrieval
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction found for {request.anonymized_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/review-queue",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"}
    }
)
async def get_review_queue(
    auth: AuthResult = Depends(verify_authentication),
    limit: int = 50
):
    """
    Get pending cases in human review queue.
    
    Args:
        auth: Authentication result
        limit: Maximum number of cases to return
    
    Returns:
        List of pending review cases
    """
    try:
        if not _human_review_queue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Human review queue not initialized"
            )
        
        pending_cases = _human_review_queue.get_pending_cases(limit=limit)
        
        return {
            "cases": [case.to_dict() for case in pending_cases],
            "total": len(pending_cases)
        }
    except Exception as e:
        logger.error(f"Error retrieving review queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/statistics",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"}
    }
)
async def get_statistics(
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Get system statistics and metrics.
    
    Args:
        auth: Authentication result
    
    Returns:
        System statistics
    """
    try:
        stats = {
            "timestamp": time.time()
        }
        
        # Review queue statistics
        if _human_review_queue:
            stats["review_queue"] = _human_review_queue.get_queue_statistics()
        
        # Model registry statistics
        if _model_registry:
            active_models = _model_registry.get_active_models()
            all_models = _model_registry.list_models()
            stats["models"] = {
                "active_count": len(active_models),
                "total_count": len(all_models)
            }
        
        return stats
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/drift-check",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"}
    }
)
async def check_drift(
    auth: AuthResult = Depends(verify_authentication)
):
    """
    Check for data and prediction drift.
    
    Args:
        auth: Authentication result
    
    Returns:
        Drift detection results
    """
    try:
        if not _drift_monitor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Drift monitor not initialized"
            )
        
        # This is a placeholder - in production, you would:
        # 1. Fetch recent data from database
        # 2. Run drift detection
        # 3. Return results
        
        return {
            "message": "Drift monitoring is active",
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


def _generate_recommendations(
    risk_level: str,
    contributing_factors: list,
    anonymized_id: str = "unknown",
    profile_data: dict = None
) -> list[ResourceRecommendation]:
    """
    Generate resource recommendations using the recommendation engine.
    
    Args:
        risk_level: Risk level (low, moderate, high, critical)
        contributing_factors: List of contributing factors
        anonymized_id: Anonymized identifier
        profile_data: Optional profile data for personalization
    
    Returns:
        List of resource recommendations
    """
    # Use global recommendation engine
    engine = _recommendation_engine or RecommendationEngine()
    
    # Create individual profile
    profile = IndividualProfile(
        anonymized_id=anonymized_id,
        risk_level=risk_level,
        contributing_factors=contributing_factors or []
    )
    
    # Add profile data if provided
    if profile_data:
        if 'age_group' in profile_data:
            profile.age_group = profile_data['age_group']
        if 'has_therapy_history' in profile_data:
            profile.has_therapy_history = profile_data['has_therapy_history']
        if 'has_medication_history' in profile_data:
            profile.has_medication_history = profile_data['has_medication_history']
        if 'has_support_system' in profile_data:
            profile.has_support_system = profile_data['has_support_system']
        if 'prefers_online' in profile_data:
            profile.prefers_online = profile_data['prefers_online']
        if 'prefers_group' in profile_data:
            profile.prefers_group = profile_data['prefers_group']
        if 'specific_conditions' in profile_data:
            profile.specific_conditions = profile_data['specific_conditions']
    
    # Get recommendations from engine
    resources = engine.get_recommendations(profile, max_recommendations=5)
    
    # Convert to API models
    recommendations = [
        ResourceRecommendation(
            resource_type=resource.resource_type.value,
            name=resource.name,
            description=resource.description,
            contact_info=resource.contact_info,
            urgency=resource.urgency.value,
            eligibility_criteria=resource.eligibility_criteria
        )
        for resource in resources
    ]
    
    return recommendations
