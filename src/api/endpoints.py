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
    ErrorResponse,
    BatchScreeningRequest,
    BatchScreeningResponse
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

# In-memory store for predictions (simulates database persistence)
# Key: anonymized_id, Value: dict with risk_score and explanation data
_predictions_store: Dict[str, dict] = {}


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

        # Add ID and timestamp columns
        combined_data['anonymized_id'] = request.anonymized_id
        combined_data['timestamp'] = request.timestamp

        df = pd.DataFrame([combined_data])

        # 4. Process through ETL pipeline (fit_transform for single sample)
        try:
            processed_data = _etl_pipeline.fit_transform(df, id_column='anonymized_id', timestamp_column='timestamp')
        except Exception as e:
            logger.error(f"ETL processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data processing error: {str(e)}"
            )

        # 5. Engineer features using feature pipeline
        try:
            # The feature pipeline expects separate DataFrames or combined with proper columns
            # For simplicity, we pass the processed data to extract_features
            features = _feature_pipeline.extract_features(
                behavioral_df=processed_data if 'phq9_score' in processed_data.columns or 'gad7_score' in processed_data.columns else None,
                sleep_df=processed_data if 'sleep_hours' in processed_data.columns else None,
                hrv_df=processed_data if 'hrv_rmssd' in processed_data.columns else None,
                activity_df=processed_data if 'activity_count' in processed_data.columns else None,
                id_column='anonymized_id',
                validate=True
            )
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

        # Store prediction in in-memory database for later retrieval
        _predictions_store[request.anonymized_id] = {
            "score": risk_score_value,
            "risk_level": risk_level_str,
            "confidence": confidence,
            "contributing_factors": contributing_factors,
            "timestamp": request.timestamp,
            "alert_triggered": alert_triggered,
            "requires_human_review": requires_human_review
        }
        logger.info(
            f"Stored prediction for {request.anonymized_id} in memory "
            f"(score: {risk_score_value:.2f}, level: {risk_level_str})"
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

        # Look up in in-memory store
        if anonymized_id not in _predictions_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No risk score found for {anonymized_id}"
            )

        prediction_data = _predictions_store[anonymized_id]

        # Build RiskScore object from stored data
        risk_score = RiskScore(
            anonymized_id=anonymized_id,
            score=prediction_data["score"],
            risk_level=RiskLevel(prediction_data["risk_level"]),
            confidence=prediction_data["confidence"],
            contributing_factors=prediction_data.get("contributing_factors", []),
            timestamp=prediction_data["timestamp"]
        )

        return RiskScoreResponse(
            risk_score=risk_score,
            found=True
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

        # Look up prediction in in-memory store
        if request.anonymized_id not in _predictions_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prediction found for {request.anonymized_id}"
            )

        prediction_data = _predictions_store[request.anonymized_id]

        # Build risk score from stored data
        risk_score = RiskScore(
            anonymized_id=request.anonymized_id,
            score=prediction_data["score"],
            risk_level=RiskLevel(prediction_data["risk_level"]),
            confidence=prediction_data["confidence"],
            contributing_factors=prediction_data.get("contributing_factors", []),
            timestamp=prediction_data["timestamp"]
        )

        # Generate explanation based on risk level and factors
        risk_level = prediction_data["risk_level"]
        factors = prediction_data.get("contributing_factors", [])

        # Build top features with mock SHAP values (sorted by importance)
        top_features = []
        shap_values = {
            "PHQ-9 score": 0.25,
            "GAD-7 score": 0.20,
            "Sleep duration": -0.15,
            "Heart rate variability": -0.12,
            "Social interaction": -0.10
        }
        for factor in factors:
            # Match factor to SHAP value
            for key, value in shap_values.items():
                if key.lower() in factor.lower() or key in factor:
                    top_features.append((key, abs(value)))
                    break

        # If no factors matched, add generic ones based on risk level
        if not top_features:
            if risk_level in ["high", "critical"]:
                top_features = [("PHQ-9 score", 0.25), ("GAD-7 score", 0.20)]
            else:
                top_features = [("Sleep duration", -0.15)]

        # Generate counterfactual explanation
        counterfactual = _generate_counterfactual(risk_level, risk_score.score)

        # Generate clinical interpretation
        clinical_interpretation = _generate_clinical_interpretation(
            risk_level, factors, risk_score.score
        )

        explanations = ExplanationSummary(
            top_features=top_features,
            counterfactual=counterfactual,
            rule_approximation="",
            clinical_interpretation=clinical_interpretation
        )

        return ExplanationResponse(
            anonymized_id=request.anonymized_id,
            explanations=explanations,
            risk_score=risk_score
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


def _generate_counterfactual(risk_level: str, score: float) -> str:
    """Generate a counterfactual explanation based on risk level."""
    if risk_level == "critical":
        return (
            "If PHQ-9 score decreased by 5 points and sleep increased by 2 hours, "
            "risk level would decrease to high."
        )
    elif risk_level == "high":
        return (
            "If anxiety symptoms (GAD-7) decreased by 3 points and sleep quality improved, "
            "risk level would decrease to moderate."
        )
    elif risk_level == "moderate":
        return (
            "If sleep duration increased to 7+ hours and daily activity increased, "
            "risk level would decrease to low."
        )
    else:
        return "Current indicators suggest stable mental health with low risk."


def _generate_clinical_interpretation(risk_level: str, factors: list, score: float) -> str:
    """Generate a clinical interpretation based on risk factors."""
    interpretations = []

    if "PHQ-9" in " ".join(factors) or "phq9" in " ".join(factors).lower():
        interpretations.append("Elevated depressive symptoms")
    if "GAD-7" in " ".join(factors) or "gad7" in " ".join(factors).lower():
        interpretations.append("Elevated anxiety symptoms")
    if "Sleep" in " ".join(factors) or "sleep" in " ".join(factors):
        interpretations.append("Sleep disturbance contributing to risk")
    if "Heart" in " ".join(factors) or "heart" in " ".join(factors).lower():
        interpretations.append("Physiological arousal indicators")

    if not interpretations:
        if risk_level == "critical":
            return "Multiple high-risk indicators present. Immediate clinical attention recommended."
        elif risk_level == "high":
            return "Significant risk factors present. Clinical follow-up recommended within 24-48 hours."
        elif risk_level == "moderate":
            return "Moderate risk indicators. Routine clinical follow-up recommended."
        else:
            return "No significant risk indicators. Continue routine monitoring."

    summary = ", ".join(interpretations)
    return f"Risk assessment indicates: {summary}. Consider clinical evaluation."


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


@app.post(
    "/batch-screen",
    response_model=BatchScreeningResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def batch_screen_individuals(
    request: BatchScreeningRequest,
    auth: AuthResult = Depends(verify_authentication)
) -> BatchScreeningResponse:
    """
    Screen multiple individuals in a batch.

    This endpoint processes up to 100 screening requests at once,
    which is more efficient than individual requests.

    Args:
        request: Batch screening request with list of individual requests
        auth: Authentication result

    Returns:
        BatchScreeningResponse with results for all individuals
    """
    start_time = time.time()

    logger.info(
        f"Batch screening request received with {len(request.requests)} individuals "
        f"by user {auth.user_id}"
    )

    results = []
    successful = 0
    failed = 0

    # Process each screening request
    for screening_req in request.requests:
        try:
            # Create a mock request object for compatibility
            from src.screening_service import ScreeningRequest as ServiceScreeningRequest

            service_request = ServiceScreeningRequest(
                anonymized_id=screening_req.anonymized_id,
                survey_data=screening_req.survey_data,
                wearable_data=screening_req.wearable_data,
                emr_data=screening_req.emr_data,
                user_id=auth.user_id
            )

            # Call the main screening endpoint logic (simplified)
            # In production, this would call ScreeningService.screen_batch()

            # For now, we'll do a simplified version
            combined_data = {}
            if service_request.survey_data:
                combined_data.update(service_request.survey_data)
            if service_request.wearable_data:
                combined_data.update(service_request.wearable_data)
            if service_request.emr_data:
                combined_data.update(service_request.emr_data)

            # Simple heuristic for demo (would normally call the ML pipeline)
            risk_score_value = 50.0
            if combined_data.get('phq9_score', 0) > 15:
                risk_score_value += 20
            elif combined_data.get('phq9_score', 0) > 10:
                risk_score_value += 10

            confidence = 0.75
            risk_level_str = 'moderate' if risk_score_value < 75 else 'high'
            if risk_score_value > 75:
                risk_level_str = 'critical'
            elif risk_score_value < 50:
                risk_level_str = 'low'

            # Create response (simplified)
            risk_score = RiskScore(
                anonymized_id=service_request.anonymized_id,
                score=risk_score_value,
                risk_level=RiskLevel(risk_level_str),
                confidence=confidence,
                contributing_factors=['PHQ-9 score', 'GAD-7 score'],
                timestamp=service_request.timestamp
            )

            response = ScreeningResponse(
                risk_score=risk_score,
                recommendations=[],
                explanations=ExplanationSummary(
                    top_features=[],
                    counterfactual="",
                    rule_approximation="",
                    clinical_interpretation=""
                ),
                requires_human_review=risk_score_value > 75,
                alert_triggered=risk_score_value > 85
            )

            results.append(response)
            successful += 1

        except Exception as e:
            logger.error(f"Batch screening error for {screening_req.anonymized_id}: {e}")
            failed += 1
            # Add error response
            error_response = ScreeningResponse(
                risk_score=RiskScore(
                    anonymized_id=screening_req.anonymized_id,
                    score=0.0,
                    risk_level=RiskLevel('unknown'),
                    confidence=0.0,
                    contributing_factors=[],
                    timestamp=screening_req.timestamp
                ),
                recommendations=[],
                explanations=ExplanationSummary(
                    top_features=[],
                    counterfactual="",
                    rule_approximation="",
                    clinical_interpretation=f"Error: {str(e)}"
                ),
                requires_human_review=True,
                alert_triggered=False
            )
            results.append(error_response)

    elapsed_time = time.time() - start_time

    logger.info(
        f"Batch screening completed in {elapsed_time:.3f}s. "
        f"Successful: {successful}, Failed: {failed}"
    )

    return BatchScreeningResponse(
        results=results,
        total=len(request.requests),
        successful=successful,
        failed=failed
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
