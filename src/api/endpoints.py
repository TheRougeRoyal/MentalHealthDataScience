"""Core API endpoints for MHRAS"""

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import (
    Explanation as ExplanationModel,
    Review as ReviewModel,
    Screening as ScreeningModel,
)

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
from src.api.auth import AuthResult, get_current_user, require_role
from src.risk_model import get_risk_model
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
security_dep = get_current_user

# Initialize app
app = FastAPI(
    title="Mental Health Risk Assessment System API",
    description="API for mental health risk screening and prediction",
    version="1.0.0"
)


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
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ScreeningResponse:
    """
    Screen an individual and generate risk score with recommendations.

    Pipeline: validate → consent → ETL → features → predict → explain →
    persist → recommend → review-gate → audit → respond.
    """
    start_time = time.time()

    try:
        logger.info(
            "Screening request received for %s by user %s",
            request.anonymized_id, auth.user_id,
        )

        # ── 1. Consent ──────────────────────────────────────────────────
        if not request.consent_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Consent not verified",
            )

        try:
            data_types: list[str] = []
            if request.survey_data:
                data_types.append("survey")
            if request.wearable_data:
                data_types.append("wearable")
            if request.emr_data:
                data_types.append("emr")

            consent_status = _consent_verifier.verify_consent(
                request.anonymized_id, data_types,
            )
            if not consent_status.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Consent verification failed: {consent_status.reason}",
                )
        except ConsentError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Consent error: {e}",
            )

        # ── 2. Validate ─────────────────────────────────────────────────
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
                detail=f"Validation error: {e}",
            )

        # ── 3. Combine raw input ────────────────────────────────────────
        combined_data: Dict[str, Any] = {}
        if request.survey_data:
            combined_data.update(request.survey_data)
        if request.wearable_data:
            combined_data.update(request.wearable_data)
        if request.emr_data:
            combined_data.update(request.emr_data)

        combined_data["anonymized_id"] = request.anonymized_id
        combined_data["timestamp"] = str(request.timestamp)

        df = pd.DataFrame([combined_data])

        # ── 4. ETL ──────────────────────────────────────────────────────
        try:
            processed_data = _etl_pipeline.fit_transform(
                df, id_column="anonymized_id", timestamp_column="timestamp",
            )
        except Exception as e:
            logger.error("ETL processing failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data processing error: {e}",
            )

        # ── 5. Feature engineering ──────────────────────────────────────
        try:
            features = _feature_pipeline.extract_features(
                behavioral_df=(
                    processed_data
                    if "phq9_score" in processed_data.columns
                    or "gad7_score" in processed_data.columns
                    else None
                ),
                sleep_df=(
                    processed_data
                    if "sleep_hours" in processed_data.columns
                    else None
                ),
                hrv_df=(
                    processed_data
                    if "hrv_rmssd" in processed_data.columns
                    else None
                ),
                activity_df=(
                    processed_data
                    if "activity_count" in processed_data.columns
                    else None
                ),
                id_column="anonymized_id",
                validate=True,
            )
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature engineering error: {e}",
            )

        # ── 6. Structured model assessment ────────────────────────────
        model = get_risk_model()
        assessment = model.assess(combined_data)

        risk_score_value = assessment.risk_score
        confidence = assessment.confidence
        risk_level_str = assessment.risk_level
        alert_triggered = assessment.alert_triggered
        contributing_factors = assessment.contributing_factors
        top_features_list = assessment.top_features
        counterfactual_text = assessment.counterfactual
        clinical_summary = assessment.clinical_interpretation

        explanations = ExplanationSummary(
            top_features=top_features_list,
            counterfactual=counterfactual_text,
            rule_approximation="",
            clinical_interpretation=clinical_summary,
        )

        # ── 8. Persist to database ──────────────────────────────────────
        requires_human_review = assessment.requires_human_review

        try:
            screening_row = ScreeningModel(
                anonymized_id=request.anonymized_id,
                risk_score=risk_score_value,
                risk_level=risk_level_str,
                input_data=combined_data,
            )
            db.add(screening_row)
            db.flush()  # materialise screening_row.id

            explanation_row = ExplanationModel(
                screening_id=screening_row.id,
                explanation_text=clinical_summary,
                factors={
                    "contributing_factors": contributing_factors,
                    "confidence": confidence,
                    "top_features": [
                        {"name": name, "value": value}
                        for name, value in top_features_list
                    ],
                    "counterfactual": counterfactual_text,
                },
            )
            db.add(explanation_row)

            if requires_human_review:
                review_row = ReviewModel(
                    screening_id=screening_row.id,
                    status="pending",
                )
                db.add(review_row)

            db.commit()
            logger.info(
                "Persisted screening %s for %s (score=%.2f, level=%s)",
                screening_row.id, request.anonymized_id,
                risk_score_value, risk_level_str,
            )
        except Exception as e:
            db.rollback()
            logger.error("Database write failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist screening result",
            )

        # ── 9. Risk score response object ───────────────────────────────
        risk_score = RiskScore(
            anonymized_id=request.anonymized_id,
            score=risk_score_value,
            risk_level=RiskLevel(risk_level_str),
            confidence=confidence,
            contributing_factors=contributing_factors,
            timestamp=request.timestamp,
        )

        # ── 10. Recommendations ─────────────────────────────────────────
        recommendations = _generate_recommendations(
            risk_level_str,
            contributing_factors,
            anonymized_id=request.anonymized_id,
        )

        # ── 11. Human-review queue ──────────────────────────────────────
        if requires_human_review and _human_review_queue:
            try:
                case_id = _human_review_queue.enqueue_case(
                    anonymized_id=request.anonymized_id,
                    risk_score=risk_score_value,
                    risk_level=risk_level_str,
                    prediction_data=prediction_result,
                    features=features.to_dict(),
                )
                logger.info("Enqueued case %s for human review", case_id)
            except Exception as e:
                logger.error("Failed to enqueue for human review: %s", e)

        # ── 12. Audit ───────────────────────────────────────────────────
        if _audit_logger:
            try:
                _audit_logger.log_screening_request(
                    request=request.dict(),
                    response={
                        "screening_id": screening_row.id,
                        "risk_score": risk_score_value,
                        "risk_level": risk_level_str,
                        "alert_triggered": alert_triggered,
                        "requires_human_review": requires_human_review,
                    },
                    anonymized_id=request.anonymized_id,
                    user_id=auth.user_id,
                )
            except Exception as e:
                logger.error("Failed to log audit trail: %s", e)

        # ── 13. Response ────────────────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > 5.0:
            logger.warning(
                "Screening request exceeded 5 s timeout: %.3f s", elapsed,
            )

        logger.info(
            "Screening completed in %.3f s — risk=%s (%.2f), "
            "alert=%s, review=%s",
            elapsed, risk_level_str, risk_score_value,
            alert_triggered, requires_human_review,
        )

        return ScreeningResponse(
            risk_score=risk_score,
            recommendations=recommendations,
            explanations=explanations,
            requires_human_review=requires_human_review,
            alert_triggered=alert_triggered,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in screening: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )


# ---------------------------------------------------------------------------
# GET /risk-score/{anonymized_id}
# ---------------------------------------------------------------------------


@app.get(
    "/risk-score/{anonymized_id}",
    response_model=RiskScoreResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"},
        404: {"model": ErrorResponse, "description": "Risk score not found"},
    },
)
async def get_risk_score(
    anonymized_id: str,
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RiskScoreResponse:
    """
    Retrieve the most recent risk score for an individual.

    Queries the *screenings* table (most-recent first) and enriches the
    response with contributing factors stored in the linked *explanations*
    row.
    """
    try:
        logger.info(
            "Risk score retrieval for %s by user %s",
            anonymized_id, auth.user_id,
        )

        screening: Optional[ScreeningModel] = (
            db.query(ScreeningModel)
            .filter(ScreeningModel.anonymized_id == anonymized_id)
            .order_by(ScreeningModel.created_at.desc())
            .first()
        )

        if screening is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No risk score found for {anonymized_id}",
            )

        # Pull contributing factors + confidence from the explanation row.
        contributing_factors: list[str] = []
        confidence: float = 0.0

        explanation: Optional[ExplanationModel] = (
            db.query(ExplanationModel)
            .filter(ExplanationModel.screening_id == screening.id)
            .order_by(ExplanationModel.created_at.desc())
            .first()
        )

        if explanation and isinstance(explanation.factors, dict):
            contributing_factors = explanation.factors.get(
                "contributing_factors", [],
            )
            confidence = float(
                explanation.factors.get("confidence", 0.0),
            )

        risk_score = RiskScore(
            anonymized_id=screening.anonymized_id,
            score=screening.risk_score,
            risk_level=RiskLevel(screening.risk_level),
            confidence=confidence,
            contributing_factors=contributing_factors,
            timestamp=screening.created_at.isoformat(),
        )

        return RiskScoreResponse(risk_score=risk_score, found=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving risk score: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        404: {"model": ErrorResponse, "description": "Prediction not found"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
)
async def explain_prediction(
    request: ExplanationRequest,
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ExplanationResponse:
    """
    Return the explanation for a screening.

    If a stored explanation exists it is returned directly.  When
    ``prediction_id`` is supplied the specific screening is used;
    otherwise the most recent screening for the given
    ``anonymized_id`` is looked up.
    """
    try:
        logger.info(
            "Explanation requested for %s by user %s",
            request.anonymized_id, auth.user_id,
        )

        # ── Resolve screening row ───────────────────────────────────────
        if request.prediction_id:
            screening: Optional[ScreeningModel] = (
                db.query(ScreeningModel)
                .filter(ScreeningModel.id == request.prediction_id)
                .first()
            )
        else:
            screening = (
                db.query(ScreeningModel)
                .filter(
                    ScreeningModel.anonymized_id == request.anonymized_id,
                )
                .order_by(ScreeningModel.created_at.desc())
                .first()
            )

        if screening is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No screening found for {request.anonymized_id}",
            )

        # ── Fetch or create explanation ─────────────────────────────────
        explanation_row: Optional[ExplanationModel] = (
            db.query(ExplanationModel)
            .filter(ExplanationModel.screening_id == screening.id)
            .order_by(ExplanationModel.created_at.desc())
            .first()
        )

        contributing_factors: list[str] = []
        top_features: list[tuple[str, float]] = []
        counterfactual_text = ""
        clinical_text = ""
        confidence: float = 0.0

        if explanation_row and isinstance(explanation_row.factors, dict):
            # ── Use stored explanation data ─────────────────────────────
            factors_data = explanation_row.factors
            contributing_factors = factors_data.get(
                "contributing_factors", [],
            )
            confidence = float(factors_data.get("confidence", 0.0))
            top_features = [
                (f["name"], f["value"])
                for f in factors_data.get("top_features", [])
                if isinstance(f, dict) and "name" in f
            ]
            counterfactual_text = factors_data.get("counterfactual", "")
            clinical_text = explanation_row.explanation_text or ""
        elif _interpretability_engine:
            # ── Regenerate via the interpretability engine ───────────────
            try:
                features_df = pd.DataFrame([screening.input_data])
                result = _interpretability_engine.generate_explanation(
                    model_id=None,
                    features=features_df,
                    include_shap=True,
                    include_counterfactuals=True,
                    include_rules=False,
                    timeout_seconds=3.0,
                )

                shap_comp = result.get("components", {}).get("shap")
                if shap_comp and shap_comp.get("top_features"):
                    for feat in shap_comp["top_features"][:5]:
                        contributing_factors.append(feat["clinical_name"])
                    top_features = [
                        (f["feature"], f["mean_shap_value"])
                        for f in shap_comp["top_features"]
                    ]

                cf_comp = result.get("components", {}).get(
                    "counterfactuals",
                )
                if cf_comp:
                    counterfactual_text = cf_comp[0].get(
                        "description", "",
                    )

                clinical_text = result.get("clinical_summary", "")

                # Persist the generated explanation for future requests.
                new_explanation = ExplanationModel(
                    screening_id=screening.id,
                    explanation_text=clinical_text,
                    factors={
                        "contributing_factors": contributing_factors,
                        "confidence": confidence,
                        "top_features": [
                            {"name": n, "value": v}
                            for n, v in top_features
                        ],
                        "counterfactual": counterfactual_text,
                    },
                )
                db.add(new_explanation)
                db.commit()

            except InterpretabilityError as e:
                logger.warning(
                    "Interpretability regeneration failed: %s", e,
                )

        # ── Build response objects ──────────────────────────────────────
        risk_score = RiskScore(
            anonymized_id=screening.anonymized_id,
            score=screening.risk_score,
            risk_level=RiskLevel(screening.risk_level),
            confidence=confidence,
            contributing_factors=contributing_factors,
            timestamp=screening.created_at.isoformat(),
        )

        explanations = ExplanationSummary(
            top_features=top_features,
            counterfactual=counterfactual_text,
            rule_approximation="",
            clinical_interpretation=clinical_text,
        )

        return ExplanationResponse(
            anonymized_id=request.anonymized_id,
            explanations=explanations,
            risk_score=risk_score,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating explanation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )


@app.get(
    "/review-queue",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"}
    }
)
async def get_review_queue(
    auth: AuthResult = Depends(get_current_user),
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
        401: {"model": ErrorResponse, "description": "Authentication error"},
    },
)
async def get_statistics(
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return real-time statistics computed from the database."""
    try:
        from sqlalchemy import func as sa_func

        total_screenings: int = (
            db.query(sa_func.count(ScreeningModel.id)).scalar() or 0
        )

        avg_risk_score: float = (
            db.query(sa_func.avg(ScreeningModel.risk_score)).scalar() or 0.0
        )

        high_risk_count: int = (
            db.query(sa_func.count(ScreeningModel.id))
            .filter(ScreeningModel.risk_level.in_(["high", "critical"]))
            .scalar() or 0
        )

        high_risk_pct: float = (
            (high_risk_count / total_screenings * 100)
            if total_screenings > 0
            else 0.0
        )

        pending_reviews: int = (
            db.query(sa_func.count(ReviewModel.id))
            .filter(ReviewModel.status == "pending")
            .scalar() or 0
        )

        return {
            "timestamp": time.time(),
            "screenings": {
                "total": total_screenings,
                "avg_risk_score": round(avg_risk_score, 2),
                "high_risk_count": high_risk_count,
                "high_risk_pct": round(high_risk_pct, 1),
            },
            "review_queue": {
                "pending_count": pending_reviews,
            },
        }
    except Exception as e:
        logger.error("Error retrieving statistics: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )


@app.post(
    "/drift-check",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"}
    }
)
async def check_drift(
    auth: AuthResult = Depends(get_current_user)
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
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BatchScreeningResponse:
    """Screen multiple individuals in a batch using the structured model layer."""
    start_time = time.time()
    model = get_risk_model()

    logger.info(
        "Batch screening: %d individuals by %s",
        len(request.requests), auth.user_id,
    )

    results: list[ScreeningResponse] = []
    successful = 0
    failed = 0

    for screening_req in request.requests:
        try:
            # ── combine raw input ────────────────────────────────────
            combined_data: Dict[str, Any] = {}
            if screening_req.survey_data:
                combined_data.update(screening_req.survey_data)
            if screening_req.wearable_data:
                combined_data.update(screening_req.wearable_data)
            if screening_req.emr_data:
                combined_data.update(screening_req.emr_data)

            # ── model assessment ─────────────────────────────────────
            a = model.assess(combined_data)

            # ── persist ──────────────────────────────────────────────
            screening_row = ScreeningModel(
                anonymized_id=screening_req.anonymized_id,
                risk_score=a.risk_score,
                risk_level=a.risk_level,
                input_data=combined_data,
            )
            db.add(screening_row)
            db.flush()

            explanation_row = ExplanationModel(
                screening_id=screening_row.id,
                explanation_text=a.clinical_interpretation,
                factors={
                    "contributing_factors": a.contributing_factors,
                    "confidence": a.confidence,
                    "top_features": [
                        {"name": n, "value": v} for n, v in a.top_features
                    ],
                    "counterfactual": a.counterfactual,
                },
            )
            db.add(explanation_row)

            if a.requires_human_review:
                db.add(ReviewModel(
                    screening_id=screening_row.id, status="pending",
                ))

            # ── build response ───────────────────────────────────────
            risk_score = RiskScore(
                anonymized_id=screening_req.anonymized_id,
                score=a.risk_score,
                risk_level=RiskLevel(a.risk_level),
                confidence=a.confidence,
                contributing_factors=a.contributing_factors,
                timestamp=screening_req.timestamp,
            )

            recommendations = _generate_recommendations(
                a.risk_level, a.contributing_factors,
                anonymized_id=screening_req.anonymized_id,
            )

            results.append(ScreeningResponse(
                risk_score=risk_score,
                recommendations=recommendations,
                explanations=ExplanationSummary(
                    top_features=a.top_features,
                    counterfactual=a.counterfactual,
                    rule_approximation="",
                    clinical_interpretation=a.clinical_interpretation,
                ),
                requires_human_review=a.requires_human_review,
                alert_triggered=a.alert_triggered,
            ))
            successful += 1

        except Exception as e:
            logger.error(
                "Batch item error for %s: %s",
                screening_req.anonymized_id, e,
            )
            failed += 1
            results.append(ScreeningResponse(
                risk_score=RiskScore(
                    anonymized_id=screening_req.anonymized_id,
                    score=0.0,
                    risk_level=RiskLevel("low"),
                    confidence=0.0,
                    contributing_factors=[],
                    timestamp=screening_req.timestamp,
                ),
                recommendations=[],
                explanations=ExplanationSummary(
                    top_features=[],
                    counterfactual="",
                    rule_approximation="",
                    clinical_interpretation=f"Error: {e}",
                ),
                requires_human_review=True,
                alert_triggered=False,
            ))

    # Commit all successful rows in one transaction.
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Batch DB commit failed: %s", e)

    elapsed = time.time() - start_time
    logger.info(
        "Batch screening done in %.3fs — %d ok, %d failed",
        elapsed, successful, failed,
    )

    return BatchScreeningResponse(
        results=results,
        total=len(request.requests),
        successful=successful,
        failed=failed,
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
