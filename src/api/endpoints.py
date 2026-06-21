"""Core API endpoints — screening, batch, risk-score, explain, statistics."""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import Screening as ScreeningModel, Explanation as ExplanationModel, Review as ReviewModel
from src.api.models import (
    ScreeningRequest, ScreeningResponse, RiskScore, RiskScoreResponse,
    ExplanationRequest, ExplanationResponse, ExplanationSummary,
    ResourceRecommendation, RiskLevel, ErrorResponse,
    BatchScreeningRequest, BatchScreeningResponse,
)
from src.api.auth import AuthResult, get_current_user
from src.risk_model import get_risk_model

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────

def _generate_recommendations(risk_level: str, contributing_factors: list) -> list[ResourceRecommendation]:
    """Generate resource recommendations based on risk level."""
    recs = []
    if risk_level in ("high", "critical"):
        recs.append(ResourceRecommendation(
            resource_type="crisis_line",
            name="988 Suicide & Crisis Lifeline",
            description="24/7 crisis support",
            contact_info="Call or text 988",
            urgency="immediate",
        ))
    if risk_level in ("moderate", "high", "critical"):
        recs.append(ResourceRecommendation(
            resource_type="therapy",
            name="Cognitive Behavioral Therapy",
            description="Evidence-based therapy for depression and anxiety",
            contact_info="Contact your healthcare provider",
            urgency="soon" if risk_level in ("high", "critical") else "routine",
        ))
    if risk_level in ("low", "moderate"):
        recs.append(ResourceRecommendation(
            resource_type="self_help",
            name="Mindfulness & Self-Care Resources",
            description="Guided meditation, journaling, and wellness apps",
            contact_info=None,
            urgency="routine",
        ))
    return recs


# ── POST /screen ───────────────────────────────────────────────────────────

@router.post(
    "/screen",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
)
async def screen_individual(
    request: ScreeningRequest,
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ScreeningResponse:
    start_time = time.time()

    if not request.consent_verified:
        raise HTTPException(status_code=403, detail="Consent not verified")

    # Combine input data
    combined: Dict[str, Any] = {}
    if request.survey_data:
        combined.update(request.survey_data)
    if request.wearable_data:
        combined.update(request.wearable_data)
    if request.emr_data:
        combined.update(request.emr_data)

    # Run risk model
    model = get_risk_model()
    assessment = model.assess(combined)

    # Persist
    screening_row = ScreeningModel(
        anonymized_id=request.anonymized_id,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        input_data=combined,
    )
    db.add(screening_row)
    db.flush()

    explanation_row = ExplanationModel(
        screening_id=screening_row.id,
        explanation_text=assessment.clinical_interpretation,
        factors={
            "contributing_factors": assessment.contributing_factors,
            "confidence": assessment.confidence,
            "top_features": [{"name": n, "value": v} for n, v in assessment.top_features],
            "counterfactual": assessment.counterfactual,
        },
    )
    db.add(explanation_row)

    if assessment.requires_human_review:
        db.add(ReviewModel(screening_id=screening_row.id, status="pending"))

    db.commit()

    risk_score = RiskScore(
        anonymized_id=request.anonymized_id,
        score=assessment.risk_score,
        risk_level=RiskLevel(assessment.risk_level),
        confidence=assessment.confidence,
        contributing_factors=assessment.contributing_factors,
        timestamp=request.timestamp,
    )

    recommendations = _generate_recommendations(assessment.risk_level, assessment.contributing_factors)

    elapsed = time.time() - start_time
    logger.info(
        "Screening %.3fs — %s score=%.1f level=%s review=%s",
        elapsed, request.anonymized_id, assessment.risk_score,
        assessment.risk_level, assessment.requires_human_review,
    )

    return ScreeningResponse(
        risk_score=risk_score,
        recommendations=recommendations,
        explanations=ExplanationSummary(
            top_features=assessment.top_features,
            counterfactual=assessment.counterfactual,
            clinical_interpretation=assessment.clinical_interpretation,
        ),
        requires_human_review=assessment.requires_human_review,
        alert_triggered=assessment.alert_triggered,
    )


# ── POST /batch-screen ────────────────────────────────────────────────────

@router.post(
    "/batch-screen",
    response_model=BatchScreeningResponse,
    status_code=status.HTTP_200_OK,
)
async def batch_screen(
    request: BatchScreeningRequest,
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BatchScreeningResponse:
    model = get_risk_model()
    results = []
    successful = 0

    for req in request.requests:
        try:
            combined: Dict[str, Any] = {}
            if req.survey_data:
                combined.update(req.survey_data)
            if req.wearable_data:
                combined.update(req.wearable_data)
            if req.emr_data:
                combined.update(req.emr_data)

            a = model.assess(combined)

            row = ScreeningModel(
                anonymized_id=req.anonymized_id,
                risk_score=a.risk_score,
                risk_level=a.risk_level,
                input_data=combined,
            )
            db.add(row)
            db.flush()

            db.add(ExplanationModel(
                screening_id=row.id,
                explanation_text=a.clinical_interpretation,
                factors={
                    "contributing_factors": a.contributing_factors,
                    "confidence": a.confidence,
                    "top_features": [{"name": n, "value": v} for n, v in a.top_features],
                    "counterfactual": a.counterfactual,
                },
            ))

            if a.requires_human_review:
                db.add(ReviewModel(screening_id=row.id, status="pending"))

            results.append(ScreeningResponse(
                risk_score=RiskScore(
                    anonymized_id=req.anonymized_id,
                    score=a.risk_score,
                    risk_level=RiskLevel(a.risk_level),
                    confidence=a.confidence,
                    contributing_factors=a.contributing_factors,
                    timestamp=req.timestamp,
                ),
                recommendations=_generate_recommendations(a.risk_level, a.contributing_factors),
                explanations=ExplanationSummary(
                    top_features=a.top_features,
                    counterfactual=a.counterfactual,
                    clinical_interpretation=a.clinical_interpretation,
                ),
                requires_human_review=a.requires_human_review,
                alert_triggered=a.alert_triggered,
            ))
            successful += 1
        except Exception as e:
            logger.error("Batch item error %s: %s", req.anonymized_id, e)
            results.append(ScreeningResponse(
                risk_score=RiskScore(
                    anonymized_id=req.anonymized_id, score=0.0,
                    risk_level=RiskLevel("low"), confidence=0.0,
                    contributing_factors=[], timestamp=req.timestamp,
                ),
                recommendations=[], explanations=ExplanationSummary(),
                requires_human_review=True, alert_triggered=False,
            ))

    db.commit()
    return BatchScreeningResponse(
        results=results, total=len(request.requests),
        successful=successful, failed=len(request.requests) - successful,
    )


# ── GET /risk-score/{anonymized_id} ───────────────────────────────────────

@router.get("/risk-score/{anonymized_id}", response_model=RiskScoreResponse)
async def get_risk_score(
    anonymized_id: str,
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    screening = (
        db.query(ScreeningModel)
        .filter(ScreeningModel.anonymized_id == anonymized_id)
        .order_by(ScreeningModel.created_at.desc())
        .first()
    )
    if not screening:
        raise HTTPException(status_code=404, detail=f"No risk score for {anonymized_id}")

    return RiskScoreResponse(
        anonymized_id=screening.anonymized_id,
        score=screening.risk_score,
        risk_level=RiskLevel(screening.risk_level),
        found=True,
    )


# ── POST /explain ─────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: ExplanationRequest,
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if request.prediction_id:
        screening = db.query(ScreeningModel).filter(ScreeningModel.id == request.prediction_id).first()
    else:
        screening = (
            db.query(ScreeningModel)
            .filter(ScreeningModel.anonymized_id == request.anonymized_id)
            .order_by(ScreeningModel.created_at.desc())
            .first()
        )
    if not screening:
        raise HTTPException(status_code=404, detail=f"No screening for {request.anonymized_id}")

    explanation = (
        db.query(ExplanationModel)
        .filter(ExplanationModel.screening_id == screening.id)
        .order_by(ExplanationModel.created_at.desc())
        .first()
    )

    factors_data = {}
    if explanation and isinstance(explanation.factors, dict):
        factors_data = explanation.factors

    return ExplanationResponse(
        anonymized_id=request.anonymized_id,
        explanations=ExplanationSummary(
            top_features=[
                (f["name"], f["value"]) for f in factors_data.get("top_features", [])
                if isinstance(f, dict) and "name" in f
            ],
            counterfactual=factors_data.get("counterfactual", ""),
            clinical_interpretation=explanation.explanation_text if explanation else "",
        ),
        risk_score=RiskScore(
            anonymized_id=screening.anonymized_id,
            score=screening.risk_score,
            risk_level=RiskLevel(screening.risk_level),
            confidence=float(factors_data.get("confidence", 0.0)),
            contributing_factors=factors_data.get("contributing_factors", []),
            timestamp=screening.created_at.isoformat() if screening.created_at else "",
        ),
    )


# ── GET /statistics ───────────────────────────────────────────────────────

@router.get("/statistics")
async def get_statistics(
    auth: AuthResult = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from sqlalchemy import func as sa_func

    total = db.query(sa_func.count(ScreeningModel.id)).scalar() or 0
    avg = db.query(sa_func.avg(ScreeningModel.risk_score)).scalar() or 0.0
    high = (
        db.query(sa_func.count(ScreeningModel.id))
        .filter(ScreeningModel.risk_level.in_(["high", "critical"]))
        .scalar() or 0
    )
    pending = (
        db.query(sa_func.count(ReviewModel.id))
        .filter(ReviewModel.status == "pending")
        .scalar() or 0
    )

    return {
        "timestamp": time.time(),
        "screenings": {
            "total": total,
            "avg_risk_score": round(avg, 2),
            "high_risk_count": high,
            "high_risk_pct": round((high / total * 100) if total else 0, 1),
        },
        "review_queue": {"pending_count": pending},
    }
