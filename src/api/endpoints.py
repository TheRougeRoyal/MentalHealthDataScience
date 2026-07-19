"""Core API endpoints — screening, batch, risk-score, explain, statistics.

All database access uses Firestore via ``src.firebase_admin.get_firestore_client()``.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional

from firebase_admin import firestore
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.models import (
    ScreeningRequest, ScreeningResponse, RiskScore, RiskScoreResponse,
    ExplanationRequest, ExplanationResponse, ExplanationSummary,
    ResourceRecommendation, RiskLevel, ErrorResponse,
    BatchScreeningRequest, BatchScreeningResponse,
)
from src.api.auth import AuthResult, get_current_user
from src.api.metrics import (
    SCREENINGS_TOTAL, SCREENING_SCORE, ALERTS_TRIGGERED,
    REVIEWS_CREATED, BATCH_SIZE, BATCH_ITEMS,
)
from src.firebase_admin import get_firestore_client
from src.risk_model import get_risk_model

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────

def _generate_recommendations(risk_level: str, contributing_factors: list) -> list[ResourceRecommendation]:
    recs = []
    if risk_level in ("high", "critical"):
        recs.append(ResourceRecommendation(
            resource_type="crisis_line", name="988 Suicide & Crisis Lifeline",
            description="24/7 crisis support", contact_info="Call or text 988", urgency="immediate",
        ))
    if risk_level in ("moderate", "high", "critical"):
        recs.append(ResourceRecommendation(
            resource_type="therapy", name="Cognitive Behavioral Therapy",
            description="Evidence-based therapy for depression and anxiety",
            contact_info="Contact your healthcare provider",
            urgency="soon" if risk_level in ("high", "critical") else "routine",
        ))
    if risk_level in ("low", "moderate"):
        recs.append(ResourceRecommendation(
            resource_type="self_help", name="Mindfulness & Self-Care Resources",
            description="Guided meditation, journaling, and wellness apps",
            contact_info=None, urgency="routine",
        ))
    return recs


# ── POST /screen ───────────────────────────────────────────────────────────

@router.post("/screen", response_model=ScreeningResponse, status_code=status.HTTP_200_OK)
async def screen_individual(
    request: ScreeningRequest,
    auth: AuthResult = Depends(get_current_user),
) -> ScreeningResponse:
    start_time = time.time()

    if not request.consent_verified:
        raise HTTPException(status_code=403, detail="Consent not verified")

    combined: Dict[str, Any] = {}
    if request.survey_data:
        combined.update(request.survey_data)
    if request.wearable_data:
        combined.update(request.wearable_data)
    if request.emr_data:
        combined.update(request.emr_data)

    model = get_risk_model()
    assessment = model.assess(combined)

    db = get_firestore_client()
    screening_id = str(uuid.uuid4())

    db.collection("screenings").document(screening_id).set({
        "id": screening_id,
        "anonymized_id": request.anonymized_id,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "input_data": combined,
        "created_at": firestore.SERVER_TIMESTAMP,
    })

    db.collection("explanations").document(screening_id).set({
        "id": screening_id,
        "screening_id": screening_id,
        "explanation_text": assessment.clinical_interpretation,
        "factors": {
            "contributing_factors": assessment.contributing_factors,
            "confidence": assessment.confidence,
            "top_features": [{"name": n, "value": v} for n, v in assessment.top_features],
            "counterfactual": assessment.counterfactual,
        },
        "created_at": firestore.SERVER_TIMESTAMP,
    })

    if assessment.requires_human_review:
        db.collection("reviews").document(screening_id).set({
            "id": screening_id,
            "screening_id": screening_id,
            "status": "pending",
            "reviewer_uid": None,
            "notes": None,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

    risk_score = RiskScore(
        anonymized_id=request.anonymized_id,
        score=assessment.risk_score,
        risk_level=RiskLevel(assessment.risk_level),
        confidence=assessment.confidence,
        contributing_factors=assessment.contributing_factors,
        timestamp=request.timestamp,
    )

    elapsed = time.time() - start_time
    logger.info("Screening %.3fs — %s score=%.1f level=%s", elapsed, request.anonymized_id, assessment.risk_score, assessment.risk_level)

    SCREENINGS_TOTAL.labels(risk_level=assessment.risk_level).inc()
    SCREENING_SCORE.observe(assessment.risk_score)
    if assessment.alert_triggered:
        ALERTS_TRIGGERED.inc()
    if assessment.requires_human_review:
        REVIEWS_CREATED.inc()

    return ScreeningResponse(
        risk_score=risk_score,
        recommendations=_generate_recommendations(assessment.risk_level, assessment.contributing_factors),
        explanations=ExplanationSummary(
            top_features=assessment.top_features,
            counterfactual=assessment.counterfactual,
            clinical_interpretation=assessment.clinical_interpretation,
        ),
        requires_human_review=assessment.requires_human_review,
        alert_triggered=assessment.alert_triggered,
    )


# ── POST /batch-screen ────────────────────────────────────────────────────

@router.post("/batch-screen", response_model=BatchScreeningResponse, status_code=status.HTTP_200_OK)
async def batch_screen(
    request: BatchScreeningRequest,
    auth: AuthResult = Depends(get_current_user),
) -> BatchScreeningResponse:
    model = get_risk_model()
    db = get_firestore_client()
    results = []
    successful = 0
    BATCH_SIZE.observe(len(request.requests))

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
            screening_id = str(uuid.uuid4())

            db.collection("screenings").document(screening_id).set({
                "id": screening_id,
                "anonymized_id": req.anonymized_id,
                "risk_score": a.risk_score,
                "risk_level": a.risk_level,
                "input_data": combined,
                "created_at": firestore.SERVER_TIMESTAMP,
            })

            db.collection("explanations").document(screening_id).set({
                "id": screening_id,
                "screening_id": screening_id,
                "explanation_text": a.clinical_interpretation,
                "factors": {
                    "contributing_factors": a.contributing_factors,
                    "confidence": a.confidence,
                    "top_features": [{"name": n, "value": v} for n, v in a.top_features],
                    "counterfactual": a.counterfactual,
                },
                "created_at": firestore.SERVER_TIMESTAMP,
            })

            if a.requires_human_review:
                db.collection("reviews").document(screening_id).set({
                    "id": screening_id,
                    "screening_id": screening_id,
                    "status": "pending",
                    "reviewer_uid": None,
                    "notes": None,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                })

            results.append(ScreeningResponse(
                risk_score=RiskScore(
                    anonymized_id=req.anonymized_id, score=a.risk_score,
                    risk_level=RiskLevel(a.risk_level), confidence=a.confidence,
                    contributing_factors=a.contributing_factors, timestamp=req.timestamp,
                ),
                recommendations=_generate_recommendations(a.risk_level, a.contributing_factors),
                explanations=ExplanationSummary(
                    top_features=a.top_features, counterfactual=a.counterfactual,
                    clinical_interpretation=a.clinical_interpretation,
                ),
                requires_human_review=a.requires_human_review,
                alert_triggered=a.alert_triggered,
            ))
            successful += 1
            SCREENINGS_TOTAL.labels(risk_level=a.risk_level).inc()
            SCREENING_SCORE.observe(a.risk_score)
            BATCH_ITEMS.labels(status="success").inc()
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
            BATCH_ITEMS.labels(status="error").inc()

    return BatchScreeningResponse(
        results=results, total=len(request.requests),
        successful=successful, failed=len(request.requests) - successful,
    )


# ── GET /risk-score/{anonymized_id} ───────────────────────────────────────

@router.get("/risk-score/{anonymized_id}", response_model=RiskScoreResponse)
async def get_risk_score(
    anonymized_id: str,
    auth: AuthResult = Depends(get_current_user),
):
    db = get_firestore_client()
    docs = list(
        db.collection("screenings")
        .where("anonymized_id", "==", anonymized_id)
        .get()
    )

    if not docs:
        raise HTTPException(status_code=404, detail=f"No risk score for {anonymized_id}")

    docs.sort(key=lambda d: d.to_dict().get("created_at", ""), reverse=True)
    data = docs[0].to_dict()
    return RiskScoreResponse(
        anonymized_id=data["anonymized_id"],
        score=data["risk_score"],
        risk_level=RiskLevel(data["risk_level"]),
        found=True,
    )


# ── POST /explain ─────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: ExplanationRequest,
    auth: AuthResult = Depends(get_current_user),
):
    db = get_firestore_client()

    if request.prediction_id:
        screening_doc = db.collection("screenings").document(request.prediction_id).get()
    else:
        docs = list(
            db.collection("screenings")
            .where("anonymized_id", "==", request.anonymized_id)
            .get()
        )
        if docs:
            docs.sort(key=lambda d: d.to_dict().get("created_at", ""), reverse=True)
        screening_doc = docs[0] if docs else None

    if not screening_doc or not screening_doc.exists:
        raise HTTPException(status_code=404, detail=f"No screening for {request.anonymized_id}")

    screening_data = screening_doc.to_dict()
    screening_id = screening_data.get("id", screening_doc.id)

    explanation_doc = db.collection("explanations").document(screening_id).get()
    factors_data = {}
    clinical_text = ""
    if explanation_doc.exists:
        exp_data = explanation_doc.to_dict()
        factors_data = exp_data.get("factors", {})
        clinical_text = exp_data.get("explanation_text", "")

    return ExplanationResponse(
        anonymized_id=request.anonymized_id,
        explanations=ExplanationSummary(
            top_features=[
                (f["name"], f["value"]) for f in factors_data.get("top_features", [])
                if isinstance(f, dict) and "name" in f
            ],
            counterfactual=factors_data.get("counterfactual", ""),
            clinical_interpretation=clinical_text,
        ),
        risk_score=RiskScore(
            anonymized_id=screening_data["anonymized_id"],
            score=screening_data["risk_score"],
            risk_level=RiskLevel(screening_data["risk_level"]),
            confidence=float(factors_data.get("confidence", 0.0)),
            contributing_factors=factors_data.get("contributing_factors", []),
            timestamp=screening_data.get("created_at", ""),
        ),
    )


# ── GET /statistics ───────────────────────────────────────────────────────

@router.get("/statistics")
async def get_statistics(auth: AuthResult = Depends(get_current_user)):
    db = get_firestore_client()

    screenings = list(db.collection("screenings").get())
    total = len(screenings)

    if total == 0:
        return {
            "timestamp": time.time(),
            "screenings": {"total": 0, "avg_risk_score": 0, "high_risk_count": 0, "high_risk_pct": 0},
            "review_queue": {"pending_count": 0},
        }

    scores = [s.to_dict().get("risk_score", 0) for s in screenings]
    avg = sum(scores) / total
    high = sum(1 for s in screenings if s.to_dict().get("risk_level") in ("high", "critical"))

    reviews = list(db.collection("reviews").where("status", "==", "pending").get())

    return {
        "timestamp": time.time(),
        "screenings": {
            "total": total,
            "avg_risk_score": round(avg, 2),
            "high_risk_count": high,
            "high_risk_pct": round((high / total * 100) if total else 0, 1),
        },
        "review_queue": {"pending_count": len(reviews)},
    }
