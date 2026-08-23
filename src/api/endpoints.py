"""Core API endpoints — screening, batch, risk-score, explain, statistics.

All database access uses Firestore via ``src.firebase_admin.get_firestore_client()``.
"""

import logging
import hashlib
import json
import time
import uuid
from typing import Any, Dict, Optional

from firebase_admin import firestore
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

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
from src.firebase_admin import get_firestore_client, persistence_enabled
from src.risk_model import get_risk_model
from src.privacy import encrypt_input, minimize_input
from src.api.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter()


def _idempotency_document(db: Any, user_id: str, key: str) -> Any:
    digest = hashlib.sha256(f"{user_id}:{key}".encode("utf-8")).hexdigest()
    return db.collection("idempotency_keys").document(digest)


def _request_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_idempotent_response(db: Any, user_id: str, key: str, fingerprint: str) -> Any | None:
    document = _idempotency_document(db, user_id, key).get()
    if not document.exists:
        return None
    stored = document.to_dict() or {}
    if stored.get("fingerprint") != fingerprint:
        raise HTTPException(status_code=409, detail="Idempotency-Key was used with a different request")
    return stored.get("response")


def _store_idempotent_response(db: Any, user_id: str, key: str, fingerprint: str, response: Any) -> None:
    _idempotency_document(db, user_id, key).set({
        "user_id": user_id,
        "fingerprint": fingerprint,
        "response": response,
        "created_at": firestore.SERVER_TIMESTAMP,
    })


def _commit_screening(db: Any, screening_id: str, user_id: str, anonymized_id: str,
                      assessment: Any, combined: Dict[str, Any]) -> None:
    """Commit the screening graph atomically, retrying transient failures."""
    for attempt in range(3):
        batch = db.batch()
        batch.set(db.collection("screenings").document(screening_id), {
            "id": screening_id,
            "user_id": user_id,
            "anonymized_id": anonymized_id,
            "risk_score": assessment.risk_score,
            "risk_level": assessment.risk_level,
            "model_version": assessment.model_version,
            "confidence_method": assessment.confidence_method,
            "input_data_encrypted": encrypt_input(combined),
            "input_data_fields": sorted(minimize_input(combined)),
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        batch.set(db.collection("explanations").document(screening_id), {
            "id": screening_id,
            "screening_id": screening_id,
            "user_id": user_id,
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
            batch.set(db.collection("reviews").document(screening_id), {
                "id": screening_id,
                "screening_id": screening_id,
                "user_id": user_id,
                "status": "pending",
                "reviewer_uid": None,
                "notes": None,
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
        batch.set(db.collection("audit_logs").document(), {
            "action": "screening_created",
            "actor_user_id": user_id,
            "resource_type": "screening",
            "resource_id": screening_id,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        try:
            batch.commit()
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning("Transient Firestore commit failure for screening %s; retrying", screening_id)


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
@limiter.limit("120/minute")
async def screen_individual(
    request: Request,
    screening_request: ScreeningRequest,
    auth: AuthResult = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> ScreeningResponse:
    start_time = time.time()

    if not screening_request.consent_verified:
        raise HTTPException(status_code=403, detail="Consent not verified")

    fingerprint = _request_fingerprint(screening_request.model_dump(mode="json"))

    combined: Dict[str, Any] = {}
    if screening_request.survey_data:
        combined.update(screening_request.survey_data)
    if screening_request.wearable_data:
        combined.update(screening_request.wearable_data)
    if screening_request.emr_data:
        combined.update(screening_request.emr_data)

    model = get_risk_model()
    assessment = model.assess(combined)

    try:
        db = get_firestore_client()
    except Exception as exc:
        logger.error("Screening persistence client unavailable: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assessment persistence is unavailable. Check Firebase and encryption configuration.",
        ) from exc
    if db is not None and idempotency_key:
        previous = _load_idempotent_response(db, auth.user_id, idempotency_key, fingerprint)
        if previous is not None:
            return ScreeningResponse.model_validate(previous)
    screening_id = str(uuid.uuid4())

    if db is not None:
        try:
            _commit_screening(db, screening_id, auth.user_id, screening_request.anonymized_id, assessment, combined)
        except Exception as exc:
            logger.error("Screening persistence failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Assessment persistence is unavailable. Check Firebase and encryption configuration.",
            ) from exc
    else:
        logger.debug("Persistence disabled — skipping Firestore writes for screening %s", screening_id)

    risk_score = RiskScore(
        anonymized_id=screening_request.anonymized_id,
        score=assessment.risk_score,
        risk_level=RiskLevel(assessment.risk_level),
        confidence=assessment.confidence,
        contributing_factors=assessment.contributing_factors,
        timestamp=screening_request.timestamp,
        model_version=assessment.model_version,
        confidence_method=assessment.confidence_method,
    )

    elapsed = time.time() - start_time
    logger.info("Screening %.3fs — %s score=%.1f level=%s", elapsed, screening_request.anonymized_id, assessment.risk_score, assessment.risk_level)

    SCREENINGS_TOTAL.labels(risk_level=assessment.risk_level).inc()
    SCREENING_SCORE.observe(assessment.risk_score)
    if assessment.alert_triggered:
        ALERTS_TRIGGERED.inc()
    if assessment.requires_human_review:
        REVIEWS_CREATED.inc()

    response = ScreeningResponse(
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
    if db is not None and idempotency_key:
        _store_idempotent_response(db, auth.user_id, idempotency_key, fingerprint, response.model_dump(mode="json"))
    return response


# ── POST /batch-screen ────────────────────────────────────────────────────

@router.post("/batch-screen", response_model=BatchScreeningResponse, status_code=status.HTTP_200_OK)
@limiter.limit("30/minute")
async def batch_screen(
    request: Request,
    batch_request: BatchScreeningRequest,
    auth: AuthResult = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> BatchScreeningResponse:
    if any(not item.consent_verified for item in batch_request.requests):
        raise HTTPException(status_code=403, detail="Consent must be verified for every batch item")

    fingerprint = _request_fingerprint(batch_request.model_dump(mode="json"))

    model = get_risk_model()
    try:
        db = get_firestore_client()
    except Exception as exc:
        logger.error("Batch screening persistence client unavailable: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assessment persistence is unavailable. Check Firebase and encryption configuration.",
        ) from exc
    if db is not None and idempotency_key:
        previous = _load_idempotent_response(db, auth.user_id, idempotency_key, fingerprint)
        if previous is not None:
            return BatchScreeningResponse.model_validate(previous)
    results = []
    successful = 0
    BATCH_SIZE.observe(len(batch_request.requests))

    for req in batch_request.requests:
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

            if db is not None:
                try:
                    _commit_screening(db, screening_id, auth.user_id, req.anonymized_id, a, combined)
                except Exception as exc:
                    logger.error("Batch screening persistence failed: %s", exc, exc_info=True)
                    raise RuntimeError("Assessment persistence is unavailable") from exc
            else:
                logger.debug("Persistence disabled — skipping Firestore writes for batch item %s", screening_id)

            results.append(ScreeningResponse(
                risk_score=RiskScore(
                    anonymized_id=req.anonymized_id, score=a.risk_score,
                    risk_level=RiskLevel(a.risk_level), confidence=a.confidence,
                    contributing_factors=a.contributing_factors, timestamp=req.timestamp,
                    model_version=a.model_version, confidence_method=a.confidence_method,
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

    response = BatchScreeningResponse(
        results=results, total=len(batch_request.requests),
        successful=successful, failed=len(batch_request.requests) - successful,
    )
    if db is not None and idempotency_key:
        _store_idempotent_response(db, auth.user_id, idempotency_key, fingerprint, response.model_dump(mode="json"))
    return response


# ── GET /risk-score/{anonymized_id} ───────────────────────────────────────

@router.get("/risk-score/{anonymized_id}", response_model=RiskScoreResponse)
async def get_risk_score(
    anonymized_id: str,
    auth: AuthResult = Depends(get_current_user),
):
    db = get_firestore_client()
    query = db.collection("screenings").where("anonymized_id", "==", anonymized_id)
    if auth.role != "admin":
        query = query.where("user_id", "==", auth.user_id)
    docs = list(query.get())

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
        if screening_doc.exists:
            data = screening_doc.to_dict()
            if auth.role != "admin" and data.get("user_id") != auth.user_id:
                raise HTTPException(status_code=403, detail="Access forbidden to this assessment")
    else:
        query = db.collection("screenings").where("anonymized_id", "==", request.anonymized_id)
        if auth.role != "admin":
            query = query.where("user_id", "==", auth.user_id)
        docs = list(query.get())
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

    if db is None:
        logger.debug("Persistence disabled — returning empty statistics")
        return {
            "timestamp": time.time(),
            "screenings": {
                "total": 0, "avg_risk_score": 0, "median_risk_score": 0,
                "min_risk_score": 0, "max_risk_score": 0,
                "high_risk_count": 0, "high_risk_pct": 0,
            },
            "risk_distribution": {level.value: 0 for level in RiskLevel},
            "review_queue": {"pending_count": 0},
        }

    if auth.role == "admin":
        screenings = list(db.collection("screenings").get())
    else:
        screenings = list(db.collection("screenings").where("user_id", "==", auth.user_id).get())

    records = [screening.to_dict() for screening in screenings]
    scores = [float(record.get("risk_score", 0)) for record in records]
    total = len(scores)
    levels = {level.value: 0 for level in RiskLevel}
    for record in records:
        level = str(record.get("risk_level", "low")).lower()
        if level in levels:
            levels[level] += 1

    if total == 0:
        return {
            "timestamp": time.time(),
            "screenings": {
                "total": 0, "avg_risk_score": 0, "median_risk_score": 0,
                "min_risk_score": 0, "max_risk_score": 0,
                "high_risk_count": 0, "high_risk_pct": 0,
            },
            "risk_distribution": levels,
            "review_queue": {"pending_count": 0},
        }

    avg = sum(scores) / total
    ordered_scores = sorted(scores)
    midpoint = total // 2
    median = ordered_scores[midpoint] if total % 2 else (ordered_scores[midpoint - 1] + ordered_scores[midpoint]) / 2
    high = levels[RiskLevel.HIGH.value] + levels[RiskLevel.CRITICAL.value]

    if auth.role == "admin":
        reviews = list(db.collection("reviews").where("status", "==", "pending").get())
    else:
        reviews = list(db.collection("reviews").where("user_id", "==", auth.user_id).where("status", "==", "pending").get())

    return {
        "timestamp": time.time(),
        "screenings": {
            "total": total,
            "avg_risk_score": round(avg, 2),
            "median_risk_score": round(median, 2),
            "min_risk_score": round(min(scores), 2),
            "max_risk_score": round(max(scores), 2),
            "high_risk_count": high,
            "high_risk_pct": round((high / total * 100) if total else 0, 1),
        },
        "risk_distribution": levels,
        "review_queue": {"pending_count": len(reviews)},
    }
