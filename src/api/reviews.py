"""Clinical review workflow endpoints (Firestore-backed).

* ``GET  /reviews``              – list reviews, filterable by status
* ``GET  /reviews/queue``        – same as GET /reviews (backward compat)
* ``GET  /reviews/{id}``         – get single review with screening context
* ``PATCH /reviews/{id}``        – update status + optional notes
* ``POST /reviews/{id}/assign``  – assign a reviewer (legacy compat)
* ``POST /reviews/{id}/comment`` – add a note (legacy compat)
* ``POST /reviews/{id}/close``   – close a review (legacy compat)
"""

from __future__ import annotations

import logging
from typing import Optional

from firebase_admin import firestore
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from src.api.auth import AuthResult, get_current_user, require_role
from src.firebase_admin import get_firestore_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class UpdateReviewRequest(BaseModel):
    status: Optional[str] = Field(None, description="pending|approved|escalated|closed")
    notes: Optional[str] = Field(None, description="Reviewer notes")


class AssignRequest(BaseModel):
    reviewer: str = Field(..., min_length=1, max_length=128)


class CommentRequest(BaseModel):
    comments: str = Field(..., min_length=1)


class ReviewOut(BaseModel):
    id: str
    screening_id: str
    status: str
    reviewer_uid: Optional[str] = None
    notes: Optional[str] = None
    created_at: str
    updated_at: str

    # Screening context (denormalised for convenience)
    anonymized_id: Optional[str] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None


def _serialise_review(review_doc: dict, screening_data: Optional[dict] = None) -> dict:
    """Build a ReviewOut-compatible dict from Firestore documents."""
    created = review_doc.get("created_at")
    updated = review_doc.get("updated_at")
    return ReviewOut(
        id=review_doc.get("id", ""),
        screening_id=review_doc.get("screening_id", ""),
        status=review_doc.get("status", "pending"),
        reviewer_uid=review_doc.get("reviewer_uid"),
        notes=review_doc.get("notes"),
        created_at=str(created) if created else "",
        updated_at=str(updated) if updated else "",
        anonymized_id=screening_data.get("anonymized_id") if screening_data else None,
        risk_score=screening_data.get("risk_score") if screening_data else None,
        risk_level=screening_data.get("risk_level") if screening_data else None,
    ).model_dump()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/reviews", tags=["Reviews"])

# Valid review statuses
_VALID_STATUSES = {"pending", "approved", "escalated", "closed"}


@router.get("/queue")
async def get_review_queue_legacy(
    status_filter: Optional[str] = Query(None, alias="status_filter"),
    limit: int = 50,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """Backward-compatible endpoint that delegates to GET /reviews."""
    return await _get_reviews(status_filter, limit, auth)


@router.get("")
async def list_reviews(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = 50,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """List reviews, optionally filtered by status."""
    return await _get_reviews(status_filter, limit, auth)


async def _get_reviews(status_filter: Optional[str], limit: int, auth: AuthResult):
    db = get_firestore_client()
    filter_val = status_filter or "pending"

    review_docs = list(
        db.collection("reviews")
        .where("status", "==", filter_val)
        .get()
    )
    review_docs.sort(key=lambda d: d.to_dict().get("created_at", ""), reverse=True)
    review_docs = review_docs[:limit]

    total_pending = len(
        list(db.collection("reviews").where("status", "==", "pending").get())
    )

    reviews = []
    for doc in review_docs:
        rd = doc.to_dict()
        screening_data = None
        if rd.get("screening_id"):
            s_doc = db.collection("screenings").document(rd["screening_id"]).get()
            if s_doc.exists:
                screening_data = s_doc.to_dict()
        reviews.append(_serialise_review(rd, screening_data))

    return {
        "reviews": reviews,
        "total": len(reviews),
        "total_pending": total_pending,
    }


@router.get("/{review_id}")
async def get_review(
    review_id: str,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """Get a single review with its linked screening context."""
    db = get_firestore_client()
    doc = db.collection("reviews").document(review_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Review not found")

    rd = doc.to_dict()
    screening_data = None
    if rd.get("screening_id"):
        s_doc = db.collection("screenings").document(rd["screening_id"]).get()
        if s_doc.exists:
            screening_data = s_doc.to_dict()

    return _serialise_review(rd, screening_data)


@router.patch("/{review_id}")
async def update_review(
    review_id: str,
    body: UpdateReviewRequest,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """Update a review's status and/or notes."""
    db = get_firestore_client()
    doc = db.collection("reviews").document(review_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Review not found")

    rd = doc.to_dict()
    updates: dict = {"updated_at": firestore.SERVER_TIMESTAMP}

    if body.status is not None:
        if body.status not in _VALID_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(sorted(_VALID_STATUSES))}",
            )
        updates["status"] = body.status

    if body.notes is not None:
        updates["notes"] = body.notes

    db.collection("reviews").document(review_id).update(updates)

    # Re-fetch and return
    updated = db.collection("reviews").document(review_id).get().to_dict()
    screening_data = None
    if updated.get("screening_id"):
        s_doc = db.collection("screenings").document(updated["screening_id"]).get()
        if s_doc.exists:
            screening_data = s_doc.to_dict()

    logger.info("Review %s updated by %s: %s", review_id, auth.user_id, updates)
    return _serialise_review(updated, screening_data)


# ── Legacy endpoints (backward compat with old frontend) ────────────────────

@router.post("/{review_id}/assign")
async def assign_reviewer(
    review_id: str,
    body: AssignRequest,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """Assign a reviewer to a review (sets status to 'approved')."""
    db = get_firestore_client()
    doc = db.collection("reviews").document(review_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Review not found")

    rd = doc.to_dict()
    if rd.get("status") == "closed":
        raise HTTPException(status_code=409, detail="Cannot assign reviewer to closed review")

    db.collection("reviews").document(review_id).update({
        "reviewer_uid": body.reviewer,
        "status": "approved",
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

    updated = db.collection("reviews").document(review_id).get().to_dict()
    screening_data = None
    if updated.get("screening_id"):
        s_doc = db.collection("screenings").document(updated["screening_id"]).get()
        if s_doc.exists:
            screening_data = s_doc.to_dict()

    logger.info("Review %s assigned to %s by %s", review_id, body.reviewer, auth.user_id)
    return _serialise_review(updated, screening_data)


@router.post("/{review_id}/comment")
async def add_comment(
    review_id: str,
    body: CommentRequest,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """Append a note to a review."""
    db = get_firestore_client()
    doc = db.collection("reviews").document(review_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Review not found")

    rd = doc.to_dict()
    existing = rd.get("notes") or ""
    prefix = f"[{auth.user_id}] "
    new_notes = f"{existing}\n{prefix}{body.comments}".strip()

    db.collection("reviews").document(review_id).update({
        "notes": new_notes,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

    updated = db.collection("reviews").document(review_id).get().to_dict()
    screening_data = None
    if updated.get("screening_id"):
        s_doc = db.collection("screenings").document(updated["screening_id"]).get()
        if s_doc.exists:
            screening_data = s_doc.to_dict()

    logger.info("Comment added to review %s by %s", review_id, auth.user_id)
    return _serialise_review(updated, screening_data)


@router.post("/{review_id}/close")
async def close_review(
    review_id: str,
    body: Optional[CommentRequest] = None,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
):
    """Close a review, optionally adding a final note."""
    db = get_firestore_client()
    doc = db.collection("reviews").document(review_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Review not found")

    rd = doc.to_dict()
    if rd.get("status") == "closed":
        raise HTTPException(status_code=409, detail="Review is already closed")

    updates: dict = {
        "status": "closed",
        "updated_at": firestore.SERVER_TIMESTAMP,
    }

    if body and body.comments:
        existing = rd.get("notes") or ""
        prefix = f"[{auth.user_id}] "
        updates["notes"] = f"{existing}\n{prefix}{body.comments}".strip()

    if not rd.get("reviewer_uid"):
        updates["reviewer_uid"] = auth.user_id

    db.collection("reviews").document(review_id).update(updates)

    updated = db.collection("reviews").document(review_id).get().to_dict()
    screening_data = None
    if updated.get("screening_id"):
        s_doc = db.collection("screenings").document(updated["screening_id"]).get()
        if s_doc.exists:
            screening_data = s_doc.to_dict()

    logger.info("Review %s closed by %s", review_id, auth.user_id)
    return _serialise_review(updated, screening_data)
