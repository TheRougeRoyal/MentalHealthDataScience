"""Clinical review workflow endpoints.

Provides DB-backed endpoints for the review queue:

* ``GET  /reviews/queue``       – list pending/filtered reviews
* ``POST /reviews/{id}/assign`` – assign a reviewer
* ``POST /reviews/{id}/comment``– add a comment
* ``POST /reviews/{id}/close``  – mark as reviewed/closed
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from src.api.auth import AuthResult, get_current_user, require_role
from src.database import get_db
from src.models import Review as ReviewModel, Screening as ScreeningModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------


class AssignRequest(BaseModel):
    reviewer: str = Field(..., min_length=1, max_length=128)


class CommentRequest(BaseModel):
    comments: str = Field(..., min_length=1)


class ReviewOut(BaseModel):
    """Serialised review row returned by all endpoints."""
    id: str
    screening_id: str
    status: str
    reviewer: Optional[str] = None
    comments: Optional[str] = None
    created_at: str

    # Screening context (denormalised for convenience)
    anonymized_id: Optional[str] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None


def _serialise(review: ReviewModel) -> dict:
    """Turn a Review ORM instance into a dict matching ReviewOut."""
    screening = review.screening
    return ReviewOut(
        id=review.id,
        screening_id=review.screening_id,
        status=review.status,
        reviewer=review.reviewer,
        comments=review.comments,
        created_at=review.created_at.isoformat() if review.created_at else "",
        anonymized_id=screening.anonymized_id if screening else None,
        risk_score=screening.risk_score if screening else None,
        risk_level=screening.risk_level if screening else None,
    ).model_dump()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/reviews", tags=["Reviews"])


@router.get("/queue")
async def get_review_queue(
    status_filter: Optional[str] = None,
    limit: int = 50,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
    db: Session = Depends(get_db),
):
    """Return reviews, optionally filtered by status.

    Defaults to ``pending`` if no filter is supplied.
    """
    filter_val = status_filter or "pending"

    reviews = (
        db.query(ReviewModel)
        .options(joinedload(ReviewModel.screening))
        .filter(ReviewModel.status == filter_val)
        .order_by(ReviewModel.created_at.desc())
        .limit(limit)
        .all()
    )

    total_pending = (
        db.query(ReviewModel)
        .filter(ReviewModel.status == "pending")
        .count()
    )

    return {
        "reviews": [_serialise(r) for r in reviews],
        "total": len(reviews),
        "total_pending": total_pending,
    }


@router.post("/{review_id}/assign")
async def assign_reviewer(
    review_id: str,
    body: AssignRequest,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
    db: Session = Depends(get_db),
):
    """Assign (or reassign) a reviewer to a review."""
    review = db.query(ReviewModel).filter(ReviewModel.id == review_id).first()
    if not review:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")

    if review.status == "closed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Cannot assign a reviewer to a closed review",
        )

    review.reviewer = body.reviewer
    review.status = "reviewed"
    db.commit()
    db.refresh(review)

    logger.info(
        "Review %s assigned to %s by %s",
        review_id, body.reviewer, auth.user_id,
    )

    return _serialise(review)


@router.post("/{review_id}/comment")
async def add_comment(
    review_id: str,
    body: CommentRequest,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
    db: Session = Depends(get_db),
):
    """Append a comment to a review."""
    review = db.query(ReviewModel).filter(ReviewModel.id == review_id).first()
    if not review:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")

    # Append to existing comments (newline-separated log)
    existing = review.comments or ""
    prefix = f"[{auth.user_id}] "
    review.comments = (
        f"{existing}\n{prefix}{body.comments}".strip()
    )
    db.commit()
    db.refresh(review)

    logger.info("Comment added to review %s by %s", review_id, auth.user_id)
    return _serialise(review)


@router.post("/{review_id}/close")
async def close_review(
    review_id: str,
    body: Optional[CommentRequest] = None,
    auth: AuthResult = Depends(require_role("admin", "reviewer")),
    db: Session = Depends(get_db),
):
    """Close a review, optionally adding a final comment."""
    review = db.query(ReviewModel).filter(ReviewModel.id == review_id).first()
    if not review:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")

    if review.status == "closed":
        raise HTTPException(status.HTTP_409_CONFLICT, "Review is already closed")

    if body and body.comments:
        existing = review.comments or ""
        prefix = f"[{auth.user_id}] "
        review.comments = f"{existing}\n{prefix}{body.comments}".strip()

    review.status = "closed"
    review.reviewer = review.reviewer or auth.user_id
    db.commit()
    db.refresh(review)

    logger.info("Review %s closed by %s", review_id, auth.user_id)
    return _serialise(review)
