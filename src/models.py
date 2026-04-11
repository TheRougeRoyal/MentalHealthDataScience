"""SQLAlchemy ORM models for MHRAS.

Defines three core tables:

* **screenings** – each risk-screening result.
* **explanations** – interpretability artefacts tied to a screening.
* **reviews** – clinical-review workflow state tied to a screening.

All primary keys are UUIDs stored as ``CHAR(36)`` on SQLite and native
``UUID`` on PostgreSQL.  JSON columns use ``JSONB`` on PostgreSQL and
plain ``JSON`` on SQLite.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from src.database import Base

# ---------------------------------------------------------------------------
# Portable column types
# ---------------------------------------------------------------------------

# UUID: native on PostgreSQL, CHAR(36) on SQLite.
UUIDType = PG_UUID(as_uuid=False).with_variant(String(36), "sqlite")

# JSON: JSONB on PostgreSQL (indexable), plain JSON elsewhere.
JSONType = JSON().with_variant(JSONB, "postgresql")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REVIEW_STATUSES = frozenset({"pending", "reviewed", "closed"})


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Screening(Base):
    """A single risk-screening result and its raw input payload."""

    __tablename__ = "screenings"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_new_uuid
    )
    anonymized_id: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )
    input_data: Mapped[dict] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # -- Relationships -------------------------------------------------------

    explanations: Mapped[List["Explanation"]] = relationship(
        back_populates="screening",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Explanation.created_at",
    )
    reviews: Mapped[List["Review"]] = relationship(
        back_populates="screening",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Review.created_at.desc()",
    )

    # -- Constraints ---------------------------------------------------------

    __table_args__ = (
        CheckConstraint(
            "risk_score >= 0.0 AND risk_score <= 100.0",
            name="ck_screenings_risk_score_range",
        ),
        Index(
            "ix_screenings_anonymized_id_created_at",
            "anonymized_id",
            "created_at",
        ),
    )

    # -- Validation ----------------------------------------------------------

    @validates("risk_score")
    def _validate_risk_score(self, _key: str, value: float) -> float:
        if value is None or not (0.0 <= value <= 100.0):
            raise ValueError("risk_score must be between 0 and 100")
        return value

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Screening id={self.id!r} anonymized_id={self.anonymized_id!r} "
            f"risk_level={self.risk_level!r}>"
        )


class Explanation(Base):
    """Interpretability artefact generated for a screening."""

    __tablename__ = "explanations"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_new_uuid
    )
    screening_id: Mapped[str] = mapped_column(
        UUIDType,
        ForeignKey("screenings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    explanation_text: Mapped[str] = mapped_column(Text, nullable=False)
    factors: Mapped[dict] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # -- Relationships -------------------------------------------------------

    screening: Mapped["Screening"] = relationship(
        back_populates="explanations"
    )

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Explanation id={self.id!r} screening_id={self.screening_id!r}>"
        )


class Review(Base):
    """Clinical-review workflow record for a screening."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(
        UUIDType, primary_key=True, default=_new_uuid
    )
    screening_id: Mapped[str] = mapped_column(
        UUIDType,
        ForeignKey("screenings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True
    )
    reviewer: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # -- Relationships -------------------------------------------------------

    screening: Mapped["Screening"] = relationship(back_populates="reviews")

    # -- Constraints ---------------------------------------------------------

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'reviewed', 'closed')",
            name="ck_reviews_valid_status",
        ),
    )

    # -- Validation ----------------------------------------------------------

    @validates("status")
    def _validate_status(self, _key: str, value: str) -> str:
        if value not in _REVIEW_STATUSES:
            raise ValueError(
                f"status must be one of: {', '.join(sorted(_REVIEW_STATUSES))}"
            )
        return value

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Review id={self.id!r} screening_id={self.screening_id!r} "
            f"status={self.status!r}>"
        )
