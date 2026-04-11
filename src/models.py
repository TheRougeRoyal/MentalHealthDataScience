"""SQLAlchemy ORM models for MHRAS database."""

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, Float, ForeignKey, Index, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship, validates

from src.database import Base


UUID_TYPE = PGUUID(as_uuid=False).with_variant(String(36), "sqlite")
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Screening(Base):
    """Stores each screening output and original structured payload."""

    __tablename__ = "screenings"

    id = Column(UUID_TYPE, primary_key=True, default=lambda: str(uuid.uuid4()))
    anonymized_id = Column(String(128), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(32), nullable=False, index=True)
    input_data = Column(JSON_TYPE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    explanations = relationship(
        "Explanation",
        back_populates="screening",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    reviews = relationship(
        "Review",
        back_populates="screening",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint("risk_score >= 0.0 AND risk_score <= 100.0", name="ck_screenings_risk_score_range"),
        Index("ix_screenings_anonymized_id_created_at", "anonymized_id", "created_at"),
    )

    @validates("risk_score")
    def validate_risk_score(self, key, value):
        if value is None or value < 0.0 or value > 100.0:
            raise ValueError("risk_score must be between 0 and 100")
        return value


class Explanation(Base):
    """Stores explanation artifacts generated for a screening."""

    __tablename__ = "explanations"

    id = Column(UUID_TYPE, primary_key=True, default=lambda: str(uuid.uuid4()))
    screening_id = Column(UUID_TYPE, ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False, index=True)
    explanation_text = Column(Text, nullable=False)
    factors = Column(JSON_TYPE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    screening = relationship("Screening", back_populates="explanations")


class Review(Base):
    """Tracks review workflow state for a screening."""

    __tablename__ = "reviews"

    id = Column(UUID_TYPE, primary_key=True, default=lambda: str(uuid.uuid4()))
    screening_id = Column(UUID_TYPE, ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(16), nullable=False, default="pending", index=True)
    reviewer = Column(String(128), nullable=True)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    screening = relationship("Screening", back_populates="reviews")

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'reviewed', 'closed')", name="ck_reviews_valid_status"),
    )

    @validates("status")
    def validate_status(self, key, value):
        allowed = {"pending", "reviewed", "closed"}
        if value not in allowed:
            raise ValueError(f"status must be one of: {', '.join(sorted(allowed))}")
        return value
