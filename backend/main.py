"""
Mental Health Risk Assessment System (MHRAS) — Backend API
===========================================================
Single-file FastAPI application. Pure-Python clinical rules model (no ML libs).
Deployed to Railway; frontend on Vercel at https://mental-health-data-science.vercel.app/
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports + logging
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mhras")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Config from env vars
# ──────────────────────────────────────────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mhras.db")
JWT_SECRET = os.getenv("SECURITY_JWT_SECRET", "dev-secret-change-me-in-production")
JWT_ALGORITHM = os.getenv("SECURITY_JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

ML_RISK_THRESHOLD_HIGH = float(os.getenv("ML_RISK_THRESHOLD_HIGH", "51.0"))
ML_RISK_THRESHOLD_CRITICAL = float(os.getenv("ML_RISK_THRESHOLD_CRITICAL", "75.0"))
GOVERNANCE_HUMAN_REVIEW_THRESHOLD = float(
    os.getenv("GOVERNANCE_HUMAN_REVIEW_THRESHOLD", "75.0")
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. SQLAlchemy engine + session factory
# ──────────────────────────────────────────────────────────────────────────────
_db_url = DATABASE_URL
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

_connect_args: dict[str, Any] = {}
if _db_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(_db_url, connect_args=_connect_args, pool_pre_ping=True)

# SQLite-specific PRAGMAs
if _db_url.startswith("sqlite"):
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
# 4. ORM tables
# ──────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class Screening(Base):
    __tablename__ = "screenings"
    __table_args__ = (
        CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="ck_risk_score_range"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    anonymized_id = Column(String(255), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    input_data = Column(JSON, nullable=True)
    explanation_text = Column(Text, nullable=True)
    factors = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    reviews = relationship("Review", back_populates="screening")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'reviewed', 'closed')",
            name="ck_review_status",
        ),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    screening_id = Column(
        String(36), ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False
    )
    status = Column(String(20), nullable=False, default="pending")
    reviewer = Column(String(255), nullable=True)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    screening = relationship("Screening", back_populates="reviews")


# ──────────────────────────────────────────────────────────────────────────────
# 5. init_db
# ──────────────────────────────────────────────────────────────────────────────
def init_db():
    logger.info("Initializing database tables …")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Clinical Rules Model
# ──────────────────────────────────────────────────────────────────────────────
class ClinicalRulesModel:
    """Pure-Python weighted clinical rules risk model. No external ML libs."""

    WEIGHTS = {
        "phq9_score": 0.30,
        "gad7_score": 0.22,
        "sleep_hours": 0.18,
        "avg_heart_rate": 0.12,
        "diagnosis_codes": 0.10,
        "medications": 0.08,
    }
    SIGMOID_STEEPNESS = 6.0

    def _normalise_phq9(self, v: float) -> float:
        return min(v / 27.0, 1.0)

    def _normalise_gad7(self, v: float) -> float:
        return min(v / 21.0, 1.0)

    def _normalise_sleep(self, v: float) -> float:
        if 7.0 <= v <= 9.0:
            return 0.0
        if v < 7.0:
            return min((7.0 - v) / 5.0, 1.0)
        return min((v - 9.0) / 5.0, 1.0)

    def _normalise_hr(self, v: float) -> float:
        if 60.0 <= v <= 80.0:
            return 0.0
        if v < 60.0:
            return min((60.0 - v) / 40.0, 1.0)
        return min((v - 80.0) / 40.0, 1.0)

    def _normalise_diagnosis(self, codes: list[str]) -> float:
        relevant = sum(
            1
            for c in codes
            if any(c.upper().startswith(prefix) for prefix in ("F1", "F2", "F3", "F4"))
        )
        return min(relevant / 3.0, 1.0)

    def _normalise_medications(self, meds: list[str]) -> float:
        return min(len(meds) / 4.0, 1.0)

    @staticmethod
    def _sigmoid(x: float, steepness: float = 6.0) -> float:
        """Sigmoid mapping centred at 0.5."""
        return 1.0 / (1.0 + math.exp(-steepness * (x - 0.5)))

    def predict(
        self,
        survey_data: dict | None,
        wearable_data: dict | None,
        emr_data: dict | None,
    ) -> dict:
        """Return risk_score (0-100), risk_level, confidence, contributing_factors, component_scores."""
        survey = survey_data or {}
        wearable = wearable_data or {}
        emr = emr_data or {}

        component_scores: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # PHQ-9
        if "phq9_score" in survey:
            norm = self._normalise_phq9(float(survey["phq9_score"]))
            component_scores["phq9_score"] = norm
            weighted_sum += norm * self.WEIGHTS["phq9_score"]
            total_weight += self.WEIGHTS["phq9_score"]

        # GAD-7
        if "gad7_score" in survey:
            norm = self._normalise_gad7(float(survey["gad7_score"]))
            component_scores["gad7_score"] = norm
            weighted_sum += norm * self.WEIGHTS["gad7_score"]
            total_weight += self.WEIGHTS["gad7_score"]

        # Sleep hours
        if "sleep_hours" in wearable:
            norm = self._normalise_sleep(float(wearable["sleep_hours"]))
            component_scores["sleep_hours"] = norm
            weighted_sum += norm * self.WEIGHTS["sleep_hours"]
            total_weight += self.WEIGHTS["sleep_hours"]

        # Heart rate
        if "avg_heart_rate" in wearable:
            norm = self._normalise_hr(float(wearable["avg_heart_rate"]))
            component_scores["avg_heart_rate"] = norm
            weighted_sum += norm * self.WEIGHTS["avg_heart_rate"]
            total_weight += self.WEIGHTS["avg_heart_rate"]

        # Diagnosis codes
        if "diagnosis_codes" in emr and emr["diagnosis_codes"]:
            norm = self._normalise_diagnosis(emr["diagnosis_codes"])
            component_scores["diagnosis_codes"] = norm
            weighted_sum += norm * self.WEIGHTS["diagnosis_codes"]
            total_weight += self.WEIGHTS["diagnosis_codes"]

        # Medications
        if "medications" in emr and emr["medications"]:
            norm = self._normalise_medications(emr["medications"])
            component_scores["medications"] = norm
            weighted_sum += norm * self.WEIGHTS["medications"]
            total_weight += self.WEIGHTS["medications"]

        # Normalise by present weight, apply sigmoid, scale to 0-100
        if total_weight > 0:
            raw_norm = weighted_sum / total_weight
        else:
            raw_norm = 0.0

        risk_score = self._sigmoid(raw_norm, self.SIGMOID_STEEPNESS) * 100.0

        # Confidence: higher when more data sources present
        max_components = 6
        present = len(component_scores)
        confidence = 0.4 + 0.6 * (present / max_components)

        # Risk level
        if risk_score < 30:
            risk_level = "low"
        elif risk_score < ML_RISK_THRESHOLD_HIGH:
            risk_level = "moderate"
        elif risk_score < ML_RISK_THRESHOLD_CRITICAL:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Contributing factors — sorted by weighted contribution
        contributing_factors = []
        for feat, norm_val in sorted(component_scores.items(), key=lambda x: -x[1]):
            if norm_val > 0.1:
                label = feat.replace("_", " ").title()
                contributing_factors.append(f"{label}: elevated ({norm_val:.2f})")

        alert_triggered = risk_score >= 85.0
        requires_human_review = risk_score >= GOVERNANCE_HUMAN_REVIEW_THRESHOLD

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "contributing_factors": contributing_factors,
            "alert_triggered": alert_triggered,
            "requires_human_review": requires_human_review,
            "component_scores": component_scores,
        }

    def explain(
        self,
        component_scores: dict[str, float],
        risk_score: float,
        risk_level: str,
    ) -> dict:
        """Generate XAI explanations: top features, counterfactual, clinical interpretation."""
        # Top features as [feature_name, contribution_value] pairs
        top_features = sorted(
            [
                [feat.replace("_", " ").title(), round(val * self.WEIGHTS.get(feat, 0), 4)]
                for feat, val in component_scores.items()
            ],
            key=lambda x: -x[1],
        )

        # Clinical interpretation
        if risk_level == "critical":
            clinical_interpretation = (
                "The assessment indicates critical risk factors requiring immediate clinical attention. "
                "Multiple indicators suggest significant mental health distress. "
                "A comprehensive clinical evaluation and safety planning are strongly recommended."
            )
        elif risk_level == "high":
            clinical_interpretation = (
                "The assessment identifies elevated risk indicators that warrant prompt clinical review. "
                "Several contributing factors suggest emerging or worsening mental health concerns. "
                "A thorough evaluation by a qualified mental health professional is recommended."
            )
        elif risk_level == "moderate":
            clinical_interpretation = (
                "The assessment reveals moderate risk indicators. While not immediately critical, "
                "there are areas of concern that would benefit from monitoring and potentially "
                "preventive intervention. Consider connecting with mental health resources."
            )
        else:
            clinical_interpretation = (
                "The assessment suggests low current risk. Indicators are generally within "
                "normal ranges. Continued monitoring and wellness support are recommended "
                "as part of a proactive mental health strategy."
            )

        # Counterfactual
        if top_features:
            top_name = top_features[0][0]
            counterfactual = (
                f"If '{top_name}' were within normal range, the overall risk score "
                f"would likely decrease by approximately {top_features[0][1] * 100:.0f}% "
                f"of its weighted contribution, potentially moving the risk level down."
            )
        else:
            counterfactual = "Insufficient data to generate a what-if scenario."

        # Rule approximation
        rule_parts = []
        for feat_name, val in top_features[:3]:
            rule_parts.append(f"{feat_name} is elevated")
        if rule_parts:
            rule_approximation = (
                f"IF {' AND '.join(rule_parts)} THEN risk_level = {risk_level}"
            )
        else:
            rule_approximation = "Insufficient data for rule approximation."

        return {
            "top_features": top_features,
            "counterfactual": counterfactual,
            "rule_approximation": rule_approximation,
            "clinical_interpretation": clinical_interpretation,
        }


risk_model = ClinicalRulesModel()


# ──────────────────────────────────────────────────────────────────────────────
# 7. Recommendations Engine
# ──────────────────────────────────────────────────────────────────────────────
def generate_recommendations(
    risk_level: str, contributing_factors: list[str]
) -> list[dict]:
    """Pure-Python rule-based recommendations engine."""
    recs: list[dict] = []

    if risk_level in ("critical", "high"):
        recs.append(
            {
                "resource_type": "crisis_hotline",
                "name": "988 Suicide & Crisis Lifeline",
                "description": (
                    "Free, confidential 24/7 support for people in suicidal crisis "
                    "or emotional distress. Call or text 988."
                ),
                "contact_info": "988 (call or text)",
                "urgency": "immediate",
                "eligibility_criteria": "Anyone in the US experiencing crisis or distress",
            }
        )
        recs.append(
            {
                "resource_type": "therapy_referral",
                "name": "Professional Therapy Referral",
                "description": (
                    "Connect with a licensed mental health professional for individual "
                    "therapy sessions. Cognitive Behavioral Therapy (CBT) and Dialectical "
                    "Behavior Therapy (DBT) are evidence-based treatments."
                ),
                "contact_info": "Contact your primary care provider or insurance for referrals",
                "urgency": "high",
                "eligibility_criteria": "Individuals with elevated mental health risk indicators",
            }
        )
        recs.append(
            {
                "resource_type": "emergency_contact",
                "name": "Emergency Services",
                "description": (
                    "If you or someone you know is in immediate danger, contact emergency "
                    "services immediately. Trained professionals can provide immediate assistance."
                ),
                "contact_info": "911 (US) or local emergency number",
                "urgency": "immediate",
                "eligibility_criteria": "Anyone in immediate danger",
            }
        )

    if risk_level == "critical":
        recs.append(
            {
                "resource_type": "inpatient_referral",
                "name": "Psychiatric Emergency Evaluation",
                "description": (
                    "Given the critical risk level, a psychiatric emergency evaluation is "
                    "recommended to assess the need for intensive treatment or stabilisation."
                ),
                "contact_info": "Nearest emergency department or psychiatric facility",
                "urgency": "immediate",
                "eligibility_criteria": "Individuals presenting with critical mental health risk",
            }
        )

    if risk_level == "moderate":
        recs.append(
            {
                "resource_type": "therapy_referral",
                "name": "Counseling Services",
                "description": (
                    "Consider scheduling an appointment with a therapist or counselor. "
                    "Early intervention can prevent escalation of mental health concerns."
                ),
                "contact_info": "SAMHSA helpline: 1-800-662-4357",
                "urgency": "moderate",
                "eligibility_criteria": "Individuals experiencing moderate mental health symptoms",
            }
        )
        recs.append(
            {
                "resource_type": "self_help",
                "name": "Mental Health Self-Help Apps",
                "description": (
                    "Evidence-based apps like Woebot, Headspace, and Calm offer guided "
                    "exercises for anxiety, depression, and stress management."
                ),
                "contact_info": "Available on iOS and Android app stores",
                "urgency": "moderate",
                "eligibility_criteria": "Anyone seeking mental wellness support",
            }
        )
        recs.append(
            {
                "resource_type": "support_group",
                "name": "Peer Support Groups",
                "description": (
                    "NAMI (National Alliance on Mental Illness) offers free peer-led "
                    "support groups. Connecting with others who share similar experiences "
                    "can provide comfort and practical coping strategies."
                ),
                "contact_info": "nami.org/Support-Education/Support-Groups",
                "urgency": "moderate",
                "eligibility_criteria": "Open to anyone affected by mental health conditions",
            }
        )

    if risk_level == "low":
        recs.append(
            {
                "resource_type": "wellness",
                "name": "Wellness & Prevention Resources",
                "description": (
                    "Maintain your mental well-being with regular exercise, balanced nutrition, "
                    "adequate sleep, and social connection. These are foundational to mental health."
                ),
                "contact_info": "mentalhealth.gov",
                "urgency": "low",
                "eligibility_criteria": "Everyone",
            }
        )
        recs.append(
            {
                "resource_type": "self_help",
                "name": "Mindfulness & Meditation Apps",
                "description": (
                    "Apps like Headspace, Calm, and Insight Timer offer guided meditation "
                    "and mindfulness exercises to support ongoing mental wellness."
                ),
                "contact_info": "Available on iOS and Android app stores",
                "urgency": "low",
                "eligibility_criteria": "Anyone interested in proactive mental wellness",
            }
        )
        recs.append(
            {
                "resource_type": "wellness",
                "name": "Regular Mental Health Check-ins",
                "description": (
                    "Schedule periodic mental health screenings with your healthcare provider "
                    "to catch any emerging concerns early."
                ),
                "contact_info": "Your primary care provider",
                "urgency": "low",
                "eligibility_criteria": "Everyone",
            }
        )

    # Factor-specific additions
    factor_text = " ".join(contributing_factors).lower()
    if "sleep" in factor_text and risk_level in ("moderate", "high", "critical"):
        recs.append(
            {
                "resource_type": "self_help",
                "name": "Sleep Hygiene Program",
                "description": (
                    "Poor sleep is a significant contributor to your risk score. "
                    "Consider CBT for Insomnia (CBT-I), which is the gold-standard "
                    "non-pharmacological treatment for sleep difficulties."
                ),
                "contact_info": "Ask your provider about CBT-I referrals",
                "urgency": "moderate",
                "eligibility_criteria": "Individuals experiencing sleep difficulties",
            }
        )

    return recs


# ──────────────────────────────────────────────────────────────────────────────
# 8. JWT helpers
# ──────────────────────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode["type"] = "access"
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(
        days=REFRESH_TOKEN_EXPIRE_DAYS
    )
    to_encode["type"] = "refresh"
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


COOKIE_SECURE = ENVIRONMENT != "development"
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"


def _set_auth_cookies(response: Response, user_id: str, role: str):
    access = create_access_token({"sub": user_id, "role": role})
    refresh = create_refresh_token({"sub": user_id, "role": role})

    response.set_cookie(
        "access_token",
        access,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        "refresh_token",
        refresh,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/",
    )


def _clear_auth_cookies(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Hardcoded users (replace with DB in production)
# ──────────────────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pre-hash passwords at module load
USERS: dict[str, dict] = {
    "admin": {
        "user_id": "admin",
        "password_hash": pwd_context.hash("admin"),
        "role": "admin",
        "display_name": "Administrator",
    },
    "reviewer": {
        "user_id": "reviewer",
        "password_hash": pwd_context.hash("reviewer"),
        "role": "reviewer",
        "display_name": "Clinical Reviewer",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 10. Auth dependencies
# ──────────────────────────────────────────────────────────────────────────────
class AuthResult(BaseModel):
    user_id: str
    role: str
    display_name: str


_DEV_ADMIN = AuthResult(user_id="admin", role="admin", display_name="Dev Admin")


def get_current_user(
    access_token: Optional[str] = Cookie(default=None),
) -> AuthResult:
    if access_token:
        try:
            payload = decode_token(access_token)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            user_id = payload.get("sub")
            role = payload.get("role", "viewer")
            user_rec = USERS.get(user_id, {})
            display_name = user_rec.get("display_name", user_id)
            return AuthResult(user_id=user_id, role=role, display_name=display_name)
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Development fallback — auto-authenticate as admin
    if ENVIRONMENT == "development":
        return _DEV_ADMIN

    raise HTTPException(status_code=401, detail="Not authenticated")


# ──────────────────────────────────────────────────────────────────────────────
# 11. Role dependency factory
# ──────────────────────────────────────────────────────────────────────────────
def require_role(*roles: str):
    def _checker(current_user: AuthResult = Depends(get_current_user)) -> AuthResult:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{current_user.role}' not in required roles {roles}",
            )
        return current_user

    return Depends(_checker)


# ──────────────────────────────────────────────────────────────────────────────
# 12. FastAPI app with lifespan
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MHRAS starting up (env=%s) …", ENVIRONMENT)
    init_db()
    logger.info("Startup complete.")
    yield
    logger.info("MHRAS shutting down.")


app = FastAPI(
    title="Mental Health Risk Assessment System",
    version="1.0.0",
    description="Backend API for MHRAS — clinical risk screening & review",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────────────────────────────────────
# 13. CORS
# ──────────────────────────────────────────────────────────────────────────────
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    _cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# 14. Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str


# --- Screening ---
class SurveyData(BaseModel):
    phq9_score: Optional[int] = Field(None, ge=0, le=27)
    gad7_score: Optional[int] = Field(None, ge=0, le=21)


class WearableData(BaseModel):
    avg_heart_rate: Optional[float] = None
    sleep_hours: Optional[float] = None


class EMRData(BaseModel):
    diagnosis_codes: Optional[list[str]] = None
    medications: Optional[list[str]] = None


class ScreenRequest(BaseModel):
    anonymized_id: str
    consent_verified: bool
    timestamp: Optional[str] = None
    survey_data: Optional[SurveyData] = None
    wearable_data: Optional[WearableData] = None
    emr_data: Optional[EMRData] = None


class BatchScreenRequest(BaseModel):
    requests: list[ScreenRequest] = Field(..., max_length=100)


# --- Reviews ---
class AssignRequest(BaseModel):
    reviewer: str


class CommentRequest(BaseModel):
    comments: str


# ──────────────────────────────────────────────────────────────────────────────
# 14. Endpoints
# ──────────────────────────────────────────────────────────────────────────────

# --- Health ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "mode": "demo" if ENVIRONMENT == "development" else "production",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --- Auth ---
@app.post("/auth/login")
def auth_login(body: LoginRequest, response: Response):
    user = USERS.get(body.username)
    if not user or not pwd_context.verify(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    _set_auth_cookies(response, user["user_id"], user["role"])

    return {
        "user_id": user["user_id"],
        "role": user["role"],
        "display_name": user["display_name"],
    }


@app.post("/auth/logout")
def auth_logout(response: Response):
    _clear_auth_cookies(response)
    return {"status": "logged_out"}


@app.post("/auth/refresh")
def auth_refresh(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None),
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload["sub"]
        role = payload.get("role", "viewer")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    _set_auth_cookies(response, user_id, role)
    user_rec = USERS.get(user_id, {})
    return {
        "user_id": user_id,
        "role": role,
        "display_name": user_rec.get("display_name", user_id),
    }


@app.get("/auth/me")
def auth_me(current_user: AuthResult = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "role": current_user.role,
        "display_name": current_user.display_name,
    }


# --- Statistics ---
@app.get("/statistics")
def statistics(db: Session = Depends(get_db)):
    try:
        total = db.query(func.count(Screening.id)).scalar() or 0
        avg_score = db.query(func.avg(Screening.risk_score)).scalar() or 0.0
        high_risk_count = (
            db.query(func.count(Screening.id))
            .filter(Screening.risk_level.in_(["high", "critical"]))
            .scalar()
            or 0
        )
        high_risk_pct = (high_risk_count / total * 100) if total > 0 else 0.0
        pending_count = (
            db.query(func.count(Review.id))
            .filter(Review.status == "pending")
            .scalar()
            or 0
        )

        return {
            "screenings": {
                "total": total,
                "avg_risk_score": round(avg_score, 2),
                "high_risk_count": high_risk_count,
                "high_risk_pct": round(high_risk_pct, 2),
            },
            "review_queue": {
                "pending_count": pending_count,
            },
        }
    except Exception as exc:
        logger.error("Statistics error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


# --- Screen ---
def _run_screening(req: ScreenRequest, db: Session) -> dict:
    """Execute a single screening and persist to DB. Returns full response dict."""
    if not req.consent_verified:
        raise HTTPException(status_code=400, detail="Patient consent must be verified")

    # Prepare data dicts
    survey = req.survey_data.model_dump(exclude_none=True) if req.survey_data else {}
    wearable = req.wearable_data.model_dump(exclude_none=True) if req.wearable_data else {}
    emr = req.emr_data.model_dump(exclude_none=True) if req.emr_data else {}

    # Run model
    result = risk_model.predict(survey, wearable, emr)
    explanations = risk_model.explain(
        result["component_scores"], result["risk_score"], result["risk_level"]
    )
    recommendations = generate_recommendations(
        result["risk_level"], result["contributing_factors"]
    )

    ts = req.timestamp or datetime.now(timezone.utc).isoformat()

    # Persist screening
    screening = Screening(
        anonymized_id=req.anonymized_id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        input_data={"survey_data": survey, "wearable_data": wearable, "emr_data": emr},
        explanation_text=explanations.get("clinical_interpretation", ""),
        factors=result["contributing_factors"],
    )
    db.add(screening)
    db.flush()

    # Auto-create review if human review required
    if result["requires_human_review"]:
        review = Review(screening_id=screening.id, status="pending")
        db.add(review)

    db.commit()

    return {
        "risk_score": {
            "anonymized_id": req.anonymized_id,
            "score": result["risk_score"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "contributing_factors": result["contributing_factors"],
            "timestamp": ts,
        },
        "recommendations": recommendations,
        "explanations": explanations,
        "requires_human_review": result["requires_human_review"],
        "alert_triggered": result["alert_triggered"],
    }


@app.post("/screen")
def screen(req: ScreenRequest, db: Session = Depends(get_db)):
    try:
        return _run_screening(req, db)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Screening error: %s", exc, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(exc)}")


@app.post("/batch-screen")
def batch_screen(body: BatchScreenRequest, db: Session = Depends(get_db)):
    if len(body.requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")

    results = []
    successful = 0
    failed = 0

    for req in body.requests:
        try:
            result = _run_screening(req, db)
            results.append(result)
            successful += 1
        except Exception as exc:
            logger.warning("Batch item failed (id=%s): %s", req.anonymized_id, exc)
            results.append(
                {
                    "risk_score": {
                        "anonymized_id": req.anonymized_id,
                        "score": 0,
                        "risk_level": "unknown",
                        "confidence": 0,
                        "contributing_factors": [],
                        "timestamp": req.timestamp
                        or datetime.now(timezone.utc).isoformat(),
                    },
                    "recommendations": [],
                    "explanations": {
                        "top_features": [],
                        "counterfactual": "",
                        "rule_approximation": "",
                        "clinical_interpretation": f"Screening failed: {str(exc)}",
                    },
                    "requires_human_review": False,
                    "alert_triggered": False,
                    "error": str(exc),
                }
            )
            failed += 1

    return {
        "results": results,
        "total": len(body.requests),
        "successful": successful,
        "failed": failed,
    }


# --- Reviews ---
@app.get("/reviews/queue")
def review_queue(
    status_filter: str = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: AuthResult = Depends(get_current_user),
):
    try:
        query = (
            db.query(Review)
            .join(Screening, Review.screening_id == Screening.id)
        )

        if status_filter != "all":
            query = query.filter(Review.status == status_filter)

        query = query.order_by(Review.created_at.desc()).limit(limit)
        reviews = query.all()

        total_pending = (
            db.query(func.count(Review.id))
            .filter(Review.status == "pending")
            .scalar()
            or 0
        )

        review_list = []
        for r in reviews:
            screening = r.screening
            review_list.append(
                {
                    "id": r.id,
                    "screening_id": r.screening_id,
                    "status": r.status,
                    "reviewer": r.reviewer,
                    "comments": r.comments,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "anonymized_id": screening.anonymized_id if screening else None,
                    "risk_score": screening.risk_score if screening else None,
                    "risk_level": screening.risk_level if screening else None,
                }
            )

        return {
            "reviews": review_list,
            "total_pending": total_pending,
        }
    except Exception as exc:
        logger.error("Review queue error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load review queue")


@app.post("/reviews/{review_id}/assign")
def assign_review(
    review_id: str,
    body: AssignRequest,
    db: Session = Depends(get_db),
    current_user: AuthResult = Depends(get_current_user),
):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    review.reviewer = body.reviewer
    if review.status == "pending":
        review.status = "reviewed"
    db.commit()

    return {"status": "assigned", "reviewer": body.reviewer, "review_id": review_id}


@app.post("/reviews/{review_id}/comment")
def comment_review(
    review_id: str,
    body: CommentRequest,
    db: Session = Depends(get_db),
    current_user: AuthResult = Depends(get_current_user),
):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    # Append comment if existing
    if review.comments:
        review.comments = review.comments + "\n---\n" + body.comments
    else:
        review.comments = body.comments

    if review.status == "pending":
        review.status = "reviewed"
    db.commit()

    return {"status": "commented", "review_id": review_id}


@app.post("/reviews/{review_id}/close")
def close_review(
    review_id: str,
    db: Session = Depends(get_db),
    current_user: AuthResult = Depends(get_current_user),
):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    review.status = "closed"
    db.commit()

    return {"status": "closed", "review_id": review_id}


# ──────────────────────────────────────────────────────────────────────────────
# 15. Main entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
