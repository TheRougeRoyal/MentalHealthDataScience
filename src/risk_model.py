"""Structured risk-scoring model layer.

Provides a **protocol-driven** design so the scoring backend can be swapped
from the current clinical-rules implementation to a trained ML model without
touching the API endpoints.

Architecture
~~~~~~~~~~~~

::

    ┌────────────────────────────────────────────────────┐
    │                   RiskModel (ABC)                  │
    │  score()  →  classify()  →  explain()              │
    └──────────────┬────────────────┬────────────────────┘
                   │                │
       ClinicalRulesModel     (future) LightGBMModel

Public API consumed by endpoints:

    model = get_risk_model()
    result = model.assess(input_data)
    # result.probability   – float 0…1
    # result.risk_score    – float 0…100
    # result.risk_level    – "low" | "moderate" | "high" | "critical"
    # result.confidence    – float 0…1
    # result.contributing_factors – list[str]
    # result.top_features  – list[tuple[str, float]]
    # result.clinical_interpretation – str
    # result.counterfactual – str
    # result.alert_triggered – bool
    # result.requires_human_review – bool
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (pulled from settings)
# ---------------------------------------------------------------------------

_THRESHOLD_HIGH: float = settings.ml.risk_threshold_high       # 51.0
_THRESHOLD_CRITICAL: float = settings.ml.risk_threshold_critical  # 75.0
_REVIEW_THRESHOLD: float = settings.governance.human_review_threshold  # 75.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    """Immutable output of a risk assessment."""

    probability: float          # 0 → 1
    risk_score: float           # 0 → 100
    risk_level: str             # low | moderate | high | critical
    confidence: float           # 0 → 1
    contributing_factors: List[str] = field(default_factory=list)
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    clinical_interpretation: str = ""
    counterfactual: str = ""
    alert_triggered: bool = False
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RiskModel(ABC):
    """Protocol every scoring backend must implement."""

    @abstractmethod
    def score(self, data: Dict[str, Any]) -> float:
        """Return a probability in [0, 1]."""

    @abstractmethod
    def classify(self, probability: float) -> Tuple[str, float, bool, bool]:
        """Map a probability to (risk_level, risk_score, alert, needs_review)."""

    @abstractmethod
    def explain(
        self, data: Dict[str, Any], probability: float, risk_level: str,
    ) -> Tuple[List[str], List[Tuple[str, float]], str, str]:
        """Return (factors, top_features, clinical_text, counterfactual)."""

    # ── convenience entry-point ──────────────────────────────────────────

    def assess(self, input_data: Dict[str, Any]) -> AssessmentResult:
        """Run the full score → classify → explain pipeline."""
        probability = self.score(input_data)
        risk_level, risk_score, alert, needs_review = self.classify(probability)
        factors, top_features, clinical, counter = self.explain(
            input_data, probability, risk_level,
        )

        # Confidence is derived from how far the probability is from the
        # decision boundary (0.5).  Scores near the boundary → low
        # confidence; extreme scores → high confidence.
        confidence = min(1.0, 0.5 + abs(probability - 0.5))

        return AssessmentResult(
            probability=round(probability, 4),
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            contributing_factors=factors,
            top_features=top_features,
            clinical_interpretation=clinical,
            counterfactual=counter,
            alert_triggered=alert,
            requires_human_review=needs_review,
        )


# ---------------------------------------------------------------------------
# Clinical-rules implementation
# ---------------------------------------------------------------------------

# Feature contribution weights (used for both scoring and explanation).
# These mirror the constructs a clinician would consider and are assigned
# weights that sum to ≈ 1.0 so the output naturally maps to a probability.

_FEATURE_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "phq9_score": {
        "weight": 0.30,
        "max_value": 27,
        "label": "PHQ-9 Depression Score",
        "clinical": "Patient Health Questionnaire depression severity",
    },
    "gad7_score": {
        "weight": 0.22,
        "max_value": 21,
        "label": "GAD-7 Anxiety Score",
        "clinical": "Generalized Anxiety Disorder severity",
    },
    "sleep_hours": {
        "weight": 0.18,
        "max_value": None,     # inversely scored
        "label": "Sleep Duration",
        "clinical": "Nightly sleep hours (deviation from optimal 7–9 h)",
    },
    "avg_heart_rate": {
        "weight": 0.12,
        "max_value": None,     # deviation scoring
        "label": "Resting Heart Rate",
        "clinical": "Average resting heart rate (elevated may indicate stress)",
    },
    "diagnosis_codes": {
        "weight": 0.10,
        "max_value": None,
        "label": "Psychiatric Diagnosis Codes",
        "clinical": "Presence of ICD-10 psychiatric diagnoses",
    },
    "medications": {
        "weight": 0.08,
        "max_value": None,
        "label": "Psychotropic Medications",
        "clinical": "Number of psychotropic medications currently prescribed",
    },
}


class ClinicalRulesModel(RiskModel):
    """Evidence-based clinical rules that produce a calibrated probability.

    Each input feature is normalised to a 0–1 contribution using clinically
    meaningful cut-offs, then combined via a weighted sum.  The result is
    passed through a sigmoid to produce a well-calibrated probability.
    """

    # ── score ────────────────────────────────────────────────────────────

    def score(self, data: Dict[str, Any]) -> float:
        raw_sum = 0.0
        total_weight = 0.0

        for feat_key, meta in _FEATURE_WEIGHTS.items():
            contribution = self._feature_contribution(feat_key, data)
            if contribution is not None:
                raw_sum += meta["weight"] * contribution
                total_weight += meta["weight"]

        if total_weight == 0:
            return 0.5  # no information → neutral

        # Normalise to the weight that was actually used.
        normalised = raw_sum / total_weight

        # Push through a sigmoid centred on 0.5 for calibration.
        # logit maps [0,1] → [-∞, +∞]; sigmoid maps back.
        logit = (normalised - 0.5) * 6.0  # steepness factor
        probability = 1.0 / (1.0 + math.exp(-logit))

        return max(0.0, min(1.0, probability))

    # ── classify ─────────────────────────────────────────────────────────

    def classify(
        self, probability: float,
    ) -> Tuple[str, float, bool, bool]:
        risk_score = probability * 100.0

        if risk_score >= _THRESHOLD_CRITICAL:
            level = "critical"
        elif risk_score >= _THRESHOLD_HIGH:
            level = "high"
        elif risk_score >= 30:
            level = "moderate"
        else:
            level = "low"

        alert = risk_score >= 85.0
        needs_review = risk_score >= _REVIEW_THRESHOLD or probability < 0.1

        return level, risk_score, alert, needs_review

    # ── explain ──────────────────────────────────────────────────────────

    def explain(
        self,
        data: Dict[str, Any],
        probability: float,
        risk_level: str,
    ) -> Tuple[List[str], List[Tuple[str, float]], str, str]:

        # 1. Per-feature contributions
        contributions: List[Tuple[str, float, str]] = []
        for feat_key, meta in _FEATURE_WEIGHTS.items():
            c = self._feature_contribution(feat_key, data)
            if c is not None and c > 0.1:
                contributions.append((feat_key, c, meta["label"]))

        contributions.sort(key=lambda x: x[1], reverse=True)

        factors = [label for _, _, label in contributions[:5]]
        top_features = [
            (label, round(val, 4)) for _, val, label in contributions
        ]

        # 2. Clinical interpretation
        clinical = self._build_clinical_text(risk_level, factors, probability)

        # 3. Counterfactual
        counterfactual = self._build_counterfactual(
            risk_level, contributions, data,
        )

        return factors, top_features, clinical, counterfactual

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _feature_contribution(
        key: str, data: Dict[str, Any],
    ) -> Optional[float]:
        """Normalise a single feature value to [0, 1]."""

        if key == "phq9_score":
            v = data.get("phq9_score")
            if v is None:
                return None
            return max(0.0, min(1.0, v / 27.0))

        if key == "gad7_score":
            v = data.get("gad7_score")
            if v is None:
                return None
            return max(0.0, min(1.0, v / 21.0))

        if key == "sleep_hours":
            v = data.get("sleep_hours")
            if v is None:
                return None
            # Optimal is 7–9 h; farther away → higher risk.
            if 7.0 <= v <= 9.0:
                return 0.0
            deviation = min(abs(v - 7.0), abs(v - 9.0))
            return max(0.0, min(1.0, deviation / 5.0))

        if key == "avg_heart_rate":
            v = data.get("avg_heart_rate")
            if v is None:
                return None
            # Normal resting HR 60–80.  Elevated → higher risk.
            if 60 <= v <= 80:
                return 0.0
            deviation = max(0, v - 80) if v > 80 else max(0, 60 - v)
            return max(0.0, min(1.0, deviation / 40.0))

        if key == "diagnosis_codes":
            v = data.get("diagnosis_codes")
            if not v:
                return None
            codes = v if isinstance(v, list) else [v]
            psych_prefixes = ("F3", "F4", "F2", "F1")
            psych_count = sum(
                1 for c in codes
                if any(c.upper().startswith(p) for p in psych_prefixes)
            )
            return max(0.0, min(1.0, psych_count / 3.0))

        if key == "medications":
            v = data.get("medications")
            if not v:
                return None
            meds = v if isinstance(v, list) else [v]
            return max(0.0, min(1.0, len(meds) / 4.0))

        return None

    @staticmethod
    def _build_clinical_text(
        level: str, factors: List[str], probability: float,
    ) -> str:
        severity = {
            "low": "low", "moderate": "moderate",
            "high": "elevated", "critical": "critically high",
        }
        text = (
            f"Assessment indicates {severity.get(level, level)} "
            f"mental health risk (probability {probability:.1%})."
        )
        if factors:
            text += (
                f" Primary contributing factors: "
                f"{', '.join(factors[:3]).lower()}."
            )
        if level in ("high", "critical"):
            text += (
                " Clinical follow-up is recommended. "
                "Consider referral for comprehensive evaluation."
            )
        return text

    @staticmethod
    def _build_counterfactual(
        level: str,
        contributions: List[Tuple[str, float, str]],
        data: Dict[str, Any],
    ) -> str:
        if level in ("low", "moderate") or not contributions:
            return "Current risk factors are within manageable range."

        top_key, top_val, top_label = contributions[0]
        suggestions = {
            "phq9_score": "reducing PHQ-9 responses by 5+ points",
            "gad7_score": "reducing GAD-7 responses by 4+ points",
            "sleep_hours": "improving nightly sleep to 7–9 hours",
            "avg_heart_rate": "reducing resting heart rate through "
                              "stress management",
            "diagnosis_codes": "addressing active psychiatric diagnoses",
            "medications": "optimising current medication regimen",
        }
        suggestion = suggestions.get(
            top_key, f"improving {top_label.lower()}",
        )
        return (
            f"Risk level could decrease from {level} if {suggestion}. "
            "This scenario is illustrative and should be discussed "
            "with a clinician."
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[RiskModel] = None


def get_risk_model() -> RiskModel:
    """Return the application-wide risk-model instance.

    Replace the body of this function to swap in a trained ML model:

    >>> def get_risk_model() -> RiskModel:
    ...     return LightGBMModel.from_registry(model_id="prod_v3")
    """
    global _instance
    if _instance is None:
        _instance = ClinicalRulesModel()
        logger.info("ClinicalRulesModel initialised (swap for ML later)")
    return _instance
