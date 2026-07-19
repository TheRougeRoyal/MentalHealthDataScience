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
       ClinicalRulesModel     OllamaRiskModel
       (scoring + classify)    (AI-generated explanations)

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

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

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
# Ollama-backed explanation model
# ---------------------------------------------------------------------------

class OllamaRiskModel(RiskModel):
    """Hybrid model: ClinicalRulesModel for scoring, Ollama for explanations.

    Uses the clinical rules engine for deterministic, calibrated scoring
    and classification, then sends the structured data + assessment to an
    Ollama API to generate natural-language clinical interpretations,
    contributing factors, and counterfactual scenarios.

    Falls back to the rules-engine explanations if Ollama is unreachable
    or returns invalid output.
    """

    _SYSTEM_PROMPT = (
        "You are a clinical decision-support assistant for a mental health "
        "risk assessment system. You generate concise, evidence-based "
        "clinical interpretations. You do NOT diagnose conditions. You "
        "provide decision-support text for licensed clinicians to review. "
        "All outputs must include a disclaimer that they are decision-support "
        "only and not a diagnosis. Respond ONLY with valid JSON."
    )

    _USER_PROMPT_TEMPLATE = (
        "Given this patient screening data and risk assessment, generate:\n"
        "1. \"factors\" — a list of the top 3-5 contributing clinical factors "
        "(short phrases, NOT full sentences)\n"
        "2. \"interpretation\" — a 2-3 sentence clinical interpretation "
        "suitable for a clinician's review. Must include a disclaimer.\n"
        "3. \"counterfactual\" — a 1-2 sentence description of what could "
        "reduce this patient's risk level. Must note it is illustrative.\n\n"
        "Patient data:\n{data}\n\n"
        "Risk score: {score:.1f}%\n"
        "Risk level: {level}\n"
        "Probability: {probability:.1%}\n\n"
        "Respond with JSON only: "
        '{{"factors": [...], "interpretation": "...", "counterfactual": "..."}}'
    )

    def __init__(self) -> None:
        self._rules = ClinicalRulesModel()
        self._base_url = settings.ml.ollama_base_url.rstrip("/")
        self._model = settings.ml.ollama_model
        self._api_key = settings.ml.ollama_api_key
        self._timeout = settings.ml.ollama_timeout

    # ── score / classify — delegated to rules engine ─────────────────────

    def score(self, data: Dict[str, Any]) -> float:
        return self._rules.score(data)

    def classify(self, probability: float) -> Tuple[str, float, bool, bool]:
        return self._rules.classify(probability)

    # ── explain — Ollama-backed ──────────────────────────────────────────

    def explain(
        self,
        data: Dict[str, Any],
        probability: float,
        risk_level: str,
    ) -> Tuple[List[str], List[Tuple[str, float]], str, str]:

        # Get rule-based features first (always available, used as fallback
        # and to provide top_features list regardless of LLM output)
        rule_factors, rule_features, rule_clinical, rule_counter = (
            self._rules.explain(data, probability, risk_level)
        )

        # Call Ollama for enhanced explanations
        try:
            llm_result = self._call_ollama(data, probability, risk_level)
            if llm_result:
                factors = llm_result.get("factors", rule_factors)
                if not factors:
                    factors = rule_factors
                clinical = llm_result.get("interpretation", rule_clinical)
                if not clinical:
                    clinical = rule_clinical
                counterfactual = llm_result.get("counterfactual", rule_counter)
                if not counterfactual:
                    counterfactual = rule_counter

                logger.info("Ollama explanation generated successfully")
                return factors, rule_features, clinical, counterfactual

        except Exception as e:
            logger.warning("Ollama explanation failed, using rules fallback: %s", e)

        return rule_factors, rule_features, rule_clinical, rule_counter

    # ── Ollama API call ──────────────────────────────────────────────────

    def _call_ollama(
        self,
        data: Dict[str, Any],
        probability: float,
        risk_level: str,
    ) -> Optional[Dict[str, Any]]:
        """Call Ollama API to generate clinical explanations."""
        prompt = self._USER_PROMPT_TEMPLATE.format(
            data=json.dumps(data, indent=2, default=str),
            score=probability * 100,
            level=risk_level,
            probability=probability,
        )

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": self._SYSTEM_PROMPT,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }

        resp = httpx.post(
            f"{self._base_url}/api/generate",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()

        response_text = resp.json().get("response", "")
        if not response_text:
            logger.warning("Ollama returned empty response")
            return None

        return self._parse_llm_json(response_text)

    @staticmethod
    def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
        """Robustly extract JSON from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fence
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    continue

        # Try to find first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM JSON response")
        return None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[RiskModel] = None


def get_risk_model() -> RiskModel:
    """Return the application-wide risk-model instance.

    Uses OllamaRiskModel for AI-generated clinical explanations with
    ClinicalRulesModel for deterministic scoring and classification.
    Falls back to ClinicalRulesModel only if Ollama is not configured.
    """
    global _instance
    if _instance is None:
        if settings.ml.ollama_api_key or settings.ml.ollama_base_url != "http://localhost:11434":
            _instance = OllamaRiskModel()
            logger.info(
                "OllamaRiskModel initialised (model=%s, url=%s)",
                settings.ml.ollama_model, settings.ml.ollama_base_url,
            )
        else:
            _instance = ClinicalRulesModel()
            logger.info("ClinicalRulesModel initialised (Ollama not configured)")
    return _instance
