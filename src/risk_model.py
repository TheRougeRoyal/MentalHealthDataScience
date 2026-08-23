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
    risk_level: str             # insufficient_data | low | moderate | high | critical
    confidence: float           # 0 → 1
    contributing_factors: List[str] = field(default_factory=list)
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    clinical_interpretation: str = ""
    counterfactual: str = ""
    alert_triggered: bool = False
    requires_human_review: bool = False
    model_version: str = settings.ml.model_version
    confidence_method: str = "heuristic_uncalibrated"


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
        if not any(input_data.get(key) is not None for key in _FEATURE_WEIGHTS):
            return AssessmentResult(
                probability=0.0,
                risk_score=0.0,
                risk_level="insufficient_data",
                confidence=0.0,
                contributing_factors=[],
                clinical_interpretation=(
                    "Insufficient structured data for a risk assessment. "
                    "This result is not a clinical determination."
                ),
                requires_human_review=True,
            )

        probability = self.score(input_data)
        risk_level, risk_score, alert, needs_review = self.classify(probability)
        factors, top_features, clinical, counter = self.explain(
            input_data, probability, risk_level,
        )

        # ── Confidence calibration ───────────────────────────────────────
        # Combines three signals:
        #   1. Boundary distance — scores near 0.5 are inherently uncertain
        #   2. Data completeness — more features → higher confidence
        #   3. Feature availability — critical features boost confidence

        boundary_confidence = min(1.0, 0.5 + abs(probability - 0.5))

        features_present = sum(
            1 for key in _FEATURE_WEIGHTS if input_data.get(key) is not None
        )
        completeness_confidence = features_present / len(_FEATURE_WEIGHTS)

        # Critical features (PHQ-9, GAD-7) carry extra weight for confidence
        critical_features = {"phq9_score", "gad7_score"}
        critical_present = sum(
            1 for f in critical_features if input_data.get(f) is not None
        )
        critical_confidence = 0.7 + 0.3 * (critical_present / len(critical_features))

        # Weighted combination
        confidence = (
            0.40 * boundary_confidence
            + 0.35 * completeness_confidence
            + 0.25 * critical_confidence
        )
        confidence = max(0.1, min(1.0, confidence))

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
        "max_value": None,
        "label": "Sleep Duration",
        "clinical": "Nightly sleep hours (deviation from optimal 7–9 h)",
    },
    "avg_heart_rate": {
        "weight": 0.12,
        "max_value": None,
        "label": "Resting Heart Rate",
        "clinical": "Average resting heart rate (elevated may indicate stress)",
    },
    "diagnosis_codes": {
        "weight": 0.10,
        "max_value": None,
        "label": "Psychiatric Diagnosis Codes",
        "clinical": "Presence and severity of ICD-10 psychiatric diagnoses",
    },
    "medications": {
        "weight": 0.08,
        "max_value": None,
        "label": "Psychotropic Medications",
        "clinical": "Number and class of psychotropic medications currently prescribed",
    },
}

# ── Feature interaction terms ─────────────────────────────────────────────
# These capture clinically meaningful compounding effects between features.
# Each interaction has a weight (added to the raw sum after normalisation)
# and a minimum contribution threshold — both features must exceed it.

_INTERACTION_TERMS: List[Dict[str, Any]] = [
    {
        "name": "depression_sleep_compound",
        "label": "Depression + Sleep Disturbance Compound",
        "features": ["phq9_score", "sleep_hours"],
        "thresholds": [0.4, 0.15],   # PHQ-9 normalised > 0.4 AND sleep deviation > 0.15
        "multiplier": 0.12,
    },
    {
        "name": "anxiety_hr_compound",
        "label": "Anxiety + Elevated Heart Rate Compound",
        "features": ["gad7_score", "avg_heart_rate"],
        "thresholds": [0.4, 0.15],
        "multiplier": 0.18,
    },
    {
        "name": "diagnosis_medication_compound",
        "label": "Multiple Diagnoses + Medication Burden",
        "features": ["diagnosis_codes", "medications"],
        "thresholds": [0.3, 0.25],
        "multiplier": 0.08,
    },
    {
        "name": "depression_anxiety_compound",
        "label": "Comorbid Depression + Anxiety",
        "features": ["phq9_score", "gad7_score"],
        "thresholds": [0.5, 0.5],
        "multiplier": 0.15,
    },
]


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
        contributions: Dict[str, float] = {}

        for feat_key, meta in _FEATURE_WEIGHTS.items():
            contribution = self._feature_contribution(feat_key, data)
            if contribution is not None:
                weighted = meta["weight"] * contribution
                raw_sum += weighted
                total_weight += meta["weight"]
                contributions[feat_key] = contribution

        # ── Feature interactions ──────────────────────────────────────────
        interaction_bonus = 0.0
        triggered_interactions: List[str] = []
        for interaction in _INTERACTION_TERMS:
            feature_names = interaction["features"]
            thresholds = interaction["thresholds"]
            if all(f in contributions for f in feature_names):
                if all(
                    contributions[f] >= t
                    for f, t in zip(feature_names, thresholds)
                ):
                    interaction_bonus += interaction["multiplier"]
                    triggered_interactions.append(interaction["label"])

        if total_weight == 0:
            return 0.5  # no information → neutral

        # Normalise to the weight that was actually used.
        normalised = (raw_sum + interaction_bonus) / (total_weight + interaction_bonus)

        # ── Dynamic sigmoid ──────────────────────────────────────────────
        # Steepness adapts to data completeness: more features → steeper
        # (more decisive); fewer features → flatter (more uncertain).
        features_available = len(contributions)
        total_features = len(_FEATURE_WEIGHTS)
        completeness = features_available / total_features

        # Steepness ranges from 3.0 (minimal data) to 8.0 (full data)
        steepness = 3.0 + completeness * 5.0

        logit = (normalised - 0.5) * steepness
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
        """Normalise a single feature value to [0, 1] using clinical zones."""

        if key == "phq9_score":
            v = data.get("phq9_score")
            if v is None:
                return None
            # PHQ-9 clinical severity bands:
            #   0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 mod-severe, 20-27 severe
            # Non-linear mapping gives higher weight to severe range.
            if v <= 4:
                return v / 4.0 * 0.15          # 0 → 0.15
            if v <= 9:
                return 0.15 + (v - 4) / 5.0 * 0.20   # 0.15 → 0.35
            if v <= 14:
                return 0.35 + (v - 9) / 5.0 * 0.25    # 0.35 → 0.60
            if v <= 19:
                return 0.60 + (v - 14) / 5.0 * 0.20   # 0.60 → 0.80
            return 0.80 + (v - 19) / 8.0 * 0.20        # 0.80 → 1.00

        if key == "gad7_score":
            v = data.get("gad7_score")
            if v is None:
                return None
            # GAD-7 clinical severity bands:
            #   0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe
            if v <= 4:
                return v / 4.0 * 0.15
            if v <= 9:
                return 0.15 + (v - 4) / 5.0 * 0.25    # 0.15 → 0.40
            if v <= 14:
                return 0.40 + (v - 9) / 5.0 * 0.30    # 0.40 → 0.70
            return 0.70 + (v - 14) / 7.0 * 0.30        # 0.70 → 1.00

        if key == "sleep_hours":
            v = data.get("sleep_hours")
            if v is None:
                return None
            # Clinical sleep zones:
            #   <4 severe insomnia → high risk
            #   4-6 moderate insomnia → moderate risk
            #   6-7 mild deviation → low risk
            #   7-9 optimal → 0
            #   9-11 mild hypersomnia → low-moderate risk
            #   >11 severe hypersomnia → high risk
            if v < 4.0:
                return 0.85 + min(0.15, (4.0 - v) / 4.0 * 0.15)
            if v < 6.0:
                return 0.40 + (6.0 - v) / 2.0 * 0.45   # 0.40 → 0.85
            if v < 7.0:
                return 0.10 + (7.0 - v) / 1.0 * 0.30   # 0.10 → 0.40
            if v <= 9.0:
                return 0.0
            if v <= 11.0:
                return (v - 9.0) / 2.0 * 0.35           # 0 → 0.35
            return 0.35 + min(0.50, (v - 11.0) / 3.0 * 0.50)  # 0.35 → 0.85

        if key == "avg_heart_rate":
            v = data.get("avg_heart_rate")
            if v is None:
                return None
            # Clinical heart rate zones (resting, adults):
            #   <50 bradycardia → moderate risk
            #   50-60 low-normal → minimal
            #   60-80 normal → 0
            #   80-100 elevated → moderate risk
            #   100-120 tachycardia → high risk
            #   >120 severe tachycardia → very high risk
            if v < 50:
                return 0.30 + min(0.40, (50 - v) / 20.0 * 0.40)
            if v < 60:
                return (60 - v) / 10.0 * 0.30           # 0 → 0.30
            if v <= 80:
                return 0.0
            if v <= 100:
                return (v - 80) / 20.0 * 0.40           # 0 → 0.40
            if v <= 120:
                return 0.40 + (v - 100) / 20.0 * 0.35   # 0.40 → 0.75
            return 0.75 + min(0.25, (v - 120) / 30.0 * 0.25)  # 0.75 → 1.00

        if key == "diagnosis_codes":
            v = data.get("diagnosis_codes")
            if not v:
                return None
            codes = v if isinstance(v, list) else [v]
            # Weight diagnoses by clinical severity using ICD-10 groups:
            #   F2x schizophrenia spectrum → 1.0 per code (highest severity)
            #   F3x mood disorders → 0.8 per code
            #   F4x anxiety/PTSD → 0.7 per code
            #   F1x substance use → 0.6 per code
            #   Other psych → 0.3 per code
            severity_map = {
                "F2": 1.0,
                "F3": 0.8,
                "F4": 0.7,
                "F1": 0.6,
            }
            total_severity = 0.0
            for c in codes:
                upper = c.upper().strip()
                matched = False
                for prefix, sev in severity_map.items():
                    if upper.startswith(prefix):
                        total_severity += sev
                        matched = True
                        break
                if not matched:
                    total_severity += 0.3
            # 1 high-severity diagnosis → 0.80; 3+ → 1.0
            return max(0.0, min(1.0, total_severity / 3.5))

        if key == "medications":
            v = data.get("medications")
            if not v:
                return None
            meds = v if isinstance(v, list) else [v]
            # Weight by psychotropic class severity:
            #   Antipsychotics (risperidone, olanzapine, quetiapine, haloperidol) → 1.0
            #   Mood stabilizers (lithium, valproate, lamotrigine) → 0.85
            #   Anxiolytics (lorazepam, alprazolam, diazepam) → 0.7
            #   Antidepressants (sertraline, fluoxetine, etc.) → 0.5
            #   Other → 0.3
            antipsychotics = {"risperidone", "olanzapine", "quetiapine", "haloperidol",
                              "aripiprazole", "clozapine", "ziprasidone", "paliperidone"}
            mood_stabilizers = {"lithium", "valproate", "valproic acid", "lamotrigine",
                                "carbamazepine", "oxcarbazepine"}
            anxiolytics = {"lorazepam", "alprazolam", "diazepam", "clonazepam",
                           "temazepam", "midazolam"}
            antidepressants = {"sertraline", "fluoxetine", "escitalopram", "citalopram",
                               "paroxetine", "venlafaxine", "duloxetine", "amitriptyline",
                               "mirtazapine", "bupropion", "trazodone", "nefazodone"}
            total_severity = 0.0
            for med in meds:
                lower = med.lower().strip()
                if lower in antipsychotics:
                    total_severity += 1.0
                elif lower in mood_stabilizers:
                    total_severity += 0.85
                elif lower in anxiolytics:
                    total_severity += 0.7
                elif lower in antidepressants:
                    total_severity += 0.5
                else:
                    total_severity += 0.3
            # 1 antidepressant → 0.50; 1 antipsychotic → 0.80; 3+ mixed → 1.0
            return max(0.0, min(1.0, total_severity / 3.0))

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
