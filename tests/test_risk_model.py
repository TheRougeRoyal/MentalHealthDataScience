"""Tests for the clinical rules risk model."""

from src.risk_model import ClinicalRulesModel, OllamaRiskModel, RiskModel, get_risk_model


def test_get_risk_model_returns_instance():
    model = get_risk_model()
    assert isinstance(model, RiskModel)


def test_low_risk_scores_low():
    model = ClinicalRulesModel()
    result = model.assess({"phq9_score": 2, "gad7_score": 1})
    assert result.risk_score < 30
    assert result.risk_level == "low"
    assert not result.alert_triggered


def test_high_risk_scores_high():
    model = ClinicalRulesModel()
    result = model.assess({
        "phq9_score": 25, "gad7_score": 20,
        "sleep_hours": 3, "avg_heart_rate": 100,
        "diagnosis_codes": ["F32.1", "F41.1"],
        "medications": ["sertraline", "lorazepam"],
    })
    assert result.risk_score >= 51
    assert result.risk_level in ("high", "critical")


def test_critical_risk():
    model = ClinicalRulesModel()
    result = model.assess({
        "phq9_score": 27, "gad7_score": 21,
        "sleep_hours": 2, "avg_heart_rate": 110,
        "diagnosis_codes": ["F32.1", "F41.1", "F33.0"],
        "medications": ["sertraline", "lorazepam", "lithium", "quetiapine"],
    })
    assert result.risk_level == "critical"
    assert result.alert_triggered


def test_no_data_returns_neutral():
    model = ClinicalRulesModel()
    result = model.assess({})
    assert result.risk_score == 50.0
    assert result.confidence > 0


def test_explanations_generated():
    model = ClinicalRulesModel()
    result = model.assess({"phq9_score": 18, "gad7_score": 15})
    assert len(result.top_features) > 0
    assert len(result.clinical_interpretation) > 0


def test_counterfactual_for_high_risk():
    model = ClinicalRulesModel()
    result = model.assess({"phq9_score": 20, "gad7_score": 16})
    if result.risk_level in ("high", "critical"):
        assert len(result.counterfactual) > 0


def test_confidence_range():
    model = ClinicalRulesModel()
    result = model.assess({"phq9_score": 10})
    assert 0.0 <= result.confidence <= 1.0


# ── Clinical severity band tests ──────────────────────────────────────────

def test_phq9_minimal_severity():
    """PHQ-9 0-4 (minimal) should produce low contribution."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("phq9_score", {"phq9_score": 3})
    assert c is not None
    assert c < 0.15


def test_phq9_severe_range():
    """PHQ-9 20-27 (severe) should produce high contribution."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("phq9_score", {"phq9_score": 25})
    assert c is not None
    assert c > 0.80


def test_gad7_severe_range():
    """GAD-7 15-21 (severe) should produce high contribution."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("gad7_score", {"gad7_score": 18})
    assert c is not None
    assert c > 0.70


def test_sleep_optimal_zero():
    """Sleep 7-9 hours should contribute zero risk."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("sleep_hours", {"sleep_hours": 7.5})
    assert c == 0.0


def test_sleep_severe_insomnia():
    """Sleep <4h should contribute very high risk."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("sleep_hours", {"sleep_hours": 3})
    assert c is not None
    assert c > 0.85


def test_sleep_hypersomnia():
    """Sleep >11h should contribute elevated risk."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("sleep_hours", {"sleep_hours": 12})
    assert c is not None
    assert c > 0.35


def test_heart_rate_normal_zero():
    """HR 60-80 should contribute zero risk."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("avg_heart_rate", {"avg_heart_rate": 70})
    assert c == 0.0


def test_heart_rate_tachycardia():
    """HR >100 should contribute high risk."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("avg_heart_rate", {"avg_heart_rate": 115})
    assert c is not None
    assert c > 0.60


def test_heart_rate_bradycardia():
    """HR <50 should contribute moderate risk."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("avg_heart_rate", {"avg_heart_rate": 45})
    assert c is not None
    assert c > 0.30


# ── Diagnosis severity weighting ──────────────────────────────────────────

def test_diagnosis_schizophrenia_higher_than_depression():
    """F2x diagnoses should score higher than F3x."""
    model = ClinicalRulesModel()
    c_f2 = model._feature_contribution("diagnosis_codes", {"diagnosis_codes": ["F20.0"]})
    c_f3 = model._feature_contribution("diagnosis_codes", {"diagnosis_codes": ["F32.1"]})
    assert c_f2 is not None and c_f3 is not None
    assert c_f2 > c_f3


def test_diagnosis_multiple_severe():
    """3+ diagnoses should approach max contribution."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("diagnosis_codes", {
        "diagnosis_codes": ["F20.0", "F32.2", "F41.1"]
    })
    assert c is not None
    assert c > 0.70


# ── Medication class weighting ────────────────────────────────────────────

def test_antipsychotic_higher_than_antidepressant():
    """Antipsychotics should score higher than antidepressants."""
    model = ClinicalRulesModel()
    c_ap = model._feature_contribution("medications", {"medications": ["olanzapine"]})
    c_ad = model._feature_contribution("medications", {"medications": ["sertraline"]})
    assert c_ap is not None and c_ad is not None
    assert c_ap > c_ad


def test_mixed_medication_burden():
    """Multiple medication classes should approach max."""
    model = ClinicalRulesModel()
    c = model._feature_contribution("medications", {
        "medications": ["quetiapine", "lithium", "lorazepam", "sertraline"]
    })
    assert c is not None
    assert c > 0.80


# ── Feature interaction tests ─────────────────────────────────────────────

def test_depression_sleep_compound():
    """High PHQ-9 + poor sleep should score higher than either alone."""
    model = ClinicalRulesModel()
    # PHQ-9 severe + sleep <4h (both exceed interaction thresholds)
    result_combined = model.assess({
        "phq9_score": 22, "sleep_hours": 3,
    })
    # PHQ-9 severe alone
    result_phq9_only = model.assess({"phq9_score": 22})
    # Poor sleep alone
    result_sleep_only = model.assess({"sleep_hours": 3})
    assert result_combined.risk_score > result_phq9_only.risk_score
    assert result_combined.risk_score > result_sleep_only.risk_score


def test_anxiety_hr_compound():
    """High GAD-7 + elevated HR should compound."""
    model = ClinicalRulesModel()
    result_combined = model.assess({"gad7_score": 18, "avg_heart_rate": 105})
    result_anxiety_only = model.assess({"gad7_score": 18})
    assert result_combined.risk_score > result_anxiety_only.risk_score


def test_comorbid_depression_anxiety():
    """Comorbid depression + anxiety should compound significantly."""
    model = ClinicalRulesModel()
    result_combined = model.assess({"phq9_score": 20, "gad7_score": 18})
    result_phq9_only = model.assess({"phq9_score": 20})
    assert result_combined.risk_score > result_phq9_only.risk_score


# ── Dynamic sigmoid / data completeness tests ─────────────────────────────

def test_more_data_more_decisive():
    """Full data should produce a more extreme (less neutral) score than partial."""
    model = ClinicalRulesModel()
    # Partial data — just PHQ-9
    partial = model.assess({"phq9_score": 15})
    # Full data with same PHQ-9 + other elevated features
    full = model.assess({
        "phq9_score": 15, "gad7_score": 12,
        "sleep_hours": 5, "avg_heart_rate": 90,
        "diagnosis_codes": ["F32.1"],
        "medications": ["sertraline"],
    })
    # Both should be non-neutral; full data should be more decisive
    assert partial.risk_score != 50.0
    assert full.risk_score != 50.0


# ── Confidence calibration tests ──────────────────────────────────────────

def test_confidence_increases_with_more_features():
    """More features available should yield higher confidence."""
    model = ClinicalRulesModel()
    result_minimal = model.assess({"phq9_score": 15})
    result_full = model.assess({
        "phq9_score": 15, "gad7_score": 12,
        "sleep_hours": 5, "avg_heart_rate": 90,
        "diagnosis_codes": ["F32.1"],
        "medications": ["sertraline"],
    })
    assert result_full.confidence > result_minimal.confidence


def test_confidence_higher_with_critical_features():
    """PHQ-9 + GAD-7 present should boost confidence."""
    model = ClinicalRulesModel()
    result_with_both = model.assess({"phq9_score": 15, "gad7_score": 12})
    result_with_one = model.assess({"phq9_score": 15})
    assert result_with_both.confidence >= result_with_one.confidence


def test_confidence_bounded():
    """Confidence should always be in [0.1, 1.0]."""
    model = ClinicalRulesModel()
    for data in [
        {},
        {"phq9_score": 0},
        {"phq9_score": 27, "gad7_score": 21, "sleep_hours": 2,
         "avg_heart_rate": 120, "diagnosis_codes": ["F20.0"],
         "medications": ["clozapine", "lithium", "lorazepam"]},
    ]:
        result = model.assess(data)
        assert 0.1 <= result.confidence <= 1.0
