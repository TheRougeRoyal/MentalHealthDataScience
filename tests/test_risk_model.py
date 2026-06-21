"""Tests for the clinical rules risk model."""

from src.risk_model import ClinicalRulesModel, get_risk_model


def test_get_risk_model_returns_instance():
    model = get_risk_model()
    assert isinstance(model, ClinicalRulesModel)


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
