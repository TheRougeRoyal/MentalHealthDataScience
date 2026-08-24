"""Tests for API endpoints with mocked Firebase/Firestore."""

import pytest
from unittest.mock import MagicMock, patch


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_live_does_not_require_dependencies(client):
    r = client.get("/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_ready_reports_dependency_failure_without_details(client, monkeypatch):
    def unavailable_firestore():
        raise RuntimeError("private connection details")

    monkeypatch.setattr("src.api.app.get_firestore_client", unavailable_firestore)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json() == {
        "status": "not_ready",
        "service": "MHRAS API",
        "version": "2.0.0",
        "checks": {"firestore": "unavailable", "model": "ok"},
    }
    assert "private connection details" not in r.text


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "2.0.0" in r.json()["version"]


def test_auth_me(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer fake-token"})
    assert r.status_code == 200
    data = r.json()
    assert data["uid"] == "test_user_001"
    assert data["role"] == "user"


def test_admin_claim_requires_mfa(client, monkeypatch):
    monkeypatch.setattr("src.firebase_admin.verify_id_token", lambda token: {
        "uid": "admin_user",
        "email": "admin@example.com",
        "role": "admin",
        "firebase": {"sign_in_provider": "password"},
    })
    r = client.get("/admin/users", headers={"Authorization": "Bearer admin-token"})
    assert r.status_code == 403
    assert r.json()["detail"] == "MFA is required for administrator access"


def test_admin_claim_with_mfa_is_authorized(client, monkeypatch):
    monkeypatch.setattr("src.firebase_admin.verify_id_token", lambda token: {
        "uid": "admin_user",
        "email": "admin@example.com",
        "role": "admin",
        "firebase": {
            "sign_in_provider": "password",
            "sign_in_second_factor": "phone",
        },
    })
    r = client.get("/admin/users", headers={"Authorization": "Bearer admin-token"})
    assert r.status_code == 200


def test_firebase_config_is_public_only(client, monkeypatch):
    monkeypatch.setenv("FIREBASE_API_KEY", "public-key")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "public-project")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_JSON", "secret-json")
    r = client.get("/auth/config")
    assert r.status_code == 200
    assert r.json()["apiKey"] == "public-key"
    assert "serviceAccount" not in r.json()


def test_auth_me_no_token_is_rejected(client):
    """Missing tokens must never receive a development identity by default."""
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_non_admin_cannot_access_review_endpoints(client):
    r = client.get("/reviews", headers={"Authorization": "Bearer fake-token"})
    assert r.status_code == 403


def test_user_cannot_access_another_users_screening(client, mock_firebase):
    other_user_doc = MagicMock(exists=True, to_dict=lambda: {
        "id": "screening-other",
        "user_id": "another-user",
        "anonymized_id": "other-001",
    })
    mock_firebase.collection.return_value.document.return_value.get.return_value = other_user_doc
    r = client.post("/explain", headers={"Authorization": "Bearer fake-token"}, json={
        "anonymized_id": "other-001",
        "prediction_id": "screening-other",
    })
    assert r.status_code == 403


@pytest.mark.parametrize("failure", ["malformed", "expired", "forged"])
def test_invalid_tokens_are_rejected(client, monkeypatch, failure):
    def reject(_token):
        raise ValueError(failure)

    monkeypatch.setattr("src.firebase_admin.verify_id_token", reject)
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {failure}-token"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid or expired token"


def test_oversized_batch_is_rejected(client):
    payload = {
        "requests": [
            {"anonymized_id": f"batch-{index}", "consent_verified": True}
            for index in range(101)
        ]
    }
    r = client.post("/batch-screen", headers={"Authorization": "Bearer fake-token"}, json=payload)
    assert r.status_code == 400


def test_firestore_unavailable_during_screening_returns_generic_error(client, monkeypatch):
    monkeypatch.setattr("src.api.endpoints.get_firestore_client", lambda: (_ for _ in ()).throw(
        RuntimeError("private Firestore details")
    ))
    r = client.post("/screen", headers={"Authorization": "Bearer fake-token"}, json={
        "anonymized_id": "unavailable-001",
        "consent_verified": True,
        "survey_data": {"phq9_score": 10},
    })
    assert r.status_code == 503
    assert r.json()["detail"] == "Assessment persistence is unavailable. Check Firebase and encryption configuration."
    assert "private Firestore details" not in r.text


def test_idempotency_rejects_key_reuse_with_different_request():
    from src.api.endpoints import _load_idempotent_response

    db = MagicMock()
    document = db.collection.return_value.document.return_value
    document.get.return_value = MagicMock(
        exists=True,
        to_dict=lambda: {"fingerprint": "original", "response": {"ok": True}},
    )
    with pytest.raises(Exception) as error:
        _load_idempotent_response(db, "user-1", "request-1", "different")
    assert error.value.status_code == 409


def test_idempotency_replays_stored_response():
    from src.api.endpoints import _load_idempotent_response

    db = MagicMock()
    document = db.collection.return_value.document.return_value
    document.get.return_value = MagicMock(
        exists=True,
        to_dict=lambda: {"fingerprint": "original", "response": {"ok": True}},
    )
    assert _load_idempotent_response(db, "user-1", "request-1", "original") == {"ok": True}


def test_rate_limit_returns_429(client):
    from src.api.app import limiter

    limiter._storage.reset()
    payload = {
        "requests": [{"anonymized_id": "rate-limit-001", "consent_verified": True}]
    }
    responses = [client.post(
        "/batch-screen",
        headers={"Authorization": "Bearer fake-token"},
        json=payload,
    ) for _ in range(31)]
    assert any(response.status_code == 429 for response in responses)
    limiter._storage.reset()


def test_screen(client, mock_firebase):
    r = client.post("/screen", headers={"Authorization": "Bearer fake-token"}, json={
        "anonymized_id": "test_001",
        "consent_verified": True,
        "survey_data": {"phq9_score": 10, "gad7_score": 8},
    })
    assert r.status_code == 200
    data = r.json()
    assert "risk_score" in data
    assert data["risk_score"]["score"] >= 0
    assert data["risk_score"]["risk_level"] in ("low", "moderate", "high", "critical")


def test_screen_persists_minimized_encrypted_input_and_audit_event(client, mock_firebase):
    r = client.post("/screen", headers={"Authorization": "Bearer fake-token"}, json={
        "anonymized_id": "private_001",
        "consent_verified": True,
        "survey_data": {"phq9_score": 10, "unrelated_note": "do not persist"},
    })
    assert r.status_code == 200

    writes = mock_firebase.batch.return_value.set.call_args_list
    screening_write = next(call.args[1] for call in writes if "input_data_encrypted" in call.args[1])
    assert "input_data" not in screening_write
    assert screening_write["input_data_fields"] == ["phq9_score"]
    assert "10" not in screening_write["input_data_encrypted"]

    audit_write = next(call.args[1] for call in writes if call.args[1].get("action") == "screening_created")
    assert audit_write["action"] == "screening_created"
    assert "unrelated_note" not in str(audit_write)


def test_screening_commit_retries_transient_firestore_failure(mock_firebase):
    from src.api.endpoints import _commit_screening

    assessment = MagicMock(
        risk_score=10.0,
        risk_level="low",
        clinical_interpretation="test",
        contributing_factors=[],
        confidence=0.5,
        top_features=[],
        counterfactual="",
        requires_human_review=False,
    )
    mock_firebase.batch.return_value.commit.side_effect = [RuntimeError("transient"), None]

    _commit_screening(mock_firebase, "screening-1", "user-1", "anon-1", assessment, {"phq9_score": 1})

    assert mock_firebase.batch.return_value.commit.call_count == 2


def test_screen_no_consent(client, mock_firebase):
    r = client.post("/screen", headers={"Authorization": "Bearer fake-token"}, json={
        "anonymized_id": "test_002",
        "consent_verified": False,
        "survey_data": {"phq9_score": 10},
    })
    assert r.status_code == 403


def test_screen_missing_id(client, mock_firebase):
    r = client.post("/screen", headers={"Authorization": "Bearer fake-token"}, json={
        "consent_verified": True,
        "survey_data": {"phq9_score": 10},
    })
    assert r.status_code in (400, 422)


def test_batch_screen(client, mock_firebase):
    r = client.post("/batch-screen", headers={"Authorization": "Bearer fake-token"}, json={
        "requests": [
            {"anonymized_id": "batch_001", "consent_verified": True, "survey_data": {"phq9_score": 5}},
            {"anonymized_id": "batch_002", "consent_verified": True, "survey_data": {"phq9_score": 20}},
        ]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["successful"] == 2


def test_batch_screen_requires_consent_for_every_item(client, mock_firebase):
    r = client.post("/batch-screen", headers={"Authorization": "Bearer fake-token"}, json={
        "requests": [
            {"anonymized_id": "batch_001", "consent_verified": True},
            {"anonymized_id": "batch_002", "consent_verified": False},
        ]
    })
    assert r.status_code == 403


def test_statistics(client, mock_firebase):
    r = client.get("/statistics", headers={"Authorization": "Bearer fake-token"})
    assert r.status_code == 200
    data = r.json()
    assert "screenings" in data
    assert "risk_distribution" in data
    assert data["screenings"]["median_risk_score"] == 0
    assert "review_queue" in data
