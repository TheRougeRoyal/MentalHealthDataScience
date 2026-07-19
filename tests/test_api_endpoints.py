"""Tests for API endpoints with mocked Firebase/Firestore."""

import pytest
from unittest.mock import MagicMock, patch


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


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


def test_auth_me_no_token_dev_mode(client):
    """In development mode, missing token falls back to dev_user."""
    r = client.get("/auth/me")
    assert r.status_code == 200


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


def test_statistics(client, mock_firebase):
    r = client.get("/statistics", headers={"Authorization": "Bearer fake-token"})
    assert r.status_code == 200
    data = r.json()
    assert "screenings" in data
    assert "review_queue" in data
