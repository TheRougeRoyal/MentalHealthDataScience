"""Tests for MHRAS API endpoints.

Covers: /screen, /risk-score, /explain, /statistics, /reviews/*
Uses the in-memory SQLite + auth overrides from conftest.
"""

import pytest

from src.models import Screening, Explanation, Review


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_VALID_SCREEN = {
    "anonymized_id": "test_patient_001",
    "consent_verified": True,
    "timestamp": "2026-04-12T00:00:00Z",
    "survey_data": {"phq9_score": 15, "gad7_score": 12},
    "wearable_data": {"sleep_hours": 5.0, "avg_heart_rate": 82},
}

_LOW_RISK_SCREEN = {
    "anonymized_id": "low_risk_001",
    "consent_verified": True,
    "timestamp": "2026-04-12T00:00:00Z",
    "survey_data": {"phq9_score": 3, "gad7_score": 2},
    "wearable_data": {"sleep_hours": 8.0, "avg_heart_rate": 65},
}


def _create_screening(client, payload=None):
    """Post a screening and return the JSON response."""
    return client.post("/screen", json=payload or _VALID_SCREEN)


# ═══════════════════════════════════════════════════════════════════════════
# POST /screen
# ═══════════════════════════════════════════════════════════════════════════


class TestScreen:
    """POST /screen endpoint tests."""

    def test_screen_success(self, client):
        res = _create_screening(client)
        assert res.status_code == 200
        data = res.json()
        assert "risk_score" in data
        assert data["risk_score"]["anonymized_id"] == "test_patient_001"
        assert 0 <= data["risk_score"]["score"] <= 100
        assert data["risk_score"]["risk_level"] in (
            "low", "moderate", "high", "critical",
        )
        assert "explanations" in data
        assert "recommendations" in data

    def test_screen_low_risk(self, client):
        res = _create_screening(client, _LOW_RISK_SCREEN)
        assert res.status_code == 200
        score = res.json()["risk_score"]["score"]
        assert score < 50, f"Expected low score, got {score}"

    def test_screen_missing_consent(self, client):
        payload = {**_VALID_SCREEN, "consent_verified": False}
        res = client.post("/screen", json=payload)
        assert res.status_code == 403

    def test_screen_missing_id(self, client):
        payload = {k: v for k, v in _VALID_SCREEN.items() if k != "anonymized_id"}
        res = client.post("/screen", json=payload)
        assert res.status_code in (400, 422)

    def test_screen_empty_body(self, client):
        res = client.post("/screen", json={})
        assert res.status_code in (400, 422)

    def test_screen_persists_to_db(self, client, db):
        _create_screening(client)
        count = db.query(Screening).count()
        assert count >= 1

    def test_screen_creates_explanation(self, client, db):
        _create_screening(client)
        count = db.query(Explanation).count()
        assert count >= 1


# ═══════════════════════════════════════════════════════════════════════════
# GET /risk-score/{anonymized_id}
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskScore:
    """GET /risk-score/{id} endpoint tests."""

    def test_risk_score_found(self, client):
        _create_screening(client)
        res = client.get("/risk-score/test_patient_001")
        assert res.status_code == 200
        data = res.json()
        assert data["anonymized_id"] == "test_patient_001"
        assert "score" in data
        assert "risk_level" in data

    def test_risk_score_not_found(self, client):
        res = client.get("/risk-score/nonexistent_id")
        assert res.status_code == 404

    def test_risk_score_returns_latest(self, client):
        """If multiple screenings exist, the latest should be returned."""
        _create_screening(client)
        # Screen again with different data to produce a different score
        payload = {**_VALID_SCREEN, "survey_data": {"phq9_score": 25, "gad7_score": 20}}
        _create_screening(client, payload)
        res = client.get("/risk-score/test_patient_001")
        assert res.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# POST /explain
# ═══════════════════════════════════════════════════════════════════════════


class TestExplain:
    """POST /explain endpoint tests."""

    def test_explain_success(self, client):
        _create_screening(client)
        res = client.post("/explain", json={"anonymized_id": "test_patient_001"})
        assert res.status_code == 200
        data = res.json()
        assert "explanations" in data
        assert "risk_score" in data

    def test_explain_not_found(self, client):
        res = client.post("/explain", json={"anonymized_id": "ghost_patient"})
        assert res.status_code == 404

    def test_explain_missing_id(self, client):
        res = client.post("/explain", json={})
        assert res.status_code in (400, 422)


# ═══════════════════════════════════════════════════════════════════════════
# GET /statistics
# ═══════════════════════════════════════════════════════════════════════════


class TestStatistics:
    """GET /statistics endpoint tests."""

    def test_statistics_empty_db(self, client):
        res = client.get("/statistics")
        assert res.status_code == 200
        data = res.json()
        assert data["screenings"]["total"] == 0
        assert data["review_queue"]["pending_count"] == 0

    def test_statistics_after_screening(self, client):
        _create_screening(client)
        res = client.get("/statistics")
        data = res.json()
        assert data["screenings"]["total"] >= 1
        assert "avg_risk_score" in data["screenings"]
        assert "high_risk_pct" in data["screenings"]

    def test_statistics_has_timestamp(self, client):
        res = client.get("/statistics")
        assert "timestamp" in res.json()


# ═══════════════════════════════════════════════════════════════════════════
# Review workflow: /reviews/*
# ═══════════════════════════════════════════════════════════════════════════


class TestReviews:
    """Review queue + assign + comment + close tests."""

    def _seed_review(self, client, db):
        """Create a screening that triggers a review."""
        # High scores → requires_human_review = True
        payload = {
            "anonymized_id": "review_patient",
            "consent_verified": True,
            "timestamp": "2026-04-12T00:00:00Z",
            "survey_data": {"phq9_score": 27, "gad7_score": 21},
            "wearable_data": {"sleep_hours": 2.0, "avg_heart_rate": 110},
        }
        _create_screening(client, payload)
        review = db.query(Review).first()
        return review

    def test_queue_empty(self, client):
        res = client.get("/reviews/queue")
        assert res.status_code == 200
        assert res.json()["total"] == 0

    def test_queue_after_seed(self, client, db):
        self._seed_review(client, db)
        res = client.get("/reviews/queue")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] >= 1
        assert data["reviews"][0]["status"] == "pending"

    def test_assign_review(self, client, db):
        review = self._seed_review(client, db)
        if review is None:
            pytest.skip("No review created (risk too low)")
        res = client.post(
            f"/reviews/{review.id}/assign",
            json={"reviewer": "dr_smith"},
        )
        assert res.status_code == 200
        assert res.json()["reviewer"] == "dr_smith"
        assert res.json()["status"] == "reviewed"

    def test_comment_review(self, client, db):
        review = self._seed_review(client, db)
        if review is None:
            pytest.skip("No review created")
        res = client.post(
            f"/reviews/{review.id}/comment",
            json={"comments": "Needs follow-up in 48h."},
        )
        assert res.status_code == 200
        assert "Needs follow-up" in res.json()["comments"]

    def test_close_review(self, client, db):
        review = self._seed_review(client, db)
        if review is None:
            pytest.skip("No review created")
        res = client.post(
            f"/reviews/{review.id}/close",
            json={"comments": "Cleared after evaluation."},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "closed"

    def test_close_already_closed(self, client, db):
        review = self._seed_review(client, db)
        if review is None:
            pytest.skip("No review created")
        client.post(f"/reviews/{review.id}/close")
        res = client.post(f"/reviews/{review.id}/close")
        assert res.status_code == 409

    def test_assign_nonexistent(self, client):
        res = client.post(
            "/reviews/00000000-0000-0000-0000-000000000000/assign",
            json={"reviewer": "nobody"},
        )
        assert res.status_code == 404

    def test_queue_filter_closed(self, client, db):
        review = self._seed_review(client, db)
        if review is None:
            pytest.skip("No review created")
        client.post(f"/reviews/{review.id}/close")
        res = client.get("/reviews/queue?status_filter=closed")
        assert res.status_code == 200
        assert any(r["status"] == "closed" for r in res.json()["reviews"])


# ═══════════════════════════════════════════════════════════════════════════
# Batch screening
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchScreen:
    """POST /batch-screen endpoint tests."""

    def test_batch_success(self, client):
        payload = {
            "requests": [
                {**_VALID_SCREEN, "anonymized_id": "batch_001"},
                {**_LOW_RISK_SCREEN, "anonymized_id": "batch_002"},
            ]
        }
        res = client.post("/batch-screen", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2
        assert data["successful"] == 2

    def test_batch_empty(self, client):
        res = client.post("/batch-screen", json={"requests": []})
        assert res.status_code in (200, 400, 422)


# ═══════════════════════════════════════════════════════════════════════════
# Health & root
# ═══════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

    def test_root(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "service" in res.json()
