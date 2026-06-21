import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.app import app
from src.database import Base, get_db
from src.models import Screening, Explanation, Review

# ── Fixtures ──────────────────────────────────────────────────────────────

SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection, _):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Risk Assessment" in r.json()["service"]


# ── Auth ──────────────────────────────────────────────────────────────────

def test_login_success(client):
    r = client.post("/auth/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "admin"
    assert data["role"] == "admin"


def test_login_wrong_password(client):
    r = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_me_without_cookie(client):
    r = client.get("/auth/me")
    assert r.status_code == 200  # dev mode returns dev_user


def test_logout(client):
    r = client.post("/auth/logout")
    assert r.status_code == 200


# ── Screening ─────────────────────────────────────────────────────────────

def test_screen_low_risk(client):
    r = client.post("/screen", json={
        "anonymized_id": "test_001",
        "consent_verified": True,
        "survey_data": {"phq9_score": 3, "gad7_score": 2},
    })
    assert r.status_code == 200
    data = r.json()
    assert data["risk_score"]["score"] < 51
    assert data["risk_score"]["risk_level"] in ("low", "moderate")


def test_screen_high_risk(client):
    r = client.post("/screen", json={
        "anonymized_id": "test_002",
        "consent_verified": True,
        "survey_data": {"phq9_score": 22, "gad7_score": 18},
        "wearable_data": {"sleep_hours": 3.5, "avg_heart_rate": 95},
        "emr_data": {"diagnosis_codes": ["F32.1", "F41.1"], "medications": ["sertraline"]},
    })
    assert r.status_code == 200
    data = r.json()
    assert data["risk_score"]["score"] >= 51
    assert data["risk_score"]["risk_level"] in ("high", "critical")


def test_screen_no_consent(client):
    r = client.post("/screen", json={
        "anonymized_id": "test_003",
        "consent_verified": False,
        "survey_data": {"phq9_score": 10},
    })
    assert r.status_code == 403


def test_screen_missing_id(client):
    r = client.post("/screen", json={
        "consent_verified": True,
        "survey_data": {"phq9_score": 10},
    })
    assert r.status_code in (400, 422)


def test_screen_generates_review_for_high_risk(client):
    r = client.post("/screen", json={
        "anonymized_id": "test_review",
        "consent_verified": True,
        "survey_data": {"phq9_score": 25, "gad7_score": 20},
        "wearable_data": {"sleep_hours": 2, "avg_heart_rate": 110},
    })
    assert r.status_code == 200
    assert r.json()["requires_human_review"] is True


# ── Batch Screening ───────────────────────────────────────────────────────

def test_batch_screen(client):
    r = client.post("/batch-screen", json={
        "requests": [
            {"anonymized_id": "batch_001", "consent_verified": True, "survey_data": {"phq9_score": 5}},
            {"anonymized_id": "batch_002", "consent_verified": True, "survey_data": {"phq9_score": 20}},
        ]
    })
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0


# ── Risk Score ────────────────────────────────────────────────────────────

def test_get_risk_score(client):
    # First create a screening
    client.post("/screen", json={
        "anonymized_id": "score_test",
        "consent_verified": True,
        "survey_data": {"phq9_score": 10},
    })
    r = client.get("/risk-score/score_test")
    assert r.status_code == 200
    assert r.json()["anonymized_id"] == "score_test"


def test_get_risk_score_not_found(client):
    r = client.get("/risk-score/nonexistent")
    assert r.status_code == 404


# ── Statistics ────────────────────────────────────────────────────────────

def test_statistics(client):
    r = client.get("/statistics")
    assert r.status_code == 200
    data = r.json()
    assert "screenings" in data
    assert "review_queue" in data


def test_statistics_after_screening(client):
    client.post("/screen", json={
        "anonymized_id": "stats_test",
        "consent_verified": True,
        "survey_data": {"phq9_score": 10},
    })
    r = client.get("/statistics")
    assert r.status_code == 200
    assert r.json()["screenings"]["total"] >= 1
