"""Shared fixtures for MHRAS API tests.

Uses an in-memory SQLite database and overrides FastAPI dependencies
so tests run completely isolated from production infrastructure.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock

# Patch the initialize_ds_components function BEFORE importing the app
# to prevent data science initialization that tries to connect to PostgreSQL
print(">>> APPLYING EARLY PATCH: initialize_ds_components")
import sys
if 'src.api.ds_endpoints' not in sys.modules:
    with patch('src.api.ds_endpoints.initialize_ds_components') as mock_init_ds:
        mock_init_ds.side_effect = Exception("Data science initialization skipped for testing")
        print(">>> EARLY PATCH APPLIED: initialize_ds_components will raise exception")

from src.models import Base
from src.database import get_db
from src.api.auth import get_current_user, AuthResult
from src.api.app import app


# ---------------------------------------------------------------------------
# Test database
# ---------------------------------------------------------------------------

_TEST_DB_URL = "sqlite://"  # in-memory

engine = create_engine(
    _TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth overrides
# ---------------------------------------------------------------------------

def _override_auth_admin() -> AuthResult:
    return AuthResult(authenticated=True, user_id="test_admin", role="admin")


def _override_auth_reviewer() -> AuthResult:
    return AuthResult(authenticated=True, user_id="test_reviewer", role="reviewer")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_db():
    """Create tables before each test and drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    """FastAPI TestClient with admin auth and test DB."""
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _override_auth_admin
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def reviewer_client():
    """TestClient authenticated as a reviewer."""
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _override_auth_reviewer
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def db():
    """Raw DB session for direct queries in assertions."""
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
