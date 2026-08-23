"""Test fixtures — mocks Firebase so tests run without credentials."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENVIRONMENT", "development")
if not os.environ.get("SECURITY_DATA_ENCRYPTION_KEY"):
    os.environ["SECURITY_DATA_ENCRYPTION_KEY"] = "ZAeWmizAsJdqNlhw3dW-rfgUh6F-8aVfqeMbIGrTBks="


@pytest.fixture(autouse=True)
def mock_firebase():
    """Mock firebase_admin so no real Firebase calls happen in tests."""
    from src.api import admin as admin_api

    mock_firestore = MagicMock()
    mock_collection = MagicMock()
    mock_firestore.collection.return_value = mock_collection

    # Default: no existing docs
    mock_doc = MagicMock(exists=False, to_dict=MagicMock(return_value={}))
    mock_collection.document.return_value = MagicMock(get=MagicMock(return_value=mock_doc))
    mock_collection.where.return_value.order_by.return_value.limit.return_value.get.return_value = []
    mock_collection.where.return_value.get.return_value = []

    with patch("firebase_admin.initialize_app"), \
         patch("firebase_admin.credentials.Certificate"), \
         patch("src.firebase_admin._app", MagicMock()), \
         patch("src.firebase_admin._db", None), \
         patch("src.firebase_admin.get_firestore_client", return_value=mock_firestore), \
         patch("src.api.endpoints.get_firestore_client", return_value=mock_firestore), \
         patch.object(admin_api, "get_firestore_client", return_value=mock_firestore), \
         patch("src.firebase_admin.verify_id_token", return_value={
             "uid": "test_user_001",
             "email": "test@example.com",
             "name": "Test User",
         }):

        yield mock_firestore


@pytest.fixture
def client(mock_firebase):
    """Create a test client with mocked Firebase."""
    from src.api.app import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
