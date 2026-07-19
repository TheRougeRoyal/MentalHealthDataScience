"""Firebase Admin SDK initialization and helpers.

Reads credentials from either a service-account JSON file or an inline
JSON string (for Railway / Vercel where mounting files is inconvenient).
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import auth, credentials, firestore

logger = logging.getLogger(__name__)

_app: firebase_admin.App | None = None
_db: firestore.Client | None = None


def _init_app() -> firebase_admin.App:
    global _app
    if _app is not None:
        return _app

    # Guard: reuse existing app if firebase_admin was already initialized
    try:
        _app = firebase_admin.get_app()
        logger.info("Firebase Admin SDK already initialised, reusing app")
        return _app
    except ValueError:
        pass  # No default app yet — continue to create one

    json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    json_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")

    if json_str:
        info = json.loads(json_str)
        cred = credentials.Certificate(info)
    elif json_path and os.path.isfile(json_path):
        cred = credentials.Certificate(json_path)
    else:
        cred = credentials.ApplicationDefault()

    _app = firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialised")
    return _app


def get_firestore_client() -> firestore.Client:
    """Return a shared Firestore client (lazy-initialised)."""
    global _db
    if _db is None:
        _init_app()
        _db = firestore.client()
    return _db


def verify_id_token(token: str) -> dict:
    """Verify a Firebase ID token and return the decoded claims.

    Raises ``firebase_admin.auth.InvalidIdTokenError`` on failure.
    """
    _init_app()
    return auth.verify_id_token(token)
