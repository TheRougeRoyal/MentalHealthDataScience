"""Authentication and authorization for MHRAS API (Firebase Auth).

- Frontend sends Firebase ID token in ``Authorization: Bearer <token>``
- Backend verifies with ``firebase_admin.auth.verify_id_token()``
- User role fetched from Firestore ``users/{uid}`` collection
- Role checks happen server-side only
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from firebase_admin import firestore as _fs
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.firebase_admin import get_firestore_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AuthResult(BaseModel):
    """Injected into every authenticated endpoint."""
    authenticated: bool
    user_id: str | None = None
    email: str | None = None
    role: str | None = None
    display_name: str | None = None
    photo_url: str | None = None
    provider: str | None = None
    mfa_verified: bool = False
    error: str | None = None


class UserInfo(BaseModel):
    uid: str
    email: str | None = None
    role: str
    display_name: str | None = None
    photo_url: str | None = None
    provider: str | None = None
    bio: str | None = None
    organization: str | None = None
    job_title: str | None = None
    phone: str | None = None
    location: str | None = None
    website: str | None = None


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None
    bio: str | None = None
    organization: str | None = None
    job_title: str | None = None
    phone: str | None = None
    location: str | None = None
    website: str | None = None


def _user_info_from_doc(uid: str, auth: AuthResult, user_data: dict | None) -> UserInfo:
    data = user_data or {}
    return UserInfo(
        uid=uid,
        email=auth.email,
        role=auth.role or "user",
        display_name=data.get("display_name", auth.display_name),
        photo_url=data.get("photo_url", auth.photo_url),
        provider=data.get("provider", auth.provider),
        bio=data.get("bio"),
        organization=data.get("organization"),
        job_title=data.get("job_title"),
        phone=data.get("phone"),
        location=data.get("location"),
        website=data.get("website"),
    )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> AuthResult:
    """Extract and verify the Firebase ID token from the Authorization header.

    A development bypass is opt-in and must never be enabled by default.
    """
    from src.firebase_admin import verify_id_token, get_firestore_client

    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        if os.environ.get("ALLOW_DEV_AUTH_BYPASS", "").lower() == "true":
            return AuthResult(
                authenticated=True,
                user_id="dev_user",
                email="dev@example.com",
                role="admin",
                display_name="Dev User",
                photo_url=None,
                provider="dev",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    token = auth_header.split("Bearer ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty token",
        )

    try:
        decoded = verify_id_token(token)
    except Exception as e:
        logger.warning("Token verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    uid = decoded.get("uid", "")
    email = decoded.get("email", "")
    display_name = decoded.get("name", decoded.get("email", uid))
    photo_url = decoded.get("photo_url") or decoded.get("picture")

    # Determine auth provider from Firebase sign_in_provider claim
    firebase_claims = decoded.get("firebase", {})
    sign_in_provider = firebase_claims.get("sign_in_provider", "unknown")
    if sign_in_provider == "google.com":
        provider = "google"
    elif sign_in_provider == "password":
        provider = "email"
    else:
        provider = sign_in_provider

    # Firebase custom claims are the runtime source of truth. Email addresses
    # are identity metadata, never authorization credentials.
    is_admin = decoded.get("admin") is True or decoded.get("role") == "admin"
    default_role = "admin" if is_admin else "user"
    role = default_role
    mfa_verified = bool(firebase_claims.get("sign_in_second_factor"))
    try:
        db = get_firestore_client()
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            stored_role = user_data.get("role")
            role = "admin" if is_admin else "user"
            if stored_role != role:
                db.collection("users").document(uid).update({"role": role})
            display_name = user_data.get("display_name", display_name)
            photo_url = user_data.get("photo_url", photo_url)
        else:
            # First login — auto-create user doc
            new_user = {
                "uid": uid,
                "email": email,
                "display_name": display_name,
                "photo_url": photo_url,
                "role": default_role,
                "provider": provider,
                "created_at": _fs.SERVER_TIMESTAMP,
            }
            db.collection("users").document(uid).set(new_user, merge=True)
            logger.info("Created Firestore user doc for %s (role=%s, provider=%s)", uid, default_role, provider)
    except Exception as e:
        logger.error("Failed to fetch/create user doc: %s", e)

    return AuthResult(
        authenticated=True,
        user_id=uid,
        email=email,
        role=role,
        display_name=display_name,
        photo_url=photo_url,
        provider=provider,
        mfa_verified=mfa_verified,
    )


def require_role(*allowed_roles: str):
    """Dependency that enforces role-based access.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role("admin"))])
        def admin_view(): ...
    """
    def _checker(auth: AuthResult = Depends(get_current_user)) -> AuthResult:
        if auth.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{auth.role}' not allowed. Required: {', '.join(allowed_roles)}",
            )
        if auth.role == "admin" and os.environ.get("REQUIRE_ADMIN_MFA", "true").lower() == "true" and not auth.mfa_verified:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="MFA is required for administrator access")
        return auth
    return _checker


# ---------------------------------------------------------------------------
# Auth router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/config")
async def firebase_config():
    """Return only the public Firebase web configuration for the frontend."""
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
        "measurementId": os.environ.get("FIREBASE_MEASUREMENT_ID", ""),
    }


@router.get("/me")
async def me(auth: AuthResult = Depends(get_current_user)):
    """Return the identity and profile of the currently authenticated user."""
    uid = auth.user_id or ""
    user_data: dict | None = None
    try:
        db = get_firestore_client()
        if db is not None:
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                user_data = doc.to_dict()
    except Exception as e:
        logger.error("Failed to fetch user profile: %s", e)
    return _user_info_from_doc(uid, auth, user_data)


@router.patch("/me")
async def update_me(
    body: UpdateProfileRequest,
    auth: AuthResult = Depends(get_current_user),
):
    """Update the current user's profile fields in Firestore."""
    uid = auth.user_id or ""
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    updates: dict = {}
    for field in ("display_name", "bio", "organization", "job_title", "phone", "location", "website"):
        value = getattr(body, field)
        if value is not None:
            cleaned = value.strip() if isinstance(value, str) else value
            if field == "bio" and cleaned and len(cleaned) > 500:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bio must be 500 characters or fewer")
            updates[field] = cleaned or None

    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No profile fields to update")

    updates["updated_at"] = _fs.SERVER_TIMESTAMP

    try:
        db = get_firestore_client()
        if db is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Profile storage unavailable")
        db.collection("users").document(uid).set(updates, merge=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user profile: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update profile")

    if "display_name" in updates and updates["display_name"]:
        try:
            from firebase_admin import auth as firebase_auth
            firebase_auth.update_user(uid, display_name=updates["display_name"])
        except Exception as e:
            logger.warning("Firebase Auth display_name sync failed: %s", e)

    doc = db.collection("users").document(uid).get()
    user_data = doc.to_dict() if doc.exists else updates
    return _user_info_from_doc(uid, auth, user_data)
