"""User profile endpoints — GET /auth/me and PATCH /auth/me.

Separated from ``src.api.auth`` so token verification logic stays minimal and
testable on its own. This module owns:

- ``UserInfo`` and ``UpdateProfileRequest`` Pydantic models
- ``_user_info_from_doc`` / ``_provision_new_user`` Firestore helpers
- The ``/auth/me`` GET and PATCH route handlers

It imports ``AuthResult``, ``get_current_user``, and ``require_role`` from
``src.api.auth`` and mounts onto the same ``/auth`` prefix so the URL surface
is unchanged.
"""

from __future__ import annotations

import logging

from firebase_admin import firestore as _fs
from firebase_admin import auth as firebase_auth
from fastapi import APIRouter, Depends, HTTPException, status
from google.api_core.exceptions import GoogleAPICallError
from pydantic import BaseModel

import src.firebase_admin as _firebase_admin_module
from src.api.auth import AuthResult, get_current_user

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Firestore helpers
# ---------------------------------------------------------------------------

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


def _provision_new_user(
    db: object,
    uid: str,
    email: str,
    display_name: str | None,
    photo_url: str | None,
    role: str,
    provider: str,
) -> None:
    """Write the initial Firestore user document on first login.

    Called from GET /auth/me when the document does not yet exist.
    Never called from ``get_current_user`` so it does not block every request.

    ``stored_role`` is normalised to the claim-derived value on creation so the
    document stays consistent with Firebase custom claims.
    """
    stored_role = role  # claim-derived; kept for role-isolation test assertions
    new_user = {
        "uid": uid,
        "email": email,
        "display_name": display_name,
        "photo_url": photo_url,
        "role": stored_role,
        "provider": provider,
        "created_at": _fs.SERVER_TIMESTAMP,
    }
    db.collection("users").document(uid).set(new_user, merge=True)
    logger.info(
        "Created Firestore user doc for %s (role=%s, provider=%s)",
        uid, stored_role, provider,
    )


# ---------------------------------------------------------------------------
# Profile router  (same /auth prefix — mounts alongside auth.router)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/me")
async def me(auth: AuthResult = Depends(get_current_user)):
    """Return the identity and profile of the currently authenticated user.

    This is the only request path that reads the Firestore ``users/{uid}``
    document. It also provisions the document on first login.
    """
    uid = auth.user_id or ""
    user_data: dict | None = None
    try:
        db = _firebase_admin_module.get_firestore_client()
        if db is not None:
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                user_data = doc.to_dict() or {}
            else:
                # First login — create the doc, seed user_data from claims
                _provision_new_user(
                    db,
                    uid,
                    auth.email or "",
                    auth.display_name,
                    auth.photo_url,
                    auth.role or "user",
                    auth.provider or "unknown",
                )
                user_data = {
                    "display_name": auth.display_name,
                    "photo_url": auth.photo_url,
                }
    except GoogleAPICallError as e:
        # Firestore / network failure — fail explicitly so the client knows
        # the profile is unavailable rather than receiving partial data.
        logger.error(
            "Firestore unavailable fetching profile for %s: %s", uid, e, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile service temporarily unavailable",
        )
    except Exception as e:
        logger.error(
            "Unexpected error fetching profile for %s: %s", uid, e, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile service temporarily unavailable",
        )
    return _user_info_from_doc(uid, auth, user_data)


@router.patch("/me")
async def update_me(
    body: UpdateProfileRequest,
    auth: AuthResult = Depends(get_current_user),
):
    """Update the current user's mutable profile fields in Firestore."""
    uid = auth.user_id or ""
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )

    updates: dict = {}
    for field in (
        "display_name", "bio", "organization",
        "job_title", "phone", "location", "website",
    ):
        value = getattr(body, field)
        if value is not None:
            cleaned = value.strip() if isinstance(value, str) else value
            if field == "bio" and cleaned and len(cleaned) > 500:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bio must be 500 characters or fewer",
                )
            updates[field] = cleaned or None

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No profile fields to update",
        )

    updates["updated_at"] = _fs.SERVER_TIMESTAMP

    try:
        db = _firebase_admin_module.get_firestore_client()
        if db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Profile storage unavailable",
            )
        db.collection("users").document(uid).set(updates, merge=True)
    except HTTPException:
        raise
    except GoogleAPICallError as e:
        logger.error(
            "Firestore unavailable updating profile for %s: %s", uid, e, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile service temporarily unavailable",
        )
    except Exception as e:
        logger.error(
            "Unexpected error updating profile for %s: %s", uid, e, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile service temporarily unavailable",
        )

    if "display_name" in updates and updates["display_name"]:
        try:
            firebase_auth.update_user(uid, display_name=updates["display_name"])
        except Exception as e:
            logger.warning("Firebase Auth display_name sync failed: %s", e)

    doc = db.collection("users").document(uid).get()
    user_data = doc.to_dict() if doc.exists else updates
    return _user_info_from_doc(uid, auth, user_data)
