"""Authentication and authorization for MHRAS API (Firebase Auth).

- Frontend sends Firebase ID token in ``Authorization: Bearer <token>``
- Backend verifies with ``firebase_admin.auth.verify_id_token()``
- Role is derived exclusively from Firebase custom claims (``admin`` claim)
- Role checks happen server-side only via ``require_role``

Profile endpoints (GET/PATCH /auth/me) live in ``src.api.profile``.
"""

from __future__ import annotations

import logging
import os

from firebase_admin.auth import RevokedIdTokenError, UserDisabledError
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model — injected into every authenticated endpoint
# ---------------------------------------------------------------------------

class AuthResult(BaseModel):
    """Carries the verified identity and role for an authenticated request."""
    authenticated: bool
    user_id: str | None = None
    email: str | None = None
    role: str | None = None
    display_name: str | None = None
    photo_url: str | None = None
    provider: str | None = None
    mfa_verified: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dev_bypass_active() -> bool:
    """Return True only when BOTH guard variables are explicitly set for development."""
    return (
        os.environ.get("ALLOW_DEV_AUTH_BYPASS", "").lower() == "true"
        and os.environ.get("ENV", "").lower() == "development"
    )


# ---------------------------------------------------------------------------
# FastAPI dependency — token verification
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> AuthResult:
    """Extract and verify the Firebase ID token from the Authorization header.

    Role is derived purely from Firebase custom claims (``admin`` claim).
    No Firestore read occurs here — profile enrichment and first-login
    provisioning happen lazily in GET /auth/me (see ``src.api.profile``).

    Dev bypass requires BOTH ``ALLOW_DEV_AUTH_BYPASS=true`` AND
    ``ENV=development``. The app startup guard in ``app.py`` refuses to start
    if the bypass is armed without ``ENV=development``.
    """
    from src.firebase_admin import verify_id_token

    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        if _dev_bypass_active():
            logger.warning(
                "!!! DEV AUTH BYPASS ACTIVE — unauthenticated request granted admin access. "
                "ALLOW_DEV_AUTH_BYPASS=true and ENV=development are both set. "
                "NEVER deploy this configuration to production !!!"
            )
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
    except (RevokedIdTokenError, UserDisabledError) as e:
        logger.warning("Token rejected — revoked or disabled account: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
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
    role = "admin" if is_admin else "user"
    # stored_role in Firestore is always normalised to match this claim-derived
    # value on first login (see _provision_new_user in src.api.profile).
    mfa_verified = bool(firebase_claims.get("sign_in_second_factor"))

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


# ---------------------------------------------------------------------------
# Role enforcement dependency
# ---------------------------------------------------------------------------

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
        if (
            auth.role == "admin"
            and os.environ.get("REQUIRE_ADMIN_MFA", "true").lower() == "true"
            and not auth.mfa_verified
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="MFA is required for administrator access",
            )
        return auth
    return _checker


# ---------------------------------------------------------------------------
# Auth router  (/auth/config only — /auth/me lives in src.api.profile)
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
