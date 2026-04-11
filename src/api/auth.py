"""Authentication and authorization for MHRAS API.

Implements:
- JWT access + refresh tokens delivered via HTTP-only cookies
- Role-based access control (admin, reviewer, user)
- Login / refresh / logout / me endpoints
- FastAPI dependency for extracting the current user from cookies
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from settings
# ---------------------------------------------------------------------------

_SECRET = settings.security.jwt_secret
_ALGORITHM = settings.security.jwt_algorithm
_ACCESS_EXPIRE_MINUTES = 30
_REFRESH_EXPIRE_DAYS = 7

_COOKIE_ACCESS = "access_token"
_COOKIE_REFRESH = "refresh_token"
_COOKIE_SECURE = settings.environment != "development"
_COOKIE_SAMESITE = "lax"

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Hardcoded user store (swap for a DB table in production)
# ---------------------------------------------------------------------------

_USERS: dict[str, dict] = {
    "admin": {
        "hashed_password": pwd_context.hash("admin"),
        "role": "admin",
        "display_name": "System Admin",
    },
    "reviewer": {
        "hashed_password": pwd_context.hash("reviewer"),
        "role": "reviewer",
        "display_name": "Clinical Reviewer",
    },
}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AuthResult(BaseModel):
    """Injected into every authenticated endpoint."""
    authenticated: bool
    user_id: str | None = None
    role: str | None = None
    error: str | None = None


class UserInfo(BaseModel):
    user_id: str
    role: str
    display_name: str


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    payload["iat"] = datetime.now(timezone.utc)
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def create_access_token(user_id: str, role: str) -> str:
    return _create_token(
        {"sub": user_id, "role": role, "type": "access"},
        timedelta(minutes=_ACCESS_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str, role: str) -> str:
    return _create_token(
        {"sub": user_id, "role": role, "type": "refresh"},
        timedelta(days=_REFRESH_EXPIRE_DAYS),
    )


def decode_token(token: str, *, expected_type: str = "access") -> dict:
    """Decode and validate a JWT. Raises ``HTTPException`` on failure."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    if payload.get("type") != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Expected {expected_type} token",
        )

    return payload


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------


def _set_auth_cookies(response: Response, user_id: str, role: str) -> None:
    """Write access + refresh tokens as HTTP-only cookies."""
    access = create_access_token(user_id, role)
    refresh = create_refresh_token(user_id, role)

    response.set_cookie(
        key=_COOKIE_ACCESS,
        value=access,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        max_age=_ACCESS_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key=_COOKIE_REFRESH,
        value=refresh,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        max_age=_REFRESH_EXPIRE_DAYS * 86400,
        path="/auth/refresh",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(_COOKIE_ACCESS, path="/")
    response.delete_cookie(_COOKIE_REFRESH, path="/auth/refresh")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def get_current_user(request: Request) -> AuthResult:
    """Extract the current user from the ``access_token`` cookie.

    In **development** mode, missing cookies fall back to a dev admin
    identity so the app remains usable without login.
    """
    token = request.cookies.get(_COOKIE_ACCESS)

    if not token:
        if settings.environment == "development":
            return AuthResult(
                authenticated=True,
                user_id="dev_user",
                role="admin",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = decode_token(token, expected_type="access")
    return AuthResult(
        authenticated=True,
        user_id=payload["sub"],
        role=payload.get("role", "user"),
    )


def require_role(*allowed_roles: str):
    """Return a dependency that enforces role-based access.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role("admin"))])
        def admin_view(): ...
    """

    def _checker(auth: AuthResult = Depends(get_current_user)) -> AuthResult:
        if auth.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{auth.role}' is not allowed. "
                       f"Required: {', '.join(allowed_roles)}",
            )
        return auth

    return _checker


# ---------------------------------------------------------------------------
# Auth router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login")
async def login(body: LoginRequest, response: Response):
    """Authenticate with username/password. Sets HTTP-only cookies."""
    user_record = _USERS.get(body.username)
    if not user_record or not pwd_context.verify(
        body.password, user_record["hashed_password"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    _set_auth_cookies(response, body.username, user_record["role"])
    logger.info("User %s logged in (role=%s)", body.username, user_record["role"])

    return {
        "message": "Login successful",
        "user_id": body.username,
        "role": user_record["role"],
        "display_name": user_record["display_name"],
    }


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    """Issue a new access token using the refresh-token cookie."""
    token = request.cookies.get(_COOKIE_REFRESH)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token",
        )

    payload = decode_token(token, expected_type="refresh")
    user_id = payload["sub"]
    role = payload.get("role", "user")

    # Issue a fresh access cookie (refresh cookie stays valid).
    access = create_access_token(user_id, role)
    response.set_cookie(
        key=_COOKIE_ACCESS,
        value=access,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        max_age=_ACCESS_EXPIRE_MINUTES * 60,
        path="/",
    )

    logger.info("Access token refreshed for user %s", user_id)
    return {"message": "Token refreshed"}


@router.post("/logout")
async def logout(response: Response):
    """Clear authentication cookies."""
    _clear_auth_cookies(response)
    return {"message": "Logged out"}


@router.get("/me")
async def me(auth: AuthResult = Depends(get_current_user)):
    """Return the identity of the currently authenticated user."""
    user_record = _USERS.get(auth.user_id, {})
    return UserInfo(
        user_id=auth.user_id,
        role=auth.role or "user",
        display_name=user_record.get("display_name", auth.user_id),
    )


# ---------------------------------------------------------------------------
# Legacy global (kept for backward compat with older code that imports it)
# ---------------------------------------------------------------------------

class Authenticator:
    """Thin wrapper kept for backward compatibility."""

    def generate_token(self, user_id: str, role: str = "user", **_) -> str:
        return create_access_token(user_id, role)

    def verify_token(self, token: str) -> AuthResult:
        try:
            payload = decode_token(token, expected_type="access")
            return AuthResult(
                authenticated=True,
                user_id=payload["sub"],
                role=payload.get("role", "user"),
            )
        except HTTPException as exc:
            return AuthResult(authenticated=False, error=exc.detail)


authenticator = Authenticator()
