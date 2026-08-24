"""Admin-only endpoints.

Currently exposes the user roster so the admin can see who has signed up
and what role they hold. Plain users get 403 from ``require_role("admin")``.
"""

from __future__ import annotations

import logging

from firebase_admin import auth as firebase_auth, firestore
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from src.api.auth import AuthResult, require_role
from src.firebase_admin import get_firestore_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


class AdminUserOut(BaseModel):
    uid: str
    email: str | None = None
    role: str
    display_name: str | None = None
    provider: str | None = None


@router.get("/users", response_model=list[AdminUserOut])
async def list_users(auth: AuthResult = Depends(require_role("admin"))):
    """Return the full user roster. Admin-only.

    Strict isolation: this endpoint is the ONLY way a human (or the admin
    frontend) can enumerate users. The Firestore rules also block clients
    from running `list` on /users, so this server-side path is the
    single source of truth.
    """
    db = get_firestore_client()
    docs = list(db.collection("users").get())
    out: list[AdminUserOut] = []
    for d in docs:
        data = d.to_dict() or {}
        out.append(AdminUserOut(
            uid=data.get("uid", d.id),
            email=data.get("email"),
            role=data.get("role", "user"),
            display_name=data.get("display_name"),
            provider=data.get("provider"),
        ))
    # Stable ordering by email for the admin table.
    out.sort(key=lambda u: (u.email or "").lower())
    return out


class UpdateUserRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(admin|user)$")


class AdminInviteRequest(BaseModel):
    email: EmailStr


@router.post("/invites", status_code=status.HTTP_201_CREATED)
async def invite_admin(
    body: AdminInviteRequest,
    auth: AuthResult = Depends(require_role("admin")),
):
    """Create an auditable admin invitation for a registered Firebase user."""
    db = get_firestore_client()
    try:
        invited = firebase_auth.get_user_by_email(str(body.email))
    except firebase_auth.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User must register before receiving an admin role")
    db.collection("admin_invites").document(invited.uid).set({
        "uid": invited.uid,
        "email": str(body.email).lower(),
        "invited_by": auth.user_id,
        "status": "pending",
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    return {"uid": invited.uid, "email": str(body.email).lower(), "status": "pending"}


@router.patch("/users/{uid}/role", response_model=AdminUserOut)
async def update_user_role(
    uid: str,
    body: UpdateUserRoleRequest,
    auth: AuthResult = Depends(require_role("admin")),
):
    """Set another user's role. Admin-only.

    Admin changes update Firebase custom claims and take effect after the user
    refreshes their ID token.
    """
    if body.role not in {"admin", "user"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="role must be 'admin' or 'user'",
        )
    db = get_firestore_client()
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    db.collection("users").document(uid).update({"role": body.role})
    target_user = firebase_auth.get_user(uid)
    claims = dict(target_user.custom_claims or {})
    if body.role == "admin":
        claims.update({"admin": True, "role": "admin"})
    else:
        claims.pop("admin", None)
        if claims.get("role") == "admin":
            claims.pop("role")
    firebase_auth.set_custom_user_claims(uid, claims or None)
    db.collection("admin_invites").document(uid).set({
        "uid": uid,
        "status": "accepted" if body.role == "admin" else "revoked",
        "updated_by": auth.user_id,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)
    data = (db.collection("users").document(uid).get().to_dict() or {})
    logger.info("Admin %s set user %s role=%s", auth.user_id, uid, body.role)
    return AdminUserOut(
        uid=uid,
        email=data.get("email"),
        role=data.get("role", body.role),
        display_name=data.get("display_name"),
        provider=data.get("provider"),
    )
