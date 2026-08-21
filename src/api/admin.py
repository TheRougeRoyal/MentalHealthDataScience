"""Admin-only endpoints.

Currently exposes the user roster so the admin can see who has signed up
and what role they hold. Plain users get 403 from ``require_role("admin")``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

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
    role: str


@router.patch("/users/{uid}/role", response_model=AdminUserOut)
async def update_user_role(
    uid: str,
    body: UpdateUserRoleRequest,
    auth: AuthResult = Depends(require_role("admin")),
):
    """Set another user's role. Admin-only.

    Note: the bootstrap admin (aakashrraj2@gmail.com) is re-promoted on every
    login by auth.py, so demoting them here is harmless but pointless.
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
    data = (db.collection("users").document(uid).get().to_dict() or {})
    logger.info("Admin %s set user %s role=%s", auth.user_id, uid, body.role)
    return AdminUserOut(
        uid=uid,
        email=data.get("email"),
        role=data.get("role", body.role),
        display_name=data.get("display_name"),
        provider=data.get("provider"),
    )
