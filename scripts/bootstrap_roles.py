"""One-shot Firestore role bootstrap.

Enforces the policy: aakashrraj2@gmail.com is the only admin; every other
user is a plain 'user'. Idempotent — safe to re-run.

Usage:
    python scripts/bootstrap_roles.py            # dry-run, prints plan
    python scripts/bootstrap_roles.py --apply    # writes to Firestore
    python scripts/bootstrap_roles.py --list     # dump current roles
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.firebase_admin import _init_app, get_firestore_client
from firebase_admin import auth as firebase_auth

ADMIN_UIDS = {
    value.strip() for value in os.environ.get("ADMIN_BOOTSTRAP_UIDS", "").split(",") if value.strip()
}


def list_roles() -> list[dict]:
    _init_app()
    db = get_firestore_client()
    if db is None:
        raise RuntimeError("Firestore is required for role bootstrap")
    out = []
    for doc in db.collection("users").get():
        data = doc.to_dict() or {}
        out.append({
            "uid": doc.id,
            "email": (data.get("email") or "").lower(),
            "role": data.get("role", "user"),
        })
    out.sort(key=lambda u: u["email"] or u["uid"])
    return out


def plan_changes(rows: list[dict]) -> list[tuple[str, str, str]]:
    """Return list of (uid, current_role, target_role) for users that need updating."""
    changes = []
    for u in rows:
        current = u["role"]
        target = "admin" if u["uid"] in ADMIN_UIDS else "user"
        if current != target:
            changes.append((u["uid"], current, target))
    return changes


def apply(changes: list[tuple[str, str, str]]) -> None:
    db = get_firestore_client()
    if db is None:
        raise RuntimeError("Firestore is required for role bootstrap")
    batch = db.batch()
    for uid, _current, target in changes:
        batch.update(db.collection("users").document(uid), {"role": target})
        print(f"  set {uid} role={target}")
    if changes:
        batch.commit()
        for uid, _current, target in changes:
            user = firebase_auth.get_user(uid)
            claims = dict(user.custom_claims or {})
            if target == "admin":
                claims.update({"admin": True, "role": "admin"})
            else:
                claims.pop("admin", None)
                if claims.get("role") == "admin":
                    claims.pop("role")
            firebase_auth.set_custom_user_claims(uid, claims or None)
        print(f"\nApplied {len(changes)} role update(s).")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap Firestore roles")
    parser.add_argument("--apply", action="store_true", help="Apply changes to Firestore")
    parser.add_argument("--list", action="store_true", help="List current roles and exit")
    args = parser.parse_args()

    rows = list_roles()
    if args.list:
        for u in rows:
            print(f"  {u['email'] or u['uid']:<40} {u['role']}")
        sys.exit(0)

    print(f"Found {len(rows)} user(s). Admin UIDs: {sorted(ADMIN_UIDS)}")
    changes = plan_changes(rows)
    if not changes:
        print("All users already match policy. Nothing to do.")
        sys.exit(0)

    print(f"\n{len(changes)} change(s) planned:")
    for uid, current, target in changes:
        print(f"  {uid}: {current!r} -> {target!r}")
    if args.apply:
        apply(changes)
    else:
        print("\nDry-run only. Re-run with --apply to commit.")
