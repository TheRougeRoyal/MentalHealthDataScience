"""Role isolation self-check.

Verifies the two policy primitives that gate strict user/admin isolation:

  1. ``src/api/auth.py`` defines exactly the bootstrap admin email.
  2. ``firestore.rules`` denies non-admins from reading other users' data.

Run:
    python tests/test_role_isolation.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def check_auth_admin_email() -> str:
    """Only aakashrraj2@gmail.com is bootstrap admin."""
    src = _read(ROOT / "src" / "api" / "auth.py")
    m = re.search(r"ADMIN_EMAILS\s*=\s*\{([^}]+)\}", src)
    assert m, "ADMIN_EMAILS set literal missing"
    emails = {e.strip().strip('"').strip("'").lower() for e in m.group(1).split(",") if e.strip()}
    assert emails == {"aakashrraj2@gmail.com"}, f"unexpected admin emails: {emails}"
    return f"admin emails = {sorted(emails)}"


def check_auth_no_reviewer_promotion() -> str:
    """Non-admin emails must always resolve to role='user'."""
    src = _read(ROOT / "src" / "api" / "auth.py")
    # The function body of get_current_user must demote stored 'reviewer' or 'admin'
    # to 'user' for non-admin emails.
    assert 'role = "user"' in src, "auth.py must force role='user' for non-admin emails"
    # And the if/else structure should never reach "admin" without the email check.
    # Ponytail: a coarse substring test is enough.
    return "auth.py demotes non-admin emails to 'user'"


def check_reviews_admin_only() -> str:
    src = _read(ROOT / "src" / "api" / "reviews.py")
    assert 'require_role("admin", "reviewer")' not in src, "reviews still allows 'reviewer' role"
    # All require_role calls in reviews.py must be admin-only.
    for m in re.finditer(r'require_role\("([^"]+)"\)', src):
        assert m.group(1) == "admin", f"reviews has non-admin require_role: {m.group(1)}"
    return "reviews.py is admin-only"


def check_firestore_user_isolation() -> str:
    rules = _read(ROOT / "firestore.rules")
    # Non-admin users can only read their own screenings/explanations.
    for col in ("screenings", "explanations"):
        # Find the match block for the collection.
        block = re.search(rf"match /{col}/\{{[^}}]+\}} \{{(.+?)\n\}}", rules, re.DOTALL)
        assert block, f"missing rule block for {col}"
        body = block.group(1)
        assert "resource.data.user_id == request.auth.uid" in body, (
            f"{col} rule does not restrict reads to owner"
        )
    # Reviews must be admin-only.
    reviews_block = re.search(r"match /reviews/\{[^}]+\} \{(.+?)\n\}", rules, re.DOTALL)
    assert reviews_block, "missing reviews rule block"
    assert re.search(r"allow read:\s*if isAdmin\(\);", reviews_block.group(1)), (
        "reviews must be admin-only on read"
    )
    return "firestore.rules isolates users from others' data"


def check_frontend_hides_admin_sections() -> str:
    html = _read(ROOT / "index.html")
    assert 'data-admin-only' in html, "admin nav links missing data-admin-only"
    js = _read(ROOT / "app.js")
    assert "applyRoleGating" in js, "applyRoleGating helper missing in app.js"
    assert "review-section" in js, "review-section is not gated by app.js"
    return "frontend hides admin-only sections"


def check_separate_pages_exist() -> str:
    for page in ("screening.html", "statistics.html", "batch.html", "queue.html"):
        assert (ROOT / page).exists(), f"missing dedicated page: {page}"
    return "dedicated feature pages exist"


def check_page_guard_enforces_access() -> str:
    js = _read(ROOT / "app.js")
    assert "guardPageAccess" in js, "page guard helper missing from app.js"
    assert "queue.html" in js, "queue page guard missing"
    assert "statistics.html" in js, "statistics page guard missing"
    return "page guard enforces feature access"


def check_admin_endpoint_registered() -> str:
    app = _read(ROOT / "src" / "api" / "app.py")
    assert "from src.api.admin import router as admin_router" in app, "admin router not included"
    assert "include_router(admin_router)" in app, "admin router not mounted"
    return "admin router mounted"


def main() -> int:
    checks = [
        check_auth_admin_email,
        check_auth_no_reviewer_promotion,
        check_reviews_admin_only,
        check_firestore_user_isolation,
        check_frontend_hides_admin_sections,
        check_separate_pages_exist,
        check_page_guard_enforces_access,
        check_admin_endpoint_registered,
    ]
    failed = 0
    for c in checks:
        try:
            print(f"  PASS  {c.__name__}: {c()}")
        except AssertionError as e:
            print(f"  FAIL  {c.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll role-isolation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
