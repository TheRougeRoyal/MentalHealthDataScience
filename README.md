# Mental Health Risk Assessment System

MHRAS is a FastAPI and Firebase-based clinical decision-support application for structured mental-health screening. It includes PHQ-9/GAD-7 scoring, optional wearable and EMR features, explanations, review workflows, encrypted persistence, and a static frontend.

> MHRAS is decision-support software, not a diagnostic tool or a substitute for professional clinical judgment. The deployment is not represented as HIPAA compliant without a separate compliance review, appropriate agreements, policies, and operational controls.

## Project Layout

```text
api/                 Vercel serverless API entrypoint
frontend/            HTML, JavaScript, and CSS application files
scripts/             Role bootstrap, Firestore seeding, and retention jobs
src/                 FastAPI application, auth, model, privacy, and evaluation code
tests/               API, model, security, and authorization tests
firestore.rules      Firestore client access rules
vercel.json          Vercel build and route configuration
requirements.txt     Pinned Python dependencies
```

## Requirements

- Python 3.12+
- A Firebase project with Authentication and Firestore enabled
- Firebase service-account credentials for persistent deployments
- Optional: Ollama for generated explanations; the rules model remains the scoring source

## Local Setup

```bash
git clone https://github.com/TheRougeRoyal/MentalHealthDataScience.git
cd MentalHealthDataScience
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` before starting the API. At minimum, persistent environments need:

```env
FIREBASE_SERVICE_ACCOUNT_PATH=./service-account.json
SECURITY_JWT_SECRET=<long-random-secret>
SECURITY_ANONYMIZATION_SALT=<long-random-secret>
SECURITY_DATA_ENCRYPTION_KEY=<valid-fernet-key>
```

Generate a Fernet key with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Do not commit `.env`, service-account files, or secret values.

## Run Locally

Start the API:

```bash
source .venv/bin/activate
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /live` confirms the process is running.
- `GET /ready` checks Firestore and model readiness.
- `GET /health` is a readiness-compatible health endpoint.
- `POST /screen` evaluates one screening.
- `POST /batch-screen` evaluates up to 100 screenings.
- `GET /risk-score/{anonymized_id}` retrieves a user-scoped result.
- `POST /explain` retrieves an authorized explanation.
- `GET /docs` opens the OpenAPI documentation.

For local development, API paths use `/...`. On Vercel, the serverless entrypoint maps `/api/...` to the same FastAPI routes.

## Data Protection

- Screening inputs are minimized to model features before persistence.
- Clinical inputs are encrypted with Fernet using `SECURITY_DATA_ENCRYPTION_KEY`.
- Screening, explanation, review, and audit records use an atomic Firestore batch.
- Transient Firestore commit failures are retried up to three times.
- `Idempotency-Key` can be supplied to `POST /screen` and `POST /batch-screen` to replay completed requests safely.
- Audit records contain metadata only and do not contain clinical values.
- Retention cleanup removes expired screenings, linked explanations/reviews, and audit events.

Run retention cleanup from a scheduled job:

```bash
python scripts/retention_cleanup.py
```

Configure retention with `GOVERNANCE_SCREENING_RETENTION_DAYS` and `GOVERNANCE_AUDIT_LOG_RETENTION_DAYS`.

## Administration

Administrator authorization uses Firebase custom claims. Admin routes require MFA by default.

To bootstrap administrators, set protected UIDs and run:

```bash
export ADMIN_BOOTSTRAP_UIDS="firebase-uid-1,firebase-uid-2"
python scripts/bootstrap_roles.py --apply
```

The admin API supports:

- `POST /admin/invites` to create an auditable invitation for an existing Firebase user.
- `PATCH /admin/users/{uid}/role` to grant or revoke the admin claim.
- `GET /admin/users` to list users.

Set `REQUIRE_ADMIN_MFA=false` only for a controlled local environment. Users must refresh their Firebase ID token after a role change.

## Firestore Deployment

Deploy rules and indexes with the Firebase CLI after selecting the target project:

```bash
firebase use <project-id>
firebase deploy --only firestore:rules,firestore:indexes
```

Review `firestore.rules` before production deployment. Backend service-account access is governed separately from client Firestore rules.

## Model Validation

The default rules-based model reports `insufficient_data` with zero confidence when no recognized model features are provided. Valid results include model version and confidence provenance metadata.

Offline validation helpers in `src/model_evaluation.py` calculate:

- Brier score and expected calibration error
- Sensitivity, specificity, PPV, and NPV
- ROC-AUC
- Per-subgroup classification metrics

These metrics require a representative labelled evaluation dataset. They do not establish clinical performance by themselves.

## Tests and Quality Checks

Run the full test suite:

```bash
pytest -q
```

Run the authorization policy self-check:

```bash
python tests/test_role_isolation.py
```

GitHub Actions in `.github/workflows/ci.yml` runs tests, Ruff formatting/linting, mypy, pip-audit, and the Firestore authorization policy check for pushes and pull requests.

## Vercel Deployment

The repository contains an explicit Vercel configuration in `vercel.json`:

- `api/index.py` is the Python serverless entrypoint.
- All frontend assets are under `frontend/`.
- Public routes such as `/screening.html`, `/queue.html`, `/app.js`, and `/docs` are mapped to their frontend files.
- API requests are served under `/api/...`.

Configure Firebase credentials, encryption, authentication, CORS, and model environment variables in the Vercel project settings. Never place private credentials in the frontend directory.

## License and Legal Notices

See the pages under `frontend/` for the current terms, privacy information, HIPAA-related information, and medical disclaimer. Replace the deployment contact notice with the operating organization’s verified legal and privacy contacts before production use.
