# Mental Health Risk Assessment System (MHRAS)

Clinical-grade mental health screening platform with structured risk modelling, explainable assessments, role-gated review workflows, AI-generated clinical interpretations, and HIPAA-compliant data handling.

## Live

- **Frontend:** [mental-health-data-science.vercel.app](https://mental-health-data-science.vercel.app)
- **API docs:** [mental-health-data-science.vercel.app/docs](https://mental-health-data-science.vercel.app/docs)

## Quick Start

```bash
# Clone
git clone https://github.com/TheRougeRoyal/MentalHealthDataScience.git
cd MentalHealthDataScience

# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Add your Firebase service-account JSON (see Firebase Setup below)
# Edit .env and set SECURITY_JWT_SECRET, SECURITY_ANONYMIZATION_SALT

# Backend
uvicorn src.api.app:app --reload --port 8000

# Frontend (separate terminal)
python -m http.server 3000
# Open http://localhost:3000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vercel)                        │
│   index.html · app.js · styles.css · legal pages               │
│   Firebase Auth (Google SSO + Email/Password)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API (Bearer token)
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend (Vercel)                     │
│   api/index.py → src/api/app.py                                 │
│   ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│   │ auth.py  │ endpoint │ reviews  │ metrics  │ middleware.py │  │
│   │ (RBAC)   │ (screen) │ (review) │ (prom)   │ (security)   │  │
│   └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
│                           │                                      │
│   ┌───────────────────────▼──────────────────────────────────┐  │
│   │              Risk Model Layer (pluggable)                 │  │
│   │  ┌─────────────────────┐    ┌──────────────────────────┐ │  │
│   │  │ ClinicalRulesModel  │    │    OllamaRiskModel       │ │  │
│   │  │  score() + classify()│    │  score() → rules engine  │ │  │
│   │  │  explain() → rules   │    │  explain() → Ollama LLM  │ │  │
│   │  └─────────────────────┘    └──────────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  Cloud Firestore (Firebase)                      │
│   users · screenings · explanations · reviews                   │
│   Firestore Security Rules (RBAC enforced server-side)          │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML/CSS/JS + Firebase Auth | SPA with Google SSO + email auth |
| Backend | FastAPI + Python 3.12 | REST API, serverless on Vercel |
| Database | Cloud Firestore | Patient data, reviews, audit trails |
| Auth | Firebase Authentication | Google + Email/Password with RBAC |
| Risk Model | Clinical rules engine + Ollama LLM | Deterministic scoring + AI explanations |
| Hosting | Vercel | Serverless functions + static hosting |
| Monitoring | Prometheus | Metrics at `/metrics` |

## Risk Model

MHRAS uses a **pluggable risk model** architecture. The `RiskModel` ABC (`src/risk_model.py`) defines a protocol that any scoring backend must implement:

```python
class RiskModel(ABC):
    def score(self, data) -> float          # probability [0, 1]
    def classify(self, probability) -> ...  # risk_level, score, alert, needs_review
    def explain(self, data, prob, level) -> # factors, features, text, counterfactual
```

### ClinicalRulesModel (default)

Evidence-based clinical rules that produce a calibrated probability from structured screening data. Each feature is normalized using clinically meaningful cut-offs and combined via a weighted sum through a sigmoid function.

**Input features:**

| Feature | Weight | Source | Clinical Basis |
|---------|--------|--------|----------------|
| PHQ-9 Score | 0.30 | Survey | Patient Health Questionnaire depression severity |
| GAD-7 Score | 0.22 | Survey | Generalized Anxiety Disorder severity |
| Sleep Hours | 0.18 | Wearable | Deviation from optimal 7–9 hours |
| Resting Heart Rate | 0.12 | Wearable | Elevated HR may indicate stress |
| Diagnosis Codes | 0.10 | EMR | ICD-10 psychiatric diagnoses (F1–F4) |
| Medications | 0.08 | EMR | Number of psychotropic medications |

**Risk classification:**

| Score | Level | Action |
|-------|-------|--------|
| 0–29 | Low | Self-help resources |
| 30–50 | Moderate | Mindfulness, CBT referral |
| 51–74 | High | Clinical follow-up recommended |
| 75–100 | Critical | Urgent evaluation, auto-flagged for review |

### OllamaRiskModel (AI-enhanced)

Hybrid model that uses the clinical rules engine for deterministic scoring and classification, then sends structured data to an Ollama LLM to generate natural-language clinical interpretations, contributing factors, and counterfactual scenarios.

**How it works:**
1. `score()` and `classify()` → delegated to `ClinicalRulesModel` (fast, deterministic)
2. `explain()` → calls Ollama API with a clinical prompt to generate:
   - Top 3–5 contributing factors
   - 2–3 sentence clinical interpretation (with disclaimer)
   - Counterfactual scenario (what could reduce risk)
3. Falls back to rules-engine explanations if Ollama is unreachable or returns invalid JSON

**Configuration (env vars):**

```
ML_OLLAMA_BASE_URL=http://localhost:11434    # or cloud endpoint
ML_OLLAMA_MODEL=llama3
ML_OLLAMA_API_KEY=your-key                   # for cloud-hosted instances
ML_OLLAMA_TIMEOUT=30
```

### Adding a Custom Model

Implement the `RiskModel` ABC and swap it in `get_risk_model()`:

```python
# src/risk_model.py
class MyCustomModel(RiskModel):
    def score(self, data):
        # Your ML inference logic
        return probability

    def classify(self, probability):
        # Reuse ClinicalRulesModel thresholds
        return ClinicalRulesModel.classify(self, probability)

    def explain(self, data, probability, risk_level):
        # Your explanation logic
        return factors, features, clinical_text, counterfactual

def get_risk_model() -> RiskModel:
    return MyCustomModel()
```

## Firebase Setup

### 1. Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/) → your project
2. **Authentication** → Sign-in method → Enable **Email/Password** and **Google**
3. **Firestore Database** → Create database (start in test mode for dev)
4. **Project Settings** → Service accounts → **Generate new private key**
5. Save the JSON file as `service-account.json` in the project root

The backend reads `FIREBASE_SERVICE_ACCOUNT_PATH=./service-account.json` by default. For Vercel, set `FIREBASE_SERVICE_ACCOUNT_JSON` env var to the full JSON string instead.

### 2. Deploy Firestore Rules
```bash
firebase deploy --only firestore:rules
```
Or copy the contents of `firestore.rules` into the Firebase Console → Firestore → Rules.

### 3. Create Your First Admin User
1. Sign up via the frontend (creates a Firebase Auth user + Firestore doc with role `"user"`)
2. In Firebase Console → Firestore → `users` collection → find your user doc
3. Change the `role` field from `"user"` to `"admin"`
4. Refresh the page — you now have admin access

### 4. User Roles
- **admin** — full access: screenings, reviews, user management
- **reviewer** — can view and update reviews
- **user** — can run screenings and view results (default for new sign-ups)

Roles are stored in Firestore at `users/{uid}/role` and checked server-side on every request.

## API Endpoints

All authenticated endpoints require `Authorization: Bearer <firebase-id-token>`.

### Core

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Health check |
| `/auth/me` | GET | Bearer | Current user info |
| `/screen` | POST | Bearer | Single screening |
| `/batch-screen` | POST | Bearer | Batch screening (≤100) |
| `/risk-score/{id}` | GET | Bearer | Retrieve risk score |
| `/explain` | POST | Bearer | Generate explanation |
| `/statistics` | GET | Bearer | Aggregate stats |

### Reviews (admin/reviewer only)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/reviews` | GET | admin/reviewer | List reviews (filterable by status) |
| `/reviews/{id}` | GET | admin/reviewer | Get single review with screening context |
| `/reviews/{id}` | PATCH | admin/reviewer | Update status + notes |
| `/reviews/{id}/assign` | POST | admin/reviewer | Assign reviewer |
| `/reviews/{id}/comment` | POST | admin/reviewer | Add clinical note |
| `/reviews/{id}/close` | POST | admin/reviewer | Close review |

### Monitoring

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/metrics` | GET | — | Prometheus metrics |

## Security

- **Authentication:** Firebase Authentication (Google SSO + Email/Password)
- **Authorization:** Role-based access control (RBAC) enforced server-side
- **Rate Limiting:** 120 requests/minute per IP via slowapi
- **CORS:** Restricted to known origins (configurable via `CORS_ORIGINS`)
- **Secrets:** Required env vars with fallback validation — app refuses to start without `SECURITY_JWT_SECRET` and `SECURITY_ANONYMIZATION_SALT`
- **Encryption:** TLS 1.2+ in transit, AES-256 at rest
- **HIPAA:** Business Associate Agreement with Google Cloud, PHI handling per 45 CFR Part 164
- **Audit Logging:** All data access and modifications logged to Firestore

## Legal

MHRAS includes comprehensive legal documentation tailored to clinical health data systems:

- **[Terms of Service](/terms.html)** — Acceptable use, clinical responsibilities, IP rights, liability
- **[Privacy Policy](/privacy.html)** — HIPAA/GDPR/CCPA compliance, data handling, patient rights
- **[HIPAA Notice](/hipaa.html)** — Notice of Privacy Practices, PHI disclosures, mental health protections
- **[Medical Disclaimer](/disclaimer.html)** — Not a diagnostic tool, algorithm limitations, emergency protocols

## Environment Variables

```bash
# ── Required ────────────────────────────────────────────────────────────────
SECURITY_JWT_SECRET=          # Random hex string (min 64 chars)
SECURITY_ANONYMIZATION_SALT=  # Random hex string for patient ID hashing
FIREBASE_SERVICE_ACCOUNT_JSON= # Full Firebase service account JSON (Vercel)
# OR
FIREBASE_SERVICE_ACCOUNT_PATH= # Path to service account JSON (local dev)

# ── Optional ────────────────────────────────────────────────────────────────
ENVIRONMENT=development        # development | production
CORS_ORIGINS=                  # Comma-separated allowed origins

# ML / Risk Model
ML_RISK_THRESHOLD_HIGH=51.0
ML_RISK_THRESHOLD_CRITICAL=75.0

# Ollama (AI-generated explanations)
ML_OLLAMA_BASE_URL=http://localhost:11434
ML_OLLAMA_MODEL=llama3
ML_OLLAMA_API_KEY=
ML_OLLAMA_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Deployment

### Vercel (production)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

Environment variables are set via `vercel env add`. The app deploys as:
- Static files: `index.html`, `app.js`, `styles.css`, legal pages
- Serverless function: `api/index.py` (FastAPI under `/api`)

### Local Development

```bash
uvicorn src.api.app:app --reload --port 8000
python -m http.server 3000
```

## Project Structure

```
MentalHealthDataScience/
├── api/
│   └── index.py              # Vercel serverless entrypoint
├── scripts/
│   └── seed_firestore.py     # Synthetic data seeder
├── src/
│   ├── api/
│   │   ├── app.py            # FastAPI app + middleware
│   │   ├── auth.py           # Firebase Auth + RBAC
│   │   ├── endpoints.py      # Screening + batch + statistics
│   │   ├── metrics.py        # Prometheus metrics
│   │   ├── models.py         # Pydantic request/response models
│   │   └── reviews.py        # Clinical review workflow
│   ├── config.py             # Pydantic settings (env-driven)
│   ├── firebase_admin.py     # Firebase SDK initialization
│   ├── logging_config.py     # Structured logging
│   └── risk_model.py         # RiskModel ABC + implementations
├── tests/
│   ├── conftest.py           # Mocked Firebase fixtures
│   ├── test_api_endpoints.py # API endpoint tests
│   └── test_risk_model.py    # Risk model unit tests
├── index.html                # Main frontend
├── app.js                    # Frontend JavaScript
├── styles.css                # Global styles
├── legal.css                 # Legal page styles
├── terms.html                # Terms of Service
├── privacy.html              # Privacy Policy
├── hipaa.html                # HIPAA Notice
├── disclaimer.html           # Medical Disclaimer
├── firestore.rules           # Firestore security rules
├── firestore.indexes.json    # Firestore composite indexes
├── vercel.json               # Vercel deployment config
├── requirements.txt          # Python dependencies
└── .env.example              # Environment template
```

## Tests

```bash
pytest tests/ -v
```

17 tests covering:
- Health check and root endpoints
- Firebase authentication (valid token, dev mode, missing token)
- Individual screening (consent validation, scoring, risk levels)
- Batch screening (multiple patients, error handling)
- Statistics aggregation
- Risk model (low/high/critical scoring, confidence range, explanations, counterfactuals)

## License

Private — All rights reserved.
