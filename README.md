# Mental Health Risk Assessment System (MHRAS)

A clinical-grade mental health screening platform with structured risk modelling, explainable assessments, role-gated review workflows, AI-generated clinical interpretations, and HIPAA-compliant data handling.

## Overview

MHRAS is a web-based mental health screening system that combines evidence-based clinical rules with AI-enhanced explanations to provide comprehensive risk assessments. The system implements a pluggable architecture for risk models, supporting both deterministic scoring and AI-generated clinical interpretations.

## Features

### Core Assessment Features
- **Multi-factor Risk Assessment**: Evaluates mental health risk using PHQ-9, GAD-7, sleep patterns, heart rate, diagnosis codes, and medications
- **Explainable AI**: Provides clinical explanations with contributing factors and counterfactual scenarios
- **Role-Based Access Control**: Admin, reviewer, and user roles with server-side enforcement
- **Clinical Review Workflow**: Enables healthcare professionals to review and manage flagged cases
- **HIPAA Compliance**: Audit logging, encryption, and PHI handling per 45 CFR Part 164

### UI-Only Mode Features
- **🎯 Individual Assessments**: Real-time risk scoring with interactive what-if simulator
- **📊 Batch Analytics**: Process up to 100 records simultaneously with visual distribution analysis
- **📈 Statistical Dashboard**: Auto-generated mock data (150+ screenings) with real-time statistics
- **💾 Export Capabilities**: Download batch results in CSV or JSON format
- **🎨 Theme Support**: Dark/Light mode with persistent preference
- **📱 Responsive Design**: Mobile-optimized interface
- **🔄 Trend Tracking**: Local storage of screening history with sparkline visualization
- **🎛️ What-If Simulator**: Interactive sliders to explore risk score changes

### Batch Processing Pipeline
```
JSON Input → Validation → Individual Scoring → Aggregation → Visualization → Export
            (consent)    (clientScore())     (distribution)  (charts)     (CSV/JSON)
```

**Batch Features:**
- Risk distribution bar chart with color-coded levels
- Summary statistics (total, successful, failed)
- Individual result cards with badges
- Alert and review flags
- One-click export to CSV or JSON

## Quick Start

### UI-Only Mode (No Backend Required)

The platform includes a **fully functional UI-only mode** with mock data pipeline. All features work immediately without backend setup:

```bash
# Simply open in browser
open index.html

# Or use a local server
python -m http.server 8080
# Navigate to http://localhost:8080
```

**Features available in UI-only mode:**
- ✅ Individual risk assessments with client-side scoring
- ✅ Batch analytics (up to 100 records)
- ✅ Statistical analysis (150 auto-generated mock screenings)
- ✅ What-if simulator with real-time scoring
- ✅ Review queue with demo cases
- ✅ CSV/JSON export for batch results
- ✅ Risk distribution visualizations
- ✅ Interactive charts and gauges

### Backend API Setup (Optional)

For production deployment with persistent data storage:

### Prerequisites

- Python 3.10+
- Firebase project with Authentication and Firestore enabled
- (Optional) Ollama for AI-enhanced explanations

### Installation

```bash
# Clone the repository
git clone https://github.com/TheRougeRoyal/MentalHealthDataScience.git
cd MentalHealthDataScience

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Firebase configuration
```

### Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable **Authentication** (Email/Password and Google)
3. Create a **Firestore Database**
4. Generate a service account key and save as `service-account.json`
5. Deploy Firestore rules: `firebase deploy --only firestore:rules`
6. Set the Firebase web config environment variables (`FIREBASE_API_KEY`,
   `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`, `FIREBASE_STORAGE_BUCKET`,
   `FIREBASE_MESSAGING_SENDER_ID`, and `FIREBASE_APP_ID`) in the backend
   deployment. The frontend loads these public values from `/api/auth/config`.

### Running the Application

```bash
# Start backend server
uvicorn src.api.app:app --reload --port 8000

# Start frontend (separate terminal)
python -m http.server 3000

# Access at http://localhost:3000
```

## Architecture

### UI-Only Mode Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (Client-Side)                        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Frontend (index.html)                       │  │
│   │   HTML5 · CSS3 · Vanilla JavaScript                      │  │
│   └───────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│   ┌───────────────────▼─────────────────────────────────────┐  │
│   │          Mock Data Pipeline (app.js)                     │  │
│   │                                                           │  │
│   │  ┌────────────────────────────────────────────────────┐ │  │
│   │  │  generateMockStatisticalData()                     │ │  │
│   │  │  → 150 realistic screening records                 │ │  │
│   │  │  → Risk score distribution                         │ │  │
│   │  └────────────────────────────────────────────────────┘ │  │
│   │                                                           │  │
│   │  ┌────────────────────────────────────────────────────┐ │  │
│   │  │  clientScore()                                     │ │  │
│   │  │  → Mirrors backend ClinicalRulesModel             │ │  │
│   │  │  → PHQ-9, GAD-7, Sleep, HR weighting              │ │  │
│   │  │  → Risk classification (Low/Med/High/Critical)    │ │  │
│   │  └────────────────────────────────────────────────────┘ │  │
│   │                                                           │  │
│   │  ┌────────────────────────────────────────────────────┐ │  │
│   │  │  Batch Processing                                  │ │  │
│   │  │  → Parse JSON array                                │ │  │
│   │  │  → Score each record                               │ │  │
│   │  │  → Generate distribution chart                     │ │  │
│   │  │  → Export to CSV/JSON                              │ │  │
│   │  └────────────────────────────────────────────────────┘ │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐    │
│   │  Local Storage                                         │    │
│   │  → Trend history                                       │    │
│   │  → Theme preference                                    │    │
│   │  → Batch results cache                                 │    │
│   └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Full Stack Architecture (Backend Connected)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend                                 │
│   index.html · app.js · styles.css · legal pages               │
│   Firebase Auth (Google SSO + Email/Password)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API (Bearer token)
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend                              │
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
| Backend | FastAPI + Python 3.12 | REST API |
| Database | Cloud Firestore | Patient data, reviews, audit trails |
| Auth | Firebase Authentication | Google + Email/Password with RBAC |
| Risk Model | Clinical rules engine + Ollama LLM | Deterministic scoring + AI explanations |

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

## Security

- **Authentication:** Firebase Authentication (Google SSO + Email/Password)
- **Authorization:** Role-based access control (RBAC) enforced server-side
- **Rate Limiting:** 120 requests/minute per IP via slowapi
- **CORS:** Restricted to known origins (configurable via `CORS_ORIGINS`)
- **Secrets:** Required env vars with fallback validation — app refuses to start without `SECURITY_JWT_SECRET` and `SECURITY_ANONYMIZATION_SALT`
- **Encryption:** TLS 1.2+ in transit, AES-256 at rest
- **HIPAA:** Business Associate Agreement with Google Cloud, PHI handling per 45 CFR Part 164
- **Audit Logging:** All data access and modifications logged to Firestore

## Environment Variables

```bash
# ── Required ────────────────────────────────────────────────────────────────
SECURITY_JWT_SECRET=          # Random hex string (min 64 chars)
SECURITY_ANONYMIZATION_SALT=  # Random hex string for patient ID hashing
FIREBASE_SERVICE_ACCOUNT_PATH= # Path to service account JSON

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

## Project Structure

```
MentalHealthDataScience/
├── api/
│   └── index.py              # Serverless entrypoint
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
├── requirements.txt          # Python dependencies
└── .env.example              # Environment template
```

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Health check and root endpoints
- Firebase authentication (valid token, dev mode, missing token)
- Individual screening (consent validation, scoring, risk levels)
- Batch screening (multiple patients, error handling)
- Statistics aggregation
- Risk model (low/high/critical scoring, confidence range, explanations, counterfactuals)

## License

Private — All rights reserved.
