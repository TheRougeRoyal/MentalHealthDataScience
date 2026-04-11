# Mental Health Risk Assessment System (MHRAS)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

A clinical-grade mental health screening platform with structured risk modelling, explainable assessments, role-gated review workflows, and resource recommendations.

---

## Architecture

| Layer | Technology | Deployment |
|-------|-----------|------------|
| **Frontend** | HTML 5, CSS 3, Vanilla JS | **Vercel** (static) |
| **Backend API** | FastAPI, Python 3.11 | **Railway** / Docker |
| **Database** | PostgreSQL (prod) · SQLite (dev) | Railway Postgres / local |
| **Auth** | JWT via HTTP-only cookies | — |
| **Risk Model** | Modular model layer (clinical rules, future ML) | — |

```
Frontend (Vercel)              Backend (Railway / Docker)
  index.html                     ┌─ FastAPI ─────────────────┐
  app.js   ──── fetch() ────→   │  /screen                  │
  styles.css                     │  /risk-score/{id}         │
                                 │  /explain                 │
                                 │  /batch-screen            │
                                 │  /statistics              │
                                 │  /reviews/*               │
                                 │  /auth/*                  │
                                 └──── PostgreSQL ───────────┘
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/MentalHealthDataScience.git
cd MentalHealthDataScience
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   SECURITY_JWT_SECRET=<random-string>
#   DATABASE_URL=         (leave blank for SQLite fallback)
#   ENVIRONMENT=development
```

### 3. Run locally

```bash
# Backend API (auto-creates tables on startup)
uvicorn src.api.app:app --reload --port 8000

# Frontend (separate terminal)
python -m http.server 3000
# Open http://localhost:3000
```

### 4. Run tests

```bash
pytest tests/test_api_endpoints.py -v
```

---

## Production Deployment

### Backend → Railway

1. Connect your GitHub repo to [Railway](https://railway.app).
2. Railway auto-detects the `Dockerfile` or uses Nixpacks via `railway.json`.
3. Set environment variables in the Railway dashboard (see table below).
4. Provision a **Railway Postgres** add-on — set `DATABASE_URL` automatically.
5. Health check: `/health`

### Backend → Docker

```bash
docker build -t mhras-api .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  -e SECURITY_JWT_SECRET=change-me \
  -e ENVIRONMENT=production \
  mhras-api
```

### Frontend → Vercel

1. Import the repo to [Vercel](https://vercel.com).
2. Set **Framework Preset** to `Other`.
3. Set **Output Directory** to `.` (root).
4. Vercel serves `index.html`, `app.js`, `styles.css` as static files.
5. In `app.js`, set `API_BASE_URL` to  your Railway backend URL.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` or `production` |
| `DATABASE_URL` | _(SQLite fallback)_ | PostgreSQL connection string |
| `SECURITY_JWT_SECRET` | `change-me-in-production` | JWT signing key |
| `SECURITY_JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `ML_RISK_THRESHOLD_HIGH` | `51.0` | Score threshold for "high" risk |
| `ML_RISK_THRESHOLD_CRITICAL` | `75.0` | Score threshold for "critical" risk |
| `GOVERNANCE_HUMAN_REVIEW_THRESHOLD` | `75.0` | Score requiring human review |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` |

See [.env.example](.env.example) for the full list.

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Health check |
| `/` | GET | — | Service info |
| `/auth/login` | POST | — | Login (returns HTTP-only cookie) |
| `/auth/refresh` | POST | cookie | Refresh token |
| `/auth/logout` | POST | cookie | Clear session |
| `/auth/me` | GET | cookie | Current user info |
| `/screen` | POST | cookie | Single screening |
| `/batch-screen` | POST | cookie | Batch screening (≤100) |
| `/risk-score/{id}` | GET | cookie | Retrieve risk score by patient ID |
| `/explain` | POST | cookie | Retrieve/generate explanation |
| `/statistics` | GET | cookie | DB-backed system stats |
| `/reviews/queue` | GET | admin/reviewer | Review queue |
| `/reviews/{id}/assign` | POST | admin/reviewer | Assign reviewer |
| `/reviews/{id}/comment` | POST | admin/reviewer | Add comment |
| `/reviews/{id}/close` | POST | admin/reviewer | Close review |

---

## Project Structure

```
├── index.html / app.js / styles.css   # Frontend
├── src/
│   ├── api/
│   │   ├── app.py                     # FastAPI application
│   │   ├── endpoints.py               # Core API routes
│   │   ├── auth.py                    # JWT + cookie auth
│   │   ├── reviews.py                 # Review workflow routes
│   │   ├── middleware.py              # Logging, error handling
│   │   └── models.py                  # Pydantic schemas
│   ├── risk_model.py                  # Structured model layer (ABC + rules)
│   ├── database.py                    # SQLAlchemy engine + session
│   ├── models.py                      # ORM models
│   ├── config.py                      # Env-based settings
│   └── logging_config.py             # Structured logging (structlog)
├── tests/
│   ├── conftest.py                    # Fixtures (in-memory DB, auth overrides)
│   └── test_api_endpoints.py          # Endpoint tests
├── Dockerfile                         # Multi-stage production image
├── railway.json                       # Railway deploy config
├── vercel.json                        # Vercel static deploy config
├── .env.example                       # All env vars documented
└── requirements.txt
```

---

## Testing

```bash
# All API endpoint tests
pytest tests/test_api_endpoints.py -v

# Full suite with coverage
pytest --cov=src tests/

# Specific test class
pytest tests/test_api_endpoints.py::TestReviews -v
```

Tests use an **in-memory SQLite** database and override auth dependencies — no external services required.

---

## Author

**Aakash Raj** · [GitHub](https://github.com/aakashraj)

## License

Proprietary — for authorized clinical use only.