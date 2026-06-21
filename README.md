# Mental Health Risk Assessment System (MHRAS)

Clinical-grade mental health screening platform with structured risk modelling, explainable assessments, and role-gated review workflows.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Backend
uvicorn src.api.app:app --reload --port 8000

# Frontend (separate terminal)
python -m http.server 3000
# Open http://localhost:3000
```

## Tests

```bash
pytest tests/ -v
```

## Architecture

| Layer | Technology | Deploy |
|-------|-----------|--------|
| Frontend | HTML/CSS/JS | Vercel |
| Backend | FastAPI + Python 3.11 | Railway / Docker |
| Database | SQLite (dev) / PostgreSQL (prod) | — |
| Auth | JWT via HTTP-only cookies | — |
| Risk Model | Clinical rules engine | — |

## Default Credentials

- **admin** / admin (admin role)
- **reviewer** / reviewer (reviewer role)
