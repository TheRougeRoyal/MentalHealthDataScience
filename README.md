# Mental Health Risk Assessment System (MHRAS)

Clinical-grade mental health screening platform with structured risk modelling, explainable assessments, and role-gated review workflows.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your Firebase service-account JSON (see Firebase Setup below)

# Backend
uvicorn src.api.app:app --reload --port 8000

# Frontend (separate terminal)
python -m http.server 3000
# Open http://localhost:3000
```

## Firebase Setup

### 1. Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/) → your project
2. **Authentication** → Sign-in method → Enable **Email/Password** and **Google**
3. **Firestore Database** → Create database (start in test mode for dev)
4. **Project Settings** → Service accounts → **Generate new private key**
5. Save the JSON file as `service-account.json` in the project root

The backend reads `FIREBASE_SERVICE_ACCOUNT_PATH=./service-account.json` by default. For Railway/Vercel, set `FIREBASE_SERVICE_ACCOUNT_JSON` env var to the full JSON string instead.

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

## Architecture

| Layer | Technology | Deploy |
|-------|-----------|--------|
| Frontend | HTML/CSS/JS + Firebase Auth (Google + Email) | Vercel |
| Backend | FastAPI + Python 3.11 | Railway / Docker |
| Database | Cloud Firestore | Firebase |
| Auth | Firebase Authentication | — |
| Risk Model | Clinical rules engine | — |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Health check |
| `/auth/me` | GET | Bearer | Current user info |
| `/screen` | POST | Bearer | Single screening |
| `/batch-screen` | POST | Bearer | Batch screening (≤100) |
| `/risk-score/{id}` | GET | Bearer | Retrieve risk score |
| `/explain` | POST | Bearer | Generate explanation |
| `/statistics` | GET | Bearer | Aggregate stats |
| `/reviews` | GET | admin/reviewer | List reviews |
| `/reviews/{id}` | GET | admin/reviewer | Get single review |
| `/reviews/{id}` | PATCH | admin/reviewer | Update review |
| `/reviews/{id}/assign` | POST | admin/reviewer | Assign reviewer |
| `/reviews/{id}/comment` | POST | admin/reviewer | Add note |
| `/reviews/{id}/close` | POST | admin/reviewer | Close review |

## Tests

```bash
pytest tests/ -v
```
