# Mental Health Risk Assessment System (MHRAS)

Minimal mental health screening app with risk scoring, batch analysis, review workflow, and optional AI explanations.

## Features

- PHQ-9 / GAD-7 based risk assessment
- Individual and batch screening
- Risk distribution dashboard and export
- Role-based review flow
- Firebase auth + Firestore backend
- Optional Ollama-based clinical explanations

## Quick start

### UI-only mode

```bash
# open directly in browser
open index.html

# or serve locally
python -m http.server 8080
```

Then open `http://localhost:8080`.

### Backend mode

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.api.app:app --reload --port 8000
```

Then open the frontend via local static server or deployed app.

## Requirements

- Python 3.10+
- Firebase project with Authentication + Firestore
- Optional: Ollama for AI-generated explanations

## Project structure

```text
MentalHealthDataScience/
├── index.html
├── app.js
├── styles.css
├── src/
│   ├── api/
│   ├── config.py
│   ├── firebase_admin.py
│   ├── logging_config.py
│   └── risk_model.py
├── tests/
├── requirements.txt
├── .env.example
├── firestore.rules
├── firestore.indexes.json
└── README.md
```

## Notes

- The app includes a working client-side mock pipeline without backend setup.
- Production mode uses FastAPI, Firebase Auth, and Firestore.
- Sensitive data handling and audit logging are included for HIPAA-oriented workflows.

## License

Private — All rights reserved.
