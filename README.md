# Mental Health Risk Assessment System (MHRAS)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/MentalHealthDataScience)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)

A comprehensive mental health screening system with ML-powered risk assessment, clinical decision support, and resource recommendations.

**[Live Demo →](https://your-project.vercel.app)**

## Features

- **Mental Health Screening** - PHQ-9, GAD-7, wearable data, EMR integration
- **ML Risk Assessment** - Ensemble models (Logistic Regression, LightGBM)
- **Interpretable Predictions** - SHAP values, counterfactuals, clinical explanations
- **Resource Recommendations** - Crisis lines, therapy, support groups
- **Demo Mode** - Full UI with simulated predictions for testing
- **Batch Processing** - Screen up to 100 individuals at once

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | FastAPI, Python 3.11 |
| ML | scikit-learn, LightGBM, SHAP |
| Database | PostgreSQL, Redis |
| Deployment | Vercel (serverless), Docker, Kubernetes |

## Quick Start

### Option 1: Deploy to Vercel (Recommended)

1. Click the **Deploy with Vercel** button above
2. Fork or clone the repository
3. Deploy - works immediately in **Demo Mode**

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MentalHealthDataScience.git
cd MentalHealthDataScience

# Run frontend (quickest)
python -m http.server 3000
# Open http://localhost:3000
```

### Option 3: Full Stack with Docker

```bash
# Start all services
docker-compose up

# Initialize database
./setup_database.sh

# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Demo Mode

The system runs in **Demo Mode** by default on Vercel, providing:

- Simulated risk predictions based on PHQ-9 and GAD-7 scores
- Full UI functionality for demonstrations
- Sample data buttons for quick testing

### Demo Features

- **Load Sample Data** - Pre-fill forms with realistic patient data
- **Batch Screening** - Test with multiple patients at once
- **Risk Visualization** - See risk scores, contributing factors, and recommendations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/statistics` | GET | Model and queue statistics |
| `/api/screen` | POST | Single patient screening |
| `/api/batch-screen` | POST | Batch screening (max 100) |

### Example Request

```bash
curl -X POST https://your-project.vercel.app/api/screen \
  -H "Content-Type: application/json" \
  -d '{
    "anonymized_id": "patient_001",
    "consent_verified": true,
    "survey_data": {
      "phq9_score": 15,
      "gad7_score": 12
    }
  }'
```

## Architecture

```
Vercel (Frontend + Serverless)
├── index.html, app.js, styles.css
└── api/
    ├── health.js      - Health check
    ├── statistics.js   - System stats
    ├── screen.js       - Single screening
    ├── batch-screen.js - Batch screening
    └── index.js        - API info
         │
         ▼ (optional proxy)
    FastAPI Backend
    ├── ML Models (LightGBM)
    ├── PostgreSQL
    └── Redis
```

## Project Structure

```
├── api/                    # Vercel serverless functions
├── src/                    # Backend source code
│   ├── api/               # FastAPI endpoints
│   ├── ds/                # Data science modules
│   ├── governance/        # Compliance & audit
│   └── database/          # Database models
├── tests/                 # Test suite
├── monitoring/            # Prometheus + Grafana configs
├── k8s/                   # Kubernetes manifests
└── examples/             # Example notebooks
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MHRAS_API_URL` | No | Backend URL for full-stack mode |

For full-stack deployment, set these additional variables:
| Variable | Description |
|----------|-------------|
| `DB_HOST` | PostgreSQL host |
| `DB_PASSWORD` | Database password |
| `REDIS_URL` | Redis connection URL |

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_screening_service.py -v
```

## Author

**Aakash Raj**
- GitHub: [@aakashraj](https://github.com/aakashraj)

## License

Proprietary - For authorized clinical use only.