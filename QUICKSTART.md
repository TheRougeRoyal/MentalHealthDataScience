# Mental Health Risk Assessment System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ (optional - for full features)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example configuration
cp config/.env.example .env

# The default configuration works for development!
# No changes needed for basic testing
```

### Step 3: Start the API Server

```bash
python run_api.py
```

The API will start at: http://localhost:8000

### Step 4: Open the Frontend

Open a new terminal:

```bash
python run_frontend.py
```

The frontend will open at: http://localhost:3000

### Step 5: Test the System

1. **Open the frontend** in your browser: http://localhost:3000
2. **Skip authentication** - it's optional in development mode
3. **Fill in the screening form**:
   - Anonymized Patient ID: `test_patient_001`
   - Check "Patient consent verified"
   - PHQ-9 Score: `15` (moderate depression)
   - GAD-7 Score: `12` (moderate anxiety)
4. **Click "Run Risk Assessment"**
5. **View results** - risk score, recommendations, and explanations

## 🔧 Development Mode Features

### No Authentication Required
The system runs in development mode by default:
- ✅ No JWT token needed
- ✅ All endpoints accessible
- ✅ Full functionality available

### Optional: Generate a Token (for testing)

If you want to test with authentication:

```bash
# Generate a token
curl -X POST "http://localhost:8000/auth/token?user_id=test_user&role=admin"
```

Copy the `access_token` and paste it in the frontend.

## 📊 API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Test the API Directly

### Health Check
```bash
curl http://localhost:8000/health
```

### Run a Screening (No Auth Required)
```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{
    "anonymized_id": "test_001",
    "consent_verified": true,
    "survey_data": {
      "phq9_score": 15,
      "gad7_score": 12
    }
  }'
```

## 🗄️ Optional: Database Setup

For full features (audit logs, review queue, etc.):

```bash
# Quick setup
./setup_database.sh

# Or manual setup
python run_cli.py db migrate
```

See [DATABASE_QUICKSTART.md](DATABASE_QUICKSTART.md) for details.

## 🎯 What's Next?

### Explore Features
- **CLI Tool**: `python run_cli.py --help`
- **Model Management**: `python run_cli.py models list`
- **System Stats**: `python run_cli.py system stats`

### Read Documentation
- [SETUP.md](SETUP.md) - Detailed setup guide
- [README.md](README.md) - Complete documentation
- [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md) - Frontend usage
- [docs/](docs/) - Component documentation

### Production Deployment
- [Dockerfile](Dockerfile) - Docker deployment
- [k8s/](k8s/) - Kubernetes manifests
- [CREDENTIALS.md](CREDENTIALS.md) - Security guide

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Change port in run_api.py or run_frontend.py
# Or kill existing process:
lsof -ti:8000 | xargs kill -9  # API
lsof -ti:3000 | xargs kill -9  # Frontend
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Database Connection Issues
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Or skip database features - system works without it!
```

## 💡 Key Features

✅ **Multi-Modal Data Processing** - Survey, wearable, EMR data  
✅ **ML Risk Prediction** - Ensemble models with confidence scores  
✅ **Interpretable Explanations** - SHAP values and counterfactuals  
✅ **Personalized Recommendations** - Risk-based resource matching  
✅ **Governance & Compliance** - Audit logs, review queue, drift monitoring  
✅ **Production Ready** - Docker, Kubernetes, monitoring

## 📞 Need Help?

- Check [README.md](README.md) for comprehensive documentation
- Review [examples/](examples/) for code samples
- See [docs/](docs/) for detailed guides

---

**Ready to go!** 🎉 The system is now running and ready for testing.
