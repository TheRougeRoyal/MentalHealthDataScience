# Mental Health Risk Assessment System - Frontend

A clean, user-friendly web interface for the MHRAS API.

## Features

- **Patient Screening**: Submit comprehensive mental health assessments
- **Risk Scoring**: View calculated risk scores with confidence levels
- **Recommendations**: Get personalized resource recommendations
- **Explanations**: Understand model predictions with interpretable explanations
- **Real-time Alerts**: Immediate notifications for high-risk cases

## Getting Started

### Prerequisites

- Python 3.8+ (for running the simple HTTP server)
- MHRAS API running on `http://localhost:8000`

### Running the Frontend

1. Start the API server (if not already running):
   ```bash
   python run_api.py
   ```

2. In a new terminal, start the frontend server:
   ```bash
   python run_frontend.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Usage

### 1. Authentication

Enter your JWT token in the Authentication section. For demo purposes, you can use a test token from your authentication system.

### 2. Patient Screening

Fill in the screening form with:

- **Anonymized Patient ID**: Unique identifier for the patient
- **Consent Verification**: Must be checked to proceed
- **Survey Data**: PHQ-9 and GAD-7 scores
- **Wearable Data**: Heart rate and sleep metrics
- **EMR Data**: Diagnosis codes and medications

### 3. View Results

After submission, you'll see:

- **Risk Score**: 0-100 scale with risk level classification
- **Contributing Factors**: Key factors influencing the risk score
- **Recommendations**: Personalized resources and interventions
- **Model Explanation**: Interpretable insights into the prediction

## Risk Levels

- **Low** (0-25): Minimal risk, routine monitoring
- **Moderate** (26-50): Some concern, preventive interventions
- **High** (51-75): Significant risk, active intervention needed
- **Critical** (76-100): Severe risk, immediate action required

## Example Data

For testing, you can use these sample values:

**Survey Data:**
- PHQ-9 Score: 15 (moderate depression)
- GAD-7 Score: 12 (moderate anxiety)

**Wearable Data:**
- Avg Heart Rate: 75 bpm
- Sleep Hours: 6.5 hours

**EMR Data:**
- Diagnosis Codes: F32.1, F41.1
- Medications: sertraline, lorazepam

## Security Notes

- Always use HTTPS in production
- Store JWT tokens securely
- Never share patient data
- Comply with HIPAA and data protection regulations

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Troubleshooting

**"Assessment failed: Failed to fetch"**
- Ensure the API server is running on port 8000
- Check that CORS is properly configured

**"Invalid authentication credentials"**
- Verify your JWT token is valid
- Check token expiration

**"Consent not verified"**
- Ensure the consent checkbox is checked before submission
