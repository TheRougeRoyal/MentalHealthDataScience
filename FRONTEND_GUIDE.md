# MHRAS Frontend - Quick Start Guide

## 🎉 Your Frontend is Ready!

The Mental Health Risk Assessment System now has a fully functional web interface.

## 🚀 Access the Application

**Frontend URL:** http://localhost:3000  
**API URL:** http://localhost:8000  
**API Docs:** http://localhost:8000/docs

## 📋 Current Status

✅ API Server: Running on port 8000  
✅ Frontend Server: Running on port 3000  
✅ CORS: Configured and enabled

## 🔑 Getting Started

### Step 1: Get an API Token

For demo purposes, you can generate a test token. Open your browser console at http://localhost:3000 and run:

```javascript
// This is a demo token - in production, use your auth system
const demoToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkZW1vX3VzZXIiLCJyb2xlIjoiY2xpbmljaWFuIn0.demo";
```

Or use the API documentation at http://localhost:8000/docs to test authentication.

### Step 2: Enter Token in Frontend

1. Open http://localhost:3000
2. Paste your token in the "API Token" field
3. Click "Save Token"

### Step 3: Submit a Screening

Fill in the form with patient data:

**Required:**
- Anonymized Patient ID (e.g., "patient_001")
- Check "Patient consent verified"

**Optional (but recommended for testing):**
- PHQ-9 Score: 15
- GAD-7 Score: 12
- Avg Heart Rate: 75
- Sleep Hours: 6.5
- Diagnosis Codes: F32.1
- Medications: sertraline

### Step 4: View Results

After submission, you'll see:
- Risk score and level
- Contributing factors
- Personalized recommendations
- Model explanations

## 🎨 Features

### Risk Assessment
- Real-time risk scoring (0-100)
- Four risk levels: Low, Moderate, High, Critical
- Confidence scores for predictions

### Recommendations
- Personalized resource suggestions
- Urgency levels (immediate, soon, routine)
- Contact information for resources

### Explanations
- Top contributing features
- Clinical interpretations
- What-if scenarios (counterfactuals)

### Alerts
- Automatic alerts for high-risk cases
- Human review queue notifications

## 🛠️ Troubleshooting

### "Failed to fetch" Error

**Problem:** Frontend can't connect to API

**Solutions:**
1. Verify API is running: `curl http://localhost:8000/health`
2. Check for firewall blocking port 8000
3. Ensure CORS is enabled (already configured)

### "Invalid authentication credentials"

**Problem:** Token is invalid or expired

**Solutions:**
1. Generate a new token
2. Check token format (should be JWT)
3. Verify token hasn't expired

### "Consent not verified"

**Problem:** Consent checkbox not checked

**Solution:** Check the "Patient consent verified" checkbox before submitting

## 📊 Testing the System

### Test Case 1: Low Risk Patient
```
Anonymized ID: patient_low_001
PHQ-9: 3
GAD-7: 2
Sleep Hours: 8
Heart Rate: 70
```

### Test Case 2: Moderate Risk Patient
```
Anonymized ID: patient_mod_001
PHQ-9: 10
GAD-7: 8
Sleep Hours: 6.5
Heart Rate: 75
```

### Test Case 3: High Risk Patient
```
Anonymized ID: patient_high_001
PHQ-9: 20
GAD-7: 18
Sleep Hours: 4
Heart Rate: 90
Diagnosis Codes: F32.2, F41.1
Medications: sertraline, lorazepam
```

## 🔒 Security Notes

⚠️ **Important for Production:**

1. **Use HTTPS:** Never use HTTP in production
2. **Secure Tokens:** Store JWT tokens securely (httpOnly cookies)
3. **Environment Variables:** Don't hardcode API URLs
4. **CORS:** Restrict allowed origins to your domain
5. **Rate Limiting:** Implement rate limiting on API
6. **Input Validation:** Always validate on both client and server
7. **HIPAA Compliance:** Ensure all data handling meets regulations

## 📱 Browser Compatibility

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+

## 🎯 Next Steps

1. **Customize Styling:** Edit `frontend/styles.css`
2. **Add Features:** Modify `frontend/app.js`
3. **Integrate Auth:** Connect to your authentication system
4. **Deploy:** Use nginx or similar for production serving

## 📞 Support

For issues or questions:
- Check API logs: Process ID 1
- Check browser console for frontend errors
- Review API documentation: http://localhost:8000/docs

---

**Enjoy using the Mental Health Risk Assessment System! 🧠💙**
