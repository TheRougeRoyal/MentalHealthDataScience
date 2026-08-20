# MindMetrics AI - Interactive Demo Guide

## 🚀 Quick Start Demo

### Open the Application
1. Open `index.html` in your browser (works offline!)
2. Or run: `python -m http.server 8080` and visit http://localhost:8080

---

## 🎯 Feature Demonstrations

### 1. Statistical Analysis Dashboard (Auto-loads on Page Load)

**What you'll see:**
- API Status showing "UI Mode" (indicates client-side mock data)
- Total Screenings: ~150 records (auto-generated)
- High Risk Flagged: Dynamic count based on mock data
- Pending Reviews: Calculated from high-risk cases

**Statistical Analysis Section:**
- Average risk score across dataset
- Median risk score
- Score range (min-max)
- Risk group distribution breakdown

**Try this:**
- Click "Refresh Analysis" to regenerate the entire mock dataset with new random values
- Each refresh creates 150 new screening records with realistic clinical distributions

---

### 2. Individual Assessment Demo

**Step-by-step:**

1. **Navigate to "Screening Tool" section**

2. **Load Sample Data:**
   - Click "Autofill Test Sample" button
   - Observe pre-filled values:
     - Anonymized ID: `demo_patient_sample`
     - PHQ-9: 15 (Moderately severe depression)
     - GAD-7: 12 (Moderate anxiety)
     - Sleep: 5.5 hours (Sleep deprivation)
     - Heart Rate: 78 bpm (Slightly elevated)

3. **Run Assessment:**
   - Ensure "Data Protection & Privacy Consent Verified" is checked
   - Click "Run Statistical Risk Model"
   - Watch the loading indicator

4. **Explore Results:**
   - **Risk Score Gauge**: Visual semicircle gauge with needle
   - **Risk Badge**: Color-coded level (Low/Moderate/High/Critical)
   - **Confidence Score**: Model confidence percentage
   - **Feature Attribution**: Bar chart showing which factors contribute most
   - **Contributing Factors**: Detailed list of elevated risk factors
   - **Recommendations**: Personalized support resources

5. **What-If Simulator:**
   - Adjust sliders for PHQ-9, GAD-7, Sleep, Heart Rate
   - Watch risk score update in real-time
   - See delta (▲/▼) from baseline score
   - Try reducing PHQ-9 to 5 → score drops significantly
   - Try reducing sleep to 3 hours → score increases

6. **Risk Trajectory:**
   - After first assessment, you'll see "Run the assessment again later to see trajectory"
   - Submit another assessment with same ID
   - Trajectory chart appears showing trend over time
   - Color-coded dots for each assessment
   - Trend arrow showing direction (▲ up, ▼ down, ◆ stable)

---

### 3. Batch Analytics Demo

**Step-by-step:**

1. **Navigate to "Batch Analytics" section**

2. **Load Sample Batch:**
   - Click "Load Sample Batch" button
   - JSON appears with 3 demo patients:
     - demo_001: High risk (PHQ-9: 18, Sleep: 4.5h)
     - demo_002: Low risk (PHQ-9: 8, Sleep: 7h)
     - demo_003: Critical risk (PHQ-9: 22, Sleep: 3.5h)

3. **Execute Batch:**
   - Click "Execute Batch Analysis"
   - Processing happens instantly (client-side)

4. **Review Batch Results:**
   
   **Summary Section:**
   - Total/Successful/Failed counts
   - Risk Distribution Bar Chart:
     - Color-coded segments (green/orange/red/dark red)
     - Shows proportion of each risk level
     - Hover to see exact counts

   **Individual Results:**
   - Each patient card shows:
     - Anonymized ID
     - Risk level badge
     - Risk score
     - Alert flag (🚨) if critical
     - Review flag (👤) if needs review

5. **Export Results:**
   - **CSV Export**: Click "📊 Export Results as CSV"
     - Downloads: `batch_results_YYYY-MM-DD.csv`
     - Includes: ID, Score, Level, Alerts, Contributing Factors
     - Ready for Excel/Google Sheets
   
   - **JSON Export**: Click "📄 Export Results as JSON"
     - Downloads: `batch_results_YYYY-MM-DD.json`
     - Full structured data
     - Ready for further analysis/processing

6. **Try Custom Batch:**
   ```json
   [
     {
       "anonymized_id": "patient_custom_001",
       "consent_verified": true,
       "survey_data": {
         "phq9_score": 20,
         "gad7_score": 17
       },
       "wearable_data": {
         "sleep_hours": 4.0,
         "avg_heart_rate": 88
       }
     },
     {
       "anonymized_id": "patient_custom_002",
       "consent_verified": true,
       "survey_data": {
         "phq9_score": 3,
         "gad7_score": 2
       },
       "wearable_data": {
         "sleep_hours": 8.5,
         "avg_heart_rate": 62
       }
     }
   ]
   ```
   - Paste this into the batch data textarea
   - Click "Execute Batch Analysis"
   - Observe different risk distributions

---

### 4. Review Queue Demo

**Step-by-step:**

1. **Navigate to "Review Queue" section**

2. **Load Demo Queue:**
   - Click "Refresh Queue"
   - Two demo review cases appear:
     - rev_101: patient_001 (High risk, 78.5 score)
     - rev_102: patient_003 (Moderate risk, 58.2 score)

3. **Select a Review:**
   - Click on any review card
   - Review detail panel expands below
   - Shows:
     - Review ID
     - Patient ID
     - Risk Level badge
     - Risk Score
     - Current Status
     - Existing notes

4. **Review Actions:**
   - **Assign Reviewer**: Enter email (e.g., reviewer@example.org)
   - **Add Note**: Enter clinical observations
   - **Close Review**: Mark as completed

5. **Filter Reviews:**
   - Use status dropdown: Pending/Approved/Escalated/Closed
   - Click "Refresh Queue" to reload with filter

---

### 5. Theme Toggle

**Try both themes:**
- Click theme toggle button (top right on desktop, menu on mobile)
- Dark Mode: Deep blue background, easy on eyes
- Light Mode: Clean white background, high contrast
- Preference persists in browser storage

---

### 6. Mobile Experience

**Test mobile view:**
1. Resize browser to mobile width (< 768px)
2. Mobile nav bar appears at top
3. Click hamburger menu to expand sidebar
4. Tap any section to navigate
5. Menu auto-closes after selection

---

## 🧪 Testing Scenarios

### Scenario 1: Low Risk Patient
```
PHQ-9: 2
GAD-7: 3
Sleep: 8 hours
Heart Rate: 65 bpm
Expected: Risk Score ~15-20, Level: Low
```

### Scenario 2: Moderate Risk Patient
```
PHQ-9: 11
GAD-7: 9
Sleep: 6 hours
Heart Rate: 75 bpm
Expected: Risk Score ~35-45, Level: Moderate
```

### Scenario 3: High Risk Patient
```
PHQ-9: 18
GAD-7: 14
Sleep: 5 hours
Heart Rate: 82 bpm
Expected: Risk Score ~58-68, Level: High
```

### Scenario 4: Critical Risk Patient
```
PHQ-9: 24
GAD-7: 19
Sleep: 3.5 hours
Heart Rate: 95 bpm
Expected: Risk Score ~80-90, Level: Critical
Alert triggered, Review required
```

---

## 🎨 Visual Features to Notice

### Color Coding
- **Green**: Low risk, safe
- **Orange**: Moderate risk, attention needed
- **Red**: High risk, intervention recommended
- **Dark Red**: Critical risk, immediate attention

### Animations
- Smooth scrolling to results
- Gauge needle animation
- Progress indicators
- Hover effects on cards

### Responsive Elements
- Cards stack on mobile
- Tables become scrollable
- Buttons resize appropriately
- Text scales for readability

---

## 📊 Data Quality Notes

### Mock Data Characteristics:
- **PHQ-9**: Distributed across 0-27 range, skewed toward lower scores (realistic)
- **GAD-7**: Correlated with PHQ-9, distributed 0-21
- **Sleep**: 4-10 hours, inverse correlation with depression
- **Heart Rate**: 55-90 bpm, slight correlation with anxiety
- **Risk Distribution**: ~60% low, ~25% moderate, ~10% high, ~5% critical (realistic clinical distribution)

### Why Mock Data?
- Demonstrates full functionality without backend setup
- Perfect for demos, testing, and UI development
- Allows exploration of all features immediately
- Can be easily replaced with real API calls

---

## 🔍 Key Technical Features

### Client-Side Scoring Engine
- Mirrors backend ClinicalRulesModel exactly
- Weighted feature scoring (PHQ-9: 30%, GAD-7: 22%, etc.)
- Sigmoid transformation for calibrated probabilities
- Non-linear score normalization curves

### Performance
- Individual assessment: < 50ms
- Batch 100 records: < 200ms
- Statistical generation: < 500ms
- All processing happens in browser

### Browser Compatibility
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- Mobile browsers: Full support

---

## 🚨 Crisis Resources (Always Visible)

The platform displays prominent crisis support information:
- **988 Suicide & Crisis Lifeline**: Call/Text 988
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: Call 911

---

## 💡 Tips for Best Experience

1. **Start with Individual Assessment** to understand the risk model
2. **Use What-If Simulator** to explore how factors affect risk
3. **Try Batch Analytics** to see distribution visualization
4. **Refresh Statistics** to see how datasets vary
5. **Toggle Theme** to find your preferred viewing mode
6. **Test on Mobile** to see responsive design
7. **Export Batch Results** to see data format

---

## 🎯 Next Steps

### For Developers:
- Review `app.js` clientScore() function for risk model logic
- Check generateMockStatisticalData() for data generation
- Explore renderForcePlot() for feature attribution viz

### For Users:
- Try different patient scenarios
- Export batch results and analyze in Excel
- Use simulator to understand risk factors
- Share with clinical team for feedback

### For Production:
- Connect to backend API by configuring API_BASE_URL
- Set up Firebase authentication
- Deploy to hosting platform
- Enable real data persistence

---

## 📞 Questions?

See README.md for full documentation including:
- Architecture details
- API endpoints
- Risk model specifications
- Security features
- Deployment guide
