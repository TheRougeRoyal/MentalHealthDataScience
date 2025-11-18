// API Configuration
const API_BASE_URL = 'http://localhost:8000';
let authToken = '';

// Save authentication token
function saveToken() {
    const token = document.getElementById('api-token').value.trim();
    if (!token) {
        showError('Please enter an API token');
        return;
    }
    authToken = token;
    localStorage.setItem('mhras_token', token);
    showSuccess('Token saved successfully');
}

// Load token from localStorage on page load
window.addEventListener('DOMContentLoaded', () => {
    const savedToken = localStorage.getItem('mhras_token');
    if (savedToken) {
        authToken = savedToken;
        document.getElementById('api-token').value = savedToken;
    }
});

// Submit screening request
async function submitScreening() {
    // Validate inputs
    const anonymizedId = document.getElementById('anonymized-id').value.trim();
    const consentVerified = document.getElementById('consent-verified').checked;
    
    if (!anonymizedId) {
        showError('Please enter an anonymized patient ID');
        return;
    }
    
    if (!consentVerified) {
        showError('Patient consent must be verified');
        return;
    }
    
    // Token is optional in development mode
    if (!authToken) {
        console.log('No API token provided - using development mode');
    }
    
    // Collect form data
    const surveyData = {};
    const phq9 = document.getElementById('phq9-score').value;
    const gad7 = document.getElementById('gad7-score').value;
    if (phq9) surveyData.phq9_score = parseInt(phq9);
    if (gad7) surveyData.gad7_score = parseInt(gad7);
    
    const wearableData = {};
    const heartRate = document.getElementById('avg-heart-rate').value;
    const sleepHours = document.getElementById('sleep-hours').value;
    if (heartRate) wearableData.avg_heart_rate = parseInt(heartRate);
    if (sleepHours) wearableData.sleep_hours = parseFloat(sleepHours);
    
    const emrData = {};
    const diagnosisCodes = document.getElementById('diagnosis-codes').value.trim();
    const medications = document.getElementById('medications').value.trim();
    if (diagnosisCodes) {
        emrData.diagnosis_codes = diagnosisCodes.split(',').map(c => c.trim());
    }
    if (medications) {
        emrData.medications = medications.split(',').map(m => m.trim());
    }
    
    // Build request payload
    const payload = {
        anonymized_id: anonymizedId,
        consent_verified: consentVerified,
        timestamp: new Date().toISOString()
    };
    
    if (Object.keys(surveyData).length > 0) payload.survey_data = surveyData;
    if (Object.keys(wearableData).length > 0) payload.wearable_data = wearableData;
    if (Object.keys(emrData).length > 0) payload.emr_data = emrData;
    
    // Show loading
    showLoading(true);
    hideError();
    hideResults();
    
    try {
        // Build headers - only add Authorization if token exists
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (authToken) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }
        
        const response = await fetch(`${API_BASE_URL}/screen`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Screening error:', error);
        showError(`Assessment failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Display risk score
    const riskScore = data.risk_score;
    document.getElementById('score-value').textContent = riskScore.score.toFixed(1);
    document.getElementById('confidence-value').textContent = (riskScore.confidence * 100).toFixed(1);
    
    const riskBadge = document.getElementById('risk-badge');
    riskBadge.textContent = riskScore.risk_level;
    riskBadge.className = `risk-badge ${riskScore.risk_level}`;
    
    // Display alerts
    const alertsDisplay = document.getElementById('alerts-display');
    alertsDisplay.innerHTML = '';
    
    if (data.alert_triggered) {
        alertsDisplay.innerHTML = `
            <div class="alert-box danger">
                <span>⚠️</span>
                <strong>Alert Triggered:</strong> Immediate attention recommended
            </div>
        `;
    }
    
    if (data.requires_human_review) {
        alertsDisplay.innerHTML += `
            <div class="alert-box warning">
                <span>👤</span>
                <strong>Human Review Required:</strong> Case flagged for clinical review
            </div>
        `;
    }
    
    // Display contributing factors
    const factorsList = document.getElementById('factors-list');
    factorsList.innerHTML = '';
    
    if (riskScore.contributing_factors && riskScore.contributing_factors.length > 0) {
        riskScore.contributing_factors.forEach(factor => {
            const li = document.createElement('li');
            li.textContent = factor;
            factorsList.appendChild(li);
        });
    } else {
        factorsList.innerHTML = '<li>No specific factors identified</li>';
    }
    
    // Display recommendations
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = '';
    
    if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach(rec => {
            const card = document.createElement('div');
            card.className = 'recommendation-card';
            card.innerHTML = `
                <h4>${rec.name}</h4>
                <span class="urgency ${rec.urgency}">${rec.urgency}</span>
                <p><strong>Type:</strong> ${rec.resource_type}</p>
                <p>${rec.description}</p>
                ${rec.contact_info ? `<p><strong>Contact:</strong> ${rec.contact_info}</p>` : ''}
            `;
            recommendationsList.appendChild(card);
        });
    } else {
        recommendationsList.innerHTML = '<p>No specific recommendations at this time</p>';
    }
    
    // Display explanations
    const explanationsContent = document.getElementById('explanations-content');
    explanationsContent.innerHTML = '';
    
    const explanations = data.explanations;
    
    // Top features
    if (explanations.top_features && explanations.top_features.length > 0) {
        const featuresDiv = document.createElement('div');
        featuresDiv.className = 'explanation-item';
        featuresDiv.innerHTML = '<h4>Top Contributing Features</h4>';
        const featuresList = document.createElement('ul');
        featuresList.className = 'feature-list';
        
        explanations.top_features.forEach(([feature, value]) => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${feature}:</strong> ${value.toFixed(3)}`;
            featuresList.appendChild(li);
        });
        
        featuresDiv.appendChild(featuresList);
        explanationsContent.appendChild(featuresDiv);
    }
    
    // Clinical interpretation
    if (explanations.clinical_interpretation) {
        const clinicalDiv = document.createElement('div');
        clinicalDiv.className = 'explanation-item';
        clinicalDiv.innerHTML = `
            <h4>Clinical Interpretation</h4>
            <p>${explanations.clinical_interpretation}</p>
        `;
        explanationsContent.appendChild(clinicalDiv);
    }
    
    // Counterfactual
    if (explanations.counterfactual) {
        const counterfactualDiv = document.createElement('div');
        counterfactualDiv.className = 'explanation-item';
        counterfactualDiv.innerHTML = `
            <h4>What-If Scenario</h4>
            <p>${explanations.counterfactual}</p>
        `;
        explanationsContent.appendChild(counterfactualDiv);
    }
}

// Utility functions
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorDisplay = document.getElementById('error-display');
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';
    errorDisplay.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('error-display').style.display = 'none';
}

function hideResults() {
    document.getElementById('results-section').style.display = 'none';
}

function showSuccess(message) {
    // Simple alert for now
    const errorDisplay = document.getElementById('error-display');
    errorDisplay.style.backgroundColor = '#d4edda';
    errorDisplay.style.color = '#155724';
    errorDisplay.style.borderLeftColor = '#28a745';
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';
    
    setTimeout(() => {
        errorDisplay.style.display = 'none';
        errorDisplay.style.backgroundColor = '';
        errorDisplay.style.color = '';
        errorDisplay.style.borderLeftColor = '';
    }, 3000);
}
