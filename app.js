// API Configuration - uses relative path for Vercel serverless or localhost for development
const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : '/api';
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

    // Check system status on load
    checkSystemStatus();
});

// Check system status
async function checkSystemStatus() {
    updateStatusElement('api-status', 'checking', 'Checking...');
    updateStatusElement('models-status', 'checking', 'Checking...');
    updateStatusElement('queue-status', 'checking', 'Checking...');

    try {
        // Check health endpoint
        const healthResponse = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: getHeaders()
        });

        if (healthResponse.ok) {
            const health = await healthResponse.json();
            updateStatusElement('api-status', 'healthy', 'Healthy');

            // Show demo banner if in demo mode
            if (health.mode === 'demo') {
                document.getElementById('demo-banner').style.display = 'block';
            }
        } else {
            updateStatusElement('api-status', 'error', 'Error');
        }
    } catch (error) {
        updateStatusElement('api-status', 'error', 'Offline');
    }

    try {
        // Check statistics endpoint
        const statsResponse = await fetch(`${API_BASE_URL}/statistics`, {
            method: 'GET',
            headers: getHeaders()
        });

        if (statsResponse.ok) {
            const stats = await statsResponse.json();

            if (stats.models) {
                updateStatusElement('models-status', 'healthy',
                    `${stats.models.active_count} active / ${stats.models.total_count} total`);
            } else {
                updateStatusElement('models-status', 'warning', 'No models');
            }

            if (stats.review_queue) {
                updateStatusElement('queue-status', 'healthy',
                    `${stats.review_queue.pending_count} pending`);
            } else {
                updateStatusElement('queue-status', 'warning', 'No data');
            }
        } else {
            updateStatusElement('models-status', 'warning', 'Unable to fetch');
            updateStatusElement('queue-status', 'warning', 'Unable to fetch');
        }
    } catch (error) {
        updateStatusElement('models-status', 'error', 'Offline');
        updateStatusElement('queue-status', 'error', 'Offline');
    }
}

function updateStatusElement(id, status, text) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
        element.className = `status-value status-${status}`;
    }
}

// Get headers for API requests
function getHeaders() {
    const headers = {
        'Content-Type': 'application/json'
    };

    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }

    return headers;
}

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
        const response = await fetch(`${API_BASE_URL}/screen`, {
            method: 'POST',
            headers: getHeaders(),
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

// Submit batch screening
async function submitBatchScreening() {
    const batchDataInput = document.getElementById('batch-data').value.trim();

    if (!batchDataInput) {
        showError('Please enter batch data');
        return;
    }

    let requests;
    try {
        requests = JSON.parse(batchDataInput);
    } catch (e) {
        showError('Invalid JSON format for batch data');
        return;
    }

    if (!Array.isArray(requests)) {
        showError('Batch data must be a JSON array');
        return;
    }

    if (requests.length === 0) {
        showError('Batch data cannot be empty');
        return;
    }

    if (requests.length > 100) {
        showError('Maximum 100 requests per batch');
        return;
    }

    // Show loading
    showLoading(true);
    hideError();

    try {
        const payload = { requests: requests };

        const response = await fetch(`${API_BASE_URL}/batch-screen`, {
            method: 'POST',
            headers: getHeaders(),
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displayBatchResults(data);

    } catch (error) {
        console.error('Batch screening error:', error);
        showError(`Batch assessment failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display batch results
function displayBatchResults(data) {
    const batchResults = document.getElementById('batch-results');
    batchResults.style.display = 'block';

    let html = `
        <div class="batch-summary">
            <h4>Batch Results Summary</h4>
            <p><strong>Total:</strong> ${data.total} |
               <strong>Successful:</strong> ${data.successful} |
               <strong>Failed:</strong> ${data.failed}</p>
        </div>
        <div class="batch-items">
    `;

    data.results.forEach((result, index) => {
        const riskScore = result.risk_score;
        const riskClass = riskScore.risk_level ? riskScore.risk_level.toLowerCase() : 'unknown';

        html += `
            <div class="batch-item">
                <div class="batch-item-header">
                    <strong>${riskScore.anonymized_id}</strong>
                    <span class="risk-badge ${riskClass}">${riskScore.risk_level || 'UNKNOWN'}</span>
                    <span>Score: ${riskScore.score.toFixed(1)}/100</span>
                    <span>Confidence: ${(riskScore.confidence * 100).toFixed(0)}%</span>
                </div>
        `;

        if (result.alert_triggered) {
            html += `<div class="alert-inline danger">Alert Triggered</div>`;
        }
        if (result.requires_human_review) {
            html += `<div class="alert-inline warning">Human Review Required</div>`;
        }

        html += `</div>`;
    });

    html += '</div>';
    batchResults.innerHTML = html;
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
            li.innerHTML = `<strong>${feature}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}`;
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
    errorDisplay.className = 'error-message';
    errorDisplay.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('error-display').style.display = 'none';
}

function hideResults() {
    document.getElementById('results-section').style.display = 'none';
}

function showSuccess(message) {
    const errorDisplay = document.getElementById('error-display');
    errorDisplay.className = 'success-message';
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';

    setTimeout(() => {
        errorDisplay.style.display = 'none';
        errorDisplay.className = '';
    }, 3000);
}

// Load sample batch data for demo
function loadSampleBatchData() {
    const sampleData = [
        {
            anonymized_id: "demo_patient_001",
            consent_verified: true,
            survey_data: { phq9_score: 18, gad7_score: 14 },
            wearable_data: { sleep_hours: 4.5, avg_heart_rate: 82 }
        },
        {
            anonymized_id: "demo_patient_002",
            consent_verified: true,
            survey_data: { phq9_score: 8, gad7_score: 6 },
            wearable_data: { sleep_hours: 7, avg_heart_rate: 68 }
        },
        {
            anonymized_id: "demo_patient_003",
            consent_verified: true,
            survey_data: { phq9_score: 22, gad7_score: 18 },
            wearable_data: { sleep_hours: 3.5, avg_heart_rate: 95 }
        },
        {
            anonymized_id: "demo_patient_004",
            consent_verified: true,
            survey_data: { phq9_score: 5, gad7_score: 4 },
            wearable_data: { sleep_hours: 8, avg_heart_rate: 62 }
        },
        {
            anonymized_id: "demo_patient_005",
            consent_verified: true,
            survey_data: { phq9_score: 12, gad7_score: 10 },
            wearable_data: { sleep_hours: 5.5, avg_heart_rate: 75 }
        }
    ];

    document.getElementById('batch-data').value = JSON.stringify(sampleData, null, 2);
    showSuccess('Sample data loaded! Click "Run Batch Assessment" to test.');
}

// Load sample single screening data
function loadSampleScreeningData() {
    document.getElementById('anonymized-id').value = 'demo_patient_sample';
    document.getElementById('consent-verified').checked = true;
    document.getElementById('phq9-score').value = '15';
    document.getElementById('gad7-score').value = '12';
    document.getElementById('sleep-hours').value = '5.5';
    document.getElementById('avg-heart-rate').value = '78';
    showSuccess('Sample data loaded! Click "Run Risk Assessment" to test.');
}
