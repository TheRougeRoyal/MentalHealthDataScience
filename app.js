// API Configuration - uses relative path for Vercel serverless or localhost for development
const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : '/api';

// ── Auth helpers (HTTP-only cookies — no JS token storage) ──────────────

async function loginUser() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value.trim();

    if (!username || !password) {
        showError('Please enter username and password');
        return;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ username, password }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Login failed');
        }

        const data = await res.json();
        showAuthState(data.user_id, data.role, data.display_name);
        showSuccess(`Signed in as ${data.display_name}`);
        checkSystemStatus();
    } catch (e) {
        showError(`Login failed: ${e.message}`);
    }
}

async function logoutUser() {
    try {
        await fetch(`${API_BASE_URL}/auth/logout`, {
            method: 'POST',
            credentials: 'include',
        });
    } catch (_) { /* ignore network errors on logout */ }

    document.getElementById('login-form').style.display = '';
    document.getElementById('user-info').style.display = 'none';
    showSuccess('Signed out');
}

async function checkSession() {
    try {
        const res = await fetch(`${API_BASE_URL}/auth/me`, {
            credentials: 'include',
        });
        if (res.ok) {
            const user = await res.json();
            showAuthState(user.user_id, user.role, user.display_name);
        }
    } catch (_) { /* not logged in */ }
}

function showAuthState(userId, role, displayName) {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('user-info').style.display = '';
    document.getElementById('user-display-name').textContent = displayName || userId;
    const badge = document.getElementById('user-role-badge');
    badge.textContent = role;
    badge.className = `risk-badge ${role === 'admin' ? 'moderate' : 'low'}`;
}

// ── Page load ───────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    checkSession();
    checkSystemStatus();
});

// ── System status ───────────────────────────────────────────────────────

async function checkSystemStatus() {
    updateStatusElement('api-status', 'checking', 'Checking...');
    updateStatusElement('models-status', 'checking', 'Checking...');
    updateStatusElement('queue-status', 'checking', 'Checking...');

    try {
        const healthResponse = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            credentials: 'include',
            headers: getHeaders(),
        });

        if (healthResponse.ok) {
            const health = await healthResponse.json();
            updateStatusElement('api-status', 'healthy', 'Healthy');

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
        const statsResponse = await fetch(`${API_BASE_URL}/statistics`, {
            method: 'GET',
            credentials: 'include',
            headers: getHeaders(),
        });

        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            const s = stats.screenings || {};
            const q = stats.review_queue || {};

            updateStatusElement('screenings-status', 'healthy',
                `${s.total || 0} total · avg ${(s.avg_risk_score || 0).toFixed(1)}`);

            const hrCount = s.high_risk_count || 0;
            const hrPct = s.high_risk_pct || 0;
            updateStatusElement('highrisk-status',
                hrPct > 20 ? 'warning' : 'healthy',
                `${hrCount} (${hrPct.toFixed(1)}%)`);

            updateStatusElement('queue-status',
                (q.pending_count || 0) > 0 ? 'warning' : 'healthy',
                `${q.pending_count || 0} pending`);
        } else {
            updateStatusElement('screenings-status', 'warning', 'Unable to fetch');
            updateStatusElement('highrisk-status', 'warning', 'Unable to fetch');
            updateStatusElement('queue-status', 'warning', 'Unable to fetch');
        }
    } catch (error) {
        updateStatusElement('screenings-status', 'error', 'Offline');
        updateStatusElement('highrisk-status', 'error', 'Offline');
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

// Get headers for API requests (no Authorization header — cookies handle auth)
function getHeaders() {
    return { 'Content-Type': 'application/json' };
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
            credentials: 'include',
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
            credentials: 'include',
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


// ── Clinical Review Queue ───────────────────────────────────────────────

let _selectedReviewId = null;

async function loadReviewQueue() {
    const filter = document.getElementById('review-status-filter').value;
    const list = document.getElementById('review-queue-list');
    list.innerHTML = '<p class="info-text">Loading…</p>';

    try {
        const res = await fetch(
            `${API_BASE_URL}/reviews/queue?status_filter=${filter}&limit=50`,
            { credentials: 'include', headers: getHeaders() },
        );

        if (res.status === 401 || res.status === 403) {
            list.innerHTML = '<p class="info-text">Sign in with an <strong>admin</strong> or <strong>reviewer</strong> account to view the queue.</p>';
            return;
        }

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();

        if (!data.reviews || data.reviews.length === 0) {
            list.innerHTML = `<p class="info-text">No ${filter} reviews.</p>`;
            document.getElementById('review-detail').style.display = 'none';
            return;
        }

        let html = `<p class="info-text" style="margin-bottom:12px;">Showing ${data.reviews.length} ${filter} review(s) · ${data.total_pending} total pending</p>`;

        data.reviews.forEach(r => {
            const riskClass = (r.risk_level || 'low').toLowerCase();
            const selected = r.id === _selectedReviewId ? ' review-item--selected' : '';
            html += `
                <div class="review-item${selected}" onclick="selectReview('${r.id}', this)" data-review='${JSON.stringify(r).replace(/'/g, "&#39;")}'>
                    <div class="review-item-header">
                        <strong>${r.anonymized_id || 'Unknown'}</strong>
                        <span class="risk-badge ${riskClass}">${r.risk_level || '—'}</span>
                        <span>Score: ${r.risk_score != null ? r.risk_score.toFixed(1) : '—'}</span>
                        <span class="review-item-status">${r.status}</span>
                    </div>
                    <div class="review-item-meta">
                        ${r.reviewer ? `Reviewer: ${r.reviewer}` : 'Unassigned'}
                        · ${new Date(r.created_at).toLocaleString()}
                    </div>
                </div>`;
        });

        list.innerHTML = html;

    } catch (e) {
        list.innerHTML = `<p class="error-message">Failed to load queue: ${e.message}</p>`;
    }
}

function selectReview(id, el) {
    _selectedReviewId = id;
    const data = JSON.parse(el.dataset.review);

    document.getElementById('review-detail').style.display = '';
    document.getElementById('review-detail-id').textContent = id.substring(0, 8) + '…';
    document.getElementById('review-detail-patient').textContent = data.anonymized_id || '—';
    document.getElementById('review-detail-risk').textContent = data.risk_level || '—';
    document.getElementById('review-detail-score').textContent = data.risk_score != null ? data.risk_score.toFixed(1) : '—';

    const badge = document.getElementById('review-detail-status');
    badge.textContent = data.status;
    badge.className = `risk-badge ${data.status === 'closed' ? 'low' : data.status === 'reviewed' ? 'moderate' : 'high'}`;

    const commentsEl = document.getElementById('review-detail-comments');
    if (data.comments) {
        commentsEl.innerHTML = `<h4>Comments</h4><pre>${data.comments}</pre>`;
    } else {
        commentsEl.innerHTML = '<p class="info-text">No comments yet.</p>';
    }

    document.getElementById('review-assign-input').value = data.reviewer || '';
    document.getElementById('review-comment-input').value = '';

    // Highlight selected row
    document.querySelectorAll('.review-item').forEach(item => item.classList.remove('review-item--selected'));
    el.classList.add('review-item--selected');

    document.getElementById('review-detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function assignReview() {
    if (!_selectedReviewId) return;
    const reviewer = document.getElementById('review-assign-input').value.trim();
    if (!reviewer) { showError('Enter a reviewer username'); return; }

    try {
        const res = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/assign`, {
            method: 'POST',
            credentials: 'include',
            headers: getHeaders(),
            body: JSON.stringify({ reviewer }),
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.status); }
        showSuccess(`Assigned to ${reviewer}`);
        loadReviewQueue();
    } catch (e) { showError(`Assign failed: ${e.message}`); }
}

async function commentReview() {
    if (!_selectedReviewId) return;
    const comments = document.getElementById('review-comment-input').value.trim();
    if (!comments) { showError('Enter a comment'); return; }

    try {
        const res = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/comment`, {
            method: 'POST',
            credentials: 'include',
            headers: getHeaders(),
            body: JSON.stringify({ comments }),
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.status); }
        showSuccess('Comment saved');
        loadReviewQueue();
    } catch (e) { showError(`Comment failed: ${e.message}`); }
}

async function closeReview() {
    if (!_selectedReviewId) return;
    const comments = document.getElementById('review-comment-input').value.trim();

    try {
        const body = comments ? { comments } : null;
        const res = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/close`, {
            method: 'POST',
            credentials: 'include',
            headers: getHeaders(),
            body: body ? JSON.stringify(body) : undefined,
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.status); }
        showSuccess('Review closed');
        _selectedReviewId = null;
        document.getElementById('review-detail').style.display = 'none';
        loadReviewQueue();
    } catch (e) { showError(`Close failed: ${e.message}`); }
}
