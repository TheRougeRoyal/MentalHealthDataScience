const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : '/api';

function getHeaders() {
    return { 'Content-Type': 'application/json' };
}

// ── Auth ──────────────────────────────────────────────────────────────────

async function loginUser() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value.trim();
    if (!username || !password) { showError('Please enter username and password'); return; }

    try {
        const res = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST', headers: getHeaders(), credentials: 'include',
            body: JSON.stringify({ username, password }),
        });
        if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Login failed'); }
        const data = await res.json();
        showAuthState(data.user_id, data.role, data.display_name);
        showSuccess(`Signed in as ${data.display_name}`);
        checkSystemStatus();
    } catch (e) { showError(`Login failed: ${e.message}`); }
}

async function logoutUser() {
    try { await fetch(`${API_BASE_URL}/auth/logout`, { method: 'POST', credentials: 'include' }); } catch (_) {}
    document.getElementById('login-form').style.display = '';
    document.getElementById('user-info').style.display = 'none';
    showSuccess('Signed out');
}

async function checkSession() {
    try {
        const res = await fetch(`${API_BASE_URL}/auth/me`, { credentials: 'include' });
        if (res.ok) { const u = await res.json(); showAuthState(u.user_id, u.role, u.display_name); }
    } catch (_) {}
}

function showAuthState(userId, role, displayName) {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('user-info').style.display = '';
    document.getElementById('user-display-name').textContent = displayName || userId;
    const badge = document.getElementById('user-role-badge');
    badge.textContent = role;
    badge.className = `risk-badge ${role === 'admin' ? 'moderate' : 'low'}`;
}

// ── Init ──────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => { checkSession(); checkSystemStatus(); });

// ── Status ────────────────────────────────────────────────────────────────

async function checkSystemStatus() {
    updateStatusElement('api-status', 'checking', 'Checking...');
    updateStatusElement('screenings-status', 'checking', 'Checking...');
    updateStatusElement('highrisk-status', 'checking', 'Checking...');
    updateStatusElement('queue-status', 'checking', 'Checking...');

    try {
        const r = await fetch(`${API_BASE_URL}/health`, { credentials: 'include', headers: getHeaders() });
        if (r.ok) { updateStatusElement('api-status', 'healthy', 'Healthy'); }
        else { updateStatusElement('api-status', 'error', 'Error'); }
    } catch (_) { updateStatusElement('api-status', 'error', 'Offline'); }

    try {
        const r = await fetch(`${API_BASE_URL}/statistics`, { credentials: 'include', headers: getHeaders() });
        if (r.ok) {
            const s = await r.json();
            const sc = s.screenings || {};
            const q = s.review_queue || {};
            updateStatusElement('screenings-status', 'healthy', `${sc.total || 0} total`);
            updateStatusElement('highrisk-status', (sc.high_risk_pct || 0) > 20 ? 'warning' : 'healthy',
                `${sc.high_risk_count || 0} (${(sc.high_risk_pct || 0).toFixed(1)}%)`);
            updateStatusElement('queue-status', (q.pending_count || 0) > 0 ? 'warning' : 'healthy',
                `${q.pending_count || 0} pending`);
        }
    } catch (_) {
        updateStatusElement('screenings-status', 'error', 'Offline');
        updateStatusElement('highrisk-status', 'error', 'Offline');
        updateStatusElement('queue-status', 'error', 'Offline');
    }
}

function updateStatusElement(id, status, text) {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; el.className = `status-value status-${status}`; }
}

// ── Screening ─────────────────────────────────────────────────────────────

async function submitScreening() {
    const anonymizedId = document.getElementById('anonymized-id').value.trim();
    const consent = document.getElementById('consent-verified').checked;
    if (!anonymizedId) { showError('Please enter an anonymized patient ID'); return; }
    if (!consent) { showError('Patient consent must be verified'); return; }

    const surveyData = {};
    const phq9 = document.getElementById('phq9-score').value;
    const gad7 = document.getElementById('gad7-score').value;
    if (phq9) surveyData.phq9_score = parseInt(phq9);
    if (gad7) surveyData.gad7_score = parseInt(gad7);

    const wearableData = {};
    const hr = document.getElementById('avg-heart-rate').value;
    const sleep = document.getElementById('sleep-hours').value;
    if (hr) wearableData.avg_heart_rate = parseInt(hr);
    if (sleep) wearableData.sleep_hours = parseFloat(sleep);

    const emrData = {};
    const dx = document.getElementById('diagnosis-codes').value.trim();
    const meds = document.getElementById('medications').value.trim();
    if (dx) emrData.diagnosis_codes = dx.split(',').map(c => c.trim());
    if (meds) emrData.medications = meds.split(',').map(m => m.trim());

    const payload = { anonymized_id: anonymizedId, consent_verified: consent, timestamp: new Date().toISOString() };
    if (Object.keys(surveyData).length) payload.survey_data = surveyData;
    if (Object.keys(wearableData).length) payload.wearable_data = wearableData;
    if (Object.keys(emrData).length) payload.emr_data = emrData;

    showLoading(true); hideError(); hideResults();
    try {
        const r = await fetch(`${API_BASE_URL}/screen`, { method: 'POST', credentials: 'include', headers: getHeaders(), body: JSON.stringify(payload) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || `HTTP ${r.status}`); }
        displayResults(await r.json());
    } catch (e) { showError(`Assessment failed: ${e.message}`); }
    finally { showLoading(false); }
}

// ── Batch ─────────────────────────────────────────────────────────────────

async function submitBatchScreening() {
    const raw = document.getElementById('batch-data').value.trim();
    if (!raw) { showError('Please enter batch data'); return; }
    let requests;
    try { requests = JSON.parse(raw); } catch (_) { showError('Invalid JSON'); return; }
    if (!Array.isArray(requests) || !requests.length) { showError('Must be a non-empty JSON array'); return; }
    if (requests.length > 100) { showError('Max 100 per batch'); return; }

    showLoading(true); hideError();
    try {
        const r = await fetch(`${API_BASE_URL}/batch-screen`, { method: 'POST', credentials: 'include', headers: getHeaders(), body: JSON.stringify({ requests }) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || `HTTP ${r.status}`); }
        displayBatchResults(await r.json());
    } catch (e) { showError(`Batch failed: ${e.message}`); }
    finally { showLoading(false); }
}

function displayBatchResults(data) {
    const el = document.getElementById('batch-results');
    el.style.display = 'block';
    let html = `<div class="batch-summary"><h4>Batch Results</h4><p><strong>Total:</strong> ${data.total} | <strong>OK:</strong> ${data.successful} | <strong>Failed:</strong> ${data.failed}</p></div><div class="batch-items">`;
    data.results.forEach(r => {
        const rs = r.risk_score;
        const cls = rs.risk_level ? rs.risk_level.toLowerCase() : 'unknown';
        html += `<div class="batch-item"><div class="batch-item-header"><strong>${rs.anonymized_id}</strong><span class="risk-badge ${cls}">${rs.risk_level || 'N/A'}</span><span>Score: ${rs.score.toFixed(1)}</span><span>Conf: ${(rs.confidence * 100).toFixed(0)}%</span></div>`;
        if (r.alert_triggered) html += `<div class="alert-inline danger">Alert Triggered</div>`;
        if (r.requires_human_review) html += `<div class="alert-inline warning">Review Required</div>`;
        html += `</div>`;
    });
    el.innerHTML = html + '</div>';
}

// ── Results ───────────────────────────────────────────────────────────────

function displayResults(data) {
    const sec = document.getElementById('results-section');
    sec.style.display = 'block';
    sec.scrollIntoView({ behavior: 'smooth' });

    const rs = data.risk_score;
    document.getElementById('score-value').textContent = rs.score.toFixed(1);
    document.getElementById('confidence-value').textContent = (rs.confidence * 100).toFixed(1);
    const badge = document.getElementById('risk-badge');
    badge.textContent = rs.risk_level;
    badge.className = `risk-badge ${rs.risk_level}`;

    const alerts = document.getElementById('alerts-display');
    alerts.innerHTML = '';
    if (data.alert_triggered) alerts.innerHTML = '<div class="alert-box danger"><span>⚠️</span><strong>Alert:</strong> Immediate attention recommended</div>';
    if (data.requires_human_review) alerts.innerHTML += '<div class="alert-box warning"><span>👤</span><strong>Human Review Required</strong></div>';

    const factors = document.getElementById('factors-list');
    factors.innerHTML = '';
    (rs.contributing_factors || []).forEach(f => { const li = document.createElement('li'); li.textContent = f; factors.appendChild(li); });
    if (!rs.contributing_factors?.length) factors.innerHTML = '<li>No specific factors identified</li>';

    const recs = document.getElementById('recommendations-list');
    recs.innerHTML = '';
    (data.recommendations || []).forEach(r => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.innerHTML = `<h4>${r.name}</h4><span class="urgency ${r.urgency}">${r.urgency}</span><p><strong>Type:</strong> ${r.resource_type}</p><p>${r.description}</p>${r.contact_info ? `<p><strong>Contact:</strong> ${r.contact_info}</p>` : ''}`;
        recs.appendChild(card);
    });
    if (!data.recommendations?.length) recs.innerHTML = '<p>No specific recommendations at this time</p>';

    const exp = document.getElementById('explanations-content');
    exp.innerHTML = '';
    const explanations = data.explanations || {};
    if (explanations.top_features?.length) {
        const div = document.createElement('div'); div.className = 'explanation-item';
        div.innerHTML = '<h4>Top Contributing Features</h4><ul class="feature-list">' +
            explanations.top_features.map(([f, v]) => `<li><strong>${f}:</strong> ${typeof v === 'number' ? v.toFixed(3) : v}</li>`).join('') + '</ul>';
        exp.appendChild(div);
    }
    if (explanations.clinical_interpretation) {
        const div = document.createElement('div'); div.className = 'explanation-item';
        div.innerHTML = `<h4>Clinical Interpretation</h4><p>${explanations.clinical_interpretation}</p>`;
        exp.appendChild(div);
    }
    if (explanations.counterfactual) {
        const div = document.createElement('div'); div.className = 'explanation-item';
        div.innerHTML = `<h4>What-If Scenario</h4><p>${explanations.counterfactual}</p>`;
        exp.appendChild(div);
    }
}

// ── Sample data ───────────────────────────────────────────────────────────

function loadSampleScreeningData() {
    document.getElementById('anonymized-id').value = 'demo_patient_sample';
    document.getElementById('consent-verified').checked = true;
    document.getElementById('phq9-score').value = '15';
    document.getElementById('gad7-score').value = '12';
    document.getElementById('sleep-hours').value = '5.5';
    document.getElementById('avg-heart-rate').value = '78';
    showSuccess('Sample data loaded!');
}

function loadSampleBatchData() {
    document.getElementById('batch-data').value = JSON.stringify([
        { anonymized_id: "demo_001", consent_verified: true, survey_data: { phq9_score: 18, gad7_score: 14 }, wearable_data: { sleep_hours: 4.5, avg_heart_rate: 82 } },
        { anonymized_id: "demo_002", consent_verified: true, survey_data: { phq9_score: 8, gad7_score: 6 }, wearable_data: { sleep_hours: 7, avg_heart_rate: 68 } },
        { anonymized_id: "demo_003", consent_verified: true, survey_data: { phq9_score: 22, gad7_score: 18 }, wearable_data: { sleep_hours: 3.5, avg_heart_rate: 95 } },
    ], null, 2);
    showSuccess('Sample batch data loaded!');
}

// ── Review Queue ──────────────────────────────────────────────────────────

let _selectedReviewId = null;

async function loadReviewQueue() {
    const filter = document.getElementById('review-status-filter').value;
    const list = document.getElementById('review-queue-list');
    list.innerHTML = '<p class="info-text">Loading...</p>';
    try {
        const r = await fetch(`${API_BASE_URL}/reviews/queue?status_filter=${filter}&limit=50`, { credentials: 'include', headers: getHeaders() });
        if (r.status === 401 || r.status === 403) { list.innerHTML = '<p class="info-text">Sign in as <strong>admin</strong> or <strong>reviewer</strong>.</p>'; return; }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (!data.reviews?.length) { list.innerHTML = `<p class="info-text">No ${filter} reviews.</p>`; document.getElementById('review-detail').style.display = 'none'; return; }

        let html = `<p class="info-text" style="margin-bottom:12px;">${data.reviews.length} ${filter}(s) shown</p>`;
        data.reviews.forEach(r => {
            const cls = (r.risk_level || 'low').toLowerCase();
            const sel = r.id === _selectedReviewId ? ' review-item--selected' : '';
            html += `<div class="review-item${sel}" onclick="selectReview('${r.id}', this)" data-review='${JSON.stringify(r).replace(/'/g, "&#39;")}'><div class="review-item-header"><strong>${r.anonymized_id || '?'}</strong><span class="risk-badge ${cls}">${r.risk_level || '-'}</span><span>Score: ${r.risk_score != null ? r.risk_score.toFixed(1) : '-'}</span><span class="review-item-status">${r.status}</span></div><div class="review-item-meta">${r.reviewer || 'Unassigned'} · ${new Date(r.created_at).toLocaleString()}</div></div>`;
        });
        list.innerHTML = html;
    } catch (e) { list.innerHTML = `<p class="error-message">Failed: ${e.message}</p>`; }
}

function selectReview(id, el) {
    _selectedReviewId = id;
    const data = JSON.parse(el.dataset.review);
    document.getElementById('review-detail').style.display = '';
    document.getElementById('review-detail-id').textContent = id.substring(0, 8) + '...';
    document.getElementById('review-detail-patient').textContent = data.anonymized_id || '-';
    document.getElementById('review-detail-risk').textContent = data.risk_level || '-';
    document.getElementById('review-detail-score').textContent = data.risk_score != null ? data.risk_score.toFixed(1) : '-';
    const badge = document.getElementById('review-detail-status');
    badge.textContent = data.status;
    badge.className = `risk-badge ${data.status === 'closed' ? 'low' : data.status === 'reviewed' ? 'moderate' : 'high'}`;
    const commentsEl = document.getElementById('review-detail-comments');
    commentsEl.innerHTML = data.comments ? `<h4>Comments</h4><pre>${data.comments}</pre>` : '<p class="info-text">No comments yet.</p>';
    document.getElementById('review-assign-input').value = data.reviewer || '';
    document.getElementById('review-comment-input').value = '';
    document.querySelectorAll('.review-item').forEach(i => i.classList.remove('review-item--selected'));
    el.classList.add('review-item--selected');
}

async function assignReview() {
    if (!_selectedReviewId) return;
    const reviewer = document.getElementById('review-assign-input').value.trim();
    if (!reviewer) { showError('Enter a reviewer username'); return; }
    try {
        const r = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/assign`, { method: 'POST', credentials: 'include', headers: getHeaders(), body: JSON.stringify({ reviewer }) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.status); }
        showSuccess(`Assigned to ${reviewer}`); loadReviewQueue();
    } catch (e) { showError(`Assign failed: ${e.message}`); }
}

async function commentReview() {
    if (!_selectedReviewId) return;
    const comments = document.getElementById('review-comment-input').value.trim();
    if (!comments) { showError('Enter a comment'); return; }
    try {
        const r = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/comment`, { method: 'POST', credentials: 'include', headers: getHeaders(), body: JSON.stringify({ comments }) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.status); }
        showSuccess('Comment saved'); loadReviewQueue();
    } catch (e) { showError(`Comment failed: ${e.message}`); }
}

async function closeReview() {
    if (!_selectedReviewId) return;
    const comments = document.getElementById('review-comment-input').value.trim();
    try {
        const body = comments ? { comments } : undefined;
        const r = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/close`, { method: 'POST', credentials: 'include', headers: getHeaders(), body: body ? JSON.stringify(body) : undefined });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.status); }
        showSuccess('Review closed'); _selectedReviewId = null;
        document.getElementById('review-detail').style.display = 'none'; loadReviewQueue();
    } catch (e) { showError(`Close failed: ${e.message}`); }
}

// ── Utilities ─────────────────────────────────────────────────────────────

function showLoading(show) { document.getElementById('loading').style.display = show ? 'block' : 'none'; }
function showError(msg) { const e = document.getElementById('error-display'); e.textContent = msg; e.style.display = 'block'; e.className = 'error-message'; e.scrollIntoView({ behavior: 'smooth' }); }
function hideError() { document.getElementById('error-display').style.display = 'none'; }
function hideResults() { document.getElementById('results-section').style.display = 'none'; }
function showSuccess(msg) { const e = document.getElementById('error-display'); e.className = 'success-message'; e.textContent = msg; e.style.display = 'block'; setTimeout(() => { e.style.display = 'none'; }, 3000); }
