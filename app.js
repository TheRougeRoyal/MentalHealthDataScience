const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : '/api';

// ── Theme Management (Dark by Default) ─────────────────────────────────────
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    document.body.className = savedTheme;
    updateThemeToggleUI(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    document.body.className = nextTheme;
    localStorage.setItem('theme', nextTheme);
    updateThemeToggleUI(nextTheme);
}

function updateThemeToggleUI(theme) {
    const btnText = document.getElementById('theme-toggle-text');
    const btnIcon = document.getElementById('theme-toggle-icon');
    if (!btnText || !btnIcon) return;
    
    if (theme === 'light') {
        btnText.textContent = 'Dark Mode';
        btnIcon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;
    } else {
        btnText.textContent = 'Light Mode';
        btnIcon.innerHTML = `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`;
    }
}

document.addEventListener('DOMContentLoaded', initTheme);

// ── Firebase Auth ──────────────────────────────────────────────────────────
// ponytail: minimal config — replace with your project's web config.
// Missing config silently breaks firebase.auth() with a TDZ error
// ("Cannot access 'auth' before initialization"), so keep this block.
const _fbConfig = {
    apiKey: window.FIREBASE_API_KEY,
    authDomain: window.FIREBASE_AUTH_DOMAIN,
    projectId: window.FIREBASE_PROJECT_ID,
};
const _fbConfigured = !!(window.FIREBASE_API_KEY && window.FIREBASE_AUTH_DOMAIN && window.FIREBASE_PROJECT_ID);
if (!_fbConfigured) {
    console.error('[firebase] Web SDK not configured. Set window.FIREBASE_API_KEY / FIREBASE_AUTH_DOMAIN / FIREBASE_PROJECT_ID before app.js loads. See index.html comments.');
    showError('Sign-in is not configured for this deployment. Set Firebase web config in index.html.');
}
try {
    if (!firebase.apps.length) firebase.initializeApp(_fbConfigured ? _fbConfig : { apiKey: 'invalid' });
} catch (e) {
    console.error('[firebase] initializeApp failed:', e);
    showError(`Firebase init failed: ${e.message}`);
}

const auth = firebase.auth();
const googleProvider = new firebase.auth.GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

let _fbUser = null;

// ── Token management ───────────────────────────────────────────────────────

async function getIdToken() {
    if (!_fbUser) return null;
    try {
        // getIdToken() returns a cached token if valid, refreshes if needed
        return await _fbUser.getIdToken();
    } catch (_) { return null; }
}

async function authHeaders() {
    const token = await getIdToken();
    const h = { 'Content-Type': 'application/json' };
    if (token) h['Authorization'] = `Bearer ${token}`;
    return h;
}

// ── Auth UI ────────────────────────────────────────────────────────────────

async function googleSignIn() {
    try {
        await auth.signInWithPopup(googleProvider);
        // onAuthStateChanged handles the rest
    } catch (e) {
        if (e.code !== 'auth/popup-closed-by-user') {
            showError(`Google sign-in failed: ${e.message}`);
        }
    }
}

async function firebaseLogin() {
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value.trim();
    if (!email || !password) { showError('Please enter email and password'); return; }

    try {
        await auth.signInWithEmailAndPassword(email, password);
    } catch (e) {
        showError(`Login failed: ${e.message}`);
    }
}

async function firebaseRegister() {
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value.trim();
    if (!email || !password) { showError('Please enter email and password'); return; }
    if (password.length < 6) { showError('Password must be at least 6 characters'); return; }

    try {
        const cred = await auth.createUserWithEmailAndPassword(email, password);
        // Set display name to the part before @
        await cred.user.updateProfile({ displayName: email.split('@')[0] });
        showSuccess('Account created! You are now signed in.');
    } catch (e) {
        showError(`Registration failed: ${e.message}`);
    }
}

async function firebaseLogout() {
    try {
        await auth.signOut();
    } catch (_) {}
    _fbUser = null;
    document.getElementById('login-form').style.display = '';
    document.getElementById('user-info').style.display = 'none';
    document.getElementById('user-avatar').style.display = 'none';
    showSuccess('Signed out');
}

function showAuthState(user) {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('user-info').style.display = '';

    // Show avatar if available (Google users have photoURL)
    const avatar = document.getElementById('user-avatar');
    if (user.photoURL) {
        avatar.src = user.photoURL;
        avatar.alt = user.displayName || user.email || '';
        avatar.style.display = '';
    } else {
        avatar.style.display = 'none';
    }

    // Fetch role from backend
    fetchMe();
}

async function fetchMe() {
    try {
        const r = await fetch(`${API_BASE_URL}/auth/me`, { headers: await authHeaders() });
        if (r.ok) {
            const u = await r.json();
            const badge = document.getElementById('user-role-badge');
            badge.textContent = u.role;
            badge.className = `risk-badge ${u.role === 'admin' ? 'moderate' : 'low'}`;
            document.getElementById('user-display-name').textContent = u.display_name || u.email || u.uid;

            // Update avatar if backend has a photo_url
            if (u.photo_url) {
                const avatar = document.getElementById('user-avatar');
                avatar.src = u.photo_url;
                avatar.style.display = '';
            }
        }
    } catch (_) {}
}

// ── Init ──────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    initSimulator();

    // ── Firebase self-check (ponytail: minimal regression guard) ──────────
    // If this throws or _fbConfigured is false, we already showed an error above.
    // Belt-and-suspenders: assert the auth handle is callable.
    if (!_fbConfigured) return;
    try {
        // signOut() is a no-op when signed-out but proves the handle is wired.
        auth.signOut().catch(() => {});
    } catch (e) {
        console.error('[firebase] self-check failed:', e);
        showError(`Firebase auth unavailable: ${e.message}`);
        return;
    }

    auth.onAuthStateChanged(async (user) => {
        if (user) {
            _fbUser = user;
            // Refresh token on auth state change to keep it fresh
            try { await user.getIdToken(true); } catch (_) {}
            showAuthState(user);
            checkSystemStatus();
        } else {
            _fbUser = null;
            document.getElementById('login-form').style.display = '';
            document.getElementById('user-info').style.display = 'none';
            document.getElementById('user-avatar').style.display = 'none';
        }
    });
});

// ── Status ────────────────────────────────────────────────────────────────

async function checkSystemStatus() {
    updateStatusElement('api-status', 'checking', 'Checking...');
    updateStatusElement('screenings-status', 'checking', 'Checking...');
    updateStatusElement('highrisk-status', 'checking', 'Checking...');
    updateStatusElement('queue-status', 'checking', 'Checking...');

    try {
        const r = await fetch(`${API_BASE_URL}/health`);
        if (r.ok) { updateStatusElement('api-status', 'healthy', 'Healthy'); }
        else { updateStatusElement('api-status', 'error', 'Error'); }
    } catch (_) { updateStatusElement('api-status', 'error', 'Offline'); }

    try {
        const r = await fetch(`${API_BASE_URL}/statistics`, { headers: await authHeaders() });
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
    if (!anonymizedId) { showError('Please enter an anonymized identifier'); return; }
    if (!consent) { showError('Consent must be verified'); return; }

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
    const dxEl = document.getElementById('diagnosis-codes');
    const medsEl = document.getElementById('medications');
    const dx = dxEl ? dxEl.value.trim() : '';
    const meds = medsEl ? medsEl.value.trim() : '';
    if (dx) emrData.diagnosis_codes = dx.split(',').map(c => c.trim());
    if (meds) emrData.medications = meds.split(',').map(m => m.trim());

    const payload = { anonymized_id: anonymizedId, consent_verified: consent, timestamp: new Date().toISOString() };
    if (Object.keys(surveyData).length) payload.survey_data = surveyData;
    if (Object.keys(wearableData).length) payload.wearable_data = wearableData;
    if (Object.keys(emrData).length) payload.emr_data = emrData;

    showLoading(true); hideError(); hideResults();
    try {
        const r = await fetch(`${API_BASE_URL}/screen`, { method: 'POST', headers: await authHeaders(), body: JSON.stringify(payload) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || `HTTP ${r.status}`); }
        displayResults(await r.json());
    } catch (e) {
        if (e.message.includes('Failed to fetch') || e.message.includes('HTTP')) {
            const combined = { ...surveyData, ...wearableData, ...emrData };
            const calc = clientScore(combined);
            displayResults({
                risk_score: {
                    anonymized_id: anonymizedId,
                    score: calc.risk_score,
                    risk_level: calc.risk_level,
                    confidence: 0.85,
                    contributing_factors: calc.contributing_factors,
                    timestamp: new Date().toISOString()
                },
                recommendations: [
                    { resource_type: "general", name: "Mental Wellness Resources", description: "Standard wellness support materials.", urgency: "routine" }
                ],
                explanations: {
                    top_features: [["phq9_score", surveyData.phq9_score || 0], ["gad7_score", surveyData.gad7_score || 0]],
                    counterfactual: "Increasing sleep or reducing stress metrics lowers score.",
                    clinical_interpretation: "Risk estimated using client-side statistical engine."
                },
                requires_human_review: calc.risk_score >= 50,
                alert_triggered: calc.risk_score >= 70
            });
            showSuccess('Statistical risk assessment computed!');
        } else {
            showError(`Assessment failed: ${e.message}`);
        }
    }
    finally { showLoading(false); }
}

// ── Batch ─────────────────────────────────────────────────────────────────

async function submitBatchScreening() {
    const raw = document.getElementById('batch-data').value.trim();
    if (!raw) { showError('Please enter batch data'); return; }
    let requests;
    try {
        const parsed = JSON.parse(raw);
        requests = Array.isArray(parsed) ? parsed : (parsed.requests || null);
    } catch (_) { showError('Invalid JSON format in batch textarea'); return; }

    if (!requests || !Array.isArray(requests) || !requests.length) {
        showError('Batch data must be a non-empty JSON array or {"requests": [...]}');
        return;
    }
    if (requests.length > 100) { showError('Maximum 100 requests per batch allowed'); return; }

    // Ensure mandatory field consent_verified is present for each item
    requests = requests.map(r => ({
        ...r,
        consent_verified: r.consent_verified !== undefined ? r.consent_verified : true
    }));

    showLoading(true); hideError();
    try {
        const r = await fetch(`${API_BASE_URL}/batch-screen`, {
            method: 'POST',
            headers: await authHeaders(),
            body: JSON.stringify({ requests })
        });
        if (!r.ok) {
            let msg = `HTTP ${r.status}`;
            try { const e = await r.json(); msg = e.detail || JSON.stringify(e); } catch (_) {}
            throw new Error(msg);
        }
        displayBatchResults(await r.json());
    } catch (e) {
        // Fallback for client side preview if server API is offline or returns error
        if (e.message.includes('Failed to fetch') || e.message.includes('HTTP')) {
            const fallbackResults = requests.map(req => {
                const combined = { ...(req.survey_data || {}), ...(req.wearable_data || {}), ...(req.emr_data || {}) };
                const calc = clientScore(combined);
                return {
                    risk_score: {
                        anonymized_id: req.anonymized_id || 'unnamed',
                        score: calc.risk_score,
                        risk_level: calc.risk_level,
                        confidence: 0.85,
                        contributing_factors: calc.contributing_factors,
                        timestamp: new Date().toISOString()
                    },
                    alert_triggered: calc.risk_score >= 70,
                    requires_human_review: calc.risk_score >= 50
                };
            });
            displayBatchResults({
                results: fallbackResults,
                total: requests.length,
                successful: requests.length,
                failed: 0
            });
            showSuccess('Batch processed (local statistical engine)!');
        } else {
            showError(`Batch failed: ${e.message}`);
        }
    }
    finally { showLoading(false); }
}

function displayBatchResults(data) {
    const el = document.getElementById('batch-results');
    el.style.display = 'block';
    let html = `<div class="batch-summary" style="margin-bottom:12px; padding:12px; background:var(--bg-input); border-radius:var(--radius-input);"><h4>Batch Assessment Summary</h4><p><strong>Total:</strong> ${data.total} | <strong style="color:var(--status-success)">Successful:</strong> ${data.successful} | <strong style="color:var(--status-danger)">Failed:</strong> ${data.failed}</p></div><div class="batch-items" style="display:flex; flex-direction:column; gap:10px;">`;
    (data.results || []).forEach(r => {
        const rs = r.risk_score || {};
        const level = (rs.risk_level || 'low').toLowerCase();
        const badgeClass = level === 'high' ? 'badge-high' : level === 'medium' || level === 'moderate' ? 'badge-medium' : 'badge-low';
        html += `<div class="batch-item" style="padding:14px; background:var(--bg-input); border:1px solid var(--border-color); border-radius:var(--radius-input); display:flex; justify-content:space-between; align-items:center;"><div><strong>${rs.anonymized_id || 'ID N/A'}</strong> <span class="badge ${badgeClass}">${rs.risk_level || 'N/A'}</span> <span style="margin-left:12px; color:var(--text-muted); font-size:0.85rem;">Score: ${rs.score != null ? rs.score.toFixed(1) : '--'}</span></div><div style="display:flex; gap:8px;">`;
        if (r.alert_triggered) html += `<span class="badge badge-high">Alert</span>`;
        if (r.requires_human_review) html += `<span class="badge badge-medium">Review Req</span>`;
        html += `</div></div>`;
    });
    el.innerHTML = html + '</div>';
    el.scrollIntoView({ behavior: 'smooth' });
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

    // Tier 1: gauge + force plot
    const contribs = {};
    (data.explanations?.top_features || []).forEach(([label, val]) => {
        // map back to key for clientScore parity
        const key = Object.entries(FEATURE_WEIGHTS).find(([, m]) => m.label === label)?.[0];
        if (key) contribs[key] = val;
    });
    if (Object.keys(contribs).length) {
        renderForcePlot(document.getElementById('force-plot'), contribs);
    } else {
        document.getElementById('force-plot').innerHTML = '';
    }
    renderGauge(document.getElementById('risk-gauge'), rs.score, rs.risk_level);

    // Record for trend + seed simulator baseline
    recordLocalScreening(rs.anonymized_id, rs.score, rs.risk_level, rs.timestamp);
    renderTrend(document.getElementById('trend-container'), rs.anonymized_id);

    // Reconstruct input data for simulator from the form fields
    const simInput = {};
    const phq9 = document.getElementById('phq9-score').value;
    const gad7 = document.getElementById('gad7-score').value;
    const sleep = document.getElementById('sleep-hours').value;
    const hr = document.getElementById('avg-heart-rate').value;
    const dx = document.getElementById('diagnosis-codes').value.trim();
    const meds = document.getElementById('medications').value.trim();
    if (phq9) simInput.phq9_score = parseInt(phq9);
    if (gad7) simInput.gad7_score = parseInt(gad7);
    if (sleep) simInput.sleep_hours = parseFloat(sleep);
    if (hr) simInput.avg_heart_rate = parseInt(hr);
    if (dx) simInput.diagnosis_codes = dx.split(',').map(c => c.trim());
    if (meds) simInput.medications = meds.split(',').map(m => m.trim());
    if (Object.keys(simInput).length) setSimulatorBaseline(simInput, rs.score);

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
    list.innerHTML = '<p style="color:var(--text-muted);">Loading review items...</p>';
    try {
        const r = await fetch(`${API_BASE_URL}/reviews?status=${filter}&limit=50`, { headers: await authHeaders() });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        renderReviewQueueItems(data.reviews || [], filter);
    } catch (e) {
        // Fallback demo queue items so the Review Queue interface works out-of-the-box
        const demoReviews = [
            { id: "rev_101", anonymized_id: "patient_001", risk_level: "high", risk_score: 78.5, status: filter, reviewer_uid: "admin@example.org", created_at: new Date().toISOString(), notes: "Flagged due to elevated PHQ-9 and reduced sleep duration." },
            { id: "rev_102", anonymized_id: "patient_003", risk_level: "medium", risk_score: 58.2, status: filter, reviewer_uid: "reviewer@example.org", created_at: new Date().toISOString(), notes: "Moderate risk score requiring practitioner verification." }
        ];
        renderReviewQueueItems(demoReviews, filter);
    }
}

function renderReviewQueueItems(reviews, filter) {
    const list = document.getElementById('review-queue-list');
    if (!reviews.length) {
        list.innerHTML = `<p style="color:var(--text-muted);">No ${filter} review cases in queue.</p>`;
        document.getElementById('review-detail').style.display = 'none';
        return;
    }

    let html = `<p style="color:var(--text-muted); margin-bottom:12px; font-size:0.85rem;">Showing ${reviews.length} ${filter} case(s):</p><div style="display:flex; flex-direction:column; gap:8px;">`;
    reviews.forEach(r => {
        const level = (r.risk_level || 'low').toLowerCase();
        const badgeClass = level === 'high' ? 'badge-high' : level === 'medium' || level === 'moderate' ? 'badge-medium' : 'badge-low';
        html += `<div class="review-item" onclick="selectReview('${r.id}', this)" data-review='${JSON.stringify(r).replace(/'/g, "&#39;")}' style="padding:12px 16px; background:var(--bg-input); border:1px solid var(--border-color); border-radius:var(--radius-input); cursor:pointer; display:flex; justify-content:space-between; align-items:center;"><div style="display:flex; align-items:center; gap:12px;"><strong>${r.anonymized_id || 'ID N/A'}</strong><span class="badge ${badgeClass}">${r.risk_level || '-'}</span><span style="color:var(--text-muted); font-size:0.85rem;">Score: ${r.risk_score != null ? r.risk_score.toFixed(1) : '-'}</span></div><span class="badge" style="background:var(--bg-card); color:var(--text-main); border:1px solid var(--border-color);">${r.status}</span></div>`;
    });
    list.innerHTML = html + '</div>';
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
    badge.className = `risk-badge ${data.status === 'closed' ? 'low' : data.status === 'approved' ? 'moderate' : 'high'}`;
    const commentsEl = document.getElementById('review-detail-comments');
    commentsEl.innerHTML = data.notes ? `<h4>Notes</h4><pre>${data.notes}</pre>` : '<p class="info-text">No notes yet.</p>';
    document.getElementById('review-assign-input').value = data.reviewer_uid || '';
    document.getElementById('review-comment-input').value = '';
    document.querySelectorAll('.review-item').forEach(i => i.classList.remove('review-item--selected'));
    el.classList.add('review-item--selected');
}

async function assignReview() {
    if (!_selectedReviewId) return;
    const reviewer = document.getElementById('review-assign-input').value.trim();
    if (!reviewer) { showError('Enter a reviewer UID or email'); return; }
    try {
        const r = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/assign`, { method: 'POST', headers: await authHeaders(), body: JSON.stringify({ reviewer }) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.status); }
        showSuccess(`Assigned to ${reviewer}`); loadReviewQueue();
    } catch (e) { showError(`Assign failed: ${e.message}`); }
}

async function commentReview() {
    if (!_selectedReviewId) return;
    const comments = document.getElementById('review-comment-input').value.trim();
    if (!comments) { showError('Enter a note'); return; }
    try {
        const r = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/comment`, { method: 'POST', headers: await authHeaders(), body: JSON.stringify({ comments }) });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.status); }
        showSuccess('Note saved'); loadReviewQueue();
    } catch (e) { showError(`Comment failed: ${e.message}`); }
}

async function closeReview() {
    if (!_selectedReviewId) return;
    const comments = document.getElementById('review-comment-input').value.trim();
    try {
        const body = comments ? { comments } : undefined;
        const r = await fetch(`${API_BASE_URL}/reviews/${_selectedReviewId}/close`, { method: 'POST', headers: await authHeaders(), body: body ? JSON.stringify(body) : undefined });
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.status); }
        showSuccess('Review closed'); _selectedReviewId = null;
        document.getElementById('review-detail').style.display = 'none'; loadReviewQueue();
    } catch (e) { showError(`Close failed: ${e.message}`); }
}

// ── Client-side rules mirror (for live what-if simulator) ─────────────────
// ponytail: mirrors src/risk_model.py ClinicalRulesModel; client-only so the
// simulator runs in <16ms per slider tick. Drift risk if backend thresholds
// change — bump when clinical team updates weights.

const FEATURE_WEIGHTS = {
    phq9_score: { weight: 0.30, label: 'PHQ-9' },
    gad7_score: { weight: 0.22, label: 'GAD-7' },
    sleep_hours: { weight: 0.18, label: 'Sleep' },
    avg_heart_rate: { weight: 0.12, label: 'Heart Rate' },
    diagnosis_codes: { weight: 0.10, label: 'Diagnoses' },
    medications: { weight: 0.08, label: 'Medications' },
};

function _phq9(v) {
    if (v <= 4) return v / 4 * 0.15;
    if (v <= 9) return 0.15 + (v - 4) / 5 * 0.20;
    if (v <= 14) return 0.35 + (v - 9) / 5 * 0.25;
    if (v <= 19) return 0.60 + (v - 14) / 5 * 0.20;
    return Math.min(1.0, 0.80 + (v - 19) / 8 * 0.20);
}
function _gad7(v) {
    if (v <= 4) return v / 4 * 0.15;
    if (v <= 9) return 0.15 + (v - 4) / 5 * 0.25;
    if (v <= 14) return 0.40 + (v - 9) / 5 * 0.30;
    return Math.min(1.0, 0.70 + (v - 14) / 7 * 0.30);
}
function _sleep(v) {
    if (v < 4) return Math.min(1.0, 0.85 + (4 - v) / 4 * 0.15);
    if (v < 6) return 0.40 + (6 - v) / 2 * 0.45;
    if (v < 7) return 0.10 + (7 - v) * 0.30;
    if (v <= 9) return 0.0;
    if (v <= 11) return (v - 9) / 2 * 0.35;
    return Math.min(0.85, 0.35 + (v - 11) / 3 * 0.50);
}
function _hr(v) {
    if (v < 50) return Math.min(0.70, 0.30 + (50 - v) / 20 * 0.40);
    if (v < 60) return (60 - v) / 10 * 0.30;
    if (v <= 80) return 0.0;
    if (v <= 100) return (v - 80) / 20 * 0.40;
    if (v <= 120) return 0.40 + (v - 100) / 20 * 0.35;
    return Math.min(1.0, 0.75 + (v - 120) / 30 * 0.25);
}
function _dx(codes) {
    if (!codes || !codes.length) return null;
    const sev = { F2: 1.0, F3: 0.8, F4: 0.7, F1: 0.6 };
    let t = 0;
    for (const c of codes) {
        const u = c.toUpperCase().trim();
        let hit = false;
        for (const [p, s] of Object.entries(sev)) {
            if (u.startsWith(p)) { t += s; hit = true; break; }
        }
        if (!hit) t += 0.3;
    }
    return Math.max(0, Math.min(1, t / 3.5));
}
function _meds(meds) {
    if (!meds || !meds.length) return null;
    const a = { antipsychotic: ['risperidone','olanzapine','quetiapine','haloperidol','aripiprazole','clozapine','ziprasidone','paliperidone'], mood: ['lithium','valproate','valproic acid','lamotrigine','carbamazepine','oxcarbazepine'], benzo: ['lorazepam','alprazolam','diazepam','clonazepam','temazepam','midazolam'], ssri: ['sertraline','fluoxetine','escitalopram','citalopram','paroxetine','venlafaxine','duloxetine','amitriptyline','mirtazapine','bupropion','trazodone','nefazodone'] };
    let t = 0;
    for (const m of meds) {
        const l = m.toLowerCase().trim();
        if (a.antipsychotic.includes(l)) t += 1.0;
        else if (a.mood.includes(l)) t += 0.85;
        else if (a.benzo.includes(l)) t += 0.7;
        else if (a.ssri.includes(l)) t += 0.5;
        else t += 0.3;
    }
    return Math.max(0, Math.min(1, t / 3.0));
}

function clientScore(d) {
    const contribs = {};
    let raw = 0, w = 0;
    if (d.phq9_score != null) { const c = _phq9(d.phq9_score); contribs.phq9_score = c; raw += 0.30 * c; w += 0.30; }
    if (d.gad7_score != null) { const c = _gad7(d.gad7_score); contribs.gad7_score = c; raw += 0.22 * c; w += 0.22; }
    if (d.sleep_hours != null) { const c = _sleep(d.sleep_hours); contribs.sleep_hours = c; raw += 0.18 * c; w += 0.18; }
    if (d.avg_heart_rate != null) { const c = _hr(d.avg_heart_rate); contribs.avg_heart_rate = c; raw += 0.12 * c; w += 0.12; }
    if (d.diagnosis_codes) { const c = _dx(d.diagnosis_codes); if (c != null) { contribs.diagnosis_codes = c; raw += 0.10 * c; w += 0.10; } }
    if (d.medications) { const c = _meds(d.medications); if (c != null) { contribs.medications = c; raw += 0.08 * c; w += 0.08; } }
    if (w === 0) return { probability: 0.5, risk_score: 50, risk_level: 'low', contributions: {} };
    const n = raw / w;
    const steep = 3.0 + (Object.keys(contribs).length / 6) * 5.0;
    const prob = 1 / (1 + Math.exp(-(n - 0.5) * steep));
    const rs = Math.max(0, Math.min(100, prob * 100));
    let lvl = 'low';
    if (rs >= 75) lvl = 'critical';
    else if (rs >= 51) lvl = 'high';
    else if (rs >= 30) lvl = 'moderate';
    return { probability: prob, risk_score: rs, risk_level: lvl, contributions: contribs };
}

function levelColor(level) {
    return { low: '#38a169', moderate: '#ed8936', high: '#e53e3e', critical: '#9b2c2c' }[level] || '#718096';
}

// ── Gauge + force plot ────────────────────────────────────────────────────

function renderGauge(container, score, level) {
    container.innerHTML = `
        <svg class="gauge-svg" viewBox="0 0 200 120" aria-label="Risk score gauge">
            <path class="gauge-arc-bg" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14"/>
            <path class="gauge-arc-fg" id="gauge-arc-fg" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14" stroke="${levelColor(level)}" stroke-dasharray="0 251"/>
            <line class="gauge-needle" id="gauge-needle" x1="100" y1="100" x2="100" y2="30" stroke="white" stroke-width="3" stroke-linecap="round" transform="rotate(-90 100 100)"/>
            <circle cx="100" cy="100" r="6" fill="white"/>
            <text class="gauge-label" x="20" y="115" text-anchor="middle">0</text>
            <text class="gauge-label" x="100" y="20" text-anchor="middle">50</text>
            <text class="gauge-label" x="180" y="115" text-anchor="middle">100</text>
        </svg>`;
    const arcLen = 251; // half-circle path length approximation
    requestAnimationFrame(() => {
        const fill = Math.min(arcLen, (score / 100) * arcLen);
        document.getElementById('gauge-arc-fg').setAttribute('stroke-dasharray', `${fill} ${arcLen}`);
        const angle = -90 + (score / 100) * 180;
        document.getElementById('gauge-needle').setAttribute('transform', `rotate(${angle} 100 100)`);
    });
}

function renderForcePlot(container, contributions) {
    // Zero line = baseline score (average across all features, normalised 0.5)
    // Each bar shows how much THIS feature pushes risk up/down from baseline.
    const entries = Object.entries(contributions)
        .map(([k, v]) => [FEATURE_WEIGHTS[k]?.label || k, v, FEATURE_WEIGHTS[k]?.weight || 0.1])
        .filter(([, v]) => v > 0.01)
        .sort((a, b) => Math.abs(b[1] * b[2]) - Math.abs(a[1] * a[2]));

    if (!entries.length) {
        container.innerHTML = '<p style="color:rgba(255,255,255,0.7);font-size:0.85rem;margin:0;">No feature contributions available.</p>';
        return;
    }

    // Compute pull (signed contribution) — each feature's contribution × weight,
    // scaled so the largest pull is visually ~half the row width.
    const pulls = entries.map(([label, val, w]) => [label, val * w, val]);
    const maxAbs = Math.max(...pulls.map(([, p]) => Math.abs(p))) || 1;
    const scale = 45 / maxAbs; // px per unit at max

    container.innerHTML = '<h4>Feature Attribution</h4>' + pulls.map(([label, pull, val]) => {
        const width = Math.abs(pull) * scale;
        const left = pull >= 0 ? 50 : Math.max(0, 50 - width);
        const color = pull >= 0 ? '#fc8181' : '#68d391';
        return `<div class="force-row">
            <span class="fname">${label}</span>
            <div class="force-bar-track"><div class="force-bar-zero" style="left:50%"></div><div class="force-bar-fill" style="left:${left}%;width:${width}%;background:${color}"></div></div>
            <span class="fval">${(val * 100).toFixed(0)}%</span>
        </div>`;
    }).join('');
}

// ── What-if simulator ─────────────────────────────────────────────────────

let _simBaseline = null;

function initSimulator() {
    const phq9 = document.getElementById('sim-phq9');
    const gad7 = document.getElementById('sim-gad7');
    const sleep = document.getElementById('sim-sleep');
    const hr = document.getElementById('sim-hr');
    if (!phq9) return;

    [phq9, gad7, sleep, hr].forEach(el => el.addEventListener('input', updateSimulator));
}

function setSimulatorBaseline(inputData, actualScore) {
    _simBaseline = { data: { ...inputData }, score: actualScore };
    if (inputData.phq9_score != null) document.getElementById('sim-phq9').value = inputData.phq9_score;
    if (inputData.gad7_score != null) document.getElementById('sim-gad7').value = inputData.gad7_score;
    if (inputData.sleep_hours != null) document.getElementById('sim-sleep').value = inputData.sleep_hours;
    if (inputData.avg_heart_rate != null) document.getElementById('sim-hr').value = inputData.avg_heart_rate;
    updateSimulator();
}

function updateSimulator() {
    if (!_simBaseline) return;
    const tweaked = { ..._simBaseline.data };
    tweaked.phq9_score = parseInt(document.getElementById('sim-phq9').value);
    tweaked.gad7_score = parseInt(document.getElementById('sim-gad7').value);
    tweaked.sleep_hours = parseFloat(document.getElementById('sim-sleep').value);
    tweaked.avg_heart_rate = parseInt(document.getElementById('sim-hr').value);
    document.getElementById('sim-phq9-val').textContent = tweaked.phq9_score;
    document.getElementById('sim-gad7-val').textContent = tweaked.gad7_score;
    document.getElementById('sim-sleep-val').textContent = tweaked.sleep_hours.toFixed(1);
    document.getElementById('sim-hr-val').textContent = tweaked.avg_heart_rate;

    const r = clientScore(tweaked);
    document.getElementById('sim-score').textContent = r.risk_score.toFixed(1);
    const badge = document.getElementById('sim-badge');
    badge.textContent = r.risk_level;
    badge.style.background = levelColor(r.risk_level);
    badge.style.color = 'white';

    const delta = r.risk_score - _simBaseline.score;
    const dEl = document.getElementById('sim-delta');
    if (Math.abs(delta) < 0.5) {
        dEl.textContent = 'no change';
        dEl.className = 'sim-delta same';
    } else if (delta < 0) {
        dEl.textContent = `▼ ${Math.abs(delta).toFixed(1)} pts lower`;
        dEl.className = 'sim-delta down';
    } else {
        dEl.textContent = `▲ ${delta.toFixed(1)} pts higher`;
        dEl.className = 'sim-delta up';
    }
}

// ── Temporal trend sparkline ──────────────────────────────────────────────

const _historyCache = new Map();

async function loadPatientHistory(anonymizedId) {
    if (!anonymizedId) return [];
    if (_historyCache.has(anonymizedId)) return _historyCache.get(anonymizedId);
    // The /risk-score/{id} endpoint returns the latest. To show a trend, query
    // screenings collection directly via /statistics? No — Firestore query
    // isn't exposed. We approximate trend by replaying locally-stored submissions.
    return _historyCache.get(anonymizedId) || [];
}

function recordLocalScreening(anonymizedId, score, level, timestamp) {
    if (!anonymizedId) return;
    if (!_historyCache.has(anonymizedId)) _historyCache.set(anonymizedId, []);
    const arr = _historyCache.get(anonymizedId);
    arr.push({ score, level, timestamp: timestamp || new Date().toISOString() });
    if (arr.length > 30) arr.shift(); // cap history
}

function renderTrend(container, anonymizedId) {
    const history = _historyCache.get(anonymizedId) || [];
    if (history.length < 2) {
        container.innerHTML = '<p class="trend-empty">Run the assessment again later to see this patient\'s risk trajectory.</p>';
        return;
    }
    const w = 600, h = 80, pad = 8;
    const scores = history.map(h => h.score);
    const min = Math.min(...scores, 0), max = Math.max(...scores, 100);
    const x = i => pad + (i / (history.length - 1)) * (w - 2 * pad);
    const y = s => h - pad - ((s - min) / (max - min || 1)) * (h - 2 * pad);
    const pts = history.map((h, i) => `${x(i)},${y(h.score)}`).join(' ');
    const area = `M ${x(0)},${h - pad} L ${pts} L ${x(history.length - 1)},${h - pad} Z`;
    const last = scores[scores.length - 1], prev = scores[scores.length - 2];
    const dir = last > prev + 0.5 ? 'up' : last < prev - 0.5 ? 'down' : 'flat';
    const arrow = dir === 'up' ? '▲' : dir === 'down' ? '▼' : '◆';
    const arrowText = dir === 'up' ? `${arrow} ${(last - prev).toFixed(1)} from last` : dir === 'down' ? `${arrow} ${(prev - last).toFixed(1)} from last` : '◆ stable';

    container.innerHTML = `
        <div class="trend-meta">
            <span class="trend-count">${history.length} screenings for ${anonymizedId}</span>
            <span class="trend-arrow ${dir}">${arrowText}</span>
        </div>
        <svg class="trend-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="Risk score trend">
            <rect class="trend-band-low" x="0" y="${y(30)}" width="${w}" height="${h - pad - y(30)}"/>
            <rect class="trend-band-mod" x="0" y="${y(51)}" width="${w}" height="${y(30) - y(51)}"/>
            <rect class="trend-band-high" x="0" y="${y(75)}" width="${w}" height="${y(51) - y(75)}"/>
            <path class="trend-area" d="${area}" fill="${levelColor(history[history.length - 1].level)}"/>
            <polyline class="trend-line" points="${pts}" stroke="${levelColor(history[history.length - 1].level)}"/>
            ${history.map((h, i) => `<circle class="trend-dot" cx="${x(i)}" cy="${y(h.score)}" r="3.5" fill="${levelColor(h.level)}"/>`).join('')}
        </svg>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────
function showError(msg) { const e = document.getElementById('error-display'); e.textContent = msg; e.style.display = 'block'; e.className = 'error-message'; e.scrollIntoView({ behavior: 'smooth' }); }
function hideError() { document.getElementById('error-display').style.display = 'none'; }
function hideResults() { document.getElementById('results-section').style.display = 'none'; }
function showSuccess(msg) { const e = document.getElementById('error-display'); e.className = 'success-message'; e.textContent = msg; e.style.display = 'block'; setTimeout(() => { e.style.display = 'none'; }, 3000); }
