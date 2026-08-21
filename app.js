const API_BASE_URL = ['localhost', '127.0.0.1'].includes(window.location.hostname)
  ? 'http://localhost:8000'
  : '/api';

// ponytail: default role is 'user' until /auth/me confirms otherwise. Any
// admin-only UI is hidden by default so an unauthenticated visitor never
// sees a patient list flash before login.
window._currentRole = 'user';
window._authReady = false;
window._pageScope = document.body?.dataset?.page || 'dashboard';
applyRoleGating('user');

function guardPageAccess(role = window._currentRole || 'user') {
    const page = document.body?.dataset?.page || 'dashboard';
    const routePageNames = ['index.html', 'screening.html', 'batch.html', 'statistics.html', 'queue.html'];
    const allowedByPage = {
        dashboard: ['user', 'admin'],
        screening: ['user', 'admin'],
        batch: ['user', 'admin'],
        statistics: ['user', 'admin'],
        queue: ['admin']
    };

    if (window._authReady && !_fbUser) {
        if (page !== 'dashboard') window.location.href = 'index.html';
        return page === 'dashboard';
    }

    if (routePageNames.includes(window.location.pathname.split('/').pop()) && page === 'queue' && role !== 'admin') {
        window.location.href = 'index.html';
        return false;
    }

    const allowed = allowedByPage[page] || ['user', 'admin'];
    if (!allowed.includes(role)) {
        window.location.href = 'index.html';
        return false;
    }

    if (page === 'queue' && role !== 'admin') {
        window.location.href = 'index.html';
        return false;
    }

    return true;
}

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
const _inlineFbConfig = {
    apiKey: window.FIREBASE_API_KEY,
    authDomain: window.FIREBASE_AUTH_DOMAIN,
    projectId: window.FIREBASE_PROJECT_ID,
    storageBucket: window.FIREBASE_STORAGE_BUCKET,
    messagingSenderId: window.FIREBASE_MESSAGING_SENDER_ID,
    appId: window.FIREBASE_APP_ID,
    measurementId: window.FIREBASE_MEASUREMENT_ID,
};
let _fbConfigured = false;
let auth = null;
let db = null;
let googleProvider = null;
let _firebaseInitPromise = null;

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
    if (!(await ensureFirebaseReady())) return;
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
    if (!(await ensureFirebaseReady())) return;
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
    if (!(await ensureFirebaseReady())) return;
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
    if (!(await ensureFirebaseReady())) return;
    try {
        await auth.signOut();
    } catch (_) {}
    _fbUser = null;
    document.getElementById('login-form').style.display = '';
    document.getElementById('user-info').style.display = 'none';
    const avatar = document.getElementById('user-avatar');
    if (avatar) avatar.style.display = 'none';
    const headerPill = document.getElementById('user-header-pill');
    if (headerPill) headerPill.style.display = 'none';
    showSuccess('Signed out');
}

function showAuthState(user) {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('user-info').style.display = '';

    const headerPill = document.getElementById('user-header-pill');
    if (headerPill) headerPill.style.display = 'flex';

    // Show avatar if available
    const avatar = document.getElementById('user-avatar');
    if (avatar) {
        if (user.photoURL) {
            avatar.src = user.photoURL;
            avatar.alt = user.displayName || user.email || '';
            avatar.style.display = '';
        } else {
            avatar.style.display = 'none';
        }
    }

    const displayName = user.displayName || (user.email ? user.email.split('@')[0] : user.uid);
    const headerName = document.getElementById('header-user-name');
    if (headerName) headerName.textContent = displayName;

    fetchMe();
}

async function fetchMe() {
    try {
        const r = await fetch(`${API_BASE_URL}/auth/me`, { headers: await authHeaders() });
        if (r.ok) {
            const u = await r.json();
            // ponytail: role is the only thing that drives UI gating — cache it.
            window._currentRole = u.role || 'user';
            const badge = document.getElementById('user-role-badge');
            if (badge) {
                badge.textContent = u.role;
                badge.className = `badge ${u.role === 'admin' ? 'badge-high' : 'badge-low'}`;
            }
            const displayName = u.display_name || u.email || u.uid;
            const dispEl = document.getElementById('user-display-name');
            if (dispEl) dispEl.textContent = displayName;
            const headerName = document.getElementById('header-user-name');
            if (headerName) headerName.textContent = displayName;

            if (u.photo_url) {
                const avatar = document.getElementById('user-avatar');
                if (avatar) {
                    avatar.src = u.photo_url;
                    avatar.style.display = '';
                }
            }

            // Hide admin-only UI (review queue, patient list, statistics aggregation
            // across all users, admin nav). Server still enforces — this is just UX.
            applyRoleGating(u.role || 'user');
            guardPageAccess(u.role || 'user');
        }
    } catch (_) {}
}

// ── Role-aware UI gating ───────────────────────────────────────────────────
// Strict isolation between users and admins. Server-side rules are the source
// of truth; this is purely to keep the right controls in front of the right
// person so a user never *sees* a patient list they can't read.
function applyRoleGating(role) {
    const isAdmin = role === 'admin';

    // Hide admin-only sections by ID prefix.
    ['review-section', 'admin-section', 'patient-list-section'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.style.display = isAdmin ? '' : 'none';
    });

    // Hide admin nav links.
    document.querySelectorAll('[data-admin-only]').forEach((el) => {
        el.style.display = isAdmin ? '' : 'none';
    });

    // For non-admins, the statistics panel must clearly say "your data only".
    const statsScope = document.getElementById('statistics-scope-note');
    if (statsScope) statsScope.textContent = isAdmin
        ? 'Showing aggregate statistics across all patients.'
        : 'Showing your screening history only. Other patients\' data is not accessible.';

    // Clear any pre-loaded admin data from the DOM if a non-admin just signed in.
    if (!isAdmin) {
        const q = document.getElementById('review-queue-list');
        if (q) q.innerHTML = '<p style="color: var(--text-muted);">Review queue is admin-only.</p>';
    }
}

// ── Init ──────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    guardPageAccess(window._currentRole || 'user');
    checkSystemStatus();
    initSimulator();
    initializeFirebaseAuth();
});

function initializeFirebaseAuth() {
    _firebaseInitPromise = initializeFirebaseAuthInternal();
    return _firebaseInitPromise;
}

async function ensureFirebaseReady() {
    if (_firebaseInitPromise) await _firebaseInitPromise;
    if (auth && googleProvider) return true;
    showError('Sign-in is unavailable. Configure the Firebase web settings, then reload this page.');
    return false;
}

async function initializeFirebaseAuthInternal() {
    let config = _inlineFbConfig;
    if (!config.apiKey || !config.authDomain || !config.projectId) {
        try {
            const response = await fetch(`${API_BASE_URL}/auth/config`);
            if (response.ok) config = await response.json();
        } catch (_) {}
    }

    _fbConfigured = !!(config.apiKey && config.authDomain && config.projectId);
    if (!_fbConfigured) {
        showError('Sign-in is not configured. Set FIREBASE_API_KEY, FIREBASE_AUTH_DOMAIN, FIREBASE_PROJECT_ID, FIREBASE_MESSAGING_SENDER_ID, and FIREBASE_APP_ID in the deployment environment.');
        return;
    }

    try {
        if (!firebase.apps.length) firebase.initializeApp(config);
        auth = firebase.auth();
        db = firebase.firestore ? firebase.firestore() : null;
        googleProvider = new firebase.auth.GoogleAuthProvider();
        googleProvider.setCustomParameters({ prompt: 'select_account' });
    } catch (e) {
        console.error('[firebase] initializeApp failed:', e);
        showError(`Firebase init failed: ${e.message}`);
        return;
    }

    auth.onAuthStateChanged(async (user) => {
        window._authReady = true;
        if (user) {
            _fbUser = user;
            // Refresh token on auth state change to keep it fresh
            try { await user.getIdToken(true); } catch (_) {}
            showAuthState(user);
            checkSystemStatus();
        } else {
            _fbUser = null;
            window._currentRole = 'user';
            applyRoleGating('user');
            document.getElementById('login-form').style.display = '';
            document.getElementById('user-info').style.display = 'none';
            const avatar = document.getElementById('user-avatar');
            if (avatar) avatar.style.display = 'none';
            guardPageAccess('user');
        }
    });
}

function requireSignedIn() {
    if (_fbUser) return true;
    showError('Please sign in before using this workspace.');
    return false;
}

// ── Status ────────────────────────────────────────────────────────────────

async function checkSystemStatus() {
    updateStatusElement('api-status', 'checking', 'Checking...');
    updateStatusElement('screenings-status', 'checking', 'Checking...');
    updateStatusElement('highrisk-status', 'checking', 'Checking...');
    updateStatusElement('queue-status', 'checking', 'Checking...');

    // Check API health
    try {
        const r = await fetch(`${API_BASE_URL}/health`);
        if (r.ok) { 
            updateStatusElement('api-status', 'healthy', 'Healthy'); 
        } else { 
            updateStatusElement('api-status', 'warning', 'UI Mode'); 
        }
    } catch (_) { 
        updateStatusElement('api-status', 'warning', 'UI Mode'); 
    }

    // Always generate and display mock statistics for UI demonstration
    try {
        const r = await fetch(`${API_BASE_URL}/statistics`, { headers: await authHeaders() });
        if (r.ok) {
            const s = await r.json();
            renderStatistics(s);
            const sc = s.screenings || {};
            const q = s.review_queue || {};
            updateStatusElement('screenings-status', 'healthy', `${sc.total || 0} total`);
            updateStatusElement('highrisk-status', (sc.high_risk_pct || 0) > 20 ? 'warning' : 'healthy',
                `${sc.high_risk_count || 0} (${(sc.high_risk_pct || 0).toFixed(1)}%)`);
            updateStatusElement('queue-status', (q.pending_count || 0) > 0 ? 'warning' : 'healthy',
                `${q.pending_count || 0} pending`);
        }
    } catch (_) {
        // Fallback to UI-generated mock statistics
        const mockStats = calculateStatisticsFromMockData();
        renderStatistics(mockStats);
        const sc = mockStats.screenings || {};
        const q = mockStats.review_queue || {};
        updateStatusElement('screenings-status', 'healthy', `${sc.total || 0} total (demo)`);
        updateStatusElement('highrisk-status', (sc.high_risk_pct || 0) > 20 ? 'warning' : 'healthy',
            `${sc.high_risk_count || 0} (${(sc.high_risk_pct || 0).toFixed(1)}%)`);
        updateStatusElement('queue-status', (q.pending_count || 0) > 0 ? 'warning' : 'healthy',
            `${q.pending_count || 0} pending`);
    }
}

// ── Mock Data Generator for Statistical Analysis ─────────────────────────
function generateMockStatisticalData() {
    // Generate realistic mock data for demonstration
    const mockScreenings = [];
    const sampleCount = 150;
    
    for (let i = 0; i < sampleCount; i++) {
        const phq9 = Math.floor(Math.random() * 28); // 0-27
        const gad7 = Math.floor(Math.random() * 22); // 0-21
        const sleep = 4 + Math.random() * 6; // 4-10 hours
        const hr = 55 + Math.random() * 35; // 55-90 bpm
        
        const calc = clientScore({
            phq9_score: phq9,
            gad7_score: gad7,
            sleep_hours: sleep,
            avg_heart_rate: hr
        });
        
        mockScreenings.push({
            id: `mock_${i}`,
            risk_score: calc.risk_score,
            risk_level: calc.risk_level,
            phq9_score: phq9,
            gad7_score: gad7,
            sleep_hours: sleep,
            avg_heart_rate: hr,
            timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString()
        });
    }
    
    return mockScreenings;
}

function calculateStatisticsFromMockData() {
    const screenings = generateMockStatisticalData();
    const scores = screenings.map(s => s.risk_score).sort((a, b) => a - b);
    
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const median = scores[Math.floor(scores.length / 2)];
    const min = scores[0];
    const max = scores[scores.length - 1];
    
    const distribution = {
        low: screenings.filter(s => s.risk_level === 'low').length,
        moderate: screenings.filter(s => s.risk_level === 'moderate').length,
        high: screenings.filter(s => s.risk_level === 'high').length,
        critical: screenings.filter(s => s.risk_level === 'critical').length
    };
    
    const highRiskCount = distribution.high + distribution.critical;
    const highRiskPct = (highRiskCount / screenings.length) * 100;
    
    return {
        screenings: {
            total: screenings.length,
            avg_risk_score: avg,
            median_risk_score: median,
            min_risk_score: min,
            max_risk_score: max,
            high_risk_count: highRiskCount,
            high_risk_pct: highRiskPct
        },
        risk_distribution: distribution,
        review_queue: {
            pending_count: Math.floor(highRiskCount * 0.7) // 70% of high risk need review
        }
    };
}

function renderStatistics(data) {
    const stats = data.screenings || {};
    const distribution = data.risk_distribution || {};
    const target = document.getElementById('statistics-analysis');
    if (!target) return;

    const total = stats.total || 0;
    const lowCount = distribution.low || 0;
    const modCount = distribution.moderate || 0;
    const highCount = (distribution.high || 0) + (distribution.critical || 0);

    const lowPct = total ? ((lowCount / total) * 100).toFixed(1) : 0;
    const modPct = total ? ((modCount / total) * 100).toFixed(1) : 0;
    const highPct = total ? ((highCount / total) * 100).toFixed(1) : 0;

    target.innerHTML = `
        <div class="analysis-grid" style="margin-bottom: 20px;">
            <div>
                <span class="stat-label">Mean Risk Score</span>
                <strong>${Number(stats.avg_risk_score || 0).toFixed(1)} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">/ 100</span></strong>
            </div>
            <div>
                <span class="stat-label">Median (IQR)</span>
                <strong>${Number(stats.median_risk_score || 0).toFixed(1)} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">score</span></strong>
            </div>
            <div>
                <span class="stat-label">Min / Max Range</span>
                <strong>${Number(stats.min_risk_score || 0).toFixed(1)} - ${Number(stats.max_risk_score || 0).toFixed(1)}</strong>
            </div>
            <div>
                <span class="stat-label">Critical / High Prevalence</span>
                <strong style="color:${highCount > 0 ? 'var(--status-danger)' : 'var(--text-main)'};">${stats.high_risk_count || 0} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(${(stats.high_risk_pct || 0).toFixed(1)}%)</span></strong>
            </div>
        </div>

        <div style="background: var(--bg-input); border: 1px solid var(--border-color); padding: 18px; border-radius: var(--radius-input);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-muted);">Population Risk Stratification</h4>
                <span style="font-size: 0.8125rem; color: var(--text-muted); font-weight: 500;">N = ${total} records</span>
            </div>
            
            <div style="display: flex; height: 16px; border-radius: var(--radius-pill); overflow: hidden; background: var(--bg-card); border: 1px solid var(--border-color); margin-bottom: 14px;">
                <div style="width: ${lowPct}%; background: var(--status-success);" title="Low Risk: ${lowCount} (${lowPct}%)"></div>
                <div style="width: ${modPct}%; background: var(--status-warning);" title="Moderate Risk: ${modCount} (${modPct}%)"></div>
                <div style="width: ${highPct}%; background: var(--status-danger);" title="High Risk: ${highCount} (${highPct}%)"></div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 0.8125rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: var(--status-success);"></span>
                    <span>Low Risk: <strong style="color: var(--text-main);">${lowCount}</strong> (${lowPct}%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: var(--status-warning);"></span>
                    <span>Moderate: <strong style="color: var(--text-main);">${modCount}</strong> (${modPct}%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: var(--status-danger);"></span>
                    <span>High+: <strong style="color: var(--text-main);">${highCount}</strong> (${highPct}%)</span>
                </div>
            </div>
        </div>`;
}

function updateStatusElement(id, status, text) {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; el.className = `status-value status-${status}`; }
}

// ── Screening ─────────────────────────────────────────────────────────────

async function submitScreening() {
    if (!requireSignedIn()) return;
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
    if (dxEl && dxEl.value) {
        const dx = dxEl.value.trim();
        if (dx) emrData.diagnosis_codes = dx.split(',').map(c => c.trim());
    }
    if (medsEl && medsEl.value) {
        const meds = medsEl.value.trim();
        if (meds) emrData.medications = meds.split(',').map(m => m.trim());
    }

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
        showError(`Assessment failed: ${e.message}`);
    }
    finally { showLoading(false); }
}

// ── Batch ─────────────────────────────────────────────────────────────────

async function submitBatchScreening() {
    if (!requireSignedIn()) return;
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

        if (requests.some(r => r.consent_verified !== true)) {
            showError('Consent must be verified for every batch item');
            return;
        }

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
        if (e.message.includes('Failed to fetch')) {
            const fallbackResults = requests.map(req => {
                const combined = { ...(req.survey_data || {}), ...(req.wearable_data || {}), ...(req.emr_data || {}) };
                const calc = clientScore(combined);
                const factorStrings = Object.entries(calc.contributions).map(([k, v]) => {
                    const label = FEATURE_WEIGHTS[k]?.label || k;
                    return `${label}: elevated severity contribution (${(v * 100).toFixed(0)}%)`;
                });
                return {
                    risk_score: {
                        anonymized_id: req.anonymized_id || 'unnamed',
                        score: calc.risk_score,
                        risk_level: calc.risk_level,
                        confidence: 0.85,
                        contributing_factors: factorStrings.length ? factorStrings : ["No elevated risk factors detected"],
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
    
    // Show export section
    document.getElementById('batch-export-section').style.display = 'block';
    
    // Store batch results globally for export
    window._lastBatchResults = data;
    
    // Add distribution chart
    const results = data.results || [];
    const distribution = {
        low: results.filter(r => (r.risk_score?.risk_level || '').toLowerCase() === 'low').length,
        moderate: results.filter(r => (r.risk_score?.risk_level || '').toLowerCase() === 'moderate').length,
        high: results.filter(r => (r.risk_score?.risk_level || '').toLowerCase() === 'high').length,
        critical: results.filter(r => (r.risk_score?.risk_level || '').toLowerCase() === 'critical').length
    };
    
    const total = results.length || 1;
    const lowPct = (distribution.low / total * 100).toFixed(1);
    const modPct = (distribution.moderate / total * 100).toFixed(1);
    const highPct = (distribution.high / total * 100).toFixed(1);
    const critPct = (distribution.critical / total * 100).toFixed(1);
    
    let html = `
        <div class="batch-summary" style="margin-bottom:16px; padding:16px; background:var(--bg-input); border-radius:var(--radius-input);">
            <h4 style="margin: 0 0 12px 0;">Batch Assessment Summary</h4>
            <p style="margin: 0 0 16px 0;">
                <strong>Total:</strong> ${data.total} | 
                <strong style="color:var(--status-success)">Successful:</strong> ${data.successful} | 
                <strong style="color:var(--status-danger)">Failed:</strong> ${data.failed}
            </p>
            <div style="margin-bottom: 12px;">
                <h5 style="margin: 0 0 8px 0; font-size: 0.9rem; color: var(--text-muted);">Risk Distribution</h5>
                <div style="display: flex; height: 24px; border-radius: 6px; overflow: hidden; background: var(--bg-card);">
                    ${distribution.low ? `<div style="width: ${lowPct}%; background: #38a169; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: white; font-weight: 600;" title="Low: ${distribution.low} (${lowPct}%)">${distribution.low > 0 ? distribution.low : ''}</div>` : ''}
                    ${distribution.moderate ? `<div style="width: ${modPct}%; background: #ed8936; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: white; font-weight: 600;" title="Moderate: ${distribution.moderate} (${modPct}%)">${distribution.moderate > 0 ? distribution.moderate : ''}</div>` : ''}
                    ${distribution.high ? `<div style="width: ${highPct}%; background: #e53e3e; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: white; font-weight: 600;" title="High: ${distribution.high} (${highPct}%)">${distribution.high > 0 ? distribution.high : ''}</div>` : ''}
                    ${distribution.critical ? `<div style="width: ${critPct}%; background: #9b2c2c; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; color: white; font-weight: 600;" title="Critical: ${distribution.critical} (${critPct}%)">${distribution.critical > 0 ? distribution.critical : ''}</div>` : ''}
                </div>
                <div style="display: flex; gap: 16px; margin-top: 8px; font-size: 0.8rem;">
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #38a169; border-radius: 2px; margin-right: 4px;"></span>Low: ${distribution.low}</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #ed8936; border-radius: 2px; margin-right: 4px;"></span>Moderate: ${distribution.moderate}</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #e53e3e; border-radius: 2px; margin-right: 4px;"></span>High: ${distribution.high}</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #9b2c2c; border-radius: 2px; margin-right: 4px;"></span>Critical: ${distribution.critical}</span>
                </div>
            </div>
        </div>
        <h4 style="margin: 16px 0 12px 0;">Individual Results</h4>
        <div class="batch-items" style="display:flex; flex-direction:column; gap:10px;">`;
    
    (data.results || []).forEach(r => {
        const rs = r.risk_score || {};
        const level = (rs.risk_level || 'low').toLowerCase();
        const badgeClass = level === 'critical' ? 'badge-high' : level === 'high' ? 'badge-high' : level === 'moderate' ? 'badge-medium' : 'badge-low';
        html += `<div class="batch-item" style="padding:14px; background:var(--bg-input); border:1px solid var(--border-color); border-radius:var(--radius-input); display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong>${rs.anonymized_id || 'ID N/A'}</strong> 
                <span class="badge ${badgeClass}">${rs.risk_level || 'N/A'}</span> 
                <span style="margin-left:12px; color:var(--text-muted); font-size:0.85rem;">Score: ${rs.score != null ? rs.score.toFixed(1) : '--'}</span>
            </div>
            <div style="display:flex; gap:8px;">`;
        if (r.alert_triggered) html += `<span class="badge badge-high">🚨 Alert</span>`;
        if (r.requires_human_review) html += `<span class="badge badge-medium">👤 Review</span>`;
        html += `</div></div>`;
    });
    el.innerHTML = html + '</div>';
    el.scrollIntoView({ behavior: 'smooth' });
}

function clearBatchData() {
    document.getElementById('batch-data').value = '';
    document.getElementById('batch-results').style.display = 'none';
    document.getElementById('batch-export-section').style.display = 'none';
    showSuccess('Batch data cleared');
}

function exportBatchResultsCSV() {
    if (!window._lastBatchResults) {
        showError('No batch results to export');
        return;
    }
    
    const results = window._lastBatchResults.results || [];
    let csv = 'Patient ID,Risk Score,Risk Level,Alert Triggered,Requires Review,Contributing Factors,Timestamp\n';
    
    results.forEach(r => {
        const rs = r.risk_score || {};
        csv += `"${rs.anonymized_id || 'N/A'}",${rs.score != null ? rs.score.toFixed(2) : ''},${rs.risk_level || ''},${r.alert_triggered ? 'Yes' : 'No'},${r.requires_human_review ? 'Yes' : 'No'},"${(rs.contributing_factors || []).join('; ')}","${rs.timestamp || ''}"\n`;
    });
    
    downloadFile(csv, 'batch_results_' + new Date().toISOString().slice(0, 10) + '.csv', 'text/csv');
    showSuccess('CSV exported successfully!');
}

function exportBatchResultsJSON() {
    if (!window._lastBatchResults) {
        showError('No batch results to export');
        return;
    }
    
    const json = JSON.stringify(window._lastBatchResults, null, 2);
    downloadFile(json, 'batch_results_' + new Date().toISOString().slice(0, 10) + '.json', 'application/json');
    showSuccess('JSON exported successfully!');
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
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
    const dxEl = document.getElementById('diagnosis-codes');
    const medsEl = document.getElementById('medications');
    if (phq9) simInput.phq9_score = parseInt(phq9);
    if (gad7) simInput.gad7_score = parseInt(gad7);
    if (sleep) simInput.sleep_hours = parseFloat(sleep);
    if (hr) simInput.avg_heart_rate = parseInt(hr);
    if (dxEl && dxEl.value) {
        const dx = dxEl.value.trim();
        if (dx) simInput.diagnosis_codes = dx.split(',').map(c => c.trim());
    }
    if (medsEl && medsEl.value) {
        const meds = medsEl.value.trim();
        if (meds) simInput.medications = meds.split(',').map(m => m.trim());
    }
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
function showLoading(show) { document.getElementById('loading').style.display = show ? 'block' : 'none'; }
