/* ============================================================================
   MindMetrics — shared workspace chrome.
   ponytail: replaces the ~50 lines of duplicated sidebar/drawer/header/footer
   that were hand-copied across 6 workspace pages. Reads `data-page` from
   <body> to mark the active link, and `data-chrome-slot` placeholders on the
   page to slot in per-page content. Renders before DOMContentLoaded so the
   existing app.js init handlers find the chrome elements they expect.
   ============================================================================ */
(function () {
    'use strict';

    // Inline SVG icons — verbatim from the original markup so we don't drift.
    const ICON = {
        dashboard: '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
        screening: '<path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/><path d="M9 12h6M9 16h4"/>',
        batch: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
        statistics: '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
        queue: '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
        profile: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    };

    const NAV = [
        { id: 'dashboard',  href: 'index.html',      label: 'Dashboard' },
        { id: 'screening',  href: 'screening.html',  label: 'Screening Tool' },
        { id: 'batch',      href: 'batch.html',      label: 'Batch Analytics' },
        { id: 'statistics', href: 'statistics.html', label: 'Statistical Analysis' },
        { id: 'queue',      href: 'queue.html',      label: 'Review Queue', admin: true },
        { id: 'profile',    href: 'profile.html',    label: 'Profile' },
    ];

    const LEGAL = [
        { href: 'privacy.html',    label: 'Privacy' },
        { href: 'hipaa.html',      label: 'HIPAA' },
        { href: 'disclaimer.html', label: 'Disclaimer' },
        { href: 'terms.html',      label: 'Terms' },
    ];

    function navItem(n, active, drawer) {
        const admin = n.admin ? ' data-admin-only' : '';
        // Drawer links close the drawer on click; sidebar links use the existing scroll/active handler.
        const close = drawer ? ' data-action="closeMobileMenu"' : '';
        return `<a href="${n.href}" class="nav-item${active ? ' active' : ''}"${admin}${close} title="${n.label}" aria-current="${active ? 'page' : 'false'}">
            <svg class="nav-icon" viewBox="0 0 24 24">${ICON[n.id]}</svg>
            <span class="nav-text">${n.label}</span>
        </a>`;
    }

    function buildShell(active) {
        return `
            <div class="mobile-nav-bar">
                <div class="mobile-brand">MindMetrics</div>
                <button id="mobile-menu-btn" class="mobile-menu-btn" data-action="toggleMobileMenu" aria-label="Open menu" aria-expanded="false">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
                </button>
            </div>
            <div id="mobile-drawer" class="mobile-drawer">
                ${NAV.map((n) => navItem(n, n.id === active, true)).join('')}
                <button data-action="toggleTheme" class="btn btn-secondary" style="margin-top:auto;">Toggle Theme</button>
            </div>
            <div class="app-wrapper" id="app-wrapper">
                <aside class="app-sidebar">
                    <div class="sidebar-brand">
                        <div class="sidebar-brand-inner">
                            <div class="brand-mark" aria-hidden="true">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h3l2.5-7 3 14 2.5-7H21"/></svg>
                            </div>
                            <div class="brand-text">
                                <div class="brand-title">MindMetrics</div>
                                <div class="brand-subtitle">Risk Data Science</div>
                            </div>
                        </div>
                    </div>
                    <nav class="sidebar-nav" aria-label="Primary">
                        ${NAV.map((n) => navItem(n, n.id === active, false)).join('')}
                    </nav>
                    <div class="sidebar-footer">
                        <button class="sidebar-collapse-btn" data-action="toggleSidebar" title="Collapse sidebar" aria-label="Collapse sidebar">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
                            <span class="nav-text">Collapse</span>
                        </button>
                    </div>
                </aside>
                <main class="app-main">
                    <header class="top-header">
                        <div data-chrome-slot="page-header"></div>
                        <div class="header-toolbar">
                            <div id="user-header-pill" class="user-header-pill">
                                <img id="user-avatar" src="" alt="Avatar" style="display:none;">
                                <span id="header-user-name" class="header-user-name">Guest</span>
                                <span id="user-role-badge" class="badge badge-low">User</span>
                            </div>
                            <button id="theme-toggle-btn" data-action="toggleTheme" class="btn btn-secondary btn-theme" title="Toggle theme">
                                <svg class="theme-icon" id="theme-toggle-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"></svg>
                                <span id="theme-toggle-text">Light mode</span>
                            </button>
                        </div>
                    </header>
                    <div data-chrome-slot="page-body"></div>
                    <footer class="page-footer">
                        <p>MindMetrics · Clinical decision support, not a medical device.</p>
                        <nav class="footer-links" aria-label="Legal">
                            ${LEGAL.map((l) => `<a href="${l.href}">${l.label}</a>`).join('')}
                        </nav>
                    </footer>
                </main>
            </div>`;
    }

    function mountChrome() {
        const slot = document.getElementById('app-chrome');
        if (!slot) return false; // legal/standalone page — skip silently

        const active = document.body.dataset.page || '';
        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildShell(active);
        slot.replaceWith(...Array.from(wrapper.children));

        // Page supplies header + body in <template data-chrome-template="page-header|page-body">…</template>
        ['page-header', 'page-body'].forEach((name) => {
            const tpl = document.querySelector(`template[data-chrome-template="${name}"]`);
            const target = document.querySelector(`[data-chrome-slot="${name}"]`);
            if (!target) return;
            if (tpl) {
                target.replaceWith(tpl.content.cloneNode(true));
            } else {
                // ponytail: visible "missing slot" marker, so authors see it.
                target.innerHTML = '<p class="chrome-missing">—</p>';
            }
        });

        return true;
    }

    // Run as early as possible so DOMContentLoaded handlers see the chrome.
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mountChrome, { once: true });
    } else {
        mountChrome();
    }
})();
