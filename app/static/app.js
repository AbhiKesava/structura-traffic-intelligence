/**
 * STRUCTURA — TRAFFIC INTELLIGENCE
 * Interactive Client Controller
 * Multi-Horizon Speed Forecasting, Corridor Monitoring, Map Tooltips & Auth Flow
 */

(function () {
  'use strict';

  // Global Application State
  const state = {
    user: null, // session object { email, name, role }
    authToken: null,
    savedState: {},
    currentView: 'overview',
    unit: 'mph', // 'mph' | 'kmh'
    activeHorizon: 30, // 0 | 15 | 30 | 60
    selectedSensorId: '7100075', // default to I-405 / I-10 corridor
    selectedHighway: 'all',
    selectedModel: 'historical_average',
    selectedTimestampIdx: -1,
    sensorsList: [],
    corridorsData: [],
    overviewData: null,
    currentTrajectoryData: null,
    departureData: null
  };

  function getAuthHeaders(extra = {}) {
    const headers = Object.assign({}, extra);
    if (state.authToken) {
      headers['Authorization'] = `Bearer ${state.authToken}`;
    }
    if (state.user && state.user.email) {
      headers['X-User-Email'] = state.user.email;
    }
    return headers;
  }

  function getInitials(name) {
    const parts = String(name || 'AT').trim().split(/\s+/).filter(Boolean);
    if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return 'AT';
  }

  let persistStateTimer = null;
  let pendingStatePatch = {};

  function persistUserState(patch) {
    if (!state.user || !state.user.email) return;
    Object.assign(pendingStatePatch, patch);
    clearTimeout(persistStateTimer);
    persistStateTimer = setTimeout(async () => {
      const toSend = Object.assign({}, pendingStatePatch);
      pendingStatePatch = {};
      try {
        await fetch('/api/user/state', {
          method: 'PUT',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ state: toSend })
        });
      } catch (e) {
        console.warn('Could not persist user state:', e);
      }
    }, 200);
  }

  function restoreUserState(saved) {
    if (!saved || typeof saved !== 'object') return;
    state.savedState = saved;

    if (saved.unit === 'kmh' || saved.unit === 'mph') {
      state.unit = saved.unit;
    }

    // Restore Module 1: Congestion inputs
    if (saved.congestion_inputs) {
      const c = saved.congestion_inputs;
      const setVal = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
      setVal('cong-road-segment', c.road_segment);
      setVal('cong-volume', c.traffic_volume);
      setVal('cong-occupancy', c.occupancy);
      setVal('cong-time', c.time_of_day);
      setVal('cong-day', c.day_of_week);
      setVal('cong-temp', c.temperature);
      setVal('cong-rain', c.rainfall);
    }

    // Restore Module 2: Speed inputs
    if (saved.speed_inputs) {
      const s = saved.speed_inputs;
      const setVal = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
      setVal('spd-road-segment', s.road_segment);
      setVal('spd-volume', s.traffic_volume);
      setVal('spd-occupancy', s.occupancy);
      setVal('spd-time', s.time_of_day);
      setVal('spd-day', s.day_of_week);
      setVal('spd-temp', s.temperature);
      setVal('spd-rain', s.rainfall);
    }

    // Restore Module 3: Forecast inputs
    if (saved.forecast_inputs) {
      const f = saved.forecast_inputs;
      const setVal = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
      setVal('fc-segment', f.road_segment);
      setVal('fc-vol', f.current_volume);
      setVal('fc-occ', f.current_occupancy);
      setVal('fc-spd', f.current_speed);
      setVal('fc-hour', f.hour);
      setVal('fc-rain', f.rainfall);
    }
  }

  // Initialization
  async function init() {
    // If user explicitly logged out and has no active token, redirect to /login
    const token = localStorage.getItem('structura_auth_token');
    const wasLoggedOut = localStorage.getItem('structura_logged_out') === '1';
    const isDashboardPage = !!document.getElementById('main-dashboard-app') && !document.getElementById('view-login');

    if (isDashboardPage && wasLoggedOut && !token) {
      window.location.href = '/login';
      return;
    }

    state.authToken = token || null;

    // Load authenticated user + persisted SQLite state BEFORE rendering ML modules
    if (state.authToken) {
      try {
        const meResp = await fetch('/api/auth/me', {
          headers: { 'Authorization': `Bearer ${state.authToken}` }
        });
        if (meResp.ok) {
          const meData = await meResp.json();
          state.user = meData.user;
          localStorage.setItem('structura_user_session', JSON.stringify(meData.user));
          restoreUserState(meData.saved_state);
        } else {
          // Expired/invalid token -> clean fallback
          localStorage.removeItem('structura_auth_token');
          state.authToken = null;
        }
      } catch (e) {
        console.warn('Auth profile restore warning:', e);
      }
    }

    if (!state.user) {
      const savedUser = localStorage.getItem('structura_user_session');
      if (savedUser) {
        try {
          state.user = JSON.parse(savedUser);
        } catch (e) {
          state.user = { email: 'alex.taylor@structura.io', name: 'Alex Taylor', role: 'Traffic analyst' };
        }
      } else {
        state.user = { email: 'alex.taylor@structura.io', name: 'Alex Taylor', role: 'Traffic analyst' };
        localStorage.setItem('structura_user_session', JSON.stringify(state.user));
      }

      // Restore saved SQLite state for this email
      try {
        const stResp = await fetch(`/api/user/state?user_email=${encodeURIComponent(state.user.email)}`);
        if (stResp.ok) {
          const stData = await stResp.json();
          restoreUserState(stData.saved_state);
        }
      } catch (e) {
        console.warn('State restore warning:', e);
      }
    }

    setupAuthFlow();
    setupNavigation();
    setupTopBar();
    setupOverviewInteractions();
    setupExploreView();
    setupCompareView();
    setupCsvUpload();
    setupMLPlatform();

    showDashboard();

    // Initial data fetch from FastAPI backend
    fetchSensors();
    fetchOverview();
    fetchCorridors();
    loadSensorTrajectory(state.selectedSensorId);
  }

  // =========================================================================
  // 1. AUTHENTICATION & LOGIN FLOW (MATCHES IMAGE 2)
  // =========================================================================
  function setupAuthFlow() {
    const signinForm = document.getElementById('signin-form');
    const registerForm = document.getElementById('register-form');
    const btnTogglePw = document.getElementById('btn-toggle-pw');
    const btnQuickDemo = document.getElementById('btn-quick-demo');
    const btnDemoSsoG = document.getElementById('btn-demo-sso-g');
    const btnDemoSsoMs = document.getElementById('btn-demo-sso-ms');
    const btnSwitchRegister = document.getElementById('btn-switch-to-register');
    const btnSwitchSignin = document.getElementById('btn-switch-to-signin');
    const btnForgotPassword = document.getElementById('btn-forgot-password');
    const forgotModal = document.getElementById('forgot-modal');
    const forgotClose = document.getElementById('forgot-modal-close');
    const forgotOk = document.getElementById('forgot-modal-ok');
    const btnSignOut = document.getElementById('btn-sign-out');
    const linkPrivacy = document.getElementById('link-privacy');
    const linkHelp = document.getElementById('link-help');

    // Show/Hide Password Toggle
    if (btnTogglePw) {
      btnTogglePw.addEventListener('click', function () {
        const input = document.getElementById('signin-password');
        const eyeIcon = document.getElementById('pw-eye-icon');
        if (input.type === 'password') {
          input.type = 'text';
          if (eyeIcon) {
            eyeIcon.innerHTML = `<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line>`;
          }
        } else {
          input.type = 'password';
          if (eyeIcon) {
            eyeIcon.innerHTML = `<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle>`;
          }
        }
      });
    }

    // Switch between Sign in and Register
    if (btnSwitchRegister && btnSwitchSignin) {
      btnSwitchRegister.addEventListener('click', function () {
        document.querySelector('.login-form-panel .form-inner-container').style.display = 'none';
        document.getElementById('register-card-panel').style.display = 'block';
      });

      btnSwitchSignin.addEventListener('click', function () {
        document.getElementById('register-card-panel').style.display = 'none';
        document.querySelector('.login-form-panel .form-inner-container').style.display = 'block';
      });
    }

    // Passkey / Instant Demo Access
    if (btnQuickDemo) {
      btnQuickDemo.addEventListener('click', function () {
        completeSignIn('alex.taylor@structura.io', 'Alex Taylor', true);
      });
    }

    // SSO Mock buttons
    if (btnDemoSsoG) {
      btnDemoSsoG.addEventListener('click', function () {
        completeSignIn('alex.taylor@gmail.com', 'Alex Taylor (Google)', true);
      });
    }
    if (btnDemoSsoMs) {
      btnDemoSsoMs.addEventListener('click', function () {
        completeSignIn('alex.taylor@microsoft.com', 'Alex Taylor (Microsoft)', true);
      });
    }

    // Sign In Form Submission
    if (signinForm) {
      signinForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const email = document.getElementById('signin-email').value.trim();
        const pw = document.getElementById('signin-password').value;
        const remember = document.getElementById('signin-remember').checked;
        const errBox = document.getElementById('signin-error-box');
        const errMsg = document.getElementById('signin-error-msg');
        const submitBtn = document.getElementById('btn-submit-signin');

        // Validation
        if (!email || !email.includes('@')) {
          errBox.style.display = 'block';
          errMsg.textContent = 'Please enter a valid email address.';
          return;
        }

        if (!pw || pw.length < 4) {
          errBox.style.display = 'block';
          errMsg.textContent = 'Password must be at least 4 characters.';
          return;
        }

        errBox.style.display = 'none';
        if (submitBtn) {
          submitBtn.innerHTML = `<span>Verifying credentials...</span>`;
          submitBtn.disabled = true;
        }

        setTimeout(() => {
          if (submitBtn) {
            submitBtn.innerHTML = `<span>Sign in to dashboard</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
            submitBtn.disabled = false;
          }
          const name = email.toLowerCase().includes('alex') ? 'Alex Taylor' : email.split('@')[0];
          completeSignIn(email, name, remember);
        }, 300);
      });
    }

    // Register Form Submission
    if (registerForm) {
      registerForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const name = document.getElementById('reg-name').value.trim();
        const email = document.getElementById('reg-email').value.trim();
        const pw = document.getElementById('reg-password').value;

        if (!name || !email || !pw) {
          showToast('Please complete all fields.');
          return;
        }

        completeSignIn(email, name, true);
        showToast(`Account created for ${name}!`);
      });
    }

    // Forgot Password Modal
    if (btnForgotPassword && forgotModal) {
      btnForgotPassword.addEventListener('click', () => { forgotModal.style.display = 'flex'; });
      if (forgotClose) forgotClose.addEventListener('click', () => { forgotModal.style.display = 'none'; });
      if (forgotOk) {
        forgotOk.addEventListener('click', () => {
          document.getElementById('signin-email').value = 'alex.taylor@structura.io';
          document.getElementById('signin-password').value = 'structura2026';
          forgotModal.style.display = 'none';
          showToast('Alex Taylor demo credentials populated.');
        });
      }
    }

    // Privacy & Help link handlers
    if (linkPrivacy) {
      linkPrivacy.addEventListener('click', (e) => {
        e.preventDefault();
        showDashboard();
        switchView('about');
      });
    }
    if (linkHelp) {
      linkHelp.addEventListener('click', (e) => {
        e.preventDefault();
        showDashboard();
        switchView('about');
      });
    }

    // Floating View Switcher: Instant 1-click toggle between Dashboard (Image 1) and Login (Image 2)
    const btnMockupToggle = document.getElementById('btn-toggle-mockup-view');
    if (btnMockupToggle) {
      btnMockupToggle.addEventListener('click', function () {
        const loginView = document.getElementById('view-login');
        if (loginView && loginView.style.display !== 'none') {
          showDashboard();
          showToast('Viewing Dashboard Screen (Image 1)');
        } else {
          showLoginScreen();
          showToast('Viewing Login Screen (Image 2)');
        }
      });
    }

    // Sign Out (clears active token while preserving all account & workspace data in SQLite)
    if (btnSignOut) {
      btnSignOut.addEventListener('click', async function (e) {
        e.preventDefault();
        try {
          if (state.authToken) {
            await fetch('/api/auth/logout', {
              method: 'POST',
              headers: getAuthHeaders()
            });
          }
        } catch (err) {
          console.warn('Logout notice:', err);
        }
        if (state.user && state.user.email) {
          localStorage.setItem('structura_last_email', state.user.email);
        }
        localStorage.removeItem('structura_auth_token');
        localStorage.removeItem('structura_user_session');
        localStorage.setItem('structura_logged_out', '1');
        state.authToken = null;
        window.location.href = '/login';
      });
    }
  }

  function completeSignIn(email, name, remember) {
    state.user = {
      email: email,
      name: name || 'Alex Taylor',
      role: 'Traffic analyst'
    };

    if (remember) {
      localStorage.setItem('structura_user_session', JSON.stringify(state.user));
    }

    showDashboard();
    showToast(`Welcome back, ${state.user.name.split(' ')[0]}!`);
  }

  function showDashboard() {
    const loginView = document.getElementById('view-login');
    const dashboardApp = document.getElementById('main-dashboard-app');
    if (loginView) loginView.style.display = 'none';
    if (dashboardApp) dashboardApp.style.display = 'flex';

    const toggleText = document.getElementById('mockup-toggle-text');
    if (toggleText) toggleText.textContent = 'Switch to Login Screen (Image 2)';

    // Update user profile pill
    const nameEl = document.getElementById('display-user-name');
    const avatarEl = document.querySelector('.profile-avatar-circle');
    const topGreeting = document.getElementById('top-greeting-heading');

    if (state.user) {
      const firstName = state.user.name.split(' ')[0];
      if (nameEl) nameEl.textContent = state.user.name;
      if (avatarEl) {
        avatarEl.textContent = getInitials(state.user.name);
      }
      if (topGreeting) topGreeting.textContent = `Good morning, ${firstName}.`;
      const exportBtn = document.getElementById('btn-export-csv');
      if (exportBtn && state.user.email) {
        exportBtn.href = `/api/predictions/export?user_email=${encodeURIComponent(state.user.email)}`;
      }
    }

    const urlParams = new URLSearchParams(window.location.search);
    const requestedView = urlParams.get('view') || 'overview';
    switchView(requestedView);
  }

  function showLoginScreen() {
    const loginView = document.getElementById('view-login');
    const dashboardApp = document.getElementById('main-dashboard-app');
    if (dashboardApp) dashboardApp.style.display = 'none';
    if (loginView) loginView.style.display = 'flex';

    const toggleText = document.getElementById('mockup-toggle-text');
    if (toggleText) toggleText.textContent = 'Switch to Dashboard (Image 1)';
  }

  // =========================================================================
  // 2. SIDEBAR NAVIGATION
  // =========================================================================
  function setupNavigation() {
    const navButtons = document.querySelectorAll('.sidebar-nav-btn');
    navButtons.forEach(btn => {
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        const view = btn.getAttribute('data-view');
        if (view) {
          switchView(view);
        }
      });
    });

    const btnViewAll = document.getElementById('btn-view-all-corridors');
    if (btnViewAll) {
      btnViewAll.addEventListener('click', () => switchView('explore'));
    }
  }

  function switchView(viewName) {
    state.currentView = viewName;

    // Normalize views: 'live-map' and 'saved-corridors' map to 'overview'
    let targetViewId = viewName;
    if (viewName === 'live-map' || viewName === 'saved-corridors') {
      targetViewId = 'overview';
    } else if (viewName === 'reports') {
      targetViewId = 'about';
    }

    // Update active nav button
    document.querySelectorAll('.sidebar-nav-btn').forEach(btn => {
      const btnView = btn.getAttribute('data-view');
      btn.classList.toggle('active', btnView === viewName);
    });

    // Hide all view panels and show target
    document.querySelectorAll('.structura-view-panel').forEach(panel => {
      panel.style.display = 'none';
      panel.classList.remove('active');
    });

    const targetPanel = document.getElementById(`view-${targetViewId}`);
    if (targetPanel) {
      targetPanel.style.display = 'block';
      targetPanel.classList.add('active');
    }

    // Context actions on view change
    if (targetViewId === 'compare') {
      runDepartureComparison();
    } else if (viewName === 'live-map') {
      const mapEl = document.querySelector('.conditions-map-card');
      if (mapEl) mapEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else if (viewName === 'saved-corridors') {
      const listEl = document.querySelector('.corridors-watch-card');
      if (listEl) listEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else if (targetViewId === 'ml-traffic') {
      runCongestionPrediction(false);
    } else if (targetViewId === 'ml-speed') {
      runSpeedPrediction(false);
    } else if (targetViewId === 'ml-forecast') {
      runForecastPrediction(false);
    } else if (targetViewId === 'ml-dataset' && !datasetLoaded) {
      loadDatasetSummary();
    } else if (targetViewId === 'ml-evaluation' && !evalLoaded) {
      loadModelEvaluation();
    } else if (targetViewId === 'ml-comparison' && !comparisonLoaded) {
      loadModelComparison();
    } else if (targetViewId === 'ml-analytics') {
      loadFeatureImportanceAndHistory();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // =========================================================================
  // 3. TOP BAR ACTIONS
  // =========================================================================
  function setupTopBar() {
    const btnUnit = document.getElementById('btn-toggle-unit');
    const unitLabel = document.getElementById('unit-pill-label');
    const btnAdd = document.getElementById('btn-add-corridor');

    if (btnUnit) {
      btnUnit.addEventListener('click', function () {
        state.unit = state.unit === 'mph' ? 'kmh' : 'mph';
        if (unitLabel) unitLabel.textContent = state.unit;
        updateSpeedDisplays();
        persistUserState({ unit: state.unit });
        showToast(`Display speed units updated to ${state.unit.toUpperCase()}`);
      });
    }

    if (btnAdd) {
      btnAdd.addEventListener('click', function () {
        showToast('Corridor monitor added to your personal watchlist.');
      });
    }

    const btnResetDs = document.getElementById('btn-reset-dataset');
    if (btnResetDs) {
      btnResetDs.addEventListener('click', async function () {
        try {
          const resp = await fetch('/api/dataset/reset', {
            method: 'POST',
            headers: getAuthHeaders()
          });
          if (resp.ok) {
            state.activeDatasetOverview = null;
            applyOverviewMetrics({ is_custom_dataset: false });
            datasetLoaded = false;
            showToast('Dashboard restored to default dataset.');
          }
        } catch (e) {
          console.warn('Reset dataset failed:', e);
        }
      });
    }
  }

  function applyOverviewMetrics(metrics) {
    const isCustom = !!(metrics && metrics.is_custom_dataset);
    state.activeDatasetOverview = isCustom ? metrics : null;

    const resetBtn = document.getElementById('btn-reset-dataset');
    if (resetBtn) resetBtn.style.display = isCustom ? 'inline-flex' : 'none';

    const syncEl = document.getElementById('top-sync-time');
    if (syncEl) {
      syncEl.textContent = isCustom
        ? `Dataset: ${metrics.filename} (${(metrics.total_rows || 0).toLocaleString()} rows)`
        : 'Data updated 2 min ago';
    }

    // 1. Network Speed card
    const kpiSpeed = document.getElementById('kpi-network-speed');
    const kpiUnit = document.getElementById('kpi-network-unit');
    const spdMph = isCustom ? metrics.network_speed_mph : 47;
    const spdKmh = isCustom ? metrics.network_speed_kmh : 76;
    if (kpiSpeed) kpiSpeed.textContent = state.unit === 'kmh' ? spdKmh : spdMph;
    if (kpiUnit) kpiUnit.textContent = state.unit;

    const trendEl = document.getElementById('kpi-speed-trend');
    const trendPct = isCustom ? metrics.trend_pct : 6;
    if (trendEl) {
      const up = trendPct >= 0;
      const cls = up ? 'trend-green' : 'trend-amber';
      const arrow = up ? '&uarr;' : '&darr;';
      trendEl.innerHTML = `<span class="${cls}">${arrow} ${Math.abs(trendPct)}%</span> ${isCustom ? 'vs. 45 mph baseline' : 'vs. yesterday'}`;
    }

    // 2. Active Slowdowns card
    const slowCount = document.getElementById('kpi-slowdowns-count');
    const slowBreak = document.getElementById('kpi-slowdowns-breakdown');
    if (slowCount) slowCount.textContent = isCustom ? metrics.active_slowdowns : 12;
    if (slowBreak) {
      const sev = isCustom ? metrics.severe_count : 3;
      const hi = isCustom ? metrics.high_count : 5;
      const mod = isCustom ? metrics.moderate_count : 4;
      slowBreak.innerHTML = `
        <span class="breakdown-dot dot-severe"></span> ${sev} severe &middot;
        <span class="breakdown-dot dot-high"></span> ${hi} high &middot;
        <span class="breakdown-dot dot-moderate"></span> ${mod} moderate
      `;
    }

    // 3. Avg Delay card
    const delayEl = document.getElementById('kpi-avg-delay');
    const delaySub = document.getElementById('kpi-delay-subtext');
    const dMin = isCustom ? metrics.avg_delay_min : 8;
    if (delayEl) delayEl.textContent = `+${dMin}`;
    if (delaySub) {
      delaySub.innerHTML = isCustom
        ? `<span class="trend-amber">15-mi corridor</span> vs. 60 mph free flow`
        : `<span class="trend-amber">&uarr; 2 min</span> in the last hour`;
    }

    // 4. Best Departure Window card
    const bestDep = document.getElementById('kpi-best-departure');
    const bestDesc = document.getElementById('kpi-best-window-desc');
    if (bestDep) bestDep.textContent = isCustom ? metrics.best_departure_window : '10:30 – 11:00';
    if (bestDesc) bestDesc.textContent = isCustom ? metrics.best_window_desc : 'Typically 18% faster than now';

    // 5. Corridor Map subtitle & Tooltip
    const mapSub = document.getElementById('map-section-sub');
    if (mapSub) {
      mapSub.textContent = isCustom ? metrics.map_subtitle : 'Los Angeles · Live and forecasted traffic';
    }
    const tooltip = document.getElementById('map-tooltip-box');
    if (tooltip) {
      const hwyText = tooltip.querySelector('text:first-of-type');
      const valText = tooltip.querySelector('text:last-of-type');
      if (isCustom && metrics.map_tooltip) {
        const t = metrics.map_tooltip;
        const dispSpd = state.unit === 'kmh' ? `${t.speed_kmh} km/h` : `${t.speed_mph} mph`;
        const cColor = t.congestion === 'SEVERE' ? '#ef4444' : (t.congestion === 'HIGH' ? '#fb923c' : (t.congestion === 'MODERATE' ? '#f59e0b' : '#7ea872'));
        if (hwyText) hwyText.textContent = t.label;
        if (valText) {
          valText.innerHTML = `<tspan fill="#ffffff">${dispSpd}</tspan> <tspan fill="#64748b">&ndash;</tspan> <tspan fill="${cColor}">${t.congestion}</tspan>`;
        }
      } else {
        if (hwyText) hwyText.textContent = 'I-405 NORTH';
        if (valText) {
          valText.innerHTML = `<tspan fill="#ffffff">32 mph</tspan> <tspan fill="#64748b">&ndash;</tspan> <tspan fill="#fb923c">HIGH</tspan>`;
        }
      }
    }

    // 6. Speed Forecast Card (headline, badge, SVG curves)
    const forecastHeadline = document.getElementById('forecast-speed-headline');
    if (forecastHeadline) {
      forecastHeadline.textContent = state.unit === 'kmh' ? `${spdKmh} km/h` : `${spdMph} mph`;
    }
    const forecastBadge = document.getElementById('forecast-trend-badge');
    if (forecastBadge) {
      const sign = trendPct >= 0 ? '+' : '';
      forecastBadge.textContent = isCustom ? `${sign}${trendPct}% vs base` : '+6% today';
    }

    const predictedCurve = document.getElementById('speed-predicted-curve');
    const baselineCurve = document.getElementById('speed-baseline-curve');
    if (predictedCurve) {
      if (isCustom && metrics.forecast_points_mph && metrics.forecast_points_mph.length >= 4) {
        const pts = metrics.forecast_points_mph;
        const speedToY = (s) => {
          const clamped = Math.max(15, Math.min(70, s));
          return Math.round(155 - ((clamped - 15) / 55) * 135);
        };
        const y0 = speedToY(pts[0]), y1 = speedToY(pts[1]), y2 = speedToY(pts[2]), y3 = speedToY(pts[3]);
        predictedCurve.setAttribute(
          'd',
          `M 35 ${y0} C 75 ${y0}, 95 ${y1}, 135 ${y1} C 175 ${y1}, 195 ${y2}, 235 ${y2} C 275 ${y2}, 295 ${y3}, 325 ${y3}`
        );
        if (baselineCurve) {
          const bY = speedToY(metrics.baseline_mph || 45);
          baselineCurve.setAttribute('d', `M 30 ${bY + 6} C 110 ${bY + 2}, 220 ${bY - 2}, 330 ${bY - 6}`);
        }
      } else {
        predictedCurve.setAttribute('d', 'M 30 112 C 60 112, 85 90, 110 88 C 135 86, 160 115, 185 110 C 215 102, 240 70, 270 65 C 295 62, 315 45, 330 25');
        if (baselineCurve) {
          baselineCurve.setAttribute('d', 'M 30 115 C 80 108, 140 102, 200 95 C 260 88, 300 80, 330 75');
        }
      }
    }

    // 7. Corridors to Watch list
    renderCorridorsWatchList();
  }

  function updateSpeedDisplays() {
    if (state.activeDatasetOverview) {
      applyOverviewMetrics(state.activeDatasetOverview);
    } else {
      const kpiSpeed = document.getElementById('kpi-network-speed');
      const kpiUnit = document.getElementById('kpi-network-unit');
      if (kpiSpeed) kpiSpeed.textContent = state.unit === 'kmh' ? '76' : '47';
      if (kpiUnit) kpiUnit.textContent = state.unit;

      const forecastHeadline = document.getElementById('forecast-speed-headline');
      if (forecastHeadline) forecastHeadline.textContent = state.unit === 'kmh' ? '76 km/h' : '47 mph';

      renderCorridorsWatchList();
    }

    if (state.currentTrajectoryData) {
      updateExploreCards(state.currentTrajectoryData);
    }
  }

  // =========================================================================
  // 4. OVERVIEW DASHBOARD & MAP INTERACTIONS (SYNCED TO ACTIVE UPLOADED DATASET)
  // =========================================================================
  function fetchOverview() {
    fetch(`/api/overview?timestamp_idx=${state.selectedTimestampIdx}`)
      .then(r => r.json())
      .then(data => {
        state.overviewData = data;
      })
      .catch(err => console.warn('Overview API warning:', err));

    // Also load dynamic dataset-backed Overview summary for the logged-in user
    fetch('/api/dashboard/summary', { headers: getAuthHeaders() })
      .then(r => r.json())
      .then(summary => {
        if (summary && summary.is_custom_dataset) {
          applyOverviewMetrics(summary);
        }
      })
      .catch(err => console.warn('Dashboard summary warning:', err));
  }

  function fetchCorridors() {
    fetch('/api/corridors')
      .then(r => r.json())
      .then(data => {
        state.corridorsData = data;
      })
      .catch(err => console.warn('Corridors fetch warning:', err));
  }

  function renderCorridorsWatchList() {
    const listWrap = document.querySelector('.corridors-watch-list');
    if (!listWrap) return;

    // Use active uploaded dataset corridors when present; otherwise default reference corridors
    const refCorridors = (state.activeDatasetOverview && state.activeDatasetOverview.corridors && state.activeDatasetOverview.corridors.length > 0)
      ? state.activeDatasetOverview.corridors
      : [
      {
        name: 'I-405 North - Sepulveda Pass',
        speed_mph: 32,
        speed_kmh: 51,
        risk: 'HIGH',
        riskClass: 'badge-high',
        status: 'Expected to clear in 24 min',
        arrowClass: 'arrow-orange',
        sensor: '7100020'
      },
      {
        name: 'US-101 South - Hollywood',
        speed_mph: 41,
        speed_kmh: 66,
        risk: 'MODERATE',
        riskClass: 'badge-moderate',
        status: 'Steady for the next hour',
        arrowClass: '',
        sensor: '7100115'
      },
      {
        name: 'I-10 East - Downtown LA',
        speed_mph: 52,
        speed_kmh: 84,
        risk: 'LOW',
        riskClass: 'badge-low',
        status: 'Flowing normally',
        arrowClass: '',
        sensor: '7100075'
      }
    ];

    listWrap.innerHTML = refCorridors.map(c => {
      const sp = state.unit === 'kmh' ? `${c.speed_kmh} km/h` : `${c.speed_mph} mph`;
      return `
        <div class="corridor-watch-row" data-sensor="${c.sensor}">
          <div class="col-road-name">
            <span class="road-title">${c.name}</span>
          </div>
          <div class="col-speed">
            <span class="road-speed-val">${sp}</span>
          </div>
          <div class="col-risk">
            <span class="risk-badge ${c.riskClass}">${c.risk}</span>
          </div>
          <div class="col-status">
            <span class="road-status-msg">${c.status}</span>
          </div>
          <div class="col-arrow">
            <span class="arrow-symbol ${c.arrowClass}">&rarr;</span>
          </div>
        </div>
      `;
    }).join('');

    // Re-attach clicks
    listWrap.querySelectorAll('.corridor-watch-row').forEach(row => {
      row.addEventListener('click', function () {
        const sId = row.getAttribute('data-sensor');
        if (sId) {
          state.selectedSensorId = sId;
          loadSensorTrajectory(sId);
          showToast(`Inspecting telemetry for ${row.querySelector('.road-title').textContent}.`);
        }
      });
    });
  }

  function setupOverviewInteractions() {
    // Map horizon selector
    const mapHorizonSelect = document.getElementById('overview-map-horizon-select');
    if (mapHorizonSelect) {
      mapHorizonSelect.addEventListener('change', function () {
        const horizon = parseInt(mapHorizonSelect.value, 10);
        state.activeHorizon = horizon;
        if (state.currentTrajectoryData) {
          updateMapTooltip(state.currentTrajectoryData);
        }
        showToast(`Map displaying ${horizon === 0 ? 'current conditions' : `forecast for +${horizon} min`}`);
      });
    }

    // Interactive sensor nodes on SVG map
    document.querySelectorAll('.interactive-map-node').forEach(node => {
      node.style.cursor = 'pointer';
      node.addEventListener('click', function () {
        const sensorId = node.getAttribute('data-sensor');
        if (sensorId) {
          state.selectedSensorId = sensorId;
          loadSensorTrajectory(sensorId);
          showToast(`Sensor ${sensorId} selected.`);
        }
      });
    });

    // Highway line clicks
    const i10Path = document.getElementById('map-hwy-i10');
    const i405Path = document.getElementById('map-hwy-i405-south');
    const bottleneckPath = document.getElementById('map-hwy-bottleneck');

    if (i10Path) {
      i10Path.style.cursor = 'pointer';
      i10Path.addEventListener('click', () => {
        state.selectedSensorId = '7100075';
        loadSensorTrajectory('7100075');
        highlightMapCorridor('I-10');
      });
    }
    if (i405Path || bottleneckPath) {
      const clickHandler = () => {
        state.selectedSensorId = '7100020';
        loadSensorTrajectory('7100020');
        highlightMapCorridor('I-405');
      };
      if (i405Path) { i405Path.style.cursor = 'pointer'; i405Path.addEventListener('click', clickHandler); }
      if (bottleneckPath) { bottleneckPath.style.cursor = 'pointer'; bottleneckPath.addEventListener('click', clickHandler); }
    }
  }

  function highlightMapCorridor(hwyName) {
    const tooltip = document.getElementById('map-tooltip-box');
    if (!tooltip) return;

    if (hwyName.includes('405')) {
      tooltip.setAttribute('transform', 'translate(370, 115)');
    } else if (hwyName.includes('10')) {
      tooltip.setAttribute('transform', 'translate(210, 160)');
    }
  }

  function updateMapTooltip(data) {
    const tooltip = document.getElementById('map-tooltip-box');
    if (!tooltip) return;

    const hwyText = tooltip.querySelector('text:first-of-type');
    const valText = tooltip.querySelector('text:last-of-type');

    let targetSpeed = data.current_speed_mph;
    let targetCong = data.current_congestion;

    if (state.activeHorizon === 15) {
      targetSpeed = data.forecast_15m;
      targetCong = data.forecast_congestion_15m;
    } else if (state.activeHorizon === 30) {
      targetSpeed = data.forecast_30m;
      targetCong = data.forecast_congestion_30m;
    } else if (state.activeHorizon === 60) {
      targetSpeed = data.forecast_60m;
      targetCong = data.forecast_congestion_60m;
    }

    const displaySpeed = state.unit === 'kmh' ? Math.round(targetSpeed * 1.60934) : Math.round(targetSpeed);
    const congColor = targetCong === 'Severe' ? '#ef4444' : (targetCong === 'High' ? '#fb923c' : (targetCong === 'Moderate' ? '#f59e0b' : '#7ea872'));

    if (hwyText) hwyText.textContent = `${data.highway} NORTH`;
    if (valText) {
      valText.innerHTML = `<tspan fill="#ffffff">${displaySpeed} ${state.unit}</tspan> <tspan fill="#64748b">&ndash;</tspan> <tspan fill="${congColor}">${targetCong.toUpperCase()}</tspan>`;
    }
  }

  function loadSensorTrajectory(sensorId) {
    const model = state.selectedModel || 'historical_average';
    fetch(`/api/sensor/${sensorId}/trajectory?model_name=${model}&timestamp_idx=${state.selectedTimestampIdx}`)
      .then(r => r.json())
      .then(data => {
        state.currentTrajectoryData = data;
        updateExploreCards(data);
      })
      .catch(err => console.warn('Trajectory fetch warning:', err));
  }

  function updateSpeedForecastCurve(data) {
    const predictedCurve = document.getElementById('speed-predicted-curve');
    const baselineCurve = document.getElementById('speed-baseline-curve');
    if (!predictedCurve) return;

    // Map speeds to SVG coordinates (viewBox 0 0 340 180)
    // Speed 20 -> y=150, Speed 65 -> y=20
    function speedToY(s) {
      const clamped = Math.max(15, Math.min(70, s));
      const fraction = (clamped - 15) / (70 - 15);
      return 155 - fraction * 135;
    }

    const yNow = speedToY(data.current_speed_mph);
    const y15 = speedToY(data.forecast_15m);
    const y30 = speedToY(data.forecast_30m);
    const y60 = speedToY(data.forecast_60m);

    // Smooth bezier curve for predicted path
    const predictedD = `M 35 ${yNow} C 80 ${yNow - 5}, 100 ${y15 + 8}, 135 ${y15} C 175 ${y15 - 8}, 200 ${y30 + 10}, 235 ${y30} C 275 ${y30 - 10}, 295 ${y60 + 5}, 325 ${y60}`;
    predictedCurve.setAttribute('d', predictedD);

    if (baselineCurve) {
      const baselineAvg = 45;
      const bY = speedToY(baselineAvg);
      const baselineD = `M 30 ${bY + 10} C 90 ${bY + 5}, 180 ${bY - 2}, 330 ${bY - 8}`;
      baselineCurve.setAttribute('d', baselineD);
    }
  }

  // =========================================================================
  // 5. FORECAST EXPLORER
  // =========================================================================
  function fetchSensors() {
    fetch('/sensors?limit=207')
      .then(r => r.json())
      .then(data => {
        state.sensorsList = data;
        populateSensorDropdown();
      })
      .catch(err => console.warn('Sensors fetch warning:', err));
  }

  function populateSensorDropdown() {
    const sensorSelect = document.getElementById('explore-sensor-select');
    if (!sensorSelect) return;
    sensorSelect.innerHTML = '';

    const selectedHwy = document.getElementById('explore-hwy-select')?.value || 'all';

    state.sensorsList.forEach(s => {
      if (selectedHwy === 'all' || s.highway === selectedHwy) {
        const opt = document.createElement('option');
        opt.value = s.sensor_id;
        opt.textContent = `${s.sensor_id} · ${s.highway} (${s.direction}B) · Speed limit: ${s.speed_limit} mph`;
        sensorSelect.appendChild(opt);
      }
    });

    if (sensorSelect.options.length > 0) {
      sensorSelect.value = state.selectedSensorId;
    }
  }

  function setupExploreView() {
    const hwySelect = document.getElementById('explore-hwy-select');
    const sensorSelect = document.getElementById('explore-sensor-select');
    const runBtn = document.getElementById('btn-run-explore');
    const dateSelect = document.getElementById('explore-date-select');
    const modelSelect = document.getElementById('explore-model-select');

    if (hwySelect) {
      hwySelect.addEventListener('change', () => {
        populateSensorDropdown();
        if (sensorSelect && sensorSelect.value) {
          state.selectedSensorId = sensorSelect.value;
          loadSensorTrajectory(state.selectedSensorId);
        }
      });
    }

    [sensorSelect, dateSelect, modelSelect].forEach(el => {
      if (el) {
        el.addEventListener('change', () => {
          if (sensorSelect && sensorSelect.value) state.selectedSensorId = sensorSelect.value;
          if (modelSelect) state.selectedModel = modelSelect.value;
          if (dateSelect) state.selectedTimestampIdx = parseInt(dateSelect.value, 10);
          loadSensorTrajectory(state.selectedSensorId);
        });
      }
    });

    if (runBtn) {
      runBtn.addEventListener('click', function () {
        if (sensorSelect && sensorSelect.value) {
          state.selectedSensorId = sensorSelect.value;
        }
        if (modelSelect) {
          state.selectedModel = modelSelect.value;
        }
        if (dateSelect) {
          state.selectedTimestampIdx = parseInt(dateSelect.value, 10);
        }

        runBtn.textContent = 'Computing estimates...';
        runBtn.disabled = true;

        setTimeout(() => {
          loadSensorTrajectory(state.selectedSensorId);
          runBtn.textContent = '⚡ Run Prediction Engine';
          runBtn.disabled = false;
          showToast('Multi-horizon trajectory updated successfully.');
        }, 200);
      });
    }
  }

  function updateExploreCards(data) {
    const curSpeed = document.getElementById('exp-current-speed');
    const curCong = document.getElementById('exp-current-cong');
    const p15Speed = document.getElementById('exp-p15-speed');
    const p15Cong = document.getElementById('exp-p15-cong');
    const p30Speed = document.getElementById('exp-p30-speed');
    const p30Cong = document.getElementById('exp-p30-cong');
    const p60Speed = document.getElementById('exp-p60-speed');
    const p60Cong = document.getElementById('exp-p60-cong');

    const toUnit = (val) => state.unit === 'kmh' ? Math.round(val * 1.60934) : Math.round(val * 10) / 10;

    if (curSpeed) curSpeed.textContent = toUnit(data.current_speed_mph);
    if (curCong) {
      curCong.textContent = data.current_congestion;
      curCong.className = `risk-badge ${getBadgeClass(data.current_congestion)}`;
    }

    if (p15Speed) p15Speed.textContent = toUnit(data.forecast_15m);
    if (p15Cong) {
      p15Cong.textContent = data.forecast_congestion_15m;
      p15Cong.className = `risk-badge ${getBadgeClass(data.forecast_congestion_15m)}`;
    }

    if (p30Speed) p30Speed.textContent = toUnit(data.forecast_30m);
    if (p30Cong) {
      p30Cong.textContent = data.forecast_congestion_30m;
      p30Cong.className = `risk-badge ${getBadgeClass(data.forecast_congestion_30m)}`;
    }

    if (p60Speed) p60Speed.textContent = toUnit(data.forecast_60m);
    if (p60Cong) {
      p60Cong.textContent = data.forecast_congestion_60m;
      p60Cong.className = `risk-badge ${getBadgeClass(data.forecast_congestion_60m)}`;
    }
  }

  function getBadgeClass(cong) {
    if (cong === 'Severe') return 'badge-severe';
    if (cong === 'High') return 'badge-high';
    if (cong === 'Moderate') return 'badge-moderate';
    return 'badge-low';
  }

  // =========================================================================
  // 6. COMPARE VIEW (DEPARTURE WINDOWS)
  // =========================================================================
  function setupCompareView() {
    // Automatically runs when entering compare view
  }

  function runDepartureComparison() {
    fetch('/api/compare/departure-times', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sensor_id: state.selectedSensorId,
        model_name: 'historical_average',
        departure_offsets: [0, 15, 30, 45, 60]
      })
    })
      .then(r => r.json())
      .then(data => {
        state.departureData = data;
        renderDepartureSlotsTable(data);
      })
      .catch(err => console.warn('Departure comparison warning:', err));
  }

  function renderDepartureSlotsTable(data) {
    const tbody = document.getElementById('departure-slots-tbody');
    if (!tbody || !data || !data.slots) return;

    tbody.innerHTML = '';
    data.slots.forEach(slot => {
      const tr = document.createElement('tr');
      const isBest = slot.departure_label === data.best_departure_label;
      const sp = state.unit === 'kmh' ? `${slot.predicted_speed_kmh} km/h` : `${slot.predicted_speed_mph} mph`;

      tr.innerHTML = `
        <td><strong>${slot.departure_label}</strong> ${isBest ? '<span class="risk-badge badge-low" style="margin-left: 8px;">RECOMMENDED</span>' : ''}</td>
        <td>${slot.clock_time}</td>
        <td><strong>${sp}</strong></td>
        <td><span class="risk-badge ${getBadgeClass(slot.congestion_class)}">${slot.congestion_class}</span></td>
        <td><strong>${slot.estimated_travel_time_min} min</strong></td>
        <td><span style="color:${slot.delay_min > 3 ? '#ef4444' : '#7ea872'}; font-weight:700;">+${slot.delay_min} min delay</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // =========================================================================
  // 7. CSV UPLOAD & TELEMETRY
  // =========================================================================
  function setupCsvUpload() {
    const dropzone = document.getElementById('csv-dropzone');
    const fileInput = document.getElementById('csv-file-input');
    const browseBtn = document.getElementById('btn-browse-csv');

    if (browseBtn && fileInput) {
      browseBtn.addEventListener('click', () => fileInput.click());
    }

    if (dropzone && fileInput) {
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#b8cca6';
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = '#1e2821';
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#1e2821';
        if (e.dataTransfer.files.length > 0) {
          uploadCsvFile(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
          uploadCsvFile(fileInput.files[0]);
        }
      });
    }
  }

  function uploadCsvFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const resultsArea = document.getElementById('csv-validation-results');
    const headerText = document.getElementById('val-header-text');
    const summaryText = document.getElementById('val-summary-text');

    if (resultsArea) resultsArea.style.display = 'block';
    if (summaryText) summaryText.textContent = 'Analyzing and validating CSV schema against METR-LA standard...';

    fetch('/api/upload-csv', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData
    })
      .then(r => r.json())
      .then(res => {
        if (headerText) headerText.textContent = res.is_valid ? '✓ CSV Validation Passed' : '✕ CSV Validation Notice';
        if (summaryText) summaryText.textContent = res.summary_message;
        if (res.is_valid && res.overview_metrics && res.overview_metrics.is_custom_dataset) {
          applyOverviewMetrics(res.overview_metrics);
          datasetLoaded = false;
        }
        showToast(res.is_valid ? `CSV validated & Overview updated with ${file.name}!` : 'CSV uploaded with warnings.');
      })
      .catch(err => {
        if (summaryText) summaryText.textContent = `Upload error: ${err}`;
      });
  }

  // =========================================================================
  // MACHINE LEARNING PLATFORM MODULES (7 EXTENSIONS)
  // =========================================================================
  let datasetLoaded = false;
  let evalLoaded = false;
  let comparisonLoaded = false;
  let analyticsLoaded = false;

  function setupMLPlatform() {
    setupMLCongestion();
    setupMLSpeed();
    setupMLForecast();
    setupMLDataset();
    setupMLEvaluation();
    setupMLComparison();
    setupMLAnalytics();
  }

  // --- MODULE 1: CONGESTION PREDICTION (MLP) ---
  let congDebounceTimer = null;

  async function runCongestionPrediction(showToastFlag = false, saveHistory = false) {
    const btn = document.getElementById('btn-predict-congestion');
    const statusText = document.getElementById('cong-status-text');
    const panel = document.getElementById('cong-result-panel');

    const roadSegEl = document.getElementById('cong-road-segment');
    const volEl = document.getElementById('cong-volume');
    const occEl = document.getElementById('cong-occupancy');
    const timeEl = document.getElementById('cong-time');
    const dayEl = document.getElementById('cong-day');
    const tempEl = document.getElementById('cong-temp');
    const rainEl = document.getElementById('cong-rain');

    if (!volEl || !occEl) return;

    const payload = {
      road_segment: roadSegEl ? roadSegEl.value : 'I-405 North · Sepulveda Pass',
      traffic_volume: parseFloat(volEl.value) || 2400,
      occupancy: parseFloat(occEl.value) || 45.0,
      time_of_day: timeEl ? timeEl.value : '08:30',
      day_of_week: dayEl ? dayEl.value : 'Monday',
      temperature: tempEl ? (parseFloat(tempEl.value) || 22.0) : 22.0,
      rainfall: rainEl ? (parseFloat(rainEl.value) || 0.0) : 0.0,
      save_history: saveHistory
    };

    if (saveHistory) {
      persistUserState({ congestion_inputs: payload });
    }

    if (btn && showToastFlag) btn.disabled = true;
    if (statusText) statusText.textContent = 'Computing exact MLP inference...';

    try {
      const resp = await fetch('/api/predict/congestion', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Prediction failed');

      const catEl = document.getElementById('res-cong-category');
      if (catEl) {
        catEl.textContent = data.predicted_congestion;
        catEl.className = 'risk-badge ' + (
          data.predicted_congestion === 'LOW' ? 'badge-low' :
          data.predicted_congestion === 'MODERATE' ? 'badge-moderate' :
          data.predicted_congestion === 'HIGH' ? 'badge-high' : 'badge-severe'
        );
      }

      const confEl = document.getElementById('res-cong-confidence');
      if (confEl) confEl.textContent = data.confidence;

      const spdEst = document.getElementById('res-cong-speed-est');
      if (spdEst && data.predicted_speed_mph) {
        spdEst.textContent = `Est. speed: ${data.predicted_speed_mph} mph (${data.predicted_speed_kmh} km/h)`;
      }

      const segEl = document.getElementById('res-cong-segment');
      if (segEl) segEl.textContent = data.road_segment;

      const tsEl = document.getElementById('res-cong-timestamp');
      if (tsEl) tsEl.textContent = data.timestamp;

      if (data.probabilities) {
        for (const [cls, val] of Object.entries(data.probabilities)) {
          const lcls = cls.toLowerCase();
          const valEl = document.getElementById(`prob-val-${lcls}`);
          const barEl = document.getElementById(`prob-bar-${lcls}`);
          if (valEl) valEl.textContent = `${val}%`;
          if (barEl) barEl.style.width = `${val}%`;
        }
      }

      const sumVol = document.getElementById('sum-vol');
      if (sumVol) sumVol.textContent = `${payload.traffic_volume} veh/h`;
      const sumOcc = document.getElementById('sum-occ');
      if (sumOcc) sumOcc.textContent = `${payload.occupancy}%`;
      const sumTime = document.getElementById('sum-time');
      if (sumTime) sumTime.textContent = payload.time_of_day;
      const sumDay = document.getElementById('sum-day');
      if (sumDay) sumDay.textContent = payload.day_of_week;
      const sumWeather = document.getElementById('sum-weather');
      if (sumWeather) sumWeather.textContent = `${payload.temperature}°C, ${payload.rainfall}mm rain`;

      if (panel) panel.style.display = 'block';
      if (showToastFlag) showToast(`MLP Predicted Congestion: ${data.predicted_congestion}`);
    } catch (err) {
      if (showToastFlag) showToast(`Error: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
      if (statusText) statusText.textContent = '';
    }
  }

  function setupMLCongestion() {
    const form = document.getElementById('form-predict-congestion');
    if (!form) return;

    // Listen to changes on ANY input/select in the form for instant real-time predictions
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(el => {
      el.addEventListener('input', () => {
        clearTimeout(congDebounceTimer);
        congDebounceTimer = setTimeout(() => runCongestionPrediction(false, true), 150);
      });
      el.addEventListener('change', () => {
        clearTimeout(congDebounceTimer);
        congDebounceTimer = setTimeout(() => runCongestionPrediction(false, true), 60);
      });
    });

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      runCongestionPrediction(true, true);
    });

    // Run initial prediction so panel is immediately visible (without duplicating history on reload)
    setTimeout(() => runCongestionPrediction(false, false), 80);
  }

  // --- MODULE 2: SPEED PREDICTION (MLP REGRESSION) ---
  let spdDebounceTimer = null;

  async function runSpeedPrediction(showToastFlag = false, saveHistory = false) {
    const btn = document.getElementById('btn-predict-speed');
    const statusText = document.getElementById('spd-status-text');
    const panel = document.getElementById('spd-result-panel');

    const roadSegEl = document.getElementById('spd-road-segment');
    const volEl = document.getElementById('spd-volume');
    const occEl = document.getElementById('spd-occupancy');
    const timeEl = document.getElementById('spd-time');
    const dayEl = document.getElementById('spd-day');
    const tempEl = document.getElementById('spd-temp');
    const rainEl = document.getElementById('spd-rain');

    if (!volEl || !occEl) return;

    const payload = {
      road_segment: roadSegEl ? roadSegEl.value : 'I-405 North · Sepulveda Pass',
      traffic_volume: parseFloat(volEl.value) || 2300,
      occupancy: parseFloat(occEl.value) || 38.0,
      time_of_day: timeEl ? timeEl.value : '09:00',
      day_of_week: dayEl ? dayEl.value : 'Monday',
      temperature: tempEl ? (parseFloat(tempEl.value) || 21.0) : 21.0,
      rainfall: rainEl ? (parseFloat(rainEl.value) || 0.0) : 0.0,
      save_history: saveHistory
    };

    if (saveHistory) {
      persistUserState({ speed_inputs: payload });
    }

    if (btn && showToastFlag) btn.disabled = true;
    if (statusText) statusText.textContent = 'Calculating speed regression...';

    try {
      const resp = await fetch('/api/predict/speed', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Speed prediction failed');

      const spdKmhEl = document.getElementById('res-spd-kmh');
      if (spdKmhEl) spdKmhEl.textContent = data.predicted_speed_kmh;

      const spdMphEl = document.getElementById('res-spd-mph-sub');
      if (spdMphEl) spdMphEl.textContent = `${data.predicted_speed_mph} mph`;

      const baseEl = document.getElementById('res-spd-baseline');
      if (baseEl) baseEl.textContent = data.baseline_speed_kmh;

      const deltaEl = document.getElementById('res-spd-delta');
      if (deltaEl) {
        const sign = data.delta_kmh >= 0 ? '+' : '';
        deltaEl.textContent = `${sign}${data.delta_kmh} km/h vs. baseline`;
        deltaEl.style.color = data.delta_kmh >= 0 ? 'var(--traffic-free)' : 'var(--traffic-high)';
      }

      const congBadge = document.getElementById('res-spd-congestion-badge');
      if (congBadge) {
        congBadge.textContent = data.congestion_level;
        congBadge.className = 'risk-badge ' + (
          data.congestion_level === 'LOW' ? 'badge-low' :
          data.congestion_level === 'MODERATE' ? 'badge-moderate' :
          data.congestion_level === 'HIGH' ? 'badge-high' : 'badge-severe'
        );
      }

      const tbody = document.getElementById('spd-trend-tbody');
      if (tbody && data.speed_trend) {
        tbody.innerHTML = data.speed_trend.map(t => `
          <tr>
            <td style="font-weight:700; color:var(--text-white);">${t.time_offset}</td>
            <td style="color:var(--sage-accent); font-weight:700;">${t.predicted_kmh} km/h</td>
            <td>${t.predicted_mph} mph</td>
            <td>${t.baseline_kmh} km/h</td>
            <td>
              <span style="color:${t.predicted_kmh >= t.baseline_kmh ? 'var(--traffic-free)' : 'var(--traffic-high)'}; font-weight:600;">
                ${t.predicted_kmh >= t.baseline_kmh ? '▲ Normal Flow' : '▼ Delay Expected'}
              </span>
            </td>
          </tr>
        `).join('');
      }

      if (panel) panel.style.display = 'block';
      if (showToastFlag) showToast(`Predicted Speed: ${data.predicted_speed_kmh} km/h (${data.predicted_speed_mph} mph)`);
    } catch (err) {
      if (showToastFlag) showToast(`Error: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
      if (statusText) statusText.textContent = '';
    }
  }

  function setupMLSpeed() {
    const form = document.getElementById('form-predict-speed');
    if (!form) return;

    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(el => {
      el.addEventListener('input', () => {
        clearTimeout(spdDebounceTimer);
        spdDebounceTimer = setTimeout(() => runSpeedPrediction(false, true), 150);
      });
      el.addEventListener('change', () => {
        clearTimeout(spdDebounceTimer);
        spdDebounceTimer = setTimeout(() => runSpeedPrediction(false, true), 60);
      });
    });

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      runSpeedPrediction(true, true);
    });

    setTimeout(() => runSpeedPrediction(false, false), 100);
  }

  // --- MODULE 3: PEAK-HOUR FORECASTING ---
  let fcDebounceTimer = null;

  async function runForecastPrediction(showToastFlag = false, saveHistory = false) {
    const btn = document.getElementById('btn-run-forecast');
    const statusText = document.getElementById('fc-status-text');
    const panel = document.getElementById('fc-result-panel');

    const segEl = document.getElementById('fc-segment');
    const volEl = document.getElementById('fc-vol');
    const occEl = document.getElementById('fc-occ');
    const spdEl = document.getElementById('fc-spd');
    const hourEl = document.getElementById('fc-hour');
    const rainEl = document.getElementById('fc-rain');

    if (!volEl || !occEl) return;

    const payload = {
      road_segment: segEl ? segEl.value : 'I-405 North · Sepulveda Pass',
      current_volume: parseFloat(volEl.value) || 2800,
      current_occupancy: parseFloat(occEl.value) || 52.0,
      current_speed: spdEl ? (parseFloat(spdEl.value) || 32.0) : 32.0,
      hour: hourEl ? (parseInt(hourEl.value, 10) || 8) : 8,
      day_of_week: 0,
      temperature: 24.0,
      rainfall: rainEl ? (parseFloat(rainEl.value) || 0.0) : 0.0,
      save_history: saveHistory
    };

    if (saveHistory) {
      persistUserState({ forecast_inputs: payload });
    }

    if (btn && showToastFlag) btn.disabled = true;
    if (statusText) statusText.textContent = 'Forecasting time horizons...';

    try {
      const resp = await fetch('/api/predict/forecast', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Forecast failed');

      const peakStatusEl = document.getElementById('fc-peak-status');
      if (peakStatusEl) peakStatusEl.textContent = data.peak_status;

      const intBadge = document.getElementById('fc-peak-intensity-badge');
      if (intBadge) {
        intBadge.textContent = `${data.peak_intensity} INTENSITY`;
        intBadge.className = 'risk-badge ' + (data.peak_intensity === 'HIGH' ? 'badge-high' : 'badge-low');
      }

      const grid = document.getElementById('fc-cards-grid');
      if (grid && data.horizons) {
        grid.innerHTML = data.horizons.map(h => `
          <div class="structura-card metric-card">
            <div class="card-inner-pad">
              <span class="metric-caption-title">HORIZON ${h.horizon}</span>
              <div class="metric-stat-row">
                <span class="big-metric-num">${h.predicted_speed_mph}</span>
                <span class="metric-unit-text">mph</span>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
                <span class="risk-badge badge-${h.congestion_level.toLowerCase()}">${h.congestion_level}</span>
                <span style="font-size:11px; color:var(--text-secondary);">${h.predicted_volume} veh/h</span>
              </div>
            </div>
          </div>
        `).join('');
      }

      const tbody = document.getElementById('fc-table-tbody');
      if (tbody && data.horizons) {
        tbody.innerHTML = data.horizons.map(h => `
          <tr>
            <td style="font-weight:700; color:var(--text-white);">${h.horizon} (${h.minutes_ahead}m ahead)</td>
            <td style="color:var(--text-primary); font-weight:600;">${h.predicted_volume} veh/hr</td>
            <td style="color:var(--sage-accent); font-weight:700;">${h.predicted_speed_mph} mph (${h.predicted_speed_kmh} km/h)</td>
            <td>${h.baseline_speed_mph} mph</td>
            <td><span class="risk-badge badge-${h.congestion_level.toLowerCase()}">${h.congestion_level}</span></td>
            <td>${h.confidence}%</td>
          </tr>
        `).join('');
      }

      if (panel) panel.style.display = 'block';
      if (showToastFlag) showToast('Peak-hour forecast generated successfully.');
    } catch (err) {
      if (showToastFlag) showToast(`Error: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
      if (statusText) statusText.textContent = '';
    }
  }

  function setupMLForecast() {
    const form = document.getElementById('form-peak-forecast');
    if (!form) return;

    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(el => {
      el.addEventListener('input', () => {
        clearTimeout(fcDebounceTimer);
        fcDebounceTimer = setTimeout(() => runForecastPrediction(false, true), 150);
      });
      el.addEventListener('change', () => {
        clearTimeout(fcDebounceTimer);
        fcDebounceTimer = setTimeout(() => runForecastPrediction(false, true), 60);
      });
    });

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      runForecastPrediction(true, true);
    });

    setTimeout(() => runForecastPrediction(false, false), 120);
  }

  // --- MODULE 4: DATASET EXPLORER & PREPROCESSING ---
  function setupMLDataset() {
    const btnBrowse = document.getElementById('btn-browse-dataset');
    const fileInput = document.getElementById('dataset-file-input');
    const dropzone = document.getElementById('dataset-dropzone');
    const btnPreprocess = document.getElementById('btn-apply-preprocessing');

    if (btnBrowse && fileInput) {
      btnBrowse.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) uploadDatasetFile(fileInput.files[0]);
      });
    }

    if (dropzone) {
      dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--sage-accent)'; });
      dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = '#1e2821'; });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#1e2821';
        if (e.dataTransfer.files.length > 0) uploadDatasetFile(e.dataTransfer.files[0]);
      });
    }

    if (btnPreprocess) {
      btnPreprocess.addEventListener('click', async () => {
        const msg = document.getElementById('prep-result-msg');
        if (msg) msg.textContent = 'Executing preprocessing pipeline...';
        try {
          const resp = await fetch('/api/dataset/preprocess', {
            method: 'POST',
            headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({
              missing_strategy: document.getElementById('prep-missing').value,
              normalize: true,
              encode_cats: true,
              test_ratio: 0.15,
              val_ratio: 0.15
            })
          });
          const res = await resp.json();
          if (msg) {
            const s = res.split_summary;
            msg.textContent = `✓ Preprocessing complete: Train: ${s.train_rows} (${s.train_percentage}%), Val: ${s.validation_rows} (${s.val_percentage}%), Test: ${s.test_rows} (${s.test_percentage}%). Imputed ${res.missing_values_handled} nulls.`;
          }
          showToast('Dataset preprocessing applied successfully!');
        } catch (e) {
          if (msg) msg.textContent = 'Preprocessing failed: ' + e;
        }
      });
    }
  }

  async function loadDatasetSummary() {
    try {
      const resp = await fetch('/api/dataset/summary', { headers: getAuthHeaders() });
      if (!resp.ok) return;
      const data = await resp.json();

      document.getElementById('ds-total-rows').textContent = data.total_rows.toLocaleString();
      document.getElementById('ds-total-cols').textContent = data.total_cols;
      document.getElementById('ds-missing-val').textContent = data.total_missing;
      document.getElementById('ds-duplicates').textContent = data.duplicate_count;

      const hContainer = document.getElementById('histograms-container');
      if (hContainer && data.histograms) {
        hContainer.innerHTML = Object.entries(data.histograms).map(([feat, h]) => {
          const maxCount = Math.max(...h.counts, 1);
          const bars = h.counts.map((c, i) => {
            const heightPct = Math.round((c / maxCount) * 100);
            return `<div style="flex:1; display:flex; flex-direction:column; align-items:center; height:100px; justify-content:flex-end;">
              <div style="width:85%; background:#7ea872; height:${heightPct}%; border-radius:3px 3px 0 0;" title="${h.bins[i]} - ${h.bins[i+1]}: ${c} records"></div>
            </div>`;
          }).join('');
          return `
            <div style="background:#0e1310; border:1px solid #161f18; border-radius:8px; padding:12px;">
              <div style="font-size:11.5px; font-weight:700; color:var(--text-white); margin-bottom:8px; text-transform:uppercase;">${feat.replace('_', ' ')}</div>
              <div style="display:flex; align-items:flex-end; gap:2px; height:100px; border-bottom:1px solid #1b261d;">${bars}</div>
              <div style="display:flex; justify-content:space-between; font-size:9.5px; color:var(--text-muted); margin-top:4px;">
                <span>${h.bins[0]}</span>
                <span>${h.bins[h.bins.length - 1]}</span>
              </div>
            </div>
          `;
        }).join('');
      }

      const corrContainer = document.getElementById('correlation-container');
      if (corrContainer && data.correlation_matrix && data.correlation_matrix.columns) {
        const cols = data.correlation_matrix.columns;
        const vals = data.correlation_matrix.values;
        let html = '<table class="corr-table"><thead><tr><th>Feature</th>';
        cols.forEach(c => html += `<th>${c.replace('_', ' ')}</th>`);
        html += '</tr></thead><tbody>';
        vals.forEach((row, rIdx) => {
          html += `<tr><td style="font-weight:700; text-align:left; color:var(--text-white);">${cols[rIdx].replace('_', ' ')}</td>`;
          row.forEach(cell => {
            const color = cell > 0.6 ? '#7ea872' : cell < -0.4 ? '#ef4444' : 'var(--text-secondary)';
            html += `<td style="color:${color}; font-weight:${Math.abs(cell) > 0.5 ? '700' : '400'};">${cell}</td>`;
          });
          html += '</tr>';
        });
        html += '</tbody></table>';
        corrContainer.innerHTML = html;
      }

      const prevTable = document.getElementById('dataset-preview-table');
      if (prevTable && data.preview && data.preview.length > 0) {
        const pCols = Object.keys(data.preview[0]);
        let pHead = '<thead><tr>' + pCols.map(c => `<th>${c.toUpperCase()}</th>`).join('') + '</tr></thead>';
        let pBody = '<tbody>' + data.preview.map(row => {
          return '<tr>' + pCols.map(c => `<td>${row[c] !== null ? row[c] : '--'}</td>`).join('') + '</tr>';
        }).join('') + '</tbody>';
        prevTable.innerHTML = pHead + pBody;
      }

      datasetLoaded = true;
    } catch (e) {
      console.error('Failed to load dataset summary', e);
    }
  }

  async function uploadDatasetFile(file) {
    const status = document.getElementById('dataset-upload-status');
    if (status) status.textContent = `Uploading ${file.name}...`;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const resp = await fetch('/api/dataset/upload', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: fd
      });
      const res = await resp.json();
      if (!resp.ok) throw new Error(res.detail || 'Upload failed');
      if (status) status.textContent = `✓ Uploaded ${file.name} — Dataset Explorer & Overview Dashboard updated!`;
      if (res.overview_metrics && res.overview_metrics.is_custom_dataset) {
        applyOverviewMetrics(res.overview_metrics);
      }
      showToast(`Overview & Dataset Explorer updated with ${file.name}`);
      loadDatasetSummary();
    } catch (e) {
      if (status) status.textContent = `Upload failed: ${e.message}`;
      showToast(`Upload failed: ${e.message}`);
    }
  }

  // --- MODULE 5: MLP MODEL EVALUATION ---
  function setupMLEvaluation() {}

  async function loadModelEvaluation() {
    try {
      const resp = await fetch('/api/model/evaluation');
      if (!resp.ok) return;
      const data = await resp.json();
      const c = data.classification || {};
      const r = data.regression || {};

      if (c.test_accuracy) document.getElementById('eval-test-acc').textContent = (c.test_accuracy * 100).toFixed(1);
      if (c.precision) document.getElementById('eval-precision').textContent = c.precision.toFixed(3);
      if (c.recall) document.getElementById('eval-recall').textContent = c.recall.toFixed(3);
      if (c.f1_score) document.getElementById('eval-f1').textContent = c.f1_score.toFixed(3);

      if (r.r2_score) document.getElementById('eval-r2').textContent = r.r2_score.toFixed(3);
      if (r.mae) document.getElementById('eval-mae').textContent = r.mae.toFixed(2);
      if (r.rmse) document.getElementById('eval-rmse').textContent = r.rmse.toFixed(2);
      if (r.mse) document.getElementById('eval-mse').textContent = r.mse.toFixed(2);

      const cmTable = document.getElementById('confusion-matrix-table');
      if (cmTable && c.confusion_matrix && c.classes) {
        const labels = c.classes;
        let html = '<thead><tr><th>Actual \\ Pred</th>';
        labels.forEach(l => html += `<th>${l}</th>`);
        html += '</tr></thead><tbody>';
        c.confusion_matrix.forEach((row, i) => {
          html += `<tr><td style="font-weight:700; color:var(--text-white);">${labels[i]}</td>`;
          row.forEach((cnt, j) => {
            const isDiag = (i === j);
            const bg = isDiag ? 'rgba(126, 168, 114, 0.28)' : (cnt > 0 ? 'rgba(239, 68, 68, 0.12)' : '#101512');
            const border = isDiag ? '1px solid rgba(126, 168, 114, 0.4)' : '1px solid #18221a';
            html += `<td class="matrix-cell" style="background:${bg}; border:${border}; color:${isDiag ? 'var(--sage-accent)' : 'var(--text-muted)'};">${cnt}</td>`;
          });
          html += '</tr>';
        });
        html += '</tbody>';
        cmTable.innerHTML = html;
      }

      const lossWrap = document.getElementById('loss-bars-wrap');
      if (lossWrap && c.loss_curve) {
        const maxLoss = Math.max(...c.loss_curve, 1);
        lossWrap.innerHTML = c.loss_curve.map((l, idx) => {
          const heightPct = Math.max(8, Math.round((l / maxLoss) * 100));
          return `<div style="flex:1; height:100%; display:flex; align-items:flex-end;">
            <div style="width:100%; background:linear-gradient(180deg, #b8cca6, #7ea872); height:${heightPct}%; border-radius:2px 2px 0 0;" title="Iteration ${idx+1}: Loss ${l.toFixed(4)}"></div>
          </div>`;
        }).join('');
      }

      evalLoaded = true;
    } catch (e) {
      console.error('Failed to load evaluation metrics', e);
    }
  }

  // --- MODULE 6: MODEL COMPARISON ---
  function setupMLComparison() {}

  async function loadModelComparison() {
    try {
      const resp = await fetch('/api/model/comparison');
      if (!resp.ok) return;
      const data = await resp.json();

      const tbody = document.getElementById('comparison-table-tbody');
      if (tbody && data.models) {
        tbody.innerHTML = data.models.map(m => {
          const isMLP = m.model_name === 'MLP';
          return `
            <tr style="${isMLP ? 'background:rgba(184, 204, 166, 0.05);' : ''}">
              <td style="font-weight:700; color:var(--text-white);">
                ${m.model_name}
                ${isMLP ? '<span class="model-winner-badge">FASTEST INFERENCE</span>' : ''}
                ${m.model_name === 'Random Forest' ? '<span class="model-winner-badge" style="color:#fbbf24; border-color:#fbbf24;">HIGHEST ACC</span>' : ''}
              </td>
              <td style="font-weight:700; color:${m.test_accuracy > 0.9 ? 'var(--sage-accent)' : 'var(--text-primary)'};">${(m.test_accuracy * 100).toFixed(1)}%</td>
              <td>${m.precision.toFixed(3)}</td>
              <td>${m.recall.toFixed(3)}</td>
              <td>${m.f1_score.toFixed(3)}</td>
              <td>${m.mae.toFixed(2)}</td>
              <td style="font-weight:700;">${m.r2_score.toFixed(3)}</td>
              <td>${m.training_time_sec.toFixed(2)}s</td>
              <td style="color:var(--sage-accent); font-weight:700;">${m.prediction_time_ms.toFixed(4)} ms</td>
            </tr>
          `;
        }).join('');
      }

      const accBars = document.getElementById('comp-acc-bars');
      if (accBars && data.models) {
        accBars.innerHTML = data.models.map(m => `
          <div>
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px;">
              <span style="font-weight:600; color:var(--text-white);">${m.model_name}</span>
              <span style="font-weight:700; color:var(--sage-accent);">${(m.test_accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-low" style="width:${(m.test_accuracy * 100).toFixed(1)}%;"></div></div>
          </div>
        `).join('');
      }

      const r2Bars = document.getElementById('comp-r2-bars');
      if (r2Bars && data.models) {
        r2Bars.innerHTML = data.models.map(m => `
          <div>
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:4px;">
              <span style="font-weight:600; color:var(--text-white);">${m.model_name}</span>
              <span style="font-weight:700; color:var(--sage-accent);">${(m.r2_score * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-low" style="width:${Math.max(10, (m.r2_score * 100)).toFixed(1)}%;"></div></div>
          </div>
        `).join('');
      }

      if (data.recommendation) {
        const rec = document.getElementById('comparison-rec-text');
        if (rec) rec.textContent = data.recommendation;
      }

      comparisonLoaded = true;
    } catch (e) {
      console.error('Failed to load model comparison', e);
    }
  }

  // --- MODULE 7: FEATURE IMPORTANCE & PREDICTION HISTORY ---
  function setupMLAnalytics() {
    const btnFilter = document.getElementById('btn-filter-history');
    if (btnFilter) {
      btnFilter.addEventListener('click', () => {
        const s = document.getElementById('history-search-input').value.trim();
        const c = document.getElementById('history-congestion-filter').value;
        loadPredictionHistoryTable(s, c);
      });
    }

    const searchInput = document.getElementById('history-search-input');
    if (searchInput) {
      searchInput.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') {
          const s = searchInput.value.trim();
          const c = document.getElementById('history-congestion-filter').value;
          loadPredictionHistoryTable(s, c);
        }
      });
    }
  }

  async function loadFeatureImportanceAndHistory() {
    try {
      const fResp = await fetch('/api/model/feature-importance');
      if (fResp.ok) {
        const fData = await fResp.json();
        const container = document.getElementById('feature-importance-bars-container');
        if (container && fData.features) {
          container.innerHTML = fData.features.map(f => `
            <div class="feat-imp-row">
              <span class="feat-name">${f.feature.replace('_', ' ').toUpperCase()}</span>
              <div>
                <div class="feat-bar-fill" style="width: ${Math.min(100, f.relative_percentage * 1.3)}%;"></div>
                <div style="font-size: 10.5px; color: var(--text-muted); margin-top: 4px;">${f.description}</div>
              </div>
              <span class="feat-pct">${f.relative_percentage}%</span>
            </div>
          `).join('');
        }
      }
    } catch (e) {
      console.error('Failed to load feature importance', e);
    }

    loadPredictionHistoryTable();
    analyticsLoaded = true;
  }

  async function loadPredictionHistoryTable(search = '', congestion = 'ALL') {
    try {
      let url = `/api/predictions/history?limit=30`;
      if (search) url += `&search=${encodeURIComponent(search)}`;
      if (congestion && congestion !== 'ALL') url += `&congestion=${encodeURIComponent(congestion)}`;

      const resp = await fetch(url, { headers: getAuthHeaders() });
      if (!resp.ok) return;
      const data = await resp.json();
      const tbody = document.getElementById('history-table-tbody');
      if (tbody && data.predictions) {
        if (data.predictions.length === 0) {
          tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-muted);">No predictions found matching current filter.</td></tr>`;
          return;
        }
        tbody.innerHTML = data.predictions.map(p => `
          <tr>
            <td style="font-family:monospace; color:var(--text-muted);">#${p.id}</td>
            <td style="font-size:11.5px; color:var(--text-secondary);">${p.timestamp || '--'}</td>
            <td style="font-weight:700; color:var(--text-white);">${p.road_segment}</td>
            <td style="font-size:11px; color:var(--text-secondary);">Vol: ${p.traffic_volume || '--'}, Occ: ${p.occupancy || '--'}%</td>
            <td>
              ${p.predicted_congestion ? `<span class="risk-badge badge-${p.predicted_congestion.toLowerCase()}">${p.predicted_congestion}</span>` : '--'}
            </td>
            <td style="color:var(--sage-accent); font-weight:700;">${p.predicted_speed ? `${p.predicted_speed} km/h` : '--'}</td>
            <td>${p.confidence ? `${p.confidence}%` : '--'}</td>
            <td style="font-size:11px; color:var(--text-muted);">${p.model_name || 'MLP'}</td>
          </tr>
        `).join('');
      }
    } catch (e) {
      console.error('Failed to load prediction history', e);
    }
  }

  // Toast Notification Utility
  function showToast(msg) {
    let toast = document.getElementById('app-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'app-toast';
      toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: #141b16;
        border: 1px solid #233026;
        color: #f2f6f3;
        font-size: 13px;
        font-weight: 500;
        padding: 12px 20px;
        border-radius: 10px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        z-index: 2000;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: opacity 0.25s ease, transform 0.25s ease;
      `;
      document.body.appendChild(toast);
    }
    toast.innerHTML = `<span style="color:#b8cca6; font-weight:bold;">✓</span> ${msg}`;
    toast.style.opacity = '1';
    toast.style.transform = 'translateY(0)';
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
    }, 3200);
  }

  // Launch on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
