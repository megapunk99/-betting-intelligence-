/* Betting Intelligence — Dashboard JS */

(function () {
  'use strict';

  // ─── Chart.js performance chart ───

  let _chartInstance = null;

  function initPerformanceChart() {
    // Destroy previous chart instance FIRST to prevent memory leak on DOM swaps,
    // even if the new DOM doesn't contain a chart (fewer than 2 data points)
    if (_chartInstance) {
      _chartInstance.destroy();
      _chartInstance = null;
    }

    const chartContainer = document.getElementById('chart-container');
    const canvas = document.getElementById('performance-chart');
    if (!canvas || !chartContainer) return;

    // Read profits from data attribute (server-side resolved bets, oldest first)
    let profits = [];
    try {
      var raw = chartContainer.getAttribute('data-profits');
      if (raw) {
        profits = JSON.parse(raw);
      }
    } catch (e) {
      console.warn('Failed to parse chart profits:', e);
    }

    // Guard against malformed data and filter to valid numbers
    if (!Array.isArray(profits)) profits = [];
    profits = profits.filter(function(p) { return typeof p === 'number' && !isNaN(p); });

    if (profits.length < 2) {
      chartContainer.style.display = 'none';
      return;
    }

    let cum = 0;
    const cumulative = profits.map(p => { cum += p; return cum; });
    const labels = profits.map((_, i) => `#${i + 1}`);

    _chartInstance = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Cumulative P&L',
          data: cumulative,
          borderColor: cumulative[cumulative.length - 1] >= 0 ? '#00e676' : '#ff5252',
          backgroundColor: (ctx) => {
            const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 240);
            gradient.addColorStop(0, cumulative[cumulative.length - 1] >= 0
              ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 82, 82, 0.15)');
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
            return gradient;
          },
          fill: true,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: cumulative[cumulative.length - 1] >= 0 ? '#00e676' : '#ff5252',
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#14142a',
            borderColor: 'rgba(255,255,255,0.08)',
            borderWidth: 1,
            titleColor: '#9494a8',
            bodyColor: '#e8e8f0',
            callbacks: {
              label: (ctx) => {
                const val = ctx.parsed.y;
                return (val >= 0 ? '+' : '') + '$' + val.toFixed(0);
              }
            }
          }
        },
        scales: {
          x: {
            display: true,
            grid: { display: false },
            ticks: { color: '#5c5c70', font: { size: 10 }, maxTicksLimit: 10 }
          },
          y: {
            display: true,
            grid: { color: 'rgba(255,255,255,0.03)' },
            ticks: {
              color: '#5c5c70', font: { size: 10 },
              callback: (val) => (val >= 0 ? '+' : '') + '$' + val.toFixed(0),
            }
          }
        },
        interaction: { intersect: false, mode: 'index' }
      }
    });
  }

  // ─── Animated counters ───

  function animateCounters() {
    document.querySelectorAll('[data-count]').forEach(el => {
      const target = parseInt(el.getAttribute('data-count'), 10);
      if (isNaN(target) || target === 0) {
        el.textContent = target;
        return;
      }
      const duration = 1000;
      const start = performance.now();

      function update(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(update);
      }
      requestAnimationFrame(update);
    });
  }

  // ─── Freshness timer ───

  let _freshnessInterval = null;

  function initFreshness() {
    const el = document.getElementById('gen-time');
    if (!el || !el.textContent) return;
    const dt = new Date(el.textContent);
    if (isNaN(dt.getTime())) return;

    // Clear any previous interval to prevent accumulation across DOM swaps
    if (_freshnessInterval) {
      clearInterval(_freshnessInterval);
    }

    function update() {
      const seconds = Math.floor((new Date() - dt) / 1000);
      let display;
      if (seconds < 60) display = seconds + 's ago';
      else if (seconds < 3600) display = Math.floor(seconds / 60) + 'm ago';
      else display = Math.floor(seconds / 3600) + 'h ago';
      el.textContent = display;
    }
    update();
    _freshnessInterval = setInterval(update, 10000);
  }

  // ─── WebSocket live connection ───

  function initWebSocket() {
    const indicator = document.getElementById('live-indicator');
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = wsProto + '//' + window.location.host + '/ws/live';
    let ws = null;
    let reconnectTimer = null;

    function connect() {
      if (ws && ws.readyState === WebSocket.OPEN) return;

      try {
        ws = new WebSocket(wsUrl);
      } catch (e) {
        scheduleReconnect();
        return;
      }

      ws.onopen = () => {
        let dot = document.getElementById('ws-dot');
        if (!dot) {
          dot = document.createElement('span');
          dot.id = 'ws-dot';
          dot.title = 'Live connection active';
          const nav = document.querySelector('.nav-links');
          if (nav) nav.appendChild(dot);
        }
        dot.style.display = 'inline-block';
        if (indicator) indicator.style.borderColor = 'rgba(0,230,118,0.3)';
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'predictions') {
            updateDashboardFromWS(data);
          }
        } catch (e) {}
      };

      ws.onclose = () => {
        const dot = document.getElementById('ws-dot');
        if (dot) dot.style.display = 'none';
        if (indicator) indicator.style.borderColor = '';
        scheduleReconnect();
      };

      ws.onerror = () => { ws.close(); };
    }

    function scheduleReconnect() {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 10000);
    }

    function updateDashboardFromWS(data) {
      const statValues = document.querySelectorAll('.stat-value');
      if (statValues.length >= 5) {
        statValues[0].textContent = data.n_bets;
        statValues[0].className = 'stat-value ' + (data.n_bets > 0 ? 'green' : 'dim');
        const subs = document.querySelectorAll('.stat-sub');
        if (subs.length > 0) subs[0].textContent = data.n_games + ' games';
        statValues[4].textContent = data.n_clear;
        statValues[4].className = 'stat-value ' + (data.n_clear > 0 ? 'green' : 'dim');
      }

      const genEl = document.getElementById('gen-time');
      if (genEl && data.generated_at) {
        genEl.textContent = data.generated_at.slice(0, 19);
      }

      if (indicator) {
        indicator.textContent = data.n_bets + ' bets';
        indicator.classList.toggle('has-data', data.n_bets > 0);
      }
    }

    setTimeout(connect, 500);
  }

  // ─── Keyboard shortcuts ───

  function initKeyboardNav() {
    document.addEventListener('keydown', e => {
      if (e.target.closest('input,textarea,button')) return;
      // Note: 'r' for reload removed — too easy to trigger accidentally
      if (e.key === 'p' && !e.ctrlKey && !e.metaKey) window.location.href = '/future-predictions';
      if (e.key === 'd' && !e.ctrlKey && !e.metaKey) window.location.href = '/';
    });
  }

  // ─── Future Predictions: client-side loading ───

  function _renderFutureCard(pred) {
    var direction = pred.direction || 'under';
    var edgePct = pred.edge_pct || 0;
    var edgeAbs = Math.abs(edgePct * 100);
    var barPct = Math.min(edgeAbs * 3, 100);
    var isBest = (pred.best_quarter_edge || 0) > 10.0;

    var quarters = ['q1', 'q2', 'q3', 'q4'];
    var halves = [['h1', '1st Half'], ['h2', '2nd Half']];

    var card = document.createElement('div');
    card.className = 'pred-card' + (isBest ? ' best-card' : '');
    card.style.animationDelay = '0s';

    var cardHeader = document.createElement('div');
    cardHeader.className = 'card-header';
    cardHeader.innerHTML =
      '<span class="card-league-badge ' + (pred.league || 'NBA').toLowerCase() + '">' + (pred.league || 'NBA') + '</span>' +
      '<span class="card-date">' + (pred.game_date || '') + '</span>' +
      '<span class="card-edge-badge ' + direction + '">' +
        '<span class="edge-arrow">' + (direction === 'over' ? '▲' : '▼') + '</span>' +
        '<span class="edge-value">' + (edgePct >= 0 ? '+' : '') + (edgePct * 100).toFixed(1) + '%</span>' +
      '</span>';
    card.appendChild(cardHeader);

    // Card body
    var cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    cardBody.innerHTML =
      '<div class="teams-section">' +
        '<div class="team-row away">' +
          '<span class="team-name">' + (pred.away_team || '') + '</span>' +
          '<span class="team-tag">' + (pred.away_team_short || '') + '</span>' +
        '</div>' +
        '<div class="matchup-vs">' +
          '<span class="vs-line"></span><span class="vs-text">@</span><span class="vs-line"></span>' +
        '</div>' +
        '<div class="team-row home">' +
          '<span class="team-name">' + (pred.home_team || '') + '</span>' +
          '<span class="team-tag">' + (pred.home_team_short || '') + '</span>' +
        '</div>' +
      '</div>' +
      '<div class="prediction-display">' +
        '<div class="prediction-main">' +
          '<span class="prediction-value">' + (pred.predicted_total || 0).toFixed(0) + '</span>' +
          '<span class="prediction-unit">pts</span>' +
        '</div>' +
        '<div class="prediction-market">Market: <strong>' + (pred.market_total || 0).toFixed(0) + '</strong></div>' +
        '<div class="prediction-bar">' +
          '<div class="pred-bar-track"><div class="pred-bar-fill ' + direction + '" style="width:' + barPct + '%;"></div></div>' +
          '<span class="pred-bar-label">' + barPct.toFixed(0) + '%</span>' +
        '</div>' +
      '</div>';
    card.appendChild(cardBody);

    // Quarters
    var qSection = document.createElement('div');
    qSection.className = 'quarters-section';
    var qHtml = '<div class="section-label">Quarter Projections</div><div class="quarters-grid">';
    quarters.forEach(function(q) {
      var qTotal = pred[q + '_total'] || 0;
      var qHome = pred[q + '_home'] || 0;
      var qAway = pred[q + '_away'] || 0;
      var qEdge = pred[q + '_edge'] || 0;
      var maxQ = 50;
      var homeH = Math.round((qHome / maxQ) * 70);
      var awayH = Math.round((qAway / maxQ) * 70);
      qHtml +=
        '<div class="quarter-cell">' +
          '<div class="quarter-label">' + q.toUpperCase() + '</div>' +
          '<div class="quarter-bars">' +
            '<div class="q-bar away" style="height:' + awayH + '%;"><span>' + qAway.toFixed(0) + '</span></div>' +
            '<div class="q-bar home" style="height:' + homeH + '%;"><span>' + qHome.toFixed(0) + '</span></div>' +
          '</div>' +
          '<div class="quarter-total">' + qTotal.toFixed(0) + '</div>' +
          '<div class="quarter-edge ' + (qEdge > 0 ? 'pos' : 'neg') + '">' + (qEdge >= 0 ? '+' : '') + (qEdge * 100).toFixed(1) + '%</div>' +
        '</div>';
    });
    qHtml += '</div>';
    qSection.innerHTML = qHtml;
    card.appendChild(qSection);

    // Halves
    var hSection = document.createElement('div');
    hSection.className = 'halves-section';
    var hHtml = '<div class="section-label">Half Projections</div><div class="halves-grid">';
    halves.forEach(function(h) {
      var key = h[0];
      var label = h[1];
      var hTotal = pred[key + '_total'] || 0;
      var hHome = pred[key + '_home'] || 0;
      var hAway = pred[key + '_away'] || 0;
      var hEdge = pred[key + '_edge'] || 0;
      hHtml +=
        '<div class="half-cell">' +
          '<div class="half-label">' + label + '</div>' +
          '<div class="half-detail-row">' +
            '<span class="half-team away">' + hAway.toFixed(0) + '</span>' +
            '<span class="half-dash">—</span>' +
            '<span class="half-team home">' + hHome.toFixed(0) + '</span>' +
          '</div>' +
          '<div class="half-total">' + hTotal.toFixed(0) + ' <span class="half-mkt">mkt ' + (pred[key + '_market'] || 0).toFixed(0) + '</span></div>' +
          '<div class="half-edge ' + (hEdge > 0 ? 'pos' : 'neg') + '">' + (hEdge >= 0 ? '+' : '') + (hEdge * 100).toFixed(1) + '%</div>' +
        '</div>';
    });
    hHtml += '</div>';
    hSection.innerHTML = hHtml;
    card.appendChild(hSection);

    // Footer
    var confidence = pred.confidence || 'low';
    var homeScore = pred.home_score || 0;
    var awayScore = pred.away_score || 0;
    var bestQ = pred.best_quarter || 'FULL';
    var bestDir = pred.best_quarter_direction || '';
    var bestEdge = pred.best_quarter_edge || 0;

    var footer = document.createElement('div');
    footer.className = 'card-footer';
    footer.innerHTML =
      '<div class="footer-item">' +
        '<span class="footer-label">Confidence</span>' +
        '<span class="footer-value conf-' + confidence + '">' + confidence.charAt(0).toUpperCase() + confidence.slice(1) + '</span>' +
      '</div>' +
      '<div class="footer-item">' +
        '<span class="footer-label">Score</span>' +
        '<span class="footer-value">' + homeScore.toFixed(0) + ' - ' + awayScore.toFixed(0) + '</span>' +
      '</div>' +
      '<div class="footer-item best-bet-item">' +
        '<span class="footer-label">Best Bet</span>' +
        '<span class="footer-value">' + bestQ + ' <span class="best-direction ' + bestDir + '">' + bestDir.toUpperCase() + '</span></span>' +
        '<span class="footer-sub">' + (bestEdge >= 0 ? '+' : '') + bestEdge.toFixed(1) + '%</span>' +
      '</div>';
    card.appendChild(footer);

    return card;
  }

  function loadFuturePredictions() {
    var skeleton = document.getElementById('future-skeleton');
    var container = document.getElementById('future-cards');
    var emptyState = document.getElementById('future-empty');
    var countEl = document.getElementById('future-count');
    var indicator = document.getElementById('future-indicator');
    var genTime = document.getElementById('gen-time');

    if (!container) return; // not on the future predictions page

    // Show skeleton
    if (skeleton) skeleton.classList.add('active');
    if (emptyState) emptyState.style.display = 'none';
    if (container) container.innerHTML = '';

    fetch('/api/future-predictions')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var preds = data.predictions || [];

        // Update hero stats & indicator
        if (countEl) {
          countEl.textContent = preds.length;
          countEl.setAttribute('data-count', preds.length);
        }
        if (indicator) {
          indicator.textContent = preds.length + ' games';
          indicator.classList.toggle('has-data', preds.length > 0);
        }
        if (genTime && data.generated_at) {
          genTime.textContent = data.generated_at.slice(0, 16);
          // Re-init freshness timer with the real timestamp (replaces the stale one from init())
          initFreshness();
        }

        // Hide skeleton
        if (skeleton) skeleton.classList.remove('active');

        if (preds.length === 0) {
          // Show empty state
          if (emptyState) emptyState.style.display = '';
          return;
        }

        // Render cards with staggered fade-in
        preds.forEach(function(pred, i) {
          var card = _renderFutureCard(pred);
          card.classList.add('fade-in');
          card.style.animationDelay = (i * 80) + 'ms';
          container.appendChild(card);
        });

        // Animate the counters
        if (typeof animateCounters === 'function') animateCounters();
      })
      .catch(function(err) {
        console.error('Failed to load future predictions:', err);
        if (skeleton) skeleton.classList.remove('active');
        // Show empty state without mutating its text (preserves original "No Predictions" message)
        if (emptyState) emptyState.style.display = '';
      });
  }

  window._loadFuturePredictions = loadFuturePredictions;
  window._renderFutureCard = _renderFutureCard;

  // ─── Theme toggle ───

  const STORAGE_KEY = 'betting-intel-theme';

  function getPreferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
    return 'dark';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
    const btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.classList.toggle('light', theme === 'light');
      btn.classList.toggle('dark', theme === 'dark');
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  }

  function initTheme() {
    const theme = getPreferredTheme();
    applyTheme(theme);

    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: light)');
      mq.addEventListener('change', (e) => {
        if (!localStorage.getItem(STORAGE_KEY)) {
          applyTheme(e.matches ? 'light' : 'dark');
        }
      });
    }

    const btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', toggleTheme);
  }

  // ─── Show/hide skeleton loader ───

  function _showSkeleton() {
    const skeleton = document.getElementById('skeleton-area');
    const emptyState = document.getElementById('empty-state');
    if (skeleton) skeleton.classList.add('active');
    if (emptyState) emptyState.style.display = 'none';
  }

  function _hideSkeleton() {
    const skeleton = document.getElementById('skeleton-area');
    const emptyState = document.getElementById('empty-state');
    if (skeleton) skeleton.classList.remove('active');
    if (emptyState) emptyState.style.display = '';  // restore default in case error occurred
  }

  // ─── Shared: swap <main> content without page reload ───
  // Exposed on window so the global resolvePredictions() can use it too.

  function _swapMainContent() {
    var controller = new AbortController();
    var timeoutId = setTimeout(function() { controller.abort(); }, 15000);

    return fetch('/', { signal: controller.signal })
      .then(function(r) { return r.text(); })
      .then(function(html) {
        clearTimeout(timeoutId);
        var parser = new DOMParser();
        var doc = parser.parseFromString(html, 'text/html');
        var newMain = doc.querySelector('main');
        var oldMain = document.querySelector('main');
        if (newMain && oldMain) {
          oldMain.replaceWith(newMain);
        }
        // Re-init components that were in the swapped <main>
        if (typeof Chart !== 'undefined') initPerformanceChart();
        animateCounters();
        initFreshness();
      })
      .catch(function(err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
          _showToast('Page load timed out', 'error', 3000);
        }
        throw err;
      });
  }

  window._swapMainContent = _swapMainContent;

  // ─── Toast notification (fixed outside <main> — survives DOM swaps) ───

  function _showToast(message, type, duration) {
    type = type || 'success';
    duration = duration || 4000;

    var root = document.getElementById('toast-root');
    if (!root) return;

    var toast = document.createElement('div');
    toast.className = 'toast-float ' + type;
    toast.textContent = message;
    root.appendChild(toast);

    // Auto-dismiss after duration
    setTimeout(function () {
      toast.classList.add('dismissing');
      // Remove from DOM after animation completes
      setTimeout(function () {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 300);
    }, duration);
  }

  window._showToast = _showToast;

  // ─── Refresh live data without page reload ───

  var REFRESH_TIMEOUT_MS = 50000;  // 50 seconds max wait for refresh
  var SKELETON_FALLBACK_MS = 55000; // Show empty state after 55s if nothing happens

  function refreshLiveData() {
    var btn = document.getElementById('refresh-btn');
    var refreshBtn2 = document.querySelector('.action-bar .btn-secondary');

    // Disable all refresh buttons
    [btn, refreshBtn2].forEach(function(b) {
      if (b) { b.disabled = true; b.textContent = 'Refreshing...'; }
    });

    // Show skeleton immediately so user sees activity
    _showSkeleton();

    // Safety timer: if refresh takes too long, hide skeleton and show empty state
    var skeletonTimer = setTimeout(function() {
      _hideSkeleton();
      _showToast('Refresh timed out. Try again or check API keys.', 'error', 6000);
      [btn, refreshBtn2].forEach(function(b) {
        if (b) { b.disabled = false; b.textContent = b === btn ? 'Refresh Now' : '\u21B5 Refresh'; }
      });
    }, SKELETON_FALLBACK_MS);

    // Create AbortController for timeout
    var controller = new AbortController();
    var timeoutId = setTimeout(function() { controller.abort(); }, REFRESH_TIMEOUT_MS);

    fetch('/api/live/refresh', {
      method: 'POST',
      signal: controller.signal,
    })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        clearTimeout(timeoutId);
        clearTimeout(skeletonTimer);

        if (data.n_total > 0) {
          // Fetch the full dashboard HTML and swap <main> content
          // instead of location.reload() to avoid the page flash
          _swapMainContent().catch(function() {
            // If HTML fetch fails after a successful API refresh, fall back
            // to a full page reload so the user still sees fresh data
            location.reload();
          });
        } else {
          _hideSkeleton();
          [btn, refreshBtn2].forEach(function(b) {
            if (b) { b.disabled = false; b.textContent = b === btn ? 'Refresh Now' : '\u21B5 Refresh'; }
          });

          if (data.error === 'timed_out') {
            _showToast('Odds sources timed out. Check API keys.', 'error', 5000);
          } else if (data.error) {
            _showToast('Error: ' + data.error, 'error', 5000);
          } else {
            _showToast('No games found. Try again later.', 'info', 4000);
          }
        }
      })
      .catch(function(err) {
        clearTimeout(timeoutId);
        clearTimeout(skeletonTimer);

        if (err.name === 'AbortError') {
          _showToast('Request timed out. Odds sources unreachable.', 'error', 6000);
        } else {
          console.error('Refresh failed:', err);
          _showToast('Connection failed. Server may be busy.', 'error', 5000);
        }

        _hideSkeleton();
        [btn, refreshBtn2].forEach(function(b) {
          if (b) { b.disabled = false; b.textContent = b === btn ? 'Refresh Now' : '\u21B5 Refresh'; }
        });
      });
  }

  window.refreshLiveData = refreshLiveData;

  // ─── Auto-refresh on first load if no data ───

  function autoRefreshIfEmpty() {
    var statValue = document.querySelector('.stat-value');
    if (!statValue) return;
    var nBets = parseInt(statValue.textContent || '0', 10);
    // Only auto-refresh if both live bets AND resolved bets are empty
    var nResolved = parseInt(document.querySelector('.stat-sub')?.textContent || '0', 10);
    if (nBets === 0 && nResolved === 0) {
      console.log('No data found — auto-refreshing from live engine...');
      refreshLiveData();
    }
  }

  // ─── Sport filter ───

  window.filterBySport = function(sport) {
    // Update active button
    document.querySelectorAll('.sport-filter-btn').forEach(function(btn) {
      btn.classList.toggle('active', btn.getAttribute('data-sport') === sport);
    });

    // Show/hide game cards
    var cards = document.querySelectorAll('.game-card');
    var visibleCount = 0;
    cards.forEach(function(card) {
      if (sport === 'all') {
        card.classList.remove('hidden-by-filter');
        visibleCount++;
      } else {
        var cardSport = card.getAttribute('data-sport');
        if (cardSport === sport) {
          card.classList.remove('hidden-by-filter');
          visibleCount++;
        } else {
          card.classList.add('hidden-by-filter');
        }
      }
    });

    // Update count badge
    var countEl = document.getElementById('total-bets-count');
    if (countEl) countEl.textContent = visibleCount;
  };

  // ─── Init ───

  function init() {
    initTheme();
    if (typeof Chart !== 'undefined') {
      initPerformanceChart();
    }
    animateCounters();
    initFreshness();
    initWebSocket();
    initKeyboardNav();

    // Detect which page we're on and act accordingly
    var futureContainer = document.getElementById('future-cards');
    if (futureContainer) {
      // Future Predictions page: load async
      loadFuturePredictions();
    } else {
      // Dashboard page: auto-refresh if empty
      setTimeout(autoRefreshIfEmpty, 1500);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* Global: resolvePredictions (called from onclick) */

function resolvePredictions() {
  var btn = document.getElementById('resolve-btn');
  var text = document.getElementById('resolve-text');
  var spinner = document.getElementById('resolve-spinner');

  if (!btn || !text || !spinner) return;

  btn.disabled = true;
  text.textContent = 'Resolving...';
  spinner.classList.remove('hidden');

  var controller = new AbortController();
  var timeoutId = setTimeout(function() { controller.abort(); }, 35000);

  fetch('/api/resolve', { signal: controller.signal })
    .then(function(r) {
      clearTimeout(timeoutId);
      if (!r.ok) return r.json().then(function(d) { throw new Error(d.error || 'Request failed'); });
      return r.json();
    })
    .then(function(data) {
      if (data.error) {
        if (window._showToast) window._showToast('Error: ' + data.error, 'error', 5000);
      } else {
        var msg = 'Resolved ' + data.resolved + ' predictions';
        if (window._showToast) window._showToast(msg, 'success', 3000);
        // Use async HTML swap instead of location.reload() to avoid page flash
        setTimeout(function() {
          var swap = window._swapMainContent;
          if (swap) {
            swap().catch(function() { location.reload(); });
          } else {
            location.reload();
          }
        }, 1000);
      }
    })
    .catch(function(err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        if (window._showToast) window._showToast('Resolve timed out', 'error', 5000);
      } else {
        if (window._showToast) window._showToast(err.message || 'Request failed', 'error', 5000);
      }
    })
    .finally(function() {
      btn.disabled = false;
      text.textContent = 'Resolve Predictions';
      spinner.classList.add('hidden');
    });
}
