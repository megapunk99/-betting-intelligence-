/* Betting Intelligence — Dashboard JS */

(function () {
  'use strict';

  // ─── Chart.js performance chart ───

  function initPerformanceChart() {
    const canvas = document.getElementById('performance-chart');
    if (!canvas) return;

    const rows = document.querySelectorAll('table tbody tr');
    const profits = [];
    rows.forEach(row => {
      const pnlCell = row.querySelector('td:last-child');
      if (!pnlCell) return;
      const text = pnlCell.textContent.trim();
      const match = text.match(/^[+\-–—$]\s*(\d+)/);
      if (match) {
        const val = parseFloat(match[1]);
        profits.push(isNaN(val) ? 0 : val);
      }
    });

    if (profits.length < 2) {
      canvas.parentElement.parentElement.style.display = 'none';
      return;
    }

    let cum = 0;
    const cumulative = profits.map(p => { cum += p; return cum; });
    const labels = profits.map((_, i) => `#${i + 1}`);

    new Chart(canvas, {
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

  function initFreshness() {
    const el = document.getElementById('gen-time');
    if (!el || !el.textContent) return;
    const dt = new Date(el.textContent);
    if (isNaN(dt.getTime())) return;

    function update() {
      const seconds = Math.floor((new Date() - dt) / 1000);
      let display;
      if (seconds < 60) display = seconds + 's ago';
      else if (seconds < 3600) display = Math.floor(seconds / 60) + 'm ago';
      else display = Math.floor(seconds / 3600) + 'h ago';
      el.textContent = display;
    }
    update();
    setInterval(update, 10000);
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
      if (e.key === 'r' && !e.ctrlKey && !e.metaKey) window.location.reload();
      if (e.key === 'p' && !e.ctrlKey && !e.metaKey) window.location.href = '/future-predictions';
      if (e.key === 'd' && !e.ctrlKey && !e.metaKey) window.location.href = '/';
    });
  }

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
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* Global: resolvePredictions (called from onclick) */

function resolvePredictions() {
  const btn = document.getElementById('resolve-btn');
  const text = document.getElementById('resolve-text');
  const spinner = document.getElementById('resolve-spinner');
  const result = document.getElementById('resolve-result');

  btn.disabled = true;
  text.textContent = 'Resolving...';
  spinner.classList.remove('hidden');

  fetch('/api/resolve')
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        result.className = 'toast error';
        result.textContent = 'Error: ' + data.error;
      } else {
        result.className = 'toast success';
        result.textContent = 'Resolved ' + data.resolved + ' predictions. Refreshing...';
        setTimeout(() => location.reload(), 1000);
      }
    })
    .catch(err => {
      result.className = 'toast error';
      result.textContent = 'Request failed: ' + err.message;
    })
    .finally(() => {
      btn.disabled = false;
      text.textContent = 'Resolve Predictions';
      spinner.classList.add('hidden');
      result.classList.remove('hidden');
    });
}
