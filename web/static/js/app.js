/* ═══════════════════════════════════════════════════════════════════════════
   Betting Intelligence — Premium Dashboard JS
   Interactive card effects, live counters, auto-refresh
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── Card Mouse Tracking (glow follows cursor) ─────────────────────────
  function initCardGlow() {
    document.querySelectorAll('.pred-card').forEach(card => {
      card.addEventListener('mousemove', e => {
        const rect = card.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        card.style.setProperty('--mouse-x', x + '%');
        card.style.setProperty('--mouse-y', y + '%');
      });
    });
  }

  // ── Animated Counters ─────────────────────────────────────────────────
  function animateCounters() {
    document.querySelectorAll('[data-count]').forEach(el => {
      const target = parseInt(el.getAttribute('data-count'), 10);
      if (isNaN(target) || target === 0) {
        el.textContent = target;
        return;
      }
      const duration = 1200;
      const start = performance.now();
      const initial = 0;

      function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(initial + (target - initial) * eased);
        el.textContent = current;
        if (progress < 1) {
          requestAnimationFrame(update);
        }
      }
      requestAnimationFrame(update);
    });
  }

  // ── Live Freshness Timer ──────────────────────────────────────────────
  function initFreshness() {
    const el = document.getElementById('gen-time');
    if (!el || !el.textContent) return;

    const dateStr = el.textContent;
    const dt = new Date(dateStr);
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

  // ── Auto-Refresh (every 5 minutes) ────────────────────────────────────
  function initAutoRefresh() {
    setInterval(() => {
      fetch('/api/future-predictions?num_games=1')
        .then(r => r.json())
        .then(data => {
          const statusEl = document.querySelector('.status-text');
          if (statusEl && data.n_predictions !== undefined) {
            statusEl.textContent = data.n_predictions + ' games';
            // Update status dot
            const dot = document.querySelector('.status-dot');
            if (dot) {
              dot.classList.toggle('live', data.n_predictions > 0);
            }
          }
        })
        .catch(() => { /* silent */ });
    }, 300000);
  }

  // ── Card Hover Sound Effect (subtle) ──────────────────────────────────
  // No audio needed — visual feedback is sufficient

  // ── Smooth Scroll for Anchor Links ────────────────────────────────────
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', e => {
        const href = a.getAttribute('href');
        if (href && href.length > 1) {
          const target = document.querySelector(href);
          if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }
      });
    });
  }

  // ── Copy Game ID on Click (power user feature) ────────────────────────
  function initCopyGameId() {
    document.querySelectorAll('.card-date').forEach(el => {
      el.style.cursor = 'pointer';
      el.title = 'Click to copy game date';
      el.addEventListener('click', () => {
        const text = el.textContent.trim();
        navigator.clipboard.writeText(text).catch(() => {});
        // Brief flash feedback
        el.style.color = 'var(--accent)';
        setTimeout(() => { el.style.color = ''; }, 500);
      });
    });
  }

  // ── Keyboard Navigation ───────────────────────────────────────────────
  function initKeyboardNav() {
    document.addEventListener('keydown', e => {
      // 'r' to refresh
      if (e.key === 'r' && !e.ctrlKey && !e.metaKey && !e.target.closest('input,textarea')) {
        window.location.reload();
      }
    });
  }

  // ── Init Everything ───────────────────────────────────────────────────
  function init() {
    initCardGlow();
    animateCounters();
    initFreshness();
    initAutoRefresh();
    initSmoothScroll();
    initCopyGameId();
    initKeyboardNav();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
