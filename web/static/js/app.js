/* ═══════════════════════════════════════════════════════════════════════════
   Betting Intelligence — Interactive Effects
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

  // ── Particle Background ──────────────────────────────────────────────────
  function initParticles() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];
    let mouse = { x: 0, y: 0 };
    let frameId;

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    class Particle {
      constructor() {
        this.reset();
      }

      reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 0.5;
        this.speedX = (Math.random() - 0.5) * 0.3;
        this.speedY = (Math.random() - 0.5) * 0.3;
        this.opacity = Math.random() * 0.5 + 0.1;
        this.pulseSpeed = Math.random() * 0.02 + 0.005;
        this.pulsePhase = Math.random() * Math.PI * 2;
        this.color = this.randomColor();
      }

      randomColor() {
        const colors = [
          { r: 0, g: 230, b: 94 },   // green
          { r: 96, g: 165, b: 250 },  // blue
          { r: 167, g: 139, b: 250 }, // purple
        ];
        return colors[Math.floor(Math.random() * colors.length)];
      }

      update() {
        this.x += this.speedX;
        this.y += this.speedY;
        this.pulsePhase += this.pulseSpeed;

        // Mouse interaction
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
          const force = (150 - dist) / 150;
          this.x -= dx * force * 0.01;
          this.y -= dy * force * 0.01;
        }

        // Wrap around edges
        if (this.x < -10) this.x = canvas.width + 10;
        if (this.x > canvas.width + 10) this.x = -10;
        if (this.y < -10) this.y = canvas.height + 10;
        if (this.y > canvas.height + 10) this.y = -10;
      }

      draw() {
        const pulseOpacity = this.opacity * (0.7 + 0.3 * Math.sin(this.pulsePhase));
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${pulseOpacity})`;
        ctx.fill();

        // Glow
        if (this.size > 1.2) {
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size * 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${pulseOpacity * 0.1})`;
          ctx.fill();
        }
      }
    }

    // Create particles based on screen size
    function createParticles() {
      const count = Math.min(Math.floor(canvas.width * canvas.height / 15000), 120);
      particles = [];
      for (let i = 0; i < count; i++) {
        particles.push(new Particle());
      }
    }
    createParticles();

    // Draw connections between nearby particles
    function drawConnections() {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 120) {
            const opacity = (1 - dist / 120) * 0.12;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(0, 230, 94, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach(p => {
        p.update();
        p.draw();
      });

      drawConnections();

      frameId = requestAnimationFrame(animate);
    }

    animate();

    // Track mouse
    document.addEventListener('mousemove', (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    });

    document.addEventListener('mouseleave', () => {
      mouse.x = -9999;
      mouse.y = -9999;
    });

    // Cleanup
    window.addEventListener('beforeunload', () => {
      if (frameId) cancelAnimationFrame(frameId);
    });

    // Recreate particles on resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        resize();
        createParticles();
      }, 300);
    });
  }

  // ── Animated Counters ────────────────────────────────────────────────────
  function initCounters() {
    document.querySelectorAll('[data-counter]').forEach(el => {
      const target = parseFloat(el.getAttribute('data-counter'));
      const suffix = el.getAttribute('data-suffix') || '';
      const duration = parseInt(el.getAttribute('data-duration')) || 1500;
      const isCurrency = el.getAttribute('data-currency') === 'true';
      const isPercent = el.getAttribute('data-percent') === 'true';
      const decimals = parseInt(el.getAttribute('data-decimals')) || 0;

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            animateCounter(el, target, duration, suffix, isCurrency, isPercent, decimals);
            observer.unobserve(el);
          }
        });
      }, { threshold: 0.3 });

      observer.observe(el);
    });
  }

  function animateCounter(el, target, duration, suffix, isCurrency, isPercent, decimals) {
    const start = performance.now();
    const startVal = 0;

    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (target - startVal) * eased;

      let display;
      if (isCurrency) {
        display = '$' + current.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      } else if (isPercent) {
        display = current.toFixed(decimals) + '%';
      } else {
        display = current.toFixed(decimals);
      }

      el.textContent = display + (suffix && !isCurrency && !isPercent ? suffix : '');

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        // Final value
        let final;
        if (isCurrency) {
          final = '$' + target.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        } else if (isPercent) {
          final = target.toFixed(decimals) + '%';
        } else {
          final = target.toFixed(decimals) + suffix;
        }
        el.textContent = final;
      }
    }

    requestAnimationFrame(update);
  }

  // ── Smooth Nav Transitions ───────────────────────────────────────────────
  function initNavTransitions() {
    document.addEventListener('htmx:beforeSwap', function(evt) {
      // Close mobile nav
      const mobileNav = document.getElementById('mobile-nav');
      if (mobileNav && !mobileNav.classList.contains('hidden')) {
        mobileNav.classList.add('hidden');
      }

      // Add fade-out to current content
      const main = document.querySelector('main');
      if (main && evt.detail.target === main) {
        main.style.opacity = '0';
        main.style.transform = 'translateY(8px)';
        main.style.transition = 'opacity 0.15s ease, transform 0.15s ease';
      }
    });

    document.addEventListener('htmx:afterSwap', function(evt) {
      const main = document.querySelector('main');
      if (main && evt.detail.target === main) {
        // Trigger reflow
        void main.offsetWidth;
        main.style.opacity = '1';
        main.style.transform = 'translateY(0)';
      }

      // Update active nav link
      document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === window.location.pathname) {
          link.classList.add('active');
        }
      });

      // Re-init effects on new content
      initCounters();
      initStaggerAnimations();
      initCardTilt();
    });
  }

  // ── Staggered Entrance Animations ────────────────────────────────────────
  function initStaggerAnimations() {
    document.querySelectorAll('.stagger-children > *').forEach((el, i) => {
      el.classList.add('fade-in-up');
      el.style.animationDelay = `${i * 0.06}s`;
    });
  }

  // ── Card Tilt Effect ─────────────────────────────────────────────────────
  function initCardTilt() {
    document.querySelectorAll('.tilt-card').forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const rotateX = (y - centerY) / centerY * -8;
        const rotateY = (x - centerX) / centerX * 8;

        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
      });

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0)';
        card.style.transition = 'transform 0.5s ease';
      });

      card.addEventListener('mouseenter', () => {
        card.style.transition = 'transform 0.1s ease';
      });
    });
  }

  // ── Refresh Toast ────────────────────────────────────────────────────────
  function initRefreshToast() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (!refreshBtn) return;

    refreshBtn.addEventListener('click', async () => {
      const spinner = refreshBtn.querySelector('.refresh-spinner');
      const icon = refreshBtn.querySelector('.refresh-icon');

      if (spinner) spinner.classList.remove('hidden');
      if (icon) icon.classList.add('hidden');
      refreshBtn.disabled = true;

      try {
        const resp = await fetch('/api/refresh');
        const data = await resp.json();
        showToast(`Data refreshed — ${data.clear_picks} clear picks, ${data.total_bets} bets`, 'success');
        // Reload after short delay
        setTimeout(() => location.reload(), 800);
      } catch (err) {
        showToast('Failed to refresh data', 'error');
        if (spinner) spinner.classList.add('hidden');
        if (icon) icon.classList.remove('hidden');
        refreshBtn.disabled = false;
      }
    });
  }

  // ── Toast Notifications ──────────────────────────────────────────────────
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) {
      // Create container
      const c = document.createElement('div');
      c.id = 'toast-container';
      c.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        gap: 8px;
        pointer-events: none;
      `;
      document.body.appendChild(c);
    }

    const toast = document.createElement('div');
    const colors = {
      success: 'rgba(0, 230, 94, 0.15)',
      error: 'rgba(248, 113, 113, 0.15)',
      info: 'rgba(96, 165, 250, 0.15)',
    };
    const borderColors = {
      success: 'rgba(0, 230, 94, 0.3)',
      error: 'rgba(248, 113, 113, 0.3)',
      info: 'rgba(96, 165, 250, 0.3)',
    };

    toast.style.cssText = `
      background: ${colors[type] || colors.info};
      backdrop-filter: blur(12px);
      border: 1px solid ${borderColors[type] || borderColors.info};
      color: #fff;
      padding: 12px 20px;
      border-radius: 10px;
      font-size: 0.85rem;
      font-family: 'Inter', sans-serif;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      animation: toastIn 0.3s ease-out;
      pointer-events: auto;
      max-width: 360px;
    `;
    toast.textContent = message;

    const container2 = document.getElementById('toast-container') || document.body.appendChild(
      Object.assign(document.createElement('div'), { id: 'toast-container', style: 'position:fixed;top:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none;' })
    );
    container2.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  // ── Live Clock ───────────────────────────────────────────────────────────
  function initLiveClock() {
    const clock = document.getElementById('live-clock');
    if (!clock) return;

    function update() {
      const now = new Date();
      clock.textContent = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short'
      });
    }

    update();
    setInterval(update, 1000);
  }

  // ── Initialize Everything ────────────────────────────────────────────────
  function init() {
    // Wait for DOM
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', run);
    } else {
      run();
    }
  }

  function run() {
    initParticles();
    initCounters();
    initNavTransitions();
    initStaggerAnimations();
    initCardTilt();
    initRefreshToast();
    initLiveClock();
  }

  // Inject toast animation keyframes
  const style = document.createElement('style');
  style.textContent = `
    @keyframes toastIn {
      from { opacity: 0; transform: translateX(20px) translateY(-8px); }
      to { opacity: 1; transform: translateX(0) translateY(0); }
    }
  `;
  document.head.appendChild(style);

  init();

})();
