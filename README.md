# 🏀 Betting Intelligence System

**Production-grade basketball betting analytics engine — identifies market inefficiencies, generates exact +EV bets, and manages risk through Kelly-optimized staking.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/megapunk99/betting-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/megapunk99/betting-intelligence/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Web Dashboard](#-web-dashboard)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Docker](#-docker)
- [Production Deployment](#-production-deployment)
- [Development](#-development)
- [CLI Reference](#-cli-reference)
- [License](#-license)

---

## 🚀 Features

### 🎯 Exact Bet Recommendations
No probabilities, no confusion — just clear, exact bets with dollar amounts:
```
$55 on Spurs -3.5           (4.2% edge)
$32 on Wemby o24.5pts     (3.8% edge)
$28 on Celtics ML           (3.1% edge)
```

### 📊 Every Major Bet Type
| Market | Description |
|--------|-------------|
| **Moneyline** | Team to win outright |
| **Point Spread** | Team -3.5 / Team +3.5 |
| **Total O/U** | Over/Under game points |
| **Team Total** | Over/Under team points |
| **1st Quarter** | Quarter winner predictions |
| **1st Half Total** | First half scoring totals |
| **Player Props** | Points, rebounds, assists, PRA |
| **Parlays** | Auto-suggested multi-leg combos |

### 🏆 Multi-League Coverage
- 🏀 **NBA** — Full season + playoffs (momentum model)
- 🇫🇷 **LNB Pro B** — French 2nd division (high inefficiency)
- 🇨🇦 **CEBL** — Canadian Elite Basketball League
- 🇧🇪🇳🇱 **BNXT** — Belgium/Netherlands combined league

### 🛡️ Production-Grade Reliability
- **Circuit Breaker Pattern** — Automatic failure detection and recovery for all external API calls (TheOddsAPI, Stripe)
- **Graceful Degradation** — Every page renders fallback data when services are unavailable — never a 500 error
- **Health Check Endpoints** — Full system monitoring with `/api/health`, `/api/health/live`, `/api/health/ready`
- **TTL Caching** — 60s cache with stale-while-revalidate for all API responses
- **Double-Checked Locking** — Thread-safe singleton initialization for all engine components

### 🧠 Model Accuracy
- **Cross-Validation R²: 0.57 ± 0.10** — Realistic predictive power (eliminated data leakage from 0.99)
- **Overfitting Score: 0.0/100** — LOW risk, strategy appears genuine
- **Proper Probability Conversion** — All probability estimates use calibrated sigmoid functions
- **ELO Ratings** — Chronological ELO (k=20) with zero lookahead bias as model features
- **7+ Models** — LightGBM, CatBoost, Random Forest, Bayesian Ridge, Stacking Ensemble, Momentum Reversion, Spread Prediction

---

## 🏗 Architecture

```
betting-intelligence/
├── web/                          # FastAPI web application
│   ├── app.py                    # ★ Main app with circuit breakers, health checks
│   ├── static/                   # CSS, JS, assets
│   │   ├── css/app.css           # Dark theme stylesheet (particle bg, glassmorphism)
│   │   └── js/app.js             # Interactive effects (particles, counters, tilt)
│   └── templates/                # Jinja2 templates
│       ├── base.html             # Base layout with nav, footer, WebSocket
│       ├── index.html            # Dashboard with stats, clear picks, market breakdown
│       ├── live.html             # Live games with real-time updates
│       ├── pre_match_prediction.html
│       ├── landing.html          # Marketing/pricing page
│       ├── subscribe_*.html      # Stripe subscription pages
│       └── *.html                # Additional pages
├── src/betting_intel/
│   ├── api/                      # Production REST API
│   │   └── app.py                # FastAPI with lifespan, CORS, Prometheus
│   ├── backtesting/              # Walk-forward validation engine
│   ├── betting/                  # Edge computation, EV, CLV tracking
│   ├── business/                 # Subscriptions, Stripe integration
│   ├── cli/                      # Click-based CLI
│   ├── config/                   # pydantic-settings (env-based config)
│   ├── data/                     # Data pipeline
│   │   ├── loader.py             # SQLite data loading
│   │   ├── features.py           # Feature engineering (v2.1, ELO, backfill-safe)
│   │   ├── odds_fetcher.py       # TheOddsAPI client with multi-book consensus
│   │   ├── live_gateway.py       # Live data orchestrator
│   │   └── small_leagues/        # LNB Pro B, CEBL, BNXT sources
│   ├── live/                     # Live prediction engine
│   ├── models/                   # ML models (predictors, ensemble, MLP)
│   ├── pipeline/                 # Full prediction pipeline orchestrator
│   ├── recommendations/          # ★ THE BET ENGINE
│   │   ├── engine.py             # Core recommendation engine
│   │   ├── bet_types.py          # Bet type definitions
│   │   ├── ranker.py             # Clear pick identification
│   │   ├── player_props.py       # Player prop predictions
│   │   └── ev_scanner.py         # +EV opportunity scanning
│   ├── risk/                     # Kelly, exposure, correlation
│   └── validation/               # Time-series CV, calibration, overfitting
├── tools/                        # Utility scripts
├── tests/                        # pytest test suite
├── data/                         # SQLite databases, cached predictions
├── models/                       # Saved model files (.pkl)
├── main.py                       # CLI entry point for full pipeline
├── docker-compose.yml            # Multi-service Docker setup
├── Makefile                      # Build automation
└── README.md                     # ← You are here
```

### 🔄 Data Flow

```
TheOddsAPI ──→ OddsAPIClient ──→ FeatureEngineer ──→ Models ──→ RecommendationEngine ──→ Web Dashboard
     │               │                  │                │               │
     │     CircuitBreaker        Backfill-safe     Ensemble       Kelly Staking
     │     (3 failures → OPEN)   (no future leak)  (7 models)     (exposure mgmt)
     │               │                  │                │               │
     └────────── Cache (60s) ──────────┘                └─── Clear Picks
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- [TheOddsAPI key](https://the-odds-api.com/) (free tier available)

### Installation

```bash
# Clone
git clone https://github.com/megapunk99/betting-intelligence.git
cd betting-intelligence

# Install
make install

# Configure
cp .env.example .env
# Edit .env with your ODDS_API_KEY
```

### Windows
```powershell
git clone https://github.com/megapunk99/betting-intelligence.git
cd betting-intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,dashboard]"
Copy-Item .env.example .env
```

### Run the Full Pipeline
```bash
# Fast mode (LightGBM + Momentum — ~2 min)
python main.py

# Full mode (all 7+ models — ~10 min)
python main.py --full --no-tune
```

---

## 🌐 Web Dashboard

The FastAPI web dashboard is the primary interface. Run:

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Opens at **[http://localhost:8000](http://localhost:8000)**

### Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with pricing tiers |
| `/dashboard` | Main dashboard: stats, clear picks, today's games, market breakdown |
| `/live` | Live games with real-time predictions |
| `/pre-match-prediction` | All games today + tomorrow with market odds |
| `/clear-picks` | High-confidence picks with HTMX filtering |
| `/todays-card` | Today's full betting card |
| `/tomorrow` | Tomorrow's predictions |
| `/all-bets` | All bets with sorting and filtering |
| `/player-props` | Player prop predictions |
| `/signals` | X/Twitter intelligence signals |
| `/api/health` | **Full system health check** |
| `/api/health/live` | **Lightweight liveness probe** |
| `/api/health/ready` | **Readiness probe (database + engine)** |

---

## 📡 API Reference

### System Health
```http
GET /api/health
```
Returns status of all critical services: database, TheOddsAPI, Stripe, engine, circuit breakers.

```json
{
  "status": "degraded",
  "services": {
    "database": {"status": "stale", "age_hours": 96.0},
    "odds_api": {"status": "missing", "has_key": false},
    "engine": {"status": "ok"}
  },
  "circuit_breakers": {
    "odds_api": {"state": "closed", "total_calls": 0, "total_failures": 0}
  },
  "warnings": ["ODDS_API_KEY not configured"]
}
```

### Live Data
```http
GET /api/live/snapshot    # Full live prediction snapshot
GET /api/live/chart-data  # Chart-ready data (edges, confidence breakdown)
GET /api/live/games       # Live games only
```

### Bets
```http
GET /api/refresh          # Force refresh cached data
GET /api/bets?league=nba&min_edge=0.02&limit=20
GET /api/clear-picks
```

### Signals
```http
GET /api/signals          # X/Twitter intelligence signals
```

### Stripe
```http
GET  /api/stripe/config                    # Publishable key
POST /api/stripe/create-checkout-session   # Create subscription
POST /api/stripe/webhook                   # Stripe webhook handler
```

---

## ⚙️ Configuration

All settings via `.env` file. Key settings:

```ini
# REQUIRED: TheOddsAPI key (get free at https://the-odds-api.com/)
ODDS_API_KEY=your-api-key-here

# Staking
INITIAL_BANKROLL=10000.0
MAX_KELLY_FRACTION=0.25
MIN_EDGE_THRESHOLD=0.02

# Stripe (for subscriptions)
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
SITE_DOMAIN=http://localhost:8000

# Monitoring
ENABLE_PROMETHEUS=false
```

---

## 🐳 Docker

```bash
# Build and start all services
docker compose up -d

# Services:
# - API server:     http://localhost:8000
# - Prometheus:     http://localhost:9091
# - Grafana:        http://localhost:3000 (user: admin / admin)
```

---

## 🏭 Production Deployment

### Prerequisites
- Paid TheOddsAPI tier (START 5M / $119mo recommended)
- Stripe account (for subscriptions)
- PostgreSQL (optional, for production scale)

### Steps
```bash
# 1. Set up environment
cp .env.example .env
# Edit ODDS_API_KEY, STRIPE_*, DATABASE_URL

# 2. Build and run
docker compose up -d --build

# 3. Verify health
curl http://localhost:8000/api/health/live
curl http://localhost:8000/api/health/ready

# 4. Set up monitoring
# Grafana dashboards in grafana/dashboards/
```

### Production Checklist
- [ ] Paid TheOddsAPI key configured
- [ ] Stripe keys configured (for subscriptions)
- [ ] Database migrated (SQLite → PostgreSQL recommended)
- [ ] SITE_DOMAIN set to production URL
- [ ] Grafana dashboards imported
- [ ] Prometheus metrics enabled
- [ ] Docker health checks configured
- [ ] SSL/TLS (reverse proxy with nginx/Caddy)
- [ ] Regular database backups

---

## 🛠 Development

```bash
# Install dev dependencies
make dev

# Run all CI checks locally
make ci

# Run tests
make test-fast     # Fast tests only
make test-slow     # Slow tests (requires NBA database)

# Type checking
make typecheck

# Lint and format
make lint
make format

# Validate YAML workflow files
make yaml-validate
```

### Windows
```powershell
.\.venv\Scripts\Activate.ps1
pytest -v -m "not slow"
ruff check src/ tests/
ruff format --check src/ tests/
```

---

## 📖 CLI Reference

```bash
Usage: betting-intel [OPTIONS] COMMAND [ARGS]...

Commands:
  pipeline         Pipeline management (run, status)
  models           Model management (list, info)
  backtest         Backtesting (run, report)
  api              API server (start)
  db               Database management (init, check)
  web              Start FastAPI web GUI
  small-leagues    Small-league data (list, fetch, teams)
  recommendations  Generate bets (list, clear-picks, player-props)
```

| Action | Command |
|--------|---------|
| Install | `make install` |
| Run pipeline | `python main.py` |
| Full pipeline | `python main.py --full --no-tune` |
| Run server | `uvicorn web.app:app --port 8000` |
| Health check | `curl http://localhost:8000/api/health` |
| Run tests | `make test` |
| CI checks | `make ci` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Sports betting involves financial risk. Predictions are based on historical data and statistical models — they do not guarantee future results. Gamble responsibly. Never bet more than you can afford to lose.
