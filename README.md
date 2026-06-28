<div align="center">

# Betting Intelligence

**ML-powered sports betting analytics platform** — detects market inefficiencies, generates live predictions, and tracks performance with Closing Line Value metrics.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-365%20passing-brightgreen)](https://github.com/megapunk99/betting-intelligence/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Code style](https://img.shields.io/badge/Code%20style-Ruff-000000)](https://github.com/astral-sh/ruff)

</div>

---

##  Features

- **Live Prediction Engine** — 60-second refresh cycle with 3-tier odds fallback (TheOddsAPI → ESPN → DraftKings)
- **Market Inefficiency Detection** — ML models trained to predict *how much the market is wrong*, not just who wins
- **Automated P&L Tracking** — resolves predictions against actual results, computes ROI, win rate, Sharpe ratio
- **Closing Line Value (CLV)** — measures whether your predicted line beats the closing market (the #1 metric in betting)
- **3% Minimum Edge Filter** — only surfaces bets with statistically significant edges
- **Web Dashboard** — real-time predictions, P&L charts, sport filters, dark/light theme
- **Kelly Criterion Staking** — optimal bet sizing based on edge, confidence, and bankroll
- **Strategy Alerts** — automatic detection of underperforming strategies within 30-day trailing window
- **342+ Automated Tests** — full test coverage with in-memory databases, no external dependencies required

##  Quick Start

### Prerequisites

- Python 3.10 or later
- (Optional) A free [TheOddsAPI key](https://the-odds-api.com/) for live odds

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/megapunk99/betting-intelligence.git
cd betting-intelligence

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional)
cp .env.example .env
# Edit .env with your TheOddsAPI key if you want live odds

# 5. Start the web dashboard
uvicorn web.app:app --reload --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Verify It Works

```bash
# Health check
curl http://localhost:8000/api/health

# Run the test suite
pytest tests/ -v
```

##  Web Dashboard

The dashboard auto-loads when you start the server. No configuration needed.

| Page | Route | Description |
|------|-------|-------------|
| **Dashboard** | `/` | Live predictions, P&L chart, resolved bets table, sport filters |
| **Future Predictions** | `/future-predictions` | AI-projected game scores with quarter/half breakdowns |

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Stats Bar** | Live games, win rate, P&L, average edge, clear picks count |
| **Refresh Button** | Fetches fresh odds on demand (never auto-refreshes on page load) |
| **Resolve Button** | Matches predictions against actual results to compute P&L |
| **Sport Filters** | Filter predictions by league (NBA, NCAAB, MLB, etc.) |
| **Clear Picks** | Bets with edge > 3% highlighted with star badge |
| **Analysis Dropdown** | Click any prediction to see model reasoning and feature importance |
| **P&L Chart** | Cumulative profit/loss over time (auto-generated from resolved bets) |
| **All Bets Table** | Complete history of predictions with results |
| **CSV Export** | Download resolved bets as CSV for external analysis |
| **Dark/Light Theme** | Persists to localStorage, follows system preference |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (DB status, engine readiness) |
| `/api/live/refresh` | POST | Force-refresh live odds from all sources |
| `/api/live/snapshot` | GET | Current prediction snapshot with all games |
| `/api/predictions?limit=50` | GET | Live predictions sorted by edge descending |
| `/api/clear-picks` | GET | Only bets with edge > 3% |
| `/api/bets` | GET | All live games as a flat list |
| `/api/resolve` | GET | Resolve predictions against actual results |
| `/api/resolved-bets/csv` | GET | Download resolved bets as CSV |
| `/api/future-predictions` | GET | AI-projected future games |
| `/api/health/ready` | GET | Readiness check (lightweight) |

##  Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  TheOddsAPI  │────│ LivePrediction   │────│  FastAPI Web     │
│  (live odds) │     │ Engine           │     │  Dashboard       │
└──────────────┘     │ (60s refresh)    │     │  (Jinja2 + JS)   │
                     │                  │     └──────────────────┘
┌──────────────┐     │  ┌────────────┐  │
│  ESPN/DK     │────│  │ Market     │  │     ┌──────────────────┐
│  Scrapers    │     │  │ Inefficiency│  │────│  Analytics       │
└──────────────┘     │  │ System     │  │     │  Tracker         │
                     │  └────────────┘  │     │  (CLV, ROI, P&L) │
┌──────────────┐     │  ┌────────────┐  │     └──────────────────┘
│  Historical   │────│  │ Totals    │  │
│  NBA Data    │     │  │ Regressor │  │     ┌──────────────────┐
└──────────────┘     │  └────────────┘  │     │  MarketOddsStore │
                     │                  │────│  (SQLite history)│
                     │  ┌────────────┐  │     └──────────────────┘
                     │  │ Kelly     │  │
                     │  │ Staker    │  │
                     │  └────────────┘  │
                     └──────────────────┘
```

### Key Components

| Component | Path | What It Does |
|-----------|------|-------------|
| **Live Engine** | `src/betting_intel/live/engine.py` | Central orchestrator — fetches odds, runs predictions, caches snapshots |
| **Robust System** | `src/betting_intel/models/robust_ensemble.py` | XGBoost + LogisticRegression + RandomForest stacking |
| **Feature Engineering** | `src/betting_intel/data/features.py` | 372 clean features with no data leakage |
| **Market Inefficiency** | `src/betting_intel/features/market_inefficiency.py` | Computes market error targets for training |
| **Analytics Tracker** | `src/betting_intel/analytics/tracker.py` | Resolves bets, computes CLV, strategy alerts |
| **Kelly Staker** | `src/betting_intel/recommendations/staking.py` | Optimal bet sizing based on bankroll management |
| **Web Dashboard** | `web/app.py` | FastAPI + Jinja2 templates + Chart.js |

##  CLV: Closing Line Value

CLV is the single most important metric in sports betting. It measures whether your predicted line was better than the closing market line.

```
CLV = our_predicted_prob - closing_market_prob
```

- **Positive CLV** = you identified value the market didn't fully price in
- **Negative CLV** = the market moved against your prediction
- **Zero CLV** = your edge was exactly the market movement (neutral)

The system computes CLV automatically for every resolved bet using:
1. **Opening line** → first odds snapshot stored for the game
2. **Your prediction** → `opening_prob + edge_pct`
3. **Closing line** → last odds snapshot before game start

##  Running Tests

```bash
# Full test suite (365+ tests)
pytest tests/

# Specific test files
pytest tests/test_market_odds_store.py -v
pytest tests/test_analytics_tracker.py -v

# With coverage report
pytest --cov=src --cov-report=term-missing
```

All tests use:
- In-memory SQLite databases (no real DB needed)
- Mocked external APIs (no network required)
- Fast execution (full suite in ~2 minutes)

##  Docker Deployment

```bash
# Build and run
docker compose up --build

# Opens at http://localhost:8000
```

For production deployment on a VPS (Linode, DigitalOcean, etc.):

```bash
# Copy to server
rsync -avz --exclude='.env' --exclude='venv' --exclude='__pycache__' ./ user@your-server:/app/

# On the server
cd /app
docker compose up -d --build
```

##  CLI Tools

```bash
# View available commands
python -m betting_intel.cli.main --help

# Refresh live predictions
python -m betting_intel.cli.main live refresh

# Backtest historical performance
python -m betting_intel.cli.main backtest run
```

##  Project Layout

```
betting-intelligence/
├── src/betting_intel/          # Core Python package
│   ├── live/                   # Prediction engine core
│   ├── data/                   # Data loaders, features, scrapers
│   ├── models/                 # ML models (ensemble, stacking)
│   ├── pipeline/               # Training, backtesting, export
│   ├── features/               # Feature engineering (market inefficiency)
│   ├── db/                     # Database schema & connection
│   ├── recommendations/        # Bet recommendation engine
│   ├── analytics/              # Performance tracking & alerts
│   ├── api/                    # FastAPI routes
│   ├── cli/                    # Click CLI
│   └── config/                 # Settings (.env)
├── web/                        # FastAPI web dashboard
│   ├── app.py                  # Server + routes
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS
├── tests/                      # Pytest test suite
├── tools/                      # Utility scripts
├── data/                       # SQLite databases, prediction logs
├── models/saved/               # Trained model artifacts
├── .env.example                # Environment template
├── Dockerfile                  # Container build
└── docker-compose.yml          # Container orchestration
```

##  License

MIT — see [LICENSE](LICENSE).

##  FAQ

**Q: Do I need an API key?**  
A: No. The dashboard works with cached data. For live odds, get a free key from [TheOddsAPI](https://the-odds-api.com/).

**Q: Does this place real bets?**  
A: No. It generates predictions and tracks performance. No money is ever moved.

**Q: What sports are supported?**  
A: Primary focus is NBA basketball. The system can be extended to other sports via `sport_configs.py`.

**Q: How accurate are the predictions?**  
A: The system tracks CLV, win rate, and ROI for every strategy. Check the dashboard's P&L chart and resolved bets table for actual performance data.
