#  Betting Intelligence System

**Professional-grade basketball betting analytics — identify market inefficiencies and get exact, actionable bets.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/megapunk99/betting-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/megapunk99/betting-intelligence/actions)

---

##   Features

###  Exact Bet Recommendations
No probabilities, no confusion — just clear, exact bets with dollar amounts:
```
 $55 on Spurs -3.5           (4.2% edge)
 $32 on Wemby o24.5pts     (3.8% edge)
 $28 on Celtics ML           (3.1% edge)
```

###  Every Major Bet Type
| Market | Examples |
|--------|----------|
| **Moneyline** | Team A to win outright |
| **Point Spread** | Team A -3.5 / Team B +3.5 |
| **Total O/U** | Over/Under total game points |
| **Team Total** | Over/Under specific team points |
| **1st Quarter** | Quarter winner predictions |
| **1st Half Total** | First half scoring totals |
| **Player Props** | Points, rebounds, assists, PRA |
| **Parlays** | Auto-suggested multi-leg combos |

###  Multi-League Coverage
-  NBA — Full season + playoffs (momentum model)
-  **LNB Pro B** — French 2nd division (high inefficiency)
-  **CEBL** — Canadian Elite Basketball League
-  **BNXT** — Belgium/Netherlands combined league

###  Market Inefficiency Detection
Smaller leagues have **less efficient markets** — algorithms trained on top-tier data misprice these games, creating consistent edge opportunities.

---

##   Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/megapunk99/betting-intelligence.git
cd betting-intelligence

# Install with all dependencies
make install

# Copy and configure environment
cp .env.example .env
```

### Quick Start (Windows)

```powershell
# Clone the repo
git clone https://github.com/megapunk99/betting-intelligence.git
cd betting-intelligence

# Run the Windows bootstrap installer (recommended)
powershell -ExecutionPolicy Bypass -File install.ps1

# Or manually:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"

# Copy and configure environment
Copy-Item .env.example .env
```

> **Note:** Some Python packages require a C compiler. If you get build errors,
> install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> and retry.

### Initialize the Database

```bash
# Using the CLI (after pip install -e .)
betting-intel db init

# Or using the standalone script (before installing the package)
python scripts/init_db.py
```

### Generate Exact Bets

```bash
# See ALL available bets, ranked by edge
betting-intel recommendations list

# Only the HIGH-CONFIDENCE picks
betting-intel recommendations clear-picks

# Today's betting card
betting-intel recommendations todays-card

# Player props for a specific matchup
betting-intel recommendations player-props Spurs Thunder
```

### Launch the Dashboard

```bash
betting-intel dashboard
# or directly:
streamlit run dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501) with 3 tabs:
- ★ **Place These Bets** — Clear picks with exact bet amounts
-  **Today's Card** — All bets for today's games
-  **All Bets** — Full list with filters

---

##   Example Output

```
★ CLEAR PICKS — PLACE THESE BETS
────────────────────────────────────
★ PICK #1 — MODERATE (78/100 confidence)
  PLACE: $55 on Spurs -3.5
  Game:   Thunder @ Spurs (NBA)
  Market: Point Spread
  →  Strong edge (4.2%)
  →  High win probability (58%)
  →  Model signal: Spread Baseline
────────────────────────────────────
★ PICK #2 — CONSERVATIVE (65/100 confidence)
  PLACE: $32 on Wemby o24.5pts
  Game:   Thunder @ Spurs (NBA)
  Market: Player Points
  →  Exceptional player form
  →  Pace-adjusted projection: 27.3pts
────────────────────────────────────
TOTAL TO BET: $87 across 2 picks
```

---

##   Architecture

```
src/betting_intel/
├── betting/           # Edge computation, bankroll management
│   ├── edge.py        # Kelly staking, edge detection
│   └── bankroll.py    # Bankroll management & risk control
├── cli/               # Click-based CLI
│   └── main.py        # All CLI commands
├── config/            # pydantic-settings configuration
│   └── settings.py    # Environment-based config
├── data/              # Data pipeline
│   ├── loader.py      # NBA game log loading
│   ├── features.py    # Feature engineering
│   └── small_leagues/ # LNB Pro B, CEBL, BNXT sources
├── models/            # ML models
│   └── predictors.py  # Momentum, XGBoost, Ensemble models
└── recommendations/   # ** THE BET ENGINE **
    ├── __init__.py     # Unified API
    ├── bet_types.py    # All bet type definitions
    ├── engine.py       # Core recommendation engine
    ├── ranker.py       # Clear pick identification
    └── player_props.py # Player prop predictions
```

---

##   Configuration

All settings via `.env` file. Copy `.env.example` to `.env` and fill in your values.

Key settings to configure:

```ini
# REQUIRED for live odds and --live mode
ODDS_API_KEY=your-api-key-here

# Staking
INITIAL_BANKROLL=10000.0    # Starting bankroll ($)
MAX_KELLY_FRACTION=0.25     # Conservative Kelly (25%)
MIN_EDGE_THRESHOLD=0.02     # Minimum edge to bet (2%)

# Leagues
ENABLE_SMALL_LEAGUES=true
ACTIVE_SMALL_LEAGUES=lnb_pro_b,cebl,bnxt,wnba

# Models
ENABLE_LINEAR_MODEL=true
ENABLE_XGBOOST_MODEL=true
ENABLE_ENSEMBLE=true
```

See [.env.example](.env.example) for the full list of all 40+ configuration options.

---

##   Docker

```bash
# Build and start all services
docker compose up -d

# Services:
# - API server:     http://localhost:8000
# - Dashboard:       http://localhost:8501
# - Prometheus:      http://localhost:9091
# - Grafana:         http://localhost:3000
```

---

##   Development

### Linux / macOS
```bash
# Install dev dependencies
make dev

# Run tests
make test

# Lint and format
make lint
make format
```

### Windows
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dev dependencies
python -m pip install -e ".[dev,dashboard]"
pip install ruff mypy pytest pytest-cov pytest-asyncio

# Run tests
pytest -v

# Run fast tests only
pytest -v -m "not slow"

# Lint and format
ruff check src/ tests/
ruff format --check src/ tests/
```

---

##   CLI Reference

```
Usage: betting-intel [OPTIONS] COMMAND [ARGS]...

Commands:
  pipeline         Pipeline management (run, status)
  models           Model management (list, info)
  backtest         Backtesting (run, report)
  api              API server (start)
  db               Database management (init, check)
  dashboard        Launch Streamlit dashboard
  web              Start FastAPI web GUI
  small-leagues    Small-league data (list, fetch, teams, bridge)
  recommendations  Generate bets (list, todays-card, tomorrow, clear-picks,
                   player-props)
```

##   Quick Reference

| Action | Command |
|--------|---------|
| Install (Linux/macOS) | `make install` |
| Install (Windows) | `powershell -File install.ps1` |
| Init database | `betting-intel db init` or `python scripts/init_db.py` |
| Run pipeline | `betting-intel pipeline run` |
| Start API | `betting-intel api start` |
| Launch dashboard | `betting-intel dashboard` |
| Run tests | `make test` (Linux) or `pytest -v` (Windows) |

---

##   Roadmap

- [x]  Live odds integration (ODDS API / TheOddsAPI) ✓
- [ ]  Live betting recommendations during games
- [ ]  Headless browser for Cloudflare-protected leagues
- [ ]  Telegram/Discord bot for real-time picks
- [ ]  Historical backtesting dashboard
- [ ]  Multi-sport support (soccer, tennis, esports)

---

##   License

MIT License — see [LICENSE](LICENSE) for details.

##   Disclaimer

**This software is for educational and research purposes only.** Sports betting involves financial risk. The predictions generated by this system are based on historical data and statistical models — they do not guarantee future results. Always gamble responsibly. Never bet more than you can afford to lose.
