# Betting Intelligence

Basketball prediction engine and web dashboard. Uses historical NBA data and market odds to identify edges, generate predictions, and track results.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn web.app:app --port 8000
```

Opens at http://localhost:8000.

## Requirements

- Python 3.10+
- [TheOddsAPI key](https://the-odds-api.com/) for live odds (optional — dashboard works with cached data)

## Project Structure

```
src/betting_intel/
  live/              Prediction engine (walk-forward, ensemble models)
  data/              Loaders, features, odds fetching
  models/            ML models (LightGBM, CatBoost, etc.)
  pipeline/          Orchestrator for data loading, training, prediction
  recommendations/   Bet engine — edge computation, Kelly staking, ranker
  web/               FastAPI dashboard with Jinja2 templates
  tools/             Forward testing, daily refresh, prediction logging
```

## Web Dashboard

Run `uvicorn web.app:app --reload --port 8000` for the development server.

Pages:

- `/` — Dashboard: stats, P&L chart, game cards, clear picks, results table
- `/future-predictions` — Upcoming game projections with quarter/half breakdowns

The dashboard auto-updates via WebSocket. Theme toggle (dark/light) persists to localStorage.

## Data Pipeline

1. Fetch historical game data from ESPN
2. Engineer features (rolling averages, ELO ratings, strength of schedule)
3. Train ensemble models (walk-forward cross-validation)
4. Fetch market odds from TheOddsAPI
5. Compute edges and generate bet recommendations with Kelly staking
6. Save predictions to JSON, served by the web dashboard

## License

MIT
