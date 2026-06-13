# Contributing

Thanks for your interest in Betting Intelligence! This is a machine learning sports betting analytics platform built with Python.

## Getting Started

1. **Fork** the repo and clone locally
2. **Set up** a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```
3. **Copy the env template:**
   ```bash
   cp .env.example .env
   ```
4. **Run the tests:**
   ```bash
   pytest tests/
   ```
5. **Start the dashboard:**
   ```bash
   make run-api
   # or: uvicorn web.app:app --reload --port 8000
   ```

## Project Structure

```
src/betting_intel/
  live/              Prediction engine (walk-forward, ensemble models)
  data/              Loaders, features, odds fetching
  models/            ML models (LightGBM, CatBoost, XGBoost, ensembles)
  pipeline/          Orchestrator for data loading, training, prediction
  features/          Feature engineering (market inefficiency, etc.)
  db/                Database connection and schema
  recommendations/   Bet engine — edge computation, Kelly staking, ranker
  analytics/         Performance tracker, CLV computation, alerts
  api/               REST API routes (FastAPI)
  cli/               Click-based CLI tool
  config/            Settings (pydantic-settings + .env)

web/                 FastAPI web dashboard (Jinja2 templates)
tests/               Pytest test suite
tools/               Scripts for daily refresh, backfill, forward testing
```

## Coding Standards

- **Python 3.10+** with type hints on all public functions
- Follow the style of existing code — `ruff` will catch most issues
- All new features should include tests (we maintain ~350+ tests)
- Use `make lint` and `make format` before committing

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_market_odds_store.py -v

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

All tests use in-memory SQLite and mocked external APIs — no network required.

## Pull Request Process

1. Create a feature branch (`git checkout -b feature/my-feature`)
2. Make your changes with tests
3. Run the full test suite: `pytest tests/`
4. Push and open a PR
5. Keep PRs focused — one feature or fix per PR

## Questions?

Open a GitHub Issue for bugs, feature requests, or questions.
