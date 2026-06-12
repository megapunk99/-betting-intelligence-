# CI/CD Pipeline Architecture

## Overview

The project has **7 GitHub Actions workflows** organized into two tiers:

### Tier 1: CI — Tests & Quality (Every Push / PR)

| Workflow | Triggers | When | What it does | Approx time |
|---|---|---|---|---|
| **ci.yml** | Push to main/master, PR | Every change | Lint → Tests (3 OS × 3 Python) → TypeCheck → Build | 3-5 min |
| **ci-tests.yml** | Push (any branch) | Every change | Quick tests (single config, fail-fast, no coverage) | 30s |

### Tier 2: Scheduled — Data, Models & Extended Tests (Nightly / Weekly / Monthly)

| Workflow | Triggers | When | What it does | Approx time |
|---|---|---|---|---|
| **ci-seed-database.yml** | Schedule (daily), workflow_dispatch | 3:00 AM UTC | Fetch NBA data from nba_api, cache for slow tests | 2-3 min |
| **ci-slow-tests.yml** | Schedule (daily), workflow_dispatch, push on core changes | 4:00 AM UTC | Run @pytest.mark.slow tests (model training tests) | 3 min |
| **daily_feature_refresh.yml** | Schedule (daily), workflow_dispatch | 6:00 AM UTC | Refresh features, retrain models (via tools/retrain_all.py), upload artifacts | varies |
| **weekly_retraining.yml** | Schedule (Monday), workflow_dispatch | 8:00 AM UTC | Retrain total + spread models (via tools/retrain_all.py --mode weekly) | 5-10 min |
| **monthly_full_retraining.yml** | Schedule (1st), workflow_dispatch | 10:00 AM UTC | Full retrain + calibration + report (via tools/retrain_all.py --mode monthly) | 10-15 min |

## Daily Timelines

```
03:00 ─ ci-seed-database.yml ─────── Fetch NBA data → cache DB
04:00 ─ ci-slow-tests.yml ────────── Restore cache → run slow tests (pytest -m "slow")
06:00 ─ daily_feature_refresh.yml ── Refresh features → retrain → upload models
08:00 ─ weekly_retraining.yml ────── (Mondays only) Retrain models (tools/retrain_all.py)
10:00 ─ monthly_full_retraining.yml ─ (1st only) Full retrain + report
```

## Pytest vs Custom Scripts

Not all workflows use pytest. Here is the breakdown:

- **ci.yml** — `pytest ... -m "not slow"` (unit tests, coverage)
- **ci-tests.yml** — `pytest ... -m "not slow"` (quick feedback, no coverage)
- **ci-slow-tests.yml** — `pytest ... -m "slow"` (model training / integration tests)
- **ci-seed-database.yml** — Custom `python scripts/populate_nba_data.py` (no pytest)
- **daily_feature_refresh.yml** — Custom `tools/retrain_all.py` (no pytest)
- **weekly_retraining.yml** — Custom `tools/retrain_all.py` (no pytest; slow-test validation uses pytest on manual trigger)
- **monthly_full_retraining.yml** — Custom `tools/retrain_all.py` (no pytest; slow-test validation uses pytest on manual trigger)

## Cache Architecture

```
ci-seed-database.yml                ci-slow-tests.yml / retraining workflows
┌──────────────────────┐            ┌──────────────────────────┐
│  nba_api: fetch data │            │  restore cache           │
│         ↓            │            │  nba-data-<run_id>       │
│  save data/nba_data.db│           │  restore-keys:           │
│         ↓            │  cache     │    nba-data-             │
│  actions/cache@v4    │ ────────→  │         ↓                │
│  key: nba-data-...   │            │  tools/check_db.py       │
│  restore-keys:       │            │  (verify ≥50 rows)       │
│    nba-data-         │            │         ↓                │
└──────────────────────┘            │  retrain / slow tests    │
                                    └──────────────────────────┘
```

All retraining workflows plus `daily_feature_refresh.yml` now restore the NBA database cache and run `tools/check_db.py` before proceeding.

## Consistency Status

### ✅ Resolved Issues

#### 1. Python Version Drift — ✅ FIXED
Standardized on Python 3.11 for all scheduled workflows; CI matrix stays as-is (3.10, 3.11, 3.12).

| Workflow | Python version(s) |
|---|---|
| ci.yml | 3.10 (env default) + matrix 3.10, 3.11, 3.12 |
| ci-tests.yml | 3.11 |
| ci-seed-database.yml | 3.11 |
| ci-slow-tests.yml | 3.10, 3.11 |
| daily_feature_refresh.yml | **3.11** (was 3.12) |
| weekly_retraining.yml | **3.11** (was 3.12) |
| monthly_full_retraining.yml | **3.11** (was 3.12) |

#### 2. Pip Caching — ✅ FIXED
All 7 workflows now use `setup-python@v5` built-in `cache: "pip"`. The redundant explicit `actions/cache@v4` for pip in `ci-slow-tests.yml` was removed.

#### 3. Missing Concurrency Groups — ✅ FIXED
Added `concurrency` groups to `daily_feature_refresh.yml`, `weekly_retraining.yml`, and `monthly_full_retraining.yml`. All 7 workflows now have concurrency control.

#### 4. Redundant Package Installs — ✅ FIXED
Removed manual `pip install xgboost lightgbm scikit-learn pandas numpy` from retraining workflows — all are already in `pyproject.toml`.

#### 5. Inline Python Code — ✅ FIXED
Extracted retraining logic to `tools/retrain_all.py` and DB validation to `tools/check_db.py`. Both are testable, lintable, and maintainable.

#### 6. Monthly Overwrites Spread Model — ✅ FIXED
`tools/retrain_all.py --mode monthly` now saves all three models: `total_model.pkl`, `spread_model.pkl`, and `win_model.pkl`.

#### 7. Missing Data Guards — ✅ FIXED
All workflows that depend on the NBA database now have a `tools/check_db.py` guard step before proceeding. If the DB is unavailable, the step is skipped with a clear notice message.

#### 8. ci-slow-tests.yml Redundant Pip Cache — ✅ FIXED
Removed the explicit `actions/cache@v4` for pip — `setup-python@v5 cache: "pip"` handles it.

### ❌ Remaining / Non-Actionable Issues

#### 9. No Secrets Validation
The retraining workflows reference `secrets.ODDS_API_KEY` and `secrets.NBA_API_KEY`. These fail gracefully (steps that depend on them use `|| echo "skipping"`). No further action needed.

#### 10. Retention Mismatch
Artifact retention varies: 30 days (daily_feature_refresh, ci-slow-tests) vs 90 days (weekly, monthly). This is intentional — weekly/monthly models are more valuable and used for longer periods. No change needed.
