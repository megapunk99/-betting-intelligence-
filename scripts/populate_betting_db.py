"""
Populate the betting_intel.db tables (Game, Bet, ModelVersion, PipelineRun)
from the prediction pipeline results.

Usage:
    python scripts/populate_betting_db.py
"""

import sys
import os
import json
import math
import warnings
from pathlib import Path
from datetime import datetime

# Ensure src/ is on path
_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
for p in (_project_root, _src):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ["LOG_LEVEL"] = "CRITICAL"
warnings.filterwarnings("ignore")

# Suppress logging from pipeline modules
import logging
logging.disable(logging.CRITICAL)

from betting_intel.db.connection import DatabaseManager
from betting_intel.db.schema import Game, Bet, ModelVersion, PipelineRun
from betting_intel.pipeline.pipeline import PredictionPipeline
from betting_intel.pipeline.cli import parse_args

print("=" * 65)
print("  POPULATE BETTING_INTEL.DB FROM PIPELINE")
print("=" * 65)


def pg(key, *alt_keys):
    """Get a value from a dict using multiple possible key names (case-insensitive)."""
    def _find(d, k):
        if k in d:
            return d[k]
        lower = k.lower()
        for dk in d:
            if dk.lower() == lower:
                return d[dk]
        return None
    for dikt in alt_keys:
        if isinstance(dikt, dict):
            for k in [key] + ([key] if isinstance(key, str) else []):
                val = _find(dikt, k)
                if val is not None:
                    return val
    return None


def get_val(d, *keys, default=None):
    """Get first existing key from dict."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def safe_float(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def parse_game_date(val, fallback=None):
    if val is None:
        return fallback or datetime.utcnow()
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            try:
                return datetime.strptime(val, "%Y-%m-%d")
            except (ValueError, TypeError):
                pass
    try:
        import pandas as pd
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime()
    except Exception:
        pass
    return fallback or datetime.utcnow()


# ── Step 1: Run the pipeline ───────────────────────────────────────────
print("\n[1/4] Running prediction pipeline...")
args = parse_args([
    "--days-history", "90",
    "--no-tune",
])
args.live = False

import time as _time
t0 = _time.time()
pipeline = PredictionPipeline(args)
results = pipeline.run()
elapsed = _time.time() - t0

print(f"  ✅ Pipeline complete in {elapsed:.1f}s")

# ── Step 2: Connect to betting_intel.db ────────────────────────────────
print("\n[2/4] Connecting to betting_intel.db...")
db_manager = DatabaseManager()
db_manager.create_tables()
session = db_manager.get_session()
print("  ✅ Connected")

# ── Step 3: Populate tables ─────────────────────────────────────────────
print("\n[3/4] Populating tables...")

now = datetime.utcnow()

# ── 3a. PipelineRun ─────────────────────────────────────────────────────
run_id = f"pipeline_{now.strftime('%Y%m%d_%H%M%S')}"
n_preds = len(results.get("predictions", []))
n_recs = len(results.get("recommendations", []))
n_clear = len(results.get("clear_picks", []))
n_props = len(results.get("player_props", []))

pipeline_run = PipelineRun(
    run_id=run_id,
    status="completed",
    started_at=now,
    completed_at=now,
    games_processed=n_preds,
    bets_generated=n_recs,
    summary=json.dumps({
        "mode": "historical",
        "days_history": 90,
        "recommendations": n_recs,
        "clear_picks": n_clear,
        "player_props": n_props,
        "ev_opportunities": len(results.get("ev_opportunities", [])),
        "arbitrage": len(results.get("arbitrage_opportunities", [])),
    }),
)
session.add(pipeline_run)
print(f"  ✅ PipelineRun: {run_id}")

# ── 3b. ModelVersion ───────────────────────────────────────────────────
models_dir = _project_root / "models" / "saved"
model_versions_data = []
added_stems = set()

# Check for EnhancedEnsemble model
ensemble_path = models_dir / "engine_ensemble.joblib"
if ensemble_path.exists():
    mv = ModelVersion(
        model_name="EnhancedEnsemble",
        version="3.0",
        artifact_path=str(ensemble_path),
        training_date=now,
        training_samples=n_preds,
        parameters=json.dumps({
            "type": "ensemble",
            "models": ["LightGBM", "Ridge", "MLP"],
            "days_history": 90,
        }),
        metrics_json=json.dumps({"n_predictions": n_preds}),
        feature_cols=json.dumps(pipeline.model_feature_cols if pipeline.model_feature_cols else []),
    )
    session.add(mv)
    model_versions_data.append("EnhancedEnsemble")
    added_stems.add("engine_ensemble")
    print("  ✅ ModelVersion: EnhancedEnsemble")

# Check for any other model files
for model_file in models_dir.glob("**/*"):
    if not model_file.is_file() or model_file.suffix not in (".joblib", ".pkl", ".pickle"):
        continue
    stem = model_file.stem.lower()
    if stem in added_stems:
        continue
    name_map = {
        "total_model": "TotalPointsPredictor",
        "ml_model": "MlPredictor",
        "calibrator": "ProbabilityCalibrator",
        "lgbm": "LightGBM",
    }
    model_name = name_map.get(stem, stem)
    mv = ModelVersion(
        model_name=model_name,
        version="1.0",
        artifact_path=str(model_file),
        training_date=now,
        training_samples=n_preds,
    )
    session.add(mv)
    model_versions_data.append(model_name)
    added_stems.add(stem)
    print(f"  ✅ ModelVersion: {model_name}")

if not model_versions_data:
    mv = ModelVersion(
        model_name="pipeline_ensemble",
        version="3.0",
        parameters=json.dumps({"type": "historical", "models_trained": True}),
        metrics_json=json.dumps({"n_predictions": n_preds}),
        feature_cols=json.dumps(pipeline.model_feature_cols if pipeline.model_feature_cols else []),
        training_date=now,
        training_samples=n_preds,
    )
    session.add(mv)
    model_versions_data.append("pipeline_ensemble")
    print("  ✅ ModelVersion: pipeline_ensemble (inferred)")

# ── 3c. Games (deduplicated by game_id) ────────────────────────────────
predictions = results.get("predictions", [])
games_added = 0
seen_game_ids = set()

print(f"\n  Debug: {len(predictions)} predictions available")

for pred in predictions:
    if not isinstance(pred, dict):
        continue

    # Try case-insensitive lookups for game_id
    game_id = get_val(pred, "GAME_ID", "game_id", "id", "")
    if not game_id:
        continue

    game_id = str(game_id)

    # Deduplicate
    if game_id in seen_game_ids:
        continue
    seen_game_ids.add(game_id)

    # Home team: try various key patterns
    home_team = get_val(pred, "home_team", "TEAM_NAME_home", "home_team_name", "HOME_TEAM", "")
    away_team = get_val(pred, "away_team", "TEAM_NAME_away", "away_team_name", "AWAY_TEAM", "")

    if not home_team and not away_team:
        continue

    game = Game(
        game_id=game_id,
        game_date=parse_game_date(get_val(pred, "game_date", "GAME_DATE", "DATE", "date")),
        season=safe_float(get_val(pred, "season", "SEASON", "SEASON_ID"), None),
        home_team_name=str(home_team),
        away_team_name=str(away_team),
        home_team_abbr=str(get_val(pred, "home_team_abbr", "TEAM_ABBREVIATION_home", "HOME_ABBR", "")) or None,
        away_team_abbr=str(get_val(pred, "away_team_abbr", "TEAM_ABBREVIATION_away", "AWAY_ABBR", "")) or None,
        home_score=safe_float(get_val(pred, "home_score", "HOME_SCORE", "home_pts"), None),
        away_score=safe_float(get_val(pred, "away_score", "AWAY_SCORE", "away_pts"), None),
        total_points=safe_float(get_val(pred, "total_points", "TOTAL_POINTS", "actual_total"), None),
        point_diff=safe_float(get_val(pred, "point_diff", "POINT_DIFF", "actual_margin"), None),
        predicted_total=safe_float(get_val(pred, "predicted_total", "PREDICTED_TOTAL"), None),
        predicted_spread=safe_float(get_val(pred, "predicted_spread", "PREDICTED_SPREAD"), None),
        prediction_confidence=safe_float(get_val(pred, "confidence", "CONFIDENCE"), None),
    )
    session.add(game)
    games_added += 1

print(f"  ✅ Games: {games_added} added ({len(seen_game_ids)} unique game_ids)")

# ── 3d. Bets ────────────────────────────────────────────────────────────
recommendations = results.get("recommendations", [])
bets_added = 0
for rec in recommendations:
    if not isinstance(rec, dict):
        continue

    team = rec.get("team", "")
    bet_type = rec.get("bet_type", "unknown")
    edge = safe_float(rec.get("edge"), 0.0)
    stake = safe_float(rec.get("stake", rec.get("recommended_stake")), 0.0)
    predicted_val = safe_float(rec.get("predicted_value", rec.get("predicted_total")), None)
    market_line = safe_float(rec.get("market_line", rec.get("market_total")), None)
    kelly = safe_float(rec.get("kelly_fraction"), 0.25)

    bet = Bet(
        game_id=rec.get("game_id", f"bet_{bets_added}"),
        game_date=now,
        strategy=rec.get("strategy", bet_type),
        model="pipeline_ensemble",
        bet_type=bet_type,
        matchup=rec.get("matchup", f"{team} - {bet_type}"),
        predicted_value=predicted_val,
        market_line=market_line,
        actual_value=None,
        edge_pct=edge,
        outcome=None,
        profit_units=None,
        kelly_fraction=kelly,
        stake_dollars=stake,
    )
    session.add(bet)
    bets_added += 1

print(f"  ✅ Bets: {bets_added} added")

# ── Commit all changes ──────────────────────────────────────────────────
print("\n[4/4] Committing to database...")
session.commit()
session.close()
db_manager.close()

print()
print("=" * 65)
print("  ✅ POPULATION COMPLETE")
print("=" * 65)
print(f"  PipelineRun:   1")
print(f"  ModelVersions: {len(model_versions_data)}")
print(f"  Games:         {games_added}")
print(f"  Bets:          {bets_added}")
print(f"  Database:      data/betting_intel.db")
print("=" * 65)
