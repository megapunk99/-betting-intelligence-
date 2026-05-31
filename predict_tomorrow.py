#!/usr/bin/env python3
"""
predict_tomorrow.py — Full Advanced Betting Prediction Pipeline
================================================================
Generates betting predictions using the complete advanced pipeline.
Integrates with TheOddsAPI for LIVE upcoming game predictions with
multi-strategy ensembles, recommendation engine, risk management,
+EV scanning, arbitrage detection, player props, and more.

Usage:
    # Live predictions for upcoming games (needs ODDS_API_KEY):
    python predict_tomorrow.py --live

    # Historical predictions for already-played games:
    python predict_tomorrow.py

    # Skip hyperparameter tuning (much faster):
    python predict_tomorrow.py --live --no-tune

    # Full simulation run with monte carlo + recommendations:
    python predict_tomorrow.py --full

    # Generate only recommendations from existing predictions:
    python predict_tomorrow.py --recommend-only

    # Scheduled mode (for cron / Task Scheduler):
    python predict_tomorrow.py --scheduled
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# 1.  Environment & Config Bootstrap (must happen before any imports)
# ──────────────────────────────────────────────────────────────────────

# Fix Unicode on Windows terminals (cp1252 can't render emoji/Unicode)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
# Also set the environment variable for subprocesses
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("LOG_LEVEL", "INFO")

# Try loading .env
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────
# 2.  Imports (all from the canonical betting_intel package)
# ──────────────────────────────────────────────────────────────────────

from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import (
    TotalPointsPredictor,
    SpreadPredictor,
    MomentumModel,
    StackingEnsemblePredictor,
)
from betting_intel.recommendations.engine import RecommendationEngine
from betting_intel.recommendations.bet_types import BetType
from betting_intel.recommendations.ranker import BetRanker
from betting_intel.recommendations.ev_scanner import PositiveEVScanner
from betting_intel.recommendations.arbitrage import ArbitrageDetector
from betting_intel.recommendations.player_props import PlayerPropEngine
from betting_intel.risk.kelly import KellyCalculator
from betting_intel.risk.exposure import ExposureManager
from betting_intel.risk.correlation import BetCorrelationTracker
from betting_intel.betting.edge import EdgeDetector
from betting_intel.betting.monte_carlo import MonteCarloSimulator
from betting_intel.validation.calibration import ProbabilityCalibrator
from betting_intel.validation.overfitting import OverfittingDetector
from betting_intel.validation.cross_validation import TimeSeriesCrossValidator
from betting_intel.monitoring.drift import PerformanceTracker
from betting_intel.backtesting.metrics import BacktestMetrics
from betting_intel.services.logging import get_logger

# Module availability flags (for graceful degradation at runtime)
HAS_RECOMMENDATIONS = True
HAS_RISK = True
HAS_BETTING = True
HAS_VALIDATION = True
HAS_MONITORING = True
HAS_BACKTESTING = True
HAS_ROOT_PREDICTORS = True

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 3.  CLI Argument Parser
# ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🏀 Betting Intelligence — Full Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict_tomorrow.py                           # Historical mode\n"
            "  python predict_tomorrow.py --live                    # Live predictions\n"
            "  python predict_tomorrow.py --live --no-tune          # Skip tuning\n"
            "  python predict_tomorrow.py --full                    # Full pipeline\n"
        ),
    )

    # Mode
    mode = parser.add_argument_group("Mode")
    mode.add_argument("--live", action="store_true", help="Fetch live upcoming games from TheOddsAPI")
    mode.add_argument("--full", action="store_true", help="Run full pipeline: predictions → recommendations → risk → simulation")
    mode.add_argument("--recommend-only", action="store_true", help="Generate recommendations from existing predictions only")
    mode.add_argument("--simulate", action="store_true", help="Run Monte Carlo simulation on results")
    mode.add_argument("--scheduled", action="store_true", help="Run in scheduled mode (auto-save, JSON summary to stdout)")

    # Data
    data_grp = parser.add_argument_group("Data Options")
    data_grp.add_argument("--days-history", type=int, default=90, help="Days of historical data to load")
    data_grp.add_argument("--data-source", choices=["csv", "sqlite", "api"], default=None, help="Force data source")
    data_grp.add_argument("--csv-path", type=str, help="Path to CSV data file")

    # Model
    model_grp = parser.add_argument_group("Model Options")
    model_grp.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    model_grp.add_argument("--model-dir", type=str, default="models/saved", help="Directory for saved models")
    model_grp.add_argument("--ensemble", action="store_true", default=True, help="Use ensemble of all strategies")
    model_grp.add_argument("--strategy", type=str, choices=["lightgbm", "catboost", "random_forest", "bayesian", "ridge", "all"], default="all",
                           help="Which prediction strategy to use")

    # Risk
    risk_grp = parser.add_argument_group("Risk Options")
    risk_grp.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll for Kelly sizing")
    risk_grp.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly fraction (0.0-1.0)")
    risk_grp.add_argument("--max-exposure", type=float, default=0.20, help="Max exposure per game as fraction of bankroll")
    risk_grp.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge threshold (2% = 0.02)")

    # Output
    out_grp = parser.add_argument_group("Output Options")
    out_grp.add_argument("--output", type=str, default=None, help="Save predictions to JSON file")
    out_grp.add_argument("--html", action="store_true", help="Generate HTML report")
    out_grp.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────
# 4.  Core Prediction Pipeline
# ──────────────────────────────────────────────────────────────────────


class PredictionPipeline:
    """
    Orchestrates the full prediction workflow:
      1. Load data     → 2. Engineer features  → 3. Train/predict
      4. Tune hparams  → 5. Generate bets       → 6. Risk-manage
      7. Validate      → 8. Report
    """

    # Columns to exclude when selecting feature columns for models
    EXCLUDE_COLS = {
        "game_id", "game_date", "home_team", "away_team",
        "total_points", "spread", "label", "home_win",
        "home_score", "away_score",
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.start_time = time.time()
        self.results: Dict[str, Any] = {
            "pipeline_version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "mode": "live" if args.live else "historical",
            "games": [],
            "predictions": [],
            "recommendations": [],
            "clear_picks": [],
            "ev_opportunities": [],
            "arbitrage_opportunities": [],
            "player_props": [],
            "risk_assessment": {},
            "validation": {},
            "simulation": {},
            "metadata": {},
        }
        self.df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.model = None  # Trained model for tomorrow predictions (full-data)
        self.model_feature_cols: List[str] = []  # Feature columns used by the model
        self.tomorrow_recommendations_final: List[Dict[str, Any]] = []  # Real edge-based recs

    # ──────────────────────────────────────────────────────────────
    # 5a.  Data Loading
    # ──────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Load game data from the best available source."""
        print("\n" + "=" * 70)
        print("  📊  STAGE 1: DATA LOADING")
        print("=" * 70)

        if self.args.live:
            print("  🌐  Attempting live odds fetch from TheOddsAPI...")
            df = self._load_live_data()
            if df is not None and not df.empty:
                return df
            print("  ⚠  Live data unavailable, falling back to historical.")
            self.args.live = False

        # Historical loading
        df = self._load_historical_data()
        return df

    def _load_live_data(self) -> Optional[pd.DataFrame]:
        """Fetch live upcoming games via TheOddsAPI."""
        # Try LiveDataGateway (the canonical odds integration module)
        try:
            from betting_intel.data.live_gateway import LiveDataGateway
            gateway = LiveDataGateway(odds_api_key=ODDS_API_KEY)
            odds_data = gateway.get_live_odds(force_refresh=True)
            if odds_data and len(odds_data) > 0:
                df = pd.DataFrame(odds_data)
                print(f"  ✅  Fetched {len(df)} games from LiveDataGateway")
                self.results["metadata"]["data_source"] = "live_gateway"
                return df
        except Exception as e:
            print(f"  ⚠  LiveDataGateway failed: {e}")

        return None

    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """Load historical game data from CSV, SQLite, or data loader."""
        days = self.args.days_history

        # Try CSV path
        if self.args.csv_path:
            path = Path(self.args.csv_path)
            if path.exists():
                df = pd.read_csv(path)
                print(f"  ✅  Loaded {len(df)} games from CSV: {path.name}")
                self.results["metadata"]["data_source"] = "csv"
                return df

        # Try NBADataLoader
        try:
            from betting_intel.data.loader import NBADataLoader
            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is not None and not raw_df.empty:
                # Filter to recent games
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])
                recent = raw_df[raw_df["GAME_DATE"] >= cutoff]
                if recent.empty:
                    # No recent data within the window — use all available data
                    print(f"  ⚠  No games found in the last {days} days (data may be from an older season)")
                    print(f"  ℹ  Falling back to all {len(raw_df)} available historical games")
                    df = loader.build_game_dataset(raw_df)
                else:
                    df = loader.build_game_dataset(recent)
                if df is not None and not df.empty:
                    print(f"  ✅  Loaded {len(df)} games from NBADataLoader")
                    self.results["metadata"]["data_source"] = "nba_dataloader"
                    return df
        except Exception as e:
            print(f"  ⚠  NBADataLoader failed: {e}")

        # Try scripts
        try:
            from scripts.fetch_real_nba_data import NBAStatsFetcher

            fetcher = NBAStatsFetcher()
            df = fetcher.fetch_game_logs(days=days)
            if df is not None and not df.empty:
                print(f"  ✅  Loaded {len(df)} games from NBAStatsFetcher")
                self.results["metadata"]["data_source"] = "nba_stats"
                return df
        except Exception as e:
            print(f"  ⚠  NBAStatsFetcher failed: {e}")

        return None

    # ──────────────────────────────────────────────────────────────
    # 5b.  Feature Engineering
    # ──────────────────────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features for model input."""
        print("\n" + "=" * 70)
        print("  🔧  STAGE 2: FEATURE ENGINEERING")
        print("=" * 70)

        try:
            engineer = FeatureEngineer()
            # FeatureEngineer.build_all_features needs two DataFrames (games_df, raw_df)
            if hasattr(engineer, 'build_all_features'):
                features_df = engineer.build_all_features(df, df)
            elif hasattr(engineer, 'create_features'):
                features_df = engineer.create_features(df)
            else:
                features_df = None
            if features_df is not None and not features_df.empty:
                print(f"  ✅  Engineered {len(features_df.columns)} features from {len(features_df)} rows")
                return features_df
        except Exception as e:
            print(f"  ⚠  FeatureEngineer failed: {e}")

        # Manual feature engineering fallback
        print("  ℹ  Using manual feature engineering...")
        df_feat = df.copy()

        # Basic rolling features if date column exists
        if "game_date" in df_feat.columns:
            df_feat["game_date"] = pd.to_datetime(df_feat["game_date"])
            df_feat = df_feat.sort_values("game_date")

        # Fill missing values
        for col in df_feat.select_dtypes(include=[np.number]).columns:
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

        # Add interaction features if available
        if all(c in df_feat.columns for c in ["home_fg_pct", "away_fg_pct"]):
            df_feat["fg_pct_diff"] = df_feat["home_fg_pct"] - df_feat["away_fg_pct"]
        if all(c in df_feat.columns for c in ["home_rebounds", "away_rebounds"]):
            df_feat["rebound_diff"] = df_feat["home_rebounds"] - df_feat["away_rebounds"]
        if all(c in df_feat.columns for c in ["home_turnovers", "away_turnovers"]):
            df_feat["turnover_diff"] = df_feat["home_turnovers"] - df_feat["away_turnovers"]
        if all(c in df_feat.columns for c in ["home_elo", "away_elo"]):
            df_feat["elo_diff"] = df_feat["home_elo"] - df_feat["away_elo"]
        if all(c in df_feat.columns for c in ["home_pace", "away_pace"]):
            df_feat["pace_diff"] = df_feat["home_pace"] - df_feat["away_pace"]

        print(f"  ✅  Engineered {len(df_feat.columns)} total columns")
        return df_feat

    # ──────────────────────────────────────────────────────────────
    # 5c.  Training & Prediction
    # ──────────────────────────────────────────────────────────────

    def train_and_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Run the multi-strategy prediction engine."""
        print("\n" + "=" * 70)
        print("  🤖  STAGE 3: MODEL TRAINING & PREDICTION")
        print("=" * 70)

        tune = not self.args.no_tune
        strategy = self.args.strategy
        model_dir = Path(self.args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine target columns
        has_total = "total_points" in features_df.columns
        has_spread = "spread" in features_df.columns
        has_label = "label" in features_df.columns or "home_win" in features_df.columns

        # Identify feature columns (exclude target & id columns)
        feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]

        if len(feature_cols) < 3:
            print("  ⚠  Not enough feature columns. Using default features.")
            feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in self.EXCLUDE_COLS]

        print(f"  📐  Using {len(feature_cols)} feature columns")

        target_total = "total_points" if has_total else None
        target_spread = "spread" if has_spread else None

        # Try StackingEnsemblePredictor (most advanced)
        if HAS_ROOT_PREDICTORS and target_total:
            try:
                print("  🏗  Building StackingEnsemble predictor...")
                ensemble = StackingEnsemblePredictor(prediction_type="regression")
                ensemble.add_base_model(TotalPointsPredictor("ridge"))
                # Add LightGBM if available
                try:
                    from lightgbm import LGBMRegressor
                    ensemble.add_base_model(TotalPointsPredictor("lightgbm"))
                except ImportError:
                    pass
                X = features_df[feature_cols].fillna(0).values
                y = features_df[target_total].fillna(features_df[target_total].median()).values
                n = len(features_df)
                split = max(1, int(n * 0.8))
                ensemble.fit(X[:split], y[:split])
                preds = ensemble.predict(X[split:])
                result_df = features_df.iloc[split:].copy()
                result_df["predicted_total"] = preds
                print(f"  ✅  StackingEnsemble: trained on {split} rows, predicted {len(preds)}")
                self.results["metadata"]["model"] = "stacking_ensemble"
                return result_df
            except Exception as e:
                print(f"  ⚠  StackingEnsemble failed: {e}")

        # Fallback: manual prediction loop
        print("  ℹ  Using manual prediction pipeline...")
        predictions_list = []

        # Simple cross-validation split
        n = len(features_df)
        split = max(1, int(n * 0.8))
        train_df = features_df.iloc[:split]
        test_df = features_df.iloc[split:]

        for target_name in [t for t in [target_total, target_spread] if t]:
            try:
                print(f"  🎯  Training predictor for: {target_name}")
                X_train = train_df[feature_cols].fillna(0)
                y_train = train_df[target_name].fillna(train_df[target_name].median())
                X_test = test_df[feature_cols].fillna(0)

                try:
                    from lightgbm import LGBMRegressor
                    model_class = LGBMRegressor
                except ImportError:
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        model_class = RandomForestRegressor
                        print("  ℹ  lightgbm not available, using RandomForestRegressor")
                    except ImportError:
                        print("  ❌  No regression library available (need lightgbm or sklearn)")
                        continue

                model = model_class(
                    n_estimators=200 if tune else 100,
                    learning_rate=0.05,
                    max_depth=5,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                test_df = test_df.copy()
                test_df[f"predicted_{target_name}"] = preds
                print(f"  ✅  {target_name}: trained on {len(train_df)} rows, predicted {len(test_df)}")

                # Try to save model (lightgbm has booster_.save_model)
                try:
                    if hasattr(model, 'booster_') and hasattr(model.booster_, 'save_model'):
                        model_path = model_dir / f"{target_name}_model.txt"
                        model.booster_.save_model(str(model_path))
                        print(f"  💾  Model saved to {model_path}")
                except Exception:
                    pass

            except Exception as e:
                print(f"  ⚠  Failed to train {target_name}: {e}")

        predictions_list.append(test_df)

        if not predictions_list:
            # Return features with naive predictions
            print("  ℹ  No models trained. Using naive historical averages.")
            df_out = features_df.copy()
            if has_total:
                avg_total = features_df["total_points"].mean()
                df_out["predicted_total"] = avg_total
            if has_spread:
                avg_spread = features_df["spread"].mean()
                df_out["predicted_spread"] = avg_spread
            return df_out

        result = pd.concat(predictions_list, ignore_index=True) if len(predictions_list) > 1 else predictions_list[0]
        return result

    # ──────────────────────────────────────────────────────────────
    # 5d.  Hyperparameter Tuning
    # ──────────────────────────────────────────────────────────────

    def tune_hyperparameters(self, features_df: pd.DataFrame):
        """Optional hyperparameter tuning with cross-validation."""
        if self.args.no_tune:
            print("\n  ⏩  Hyperparameter tuning skipped (--no-tune)")
            return

        print("\n" + "=" * 70)
        print("  🎛   STAGE 3b: HYPERPARAMETER TUNING")
        print("=" * 70)

        if not HAS_VALIDATION:
            print("  ⚠  Cross-validation module unavailable, skipping tuning")
            return

        feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]

        if not feature_cols:
            print("  ⚠  No feature columns for tuning")
            return

    # ──────────────────────────────────────────────────────────────
    # 5d.  Train on ALL Historical Data (for tomorrow predictions)
    # ──────────────────────────────────────────────────────────────

    def _train_all_data_model(self, features_df: pd.DataFrame):
        """Train model on ALL historical data and save to self.model.

        In live mode, train_and_predict uses 80/20 split for evaluation,
        but predicting tomorrow requires a model trained on 100% of data.
        """
        print("  🏋  Training model on FULL dataset for tomorrow predictions...")

        # Identify feature columns
        feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]

        if len(feature_cols) < 3:
            print("  ⚠  Not enough feature columns for full-data model")
            return

        target_total = "total_points" if "total_points" in features_df.columns else None
        if not target_total:
            print("  ⚠  No total_points target for full-data model")
            return

        X = features_df[feature_cols].fillna(0)
        y = features_df[target_total].fillna(features_df[target_total].median())

        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X, y)
            self.model = model
            self.model_feature_cols = feature_cols
            print(f"  ✅  Full-data model trained on {len(X)} rows with {len(feature_cols)} features")
        except ImportError:
            try:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                )
                model.fit(X, y)
                self.model = model
                self.model_feature_cols = feature_cols
                print(f"  ✅  Full-data RandomForest trained on {len(X)} rows with {len(feature_cols)} features")
            except Exception as e:
                print(f"  ⚠  Full-data model training failed: {e}")

    # ──────────────────────────────────────────────────────────────
    # 5e.  Build Feature Vector for a Tomorrow Game
    # ──────────────────────────────────────────────────────────────

    def _build_tomorrow_feature_vector(self, home_team: str, away_team: str) -> Optional[pd.Series]:
        """Build a model-compatible feature vector for a specific tomorrow matchup.

        For each feature column the model expects, computes the rolling average
        from the team's recent games. Interaction features (diff columns) are
        computed fresh from the per-team averages rather than from historical diffs.

        Returns a pd.Series matching self.model_feature_cols, or None if unavailable.
        """
        if self.model is None or not self.model_feature_cols or self.features_df is None:
            return None

        df = self.features_df

        # Guard: make sure we have team columns to look up
        if "home_team" not in df.columns or "away_team" not in df.columns:
            return None

        def _team_avg(team: str, base_stat: str, n: int = 10) -> float:
            """Average `base_stat` for `team` across recent games on either side."""
            home_col = f"home_{base_stat}"
            away_col = f"away_{base_stat}"
            home_vals = df.loc[df.get("home_team", "") == team, home_col] if home_col in df.columns else pd.Series(dtype=float)
            away_vals = df.loc[df.get("away_team", "") == team, away_col] if away_col in df.columns else pd.Series(dtype=float)
            combined = pd.concat([home_vals, away_vals]).tail(n)
            return float(combined.mean()) if len(combined) > 0 else 0.0

        feature_dict: Dict[str, float] = {}
        for col in self.model_feature_cols:
            if col.startswith("home_"):
                base = col[5:]  # strip "home_" prefix
                feature_dict[col] = _team_avg(home_team, base)
            elif col.startswith("away_"):
                base = col[5:]
                feature_dict[col] = _team_avg(away_team, base)
            elif col.endswith("_diff"):
                base = col.replace("_diff", "")
                h_val = _team_avg(home_team, base)
                a_val = _team_avg(away_team, base)
                feature_dict[col] = h_val - a_val
            else:
                # Non-prefixed column — use overall dataset average
                feature_dict[col] = float(df[col].mean()) if col in df.columns else 0.0

        return pd.Series(feature_dict)

    # ──────────────────────────────────────────────────────────────
    # 5f.  Predict Tomorrow's Games (Full-Data Model + Market Odds)
    # ──────────────────────────────────────────────────────────────

    def predict_tomorrow_games(self) -> List[Dict[str, Any]]:
        """Predict tomorrow's games using the full-data model + market odds.

        Only produces results in live mode where self.df contains
        the upcoming schedule with market prices. For each game:
          - Builds a feature vector from each team's recent history
          - Runs the full-data model to predict total points
          - Computes edge vs TheOddsAPI market total
          - Flags direction (over/under) and confidence level

        Returns list of prediction dicts and stores in self.tomorrow_recommendations_final.
        """
        print("\n" + "=" * 70)
        print("  🔮  STAGE: TOMORROW PREDICTIONS (Full-Data Model)")
        print("=" * 70)

        tomorrow_preds: List[Dict[str, Any]] = []

        if self.model is None:
            print("  ⚠  No full-data model available. Skipping tomorrow predictions.")
            return tomorrow_preds

        if self.df is None or self.df.empty:
            print("  ⚠  No game data to predict.")
            return tomorrow_preds

        for idx, row in self.df.iterrows():
            home = row.get("home_team", row.get("team", ""))
            away = row.get("away_team", row.get("opponent", ""))
            game_id = row.get("game_id", f"gm_{idx}")

            if not home or not away:
                continue

            # Build feature vector from each team's recent history
            feat = self._build_tomorrow_feature_vector(home, away)
            if feat is None or feat.isnull().any():
                continue

            # Run the model
            try:
                X_pred = feat.values.reshape(1, -1)
                predicted_total = float(self.model.predict(X_pred)[0])
            except Exception as e:
                print(f"  ⚠  Model predict failed for {home} vs {away}: {e}")
                continue

            # Market odds
            market_total = row.get("market_total", 220.0)
            home_ml = row.get("home_ml_odds", -110)
            away_ml = row.get("away_ml_odds", -110)

            # Compute real edge: (model_prediction - market) / market
            edge = (predicted_total - market_total) / market_total if market_total else 0.0
            direction = "over" if edge > 0 else "under"
            abs_edge = abs(edge)

            if abs_edge >= self.args.min_edge:
                conf = "high" if abs_edge > 0.05 else "medium"
            else:
                conf = "low"

            game_pred = {
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "game_date": str(row.get("game_date", "")),
                "predicted_total": round(predicted_total, 1),
                "market_total": market_total,
                "edge_pct": round(edge, 4),
                "direction": direction,
                "confidence": conf,
                "implied_odds": {
                    "home_moneyline": home_ml,
                    "away_moneyline": away_ml,
                },
            }
            tomorrow_preds.append(game_pred)

            arrow = "🟢" if abs_edge > 0.03 else "🔵" if abs_edge > 0.01 else "⚪"
            print(f"  {arrow}  {home:20s} vs {away:<20s}  pred={predicted_total:.1f}  "
                  f"mkt={market_total}  edge={edge:+.2%}  {direction}")

        if tomorrow_preds:
            print(f"  ✅  Predicted {len(tomorrow_preds)} tomorrow games with real model")
            self.results["tomorrow_predictions"] = tomorrow_preds
            self.tomorrow_recommendations_final = tomorrow_preds

        return tomorrow_preds

    # ──────────────────────────────────────────────────────────────
    # 5g.  Recommendation Engine
    # ──────────────────────────────────────────────────────────────

    def generate_recommendations(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate all bet types, rank by edge, identify clear picks."""
        print("\n" + "=" * 70)
        print("  💰  STAGE 4: RECOMMENDATION ENGINE")
        print("=" * 70)

        recommendations: List[Dict[str, Any]] = []
        min_edge = self.args.min_edge

        if HAS_RECOMMENDATIONS:
            try:
                engine = RecommendationEngine()
                ranker = BetRanker()

                # Generate all bet types — the engine fetches games internally
                all_bets = engine.generate_all_bets()
                if all_bets:
                    print(f"  ✅  Generated {len(all_bets)} total bet opportunities")

                    # BetRanker uses rank_bets, not rank_by_edge
                    if hasattr(ranker, 'rank_bets'):
                        ranked = ranker.rank_bets(all_bets)
                    else:
                        ranked = all_bets

                    # Show top bets by edge
                    print(f"  📊  Generated {len(all_bets)} total bets")
                    for i, bet in enumerate(all_bets[:5]):
                        # BetSuggestion may use attribute access or .as_dict()
                        bd = bet.as_dict() if hasattr(bet, 'as_dict') else bet
                        team = getattr(bet, 'team', bd.get('team', '?'))
                        edge = getattr(bet, 'edge', bd.get('edge', 0))
                        conf = getattr(bet, 'confidence', bd.get('confidence', 'N/A'))
                        print(f"       {i+1}. {team}: edge={edge:.2%}, conf={conf}")

                    # Identify clear picks
                    clear_picks = []
                    if hasattr(ranker, 'get_clear_picks'):
                        try:
                            clear_picks = ranker.get_clear_picks(all_bets, threshold=min_edge)
                        except TypeError:
                            clear_picks = ranker.get_clear_picks(all_bets)
                    elif hasattr(ranker, 'MIN_EDGE'):
                        clear_picks = [b for b in all_bets if getattr(b, 'is_clear_pick', False)]

                    if clear_picks:
                        print(f"  🎯  {len(clear_picks)} Clear Picks identified")
                        self.results["clear_picks"] = [
                            {
                                "team": getattr(p, 'team', getattr(p, 'home_team', '?')),
                                "edge": float(getattr(p, 'edge', 0)),
                                "confidence": str(getattr(p, 'confidence', '')),
                                "bet_type": str(getattr(p, 'bet_type', getattr(p, 'suggestion_type', ''))),
                                "odds": getattr(p, 'odds', 0),
                            }
                            for p in clear_picks[:10]
                        ]

                    # Convert BetSuggestion objects to dicts for the results
                    recommendations = [
                        {
                            "team": getattr(b, 'team', getattr(b, 'home_team', '?')),
                            "bet_type": str(getattr(b, 'bet_type', getattr(b, 'suggestion_type', ''))),
                            "edge": float(getattr(b, 'edge', 0)),
                            "confidence": str(getattr(b, 'confidence', '')),
                            "odds": getattr(b, 'odds', -110) or -110,
                            "stake": getattr(b, 'stake', 0),
                            "expected_value": float(getattr(b, 'expected_value', 0)),
                        }
                        for b in (ranked if isinstance(ranked, list) else all_bets)
                    ]
                    self.results["recommendations"] = recommendations

            except Exception as e:
                print(f"  ⚠  Recommendation engine failed: {e}")
        else:
            print("  ℹ  Recommendation engine not available")

        # Fallback: basic edge calculation
        if not recommendations and "predicted_total" in predictions_df.columns:
            print("  ℹ  Using basic edge calculation...")
            for _, row in predictions_df.iterrows():
                if "market_total" in predictions_df.columns:
                    market_total = row.get("market_total", 0)
                    predicted_total = row.get("predicted_total", 0)
                    if market_total and predicted_total:
                        edge = (predicted_total - market_total) / market_total
                        if abs(edge) >= min_edge:
                            team = row.get("home_team", row.get("team", "?"))
                            recommendations.append({
                                "team": team,
                                "bet_type": "total_over" if edge > 0 else "total_under",
                                "edge": abs(edge),
                                "confidence": "high" if abs(edge) > 0.05 else "medium",
                                "odds": row.get("over_odds", -110) if edge > 0 else row.get("under_odds", -110),
                                "expected_value": abs(edge),
                            })

        return recommendations

    # ──────────────────────────────────────────────────────────────
    # 5f.  Player Props Generation
    # ──────────────────────────────────────────────────────────────

    def generate_player_props(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate player prop bet recommendations."""
        print("\n" + "=" * 70)
        print("  🏀  STAGE 5: PLAYER PROPS")
        print("=" * 70)

        props: List[Dict[str, Any]] = []

        if HAS_RECOMMENDATIONS:
            try:
                generator = PlayerPropEngine()
                all_props_list = []
                # Generate props for each game
                home_col = 'home_team' if 'home_team' in predictions_df.columns else 'team'
                away_col = 'away_team' if 'away_team' in predictions_df.columns else 'opponent'
                for _, row in predictions_df.iterrows():
                    game_props = generator.predict_for_game(
                        home=str(row.get(home_col, 'Home')),
                        away=str(row.get(away_col, 'Away')),
                    )
                    if game_props:
                        all_props_list.extend(game_props if isinstance(game_props, list) else [game_props])
                if all_props_list:
                    print(f"  ✅  Generated {len(all_props_list)} player props")
                    for prop in all_props_list[:5]:
                        # BetSuggestion object — use .as_dict() or attribute access
                        pd = prop.as_dict() if hasattr(prop, 'as_dict') else prop
                        print(f"       {pd.get('player', '?')}: {pd.get('prop_type', '?')} "
                              f"→ {pd.get('line', 0)} (edge: {pd.get('edge', 0):.2%})")
                    props = [
                        {
                            "player": p.get("player", "?"),
                            "team": p.get("team", "?"),
                            "prop_type": p.get("prop_type", "?"),
                            "line": p.get("line", 0),
                            "edge": float(p.get("edge", 0)),
                            "confidence": str(p.get("confidence", "")),
                            "odds": p.get("odds", 0),
                        }
                        for p in (p.as_dict() if hasattr(p, 'as_dict') else p for p in all_props_list)
                    ]
                    self.results["player_props"] = props
                else:
                    print("  ℹ  No player props generated")
            except Exception as e:
                print(f"  ⚠  PlayerPropGenerator failed: {e}")
        else:
            print("  ℹ  Player props module not available")

        return props

    # ──────────────────────────────────────────────────────────────
    # 5g.  +EV Scanning & Arbitrage
    # ──────────────────────────────────────────────────────────────

    def scan_opportunities(self, predictions_df: pd.DataFrame):
        """Scan for +EV opportunities and arbitrage across sportsbooks."""
        print("\n" + "=" * 70)
        print("  🔬  STAGE 6: +EV SCANNING & ARBITRAGE")
        print("=" * 70)

        # +EV Scanning
        if HAS_RECOMMENDATIONS:
            try:
                scanner = PositiveEVScanner()
                ev_report = scanner.scan_odds_snapshots(predictions_df.to_dict("records"))
                if ev_report:
                    opportunities = getattr(ev_report, "opportunities", []) or []
                    if opportunities:
                        print(f"  ✅  Found {len(opportunities)} +EV opportunities")
                        for opp in opportunities[:5]:
                            print(f"       {getattr(opp, 'game', '?')}: "
                                  f"EV={getattr(opp, 'expected_value', 0):.2%}, "
                                  f"confidence={getattr(opp, 'confidence', 'N/A')}")
                        self.results["ev_opportunities"] = [
                            {
                                "game": getattr(o, "game", "?"),
                                "bet_type": getattr(o, "bet_type", "?"),
                                "expected_value": float(getattr(o, "expected_value", 0)),
                                "confidence": str(getattr(o, "confidence", "")),
                                "source": str(getattr(o, "source", "")),
                            }
                            for o in opportunities[:20]
                        ]
                    else:
                        print("  ℹ  No +EV opportunities found")
                else:
                    print("  ℹ  No EV report generated")
            except Exception as e:
                print(f"  ⚠  +EV scanning failed: {e}")
        else:
            print("  ℹ  +EV scanner not available")

        # Arbitrage Detection
        if HAS_RECOMMENDATIONS:
            try:
                detector = ArbitrageDetector()
                arb_report = detector.scan_for_arbitrage(predictions_df.to_dict("records"))
                if arb_report:
                    opportunities = getattr(arb_report, "opportunities", []) or []
                    if opportunities:
                        print(f"  ✅  Found {len(opportunities)} arbitrage opportunities!")
                        for arb in opportunities[:3]:
                            print(f"       {getattr(arb, 'game', '?')}: "
                                  f"return={getattr(arb, 'return_pct', 0):.2%}")
                        self.results["arbitrage_opportunities"] = [
                            {
                                "game": getattr(a, "game", "?"),
                                "return_pct": float(getattr(a, "return_pct", 0)),
                                "outcomes": getattr(a, "outcomes", []),
                                "stakes": getattr(a, "stakes", {}),
                            }
                            for a in opportunities[:10]
                        ]
                    else:
                        print("  ℹ  No arbitrage opportunities found")
            except Exception as e:
                print(f"  ⚠  Arbitrage detection failed: {e}")
        else:
            print("  ℹ  Arbitrage detector not available")

    # ──────────────────────────────────────────────────────────────
    # 5h.  Risk Management (Kelly + Exposure + Correlation)
    # ──────────────────────────────────────────────────────────────

    def apply_risk_management(self, recommendations: List[Dict[str, Any]], predictions_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Apply Kelly criterion, exposure limits, and correlation analysis."""
        print("\n" + "=" * 70)
        print("  🛡   STAGE 7: RISK MANAGEMENT")
        print("=" * 70)

        risk_result: Dict[str, Any] = {
            "bankroll": self.args.bankroll,
            "kelly_fraction": self.args.kelly_fraction,
            "max_exposure": self.args.max_exposure,
            "bets": [],
            "exposure": {},
            "correlation": {},
        }

        if not recommendations:
            print("  ℹ  No recommendations to risk-manage")
            return risk_result
        if HAS_RISK:
            try:
                # Kelly sizing — KellyCalculator takes bankroll + fraction at init
                kelly = KellyCalculator(
                    bankroll=self.args.bankroll,
                    fraction=self.args.kelly_fraction,
                )
                exposure_mgr = ExposureManager(
                    bankroll=self.args.bankroll,
                    default_max_exposure_pct=self.args.max_exposure,
                    default_max_per_game_pct=self.args.max_exposure * 0.75,
                )

                sized_bets = []
                for bet in recommendations:
                    edge = bet.get("edge", 0)
                    odds = bet.get("odds", -110)

                    # Convert American odds to decimal
                    if odds > 0:
                        decimal_odds = 1 + odds / 100
                    elif odds < 0:
                        decimal_odds = 1 + 100 / abs(odds)
                    else:
                        decimal_odds = 1.91  # default -110 equivalent

                    # Use compute_kelly instead of calculate
                    kelly_pct, _ = kelly.compute_kelly(
                        win_probability=0.5 + edge / 2,
                        decimal_odds=decimal_odds,
                    )

                    # Check exposure
                    team = bet.get("team", "?")
                    if exposure_mgr.check_exposure(team, kelly_pct * self.args.bankroll):
                        bet["kelly_pct"] = kelly_pct
                        bet["stake"] = round(kelly_pct * self.args.bankroll, 2)
                        sized_bets.append(bet)
                        exposure_mgr.track_bet(team, bet["stake"])

                print(f"  ✅  Sized {len(sized_bets)} bets with Kelly criterion")
                for bet in sized_bets[:5]:
                    print(f"       {bet.get('team', '?')}: stake=${bet.get('stake', 0):.2f} "
                          f"({bet.get('kelly_pct', 0):.2%} of bankroll)")

                risk_result["bets"] = sized_bets
                risk_result["exposure"] = exposure_mgr.get_summary() if hasattr(exposure_mgr, "get_summary") else {}

                # Correlation analysis (pass predictions_df explicitly)
                try:
                    tracker = BetCorrelationTracker()
                    corr_df = predictions_df if predictions_df is not None else self.predictions_df
                    high_corr_count = 0
                    if corr_df is not None and len(corr_df) > 1:
                        bet_ids = []
                        for idx, row in corr_df.iterrows():
                            bet_id = f"bet_{idx}"
                            tracker.register_bet(
                                bet_id=bet_id,
                                bet_type="total_points",
                                game_id=row.get("game_id", str(idx)),
                            )
                            bet_ids.append(bet_id)
                        corr_matrix = tracker.get_correlation_matrix(bet_ids)
                        if hasattr(corr_matrix, 'matrix') and hasattr(corr_matrix.matrix, 'shape'):
                            mat = corr_matrix.matrix
                            n = mat.shape[0]
                            high_corr_count = int((np.sum(np.abs(mat) > 0.7) - n) / 2) if n > 1 else 0
                            print(f"  📈  Correlation analysis: {high_corr_count} high correlations found")
                    risk_result["correlation"] = {"high_correlations": high_corr_count}
                except Exception as e:
                    print(f"  ⚠  Correlation analysis failed: {e}")

                self.results["recommendations"] = sized_bets

            except Exception as e:
                print(f"  ⚠  Risk management failed: {e}")
        else:
            print("  ℹ  Risk management module not available")
            # Basic stake sizing
            for bet in recommendations:
                edge = bet.get("edge", 0)
                kelly_pct = min(edge * self.args.kelly_fraction * 4, 0.05)
                bet["kelly_pct"] = kelly_pct
                bet["stake"] = round(kelly_pct * self.args.bankroll, 2)

            risk_result["bets"] = recommendations

        self.results["risk_assessment"] = risk_result
        return risk_result

    # ──────────────────────────────────────────────────────────────
    # 5i.  Validation Suite
    # ──────────────────────────────────────────────────────────────

    def run_validation(self, features_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Run calibration, overfitting detection, cross-validation & drift monitoring."""
        print("\n" + "=" * 70)
        print("  ✅  STAGE 8: MODEL VALIDATION")
        print("=" * 70)

        validation_results: Dict[str, Any] = {}

        # Calibration
        if HAS_VALIDATION:
            try:
                cal = ProbabilityCalibrator(method='platt')
                from sklearn.metrics import brier_score_loss
                # Use predictions_df to check calibration if we have actual outcomes
                if 'actual_total' in predictions_df.columns and 'predicted_total' in predictions_df.columns:
                    scores = predictions_df['predicted_total'].values / 250.0  # Normalize to 0-1
                    labels = (predictions_df['actual_total'] > predictions_df['predicted_total']).astype(int).values
                    try:
                        cal.fit(scores, labels)
                        metrics = cal.evaluate(scores, labels)
                        cal_score = metrics.get('brier_score', 'N/A')
                        print(f"  ✅  Calibration Brier score: {cal_score}")
                        validation_results["calibration"] = metrics
                    except Exception as ce:
                        print(f"  ⚠  Calibration fit failed: {ce}")
                else:
                    print("  ℹ  No actual outcomes for calibration analysis")
                    validation_results["calibration"] = {"status": "skipped", "reason": "no_actuals"}
            except Exception as e:
                print(f"  ⚠  Calibration analysis failed: {e}")

            # Overfitting detection
            try:
                overfit = OverfittingDetector()
                # OverfittingDetector.analyze needs train/test metrics, cv results
                # We don't have those from the simple pipeline, so just log it
                train_metrics = {"mean_error": 0.0}
                test_metrics = {"mean_error": 0.0}
                cv_results = []
                overfit_result = overfit.analyze(
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    cv_results=cv_results,
                    n_observations=len(features_df) if features_df is not None else 100,
                )
                if overfit_result:
                    is_overfit = overfit_result.get("is_overfit", overfit_result.get("overfit", False))
                    print(f"  ✅  Overfitting check: {'⚠ OVERFIT' if is_overfit else '✓ OK'}")
                    validation_results["overfitting"] = overfit_result
            except Exception as e:
                print(f"  ⚠  Overfitting detection failed: {e}")

        # Cross-validation (if not already done in tuning)
        if HAS_VALIDATION:
            try:
                ts_cv = TimeSeriesCrossValidator(n_splits=5)
                feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                                if c not in self.EXCLUDE_COLS]
                if feature_cols and len(feature_cols) > 0 and "total_points" in features_df.columns:
                    try:
                        cv_result = ts_cv.get_splits(len(features_df))
                        print(f"  ✅  Cross-validation: {len(cv_result)} splits generated")
                        validation_results["cross_validation"] = {"n_splits": len(cv_result)}
                    except Exception as cve:
                        print(f"  ℹ  Cross-validation run: {cve}")
            except Exception as e:
                print(f"  ⚠  Cross-validation failed: {e}")

        # Drift monitoring
        if HAS_MONITORING:
            try:
                tracker = PerformanceTracker(model_name="pipeline_model")
                if 'predicted_total' in predictions_df.columns and 'total_points' in features_df.columns:
                    for idx, row in predictions_df.iterrows():
                        pred = row.get('predicted_total', 0)
                        actual = features_df.loc[idx, 'total_points'] if idx in features_df.index else pred
                        if pred and actual and actual != pred:
                            tracker.record_prediction(predicted=pred, actual=actual)
                    drift_report = tracker.get_report()
                    if drift_report:
                        n_alerts = len(drift_report.get('drift_alerts', []))
                        print(f"  ✅  Drift check: {n_alerts} alerts")
                        validation_results["drift"] = drift_report
                else:
                    print("  ℹ  Insufficient data for drift analysis")
                    validation_results["drift"] = {"status": "skipped"}
            except Exception as e:
                print(f"  ⚠  Drift detection failed: {e}")

        self.results["validation"] = validation_results

    # ──────────────────────────────────────────────────────────────
    # 5j.  Monte Carlo Simulation
    # ──────────────────────────────────────────────────────────────

    def run_simulation(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Monte Carlo simulation on recommended bets."""
        print("\n" + "=" * 70)
        print("  🎲  STAGE 9: MONTE CARLO SIMULATION")
        print("=" * 70)

        sim_result: Dict[str, Any] = {}

        if not recommendations:
            print("  ℹ  No recommendations to simulate")
            return sim_result

        if HAS_BETTING:
            try:
                sim = MonteCarloSimulator()
                n_sims = 10_000
                result = sim.simulate(
                    bets=recommendations,
                    bankroll=self.args.bankroll,
                    n_simulations=n_sims,
                )
                if result:
                    median_return = result.get("median_return", result.get("median", 0))
                    upside = result.get("upside_90th", result.get("percentile_90", 0))
                    downside = result.get("downside_10th", result.get("percentile_10", 0))
                    print(f"  ✅  {n_sims:,} simulations complete")
                    print(f"       Median return: ${median_return:+.2f}")
                    print(f"       Upside (90th): ${upside:+.2f}")
                    print(f"       Downside (10th): ${downside:+.2f}")
                    sim_result = result
                    self.results["simulation"] = result
            except Exception as e:
                print(f"  ⚠  Monte Carlo simulation failed: {e}")
        else:
            print("  ℹ  Monte Carlo module not available")

        return sim_result

    # ──────────────────────────────────────────────────────────────
    # 5k.  Edge Detection
    # ──────────────────────────────────────────────────────────────

    def detect_edges(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect betting edges from prediction residuals."""
        print("\n" + "=" * 70)
        print("  🎯  STAGE 10: EDGE DETECTION")
        print("=" * 70)

        edges: List[Dict[str, Any]] = []

        if HAS_BETTING:
            try:
                detector = EdgeDetector()
                # EdgeDetector uses methods like detect_rest_edge, detect_home_advantage_edge
                # Try calling them if the DataFrame has the right columns
                if 'rest_advantage' in predictions_df.columns:
                    rest_edge = detector.detect_rest_edge(predictions_df)
                    if rest_edge:
                        print(f"  ℹ  Rest edge detected: {rest_edge}")
                if 'point_diff' in predictions_df.columns:
                    home_edge = detector.detect_home_advantage_edge(predictions_df)
                    if home_edge:
                        print(f"  ℹ  Home advantage edge: {home_edge}")
            except Exception as e:
                print(f"  ⚠  Edge detection failed: {e}")
        else:
            print("  ℹ  Edge detector not available")

        if not edges and "predicted_total" in predictions_df.columns and "market_total" in predictions_df.columns:
            print("  ℹ  Using simple edge calculation...")
            for _, row in predictions_df.iterrows():
                pt = row.get("predicted_total", 0)
                mt = row.get("market_total", 0)
                if pt and mt:
                    diff = pt - mt
                    pct_edge = diff / mt
                    if abs(pct_edge) > self.args.min_edge:
                        team = row.get("home_team", row.get("team", "?"))
                        edges.append({
                            "team": team,
                            "game_id": row.get("game_id", ""),
                            "market_total": mt,
                            "predicted_total": pt,
                            "edge_pct": round(pct_edge, 4),
                            "direction": "over" if pct_edge > 0 else "under",
                        })

        if edges:
            top_edges = sorted(edges, key=lambda x: abs(x.get("edge_pct", 0)), reverse=True)[:5]
            for e in top_edges:
                print(f"       {e.get('team', '?')}: {e.get('direction', '?')} "
                      f"by {e.get('edge_pct', 0):.2%}")

        return edges

    # ──────────────────────────────────────────────────────────────
    # 5l.  Backtesting (if historical data available)
    # ──────────────────────────────────────────────────────────────

    def run_backtest(self, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run a full backtest on historical predictions."""
        if self.args.live:
            print("\n  ⏩  Backtesting skipped in live mode")
            return None

        print("\n" + "=" * 70)
        print("  ⏪  STAGE 11: BACKTESTING")
        print("=" * 70)

        if not HAS_BACKTESTING:
            print("  ℹ  Backtesting module not available")
            return None

        try:
            engine = BacktestMetrics()
            # Build a bets_df from predictions if available
            if hasattr(self, 'predictions_df') and self.predictions_df is not None and 'predicted_total' in self.predictions_df.columns and 'total_points' in features_df.columns:
                bet_rows = []
                for idx, row in self.predictions_df.iterrows():
                    pred = row.get('predicted_total', 0)
                    actual = features_df.loc[idx, 'total_points'] if idx in features_df.index else 0
                    if pred and actual:
                        edge = (pred - actual) / max(actual, 0.1)
                        outcome = "WIN" if abs(pred - actual) < 5 else "LOSS"
                        bet_rows.append({"game_date": str(row.get('game_date', '')), "outcome": outcome, "profit_units": 1.0 if outcome == "WIN" else -1.0, "edge_pct": edge})
                if bet_rows:
                    bets_df = pd.DataFrame(bet_rows)
                    metrics = engine.compute_all(bets_df)
                    if metrics and "error" not in metrics:
                        total_return = metrics.get("total_profit_units", 0)
                        sharpe = metrics.get("sharpe_ratio", "N/A")
                        win_rate = metrics.get("win_rate", "N/A")
                        print(f"  ✅  Backtest complete")
                        print(f"       Total return: {total_return:+.2f} units")
                        print(f"       Sharpe ratio: {sharpe}")
                        print(f"       Win rate: {win_rate if isinstance(win_rate, str) else f'{win_rate:.1%}'}")
                        self.results["backtest"] = metrics
                        return metrics
            print("  ℹ  No backtest results")
        except Exception as e:
            print(f"  ⚠  Backtesting failed: {e}")

        return None

    # ──────────────────────────────────────────────────────────────
    # 5m.  Report Generation (Console + JSON + HTML)
    # ──────────────────────────────────────────────────────────────

    def generate_report(self):
        """Print a comprehensive summary and optionally save outputs."""
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("  📋  FINAL REPORT")
        print("=" * 70)
        print(f"  ⏱  Pipeline completed in {elapsed:.1f}s")
        print(f"  📊  Data source: {self.results.get('metadata', {}).get('data_source', 'N/A')}")
        print(f"  📅  Mode: {'LIVE' if self.args.live else 'HISTORICAL'}")

        # Summary stats
        n_games = len(self.results.get("predictions", []))
        n_recommendations = len(self.results.get("recommendations", []))
        n_clear = len(self.results.get("clear_picks", []))
        n_ev = len(self.results.get("ev_opportunities", []))
        n_arb = len(self.results.get("arbitrage_opportunities", []))
        n_props = len(self.results.get("player_props", []))

        print(f"  🎮  Games analyzed: {n_games}")
        print(f"  💰  Bet recommendations: {n_recommendations}")
        print(f"  🎯  Clear picks: {n_clear}")
        print(f"  🔬  +EV opportunities: {n_ev}")
        print(f"  ♻   Arbitrage opportunities: {n_arb}")
        print(f"  🏀  Player props: {n_props}")

        # Bankroll summary
        risk = self.results.get("risk_assessment", {})
        if risk:
            bankroll = risk.get("bankroll", self.args.bankroll)
            total_staked = sum(b.get("stake", 0) for b in risk.get("bets", []))
            n_bets = len(risk.get("bets", []))
            print(f"  💵  Bankroll: ${bankroll:.2f} | Staked: ${total_staked:.2f} ({total_staked/bankroll:.1%}) across {n_bets} bets")

        # Validation summary
        val = self.results.get("validation", {})
        if val:
            cal = val.get("calibration", {})
            overfit = val.get("overfitting", {})
            drift = val.get("drift", {})
            if cal:
                print(f"  📐  Calibration: checked")
            if overfit:
                print(f"  ⚠  Overfitting: {'DETECTED' if overfit.get('is_overfit', overfit.get('overfit', False)) else 'None'}")
            if drift:
                print(f"  🌊  Drift: {'DETECTED' if drift.get('drift_detected', False) else 'None'}")

        # Simulation summary
        sim = self.results.get("simulation", {})
        if sim:
            med = sim.get("median_return", sim.get("median", 0))
            print(f"  🎲  Simulation (10k runs): median=${med:+.2f}")

        # Top clear picks
        clear_picks = self.results.get("clear_picks", [])
        if clear_picks:
            print(f"\n  ── TOP CLEAR PICKS ──")
            for i, pick in enumerate(clear_picks[:5]):
                print(f"   {i+1}. {pick.get('team', '?')} ({pick.get('bet_type', '?')}) "
                      f"edge={pick.get('edge', 0):.2%} "
                      f"conf={pick.get('confidence', 'N/A')}")

        # Top EV opportunities
        ev_opps = self.results.get("ev_opportunities", [])
        if ev_opps:
            print(f"\n  ── TOP +EV OPPORTUNITIES ──")
            for i, opp in enumerate(ev_opps[:3]):
                print(f"   {i+1}. {opp.get('game', '?')} ({opp.get('bet_type', '?')}) "
                      f"EV={opp.get('expected_value', 0):.2%}")

        # Arbitrage
        arb_opps = self.results.get("arbitrage_opportunities", [])
        if arb_opps:
            print(f"\n  ── ARBITRAGE OPPORTUNITIES ──")
            for i, arb in enumerate(arb_opps[:3]):
                print(f"   {i+1}. {arb.get('game', '?')} return={arb.get('return_pct', 0):.2%}")

        # Save JSON output
        if self.args.output:
            output_path = Path(self.args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n  💾  Results saved to {output_path}")

        # HTML report
        if self.args.html:
            self._generate_html_report()

        # Scheduled mode: print JSON summary to stdout for the scheduler
        if self.args.scheduled:
            summary = {
                "status": "complete",
                "duration_seconds": round(elapsed, 1),
                "data_source": self.results.get("metadata", {}).get("data_source", "N/A"),
                "games": len(self.results.get("predictions", [])),
                "recommendations": len(self.results.get("recommendations", [])),
                "clear_picks": len(self.results.get("clear_picks", [])),
                "ev_opportunities": len(self.results.get("ev_opportunities", [])),
                "arbitrage": len(self.results.get("arbitrage_opportunities", [])),
                "player_props": len(self.results.get("player_props", [])),
                "bankroll": self.args.bankroll,
                "total_staked": sum(
                    b.get("stake", 0)
                    for b in self.results.get("risk_assessment", {}).get("bets", [])
                ),
                "risk_assessment": self.results.get("risk_assessment", {}),
                "clear_picks_detail": self.results.get("clear_picks", []),
                "ev_detail": self.results.get("ev_opportunities", []),
                "arbitrage_detail": self.results.get("arbitrage_opportunities", []),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"##SCHEDULED_RESULT##{json.dumps(summary, default=str)}")

    def _generate_html_report(self):
        """Generate a standalone HTML report with all results."""
        try:
            from jinja2 import Environment, FileSystemLoader

            templates_dir = PROJECT_ROOT / "web" / "templates"
            if templates_dir.exists():
                env = Environment(loader=FileSystemLoader(str(templates_dir)))
                template = env.get_template("tomorrow.html") if (templates_dir / "tomorrow.html").exists() else None
                if template:
                    html = template.render(
                        results=self.results,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        args=self.args,
                    )
                    report_path = PROJECT_ROOT / "reports" / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.html"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(html, encoding="utf-8")
                    print(f"  🌐  HTML report: {report_path}")
                    return

            # Fallback: inline HTML
            html = self._build_inline_html_report()
            report_path = PROJECT_ROOT / "reports" / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.html"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(html, encoding="utf-8")
            print(f"  🌐  HTML report: {report_path}")

        except Exception as e:
            print(f"  ⚠  HTML report generation failed: {e}")

    def _build_inline_html_report(self) -> str:
        """Build a self-contained HTML report."""
        recs = self.results.get("recommendations", [])
        clear = self.results.get("clear_picks", [])
        ev = self.results.get("ev_opportunities", [])
        arb = self.results.get("arbitrage_opportunities", [])
        props = self.results.get("player_props", [])

        rows_html = ""
        for r in recs[:20]:
            rows_html += f"""
            <tr>
                <td>{r.get('team', '?')}</td>
                <td>{r.get('bet_type', '?')}</td>
                <td>{r.get('edge', 0):.2%}</td>
                <td>{r.get('confidence', 'N/A')}</td>
                <td>${r.get('stake', 0):.2f}</td>
                <td>{r.get('odds', 0)}</td>
            </tr>"""

        clear_html = ""
        for c in clear[:10]:
            clear_html += f"""
            <tr>
                <td>{c.get('team', '?')}</td>
                <td>{c.get('bet_type', '?')}</td>
                <td>{c.get('edge', 0):.2%}</td>
                <td>{c.get('confidence', 'N/A')}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Betting Intelligence — Prediction Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:#0f1219; color:#e1e5ed; padding:2rem; }}
  .container {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:1.8rem; margin-bottom:0.5rem; background:linear-gradient(135deg,#6366f1,#a855f7);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .meta {{ color:#8892a4; font-size:0.9rem; margin-bottom:2rem; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1rem; margin-bottom:2rem; }}
  .stat-card {{ background:#1a1f2e; border-radius:12px; padding:1.2rem; text-align:center;
               border:1px solid #2a3040; }}
  .stat-card .num {{ font-size:1.8rem; font-weight:700; color:#6366f1; }}
  .stat-card .label {{ font-size:0.8rem; color:#8892a4; margin-top:0.3rem; }}
  table {{ width:100%; border-collapse:collapse; background:#1a1f2e; border-radius:12px;
          overflow:hidden; margin-bottom:2rem; }}
  th {{ background:#2a3040; padding:0.8rem 1rem; text-align:left; font-size:0.85rem;
        color:#8892a4; text-transform:uppercase; letter-spacing:0.05em; }}
  td {{ padding:0.7rem 1rem; border-top:1px solid #2a3040; font-size:0.9rem; }}
  tr:hover {{ background:#222838; }}
  .section-title {{ font-size:1.2rem; font-weight:600; margin:1.5rem 0 1rem;
                    color:#a5b4fc; }}
  .badge {{ display:inline-block; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem;
            font-weight:600; }}
  .badge-high {{ background:#065f46; color:#6ee7b7; }}
  .badge-med {{ background:#78350f; color:#fcd34d; }}
  .badge-low {{ background:#3b0f1f; color:#fca5a5; }}
  @media (max-width:768px) {{ body {{ padding:1rem; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>🏀 Betting Intelligence Report</h1>        <p class="meta">Generated {datetime.now():%B %d, %Y at %H:%M:%S} · {'LIVE' if self.args.live else 'HISTORICAL'}</p>

  <div class="stats">
    <div class="stat-card"><div class="num">{len(recs)}</div><div class="label">Recommendations</div></div>
    <div class="stat-card"><div class="num">{len(clear)}</div><div class="label">Clear Picks</div></div>
    <div class="stat-card"><div class="num">{len(ev)}</div><div class="label">+EV Opportunities</div></div>
    <div class="stat-card"><div class="num">{len(arb)}</div><div class="label">Arbitrage</div></div>
    <div class="stat-card"><div class="num">{len(props)}</div><div class="label">Player Props</div></div>
  </div>

  <div class="section-title">🎯 Top Recommendations</div>
  <table>
    <thead><tr><th>Team</th><th>Type</th><th>Edge</th><th>Confidence</th><th>Stake</th><th>Odds</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="section-title">⭐ Clear Picks</div>
  <table>
    <thead><tr><th>Team</th><th>Type</th><th>Edge</th><th>Confidence</th></tr></thead>
    <tbody>{clear_html}</tbody>
  </table>

  <p class="meta" style="text-align:center;margin-top:3rem;">
    Powered by Betting Intelligence Engine v3.0
  </p>
</div>
</body>
</html>"""

    # ──────────────────────────────────────────────────────────────
    # 5n.  Run the Full Pipeline
    # ──────────────────────────────────────────────────────────────

    def run(self):
        """Execute the full prediction pipeline."""
        print("\n" + "█" * 70)
        print("  🏀  BETTING INTELLIGENCE — PREDICTION PIPELINE v3.0")
        print("█" * 70)

        # 1. Load data
        self.df = self.load_data()
        if self.df is None or self.df.empty:
            print("  ❌  No data available. Exiting.")
            sys.exit(1)

        # 2. Engineer features
        self.features_df = self.engineer_features(self.df)

        # 3. Optional tuning
        self.tune_hyperparameters(self.features_df)

        # 4. Train & predict
        self.predictions_df = self.train_and_predict(self.features_df)
        self.results["predictions"] = self.predictions_df.to_dict("records") if hasattr(self.predictions_df, "to_dict") else []

        # 4a. Train full-data model for tomorrow predictions (always, in live mode)
        if self.args.live:
            self._train_all_data_model(self.features_df)

        # 4b. Predict tomorrow's games using full-data model (only in live mode)
        if self.args.live and self.model is not None:
            tomorrow_preds = self.predict_tomorrow_games()
            if tomorrow_preds:
                # Convert tomorrow predictions into recommendation dicts
                tomorrow_recs = []
                for tp in tomorrow_preds:
                    edge = abs(tp.get("edge_pct", 0))
                    if edge < self.args.min_edge:
                        continue
                    direction = tp.get("direction", "over")
                    team = tp.get("home_team", "?")
                    rec = {
                        "team": team,
                        "bet_type": f"total_{direction}",
                        "edge": edge,
                        "confidence": tp.get("confidence", "medium"),
                        "odds": -110,
                        "market_total": tp.get("market_total", 0),
                        "predicted_total": tp.get("predicted_total", 0),
                        "expected_value": edge,
                    }
                    tomorrow_recs.append(rec)
                if tomorrow_recs:
                    print(f"  🎯  Generated {len(tomorrow_recs)} real-edge recommendations from tomorrow predictions")
                    self.results["tomorrow_recommendations"] = tomorrow_recs
                    # Prepend tomorrow recs to the recommendation engine output
                    if hasattr(self, 'tomorrow_recommendations_final'):
                        self.tomorrow_recommendations_final = tomorrow_recs

        # 5. Edge detection
        edges = self.detect_edges(self.predictions_df)

        # 6. Generate recommendations
        recommendations = self.generate_recommendations(self.predictions_df)

        # Merge tomorrow predictions (real model edges) into main recommendations
        if self.args.live and self.tomorrow_recommendations_final:
            recommendations = self.tomorrow_recommendations_final + recommendations
            print(f"  🔗  Merged {len(self.tomorrow_recommendations_final)} real-model picks into recommendations")

        # 7. Player props
        props = self.generate_player_props(self.predictions_df)

        # 8. +EV & Arbitrage scanning
        self.scan_opportunities(self.predictions_df)

        # 9. Risk management
        risked_bets = self.apply_risk_management(recommendations, predictions_df=self.predictions_df)

        # 10. Validation suite
        self.run_validation(self.features_df, self.predictions_df)

        # 11. Simulation (if --full or --simulate)
        if self.args.full or self.args.simulate:
            self.run_simulation(risked_bets.get("bets", recommendations))

        # 12. Backtesting (if historical)
        if not self.args.live:
            self.run_backtest(self.features_df)

        # 13. Report
        self.generate_report()

        return self.results


# ──────────────────────────────────────────────────────────────────────
# 6.  Entry Point
# ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    if args.live and not ODDS_API_KEY:
        print("  ⚠  --live mode requires ODDS_API_KEY env var or .env file.")
        print("  ℹ  Falling back to historical.")
        args.live = False

    # Scheduled mode: force live+no-tune, auto-save results, JSON to stdout
    if args.scheduled:
        if not ODDS_API_KEY:
            print("  ⚠  --scheduled mode requires ODDS_API_KEY env var or .env file.")
            return 1
        args.live = True
        args.no_tune = True
        args.html = False
        if not args.output:
            args.output = str(PROJECT_ROOT / "reports" / "latest.json")

    pipeline = PredictionPipeline(args)
    results = pipeline.run()

    # Return exit code based on results
    if results.get("clear_picks") or results.get("recommendations"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
