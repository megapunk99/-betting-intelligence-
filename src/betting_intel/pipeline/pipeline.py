"""
PredictionPipeline — the main pipeline orchestrator.

Composes all mixin modules into a single PredictionPipeline class that
executes the full betting prediction workflow:
  1. Load data     → 2. Engineer features  → 3. Train/predict
  4. Tune hparams  → 5. Generate bets       → 6. Risk-manage
  7. Validate      → 8. Report
"""

from __future__ import annotations

import time
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from betting_intel.pipeline.bootstrap import (
    PROJECT_ROOT, logger,
)
from betting_intel.pipeline.data_loading import DataLoadingMixin
from betting_intel.pipeline.modeling import ModelingMixin
from betting_intel.pipeline.staking import StakingMixin
from betting_intel.pipeline.risk_analysis import RiskAnalysisMixin
from betting_intel.pipeline.validation import ValidationMixin
from betting_intel.pipeline.reporting import ReportingMixin

# ── Inline Pipeline Monitor ──────────────────────────────────────────────
# Lightweight replacement for deleted monitoring package.

class _InlinePipelineMonitor:
    """Prints stage-level timing to console during pipeline runs."""
    def __init__(self):
        self._start_times = {}
    def start_run(self, mode="historical", season="Current"):
        import time
        run_id = f"run_{int(time.time())}"
        self._start_times[run_id] = time.time()
        logger.info(f"Pipeline {run_id} started — mode={mode}, season={season}")
        return run_id
    def record_stage(self, run_id, stage_name, status="ok", duration=0.0, **kwargs):
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items() if v)
        logger.info(f"  Stage {stage_name}: {status} ({duration:.1f}s) {extra}")
    def complete_run(self, run_id, status="success", games=0, predictions=0, metrics=None):
        import time
        elapsed = time.time() - self._start_times.get(run_id, time.time())
        logger.info(f"Pipeline {run_id} {status}: {games} games, {predictions} preds, {elapsed:.1f}s")

class _InlineATSTracker:
    """Tracks ATS (against the spread) prediction records."""
    def __init__(self):
        self._records = []
    def record_prediction(self, game_id="", game_date="", matchup="",
                          model_name="", predicted_spread=0.0,
                          vegas_spread=0.0, actual_margin=0.0):
        # Determine ATS result: compare model prediction error vs Vegas error
        pred_error = abs(predicted_spread - actual_margin)
        vegas_error = abs(vegas_spread - actual_margin)
        result = "WIN" if pred_error < vegas_error else ("LOSS" if pred_error > vegas_error else "PUSH")
        self._records.append({
            "game_id": game_id, "model_name": model_name, "result": result,
        })
    def get_summary(self, model_name="pipeline_ensemble"):
        recs = [r for r in self._records if r["model_name"] == model_name]
        wins = sum(1 for r in recs if r["result"] == "WIN")
        losses = sum(1 for r in recs if r["result"] == "LOSS")
        pushes = sum(1 for r in recs if r["result"] == "PUSH")
        total = wins + losses
        return {
            "wins": wins, "losses": losses, "pushes": pushes,
            "win_rate": round(wins / total, 4) if total > 0 else 0.0,
        }


class PredictionPipeline(
    DataLoadingMixin,
    ModelingMixin,
    StakingMixin,
    RiskAnalysisMixin,
    ValidationMixin,
    ReportingMixin,
):
    """
    Orchestrates the full prediction workflow:
      1. Load data     → 2. Engineer features  → 3. Train/predict
      4. Tune hparams  → 5. Generate bets       → 6. Risk-manage
      7. Validate      → 8. Report
    """

    def __init__(self, args):
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
        self.model = None
        self.model_feature_cols: List[str] = []
        self.tomorrow_recommendations_final: List[Dict[str, Any]] = []
        self._upcoming_games_df: Optional[pd.DataFrame] = None
        self.bet_journal: Optional[Any] = None
        self._feature_pipeline = None
        self._pipeline_raw_df: Optional[pd.DataFrame] = None

        # NBA_AI-inspired pipeline & ATS monitoring
        self.pipeline_monitor = _InlinePipelineMonitor()
        self.ats_tracker = _InlineATSTracker()
        self._pipeline_run_id = None

    def _init_journal(self):
        """Lazy-initialize the bet journal (stub — BetJournal deleted)."""
        if self.bet_journal is None:
            from betting_intel.pipeline.modeling import _StubBetJournal
            self.bet_journal = _StubBetJournal()
        return self.bet_journal

    # ── Orchestrator ───────────────────────────────────────────────

    def run(self):
        """Execute the full prediction pipeline with NBA_AI-inspired monitoring."""
        print("\n" + "█" * 70)
        print("  🏀  BETTING INTELLIGENCE — PREDICTION PIPELINE v3.0")
        print("     Now with: Enhanced Ensemble (MLP + LightGBM + Ridge)")
        print("     Spread Uncertainty · ATS Tracking · Run Monitor")
        print("█" * 70)

        # ── Start monitoring ───────────────────────────────────────
        if self.pipeline_monitor is not None:
            mode_str = "live" if self.args.live else "historical"
            self._pipeline_run_id = self.pipeline_monitor.start_run(
                mode=mode_str, season=self.args.season if hasattr(self.args, 'season') else "Current"
            )

        # ── 1. Load data ──────────────────────────────────────────────
        t0 = time.time()
        if self.args.live:
            self._load_for_live_mode()
        else:
            print("\n" + "=" * 70)
            print("  📊  STAGE 1: DATA LOADING")
            print("=" * 70)
            self.df = self._load_historical_data()
            if self.df is None or self.df.empty:
                print("  ❌  No historical data available. Exiting.")
                sys.exit(1)
            self._upcoming_games_df = None
        if self.pipeline_monitor is not None and self._pipeline_run_id:
            self.pipeline_monitor.record_stage(
                self._pipeline_run_id, "data_loading",
                status="ok", duration=time.time() - t0,
                rows=len(self.df) if self.df is not None else 0,
            )

        # ── 2. Engineer features ──────────────────────────────────────
        self.features_df = self.engineer_features(self.df)

        # ── 3. Optional tuning ────────────────────────────────────────
        self.tune_hyperparameters(self.features_df)

        # ── 4. Train & predict (backtest split) ───────────────────────
        t0 = time.time()
        self.predictions_df = self.train_and_predict(self.features_df)
        self.results["predictions"] = (
            self.predictions_df.to_dict("records") if hasattr(self.predictions_df, "to_dict") else []
        )
        if self.pipeline_monitor is not None and self._pipeline_run_id:
            self.pipeline_monitor.record_stage(
                self._pipeline_run_id, "train_predict",
                status="ok", duration=time.time() - t0,
                n_predictions=len(self.results["predictions"]),
            )

        self._prepare_predictions_for_recommendations()

        # ── 4a. Train full-data model on ALL historical data ───────────
        if self.args.live:
            self._train_all_data_model(self.features_df)

        # ── 4a2. Train multi-league models (WNBA, Euroleague, NCAAB...) ─
        if self.args.live:
            self.train_multi_league_models()

        # ── 4b. Predict tomorrow's games (live mode) ───────────────────
        # _run_tomorrow_predictions -> predict_tomorrow_games handles the fallback
        # to nba_api when TheOddsAPI is unavailable, so we call it even when
        # _upcoming_games_df is None (the fallback generates the schedule).
        if self.args.live and self.model is not None:
            self._run_tomorrow_predictions()

        # ── 4c. Predict multi-league games (WNBA, Euroleague, NCAAB...) ─
        if self.args.live:
            multi_league_preds = self.predict_multi_league_games()
            if multi_league_preds:
                self._merge_multi_league_predictions(multi_league_preds)

        # ── 5. Edge detection ─────────────────────────────────────────
        edges = self.detect_edges(self.predictions_df)

        # ── 6. Generate recommendations ───────────────────────────────
        recommendations = self.generate_recommendations(self.predictions_df)

        if self.args.live and self.tomorrow_recommendations_final:
            recommendations = self.tomorrow_recommendations_final + recommendations
            print(f"  🔗  Merged {len(self.tomorrow_recommendations_final)} real-model picks into recommendations")

        # ── 7. Player props ───────────────────────────────────────────
        props = self.generate_player_props(self.predictions_df)

        # ── 8. +EV & Arbitrage scanning ───────────────────────────────
        self.scan_opportunities(self.predictions_df)

        # ── 9. Risk management ────────────────────────────────────────
        risked_bets = self.apply_risk_management(recommendations, predictions_df=self.predictions_df)

        # ── 10. Validation suite ──────────────────────────────────────
        self.run_validation(self.features_df, self.predictions_df)

        # ── 11. Simulation (if --full or --simulate) ──────────────────
        if self.args.full or self.args.simulate:
            self.run_simulation(risked_bets.get("bets", recommendations))

        # ── 12. Backtesting (if historical, with ATS tracking) ────────
        if not self.args.live:
            self.run_backtest(self.features_df)
            self._track_ats_performance()

        # ── 13. Report ────────────────────────────────────────────────
        self.generate_report()

        # ── Complete monitoring ─────────────────────────────────────
        n_preds = len(self.results.get("predictions", []))
        n_games = len(self.results.get("games", [])) or len(self.results.get("recommendations", []))
        if self.pipeline_monitor is not None and self._pipeline_run_id:
            self.pipeline_monitor.complete_run(
                self._pipeline_run_id,
                status="success",
                games=n_games,
                predictions=n_preds,
                metrics={
                    "duration": round(time.time() - self.start_time, 1),
                    "n_models": len(self.model_feature_cols) if self.model_feature_cols else 0,
                },
            )

        return self.results

    # ── ATS & Live Mode Helpers ─────────────────────────────────────

    def _track_ats_performance(self):
        """Track ATS performance if actual results are available."""
        if (self.ats_tracker is None
                or self.predictions_df is None
                or self.predictions_df.empty):
            return
        try:
            if ('predicted_spread' not in self.predictions_df.columns
                    or 'spread' not in self.predictions_df.columns):
                return

            # Determine the actual margin column — total_points is NOT margin!
            margin_col = None
            for col in ["actual_margin", "point_diff", "home_score"]:
                if col in self.predictions_df.columns:
                    margin_col = col
                    break
            if margin_col is None:
                logger.debug("No margin column found for ATS tracking — skipping")
                return

            n_tracked = 0
            for idx, row in self.predictions_df.iterrows():
                game_id = str(row.get("game_id", f"game_{idx}"))
                matchup = f"{row.get('home_team', '?')} vs {row.get('away_team', '?')}"
                margin = float(row[margin_col])
                self.ats_tracker.record_prediction(
                    game_id=game_id,
                    game_date=str(row.get("game_date", "")),
                    matchup=matchup,
                    model_name="pipeline_ensemble",
                    predicted_spread=float(row["predicted_spread"]),
                    vegas_spread=float(row["spread"]),
                    actual_margin=margin,
                )
                n_tracked += 1
            if n_tracked:
                summary = self.ats_tracker.get_summary(model_name="pipeline_ensemble")
                print(f"  📊  ATS Record: {summary.get('wins', 0)}-{summary.get('losses', 0)} "
                      f"({summary.get('win_rate', 0):.1%}) across {n_tracked} games")
                self.results["ats_summary"] = summary
        except Exception as e:
            logger.debug(f"ATS tracking failed (non-fatal): {e}")

    def _load_for_live_mode(self):
        """Load BOTH historical data (for training) AND live odds (for tomorrow predictions)."""
        print("\n" + "=" * 70)
        print("  📊  STAGE 1: DATA LOADING (LIVE MODE)")
        print("=" * 70)

        # Step A: Load historical data for feature engineering & model training
        print("  📚  Loading historical data for model training...")
        self.df = self._load_historical_data()
        if self.df is None or self.df.empty:
            print("  ❌  No historical data available for training. Exiting.")
            sys.exit(1)
        print(f"  ✅  Loaded {len(self.df)} historical games")

        # Step B: Load live odds for tomorrow's games
        print("  🌐  Loading live odds for upcoming games...")
        live_df = self._load_live_data()
        if live_df is not None and not live_df.empty:
            self._upcoming_games_df = live_df
            print(f"  ✅  Loaded {len(live_df)} upcoming games from live odds")
        else:
            print("  ⚠  No live odds available. Using NBA static data for schedule.")
            print("  ℹ  Predictions will use league-average market totals for edge estimates.")
            self._upcoming_games_df = None

    def _prepare_predictions_for_recommendations(self):
        """Ensure predictions_df has columns needed by the recommendation engine."""
        if self.predictions_df is None or self.predictions_df.empty:
            return

        pred_copy = self.predictions_df.copy()
        if "home_team" not in pred_copy.columns and "TEAM_NAME_home" in pred_copy.columns:
            pred_copy["home_team"] = pred_copy["TEAM_NAME_home"]
        if "away_team" not in pred_copy.columns and "TEAM_NAME_away" in pred_copy.columns:
            pred_copy["away_team"] = pred_copy["TEAM_NAME_away"]
        if "market_total" not in pred_copy.columns and "market_line_baseline" in pred_copy.columns:
            pred_copy["market_total"] = pred_copy["market_line_baseline"]

        if "edge_pct" not in pred_copy.columns:
            if "market_total" in pred_copy.columns:
                pred_copy["edge_pct"] = np.where(
                    pred_copy["market_total"] > 0,
                    (pred_copy["predicted_total"] - pred_copy["market_total"]) / pred_copy["market_total"],
                    0.0,
                )
            elif "market_line_baseline" in pred_copy.columns and "predicted_total" in pred_copy.columns:
                pred_copy["edge_pct"] = np.where(
                    pred_copy["market_line_baseline"] > 0,
                    (pred_copy["predicted_total"] - pred_copy["market_line_baseline"]) / pred_copy["market_line_baseline"],
                    0.0,
                )

        if "direction" not in pred_copy.columns and "edge_pct" in pred_copy.columns:
            pred_copy["direction"] = np.where(pred_copy["edge_pct"] > 0, "over", "under")

        self.predictions_df = pred_copy

    def _run_tomorrow_predictions(self):
        """Run tomorrow predictions and merge into recommendations."""
        tomorrow_preds = self.predict_tomorrow_games()
        if tomorrow_preds:
            tomorrow_recs = []
            for tp in tomorrow_preds:
                edge = abs(tp.get("edge_pct", 0))
                if edge < self.args.min_edge:
                    continue
                direction = tp.get("direction", "over")
                odds_val = tp.get("odds", tp.get("implied_odds", {})
                                  .get("home_moneyline" if tp.get("direction") == "over" else "away_moneyline", -110))
                rec = {
                    "team": tp.get("home_team", "?"),
                    "bet_type": f"total_{direction}",
                    "edge": edge,
                    "confidence": tp.get("confidence", "medium"),
                    "odds": odds_val,
                    "market_total": tp.get("market_total", 0),
                    "predicted_total": tp.get("predicted_total", 0),
                    "expected_value": edge,
                }
                tomorrow_recs.append(rec)
            if tomorrow_recs:
                print(f"  🎯  Generated {len(tomorrow_recs)} real-edge recommendations from tomorrow predictions")
                self.results["tomorrow_recommendations"] = tomorrow_recs
                self.tomorrow_recommendations_final = tomorrow_recs

    def _merge_multi_league_predictions(self, multi_league_preds: List[Dict[str, Any]]):
        """Merge multi-league predictions into tomorrow recommendations."""
        if not multi_league_preds:
            return

        # Convert to recommendation format
        ml_recs = []
        for tp in multi_league_preds:
            edge = abs(tp.get("edge_pct", 0))
            if edge < self.args.min_edge:
                continue
            direction = tp.get("direction", "over")
            league = tp.get("league", "nba")
            rec = {
                "team": tp.get("home_team", "?"),
                "bet_type": f"total_{direction}",
                "edge": edge,
                "confidence": tp.get("confidence", "medium"),
                "odds": tp.get("implied_odds", {}).get("home_moneyline" if direction == "over" else "away_moneyline", -110),
                "market_total": tp.get("market_total", 0),
                "predicted_total": tp.get("predicted_total", 0),
                "expected_value": edge,
                "league": league,
            }
            ml_recs.append(rec)

        if ml_recs:
            # Merge with existing tomorrow recommendations
            existing = self.tomorrow_recommendations_final
            self.tomorrow_recommendations_final = existing + ml_recs
            leagues = set(r.get("league", "?") for r in ml_recs)
            print(f"  🔗  Merged {len(ml_recs)} multi-league predictions into recommendations")
            print(f"      Leagues: {', '.join(leagues)}")
