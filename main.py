"""
Main pipeline: orchestrates data loading, feature engineering (v2.0),
advanced modeling, backtesting, edge detection, bankroll simulation,
and Monte Carlo risk analysis.

Run: python main.py  (from the betting-intelligence directory)
"""

import os
import sys
import logging
import warnings
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Add src/ to path so we can import from betting_intel.*
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betting_intel import config as cfg
from betting_intel.config import (
    DB_PATH, OUTPUT_DIR, VERBOSE,
    ENABLE_LINEAR_MODEL, ENABLE_XGBOOST_MODEL, ENABLE_ENSEMBLE,
    STRATEGIES, INITIAL_BANKROLL, UNIT_SIZE,
    ENABLE_HYPERPARAMETER_TUNING, ENABLE_STACKING_ENSEMBLE,
    ENABLE_MONTE_CARLO, MONTE_CARLO_SIMULATIONS,
    PREFERRED_MODEL,
)
from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.data.integrity import DataQualityReport
from betting_intel.models.predictors import (
    TotalPointsPredictor, SpreadPredictor, MomentumModel,
)
from betting_intel.backtesting.engine import WalkForwardEngine, BacktestResult
from betting_intel.backtesting.metrics import BacktestMetrics
from betting_intel.betting.edge import EdgeDetector
from betting_intel.betting.bankroll import BankrollManager
from betting_intel.betting.monte_carlo import MonteCarloSimulator
from betting_intel.validation.cross_validation import TimeSeriesCrossValidator
from betting_intel.validation.overfitting import OverfittingDetector
from betting_intel.risk.kelly import MultiBetKelly, KellyCalculator
from betting_intel.risk.exposure import ExposureManager, ActiveBet
from betting_intel.risk.correlation import BetCorrelationTracker
from betting_intel.monitoring.drift import PerformanceTracker, FeatureDriftDetector


class BettingIntelligenceSystem:
    """Orchestrates the entire v2.0 betting intelligence pipeline.

    Integrates:
    - Data integrity & leakage validation
    - Time-series cross-validation & calibration
    - Feature engineering (v2.1: opponent-adjusted, SOS, style features)
    - Walk-forward backtesting
    - Edge detection & bankroll simulation
    - Monte Carlo risk analysis
    - Concept drift tracking
    - Multi-bet Kelly & exposure management
    """

    def __init__(self):
        self.loader = NBADataLoader()
        self.feature_engineer = FeatureEngineer()
        self.backtester = WalkForwardEngine()
        self.edge_detector = EdgeDetector()
        self.bankroll = BankrollManager()
        self.monte_carlo = MonteCarloSimulator(n_simulations=MONTE_CARLO_SIMULATIONS)
        self.results: dict = {}
        
        # New modules
        self.data_quality = DataQualityReport()
        self.performance_tracker = PerformanceTracker(model_name="ensemble")
        self.feature_drift_detector = None  # Initialized after feature selection
        self.exposure_manager = ExposureManager(bankroll=INITIAL_BANKROLL)
        self.correlation_tracker = BetCorrelationTracker()

    def run_full_pipeline(self) -> dict:
        """Execute the complete v2.1 pipeline with all new modules."""
        print("=" * 60)
        print("  BETTING INTELLIGENCE v2.1 - FULL PIPELINE")
        print("  Data integrity + Validation + Drift tracking + Risk management")
        print("=" * 60)

        # ── 0. Data Integrity & Quality ────────────────────────────────
        print("\n[0/7] Validating data integrity...")
        raw_df = self.loader.load_game_logs()
        games_df = self.loader.build_game_dataset(raw_df)
        raw_df = self.loader.compute_rest_days(raw_df)
        
        quality_report = self.data_quality.generate(
            df=games_df.head(1000),
            feature_cols=[c for c in games_df.columns if games_df[c].dtype in ("float64", "int64")],
            db_path=DB_PATH,
        )
        self.results["data_quality"] = quality_report
        print(f"  Data Quality Score: {quality_report.get('overall_score', {}).get('grade', 'N/A')}")
        print(f"    Rows: {quality_report['dataset_shape'][0]:,} x {quality_report['dataset_shape'][1]:,}")
        dq = quality_report.get('data_quality', {})
        missing = dq.get('missing_values', {}).get('missing_pct', 0)
        print(f"    Missing: {missing:.1f}% | Duplicates: {dq.get('duplicate_games', 0)}")

        # ── 1. Data Loading ───────────────────────────────────────────
        print("\n[1/7] Loading & cleaning data...")
        print(f"  Raw game logs: {len(raw_df)} rows")
        print(f"  Merged games:  {len(games_df)} rows")
        print(f"  Date range:    {games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}")
        print(f"  Unique teams:  {games_df['TEAM_NAME_home'].nunique()}")

        self.results["raw_data_shape"] = len(raw_df)
        self.results["games_data_shape"] = len(games_df)

        # ── 2. Feature Engineering (v2.1) ─────────────────────────────
        print("\n[2/7] Engineering advanced features (v2.1)...")
        print("  - Rolling averages & momentum features")
        print("  - Opponent-adjusted stats (off/def adjusted)")
        print("  - Strength of schedule (SOS) & SOS trend")
        print("  - Play-style features (3PT rate, FT rate, TS%, AST ratio)")
        print("  - Market line baselines (excluded from model training)")

        feature_df = self.feature_engineer.build_all_features(games_df, raw_df)
        feature_cols = self.feature_engineer.select_features(feature_df)

        print(f"\n  Features created: {len(feature_cols)}")
        if feature_cols:
            print(f"  Sample features: {feature_cols[:10]}...")

        # Remove rows with NaN features
        clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
        clean_df = clean_df.reset_index(drop=True)
        print(f"  Clean samples: {len(clean_df)} (dropped {len(feature_df) - len(clean_df)} incomplete rows)")

        self.results["feature_cols"] = feature_cols
        self.results["clean_df"] = clean_df

        # ── 3. Train, Backtest & Validate Models ──────────────────────
        print("\n[3/7] Running v2.1 model backtests + validation...")
        backtest_results = self._run_backtests(clean_df, feature_cols)
        self.results["backtest_results"] = backtest_results
        
        # Time-series cross-validation
        if len(clean_df) >= 300 and len(feature_cols) >= 5:
            cv = self._run_cross_validation(clean_df, feature_cols)
            self.results["cross_validation"] = cv

        # ── 4. Edge Detection ──────────────────────────────────────────
        print("\n[4/7] Detecting market edges...")
        edge_signals = self.edge_detector.detect_all(clean_df)
        self.results["edge_signals"] = edge_signals

        # ── 5. Bankroll + Kelly + Exposure Simulation ──────────────────
        print("\n[5/7] Simulating bankroll with Kelly + exposure control...")
        bankroll_results = self._simulate_bankroll(clean_df, feature_cols)
        self.results["bankroll_results"] = bankroll_results
        
        # Exposure report
        exposure_report = self.exposure_manager.get_report()
        self.results["exposure_report"] = exposure_report
        
        if exposure_report.violations:
            for v in exposure_report.violations:
                print(f"  [!] {v}")

        # ── 6. Monte Carlo + Overfitting Analysis ─────────────────────
        print("\n[6/7] Risk & overfitting analysis...")
        mc_results = self._run_monte_carlo(backtest_results)
        self.results["monte_carlo"] = mc_results
        
        # Overfitting detection
        of_result = self._check_overfitting(backtest_results)
        self.results["overfitting"] = of_result

        # ── 7. Summary + Drift Report ──────────────────────────────────
        print("\n[7/7] Generating v2.1 summary...")
        summary = self._generate_summary()
        self.results["summary"] = summary

        # Save results
        self._save_results()

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE (v2.1)")
        print("=" * 60)

        return self.results

    def _run_backtests(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Run all v2.0 backtest strategies.

        In FAST_MODE, runs only LightGBM + Momentum (2 models) for speed.
        In full mode, runs all 7+ model strategies.
        """
        results = {}

        if cfg.FAST_MODE:
            print("  [Fast Mode] Running only essential models (LightGBM + Momentum)")
            print("     Use python main.py --full for the full 7-model comparison.\n")

        # ── Baseline model for edge calculation ───────────────────────
        # Use a LightGBM model on BASIC features only (rolling averages,
        # pace, rest — like a modestly competent oddsmaker). The main model
        # has access to ALL 159 advanced features (opponent-adjusted, SOS,
        # play-style, momentum). Edge is the incremental value of advanced
        # features over basic ones, calculated in-fold (NO leakage).
        #
        # Key: using the same model type (LightGBM) for both ensures we measure
        # FEATURE value, not MODEL superiority. A Ridge on 35 basic features
        # is too biased — LightGBM on the same basics is a fairer benchmark.
        basic_feature_prefixes = ["avg_pts_", "avg_pm_", "avg_ts_", "avg_efg_",
                                   "pace_", "rest_", "is_b2b_", "tz_",
                                   "home_rest", "away_rest", "home_is_b2b", "away_is_b2b",
                                   "net_rating_", "opp_avg_pts_allowed_", "opp_avg_pts_scored_"]

        def _get_basic_features(all_cols):
            selected = []
            for col in all_cols:
                for prefix in basic_feature_prefixes:
                    if col.startswith(prefix):
                        selected.append(col)
                        break
            return selected if len(selected) >= 5 else all_cols[:20]

        def _baseline_lgbm():
            return TotalPointsPredictor("lightgbm")

        # ── Strategy 1: Total Points (LightGBM - v2.0) ───────────────
        print("  Running: Total Points (LightGBM - v2.0)...")
        result_lgbm = self.backtester.run_walk_forward(
            df=df,
            feature_cols=feature_cols,
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
            baseline_model_builder=_baseline_lgbm,
            baseline_feature_cols=_get_basic_features(feature_cols),
            strategy_name="pace_total",
            model_name=f"LightGBM_{PREFERRED_MODEL}",
            prediction_type="regression",
            make_bets=True,
        )
        results["total_lgbm"] = result_lgbm
        self._print_result(result_lgbm)

        if not cfg.FAST_MODE:
            # ── Strategy 2: Total Points (CatBoost - v2.0) ─────────────
            print("  Running: Total Points (CatBoost - v2.0)...")
            result_cb = self.backtester.run_walk_forward(
                df=df,
                feature_cols=feature_cols,
                target_col="total_points",
                model_builder=lambda: TotalPointsPredictor("catboost"),
                baseline_model_builder=_baseline_lgbm,
                baseline_feature_cols=_get_basic_features(feature_cols),
                strategy_name="pace_total",
                model_name="CatBoost",
                prediction_type="regression",
                make_bets=True,
            )
            results["total_catboost"] = result_cb
            self._print_result(result_cb)

            # ── Strategy 3: Total Points (Bayesian Ridge - v2.0) ─────────
            print("  Running: Total Points (Bayesian Ridge - v2.0)...")
            result_br = self.backtester.run_walk_forward(
                df=df,
                feature_cols=feature_cols,
                target_col="total_points",
                model_builder=lambda: TotalPointsPredictor("bayesian"),
                baseline_model_builder=_baseline_lgbm,
                baseline_feature_cols=_get_basic_features(feature_cols),
                strategy_name="pace_total",
                model_name="BayesianRidge",
                prediction_type="regression",
                make_bets=True,
            )
            results["total_bayesian"] = result_br
            self._print_result(result_br)

            # ── Strategy 4: Total Points (Random Forest - v2.0) ──────────
            print("  Running: Total Points (Random Forest - v2.0)...")
            result_rf = self.backtester.run_walk_forward(
                df=df,
                feature_cols=feature_cols,
                target_col="total_points",
                model_builder=lambda: TotalPointsPredictor("random_forest"),
                baseline_model_builder=_baseline_lgbm,
                baseline_feature_cols=_get_basic_features(feature_cols),
                strategy_name="pace_total",
                model_name="RandomForest",
                prediction_type="regression",
                make_bets=True,
            )
            results["total_rf"] = result_rf
            self._print_result(result_rf)

            # ── Strategy 5: Spread Prediction (LightGBM - v2.0) ──────────
            print("  Running: Spread Prediction (LightGBM - v2.0)...")
            result_spread = self.backtester.run_walk_forward(
                df=df,
                feature_cols=feature_cols,
                target_col="point_diff",
                model_builder=lambda: SpreadPredictor("lightgbm"),
                strategy_name="spread_model",
                model_name="LightGBM",
                prediction_type="regression",
                make_bets=True,
            )
            results["spread"] = result_spread
            self._print_result(result_spread)

            # ── Strategy 7: Stacking Ensemble (v2.0) ─────────────────────
            if ENABLE_STACKING_ENSEMBLE:
                print("  Running: Stacking Ensemble (v2.0)...")
                results["ensemble"] = self._build_ensemble(results)

            # ── Strategy 8: Elo-Based Prediction (v2.0) ──────────────────
            if "elo_home_pre" in df.columns:
                elo_features = [c for c in feature_cols if "elo" in c]
                if len(elo_features) >= 3:
                    print("  Running: Elo-Based Prediction (v2.0)...")
                    result_elo = self.backtester.run_walk_forward(
                        df=df,
                        feature_cols=elo_features,
                        target_col="total_points",
                        model_builder=lambda: TotalPointsPredictor("ridge"),
                        strategy_name="pace_total",
                        model_name="EloOnly",
                        prediction_type="regression",
                        make_bets=True,
                    )
                    results["elo_based"] = result_elo
                    self._print_result(result_elo)

        # ── Strategy 6: Momentum Reversion (ran in both modes) ────────
        print("  Running: Momentum Reversion (Calibrated LGBM - v2.0)...")

        # Select momentum features
        momentum_features = [c for c in feature_cols if any(
            kw in c for kw in ["streak", "momentum", "win_pct", "margin_volatility",
                               "elo_", "form_", "weighted_", "win_prob",
                               "rest_", "fatigue", "travel", "net_rating",
                               "mom_vs_opp", "sos_", "home_advantage",
                               "avg_pm_", "avg_pts_", "avg_ts_", "avg_efg_",
                               "tz_", "pace_"]
        )]
        if len(momentum_features) < 10:
            momentum_features = feature_cols

        df["home_win"] = (df["point_diff"] > 0).astype(int)
        result_momentum = self.backtester.run_walk_forward(
            df=df,
            feature_cols=momentum_features,
            target_col="home_win",
            model_builder=lambda: MomentumModel("lightgbm", calibrate=True),
            strategy_name="momentum",
            model_name="CalibratedLGBM",
            prediction_type="classification",
            make_bets=True,
        )
        results["momentum"] = result_momentum
        self._print_result(result_momentum)

        return results

    def _print_result(self, result: BacktestResult):
        """Print a backtest result summary."""
        if result.errors:
            print(f"    [!] Errors ({len(result.errors)}): {result.errors[0][:120]}")
        if result.total_bets > 0:
            print(f"    Bets: {result.total_bets} | "
                  f"Win: {result.win_rate:.1%} | "
                  f"Profit: {result.total_profit:+.1f}u | "
                  f"ROI: {result.roi:+.1f}% | "
                  f"Sharpe: {result.sharpe_ratio:.2f} | "
                  f"Max DD: {result.max_drawdown:.1f}u")
        elif result.model_metrics:
            m = result.model_metrics
            print(f"    Predictions: {m.get('n_predictions', 0)} | "
                  f"MAE: {m.get('mae', 0):.1f} | "
                  f"R2: {m.get('r2', 0):.3f}")

    def _build_ensemble(self, results: dict) -> BacktestResult:
        """Build a stacking ensemble from individual model results.

        Trains a Ridge regression meta-learner on out-of-fold predictions
        from each base model using TimeSeriesSplit to prevent look-ahead bias.
        """
        # Collect all regression results
        regression_keys = [k for k in results.keys()
                          if k.startswith("total_") and results[k].total_bets > 0]
        if len(regression_keys) < 2:
            return BacktestResult("ensemble", "Ensemble")

        # Use the best models for ensemble
        best_keys = sorted(regression_keys,
                          key=lambda k: results[k].sharpe_ratio,
                          reverse=True)[:3]

        if not best_keys:
            return BacktestResult("ensemble", "Ensemble")

        base_result = results[best_keys[0]]
        if base_result.bets_df.empty:
            return BacktestResult("ensemble", "Ensemble")

        ensemble_df = base_result.bets_df[["game_id", "game_date", "matchup",
                                            "actual_total", "market_line"]].copy()

        # Add predictions from other models
        valid_count = 1
        for key in best_keys[1:]:
            other = results[key]
            if not other.bets_df.empty and "predicted_total" in other.bets_df.columns:
                other_preds = other.bets_df[["game_id", "predicted_total"]].copy()
                other_preds.columns = ["game_id", f"pred_{key}"]
                ensemble_df = ensemble_df.merge(other_preds, on="game_id", how="inner")
                valid_count += 1

        if valid_count < 2 or len(ensemble_df) == 0:
            return BacktestResult("ensemble", "Ensemble")

        pred_cols = [c for c in ensemble_df.columns if c.startswith("pred_")]
        if len(pred_cols) == 0:
            return BacktestResult("ensemble", "Ensemble")

        # Sort by date for time-series split
        ensemble_df = ensemble_df.sort_values("game_date").reset_index(drop=True)

        # ── Train Ridge meta-learner with TimeSeriesSplit ──────────────
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import TimeSeriesSplit

        X_meta = ensemble_df[pred_cols].fillna(0).values
        y_meta = ensemble_df["actual_total"].values

        tscv = TimeSeriesSplit(n_splits=5)
        meta_preds = np.full(len(ensemble_df), np.nan)

        for train_idx, test_idx in tscv.split(X_meta):
            if len(train_idx) < 30:
                # Use all but last 20% for first fold
                split = int(len(ensemble_df) * 0.8)
                train_idx = np.arange(split)
                test_idx = np.arange(split, len(ensemble_df))

            X_train_fold = X_meta[train_idx]
            y_train_fold = y_meta[train_idx]
            X_test_fold = X_meta[test_idx]

            meta_model = Ridge(alpha=1.0)
            meta_model.fit(X_train_fold, y_train_fold)
            meta_preds[test_idx] = meta_model.predict(X_test_fold)

        # Fall back to simple average for any rows where meta-learner failed
        valid_meta = ~np.isnan(meta_preds)
        ensemble_df["ensemble_pred"] = np.where(
            valid_meta,
            meta_preds,
            ensemble_df[pred_cols].mean(axis=1).values,
        )

        n_meta = int(valid_meta.sum())
        n_avg = len(ensemble_df) - n_meta
        if n_meta > 10:
            model_label = "StackingEnsemble(Ridge+TSCV)"
        else:
            model_label = "StackingEnsemble(Avg)"

        print(f"    Ensemble: {n_meta} Ridge + {n_avg} avg predictions [{model_label}]")

        # Generate bets
        bet_records = []
        for _, row in ensemble_df.iterrows():
            edge_pct = (row["ensemble_pred"] - row["market_line"]) / max(row["market_line"], 1)

            if abs(edge_pct) < 0.02:
                continue

            if edge_pct > 0:
                bet_side = "OVER"
                won = row["actual_total"] > row["market_line"]
            else:
                bet_side = "UNDER"
                won = row["actual_total"] < row["market_line"]

            profit = 1.0 if won else -1.0
            bet_records.append({
                "game_date": row["game_date"],
                "game_id": row["game_id"],
                "matchup": row.get("matchup", ""),
                "strategy": "ensemble",
                "model": model_label,
                "bet_type": f"TOTAL_{bet_side}",
                "predicted_total": float(row["ensemble_pred"]),
                "market_line": float(row["market_line"]),
                "actual_total": float(row["actual_total"]),
                "edge_pct": float(edge_pct),
                "decimal_odds": 1.91,
                "outcome": "WIN" if won else "LOSS",
                "profit_units": profit,
            })

        ensemble = BacktestResult("ensemble", model_label)
        if bet_records:
            ensemble.bets_df = pd.DataFrame(bet_records)
            self.backtester._compute_performance(ensemble)

        self._print_result(ensemble)
        return ensemble

    def _simulate_bankroll(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Simulate real bankroll management with v2.0 models."""
        bankroll = BankrollManager(initial_bankroll=INITIAL_BANKROLL)
        results = {"bankroll": bankroll, "snapshots": []}

        # Use the best model for simulation (dynamically find from all backtest results)
        best_result = None
        for key, r in self.results.get("backtest_results", {}).items():
            if isinstance(r, BacktestResult) and r.total_bets > 0:
                if best_result is None or r.sharpe_ratio > best_result.sharpe_ratio:
                    best_result = r

        if best_result is None or best_result.bets_df.empty:
            print("  [!] No bets to simulate bankroll on")
            return results

        bets_df = best_result.bets_df.sort_values("game_date")

        for _, bet in bets_df.iterrows():
            edge = abs(bet.get("edge_pct", 0.03))
            prob = 0.5 + edge
            decimal_odds = float(bet.get("decimal_odds", 1.91))
            if decimal_odds <= 0:
                if "decimal_odds" not in bet or float(bet.get("decimal_odds", 0)) <= 0:
                    logger.warning("decimal_odds missing or invalid, defaulting to 1.91")
                decimal_odds = 1.91

            stake = bankroll.compute_kelly_stake(
                win_probability=prob,
                decimal_odds=decimal_odds,
                edge_pct=edge,
            )

            if stake[0] > 0:
                bankroll.place_bet(
                    game_id=str(bet["game_id"]),
                    strategy=best_result.strategy_name,
                    win_probability=prob,
                    decimal_odds=decimal_odds,
                    edge_pct=edge,
                )

                won = bet["outcome"] == "WIN"
                if bankroll.bets_placed:
                    bankroll.record_result(bankroll.bets_placed[-1], won)

                # Track exposure through the exposure manager
                if hasattr(self, 'exposure_manager'):
                    self.exposure_manager.add_bet(
                        ActiveBet(
                            bet_id=str(bet["game_id"]) + "_" + str(bet.get("game_date", "")),
                            game_id=str(bet["game_id"]),
                            matchup=bet.get("matchup", ""),
                            league="NBA",
                            bet_type="total",
                            side=bet.get("bet_type", "OVER").split("_")[-1],
                            stake_dollars=float(stake[0]),
                            decimal_odds=decimal_odds,
                            edge_pct=edge,
                            win_probability=prob,
                        )
                    )

            # Weekly snapshots
            if bankroll.total_bets % 10 == 0:
                bankroll.take_snapshot(str(bet["game_date"]))

        metrics = bankroll.get_metrics()
        results["metrics"] = metrics

        print(f"  Initial:  ${INITIAL_BANKROLL:,.2f}")
        print(f"  Final:    ${metrics['current_bankroll']:,.2f}")
        print(f"  Return:   {metrics['total_return_pct']:+.1f}%")
        print(f"  Drawdown: {metrics['drawdown_pct']:.1f}%")
        print(f"  Bets:     {metrics['total_bets']}")

        return results

    def _run_cross_validation(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Run time-series cross-validation for model stability assessment."""
        validator = TimeSeriesCrossValidator(n_splits=5, embargo=5)

        results = validator.validate(
            df=df,
            feature_cols=feature_cols,
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
            prediction_type="regression",
        )

        cv_summary = validator.get_summary()
        stability = validator.get_prediction_stability()

        print(f"  CV Folds: {cv_summary.get('n_folds', 0)}")
        print(f"  CV MAE:   {cv_summary.get('mae_mean', 0):.1f} +/- {cv_summary.get('mae_std', 0):.1f}")
        print(f"  CV R2:    {cv_summary.get('r2_mean', 0):.3f} +/- {cv_summary.get('r2_std', 0):.3f}")
        if stability:
            print(f"  Stability: CV(MAE)={stability.get('mae_cv', 0):.2f} "
                  f"(lower = more stable across time periods)")

        return {
            "folds": len(results),
            "summary": cv_summary,
            "stability": stability,
        }

    def _check_overfitting(self, backtest_results: dict) -> dict:
        """Run overfitting detection on backtest results.

        Uses REAL out-of-fold test metrics from the walk-forward backtest
        (per-fold train/test metrics stored in BacktestResult.fold_metrics)
        instead of copying train metrics.
        """
        detector = OverfittingDetector()

        # Find models with enough bets
        valid_results = []
        for key, result in backtest_results.items():
            if isinstance(result, BacktestResult) and result.total_bets >= 20:
                valid_results.append((key, result))

        if not valid_results:
            return {"error": "No valid backtest results for overfitting analysis"}

        # Analyze the best model (highest Sharpe)
        best_key, best_result = max(valid_results, key=lambda x: x[1].sharpe_ratio)

        # Use REAL out-of-fold test metrics from fold_metrics instead of copying train_metrics
        if best_result.fold_metrics:
            # Average test metrics across all folds
            avg_test_r2 = float(np.mean([fm.get("test_r2", 0) for fm in best_result.fold_metrics]))
            avg_test_mae = float(np.mean([fm.get("test_mae", 0) for fm in best_result.fold_metrics]))
            avg_train_r2 = float(np.mean([fm.get("train_r2", 0) for fm in best_result.fold_metrics]))
            avg_train_mae = float(np.mean([fm.get("train_mae", 0) for fm in best_result.fold_metrics]))

            train_metrics = {
                "win_rate": best_result.win_rate,
                "r2": avg_train_r2,
                "mae": avg_train_mae,
                "sharpe_ratio": best_result.sharpe_ratio,
            }
            test_metrics = {
                "win_rate": best_result.win_rate,
                "r2": avg_test_r2,
                "mae": avg_test_mae,
                "sharpe_ratio": best_result.sharpe_ratio,
            }
        else:
            # Fallback: use overall model metrics (still better than copy)
            train_metrics = {
                "win_rate": best_result.win_rate,
                "r2": best_result.model_metrics.get("avg_train_r2", best_result.model_metrics.get("r2", 0)),
                "mae": best_result.model_metrics.get("mae", 0),
                "sharpe_ratio": best_result.sharpe_ratio,
            }
            test_metrics = {
                "win_rate": best_result.win_rate,
                "r2": best_result.model_metrics.get("r2", 0),
                "mae": best_result.model_metrics.get("mae", 0),
                "sharpe_ratio": best_result.sharpe_ratio,
            }

        # CV results (if available)
        cv_results = self.results.get("cross_validation", {}).get("summary", {})
        cv_list = [{"r2": cv_results.get("r2_mean", 0)}] if cv_results else []

        analysis = detector.analyze(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            cv_results=cv_list,
            n_strategies_tested=len(valid_results),
            n_observations=best_result.total_bets,
            sharpe_ratio=best_result.sharpe_ratio,
        )

        print(f"  Model: {best_key}")
        print(f"  Overfitting Score: {analysis.get('overfitting_score', 0):.1f}/100")
        print(f"  Verdict: {analysis.get('verdict', 'N/A')}")
        print(f"  Train R2: {train_metrics['r2']:.3f} | Test R2: {test_metrics['r2']:.3f}")
        print(f"  Train MAE: {train_metrics['mae']:.1f} | Test MAE: {test_metrics['mae']:.1f}")
        print(f"  Deflated Sharpe: {analysis.get('deflated_sharpe', 0):.2f}")

        return analysis

    def _run_monte_carlo(self, backtest_results: dict) -> Optional[dict]:
        """Run Monte Carlo simulation on backtest results."""
        # Find the best result with enough bets
        best_bets_df = None
        best_name = None

        for key, result in backtest_results.items():
            if isinstance(result, BacktestResult) and not result.bets_df.empty and result.total_bets >= 20:
                if best_bets_df is None or result.total_bets > len(best_bets_df):
                    best_bets_df = result.bets_df
                    best_name = key

        if best_bets_df is None:
            print("  [!] No backtest results with sufficient bets for Monte Carlo")
            return None

        print(f"  Simulating from '{best_name}' ({len(best_bets_df)} bets)...")
        sim_result = self.monte_carlo.simulate_from_bets(
            bets_df=best_bets_df,
            n_games_per_season=500,
            initial_bankroll=INITIAL_BANKROLL,
            stake_per_bet=UNIT_SIZE,
        )

        print(f"  Median profit: ${sim_result.median_profit:,.0f}")
        print(f"  95% CI: ${sim_result.profit_ci_95[0]:,.0f} to ${sim_result.profit_ci_95[1]:,.0f}")
        print(f"  P(Profitable): {sim_result.probability_profit:.1%}")
        print(f"  Risk of Ruin:   {sim_result.risk_of_ruin:.1%}")

        return {
            "simulation": sim_result,
            "source_strategy": best_name,
        }

    def _generate_summary(self) -> str:
        """Generate comprehensive v2.1 summary with new modules."""
        lines = [
            "=" * 60,
            "  BETTING INTELLIGENCE v2.1 - SYSTEM SUMMARY",
            "=" * 60,
            "",
            f"  Data analyzed: {self.results.get('games_data_shape', 0):,} games",
            f"  Features engineered (v2.1): {len(self.results.get('feature_cols', []))}",
            f"  (Opponent-adj, SOS, play-style, rolling averages, momentum)",
        ]
        
        # Data quality
        dq = self.results.get("data_quality", {})
        if dq:
            overall = dq.get("overall_score", {})
            lines.extend([
                "",
                "  -- Data Quality & Integrity --",
                f"  Quality: {overall.get('grade', 'N/A')} ({overall.get('score', 0):.0f}/100)",
            ])
            for d in overall.get("deductions", []):
                lines.append(f"    [!] {d}")

        lines.extend([
            "",
            "  -- Backtest Results (v2.1 Models) --",
        ])

        for key, result in self.results.get("backtest_results", {}).items():
            if isinstance(result, BacktestResult):
                lines.append(
                    f"  {key:18s}: {result.total_bets:3d} bets | "
                    f"WR {result.win_rate:.1%} | "
                    f"{result.total_profit:+.1f}u | "
                    f"ROI {result.roi:+.1f}% | "
                    f"Sharpe {result.sharpe_ratio:.2f}"
                )

        lines.append("")
        lines.append("  -- Edge Detection --")
        for signal in self.results.get("edge_signals", []):
            lines.append(f"  {signal.strategy:12s}: {signal.description[:60]}")

        # Cross-validation
        cv = self.results.get("cross_validation", {})
        if cv:
            cv_summary = cv.get("summary", {})
            lines.extend([
                "",
                "  -- Time-Series Cross-Validation --",
                f"  Folds: {cv_summary.get('n_folds', 0)}",
                f"  MAE:    {cv_summary.get('mae_mean', 0):.1f} +/- {cv_summary.get('mae_std', 0):.1f}",
                f"  R2:     {cv_summary.get('r2_mean', 0):.3f} +/- {cv_summary.get('r2_std', 0):.3f}",
            ])

        bankroll_metrics = self.results.get("bankroll_results", {}).get("metrics", {})
        if bankroll_metrics:
            lines.extend([
                "",
                "  -- Bankroll + Kelly Simulation --",
                f"  Start: ${INITIAL_BANKROLL:,.0f} -> End: ${bankroll_metrics.get('current_bankroll', 0):,.0f}",
                f"  Return: {bankroll_metrics.get('total_return_pct', 0):+.1f}% | "
                f"Max DD: {bankroll_metrics.get('drawdown_pct', 0):.1f}%",
            ])

        # Exposure report
        exposure = self.results.get("exposure_report")
        if exposure:
            lines.extend([
                "",
                "  -- Exposure Management --",
                f"  Active bets: {exposure.n_active_bets}",
                f"  Total exposure: ${exposure.total_exposure:,.0f} ({exposure.bankroll_pct:.1%} of bankroll)",
            ])
            if exposure.violations:
                for v in exposure.violations[:3]:
                    lines.append(f"    [!] {v}")

        # Monte Carlo results
        mc = self.results.get("monte_carlo", {})
        if mc and mc.get("simulation"):
            sim = mc["simulation"]
            lines.extend([
                "",
                "  -- Monte Carlo Risk Analysis --",
                f"  Simulations: {sim.n_simulations:,}",
                f"  Median Profit: ${sim.median_profit:,.0f}",
                f"  95% CI: ${sim.profit_ci_95[0]:,.0f} to ${sim.profit_ci_95[1]:,.0f}",
                f"  P(Profitable): {sim.probability_profit:.1%}",
                f"  Risk of Ruin:   {sim.risk_of_ruin:.1%}",
            ])

        # Overfitting analysis
        of = self.results.get("overfitting", {})
        if of and "error" not in of:
            lines.extend([
                "",
                "  -- Overfitting Detection --",
                f"  Score: {of.get('overfitting_score', 0):.1f}/100",
                f"  Verdict: {of.get('verdict', 'N/A')}",
                f"  Deflated Sharpe: {of.get('deflated_sharpe', 0):.2f}",
            ])
            if of.get("warnings"):
                for w in of["warnings"][:3]:
                    lines.append(f"    [!] {w}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _save_results(self):
        """Save all v2.0 results to disk."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_path = output_dir / f"summary_v2_{timestamp}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.results.get("summary", ""))

        # Save backtest results as CSV
        for key, result in self.results.get("backtest_results", {}).items():
            if isinstance(result, BacktestResult) and not result.bets_df.empty:
                csv_path = output_dir / f"bets_{key}_v2_{timestamp}.csv"
                result.bets_df.to_csv(csv_path, index=False)
                print(f"  Saved: {csv_path.name}")

        # Save bankroll history
        bankroll_results = self.results.get("bankroll_results", {})
        if "snapshots" in bankroll_results:
            snapshots = bankroll_results["snapshots"]
            if snapshots:
                import json
                snap_data = [
                    {"date": s.date, "bankroll": s.bankroll, "bets": s.total_bets,
                     "drawdown": s.current_drawdown}
                    for s in snapshots
                ]
                snap_path = output_dir / f"bankroll_v2_{timestamp}.json"
                with open(snap_path, "w", encoding="utf-8") as f:
                    json.dump(snap_data, f, indent=2)

        # Save Monte Carlo report
        mc = self.results.get("monte_carlo", {})
        if mc and mc.get("simulation"):
            mc_path = output_dir / f"monte_carlo_v2_{timestamp}.txt"
            with open(mc_path, "w", encoding="utf-8") as f:
                f.write(self.monte_carlo.format_report(mc["simulation"]))

        print(f"\n  Results saved to: {output_dir}/")
        print(f"  Summary: summary_v2_{timestamp}.txt")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Betting Intelligence v2.0 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Fast mode: LightGBM + Momentum only (default)
  python main.py --full                    # Full pipeline: all 7+ models
  python main.py --tune                    # Enable hyperparameter tuning (Optuna)
  python main.py --live                    # Live predictions for upcoming games
        """
    )
    parser.add_argument("--live", action="store_true",
                        help="Run live prediction mode: fetch upcoming games from TheOddsAPI")
    parser.add_argument("--full", action="store_true",
                        help="Run full pipeline with all models (default is fast mode)")
    parser.add_argument("--tune", action="store_true",
                        help="Enable Optuna hyperparameter tuning (off by default for speed)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Explicitly skip hyperparameter tuning")
    args = parser.parse_args()

    # Override config based on CLI args
    if args.full:
        cfg.FAST_MODE = False

    tuning = not args.no_tune if args.no_tune else args.tune
    if tuning:
        from betting_intel import config as cfg_inner
        cfg_inner.ENABLE_HYPERPARAMETER_TUNING = True

    if args.live:
        # Run the live prediction pipeline (modular pipeline)
        print("Starting LIVE prediction pipeline...\n")
        from betting_intel.pipeline import PredictionPipeline
        live_args = argparse.Namespace(
            live=True,
            full=False,
            recommend_only=False,
            simulate=False,
            scheduled=False,
            days_history=90,
            data_source=None,
            csv_path=None,
            no_tune=not tuning,
            model_dir="models/saved",
            ensemble=True,
            strategy="all",
            bankroll=INITIAL_BANKROLL,
            kelly_fraction=0.25,
            max_exposure=0.20,
            min_edge=0.02,
            output=None,
            html=False,
            verbose=False,
        )
        pipeline = PredictionPipeline(live_args)
        results = pipeline.run()
    else:
        # Run the backtesting pipeline
        mode_label = "FAST" if cfg.FAST_MODE else "FULL"
        print(f"Running {mode_label} pipeline...")
        if cfg.FAST_MODE:
            print("  Use --full for all models, --tune for hyperparameter tuning\n")
        else:
            print("  Running all 7+ models. This may take a few minutes...\n")

        system = BettingIntelligenceSystem()
        results = system.run_full_pipeline()

        # Print final summary
        print("\n" + results.get("summary", ""))


if __name__ == "__main__":
    main()
