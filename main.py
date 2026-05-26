"""
Main pipeline: orchestrates data loading, feature engineering (v2.0),
advanced modeling, backtesting, edge detection, bankroll simulation,
and Monte Carlo risk analysis.

Run: python main.py  (from the betting-intelligence directory)
"""

import sys
import os
import warnings
from typing import Optional, Dict, List
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DB_PATH, OUTPUT_DIR, VERBOSE,
    ENABLE_LINEAR_MODEL, ENABLE_XGBOOST_MODEL, ENABLE_ENSEMBLE,
    STRATEGIES, INITIAL_BANKROLL, UNIT_SIZE,
    ENABLE_HYPERPARAMETER_TUNING, ENABLE_STACKING_ENSEMBLE,
    ENABLE_MONTE_CARLO, MONTE_CARLO_SIMULATIONS,
    PREFERRED_MODEL, FAST_MODE,
)
from data.loader import NBADataLoader
from data.features import FeatureEngineer
from models.predictors import (
    TotalPointsPredictor, SpreadPredictor, MomentumModel,
    StackingEnsemblePredictor, create_best_model, create_tuned_lgbm_regressor
)
from backtesting.engine import WalkForwardEngine, BacktestResult
from backtesting.metrics import BacktestMetrics
from betting.edge import EdgeDetector
from betting.bankroll import BankrollManager
from betting.monte_carlo import MonteCarloSimulator


class BettingIntelligenceSystem:
    """Orchestrates the entire v2.0 betting intelligence pipeline."""

    def __init__(self):
        self.loader = NBADataLoader()
        self.feature_engineer = FeatureEngineer()
        self.backtester = WalkForwardEngine()
        self.edge_detector = EdgeDetector()
        self.bankroll = BankrollManager()
        self.monte_carlo = MonteCarloSimulator(n_simulations=MONTE_CARLO_SIMULATIONS)
        self.results: dict = {}

    def run_full_pipeline(self) -> dict:
        """Execute the complete v2.0 pipeline."""
        print("=" * 60)
        print("  BETTING INTELLIGENCE v2.0 - FULL PIPELINE")
        print("  Advanced features + State-of-the-art models + Monte Carlo")
        print("=" * 60)

        # ── 1. Load Data ──────────────────────────────────────────────
        print("\n[1/7] Loading data...")
        raw_df = self.loader.load_game_logs()
        games_df = self.loader.build_game_dataset(raw_df)
        raw_df = self.loader.compute_rest_days(raw_df)

        print(f"  Raw game logs: {len(raw_df)} rows")
        print(f"  Merged games:  {len(games_df)} rows")
        print(f"  Date range:    {games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}")
        print(f"  Unique teams:  {games_df['TEAM_NAME_home'].nunique()}")

        self.results["raw_data_shape"] = len(raw_df)
        self.results["games_data_shape"] = len(games_df)

        # ── 2. Feature Engineering (v2.0) ─────────────────────────────
        print("\n[2/7] Engineering advanced features (v2.0)...")
        print("  - Elo ratings with K-factor optimization")
        print("  - True Shooting %, Points Per Possession")
        print("  - Opponent-adjusted stats & strength of schedule")
        print("  - Travel distance & schedule fatigue")
        print("  - Weighted/decay-based momentum features")
        print("  - Scoring consistency & volatility metrics")

        feature_df = self.feature_engineer.build_all_features(games_df, raw_df)
        feature_cols = self.feature_engineer.select_features(feature_df)

        print(f"\n  Features created: {len(feature_cols)}")
        print(f"  Sample features: {feature_cols[:10]}...")

        # Remove rows with NaN features
        clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
        clean_df = clean_df.reset_index(drop=True)
        print(f"  Clean samples: {len(clean_df)} (dropped {len(feature_df) - len(clean_df)} incomplete rows)")

        self.results["feature_cols"] = feature_cols
        self.results["clean_df"] = clean_df

        # ── 3. Train & Backtest Advanced Models ───────────────────────
        print("\n[3/7] Running v2.0 model backtests...")
        backtest_results = self._run_backtests(clean_df, feature_cols)
        self.results["backtest_results"] = backtest_results

        # ── 4. Edge Detection (v2.0) ──────────────────────────────────
        print("\n[4/7] Detecting market edges...")
        edge_signals = self.edge_detector.detect_all(clean_df)
        self.results["edge_signals"] = edge_signals

        # ── 5. Bankroll Simulation ────────────────────────────────────
        print("\n[5/7] Simulating bankroll management...")
        bankroll_results = self._simulate_bankroll(clean_df, feature_cols)
        self.results["bankroll_results"] = bankroll_results

        # ── 6. Monte Carlo Risk Analysis (v2.0) ──────────────────────
        print("\n[6/7] Running Monte Carlo risk analysis...")
        mc_results = self._run_monte_carlo(backtest_results)
        self.results["monte_carlo"] = mc_results

        # ── 7. Summary ────────────────────────────────────────────────
        print("\n[7/7] Generating v2.0 summary...")
        summary = self._generate_summary()
        self.results["summary"] = summary

        # Save results
        self._save_results()

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)

        return self.results

    def _run_backtests(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Run all v2.0 backtest strategies.

        In FAST_MODE, runs only LightGBM + Momentum (2 models) for speed.
        In full mode, runs all 7+ model strategies.
        """
        results = {}

        if FAST_MODE:
            print("  [Fast Mode] Running only essential models (LightGBM + Momentum)")
            print("     Use python main.py for the full 7-model comparison.\n")

        # ── Strategy 1: Total Points (LightGBM - v2.0) ───────────────
        print("  Running: Total Points (LightGBM - v2.0)...")
        result_lgbm = self.backtester.run_walk_forward(
            df=df,
            feature_cols=feature_cols,
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
            strategy_name="pace_total",
            model_name=f"LightGBM_{PREFERRED_MODEL}",
            prediction_type="regression",
            make_bets=True,
        )
        results["total_lgbm"] = result_lgbm
        self._print_result(result_lgbm)

        if not FAST_MODE:
            # ── Strategy 2: Total Points (CatBoost - v2.0) ─────────────
            print("  Running: Total Points (CatBoost - v2.0)...")
            result_cb = self.backtester.run_walk_forward(
                df=df,
                feature_cols=feature_cols,
                target_col="total_points",
                model_builder=lambda: TotalPointsPredictor("catboost"),
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
                  f"R²: {m.get('r2', 0):.3f}")

    def _build_ensemble(self, results: dict) -> BacktestResult:
        """Build a stacking ensemble from individual model results."""
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

        # Ensemble prediction = simple average
        pred_cols = [c for c in ensemble_df.columns if c.startswith("pred_")]
        if len(pred_cols) == 0:
            return BacktestResult("ensemble", "Ensemble")

        ensemble_df["ensemble_pred"] = ensemble_df[pred_cols].mean(axis=1)

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
                "model": "StackingEnsemble",
                "bet_type": f"TOTAL_{bet_side}",
                "predicted_total": float(row["ensemble_pred"]),
                "market_line": float(row["market_line"]),
                "actual_total": float(row["actual_total"]),
                "edge_pct": float(edge_pct),
                "outcome": "WIN" if won else "LOSS",
                "profit_units": profit,
            })

        ensemble = BacktestResult("ensemble", "StackingEnsemble")
        if bet_records:
            ensemble.bets_df = pd.DataFrame(bet_records)
            self.backtester._compute_performance(ensemble)

        self._print_result(ensemble)
        return ensemble

    def _simulate_bankroll(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Simulate real bankroll management with v2.0 models."""
        bankroll = BankrollManager(initial_bankroll=INITIAL_BANKROLL)
        results = {"bankroll": bankroll, "snapshots": []}

        # Use the best model for simulation
        best_result = None
        for key in ["total_lgbm", "total_catboost", "total_bayesian", "ensemble"]:
            if key in self.results.get("backtest_results", {}):
                r = self.results["backtest_results"][key]
                if r.total_bets > 0 and (best_result is None or r.sharpe_ratio > best_result.sharpe_ratio):
                    best_result = r

        if best_result is None or best_result.bets_df.empty:
            print("  [!] No bets to simulate bankroll on")
            return results

        bets_df = best_result.bets_df.sort_values("game_date")

        for _, bet in bets_df.iterrows():
            edge = abs(bet.get("edge_pct", 0.03))
            prob = 0.5 + edge

            stake = bankroll.compute_kelly_stake(
                win_probability=prob,
                decimal_odds=1.91,
                edge_pct=edge,
            )

            if stake[0] > 0:
                bankroll.place_bet(
                    game_id=str(bet["game_id"]),
                    strategy=best_result.strategy_name,
                    win_probability=prob,
                    decimal_odds=1.91,
                    edge_pct=edge,
                )

                won = bet["outcome"] == "WIN"
                if bankroll.bets_placed:
                    bankroll.record_result(bankroll.bets_placed[-1], won)

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
        """Generate comprehensive v2.0 summary."""
        lines = [
            "=" * 60,
            "  BETTING INTELLIGENCE v2.0 - SYSTEM SUMMARY",
            "=" * 60,
            "",
            f"  Data analyzed: {self.results.get('games_data_shape', 0):,} games",
            f"  Features engineered (v2.0): {len(self.results.get('feature_cols', []))}",
            f"  (Elo, TS%, opponent-adj, travel fatigue, weighted momentum)",
            "",
            "  -- Backtest Results (v2.0 Models) --",
        ]

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

        bankroll_metrics = self.results.get("bankroll_results", {}).get("metrics", {})
        if bankroll_metrics:
            lines.extend([
                "",
                "  -- Bankroll Simulation --",
                f"  Start: ${INITIAL_BANKROLL:,.0f} -> End: ${bankroll_metrics.get('current_bankroll', 0):,.0f}",
                f"  Return: {bankroll_metrics.get('total_return_pct', 0):+.1f}% | "
                f"Max DD: {bankroll_metrics.get('drawdown_pct', 0):.1f}%",
            ])

        # Monte Carlo results
        mc = self.results.get("monte_carlo", {})
        if mc and mc.get("simulation"):
            sim = mc["simulation"]
            lines.extend([
                "",
                "  -- Monte Carlo Risk Analysis (v2.0) --",
                f"  Simulations: {sim.n_simulations:,}",
                f"  Median Profit: ${sim.median_profit:,.0f}",
                f"  95% CI: ${sim.profit_ci_95[0]:,.0f} to ${sim.profit_ci_95[1]:,.0f}",
                f"  P(Profitable): {sim.probability_profit:.1%}",
                f"  Risk of Ruin:   {sim.risk_of_ruin:.1%}",
            ])

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
  python main.py --live --demo             # Live predictions in demo mode (no API key)
        """
    )
    parser.add_argument("--live", action="store_true",
                        help="Run live prediction mode: fetch upcoming games from TheOddsAPI")
    parser.add_argument("--demo", action="store_true",
                        help="Run with demo data (no API key needed)")
    parser.add_argument("--full", action="store_true",
                        help="Run full pipeline with all models (default is fast mode)")
    parser.add_argument("--tune", action="store_true",
                        help="Enable Optuna hyperparameter tuning (off by default for speed)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Explicitly skip hyperparameter tuning")
    args = parser.parse_args()

    # Override config based on CLI args
    if args.full:
        import config
        config.FAST_MODE = False

    tuning = not args.no_tune if args.no_tune else args.tune
    if tuning:
        import config
        config.ENABLE_HYPERPARAMETER_TUNING = True

    if args.live:
        # Run the live prediction engine instead
        print("Starting LIVE prediction engine with TheOddsAPI...\n")
        from predict_tomorrow import AdvancedPredictionEngine
        engine = AdvancedPredictionEngine(
            tune_hyperparams=tuning,
            live_mode=True,
            demo_mode=args.demo,
        )
        results = engine.run()
    else:
        # Run the backtesting pipeline
        mode_label = "FAST" if FAST_MODE else "FULL"
        print(f"Running {mode_label} pipeline...")
        print("  Use --full for all models, --tune for hyperparameter tuning\n")

        system = BettingIntelligenceSystem()
        results = system.run_full_pipeline()

        # Print final summary
        print("\n" + results.get("summary", ""))


if __name__ == "__main__":
    main()
