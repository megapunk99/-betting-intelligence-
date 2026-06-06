"""
Risk analysis mixin — risk management, Monte Carlo simulation, edge detection, and backtesting.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from betting_intel.pipeline.bootstrap import (
    logger, HAS_RISK, HAS_BETTING, HAS_BACKTESTING,
    KellyCalculator, ExposureManager, BetCorrelationTracker,
    MonteCarloSimulator, EdgeDetector, BacktestMetrics,
)


class RiskAnalysisMixin:
    """Mixin providing risk management, simulation, edge detection, and backtesting."""

    # ── Risk Management ─────────────────────────────────────────────

    def apply_risk_management(self, recommendations: List[Dict[str, Any]],
                               predictions_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
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
            risk_result = self._apply_kelly_risk(recommendations, predictions_df, risk_result)
        else:
            risk_result = self._basic_stake_sizing(recommendations, risk_result)

        self.results["risk_assessment"] = risk_result
        return risk_result

    def _apply_kelly_risk(self, recommendations: List[Dict[str, Any]],
                           predictions_df: Optional[pd.DataFrame],
                           risk_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply full Kelly criterion with exposure management."""
        try:
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

                if odds > 0:
                    decimal_odds = 1 + odds / 100
                elif odds < 0:
                    decimal_odds = 1 + 100 / abs(odds)
                else:
                    decimal_odds = 1.91

                kelly_pct, _ = kelly.compute_kelly(
                    win_probability=0.5 + edge / 2,
                    decimal_odds=decimal_odds,
                )

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

            self._run_correlation_analysis(predictions_df, risk_result)
            self.results["recommendations"] = sized_bets

        except Exception as e:
            print(f"  ⚠  Risk management failed: {e}")

        return risk_result

    def _run_correlation_analysis(self, predictions_df: Optional[pd.DataFrame],
                                    risk_result: Dict[str, Any]):
        """Run bet correlation analysis."""
        try:
            tracker = BetCorrelationTracker()
            corr_df = predictions_df if predictions_df is not None else getattr(self, 'predictions_df', None)
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

    def _basic_stake_sizing(self, recommendations: List[Dict[str, Any]],
                             risk_result: Dict[str, Any]) -> Dict[str, Any]:
        """Basic stake sizing when full risk module is unavailable."""
        print("  ℹ  Recommendation engine not available")
        for bet in recommendations:
            edge = bet.get("edge", 0)
            kelly_pct = min(edge * self.args.kelly_fraction * 4, 0.05)
            bet["kelly_pct"] = kelly_pct
            bet["stake"] = round(kelly_pct * self.args.bankroll, 2)
        risk_result["bets"] = recommendations
        return risk_result

    # ── Monte Carlo Simulation ──────────────────────────────────────

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
                bet_rows = []
                for rec in recommendations:
                    outcome = "WIN" if rec.get("edge", 0) > rec.get("min_edge", 0.02) else "LOSS"
                    bet_rows.append({
                        "game_id": rec.get("game_id", ""),
                        "game_date": rec.get("game_date", ""),
                        "outcome": outcome,
                        "profit_units": 1.0 if outcome == "WIN" else -1.0,
                        "edge_pct": rec.get("edge", 0),
                    })

                if bet_rows:
                    bets_df = pd.DataFrame(bet_rows)
                    result = sim.simulate_from_bets(
                        bets_df=bets_df,
                        n_games_per_season=len(bets_df),
                        initial_bankroll=self.args.bankroll,
                        stake_per_bet=0.02,
                    )
                    if result:
                        print(f"  ✅  10,000 simulations complete")
                        print(f"       Median return: ${result.median_profit:+.2f}")
                        print(f"       Upside (90th): ${result.percentile_90:+.2f}")
                        print(f"       Downside (10th): ${result.percentile_10:+.2f}")
                        sim_result = {
                            "median_return": result.median_profit,
                            "median": result.median_profit,
                            "upside_90th": result.percentile_90,
                            "percentile_90": result.percentile_90,
                            "downside_10th": result.percentile_10,
                            "percentile_10": result.percentile_10,
                        }
                        self.results["simulation"] = sim_result
            except Exception as e:
                print(f"  ⚠  Monte Carlo simulation failed: {e}")

        return sim_result

    # ── Edge Detection ──────────────────────────────────────────────

    def detect_edges(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect betting edges from prediction residuals."""
        print("\n" + "=" * 70)
        print("  🎯  STAGE 10: EDGE DETECTION")
        print("=" * 70)

        edges: List[Dict[str, Any]] = []

        if HAS_BETTING:
            try:
                detector = EdgeDetector()
                if 'rest_advantage' in predictions_df.columns:
                    rest_edge = detector.detect_rest_edge(predictions_df)
                    if rest_edge:
                        print(f"  ℹ  Rest edge detected: {rest_edge}")
                if 'point_diff' in predictions_df.columns or 'TEAM_NAME_home' in predictions_df.columns:
                    home_edge = detector.detect_home_court_edge(predictions_df)
                    if home_edge:
                        print(f"  ℹ  Home court advantage edge: {home_edge}")
            except Exception as e:
                print(f"  ⚠  Edge detection failed: {e}")

        # Simple edge calculation from prediction residuals
        if not edges and "predicted_total" in predictions_df.columns and "market_total" in predictions_df.columns:
            print("  ℹ  Using simple edge calculation...")
            for _, row in predictions_df.iterrows():
                pt = row.get("predicted_total", 0)
                mt = row.get("market_total", 0)
                if pt and mt:
                    pct_edge = (pt - mt) / mt
                    if abs(pct_edge) > self.args.min_edge:
                        edges.append({
                            "team": row.get("home_team", row.get("team", "?")),
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

    # ── Backtesting ────────────────────────────────────────────────

    def run_backtest(self, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run backtest on historical predictions using ONLY out-of-sample rows.

        Walk-forward validation sets _is_oos=True on test fold predictions.
        We compute metrics ONLY on those rows, preventing the illusion of
        100% win rates from training data leakage.

        Additionally tracks:
        - Train vs Test R² (gap = overfitting signal)
        - Per-fold metrics from walk-forward
        - Number of OOS vs total predictions
        """
        if self.args.live:
            print("\n  ⏩  Backtesting skipped in live mode")
            return None

        print("\n" + "=" * 70)
        print("  ⏪  STAGE 11: BACKTESTING (OOS only)")
        print("=" * 70)

        if not HAS_BACKTESTING:
            print("  ℹ  Backtesting module not available")
            return None

        try:
            if (hasattr(self, 'predictions_df') and self.predictions_df is not None
                    and 'predicted_total' in self.predictions_df.columns):

                # ── Only compute on OOS rows ───────────────────────────────
                df = self.predictions_df
                if '_is_oos' in df.columns:
                    oos_df = df[df['_is_oos']].copy()
                    total_with_preds = df['predicted_total'].notna().sum()
                    n_oos = len(oos_df)
                    print(f"  📊  OOS predictions: {n_oos:,} of {total_with_preds:,} total")
                    if n_oos == 0:
                        print("  ⚠  No OOS predictions found — backtest skipped")
                        return None
                else:
                    # Fallback: use all rows with predictions (legacy mode)
                    oos_df = df[df['predicted_total'].notna()].copy()
                    print(f"  ℹ  No _is_oos flag found — using all {len(oos_df)} predictions")
                    n_oos = len(oos_df)

                # ── Build bet records from OOS predictions ─────────────────
                bet_rows = []
                for idx, row in oos_df.iterrows():
                    pred = row.get('predicted_total', 0)
                    if not pred or np.isnan(pred):
                        continue

                    # Get actual total points
                    actual = features_df.loc[idx, 'total_points'] if idx in features_df.index else None
                    if actual is None or np.isnan(actual):
                        continue

                    # Market line: prefer market_total, then market_line_baseline
                    market_line = row.get(
                        'market_total',
                        row.get('market_line_baseline', 0)
                    )
                    if market_line is None or market_line <= 0:
                        # Use trailing average as fallback
                        market_line = row.get('trailing_avg_total_10g', 0)

                    if market_line <= 0:
                        continue

                    # Check if we have a genuine edge
                    edge = (pred - market_line) / market_line
                    if abs(edge) < self.args.min_edge:
                        continue

                    # Determine outcome
                    if (edge > 0 and actual > market_line) or (edge < 0 and actual < market_line):
                        outcome = "WIN"
                        # Use actual decimal odds (default -110 = 1.909) for realistic payout
                        decimal_odds = row.get('home_odds' if edge > 0 else 'away_odds', row.get('decimal_odds', 1.909))
                        try:
                            dec = float(decimal_odds)
                            if dec < 1.01:
                                dec = 1.909
                        except (ValueError, TypeError):
                            dec = 1.909
                        profit = dec - 1.0  # net profit after stake
                    elif abs(actual - market_line) < 0.5:
                        outcome = "PUSH"
                        profit = 0.0
                    else:
                        outcome = "LOSS"
                        profit = -1.0  # lose the full stake

                    bet_rows.append({
                        "game_date": str(row.get('game_date', '')),
                        "outcome": outcome,
                        "profit_units": profit,
                        "edge_pct": edge,
                    })

                n_bets = len(bet_rows)
                if n_bets == 0:
                    print("  ℹ  No bets cleared edge threshold in OOS data")
                    return None

                # ── Compute metrics ────────────────────────────────────────
                bets_df = pd.DataFrame(bet_rows)
                metrics = BacktestMetrics.compute_all(bets_df)

                if metrics and "error" not in metrics:
                    wins = metrics.get("wins", 0)
                    losses = metrics.get("losses", 0)
                    total_return = metrics.get("total_profit_units", 0)
                    sharpe = metrics.get("sharpe_ratio", 0)
                    win_rate = metrics.get("win_rate", 0)
                    p_value = metrics.get("p_value_gt_50pct", 1.0)
                    max_dd = metrics.get("max_drawdown_units", 0)

                    print(f"  ✅  Backtest complete ({n_bets} bets from {n_oos} OOS predictions)")
                    print(f"       Record: {wins}W - {losses}L (win rate: {win_rate:.1%})")
                    print(f"       Total return: {total_return:+.2f} units")
                    print(f"       Sharpe ratio: {sharpe:.2f}")
                    print(f"       Max drawdown: {max_dd:.1f} units")
                    print(f"       P-value (>50%): {p_value:.4f}")
                    print(f"       Significant: {'YES' if metrics.get('is_significant', False) else 'NO'}")

                    # Overfitting flag from metadata
                    diag = self.results.get("metadata", {}).get("overfitting_diag", {})
                    if diag.get("overfit", False):
                        print(f"       ⚠  OVERFITTING: Train R²={diag.get('avg_train_r2', 0):.2f}, "
                              f"Test R²={diag.get('avg_test_r2', 0):.2f}")

                    self.results["backtest"] = metrics
                    self.results["backtest_meta"] = {
                        "n_oos": int(n_oos),
                        "n_bets": n_bets,
                        "oos_pct": round(n_oos / max(total_with_preds, 1), 4) if '_is_oos' in df.columns else 1.0,
                    }
                    return metrics

            print("  ℹ  No backtest results")
        except Exception as e:
            print(f"  ⚠  Backtesting failed: {e}")
            import traceback
            traceback.print_exc()

        return None
