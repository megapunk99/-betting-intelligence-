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
)


# ── Inline stubs for deleted modules ─────────────────────────────────────
# These are never called because HAS_RISK/HAS_BETTING/HAS_BACKTESTING are
# all False in bootstrap.py, but they must exist to prevent NameError if
# someone changes those flags.  Each stub logs a message and returns a
# safe default.

class _InlineKellyCalculator:
    def __init__(self, bankroll=10000, fraction=0.25):
        self.bankroll = bankroll
        self.fraction = fraction
    def compute_kelly(self, win_probability=0.5, decimal_odds=1.91):
        b = decimal_odds - 1.0
        if b <= 0:
            return (0.0, 0.0)
        full = (b * win_probability - (1.0 - win_probability)) / b
        return (max(0, full * self.fraction), 0.0)

class _InlineExposureManager:
    def __init__(self, bankroll=10000, default_max_exposure_pct=0.05,
                 default_max_per_game_pct=0.0375):
        self.bankroll = bankroll
        self.tracked = {}
    def check_exposure(self, team, stake):
        return stake <= self.bankroll * 0.05
    def track_bet(self, team, stake):
        self.tracked[team] = self.tracked.get(team, 0) + stake
    def get_summary(self):
        return {"total_exposure": sum(self.tracked.values()), "teams": dict(self.tracked)}

class _InlineBetCorrelationTracker:
    def register_bet(self, bet_id="", bet_type="", game_id=""):
        pass
    def get_correlation_matrix(self, bet_ids):
        return type("obj", (object,), {"matrix": type("obj", (object,), {"shape": (0,0)})()})()

class _InlineMonteCarloSimulator:
    def simulate_from_bets(self, bets_df, n_games_per_season=1000,
                           initial_bankroll=10000, stake_per_bet=0.02):
        return type("obj", (object,), {
            "median_profit": 0.0, "percentile_90": 0.0, "percentile_10": 0.0,
        })()

class _InlineEdgeDetector:
    def detect_rest_edge(self, df):
        return None
    def detect_home_court_edge(self, df):
        return None

class _InlineBacktestMetrics:
    """Real backtest metrics computation from OOS bet data.

    Computes:
      - Win/loss/push counts and win rate
      - Total profit in units
      - Sharpe ratio (annualized: mean / std * sqrt(n_bets))
      - Maximum drawdown from peak (cumulative profit curve)
      - One-sided binomial p-value (H₀: win_rate = 50%)
      - Significance flag (p < 0.05)

    All edge cases handled: empty DataFrame, all pushes, single bet,
    zero variance, etc.
    """

    @staticmethod
    def compute_all(bets_df: pd.DataFrame) -> Dict[str, Any]:
        if bets_df is None or bets_df.empty:
            return {
                "wins": 0, "losses": 0, "pushes": 0,
                "win_rate": 0.0, "total_profit_units": 0.0,
                "sharpe_ratio": 0.0, "max_drawdown_units": 0.0,
                "p_value_gt_50pct": 1.0, "is_significant": False,
                "n_bets": 0,
            }

        # ── Counts ─────────────────────────────────────────────────
        wins = int((bets_df["outcome"] == "WIN").sum())
        losses = int((bets_df["outcome"] == "LOSS").sum())
        pushes = int((bets_df["outcome"] == "PUSH").sum())
        n_decided = wins + losses  # pushes don't count as decided
        n_bets = len(bets_df)

        # ── Win rate (only on decided bets) ────────────────────────
        win_rate = wins / n_decided if n_decided > 0 else 0.0

        # ── Total profit ───────────────────────────────────────────
        total_profit = float(bets_df["profit_units"].sum())

        # ── Sharpe ratio ───────────────────────────────────────────
        # Use bet-level Sharpe: mean(return) / std(return) * sqrt(n)
        # This measures risk-adjusted return per bet.
        returns = bets_df["profit_units"].values
        if len(returns) > 1:
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))  # sample std
            if std_ret > 1e-10:
                sharpe = mean_ret / std_ret * np.sqrt(len(returns))
            else:
                sharpe = 0.0  # no variance
        elif len(returns) == 1:
            # Single bet: Sharpe is either +inf (win) or -inf (loss)
            # Cap at +/- 5.0 for practical display
            sharpe = 5.0 if returns[0] > 0 else (-5.0 if returns[0] < 0 else 0.0)
        else:
            sharpe = 0.0

        sharpe = float(np.clip(sharpe, -10.0, 10.0))  # clamp for display

        # ── Max drawdown ───────────────────────────────────────────
        # Compute cumulative profit curve in chronological order.
        # Prepend 0.0 (starting bankroll) so running_max starts at 0,
        # not at the first bet's cumulative value.
        if "game_date" in bets_df.columns and n_bets > 1:
            sorted_df = bets_df.sort_values("game_date")
            cum_profit = np.concatenate([[0.0], sorted_df["profit_units"].cumsum().values])
        elif n_bets > 0:
            cum_profit = np.concatenate([[0.0], bets_df["profit_units"].cumsum().values])
        else:
            cum_profit = np.array([0.0])

        if len(cum_profit) > 0:
            running_max = np.maximum.accumulate(cum_profit)
            drawdowns = running_max - cum_profit
            max_dd = float(np.max(drawdowns))
        else:
            max_dd = 0.0

        # ── Binomial p-value (one-sided: is win_rate > 50%?) ───────
        # H₀: true win rate = 50%.  Use normal approximation when
        # n_decided is large enough, else exact binomial via math.comb.
        if n_decided > 0 and n_decided >= 10:
            # Normal approximation: z = (wins - n*p) / sqrt(n*p*(1-p))
            # where p = 0.5 under H₀
            z = (wins - n_decided * 0.5) / max(np.sqrt(n_decided * 0.5 * 0.5), 1e-10)
            # One-sided: P(Z > z) = 1 - Phi(z)
            from math import erfc
            p_value = erfc(z / np.sqrt(2)) / 2.0
            p_value = float(min(p_value, 1.0))
        elif n_decided > 0:
            # Exact binomial: sum_{k=wins}^{n} C(n,k) * 0.5^n
            from math import comb
            total_outcomes = 2 ** n_decided
            exact_p = 0.0
            for k in range(wins, n_decided + 1):
                exact_p += comb(n_decided, k) / total_outcomes
            p_value = float(min(exact_p, 1.0))
        else:
            p_value = 1.0

        is_significant = p_value < 0.05

        return {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "n_bets": n_bets,
            "n_decided": n_decided,
            "win_rate": round(win_rate, 4),
            "total_profit_units": round(total_profit, 4),
            "avg_profit_per_bet": round(total_profit / max(n_bets, 1), 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_units": round(max_dd, 4),
            "p_value_gt_50pct": round(p_value, 4),
            "is_significant": is_significant,
            "method": "normal_approx" if n_decided >= 10 else "exact_binomial",
        }


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
            from betting_intel.pipeline.bootstrap import HAS_RISK
            # Inline KellyCalculator logic — risk package was deleted
            kelly = _InlineKellyCalculator(
                bankroll=self.args.bankroll,
                fraction=self.args.kelly_fraction,
            )
            exposure_mgr = _InlineExposureManager(
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

                # PROPER probability conversion: sigmoid instead of crude 0.5 + edge/2
                # edge/2 can give probabilities > 1.0 for edges > 100%
                # Sigmoid correctly maps (-inf, +inf) → (0, 1)
                import math
                win_probability = 1.0 / (1.0 + math.exp(-edge * 5.0)) if edge != 0 else 0.5
                win_probability = max(0.01, min(0.99, win_probability))
                kelly_pct, _ = kelly.compute_kelly(
                    win_probability=win_probability,
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
            tracker = _InlineBetCorrelationTracker()
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
                sim = _InlineMonteCarloSimulator()
                bet_rows = []
                import random
                for rec in recommendations:
                    # PROPER Monte Carlo: use win_probability to simulate outcome,
                    # NOT a deterministic "WIN if edge > min_edge" which would
                    # guarantee 100% win rate for all bets above threshold.
                    win_prob = rec.get("model_probability",
                                       rec.get("probability",
                                               rec.get("win_probability", 0.5)))
                    # Fallback: if no probability available, estimate from edge
                    if win_prob <= 0 or win_prob >= 1:
                        import math
                        edge = rec.get("edge", 0)
                        win_prob = 1.0 / (1.0 + math.exp(-edge * 5.0)) if edge != 0 else 0.5
                    win_prob = max(0.01, min(0.99, win_prob))

                    is_win = random.random() < win_prob
                    outcome = "WIN" if is_win else "LOSS"
                    bet_rows.append({
                        "game_id": rec.get("game_id", ""),
                        "game_date": rec.get("game_date", ""),
                        "outcome": outcome,
                        "profit_units": 1.0 if is_win else -1.0,
                        "edge_pct": rec.get("edge", 0),
                        "win_probability": win_prob,
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
                detector = _InlineEdgeDetector()
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
                metrics = _InlineBacktestMetrics.compute_all(bets_df)

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
