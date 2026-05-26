"""
Edge detection: identifies market inefficiencies and quantifies betting edges.
Focuses on detecting systematic biases in market pricing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EdgeSignal:
    """A detected edge opportunity."""

    strategy: str
    edge_type: str
    description: str
    avg_edge_pct: float
    sample_size: int
    win_rate: float
    expected_value: float
    confidence: float  # 0-1, how reliable we think this edge is
    conditions: Dict = field(default_factory=dict)
    is_actionable: bool = False


class EdgeDetector:
    """
    Identifies betting edges by analyzing prediction residuals and
    detecting systematic biases in the data.

    Core philosophy: An edge exists when our model consistently
    predicts outcomes that differ from the market expectation.
    """

    def __init__(self):
        self.signals: List[EdgeSignal] = []

    def detect_rest_edge(self, df: pd.DataFrame) -> Optional[EdgeSignal]:
        """
        Detect if rest advantage creates betting edge.
        Hypothesis: Teams with more rest outperform market expectations.
        """
        if "rest_advantage" not in df.columns or "point_diff" not in df.columns:
            return None

        # Significant rest advantage (3+ days more rest than opponent)
        significant_rest = df[df["rest_advantage"] >= 2].copy()

        if len(significant_rest) < 20:
            return None

        win_rate = (significant_rest["point_diff"] > 0).mean()
        avg_margin = significant_rest["point_diff"].mean()
        sample = len(significant_rest)

        # Normal rest advantage
        normal_rest = df[df["rest_advantage"].between(-1, 1)]
        normal_wr = (normal_rest["point_diff"] > 0).mean() if len(normal_rest) > 0 else 0.5

        edge = win_rate - normal_wr
        ev = (win_rate * 1.0) - ((1 - win_rate) * 1.0)  # Simplified at -110 odds

        if edge > 0.02 and sample >= 30:
            return EdgeSignal(
                strategy="rest_edge",
                edge_type="FATIGUE",
                description=f"Teams with {2}+ rest advantage over opponent",
                avg_edge_pct=edge,
                sample_size=sample,
                win_rate=win_rate,
                expected_value=ev,
                confidence=min(sample / 200, 0.9),
                conditions={
                    "min_rest_advantage": 2,
                    "avg_margin": round(avg_margin, 1),
                    "baseline_win_rate": round(normal_wr, 3),
                },
                is_actionable=edge > 0.03 and sample >= 50,
            )
        return None

    def detect_momentum_edge(self, df: pd.DataFrame) -> Optional[EdgeSignal]:
        """
        Detect momentum reversion opportunities.
        Hypothesis: Teams on extreme streaks are overvalued.
        """
        if "win_streak_home" not in df.columns or "point_diff" not in df.columns:
            return None

        # Hot teams (3+ wins)
        hot_mask = df["win_streak_home"] >= 3
        cold_mask = df["win_streak_home"] <= -3

        hot_teams = df[hot_mask].copy()
        cold_teams = df[cold_mask].copy()

        results = []

        if len(hot_teams) >= 20:
            next_game_wr = (hot_teams["point_diff"] > 0).mean()
            # Should be less than streak would suggest (regression)
            results.append(("hot_streak", hot_teams, next_game_wr))

        if len(cold_teams) >= 20:
            next_game_wr = (cold_teams["point_diff"] > 0).mean()
            # Should be higher than streak would suggest (regression)
            results.append(("cold_streak", cold_teams, next_game_wr))

        if not results:
            return None

        best_signal = max(results, key=lambda x: abs(x[2] - 0.5))
        signal_type, data, wr = best_signal

        if signal_type == "hot_streak":
            edge = 0.5 - wr  # Fade the streak
            desc = f"Teams on 3+ win streaks regress: win rate {wr:.1%}"
        else:
            edge = wr - 0.5  # Back the bounce-back
            desc = f"Teams on 3+ loss streaks bounce back: win rate {wr:.1%}"

        ev = (wr * 1.0) - ((1 - wr) * 1.0)

        if abs(edge) > 0.02 and len(data) >= 30:
            return EdgeSignal(
                strategy="momentum",
                edge_type="REVERSION",
                description=desc,
                avg_edge_pct=edge,
                sample_size=len(data),
                win_rate=wr,
                expected_value=ev,
                confidence=min(len(data) / 300, 0.8),
                conditions={
                    "signal_type": signal_type,
                    "min_streak": 3,
                },
                is_actionable=abs(edge) > 0.03 and len(data) >= 50,
            )
        return None

    def detect_pace_edge(self, df: pd.DataFrame) -> Optional[EdgeSignal]:
        """
        Detect pace-based edges for totals betting.
        Hypothesis: Pace changes (roster changes, coaching) are
        not immediately priced into totals.
        """
        # Try both old and new pace column names
        pace_col = "predicted_pace" if "predicted_pace" in df.columns else (
            "market_line_pace_adj" if "market_line_pace_adj" in df.columns else None
        )
        baseline_col = "market_line_baseline" if "market_line_baseline" in df.columns else (
            "predicted_total_base" if "predicted_total_base" in df.columns else None
        )

        if pace_col is None or "total_points" not in df.columns:
            return None

        # High pace games
        high_pace = df[df[pace_col] > df[pace_col].quantile(0.75)].copy()
        low_pace = df[df[pace_col] < df[pace_col].quantile(0.25)].copy()

        if len(high_pace) < 20 or len(low_pace) < 20:
            return None

        # If actual totals exceed predictions more in high-pace games
        market_line = baseline_col or "total_points"
        high_pace["total_pred"] = high_pace.get(market_line, high_pace["total_points"].mean())
        low_pace["total_pred"] = low_pace.get(market_line, low_pace["total_points"].mean())

        high_residual = (high_pace["total_points"] - high_pace["total_pred"]).mean()
        low_residual = (low_pace["total_points"] - low_pace["total_pred"]).mean()

        # Over proportion in high pace
        over_rate_high = (high_pace["total_points"] > high_pace["total_pred"]).mean()
        over_rate_low = (low_pace["total_points"] > low_pace["total_pred"]).mean()

        edge = over_rate_high - over_rate_low

        if abs(edge) > 0.03:
            return EdgeSignal(
                strategy="pace_total",
                edge_type="OVER_UNDER",
                description=f"High-pace games go over at {over_rate_high:.1%} vs {over_rate_low:.1%} for low-pace",
                avg_edge_pct=edge,
                sample_size=len(high_pace) + len(low_pace),
                win_rate=over_rate_high,
                expected_value=(over_rate_high * 1.0) - ((1 - over_rate_high) * 1.0),
                confidence=min(len(high_pace) / 200, 0.8),
                conditions={
                    "high_pace_n": len(high_pace),
                    "low_pace_n": len(low_pace),
                    "high_residual": round(high_residual, 1),
                    "low_residual": round(low_residual, 1),
                },
                is_actionable=abs(edge) > 0.05 and len(high_pace) >= 40,
            )
        return None

    def detect_home_court_edge(self, df: pd.DataFrame) -> Optional[EdgeSignal]:
        """
        Detect if home court advantage is mispriced.
        Hypothesis: Home court advantage varies significantly by team
        and is not uniformly priced.
        """
        if "TEAM_NAME_home" not in df.columns or "point_diff" not in df.columns:
            return None

        team_home_margins = df.groupby("TEAM_NAME_home")["point_diff"].agg(["mean", "count", "std"]).reset_index()
        team_home_margins.columns = ["team", "avg_home_margin", "games", "std"]

        league_avg_home = team_home_margins["avg_home_margin"].mean()

        # Teams with significantly different home court advantage
        team_home_margins["edge"] = team_home_margins["avg_home_margin"] - league_avg_home
        team_home_margins["edge_z"] = team_home_margins["edge"] / (team_home_margins["std"] / np.sqrt(team_home_margins["games"]))

        significant = team_home_margins[team_home_margins["games"] >= 10].copy()
        if len(significant) == 0:
            return None

        best = significant.loc[significant["edge"].idxmax()]
        worst = significant.loc[significant["edge"].idxmin()]

        edge_range = best["edge"] - worst["edge"]

        if edge_range > 3:  # 3+ points difference
            return EdgeSignal(
                strategy="spread_model",
                edge_type="HOME_COURT_MISPRICING",
                description=f"Home court advantage varies by {edge_range:.1f} points across teams",
                avg_edge_pct=edge_range / 10,
                sample_size=len(significant),
                win_rate=0.5 + best["edge"] / 20,
                expected_value=best["edge"] / 20,
                confidence=0.6,
                conditions={
                    "best_home_team": best["team"],
                    "best_margin": round(best["avg_home_margin"], 1),
                    "worst_home_team": worst["team"],
                    "worst_margin": round(worst["avg_home_margin"], 1),
                    "league_avg": round(league_avg_home, 1),
                },
                is_actionable=edge_range > 5,
            )
        return None

    def detect_all(self, df: pd.DataFrame) -> List[EdgeSignal]:
        """Run all edge detection heuristics."""
        self.signals = []

        detectors = [
            self.detect_rest_edge,
            self.detect_momentum_edge,
            self.detect_pace_edge,
            self.detect_home_court_edge,
        ]

        for detector in detectors:
            try:
                signal = detector(df)
                if signal:
                    self.signals.append(signal)
            except Exception as e:
                continue

        return sorted(self.signals, key=lambda s: s.confidence, reverse=True)

    def report(self) -> str:
        """Generate a readable report of detected edges."""
        if not self.signals:
            return "No significant edges detected in the data."

        lines = [
            "═" * 60,
            "EDGE DETECTION REPORT",
            "═" * 60,
        ]

        for i, signal in enumerate(self.signals, 1):
            action = "[ACTIONABLE]" if signal.is_actionable else "[MONITOR]"
            lines.extend([
                f"\n{i}. {signal.description}",
                f"   Type:     {signal.edge_type} | {action}",
                f"   Edge:     {signal.avg_edge_pct:.2%}",
                f"   Win Rate: {signal.win_rate:.1%}",
                f"   EV:       {signal.expected_value:.3f} units",
                f"   Sample:   {signal.sample_size} games",
                f"   Confidence: {signal.confidence:.0%}",
            ])
            if signal.conditions:
                for k, v in signal.conditions.items():
                    lines.append(f"   {k}: {v}")

        lines.append("\n" + "═" * 60)
        return "\n".join(lines)
