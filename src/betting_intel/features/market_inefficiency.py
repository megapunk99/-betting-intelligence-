"""
Market Inefficiency — compute market-implied probabilities and error targets.

The key insight: instead of training a model to predict "who will win" (home_win),
train it to predict "where is the market wrong?" (market_inefficiency).

Architecture
────────────
  1. Market Proxy: Build a market-implied win probability for each historical
     game using available data (ELO ratings as the proxy for market belief)
  2. Error Target: market_error = actual_outcome - market_implied_prob
     - Positive = market underestimated (team outperformed expectations)
     - Negative = market overestimated (team underperformed expectations)
  3. Edge Signal: Predict the error — learn patterns of market mispricing
  4. Inference: predicted_error + current_market_prob = our win probability

Data Flow
─────────
  FeatureEngineer produces:
    - elo_home_prob: ELO-based home win probability (runs chronologically)
    - market_line_baseline: Trailing-average total proxy
    - point_diff: Actual game outcome margin

  This module consumes those to produce:
    - market_implied_home_prob: Best estimate of market's belief
    - market_error: home_win - market_implied_home_prob (the inefficiency)
    - total_market_error: total_points - market_line_baseline
    - avg_market_error: rolling average of recent market errors
    - market_error_volatility: std of market errors over recent games
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET PROXY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def spread_to_implied_prob(spread: float, home: bool = True) -> float:
    """
    Convert a point spread to an implied win probability.

    In the NBA, each point of spread corresponds to roughly 3.5-4% win
    probability. The standard formula uses a logistic transformation:

        P(home_win) = 1 / (1 + 10^(-spread / spread_scale))

    where spread_scale ≈ 14 for the NBA (calibrated from ~25 years of data).

    TheOddsAPI convention: a negative spread means the home team is favored.
    E.g., spread = -5.5 means home is favored by 5.5 points → home win
    probability ~ 0.71. The formula handles this correctly — a negative
    spread produces a higher P(home_win).

    Args:
        spread: Point spread (negative = home favored, positive = away favored)
        home: If True, return home team's win probability. If False, away.

    Returns:
        Implied win probability (0-1)
    """
    if spread is None or np.isnan(spread):
        return 0.5

    spread_scale = 14.0  # Calibrated for NBA
    if not home:
        spread = -spread
    # Flip sign: negative spread (home favored) → higher home win prob
    prob = 1.0 / (1.0 + 10.0 ** (spread / spread_scale))
    return float(np.clip(prob, 0.01, 0.99))


def margin_to_implied_prob(
    expected_margin: float,
    home: bool = True,
) -> float:
    """
    Convert an expected margin to implied win probability.

    Same logic as spread_to_implied_prob but for model-predicted margins.
    Each ~0.06 of win probability ≈ 1 point of margin in NBA.

    Args:
        expected_margin: Expected point differential (positive = home favored)
        home: Return probability for home team

    Returns:
        Implied win probability (0-1)
    """
    return spread_to_implied_prob(-expected_margin if home else expected_margin)


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    return 0.5


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Normalize implied probabilities by removing the vig."""
    total = home_prob + away_prob
    if total > 0:
        return (home_prob / total, away_prob / total)
    return (home_prob, away_prob)


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET INEFFICIENCY COMPUTER
# ═══════════════════════════════════════════════════════════════════════════


class MarketInefficiencyComputer:
    """
    Compute market inefficiency targets from the feature engineering pipeline.

    This is NOT a model — it computes the TARGET variable that we train
    a model to predict. The model learns to predict where the market's
    estimate deviates from reality.

    Market Proxy Strategy (in priority order):
      Tier 1: Spread-based — use actual point_diff to derive implied odds
      Tier 2: ELO-based — use elo_home_prob as the market proxy
      Tier 3: Baseline — use league-average 0.5
    """

    def __init__(self, use_elo_proxy: bool = True, use_spread_proxy: bool = True):
        self.use_elo_proxy = use_elo_proxy
        self.use_spread_proxy = use_spread_proxy

    def compute_targets(
        self,
        df: pd.DataFrame,
        market_prob_overrides: Optional[dict[tuple[str, str, str], float]] = None,
    ) -> pd.DataFrame:
        """
        Add market inefficiency target columns to a features DataFrame.

        Source priority for market-implied probability:
          1. market_prob_overrides dict keyed by (home_team, away_team, game_date)
             — REAL market data from MarketOddsStore (highest quality)
          2. elo_home_prob — ELO-based proxy (used when no real data available)
          3. point_diff — fallback spread conversion (noisy, avoid)
          4. 0.5 — last resort

        Args:
            df: Features DataFrame from FeatureEngineer
            market_prob_overrides: Optional dict mapping
                (home_team_name, away_team_name, game_date_iso) → vig-free home prob
                Provided by the training pipeline from MarketOddsStore queries.

        Requires columns:
          - home_win: 0 or 1 (actual outcome)
          - elo_home_prob: ELO-based home win probability (pre-game)
          - market_line_baseline: trailing-average total proxy
          - total_points: actual total points scored

        Adds columns:
          - market_proxy_source: "real_odds", "elo_proxy", "spread_proxy", or "default"
          - market_implied_home_prob: Best estimate of market's pre-game belief
          - market_error: home_win - market_implied_home_prob (continuous, -1 to +1)
          - abs_market_error: |market_error| (magnitude of market's miss)
          - total_market_error: total_points - market_line_baseline
          - market_error_clipped: market_error clipped to [-0.5, 0.5] for stability
          - market_error_binary: 1 if model beats market, 0 otherwise
          - elo_error: home_win - elo_home_prob (raw ELO miss)
          - weighted_market_error: Blend of moneyline + total error signals
        """
        df = df.copy()

        # ── 1. Build market-implied home win probability ─────────────────
        # Priority: real odds > ELO proxy > spread proxy > default
        source_col = []

        # Determine team name columns
        home_name_col = (
            "TEAM_NAME_home" if "TEAM_NAME_home" in df.columns else "home_team"
        )
        away_name_col = (
            "TEAM_NAME_away" if "TEAM_NAME_away" in df.columns else "away_team"
        )
        date_col = "GAME_DATE" if "GAME_DATE" in df.columns else "game_date"

        market_implied = []

        for idx, row in df.iterrows():
            home_team = str(row.get(home_name_col, "")).strip()
            away_team = str(row.get(away_name_col, "")).strip()
            game_date = str(row.get(date_col, ""))[:10]

            real_prob = None

            # Tier 1: Real market data from MarketOddsStore
            if market_prob_overrides is not None:
                key = (home_team, away_team, game_date)
                reverse_key = (away_team, home_team, game_date)
                if key in market_prob_overrides:
                    real_prob = market_prob_overrides[key]
                elif reverse_key in market_prob_overrides:
                    real_prob = 1.0 - market_prob_overrides[reverse_key]

            if real_prob is not None:
                market_implied.append(float(np.clip(real_prob, 0.01, 0.99)))
                source_col.append("real_odds")
                continue

            # Tier 2: ELO-based probability
            if self.use_elo_proxy and "elo_home_prob" in df.columns:
                elo_val = row.get("elo_home_prob", 0.5)
                if pd.notna(elo_val):
                    market_implied.append(float(np.clip(elo_val, 0.01, 0.99)))
                    source_col.append("elo_proxy")
                    continue

            # Tier 3: Spread-based proxy (noisy)
            if self.use_spread_proxy and "point_diff" in df.columns:
                pd_val = row.get("point_diff", 0)
                if pd.notna(pd_val):
                    market_implied.append(
                        margin_to_implied_prob(float(pd_val), home=True)
                    )
                    source_col.append("spread_proxy")
                    continue

            # Tier 4: Default
            market_implied.append(0.5)
            source_col.append("default")

        df["market_proxy_source"] = source_col
        df["market_implied_home_prob"] = market_implied

        # ── 2. Compute market error (the inefficiency target) ────────────
        # market_error = 1.0 if home won and market said < 100%
        #                -1.0 if home lost and market said > 0%
        # This is the continuous target: predict how much the market was wrong
        df["market_error"] = (
            df["home_win"].astype(float) - df["market_implied_home_prob"]
        )
        df["abs_market_error"] = df["market_error"].abs()

        # Clipped version for stability in regression training
        df["market_error_clipped"] = df["market_error"].clip(-0.5, 0.5)

        # Binary: did our model (using ELO as market proxy) get it wrong?
        # 1 if home_win and market said < 50%, or home_loss and market said > 50%
        # This tells us if the market's favorite or underdog won
        df["market_error_binary"] = (
            (df["home_win"] == 1) & (df["market_implied_home_prob"] < 0.5)
        ) | ((df["home_win"] == 0) & (df["market_implied_home_prob"] > 0.5)).astype(int)

        # ── 3. ELO-specific error (how much did ELO itself miss?) ────────
        if "elo_home_prob" in df.columns:
            df["elo_error"] = df["home_win"].astype(float) - df["elo_home_prob"].clip(
                0.01, 0.99
            )
        else:
            df["elo_error"] = df["market_error"]

        # ── 4. Total market error (for totals prediction) ────────────────
        if "total_points" in df.columns and "market_line_baseline" in df.columns:
            df["total_market_error"] = df["total_points"] - df["market_line_baseline"]
        else:
            df["total_market_error"] = 0.0

        # ── 5. Weighted combined edge signal ────────────────────────────
        # Blend moneyline error (70%) and total error (30%)
        # Normalize total_market_error to ~0-1 scale (NBA games range -40 to +40)
        total_error_norm = df["total_market_error"].clip(-40, 40) / 40.0
        df["weighted_market_error"] = (
            0.70 * df["market_error_clipped"] + 0.30 * total_error_norm
        )

        return df

    def compute_trend_features(
        self,
        df: pd.DataFrame,
        team_id_col: str = "TEAM_ID_home",
    ) -> pd.DataFrame:
        """
        Add rolling market error trend features.

        These capture whether a team has been consistently beating or
        falling short of market expectations — a key inefficiency signal.

        Features:
          - market_error_ma_5: 5-game rolling average of market error
          - market_error_ma_10: 10-game rolling average
          - market_error_trend: Slope of market error over last 5 games
          - recent_edge_streak: Consecutive games with |market_error| > 0.05
        """
        df = df.copy()

        if "market_error" not in df.columns:
            return df

        # Rolling averages of market error (positive = beating expectations)
        for window in [5, 10]:
            df[f"market_error_ma_{window}g"] = df.groupby(team_id_col)[
                "market_error"
            ].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))

        # Market error trend (are they increasingly beating/failing expectations?)
        if "market_error" in df.columns:
            df["market_error_trend_home"] = df.groupby(team_id_col)[
                "market_error"
            ].transform(lambda x: self._compute_slope(x, window=5))

        # Recent edge streak: how many of last 5 games had |market_error| > 0.05
        # i.e., games where the market was meaningfully wrong
        if "abs_market_error" in df.columns:
            df["recent_edge_streak"] = df.groupby(team_id_col)[
                "abs_market_error"
            ].transform(
                lambda x: (
                    x.rolling(5, min_periods=1).apply(
                        lambda s: int((s > 0.05).sum()),
                        raw=True,
                    )
                ).shift(1)
            )

        return df

    @staticmethod
    def _compute_slope(values: pd.Series, window: int = 5) -> pd.Series:
        """Compute linear trend slope over rolling window."""
        n = len(values)
        result = np.full(n, 0.0)

        for i in range(n):
            if i < window:
                continue
            window_vals = values.iloc[max(0, i - window) : i].values
            if len(window_vals) < 2:
                continue
            x = np.arange(len(window_vals))
            y = window_vals
            x_mean = x.mean()
            y_mean = y.mean()
            num = np.sum((x - x_mean) * (y - y_mean))
            den = np.sum((x - x_mean) ** 2)
            result[i] = num / den if den > 0 else 0.0

        return pd.Series(result, index=values.index)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════


def compute_market_inefficiency_targets(
    df: pd.DataFrame,
    use_elo: bool = True,
    add_trends: bool = True,
    market_prob_overrides: Optional[dict[tuple[str, str, str], float]] = None,
) -> pd.DataFrame:
    """
    One-call convenience: add all market inefficiency targets to a DataFrame.

    Args:
        df: Features DataFrame from FeatureEngineer
        use_elo: Use ELO as market proxy (recommended)
        add_trends: Add rolling trend features
        market_prob_overrides: Real market data from MarketOddsStore
            Dict keyed by (home_team, away_team, game_date) → vig-free home prob

    Returns:
        DataFrame with market inefficiency target columns
    """
    computer = MarketInefficiencyComputer(use_elo_proxy=use_elo)
    result = computer.compute_targets(df, market_prob_overrides=market_prob_overrides)
    if add_trends:
        result = computer.compute_trend_features(result)
    return result


__all__ = [
    "MarketInefficiencyComputer",
    "compute_market_inefficiency_targets",
    "spread_to_implied_prob",
    "margin_to_implied_prob",
    "american_to_implied_prob",
    "remove_vig",
]
