"""
SoccerPredictor — ELO-based prediction model for soccer (EPL).

Uses a simplified ELO rating system with:
  - Home advantage (~0.38 expected goals, ~62% historic home win/draw%)
  - Poisson distribution for goals prediction
  - Implied probabilities for home/draw/away from ELO differential

The model predicts:
  1. Match outcome probabilities (home/draw/away) via ELO
  2. Total goals edge (over/under) via Poisson
  3. Compares predictions with market odds to find edges

ELO ratings are maintained in memory and decay during the offseason.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, date
from typing import Optional

from betting_intel.live.models import LiveGame, MIN_EDGE_THRESHOLD

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# ELO parameters (standard soccer values)
ELO_K = 32                # K-factor for ELO updates
ELO_HOME_ADVANTAGE = 75   # ~75 ELO points ≈ 0.38 expected goals advantage
ELO_INITIAL = 1500        # Starting ELO for all teams
ELO_DECAY_OFFSEASON = 50  # ELO decay during offseason (per month)

# Poisson expected goals conversion
# ELO diff → expected goals ratio:
#   exp(elo_diff / 400) gives odds ratio for outcome
#   Convert ELO diff to expected goals via logistic

# Average EPL stats
AVG_GOALS_HOME = 1.53     # Avg home goals per game in EPL
AVG_GOALS_AWAY = 1.19     # Avg away goals per game in EPL
AVG_TOTAL_GOALS = 2.72    # Avg total goals per game

# Home advantage in ELO points
# 75 ELO = 60.6% home win probability vs equal opponent
# Actual EPL home win rate is ~43%, draw ~25%, away ~32%
# The 3-way market means home advantage is spread across win/draw


class EPLSoccerPredictor:
    """ELO-based soccer prediction for EPL matches.

    Maintains ELO ratings for all 20 EPL teams across sessions.
    Ratings are stored in memory and initialized with reasonable defaults
    based on recent historical performance.
    """

    # Class-level ELO ratings (shared across engine instances)
    _elo_ratings: dict[str, int] = {
        # Top tier — title contenders
        "Man City": 1980,
        "Arsenal": 1920,
        "Liverpool": 1890,
        # Champions League contenders
        "Chelsea": 1850,
        "Man United": 1840,
        "Tottenham": 1820,
        "Newcastle": 1810,
        "Aston Villa": 1800,
        # Mid-table
        "Brighton": 1760,
        "West Ham": 1750,
        "Brentford": 1720,
        "Fulham": 1710,
        "Crystal Palace": 1705,
        "Everton": 1690,
        "Bournemouth": 1680,
        "Nott'm Forest": 1670,
        "Wolves": 1660,
        # Relegation candidates
        "Leicester": 1640,
        "Southampton": 1620,
        "Ipswich": 1600,
    }

    _last_update_date: Optional[str] = None

    @classmethod
    def get_elo(cls, team: str) -> int:
        """Get ELO rating for a team, with default for unknown teams."""
        return cls._elo_ratings.get(team, ELO_INITIAL)

    @classmethod
    def expected_goals(cls, team_elo: int, opponent_elo: int, home: bool) -> float:
        """Compute expected goals for a team based on ELO differential.

        Uses the formula:
            expected_goals = avg_goals * exp((elo_diff + home_adv) / 400)

        Where:
            - elo_diff = team_elo - opponent_elo
            - home_adv = 75 if home, 0 if away
            - avg_goals = AVG_GOALS_HOME if home, AVG_GOALS_AWAY if away
        """
        elo_diff = team_elo - opponent_elo
        if home:
            elo_diff += ELO_HOME_ADVANTAGE

        avg_goals = AVG_GOALS_HOME if home else AVG_GOALS_AWAY
        return avg_goals * math.exp(elo_diff / 400.0)

    @classmethod
    def poisson_prob(cls, k: int, lam: float) -> float:
        """Poisson probability of exactly k goals given expected rate lam."""
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    @classmethod
    def match_outcome_probs(
        cls,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float, float]:
        """Compute home/draw/away probabilities using Poisson distribution.

        Returns:
            (home_win_prob, draw_prob, away_win_prob)
        """
        home_elo = cls.get_elo(home_team)
        away_elo = cls.get_elo(away_team)

        # Expected goals for each team
        home_xg = cls.expected_goals(home_elo, away_elo, home=True)
        away_xg = cls.expected_goals(away_elo, home_elo, home=False)

        # Compute match outcome probabilities using Poisson
        # Sum over all scorelines up to 10 goals
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for i in range(11):  # home goals
            p_home = cls.poisson_prob(i, home_xg)
            for j in range(11):  # away goals
                p_away = cls.poisson_prob(j, away_xg)
                joint = p_home * p_away
                if i > j:
                    home_win += joint
                elif i == j:
                    draw += joint
                else:
                    away_win += joint

        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total

        return (home_win, draw, away_win)

    @classmethod
    def expected_total_goals(cls, home_team: str, away_team: str) -> float:
        """Compute expected total goals for a match."""
        home_elo = cls.get_elo(home_team)
        away_elo = cls.get_elo(away_team)
        home_xg = cls.expected_goals(home_elo, away_elo, home=True)
        away_xg = cls.expected_goals(away_elo, home_elo, home=False)
        return home_xg + away_xg

    @classmethod
    def over_under_probs(
        cls, home_team: str, away_team: str, market_total: float
    ) -> tuple[float, float]:
        """Compute over/under probabilities for a given market total line.

        Uses Poisson distribution for total goals.

        Returns:
            (over_prob, under_prob)
        """
        expected_total = cls.expected_total_goals(home_team, away_team)

        # Poisson probability of total goals
        # P(over) = 1 - P(<= market_total)
        under_prob = 0.0
        max_goals = int(market_total + 4)  # compute up to reasonable max
        for k in range(max_goals + 1):
            under_prob += cls.poisson_prob(k, expected_total)

        # Adjust for fractional lines (e.g., 2.5)
        if market_total != int(market_total):
            # For fractional lines, under means <= floor(market_total)
            floor_total = int(market_total)
            under_prob = 0.0
            for k in range(floor_total + 1):
                under_prob += cls.poisson_prob(k, expected_total)
        else:
            # For integer lines (e.g., 3.0), push is possible
            # P(under) = P(<= market_total - 1) + 0.5 * P(market_total)
            push_prob = cls.poisson_prob(int(market_total), expected_total)
            under_prob = under_prob - push_prob + 0.5 * push_prob

        over_prob = 1.0 - under_prob
        return (over_prob, under_prob)

    # ── Public Prediction API ───────────────────────────────────────────

    @classmethod
    def predict_moneyline(cls, game: LiveGame) -> LiveGame:
        """Predict moneyline edge for a soccer game.

        Compares our ELO-based home/draw/away probabilities with
        the market-implied probabilities to find edges.
        """
        if game.sport_group not in ("Soccer",):
            return game

        home_team = game.home_team_short
        away_team = game.away_team_short

        # Get our model probabilities
        home_prob, draw_prob, away_prob = cls.match_outcome_probs(home_team, away_team)

        # Get market-implied probabilities (remove vig)
        market_home_prob = None
        market_draw_prob = None
        market_away_prob = None

        if game.home_ml is not None and game.away_ml is not None and game.draw_ml is not None:
            from betting_intel.features.market_inefficiency import (
                american_to_implied_prob,
            )

            market_home_prob = american_to_implied_prob(game.home_ml)
            market_draw_prob = american_to_implied_prob(game.draw_ml) if game.draw_ml else 0.0
            market_away_prob = american_to_implied_prob(game.away_ml)

            # For soccer, we need 3-way vig removal
            total = market_home_prob + market_draw_prob + market_away_prob
            if total > 0:
                market_home_prob /= total
                market_draw_prob /= total
                market_away_prob /= total

            # Find the best edge among home/draw/away
            home_edge = home_prob - market_home_prob if market_home_prob else 0.0
            draw_edge = draw_prob - market_draw_prob if market_draw_prob else 0.0
            away_edge = away_prob - market_away_prob if market_away_prob else 0.0

            # Pick the best edge
            edges = [
                (home_edge, "home", home_prob),
                (draw_edge, "neutral", None),
                (away_edge, "away", away_prob),
            ]

            best_edge, best_direction, best_prob = max(edges, key=lambda e: abs(e[0]))

            if abs(best_edge) >= MIN_EDGE_THRESHOLD:
                game.edge_pct = best_edge
                game.direction = best_direction
                if abs(best_edge) >= 0.08:
                    game.confidence = "high"
                elif abs(best_edge) >= 0.05:
                    game.confidence = "medium"
                else:
                    game.confidence = "low"

                # Simple stake based on edge magnitude
                # (Full Kelly staker is owned by the engine, not available here)
                game.stake_dollars = round(abs(best_edge) * 500, 2)
            else:
                game.edge_pct = 0.0
                game.direction = "neutral"
                game.confidence = "low"
                game.stake_dollars = 0.0

            game.predicted_total = round(home_prob, 3)

        else:
            # No market odds — just set model probabilities
            game.edge_pct = 0.0
            game.direction = "neutral"
            game.confidence = "low"
            game.stake_dollars = 0.0
            game.predicted_total = round(home_prob, 3)

        game.predicted_at = datetime.now().isoformat()
        return game

    @classmethod
    def predict_totals(cls, game: LiveGame) -> LiveGame:
        """Predict over/under edge for a soccer game.

        Compares our Poisson-based expected total goals with
        the market total line.
        """
        if game.sport_group not in ("Soccer",) or not game.market_total:
            return game

        home_team = game.home_team_short
        away_team = game.away_team_short

        # Model prediction
        expected_total = cls.expected_total_goals(home_team, away_team)
        over_prob, under_prob = cls.over_under_probs(home_team, away_team, game.market_total)

        # Market-implied over/under probability
        if game.over_odds is not None and game.under_odds is not None:
            from betting_intel.features.market_inefficiency import american_to_implied_prob
            market_over_prob = american_to_implied_prob(game.over_odds)
            market_under_prob = american_to_implied_prob(game.under_odds)

            # Remove vig
            total = market_over_prob + market_under_prob
            if total > 0:
                market_over_prob /= total
                market_under_prob /= total

            over_edge = over_prob - market_over_prob
            under_edge = under_prob - market_under_prob

            if abs(over_edge) > abs(under_edge) and abs(over_edge) >= 0.02:
                game.total_prediction = round(expected_total, 2)
                game.total_edge_pct = over_edge
                game.total_direction = "over"
                game.total_confidence = "high" if abs(over_edge) >= 0.08 else ("medium" if abs(over_edge) >= 0.05 else "low")
            elif abs(under_edge) >= 0.02:
                game.total_prediction = round(expected_total, 2)
                game.total_edge_pct = under_edge
                game.total_direction = "under"
                game.total_confidence = "high" if abs(under_edge) >= 0.08 else ("medium" if abs(under_edge) >= 0.05 else "low")

        return game
