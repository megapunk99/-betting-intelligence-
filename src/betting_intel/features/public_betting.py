"""
Public Betting Percentages — track and model public betting patterns.

INSIGHT: The public bets favorites, overs, and popular teams.
Fading the public (betting against the majority) has been a profitable
long-term strategy if you can identify WHEN the public is wrong.

DATA SOURCE REQUIREMENTS:
  - Action Network API (paid, most reliable)
  - SportsBettingDime (free, ~24h delayed)
  - ESPN / CBS Sports tracking (free, limited)
  - Twitter scraping of public polls (free, noisy)

This module provides:
  1. PublicBettingData model — structure for storing betting % data
  2. PublicBettingFetcher — abstract base with methods to fetch from sources
  3. FadeThePublicAnalyzer — computes whether to fade the public
  4. Stub implementation that returns neutral values for now

USAGE:
    from betting_intel.features.public_betting import get_public_betting_features
    features = get_public_betting_features(home_team, away_team)
    # Returns zero-vector (stub) until data sources are connected
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data Model ───────────────────────────────────────────────────────────

class PublicBettingData:
    """Public betting percentages for a single game.
    
    All values are percentages (0-100).
    """
    def __init__(
        self,
        home_ml_pct: float = 50.0,       # % of bets on home moneyline
        away_ml_pct: float = 50.0,       # % of bets on away moneyline
        spread_home_pct: float = 50.0,   # % of spread bets on home
        spread_away_pct: float = 50.0,   # % of spread bets on away
        over_pct: float = 50.0,          # % of total bets on over
        under_pct: float = 50.0,         # % of total bets on under
        n_bets_total: int = 0,           # Total bets tracked
        source: str = "stub",            # Data source name
    ):
        self.home_ml_pct = home_ml_pct
        self.away_ml_pct = away_ml_pct
        self.spread_home_pct = spread_home_pct
        self.spread_away_pct = spread_away_pct
        self.over_pct = over_pct
        self.under_pct = under_pct
        self.n_bets_total = n_bets_total
        self.source = source


# ── Public Betting Fetcher ───────────────────────────────────────────────

class PublicBettingFetcher:
    """Fetch public betting percentages from external sources.
    
    Currently a stub — returns neutral values until a data source
    is connected.
    
    TO INTEGRATE A DATA SOURCE:
      1. Create a subclass that overrides fetch()
      2. Use the source's API to get betting percentages
      3. Return a PublicBettingData instance
    """
    
    def __init__(self):
        self._cache: dict[str, PublicBettingData] = {}
    
    def fetch(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
    ) -> PublicBettingData:
        """Fetch public betting data for a game.
        
        Currently a stub — returns neutral 50/50 data.
        Override this method to integrate a real data source.
        
        Args:
            game_id: Unique game identifier.
            home_team: Home team name.
            away_team: Away team name.
            
        Returns:
            PublicBettingData with betting percentages.
        """
        # Stub: return neutral data
        return self._get_neutral(game_id)
    
    def _get_neutral(self, game_id: str) -> PublicBettingData:
        """Return neutral (50/50) betting data as default."""
        if game_id not in self._cache:
            self._cache[game_id] = PublicBettingData(source="stub")
        return self._cache[game_id]
    
    def clear_cache(self) -> None:
        self._cache.clear()


# ── Fade the Public Analyzer ─────────────────────────────────────────────

class FadeThePublicAnalyzer:
    """Analyze whether to fade the public based on betting percentages.
    
    Core logic:
      1. If 70%+ of bets are on one side, that side is the "public side"
      2. If the line is moving TOWARD the public side, follow the public
         (this is a sucker move, avoid it)
      3. If the line is moving AGAINST the public side (reverse line
         movement), the sharps are fading the public — FOLLOW THE SHARPS
      4. The strongest signal is: public heavy on side A, line moving to side B
    """
    
    @staticmethod
    def compute_features(
        public_data: PublicBettingData,
        home_ml: Optional[float] = None,
        away_ml: Optional[float] = None,
    ) -> dict[str, float]:
        """Compute public betting features for ML model.
        
        Args:
            public_data: PublicBettingData for the game.
            home_ml: Home moneyline (for RLM detection).
            away_ml: Away moneyline.
            
        Returns:
            Feature dict with:
              - public_home_ml_pct: % of bets on home ML
              - public_away_ml_pct: % of bets on away ML
              - public_over_pct: % of bets on over
              - public_under_pct: % of bets on under
              - public_fade_signal: 1.0 = fade public (sharp on other side),
                -1.0 = follow public, 0.0 = neutral
              - public_extreme_flag: 1.0 if public > 70% on one side
        """
        # Compute fade signal based on deviation from 50/50
        home_deviation = public_data.home_ml_pct - 50.0
        fade_signal = home_deviation / 50.0  # -1 to +1
        
        # Extreme flag: 70%+ on one side
        extreme = 1.0 if abs(home_deviation) >= 20.0 else 0.0
        
        return {
            "public_home_ml_pct": public_data.home_ml_pct,
            "public_away_ml_pct": public_data.away_ml_pct,
            "public_over_pct": public_data.over_pct,
            "public_under_pct": public_data.under_pct,
            "public_fade_signal": round(fade_signal, 3),
            "public_extreme_flag": extreme,
        }


# ── Convenience ──────────────────────────────────────────────────────────

_public_fetcher = PublicBettingFetcher()
_public_analyzer = FadeThePublicAnalyzer()


def get_public_betting_features(
    game_id: str,
    home_team: str,
    away_team: str,
    home_ml: Optional[float] = None,
    away_ml: Optional[float] = None,
) -> dict[str, float]:
    """One-shot: get public betting features for a game."""
    data = _public_fetcher.fetch(game_id, home_team, away_team)
    return _public_analyzer.compute_features(data, home_ml, away_ml)


__all__ = [
    "PublicBettingData", "PublicBettingFetcher",
    "FadeThePublicAnalyzer", "get_public_betting_features",
]
