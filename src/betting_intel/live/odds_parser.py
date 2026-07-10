"""
OddsParser — parses raw TheOddsAPI / scraper events into LiveGame objects.

Handles:
  - Team name mapping (TheOddsAPI full names -> short display names)
  - Market line extraction (moneyline, spreads, totals)
  - Consensus computation (median across sportsbooks)
  - Age filtering (drops games that have already started — only upcoming)
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

from betting_intel.live.models import LiveGame

logger = logging.getLogger(__name__)


class OddsParser:
    """Parses raw odds dicts into LiveGame domain objects."""

    @staticmethod
    def median_or_none(values: list) -> Optional[float]:
        """Median of values, or None if empty."""
        if not values:
            return None
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return sorted_vals[n // 2]

    @staticmethod
    def std_or_none(values: list) -> Optional[float]:
        """Std dev of values, or None if fewer than 2 values."""
        if not values or len(values) < 2:
            return None
        try:
            return statistics.stdev(values)
        except (statistics.StatisticsError, ValueError, TypeError):
            return None

    @staticmethod
    def _resolve_short_name(full_name: str, sport_key: str) -> str:
        """Resolve a full team name to a short display name using SportConfig.

        Uses the sport's team_name_map from SportConfig first.
        Falls back to the global ODDS_TO_SHORT_NAME dict (NBA teams).
        Finally, uses the last word of the team name as a fallback.
        """
        from betting_intel.live.sport_configs import SPORT_KEY_TO_CONFIG
        from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME

        if not full_name:
            return ""

        # 1. Try SportConfig's team_name_map (handles NCAAB, future leagues)
        config = SPORT_KEY_TO_CONFIG.get(sport_key)
        if config and config.team_name_map:
            short = config.team_name_map.get(full_name)
            if short:
                return short

        # 2. Try the global ODDS_TO_SHORT_NAME dict (NBA teams)
        short = ODDS_TO_SHORT_NAME.get(full_name)
        if short:
            return short

        # 3. Fallback: use the last word of the team name
        #    For NCAAB un-mapped teams, this may not be ideal (e.g.,
        #    "Duke Blue Devils" -> "Devils"), but mapped teams are
        #    handled in step 1. This fallback is for truly unknown teams.
        parts = full_name.split()
        if len(parts) > 1:
            return parts[-1]
        return full_name

    def parse_games(self, raw_odds: list[dict]) -> list[LiveGame]:
        """Parse raw TheOddsAPI events into LiveGame objects."""
        if not raw_odds:
            return []

        from betting_intel.live.sport_configs import (
            league_from_sport_key,
            sport_key_to_group,
        )

        games: list[LiveGame] = []
        now_utc = datetime.now(timezone.utc)

        for event in raw_odds:
            try:
                # Guard: event must be a dict
                if not isinstance(event, dict):
                    logger.debug("Skipping non-dict event in raw_odds")
                    continue

                home_full = event.get("home_team", "") or ""
                away_full = event.get("away_team", "") or ""
                if not home_full or not away_full:
                    continue

                sport_key = (
                    event.get(
                        "_sport_config_key", event.get("sport_key", "basketball_nba")
                    )
                    or "basketball_nba"
                )
                league_name = league_from_sport_key(sport_key)
                sport_group = sport_key_to_group(sport_key)

                # Team name mapping using SportConfig-aware resolution
                home_short = self._resolve_short_name(home_full, sport_key)
                away_short = self._resolve_short_name(away_full, sport_key)

                commence_time = event.get("commence_time", "") or ""
                game_date = commence_time[:10] if commence_time else ""

                # Filter out games that have already started.
                # A 15-minute buffer allows games that just tipped off to still appear.
                try:
                    commence_dt = datetime.fromisoformat(
                        commence_time.replace("Z", "+00:00")
                    )
                    if commence_dt < now_utc - timedelta(minutes=15):
                        continue
                except (ValueError, TypeError):
                    pass

                # Extract market lines from all sportsbooks
                home_ml_values: list[float] = []
                away_ml_values: list[float] = []
                draw_ml_values: list[float] = []  # Soccer 3-way market
                total_values: list[float] = []
                over_odds_values: list[float] = []
                under_odds_values: list[float] = []
                spread_values: list[float] = []

                bookmakers = event.get("bookmakers", [])
                if not isinstance(bookmakers, list):
                    bookmakers = []

                for book in bookmakers:
                    if not isinstance(book, dict):
                        continue
                    markets = book.get("markets", [])
                    if not isinstance(markets, list):
                        continue
                    for market in markets:
                        if not isinstance(market, dict):
                            continue
                        key = market.get("key", "")
                        outcomes = market.get("outcomes", [])
                        if not isinstance(outcomes, list):
                            continue
                        if key == "h2h":
                            for o in outcomes:
                                if not isinstance(o, dict):
                                    continue
                                name = o.get("name", "")
                                price = o.get("price")
                                if price is not None:
                                    try:
                                        price_f = float(price)
                                    except (ValueError, TypeError):
                                        continue
                                    if name == home_full:
                                        home_ml_values.append(price_f)
                                    elif name == away_full:
                                        away_ml_values.append(price_f)
                                    elif name and name.lower() == "draw":
                                        draw_ml_values.append(price_f)
                        elif key == "spreads":
                            for o in outcomes:
                                if not isinstance(o, dict):
                                    continue
                                point = o.get("point")
                                if point is not None and o.get("name", "") == home_full:
                                    try:
                                        spread_values.append(float(point))
                                    except (ValueError, TypeError):
                                        continue
                        elif key == "totals":
                            for o in outcomes:
                                if not isinstance(o, dict):
                                    continue
                                point = o.get("point")
                                price = o.get("price")
                                if point is not None:
                                    try:
                                        total_values.append(float(point))
                                    except (ValueError, TypeError):
                                        continue
                                    if price is not None:
                                        name = o.get("name", "")
                                        try:
                                            price_f = float(price)
                                        except (ValueError, TypeError):
                                            continue
                                        if name == "Over":
                                            over_odds_values.append(price_f)
                                        elif name == "Under":
                                            under_odds_values.append(price_f)

                consensus_home_ml = self.median_or_none(home_ml_values)
                consensus_away_ml = self.median_or_none(away_ml_values)
                consensus_draw_ml = self.median_or_none(draw_ml_values)
                consensus_total = self.median_or_none(total_values)
                consensus_spread = self.median_or_none(spread_values)
                consensus_over_odds = self.median_or_none(over_odds_values)
                consensus_under_odds = self.median_or_none(under_odds_values)

                game = LiveGame(
                    game_id=str(
                        event.get(
                            "id",
                            f"{sport_key}_{home_short}_{away_short}_{game_date}",
                        )
                    ),
                    sport_key=sport_key,
                    league=league_name,
                    sport_group=sport_group,
                    home_team=home_full,
                    away_team=away_full,
                    home_team_short=home_short,
                    away_team_short=away_short,
                    commence_time=commence_time,
                    game_date=game_date,
                    home_ml=consensus_home_ml,
                    away_ml=consensus_away_ml,
                    draw_ml=consensus_draw_ml,
                    spread=consensus_spread,
                    market_total=consensus_total,
                    over_odds=consensus_over_odds,
                    under_odds=consensus_under_odds,
                    n_books_ml=len(home_ml_values),
                    n_books_total=len(total_values),
                    ml_std=self.std_or_none(home_ml_values),
                    odds_fetched_at=datetime.now().isoformat(),
                )
                games.append(game)

            except Exception as e:
                logger.debug(f"Skipping malformed game event: {e}")
                continue

        return games
