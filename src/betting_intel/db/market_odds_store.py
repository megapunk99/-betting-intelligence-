"""
MarketOddsStore — persist real market odds to the local database.

Architecture
────────────
  Every time LivePredictionEngine fetches odds from TheOddsAPI (or scrapers),
  each game's consensus lines are logged to the `market_odds` table. Over
  time this builds a rich history of real market data.

  During training (MarketInefficiencySystem.fit), the store is queried for
  real market lines matching each historical game. When found, those real
  lines are used instead of the ELO proxy. This is the key upgrade:
  v5.0 used ELO as a proxy; v5.1 uses REAL market data.

Table: market_odds
  - game_id        TheOddsAPI event ID
  - game_date      YYYY-MM-DD
  - home_team      Full team name
  - away_team      Full team name
  - home_ml        Consensus home moneyline
  - away_ml        Consensus away moneyline
  - spread         Consensus point spread
  - market_total   Consensus total points line
  - home_implied_prob   Market-implied home win probability (with vig)
  - vig_removed_home_prob  Market-implied home prob (vig removed)
  - n_books_ml     Number of books contributing to moneyline consensus
  - captured_at    ISO-8601 timestamp of this snapshot
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from betting_intel.db.connection import DatabaseManager
from betting_intel.live.models import LiveGame

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  PROBABILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (with vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def _remove_vig(home_p: float, away_p: float) -> float:
    """Remove vig from two implied probabilities, return home vig-free prob."""
    total = home_p + away_p
    if total > 0:
        return home_p / total
    return home_p


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET ODDS STORE
# ═══════════════════════════════════════════════════════════════════════════


class MarketOddsStore:
    """
    Persistence layer for historical market odds.

    Writes odds snapshots to the `market_odds` table whenever the engine
    refreshes, and reads them back during model training to supply real
    market-implied probabilities.

    Usage:
        store = MarketOddsStore()
        store.log_snapshot(game_id, home_team, away_team, home_ml=-150, ...)
        df = store.get_odds_for_date("2024-03-15")
        prob = store.get_market_prob_for_game(game_id="...", home_team="Celtics")
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self._db = db_manager or DatabaseManager()
        self.ensure_table()

    # ── Write ─────────────────────────────────────────────────────────────

    def log_snapshot(
        self,
        game_id: str,
        game_date: str,
        home_team: str,
        away_team: str,
        home_team_short: str = "",
        away_team_short: str = "",
        home_ml: Optional[float] = None,
        away_ml: Optional[float] = None,
        spread: Optional[float] = None,
        market_total: Optional[float] = None,
        over_odds: Optional[float] = None,
        under_odds: Optional[float] = None,
        n_books_ml: int = 0,
        n_books_total: int = 0,
        ml_std: Optional[float] = None,
        source: str = "theoddsapi",
        sport_key: str = "basketball_nba",
    ) -> bool:
        """
        Log a single game's odds to the market_odds table.

        This is called for EVERY game during EVERY refresh cycle, building
        up a rich time series of market data.

        Returns:
            True if written successfully, False on error.
        """
        try:
            # Compute implied probabilities at write time
            home_implied = (
                _american_to_implied(home_ml) if home_ml is not None else None
            )
            away_implied = (
                _american_to_implied(away_ml) if away_ml is not None else None
            )
            vig_free = (
                _remove_vig(home_implied, away_implied)
                if home_implied is not None and away_implied is not None
                else None
            )

            from betting_intel.db.schema import MarketOdds

            session = self._db.get_session()
            try:
                record = MarketOdds(
                    game_id=game_id,
                    game_date=game_date,
                    sport_key=sport_key,
                    home_team=home_team,
                    away_team=away_team,
                    home_team_short=home_team_short or home_team.split()[-1]
                    if " " in home_team
                    else home_team,
                    away_team_short=away_team_short or away_team.split()[-1]
                    if " " in away_team
                    else away_team,
                    home_ml=home_ml,
                    away_ml=away_ml,
                    spread=spread,
                    market_total=market_total,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    home_implied_prob=home_implied,
                    away_implied_prob=away_implied,
                    vig_removed_home_prob=vig_free,
                    n_books_ml=n_books_ml,
                    n_books_total=n_books_total,
                    ml_std=ml_std,
                    source=source,
                    captured_at=datetime.now().isoformat(),
                )
                session.add(record)
                session.commit()
                return True
            except Exception:
                session.rollback()
                logger.debug(
                    f"Failed to log odds snapshot for {game_id}", exc_info=True
                )
                return False
            finally:
                session.close()
        except Exception:
            logger.debug("Failed to log odds snapshot", exc_info=True)
            return False

    def log_snapshot_from_live_game(self, game: "LiveGame", source: str = "") -> bool:
        """
        Convenience: log odds from a LiveGame object.

        Called during engine refresh — one call per game in the snapshot.
        """
        return self.log_snapshot(
            game_id=game.game_id,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
            home_team_short=game.home_team_short,
            away_team_short=game.away_team_short,
            home_ml=game.home_ml,
            away_ml=game.away_ml,
            spread=game.spread,
            market_total=game.market_total,
            over_odds=game.over_odds,
            under_odds=game.under_odds,
            n_books_ml=game.n_books_ml,
            n_books_total=game.n_books_total,
            ml_std=game.ml_std,
            source=source or "engine",
            sport_key=game.sport_key,
        )

    def log_batch(self, games: list["LiveGame"], source: str = "") -> int:
        """
        Log odds for a batch of LiveGame objects.

        Args:
            games: List of LiveGame objects from a snapshot
            source: Source identifier (e.g., "theoddsapi")

        Returns:
            Number of successfully logged games.
        """
        count = 0
        for game in games:
            if game.home_ml is not None or game.market_total is not None:
                if self.log_snapshot_from_live_game(game, source=source):
                    count += 1
        if count > 0:
            logger.info(f"Logged {count} odds snapshots to market_odds table")
        return count

    # ── Read ──────────────────────────────────────────────────────────────

    def get_odds_for_date(
        self,
        game_date: str,
        sport_key: str = "basketball_nba",
        latest_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get all odds snapshots for a specific date.

        Args:
            game_date: YYYY-MM-DD
            sport_key: Filter by sport (default basketball_nba)
            latest_only: If True, return only the most recent snapshot per game

        Returns:
            DataFrame with one row per game (or per snapshot if latest_only=False)
        """
        from betting_intel.db.schema import MarketOdds

        session = self._db.get_session()
        try:
            query = (
                session.query(MarketOdds)
                .filter(
                    MarketOdds.game_date == game_date,
                    MarketOdds.sport_key == sport_key,
                )
                .order_by(MarketOdds.captured_at.desc())
            )
            records = query.all()

            if not records:
                return pd.DataFrame()

            data = [
                {
                    "game_id": r.game_id,
                    "home_team": r.home_team,
                    "away_team": r.away_team,
                    "home_team_short": r.home_team_short,
                    "away_team_short": r.away_team_short,
                    "home_ml": r.home_ml,
                    "away_ml": r.away_ml,
                    "spread": r.spread,
                    "market_total": r.market_total,
                    "home_implied_prob": r.home_implied_prob,
                    "vig_removed_home_prob": r.vig_removed_home_prob,
                    "n_books_ml": r.n_books_ml,
                    "captured_at": r.captured_at,
                }
                for r in records
            ]
            df = pd.DataFrame(data)

            if latest_only and not df.empty:
                # Keep only the most recent snapshot per game_id
                df = df.sort_values("captured_at", ascending=False)
                df = df.drop_duplicates(subset=["game_id"], keep="first")

            return df

        except Exception:
            logger.debug("Failed to query market odds", exc_info=True)
            return pd.DataFrame()
        finally:
            session.close()

    def get_market_prob_for_game(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
    ) -> Optional[float]:
        """
        Get the vig-removed home win probability from real market odds.

        This is THE key method for the MarketInefficiencySystem upgrade.
        Instead of using ELO as a proxy for market belief, this returns
        the REAL market-implied probability from stored odds.

        Uses a single query with OR conditions to handle both normal and
        swapped home/away team name ordering.

        Args:
            home_team: Home team full name (or short name)
            away_team: Away team full name (or short name)
            game_date: YYYY-MM-DD of the game

        Returns:
            Vig-removed home win probability (0-1), or None if not found.
        """
        from betting_intel.db.schema import MarketOdds

        session = self._db.get_session()
        try:
            from sqlalchemy import or_, and_

            # Single query: match either (home=A, away=B) or (home=B, away=A)
            record = (
                session.query(MarketOdds)
                .filter(
                    MarketOdds.game_date == game_date,
                    or_(
                        and_(
                            or_(
                                MarketOdds.home_team == home_team,
                                MarketOdds.home_team_short == home_team,
                            ),
                            or_(
                                MarketOdds.away_team == away_team,
                                MarketOdds.away_team_short == away_team,
                            ),
                        ),
                        and_(
                            or_(
                                MarketOdds.home_team == away_team,
                                MarketOdds.home_team_short == away_team,
                            ),
                            or_(
                                MarketOdds.away_team == home_team,
                                MarketOdds.away_team_short == home_team,
                            ),
                        ),
                    ),
                )
                .order_by(MarketOdds.captured_at.desc())
                .first()
            )

            if record is not None and record.vig_removed_home_prob is not None:
                # Determine orientation: check BOTH full name and short name
                # against the requested team. The match could be via
                # home_team="Boston Celtics" OR home_team_short="Celtics".
                stored_full = (record.home_team or "").lower()
                stored_short = (record.home_team_short or "").lower()
                target = home_team.lower()
                if stored_full == target or stored_short == target:
                    return float(record.vig_removed_home_prob)
                else:
                    return float(1.0 - record.vig_removed_home_prob)

            return None

        except Exception:
            logger.debug(
                f"Failed to get market prob for {home_team} vs {away_team}",
                exc_info=True,
            )
            return None
        finally:
            session.close()

    def get_market_probs_for_date_range(
        self,
        start_date: str,
        end_date: str,
        sport_key: str = "basketball_nba",
    ) -> dict[tuple[str, str, str], float]:
        """
        Get all market probabilities for games in a date range.

        Returns:
            Dict keyed by (home_team, away_team, game_date) → vig-free home prob
        """
        from betting_intel.db.schema import MarketOdds

        session = self._db.get_session()
        try:
            records = (
                session.query(MarketOdds)
                .filter(
                    MarketOdds.game_date >= start_date,
                    MarketOdds.game_date <= end_date,
                    MarketOdds.sport_key == sport_key,
                    MarketOdds.vig_removed_home_prob.isnot(None),
                )
                .order_by(MarketOdds.captured_at.desc())
                .all()
            )

            result: dict[tuple[str, str, str], float] = {}
            for r in records:
                key = (r.home_team, r.away_team, r.game_date)
                if key not in result and r.vig_removed_home_prob is not None:
                    result[key] = float(r.vig_removed_home_prob)

            return result

        except Exception:
            logger.debug("Failed to get market probs for date range", exc_info=True)
            return {}
        finally:
            session.close()

    def get_snapshot_count(self) -> int:
        """Get total number of odds snapshots stored."""
        from betting_intel.db.schema import MarketOdds

        session = self._db.get_session()
        try:
            return session.query(MarketOdds).count()
        except Exception:
            return 0
        finally:
            session.close()

    def get_unique_games_count(self) -> int:
        """Get number of unique games with odds data."""
        from betting_intel.db.schema import MarketOdds

        session = self._db.get_session()
        try:
            return session.query(MarketOdds.game_id).distinct().count()
        except Exception:
            return 0
        finally:
            session.close()

    def get_closing_vs_opening_prob(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Get both the opening and closing vig-free home win probabilities.

        Opening = first snapshot logged for this game.
        Closing = last (most recent) snapshot before game start.

        Returns:
            (opening_home_prob, closing_home_prob) or (None, None) if no data.
        """
        from betting_intel.db.schema import MarketOdds

        session = self._db.get_session()
        try:
            from sqlalchemy import or_, and_, asc

            records = (
                session.query(MarketOdds)
                .filter(
                    MarketOdds.game_date == game_date,
                    or_(
                        and_(
                            or_(
                                MarketOdds.home_team == home_team,
                                MarketOdds.home_team_short == home_team,
                            ),
                            or_(
                                MarketOdds.away_team == away_team,
                                MarketOdds.away_team_short == away_team,
                            ),
                        ),
                        and_(
                            or_(
                                MarketOdds.home_team == away_team,
                                MarketOdds.home_team_short == away_team,
                            ),
                            or_(
                                MarketOdds.away_team == home_team,
                                MarketOdds.away_team_short == home_team,
                            ),
                        ),
                    ),
                )
                .order_by(asc(MarketOdds.captured_at))
                .all()
            )

            if not records:
                return (None, None)

            first = records[0]
            last = records[-1]

            def _prob_for_home(r, home_target: str) -> Optional[float]:
                if r.vig_removed_home_prob is None:
                    return None
                stored_full = (r.home_team or "").lower()
                stored_short = (r.home_team_short or "").lower()
                target = home_target.lower()
                if stored_full == target or stored_short == target:
                    return float(r.vig_removed_home_prob)
                return float(1.0 - r.vig_removed_home_prob)

            opening = _prob_for_home(first, home_team)
            closing = _prob_for_home(last, home_team)

            return (opening, closing)

        except Exception:
            logger.debug(
                f"Failed to get opening/closing probs for {home_team} vs {away_team}",
                exc_info=True,
            )
            return (None, None)
        finally:
            session.close()

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get summary stats about stored market odds."""
        return {
            "total_snapshots": self.get_snapshot_count(),
            "unique_games": self.get_unique_games_count(),
        }

    # ── Table Creation ────────────────────────────────────────────────────

    def ensure_table(self):
        """Ensure the market_odds table exists."""
        from betting_intel.db.schema import Base, MarketOdds

        Base.metadata.create_all(self._db.engine, tables=[MarketOdds.__table__])
        logger.debug("market_odds table ensured")


__all__ = [
    "MarketOddsStore",
]
