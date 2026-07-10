"""
SnapshotBuilder — assembles LivePredictionSnapshot from already-prepared data.

This is a PURE ASSEMBLER. It does NOT call auto-resolve, fetch odds, or
run predictions. The engine handles those steps upfront so tests can
patch engine methods directly.

Responsibilities:
  1. Log parsed games to the market odds store
  2. Classify games as live / today / tomorrow
  3. Build and return the LivePredictionSnapshot
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from betting_intel.live.models import (
    LiveGame,
    LivePredictionSnapshot,
    LIVE_GAME_LEEWAY_MINUTES,
)

logger = logging.getLogger(__name__)


class SnapshotBuilder:
    """
    Assembles prediction snapshots from already-parsed + predicted games.

    No fetching, no prediction, no auto-resolve. Pure data assembly.
    """

    def __init__(self, market_odds_store: Any):
        self._market_odds_store = market_odds_store
        self._last_auto_resolve: str | None = None

    def build_snapshot(
        self,
        all_games: list[LiveGame],
        fresh_odds: bool,
    ) -> LivePredictionSnapshot:
        """
        Assemble a prediction snapshot from already-parsed games.

        Args:
            all_games: Parsed and predicted LiveGame objects.
            fresh_odds: Whether the odds were freshly fetched.

        Returns:
            Populated LivePredictionSnapshot.
        """
        if not all_games:
            logger.info("No games to assemble — returning empty snapshot")
            return LivePredictionSnapshot(fresh_odds=fresh_odds)

        # Step 1: Log to market odds store
        try:
            if hasattr(self._market_odds_store, "log_batch"):
                self._market_odds_store.log_batch(all_games, source="engine_refresh")
        except Exception:
            logger.debug("Failed to log odds snapshots (non-critical)", exc_info=True)

        # Step 2: Classify games
        try:
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")
            tomorrow_str = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")
            day_after_str = (now_utc + timedelta(days=2)).strftime("%Y-%m-%d")
        except Exception:
            logger.error("Failed to compute date strings for classification")
            return LivePredictionSnapshot(fresh_odds=fresh_odds)

        live_games: list[LiveGame] = []
        today_games: list[LiveGame] = []
        tomorrow_games: list[LiveGame] = []
        day_after_games: list[LiveGame] = []

        for game in all_games:
            try:
                if game is None:
                    continue
                game.is_today = bool(game.game_date == today_str)
                game.is_tomorrow = bool(game.game_date == tomorrow_str)
                try:
                    commence_dt = game.commence_datetime
                except Exception:
                    commence_dt = None
                if commence_dt and commence_dt < now_utc:
                    age_minutes = (now_utc - commence_dt).total_seconds() / 60
                    game.is_live = age_minutes < LIVE_GAME_LEEWAY_MINUTES
                if game.is_live:
                    live_games.append(game)
                if game.is_today:
                    today_games.append(game)
                if game.is_tomorrow:
                    tomorrow_games.append(game)
                if game.game_date == day_after_str:
                    day_after_games.append(game)
            except Exception as e:
                logger.debug(f"Skipping malformed game in snapshot: {e}")
                continue

        # Step 3: Assemble next_two_days in deduped order
        seen_ids: set[str] = set()
        next_two_days: list[LiveGame] = []
        for game in today_games + tomorrow_games + day_after_games:
            if game.game_id not in seen_ids:
                next_two_days.append(game)
                seen_ids.add(game.game_id)

        snapshot = LivePredictionSnapshot(
            live_games=live_games,
            today_games=[g for g in today_games if not g.is_live] + live_games,
            tomorrow_games=tomorrow_games,
            next_two_days=next_two_days,
            generated_at=datetime.now().isoformat(),
            fresh_odds=fresh_odds,
        )

        if live_games:
            logger.info(f"LIVE: {len(live_games)} games in progress")
        logger.info(
            f"Snapshot built: {len(today_games)} today, "
            f"{len(tomorrow_games)} tomorrow, "
            f"{len(day_after_games)} day after, "
            f"{len(next_two_days)} total in window"
        )
        return snapshot

    def set_auto_resolve_timestamp(self, timestamp: str | None):
        """Store the timestamp of the last auto-resolve (called by engine)."""
        self._last_auto_resolve = timestamp
