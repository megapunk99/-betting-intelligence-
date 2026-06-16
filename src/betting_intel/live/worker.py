"""
LivePredictionWorker — background worker that continuously refreshes predictions.

Designed to be run in a separate thread or asyncio task.
Updates the shared engine's snapshot every refresh_interval seconds.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from betting_intel.live.engine import LivePredictionEngine

logger = logging.getLogger(__name__)


class LivePredictionWorker:
    """Background worker that continuously refreshes predictions."""

    def __init__(self, engine: LivePredictionEngine):
        self.engine = engine
        self._running = False

    def start(self):
        """Start the continuous refresh loop (blocking)."""
        self._running = True
        logger.info("Live prediction worker started")

        while self._running:
            try:
                snapshot = self.engine.refresh_now()
                if snapshot.n_total > 0:
                    logger.info(
                        f"Refreshed: {snapshot.n_live} live, "
                        f"{snapshot.n_today - snapshot.n_live} upcoming today, "
                        f"{snapshot.n_tomorrow} tomorrow — "
                        f"{snapshot.n_total} total in 2-day window"
                    )
                else:
                    logger.info("Refresh complete — no real games available")
            except Exception as e:
                logger.error(f"Refresh cycle failed: {e}")

            time.sleep(self._refresh_interval)

    def stop(self):
        """Stop the refresh loop."""
        self._running = False
        logger.info("Live prediction worker stopped")

    @property
    def _refresh_interval(self) -> int:
        return self.engine._refresh_interval
