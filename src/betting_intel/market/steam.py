"""
Steam Move Detector — Phase 3.11 of the Professional Betting Intelligence Platform.

Detects:
    - rapid_line_movement: lines moving faster than a configurable threshold
    - cross_book_movement: multiple books moving in the same direction simultaneously

Generates alerts when:
    - multiple_books_move in the same direction
    - within_short_time_window (configurable, default 30 minutes)
    - velocity exceeds threshold (configurable, default 0.5 pts/hour)

A steam move is one of the strongest signals in sports betting — it indicates
"smart money" (sharp bettors) is entering the market.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from betting_intel.market.movement import MarketMovementEngine, LineMovement, MarketTrend

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class SteamMoveType(Enum):
    RAPID_LINE_MOVE = "rapid_line_move"           # Single book, fast movement
    CROSS_BOOK_MOVE = "cross_book_move"           # Multiple books, same direction
    CONSENSUS_SHIFT = "consensus_shift"           # Market-wide consensus change
    SHARP_REVERSAL = "sharp_reversal"             # Line reverses direction (fade the public)


@dataclass
class SteamAlert:
    """A detected steam move alert."""
    game_id: str
    home_team: str
    away_team: str
    market_type: str          # 'totals', 'spreads', 'h2h'
    move_type: str            # SteamMoveType value

    # Movement details
    from_line: float = 0.0
    to_line: float = 0.0
    velocity: float = 0.0     # Points per hour
    num_books: int = 1
    agreement_pct: float = 1.0

    # Direction
    direction: str = "up"     # 'up' or 'down'
    is_sharp: bool = True     # Steam moves are assumed sharp by default

    # Timing
    detected_at: str = ""
    time_window_minutes: float = 30.0

    # Strength
    confidence: float = 0.7   # 0-1
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()

    @property
    def label(self) -> str:
        return (
            f"{'🔥' if self.confidence > 0.8 else '⚡'} "
            f"{self.away_team} @ {self.home_team} | "
            f"{self.market_type.upper()} "
            f"{self.from_line} -> {self.to_line} "
            f"({self.direction.upper()})"
        )

    def summary(self) -> str:
        return (
            f"{self.label}\n"
            f"  Type: {self.move_type.replace('_', ' ').title()}\n"
            f"  Velocity: {self.velocity:.2f} pts/hr | "
            f"Books: {self.num_books} | "
            f"Confidence: {self.confidence:.0%}\n"
            f"  Detected: {self.detected_at}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  STEAM MOVE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class SteamMoveDetector:
    """
    Detects steam moves by analyzing market movements across sportsbooks.

    Configuration:
        - velocity_threshold: minimum pts/hr to trigger (default 0.5)
        - time_window_minutes: lookback window for cross-book detection (default 30)
        - min_books_for_cross: minimum books needed for cross-book alert (default 2)
        - min_agreement_pct: minimum agreement % for cross-book (default 0.7)

    Usage:
        detector = SteamMoveDetector(DB_PATH)
        alerts = detector.scan_all_active_games()
        for alert in alerts:
            print(alert.summary())
            dispatch_alert(alert)
    """

    def __init__(
        self,
        db_path: Path,
        velocity_threshold: float = 0.5,
        time_window_minutes: float = 30.0,
        min_books_for_cross: int = 2,
        min_agreement_pct: float = 0.7,
    ):
        self.movement_engine = MarketMovementEngine(db_path)
        self.velocity_threshold = velocity_threshold
        self.time_window_minutes = time_window_minutes
        self.min_books_for_cross = min_books_for_cross
        self.min_agreement_pct = min_agreement_pct

    # ═══════════════════════════════════════════════════════════════════
    #  SCANNING
    # ═══════════════════════════════════════════════════════════════════

    def scan_all_active_games(self) -> List[SteamAlert]:
        """
        Scan all active (unfinished) games for steam moves.

        Returns:
            List of SteamAlert sorted by confidence (descending)
        """
        movements = self.movement_engine.get_all_active_movements()
        all_alerts: List[SteamAlert] = []

        for game_id, records in movements.items():
            for record in records:
                alerts = self._scan_single_market(record)
                all_alerts.extend(alerts)

        # Sort by confidence descending
        all_alerts.sort(key=lambda a: a.confidence, reverse=True)
        return all_alerts

    def scan_game(self, game_id: str) -> List[SteamAlert]:
        """Scan a specific game for steam moves across all market types."""
        records = []
        for market in ["totals", "spreads", "h2h"]:
            rec = self.movement_engine.get_market_record(game_id, market)
            if rec:
                records.append(rec)

        alerts = []
        for record in records:
            alerts.extend(self._scan_single_market(record))

        alerts.sort(key=lambda a: a.confidence, reverse=True)
        return alerts

    def _scan_single_market(self, record) -> List[SteamAlert]:
        """Check a single market record for steam signals."""
        alerts = []

        if not record:
            return alerts

        home = record.home_team
        away = record.away_team

        # 1. Check rapid line movement from individual movements
        if record.opening_to_current and record.opening_to_current.velocity >= self.velocity_threshold:
            mov = record.opening_to_current
            severity = self._determine_severity(mov.velocity, 1)
            alerts.append(SteamAlert(
                game_id=record.game_id,
                home_team=home,
                away_team=away,
                market_type=record.market_type,
                move_type=SteamMoveType.RAPID_LINE_MOVE.value,
                from_line=mov.from_odds or 0,
                to_line=mov.to_odds or 0,
                velocity=mov.velocity,
                num_books=1,
                direction=mov.direction,
                time_window_minutes=self.time_window_minutes,
                confidence=min(0.5 + mov.velocity * 0.1, 0.95),
                severity=severity,
            ))

        # 2. Check cross-book movement via trends
        trends = self.movement_engine.get_trends(record.game_id)
        for trend in trends:
            if trend.market_type != record.market_type:
                continue

            if (trend.num_books >= self.min_books_for_cross
                    and trend.agreement_pct >= self.min_agreement_pct
                    and abs((trend.consensus_current or 0) - (trend.consensus_opening or 0)) >= 1.0):

                velocity = abs((trend.consensus_current or 0) - (trend.consensus_opening or 0)) / max(trend.num_books, 1)
                if velocity >= self.velocity_threshold:
                    severity = self._determine_severity(velocity, trend.num_books)
                    alerts.append(SteamAlert(
                        game_id=record.game_id,
                        home_team=home,
                        away_team=away,
                        market_type=trend.market_type,
                        move_type=SteamMoveType.CROSS_BOOK_MOVE.value,
                        from_line=trend.consensus_opening or 0,
                        to_line=trend.consensus_current or 0,
                        velocity=velocity,
                        num_books=trend.num_books,
                        agreement_pct=trend.agreement_pct,
                        direction=trend.move_direction,
                        time_window_minutes=self.time_window_minutes,
                        confidence=min(0.6 + trend.agreement_pct * 0.3, 0.98),
                        severity=severity,
                    ))

        # 3. Check for sharp reversal
        if record.opening_to_current and record.current_to_closing:
            op = record.opening_to_current
            cl = record.current_to_closing
            if op and cl and op.direction != cl.direction:
                if abs(op.raw_difference) >= 1.0 and abs(cl.raw_difference) >= 1.0:
                    alerts.append(SteamAlert(
                        game_id=record.game_id,
                        home_team=home,
                        away_team=away,
                        market_type=record.market_type,
                        move_type=SteamMoveType.SHARP_REVERSAL.value,
                        from_line=op.to_odds or 0,
                        to_line=cl.to_odds or 0,
                        velocity=max(op.velocity, cl.velocity),
                        num_books=1,
                        direction=cl.direction,
                        time_window_minutes=self.time_window_minutes,
                        confidence=0.85,
                        severity="high",
                    ))

        return alerts

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _determine_severity(self, velocity: float, num_books: int) -> str:
        """Determine severity based on velocity and number of books."""
        score = velocity * (1 + num_books * 0.5)
        if score >= 5.0:
            return "critical"
        if score >= 2.0:
            return "high"
        if score >= 1.0:
            return "medium"
        return "low"

    def format_alerts(self, alerts: List[SteamAlert]) -> str:
        """Format steam alerts for display/dispatch."""
        if not alerts:
            return "No steam moves detected."

        lines = [f"🔥 STEAM MOVE ALERTS ({len(alerts)} detected)", "═" * 50]
        for alert in alerts:
            lines.append(f"\n{alert.summary()}")
            lines.append("─" * 40)

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "velocity_threshold": self.velocity_threshold,
            "time_window_minutes": self.time_window_minutes,
            "min_books_for_cross": self.min_books_for_cross,
            "min_agreement_pct": self.min_agreement_pct,
        }
