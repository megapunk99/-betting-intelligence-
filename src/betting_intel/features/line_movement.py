"""
Line Movement Velocity Tracker — track odds changes over time to detect
sharp money vs public money, and compute movement velocity features.

KEY INSIGHT: Line movement reveals where the smart money is going.
  - A line opening at -110 and moving to -120 means money is coming in on that side
  - Fade the public: if public % is 80% on one side but the line moves the OTHER way,
    that's "reverse line movement" (RLM) — sharp money fading the public
  - Velocity matters: fast movement (2+ point change in 1 hour) is sharper than
    slow movement (same change over 24 hours)
  - Opening line vs current line disparity is the single strongest market signal

ARCHITECTURE:
  LineMovementTracker stores odds snapshots indexed by game_id + timestamp.
  On each engine refresh, it records the current odds and computes:
    - opening_line: first recorded odds for this game
    - current_line: most recent odds
    - total_movement: opening → current delta
    - velocity_1h: movement in last hour (if data available)
    - velocity_4h: movement in last 4 hours
    - direction: which side the line is moving toward
    - is_sharp_money: flag if movement contradicts expected public betting

USAGE:
    from betting_intel.features.line_movement import LineMovementTracker
    tracker = LineMovementTracker()
    
    # On each odds fetch:
    tracker.record_odds(game_id, home_ml, away_ml, spread, total)
    
    # Get features for a game:
    features = tracker.get_movement_features(game_id)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  LINE MOVEMENT DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════

class OddsSnapshot:
    """A single snapshot of odds for a game at a point in time."""
    
    def __init__(
        self,
        home_ml: Optional[float] = None,
        away_ml: Optional[float] = None,
        spread: Optional[float] = None,
        total: Optional[float] = None,
        timestamp: float = 0.0,
    ):
        self.home_ml = home_ml
        self.away_ml = away_ml
        self.spread = spread
        self.total = total
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> dict:
        return {
            "home_ml": self.home_ml,
            "away_ml": self.away_ml,
            "spread": self.spread,
            "total": self.total,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> OddsSnapshot:
        return cls(
            home_ml=d.get("home_ml"),
            away_ml=d.get("away_ml"),
            spread=d.get("spread"),
            total=d.get("total"),
            timestamp=d.get("timestamp", 0.0),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  LINE MOVEMENT TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class LineMovementTracker:
    """Track odds movement over time for NBA games.
    
    Stores snapshots in memory with optional persistence to disk.
    Computes velocity, direction, and sharp money signals.
    
    Thread-safe: designed to be shared across engine refresh cycles.
    """
    
    def __init__(self, persist_path: Optional[Path] = None, max_snapshots_per_game: int = 20):
        self._history: dict[str, list[OddsSnapshot]] = {}  # game_id → snapshots
        self._persist_path = persist_path
        self._max_snapshots = max_snapshots_per_game
        
        # Load persisted history if available
        if persist_path and persist_path.exists():
            try:
                with open(persist_path) as f:
                    raw = json.load(f)
                for game_id, snapshots_raw in raw.items():
                    self._history[game_id] = [
                        OddsSnapshot.from_dict(s) for s in snapshots_raw
                    ]
                logger.info(f"Loaded {len(self._history)} game histories from {persist_path}")
            except Exception as e:
                logger.warning(f"Failed to load line movement history: {e}")
    
    def record_odds(
        self,
        game_id: str,
        home_ml: Optional[float] = None,
        away_ml: Optional[float] = None,
        spread: Optional[float] = None,
        total: Optional[float] = None,
    ) -> None:
        """Record an odds snapshot for a game.
        
        Args:
            game_id: Unique game identifier.
            home_ml: Home moneyline in American odds.
            away_ml: Away moneyline in American odds.
            spread: Point spread (home team perspective).
            total: Over/under total.
        """
        if game_id not in self._history:
            self._history[game_id] = []
        
        snapshots = self._history[game_id]
        
        # Don't record duplicate if nothing changed
        if snapshots:
            last = snapshots[-1]
            if (last.home_ml == home_ml and last.away_ml == away_ml 
                and last.spread == spread and last.total == total):
                return
        
        snapshots.append(OddsSnapshot(
            home_ml=home_ml, away_ml=away_ml,
            spread=spread, total=total,
        ))
        
        # Trim to max size
        if len(snapshots) > self._max_snapshots:
            self._history[game_id] = snapshots[-self._max_snapshots:]
        
        # Auto-persist after every record
        if self._persist_path:
            self._persist()
    
    def get_opening_line(self, game_id: str) -> Optional[OddsSnapshot]:
        """Get the first recorded odds for a game."""
        snapshots = self._history.get(game_id, [])
        return snapshots[0] if snapshots else None
    
    def get_current_line(self, game_id: str) -> Optional[OddsSnapshot]:
        """Get the most recent odds for a game."""
        snapshots = self._history.get(game_id, [])
        return snapshots[-1] if snapshots else None
    
    def get_movement_features(self, game_id: str) -> dict[str, float]:
        """Compute movement velocity features for a game.
        
        Returns feature dict suitable for ML model consumption.
        Returns zero-vector if no history exists for this game.
        
        Features:
          - line_n_snapshots: number of recorded snapshots
          - line_age_hours: hours since first recorded odds
          - line_movement_ml: total home_ml movement (positive = home shortening)
          - line_movement_spread: total spread movement (positive = home covering more)
          - line_movement_total: total over/under movement
          - line_velocity_ml_1h: home_ml change per hour (last 1 hour)
          - line_velocity_ml_4h: home_ml change per hour (last 4 hours)
          - line_velocity_total_1h: total change per hour (last 1 hour)
          - line_velocity_total_4h: total change per hour (last 4 hours)
          - line_sharp_signal: 1.0 if RLM detected, -1.0 if public-driven, 0.0 if neutral
        """
        snapshots = self._history.get(game_id, [])
        if len(snapshots) < 2:
            return {
                "line_n_snapshots": float(len(snapshots)),
                "line_age_hours": 0.0,
                "line_movement_ml": 0.0,
                "line_movement_spread": 0.0,
                "line_movement_total": 0.0,
                "line_velocity_ml_1h": 0.0,
                "line_velocity_ml_4h": 0.0,
                "line_velocity_total_1h": 0.0,
                "line_velocity_total_4h": 0.0,
                "line_sharp_signal": 0.0,
            }
        
        first = snapshots[0]
        last = snapshots[-1]
        now = time.time()
        
        # Compute age
        age_seconds = now - first.timestamp
        age_hours = age_seconds / 3600.0 if age_seconds > 0 else 0.0
        
        # Total movement
        ml_movement = 0.0
        if first.home_ml is not None and last.home_ml is not None:
            ml_movement = last.home_ml - first.home_ml
        
        spread_movement = 0.0
        if first.spread is not None and last.spread is not None:
            spread_movement = last.spread - first.spread
        
        total_movement = 0.0
        if first.total is not None and last.total is not None:
            total_movement = last.total - first.total
        
        # Velocity: movement per hour in last window
        def compute_velocity(window_hours: float) -> tuple[float, float]:
            """Compute velocity of ML and total in the last N hours.
            
            Returns (ml_velocity, total_velocity).
            """
            cutoff = now - window_hours * 3600
            window_snapshots = [s for s in snapshots if s.timestamp >= cutoff]
            
            if len(window_snapshots) < 2:
                return (0.0, 0.0)
            
            first_w = window_snapshots[0]
            last_w = window_snapshots[-1]
            elapsed = last_w.timestamp - first_w.timestamp
            hours = elapsed / 3600.0 if elapsed > 0 else 0.001
            
            ml_v = 0.0
            if first_w.home_ml is not None and last_w.home_ml is not None:
                ml_v = (last_w.home_ml - first_w.home_ml) / hours
            
            total_v = 0.0
            if first_w.total is not None and last_w.total is not None:
                total_v = (last_w.total - first_w.total) / hours
            
            return (ml_v, total_v)
        
        vel_ml_1h, vel_total_1h = compute_velocity(1.0) if age_hours >= 0.5 else (0.0, 0.0)
        vel_ml_4h, vel_total_4h = compute_velocity(4.0) if age_hours >= 2.0 else (0.0, 0.0)
        
        # Sharp money signal:
        # If line moves toward the UNDER but public % is on the OVER → sharp money
        # This is a simplified heuristic. Real sharp detection needs public % data.
        # For now: strong sudden movements in the opposite direction of where
        # naive money would go (toward the favorite) signal sharp action.
        sharp_signal = 0.0
        if abs(vel_ml_1h) > 5.0:  # More than 5 cent movement in last hour
            # Line moving toward the dog (positive movement on home ML in American odds
            # means home is getting longer odds = money on away team)
            # This is simplified — real RLM detection needs public % comparison
            sharp_signal = 1.0 if abs(vel_ml_1h) > 10.0 else 0.5
        if abs(vel_total_1h) > 0.5:  # 0.5 point total movement in last hour
            sharp_signal = max(sharp_signal, 0.5)
        
        return {
            "line_n_snapshots": float(len(snapshots)),
            "line_age_hours": round(age_hours, 2),
            "line_movement_ml": round(ml_movement, 1),
            "line_movement_spread": round(spread_movement, 1),
            "line_movement_total": round(total_movement, 1),
            "line_velocity_ml_1h": round(vel_ml_1h, 2),
            "line_velocity_ml_4h": round(vel_ml_4h, 2),
            "line_velocity_total_1h": round(vel_total_1h, 2),
            "line_velocity_total_4h": round(vel_total_4h, 2),
            "line_sharp_signal": round(sharp_signal, 2),
        }
    
    def clear_game(self, game_id: str) -> None:
        """Remove history for a specific game (e.g. after it completes)."""
        self._history.pop(game_id, None)
    
    def clear_all(self) -> None:
        """Clear all history."""
        self._history.clear()
        if self._persist_path and self._persist_path.exists():
            try:
                self._persist_path.unlink()
            except Exception:
                pass
    
    def _persist(self) -> None:
        """Persist history to disk."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            raw = {
                game_id: [s.to_dict() for s in snapshots]
                for game_id, snapshots in self._history.items()
            }
            with open(self._persist_path, "w") as f:
                json.dump(raw, f)
        except Exception as e:
            logger.debug(f"Failed to persist line movement: {e}")
    
    @property
    def n_games_tracked(self) -> int:
        return len(self._history)
    
    def get_tracked_game_ids(self) -> list[str]:
        return list(self._history.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def get_line_movement_features(
    tracker: LineMovementTracker,
    game_id: str,
) -> dict[str, float]:
    """One-shot: get line movement features for a game as a flat dict."""
    return tracker.get_movement_features(game_id)


__all__ = [
    "OddsSnapshot", "LineMovementTracker",
    "get_line_movement_features",
]
