"""
Live data models for the prediction engine.

Pure dataclasses and shared constants — no business logic beyond
property accessors and serialization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

from betting_intel.config import ODDS_CACHE_TTL_SECONDS


# ── Thresholds ────────────────────────────────────────────────────────────
PREDICTION_REFRESH_INTERVAL = 60   # Re-generate predictions every 60s
LIVE_GAME_LEEWAY_MINUTES = 60      # Game is "live" if started within 60 min of now
MIN_EDGE_THRESHOLD = 0.03          # No recommendations below 3% edge


# ── Data Models ───────────────────────────────────────────────────────────

@dataclass
class LiveGame:
    """A single game — live or upcoming — with real market data."""
    game_id: str
    sport_key: str
    home_team: str
    away_team: str
    home_team_short: str
    away_team_short: str
    commence_time: str  # ISO 8601
    game_date: str      # YYYY-MM-DD

    # League / sport identification
    league: str = "NBA"
    sport_group: str = "Basketball"

    # Market lines (from consensus across sportsbooks)
    home_ml: Optional[float] = None
    away_ml: Optional[float] = None
    spread: Optional[float] = None
    market_total: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None

    # Consensus metadata
    n_books_ml: int = 0
    n_books_total: int = 0
    ml_std: Optional[float] = None

    # Status
    is_live: bool = False
    is_today: bool = False
    is_tomorrow: bool = False

    # ML predictions — moneyline edge (filled by MarketInefficiencySystem)
    predicted_total: Optional[float] = None  # home_win_prob (0-1) from robust system
    edge_pct: Optional[float] = None         # predicted market error for moneyline
    direction: Optional[str] = None           # "home" or "away"
    confidence: Optional[str] = None          # "high", "medium", "low", "neutral"

    # ML predictions — totals edge (filled by TotalsRegressor)
    total_prediction: Optional[float] = None  # predicted total points (e.g. 225.5)
    total_edge_pct: Optional[float] = None    # edge on the total (positive = over)
    total_direction: Optional[str] = None     # "over" or "under" or "neutral"
    total_confidence: Optional[str] = None    # "high", "medium", "low"

    # Quarter & Half projections
    q1_home: Optional[float] = None
    q1_away: Optional[float] = None
    q1_total: Optional[float] = None
    q2_home: Optional[float] = None
    q2_away: Optional[float] = None
    q2_total: Optional[float] = None
    q3_home: Optional[float] = None
    q3_away: Optional[float] = None
    q3_total: Optional[float] = None
    q4_home: Optional[float] = None
    q4_away: Optional[float] = None
    q4_total: Optional[float] = None
    h1_home: Optional[float] = None
    h1_away: Optional[float] = None
    h1_total: Optional[float] = None
    h2_home: Optional[float] = None
    h2_away: Optional[float] = None
    h2_total: Optional[float] = None

    # Kelly stake (computed by KellyStaker in _predict_with_robust_system)
    stake_dollars: float = 0.0

    # Feature importance (how the model arrived at this prediction)
    feature_importance: Optional[dict[str, float]] = None  # {human_readable_name: importance_weight}

    # Bet recommendations
    recommended_quarter: Optional[str] = None   # e.g. "Q1", "Q2", "1H"
    recommended_direction: Optional[str] = None # "over" or "under"

    # Timestamps
    odds_fetched_at: Optional[str] = None
    predicted_at: Optional[str] = None

    @property
    def matchup(self) -> str:
        return f"{self.away_team_short} @ {self.home_team_short}"

    @property
    def commence_datetime(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.commence_time.replace("Z", "+00:00"))
        except Exception:
            return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LivePredictionSnapshot:
    """Complete snapshot of all live + upcoming predictions."""
    live_games: list[LiveGame] = field(default_factory=list)
    today_games: list[LiveGame] = field(default_factory=list)
    tomorrow_games: list[LiveGame] = field(default_factory=list)
    next_two_days: list[LiveGame] = field(default_factory=list)

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    n_live: int = 0
    n_today: int = 0
    n_tomorrow: int = 0
    n_total: int = 0
    fresh_odds: bool = False

    # Pre-computed chart data (built once when snapshot is constructed)
    chart_data: Optional[dict] = None

    # Fields to exclude from serialization (chart_data, internal state)
    _exclude_from_dict: set = field(default_factory=lambda: {"chart_data", "_exclude_from_dict"})

    def __post_init__(self):
        self.n_live = len(self.live_games)
        self.n_today = len(self.today_games)
        self.n_tomorrow = len(self.tomorrow_games)
        self.n_total = len(self.next_two_days)
        self.chart_data = self._build_chart_data()

    def _build_chart_data(self) -> dict:
        edges = []
        confidence_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "neutral": 0}
        over_count = 0
        under_count = 0
        neutral_count = 0

        for g in self.next_two_days:
            d = g.to_dict()
            edge_pct = d.get("edge_pct")
            if edge_pct is not None and edge_pct != 0:
                edges.append({
                    "matchup": g.matchup,
                    "edge_pct": round(edge_pct * 100, 1),
                    "predicted_total": d.get("predicted_total"),
                    "market_total": d.get("market_total"),
                    "is_live": d.get("is_live", False),
                    "confidence": d.get("confidence", "low"),
                    "home_team": d.get("home_team_short", ""),
                    "away_team": d.get("away_team_short", ""),
                    "spread": d.get("spread"),
                    "n_books_ml": d.get("n_books_ml", 0),
                    "direction": d.get("direction", "neutral"),
                })

            c = d.get("confidence", "low") or "low"
            if c in confidence_counts:
                confidence_counts[c] += 1
            else:
                confidence_counts[c] = 1

            direction = d.get("direction", "neutral")
            if direction == "over":
                over_count += 1
            elif direction == "under":
                under_count += 1
            else:
                neutral_count += 1

        return {
            "n_live": self.n_live,
            "n_today": self.n_today,
            "n_tomorrow": self.n_tomorrow,
            "n_total": self.n_total,
            "edges": edges,
            "confidence_breakdown": confidence_counts,
            "direction_breakdown": {
                "over": over_count,
                "under": under_count,
                "neutral": neutral_count,
            },
            "generated_at": self.generated_at,
            "fresh_odds": self.fresh_odds,
        }

    def to_dict(self) -> dict:
        d = {
            "live_games": [g.to_dict() for g in self.live_games],
            "today_games": [g.to_dict() for g in self.today_games],
            "tomorrow_games": [g.to_dict() for g in self.tomorrow_games],
            "next_two_days": [g.to_dict() for g in self.next_two_days],
            "generated_at": self.generated_at,
            "n_live": self.n_live,
            "n_today": self.n_today,
            "n_tomorrow": self.n_tomorrow,
            "n_total": self.n_total,
            "fresh_odds": self.fresh_odds,
        }
        exclude = getattr(self, '_exclude_from_dict', set())
        for field_name in exclude:
            d.pop(field_name, None)
        return d
