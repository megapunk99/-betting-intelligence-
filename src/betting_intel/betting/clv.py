"""
Closing Line Value (CLV) Tracker — Phase 1.5 of the Professional Betting Intelligence Platform.

CLV is the single most important metric for serious bettors.
If you consistently beat the closing line, you have a true edge.

CLV = (bet_odds - closing_odds) / abs(closing_odds)   for negative odds
    = (closing_odds - bet_odds) / bet_odds              for positive odds

In practice:
    CLV difference: the difference between the odds you bet and the closing line
    CLV percentage: the percentage edge you got vs the market

Stores:
    bet_odds
    closing_odds

Calculates:
    clv_difference
    clv_percentage

Dashboard metrics:
    average_clv
    positive_clv_rate

CLV must be tracked for every recommendation.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CLVRecord:
    """A single CLV tracking record."""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    market_type: str        # 'moneyline', 'spread', 'total'
    bet_side: str            # 'home', 'away', 'over', 'under'

    # What we bet
    bet_odds_american: float
    bet_timestamp: str

    # What the market closed at
    closing_odds_american: float

    # Computed
    clv_difference: float = 0.0
    clv_percentage: float = 0.0

    # Outcome (optional, for later recording)
    bet_won: Optional[bool] = None
    model_probability: float = 0.0
    edge_at_bet_time: float = 0.0

    # Metadata
    id: int = 0
    created_at: str = ""


@dataclass
class CLVAggregate:
    """Aggregated CLV statistics."""
    total_bets: int = 0
    total_won: int = 0
    total_lost: int = 0

    average_clv_pct: float = 0.0
    positive_clv_rate: float = 0.0
    median_clv_pct: float = 0.0

    # By market
    ml_clv: float = 0.0
    spread_clv: float = 0.0
    total_clv: float = 0.0

    # Best/worst
    best_clv: float = 0.0
    worst_clv: float = 0.0

    # ROI
    total_staked: float = 0.0
    total_return: float = 0.0
    roi_pct: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  CLV TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class CLVTracker:
    """
    Tracks Closing Line Value for every bet.

    Usage:
        tracker = CLVTracker(DB_PATH)
        tracker.record_bet(
            game_id="...",
            home_team="Spurs",
            away_team="Knicks",
            bet_side="under",
            market_type="total",
            bet_odds_american=-110,
            closing_odds_american=-115,
            model_probability=0.55,
            edge_at_bet_time=0.03,
        )
        stats = tracker.get_aggregate_stats()
        print(f"Average CLV: {stats.average_clv_pct:.2%}")
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Create CLV tracking tables."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS clv_tracking (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id         TEXT NOT NULL,
                    home_team       TEXT NOT NULL,
                    away_team       TEXT NOT NULL,
                    game_date       TEXT NOT NULL,
                    market_type     TEXT NOT NULL,
                    bet_side        TEXT NOT NULL,
                    bet_odds_american REAL NOT NULL,
                    closing_odds_american REAL NOT NULL,
                    clv_difference  REAL DEFAULT 0,
                    clv_percentage  REAL DEFAULT 0,
                    bet_won         INTEGER,
                    model_probability REAL DEFAULT 0,
                    edge_at_bet_time REAL DEFAULT 0,
                    bet_timestamp   TEXT NOT NULL,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(game_id, market_type, bet_side)
                );

                CREATE INDEX IF NOT EXISTS idx_clv_game_id ON clv_tracking(game_id);
                CREATE INDEX IF NOT EXISTS idx_clv_game_date ON clv_tracking(game_date);
                CREATE INDEX IF NOT EXISTS idx_clv_won ON clv_tracking(bet_won);
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ═══════════════════════════════════════════════════════════════════
    #  RECORDING
    # ═══════════════════════════════════════════════════════════════════

    def record_bet(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        bet_side: str,
        market_type: str = "moneyline",
        bet_odds_american: Optional[float] = None,
        closing_odds_american: Optional[float] = None,
        model_probability: float = 0.0,
        edge_at_bet_time: float = 0.0,
        game_date: Optional[str] = None,
        bet_timestamp: Optional[str] = None,
    ) -> CLVRecord:
        """
        Record a bet and compute CLV.

        Args:
            game_id: Game identifier
            home_team, away_team: Team names
            bet_side: 'home', 'away', 'over', 'under'
            market_type: 'moneyline', 'spread', 'total'
            bet_odds_american: The odds you bet at (American format)
            closing_odds_american: The closing line odds (American format)
            model_probability: Model's estimated probability at bet time
            edge_at_bet_time: Edge percentage at bet time
            game_date: Date of the game (auto if None)
            bet_timestamp: When the bet was placed (auto if None)

        Returns:
            CLVRecord with computed CLV values
        """
        if bet_odds_american is None or closing_odds_american is None:
            raise ValueError("bet_odds_american and closing_odds_american are required")

        if game_date is None:
            game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if bet_timestamp is None:
            bet_timestamp = datetime.now(timezone.utc).isoformat()

        clv_diff, clv_pct = self._compute_clv(bet_odds_american, closing_odds_american)

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO clv_tracking
                   (game_id, home_team, away_team, game_date,
                    market_type, bet_side,
                    bet_odds_american, closing_odds_american,
                    clv_difference, clv_percentage,
                    model_probability, edge_at_bet_time, bet_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (game_id, home_team, away_team, game_date,
                 market_type, bet_side,
                 bet_odds_american, closing_odds_american,
                 clv_diff, clv_pct,
                 model_probability, edge_at_bet_time, bet_timestamp)
            )
            conn.commit()

            # Get the auto-generated ID
            record_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        return CLVRecord(
            id=record_id,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            market_type=market_type,
            bet_side=bet_side,
            bet_odds_american=bet_odds_american,
            closing_odds_american=closing_odds_american,
            clv_difference=clv_diff,
            clv_percentage=clv_pct,
            model_probability=model_probability,
            edge_at_bet_time=edge_at_bet_time,
            bet_timestamp=bet_timestamp,
        )

    def _compute_clv(self, bet_odds: float, closing_odds: float) -> Tuple[float, float]:
        """
        Compute CLV difference and percentage.

        CLV measures how much better your line was vs the closing line.

        For negative odds (favorites):
            CLV% = (closing_odds - bet_odds) / abs(bet_odds)
            A positive CLV means you got better odds than the closing line.

        For positive odds (underdogs):
            CLV% = (bet_odds - closing_odds) / closing_odds
            A positive CLV means you got better odds than the closing line.

        Args:
            bet_odds: The odds you bet at (American)
            closing_odds: The closing line (American)

        Returns:
            (clv_difference, clv_percentage)
        """
        # Handle edge cases
        if closing_odds == 0 or bet_odds == 0:
            return (0.0, 0.0)

        clv_diff = bet_odds - closing_odds

        # CLV percentage depends on sign
        if bet_odds < 0 and closing_odds < 0:
            # Both negative (favorites)
            clv_pct = (closing_odds - bet_odds) / abs(bet_odds)
        elif bet_odds > 0 and closing_odds > 0:
            # Both positive (underdogs)
            clv_pct = (bet_odds - closing_odds) / closing_odds
        elif bet_odds < 0 < closing_odds:
            # Bet favorite, closed underdog
            clv_pct = abs(closing_odds - bet_odds) / abs(bet_odds)
            if bet_odds < closing_odds:
                clv_pct = -clv_pct
        else:
            # Bet underdog, closed favorite
            clv_pct = abs(closing_odds - bet_odds) / max(abs(closing_odds), 1)
            if bet_odds > closing_odds:
                clv_pct = -clv_pct

        return (round(clv_diff, 2), round(clv_pct, 6))

    # ═══════════════════════════════════════════════════════════════════
    #  OUTCOME RECORDING
    # ═══════════════════════════════════════════════════════════════════

    def record_outcome(self, game_id: str, market_type: str, bet_side: str, won: bool):
        """Record whether a tracked bet won or lost."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE clv_tracking SET bet_won = ? WHERE game_id = ? AND market_type = ? AND bet_side = ?",
                (1 if won else 0, game_id, market_type, bet_side)
            )
            conn.commit()

    # ═══════════════════════════════════════════════════════════════════
    #  RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════

    def get_bet(self, game_id: str, market_type: str = "moneyline",
                bet_side: str = "home") -> Optional[CLVRecord]:
        """Get a specific CLV record."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM clv_tracking
                   WHERE game_id = ? AND market_type = ? AND bet_side = ?""",
                (game_id, market_type, bet_side)
            ).fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_all_bets(self, limit: int = 100, offset: int = 0) -> List[CLVRecord]:
        """Get all tracked bets."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM clv_tracking ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_recent_bets(self, days: int = 7) -> List[CLVRecord]:
        """Get bets from the last N days."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM clv_tracking
                   WHERE created_at >= datetime('now', ?)
                   ORDER BY created_at DESC""",
                (f"-{days} days",)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row) -> CLVRecord:
        return CLVRecord(
            id=row["id"],
            game_id=row["game_id"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            game_date=row["game_date"],
            market_type=row["market_type"],
            bet_side=row["bet_side"],
            bet_odds_american=row["bet_odds_american"],
            closing_odds_american=row["closing_odds_american"],
            clv_difference=row["clv_difference"],
            clv_percentage=row["clv_percentage"],
            bet_won=bool(row["bet_won"]) if row["bet_won"] is not None else None,
            model_probability=row["model_probability"],
            edge_at_bet_time=row["edge_at_bet_time"],
            bet_timestamp=row["bet_timestamp"],
            created_at=row["created_at"],
        )

    # ═══════════════════════════════════════════════════════════════════
    #  AGGREGATE STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def get_aggregate_stats(self) -> CLVAggregate:
        """Get aggregate CLV statistics."""
        with self._connect() as conn:
            # Basic counts
            total = conn.execute("SELECT COUNT(*) FROM clv_tracking").fetchone()[0]
            won = conn.execute("SELECT COUNT(*) FROM clv_tracking WHERE bet_won = 1").fetchone()[0]
            lost = conn.execute("SELECT COUNT(*) FROM clv_tracking WHERE bet_won = 0").fetchone()[0]

            # CLV stats
            avg_clv = conn.execute(
                "SELECT AVG(clv_percentage) FROM clv_tracking"
            ).fetchone()[0] or 0.0

            pos_clv = conn.execute(
                "SELECT COUNT(*) FROM clv_tracking WHERE clv_percentage > 0"
            ).fetchone()[0]

            # Median
            rows = conn.execute(
                "SELECT clv_percentage FROM clv_tracking ORDER BY clv_percentage"
            ).fetchall()
            clv_values = [r["clv_percentage"] for r in rows]

            # By market
            ml_clv = conn.execute(
                "SELECT AVG(clv_percentage) FROM clv_tracking WHERE market_type = 'moneyline'"
            ).fetchone()[0] or 0.0

            spread_clv = conn.execute(
                "SELECT AVG(clv_percentage) FROM clv_tracking WHERE market_type = 'spread'"
            ).fetchone()[0] or 0.0

            total_clv = conn.execute(
                "SELECT AVG(clv_percentage) FROM clv_tracking WHERE market_type = 'total'"
            ).fetchone()[0] or 0.0

            # Best/worst
            best = conn.execute(
                "SELECT MAX(clv_percentage) FROM clv_tracking"
            ).fetchone()[0] or 0.0
            worst = conn.execute(
                "SELECT MIN(clv_percentage) FROM clv_tracking"
            ).fetchone()[0] or 0.0

            # ROI (if we had stake info)
            total_staked = conn.execute(
                "SELECT SUM(edge_at_bet_time * 1000) FROM clv_tracking"
            ).fetchone()[0] or 0.0  # rough estimate

        pos_rate = pos_clv / total if total > 0 else 0.0
        median_clv = _median_list(clv_values) if clv_values else 0.0

        return CLVAggregate(
            total_bets=total,
            total_won=won,
            total_lost=lost,
            average_clv_pct=avg_clv,
            positive_clv_rate=pos_rate,
            median_clv_pct=median_clv,
            ml_clv=ml_clv,
            spread_clv=spread_clv,
            total_clv=total_clv,
            best_clv=best,
            worst_clv=worst,
        )

    def get_clv_by_date(self, days: int = 30) -> List[Dict]:
        """Get daily CLV averages for charting."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT game_date,
                          COUNT(*) as n_bets,
                          AVG(clv_percentage) as avg_clv,
                          SUM(CASE WHEN clv_percentage > 0 THEN 1 ELSE 0 END) as positive_count
                   FROM clv_tracking
                   WHERE created_at >= datetime('now', ?)
                   GROUP BY game_date
                   ORDER BY game_date""",
                (f"-{days} days",)
            ).fetchall()
        return [dict(r) for r in rows]

    def format_stats(self) -> str:
        """Format aggregate stats for display."""
        stats = self.get_aggregate_stats()
        return (
            f"📊 CLV STATISTICS\n"
            f"{'─' * 40}\n"
            f"Total bets tracked: {stats.total_bets}\n"
            f"Won/Lost: {stats.total_won}/{stats.total_lost}\n"
            f"\n"
            f"Average CLV:      {stats.average_clv_pct:+.4%}\n"
            f"Median CLV:       {stats.median_clv_pct:+.4%}\n"
            f"Positive CLV rate: {stats.positive_clv_rate:.1%}\n"
            f"Best CLV:         {stats.best_clv:+.4%}\n"
            f"Worst CLV:        {stats.worst_clv:+.4%}\n"
            f"\n"
            f"By Market:\n"
            f"  Moneyline: {stats.ml_clv:+.4%}\n"
            f"  Spread:    {stats.spread_clv:+.4%}\n"
            f"  Total:     {stats.total_clv:+.4%}\n"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def _median_list(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
