"""
BetJournal — persistent journal of all predictions, bets, and outcomes.

Provides a cumulative SQLite-backed history that enables:
  - Historical P&L tracking across multiple days
  - PerformanceTracker integration for drift detection
  - Backtesting dashboard data source
  - Model version comparison

Schema:
  bets_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    game_date TEXT NOT NULL,
    game_id TEXT,
    home_team TEXT,
    away_team TEXT,
    bet_type TEXT,
    side TEXT,
    model_prediction REAL,
    market_line REAL,
    edge_pct REAL,
    decimal_odds REAL,
    stake_dollars REAL,
    outcome TEXT DEFAULT 'PENDING',
    profit_loss REAL,
    model_version TEXT,
    strategy TEXT,
    league TEXT,
    confidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BetJournal:
    """
    Persistent journal for all betting predictions and outcomes.

    Writes to a SQLite database, creating the bets_journal table
    if it doesn't exist. Designed to be the single source of truth
    for historical betting activity.

    Usage:
        journal = BetJournal(db_path="./data/bets_journal.db")
        journal.record_bets([...])          # Write open bets
        journal.update_outcomes(results)     # Update after games finish
        history = journal.get_history()      # Retrieve all bets
        perf = journal.get_performance()     # Aggregate metrics
    """

    def __init__(self, db_path: str | Path = "./data/bets_journal.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _ensure_table(self):
        """Create the bets_journal table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bets_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    game_id TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    bet_type TEXT,
                    side TEXT,
                    model_prediction REAL,
                    market_line REAL,
                    edge_pct REAL,
                    decimal_odds REAL,
                    stake_dollars REAL,
                    outcome TEXT DEFAULT 'PENDING',
                    profit_loss REAL,
                    model_version TEXT,
                    strategy TEXT,
                    league TEXT,
                    confidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bets_journal_run_date
                ON bets_journal(run_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bets_journal_game_id
                ON bets_journal(game_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bets_journal_outcome
                ON bets_journal(outcome)
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the journal database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def record_bets(self, bets: List[Dict[str, Any]]):
        """Record a batch of open bets (predictions).

        Args:
            bets: List of dicts with keys matching the schema:
                  run_date, game_date, game_id, home_team, away_team,
                  bet_type, side, model_prediction, market_line, edge_pct,
                  decimal_odds, stake_dollars, model_version, strategy, league, confidence
        """
        if not bets:
            return

        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            for bet in bets:
                conn.execute(
                    """
                    INSERT INTO bets_journal (
                        run_date, game_date, game_id, home_team, away_team,
                        bet_type, side, model_prediction, market_line, edge_pct,
                        decimal_odds, stake_dollars, outcome, model_version,
                        strategy, league, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
                    """,
                    (
                        run_date,
                        str(bet.get("game_date", "")),
                        str(bet.get("game_id", "")),
                        str(bet.get("home_team", bet.get("team", ""))),
                        str(bet.get("away_team", "")),
                        str(bet.get("bet_type", "")),
                        str(bet.get("side", "")),
                        bet.get("model_prediction", bet.get("predicted_total", None)),
                        bet.get("market_line", None),
                        bet.get("edge_pct", None),
                        bet.get("decimal_odds", None),
                        bet.get("stake_dollars", bet.get("stake", None)),
                        str(bet.get("model_version", "")),
                        str(bet.get("strategy", "")),
                        str(bet.get("league", "NBA")),
                        str(bet.get("confidence", "")),
                    ),
                )
            conn.commit()
        logger.info(f"Recorded {len(bets)} bets in journal")

    def update_outcomes(self, results: List[Dict[str, Any]]):
        """Update outcomes for previously recorded PENDING bets.

        Args:
            results: List of dicts with game_id, outcome ('WIN'/'LOSS'/'PUSH'),
                     profit_loss (float), and optionally decimal_odds.
        """
        if not results:
            return

        updated = 0
        with self._connect() as conn:
            for result in results:
                game_id = result.get("game_id")
                outcome = result.get("outcome", "PENDING")
                profit_loss = result.get("profit_loss", result.get("profit_units", 0))

                if not game_id:
                    continue

                conn.execute(
                    """
                    UPDATE bets_journal
                    SET outcome = ?,
                        profit_loss = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE game_id = ? AND outcome = 'PENDING'
                    """,
                    (outcome, profit_loss, game_id),
                )
                updated += conn.total_changes
            conn.commit()
        if updated:
            logger.info(f"Updated {updated} outcomes in journal")

    def update_outcomes_by_date(self, game_date: str, results: List[Dict[str, Any]]):
        """Update outcomes for all PENDING bets on a specific game date."""
        updated = 0
        with self._connect() as conn:
            for result in results:
                outcome = result.get("outcome", "PENDING")
                profit_loss = result.get("profit_loss", result.get("profit_units", 0))

                conn.execute(
                    """
                    UPDATE bets_journal
                    SET outcome = ?,
                        profit_loss = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE game_date = ? AND outcome = 'PENDING'
                    """,
                    (outcome, profit_loss, game_date),
                )
                updated += conn.total_changes
            conn.commit()
        if updated:
            logger.info(f"Updated {updated} outcomes for date {game_date}")

    def get_history(
        self,
        limit: int = 1000,
        offset: int = 0,
        outcome: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve bet history with optional filters."""
        query = "SELECT * FROM bets_journal WHERE 1=1"
        params: List[Any] = []

        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        if start_date:
            query += " AND game_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND game_date <= ?"
            params.append(end_date)

        query += " ORDER BY run_date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_performance(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate performance metrics from journal."""
        query = "SELECT * FROM bets_journal WHERE outcome != 'PENDING'"
        params: List[Any] = []

        if start_date:
            query += " AND game_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND game_date <= ?"
            params.append(end_date)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return {"total_bets": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                    "total_profit": 0.0, "roi": 0.0}

        wins = sum(1 for r in rows if r["outcome"] == "WIN")
        losses = sum(1 for r in rows if r["outcome"] == "LOSS")
        pushes = sum(1 for r in rows if r["outcome"] == "PUSH")
        total_decided = wins + losses

        profit = sum(r["profit_loss"] or 0 for r in rows if r["profit_loss"] is not None)

        return {
            "total_bets": len(rows),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": wins / total_decided if total_decided > 0 else 0.0,
            "total_profit": profit,
            "roi": profit / len(rows) * 100 if len(rows) > 0 else 0.0,
        }

    def get_model_comparison(self) -> List[Dict[str, Any]]:
        """Compare performance across model versions."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    model_version,
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(profit_loss), 0) as total_profit
                FROM bets_journal
                WHERE outcome != 'PENDING' AND model_version != ''
                GROUP BY model_version
                ORDER BY total_profit DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_unresolved(self) -> List[Dict[str, Any]]:
        """Get the most recent unresolved (PENDING) bets."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM bets_journal
                WHERE outcome = 'PENDING'
                ORDER BY game_date DESC
                LIMIT 50
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def wire_into_performance_tracker(
        self,
        performance_tracker: Any,
    ) -> None:
        """Feed journal history into a PerformanceTracker for drift detection."""
        history = self.get_history(limit=5000)
        for record in history:
            if record["outcome"] != "PENDING" and record["model_prediction"] is not None:
                performance_tracker.record_prediction(
                    predicted=record["model_prediction"],
                    actual=record["market_line"] or 0,
                    won=(record["outcome"] == "WIN"),
                    edge_pct=record["edge_pct"] or 0,
                    profit_units=record["profit_loss"] or 0,
                )

    def close(self):
        """No-op for compatibility; connections are context-managed."""
        pass
