"""
Odds Ingestion Engine — Phase 1.1 of the Professional Betting Intelligence Platform.

Stores time-series odds snapshots from TheOddsAPI into the database.
Tracks opening odds (first seen), current odds (latest), and closing odds (last before game start).

Schema (from spec):
    odds(
        game_id,
        sportsbook,
        market,
        odds,
        timestamp
    )

Output:
    get_latest_odds(game_id)  -> current market data
    get_closing_odds(game_id) -> final odds before game starts
    get_opening_odds(game_id) -> first odds ever recorded
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class OddsSummary:
    """Human-readable summary of odds for a game at a point in time."""
    __slots__ = (
        "game_id", "home_team", "away_team", "game_date",
        "sportsbooks_available", "home_moneyline", "away_moneyline",
        "implied_home_win_prob", "total_over_under", "home_spread",
        "timestamp", "is_opening", "is_closing",
    )

    def __init__(self, game_id: str, home_team: str, away_team: str,
                 game_date: str, sportsbooks_available: int,
                 timestamp: str = "", is_opening: bool = False,
                 is_closing: bool = False):
        self.game_id = game_id
        self.home_team = home_team
        self.away_team = away_team
        self.game_date = game_date
        self.sportsbooks_available = sportsbooks_available
        self.timestamp = timestamp
        self.is_opening = is_opening
        self.is_closing = is_closing
        self.home_moneyline: Optional[float] = None
        self.away_moneyline: Optional[float] = None
        self.implied_home_win_prob: Optional[float] = None
        self.total_over_under: Optional[float] = None
        self.home_spread: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
#  ODDS INGESTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class OddsIngestionEngine:
    """
    Ingests, stores, and retrieves historical odds data.

    Design:
    - Each call to ingest_snapshot() stores a complete snapshot of all
      sportsbook odds for all upcoming games at that moment.
    - The FIRST snapshot for a game = opening odds.
    - The LATEST snapshot for a game (before it starts) = current odds.
    - The LAST snapshot before game commencement = closing odds.
    - Retrieval methods provide both raw records and computed summaries.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Create the odds schema if it doesn't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS odds (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id         TEXT NOT NULL,
                    api_game_id     TEXT DEFAULT '',
                    home_team       TEXT DEFAULT '',
                    away_team       TEXT DEFAULT '',
                    sportsbook      TEXT NOT NULL,
                    market          TEXT NOT NULL,
                    odds_value      TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    game_date       TEXT DEFAULT '',
                    odds_type       TEXT DEFAULT 'current',
                    UNIQUE(game_id, sportsbook, market, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_odds_game_id ON odds(game_id);
                CREATE INDEX IF NOT EXISTS idx_odds_timestamp ON odds(timestamp);
                CREATE INDEX IF NOT EXISTS idx_odds_game_date ON odds(game_date);
                CREATE INDEX IF NOT EXISTS idx_odds_type ON odds(odds_type);
                CREATE INDEX IF NOT EXISTS idx_odds_sportsbook ON odds(sportsbook);

                CREATE TABLE IF NOT EXISTS odds_meta (
                    game_id         TEXT PRIMARY KEY,
                    home_team       TEXT NOT NULL,
                    away_team       TEXT NOT NULL,
                    commence_time   TEXT NOT NULL,
                    first_seen      TEXT NOT NULL,
                    last_updated    TEXT NOT NULL,
                    is_finished     INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_odds_meta_finished ON odds_meta(is_finished);
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the database."""
        conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ═══════════════════════════════════════════════════════════════════
    #  INGESTION — batched in a single connection
    # ═══════════════════════════════════════════════════════════════════

    def ingest_snapshot(self, games: List[Any], source: str = "the_odds_api") -> int:
        """
        Store a complete odds snapshot for all games using a single DB connection.

        Args:
            games: List of OddsGame objects (or dict-like objects with
                   home_team, away_team, all_books, commence_time, id, etc.)
            source: Source identifier for logging

        Returns:
            Number of odds records stored
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        count = 0

        with self._connect() as conn:
            for game in games:
                game_id = self._get_game_id(game)
                home = self._get_home_team(game)
                away = self._get_away_team(game)
                commence = self._get_commence_time(game)
                game_date = commence[:10] if commence else now_iso[:10]

                # Upsert metadata within the same connection
                self._upsert_meta_conn(conn, game_id, home, away, commence, now_iso)

                books = self._get_books(game)
                if not books:
                    continue

                for book in books:
                    book_key = self._get_book_key(book)
                    markets = self._get_markets(book)

                    for market in markets:
                        market_key = market.get("key", "unknown")
                        outcomes = market.get("outcomes", [])
                        odds_value = json.dumps(outcomes)

                        try:
                            conn.execute(
                                """INSERT OR IGNORE INTO odds
                                   (game_id, api_game_id, home_team, away_team,
                                    sportsbook, market, odds_value, timestamp,
                                    game_date, odds_type)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'current')""",
                                (game_id, game_id, home, away,
                                 book_key, market_key, odds_value, now_iso,
                                 game_date)
                            )
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to store odds: {e}")

            conn.commit()

        return count

    def _upsert_meta_conn(self, conn: sqlite3.Connection, game_id: str,
                          home: str, away: str, commence: str, now_iso: str):
        """Upsert game metadata within an existing connection."""
        try:
            existing = conn.execute(
                "SELECT first_seen FROM odds_meta WHERE game_id = ?",
                (game_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE odds_meta SET last_updated = ? WHERE game_id = ?",
                    (now_iso, game_id)
                )
            else:
                conn.execute(
                    """INSERT INTO odds_meta
                       (game_id, home_team, away_team, commence_time,
                        first_seen, last_updated, is_finished)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    (game_id, home, away, commence, now_iso, now_iso)
                )
        except Exception as e:
            logger.warning(f"Failed to upsert meta: {e}")

    def _get_game_id(self, game) -> str:
        if hasattr(game, "id"):
            return game.id
        if isinstance(game, dict):
            return game.get("id", "")
        return ""

    def _get_home_team(self, game) -> str:
        if hasattr(game, "home_team"):
            return game.home_team
        if isinstance(game, dict):
            return game.get("home_team", "")
        return ""

    def _get_away_team(self, game) -> str:
        if hasattr(game, "away_team"):
            return game.away_team
        if isinstance(game, dict):
            return game.get("away_team", "")
        return ""

    def _get_commence_time(self, game) -> str:
        if hasattr(game, "commence_time"):
            return game.commence_time
        if isinstance(game, dict):
            return game.get("commence_time", "")
        return ""

    def _get_books(self, game) -> list:
        if hasattr(game, "all_books"):
            return game.all_books
        if isinstance(game, dict):
            return game.get("bookmakers", game.get("all_books", []))
        return []

    def _get_book_key(self, book) -> str:
        if hasattr(book, "book_key"):
            return book.book_key
        if isinstance(book, dict):
            return book.get("key", book.get("book_key", "unknown"))
        return "unknown"

    def _get_markets(self, book) -> list:
        if hasattr(book, "markets"):
            return book.markets
        if isinstance(book, dict):
            return book.get("markets", [])
        return self._book_odds_to_markets(book)

    def _book_odds_to_markets(self, book) -> list:
        """Convert a BookOdds dataclass into market dicts for storage."""
        markets = []

        h2h_outcomes = []
        if hasattr(book, "home_moneyline") and book.home_moneyline is not None:
            h2h_outcomes.append({"name": "home", "price": book.home_moneyline})
        if hasattr(book, "away_moneyline") and book.away_moneyline is not None:
            h2h_outcomes.append({"name": "away", "price": book.away_moneyline})
        if h2h_outcomes:
            markets.append({"key": "h2h", "outcomes": h2h_outcomes})

        spread_outcomes = []
        if hasattr(book, "home_spread") and book.home_spread is not None:
            spread_outcomes.append({
                "name": "home",
                "point": book.home_spread,
                "price": getattr(book, "home_spread_odds", None)
            })
        if hasattr(book, "away_spread") and book.away_spread is not None:
            spread_outcomes.append({
                "name": "away",
                "point": book.away_spread,
                "price": getattr(book, "away_spread_odds", None)
            })
        if spread_outcomes:
            markets.append({"key": "spreads", "outcomes": spread_outcomes})

        total_outcomes = []
        if hasattr(book, "total_over") and book.total_over is not None:
            total_outcomes.append({
                "name": "Over",
                "point": book.total_over,
                "price": getattr(book, "total_over_odds", None)
            })
        if hasattr(book, "total_under") and book.total_under is not None:
            total_outcomes.append({
                "name": "Under",
                "point": book.total_under,
                "price": getattr(book, "total_under_odds", None)
            })
        if total_outcomes:
            markets.append({"key": "totals", "outcomes": total_outcomes})

        return markets

    def _upsert_meta(self, game_id: str, home: str, away: str,
                     commence: str, now_iso: str):
        """Upsert using its own connection (for standalone calls)."""
        with self._connect() as conn:
            self._upsert_meta_conn(conn, game_id, home, away, commence, now_iso)
            conn.commit()

    # ═══════════════════════════════════════════════════════════════════
    #  RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════

    def get_latest_odds(self, game_id: str) -> Optional[OddsSummary]:
        """Get the most recent odds for a game (CURRENT market)."""
        return self._build_summary(game_id, "current")

    def get_opening_odds(self, game_id: str) -> Optional[OddsSummary]:
        """Get the FIRST odds recorded for a game (OPENING line)."""
        return self._build_summary(game_id, "opening")

    def get_closing_odds(self, game_id: str) -> Optional[OddsSummary]:
        """Get the LAST odds before game start (CLOSING line)."""
        return self._build_summary(game_id, "closing")

    def _build_summary(self, game_id: str,
                       odds_type: str = "current") -> Optional[OddsSummary]:
        """Build an OddsSummary by fetching records at the right timestamp."""
        meta = None
        records = []

        with self._connect() as conn:
            meta = conn.execute(
                "SELECT * FROM odds_meta WHERE game_id = ?", (game_id,)
            ).fetchone()

            if not meta:
                # Fallback: try api_game_id lookup
                row = conn.execute(
                    "SELECT DISTINCT game_id FROM odds WHERE api_game_id = ? LIMIT 1",
                    (game_id,)
                ).fetchone()
                if row:
                    meta = conn.execute(
                        "SELECT * FROM odds_meta WHERE game_id = ?",
                        (row["game_id"],)
                    ).fetchone()

            if not meta:
                return None

            if odds_type == "opening":
                target_ts = meta["first_seen"]
                records = conn.execute(
                    "SELECT * FROM odds WHERE game_id = ? AND timestamp = ? LIMIT 50",
                    (game_id, target_ts)
                ).fetchall()
            elif odds_type == "closing":
                target_ts = meta["last_updated"]
                records = conn.execute(
                    "SELECT * FROM odds WHERE game_id = ? AND timestamp = ? LIMIT 50",
                    (game_id, target_ts)
                ).fetchall()
            else:  # current - just get latest
                records = conn.execute(
                    "SELECT * FROM odds WHERE game_id = ? ORDER BY timestamp DESC LIMIT 50",
                    (game_id,)
                ).fetchall()

        if not records:
            return None

        return self._records_to_summary(records, dict(meta), odds_type)

    def _records_to_summary(self, records: list, meta: dict,
                            odds_type: str) -> OddsSummary:
        """Convert raw odds records into an OddsSummary."""
        summary = OddsSummary(
            game_id=meta["game_id"],
            home_team=meta["home_team"],
            away_team=meta["away_team"],
            game_date=meta.get("commence_time", "")[:10] if meta.get("commence_time") else "",
            sportsbooks_available=len(set(r["sportsbook"] for r in records)),
            timestamp=records[0]["timestamp"],
            is_opening=(odds_type == "opening"),
            is_closing=(odds_type == "closing"),
        )

        home_mls = []
        away_mls = []
        totals = []
        spreads = []

        for r in records:
            try:
                outcomes = json.loads(r["odds_value"])
            except (json.JSONDecodeError, TypeError):
                continue

            market = r["market"]

            if market == "h2h":
                for o in outcomes:
                    name = o.get("name", "")
                    price = o.get("price")
                    if price is not None:
                        if "home" in name.lower() or name == r.get("home_team", ""):
                            home_mls.append(float(price))
                        elif "away" in name.lower() or name == r.get("away_team", ""):
                            away_mls.append(float(price))

            elif market == "totals":
                for o in outcomes:
                    point = o.get("point")
                    if point is not None:
                        totals.append(float(point))

            elif market == "spreads":
                for o in outcomes:
                    point = o.get("point")
                    if point is not None and "home" in o.get("name", "").lower():
                        spreads.append(float(point))

        if home_mls:
            summary.home_moneyline = _median(home_mls)
        if away_mls:
            summary.away_moneyline = _median(away_mls)
        if totals:
            summary.total_over_under = _median(totals)
        if spreads:
            summary.home_spread = _median(spreads)

        if summary.home_moneyline and summary.away_moneyline:
            summary.implied_home_win_prob = _implied_prob(
                summary.home_moneyline, summary.away_moneyline
            )

        return summary

    # ═══════════════════════════════════════════════════════════════════
    #  BULK RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════

    def get_all_current_games(self) -> List[OddsSummary]:
        """Get current odds for all tracked games that haven't started yet."""
        with self._connect() as conn:
            active = conn.execute(
                "SELECT game_id FROM odds_meta WHERE is_finished = 0 ORDER BY commence_time ASC"
            ).fetchall()

        results = []
        for row in active:
            summary = self.get_latest_odds(row["game_id"])
            if summary:
                results.append(summary)
        return results

    def get_all_closing_games(self) -> List[OddsSummary]:
        """Get closing odds for all finished games."""
        with self._connect() as conn:
            finished = conn.execute(
                "SELECT game_id FROM odds_meta WHERE is_finished = 1 ORDER BY commence_time DESC"
            ).fetchall()

        results = []
        for row in finished:
            summary = self.get_closing_odds(row["game_id"])
            if summary:
                results.append(summary)
        return results

    def get_odds_history(self, game_id: str,
                         market: Optional[str] = None,
                         sportsbook: Optional[str] = None,
                         limit: int = 100) -> List[dict]:
        """Get the full time-series history of odds for a game (newest first)."""
        query = "SELECT * FROM odds WHERE game_id = ?"
        params: list[Any] = [game_id]

        if market:
            query += " AND market = ?"
            params.append(market)
        if sportsbook:
            query += " AND sportsbook = ?"
            params.append(sportsbook)

        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    # ═══════════════════════════════════════════════════════════════════
    #  MAINTENANCE
    # ═══════════════════════════════════════════════════════════════════

    def mark_game_finished(self, game_id: str):
        """Mark a game as finished. Last snapshot = closing line."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE odds_meta SET is_finished = 1, last_updated = ? WHERE game_id = ?",
                (now_iso, game_id)
            )
            conn.commit()

    def auto_mark_finished_games(
        self,
        game_duration_hours: Optional[float] = None,
        sport: Optional[str] = None,
    ) -> int:
        """Mark games as finished once their scheduled end time has passed.

        Accepts either a raw duration in hours or a sport abbreviation.
        If both are provided, sport takes precedence. If neither is provided,
        defaults to 3.0 hours (NBA).

        Args:
            game_duration_hours: Hours to add to commence_time before
                marking the game as finished. Optional if sport is given.
            sport: Sport abbreviation (e.g., "NBA" → 3h, "NFL" → 3.5h,
                   "MLB" → 3h, "NHL" → 2.5h). Case-insensitive.
                   Overrides game_duration_hours if both are provided.

        Returns:
            Number of games marked as finished.

        Raises:
            KeyError: If an unknown sport abbreviation is provided.
        """
        # Resolve duration: sport takes precedence over raw hours
        if sport is not None:
            duration = get_sport_duration(sport)
        elif game_duration_hours is not None:
            duration = game_duration_hours
        else:
            duration = 3.0  # Default: NBA

        now = datetime.now(timezone.utc)
        count = 0

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT game_id, commence_time FROM odds_meta WHERE is_finished = 0"
            ).fetchall()

            for row in rows:
                try:
                    commence = datetime.fromisoformat(row["commence_time"])
                    if commence.tzinfo is None:
                        commence = commence.replace(tzinfo=timezone.utc)
                    end_time = commence + timedelta(hours=duration)

                    if now >= end_time:
                        conn.execute(
                            "UPDATE odds_meta SET is_finished = 1 WHERE game_id = ?",
                            (row["game_id"],)
                        )
                        count += 1
                except (ValueError, TypeError):
                    continue

            conn.commit()

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        with self._connect() as conn:
            return {
                "total_odds_records": conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0],
                "total_games_tracked": conn.execute("SELECT COUNT(*) FROM odds_meta").fetchone()[0],
                "active_games": conn.execute("SELECT COUNT(*) FROM odds_meta WHERE is_finished = 0").fetchone()[0],
                "total_snapshots": conn.execute("SELECT COUNT(DISTINCT timestamp) FROM odds").fetchone()[0],
                "unique_sportsbooks": conn.execute("SELECT COUNT(DISTINCT sportsbook) FROM odds").fetchone()[0],
            }


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _median(values: List[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def _implied_prob(home_ml: float, away_ml: float) -> float:
    def ml_to_prob(odds: float) -> float:
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    home_p = ml_to_prob(home_ml)
    away_p = ml_to_prob(away_ml)
    total = home_p + away_p
    return home_p / total if total > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  SPORT DURATION LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

SPORT_DURATIONS: Dict[str, float] = {
    "NBA": 3.0,    # ~48 min play + halftime + clock stops
    "NFL": 3.5,    # ~60 min play + halftime + commercials
    "MLB": 3.0,    # ~9 innings, no clock
    "NHL": 2.5,    # ~60 min play + intermissions
    "WNBA": 2.25,  # ~40 min play
    "CFB": 3.5,    # College football, similar to NFL
    "UFC": 2.0,    # ~3 x 5 min rounds + buffer
    "MLS": 2.25,   # ~90 min play + halftime (Soccer)
    "EPL": 2.25,   # Same as MLS
    "LIGA": 2.25,  # La Liga
    "SERIE_A": 2.25,
    "BUNDESLIGA": 2.25,
}


def get_sport_duration(sport: str) -> float:
    """Look up the typical game duration for a sport.

    Args:
        sport: Sport abbreviation (e.g., "NBA", "NFL", "MLB", "NHL").
               Case-insensitive.

    Returns:
        Duration in hours.

    Raises:
        KeyError: If the sport is not in the lookup table.
    """
    key = sport.upper().strip()
    if key in SPORT_DURATIONS:
        return SPORT_DURATIONS[key]
    raise KeyError(
        f"Unknown sport '{sport}'. Supported sports: "
        f"{', '.join(sorted(SPORT_DURATIONS.keys()))}"
    )


def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    return 1.0 + (100.0 / abs(american_odds))


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return float("nan")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def format_american(odds: Optional[float]) -> str:
    if odds is None:
        return "N/A"
    if odds > 0:
        return f"+{odds:.0f}"
    return f"{odds:.0f}"
