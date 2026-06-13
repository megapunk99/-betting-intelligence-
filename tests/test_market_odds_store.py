"""
Unit tests for MarketOddsStore — the persistence layer for historical market odds.

All tests use an in-memory SQLite database so they run fast with zero external
dependencies. The MarketOddsStore accepts an optional DatabaseManager in its
constructor, making this trivial to inject.

Test coverage:
  - log_snapshot: basic, all fields, without odds, duplicate writes
  - log_snapshot_from_live_game: LiveGame object path
  - log_batch: batch of games, filtered by odds availability
  - get_odds_for_date: exact match, no match, latest_only, multiple snapshots
  - get_market_prob_for_game: full name, short name, swapped orientation,
    no match, team name variants (full, short, swapped)
  - get_market_probs_for_date_range: date range, empty range, partial range
  - get_stats: snapshot count, unique games count
  - Edge cases: invalid dates, None values, concurrent snapshots
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from betting_intel.db.connection import DatabaseManager
from betting_intel.db.market_odds_store import MarketOddsStore


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def memory_db() -> DatabaseManager:
    """Create an in-memory SQLite DatabaseManager for testing."""
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    # Create the MarketOdds table
    from betting_intel.db.schema import Base, MarketOdds
    Base.metadata.create_all(manager.engine, tables=[MarketOdds.__table__])
    return manager


@pytest.fixture
def store(memory_db: DatabaseManager) -> MarketOddsStore:
    """MarketOddsStore backed by in-memory SQLite."""
    return MarketOddsStore(db_manager=memory_db)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: Minimal LiveGame stub
# ═══════════════════════════════════════════════════════════════════════════


class _LiveGameStub:
    """Minimal LiveGame-like object for testing log_snapshot_from_live_game."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def sample_live_game() -> _LiveGameStub:
    """A fully populated LiveGame-like object."""
    return _LiveGameStub(
        game_id="test_live_001",
        game_date="2025-01-20",
        home_team="Boston Celtics",
        away_team="Los Angeles Lakers",
        home_team_short="Celtics",
        away_team_short="Lakers",
        home_ml=-150.0,
        away_ml=+130.0,
        spread=-4.5,
        market_total=224.5,
        over_odds=-110.0,
        under_odds=-110.0,
        n_books_ml=5,
        n_books_total=4,
        ml_std=12.5,
        sport_key="basketball_nba",
    )


# ====================================================================
#  SECTION 1: log_snapshot
# ====================================================================


class TestLogSnapshot:
    """Unit tests for the log_snapshot method."""

    def test_basic_snapshot(self, store: MarketOddsStore):
        """Log a snapshot with minimal required fields."""
        ok = store.log_snapshot(
            game_id="GAME_001",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
        )
        assert ok is True

        stats = store.get_stats()
        assert stats["total_snapshots"] == 1

    def test_snapshot_with_all_fields(self, store: MarketOddsStore):
        """Log a snapshot with the full set of fields."""
        ok = store.log_snapshot(
            game_id="GAME_002",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=-150.0,
            away_ml=+130.0,
            spread=-4.5,
            market_total=224.5,
            over_odds=-110.0,
            under_odds=-110.0,
            n_books_ml=5,
            n_books_total=4,
            ml_std=12.5,
            source="test_backfill",
            sport_key="basketball_nba",
        )
        assert ok is True

        # Verify by querying back
        df = store.get_odds_for_date("2025-01-15")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["home_team"] == "Boston Celtics"
        assert row["away_team"] == "Los Angeles Lakers"
        assert row["home_ml"] == -150.0
        assert row["away_ml"] == +130.0
        assert row["spread"] == -4.5
        assert row["market_total"] == 224.5
        assert row["n_books_ml"] == 5
        assert row["captured_at"] is not None

    def test_snapshot_without_odds(self, store: MarketOddsStore):
        """Log a snapshot with no moneyline data (score-only mode)."""
        ok = store.log_snapshot(
            game_id="GAME_003",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_ml=None,
            away_ml=None,
        )
        assert ok is True

        df = store.get_odds_for_date("2025-01-15")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["home_ml"] is None or pd.isna(row["home_ml"])
        assert row["vig_removed_home_prob"] is None or pd.isna(row["vig_removed_home_prob"])

    def test_snapshot_with_home_favorite_odds(self, store: MarketOddsStore):
        """Log a snapshot where home team is favored (-150). Checks vig-free prob."""
        ok = store.log_snapshot(
            game_id="GAME_004",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=-150.0,
            away_ml=+130.0,
        )
        assert ok is True

        df = store.get_odds_for_date("2025-01-15")
        row = df.iloc[0]
        # -150 -> 0.60, +130 -> 0.4348, vig-free = 0.60 / (0.60 + 0.4348) ≈ 0.58
        assert row["home_implied_prob"] is not None
        assert row["vig_removed_home_prob"] is not None
        assert 0.55 < row["vig_removed_home_prob"] < 0.60

    def test_snapshot_with_away_favorite_odds(self, store: MarketOddsStore):
        """Log a snapshot where away team is favored (home is +160 underdog)."""
        ok = store.log_snapshot(
            game_id="GAME_005",
            game_date="2025-01-15",
            home_team="Utah Jazz",
            away_team="Boston Celtics",
            home_team_short="Jazz",
            away_team_short="Celtics",
            home_ml=+160.0,
            away_ml=-190.0,
        )
        assert ok is True

        df = store.get_odds_for_date("2025-01-15")
        row = df.iloc[0]
        # Home underdog: hom_implied = 100/(160+100) ≈ 0.3846
        # away_implied = 190/(190+100) ≈ 0.6552
        # vig-free home = 0.3846 / (0.3846 + 0.6552) ≈ 0.37
        assert row["vig_removed_home_prob"] is not None
        assert 0.35 < row["vig_removed_home_prob"] < 0.40

    def test_duplicate_game_id(self, store: MarketOddsStore):
        """Multiple snapshots for the same game_id are allowed (no unique constraint)."""
        ok1 = store.log_snapshot(
            game_id="GAME_006",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_ml=-150.0,
            away_ml=+130.0,
        )
        ok2 = store.log_snapshot(
            game_id="GAME_006",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_ml=-160.0,  # Line moved
            away_ml=+140.0,
        )
        assert ok1 is True
        assert ok2 is True

        # Both snapshots stored
        df = store.get_odds_for_date("2025-01-15", latest_only=False)
        assert len(df) == 2

        # latest_only=True should return only the most recent
        df_latest = store.get_odds_for_date("2025-01-15", latest_only=True)
        assert len(df_latest) == 1
        assert df_latest.iloc[0]["home_ml"] == -160.0  # Most recent

    def test_snapshot_with_short_name_fallback(self, store: MarketOddsStore):
        """When home_team_short is not provided, it falls back to last word of full name."""
        ok = store.log_snapshot(
            game_id="GAME_007",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="",  # Will fall back to "Celtics"
            away_team_short="",   # Will fall back to "Lakers"
        )
        assert ok is True

        # Verify short names were auto-populated
        from betting_intel.db.schema import MarketOdds
        session = store._db.get_session()
        try:
            record = session.query(MarketOdds).filter_by(game_id="GAME_007").first()
            assert record is not None
            assert record.home_team_short == "Celtics"
            assert record.away_team_short == "Lakers"
        finally:
            session.close()

    def test_snapshot_one_team_single_word_name(self, store: MarketOddsStore):
        """Team names with no space (e.g., 'Heat') get empty string fallback."""
        ok = store.log_snapshot(
            game_id="GAME_008",
            game_date="2025-01-15",
            home_team="Miami Heat",
            away_team="Jazz",  # Single-word name
            home_team_short="",
            away_team_short="",
        )
        assert ok is True

        from betting_intel.db.schema import MarketOdds
        session = store._db.get_session()
        try:
            record = session.query(MarketOdds).filter_by(game_id="GAME_008").first()
            assert record.home_team_short == "Heat"  # Last word of "Miami Heat"
            assert record.away_team_short == "Jazz"  # Single word, no fallback needed
        finally:
            session.close()

    def test_snapshot_empty_game_id(self, store: MarketOddsStore):
        """Empty game_id can still be stored (the column is not nullable)."""
        ok = store.log_snapshot(
            game_id="",
            game_date="2025-01-15",
            home_team="Home Team",
            away_team="Away Team",
        )
        assert ok is True

    def test_snapshot_invalid_date_format(self, store: MarketOddsStore):
        """Date is stored as-is (no validation, just a string column)."""
        ok = store.log_snapshot(
            game_id="GAME_009",
            game_date="not-a-date",
            home_team="Home Team",
            away_team="Away Team",
        )
        assert ok is True
        df = store.get_odds_for_date("not-a-date")
        assert len(df) == 1

    def test_snapshot_many_games(self, store: MarketOddsStore):
        """Log 100 snapshots to ensure performance is reasonable."""
        for i in range(100):
            ok = store.log_snapshot(
                game_id=f"GAME_{i:04d}",
                game_date="2025-01-15",
                home_team=f"Home Team {i}",
                away_team=f"Away Team {i}",
            )
            assert ok is True

        stats = store.get_stats()
        assert stats["total_snapshots"] == 100


# ====================================================================
#  SECTION 2: log_snapshot_from_live_game
# ====================================================================


class TestLogSnapshotFromLiveGame:
    """Unit tests for log_snapshot_from_live_game."""

    def test_from_live_game_basic(self, store: MarketOddsStore, sample_live_game: _LiveGameStub):
        """Log a snapshot from a LiveGame-like object."""
        ok = store.log_snapshot_from_live_game(sample_live_game)
        assert ok is True

        stats = store.get_stats()
        assert stats["total_snapshots"] == 1

    def test_from_live_game_all_fields_persisted(self, store: MarketOddsStore, sample_live_game: _LiveGameStub):
        """All fields from the LiveGame object are correctly persisted."""
        store.log_snapshot_from_live_game(sample_live_game)

        df = store.get_odds_for_date("2025-01-20")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["home_team"] == "Boston Celtics"
        assert row["away_team"] == "Los Angeles Lakers"
        assert row["home_ml"] == -150.0
        assert row["away_ml"] == +130.0
        assert row["spread"] == -4.5
        assert row["market_total"] == 224.5

    def test_from_live_game_no_odds(self, store: MarketOddsStore):
        """LiveGame with no moneyline data still logs the snapshot."""
        game = _LiveGameStub(
            game_id="no_odds_001",
            game_date="2025-01-20",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=None,
            away_ml=None,
            spread=None,
            market_total=None,
            over_odds=None,
            under_odds=None,
            n_books_ml=0,
            n_books_total=0,
            ml_std=None,
            sport_key="basketball_nba",
        )
        ok = store.log_snapshot_from_live_game(game)
        assert ok is True

    def test_from_live_game_multiple_teams(self, store: MarketOddsStore):
        """Multiple LiveGames on the same date store separate records."""
        games = [
            _LiveGameStub(
                game_id=f"multi_{i}",
                game_date="2025-01-20",
                home_team=f"Home Team {i}",
                away_team=f"Away Team {i}",
                home_team_short=f"Home{i}",
                away_team_short=f"Away{i}",
                home_ml=-110.0,
                away_ml=-110.0,
                spread=0.0,
                market_total=220.0,
                over_odds=-110.0,
                under_odds=-110.0,
                n_books_ml=3,
                n_books_total=3,
                ml_std=5.0,
                sport_key="basketball_nba",
            )
            for i in range(5)
        ]
        for g in games:
            assert store.log_snapshot_from_live_game(g) is True

        df = store.get_odds_for_date("2025-01-20")
        assert len(df) == 5


# ====================================================================
#  SECTION 3: log_batch
# ====================================================================


class TestLogBatch:
    """Unit tests for log_batch."""

    def test_batch_all_with_odds(self, store: MarketOddsStore):
        """All games in batch have odds — all logged."""
        games = [
            _LiveGameStub(
                game_id=f"batch_{i}",
                game_date="2025-01-25",
                home_team="Home Team",
                away_team="Away Team",
                home_team_short="Home",
                away_team_short="Away",
                home_ml=-110.0 + i,
                away_ml=-110.0 - i,
                spread=0.0,
                market_total=220.0,
                over_odds=-110.0,
                under_odds=-110.0,
                n_books_ml=3,
                n_books_total=3,
                ml_std=5.0,
                sport_key="basketball_nba",
            )
            for i in range(10)
        ]
        count = store.log_batch(games)
        assert count == 10
        assert store.get_stats()["total_snapshots"] == 10

    def test_batch_some_without_odds(self, store: MarketOddsStore):
        """Games without odds are filtered out by log_batch's condition."""
        games = [
            _LiveGameStub(
                game_id=f"no_odds_{i}",
                game_date="2025-01-25",
                home_team="Home Team",
                away_team="Away Team",
                home_team_short="Home",
                away_team_short="Away",
                home_ml=None,
                away_ml=None,
                spread=None,
                market_total=None,
                over_odds=None,
                under_odds=None,
                n_books_ml=0,
                n_books_total=0,
                ml_std=None,
                sport_key="basketball_nba",
            )
            for i in range(5)
        ]
        # log_batch only logs games where home_ml or market_total is not None
        count = store.log_batch(games)
        assert count == 0  # None have odds

    def test_batch_mixed_odds(self, store: MarketOddsStore):
        """Mix of games with and without odds — only those with odds are logged."""
        games = []
        for i in range(5):
            games.append(_LiveGameStub(
                game_id=f"with_odds_{i}",
                game_date="2025-01-25",
                home_team="Home Team",
                away_team="Away Team",
                home_team_short="Home",
                away_team_short="Away",
                home_ml=-110.0,
                away_ml=-110.0,
                spread=0.0,
                market_total=220.0,
                over_odds=-110.0,
                under_odds=-110.0,
                n_books_ml=3,
                n_books_total=3,
                ml_std=5.0,
                sport_key="basketball_nba",
            ))
        for i in range(5):
            games.append(_LiveGameStub(
                game_id=f"no_odds_{i}",
                game_date="2025-01-25",
                home_team="Home Team",
                away_team="Away Team",
                home_team_short="Home",
                away_team_short="Away",
                home_ml=None,
                away_ml=None,
                spread=None,
                market_total=None,
                over_odds=None,
                under_odds=None,
                n_books_ml=0,
                n_books_total=0,
                ml_std=None,
                sport_key="basketball_nba",
            ))

        count = store.log_batch(games)
        assert count == 5  # Only the 5 with odds

    def test_batch_with_total_only_no_ml(self, store: MarketOddsStore):
        """Games with only market_total (no moneyline) should still be logged."""
        games = [
            _LiveGameStub(
                game_id="total_only_001",
                game_date="2025-01-25",
                home_team="Home Team",
                away_team="Away Team",
                home_team_short="Home",
                away_team_short="Away",
                home_ml=None,
                away_ml=None,
                spread=None,
                market_total=220.5,  # Only total available
                over_odds=-110.0,
                under_odds=-110.0,
                n_books_ml=0,
                n_books_total=3,
                ml_std=None,
                sport_key="basketball_nba",
            )
        ]
        count = store.log_batch(games)
        assert count == 1  # Has market_total → logged

    def test_batch_empty_list(self, store: MarketOddsStore):
        """Empty batch list logs 0 games."""
        count = store.log_batch([])
        assert count == 0

    def test_batch_counts_sequential_calls(self, store: MarketOddsStore):
        """Multiple batch calls accumulate snapshots."""
        games_batch = [
            _LiveGameStub(
                game_id=f"batch_call_1_game_{i}",
                game_date="2025-01-25",
                home_team="Team A",
                away_team="Team B",
                home_team_short="A",
                away_team_short="B",
                home_ml=-110.0,
                away_ml=-110.0,
                spread=0.0,
                market_total=220.0,
                over_odds=-110.0,
                under_odds=-110.0,
                n_books_ml=3,
                n_books_total=3,
                ml_std=5.0,
                sport_key="basketball_nba",
            )
            for i in range(3)
        ]
        assert store.log_batch(games_batch) == 3
        assert store.log_batch(games_batch) == 3  # Same games again
        assert store.get_stats()["total_snapshots"] == 6  # Not deduped


# ====================================================================
#  SECTION 4: get_odds_for_date
# ====================================================================


class TestGetOddsForDate:
    """Unit tests for get_odds_for_date."""

    def test_returns_dataframe(self, store: MarketOddsStore):
        """Returns a DataFrame (not None) even when no data."""
        df = store.get_odds_for_date("2025-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_empty_date(self, store: MarketOddsStore):
        """Date with no records returns empty DataFrame."""
        df = store.get_odds_for_date("2025-01-01")
        assert df.empty

    def test_exact_match(self, store: MarketOddsStore):
        """Single game on a specific date is returned."""
        store.log_snapshot(
            game_id="GAME_010", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_ml=-150.0, away_ml=+130.0,
        )
        df = store.get_odds_for_date("2025-01-15")
        assert len(df) == 1
        assert df.iloc[0]["game_id"] == "GAME_010"

    def test_multiple_games_same_date(self, store: MarketOddsStore):
        """Multiple games on the same date all appear."""
        for i in range(5):
            store.log_snapshot(
                game_id=f"GAME_{i:04d}", game_date="2025-01-15",
                home_team=f"Home {i}", away_team=f"Away {i}",
            )
        df = store.get_odds_for_date("2025-01-15")
        assert len(df) == 5

    def test_latest_only_false_returns_all_snapshots(self, store: MarketOddsStore):
        """With latest_only=False, all snapshots for the same game are returned."""
        store.log_snapshot(
            game_id="GAME_011", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_ml=-150.0, away_ml=+130.0,
        )
        store.log_snapshot(
            game_id="GAME_011", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_ml=-160.0, away_ml=+140.0,
        )

        df_all = store.get_odds_for_date("2025-01-15", latest_only=False)
        assert len(df_all) == 2

        df_latest = store.get_odds_for_date("2025-01-15", latest_only=True)
        assert len(df_latest) == 1
        # Should return the most recent (-160)
        assert df_latest.iloc[0]["home_ml"] == -160.0

    def test_multiple_games_with_multiple_snapshots(self, store: MarketOddsStore):
        """Multiple games each with multiple snapshots — latest_only returns one per game."""
        for game_idx in range(3):
            for snap_idx in range(2):
                store.log_snapshot(
                    game_id=f"GAME_M{game_idx}",
                    game_date="2025-01-15",
                    home_team=f"Home {game_idx}",
                    away_team=f"Away {game_idx}",
                    home_ml=-100.0 - snap_idx,
                    away_ml=-120.0 + snap_idx,
                )

        df_latest = store.get_odds_for_date("2025-01-15", latest_only=True)
        assert len(df_latest) == 3  # One per unique game

    def test_date_filter_respects_sport_key(self, store: MarketOddsStore):
        """get_odds_for_date filters by sport_key."""
        store.log_snapshot(
            game_id="NBA_001", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            sport_key="basketball_nba",
        )
        store.log_snapshot(
            game_id="WNBA_001", game_date="2025-01-15",
            home_team="NY Liberty", away_team="Las Vegas Aces",
            sport_key="basketball_wnba",
        )

        nba_df = store.get_odds_for_date("2025-01-15", sport_key="basketball_nba")
        assert len(nba_df) == 1
        assert nba_df.iloc[0]["game_id"] == "NBA_001"

        wnba_df = store.get_odds_for_date("2025-01-15", sport_key="basketball_wnba")
        assert len(wnba_df) == 1
        assert wnba_df.iloc[0]["game_id"] == "WNBA_001"

        empty_df = store.get_odds_for_date("2025-01-15", sport_key="basketball_ncaab")
        assert empty_df.empty

    def test_no_match_different_date(self, store: MarketOddsStore):
        """Querying for a date with no stored games returns empty DataFrame."""
        store.log_snapshot(
            game_id="GAME_012", game_date="2025-01-10",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
        )
        df = store.get_odds_for_date("2025-01-15")
        assert df.empty


# ====================================================================
#  SECTION 5: get_market_prob_for_game
# ====================================================================


class TestGetMarketProbForGame:
    """Unit tests for get_market_prob_for_game — the core lookup for training.

    This method is used by the MarketInefficiencySystem training pipeline to
    retrieve real market-implied probabilities. It must handle:
    - Full team name ("Boston Celtics")
    - Short team name ("Celtics")
    - Swapped home/away ordering (asked for home=Celtics/away=Lakers but
      the record might have home=Boston/away=LA or vice versa)
    """

    def _seed_game(
        self,
        store: MarketOddsStore,
        game_id: str = "GAME_P_001",
        date: str = "2025-01-15",
        home_team: str = "Boston Celtics",
        away_team: str = "Los Angeles Lakers",
        home_short: str = "Celtics",
        away_short: str = "Lakers",
        home_ml: float = -150.0,
        away_ml: float = +130.0,
    ):
        """Helper: log a single game for prob lookup tests."""
        store.log_snapshot(
            game_id=game_id, game_date=date,
            home_team=home_team, away_team=away_team,
            home_team_short=home_short, away_team_short=away_short,
            home_ml=home_ml, away_ml=away_ml,
        )

    # ── Full Name Variants ───────────────────────────────────────────

    def test_full_name_match(self, store: MarketOddsStore):
        """Full team name match: home='Boston Celtics', away='Los Angeles Lakers'."""
        self._seed_game(store)
        prob = store.get_market_prob_for_game("Boston Celtics", "Los Angeles Lakers", "2025-01-15")
        assert prob is not None
        assert 0.55 < prob < 0.60  # -150/+130 → ~0.58 vig-free

    def test_full_name_swapped_orientation(self, store: MarketOddsStore):
        """Teams are queried in swapped order — home=away_record, away=home_record.
        Should return 1 - home_prob (since home team on record is the away team in query)."""
        self._seed_game(store, home_team="Boston Celtics", away_team="Los Angeles Lakers")
        prob = store.get_market_prob_for_game("Los Angeles Lakers", "Boston Celtics", "2025-01-15")
        assert prob is not None
        # Since we queried with Lakers as 'home' and Celtics as 'away', but the
        # record has Celtics as home (favored at -150), the prob should be ~0.42
        assert 0.40 < prob < 0.45

    # ── Short Name Variants ──────────────────────────────────────────

    def test_short_name_match(self, store: MarketOddsStore):
        """Short team name match: home='Celtics', away='Lakers'."""
        self._seed_game(store)
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob is not None
        assert 0.55 < prob < 0.60

    def test_short_name_swapped_orientation(self, store: MarketOddsStore):
        """Short names, swapped order: home='Lakers', away='Celtics'."""
        self._seed_game(store)
        prob = store.get_market_prob_for_game("Lakers", "Celtics", "2025-01-15")
        assert prob is not None
        assert 0.40 < prob < 0.45

    def test_short_name_full_name_mix(self, store: MarketOddsStore):
        """Mix of short and full names: home='Celtics', away='Los Angeles Lakers'."""
        self._seed_game(store)
        prob = store.get_market_prob_for_game("Celtics", "Los Angeles Lakers", "2025-01-15")
        assert prob is not None
        assert 0.55 < prob < 0.60

    # ── No Match / Edge Cases ────────────────────────────────────────

    def test_no_match_returns_none(self, store: MarketOddsStore):
        """No stored game for this matchup returns None."""
        prob = store.get_market_prob_for_game("Miami Heat", "New York Knicks", "2025-01-15")
        assert prob is None

    def test_no_match_wrong_date(self, store: MarketOddsStore):
        """Game exists but on wrong date returns None."""
        self._seed_game(store, date="2025-01-10")
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob is None

    def test_no_match_wrong_teams(self, store: MarketOddsStore):
        """Correct date but wrong teams returns None."""
        self._seed_game(store, home_team="Miami Heat", away_team="New York Knicks",
                        home_short="Heat", away_short="Knicks")
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob is None

    def test_return_none_when_no_vig_free_prob(self, store: MarketOddsStore):
        """Game stored without odds (no moneyline) returns None."""
        store.log_snapshot(
            game_id="GAME_NO_ML",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=None,
            away_ml=None,
        )
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob is None

    def test_latest_snapshot_is_used(self, store: MarketOddsStore):
        """When multiple snapshots exist, the latest (closing line) is returned."""
        self._seed_game(store, game_id="GAME_LINE",
                         home_ml=-150.0, away_ml=+130.0)
        # Line movement: home became bigger favorite
        store.log_snapshot(
            game_id="GAME_LINE",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=-300.0,  # Much bigger favorite
            away_ml=+250.0,
        )
        # Should return the more recent line (~0.75 vig-free)
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob is not None
        assert prob > 0.70

    def test_unknown_team_names(self, store: MarketOddsStore):
        """Completely unrecognized team names return None."""
        prob = store.get_market_prob_for_game("Nonexistent Team", "Also Fake", "2025-01-15")
        assert prob is None

    def test_favorite_underdog_direction_correct(self, store: MarketOddsStore):
        """The prob direction is always home-team probability from the QUERY perspective.

        When the stored record has home=Boston Celtics (favored at -150, prob ~0.58),
        querying for home='Celtics' should return ~0.58, not ~0.42.
        """
        self._seed_game(store)
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob is not None
        assert prob > 0.5  # Home team is favored → prob > 0.5

    def test_underdog_direction_correct(self, store: MarketOddsStore):
        """When stored home team is the underdog, querying with that team returns <0.5."""
        store.log_snapshot(
            game_id="GAME_UD",
            game_date="2025-01-15",
            home_team="Utah Jazz",
            away_team="Boston Celtics",
            home_team_short="Jazz",
            away_team_short="Celtics",
            home_ml=+160.0,  # Home is underdog
            away_ml=-190.0,
        )
        # Jazz are home underdogs → prob < 0.5
        prob = store.get_market_prob_for_game("Jazz", "Celtics", "2025-01-15")
        assert prob is not None
        assert prob < 0.5


# ====================================================================
#  SECTION 6: get_market_probs_for_date_range
# ====================================================================


class TestGetMarketProbsForDateRange:
    """Unit tests for get_market_probs_for_date_range."""

    def _seed_game(
        self,
        store: MarketOddsStore,
        game_id: str,
        date: str,
        home_team: str,
        away_team: str,
        home_short: str,
        away_short: str,
        home_ml: float = -110.0,
        away_ml: float = -110.0,
    ):
        store.log_snapshot(
            game_id=game_id, game_date=date,
            home_team=home_team, away_team=away_team,
            home_team_short=home_short, away_team_short=away_short,
            home_ml=home_ml, away_ml=away_ml,
        )

    def test_single_date(self, store: MarketOddsStore):
        """Games on a single date within the range returned."""
        self._seed_game(store, "G001", "2025-01-15",
                        "Boston Celtics", "Los Angeles Lakers", "Celtics", "Lakers",
                        home_ml=-150.0, away_ml=+130.0)
        result = store.get_market_probs_for_date_range("2025-01-15", "2025-01-15")
        assert len(result) == 1
        key = ("Boston Celtics", "Los Angeles Lakers", "2025-01-15")
        assert key in result
        assert 0.55 < result[key] < 0.60

    def test_date_range_multiple_games(self, store: MarketOddsStore):
        """Multiple games across a date range all returned."""
        dates = ["2025-01-15", "2025-01-16", "2025-01-17"]
        for i, d in enumerate(dates):
            self._seed_game(store, f"G00{i}", d,
                            f"Home {i}", f"Away {i}", f"H{i}", f"A{i}")
        result = store.get_market_probs_for_date_range("2025-01-15", "2025-01-17")
        assert len(result) == 3

    def test_empty_range(self, store: MarketOddsStore):
        """No games in date range returns empty dict."""
        self._seed_game(store, "G001", "2025-01-15",
                        "Boston Celtics", "Los Angeles Lakers", "Celtics", "Lakers")
        result = store.get_market_probs_for_date_range("2025-02-01", "2025-02-28")
        assert result == {}

    def test_partial_range(self, store: MarketOddsStore):
        """Only games within the date range are returned (not all stored games)."""
        self._seed_game(store, "G001", "2025-01-10",
                        "Team A", "Team B", "A", "B")
        self._seed_game(store, "G002", "2025-01-15",
                        "Team C", "Team D", "C", "D")
        self._seed_game(store, "G003", "2025-01-20",
                        "Team E", "Team F", "E", "F")

        result = store.get_market_probs_for_date_range("2025-01-12", "2025-01-18")
        assert len(result) == 1
        assert ("Team C", "Team D", "2025-01-15") in result

    def test_dedup_latest_snapshot(self, store: MarketOddsStore):
        """Multiple snapshots for the same game: only the latest is included."""
        # First snapshot
        store.log_snapshot(
            game_id="G_DUP", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_team_short="Celtics", away_team_short="Lakers",
            home_ml=-150.0, away_ml=+130.0,
        )
        # Second snapshot (line moved)
        store.log_snapshot(
            game_id="G_DUP", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_team_short="Celtics", away_team_short="Lakers",
            home_ml=-300.0, away_ml=+250.0,
        )

        result = store.get_market_probs_for_date_range("2025-01-15", "2025-01-15")
        assert len(result) == 1  # Deduped
        key = ("Boston Celtics", "Los Angeles Lakers", "2025-01-15")
        assert key in result
        assert result[key] > 0.70  # Latest snapshot (-300/+250)

    def test_only_games_with_vig_free_prob(self, store: MarketOddsStore):
        """Games without vig_removed_home_prob are excluded from results."""
        # Game with odds (will have vig-free prob)
        store.log_snapshot(
            game_id="G_WITH", game_date="2025-01-15",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_team_short="Celtics", away_team_short="Lakers",
            home_ml=-150.0, away_ml=+130.0,
        )
        # Game without odds (no vig-free prob)
        store.log_snapshot(
            game_id="G_WITHOUT", game_date="2025-01-15",
            home_team="Miami Heat", away_team="New York Knicks",
            home_team_short="Heat", away_team_short="Knicks",
            home_ml=None, away_ml=None,
        )

        result = store.get_market_probs_for_date_range("2025-01-15", "2025-01-15")
        assert len(result) == 1  # Only the game with odds
        assert ("Boston Celtics", "Los Angeles Lakers", "2025-01-15") in result


# ====================================================================
#  SECTION 7: get_stats
# ====================================================================


class TestGetStats:
    """Unit tests for get_stats."""

    def test_empty_store(self, store: MarketOddsStore):
        """Empty store returns zeros."""
        stats = store.get_stats()
        assert stats["total_snapshots"] == 0
        assert stats["unique_games"] == 0

    def test_single_snapshot(self, store: MarketOddsStore):
        """Single snapshot: count=1, unique=1."""
        store.log_snapshot(
            game_id="GAME_STAT_1", game_date="2025-01-15",
            home_team="Team A", away_team="Team B",
        )
        stats = store.get_stats()
        assert stats["total_snapshots"] == 1
        assert stats["unique_games"] == 1

    def test_multiple_snapshots_same_game(self, store: MarketOddsStore):
        """Multiple snapshots of same game: count=2, unique=1."""
        for _ in range(2):
            store.log_snapshot(
                game_id="GAME_STAT_2", game_date="2025-01-15",
                home_team="Team A", away_team="Team B",
            )
        stats = store.get_stats()
        assert stats["total_snapshots"] == 2
        assert stats["unique_games"] == 1

    def test_multiple_games(self, store: MarketOddsStore):
        """Multiple games with unique IDs: both count and unique reflect the total."""
        for i in range(10):
            store.log_snapshot(
                game_id=f"GAME_STAT_{i:03d}", game_date="2025-01-15",
                home_team=f"Team {i}", away_team=f"Opp {i}",
            )
        stats = store.get_stats()
        assert stats["total_snapshots"] == 10
        assert stats["unique_games"] == 10


# ====================================================================
#  SECTION 8: Edge Cases & Integration
# ====================================================================


class TestEdgeCases:
    """Edge case and integration-style tests."""

    def test_full_round_trip(self, store: MarketOddsStore):
        """Log a game, query it back by date and by team name, verify consistency."""
        # Log
        store.log_snapshot(
            game_id="RT_001",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=-150.0,
            away_ml=+130.0,
            spread=-4.5,
            market_total=224.5,
            n_books_ml=5,
        )

        # Query by date
        df = store.get_odds_for_date("2025-01-15")
        assert len(df) == 1
        assert df.iloc[0]["home_ml"] == -150.0

        # Query by team (full name)
        prob_full = store.get_market_prob_for_game("Boston Celtics", "Los Angeles Lakers", "2025-01-15")
        assert prob_full is not None

        # Query by team (short name)
        prob_short = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-15")
        assert prob_short is not None

        # Both queries return the same probability
        assert prob_full == prob_short

        # Query by date range
        overrides = store.get_market_probs_for_date_range("2025-01-15", "2025-01-15")
        assert len(overrides) == 1

    def test_multiple_games_different_dates(self, store: MarketOddsStore):
        """Games on different dates can be queried independently."""
        store.log_snapshot(
            game_id="MD_001", game_date="2025-01-10",
            home_team="Boston Celtics", away_team="Los Angeles Lakers",
            home_team_short="Celtics", away_team_short="Lakers",
            home_ml=-150.0, away_ml=+130.0,
        )
        store.log_snapshot(
            game_id="MD_002", game_date="2025-01-12",
            home_team="Miami Heat", away_team="New York Knicks",
            home_team_short="Heat", away_team_short="Knicks",
            home_ml=-110.0, away_ml=-110.0,
        )

        # Query date 1
        df1 = store.get_odds_for_date("2025-01-10")
        assert len(df1) == 1
        assert df1.iloc[0]["home_team"] == "Boston Celtics"

        # Query date 2
        df2 = store.get_odds_for_date("2025-01-12")
        assert len(df2) == 1
        assert df2.iloc[0]["home_team"] == "Miami Heat"

        # Query both
        all_probs = store.get_market_probs_for_date_range("2025-01-10", "2025-01-12")
        assert len(all_probs) == 2

    def test_get_market_prob_for_game_exact_via_full_name(self, store: MarketOddsStore):
        """Full name match via home_team column (not short name) returns correct prob."""
        store.log_snapshot(
            game_id="FN_001",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=-110.0,
            away_ml=-110.0,
        )
        prob = store.get_market_prob_for_game(
            "Boston Celtics", "Los Angeles Lakers", "2025-01-15"
        )
        assert prob is not None
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_get_market_prob_for_game_exact_via_short_name(self, store: MarketOddsStore):
        """Short name match via home_team_short column returns correct prob."""
        store.log_snapshot(
            game_id="SN_001",
            game_date="2025-01-15",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_team_short="Celtics",
            away_team_short="Lakers",
            home_ml=-110.0,
            away_ml=-110.0,
        )
        prob = store.get_market_prob_for_game(
            "Celtics", "Lakers", "2025-01-15"
        )
        assert prob is not None

    def test_log_batch_then_get_market_prob(self, store: MarketOddsStore):
        """Batch-logged games can be looked up individually by team name."""
        games = [
            _LiveGameStub(
                game_id=f"BATCH_PROB_{i}",
                game_date="2025-01-25",
                home_team="Boston Celtics" if i % 2 == 0 else "Miami Heat",
                away_team="Los Angeles Lakers" if i % 2 == 0 else "New York Knicks",
                home_team_short="Celtics" if i % 2 == 0 else "Heat",
                away_team_short="Lakers" if i % 2 == 0 else "Knicks",
                home_ml=-150.0 if i % 2 == 0 else +120.0,
                away_ml=+130.0 if i % 2 == 0 else -140.0,
                spread=-4.5 if i % 2 == 0 else 2.5,
                market_total=224.5 if i % 2 == 0 else 215.0,
                over_odds=-110.0,
                under_odds=-110.0,
                n_books_ml=4,
                n_books_total=3,
                ml_std=8.0,
                sport_key="basketball_nba",
            )
            for i in range(4)
        ]
        store.log_batch(games)

        # Query first game
        prob = store.get_market_prob_for_game("Celtics", "Lakers", "2025-01-25")
        assert prob is not None
        assert 0.55 < prob < 0.60

        # Query second game
        prob2 = store.get_market_prob_for_game("Heat", "Knicks", "2025-01-25")
        assert prob2 is not None
        # Heat are underdogs (+120) → prob < 0.5
        assert prob2 < 0.5
