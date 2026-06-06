"""Unit tests for PlayerStatsManager and helper functions."""

from __future__ import annotations

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from unittest.mock import patch

import pytest

from betting_intel.data.player_stats import (
    PlayerStatsManager,
    _parse_minutes,
    _fmt_pct,
    _parse_stat_pair,
    _PLAYER_GAME_LOGS_DDL,
    _PLAYER_TRACKING_DDL,
)


# _derive_season_id is a @staticmethod on PlayerStatsManager, not a module-level function.
# Alias it here for convenience in test assertions.
_derive_season_id = PlayerStatsManager._derive_season_id


# ── Helper Function Tests ──────────────────────────────────────────────────


class TestParseMinutes:
    """Tests for the _parse_minutes helper."""

    def test_standard_format(self):
        assert _parse_minutes("PT26M01.00S") == 26

    def test_zero_seconds(self):
        assert _parse_minutes("PT10M00.00S") == 10

    def test_single_minute(self):
        assert _parse_minutes("PT1M15.50S") == 1

    def test_none_input(self):
        assert _parse_minutes(None) == 0

    def test_empty_string(self):
        assert _parse_minutes("") == 0

    def test_unusual_format(self):
        # Player who played 0 minutes
        assert _parse_minutes("PT0M00.00S") == 0


class TestFmtPct:
    """Tests for the _fmt_pct helper."""

    def test_normal_value(self):
        assert _fmt_pct(0.456) == 0.456

    def test_none_input(self):
        assert _fmt_pct(None) == 0.0

    def test_rounding(self):
        assert _fmt_pct(0.45678) == 0.457

    def test_zero(self):
        assert _fmt_pct(0.0) == 0.0

    def test_one(self):
        assert _fmt_pct(1.0) == 1.0


class TestParseStatPair:
    """Tests for the _parse_stat_pair helper (ESPN compound stats)."""

    def test_standard_fg(self):
        """Parse '10-15' -> (10, 15)."""
        assert _parse_stat_pair("10-15") == (10, 15)

    def test_zero_attempts(self):
        """Parse '0-0' -> (0, 0)."""
        assert _parse_stat_pair("0-0") == (0, 0)

    def test_single_digits(self):
        """Parse '2-5' -> (2, 5)."""
        assert _parse_stat_pair("2-5") == (2, 5)

    def test_empty_string(self):
        """Empty string -> (0, 0)."""
        assert _parse_stat_pair("") == (0, 0)

    def test_none_input(self):
        """None -> (0, 0)."""
        assert _parse_stat_pair(None) == (0, 0)

    def test_no_dash(self):
        """String without dash -> (0, 0)."""
        assert _parse_stat_pair("5") == (0, 0)

    def test_malformed_string(self):
        """Malformed string -> (0, 0)."""
        assert _parse_stat_pair("abc-def") == (0, 0)


class TestDeriveSeasonId:
    """Tests for the _derive_season_id static method."""

    def test_october_game(self):
        # October game -> belongs to the season that started this year
        # f"{2025}{2026}" = "20252026" -> 20252026
        assert _derive_season_id("2025-10-15") == 20252026

    def test_november_game(self):
        assert _derive_season_id("2025-11-10") == 20252026

    def test_january_game(self):
        # January game -> second half of the season that started previous year
        # year=2026, month=1 -> 1 < 10 so display_year=2025 -> f"{2025}{2026}" = "20252026"
        assert _derive_season_id("2026-01-15") == 20252026

    def test_may_game(self):
        # May game -> still same season
        assert _derive_season_id("2026-05-30") == 20252026

    def test_september_game(self):
        # September -> before the new season starts, previous season
        # year=2026, month=9 -> 9 < 10 so display_year=2025 -> f"{2025}{2026}" = "20252026"
        assert _derive_season_id("2026-09-15") == 20252026

    def test_empty_string(self):
        assert _derive_season_id("") == 202526

    def test_invalid_string(self):
        assert _derive_season_id("not-a-date") == 202526

    def test_edge_case_october_first(self):
        # October game -> new season
        season_id = _derive_season_id("2025-10-01")
        assert season_id == 20252026

    def test_edge_case_september_thirtieth(self):
        # Sept 30 -> still in the previous season
        # year=2025, month=9 -> 9 < 10 so display_year = 2024
        # f"{2024}{2025}" = "20242025" -> 20242025
        season_id = _derive_season_id("2025-09-30")
        assert season_id == 20242025


# ── PlayerStatsManager DB Tests ────────────────────────────────────────────


@pytest.fixture
def temp_db(tmp_path) -> Path:
    """Create a temporary SQLite database for testing."""
    db_path = tmp_path / "test_nba_data.db"
    return db_path


@pytest.fixture
def populated_manager(temp_db) -> PlayerStatsManager:
    """Create a PlayerStatsManager with pre-populated test data."""
    manager = PlayerStatsManager(db_path=temp_db)

    # Insert test data into player_game_logs
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.cursor()

    # Insert game_logs entries for team abbreviation joins
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_logs (
            GAME_ID TEXT,
            TEAM_ID INTEGER,
            TEAM_ABBREVIATION TEXT,
            GAME_DATE TEXT
        )
    """)
    cursor.execute("INSERT INTO game_logs VALUES ('GAME001', 1, 'NYK', '2026-01-15')")
    cursor.execute("INSERT INTO game_logs VALUES ('GAME001', 2, 'SAS', '2026-01-15')")
    cursor.execute("INSERT INTO game_logs VALUES ('GAME002', 1, 'NYK', '2026-01-17')")
    cursor.execute("INSERT INTO game_logs VALUES ('GAME002', 2, 'BOS', '2026-01-17')")
    cursor.execute("INSERT INTO game_logs VALUES ('GAME003', 1, 'NYK', '2026-01-20')")
    cursor.execute("INSERT INTO game_logs VALUES ('GAME003', 2, 'LAL', '2026-01-20')")

    # Insert player_game_logs entries
    test_players = [
        # (GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME, PTS, MINUTES, REB, AST, FGM, FGA, FG3M)
        ("GAME001", 1, 101, "Jalen Brunson", 28, 36, 5, 7, 10, 20, 3),
        ("GAME001", 1, 102, "Julius Randle", 22, 34, 9, 4, 8, 17, 2),
        ("GAME001", 1, 103, "Mikal Bridges", 18, 32, 4, 3, 7, 14, 2),
        ("GAME001", 2, 201, "Victor Wembanyama", 24, 30, 12, 5, 9, 18, 2),
        ("GAME001", 2, 202, "Devin Vassell", 16, 28, 3, 4, 6, 13, 3),
        ("GAME002", 1, 101, "Jalen Brunson", 32, 38, 4, 8, 12, 22, 4),
        ("GAME002", 1, 102, "Julius Randle", 20, 32, 8, 3, 7, 16, 1),
        ("GAME002", 1, 104, "OG Anunoby", 15, 30, 5, 2, 6, 12, 2),
        ("GAME002", 2, 301, "Jayson Tatum", 30, 36, 8, 5, 11, 22, 4),
        ("GAME002", 2, 302, "Jaylen Brown", 24, 34, 6, 4, 9, 18, 3),
        ("GAME003", 1, 101, "Jalen Brunson", 26, 35, 6, 9, 9, 19, 3),
        ("GAME003", 1, 102, "Julius Randle", 24, 33, 10, 5, 9, 18, 3),
        ("GAME003", 2, 401, "LeBron James", 28, 34, 7, 8, 10, 20, 2),
        ("GAME003", 2, 402, "Anthony Davis", 22, 32, 11, 3, 9, 16, 0),
    ]

    for row in test_players:
        game_id, team_id, player_id, name, pts, minutes, reb, ast, fgm, fga, fg3m = row
        cursor.execute(f"""
            INSERT OR IGNORE INTO player_game_logs
            (GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME, PTS, MINUTES, REB, AST, FGM, FGA, FG3M,
             FG_PCT, FG3_PCT, FG3A, FTM, FTA, FT_PCT, OREB, DREB, STL, BLK, TOV, PF,
             PLUS_MINUS, SEASON_ID, GAME_DATE, POSITION, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    0.0, 0.0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0,
                    0, 0, '', '', '')
        """, (game_id, team_id, player_id, name, pts, minutes, reb, ast, fgm, fga, fg3m))

    # Rebuild the tracking table so it reflects the test data
    conn.commit()
    conn.close()

    manager._rebuild_tracking()
    return manager


class TestPlayerStatsManager:
    """Tests for the PlayerStatsManager class."""

    def test_init_creates_tables(self, temp_db):
        """Initializing should create the required tables."""
        manager = PlayerStatsManager(db_path=temp_db)
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        # Check player_game_logs exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_game_logs'")
        assert cursor.fetchone() is not None

        # Check player_tracking exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_tracking'")
        assert cursor.fetchone() is not None

        conn.close()

    def test_get_player_ppg_exact_match(self, populated_manager):
        """get_player_ppg should return correct PPG for exact match."""
        ppg = populated_manager.get_player_ppg("Jalen Brunson")
        # Brunson played 3 games: 28, 32, 26 -> avg = 28.7
        assert ppg == pytest.approx(28.7, abs=0.1)

    def test_get_player_ppg_missing_player(self, populated_manager):
        """get_player_ppg should return 0.0 for unknown player."""
        ppg = populated_manager.get_player_ppg("Nonexistent Player")
        assert ppg == 0.0

    def test_get_player_stats_full(self, populated_manager):
        """get_player_stats should return all stat keys."""
        stats = populated_manager.get_player_stats("Jalen Brunson")
        assert stats["ppg"] == pytest.approx(28.7, abs=0.1)
        assert stats["games_played"] == 3
        assert stats["team"] == "NYK"
        assert "min" in stats
        assert "reb" in stats
        assert "ast" in stats

    def test_get_player_stats_missing(self, populated_manager):
        """get_player_stats should return empty dict for unknown player."""
        stats = populated_manager.get_player_stats("Nobody")
        assert stats == {}

    def test_get_team_players_returns_sorted(self, populated_manager):
        """get_team_players should return players sorted by PPG descending."""
        players = populated_manager.get_team_players("NYK")
        assert len(players) >= 3
        ppgs = [p["ppg"] for p in players]
        assert ppgs == sorted(ppgs, reverse=True)

        # Jalen Brunson should be first (highest PPG on NYK)
        assert players[0]["name"] == "Jalen Brunson"

    def test_get_team_players_unknown_team(self, populated_manager):
        """get_team_players should return empty list for unknown team."""
        players = populated_manager.get_team_players("ZZZ")
        assert players == []

    def test_search_player_matches(self, populated_manager):
        """search_player should find partial matches."""
        results = populated_manager.search_player("Brun")
        assert len(results) > 0
        assert "Jalen Brunson" in [r["name"] for r in results]

    def test_search_player_no_match(self, populated_manager):
        """search_player should return empty list for no matches."""
        results = populated_manager.search_player("Xyzzy")
        assert results == []

    def test_search_player_case_insensitive(self, populated_manager):
        """search_player should be case-insensitive."""
        results_lower = populated_manager.search_player("brunson")
        results_upper = populated_manager.search_player("BRUNSON")
        assert len(results_lower) > 0
        assert len(results_lower) == len(results_upper)

    def test_get_team_missing_ppg_clean_names(self, populated_manager):
        """get_team_missing_ppg should sum PPG for clean player names."""
        total = populated_manager.get_team_missing_ppg("NYK", ["Julius Randle", "OG Anunoby"])
        # Randle: (22+20+24)/3 = 22.0, Anunoby: 15 -> total ~37.0
        assert total == pytest.approx(37.0, abs=1.0)

    def test_get_team_missing_ppg_with_suffix(self, populated_manager):
        """get_team_missing_ppg should strip role suffix from player names."""
        total = populated_manager.get_team_missing_ppg(
            "NYK",
            ["Julius Randle (22 PPG, STAR)", "OG Anunoby (15 PPG, STARTER)"],
        )
        assert total == pytest.approx(37.0, abs=1.0)

    def test_get_team_missing_ppg_empty(self, populated_manager):
        """get_team_missing_ppg with empty list should return 0."""
        total = populated_manager.get_team_missing_ppg("NYK", [])
        assert total == 0.0

    def test_count_unprocessed_games_empty(self, temp_db):
        """With empty game_logs table, count should be 0."""
        manager = PlayerStatsManager(db_path=temp_db)
        # Create game_logs table (required by the query even if empty)
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE IF NOT EXISTS game_logs (GAME_ID TEXT, TEAM_ID INTEGER, TEAM_ABBREVIATION TEXT, GAME_DATE TEXT)")
        conn.close()
        count = manager.count_unprocessed_games()
        assert count == 0

    def test_rebuild_tracking_after_insert(self, temp_db):
        """After inserting data, rebuild should update tracking correctly."""
        manager = PlayerStatsManager(db_path=temp_db)

        # Manually insert a player_game_logs row
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_logs (
                GAME_ID TEXT, TEAM_ID INTEGER, TEAM_ABBREVIATION TEXT, GAME_DATE TEXT
            )
        """)
        cursor.execute("INSERT INTO game_logs VALUES ('G999', 1, 'NYK', '2026-03-01')")
        cursor.execute(f"""
            INSERT INTO player_game_logs
            (GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME, PTS, MINUTES, GAME_DATE, scraped_at)
            VALUES ('G999', 1, 999, 'Test Player', 15, 30, '2026-03-01', '')
        """)
        conn.commit()
        conn.close()

        # Rebuild and check
        manager._rebuild_tracking()
        ppg = manager.get_player_ppg("Test Player")
        assert ppg == 15.0

    def test_empty_database_operations(self, temp_db):
        """All query methods should handle empty database gracefully."""
        manager = PlayerStatsManager(db_path=temp_db)

        # Create game_logs table (required by count_unprocessed_games query even if empty)
        conn = sqlite3.connect(str(temp_db))
        conn.execute("CREATE TABLE IF NOT EXISTS game_logs (GAME_ID TEXT, TEAM_ID INTEGER, TEAM_ABBREVIATION TEXT, GAME_DATE TEXT)")
        conn.close()

        assert manager.get_player_ppg("Any Player") == 0.0
        assert manager.get_player_stats("Any Player") == {}
        assert manager.get_team_players("NYK") == []
        assert manager.search_player("Test") == []
        assert manager.get_team_missing_ppg("NYK", ["Player"]) == 0.0
        assert manager.count_unprocessed_games() == 0

    # ── ESPN Dispatch Tests ────────────────────────────────────────────

    def test_fetch_and_store_game_dispatches_espn(self, temp_db):
        """ESPN-style game IDs (4017...) should route to _fetch_and_store_game_espn."""
        manager = PlayerStatsManager(db_path=temp_db)
        with patch.object(manager, '_fetch_and_store_game_espn', return_value=[{'PLAYER_NAME': 'Test'}]) as mock_espn:
            result = manager._fetch_and_store_game('401716954')
            mock_espn.assert_called_once_with('401716954')
            assert result is not None

    def test_fetch_and_store_game_dispatches_nba(self, temp_db):
        """NBA-style game IDs (0022...) should route to _fetch_and_store_game_nba."""
        manager = PlayerStatsManager(db_path=temp_db)
        with patch.object(manager, '_fetch_and_store_game_nba', return_value=[{'PLAYER_NAME': 'Test'}]) as mock_nba:
            result = manager._fetch_and_store_game('0022500001')
            mock_nba.assert_called_once_with('0022500001')
            assert result is not None

    def test_count_unprocessed_games_includes_espn(self, temp_db):
        """count_unprocessed_games should now include ESPN-prefixed game IDs."""
        manager = PlayerStatsManager(db_path=temp_db)
        conn = sqlite3.connect(str(temp_db))
        c = conn.cursor()
        # Create game_logs
        c.execute("CREATE TABLE IF NOT EXISTS game_logs (GAME_ID TEXT, TEAM_ID INTEGER, TEAM_ABBREVIATION TEXT, GAME_DATE TEXT)")  # noqa: E501
        # Insert an ESPN game - unprocessed
        c.execute("INSERT INTO game_logs VALUES ('4017000001', 1, 'NYK', '2026-03-01')")
        conn.commit()
        conn.close()

        # The ESPN game should be counted as unprocessed
        count = manager.count_unprocessed_games()
        assert count == 1, f"Expected 1 ESPN unprocessed game, got {count}"

    def test_get_unprocessed_game_ids_includes_espn(self, temp_db):
        """get_unprocessed_game_ids should now include ESPN-prefixed game IDs."""
        manager = PlayerStatsManager(db_path=temp_db)
        conn = sqlite3.connect(str(temp_db))
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS game_logs (GAME_ID TEXT, TEAM_ID INTEGER, TEAM_ABBREVIATION TEXT, GAME_DATE TEXT)")  # noqa: E501
        # Insert an NBA game and an ESPN game - both unprocessed
        c.execute("INSERT INTO game_logs VALUES ('0022500001', 1, 'NYK', '2026-03-01')")
        c.execute("INSERT INTO game_logs VALUES ('4017000001', 2, 'BOS', '2026-03-01')")
        conn.commit()
        conn.close()

        ids = manager.get_unprocessed_game_ids(limit=10)
        assert '0022500001' in ids, "NBA ID should be in unprocessed list"
        assert '4017000001' in ids, "ESPN ID should be in unprocessed list"
