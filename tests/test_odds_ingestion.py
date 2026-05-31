"""
Unit tests for odds_ingestion.py — auto_mark_finished_games and get_sport_duration.

Covers:
  - get_sport_duration: known sports, case-insensitive, unknown sport error
  - auto_mark_finished_games: finished, in-progress, future, naive dates,
    malformed dates, already-finished games, empty DB, sport param, defaults,
    sport-overrides-hours precedence

Mock strategy: patch the module-level `datetime` reference BUT delegate
`fromisoformat` to the real implementation so real datetime objects are
returned and comparisons/arithmetic work correctly.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from betting_intel.data.odds_ingestion import (
    OddsIngestionEngine,
    SPORT_DURATIONS,
    get_sport_duration,
)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """Create an engine backed by a temporary file (not :memory:).

    Using a file avoids SQLite's per-connection isolation with :memory:
    where tables created in _init_database() are invisible to subsequent
    calls to _connect().
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    eng = OddsIngestionEngine(Path(tmp.name))
    yield eng
    os.unlink(tmp.name)


def _insert_game(
    eng: OddsIngestionEngine,
    game_id: str,
    commence_time: str,
    is_finished: int = 0,
    home: str = "Lakers",
    away: str = "Celtics",
):
    """Insert a row directly into odds_meta.

    Uses the real (unpatched) datetime for its own timestamps since this
    helper runs during test setup, before the method under test calls
    datetime.now().
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with eng._connect() as conn:
        conn.execute(
            """INSERT INTO odds_meta
               (game_id, home_team, away_team, commence_time,
                first_seen, last_updated, is_finished)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (game_id, home, away, commence_time,
             now_iso, now_iso, is_finished),
        )
        conn.commit()


def _count_finished(eng: OddsIngestionEngine) -> int:
    with eng._connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM odds_meta WHERE is_finished = 1"
        ).fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════
#  FIXED REFERENCE TIME
# ═══════════════════════════════════════════════════════════════════════════
#
# Tests freeze datetime at this moment for deterministic results.
#
#   REFERENCE_NOW = 2026-06-01 12:00:00 UTC
#
#  Game states relative to REFERENCE_NOW:
#    FINISHED    -> commence 4h ago (past + 3h = 1h ago)        → finished
#    IN_PROGRESS -> commence 2h ago (past + 3h = 1h ahead)     → NOT finished
#    FUTURE      -> commence 1h ahead                           → NOT finished
# ═══════════════════════════════════════════════════════════════════════════

REFERENCE_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(hours_ago: float) -> str:
    """ISO-8601 timestamp *hours_ago* before REFERENCE_NOW."""
    return (REFERENCE_NOW - timedelta(hours=hours_ago)).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
#  PATCH HELPER — for time-dependent tests
# ═══════════════════════════════════════════════════════════════════════════
#
# We patch the module-level `datetime` reference (which is the built-in
# datetime class after `from datetime import datetime`).
#
# CRITICAL: we delegate .fromisoformat and the constructor to the real
# datetime so that real datetime objects are returned.  This keeps .tzinfo,
# .replace(), ==, >=, and + timedelta working with genuine objects instead
# of MagicMock instances.
#
# We CANNOT use @patch("...datetime.now") because datetime is a C extension
# type and its attributes are immutable.
# ═══════════════════════════════════════════════════════════════════════════

_real_dt = datetime  # keep a reference before any patching


def _patch_dt_now():
    """Return a (patcher, mock) tuple that freezes datetime.now()."""
    patcher = patch("betting_intel.data.odds_ingestion.datetime", autospec=True)
    mock_dt = patcher.start()
    mock_dt.now.return_value = REFERENCE_NOW
    mock_dt.fromisoformat = _real_dt.fromisoformat
    # Make datetime(...) constructor work if called
    mock_dt.side_effect = _real_dt
    return patcher, mock_dt


# ═══════════════════════════════════════════════════════════════════════════
#  1. get_sport_duration
# ═══════════════════════════════════════════════════════════════════════════


class TestGetSportDuration:
    """Module-level get_sport_duration() lookups."""

    def test_nba(self):
        assert get_sport_duration("NBA") == 3.0

    def test_nfl(self):
        assert get_sport_duration("NFL") == 3.5

    def test_nhl(self):
        assert get_sport_duration("NHL") == 2.5

    def test_mlb(self):
        assert get_sport_duration("MLB") == 3.0

    def test_case_insensitive(self):
        assert get_sport_duration("nba") == 3.0
        assert get_sport_duration("Nba") == 3.0

    def test_trim_whitespace(self):
        assert get_sport_duration("  NFL  ") == 3.5

    def test_all_defined_sports_return_positive(self):
        for sport in SPORT_DURATIONS:
            assert get_sport_duration(sport) > 0

    def test_unknown_sport_raises_keyerror(self):
        with pytest.raises(KeyError) as exc:
            get_sport_duration("CRICKET")
        msg = str(exc.value)
        assert "CRICKET" in msg
        assert "NBA" in msg


# ═══════════════════════════════════════════════════════════════════════════
#  2. auto_mark_finished_games — empty / baseline (no time mock needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoMarkEmptyDB:
    """Behaviour when there are no games to process."""

    def test_no_games_returns_zero(self, engine):
        count = engine.auto_mark_finished_games()
        assert count == 0

    def test_no_unfinished_games_returns_zero(self, engine):
        _insert_game(engine, "G1", "2020-01-01T00:00:00Z", is_finished=1)
        _insert_game(engine, "G2", "2020-01-01T00:00:00Z", is_finished=1)
        count = engine.auto_mark_finished_games()
        assert count == 0
        assert _count_finished(engine) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  3. auto_mark_finished_games — time-dependent core logic
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoMarkCoreLogic:
    """Core finished / in-progress / future detection."""

    def test_finished_game_is_marked(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "FINISHED", _ts(4.0))  # 4h ago + 3h = done
            count = engine.auto_mark_finished_games()
            assert count == 1
            assert _count_finished(engine) == 1
        finally:
            patcher.stop()

    def test_in_progress_game_not_marked(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "IN_PROGRESS", _ts(2.0))  # 2h ago + 3h = live
            count = engine.auto_mark_finished_games()
            assert count == 0
        finally:
            patcher.stop()

    def test_future_game_not_marked(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "FUTURE", _ts(-1.0))  # 1h in future
            count = engine.auto_mark_finished_games()
            assert count == 0
        finally:
            patcher.stop()

    def test_just_started_not_marked(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "JUST_STARTED", _ts(0.1))  # 6 min ago
            count = engine.auto_mark_finished_games()
            assert count == 0
        finally:
            patcher.stop()

    def test_mixed_states_only_finished_marked(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "DONE_1", _ts(5.0))
            _insert_game(engine, "DONE_2", _ts(4.5))
            _insert_game(engine, "LIVE_1", _ts(2.0))
            _insert_game(engine, "LIVE_2", _ts(1.0))
            _insert_game(engine, "FUTURE", _ts(-2.0))
            count = engine.auto_mark_finished_games()
            assert count == 2
            assert _count_finished(engine) == 2
        finally:
            patcher.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  4. auto_mark_finished_games — date/time edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoMarkDateEdgeCases:
    """Naive datetimes, malformed dates, and already-finished games."""

    def test_naive_date_treated_as_utc(self, engine):
        """A naive commence_time (no tzinfo) should be treated as UTC."""
        patcher, _ = _patch_dt_now()
        try:
            naive_dt = REFERENCE_NOW - timedelta(hours=4)
            naive_str = naive_dt.replace(tzinfo=None).isoformat()
            assert "T" in naive_str
            assert "Z" not in naive_str and "+" not in naive_str
            _insert_game(engine, "NAIVE", naive_str)
            count = engine.auto_mark_finished_games()
            assert count == 1
        finally:
            patcher.stop()

    def test_malformed_date_skipped_gracefully(self, engine):
        """Unparseable commence_time rows are skipped (no crash)."""
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "GOOD", _ts(5.0))
            _insert_game(engine, "BAD", "not-a-date")
            _insert_game(engine, "EMPTY", "")
            count = engine.auto_mark_finished_games()
            assert count == 1
            assert _count_finished(engine) == 1
        finally:
            patcher.stop()

    def test_already_finished_ignored(self, engine):
        """Games with is_finished=1 are skipped by the SQL query."""
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "ALREADY_DONE", _ts(5.0), is_finished=1)
            _insert_game(engine, "NEEDS_DONE", _ts(5.0), is_finished=0)
            count = engine.auto_mark_finished_games()
            assert count == 1
            assert _count_finished(engine) == 2
        finally:
            patcher.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  5. auto_mark_finished_games — sport parameter & duration resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoMarkSportDuration:
    """sport= parameter resolves correct durations."""

    def test_nba_sport(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "NBA_GAME", _ts(3.5))
            count = engine.auto_mark_finished_games(sport="NBA")
            assert count == 1
        finally:
            patcher.stop()

    def test_nfl_sport(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "NFL_GAME", _ts(3.5))
            count = engine.auto_mark_finished_games(sport="NFL")
            assert count == 1
        finally:
            patcher.stop()

    def test_nhl_sport(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "DONE", _ts(3.0))
            _insert_game(engine, "LIVE", _ts(2.0))
            count = engine.auto_mark_finished_games(sport="NHL")
            assert count == 1
        finally:
            patcher.stop()

    def test_sport_overrides_game_duration_hours(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "NFL_GAME", _ts(3.5))
            count = engine.auto_mark_finished_games(
                game_duration_hours=1.0,
                sport="NFL",
            )
            # sport="NFL"=3.5h wins over 1.0h
            assert count == 1
        finally:
            patcher.stop()

    def test_default_duration_three_hours(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "DONE", _ts(3.5))
            _insert_game(engine, "LIVE", _ts(2.0))
            count = engine.auto_mark_finished_games()
            assert count == 1
        finally:
            patcher.stop()

    def test_explicit_game_duration_hours(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "SHORT_GAME", _ts(2.5))
            _insert_game(engine, "LONG_GAME", _ts(1.5))
            count = engine.auto_mark_finished_games(game_duration_hours=2.0)
            assert count == 1
        finally:
            patcher.stop()

    def test_unknown_sport_raises_error(self, engine):
        patcher, _ = _patch_dt_now()
        try:
            _insert_game(engine, "G1", _ts(5.0))
            with pytest.raises(KeyError):
                engine.auto_mark_finished_games(sport="CRICKET")
        finally:
            patcher.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  6. SPORT_DURATIONS integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestSportDurationsIntegrity:
    """The SPORT_DURATIONS table is self-consistent."""

    def test_all_values_positive(self):
        for sport, hours in SPORT_DURATIONS.items():
            assert hours > 0, f"{sport} has non-positive duration {hours}"

    def test_all_keys_uppercase(self):
        for sport in SPORT_DURATIONS:
            assert sport == sport.upper(), f"{sport} is not uppercase"

    def test_minimum_sports_defined(self):
        for required in ("NBA", "NFL", "MLB", "NHL"):
            assert required in SPORT_DURATIONS
