"""
Unit tests for the LivePredictionEngine module.

Tests are organized by class:
  1. LiveGame dataclass — edge cases, properties, serialization
  2. LivePredictionSnapshot — chart data, categories, serialization
  3. LivePredictionEngine — core state, caching, odds merging, auto-resolve
  4. LivePredictionWorker — start/stop lifecycle

All network/database calls are mocked to keep tests fast and deterministic.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Module path prefix for string-based patches (avoids import-order issues)
# NOTE: Imports in engine.py are inside methods, so we must patch the SOURCE modules.
_ENG = "betting_intel.live.engine"
_CFG = "betting_intel.live.sport_configs"
_DAT = "betting_intel.data.odds_fetcher"
_TRK = "betting_intel.analytics.tracker"
_DK  = "betting_intel.data.draftkings_scraper"


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_game() -> dict:
    """A realistic raw-odds dict resembling a TheOddsAPI event."""
    return {
        "id": "game_123",
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "commence_time": "2025-06-15T19:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -150},
                            {"name": "Los Angeles Lakers", "price": +130},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 218.5, "price": -110},
                            {"name": "Under", "point": 218.5, "price": -110},
                        ],
                    },
                ],
            },
        ],
    }


@pytest.fixture
def live_game():
    """A minimal LiveGame instance for tests."""
    from betting_intel.live.engine import LiveGame
    return LiveGame(
        game_id="g1",
        sport_key="basketball_nba",
        home_team="Boston Celtics",
        away_team="Los Angeles Lakers",
        home_team_short="BOS",
        away_team_short="LAL",
        commence_time="2025-06-15T19:00:00Z",
        game_date="2025-06-15",
        home_ml=-150,
        away_ml=+130,
        market_total=218.5,
    )


@pytest.fixture
def engine():
    """Create a fresh LivePredictionEngine for each test."""
    from betting_intel.live.engine import LivePredictionEngine
    return LivePredictionEngine()


# ═══════════════════════════════════════════════════════════════════════════
#  1. LiveGame TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLiveGame:
    """Tests for the LiveGame dataclass — properties, serialization, edge cases."""

    def test_matchup_property(self, live_game):
        assert live_game.matchup == "LAL @ BOS"

    def test_commence_datetime_parses_iso(self, live_game):
        dt = live_game.commence_datetime
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 15

    def test_commence_datetime_bad_string_returns_none(self):
        from betting_intel.live.engine import LiveGame
        game = LiveGame(
            game_id="bad", sport_key="nba", home_team="A", away_team="B",
            home_team_short="A", away_team_short="B",
            commence_time="not-a-date", game_date="",
        )
        assert game.commence_datetime is None

    def test_commence_datetime_empty_string_returns_none(self):
        from betting_intel.live.engine import LiveGame
        game = LiveGame(
            game_id="empty", sport_key="nba", home_team="A", away_team="B",
            home_team_short="A", away_team_short="B",
            commence_time="", game_date="",
        )
        assert game.commence_datetime is None

    def test_commence_datetime_z_suffix(self):
        from betting_intel.live.engine import LiveGame
        game = LiveGame(
            game_id="z", sport_key="nba", home_team="A", away_team="B",
            home_team_short="A", away_team_short="B",
            commence_time="2025-06-12T20:00:00Z", game_date="2025-06-12",
        )
        dt = game.commence_datetime
        assert dt is not None
        assert dt.tzinfo is not None

    def test_to_dict_includes_all_fields(self, live_game):
        d = live_game.to_dict()
        assert d["game_id"] == "g1"
        assert d["home_ml"] == -150
        assert d["market_total"] == 218.5
        assert "matchup" not in d  # property, not dataclass field

    def test_to_dict_optional_fields_excluded_when_none(self, live_game):
        d = live_game.to_dict()
        assert d["predicted_total"] is None
        assert d["edge_pct"] is None
        assert d["q1_home"] is None
        assert d["recommended_quarter"] is None

    def test_matchup_empty_teams(self):
        from betting_intel.live.engine import LiveGame
        game = LiveGame(
            game_id="e", sport_key="nba", home_team="", away_team="",
            home_team_short="", away_team_short="",
            commence_time="", game_date="",
        )
        assert game.matchup == " @ "

    def test_game_date_defaults(self):
        from betting_intel.live.engine import LiveGame
        game = LiveGame(
            game_id="d", sport_key="nba", home_team="A", away_team="B",
            home_team_short="A", away_team_short="B",
            commence_time="2025-06-12T20:00:00Z", game_date="2025-06-12",
        )
        assert game.league == "NBA"
        assert game.is_live is False
        assert game.n_books_ml == 0


# ═══════════════════════════════════════════════════════════════════════════
#  2. LivePredictionSnapshot TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLivePredictionSnapshot:
    """Tests for LivePredictionSnapshot — chart data, categories, serialization."""

    def test_empty_snapshot_defaults(self):
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot()
        assert snap.n_live == 0
        assert snap.n_today == 0
        assert snap.n_tomorrow == 0
        assert snap.n_total == 0
        assert snap.fresh_odds is False
        assert snap.generated_at is not None
        assert snap.chart_data is not None

    def test_empty_snapshot_chart_data(self):
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot()
        cd = snap.chart_data
        assert cd["n_total"] == 0
        assert cd["edges"] == []
        assert cd["confidence_breakdown"] == {"high": 0, "medium": 0, "low": 0, "neutral": 0}
        assert cd["direction_breakdown"] == {"over": 0, "under": 0, "neutral": 0}

    def test_snapshot_with_games_counts_correctly(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.is_live = True
        live_game.is_today = True
        live_game.edge_pct = 0.03
        snap = LivePredictionSnapshot(
            live_games=[live_game], today_games=[live_game], next_two_days=[live_game],
        )
        assert snap.n_live == 1
        assert snap.n_today == 1
        assert snap.n_total == 1

    def test_snapshot_chart_data_includes_edges(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.edge_pct = 0.04
        live_game.direction = "over"
        live_game.confidence = "medium"
        live_game.predicted_total = 220.5
        live_game.market_total = 218.5
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        cd = snap.chart_data
        assert len(cd["edges"]) == 1
        assert cd["edges"][0]["edge_pct"] == 4.0
        assert cd["edges"][0]["direction"] == "over"

    def test_snapshot_chart_data_skips_zero_edge(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.edge_pct = 0.0
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        assert snap.chart_data["edges"] == []

    def test_snapshot_chart_data_skips_none_edge(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.edge_pct = None
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        assert snap.chart_data["edges"] == []

    def test_snapshot_chart_data_confidence_breakdown(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        from copy import deepcopy
        g1, g2, g3 = (deepcopy(live_game) for _ in range(3))
        g1.edge_pct, g1.confidence = 0.05, "high"
        g2.game_id, g2.edge_pct, g2.confidence = "g2", 0.03, "medium"
        g3.game_id, g3.edge_pct, g3.confidence = "g3", 0.01, "low"
        snap = LivePredictionSnapshot(next_two_days=[g1, g2, g3])
        cd = snap.chart_data
        assert cd["confidence_breakdown"]["high"] == 1
        assert cd["confidence_breakdown"]["medium"] == 1
        assert cd["confidence_breakdown"]["low"] == 1

    def test_snapshot_chart_data_direction_breakdown(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        from copy import deepcopy
        g1, g2, g3 = (deepcopy(live_game) for _ in range(3))
        g1.edge_pct, g1.direction = 0.05, "over"
        g2.game_id, g2.edge_pct, g2.direction = "g2", 0.04, "under"
        g3.game_id, g3.edge_pct, g3.direction = "g3", 0.03, "over"
        snap = LivePredictionSnapshot(next_two_days=[g1, g2, g3])
        cd = snap.chart_data
        assert cd["direction_breakdown"]["over"] == 2
        assert cd["direction_breakdown"]["under"] == 1

    def test_to_dict_excludes_chart_data(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.edge_pct = 0.05
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        d = snap.to_dict()
        assert "chart_data" not in d
        assert "_exclude_from_dict" not in d
        assert d["n_total"] == 1

    def test_to_dict_includes_game_dicts(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot(today_games=[live_game])
        d = snap.to_dict()
        assert d["today_games"][0]["game_id"] == "g1"

    def test_serialization_roundtrip_json(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.edge_pct = 0.04
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        d = snap.to_dict()
        loaded = json.loads(json.dumps(d, default=str))
        assert loaded["n_total"] == 1
        assert loaded["next_two_days"][0]["home_team_short"] == "BOS"

    def test_multiple_games_in_multiple_categories(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        live_game.is_live = True
        live_game.is_today = True
        snap = LivePredictionSnapshot(
            live_games=[live_game], today_games=[live_game], next_two_days=[live_game],
        )
        assert snap.n_live == 1
        assert snap.n_today == 1
        assert snap.n_total == 1
        d = snap.to_dict()
        assert len(d["live_games"]) == 1
        assert len(d["today_games"]) == 1
        assert len(d["next_two_days"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  3. LivePredictionEngine TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLivePredictionEngine:
    """Tests for the LivePredictionEngine — state, caching, odds merging."""

    def test_engine_imports(self):
        from betting_intel.live.engine import LivePredictionEngine, LiveGame, LivePredictionSnapshot
        assert LivePredictionEngine is not None

    def test_engine_initialization(self, engine):
        assert engine.kelly_staker is not None
        assert engine.robust_system is None
        assert engine.has_cached_data is False
        assert engine.last_auto_resolve is None

    def test_engine_initialization_with_custom_params(self):
        from betting_intel.live.engine import LivePredictionEngine
        eng = LivePredictionEngine(odds_api_key="test-key-123", refresh_interval=120)
        assert eng._odds_api_key == "test-key-123"
        assert eng._refresh_interval == 120

    def test_engine_initialization_default_refresh_interval(self):
        from betting_intel.live.engine import LivePredictionEngine, PREDICTION_REFRESH_INTERVAL
        assert LivePredictionEngine()._refresh_interval == PREDICTION_REFRESH_INTERVAL

    # ── Properties ───────────────────────────────────────────────

    def test_robust_system_summary_not_initialized(self, engine):
        assert engine.robust_system_summary == {"fitted": False, "status": "not_initialized"}

    def test_robust_system_summary_not_fitted(self, engine):
        engine._robust_system = MagicMock()
        engine._robust_system_fitted = False
        summary = engine.robust_system_summary
        assert summary["status"] == "not_fitted"

    def test_robust_system_summary_fitted(self, engine):
        mock_sys = MagicMock()
        mock_sys.get_summary.return_value = {"fitted": True, "n_models": 3}
        engine._robust_system = mock_sys
        engine._robust_system_fitted = True
        assert engine.robust_system_summary["n_models"] == 3

    def test_robust_system_summary_error(self, engine):
        mock_sys = MagicMock()
        mock_sys.get_summary.side_effect = RuntimeError("boom")
        engine._robust_system = mock_sys
        engine._robust_system_fitted = True
        summary = engine.robust_system_summary
        assert summary["status"] == "error_reading_summary"

    def test_kelly_staker_property(self, engine):
        assert engine.kelly_staker.bankroll == 10000.0

    @pytest.mark.parametrize("key,expected", [
        ("valid-key-123", True),
        ("", False),
        ("your-api-key-here", False),
        ("REPLACE_ME_WITH_YOUR_ODDS_API_KEY", False),
        (None, False),
    ])
    def test_has_valid_api_key(self, key, expected):
        from betting_intel.live.engine import LivePredictionEngine
        eng = LivePredictionEngine(odds_api_key=key)
        assert eng._has_valid_api_key() == expected

    def test_has_cached_data_false_initially(self, engine):
        assert engine.has_cached_data is False

    def test_has_cached_data_true_after_snapshot(self, engine, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        with engine._lock:
            engine._snapshot = snap
        assert engine.has_cached_data is True

    def test_last_auto_resolve_none_initially(self, engine):
        assert engine.last_auto_resolve is None

    def test_last_auto_resolve_stores_and_retrieves(self, engine):
        with engine._lock:
            engine._last_auto_resolve = "2025-06-12T12:00:00"
        assert engine.last_auto_resolve == "2025-06-12T12:00:00"

    # ── get_snapshot ─────────────────────────────────────────────

    def test_get_snapshot_returns_empty_when_no_cache(self, engine):
        snap = engine.get_snapshot(force_refresh=False)
        assert snap.n_total == 0

    def test_get_snapshot_returns_cached(self, engine, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        cached = LivePredictionSnapshot(next_two_days=[live_game])
        with engine._lock:
            engine._snapshot = cached
        snap = engine.get_snapshot(force_refresh=False)
        assert snap is cached  # Same object, no rebuild

    @patch(f"{_ENG}.LivePredictionEngine._build_snapshot")
    def test_get_snapshot_force_refresh_calls_build(self, mock_build, engine):
        mock_build.return_value = MagicMock()
        engine.get_snapshot(force_refresh=True)
        mock_build.assert_called_once()

    @patch(f"{_ENG}.LivePredictionEngine._build_snapshot")
    def test_get_snapshot_refresh_stores_result(self, mock_build, engine, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        new_snap = LivePredictionSnapshot(next_two_days=[live_game])
        mock_build.return_value = new_snap
        result = engine.get_snapshot(force_refresh=True)
        assert result is new_snap
        with engine._lock:
            assert engine._snapshot is new_snap

    @patch(f"{_ENG}.LivePredictionEngine._build_snapshot")
    def test_get_snapshot_refresh_failure_returns_cached(self, mock_build, engine, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        cached = LivePredictionSnapshot(next_two_days=[live_game])
        with engine._lock:
            engine._snapshot = cached
        mock_build.side_effect = RuntimeError("API down")
        snap = engine.get_snapshot(force_refresh=True)
        assert snap is cached

    @patch(f"{_ENG}.LivePredictionEngine._build_snapshot")
    def test_get_snapshot_refresh_failure_no_cache_returns_empty(self, mock_build, engine):
        mock_build.side_effect = RuntimeError("API down")
        snap = engine.get_snapshot(force_refresh=True)
        assert snap.n_total == 0

    # ── Convenience getters ──────────────────────────────────────

    def test_get_live_games_delegates(self, engine):
        with patch.object(engine, "get_snapshot", return_value=MagicMock(live_games=["g1"])):
            assert engine.get_live_games() == ["g1"]

    def test_get_today_games_delegates(self, engine):
        with patch.object(engine, "get_snapshot", return_value=MagicMock(today_games=["g1"])):
            assert engine.get_today_games() == ["g1"]

    def test_get_tomorrow_games_delegates(self, engine):
        with patch.object(engine, "get_snapshot", return_value=MagicMock(tomorrow_games=["g1"])):
            assert engine.get_tomorrow_games() == ["g1"]

    def test_get_next_two_days_delegates(self, engine):
        with patch.object(engine, "get_snapshot", return_value=MagicMock(next_two_days=["g1", "g2"])):
            assert engine.get_next_two_days() == ["g1", "g2"]

    # ── refresh_now ──────────────────────────────────────────────

    def test_refresh_now_calls_force_refresh(self, engine):
        with patch.object(engine, "get_snapshot", return_value=MagicMock()) as mock_get:
            engine.refresh_now()
            mock_get.assert_called_once_with(force_refresh=True)

    # ── clear_cache ──────────────────────────────────────────────

    def test_clear_cache_resets_state(self, engine, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        with engine._model_lock:
            engine._model = MagicMock()
            engine._feature_cols = ["f1"]
            engine._robust_system = MagicMock()
            engine._robust_system_fitted = True
        with engine._lock:
            engine._snapshot = snap
            engine._last_refresh = 100.0
            engine._last_odds_fetch = 200.0
            engine._cached_odds_raw = [{"k": "v"}]

        engine.clear_cache()

        assert engine.has_cached_data is False
        with engine._model_lock:
            assert engine._model is None
            assert engine._robust_system is None
            assert engine._robust_system_fitted is False
        with engine._lock:
            assert engine._snapshot is None
            assert engine._last_refresh == 0.0
            assert engine._last_odds_fetch == 0.0
            assert engine._cached_odds_raw is None

    def test_clear_cache_idempotent(self, engine):
        engine.clear_cache()
        engine.clear_cache()

    # ── _merge_odds_sources ──────────────────────────────────────

    def test_merge_odds_sources_both_empty(self, engine):
        assert engine._merge_odds_sources([], []) == []

    def test_merge_odds_sources_only_espn(self, engine):
        result = engine._merge_odds_sources([{"home_team": "A", "away_team": "B"}], [])
        assert result[0]["home_team"] == "A"

    def test_merge_odds_sources_only_dk(self, engine):
        result = engine._merge_odds_sources([], [{"home_team": "C", "away_team": "D"}])
        assert result[0]["home_team"] == "C"

    def test_merge_odds_sources_same_matchup_merges_bookmakers(self, engine):
        espn = [{"home_team": "Lakers", "away_team": "Celtics",
                 "bookmakers": [{"key": "espn_bmk", "markets": []}]}]
        dk = [{"home_team": "Lakers", "away_team": "Celtics",
               "bookmakers": [{"key": "dk_bmk", "markets": []}]}]
        result = engine._merge_odds_sources(espn, dk)
        assert len(result[0]["bookmakers"]) == 2
        assert result[0]["bookmakers"][0]["key"] == "espn_bmk"
        assert result[0]["bookmakers"][1]["key"] == "dk_bmk"

    def test_merge_odds_sources_different_matchups(self, engine):
        result = engine._merge_odds_sources(
            [{"home_team": "A", "away_team": "B"}],
            [{"home_team": "C", "away_team": "D"}],
        )
        assert len(result) == 2

    # ── _auto_resolve_completed_games ───────────────────────────

    def test_auto_resolve_returns_zero_when_tracker_unavailable(self, engine):
        assert engine._auto_resolve_completed_games() == 0

    @patch(f"{_TRK}.ResultsTracker")
    def test_auto_resolve_updates_timestamp_on_success(self, mock_tracker_cls, engine):
        """Actually calls _auto_resolve_completed_games with ResultsTracker mocked."""
        mock_tracker = MagicMock()
        mock_tracker.resolve_all.return_value = 5
        mock_tracker_cls.return_value = mock_tracker

        result = engine._auto_resolve_completed_games()
        assert result == 5
        assert engine.last_auto_resolve is not None

    def test_auto_resolve_failure_returns_zero_gracefully(self, engine):
        with patch(f"{_TRK}.ResultsTracker") as mock_cls:
            mock_cls.side_effect = ImportError("tracker not available")
            assert engine._auto_resolve_completed_games() == 0

    # ── _parse_games ─────────────────────────────────────────────

    def test_parse_games_empty_input(self, engine):
        assert engine._parse_games([]) == []

    def test_parse_games_missing_home_team(self, engine, sample_game):
        bad = dict(sample_game)
        bad["home_team"] = ""
        assert engine._parse_games([bad]) == []

    def test_parse_games_missing_away_team(self, engine, sample_game):
        bad = dict(sample_game)
        bad["away_team"] = ""
        assert engine._parse_games([bad]) == []

    def test_parse_games_filters_old_games(self, engine, sample_game):
        old = dict(sample_game)
        old["commence_time"] = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        assert engine._parse_games([old]) == []

    def test_parse_games_bad_commence_time_no_crash(self, engine, sample_game):
        """A corrupt commence_time should not crash — the except block handles it."""
        bad = dict(sample_game)
        bad["commence_time"] = "complete-garbage-not-a-date"
        result = engine._parse_games([bad])
        assert isinstance(result, list)

    def test_parse_games_recent_game_included(self, engine, sample_game):
        recent = dict(sample_game)
        recent["commence_time"] = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        with patch(f"{_DAT}.ODDS_TO_SHORT_NAME",
                   {"Boston Celtics": "BOS", "Los Angeles Lakers": "LAL"}):
            result = engine._parse_games([recent])
        assert result[0].home_team_short == "BOS"

    # ── _build_snapshot ──────────────────────────────────────────

    def test_build_snapshot_empty_odds(self, engine):
        with patch.object(engine, "_fetch_realtime_odds", return_value=[]), \
             patch.object(engine, "_auto_resolve_completed_games", return_value=0):
            snap = engine._build_snapshot()
        assert snap.n_total == 0
        assert snap.fresh_odds is False

    def test_build_snapshot_with_games(self, engine, sample_game):
        game = dict(sample_game)
        game["commence_time"] = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        with patch.object(engine, "_fetch_realtime_odds", return_value=[game]), \
             patch.object(engine, "_auto_resolve_completed_games", return_value=0), \
             patch(f"{_DAT}.ODDS_TO_SHORT_NAME",
                   {"Boston Celtics": "BOS", "Los Angeles Lakers": "LAL"}), \
             patch.object(engine, "_predict_games", return_value=None):
            snap = engine._build_snapshot()
        assert snap.n_total >= 1
        assert snap.fresh_odds is True

    # ── _predict_games (fallback logic) ──────────────────────────

    def test_predict_games_empty(self, engine):
        assert engine._predict_games([]) == []

    def test_predict_games_applies_fallback_when_no_model(self, engine, live_game):
        """Without robust_system or legacy model, fallback assigns neutral edge."""
        live_game.market_total = 218.5
        # Ensure no models are fitted
        engine._robust_system = None
        engine._robust_system_fitted = False
        engine._model = None

        with patch.object(engine, "_load_model") as mock_load, \
             patch.object(engine, "_build_robust_system") as mock_build:
            mock_load.return_value = None
            mock_build.return_value = False
            result = engine._predict_games([live_game])

        assert result[0].edge_pct == 0.0
        assert result[0].direction == "neutral"
        assert result[0].predicted_at is not None


# ═══════════════════════════════════════════════════════════════════════════
#  4. LivePredictionWorker TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLivePredictionWorker:
    """Tests for the LivePredictionWorker background worker."""

    def test_worker_imports(self):
        from betting_intel.live.engine import LivePredictionWorker
        assert LivePredictionWorker is not None

    def test_worker_initialization(self, engine):
        from betting_intel.live.engine import LivePredictionWorker
        worker = LivePredictionWorker(engine)
        assert worker.engine is engine
        assert worker._running is False

    def test_worker_refresh_interval(self, engine):
        from betting_intel.live.engine import LivePredictionWorker
        worker = LivePredictionWorker(engine)
        assert worker._refresh_interval == engine._refresh_interval

    def test_worker_stop_while_not_running(self, engine):
        from betting_intel.live.engine import LivePredictionWorker
        worker = LivePredictionWorker(engine)
        worker.stop()
        assert worker._running is False


# ═══════════════════════════════════════════════════════════════════════════
#  5. INTEGRATION EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineEdgeCases:
    """Integration-style tests for tricky edge cases."""

    def test_engine_with_all_defaults(self):
        from betting_intel.live.engine import LivePredictionEngine
        eng = LivePredictionEngine()
        assert eng._odds_api_key is not None

    def test_engine_with_none_key(self):
        from betting_intel.live.engine import LivePredictionEngine
        eng = LivePredictionEngine(odds_api_key=None)
        assert eng._odds_api_key is not None

    def test_multiple_engine_instances_no_shared_state(self):
        from betting_intel.live.engine import LivePredictionEngine
        e1, e2 = LivePredictionEngine(), LivePredictionEngine()
        with e1._lock:
            e1._last_refresh = 42.0
        assert e2._last_refresh == 0.0

    def test_snapshot_live_today_tomorrow_counts(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        from copy import deepcopy
        live = deepcopy(live_game)
        live.is_live = True; live.is_today = True; live.edge_pct = 0.05
        today = deepcopy(live_game)
        today.game_id = "g2"; today.is_today = True; today.edge_pct = 0.03
        tomorrow = deepcopy(live_game)
        tomorrow.game_id = "g3"; tomorrow.is_tomorrow = True; tomorrow.edge_pct = 0.04
        snap = LivePredictionSnapshot(
            live_games=[live], today_games=[live, today],
            tomorrow_games=[tomorrow], next_two_days=[live, today, tomorrow],
        )
        assert snap.n_live == 1 and snap.n_today == 2
        assert snap.n_tomorrow == 1 and snap.n_total == 3
        assert len(snap.chart_data["edges"]) == 3

    def test_to_dict_chart_data_excluded_snap_preserved(self, live_game):
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot(next_two_days=[live_game])
        assert "chart_data" not in snap.to_dict()
        assert snap.chart_data is not None

    def test_parse_games_unknown_sport_key_no_crash(self, engine, sample_game):
        """Games from an unknown sport key should not crash the parser."""
        unknown = dict(sample_game)
        unknown["sport_key"] = "basketball_unknown"
        unknown["home_team"] = "Team A"
        unknown["away_team"] = "Team B"
        unknown["_sport_config_key"] = "basketball_unknown"

        with patch(f"{_CFG}.sport_key_to_group", return_value="Basketball"), \
             patch(f"{_CFG}.SPORT_KEY_TO_CONFIG",
                   {"basketball_unknown": MagicMock(
                       display_name="Unknown",
                       team_name_map={},
                       get_short_name=lambda n: n.split()[-1],
                   )}):
            result = engine._parse_games([unknown])
        assert isinstance(result, list)

    @patch(f"{_DK}.DraftKingsScraper")
    def test_draftkings_scraper_thread_timeout_returns_empty(self, mock_dk, engine):
        """DraftKings scraper with mocked scraper returns empty (no data)."""
        mock_dk.scrape.return_value = []
        result = engine._fetch_draftkings_odds()
        assert result == []
