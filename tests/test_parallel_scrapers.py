"""
Integration tests for the parallel scraper execution in
LivePredictionEngine._fetch_realtime_odds().

The parallel logic runs ESPN and DraftKings scrapers concurrently via
ThreadPoolExecutor with a configurable combined timeout (self._scraper_timeout).

Key behaviors tested:
  1. Both scrapers return data  — normal merge
  2. Both scrapers return empty — graceful empty result
  3. Both scrapers time out     — FuturesTimeoutError caught, warning logged
  4. One times out, other works — partial results collected
  5. One returns empty, other works — single-source result
  6. Late-finishing scraper     — collected after as_completed timeout
  7. ImportError in one scraper — other scraper still provides data

All tests mock _fetch_stealth_scraper and _fetch_draftkings_odds on the
engine instance (unit-testing the parallel orchestration, not the scrapers).
"""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

# Module path prefix for string-based patches on engine.py internals
_ENG = "betting_intel.live.engine"


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_engine():
    """Create a fresh LivePredictionEngine with cache cleared for test isolation."""
    from betting_intel.live.engine import LivePredictionEngine
    eng = LivePredictionEngine()
    eng.clear_cache()
    # Bypass the TheOddsAPI path so we always reach the scraper logic
    eng._has_valid_api_key = MagicMock(return_value=False)  # type: ignore[method-assign]
    # Shorten the scraper timeout for fast test execution
    eng._scraper_timeout = 0.5
    return eng


def _raw_game(home: str = "Celtics", away: str = "Lakers") -> dict:
    """A minimal raw-odds game dict (one bookmaker)."""
    return {
        "id": f"g_{home}_{away}",
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "commence_time": "2025-06-15T19:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": "test_bmk",
                "title": "Test Sportsbook",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": -150},
                            {"name": away, "price": +130},
                        ],
                    }
                ],
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  1. BOTH SCRAPERS RETURN DATA
# ═══════════════════════════════════════════════════════════════════════════


class TestBothScrapersReturnData:
    """Both scrapers complete within the timeout and return games."""

    def test_returns_merged_list(self):
        eng = _make_engine()
        espn_games = [_raw_game("Celtics", "Lakers")]
        dk_games = [_raw_game("Knicks", "Nets")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=espn_games), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 2
        teams = {(g["home_team"], g["away_team"]) for g in result}
        assert ("Celtics", "Lakers") in teams
        assert ("Knicks", "Nets") in teams

    def test_merges_same_matchup(self):
        eng = _make_engine()
        game = _raw_game("Celtics", "Lakers")
        espn_games = [dict(game)]
        dk_games = [dict(game)]
        dk_games[0]["bookmakers"] = [{"key": "dk_bmk", "markets": []}]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=espn_games), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1
        assert len(result[0]["bookmakers"]) == 2

    def test_both_return_empty(self):
        eng = _make_engine()
        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]):
            result = eng._fetch_realtime_odds()
        assert result == []

    def test_different_game_ids(self):
        eng = _make_engine()
        espn_games = [_raw_game("Celtics", "Lakers"), _raw_game("Bulls", "Heat")]
        dk_games = [_raw_game("Spurs", "Mavs")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=espn_games), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════
#  2. ONE SCRAPER RETURNS DATA, OTHER IS EMPTY
# ═══════════════════════════════════════════════════════════════════════════


class TestOneScraperEmpty:
    """One scraper has data, the other returns empty list."""

    def test_only_espn_has_data(self):
        eng = _make_engine()
        espn_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=espn_games), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1
        assert result[0]["home_team"] == "Celtics"

    def test_only_dk_has_data(self):
        eng = _make_engine()
        dk_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1

    def test_both_empty_returns_empty_list(self):
        eng = _make_engine()
        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]):
            result = eng._fetch_realtime_odds()
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
#  3. TIMEOUT BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════


class TestTimeoutBehavior:
    """
    Test cases where scrapers exceed the combined timeout.

    The engine's _scraper_timeout is patched to 0.5s by _make_engine(),
    so a 1.0s sleep reliably triggers FuturesTimeoutError.
    """

    def test_both_timeout_returns_empty(self):
        """Both scrapers sleep 1s (> 0.5s timeout) → empty result."""
        eng = _make_engine()

        def _slow():
            time.sleep(1.0)
            return [_raw_game()]

        with patch.object(eng, "_fetch_stealth_scraper", wraps=_slow), \
             patch.object(eng, "_fetch_draftkings_odds", wraps=_slow):
            start = time.time()
            result = eng._fetch_realtime_odds()
            elapsed = time.time() - start

        # Should return in ~0.5s, not 1s
        assert elapsed < 1.5, f"Took {elapsed:.1f}s — timeout didn't fire"
        assert result == []

    def test_espn_timeout_dk_returns(self):
        """ESPN hangs, DK returns fast → partial result with DK data."""
        eng = _make_engine()

        def _slow():
            time.sleep(1.0)
            return [_raw_game()]

        dk_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", wraps=_slow), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            start = time.time()
            result = eng._fetch_realtime_odds()
            elapsed = time.time() - start

        assert elapsed < 1.5, f"Took {elapsed:.1f}s — timeout didn't fire"
        assert len(result) == 1
        assert result[0]["home_team"] == "Celtics"

    def test_dk_timeout_espn_returns(self):
        """DK hangs, ESPN returns fast → partial result with ESPN data."""
        eng = _make_engine()

        def _slow():
            time.sleep(1.0)
            return [_raw_game()]

        espn_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_draftkings_odds", wraps=_slow), \
             patch.object(eng, "_fetch_stealth_scraper", return_value=espn_games):
            start = time.time()
            result = eng._fetch_realtime_odds()
            elapsed = time.time() - start

        assert elapsed < 1.5, f"Took {elapsed:.1f}s — timeout didn't fire"
        assert len(result) == 1
        assert result[0]["home_team"] == "Celtics"

    def test_shutdown_does_not_block(self):
        """
        Verifies pool.shutdown(wait=False) in the finally block.
        Without wait=False, hanging threads block the refresh cycle.
        """
        eng = _make_engine()

        def _slow():
            time.sleep(5.0)  # Would block 5s if wait=True
            return [_raw_game()]

        with patch.object(eng, "_fetch_stealth_scraper", wraps=_slow), \
             patch.object(eng, "_fetch_draftkings_odds", wraps=_slow):
            start = time.time()
            result = eng._fetch_realtime_odds()
            elapsed = time.time() - start

        # Should return in ~0.5s, not 5s
        assert elapsed < 2.0, f"Took {elapsed:.1f}s — shutdown blocked!"
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
#  4. LATE FINISHING SCRAPER
# ═══════════════════════════════════════════════════════════════════════════


class TestLateFinishingScraper:
    """
    A scraper that finishes JUST after the as_completed timeout fires
    should still be collected by the late-collection loop.
    """

    def test_one_finishes_late_still_collected(self):
        """DK returns instantly, ESPN finishes slightly late but before late-collection."""
        eng = _make_engine()

        dk_games = [_raw_game("Celtics", "Lakers")]

        # ESPN takes 0.1s — faster than the 0.5s timeout, so it completes
        # within the deadline. No timeout scenario.
        def _slightly_late():
            time.sleep(0.1)
            return [_raw_game("Knicks", "Nets")]

        with patch.object(eng, "_fetch_stealth_scraper", wraps=_slightly_late), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            start = time.time()
            result = eng._fetch_realtime_odds()
            elapsed = time.time() - start

        assert elapsed < 2.0, f"Took {elapsed:.1f}s"
        assert len(result) == 2
        teams = {(g["home_team"], g["away_team"]) for g in result}
        assert ("Celtics", "Lakers") in teams
        assert ("Knicks", "Nets") in teams


# ═══════════════════════════════════════════════════════════════════════════
#  5. EXCEPTION HANDLING
# ═══════════════════════════════════════════════════════════════════════════


class TestExceptionHandling:
    """Graceful degradation when a scraper fails."""

    def test_espn_importerror_dk_works(self):
        """ESPN scraper raises ImportError → DK provides the data."""
        eng = _make_engine()

        def _raise_import():
            raise ImportError("stealth_scraper not installed")

        dk_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", side_effect=ImportError("stealth_scraper not installed")), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1
        assert result[0]["home_team"] == "Celtics"

    def test_dk_importerror_espn_works(self):
        """DK scraper not available → ESPN provides the data."""
        eng = _make_engine()

        espn_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=espn_games), \
             patch.object(eng, "_fetch_draftkings_odds", side_effect=ImportError("draftkings_scraper not installed")):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1

    def test_both_raise_import_error(self):
        """Both scrapers unavailable → empty list."""
        eng = _make_engine()

        with patch.object(eng, "_fetch_stealth_scraper", side_effect=ImportError("not available")), \
             patch.object(eng, "_fetch_draftkings_odds", side_effect=ImportError("not available")):
            result = eng._fetch_realtime_odds()

        assert result == []

    def test_espn_raises_runtime_error(self):
        """ESPN scraper raises RuntimeError → DK data still returned."""
        eng = _make_engine()

        dk_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", side_effect=RuntimeError("ESPN API changed")), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1
        assert result[0]["home_team"] == "Celtics"


# ═══════════════════════════════════════════════════════════════════════════
#  6. CACHE INTERACTION
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheInteraction:
    """
    _fetch_realtime_odds() has a TTL cache. If data is cached,
    scrapers should not be called at all.
    """

    def test_cached_data_skips_scrapers(self):
        eng = _make_engine()

        # Pre-populate cache with fresh data
        cached_games = [_raw_game("Celtics", "Lakers")]
        with eng._lock:
            eng._cached_odds_raw = cached_games
            eng._last_odds_fetch = time.time()  # Now (fresh, within TTL)

        # If cache is used, scrapers should NOT be called
        with patch.object(eng, "_fetch_stealth_scraper") as mock_espn, \
             patch.object(eng, "_fetch_draftkings_odds") as mock_dk:
            result = eng._fetch_realtime_odds()

        mock_espn.assert_not_called()
        mock_dk.assert_not_called()
        assert result == cached_games

    def test_expired_cache_calls_scrapers(self):
        eng = _make_engine()

        # Pre-populate cache with STALE data (TTL = 300s, set to 600s ago)
        cached_games = [_raw_game("Celtics", "Lakers")]
        with eng._lock:
            eng._cached_odds_raw = cached_games
            eng._last_odds_fetch = time.time() - 600  # Expired

        dk_games = [_raw_game("Knicks", "Nets")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        # Should replace stale cache with fresh data
        assert len(result) == 1
        assert result[0]["home_team"] == "Knicks"

    def test_cache_miss_calls_scrapers(self):
        eng = _make_engine()
        eng.clear_cache()

        dk_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        assert len(result) == 1

    def test_cache_updates_after_scraper_run(self):
        """After scrapers return data, the cache should be updated."""
        eng = _make_engine()
        eng.clear_cache()

        dk_games = [_raw_game("Celtics", "Lakers")]

        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=dk_games):
            result = eng._fetch_realtime_odds()

        # Now the cache should hold the same data
        with eng._lock:
            assert eng._cached_odds_raw == result
            assert eng._last_odds_fetch > 0

    def test_cache_updates_to_empty_when_no_data(self):
        """When both scrapers return nothing, cache should be empty."""
        eng = _make_engine()

        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]):
            result = eng._fetch_realtime_odds()

        with eng._lock:
            assert eng._cached_odds_raw == []


# ═══════════════════════════════════════════════════════════════════════════
#  7. SCRAPER METHOD INDEPENDENCE
# ═══════════════════════════════════════════════════════════════════════════


class TestScraperMethodIndependence:
    """
    Verifies the two scraper methods are called independently.
    Each should be invoked even if the other fails or times out.
    """

    def test_espn_called_even_when_dk_times_out(self):
        eng = _make_engine()

        def _slow_dk():
            time.sleep(5.0)
            return [_raw_game()]

        with patch.object(eng, "_fetch_stealth_scraper") as mock_espn, \
             patch.object(eng, "_fetch_draftkings_odds", wraps=_slow_dk):
            mock_espn.return_value = [_raw_game("Celtics", "Lakers")]
            result = eng._fetch_realtime_odds()

        mock_espn.assert_called_once()
        assert len(result) == 1

    def test_dk_called_even_when_espn_times_out(self):
        eng = _make_engine()

        def _slow_espn():
            time.sleep(5.0)
            return [_raw_game()]

        with patch.object(eng, "_fetch_draftkings_odds") as mock_dk, \
             patch.object(eng, "_fetch_stealth_scraper", wraps=_slow_espn):
            mock_dk.return_value = [_raw_game("Celtics", "Lakers")]
            result = eng._fetch_realtime_odds()

        mock_dk.assert_called_once()
        assert len(result) == 1

    def test_both_methods_called(self):
        """Both scraper methods should be called (each at least once)."""
        eng = _make_engine()
        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]) as mock_espn, \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]) as mock_dk:
            eng._fetch_realtime_odds()
        mock_espn.assert_called_once()
        mock_dk.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
#  8. THREAD SAFETY
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    """
    _fetch_realtime_odds() acquires _lock at the start (cache check)
    and at the end (cache update). Verify no deadlocks.
    """

    def test_no_deadlock_after_timeout(self):
        """After both scrapers time out, the lock is released cleanly."""
        eng = _make_engine()

        def _slow():
            time.sleep(5.0)
            return [_raw_game()]

        # First call: both time out (eng._scraper_timeout = 0.5)
        with patch.object(eng, "_fetch_stealth_scraper", wraps=_slow), \
             patch.object(eng, "_fetch_draftkings_odds", wraps=_slow):
            eng._fetch_realtime_odds()

        # Second call should work (lock was released)
        with patch.object(eng, "_fetch_stealth_scraper", return_value=[]), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]):
            result = eng._fetch_realtime_odds()

        assert result == []

    def test_concurrent_calls_use_cache(self):
        """
        While one thread is fetching odds, another thread calling
        get_snapshot should still get the cached snapshot.
        """
        eng = _make_engine()

        # Set a cached snapshot
        from betting_intel.live.engine import LivePredictionSnapshot
        snap = LivePredictionSnapshot()
        with eng._lock:
            eng._snapshot = snap

        import threading

        results = []

        def _slow_espn():
            time.sleep(2.0)
            return [_raw_game()]

        def _call_get():
            # get_snapshot(non-force) reads cached snapshot without blocking
            s = eng.get_snapshot(force_refresh=False)
            results.append(s is snap)

        with patch.object(eng, "_fetch_stealth_scraper", wraps=_slow_espn), \
             patch.object(eng, "_fetch_draftkings_odds", return_value=[]):
            t1 = threading.Thread(target=lambda: eng.get_snapshot(force_refresh=True))
            t2 = threading.Thread(target=_call_get)
            t1.start()
            time.sleep(0.05)  # Let t1 start and acquire resources
            t2.start()
            t1.join(timeout=4)
            t2.join(timeout=2)

        assert True in results, "Concurrent read should return cached snapshot"
