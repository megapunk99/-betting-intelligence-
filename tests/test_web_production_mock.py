"""
Production-style integration test.

Mocks the ENTIRE odds fetching chain (TheOddsAPI -> stealth_scraper -> undetectable_scraper)
and verifies that every web page and JSON API endpoint renders realistic game data
as if live odds were actually flowing in.

What this tests:
  1. The LivePredictionEngine correctly processes realistic TheOddsAPI-format data
  2. Every HTML page renders team names, odds, and edges without errors
  3. Every JSON API endpoint returns the expected data structure
  4. Chart data is correctly computed from game data
  5. Adapter functions (_livegame_to_bet, _livegame_to_clear_pick, etc.) work end-to-end
  6. Graceful degradation when no odds data is available (empty state)
  7. Graceful degradation when the engine fails to initialize
  8. State isolation: each test class gets a clean engine via clear_cache()

ARCHITECTURE:
  Each test class uses a class-scoped autouse fixture that:
    1. Patches LivePredictionEngine._fetch_realtime_odds with appropriate return/side-effect
    2. Calls POST /api/live/clear-cache to reset any stale state from previous classes
    3. Calls POST /api/live/refresh to populate the engine with patched data
    4. Yields (keeping the patch active for all tests in the class)
    5. The with-block cleanup tears down the patch after the class completes

  This ensures each test class has isolated, predictable state regardless of
  test execution order.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

# Apply the 'integration' marker to ALL tests in this module so CI can
# run them as a separate, explicit step alongside the main test matrix.
pytestmark = pytest.mark.integration

# Ensure src is on the path (web.app does this at runtime)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
#  REALISTIC SAMPLE DATA — matches real TheOddsAPI response format
#
#  NOTE: All commence_time values are set to +6 hours from now to ensure the
#  _parse_games age filter (games older than 3 hours are skipped) never
#  filters them out, regardless of when the tests run.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sample_games():
    """Create a realistic 4-game NBA slate (2 today + 2 tomorrow) with full odds.

    All commence times are offset +6h from right now so the 3-hour age filter
    in _parse_games() never fires — games always appear fresh.
    """
    now = datetime.now(timezone.utc)

    # Commence time strings at +6h from now so they're always in the future
    t0 = (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:00Z")
    t1 = (now + timedelta(hours=7, minutes=30)).strftime("%Y-%m-%dT%H:%M:00Z")
    t2 = (now + timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:00Z")
    t3 = (now + timedelta(days=1, hours=6)).strftime("%Y-%m-%dT%H:%M:00Z")
    t4 = (now + timedelta(days=1, hours=8)).strftime("%Y-%m-%dT%H:%M:00Z")

    return [
        # -- Game 1: Today, marquee matchup (2 bookmakers: DraftKings + FanDuel) --
        {
            "id": "game_celtics_lakers",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t0,
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": t0,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Boston Celtics", "price": -200},
                            {"name": "Los Angeles Lakers", "price": 175},
                        ]},
                        {"key": "spreads", "outcomes": [
                            {"name": "Boston Celtics", "point": -4.5, "price": -110},
                            {"name": "Los Angeles Lakers", "point": 4.5, "price": -110},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 218.5, "price": -110},
                            {"name": "Under", "point": 218.5, "price": -110},
                        ]},
                    ],
                },
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": t0,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Boston Celtics", "price": -195},
                            {"name": "Los Angeles Lakers", "price": 170},
                        ]},
                        {"key": "spreads", "outcomes": [
                            {"name": "Boston Celtics", "point": -4.5, "price": -110},
                            {"name": "Los Angeles Lakers", "point": 4.5, "price": -110},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 218.5, "price": -115},
                            {"name": "Under", "point": 218.5, "price": -105},
                        ]},
                    ],
                },
            ],
        },
        # -- Game 2: Today, second game (1 bookmaker: DraftKings) --
        {
            "id": "game_warriors_nuggets",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t1,
            "home_team": "Golden State Warriors",
            "away_team": "Denver Nuggets",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": t1,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Golden State Warriors", "price": -110},
                            {"name": "Denver Nuggets", "price": -110},
                        ]},
                        {"key": "spreads", "outcomes": [
                            {"name": "Golden State Warriors", "point": -1.5, "price": -110},
                            {"name": "Denver Nuggets", "point": 1.5, "price": -110},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 229.0, "price": -110},
                            {"name": "Under", "point": 229.0, "price": -110},
                        ]},
                    ],
                }
            ],
        },
        # -- Game 3: Tomorrow, marquee --
        {
            "id": "game_bucks_thunder",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t3,
            "home_team": "Milwaukee Bucks",
            "away_team": "Oklahoma City Thunder",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": t3,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Milwaukee Bucks", "price": -240},
                            {"name": "Oklahoma City Thunder", "price": 200},
                        ]},
                        {"key": "spreads", "outcomes": [
                            {"name": "Milwaukee Bucks", "point": -5.5, "price": -110},
                            {"name": "Oklahoma City Thunder", "point": 5.5, "price": -110},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 222.0, "price": -110},
                            {"name": "Under", "point": 222.0, "price": -110},
                        ]},
                    ],
                }
            ],
        },
        # -- Game 4: Tomorrow, second game --
        {
            "id": "game_knicks_heat",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t4,
            "home_team": "New York Knicks",
            "away_team": "Miami Heat",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": t4,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "New York Knicks", "price": -165},
                            {"name": "Miami Heat", "price": 145},
                        ]},
                        {"key": "spreads", "outcomes": [
                            {"name": "New York Knicks", "point": -2.5, "price": -110},
                            {"name": "Miami Heat", "point": 2.5, "price": -110},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 212.0, "price": -110},
                            {"name": "Under", "point": 212.0, "price": -110},
                        ]},
                    ],
                }
            ],
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES — Shared at module scope
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient — uses ASGI directly (no real server process, module-scoped for speed)."""
    import os
    from web.app import app
    api_key = os.environ.get("API_KEY", "change-me-to-a-random-secret")
    with TestClient(app, headers={"X-API-Key": api_key}) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════
#  HAPPY PATH — HTML Pages render with mocked game data
#
#   Uses class-scoped autouse fixture for state isolation:
#     1. Clear any stale cache from previous test classes
#     2. Patch _fetch_realtime_odds to return sample data
#     3. Refresh to populate the engine with mock data
#     4. All tests in the class share this state
#     5. Patch is torn down automatically after the last test
# ═══════════════════════════════════════════════════════════════════════════

class TestHtmlPagesHappyPath:
    """All HTML pages render 200 with correct team names from mock data."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, sample_games):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=sample_games):
            client.post("/api/live/clear-cache")
            resp = client.post("/api/live/refresh")
            assert resp.status_code == 200, f"Refresh failed: {resp.json()}"
            yield

    def test_landing_page_renders(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_dashboard_renders_with_todays_games(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        html = resp.text
        assert "Celtics" in html
        assert "Lakers" in html
        assert "Warriors" in html
        assert "Nuggets" in html

    def test_live_page_renders(self, client: TestClient):
        resp = client.get("/live")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_todays_card_renders_with_teams(self, client: TestClient):
        resp = client.get("/todays-card")
        assert resp.status_code == 200
        assert "Celtics" in resp.text or "Lakers" in resp.text

    def test_tomorrow_page_renders_with_teams(self, client: TestClient):
        resp = client.get("/tomorrow")
        assert resp.status_code == 200
        assert "Bucks" in resp.text or "Thunder" in resp.text or "Knicks" in resp.text

    def test_pre_match_prediction_renders(self, client: TestClient):
        resp = client.get("/pre-match-prediction")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_all_bets_renders(self, client: TestClient):
        resp = client.get("/all-bets")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_clear_picks_renders(self, client: TestClient):
        resp = client.get("/clear-picks")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_no_synthetic_teams_appear(self, client: TestClient):
        """Only teams from our sample data should render, not random/fake ones."""
        resp = client.get("/todays-card")
        html = resp.text
        assert "Celtics" in html
        assert "Warriors" in html
        assert "Raptors" not in html
        assert "Spurs" not in html


# ═══════════════════════════════════════════════════════════════════════════
#  HAPPY PATH — JSON API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

class TestJsonApiHappyPath:
    """JSON API endpoints return correct data structures with mock data."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, sample_games):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=sample_games):
            client.post("/api/live/clear-cache")
            resp = client.post("/api/live/refresh")
            assert resp.status_code == 200
            yield

    def test_live_snapshot_contains_all_games(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 4
        assert data["n_today"] == 2
        assert data["n_tomorrow"] == 2
        assert data["fresh_odds"] is True

        game = data["next_two_days"][0]
        assert "home_team" in game
        assert "away_team" in game
        assert "home_team_short" in game
        assert "away_team_short" in game

    def test_live_chart_data_has_correct_counts(self, client: TestClient):
        resp = client.get("/api/live/chart-data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 4
        assert data["n_today"] == 2
        assert data["n_tomorrow"] == 2
        assert "edges" in data
        assert "confidence_breakdown" in data
        assert "direction_breakdown" in data

    def test_live_games_returns_list(self, client: TestClient):
        resp = client.get("/api/live/games")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "game_id" in data[0]
            assert "home_team" in data[0]

    def test_refresh_is_idempotent(self, client: TestClient):
        resp1 = client.post("/api/live/refresh")
        d1 = resp1.json()
        resp2 = client.post("/api/live/refresh")
        d2 = resp2.json()
        assert d1["n_total"] == d2["n_total"]
        assert d1["n_today"] == d2["n_today"]
        assert d1["n_tomorrow"] == d2["n_tomorrow"]

    def test_bets_api_returns_bet_dicts(self, client: TestClient):
        resp = client.get("/api/bets")
        assert resp.status_code == 200
        bets = resp.json()
        assert isinstance(bets, list)
        if len(bets) > 0:
            bet = bets[0]
            for key in ("game_id", "game_date", "matchup", "league", "bet_type",
                        "bet_type_display", "bet_side", "edge_pct", "stake_dollars",
                        "confidence", "is_clear_pick", "reasoning", "model_name"):
                assert key in bet, f"Missing key {key} in bet dict"

    def test_clear_picks_api_returns_picks(self, client: TestClient):
        resp = client.get("/api/clear-picks")
        assert resp.status_code == 200
        picks = resp.json()
        assert isinstance(picks, list)
        if len(picks) > 0:
            pick = picks[0]
            assert "bet" in pick
            assert "clear_score" in pick
            assert "risk_level" in pick
            assert "reasons" in pick

    def test_health_endpoints_all_return_200(self, client: TestClient):
        for path in ("/api/health", "/api/health/live", "/api/health/ready"):
            resp = client.get(path)
            assert resp.status_code == 200, f"{path} returned {resp.status_code}"

    def test_api_refresh_returns_summary(self, client: TestClient):
        resp = client.get("/api/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_bets" in data
        assert "games_available" in data
        assert "total_stake" in data
        assert "generated_at" in data


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CORRECTNESS — Verify data integrity through the pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Verify mock data flows correctly through the entire pipeline."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, sample_games):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=sample_games):
            client.post("/api/live/clear-cache")
            client.post("/api/live/refresh")
            yield

    def test_celtics_home_ml_preserved(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()

        celtics_game = None
        for game in data["next_two_days"]:
            if "Celtics" in game["home_team"]:
                celtics_game = game
                break

        assert celtics_game is not None
        assert celtics_game["home_ml"] is not None
        assert celtics_game["away_ml"] is not None

    def test_today_tomorrow_grouping(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_today"] == 2
        assert data["n_tomorrow"] == 2
        assert data["n_total"] == 4

    def test_short_names_shorter_than_full(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        for game in data["next_two_days"]:
            assert len(game["home_team_short"]) < len(game["home_team"]), \
                f"Short name '{game['home_team_short']}' not shorter than '{game['home_team']}'"
            assert len(game["away_team_short"]) < len(game["away_team"]), \
                f"Short name '{game['away_team_short']}' not shorter than '{game['away_team']}'"

    def test_n_books_reflects_multi_bookmaker(self, client: TestClient):
        """Celtics-Lakers has 2 bookmakers, so n_books_ml should be 2."""
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        celtics_game = next(
            (g for g in data["next_two_days"] if "Celtics" in g["home_team"]), None
        )
        assert celtics_game is not None
        assert celtics_game["n_books_ml"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  GRACEFUL DEGRADATION — Empty state when no odds are available
# ═══════════════════════════════════════════════════════════════════════════

class TestEmptyState:
    """When the Odds API returns nothing, every page should degrade gracefully."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=[]):
            # Clear any stale state from previous happy-path runs
            client.post("/api/live/clear-cache")
            # Refresh with empty data
            resp = client.post("/api/live/refresh")
            assert resp.status_code == 200
            yield

    def test_dashboard_renders_empty(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_todays_card_renders_empty(self, client: TestClient):
        resp = client.get("/todays-card")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_tomorrow_card_renders_empty(self, client: TestClient):
        resp = client.get("/tomorrow")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_live_page_renders_empty(self, client: TestClient):
        resp = client.get("/live")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_pre_match_renders_empty(self, client: TestClient):
        resp = client.get("/pre-match-prediction")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_live_snapshot_empty(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 0
        assert data["next_two_days"] == []

    def test_refresh_returns_zero_games(self, client: TestClient):
        resp = client.post("/api/live/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 0
        assert data["refreshed"] is True

    def test_bets_api_returns_empty(self, client: TestClient):
        resp = client.get("/api/bets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_clear_picks_api_returns_empty(self, client: TestClient):
        resp = client.get("/api/clear-picks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_chart_data_empty(self, client: TestClient):
        resp = client.get("/api/live/chart-data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 0
        assert data["edges"] == []

    def test_no_fake_generated_games(self, client: TestClient):
        """Even with stale cached state, clearing cache and refreshing with empty data
        must produce zero games — no synthetic fallback."""
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 0
        for game in data["next_two_days"]:
            assert game["home_team"] != "Fake Team"
            assert game["away_team"] != "Synthetic Team"


# ═══════════════════════════════════════════════════════════════════════════
#  GRACEFUL DEGRADATION — Engine unavailable
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineFailure:
    """When the engine fails, every page should render 200 with empty defaults."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(
            LivePredictionEngine, "_fetch_realtime_odds",
            side_effect=RuntimeError("Engine failed to initialize"),
        ):
            # Clear any stale state from previous runs
            client.post("/api/live/clear-cache")
            # Refresh will fail, but that's expected
            try:
                client.post("/api/live/refresh")
            except Exception:
                pass
            yield

    def test_dashboard_still_renders(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_live_page_still_renders(self, client: TestClient):
        resp = client.get("/live")
        assert resp.status_code == 200

    def test_todays_card_still_renders(self, client: TestClient):
        resp = client.get("/todays-card")
        assert resp.status_code == 200

    def test_tomorrow_page_still_renders(self, client: TestClient):
        resp = client.get("/tomorrow")
        assert resp.status_code == 200

    def test_pre_match_still_renders(self, client: TestClient):
        resp = client.get("/pre-match-prediction")
        assert resp.status_code == 200

    def test_all_bets_still_renders(self, client: TestClient):
        resp = client.get("/all-bets")
        assert resp.status_code == 200

    def test_snapshot_api_returns_empty(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_total" in data
        assert data["n_total"] == 0

    def test_health_returns_degraded(self, client: TestClient):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES — Cache lifecycle and state transitions
# ═══════════════════════════════════════════════════════════════════════════

class TestCacheLifecycle:
    """Verify clear_cache(), has_cached_data, and snapshot state transitions."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client):
        # First ensure clean state by clearing cache
        client.post("/api/live/clear-cache")
        yield

    def test_clear_cache_resets_snapshot(self, client: TestClient):
        """After clear_cache, force_refresh=False should return empty snapshot."""
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 0
        assert data["next_two_days"] == []

    def test_clear_cache_is_idempotent(self, client: TestClient):
        """Calling clear_cache multiple times should be safe."""
        resp1 = client.post("/api/live/clear-cache")
        resp2 = client.post("/api/live/clear-cache")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["cleared"] is True
        assert resp2.json()["cleared"] is True


class TestStateTransitions:
    """State transitions: empty -> populated -> cleared -> empty."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, sample_games):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=sample_games):
            yield

    def test_freshly_cleared_engine_is_empty(self, client: TestClient):
        """After clear_cache, cached snapshot should be gone."""
        client.post("/api/live/clear-cache")
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 0

    def test_refresh_after_clear_populates(self, client: TestClient):
        """After clear_cache + refresh with data, games should appear."""
        resp = client.post("/api/live/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 4

    def test_subsequent_reads_use_cache(self, client: TestClient):
        """After initial refresh, subsequent reads should not re-fetch (same n_total)."""
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 4

    def test_clear_cache_mid_session_empties(self, client: TestClient):
        """Clearing cache mid-session should reset everything."""
        # Verify data exists first
        resp = client.get("/api/live/snapshot")
        assert resp.json()["n_total"] == 4

        # Clear
        client.post("/api/live/clear-cache")

        # Verify empty
        resp = client.get("/api/live/snapshot")
        assert resp.json()["n_total"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES — Partial odds and missing fields
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def partial_odds_games():
    """
    Games with missing/incomplete odds fields to test graceful handling.
    All commence times are +6 hours from now to avoid the age filter.
    """
    now = datetime.now(timezone.utc)
    t0 = (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:00Z")
    t1 = (now + timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:00Z")
    t2 = (now + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:00Z")
    t3 = (now + timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:00Z")
    t4 = (now + timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:00Z")

    return [
        # Game 1: Full odds
        {
            "id": "game_full_odds",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t0,
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": t0,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Boston Celtics", "price": -200},
                            {"name": "Los Angeles Lakers", "price": 175},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 218.5, "price": -110},
                            {"name": "Under", "point": 218.5, "price": -110},
                        ]},
                    ],
                }
            ],
        },
        # Game 2: Moneyline only
        {
            "id": "game_ml_only",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t1,
            "home_team": "Golden State Warriors",
            "away_team": "Denver Nuggets",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": t1,
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Golden State Warriors", "price": -110},
                            {"name": "Denver Nuggets", "price": -110},
                        ]},
                    ],
                }
            ],
        },
        # Game 3: Totals only
        {
            "id": "game_totals_only",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t2,
            "home_team": "Milwaukee Bucks",
            "away_team": "Oklahoma City Thunder",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": t2,
                    "markets": [
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 222.0, "price": -110},
                            {"name": "Under", "point": 222.0, "price": -110},
                        ]},
                    ],
                }
            ],
        },
        # Game 4: No bookmakers
        {
            "id": "game_no_bookmakers",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t3,
            "home_team": "New York Knicks",
            "away_team": "Miami Heat",
            "bookmakers": [],
        },
        # Game 5: Bookmaker with empty markets
        {
            "id": "game_empty_markets",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": t4,
            "home_team": "Phoenix Suns",
            "away_team": "Philadelphia 76ers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": t4,
                    "markets": [],
                }
            ],
        },
    ]


class TestPartialOdds:
    """Games with missing odds fields should still render correctly."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, partial_odds_games):
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=partial_odds_games):
            client.post("/api/live/clear-cache")
            client.post("/api/live/refresh")
            yield

    def test_full_odds_game_has_total(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        game = next((g for g in data["next_two_days"] if g["game_id"] == "game_full_odds"), None)
        assert game is not None
        assert game["market_total"] is not None

    def test_ml_only_game_has_no_total(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        game = next((g for g in data["next_two_days"] if g["game_id"] == "game_ml_only"), None)
        assert game is not None
        assert game["market_total"] is None
        assert game["home_ml"] is not None

    def test_totals_only_game_has_total_but_no_ml(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        game = next((g for g in data["next_two_days"] if g["game_id"] == "game_totals_only"), None)
        assert game is not None
        assert game["market_total"] is not None
        assert game["home_ml"] is None
        assert game["away_ml"] is None

    def test_no_bookmakers_game_has_no_odds(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        game = next((g for g in data["next_two_days"] if g["game_id"] == "game_no_bookmakers"), None)
        # Game without bookmakers still gets a LiveGame created (has team names)
        if game:
            assert game["home_ml"] is None
            assert game["market_total"] is None

    def test_empty_markets_game_has_no_odds(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        game = next((g for g in data["next_two_days"] if g["game_id"] == "game_empty_markets"), None)
        if game:
            assert game["home_ml"] is None
            assert game["market_total"] is None

    def test_all_five_games_parsed(self, client: TestClient):
        """All 5 games have valid team names + commence times, so all should be parsed."""
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 5, (
            f"Expected 5 games, got {data['n_total']}. "
            f"IDs: {[g['game_id'] for g in data['next_two_days']]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES — has_cached_data property
# ═══════════════════════════════════════════════════════════════════════════

class TestHasCachedData:
    """Verify the has_cached_data property on LivePredictionEngine."""

    def test_has_cached_data_false_when_empty(self, client: TestClient):
        """After engine init + clear_cache, has_cached_data should be False."""
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=[]):
            client.post("/api/live/clear-cache")
            # Engine has no cached data
            engine = LivePredictionEngine()
            engine.clear_cache()
            assert engine.has_cached_data is False

    def test_has_cached_data_true_after_refresh(self, client: TestClient, sample_games):
        """After refresh with data, has_cached_data should be True."""
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=sample_games):
            client.post("/api/live/clear-cache")
            client.post("/api/live/refresh")
            from web.app import get_live_engine
            live_eng = get_live_engine()
            assert live_eng is not None
            assert live_eng.has_cached_data is True


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES — Clear cache endpoint directly
# ═══════════════════════════════════════════════════════════════════════════

class TestClearCacheEndpoint:
    """Direct tests for the /api/live/clear-cache endpoint.
    
    These tests DON'T use a class-level autouse fixture because they need
    to verify clear-cache behavior across different engine states.
    Each test is self-contained with its own patch context.
    """

    def test_clear_cache_returns_200(self, client: TestClient):
        resp = client.post("/api/live/clear-cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cleared"] is True

    def test_clear_cache_affects_snapshot(self, client: TestClient, sample_games):
        """Clear -> refresh -> clear -> snapshot should be empty."""
        from betting_intel.live.engine import LivePredictionEngine
        with patch.object(LivePredictionEngine, "_fetch_realtime_odds", return_value=sample_games):
            client.post("/api/live/clear-cache")
            # Populate
            client.post("/api/live/refresh")
            resp = client.get("/api/live/snapshot")
            assert resp.json()["n_total"] == 4

            # Clear
            client.post("/api/live/clear-cache")

            # Verify empty
            resp = client.get("/api/live/snapshot")
            assert resp.json()["n_total"] == 0
