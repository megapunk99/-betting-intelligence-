"""
Scraper-level mock integration test.

Mocks the stealth_scraper module and verifies the full engine → scraper
call chain works end-to-end.

What this tests:
  1. The engine correctly IMPORTS and INVOKES the stealth_scraper module
  2. The engine's fallback chain (ESPN → TheOddsAPI) runs correctly
     when the TheOddsAPI returns nothing (no API key configured)
  3. Graceful degradation when the scraper fails (ImportError, empty data)

ARCHITECTURE:
  - Patches at the scraper MODULE level
    e.g. patch('betting_intel.data.stealth_scraper.StealthBrowser.sync_scrape_live_odds')
  - Lets the engine's actual _fetch_realtime_odds → _fetch_stealth_scraper methods
    run naturally
  - Each test class uses a class-scoped autouse fixture for state isolation
  - All tests carry the 'integration' pytest marker for CI visibility
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STEALTH_MODULE = "betting_intel.data.stealth_scraper.StealthBrowser.sync_scrape_live_odds"


# ═══════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sample_espn_format():
    """Games in TheOddsAPI response format — what stealth_scraper returns."""
    now = datetime.now(timezone.utc)
    t0 = (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:00Z")

    return [
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
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 218.5, "price": -110},
                            {"name": "Under", "point": 218.5, "price": -110},
                        ]},
                    ],
                }
            ],
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient — module-scoped for speed."""
    import os
    from web.app import app
    api_key = os.environ.get("API_KEY", "change-me-to-a-random-secret")
    with TestClient(app, headers={"X-API-Key": api_key}) as c:
        yield c


def _clear_and_refresh(client):
    """Clear engine cache then refresh with ODDS_API_KEY removed so ESPN is tried first."""
    old_key = os.environ.pop("ODDS_API_KEY", None)
    try:
        client.post("/api/live/clear-cache")
        return client.post("/api/live/refresh")
    finally:
        if old_key is not None:
            os.environ["ODDS_API_KEY"] = old_key


# ═══════════════════════════════════════════════════════════════════════════
#  STEALTH SCRAPER SUCCEEDS
# ═══════════════════════════════════════════════════════════════════════════

class TestStealthScraperSucceeds:
    """Stealth scraper returns data — engine imports and calls scraper module."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, sample_espn_format):
        with patch(STEALTH_MODULE, return_value=sample_espn_format) as mock_scraper:
            resp = _clear_and_refresh(client)
            assert resp.status_code == 200
            assert resp.json()["n_total"] > 0
            mock_scraper.assert_called_once()
            yield

    def test_dashboard_renders(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Celtics" in resp.text

    def test_snapshot_has_games(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 1
        assert data["fresh_odds"] is True

    def test_team_names_preserved(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        game = data["next_two_days"][0]
        assert game["home_team"] == "Boston Celtics"
        assert game["away_team"] == "Los Angeles Lakers"

    def test_no_synthetic_teams(self, client: TestClient):
        resp = client.get("/todays-card")
        assert "Raptors" not in resp.text
        assert "Spurs" not in resp.text


# ═══════════════════════════════════════════════════════════════════════════
#  ALL SOURCES FAIL
# ═══════════════════════════════════════════════════════════════════════════

class TestAllSourcesFail:
    """All odds sources return empty — graceful degradation."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client):
        with patch(STEALTH_MODULE, return_value=[]) as mock_scraper:
            resp = _clear_and_refresh(client)
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_total"] == 0
            mock_scraper.assert_called_once()
            yield

    def test_dashboard_renders_empty(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_snapshot_empty(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 0
        assert data["next_two_days"] == []

    def test_chart_data_empty(self, client: TestClient):
        resp = client.get("/api/live/chart-data")
        data = resp.json()
        assert data["n_total"] == 0
        assert data["edges"] == []

    def test_bets_empty(self, client: TestClient):
        resp = client.get("/api/bets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_clear_picks_empty(self, client: TestClient):
        resp = client.get("/api/clear-picks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_no_fake_games_generated(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert len(data["next_two_days"]) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  SCRAPER IMPORT ERROR
# ═══════════════════════════════════════════════════════════════════════════

class TestScraperImportError:
    """Stealth scraper fails with ImportError — graceful empty."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client):
        with patch(STEALTH_MODULE, side_effect=ImportError("playwright not installed")) as mock_scraper:
            resp = _clear_and_refresh(client)
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_total"] == 0
            mock_scraper.assert_called_once()
            yield

    def test_dashboard_still_renders(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_snapshot_empty(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 0

    def test_todays_card_renders(self, client: TestClient):
        resp = client.get("/todays-card")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
#  THEODDSAPI FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

class TestTheOddsApiFallback:
    """When ESPN returns empty, TheOddsAPI is tried as fallback."""

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, client, sample_espn_format):
        from web.app import get_live_engine
        engine = get_live_engine()
        assert engine is not None
        old_key = getattr(engine, "_odds_api_key", None)
        engine._odds_api_key = "test_key_for_fallback_test"

        try:
            with patch.object(engine, "_fetch_via_theoddsapi", return_value=sample_espn_format) as mock_fetch:
                with patch(STEALTH_MODULE, return_value=[]) as mock_espn:
                    client.post("/api/live/clear-cache")
                    resp = client.post("/api/live/refresh")
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["n_total"] > 0
                    mock_espn.assert_called_once()
                    mock_fetch.assert_called_once()
                    yield
        finally:
            engine._odds_api_key = old_key

    def test_dashboard_displays_games(self, client: TestClient):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Celtics" in resp.text

    def test_snapshot_from_odds_api(self, client: TestClient):
        resp = client.get("/api/live/snapshot")
        data = resp.json()
        assert data["n_total"] == 1
        assert data["fresh_odds"] is True
