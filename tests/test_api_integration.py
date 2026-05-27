"""FastAPI TestClient integration tests.

Covers:
    - Health endpoints (/health, /ready, /live)
    - Alert endpoints (/alerts/health, /alerts/stats, /alerts/config, /alerts/test, /alerts/bet)
    - League registry endpoint (/leagues)
    - WebSocket endpoint (/ws/odds)
    - Middleware (CORS headers, request timing)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_app() -> FastAPI:
    """Create the FastAPI app instance for integration testing.

    Uses the default configuration (no real API keys, no alert channels,
    no live odds). The lifespan's try/except blocks gracefully skip
    non-functional components, so the app starts cleanly.
    """
    from betting_intel.api.app import create_app
    return create_app()


@pytest.fixture(scope="module")
def client(api_app: FastAPI) -> TestClient:
    """TestClient bound to the API app."""
    with TestClient(api_app) as c:
        yield c


# ── Health Endpoints ────────────────────────────────────────────────────────


class TestHealthEndpoints:
    """GET /health, /ready, /live — basic health checks."""

    def test_health_returns_200(self, client: TestClient):
        """/health should return 200 with status, version, and db info."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert "version" in body
        assert body["database"] in ("connected", "disconnected")
        assert body["uptime_seconds"] >= 0
        assert isinstance(body["models_loaded"], int)

    def test_ready_returns_200(self, client: TestClient):
        """/ready should return 200 when database is reachable."""
        resp = client.get("/ready")
        # Accept either 200 (ready) or 503 (db not ready)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert resp.json()["status"] == "ready"
        else:
            assert resp.json()["status"] == "not_ready"

    def test_live_returns_200(self, client: TestClient):
        """/live should always return 200 with alive status."""
        resp = client.get("/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_health_has_expected_fields(self, client: TestClient):
        """/health response should contain all expected fields."""
        resp = client.get("/health")
        body = resp.json()
        expected = {"status", "version", "database", "uptime_seconds", "models_loaded"}
        assert expected.issubset(body.keys())


# ── Alert Endpoints ─────────────────────────────────────────────────────────


class TestAlertHealthEndpoint:
    """GET /alerts/health — alert system status."""

    def test_alerts_health_returns_200(self, client: TestClient):
        """Alert health should return 200 with channel info."""
        resp = client.get("/alerts/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("active", "inactive")
        assert isinstance(body["channels"], int)
        assert "config" in body

    def test_alerts_health_has_config(self, client: TestClient):
        """Alert health response should include min_edge_pct et al."""
        resp = client.get("/alerts/health")
        config = resp.json()["config"]
        assert "min_edge_pct" in config
        assert "min_confidence" in config
        assert "min_stake" in config

    def test_alerts_health_shows_total_dispatched(self, client: TestClient):
        """Alert health should include total_dispatched counter."""
        resp = client.get("/alerts/health")
        assert isinstance(resp.json()["total_dispatched"], int)


class TestAlertStatsEndpoint:
    """GET /alerts/stats — alert dispatch statistics."""

    def test_alert_stats_returns_200(self, client: TestClient):
        """Alert stats should return 200 with stats dict."""
        resp = client.get("/alerts/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert "channels_configured" in stats
        assert "total_dispatched" in stats

    def test_alert_stats_structure(self, client: TestClient):
        """Alert stats should have expected top-level keys."""
        resp = client.get("/alerts/stats")
        stats = resp.json()
        expected_keys = {
            "channels_configured", "total_dispatched",
            "rate_limit_remaining", "channels",
        }
        assert expected_keys.issubset(stats.keys())


class TestAlertTestEndpoint:
    """POST /alerts/test — send test alert."""

    def test_test_alert_no_channel_returns_404(self, client: TestClient):
        """Without a configured channel, should return 404."""
        resp = client.post("/alerts/test", params={
            "channel": "telegram",
            "message": "Integration test",
        })
        # If no channel configured, expect 404
        assert resp.status_code == 404
        assert "not configured" in resp.json()["detail"].lower()


class TestAlertBetEndpoint:
    """POST /alerts/bet — trigger a bet alert."""

    def test_bet_alert_returns_filtered_when_no_channels(self, client: TestClient):
        """Without channels, bet alert should be 'filtered'."""
        resp = client.post("/alerts/bet", params={
            "matchup": "LAL @ BOS",
            "bet_type": "OVER 220.5",
            "edge_pct": 5.0,
            "confidence": 0.75,
            "stake": 200.0,
            "league": "NBA",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "filtered"  # no channels registered
        assert body["channels"] == []
        assert "alert" in body


class TestAlertConfigEndpoint:
    """POST /alerts/config — update alert configuration."""

    def test_update_config_returns_200(self, client: TestClient):
        """Updating alert config should return updated values."""
        resp = client.post("/alerts/config", params={
            "min_edge_pct": 4.0,
            "min_confidence": 0.60,
            "min_stake": 100.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "updated"
        assert body["config"]["min_edge_pct"] == 4.0
        assert body["config"]["min_confidence"] == 0.60
        assert body["config"]["min_stake"] == 100.0

    def test_update_config_partial(self, client: TestClient):
        """Partial update should only change specified fields."""
        resp = client.post("/alerts/config", params={
            "min_edge_pct": 5.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"]["min_edge_pct"] == 5.0

    def test_update_config_booleans(self, client: TestClient):
        """Boolean config fields should accept true/false."""
        resp = client.post("/alerts/config", params={
            "enable_live_movement_alerts": True,
            "enable_daily_summary": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"]["enable_live_movement_alerts"] is True
        assert body["config"]["enable_daily_summary"] is False


# ── League Registry Endpoint ────────────────────────────────────────────────


class TestLeaguesEndpoint:
    """GET /leagues — list all registered leagues with health."""

    def test_leagues_returns_200(self, client: TestClient):
        """/leagues should return 200."""
        resp = client.get("/leagues")
        assert resp.status_code == 200

    def test_leagues_has_leagues_key(self, client: TestClient):
        """Response should contain 'leagues' dict."""
        resp = client.get("/leagues")
        body = resp.json()
        assert "leagues" in body

    def test_leagues_health_data(self, client: TestClient):
        """Each league entry should have health fields."""
        resp = client.get("/leagues")
        leagues = resp.json()["leagues"]
        if leagues:
            for key, data in leagues.items():
                assert "status" in data
                assert isinstance(data, dict)


# ── WebSocket Endpoint ──────────────────────────────────────────────────────


class TestWebSocketOddsEndpoint:
    """WebSocket /ws/odds — connect/disconnect without live odds configured."""

    def test_websocket_receives_error_when_disabled(self, client: TestClient):
        """Without live odds, WebSocket should receive error and close."""
        with client.websocket_connect("/ws/odds") as ws:
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Live odds not enabled" in data["message"]

    def test_websocket_receive_expected_fields(self, client: TestClient):
        """Error message should have type and message fields."""
        with client.websocket_connect("/ws/odds") as ws:
            data = ws.receive_json()
            assert {"type", "message"}.issubset(data.keys())

    def test_websocket_closes_after_error(self, client: TestClient):
        """After sending the error, the server should close the connection."""
        from starlette.websockets import WebSocketDisconnect

        with client.websocket_connect("/ws/odds") as ws:
            ws.receive_json()
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


# ── CORS & Middleware ───────────────────────────────────────────────────────


class TestMiddleware:
    """CORS headers and request timing middleware."""

    def test_cors_preflight_returns_allow_origin(self, client: TestClient):
        """OPTIONS preflight should include CORS allow-origin header."""
        resp = client.options(
            "/health",
            headers={"origin": "https://example.com", "access-control-request-method": "GET"},
        )
        assert resp.status_code == 200
        origin = resp.headers["access-control-allow-origin"]
        assert origin in ("*", "https://example.com")  # Credentials=True echoes origin

    def test_process_time_header(self, client: TestClient):
        """Responses should include X-Process-Time-MS header."""
        resp = client.get("/health")
        assert "x-process-time-ms" in resp.headers
        value = float(resp.headers["x-process-time-ms"])
        assert isinstance(value, float)


# ── Error Handling ─────────────────────────────────────────────────────────


class TestErrorHandling:
    """Global exception handler and error responses."""

    def test_404_returns_json(self, client: TestClient):
        """Unknown routes should return JSON, not HTML."""
        resp = client.get("/nonexistent_route")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/json")

    def test_invalid_method_returns_405(self, client: TestClient):
        """Using wrong HTTP method should return 405."""
        resp = client.put("/health")
        assert resp.status_code == 405


# ── Schema Validation ───────────────────────────────────────────────────────


class TestSchemaValidation:
    """Pydantic schema validation on request bodies."""

    def test_bet_alert_validates_required_params(self, client: TestClient):
        """Missing required params for bet alert should return 422."""
        resp = client.post("/alerts/bet", params={
            # Missing matchup, bet_type, edge_pct, etc.
        })
        assert resp.status_code == 422

    def test_bet_alert_validates_types(self, client: TestClient):
        """Invalid types should return 422."""
        resp = client.post("/alerts/bet", params={
            "matchup": "LAL @ BOS",
            "bet_type": "OVER 220.5",
            "edge_pct": "not_a_number",  # should be float
            "confidence": 0.75,
            "stake": 200.0,
        })
        assert resp.status_code == 422


# ── /ws/stats Endpoint (conditional) ───────────────────────────────────────


class TestWebSocketStatsEndpoint:
    """GET /ws/stats — WebSocket connection stats.

    This endpoint is only registered when the live odds manager is initialized.
    With default settings (no ODDS_API_KEY), it should not be available,
    so /ws/stats should return 404.
    """

    def test_ws_stats_not_available(self, client: TestClient):
        """Without live odds configured, /ws/stats should 404."""
        resp = client.get("/ws/stats")
        assert resp.status_code == 404
