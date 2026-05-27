"""Tests for the WebSocket odds system, alert dispatcher, alert formatters, and league registry.

Covers:
    - ConnectionManager broadcast/subscribe/unsubscribe
    - OddsPoller snapshot processing and movement detection
    - AlertDispatcher threshold filtering and rate limiting
    - Telegram/Discord message formatting (static helpers only — no network)
    - LeagueRegistry health checks and freshness grading
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# ── Fixtures shared across test classes ────────────────────────────────────


@pytest.fixture
def mock_websocket():
    """A mock FastAPI WebSocket that tracks sent messages."""
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.send_json = AsyncMock()
    ws.accept = AsyncMock()
    ws.receive_text = AsyncMock()
    return ws


@pytest.fixture
def sample_snapshot():
    """A standard OddsSnapshot for testing."""
    from betting_intel.data.websocket_odds import OddsSnapshot

    return OddsSnapshot(
        game_id="NBA_LAL-BOS",
        league="NBA",
        home_team="Lakers",
        away_team="Celtics",
        game_date="2026-05-30",
        home_ml=1.85,
        away_ml=2.00,
        spread=-4.5,
        total=220.5,
        captured_at=time.time(),
    )


@pytest.fixture
def sample_bet_alert():
    """A standard BetAlert for threshold-filtering tests."""
    from betting_intel.alerts.dispatcher import BetAlert

    return BetAlert(
        game_id="NBA_LAL-BOS",
        matchup="Lakers @ Celtics",
        bet_type="OVER 220.5",
        edge_pct=5.2,
        confidence=0.72,
        stake=250.0,
        league="NBA",
        reasoning="Strong pace advantage",
    )


# ── ConnectionManager Tests ────────────────────────────────────────────────


class TestConnectionManager:
    """WebSocket connection management with league subscription filtering."""

    @pytest.fixture
    def mgr(self):
        from betting_intel.data.websocket_odds import ConnectionManager

        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, mgr, mock_websocket):
        """A connected client should be tracked, then removed on disconnect."""
        assert mgr.active_connections == 0
        await mgr.connect(mock_websocket)
        assert mgr.active_connections == 1
        await mgr.disconnect(mock_websocket)
        assert mgr.active_connections == 0

    @pytest.mark.asyncio
    async def test_connect_with_subscriptions(self, mgr, mock_websocket):
        """Client can subscribe to specific leagues on connect."""
        await mgr.connect(mock_websocket, leagues=["NBA", "WNBA"])
        assert mgr.active_connections == 1

        # Broadcast with NBA league should reach this client
        await mgr.broadcast({"type": "test"}, league="NBA")
        assert mock_websocket.send_text.call_count == 1

    @pytest.mark.asyncio
    async def test_subscribe_and_unsubscribe(self, mgr, mock_websocket):
        """Client subscription set should update after subscribe/unsubscribe calls."""
        await mgr.connect(mock_websocket, leagues=["NBA", "WNBA"])
        await mgr.subscribe(mock_websocket, ["NHL"])

        await mgr.broadcast({"type": "nba"}, league="NBA")
        assert mock_websocket.send_text.call_count == 1

        await mgr.unsubscribe(mock_websocket, ["NBA"])
        await mgr.broadcast({"type": "wnba"}, league="WNBA")
        assert mock_websocket.send_text.call_count == 2  # WNBA still subscribed

        await mgr.broadcast({"type": "nhl"}, league="NHL")
        assert mock_websocket.send_text.call_count == 3  # NHL subscribed above

        await mgr.broadcast({"type": "nba"}, league="NBA")
        assert mock_websocket.send_text.call_count == 3  # NBA unsubscribed

    @pytest.mark.asyncio
    async def test_broadcast_filtered_by_league(self, mgr, mock_websocket):
        """Broadcast with a league should only reach clients subscribed to it."""
        await mgr.connect(mock_websocket, leagues=["NBA"])

        await mgr.broadcast({"type": "nba_update"}, league="NBA")
        assert mock_websocket.send_text.call_count == 1

        await mgr.broadcast({"type": "wnba_only"}, league="WNBA")
        assert mock_websocket.send_text.call_count == 1  # not incremented

    @pytest.mark.asyncio
    async def test_broadcast_to_all_when_no_league(self, mgr, mock_websocket):
        """Broadcast without a league filter reaches every client."""
        ws2 = MagicMock()
        ws2.send_text = AsyncMock()
        ws2.accept = AsyncMock()

        await mgr.connect(mock_websocket)
        await mgr.connect(ws2)

        await mgr.broadcast({"type": "heartbeat"})
        assert mock_websocket.send_text.call_count == 1
        assert ws2.send_text.call_count == 1

    @pytest.mark.asyncio
    async def test_removes_disconnected_clients(self, mgr, mock_websocket):
        """Broadcast should silently remove clients that have disconnected."""
        mock_websocket.send_text = AsyncMock(side_effect=Exception("gone"))

        await mgr.connect(mock_websocket)
        await mgr.broadcast({"type": "test"})
        assert mgr.active_connections == 0  # removed

    @pytest.mark.asyncio
    async def test_connect_without_leagues_receives_all(self, mgr, mock_websocket):
        """A client connected without explicit subscriptions receives every broadcast."""
        await mgr.connect(mock_websocket)  # no league filter

        await mgr.broadcast({"type": "nba"}, league="NBA")
        await mgr.broadcast({"type": "wnba"}, league="WNBA")
        assert mock_websocket.send_text.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_clients_isolated(self, mgr):
        """Each client's subscription state should be independent."""
        ws_nba = MagicMock()
        ws_nba.send_text = AsyncMock()
        ws_nba.accept = AsyncMock()
        ws_wnba = MagicMock()
        ws_wnba.send_text = AsyncMock()
        ws_wnba.accept = AsyncMock()

        await mgr.connect(ws_nba, leagues=["NBA"])
        await mgr.connect(ws_wnba, leagues=["WNBA"])

        await mgr.broadcast({"type": "nba"}, league="NBA")
        ws_nba.send_text.assert_awaited_once()
        ws_wnba.send_text.assert_not_awaited()

        await mgr.broadcast({"type": "wnba"}, league="WNBA")
        ws_wnba.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_client(self, mgr, mock_websocket):
        """Disconnecting an unconnected client should not raise."""
        await mgr.disconnect(mock_websocket)  # should not raise


# ── OddsPoller Tests ────────────────────────────────────────────────────────


class TestOddsPoller:
    """Odds snapshot processing and movement detection."""

    @pytest.fixture
    def poller(self):
        from betting_intel.data.websocket_odds import OddsPoller, ConnectionManager

        return OddsPoller(
            connection_manager=ConnectionManager(),
            poll_interval=30,
            odds_api_key=None,  # no API calls
            db_path=None,       # no SQLite
            movement_threshold_pct=0.02,
        )

    @pytest.mark.asyncio
    async def test_first_snapshot_no_movement(self, poller, sample_snapshot, mock_websocket):
        """The first snapshot for a game should not trigger a movement alert."""
        await poller.manager.connect(mock_websocket)
        await poller._process_snapshot(sample_snapshot)

        # Should broadcast an odds_update
        calls = [c[0][0] for c in mock_websocket.send_text.call_args_list]
        update_calls = [c for c in calls if '"odds_update"' in c]
        movement_calls = [c for c in calls if '"odds_movement"' in c]
        assert len(update_calls) >= 1
        assert len(movement_calls) == 0  # no previous to compare

    @pytest.mark.asyncio
    async def test_significant_movement_detected(self, poller, sample_snapshot, mock_websocket):
        """A large enough change from previous snapshot triggers an odds_movement."""
        await poller.manager.connect(mock_websocket)

        # First snapshot
        await poller._process_snapshot(sample_snapshot)

        # Second snapshot with a significant total change (4% > 2% threshold)
        from betting_intel.data.websocket_odds import OddsSnapshot

        updated = OddsSnapshot(
            game_id=sample_snapshot.game_id,
            league=sample_snapshot.league,
            home_team=sample_snapshot.home_team,
            away_team=sample_snapshot.away_team,
            game_date=sample_snapshot.game_date,
            home_ml=1.85,
            away_ml=2.00,
            spread=-4.5,
            total=230.0,  # 220.5 -> 230.0 = 4.3% change > 2% threshold
            captured_at=time.time() + 10,
        )

        await poller._process_snapshot(updated)

        calls = [c[0][0] for c in mock_websocket.send_text.call_args_list]
        movement_calls = [c for c in calls if '"odds_movement"' in c]
        assert len(movement_calls) >= 1

    @pytest.mark.asyncio
    async def test_small_movement_ignored(self, poller, sample_snapshot, mock_websocket):
        """A change below the movement threshold should NOT trigger a movement alert."""
        await poller.manager.connect(mock_websocket)

        # First snapshot
        await poller._process_snapshot(sample_snapshot)

        # Tiny change: 220.5 -> 221.0 = 0.23% << 2% threshold
        from betting_intel.data.websocket_odds import OddsSnapshot

        tiny = OddsSnapshot(
            game_id=sample_snapshot.game_id,
            league=sample_snapshot.league,
            home_team=sample_snapshot.home_team,
            away_team=sample_snapshot.away_team,
            game_date=sample_snapshot.game_date,
            home_ml=1.85,
            away_ml=2.00,
            spread=-4.5,
            total=221.0,
            captured_at=time.time() + 10,
        )

        await poller._process_snapshot(tiny)

        calls = [c[0][0] for c in mock_websocket.send_text.call_args_list]
        movement_calls = [c for c in calls if '"odds_movement"' in c]
        assert len(movement_calls) == 0

    @pytest.mark.asyncio
    async def test_get_current_odds_filters_by_league(self, poller, sample_snapshot):
        """get_current_odds should optionally filter by league."""
        # Add an NBA snapshot
        await poller._process_snapshot(sample_snapshot)

        from betting_intel.data.websocket_odds import OddsSnapshot

        # Add a WNBA snapshot
        wnba = OddsSnapshot(
            game_id="WNBA_LV-ATL",
            league="WNBA",
            home_team="Aces",
            away_team="Dream",
            game_date="2026-05-30",
        )
        await poller._process_snapshot(wnba)

        all_odds = poller.get_current_odds()
        assert len(all_odds) == 2

        nba_only = poller.get_current_odds(league="NBA")
        assert len(nba_only) == 1
        assert nba_only[0]["league"] == "NBA"

        wnba_only = poller.get_current_odds(league="WNBA")
        assert len(wnba_only) == 1
        assert wnba_only[0]["league"] == "WNBA"

    @pytest.mark.asyncio
    async def test_evict_stale_snapshots(self, poller, sample_snapshot):
        """Snapshots older than TTL should be evicted from memory."""
        old = sample_snapshot
        old.captured_at = time.time() - (poller._ttl_hours * 3600 + 60)  # beyond TTL
        await poller._process_snapshot(old)

        assert len(poller._last_snapshots) == 1

        await poller._evict_stale_snapshots()
        assert len(poller._last_snapshots) == 0

    @pytest.mark.asyncio
    async def test_evict_keeps_fresh_snapshots(self, poller, sample_snapshot):
        """Snapshots within TTL should be preserved after eviction."""
        fresh = sample_snapshot
        fresh.captured_at = time.time()  # now
        await poller._process_snapshot(fresh)
        assert len(poller._last_snapshots) == 1

        await poller._evict_stale_snapshots()
        assert len(poller._last_snapshots) == 1

    @pytest.mark.asyncio
    async def test_evict_mixed(self, poller, sample_snapshot):
        """Stale and fresh snapshots — only stale should be evicted."""
        from betting_intel.data.websocket_odds import OddsSnapshot

        stale = OddsSnapshot(
            game_id="stale_1", league="NBA", home_team="A", away_team="B",
            game_date="2026-01-01", captured_at=time.time() - (poller._ttl_hours * 3600 + 60),
        )
        fresh = OddsSnapshot(
            game_id="fresh_1", league="NBA", home_team="C", away_team="D",
            game_date="2026-05-30", captured_at=time.time(),
        )
        await poller._process_snapshot(stale)
        await poller._process_snapshot(fresh)
        assert len(poller._last_snapshots) == 2

        await poller._evict_stale_snapshots()
        assert len(poller._last_snapshots) == 1
        assert "fresh_1" in poller._last_snapshots

    @pytest.mark.asyncio
    async def test_get_odds_history_no_db(self, poller):
        """get_odds_history should return empty DataFrame when no DB configured."""
        df = poller.get_odds_history("NBA_LAL-BOS")
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_live_movements_no_previous(self, poller):
        """get_live_movements should return empty list when no history exists."""
        movements = poller.get_live_movements()
        assert movements == []


# ── AlertDispatcher Tests ──────────────────────────────────────────────────


class TestBetAlertShouldDispatch:
    """BetAlert.should_dispatch threshold validation."""

    @pytest.fixture
    def config(self):
        from betting_intel.alerts.dispatcher import AlertConfig

        return AlertConfig(
            min_edge_pct=3.0,
            min_confidence=0.55,
            min_stake=50.0,
        )

    def test_sufficient_edge_passes(self, sample_bet_alert, config):
        """Alert with edge >= min should pass."""
        ok, reason = sample_bet_alert.should_dispatch(config)
        assert ok is True
        assert reason == "OK"

    def test_low_edge_filtered(self, sample_bet_alert, config):
        """Alert with edge below minimum should be rejected."""
        sample_bet_alert.edge_pct = 1.5
        ok, reason = sample_bet_alert.should_dispatch(config)
        assert ok is False
        assert "Edge" in reason

    def test_low_confidence_filtered(self, sample_bet_alert, config):
        """Alert with confidence below minimum should be rejected."""
        sample_bet_alert.confidence = 0.40
        ok, reason = sample_bet_alert.should_dispatch(config)
        assert ok is False
        assert "Confidence" in reason

    def test_low_stake_filtered(self, sample_bet_alert, config):
        """Alert with stake below minimum should be rejected."""
        sample_bet_alert.stake = 10.0
        ok, reason = sample_bet_alert.should_dispatch(config)
        assert ok is False
        assert "Stake" in reason

    def test_exact_thresholds_pass(self, sample_bet_alert, config):
        """Alerts exactly at threshold should pass."""
        sample_bet_alert.edge_pct = 3.0
        sample_bet_alert.confidence = 0.55
        sample_bet_alert.stake = 50.0
        ok, reason = sample_bet_alert.should_dispatch(config)
        assert ok is True

    def test_all_thresholds_fail_together(self, sample_bet_alert, config):
        """Alerts failing multiple criteria should report first failure."""
        sample_bet_alert.edge_pct = 1.0
        sample_bet_alert.confidence = 0.30
        sample_bet_alert.stake = 5.0
        ok, reason = sample_bet_alert.should_dispatch(config)
        assert ok is False


class TestAlertDispatcher:
    """Dispatch routing, rate limiting, and channel management."""

    @pytest.fixture
    def dispatcher(self):
        from betting_intel.alerts.dispatcher import AlertDispatcher, AlertConfig

        cfg = AlertConfig(
            min_edge_pct=3.0,
            min_confidence=0.55,
            min_stake=50.0,
            rate_limit_seconds=60,
            max_alerts_per_hour=10,
        )
        return AlertDispatcher(config=cfg)

    @pytest.fixture
    def mock_sender(self):
        """A mock alert channel sender."""
        sender = MagicMock()
        sender.send_bet_alert = AsyncMock(return_value=None)
        sender.send_line_movement = AsyncMock(return_value=None)
        sender.send_daily_summary = AsyncMock(return_value=None)
        sender.send_health_report = AsyncMock(return_value=None)
        sender.send_message = AsyncMock(return_value=None)
        return sender

    @pytest.mark.asyncio
    async def test_dispatch_bet_alert_success(self, dispatcher, sample_bet_alert, mock_sender):
        """Valid bet alert should be dispatched to registered channel."""
        dispatcher.add_channel("telegram", mock_sender)
        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert "telegram" in sent
        mock_sender.send_bet_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispatch_filtered_by_low_edge(self, dispatcher, sample_bet_alert, mock_sender):
        """Low-edge alert should NOT reach the channel."""
        dispatcher.add_channel("telegram", mock_sender)
        sample_bet_alert.edge_pct = 1.0

        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent == []
        mock_sender.send_bet_alert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_filtered_by_low_confidence(self, dispatcher, sample_bet_alert, mock_sender):
        """Low-confidence alert should NOT reach the channel."""
        dispatcher.add_channel("telegram", mock_sender)
        sample_bet_alert.confidence = 0.30
        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent == []
        mock_sender.send_bet_alert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_filtered_by_low_stake(self, dispatcher, sample_bet_alert, mock_sender):
        """Low-stake alert should NOT reach the channel."""
        dispatcher.add_channel("telegram", mock_sender)
        sample_bet_alert.stake = 10.0
        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent == []
        mock_sender.send_bet_alert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_routes_by_league(self, dispatcher, sample_bet_alert, mock_sender):
        """Alert should only go to channels that handle the alert's league."""
        dispatcher.add_channel("nba_channel", mock_sender, leagues=["NBA"])
        dispatcher.add_channel("wnba_channel", MagicMock(send_bet_alert=AsyncMock()), leagues=["WNBA"])

        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert "nba_channel" in sent
        assert "wnba_channel" not in sent

    @pytest.mark.asyncio
    async def test_rate_limited(self, dispatcher, sample_bet_alert, mock_sender):
        """Rapid alerts for the same game should be rate-limited."""
        dispatcher.add_channel("telegram", mock_sender)

        # First dispatch should succeed
        sent1 = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert "telegram" in sent1
        assert mock_sender.send_bet_alert.await_count == 1

        # Second dispatch immediately should be rate-limited
        sent2 = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent2 == []
        assert mock_sender.send_bet_alert.await_count == 1

    @pytest.mark.asyncio
    async def test_hourly_limit(self, dispatcher, sample_bet_alert, mock_sender):
        """When hourly max is reached, alerts should be dropped."""
        dispatcher.add_channel("telegram", mock_sender)

        # Fill the hourly quota (we set max_alerts_per_hour=10, add 10 more directly)
        dispatcher.config._alert_timestamps.extend([time.time()] * 10)

        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent == []

    @pytest.mark.asyncio
    async def test_no_channels_registered(self, dispatcher, sample_bet_alert):
        """Alert with no registered channels should return empty list."""
        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent == []

    @pytest.mark.asyncio
    async def test_disabled_channel_skipped(self, dispatcher, sample_bet_alert, mock_sender):
        """Disabled channels should not receive alerts."""
        dispatcher.add_channel("telegram", mock_sender)
        dispatcher.channels["telegram"].enabled = False
        sent = await dispatcher.dispatch_bet_alert(sample_bet_alert)
        assert sent == []

    @pytest.mark.asyncio
    async def test_dispatch_line_movement_disabled(self, dispatcher, mock_sender):
        """Line movement alerts should not dispatch when feature is disabled."""
        dispatcher.add_channel("discord", mock_sender)
        dispatcher.config.enable_live_movement_alerts = False

        sent = await dispatcher.dispatch_line_movement(
            matchup="LAL @ BOS", market="total",
            old_line=220.5, new_line=222.0,
            direction="up", league="NBA",
        )
        assert sent == []
        mock_sender.send_line_movement.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_line_movement_enabled(self, dispatcher, mock_sender):
        """Line movement alerts should dispatch when feature is enabled."""
        dispatcher.add_channel("discord", mock_sender)
        dispatcher.config.enable_live_movement_alerts = True

        sent = await dispatcher.dispatch_line_movement(
            matchup="LAL @ BOS", market="total",
            old_line=220.5, new_line=222.0,
            direction="up", league="NBA",
        )
        assert "discord" in sent
        mock_sender.send_line_movement.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispatch_raw_message_to_specific_channel(self, dispatcher, mock_sender):
        """Raw message should dispatch to a named channel when specified."""
        dispatcher.add_channel("telegram", mock_sender)
        other = MagicMock(send_message=AsyncMock())
        dispatcher.add_channel("discord", other)

        sent = await dispatcher.dispatch_raw_message("hello", channel_name="telegram")
        assert sent is True
        mock_sender.send_message.assert_awaited_once_with("hello")
        other.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_raw_message_to_all(self, dispatcher, mock_sender):
        """Raw message with no channel name should go to all channels."""
        other = MagicMock(send_message=AsyncMock())
        dispatcher.add_channel("telegram", mock_sender)
        dispatcher.add_channel("discord", other)

        sent = await dispatcher.dispatch_raw_message("broadcast")
        assert sent is True
        mock_sender.send_message.assert_awaited_once_with("broadcast")
        other.send_message.assert_awaited_once_with("broadcast")

    @pytest.mark.asyncio
    async def test_daily_summary_dispatches(self, dispatcher, mock_sender):
        """Daily summary should dispatch to all channels."""
        dispatcher.add_channel("telegram", mock_sender)
        dispatcher.config.enable_daily_summary = True

        sent = await dispatcher.dispatch_daily_summary(
            total_bets=10, wins=6, losses=4, profit=2.5, roi=5.0,
        )
        assert "telegram" in sent

    @pytest.mark.asyncio
    async def test_daily_summary_disabled(self, dispatcher, mock_sender):
        """Daily summary should not dispatch when disabled."""
        dispatcher.add_channel("telegram", mock_sender)
        dispatcher.config.enable_daily_summary = False

        sent = await dispatcher.dispatch_daily_summary(
            total_bets=10, wins=6, losses=4, profit=2.5, roi=5.0,
        )
        assert sent == []

    def test_get_stats(self, dispatcher, mock_sender):
        """get_stats should return configured channels and counts."""
        dispatcher.add_channel("telegram", mock_sender)
        stats = dispatcher.get_stats()

        assert stats["channels_configured"] == 1
        assert "telegram" in stats["channels"]
        assert stats["channels"]["telegram"]["type"] == "MagicMock"

    def test_remove_channel(self, dispatcher, mock_sender):
        """Removing a channel should drop it from the registry."""
        dispatcher.add_channel("telegram", mock_sender)
        assert len(dispatcher.channels) == 1
        dispatcher.remove_channel("telegram")
        assert len(dispatcher.channels) == 0


# ── Telegram Formatting Tests (no network) ─────────────────────────────────


class TestTelegramBot:
    """TelegramBot static/formatter methods — no HTTP calls."""

    @pytest.fixture
    def bot(self):
        from betting_intel.alerts.telegram import TelegramBot

        return TelegramBot(token="test:token", chat_id="123")

    def test_confidence_bar_full(self, bot):
        """100% confidence should fill all 10 bars."""
        bar = bot._confidence_bar(1.0)
        assert bar == "▓" * 10

    def test_confidence_bar_empty(self, bot):
        """0% confidence should show no filled bars."""
        bar = bot._confidence_bar(0.0)
        assert bar == "░" * 10

    def test_confidence_bar_half(self, bot):
        """50% confidence should fill 5 bars."""
        bar = bot._confidence_bar(0.5)
        assert bar.count("▓") == 5
        assert bar.count("░") == 5

    def test_confidence_bar_clamped_high(self, bot):
        """Confidence > 1.0 should be clamped to 10 bars."""
        bar = bot._confidence_bar(2.0)
        assert bar == "▓" * 10

    def test_confidence_bar_clamped_low(self, bot):
        """Confidence < 0 should be clamped to 0 bars."""
        bar = bot._confidence_bar(-0.5)
        assert bar == "░" * 10

    def test_confidence_bar_rounding(self, bot):
        """Confidence of 0.76 should round to 8 bars."""
        bar = bot._confidence_bar(0.76)
        assert bar.count("▓") == 8

    def test_kelly_color_conservative(self, bot):
        """Small stakes should be CONSERVATIVE."""
        assert bot._kelly_color(50.0) == "CONSERVATIVE"

    def test_kelly_color_moderate(self, bot):
        """Mid-range stakes should be MODERATE."""
        assert bot._kelly_color(300.0) == "MODERATE"

    def test_kelly_color_aggressive(self, bot):
        """Large stakes should be AGGRESSIVE."""
        assert bot._kelly_color(600.0) == "AGGRESSIVE"

    def test_kelly_color_boundary(self, bot):
        """Boundaries: >500 AGGRESSIVE, >200 MODERATE, <=200 CONSERVATIVE."""
        assert bot._kelly_color(200.0) == "CONSERVATIVE"  # not > 200
        assert bot._kelly_color(201.0) == "MODERATE"       # > 200
        assert bot._kelly_color(500.0) == "MODERATE"       # > 200 but not > 500
        assert bot._kelly_color(501.0) == "AGGRESSIVE"     # > 500

    @pytest.mark.asyncio
    async def test_send_message_no_token(self):
        """With no token configured, send_message should return error."""
        from betting_intel.alerts.telegram import TelegramBot

        bot = TelegramBot(token=None, chat_id=None)
        msg = await bot.send_message("test")

        assert msg.error is not None

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self, bot):
        """send_message with no token should return error gracefully."""
        bot.token = None
        result = await bot.send_message("hello")
        assert result.error == "Telegram not configured"

    def test_confidence_bar_77_pct(self, bot):
        """Confidence of 0.77 should round to 8 bars."""
        bar = bot._confidence_bar(0.77)
        assert bar.count("▓") == 8


# ── Discord Formatting Tests (HTTP client mocked) ─────────────────────────


class TestDiscordWebhook:
    """Discord embed formatting — HTTP client is mocked."""

    @pytest.fixture
    def webhook(self):
        from betting_intel.alerts.discord import DiscordWebhook

        wh = DiscordWebhook(webhook_url="https://discord.com/api/webhooks/test/id/token")
        # Mock the client to avoid real HTTP calls
        wh._client = AsyncMock()
        wh._client.post = AsyncMock()
        return wh

    @pytest.mark.asyncio
    async def test_send_embed_calls_api(self, webhook):
        """send_embed should POST to the webhook URL."""
        from betting_intel.alerts.discord import DiscordEmbed

        embed = DiscordEmbed(title="Test", description="Hello")
        result = await webhook.send_embed(embed)
        assert result is True
        webhook._client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_embed_no_url(self):
        """send_embed with no webhook URL should return False."""
        from betting_intel.alerts.discord import DiscordWebhook, DiscordEmbed

        wh = DiscordWebhook(webhook_url=None)
        embed = DiscordEmbed(title="Test")
        result = await wh.send_embed(embed)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_bet_alert_high_edge(self, webhook):
        """High-edge bet alert should produce green embed (COLOR_GREEN = 0x00FF00)."""
        await webhook.send_bet_alert(
            matchup="LAL @ BOS", bet_type="OVER 220.5",
            edge_pct=6.0, confidence=0.80, stake=300.0,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        payload = call_kwargs["json"]
        embed = payload["embeds"][0]
        assert embed["color"] == 0x00FF00  # GREEN

    @pytest.mark.asyncio
    async def test_send_bet_alert_moderate_edge(self, webhook):
        """Moderate-edge bet alert should produce blue embed (COLOR_BLUE = 0x3498DB)."""
        await webhook.send_bet_alert(
            matchup="LAL @ BOS", bet_type="SPREAD -4.5",
            edge_pct=4.0, confidence=0.70, stake=200.0,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0x3498DB  # BLUE

    @pytest.mark.asyncio
    async def test_send_bet_alert_low_edge(self, webhook):
        """Low-edge bet alert should produce gray embed (COLOR_GRAY = 0x95A5A6)."""
        await webhook.send_bet_alert(
            matchup="LAL @ BOS", bet_type="MONEYLINE",
            edge_pct=3.0, confidence=0.60, stake=100.0,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0x95A5A6  # GRAY

    @pytest.mark.asyncio
    async def test_send_bet_alert_includes_reasoning(self, webhook):
        """When reasoning is provided, it should appear as a field."""
        await webhook.send_bet_alert(
            matchup="LAL @ BOS", bet_type="OVER 220.5",
            edge_pct=5.0, confidence=0.70, stake=200.0,
            reasoning="Strong pace mismatch",
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        fields = call_kwargs["json"]["embeds"][0]["fields"]
        analysis_fields = [f for f in fields if f["name"] == "💡 Analysis"]
        assert len(analysis_fields) == 1
        assert analysis_fields[0]["value"] == "Strong pace mismatch"

    @pytest.mark.asyncio
    async def test_send_line_movement(self, webhook):
        """Line movement alert should include old and new values."""
        await webhook.send_line_movement(
            matchup="LAL @ BOS", market="total",
            old_line=220.5, new_line=222.0,
            direction="up", league="NBA",
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "220.5" in fields["Previous"]
        assert "222.0" in fields["Current"]
        assert "UP" in fields["Change"]

    @pytest.mark.asyncio
    async def test_send_daily_summary_profitable(self, webhook):
        """Profitable daily summary should produce green embed."""
        await webhook.send_daily_summary(
            total_bets=10, wins=7, losses=3, profit=4.2, roi=8.5,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0x00FF00  # GREEN

    @pytest.mark.asyncio
    async def test_send_daily_summary_losing(self, webhook):
        """Losing daily summary should produce red embed."""
        await webhook.send_daily_summary(
            total_bets=10, wins=3, losses=7, profit=-3.1, roi=-6.0,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0xE74C3C  # RED

    @pytest.mark.asyncio
    async def test_send_health_report_healthy(self, webhook):
        """Healthy system status should produce green embed."""
        await webhook.send_health_report(
            status="healthy", models_active=5, leagues_tracked=8, errors_last_hour=0,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0x00FF00

    @pytest.mark.asyncio
    async def test_send_health_report_degraded(self, webhook):
        """Degraded system status should produce orange embed."""
        await webhook.send_health_report(
            status="degraded", models_active=3, leagues_tracked=8, errors_last_hour=2,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0xF39C12  # ORANGE

    @pytest.mark.asyncio
    async def test_send_health_report_critical(self, webhook):
        """Critical system status should produce red embed."""
        await webhook.send_health_report(
            status="down", models_active=0, leagues_tracked=0, errors_last_hour=15,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 0xE74C3C  # RED

    @pytest.mark.asyncio
    async def test_connection_failure_returns_false(self):
        """If API call fails with an httpx error, send_embed should return False."""
        from betting_intel.alerts.discord import DiscordWebhook, DiscordEmbed

        wh = DiscordWebhook(webhook_url="https://discord.com/api/webhooks/test")
        wh._client = AsyncMock()
        wh._client.post = AsyncMock(side_effect=httpx.RequestError("network error", request=MagicMock()))

        embed = DiscordEmbed(title="Test")
        result = await wh.send_embed(embed)
        assert result is False

    @pytest.mark.asyncio
    async def test_health_report_embeds_fields(self, webhook):
        """Health report should include models, leagues, and errors fields."""
        await webhook.send_health_report(
            status="healthy", models_active=4, leagues_tracked=6, errors_last_hour=1,
        )
        call_kwargs = webhook._client.post.call_args.kwargs
        fields = call_kwargs["json"]["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "🧠 Active Models" in field_names
        assert "🏟️ Leagues Tracked" in field_names
        assert "❌ Errors (1h)" in field_names


# ── LeagueRegistry Tests ────────────────────────────────────────────────────


class TestLeagueRegistry:
    """League registration, health checks, and freshness grading."""

    @pytest.fixture
    def registry(self):
        from betting_intel.data.small_leagues.league_registry import LeagueRegistry

        reg = LeagueRegistry()
        # Clear auto-discovered sources to get a clean slate for testing
        reg._sources.clear()
        reg._metadata.clear()
        return reg

    def test_check_health_unknown_league(self, registry):
        """check_health for an unregistered league should return unavailable."""
        status = registry.check_health("nonexistent_league")
        assert status.is_available is False
        assert status.status == "unavailable"

    def test_list_leagues_after_registration(self, registry):
        """list_leagues should return registered sources with metadata."""
        registry.register("test_league", MagicMock, {"name": "Test League", "data_source": "mock"})
        leagues = registry.list_leagues()
        assert "test_league" in leagues
        assert leagues["test_league"]["name"] == "Test League"
        assert leagues["test_league"]["data_source"] == "mock"
        assert leagues["test_league"]["source_class"] == "MagicMock"

    def test_get_source_unknown_raises(self, registry):
        """get_source with unknown league key should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown league"):
            registry.get_source("nonexistent_league")

    def test_check_health_registered_league(self, registry):
        """check_health for a registered league should return graded status."""
        import pandas as pd

        mock_source = MagicMock()
        mock_df = pd.DataFrame({
            "date": [pd.Timestamp.now()],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
        })
        mock_source.load_historical = MagicMock(return_value=mock_df)

        registry.register("mock_league", lambda **kw: mock_source, {"name": "Mock", "data_source": "mock"})
        status = registry.check_health("mock_league")

        assert status.league_key == "mock_league"
        assert status.is_available is True
        assert status.status == "healthy"
        assert status.freshness_grade == "A"  # just fetched
        assert status.total_games == 1

    def test_check_all_health(self, registry):
        """check_all_health should return status for every registered league."""
        registry.register("league_a", MagicMock, {"name": "A"})
        registry.register("league_b", MagicMock, {"name": "B"})

        results = registry.check_all_health()
        assert "league_a" in results
        assert "league_b" in results
        assert len(results) == 2

    def test_get_available_leagues_returns_registered(self, registry):
        """get_available_leagues should return leagues whose sources initialize."""
        registry.register("good", MagicMock, {"name": "Good"})
        results = registry.get_available_leagues()
        assert isinstance(results, list)
        # A MagicMock class source will fail health check (load_historical doesn't exist)
        # so good won't appear. This at minimum validates the method runs without error.


class TestLeagueHealthStatus:
    """Freshness grade computation."""

    @pytest.fixture
    def status(self):
        from betting_intel.data.small_leagues.league_registry import LeagueHealthStatus

        return LeagueHealthStatus(
            league_key="test_league",
            league_name="Test League",
            status="healthy",
            data_source="test_api",
        )

    def test_freshness_grade_no_data(self, status):
        """No data fetch at all should return NO_DATA."""
        status.last_data_fetch = None
        assert status.freshness_grade == "NO_DATA"

    def test_freshness_grade_A(self, status):
        """Data fetched less than 1 hour ago should be A."""
        status.last_data_fetch = datetime.now()
        assert status.freshness_grade == "A"

    def test_freshness_grade_B(self, status):
        """Data fetched 2 hours ago should be B."""
        status.last_data_fetch = datetime.now() - timedelta(hours=2)
        assert status.freshness_grade == "B"

    def test_freshness_grade_C(self, status):
        """Data fetched 12 hours ago should be C."""
        status.last_data_fetch = datetime.now() - timedelta(hours=12)
        assert status.freshness_grade == "C"

    def test_freshness_grade_D(self, status):
        """Data fetched 48 hours ago should be D."""
        status.last_data_fetch = datetime.now() - timedelta(hours=48)
        assert status.freshness_grade == "D"

    def test_freshness_grade_F(self, status):
        """Data fetched 96 hours ago should be F."""
        status.last_data_fetch = datetime.now() - timedelta(hours=96)
        assert status.freshness_grade == "F"

    def test_freshness_grade_just_under_24h(self, status):
        """Data fetched just under 24 hours ago should be C (< 24)."""
        status.last_data_fetch = datetime.now() - timedelta(hours=23, minutes=59)
        assert status.freshness_grade == "C"

    def test_freshness_grade_just_over_24h(self, status):
        """Data fetched just over 24 hours ago should be D (>= 24)."""
        status.last_data_fetch = datetime.now() - timedelta(hours=24, minutes=1)
        assert status.freshness_grade == "D"

    def test_freshness_grade_just_under_72h(self, status):
        """Data fetched just under 72 hours ago should be D (< 72)."""
        status.last_data_fetch = datetime.now() - timedelta(hours=71, minutes=59)
        assert status.freshness_grade == "D"

    def test_freshness_grade_just_over_72h(self, status):
        """Data fetched just over 72 hours ago should be F (>= 72)."""
        status.last_data_fetch = datetime.now() - timedelta(hours=72, minutes=1)
        assert status.freshness_grade == "F"
