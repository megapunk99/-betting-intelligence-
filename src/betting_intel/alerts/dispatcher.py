"""
Alert dispatcher — configurable rules engine for sending betting alerts.

Routes bet alerts, line movements, and system health updates to the
appropriate channels (Telegram, Discord, etc.) based on configurable rules:

- Minimum edge threshold (e.g., only send alerts for >= 3% EV)
- Minimum confidence threshold
- Per-league routing (send WNBA alerts to one channel, NBA to another)
- Rate limiting (avoid spamming during live games)

Usage:
    dispatcher = AlertDispatcher()
    dispatcher.add_channel("telegram", TelegramBot(token="..."))
    dispatcher.add_channel("discord", DiscordWebhook(url="..."))

    # Send a bet alert (respects rules and thresholds)
    await dispatcher.dispatch_bet_alert(
        game_id="NBA_LAL-BOS",
        matchup="LAL @ BOS",
        bet_type="OVER 220.5",
        edge_pct=5.2,
        confidence=0.72,
        stake=250.0,
        league="NBA",
        reasoning="Historical trend + injury advantage",
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Priority level for alerts."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configuration for the alert dispatcher."""

    min_edge_pct: float = 3.0  # Minimum edge % to send alert
    min_confidence: float = 0.55  # Minimum confidence to send alert
    min_stake: float = 50.0  # Minimum stake $ to send alert
    rate_limit_seconds: int = 60  # Min seconds between alerts for same game
    max_alerts_per_hour: int = 30  # Max alerts per hour across all games
    enable_live_movement_alerts: bool = True
    enable_daily_summary: bool = True
    enable_health_reports: bool = True
    health_report_interval_hours: int = 6

    # Per-league routing
    league_channel_map: dict[str, str] = field(default_factory=dict)

    # Cooldown tracking
    _last_game_alert: dict[str, float] = field(default_factory=dict)
    _alert_timestamps: list[float] = field(default_factory=list)


@dataclass
class BetAlert:
    """A structured bet alert ready for dispatch."""

    game_id: str
    matchup: str
    bet_type: str
    edge_pct: float
    confidence: float
    stake: float
    league: str = "NBA"
    reasoning: str = ""
    model: str = ""
    priority: AlertPriority = AlertPriority.NORMAL
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def should_dispatch(self, config: AlertConfig) -> tuple[bool, str]:
        """Check if this alert meets dispatch criteria."""
        if self.edge_pct < config.min_edge_pct:
            return False, f"Edge {self.edge_pct:.1f}% < min {config.min_edge_pct}%"
        if self.confidence < config.min_confidence:
            return False, f"Confidence {self.confidence:.0%} < min {config.min_confidence:.0%}"
        if self.stake < config.min_stake:
            return False, f"Stake ${self.stake:.0f} < min ${config.min_stake:.0f}"
        return True, "OK"


@dataclass
class AlertChannel:
    """A configured alert channel."""

    name: str
    sender: Any  # TelegramBot | DiscordWebhook
    enabled: bool = True
    leagues: set[str] = field(default_factory=lambda: {"*"})  # "*" = all leagues
    min_priority: AlertPriority = AlertPriority.NORMAL


class AlertDispatcher:
    """Dispatches alerts to configured channels with rule-based filtering.

    Supports:
    - Multiple channels (Telegram, Discord, etc.)
    - Per-league channel routing
    - Rate limiting and throttling
    - Priority-based filtering
    - Scheduled daily summaries
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.channels: dict[str, AlertChannel] = {}
        self._dispatch_count: int = 0

    def add_channel(
        self,
        name: str,
        sender: Any,
        leagues: Optional[list[str]] = None,
        min_priority: AlertPriority = AlertPriority.NORMAL,
    ) -> None:
        """Register an alert channel."""
        self.channels[name] = AlertChannel(
            name=name,
            sender=sender,
            leagues=set(leagues or ["*"]),
            min_priority=min_priority,
        )
        logger.info(f"Alert channel '{name}' registered ({sender.__class__.__name__})")

    def remove_channel(self, name: str) -> None:
        """Unregister an alert channel."""
        self.channels.pop(name, None)
        logger.info(f"Alert channel '{name}' removed")

    def _rate_limited(self, game_id: str) -> bool:
        """Check if we're rate-limited for this game."""
        now = time.time()

        # Clean old timestamps
        one_hour_ago = now - 3600
        self.config._alert_timestamps = [
            t for t in self.config._alert_timestamps if t > one_hour_ago
        ]

        # Check per-game cooldown
        last_alert = self.config._last_game_alert.get(game_id, 0)
        if now - last_alert < self.config.rate_limit_seconds:
            return True

        # Check hourly limit
        if len(self.config._alert_timestamps) >= self.config.max_alerts_per_hour:
            return True

        return False

    def _record_alert(self, game_id: str) -> None:
        """Record an alert dispatch for rate limiting."""
        now = time.time()
        self.config._last_game_alert[game_id] = now
        self.config._alert_timestamps.append(now)
        self._dispatch_count += 1

    def _get_channels_for_league(self, league: str) -> list[AlertChannel]:
        """Get channels that should receive alerts for this league."""
        result = []
        for channel in self.channels.values():
            if not channel.enabled:
                continue
            if "*" in channel.leagues or league in channel.leagues:
                result.append(channel)
        return result

    # ── Public Dispatch Methods (all async) ─────────────────────────────────

    async def dispatch_bet_alert(self, alert: BetAlert) -> list[str]:
        """Dispatch a bet alert to all relevant channels.

        Returns:
            List of channel names that received the alert.
        """
        # Check criteria
        should_send, reason = alert.should_dispatch(self.config)
        if not should_send:
            logger.debug(f"Bet alert filtered: {reason}")
            return []

        # Rate limiting
        if self._rate_limited(alert.game_id):
            logger.debug(f"Bet alert rate-limited: {alert.game_id}")
            return []

        # Find eligible channels
        channels = self._get_channels_for_league(alert.league)
        if not channels:
            return []

        sent_to = []
        for channel in channels:
            try:
                sender = channel.sender
                if hasattr(sender, "send_bet_alert"):
                    await sender.send_bet_alert(
                        matchup=alert.matchup,
                        bet_type=alert.bet_type,
                        edge_pct=alert.edge_pct,
                        confidence=alert.confidence,
                        stake=alert.stake,
                        league=alert.league,
                        reasoning=alert.reasoning,
                    )
                    sent_to.append(channel.name)
            except Exception as exc:
                logger.error(f"Failed to dispatch to {channel.name}: {exc}")

        self._record_alert(alert.game_id)
        if sent_to:
            logger.info(
                f"Bet alert dispatched: {alert.matchup} {alert.bet_type} "
                f"(edge={alert.edge_pct:.1f}%, channels={sent_to})"
            )
        return sent_to

    async def dispatch_line_movement(
        self,
        matchup: str,
        market: str,
        old_line: float,
        new_line: float,
        direction: str,
        league: str = "NBA",
    ) -> list[str]:
        """Dispatch a line movement alert."""
        if not self.config.enable_live_movement_alerts:
            return []

        channels = self._get_channels_for_league(league)
        sent_to = []

        for channel in channels:
            sender = channel.sender
            if hasattr(sender, "send_line_movement"):
                try:
                    await sender.send_line_movement(
                        matchup=matchup,
                        market=market,
                        old_line=old_line,
                        new_line=new_line,
                        direction=direction,
                        league=league,
                    )
                    sent_to.append(channel.name)
                except Exception as exc:
                    logger.error(f"Line movement dispatch failed to {channel.name}: {exc}")

        return sent_to

    async def dispatch_daily_summary(
        self,
        total_bets: int,
        wins: int,
        losses: int,
        profit: float,
        roi: float,
        best_bet: str = "",
    ) -> list[str]:
        """Dispatch daily performance summary to all channels."""
        if not self.config.enable_daily_summary:
            return []

        sent_to = []
        for channel in self.channels.values():
            if not channel.enabled:
                continue
            sender = channel.sender
            if hasattr(sender, "send_daily_summary"):
                try:
                    await sender.send_daily_summary(
                        total_bets=total_bets,
                        wins=wins,
                        losses=losses,
                        profit=profit,
                        roi=roi,
                        best_bet=best_bet,
                    )
                    sent_to.append(channel.name)
                except Exception as exc:
                    logger.error(f"Daily summary dispatch failed to {channel.name}: {exc}")

        return sent_to

    async def dispatch_health_report(
        self,
        status: str,
        models_active: int,
        leagues_tracked: int,
        errors_last_hour: int,
    ) -> list[str]:
        """Dispatch system health report."""
        if not self.config.enable_health_reports:
            return []

        sent_to = []
        for channel in self.channels.values():
            if not channel.enabled:
                continue
            sender = channel.sender
            if hasattr(sender, "send_health_report"):
                try:
                    await sender.send_health_report(
                        status=status,
                        models_active=models_active,
                        leagues_tracked=leagues_tracked,
                        errors_last_hour=errors_last_hour,
                    )
                    sent_to.append(channel.name)
                except Exception as exc:
                    logger.error(f"Health report dispatch failed to {channel.name}: {exc}")

        return sent_to

    async def dispatch_raw_message(
        self,
        message: str,
        channel_name: Optional[str] = None,
    ) -> bool:
        """Dispatch a raw text message to a specific channel or all."""
        targets = (
            [self.channels[channel_name]]
            if channel_name and channel_name in self.channels
            else list(self.channels.values())
        )

        sent = False
        for channel in targets:
            if not channel.enabled:
                continue
            try:
                sender = channel.sender
                if hasattr(sender, "send_message"):
                    await sender.send_message(message)
                    sent = True
            except Exception as exc:
                logger.error(f"Raw message dispatch failed to {channel.name}: {exc}")

        return sent

    def get_stats(self) -> dict:
        """Get dispatcher statistics."""
        return {
            "total_dispatched": self._dispatch_count,
            "channels_configured": len(self.channels),
            "rate_limit_remaining": max(
                0,
                self.config.max_alerts_per_hour - len(self.config._alert_timestamps),
            ),
            "alerts_last_hour": len(self.config._alert_timestamps),
            "channels": {
                name: {
                    "type": ch.sender.__class__.__name__,
                    "enabled": ch.enabled,
                    "leagues": list(ch.leagues),
                }
                for name, ch in self.channels.items()
            },
        }


# Global alert dispatcher
alert_dispatcher = AlertDispatcher()
