"""
Discord webhook integration for betting alerts.

Sends rich embeds to Discord channels via webhook URLs.
Supports:
- Bet recommendation embeds (color-coded by edge)
- Line movement alerts
- Daily performance embeds
- System health embeds

Usage:
    webhook = DiscordWebhook(url="https://discord.com/api/webhooks/...")
    webhook.send_bet_alert("LAL @ BOS", "OVER 220.5", 5.2, 0.72, 250.0)

Environment variables:
    DISCORD_WEBHOOK_URL — Full webhook URL from Discord channel settings
    DISCORD_WEBHOOK_ID — Webhook ID (alternative)
    DISCORD_WEBHOOK_TOKEN — Webhook token (alternative)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# Discord embed color constants
COLOR_GREEN = 0x00FF00  # High edge (>5%)
COLOR_BLUE = 0x3498DB   # Moderate edge (3-5%)
COLOR_GRAY = 0x95A5A6   # Low edge (<3%)
COLOR_RED = 0xE74C3C    # Loss / warning
COLOR_PURPLE = 0x9B59B6  # Line movement
COLOR_ORANGE = 0xF39C12  # System alert


@dataclass
class DiscordEmbed:
    """A Discord embed object."""

    title: str
    description: str = ""
    color: int = COLOR_BLUE
    fields: list[dict] = field(default_factory=list)
    footer: Optional[dict] = None
    timestamp: Optional[str] = None


class DiscordWebhook:
    """Sends structured alerts to Discord via webhook.

    Uses async HTTP calls (httpx.AsyncClient) to avoid blocking
    the event loop when dispatched from async FastAPI route handlers.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        webhook_id: Optional[str] = None,
        webhook_token: Optional[str] = None,
        timeout: int = 10,
        username: str = "Betting Intel",
        avatar_url: str = "",
    ):
        # Allow passing URL directly or as ID + token
        if webhook_url:
            self.webhook_url = webhook_url
        elif webhook_id and webhook_token:
            self.webhook_url = (
                f"https://discord.com/api/webhooks/{webhook_id}/{webhook_token}"
            )
        else:
            self.webhook_url = None

        self.timeout = timeout
        self.username = username
        self.avatar_url = avatar_url
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_embed(
        self,
        embed: DiscordEmbed,
        content: str = "",
    ) -> bool:
        """Send a Discord embed message."""
        if not self.webhook_url:
            logger.warning("Discord webhook not configured")
            return False

        payload: dict[str, Any] = {
            "username": self.username,
        }
        if content:
            payload["content"] = content
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        embed_dict = {
            "title": embed.title,
            "description": embed.description,
            "color": embed.color,
            "fields": embed.fields,
        }
        if embed.footer:
            embed_dict["footer"] = embed.footer
        if embed.timestamp:
            embed_dict["timestamp"] = embed.timestamp

        payload["embeds"] = [embed_dict]

        client = await self.get_client()
        try:
            resp = await client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            logger.debug(f"Discord embed sent: {embed.title}")
            return True
        except httpx.HTTPStatusError as exc:
            logger.error(f"Discord API error {exc.response.status_code}: {exc.response.text}")
            return False
        except httpx.RequestError as exc:
            logger.error(f"Discord connection error: {exc}")
            return False

    async def send_bet_alert(
        self,
        matchup: str,
        bet_type: str,
        edge_pct: float,
        confidence: float,
        stake: float,
        league: str = "NBA",
        reasoning: str = "",
    ) -> bool:
        """Send a structured bet alert as a Discord embed."""
        color = COLOR_GREEN if edge_pct > 5 else COLOR_BLUE if edge_pct > 3 else COLOR_GRAY
        confidence_stars = "⭐" * int(round(confidence * 5)) or "☆"

        embed = DiscordEmbed(
            title=f"🎯 {bet_type}",
            description=f"**{matchup}** | {league}",
            color=color,
            fields=[
                {"name": "📊 Edge", "value": f"**{edge_pct:.1f}%**", "inline": True},
                {"name": "🎯 Confidence", "value": f"{confidence_stars} ({confidence:.0%})", "inline": True},
                {"name": "💰 Stake", "value": f"**${stake:.0f}**", "inline": True},
            ],
            footer={"text": f"Betting Intelligence • {datetime.now().strftime('%H:%M UTC')}"},
            timestamp=datetime.utcnow().isoformat(),
        )

        if reasoning:
            embed.fields.append({"name": "💡 Analysis", "value": reasoning, "inline": False})

        return await self.send_embed(embed)

    async def send_line_movement(
        self,
        matchup: str,
        market: str,
        old_line: float,
        new_line: float,
        direction: str,
        league: str = "NBA",
    ) -> bool:
        """Send a line movement alert."""
        arrow = "📈" if direction == "up" else "📉"
        embed = DiscordEmbed(
            title=f"{arrow} {market} Movement",
            description=f"**{matchup}** | {league}",
            color=COLOR_PURPLE,
            fields=[
                {"name": "Previous", "value": f"{old_line:.1f}", "inline": True},
                {"name": "Current", "value": f"**{new_line:.1f}**", "inline": True},
                {"name": "Change", "value": f"{abs(new_line - old_line):.1f} pts ({direction.upper()})", "inline": True},
            ],
            footer={"text": f"Line Movement Alert • {datetime.now().strftime('%H:%M UTC')}"},
            timestamp=datetime.utcnow().isoformat(),
        )
        return await self.send_embed(embed)

    async def send_daily_summary(
        self,
        total_bets: int,
        wins: int,
        losses: int,
        profit: float,
        roi: float,
        best_bet: str = "",
    ) -> bool:
        """Send a daily summary embed."""
        win_rate = wins / max(total_bets, 1)
        color = COLOR_GREEN if profit > 0 else COLOR_RED if profit < 0 else COLOR_GRAY

        embed = DiscordEmbed(
            title="📊 Daily Performance Summary",
            description=f"**{datetime.now().strftime('%B %d, %Y')}**",
            color=color,
            fields=[
                {"name": "🎯 Bets", "value": f"{total_bets} ({wins}W / {losses}L)", "inline": True},
                {"name": "📈 Win Rate", "value": f"{win_rate:.1%}", "inline": True},
                {"name": "💰 P&L", "value": f"**{profit:+.1f}u** ({roi:+.1f}%)", "inline": False},
            ],
            footer={"text": "Betting Intelligence • Daily Report"},
            timestamp=datetime.utcnow().isoformat(),
        )
        if best_bet:
            embed.fields.append({"name": "⭐ Best Bet", "value": best_bet, "inline": False})

        return await self.send_embed(embed)

    async def send_health_report(
        self,
        status: str,
        models_active: int,
        leagues_tracked: int,
        errors_last_hour: int,
    ) -> bool:
        """Send system health embed."""
        if status == "healthy":
            color = COLOR_GREEN
            emoji = "✅"
        elif status == "degraded":
            color = COLOR_ORANGE
            emoji = "⚠️"
        else:
            color = COLOR_RED
            emoji = "🚨"

        embed = DiscordEmbed(
            title=f"{emoji} System Health Report",
            description=f"**Status: {status.upper()}**",
            color=color,
            fields=[
                {"name": "🧠 Active Models", "value": str(models_active), "inline": True},
                {"name": "🏟️ Leagues Tracked", "value": str(leagues_tracked), "inline": True},
                {"name": "❌ Errors (1h)", "value": str(errors_last_hour), "inline": True},
            ],
            footer={"text": f"Betting Intelligence • Health Check • {datetime.now().strftime('%H:%M UTC')}"},
            timestamp=datetime.utcnow().isoformat(),
        )
        return await self.send_embed(embed)

    async def test_connection(self) -> bool:
        """Test if the webhook is reachable."""
        try:
            client = await self.get_client()
            resp = await client.post(
                self.webhook_url,
                json={"content": "✅ Betting Intelligence system connected", "username": self.username},
            )
            resp.raise_for_status()
            logger.info("Discord webhook test: OK")
            return True
        except Exception as exc:
            logger.error(f"Discord webhook test failed: {exc}")
            return False
