"""
Telegram bot integration for sending betting alerts.

Uses python-telegram-bot (if available) or raw HTTP API requests.
Sends structured messages with:
- Bet recommendations (matchup, line, edge, stake)
- Line movement alerts
- Daily performance summaries
- Health/error reports

Usage:
    bot = TelegramBot(token="123456:ABC-DEF1234")
    bot.send_bet_alert("LAL @ BOS", "OVER 220.5", 5.2, 0.72, 250.0)
    bot.send_message("Daily summary: +3.2u, win rate 58.3%")

Environment variables:
    TELEGRAM_BOT_TOKEN — Bot token from @BotFather
    TELEGRAM_CHAT_ID — Target chat ID (send alerts to this chat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TelegramMessage:
    """A structured message ready to send via Telegram."""

    text: str
    parse_mode: str = "HTML"
    disable_web_page_preview: bool = True
    sent_at: Optional[datetime] = None
    error: Optional[str] = None


class TelegramBot:
    """Sends betting alerts via Telegram Bot API.

    Uses async HTTP calls (httpx.AsyncClient) to avoid blocking
    the event loop when dispatched from async FastAPI route handlers.
    """

    API_BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 10,
    ):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
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

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        chat_id: Optional[str] = None,
    ) -> TelegramMessage:
        """Send a plain text message to Telegram."""
        target_chat = chat_id or self.chat_id
        if not self.token or not target_chat:
            logger.warning("Telegram not configured: missing token or chat_id")
            return TelegramMessage(
                text=text,
                error="Telegram not configured",
            )

        url = self.API_BASE.format(token=self.token, method="sendMessage")
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        client = await self.get_client()
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            msg = TelegramMessage(text=text, sent_at=datetime.now())
            logger.debug(f"Telegram message sent: {text[:60]}...")
            return msg
        except httpx.HTTPStatusError as exc:
            error = f"Telegram API error {exc.response.status_code}: {exc.response.text}"
            logger.error(error)
            return TelegramMessage(text=text, error=error)
        except httpx.RequestError as exc:
            error = f"Telegram connection error: {exc}"
            logger.error(error)
            return TelegramMessage(text=text, error=error)

    async def send_bet_alert(
        self,
        matchup: str,
        bet_type: str,
        edge_pct: float,
        confidence: float,
        stake: float,
        league: str = "NBA",
        reasoning: str = "",
        chat_id: Optional[str] = None,
    ) -> TelegramMessage:
        """Send a structured bet alert with formatting."""
        confidence_bar = self._confidence_bar(confidence)
        emoji = "🟢" if edge_pct > 5 else "🔵" if edge_pct > 3 else "⚪"

        text = (
            f"{emoji} <b>BET ALERT</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏀 {matchup}\n"
            f"📋 <b>{bet_type}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 Edge: <b>{edge_pct:.1f}%</b>\n"
            f"🎯 Confidence: {confidence_bar} ({confidence:.0%})\n"
            f"💰 Stake: <b>${stake:.0f}</b> ({self._kelly_color(stake)} Kelly)\n"
            f"🏷️ League: {league}\n"
        )

        if reasoning:
            text += f"💡 <i>{reasoning}</i>\n"

        text += (
            f"━━━━━━━━━━━━━━━\n"
            f"🕐 {datetime.now().strftime('%H:%M UTC')}\n"
        )

        return await self.send_message(text, chat_id=chat_id)

    async def send_line_movement(
        self,
        matchup: str,
        market: str,
        old_line: float,
        new_line: float,
        direction: str,
        league: str = "NBA",
        chat_id: Optional[str] = None,
    ) -> TelegramMessage:
        """Send a line movement alert."""
        arrow = "📈" if direction == "up" else "📉"
        text = (
            f"{arrow} <b>LINE MOVEMENT</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏀 {matchup}\n"
            f"📋 {market}: {old_line:.1f} → <b>{new_line:.1f}</b>\n"
            f"📊 Change: {abs(new_line - old_line):.1f} points ({direction.upper()})\n"
            f"🏷️ {league}\n"
            f"🕐 {datetime.now().strftime('%H:%M UTC')}\n"
        )
        return await self.send_message(text, chat_id=chat_id)

    async def send_daily_summary(
        self,
        total_bets: int,
        wins: int,
        losses: int,
        profit: float,
        roi: float,
        best_bet: str = "",
        chat_id: Optional[str] = None,
    ) -> TelegramMessage:
        """Send a daily betting performance summary."""
        win_rate = wins / max(total_bets, 1)
        profit_emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"

        text = (
            f"📊 <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🎯 Bets: {total_bets} ({wins}W / {losses}L)\n"
            f"📈 Win Rate: {win_rate:.1%}\n"
            f"{profit_emoji} P&L: <b>{profit:+.1f}u</b> ({roi:+.1f}%)\n"
        )

        if best_bet:
            text += f"⭐ Best: {best_bet}\n"

        text += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
        return await self.send_message(text, chat_id=chat_id)

    async def send_health_report(
        self,
        status: str,
        models_active: int,
        leagues_tracked: int,
        errors_last_hour: int,
        chat_id: Optional[str] = None,
    ) -> TelegramMessage:
        """Send system health report."""
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "🚨"
        text = (
            f"{status_emoji} <b>SYSTEM HEALTH</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Status: <b>{status.upper()}</b>\n"
            f"🧠 Models: {models_active}\n"
            f"🏟️ Leagues: {leagues_tracked}\n"
            f"❌ Errors (1h): {errors_last_hour}\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
        )
        return await self.send_message(text, chat_id=chat_id)

    async def test_connection(self) -> bool:
        """Test if the bot can connect to Telegram."""
        try:
            url = self.API_BASE.format(token=self.token, method="getMe")
            client = await self.get_client()
            resp = await client.get(url)
            resp.raise_for_status()
            logger.info("Telegram connection test: OK")
            return True
        except Exception as exc:
            logger.error(f"Telegram connection test failed: {exc}")
            return False

    @staticmethod
    def _confidence_bar(confidence: float) -> str:
        """Create a visual confidence bar."""
        filled = int(round(confidence * 10))
        filled = max(0, min(10, filled))
        return "▓" * filled + "░" * (10 - filled)

    @staticmethod
    def _kelly_color(stake: float) -> str:
        """Color-code based on Kelly aggressiveness."""
        if stake > 500:
            return "AGGRESSIVE"
        elif stake > 200:
            return "MODERATE"
        return "CONSERVATIVE"
