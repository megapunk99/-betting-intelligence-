"""
TelegramNotifier — sends high-confidence betting pick alerts via Telegram.

Uses the Telegram Bot API directly via httpx (no python-telegram-bot dependency).
Zero runtime overhead when not configured — all imports are lazy.

Usage:
    notifier = TelegramNotifier()
    if notifier.is_configured:
        await notifier.send_pick_alert(game)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send high-confidence betting pick alerts to a Telegram chat.

    Configured via environment variables:
      TELEGRAM_BOT_TOKEN  — bot token from @BotFather
      TELEGRAM_CHAT_ID    — chat ID (user, group, or channel) to send to

    Thread-safe for single-message sends.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self._bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._notified_game_ids: set[str] = set()

    # ── Configuration ─────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """Return True if both token and chat_id are set."""
        return bool(self._bot_token) and bool(self._chat_id)

    @property
    def bot_token(self) -> str:
        return self._bot_token

    @bot_token.setter
    def bot_token(self, value: str):
        self._bot_token = value

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @chat_id.setter
    def chat_id(self, value: str):
        self._chat_id = value

    # ── Notification Tracking ─────────────────────────────────────────────

    def is_notified(self, game_id: str) -> bool:
        """Check if a game has already been notified."""
        return game_id in self._notified_game_ids

    def mark_as_notified(self, game_id: str):
        """Mark a game as already notified."""
        self._notified_game_ids.add(game_id)

    def reset_tracking(self):
        """Clear all notification tracking (e.g., on new day)."""
        self._notified_game_ids.clear()

    # ── Message Formatting ────────────────────────────────────────────────

    @staticmethod
    def _format_pick_message(game) -> str:
        """Format a high-confidence pick as a Telegram message.

        Uses clean ASCII formatting — no emoji, no unicode decorative chars.
        """

        # Direction
        direction_str = game.direction.upper() if game.direction else "?"
        side = "HOME" if direction_str == "HOME" else "AWAY"

        # Edge
        edge_pct = (game.edge_pct or 0.0) * 100

        # Confidence label
        conf = (game.confidence or "low").upper()

        # Total prediction
        total_str = ""
        if game.total_prediction is not None and game.market_total:
            total_edge = (game.total_edge_pct or 0.0) * 100
            total_dir = (game.total_direction or "neutral").upper()
            total_str = (
                f"\n  Total: {game.total_prediction:.0f} (market {game.market_total:.0f}) "
                f"| Edge: {total_edge:+.1f}% | {total_dir}"
            )

        # Stake
        stake_str = f" | Stake: ${game.stake_dollars:.0f}" if game.stake_dollars else ""

        # League context
        league = game.league or "NBA"

        # Build message
        msg = (
            f"[HIGH-CONFIDENCE PICK]\n"
            f"{game.matchup}\n"
            f"League: {league}\n"
            f"Side: {side} ML | Edge: {edge_pct:+.1f}% | Confidence: {conf}{stake_str}"
            f"{total_str}"
        )

        # Timestamp
        if game.commence_time:
            msg += f"\nTip-off: {game.commence_time[:16]}"

        return msg

    @staticmethod
    def _format_digest(games: list) -> str:
        """Format multiple high-confidence picks as a single digest message."""
        if not games:
            return ""
        lines = ["[DIGEST] High-Confidence Picks"]
        for i, game in enumerate(games, 1):
            direction_str = game.direction.upper() if game.direction else "?"
            edge_pct = (game.edge_pct or 0.0) * 100
            conf = (game.confidence or "low").upper()
            stake_str = f" ${game.stake_dollars:.0f}" if game.stake_dollars else ""
            lines.append(
                f"{i}. {game.matchup} | {direction_str} | "
                f"Edge: {edge_pct:+.1f}% | {conf}{stake_str}"
            )
        return "\n".join(lines)

    # ── Sending ───────────────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "") -> bool:
        """Send a plain text message to the configured Telegram chat.

        Args:
            text: Message content.
            parse_mode: Optional parse mode (e.g., "HTML", "MarkdownV2").

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self.is_configured:
            logger.debug("Telegram not configured — skipping message")
            return False

        try:
            import httpx

            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": text,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                result = resp.json()
                if result.get("ok"):
                    logger.debug("Telegram message sent successfully")
                    return True
                else:
                    logger.warning(
                        f"Telegram API error: {result.get('description', 'unknown')}"
                    )
                    return False
        except ImportError:
            logger.warning("httpx not installed — cannot send Telegram message")
            return False
        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")
            return False

    async def send_pick_alert(self, game) -> bool:
        """Send a high-confidence pick alert for a single game.

        Skips if the game was already notified.

        Args:
            game: A LiveGame instance with predictions populated.

        Returns:
            True if sent, False if skipped or failed.
        """
        if not self.is_configured:
            return False

        if self.is_notified(game.game_id):
            logger.debug(f"Game {game.game_id} already notified — skipping")
            return False

        msg = self._format_pick_message(game)
        success = await self.send_message(msg)

        if success:
            self.mark_as_notified(game.game_id)

        return success

    async def send_digest(self, games: list) -> int:
        """Send a digest of multiple high-confidence picks as one message.

        Args:
            games: List of LiveGame instances with predictions.

        Returns:
            Number of games in the digest.
        """
        if not self.is_configured or not games:
            return 0

        # Filter out already-notified games
        new_games = [g for g in games if not self.is_notified(g.game_id)]
        if not new_games:
            return 0

        msg = self._format_digest(new_games)
        success = await self.send_message(msg)

        if success:
            for g in new_games:
                self.mark_as_notified(g.game_id)

        return len(new_games) if success else 0

    # ── Test / Verification ──────────────────────────────────────────────

    async def send_test_message(self) -> str:
        """Send a test message to verify the Telegram configuration.

        Returns:
            Status message describing the result.
        """
        if not self.is_configured:
            return (
                "Telegram not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or environment."
            )

        msg = (
            "[TEST] Betting Intelligence System\n"
            "Telegram notifications configured correctly.\n"
            "You will receive high-confidence pick alerts here."
        )

        success = await self.send_message(msg)
        if success:
            return "Test message sent successfully! Check your Telegram chat."
        else:
            return "Failed to send test message. Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."

    # ── Synchronous convenience ──────────────────────────────────────────

    def send_test_message_sync(self) -> str:
        """Synchronous wrapper for send_test_message."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # If a loop is running (e.g., inside the live engine), use it
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(
                    self.send_test_message(), loop
                ).result(timeout=15)
        except RuntimeError:
            pass
        return asyncio.run(self.send_test_message())

    def send_pick_alert_sync(self, game) -> bool:
        """Synchronous wrapper for send_pick_alert."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(
                    self.send_pick_alert(game), loop
                ).result(timeout=15)
        except RuntimeError:
            pass
        return asyncio.run(self.send_pick_alert(game))

    def send_digest_sync(self, games: list) -> int:
        """Synchronous wrapper for send_digest."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(
                    self.send_digest(games), loop
                ).result(timeout=15)
        except RuntimeError:
            pass
        return asyncio.run(self.send_digest(games))
