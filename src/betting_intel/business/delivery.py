"""
Delivery System — THE DISTRIBUTION ENGINE.

Automated delivery of betting picks to subscribers via:
  - Telegram (primary — highest engagement)
  - Email (secondary — professional)
  - Discord (community)

Every morning (configurable), the system:
  1. Runs the full prediction pipeline
  2. Generates the DailyBettingCard
  3. Filters by subscriber tier
  4. Delivers to each subscriber's preferred channel

Usage:
    deliverer = PickDeliverer()
    deliverer.distribute_daily_picks()
"""

from __future__ import annotations

import json
import logging
import os
import random
import smtplib
import time as time_module
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from betting_intel.business.report import (
    GameAnalysisGenerator,
    DailyBettingCard,
    GameAnalysisReport,
)
from betting_intel.business.subscriptions import (
    SubscriptionManager,
    SubscriptionTier,
)

logger = logging.getLogger(__name__)


# ── Retry helper for external API calls ──────────────────────────────────

def _retry_with_backoff(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Call fn with exponential backoff + jitter. Returns (ok, result).

    BLOCKING — uses time.sleep(), do not call from async handlers.
    """
    for attempt in range(max_retries):
        try:
            result = fn()
            return True, result
        except Exception as e:
            if attempt == max_retries - 1:
                return False, e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.debug(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
            time_module.sleep(delay)
    return False, None  # unreachable but satisfies type checker


# ═══════════════════════════════════════════════════════════════════════════
#  TELEGRAM BOT — PRIMARY DELIVERY CHANNEL
# ═══════════════════════════════════════════════════════════════════════════

class TelegramDeliverer:
    """
    Delivers picks to subscribers via Telegram.

    Uses the Telegram Bot API directly (no heavy framework needed).
    Supports: private messages, channel posts, group delivery.

    Usage:
        bot = TelegramDeliverer(token="your-bot-token")
        bot.send_pick(chat_id="123456", "🎯 Pick of the Day: ...")
    """

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._enabled = bool(self.token) and self.token != "your-telegram-bot-token"

    def is_enabled(self) -> bool:
        return self._enabled

    def send_message(self, chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to a Telegram chat with retry + backoff."""
        if not self._enabled:
            logger.debug(f"Telegram not configured. Would send to {chat_id}: {text[:100]}...")
            return False

        def _do_send():
            import httpx
            url = self.BASE_URL.format(token=self.token, method="sendMessage")
            response = httpx.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }, timeout=10)
            response.raise_for_status()
            return response.status_code == 200

        ok, result = _retry_with_backoff(_do_send, max_retries=3, base_delay=1.0)
        if not ok:
            logger.error(f"Telegram send failed after retries: {result}")
        return ok

    def send_card(self, chat_id: str, card: DailyBettingCard) -> bool:
        """Send a complete daily betting card."""
        return self.send_message(chat_id, card.to_telegram())

    def send_report(self, chat_id: str, report: GameAnalysisReport) -> bool:
        """Send a single game analysis report."""
        return self.send_message(chat_id, report.to_telegram())

    def broadcast(self, chat_ids: list[str], text: str) -> dict[str, bool]:
        """Send the same message to multiple chats. Returns {chat_id: success}."""
        results = {}
        for cid in chat_ids:
            results[cid] = self.send_message(cid, text)
        return results

    def broadcast_card(self, subscribers: list, card: DailyBettingCard) -> dict[str, bool]:
        """Send the daily card to all subscribers with Telegram delivery."""
        results = {}
        for sub in subscribers:
            if sub.telegram_chat_id:
                # Filter content based on tier
                if sub.tier == SubscriptionTier.FREE:
                    # Free tier gets only the best play
                    if card.best_play:
                        text = (
                            f"🎯 **FREE PICK — {datetime.now().strftime('%b %d')}**\n\n"
                            f"{card.best_play['description']}\n\n"
                            f"📊 Upgrade to Basic ($49/mo) for the full daily card!\n"
                            f"👉 https://yourdomain.com/pricing"
                        )
                        results[sub.telegram_chat_id] = self.send_message(sub.telegram_chat_id, text)
                elif sub.tier == SubscriptionTier.BASIC:
                    results[sub.telegram_chat_id] = self.send_card(sub.telegram_chat_id, card)
                elif sub.tier in (SubscriptionTier.PREMIUM, SubscriptionTier.ELITE):
                    # Premium gets enhanced card with more detail
                    results[sub.telegram_chat_id] = self.send_card(sub.telegram_chat_id, card)
            else:
                results["no_chat_id"] = False
        return results

    def send_live_alert(self, chat_id: str, alert_text: str) -> bool:
        """Send a live in-game alert (Premium+ feature)."""
        return self.send_message(chat_id, f"🔴 **LIVE ALERT**\n\n{alert_text}")


# ═══════════════════════════════════════════════════════════════════════════
#  EMAIL DELIVERER — SECONDARY CHANNEL
# ═══════════════════════════════════════════════════════════════════════════

class EmailDeliverer:
    """
    Delivers picks via email (for subscribers who prefer email over Telegram).

    Uses SMTP (SendGrid, Mailgun, or direct SMTP).
    """

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: Optional[str] = None,
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME", "")
        self.password = password or os.getenv("SMTP_PASSWORD", "")
        self.from_addr = from_addr or os.getenv("SMTP_FROM_ADDR", "picks@yourdomain.com")
        self._enabled = bool(self.smtp_server and self.username)

    def is_enabled(self) -> bool:
        return self._enabled

    def send_card(self, to_email: str, card: DailyBettingCard) -> bool:
        """Send the daily card as a formatted email."""
        if not self._enabled:
            logger.debug(f"Email not configured. Would send to {to_email}")
            return False

        try:
            msg = MIMEText(card.to_markdown(), "plain", "utf-8")
            msg["Subject"] = f"🎯 Daily Betting Card — {card.date}"
            msg["From"] = self.from_addr
            msg["To"] = to_email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent to {to_email}")
            return True
        except Exception as e:
            logger.warning(f"Email delivery failed for {to_email}: {e}")
            return False

    def broadcast_card(self, subscribers: list, card: DailyBettingCard) -> dict[str, bool]:
        """Send the daily card to all subscribers via a single SMTP connection."""
        results: dict[str, bool] = {}
        if not self._enabled:
            for sub in subscribers:
                if sub.email:
                    results[sub.email] = False
            return results

        # Build all messages first
        messages = []
        for sub in subscribers:
            if sub.email:
                msg = MIMEText(card.to_markdown(), "plain", "utf-8")
                msg["Subject"] = f"🎯 Daily Betting Card — {card.date}"
                msg["From"] = self.from_addr
                msg["To"] = sub.email
                messages.append((sub.email, msg))

        if not messages:
            return results

        # Single SMTP connection for all recipients
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                for email, msg in messages:
                    try:
                        server.send_message(msg)
                        results[email] = True
                    except Exception as e:
                        logger.warning(f"Email delivery failed for {email}: {e}")
                        results[email] = False
        except Exception as e:
            logger.error(f"SMTP connection failed: {e}")
            for email, _ in messages:
                results[email] = False

        sent = sum(1 for v in results.values() if v)
        logger.info(f"Email broadcast: {sent}/{len(results)} sent via single SMTP connection")
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER DELIVERER — ORCHESTRATES ALL CHANNELS
# ═══════════════════════════════════════════════════════════════════════════

class PickDeliverer:
    """
    Master delivery orchestrator.

    Every morning:
      1. Generates the daily betting card via the pipeline
      2. Filters content per subscriber tier
      3. Delivers via Telegram and/or email
      4. Logs delivery results

    Usage:
        deliverer = PickDeliverer()
        results = deliverer.distribute_daily_picks()
        print(f"Delivered to {results['success_count']}/{results['total_count']} subscribers")
    """

    def __init__(
        self,
        subscribers_db: str = "data/subscribers.json",
        bankroll: float = 10_000.0,
    ):
        self.sub_manager = SubscriptionManager(subscribers_db)
        self.telegram = TelegramDeliverer()
        self.email = EmailDeliverer()
        self.generator = GameAnalysisGenerator(bankroll=bankroll)

    def distribute_daily_picks(self, pipeline_results: Optional[dict] = None) -> dict:
        """
        Full daily distribution workflow.

        1. Get active subscribers (single lookup)
        2. Build the daily betting card from pipeline results
        3. Deliver to each subscriber based on tier
        4. Return delivery results

        Args:
            pipeline_results: Output from PredictionPipeline.run(). If None,
                              the method generates an empty card and returns early
                              with an error — no placeholder data is distributed.

        Returns:
            dict with delivery counts and errors
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        results = {
            "date": today,
            "telegram_sent": 0,
            "email_sent": 0,
            "total_count": 0,
            "success_count": 0,
            "errors": [],
        }

        # ── Generate the daily card (REQUIRES real pipeline results) ─
        if not pipeline_results or not isinstance(pipeline_results, dict):
            results["error"] = "No pipeline results provided. Pass output from PredictionPipeline.run()."
            results["note"] = "Daily card requires real predictions — no placeholder data distributed."
            logger.warning(results["error"])
            return results

        try:
            card = self.generator.generate_daily_card(pipeline_results)
        except Exception as e:
            results["error"] = f"Failed to generate daily card: {e}"
            results["note"] = "No data distributed due to card generation failure."
            logger.error(results["error"])
            return results

        # ── Get subscribers (one lookup per paid tier) ──────────────
        all_subs = self.sub_manager.get_subscribers_for_telegram("basic")
        all_subs += self.sub_manager.get_subscribers_for_telegram("premium")
        all_subs += self.sub_manager.get_subscribers_for_telegram("elite")
        free_subs = self.sub_manager.get_subscribers_for_telegram("free")

        if not all_subs and not free_subs:
            logger.info("No subscribers to deliver to")
            results["note"] = "No active subscribers"
            return results

        results["total_count"] = len(all_subs) + len(free_subs)

        # ── Deliver to subscribers ─────────────────────────────────
        if free_subs:
            free_pick = card.best_play or {
                "description": f"🎯 Free pick for {today} — upgrade for the full card.",
            }
            free_text = (
                f"🎯 **FREE PICK — {today}**\n\n"
                f"{free_pick['description']}\n\n"
                f"📊 Upgrade to Basic ($49/mo) for the full daily card\n"
                f"👉 https://yourdomain.com/pricing"
            )
            tg_results = self.telegram.broadcast(
                [s.telegram_chat_id for s in free_subs if s.telegram_chat_id],
                free_text,
            )
            results["telegram_sent"] += sum(1 for v in tg_results.values() if v)

        # Basic+ subscribers get the full card
        if all_subs:
            tg_results = self.telegram.broadcast_card(all_subs, card)
            results["telegram_sent"] += sum(1 for v in tg_results.values() if v)

            email_subs = [s for s in all_subs if s.email]
            if email_subs and self.email.is_enabled():
                email_results = self.email.broadcast_card(email_subs, card)
                results["email_sent"] += sum(1 for v in email_results.values() if v)

        results["success_count"] = results["telegram_sent"]
        logger.info(
            f"Daily picks distributed: {results['telegram_sent']} Telegram, "
            f"{results['email_sent']} Email to {results['total_count']} subscribers"
        )

        return results
