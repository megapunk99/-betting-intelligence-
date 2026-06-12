"""
Alert Dispatcher — Sends underperformance alerts via Slack and Email.

Integrates with ResultsTracker.check_alerts() to push notifications
when any betting strategy drops below the -5% ROI threshold.

Two channels:
  1. Slack Webhook (primary — immediate)
  2. Email (secondary — durable record)

Usage:
    from betting_intel.analytics.alerting import AlertDispatcher
    dispatcher = AlertDispatcher()
    dispatcher.send_strategy_alert(strategy_perf)
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  SLACK ALERT DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

class SlackAlertSender:
    """Send underperformance alerts to a Slack channel via webhook.

    Configure with SLACK_WEBHOOK_URL environment variable.
    Falls back to a no-op when not configured.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        self._enabled = bool(self.webhook_url) and "hooks.slack.com" in self.webhook_url

    def is_enabled(self) -> bool:
        return self._enabled

    def send_alert(self, strategy_name: str, roi: float, n_bets: int,
                   total_profit: float, trailing_profit: float,
                   league: str, model: str, bet_type: str) -> bool:
        """Send a formatted strategy underperformance alert to Slack."""
        if not self._enabled:
            logger.info(f"Slack not configured. Would alert: {strategy_name} ROI={roi:.1%}")
            return False

        color = "#ef4444"  # Red for underperformance
        emoji = "🚨"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Strategy Underperformance Alert",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Strategy:*\n{strategy_name}"},
                    {"type": "mrkdwn", "text": f"*ROI:*\n{roi:.1%}"},
                    {"type": "mrkdwn", "text": f"*Bets:*\n{n_bets}"},
                    {"type": "mrkdwn", "text": f"*Total P&L:*\n${total_profit:+.0f}"},
                    {"type": "mrkdwn", "text": f"*Trailing 30d:*\n${trailing_profit:+.0f}"},
                    {"type": "mrkdwn", "text": f"*League:*\n{league}"},
                    {"type": "mrkdwn", "text": f"*Model:*\n{model}"},
                    {"type": "mrkdwn", "text": f"*Bet Type:*\n{bet_type}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"⚠️ This strategy has fallen below the -5% ROI threshold. "
                        f"Consider pausing or re-tuning this strategy immediately."
                    ),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Betting Intelligence · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                    },
                ],
            },
        ]

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                    "mrkdwn_in": ["text", "fields"],
                },
            ],
        }

        try:
            import httpx
            resp = httpx.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"Slack alert sent for {strategy_name}")
            return True
        except Exception as e:
            logger.warning(f"Slack alert failed for {strategy_name}: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  EMAIL ALERT SENDER
# ═══════════════════════════════════════════════════════════════════════════

class EmailAlertSender:
    """Send underperformance alerts via email.

    Configure with SMTP_* environment variables (same as existing EmailDeliverer).
    Falls back to a no-op when not configured.
    """

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: Optional[str] = None,
        to_addr: Optional[str] = None,
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.username = username or os.getenv("SMTP_USERNAME", "")
        self.password = password or os.getenv("SMTP_PASSWORD", "")
        self.from_addr = from_addr or os.getenv("SMTP_FROM_ADDR", "alerts@bettingintel.com")
        self.to_addr = to_addr or os.getenv("SMTP_ALERT_TO", "")
        self._enabled = bool(self.smtp_server and self.username and self.to_addr)

    def is_enabled(self) -> bool:
        return self._enabled

    def send_alert(self, strategy_name: str, roi: float, n_bets: int,
                   total_profit: float, trailing_profit: float,
                   league: str, model: str, bet_type: str) -> bool:
        """Send a formatted strategy underperformance alert via email."""
        if not self._enabled:
            logger.info(f"Email not configured. Would alert about {strategy_name}")
            return False

        subject = f"🚨 Strategy Alert: {strategy_name} ROI={roi:.1%}"

        body = f"""
=== STRATEGY UNDERPERFORMANCE ALERT ===

Strategy:     {strategy_name}
ROI:          {roi:.1%}
Total P&L:    ${total_profit:+.0f}
Trailing 30d: ${trailing_profit:+.0f}
Bets Placed:  {n_bets}
League:       {league}
Model:        {model}
Bet Type:     {bet_type}

⚠️ This strategy has fallen below the -5% ROI threshold.
Consider pausing or re-tuning this strategy immediately.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = self.to_addr

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Alert email sent for {strategy_name} to {self.to_addr}")
            return True
        except Exception as e:
            logger.warning(f"Alert email failed for {strategy_name}: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER ALERT DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

class AlertDispatcher:
    """Dispatch strategy alerts through ALL configured channels.

    Usage:
        dispatcher = AlertDispatcher()
        alerts_sent = dispatcher.dispatch_alerts(report.alerted_strategies)
        # Returns {strategy_name: {"slack": bool, "email": bool}}
    """

    def __init__(self):
        self.slack = SlackAlertSender()
        self.email = EmailAlertSender()
        self._any_enabled = self.slack.is_enabled() or self.email.is_enabled()

    def is_enabled(self) -> bool:
        return self._any_enabled

    def dispatch_alerts(self, alerted_strategies: list) -> dict[str, dict[str, bool]]:
        """Dispatch alerts for all underperforming strategies through all channels.

        Args:
            alerted_strategies: List of StrategyPerformance objects that are alerted.

        Returns:
            Dict mapping strategy_name -> {"slack": success, "email": success}
        """
        results: dict[str, dict[str, bool]] = {}

        for strategy in alerted_strategies:
            name = strategy.strategy_name
            results[name] = {"slack": False, "email": False}

            if self.slack.is_enabled():
                ok = self.slack.send_alert(
                    strategy_name=name,
                    roi=strategy.roi,
                    n_bets=strategy.n_bets,
                    total_profit=strategy.total_profit,
                    trailing_profit=sum(strategy.trailing_profits),
                    league=strategy.league,
                    model=strategy.model,
                    bet_type=strategy.bet_type,
                )
                results[name]["slack"] = ok

            if self.email.is_enabled():
                ok = self.email.send_alert(
                    strategy_name=name,
                    roi=strategy.roi,
                    n_bets=strategy.n_bets,
                    total_profit=strategy.total_profit,
                    trailing_profit=sum(strategy.trailing_profits),
                    league=strategy.league,
                    model=strategy.model,
                    bet_type=strategy.bet_type,
                )
                results[name]["email"] = ok

        n_total = len(alerted_strategies)
        n_sent = sum(1 for r in results.values() if r.get("slack") or r.get("email"))
        if n_total > 0:
            logger.info(f"Alert dispatch: {n_sent}/{n_total} strategies notified")

        return results


__all__ = ["AlertDispatcher", "SlackAlertSender", "EmailAlertSender"]
