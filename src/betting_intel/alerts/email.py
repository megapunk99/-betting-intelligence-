"""
Email notification channel for betting alerts.

Sends pipeline completion/failure alerts and daily summaries via SMTP.
Supports Gmail (smtp.gmail.com:587) with App Passwords.

Usage:
    notifier = EmailNotifier(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="you@gmail.com",
        password="your-app-password",
        from_addr="you@gmail.com",
        to_addr="you@gmail.com",
    )
    notifier.send_message("Pipeline completed successfully")

Environment variables:
    SMTP_SERVER       — SMTP server hostname (default: smtp.gmail.com)
    SMTP_PORT         — SMTP server port (default: 587)
    SMTP_USERNAME     — SMTP login username (usually your email)
    SMTP_PASSWORD     — SMTP login password (Gmail App Password recommended)
    SMTP_FROM_ADDR    — From address for emails (default: same as username)
    SMTP_TO_ADDR      — To address for notifications (default: same as username)
    ENABLE_EMAIL      — Set to "true" to enable email notifications
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    """A structured email message ready to send."""

    subject: str
    body: str
    html: bool = False
    sent_at: Optional[datetime] = None
    error: Optional[str] = None


class EmailNotifier:
    """Sends pipeline notifications via SMTP email.

    Designed for Gmail with App Passwords (smtp.gmail.com:587, TLS).
    Follows the same pattern as TelegramBot and DiscordWebhook for
    compatibility with the AlertDispatcher system.
    """

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: Optional[str] = None,
        to_addr: Optional[str] = None,
        use_tls: bool = True,
        timeout: int = 30,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr or username or ""
        self.to_addr = to_addr or from_addr or username or ""
        self.use_tls = use_tls
        self.timeout = timeout

    async def _send_raw(self, subject: str, body: str, html: bool = False) -> EmailMessage:
        """Send an email via SMTP asynchronously."""
        return await asyncio.to_thread(self._send_raw_sync, subject, body, html)

    def _send_raw_sync(self, subject: str, body: str, html: bool = False) -> EmailMessage:
        """Synchronous implementation of email sending. Wrapped by _send_raw for async usage."""
        if not self.username or not self.password:
            return EmailMessage(
                subject=subject,
                body=body,
                error="EmailNotifier not configured: missing username or password",
            )
        if not self.to_addr:
            return EmailMessage(
                subject=subject,
                body=body,
                error="EmailNotifier not configured: missing to_addr",
            )

        msg = MIMEMultipart("alternative")
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg["Subject"] = subject

        # Attach plain text and HTML versions
        msg.attach(MIMEText(body, "html" if html else "plain"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, [self.to_addr], msg.as_string())

            logger.info(f"Email sent: {subject}")
            return EmailMessage(subject=subject, body=body, sent_at=datetime.now())

        except smtplib.SMTPAuthenticationError as exc:
            error = f"SMTP auth failed: {exc}"
            logger.error(error)
            return EmailMessage(subject=subject, body=body, error=error)
        except smtplib.SMTPException as exc:
            error = f"SMTP error: {exc}"
            logger.error(error)
            return EmailMessage(subject=subject, body=body, error=error)
        except OSError as exc:
            error = f"Connection error: {exc}"
            logger.error(error)
            return EmailMessage(subject=subject, body=body, error=error)

    # ── Public API (compatible with TelegramBot / DiscordWebhook) ─────

    async def send_message(self, text: str, **kwargs) -> EmailMessage:
        """Send a plain text notification email."""
        return await self._send_raw(
            subject="Betting Intelligence — Notification",
            body=text,
        )

    async def send_pipeline_alert(
        self,
        status: str,
        exit_code: int,
        duration_seconds: float,
        summary: str = "",
        log_path: str = "",
    ) -> EmailMessage:
        """Send a structured pipeline completion/failure alert."""
        if status == "success":
            emoji = "✅"
            subject = f"✅ Pipeline Completed Successfully"
        elif status == "partial":
            emoji = "⚠️"
            subject = f"⚠️ Pipeline Completed with Warnings"
        else:
            emoji = "❌"
            subject = f"❌ Pipeline Failed"

        duration_str = f"{duration_seconds:.0f}s" if duration_seconds < 120 else f"{duration_seconds / 60:.1f}m"

        html = f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<table cellpadding="0" cellspacing="0" style="width: 100%; border-collapse: collapse;">
<tr>
<td style="text-align: center; padding: 30px 0; font-size: 48px;">
{emoji}
</td>
</tr>
<tr>
<td style="background: {'#d4edda' if status == 'success' else '#fff3cd' if status == 'partial' else '#f8d7da'}; border-radius: 8px; padding: 24px;">
<h1 style="margin: 0 0 8px; font-size: 20px; color: {'#155724' if status == 'success' else '#856404' if status == 'partial' else '#721c24'};">
{subject}
</h1>
<p style="margin: 0; color: {'#155724' if status == 'success' else '#856404' if status == 'partial' else '#721c24'}; font-size: 14px;">
{datetime.now().strftime('%B %d, %Y at %H:%M')}
</p>
</td>
</tr>
</table>

<table cellpadding="0" cellspacing="0" style="width: 100%; margin-top: 20px; border-collapse: collapse;">
<tr>
<td style="padding: 12px; border-bottom: 1px solid #eee;"><strong>Status</strong></td>
<td style="padding: 12px; border-bottom: 1px solid #eee;">{status.upper()}</td>
</tr>
<tr>
<td style="padding: 12px; border-bottom: 1px solid #eee;"><strong>Exit Code</strong></td>
<td style="padding: 12px; border-bottom: 1px solid #eee;">{exit_code}</td>
</tr>
<tr>
<td style="padding: 12px; border-bottom: 1px solid #eee;"><strong>Duration</strong></td>
<td style="padding: 12px; border-bottom: 1px solid #eee;">{duration_str}</td>
</tr>
</table>

{f'<div style="margin-top: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px; font-family: monospace; font-size: 12px; white-space: pre-wrap;">{summary}</div>' if summary else ''}

{f'<div style="margin-top: 16px; font-size: 12px; color: #888;">Log: {log_path}</div>' if log_path else ''}

<div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #eee; font-size: 11px; color: #aaa; text-align: center;">
Betting Intelligence Pipeline &bull; {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
</div>
</body>
</html>"""

        return await self._send_raw(subject=subject, body=html, html=True)

    async def send_bet_alert(
        self,
        matchup: str,
        bet_type: str,
        edge_pct: float,
        confidence: float,
        stake: float,
        league: str = "NBA",
        reasoning: str = "",
    ) -> EmailMessage:
        """Send a structured bet alert via email."""
        emoji = "🟢" if edge_pct > 5 else "🔵" if edge_pct > 3 else "⚪"
        subject = f"{emoji} Bet Alert: {matchup} — {bet_type} ({edge_pct:.1f}%)"

        html = f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="font-size: 20px;">{emoji} Bet Alert</h1>
<table cellpadding="8" cellspacing="0" style="width: 100%; border-collapse: collapse;">
<tr><td style="border-bottom: 1px solid #eee;"><strong>Matchup</strong></td><td>{matchup}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Bet</strong></td><td>{bet_type}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Edge</strong></td><td>{edge_pct:.1f}%</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Confidence</strong></td><td>{confidence:.0%}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Stake</strong></td><td>${stake:.0f}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>League</strong></td><td>{league}</td></tr>
</table>
{f'<p style="margin-top: 16px; font-style: italic;">{reasoning}</p>' if reasoning else ''}
<p style="margin-top: 16px; font-size: 11px; color: #aaa;">{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
</body>
</html>"""

        return await self._send_raw(subject=subject, body=html, html=True)

    async def send_daily_summary(
        self,
        total_bets: int,
        wins: int,
        losses: int,
        profit: float,
        roi: float,
        best_bet: str = "",
    ) -> EmailMessage:
        """Send a daily betting performance summary."""
        profit_emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
        win_rate = wins / max(total_bets, 1)
        subject = f"{profit_emoji} Daily Summary: {wins}W/{losses}L ({profit:+.1f}u)"

        html = f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="font-size: 20px;">📊 Daily Performance Summary</h1>
<h2 style="font-size: 14px; color: #666;">{datetime.now().strftime('%B %d, %Y')}</h2>
<table cellpadding="8" cellspacing="0" style="width: 100%; border-collapse: collapse;">
<tr><td style="border-bottom: 1px solid #eee;"><strong>Bets</strong></td><td>{total_bets} ({wins}W / {losses}L)</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Win Rate</strong></td><td>{win_rate:.1%}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>P&amp;L</strong></td><td><strong>{profit:+.1f}u</strong> ({roi:+.1f}%)</td></tr>
</table>
{f'<p style="margin-top: 16px;"><strong>⭐ Best Bet:</strong> {best_bet}</p>' if best_bet else ''}
<p style="margin-top: 16px; font-size: 11px; color: #aaa;">{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
</body>
</html>"""

        return await self._send_raw(subject=subject, body=html, html=True)

    async def send_health_report(
        self,
        status: str,
        models_active: int,
        leagues_tracked: int,
        errors_last_hour: int,
    ) -> EmailMessage:
        """Send system health report."""
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "🚨"
        subject = f"{status_emoji} System Health: {status.upper()}"

        html = f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="font-size: 20px;">{status_emoji} System Health Report</h1>
<table cellpadding="8" cellspacing="0" style="width: 100%; border-collapse: collapse;">
<tr><td style="border-bottom: 1px solid #eee;"><strong>Status</strong></td><td>{status.upper()}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Active Models</strong></td><td>{models_active}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Leagues Tracked</strong></td><td>{leagues_tracked}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Errors (1h)</strong></td><td>{errors_last_hour}</td></tr>
</table>
<p style="margin-top: 16px; font-size: 11px; color: #aaa;">{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
</body>
</html>"""

        return await self._send_raw(subject=subject, body=html, html=True)

    async def send_line_movement(
        self,
        matchup: str,
        market: str,
        old_line: float,
        new_line: float,
        direction: str,
        league: str = "NBA",
    ) -> EmailMessage:
        """Send a line movement alert."""
        arrow = "📈" if direction == "up" else "📉"
        subject = f"{arrow} Line Movement: {matchup} — {market} ({old_line:.1f} → {new_line:.1f})"

        html = f"""\
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="font-size: 20px;">{arrow} Line Movement</h1>
<table cellpadding="8" cellspacing="0" style="width: 100%; border-collapse: collapse;">
<tr><td style="border-bottom: 1px solid #eee;"><strong>Matchup</strong></td><td>{matchup}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Market</strong></td><td>{market}</td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Change</strong></td><td>{old_line:.1f} → <strong>{new_line:.1f}</strong></td></tr>
<tr><td style="border-bottom: 1px solid #eee;"><strong>Direction</strong></td><td>{direction.upper()}</td></tr>
</table>
<p style="margin-top: 16px; font-size: 11px; color: #aaa;">{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
</body>
</html>"""

        return await self._send_raw(subject=subject, body=html, html=True)

    async def test_connection(self) -> bool:
        """Test SMTP connectivity by attempting to connect and authenticate."""
        if not self.username or not self.password:
            logger.error("EmailNotifier test failed: not configured")
            return False
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
            logger.info("Email SMTP connection test: OK")
            return True
        except Exception as exc:
            logger.error(f"Email SMTP connection test failed: {exc}")
            return False
