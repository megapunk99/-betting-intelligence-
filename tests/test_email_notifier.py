"""Tests for the EmailNotifier alert channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from betting_intel.alerts.email import EmailMessage, EmailNotifier


@pytest.fixture
def notifier():
    """An EmailNotifier configured with mock credentials."""
    return EmailNotifier(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="test@gmail.com",
        password="app-password-1234",
        from_addr="test@gmail.com",
        to_addr="test@gmail.com",
    )


@pytest.fixture
def unconfigured_notifier():
    """An EmailNotifier with no credentials (should skip sending)."""
    return EmailNotifier()


class TestEmailNotifierInit:
    """Constructor and configuration tests."""

    def test_defaults(self):
        """Defaults should be smtp.gmail.com:587."""
        n = EmailNotifier()
        assert n.smtp_server == "smtp.gmail.com"
        assert n.smtp_port == 587
        assert n.username is None
        assert n.password is None

    def test_custom_values(self):
        n = EmailNotifier(
            smtp_server="smtp.example.com",
            smtp_port=465,
            username="user@example.com",
            password="secret",
            from_addr="alerts@example.com",
            to_addr="admin@example.com",
        )
        assert n.smtp_server == "smtp.example.com"
        assert n.smtp_port == 465
        assert n.username == "user@example.com"
        assert n.password == "secret"
        assert n.from_addr == "alerts@example.com"
        assert n.to_addr == "admin@example.com"

    def test_from_addr_defaults_to_username(self):
        n = EmailNotifier(username="user@example.com", password="pw")
        assert n.from_addr == "user@example.com"

    def test_to_addr_defaults_to_from_addr(self):
        n = EmailNotifier(
            username="user@example.com",
            password="pw",
            from_addr="alerts@example.com",
        )
        assert n.to_addr == "alerts@example.com"


class TestEmailNotifierSendRaw:
    """Tests for the core _send_raw method (via `await`)."""

    @pytest.mark.asyncio
    async def test_no_credentials_returns_error(self, unconfigured_notifier):
        """Missing credentials should return an error message, not crash."""
        msg = await unconfigured_notifier._send_raw(
            subject="Test", body="Hello"
        )
        assert msg.error is not None
        assert "not configured" in msg.error

    @pytest.mark.asyncio
    async def test_missing_to_addr(self):
        n = EmailNotifier(username="u@x.com", password="pw")
        msg = await n._send_raw(subject="Test", body="Hello")
        assert msg.error is not None

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_successful_send(self, mock_smtp, notifier):
        """A successful SMTP send should return a message with sent_at."""
        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance

        msg = await notifier._send_raw(subject="Test Subj", body="Test Body")

        assert msg.error is None
        assert msg.subject == "Test Subj"
        assert msg.body == "Test Body"
        assert msg.sent_at is not None
        mock_instance.sendmail.assert_called_once()
        mock_instance.starttls.assert_called_once()

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_sends_html_content(self, mock_smtp, notifier):
        """HTML content type should be used when html=True."""
        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance

        msg = await notifier._send_raw(
            subject="HTML Test", body="<h1>Hello</h1>", html=True
        )

        assert msg.error is None
        # The MIME type should be multipart/alternative
        call_args = mock_instance.sendmail.call_args
        sent_message = call_args[0][2]
        assert "text/html" in sent_message

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_auth_failure(self, mock_smtp, notifier):
        """SMTP auth failure should return an error message."""
        from smtplib import SMTPAuthenticationError

        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance
        mock_instance.login.side_effect = SMTPAuthenticationError(
            535, b"Authentication failed"
        )

        msg = await notifier._send_raw(subject="Auth Fail", body="test")
        assert msg.error is not None
        assert "auth failed" in msg.error.lower()

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_connection_error(self, mock_smtp, notifier):
        """Connection error should return an error message."""
        import socket

        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance
        mock_instance.starttls.side_effect = OSError("Connection refused")

        msg = await notifier._send_raw(subject="Conn Fail", body="test")
        assert msg.error is not None
        assert "connection" in msg.error.lower() or "error" in msg.error.lower()


class TestEmailNotifierPublicAPI:
    """Tests for the public API methods (send_message, send_pipeline_alert, etc.)."""

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_message(self, mock_send_raw, notifier):
        """send_message should call _send_raw with the right args."""
        await notifier.send_message("Hello world")
        mock_send_raw.assert_awaited_once()
        args, kwargs = mock_send_raw.call_args
        assert "Notification" in kwargs.get("subject", "")
        assert kwargs.get("body") == "Hello world"

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_pipeline_alert_success(self, mock_send_raw, notifier):
        """Success pipeline alert should have ✅ in subject and HTML body."""
        await notifier.send_pipeline_alert(
            status="success",
            exit_code=0,
            duration_seconds=45.2,
            summary="All good",
            log_path="/tmp/pipeline.log",
        )
        mock_send_raw.assert_awaited_once()
        args, kwargs = mock_send_raw.call_args
        assert "✅" in kwargs["subject"]
        assert kwargs["html"] is True
        assert "All good" in kwargs["body"]

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_pipeline_alert_failed(self, mock_send_raw, notifier):
        """Failure pipeline alert should have ❌ in subject."""
        await notifier.send_pipeline_alert(
            status="failed",
            exit_code=1,
            duration_seconds=10.0,
        )
        args, kwargs = mock_send_raw.call_args
        assert "❌" in kwargs["subject"]

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_pipeline_alert_partial(self, mock_send_raw, notifier):
        """Partial/warning alert should have ⚠️ in subject."""
        await notifier.send_pipeline_alert(
            status="partial",
            exit_code=0,
            duration_seconds=120.0,
        )
        args, kwargs = mock_send_raw.call_args
        assert "⚠️" in kwargs["subject"]

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_bet_alert(self, mock_send_raw, notifier):
        """Bet alert should contain matchup, bet type, edge."""
        await notifier.send_bet_alert(
            matchup="LAL @ BOS",
            bet_type="OVER 220.5",
            edge_pct=5.2,
            confidence=0.72,
            stake=250.0,
        )
        mock_send_raw.assert_awaited_once()
        args, kwargs = mock_send_raw.call_args
        assert "LAL @ BOS" in kwargs["subject"]
        assert kwargs["html"] is True

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_daily_summary(self, mock_send_raw, notifier):
        """Daily summary should show W/L and P&L."""
        await notifier.send_daily_summary(
            total_bets=10,
            wins=6,
            losses=4,
            profit=3.2,
            roi=8.5,
            best_bet="LAL -4.5",
        )
        mock_send_raw.assert_awaited_once()
        args, kwargs = mock_send_raw.call_args
        assert "6W" in kwargs["subject"]
        assert kwargs["html"] is True

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_health_report(self, mock_send_raw, notifier):
        """Health report should show status."""
        await notifier.send_health_report(
            status="healthy",
            models_active=4,
            leagues_tracked=6,
            errors_last_hour=0,
        )
        mock_send_raw.assert_awaited_once()
        args, kwargs = mock_send_raw.call_args
        assert "healthy" in kwargs["subject"].lower()

    @pytest.mark.asyncio
    @patch.object(EmailNotifier, "_send_raw", new_callable=AsyncMock)
    async def test_send_line_movement(self, mock_send_raw, notifier):
        """Line movement should show old and new lines."""
        await notifier.send_line_movement(
            matchup="LAL @ BOS",
            market="Spread",
            old_line=-3.5,
            new_line=-4.5,
            direction="up",
        )
        mock_send_raw.assert_awaited_once()
        args, kwargs = mock_send_raw.call_args
        assert "-3.5" in kwargs["subject"]


class TestEmailNotifierConnection:
    """Tests for test_connection()."""

    @pytest.mark.asyncio
    async def test_unconfigured(self, unconfigured_notifier):
        """test_connection should fail when not configured."""
        result = await unconfigured_notifier.test_connection()
        assert result is False

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_success(self, mock_smtp, notifier):
        """test_connection should return True on success."""
        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance
        result = await notifier.test_connection()
        assert result is True
        mock_instance.login.assert_called_once()

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_failure(self, mock_smtp, notifier):
        """test_connection should return False on failure."""
        from smtplib import SMTPAuthenticationError

        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance
        mock_instance.login.side_effect = SMTPAuthenticationError(
            535, b"bad"
        )
        result = await notifier.test_connection()
        assert result is False
