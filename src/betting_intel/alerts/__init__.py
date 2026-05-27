"""
Alert and notification system for the betting intelligence platform.

Provides:
1. **Telegram bot** — sends bet alerts, movement alerts, and health reports
2. **Discord webhooks** — sends structured alerts to Discord channels
3. **Alert dispatcher** — flexible dispatch rules (by EV, confidence, league)
4. **Scheduled reports** — daily summary, significant movement alerts

Usage:
    from betting_intel.alerts.dispatcher import AlertDispatcher, AlertConfig

    dispatcher = AlertDispatcher()
    dispatcher.add_channel("telegram", TelegramBot(token="..."))
    dispatcher.add_channel("discord", DiscordWebhook(url="..."))

    await dispatcher.dispatch_bet_alert(
        game="LAL @ BOS",
        bet_type="OVER 220.5",
        edge_pct=5.2,
        confidence=0.72,
        stake=250.0,
    )
"""

from betting_intel.alerts.telegram import TelegramBot
from betting_intel.alerts.discord import DiscordWebhook
from betting_intel.alerts.dispatcher import AlertDispatcher, AlertConfig, AlertChannel, BetAlert

__all__ = [
    "TelegramBot",
    "DiscordWebhook",
    "AlertDispatcher",
    "AlertConfig",
    "AlertChannel",
    "BetAlert",
]
