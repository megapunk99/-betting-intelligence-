#!/usr/bin/env python3
"""
Pipeline Notification Script

Called by run_pipeline_daily.bat after the pipeline finishes.
Reads the last pipeline log file and sends an email notification
with the pipeline status (success/partial/failure).

Usage:
    python tools/notify_pipeline.py <log_file_path> <exit_code>

Environment variables (from .env):
    ENABLE_EMAIL         — "true" to enable email notifications
    SMTP_USERNAME        — Gmail address
    SMTP_PASSWORD        — Gmail App Password
    SMTP_TO_ADDR         — Recipient email (defaults to SMTP_USERNAME)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ensure src is on the path
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))


def get_last_log_path() -> Path | None:
    """Find the most recent pipeline log file."""
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    if not logs_dir.exists():
        return None

    pipeline_logs = sorted(logs_dir.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pipeline_logs[0] if pipeline_logs else None


def _read_log_lines(log_path: Path) -> list[str]:
    """Read all lines from the log file."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return [l.rstrip("\n\r") for l in f.readlines()]
    except Exception:
        return []


def extract_summary(log_path: Path, max_lines: int = 20) -> str:
    """Extract the last N lines of the log for the email body."""
    lines = _read_log_lines(log_path)
    if not lines:
        return "(could not read log)"
    tail = lines[-max_lines:]
    return "\n".join(tail)


def extract_duration_seconds(log_path: Path) -> float:
    """Parse start and end timestamps from the log to calculate duration.

    The batch file logs:
        [Mon 5/31/2026 10:42:00.00] Starting daily prediction pipeline...
        ...
        [Mon 5/31/2026 10:45:30.00] Exit code: 0

    Note: we track parsed datetimes directly so that non-timestamp bracket
    lines (e.g. "[notify_pipeline] ...") are silently ignored.
    """
    import datetime as dt

    lines = _read_log_lines(log_path)
    if not lines:
        return 0.0

    def _parse(raw: str) -> dt.datetime | None:
        """Parse a bracketed timestamp like '[Mon 5/31/2026 10:42:00.00]'."""
        # Extract content inside brackets
        m = re.match(r"^\[([^\]]+)\]", raw)
        if not m:
            return None
        inner = m.group(1)
        # Remove day-of-week prefix (e.g. "Mon ")
        inner = re.sub(r"^\w+\s+", "", inner)
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return dt.datetime.strptime(inner, fmt)
            except ValueError:
                continue
        return None

    start_dt: dt.datetime | None = None
    end_dt: dt.datetime | None = None

    for line in lines:
        ts = _parse(line)
        if ts is not None:
            if start_dt is None:
                start_dt = ts
            end_dt = ts

    if start_dt and end_dt and start_dt != end_dt:
        return (end_dt - start_dt).total_seconds()
    return 0.0


def send_notification(log_path: Path, exit_code: int) -> None:
    """Read the log and send an email notification."""
    # Determine status from exit code and log content
    summary = extract_summary(log_path)

    duration_seconds = extract_duration_seconds(log_path)

    if exit_code != 0:
        status = "failed"
    else:
        # Check for warnings/errors in the log
        has_errors = bool(re.search(r"error|failed|traceback|exception", summary, re.IGNORECASE))
        has_warnings = bool(re.search(r"warning|401|unauthorized|missing|skipping", summary, re.IGNORECASE))
        if has_errors:
            status = "partial"
        elif has_warnings:
            status = "partial"
        else:
            status = "success"

    # Build the email
    from betting_intel.alerts.email import EmailNotifier

    import asyncio

    notifier = EmailNotifier(
        smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        username=os.getenv("SMTP_USERNAME", ""),
        password=os.getenv("SMTP_PASSWORD", ""),
        from_addr=os.getenv("SMTP_FROM_ADDR", os.getenv("SMTP_USERNAME", "")),
        to_addr=os.getenv("SMTP_TO_ADDR", os.getenv("SMTP_USERNAME", "")),
    )

    result = asyncio.run(
        notifier.send_pipeline_alert(
            status=status,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            summary=summary,
            log_path=str(log_path),
        )
    )

    if result.error:
        print(f"[notify_pipeline] Failed to send email: {result.error}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[notify_pipeline] Email sent: {result.subject}")


def main() -> None:
    log_path: Path | None = None
    exit_code: int = 0

    # Accept log path and exit code as CLI args
    if len(sys.argv) >= 2:
        log_path = Path(sys.argv[1])
        if not log_path.exists():
            print(f"[notify_pipeline] Log not found: {log_path}", file=sys.stderr)
            log_path = None
    if len(sys.argv) >= 3:
        exit_code = int(sys.argv[2])

    # Fallback: find the last log file
    if log_path is None:
        log_path = get_last_log_path()
        if log_path is None:
            print("[notify_pipeline] No pipeline log found", file=sys.stderr)
            sys.exit(1)

    # Check if email is enabled
    if os.getenv("ENABLE_EMAIL", "").lower() not in ("true", "1", "yes"):
        print("[notify_pipeline] Email notifications disabled (set ENABLE_EMAIL=true in .env)")
        return

    if not os.getenv("SMTP_USERNAME") or not os.getenv("SMTP_PASSWORD"):
        print("[notify_pipeline] SMTP credentials not configured (set SMTP_USERNAME and SMTP_PASSWORD in .env)")
        return

    send_notification(log_path, exit_code)


if __name__ == "__main__":
    main()
