#!/usr/bin/env python3
"""
Schedule Daily Refresh as a Windows Scheduled Task.

Usage:
    python tools/schedule_daily_windows.py create        # Create the scheduled task
    python tools/schedule_daily_windows.py create --time 09:00  # Custom time
    python tools/schedule_daily_windows.py create --time "06:00" --days-back 3
    python tools/schedule_daily_windows.py delete        # Remove the scheduled task
    python tools/schedule_daily_windows.py status        # Check if task exists
    python tools/schedule_daily_windows.py run-now       # Run the task immediately

Requirements:
    - Windows with schtasks.exe available
    - Python must be on PATH
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

TASK_NAME = "BettingIntelDailyRefresh"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = PROJECT_ROOT / "tools" / "daily_refresh.py"


def find_python() -> str:
    """Find the Python executable path."""
    # First try the current interpreter
    if os.path.exists(sys.executable):
        return sys.executable

    # Try common Windows Python paths
    candidates = [
        r"C:\Python312\python.exe",
        r"C:\Python311\python.exe",
        r"C:\Python310\python.exe",
        r"C:\Program Files\Python312\python.exe",
        r"C:\Program Files\Python311\python.exe",
        r"C:\Program Files\Python310\python.exe",
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    # Last resort: try `where python`
    try:
        result = subprocess.run(
            ["where", "python"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return sys.executable


def create_task(time_str: str = "08:00", days_back: int = 3) -> bool:
    """Create a Windows scheduled task for the daily refresh."""
    python_path = find_python()
    script_path = DEFAULT_SCRIPT

    if not script_path.exists():
        print("FAIL: Script not found at", script_path)
        return False

    print("Python:", python_path)
    print("Script:", script_path)
    print("Time:  ", time_str)
    print("Task:  ", TASK_NAME)

    # Build the command that schtasks will run
    cmd = f'"{python_path}" "{script_path}" --days-back {days_back}'

    # SECURITY NOTE: This task runs with your user privileges.
    # The /IT flag means it only runs when you're logged in.
    schtasks_cmd = [
        "schtasks.exe",
        "/CREATE",
        "/SC", "DAILY",
        "/TN", TASK_NAME,
        "/TR", cmd,
        "/ST", time_str,
        "/F",     # Force overwrite if exists
        "/IT",    # Interactive - run only when user is logged on
    ]

    print()
    print("Creating scheduled task...")
    print("schtasks /CREATE /SC DAILY /TN", TASK_NAME, "/ST", time_str, "/F /IT /RL HIGHEST")
    print()

    try:
        result = subprocess.run(schtasks_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print("SUCCESS: Task created!")
            print("  Name:    ", TASK_NAME)
            print("  Schedule: Daily at", time_str)
            print("  Runs:    ", cmd)
            return True
        else:
            print("FAILED:", result.stderr.strip())
            print()
            print("You may need to run as Administrator.")
            print("Try: Right-click Command Prompt -> Run as Administrator")
            print("Then run: python", __file__, "create --time", time_str)
            return False
    except FileNotFoundError:
        print("FAIL: schtasks.exe not found. This script requires Windows.")
        return False
    except subprocess.TimeoutExpired:
        print("FAIL: Timed out waiting for schtasks.exe")
        return False


def delete_task() -> bool:
    """Delete the scheduled task."""
    print("Deleting task", TASK_NAME, "...")

    cmd = ["schtasks.exe", "/DELETE", "/TN", TASK_NAME, "/F"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("SUCCESS: Task deleted.")
            return True
        else:
            print("FAILED:", result.stderr.strip())
            return False
    except FileNotFoundError:
        print("FAIL: schtasks.exe not found.")
        return False


def check_status() -> bool:
    """Check if the scheduled task exists and show its status."""
    # Use /FO CSV which is more reliable on Windows
    cmd = ["schtasks.exe", "/QUERY", "/TN", TASK_NAME, "/FO", "CSV", "/V"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("TASK EXISTS:")
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                headers = lines[0].strip('"').split('","')
                values = lines[1].strip('"').split('","')
                for h, v in zip(headers, values):
                    if h.strip() in ("TaskName", "Status", "Schedule", "Start Time",
                                      "Task To Run", "Next Run Time", "Last Run Time",
                                      "Last Result"):
                        print("  " + h.strip() + ": " + v.strip())
            else:
                print("  (raw output available)")
            return True
        else:
            print("TASK DOES NOT EXIST.")
            print("Create it with: python", __file__, "create")
            return False
    except FileNotFoundError:
        print("FAIL: schtasks.exe not found.")
        return False


def run_now() -> bool:
    """Run the daily refresh script immediately by invoking schtasks."""
    print("Triggering immediate run via schtasks...")

    cmd = ["schtasks.exe", "/RUN", "/TN", TASK_NAME]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("SUCCESS: Task triggered. It will run in the background.")
            return True
        else:
            print("FAILED:", result.stderr.strip())
            return False
    except FileNotFoundError:
        print("FAIL: schtasks.exe not found.")
        return False
    except subprocess.TimeoutExpired:
        print("FAIL: Timed out.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Schedule daily refresh as a Windows Scheduled Task",
    )
    parser.add_argument(
        "action", type=str, choices=["create", "delete", "status", "run-now"],
        help="Action to perform",
    )
    parser.add_argument(
        "--time", type=str, default="08:00",
        help="Time to run daily (HH:MM, 24-hour format, default: 08:00)",
    )
    parser.add_argument(
        "--days-back", type=int, default=3,
        help="Number of past days to fetch each run (default: 3)",
    )

    args = parser.parse_args()

    if args.action == "create":
        success = create_task(time_str=args.time, days_back=args.days_back)
        sys.exit(0 if success else 1)
    elif args.action == "delete":
        success = delete_task()
        sys.exit(0 if success else 1)
    elif args.action == "status":
        check_status()
        sys.exit(0)
    elif args.action == "run-now":
        success = run_now()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
