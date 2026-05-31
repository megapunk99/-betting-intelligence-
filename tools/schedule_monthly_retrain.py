#!/usr/bin/env python3
"""
Schedule Monthly Retraining — Setup Windows Task Scheduler or cron.

Generates the appropriate scheduling command for automatic monthly retraining
of all betting models. Supports Windows Task Scheduler and Linux cron.

Usage:
    python tools/schedule_monthly_retrain.py                       # Interactive guide
    python tools/schedule_monthly_retrain.py --generate            # Print commands
    python tools/schedule_monthly_retrain.py --install             # Attempt auto-install

The retraining runs on the 1st of each month at 3:00 AM and:
  1. Loads all available NBA data
  2. Retrains all 4 models (Ridge totals, XGBoost totals, Momentum, Momentum calibrated)
  3. Runs walk-forward backtest to measure performance
  4. Saves models with versioned metadata to the registry
  5. Compares vs previous month and detects feature drift
  6. Saves a JSON report to output/retrain/
"""

import sys
import platform
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RETRAIN_SCRIPT = PROJECT_ROOT / "tools" / "monthly_retrain.py"


def detect_os() -> str:
    """Detect the operating system."""
    system = platform.system().lower()
    if "windows" in system:
        return "windows"
    elif "linux" in system:
        return "linux"
    elif "darwin" in system:
        return "macos"
    return "unknown"


def windows_task_command() -> str:
    """Generate the Windows Task Scheduler command."""
    task_name = "BettingIntelMonthlyRetrain"
    python_path = PYTHON
    script_path = RETRAIN_SCRIPT
    working_dir = PROJECT_ROOT

    # schtasks creates a scheduled task
    cmd = (
        f'schtasks /Create /SC MONTHLY /D 1 /TN "{task_name}" '
        f'/TR "{python_path} {script_path}" '
        f'/ST 03:00 /RL HIGHEST /F'
    )
    return cmd


def linux_cron_command() -> str:
    """Generate the Linux cron job entry."""
    python_path = PYTHON
    script_path = RETRAIN_SCRIPT
    log_path = PROJECT_ROOT / "output" / "retrain" / "cron.log"
    # Runs at 3:00 AM on the 1st of every month
    cron_line = f"0 3 1 * * cd {PROJECT_ROOT} && {python_path} {script_path} >> {log_path} 2>&1"
    return cron_line


def macos_cron_command() -> str:
    """macOS also uses cron (same as Linux)."""
    return linux_cron_command()


def print_schedule_guide():
    """Print a step-by-step guide for each OS."""
    os_name = detect_os()
    print(f"\n{'=' * 60}")
    print(f"  SCHEDULE MONTHLY RETRAINING")
    print(f"  Detected OS: {os_name.upper()}")
    print(f"{'=' * 60}")

    print(f"\n  Retrain script: {RETRAIN_SCRIPT}")
    print(f"  Python:         {PYTHON}")
    print(f"  Project root:   {PROJECT_ROOT}")
    print()

    if os_name == "windows":
        print("  WINDOWS TASK SCHEDULER")
        print("  " + "-" * 40)
        print(f"\n  Option 1 — Run this command as Administrator:")
        print(f"\n    {windows_task_command()}")
        print()
        print(f"  Option 2 — Manual setup:")
        print(f"    1. Open Task Scheduler (Win+R -> taskschd.msc)")
        print(f"    2. Click 'Create Task'")
        print(f"    3. Name: 'BettingIntelMonthlyRetrain'")
        print(f"    4. Triggers: Monthly, Day 1, 3:00 AM")
        print(f"    5. Action: Start a program")
        print(f"       Program: {PYTHON}")
        print(f"       Args: {RETRAIN_SCRIPT}")
        print(f"       Start in: {PROJECT_ROOT}")
        print()

    elif os_name == "linux":
        print("  LINUX CRON")
        print("  " + "-" * 40)
        log_dir = PROJECT_ROOT / "output" / "retrain"
        print(f"\n  Run: crontab -e")
        print(f"\n  Add this line:")
        print(f"\n    {linux_cron_command()}")
        print()
        print(f"  Make sure the log directory exists:")
        print(f"\n    mkdir -p {log_dir}")
        print()

    elif os_name == "macos":
        print("  MACOS CRON (or launchd)")
        print("  " + "-" * 40)
        print(f"\n  Using cron: crontab -e")
        print(f"\n    {macos_cron_command()}")
        print()
        print(f"  Or use launchd for better macOS integration.")
        print()

    else:
        print(f"  Unknown OS: {os_name}")
        print(f"  To schedule manually, run this command monthly:")
        print(f"\n    {PYTHON} {RETRAIN_SCRIPT}")
        print()

    print(f"{'=' * 60}")
    print(f"\n  What the retrain does each month:")
    print(f"  " + "-" * 40)
    print(f"  1. Load latest NBA data from database")
    print(f"  2. Train 4 models: Ridge totals, XGBoost totals,")
    print(f"     Momentum (uncalibrated), Momentum (Platt calibrated)")
    print(f"  3. Run walk-forward backtest for each model")
    print(f"  4. Save models with versioned metadata")
    print(f"  5. Compare vs previous month's performance")
    print(f"  6. Detect feature importance drift")
    print(f"  7. Save JSON report to output/retrain/")
    print(f"\n  Check results anytime:")
    print(f"    python tools/monthly_retrain.py --compare-only")
    print()


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def try_install():
    """Attempt to auto-install the scheduled task."""
    os_name = detect_os()

    if os_name == "windows":
        cmd = windows_task_command()
        print(f"  Installing Windows Task Scheduler...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"  {GREEN}Success!{RESET} Task 'BettingIntelMonthlyRetrain' created.")
                print(f"  Runs on the 1st of each month at 3:00 AM.")
            else:
                print(f"  {RED}Failed:{RESET} {result.stderr.strip()}")
                print(f"  Try running as Administrator, or use --generate to see the command.")
        except subprocess.TimeoutExpired:
            print(f"  {RED}Command timed out.{RESET}")
        except Exception as e:
            print(f"  {RED}Error:{RESET} {e}")

    elif os_name in ("linux", "macos"):
        cron_line = linux_cron_command()
        print(f"  To install, run:")
        print(f"\n    (crontab -l 2>/dev/null; echo \"{cron_line}\") | crontab -")
        print()
        print(f"  This appends the cron job to your existing crontab.")
        print(f"  Run the command above manually to install.")

    else:
        print(f"  Auto-install not supported on {os_name}.")
        print(f"  Use --generate to see manual setup instructions.")


def main():
    parser = argparse.ArgumentParser(
        description="Schedule monthly model retraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generate", action="store_true",
                        help="Print the scheduling command")
    parser.add_argument("--install", action="store_true",
                        help="Attempt auto-install the scheduled task")

    args = parser.parse_args()

    if args.install:
        try_install()
    elif args.generate:
        os_name = detect_os()
        if os_name == "windows":
            print(windows_task_command())
        elif os_name in ("linux", "macos"):
            print(linux_cron_command())
        else:
            print(f"Unsupported OS: {os_name}")
            sys.exit(1)
    else:
        print_schedule_guide()

    return 0


if __name__ == "__main__":
    sys.exit(main())
