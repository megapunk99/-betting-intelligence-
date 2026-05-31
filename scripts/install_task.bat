@echo off
REM Register the NBA Daily Data Update scheduled task
REM Runs daily at 3:00 AM
REM
REM Usage:
REM   scripts\install_task.bat              -- try SYSTEM (needs admin)
REM   scripts\install_task.bat --interactive -- run only when logged on (no admin needed)

setlocal enabledelayedexpansion

set "BATCH_SCRIPT=%~dp0run_daily_update.bat"
set "TASK_NAME=NBA Daily Data Update"

echo Installing scheduled task: %TASK_NAME%
echo.

if /i "%1"=="--interactive" goto interactive
if /i "%1"=="--admin" goto admin

REM Try SYSTEM mode first (needs admin)
:admin
echo Attempting SYSTEM mode (requires Administrator)...
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "%BATCH_SCRIPT%" ^
    /sc daily ^
    /st 03:00 ^
    /ru SYSTEM ^
    /f

if %ERRORLEVEL% equ 0 goto success

echo.
echo [!] SYSTEM mode failed - Access Denied.
echo.
echo You have two options:
echo.
echo   Option 1: Run as Administrator (recommended)
echo     Right-click Command Prompt ^> "Run as Administrator", then run:
echo       scripts\install_task.bat --admin
echo.
echo   Option 2: Interactive mode (runs only when you're logged in)
echo     No admin needed. Run:
echo       scripts\install_task.bat --interactive
echo.
exit /b 1

:interactive
echo Installing in interactive mode (runs when logged on)...
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "%BATCH_SCRIPT%" ^
    /sc daily ^
    /st 03:00 ^
    /ru "%USERNAME%" ^
    /it ^
    /f

if %ERRORLEVEL% equ 0 goto success

echo [!] Interactive mode also failed.
echo Try running this script as Administrator.
exit /b 1

:success
echo.
echo [OK] Task registered successfully!
echo.
schtasks /query /tn "%TASK_NAME%" /v /fo LIST 2>nul | findstr /i "TaskName: Next Run: Status:"
echo.
echo You can run it immediately with:
echo   Start-ScheduledTask -TaskName "%TASK_NAME%"
echo.
echo Logs will appear in: logs\scheduler_*.log
