@echo off
REM =============================================================================
REM  Register NBA betting scheduled tasks
REM
REM  Tasks registered:
REM    1. NBA Daily Data Update      — runs at 3:00 AM  (data refresh)
REM    2. NBA Daily Pipeline         — runs at 6:00 AM  (prediction pipeline)
REM
REM  Usage:
REM    scripts\install_task.bat              -- try SYSTEM (needs admin)
REM    scripts\install_task.bat --interactive -- run only when logged on
REM =============================================================================

setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0..\"
set "_FLAG=%~1"

REM =============================================================================
REM  Task 1: NBA Daily Data Update
REM =============================================================================

set "BATCH_SCRIPT=%~dp0run_daily_update.bat"
set "TASK1_NAME=NBA Daily Data Update"
set "TASK1_TIME=03:00"

echo ============================================================
echo Installing Task 1: %TASK1_NAME%
echo   Script: %BATCH_SCRIPT%
echo   Time:   %TASK1_TIME%
echo ============================================================
echo.

call :install_task "%TASK1_NAME%" "%BATCH_SCRIPT%" "%TASK1_TIME%"
if %ERRORLEVEL% neq 0 (
    echo [!] Failed to install Task 1. Aborting.
    exit /b 1
)

echo.
echo.

REM =============================================================================
REM  Task 2: NBA Daily Prediction Pipeline
REM =============================================================================

set "PIPELINE_SCRIPT=%ROOT_DIR%run_pipeline_daily.bat"
set "TASK2_NAME=NBA Daily Prediction Pipeline"
set "TASK2_TIME=06:00"

echo ============================================================
echo Installing Task 2: %TASK2_NAME%
echo   Script: %PIPELINE_SCRIPT%
echo   Time:   %TASK2_TIME%
echo ============================================================
echo.

call :install_task "%TASK2_NAME%" "%PIPELINE_SCRIPT%" "%TASK2_TIME%"
if %ERRORLEVEL% neq 0 (
    echo [!] Failed to install Task 2.
    exit /b 1
)

echo.
echo ============================================================
echo  BOTH tasks installed successfully!
echo ============================================================
echo.
echo Task summary:
echo   %TASK1_NAME%  at %TASK1_TIME%  -- data/feature refresh
echo   %TASK2_NAME%  at %TASK2_TIME%  -- prediction pipeline
echo.
echo Verify with:
echo   schtasks /query /tn "NBA Daily Data Update" /v /fo LIST ^| findstr /i "TaskName Next Run Status"
echo   schtasks /query /tn "NBA Daily Prediction Pipeline" /v /fo LIST ^| findstr /i "TaskName Next Run Status"
echo.
echo Run immediately (PowerShell):
echo   Start-ScheduledTask -TaskName "NBA Daily Data Update"
echo   Start-ScheduledTask -TaskName "NBA Daily Prediction Pipeline"
echo.

exit /b 0


REM =============================================================================
REM  Subroutine: install_task
REM  Parameters:
REM    %1 = Task name
REM    %2 = Batch script path
REM    %3 = Start time (HH:MM)
REM =============================================================================
:install_task
setlocal enabledelayedexpansion
set "_TASK_NAME=%~1"
set "_SCRIPT=%~2"
set "_STIME=%~3"

if /i "%_FLAG%"=="--interactive" goto :it_interactive

REM Try SYSTEM mode first
schtasks /create ^
    /tn "%_TASK_NAME%" ^
    /tr "%_SCRIPT%" ^
    /sc daily ^
    /st "%_STIME%" ^
    /ru SYSTEM ^
    /f

if %ERRORLEVEL% equ 0 (
    echo [OK] "%_TASK_NAME%" registered (SYSTEM mode).
    endlocal & exit /b 0
)

echo.
echo [!] SYSTEM mode failed - Access Denied for "%_TASK_NAME%".
echo.
echo Options:
echo   1. Run as Administrator: Right-click Command Prompt ^> "Run as Admin", then run:
echo        scripts\install_task.bat --admin
echo.
echo   2. Interactive mode (no admin needed, runs only when logged in):
echo        scripts\install_task.bat --interactive
echo.
endlocal & exit /b 1

:it_interactive
schtasks /create ^
    /tn "%_TASK_NAME%" ^
    /tr "%_SCRIPT%" ^
    /sc daily ^
    /st "%_STIME%" ^
    /ru "%USERNAME%" ^
    /it ^
    /f

if %ERRORLEVEL% equ 0 (
    echo [OK] "%_TASK_NAME%" registered (interactive mode).
    endlocal & exit /b 0
)

echo [!] Failed to register "%_TASK_NAME%" in interactive mode.
endlocal & exit /b 1
