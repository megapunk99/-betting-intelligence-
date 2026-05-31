@echo off
REM =============================================================================
REM  NBA Daily Pipeline — Windows Task Scheduler Wrapper
REM  Runs every day at 3:00 AM to execute the full NBA data pipeline:
REM    1. Scrape new completed games from the NBA CDN API
REM    2. Update player stats from boxscore data
REM    3. Data quality report
REM    4. Forward test with real sportsbook odds
REM  Logs are written to betting-intelligence/logs/daily_*.log
REM =============================================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0.."
set "PYTHON=C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe"
set "SCRIPT=%PROJECT_DIR%\tools\daily_run.py"
set "LOGS_DIR=%PROJECT_DIR%\logs"

REM Ensure logs directory exists
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

REM Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set "DT=%%I"
if defined DT (
    set "TIMESTAMP=!DT:~0,4!-!DT:~4,2!-!DT:~6,2!_!DT:~8,2!!DT:~10,2!!DT:~12,2!"
) else (
    REM Fallback if wmic not available
    set "TIMESTAMP=%DATE:/=-%_%TIME::=%
"
    set "TIMESTAMP=!TIMESTAMP: =0!"
)

set "LOG_FILE=%LOGS_DIR%\scheduler_%TIMESTAMP%.log"

echo [%DATE% %TIME%] Starting NBA daily pipeline... > "%LOG_FILE%"
echo  Project: %PROJECT_DIR% >> "%LOG_FILE%"
echo  Python:  %PYTHON% >> "%LOG_FILE%"
echo  Script:  %SCRIPT% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

cd /d "%PROJECT_DIR%"

"%PYTHON%" -B "%SCRIPT%" >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo. >> "%LOG_FILE%"
echo [%DATE% %TIME%] Exit code: %EXIT_CODE% >> "%LOG_FILE%"

exit /b %EXIT_CODE%
