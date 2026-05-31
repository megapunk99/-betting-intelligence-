@echo off
REM =============================================================================
REM  Daily Prediction Pipeline -- Windows Task Scheduler Wrapper
REM  Runs every day at 6:00 AM to execute:
REM    1. predict_tomorrow.py --scheduled (live odds, no tuning, auto-save)
REM    2. JSON summary printed to stdout for scheduler/alert parsing
REM  Logs are written to betting-intelligence/logs/pipeline_*.log
REM =============================================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PYTHON=C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe"
set "SCRIPT=%PROJECT_DIR%predict_tomorrow.py"
set "LOGS_DIR=%PROJECT_DIR%logs"

REM Ensure logs directory exists
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

REM Generate timestamp via wmic (Windows Management Instrumentation)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set "DT=%%I"
if defined DT (
    set "TIMESTAMP=!DT:~0,4!-!DT:~4,2!-!DT:~6,2!_!DT:~8,2!!DT:~10,2!!DT:~12,2!"
) else (
    REM Fallback if wmic not available
    set "TIMESTAMP=%DATE:/=-%_%TIME::=%"
    set "TIMESTAMP=!TIMESTAMP: =0!"
)

set "LOG_FILE=%LOGS_DIR%\pipeline_%TIMESTAMP%.log"

echo [%DATE% %TIME%] Starting daily prediction pipeline... > "%LOG_FILE%"
echo  Project: %PROJECT_DIR% >> "%LOG_FILE%"
echo  Python:  %PYTHON% >> "%LOG_FILE%"
echo  Script:  %SCRIPT% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

cd /d "%PROJECT_DIR%"

"%PYTHON%" -B "%SCRIPT%" --scheduled >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo. >> "%LOG_FILE%"
echo [%DATE% %TIME%] Exit code: %EXIT_CODE% >> "%LOG_FILE%"

REM Send email notification (reads the log file we just wrote)
"%PYTHON%" -B "%PROJECT_DIR%tools\notify_pipeline.py" "%LOG_FILE%" %EXIT_CODE% >> "%LOG_FILE%" 2>&1

exit /b %EXIT_CODE%
