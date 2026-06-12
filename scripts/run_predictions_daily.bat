@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  Betting Intelligence — Daily Prediction Runner
REM  Called by Windows Task Scheduler every morning at 9 AM.
REM  Runs tools/generate_recommendations.py to generate bet picks.
REM ═══════════════════════════════════════════════════════════════════════

REM Get the directory where this batch file is located (project root)
SET PROJECT_DIR=%~dp0..\

REM Path to Python (use system Python, not venv)
SET PYTHON_EXE=python

REM Log file with date
SET LOG_DIR=%PROJECT_DIR%logs
IF NOT EXIST "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Get date in YYYY-MM-DD format (works on all Windows locales)
FOR /F "tokens=2 delims==." %%I IN ('wmic os get localtime /value 2^>NUL') DO SET LOG_DATE=%%I
IF NOT DEFINED LOG_DATE SET LOG_DATE=%DATE:/=-%
SET LOG_FILE=%LOG_DIR%predictions_%LOG_DATE:~0,8%.log

REM Run the prediction generator
cd /d "%PROJECT_DIR%"
echo [%DATE% %TIME%] Starting prediction run... >> "%LOG_FILE%"
"%PYTHON_EXE%" -X utf8 tools/generate_recommendations.py >> "%LOG_FILE%" 2>&1

REM Exit with the Python script's exit code
EXIT /B %ERRORLEVEL%
