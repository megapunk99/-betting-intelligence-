@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  Betting Intelligence — Scheduled Daily Runner (Batch)
REM  Called by Windows Task Scheduler every morning at 8 AM.
REM ═══════════════════════════════════════════════════════════════════════

REM Get the directory where this batch file is located
SET PROJECT_DIR=%~dp0

REM Path to the Python virtual environment
SET PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe

REM Log file with date (locale-independent using wmic)
SET LOG_DIR=%PROJECT_DIR%logs
IF NOT EXIST "%LOG_DIR%" mkdir "%LOG_DIR%"
FOR /F "tokens=2 delims==." %%I IN ('wmic os get localtime /value 2^>NUL') DO SET LOG_DATE=%%I
IF NOT DEFINED LOG_DATE SET LOG_DATE=%DATE:/=-%
SET LOG_FILE=%LOG_DIR%scheduled_%LOG_DATE:~0,8%.log

REM Check if ODDS_API_KEY is set to decide whether to skip pipeline
SET ODDS_KEY=
IF EXIST "%PROJECT_DIR%.env" FOR /F "usebackq tokens=2 delims==" %%K IN (`type "%PROJECT_DIR%.env" ^| findstr /V "^#" ^| findstr "ODDS_API_KEY="`) DO SET ODDS_KEY=%%K
SET PIPELINE_FLAG=--skip-pipeline
IF NOT "%ODDS_KEY%"=="" IF NOT "%ODDS_KEY%"=="REPLACE_ME_WITH_YOUR_ODDS_API_KEY" SET PIPELINE_FLAG=

REM Run the daily card generator
"%PYTHON_EXE%" -X utf8 "%PROJECT_DIR%run_daily.py" --scheduled %PIPELINE_FLAG% >> "%LOG_FILE%" 2>&1

REM Exit with the Python script's exit code
EXIT /B %ERRORLEVEL%
