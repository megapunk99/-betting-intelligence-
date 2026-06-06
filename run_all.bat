@echo off
REM =============================================================================
REM  Betting Intelligence — One-Click Launcher
REM
REM  Does everything:
REM    1. Activates virtual environment (creates it if missing)
REM    2. Installs all dependencies
REM    3. Runs model comparison (LightGBM vs Ridge vs MLP vs Ensemble)
REM    4. Runs the prediction pipeline (historical mode)
REM    5. Starts the web dashboard at http://localhost:8000
REM    6. Opens your browser
REM
REM  Usage:
REM    Double-click run_all.bat
REM    Or from terminal: run_all.bat
REM =============================================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "VENV_DIR=%PROJECT_DIR%.venv"
set "PYTHON_EXE="
set "LOG_DIR=%PROJECT_DIR%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set "DT=%%I"
if defined DT (
    set "TIMESTAMP=!DT:~0,4!-!DT:~4,2!-!DT:~6,2!_!DT:~8,2!!DT:~10,2!!DT:~12,2!"
) else (
    set "TIMESTAMP=%DATE:/=-%_%TIME::=%"
    set "TIMESTAMP=!TIMESTAMP: =0!"
)

echo =============================================================================
echo   BETTING INTELLIGENCE — Full System Launcher
echo   %DATE% %TIME%
echo =============================================================================
echo.

REM ── Step 0: Find Python ────────────────────────────────────────────────
echo [1/6] Locating Python...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   [FAIL] Python not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f''{sys.version_info.major}.{sys.version_info.minor}'')" 2^>nul') do set "PY_VER=%%i"
if defined PY_VER (
    echo   Found Python %PY_VER%
) else (
    echo   [WARN] Could not determine Python version. Continuing...
)

REM ── Step 1: Set up virtual environment ─────────────────────────────────
echo.
echo [2/6] Setting up Python environment...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo   Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo   [FAIL] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo   Virtual environment created.
) else (
    echo   Virtual environment ready.
)

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM ── Step 2: Install dependencies ────────────────────────────────────────
echo.
echo [3/6] Installing dependencies...
echo   This may take a few minutes the first time...
"%PYTHON_EXE%" -m pip install -r "%PROJECT_DIR%requirements.txt" --quiet
if !ERRORLEVEL! neq 0 (
    echo   [WARN] Some packages may have failed. Continuing anyway...
)

REM Check for optional packages
"%PYTHON_EXE%" -c "import requests" >nul 2>&1
if %ERRORLEVEL% neq 0 "%PYTHON_EXE%" -m pip install requests --quiet
"%PYTHON_EXE%" -c "import bs4" >nul 2>&1
if %ERRORLEVEL% neq 0 "%PYTHON_EXE%" -m pip install beautifulsoup4 --quiet

echo   Dependencies installed.

REM ── Step 3: Model Comparison ────────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════════════════════════
echo   [4/6] Running Model Comparison (LightGBM vs Ridge vs MLP vs Ensemble)
echo   This takes ~90 seconds...
echo ════════════════════════════════════════════════════════════════════════
echo.

"%PYTHON_EXE%" tools/compare_models.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo   [WARN] Model comparison had an error. Check output above.
) else (
    echo.
    echo   [OK] Model comparison complete.
)

REM ── Step 4: Run Prediction Pipeline ────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════════════════════════
echo   [5/6] Running Prediction Pipeline (historical mode)
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   Press Ctrl+C to skip... starting in 3 seconds.
ping -n 3 127.0.0.1 >nul

"%PYTHON_EXE%" predict_tomorrow.py --no-tune 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo   [WARN] Pipeline had an error. Check output above.
) else (
    echo.
    echo   [OK] Prediction pipeline complete.
)

REM ── Step 5: Start Web Dashboard ────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════════════════════════
echo   [6/6] Starting Web Dashboard
echo   Opening http://localhost:8000 in your browser...
echo ════════════════════════════════════════════════════════════════════════
echo.
echo   Press Ctrl+C in this window to stop the server.
echo.

REM Wait for server to start, then open browser
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"

REM Start the FastAPI server (blocks until Ctrl+C)
"%PYTHON_EXE%" web/app.py

echo.
echo   Server stopped.
pause
