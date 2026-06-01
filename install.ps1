<#
.SYNOPSIS
    Bootstrap installer for the Betting Intelligence System on Windows.

.DESCRIPTION
    Creates a Python virtual environment, installs all dependencies,
    copies the .env.example to .env, and initializes the database.

    Run this from the project root directory:
        powershell -ExecutionPolicy Bypass -File install.ps1

.NOTES
    Requires:  Python 3.10+, PowerShell 5.1+
    Author:    Betting Intelligence Team
#>

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Betting Intelligence — Windows Installer"

# ── Helper Functions ──────────────────────────────────────────────────────
function Write-Step {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host "`n→ $Message" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor "Green"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor "Yellow"
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor "Red"
}

# ── Check Prerequisites ───────────────────────────────────────────────────
Write-Step "Checking prerequisites..."

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   Found: $pythonVersion"
} catch {
    Write-Error "Python is not installed or not on PATH."
    Write-Host "   Download Python 3.10+ from https://www.python.org/downloads/"
    Write-Host "   Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

# Check Python version (must be 3.10+)
$versionMatch = [regex]::Match($pythonVersion, '(\d+)\.(\d+)')
if ($versionMatch.Success) {
    $major = [int]$versionMatch.Groups[1].Value
    $minor = [int]$versionMatch.Groups[2].Value
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Error "Python 3.10+ is required. Found: $major.$minor"
        exit 1
    }
}

# Check pip
try {
    $pipVersion = python -m pip --version 2>&1
    Write-Host "   Found: pip"
} catch {
    Write-Error "pip is not available."
    exit 1
}

# Check if running from project root
$projectFiles = @("pyproject.toml", "src", "install.ps1")
$missingFiles = $projectFiles | Where-Object { -not (Test-Path $_) }
if ($missingFiles.Count -gt 0) {
    Write-Error "Not in the project root directory."
    Write-Host "   Missing files: $($missingFiles -join ', ')"
    Write-Host "   Make sure you're in the betting-intelligence directory."
    exit 1
}

# ── Virtual Environment ───────────────────────────────────────────────────
Write-Step "Creating virtual environment..."
$venvPath = ".venv"

if (Test-Path $venvPath) {
    Write-Host "   Virtual environment already exists at .venv"
} else {
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    Write-Success "Virtual environment created at .venv"
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Success "Virtual environment activated"
} else {
    Write-Error "Could not find activation script: $activateScript"
    exit 1
}

# ── Upgrade pip ───────────────────────────────────────────────────────────
Write-Step "Upgrading pip..."
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip upgrade failed (non-fatal)"
}

# ── Install Dependencies ──────────────────────────────────────────────────
Write-Step "Installing package with dependencies..."
Write-Host "   This may take a few minutes..."

# First try the editable install (preferred)
python -m pip install -e ".[dev,dashboard]"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Editable install failed, trying requirements.txt..."

    if (-not (Test-Path "requirements.txt")) {
        Write-Error "requirements.txt not found."
        exit 1
    }

    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency installation failed."
        Write-Host "   Check the error messages above."
        Write-Host "   You may need to install the Build Tools for Visual Studio:"
        Write-Host "   https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        exit 1
    }
}

Write-Success "Dependencies installed"

# ── Environment File ──────────────────────────────────────────────────────
Write-Step "Setting up environment file..."

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Success "Created .env from .env.example"
        Write-Host ""
        Write-Host "   ⚠  IMPORTANT: Edit .env with your API keys before running the system!"
        Write-Host "       Open .env in a text editor and fill in the required values:"
        Write-Host "       - ODDS_API_KEY    (get one at https://the-odds-api.com/)"
        Write-Host "       - DATABASE_URL    (SQLite works out of the box)"
        Write-Host "       - API_KEY         (change from the default)"
        Write-Host ""
    } else {
        Write-Warning ".env.example not found, skipping .env creation"
    }
} else {
    Write-Host "   .env already exists (keeping existing configuration)"
}

# ── Database Initialization ───────────────────────────────────────────────
Write-Step "Initializing database..."

$initResult = python scripts/init_db.py 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Success "Database initialized"
} else {
    Write-Warning "Database initialization had issues:"
    Write-Host "   $initResult"
    Write-Host "   You can fix and re-run later:  python scripts/init_db.py"
}

# ── Summary ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor "Green"
Write-Host "║           Betting Intelligence — Installation Complete       ║" -ForegroundColor "Green"
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor "Green"
Write-Host ""

Write-Host "   📁 Virtual env:  .venv" -ForegroundColor "White"
Write-Host "   🔧 Config:       .env (edit with your API keys)" -ForegroundColor "White"
Write-Host "   🗄️  Database:    SQLite at ./data/betting_intel.db" -ForegroundColor "White"
Write-Host ""

Write-Host "   Next steps:" -ForegroundColor "Cyan"
Write-Host "   1. Edit .env with your API keys" -ForegroundColor "Gray"
Write-Host "   2. Activate the environment:  .\.venv\Scripts\Activate.ps1" -ForegroundColor "Gray"
Write-Host "   3. Run the pipeline:          betting-intel pipeline run" -ForegroundColor "Gray"
Write-Host "   4. Start the API:             betting-intel api start" -ForegroundColor "Gray"
Write-Host "   5. Launch dashboard:          betting-intel dashboard" -ForegroundColor "Gray"
Write-Host ""

# Keep the activation
Write-Host "   💡 The virtual environment is already activated for this shell." -ForegroundColor "Yellow"
