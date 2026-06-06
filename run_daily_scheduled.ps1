<#
.SYNOPSIS
    Scheduled runner for the Betting Intelligence daily card generator.
    Designed to be called by Windows Task Scheduler every morning at 8 AM.

.DESCRIPTION
    This script:
      1. Changes to the project directory
      2. Activates the Python virtual environment
      3. Runs the daily card generator in scheduled mode
      4. Logs output to a timestamped file
      5. Exits with the correct exit code

    Exit codes:
      0 - Success (picks generated)
      1 - Error (check logs)

.LINK
    https://github.com/megapunk99/betting-intelligence
#>

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# The project root directory (where run_daily.py lives)
# Script is stored inside the betting-intelligence project
$ProjectRoot = $PSScriptRoot

# Path to the Python executable in the virtual environment
$PythonExe = Join-Path $ProjectRoot ".venv" "Scripts" "python.exe"

# Log directory
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Log file with timestamp
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "scheduled_daily_$Timestamp.log"

# The script to run
$ScriptToRun = Join-Path $ProjectRoot "run_daily.py"

# ═══════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════════

# Check that the Python venv exists
if (-not (Test-Path $PythonExe)) {
    $ErrorMsg = "[FATAL] Python virtual environment not found at: $PythonExe"
    Write-Error $ErrorMsg
    Add-Content -Path $LogFile -Value $ErrorMsg
    exit 1
}

# Check that the script exists
if (-not (Test-Path $ScriptToRun)) {
    $ErrorMsg = "[FATAL] Runner script not found at: $ScriptToRun"
    Write-Error $ErrorMsg
    Add-Content -Path $LogFile -Value $ErrorMsg
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════
#  EXECUTION
# ═══════════════════════════════════════════════════════════════════════

$StartTime = Get-Date

# Log header
@"
═══════════════════════════════════════════════════════════════════════
  BETTING INTELLIGENCE — SCHEDULED DAILY RUNNER
  Started: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Project: $ProjectRoot
  Log:     $LogFile
═══════════════════════════════════════════════════════════════════════
"@ | Out-File -FilePath $LogFile -Encoding utf8

# Change to the project directory
Set-Location -Path $ProjectRoot

# Run the daily card generator in scheduled mode
$Arguments = @(
    "-X", "utf8",
    $ScriptToRun,
    "--scheduled",
    "--skip-pipeline"
)

# If ODDS_API_KEY is set, remove --skip-pipeline to run the full ML pipeline
$OddsApiKey = [System.Environment]::GetEnvironmentVariable("ODDS_API_KEY")
if ($OddsApiKey -and $OddsApiKey -ne "REPLACE_ME_WITH_YOUR_ODDS_API_KEY") {
    $Arguments = @(
        "-X", "utf8",
        $ScriptToRun,
        "--scheduled"
    )
}

# Capture output
try {
    $Output = & $PythonExe $Arguments 2>&1
    $ExitCode = $LASTEXITCODE
} catch {
    $Output = "Exception: $_"
    $ExitCode = 1
}

# Log output
$Output | Out-File -FilePath $LogFile -Encoding utf8 -Append

# Log footer
$EndTime = Get-Date
$Duration = ($EndTime - $StartTime).TotalSeconds
@"

═══════════════════════════════════════════════════════════════════════
  COMPLETED
  Finished: $($EndTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Duration: $([math]::Round($Duration, 1))s
  Exit Code: $ExitCode
═══════════════════════════════════════════════════════════════════════
"@ | Out-File -FilePath $LogFile -Encoding utf8 -Append

exit $ExitCode
