<#
.SYNOPSIS
    Registers the Betting Intelligence Daily & Weekly Retrain in Windows Task Scheduler.

.DESCRIPTION
    Creates two scheduled tasks:
      1. "BettingIntelligenceDaily" — Runs daily at 8:00 AM (fast retrain)
      2. "BettingIntelligenceWeekly" — Runs Sundays at 9:00 AM (full pipeline)

    Run this script once to set up automation:
        powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1

    To remove tasks later:
        Unregister-ScheduledTask -TaskName "BettingIntelligenceDaily" -Confirm:$false
        Unregister-ScheduledTask -TaskName "BettingIntelligenceWeekly" -Confirm:$false
#>

$DailyTaskName = "BettingIntelligenceDaily"
$WeeklyTaskName = "BettingIntelligenceWeekly"
$ProjectPath = $PSScriptRoot
$DailyScript = [IO.Path]::Combine($ProjectPath, "scripts", "daily_retrain.py")
$WeeklyScript = [IO.Path]::Combine($ProjectPath, "scripts", "weekly_retrain.py")
$PythonExe = [IO.Path]::Combine($ProjectPath, ".venv", "Scripts", "python.exe")

Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  BETTING INTELLIGENCE — SCHEDULER SETUP" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

# ── Validate ───────────────────────────────────────────────────────────
$Errors = @()
if (-not (Test-Path $ProjectPath))     { $Errors += "Project directory not found: $ProjectPath" }
if (-not (Test-Path $DailyScript))     { $Errors += "Daily script not found: $DailyScript" }
if (-not (Test-Path $WeeklyScript))    { $Errors += "Weekly script not found: $WeeklyScript" }
if (-not (Test-Path $PythonExe))       { $Errors += "Python venv not found: $PythonExe" }

if ($Errors.Count -gt 0) {
    Write-Host "  ❌  VALIDATION FAILED:" -ForegroundColor Red
    foreach ($err in $Errors) {
        Write-Host "       - $err" -ForegroundColor Red
    }
    Write-Host "`n  Fix the issues above and re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✅  All paths validated" -ForegroundColor Green

# ── Helper function to register a task ─────────────────────────────────
function Register-RetrainingTask {
    param(
        [string]$TaskName,
        [string]$Description,
        [string]$ScriptPath,
        [string]$TriggerDescription,
        $Trigger
    )

    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Existing) {
        Write-Host "  ⚠  Task '$TaskName' already exists. It will be overwritten." -ForegroundColor Yellow
    }

    try {
        $Action = New-ScheduledTaskAction `
            -Execute "$PythonExe" `
            -Argument "`"$ScriptPath`" --scheduled"

        $Principal = New-ScheduledTaskPrincipal `
            -UserId "SYSTEM" `
            -LogonType ServiceAccount `
            -RunLevel Highest

        $Settings = New-ScheduledTaskSettingsSet `
            -StartWhenAvailable `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 2)

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Principal $Principal `
            -Settings $Settings `
            -Description $Description `
            -Force

        Write-Host "  ✅  '$TaskName' registered!" -ForegroundColor Green

        # Verify
        $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($Task) {
            $NextRun = $Task.NextRunTime
            Write-Host "      Next run: $($NextRun.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
            Write-Host "      $TriggerDescription" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  ❌  Failed to register '$TaskName': $_" -ForegroundColor Red
        throw
    }
}

# ── Register Daily Task (8:00 AM) ────────────────────────────────────
Write-Host "`n  ── Task 1: Daily Fast Retrain ──" -ForegroundColor Cyan
$DailyTrigger = New-ScheduledTaskTrigger -Daily -At 08:00AM
Register-RetrainingTask `
    -TaskName $DailyTaskName `
    -Description "Daily momentum model retrain at 8AM. Refreshes NBA data, retrains the RecommendationEngine momentum model, regenerates caches, and generates the daily betting card." `
    -ScriptPath $DailyScript `
    -TriggerDescription "Schedule: Daily at 8:00 AM" `
    -Trigger $DailyTrigger

# ── Register Weekly Task (Sunday 9:00 AM) ────────────────────────────
Write-Host "`n  ── Task 2: Weekly Full Retrain ──" -ForegroundColor Cyan
$WeeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 09:00AM
Register-RetrainingTask `
    -TaskName $WeeklyTaskName `
    -Description "Weekly full pipeline retrain every Sunday at 9AM. Runs all 7+ models with walk-forward backtesting, updates the model registry, then regenerates the recommendation engine caches." `
    -ScriptPath $WeeklyScript `
    -TriggerDescription "Schedule: Sundays at 9:00 AM" `
    -Trigger $WeeklyTrigger

# ── Summary ───────────────────────────────────────────────────────────
Write-Host "`n══════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  ✅  AUTOMATION READY" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Daily:  $DailyTaskName  — runs daily at 8:00 AM"
Write-Host "  Weekly: $WeeklyTaskName  — runs Sundays at 9:00 AM"
Write-Host ""
Write-Host "  Project: $ProjectPath" -ForegroundColor Gray
Write-Host "  Python:  $PythonExe" -ForegroundColor Gray
Write-Host ""
Write-Host "  To run the daily task manually:" -ForegroundColor Yellow
Write-Host "    $PythonExe $DailyScript --scheduled" -ForegroundColor White
Write-Host ""
Write-Host "  To run the weekly task manually:" -ForegroundColor Yellow
Write-Host "    $PythonExe $WeeklyScript --scheduled" -ForegroundColor White
Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
