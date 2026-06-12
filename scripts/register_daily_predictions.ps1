<#
.SYNOPSIS
    Registers the Daily Betting Predictions task in Windows Task Scheduler.

.DESCRIPTION
    Creates a scheduled task that runs tools/generate_recommendations.py
    every morning at 9:00 AM. Uses the system Python installation.

    Run once to set up automation:
        powershell -ExecutionPolicy Bypass -File scripts\register_daily_predictions.ps1

    To remove later:
        Unregister-ScheduledTask -TaskName "BettingIntelDailyPredictions" -Confirm:$false
#>

$TaskName = "BettingIntelDailyPredictions"
$ProjectPath = [IO.Path]::GetDirectoryName($PSScriptRoot)  # Goes up from scripts\ to project root
$BatchScript = [IO.Path]::Combine($ProjectPath, "scripts", "run_predictions_daily.bat")

Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  BETTING INTELLIGENCE — DAILY PREDICTIONS" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

# ── Validate ───────────────────────────────────────────────────────────
$Errors = @()
if (-not (Test-Path $ProjectPath))        { $Errors += "Project directory not found: $ProjectPath" }
if (-not (Test-Path $BatchScript))        { $Errors += "Batch script not found: $BatchScript" }

if ($Errors.Count -gt 0) {
    Write-Host "  [FAIL] VALIDATION FAILED:" -ForegroundColor Red
    foreach ($err in $Errors) {
        Write-Host "       - $err" -ForegroundColor Red
    }
    Write-Host "`n  Fix the issues above and re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "  [OK] All paths validated" -ForegroundColor Green
Write-Host "       Project: $ProjectPath"
Write-Host "       Script:  $BatchScript"
Write-Host ""

# ── Check for existing task ────────────────────────────────────────────
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "  [WARN] Task '$TaskName' already exists. It will be overwritten." -ForegroundColor Yellow
}

# ── Register the task ──────────────────────────────────────────────────
try {
    $Action = New-ScheduledTaskAction `
        -Execute $BatchScript

    $Trigger = New-ScheduledTaskTrigger -Daily -At 09:00AM

    # Try SYSTEM mode first (requires admin)
    try {
        $Principal = New-ScheduledTaskPrincipal `
            -UserId "SYSTEM" `
            -LogonType ServiceAccount `
            -RunLevel Highest

        $Settings = New-ScheduledTaskSettingsSet `
            -StartWhenAvailable `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1)

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Principal $Principal `
            -Settings $Settings `
            -Description "Runs betting intelligence predictions daily at 9 AM using the trained ML model and live ESPN odds" `
            -Force

        Write-Host "  [OK] Task registered (SYSTEM mode)!" -ForegroundColor Green
        Write-Host "       Runs as SYSTEM - no login required."
    }
    catch {
        Write-Host "  [INFO] SYSTEM mode failed (need admin rights). Trying interactive mode..." -ForegroundColor Yellow
        Write-Host "  [INFO] Fallback: task will only run when you're logged in." -ForegroundColor Yellow

        $Principal = New-ScheduledTaskPrincipal `
            -UserId "$env:USERNAME" `
            -LogonType Interactive

        $Settings = New-ScheduledTaskSettingsSet `
            -StartWhenAvailable `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1)

        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Principal $Principal `
            -Settings $Settings `
            -Description "Runs betting intelligence predictions daily at 9 AM using the trained ML model and live ESPN odds" `
            -Force

        Write-Host "  [OK] Task registered (interactive mode)!" -ForegroundColor Green
    }

    # ── Verify ─────────────────────────────────────────────────────────
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($Task) {
        Write-Host ""
        Write-Host "  ── Task Details ──" -ForegroundColor Cyan
        Write-Host "  Name:      $TaskName"
        Write-Host "  Schedule:  Daily at 9:00 AM"
        Write-Host "  Script:    $BatchScript"
        Write-Host "  State:     $($Task.State)"

        if ($Task.NextRunTime -and $Task.NextRunTime -ne [datetime]::MaxValue) {
            Write-Host "  Next run:  $($Task.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"
        } else {
            Write-Host "  Next run:  (will trigger at next 9:00 AM)"
        }

        Write-Host ""
        Write-Host "  To run immediately:" -ForegroundColor Yellow
        Write-Host "    Start-ScheduledTask -TaskName `"$TaskName`"" -ForegroundColor White
        Write-Host ""
        Write-Host "  To remove later:" -ForegroundColor Yellow
        Write-Host "    Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false" -ForegroundColor White
    }

} catch {
    Write-Host "  [FAIL] Failed to register task: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  [OK] AUTOMATION READY" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Task:      $TaskName"
Write-Host "  Schedule:  Daily at 9:00 AM"
Write-Host "  What runs: tools/generate_recommendations.py"
Write-Host "  Logs:      logs/predictions_YYYYMMDD.log"
Write-Host ""
