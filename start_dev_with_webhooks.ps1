<#
.SYNOPSIS
    Start the Exact Bets dev server with Stripe webhook forwarding.
.DESCRIPTION
    Launches both the web application and Stripe CLI webhook forwarding
    in a single terminal session using PowerShell jobs.
.PARAMETER Port
    Port for the web application (default: 8000).
.PARAMETER Reload
    Enable auto-reload on code changes (default: $true).
.EXAMPLE
    .\start_dev_with_webhooks.ps1
    .\start_dev_with_webhooks.ps1 -Port 8080 -Reload $true
#>

param(
    [int]$Port = 8000,
    [bool]$Reload = $true
)

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Exact Bets — Dev Mode + Stripe Webhooks          ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# ── 1. Check prerequisites ──
Write-Host "[1/3] Checking prerequisites..." -ForegroundColor Cyan

if (-not (Test-Path $VenvPython)) {
    Write-Host "  ✗ Python venv not found at $VenvPython" -ForegroundColor Red
    exit 1
}

$stripeCheck = stripe config --list 2>&1 | Select-String -Pattern "test_mode_key" -SimpleMatch
if (-not $stripeCheck) {
    Write-Host "  ⚠ Stripe not authenticated. Run 'stripe login' first." -ForegroundColor Yellow
    Write-Host "    You can still start the web server — webhooks won't forward." -ForegroundColor Yellow
}
Write-Host "  ✓ Prerequisites checked" -ForegroundColor Green

# ── 2. Start web server in background job ──
Write-Host "[2/3] Starting web server on http://localhost:$Port ..." -ForegroundColor Cyan

$WebJob = Start-Job -Name "ExactBetsWeb" -ScriptBlock {
    param($Port, $Reload, $ProjectRoot)
    Set-Location $ProjectRoot
    $env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot"
    $python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    # Use the betting-intel CLI entry point for reliability
    if ($Reload) {
        & $python -m uvicorn web.app:app --host 0.0.0.0 --port $Port --reload --log-level info 2>&1
    } else {
        & $python -m uvicorn web.app:app --host 0.0.0.0 --port $Port --log-level info 2>&1
    }
} -ArgumentList $Port, $Reload, $ProjectRoot

Start-Sleep -Seconds 3

# Check if web server started
$WebJobState = (Get-Job -Name "ExactBetsWeb").State
if ($WebJobState -eq "Failed") {
    $ErrorMsg = Receive-Job -Name "ExactBetsWeb"
    Write-Host "  ✗ Web server failed to start:" -ForegroundColor Red
    Write-Host $ErrorMsg -ForegroundColor Red
    Remove-Job -Name "ExactBetsWeb" -Force
    exit 1
}
Write-Host "  ✓ Web server is running" -ForegroundColor Green

# ── 3. Start Stripe webhook forwarding (if authenticated) ──
Write-Host "[3/3] Starting Stripe webhook forwarding..." -ForegroundColor Cyan

$ForwardUrl = "http://localhost:$Port/api/stripe/webhook"
$StripeJob = $null

if ($stripeCheck) {
    $StripeJob = Start-Job -Name "StripeWebhook" -ScriptBlock {
        param($ForwardUrl, $ProjectRoot)
        Set-Location $ProjectRoot
        $LogDir = Join-Path $ProjectRoot "logs"
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        $LogFile = Join-Path $LogDir "stripe_webhook.log"

        stripe listen `
            --forward-to $ForwardUrl `
            --events checkout.session.completed,customer.subscription.updated,customer.subscription.deleted,invoice.payment_succeeded,invoice.payment_failed `
            --log-level info 2>&1 | ForEach-Object {
                $_ | Out-File -FilePath $LogFile -Append -Encoding utf8
            }
    } -ArgumentList $ForwardUrl, $ProjectRoot

    Start-Sleep -Seconds 2
    Write-Host "  ✓ Stripe webhook forwarding started" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Skipped — run 'stripe login' to enable webhook forwarding" -ForegroundColor Yellow
}

# ── 3. Show status ──
Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  All systems running!" -ForegroundColor Green
Write-Host ""
Write-Host "  Web App:        http://localhost:$Port" -ForegroundColor Cyan
Write-Host "  Webhook URL:    $ForwardUrl" -ForegroundColor Cyan
Write-Host "  Dashboard:      http://localhost:$Port/dashboard" -ForegroundColor Cyan
Write-Host "  Checkout page:  http://localhost:$Port/#pricing" -ForegroundColor Cyan
Write-Host "  Manage sub:     http://localhost:$Port/subscribe/manage" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To test webhooks, run in another terminal:" -ForegroundColor Yellow
Write-Host "    stripe trigger checkout.session.completed" -ForegroundColor White
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor Gray
Write-Host ""

# ── 4. Wait for Ctrl+C, then clean up ──
try {
    while ($true) {
        Start-Sleep -Seconds 3

        # Check jobs are still running
        $webState = (Get-Job -Name "ExactBetsWeb" -ErrorAction SilentlyContinue).State
        $stripeState = if ($StripeJob) { (Get-Job -Name "StripeWebhook" -ErrorAction SilentlyContinue).State } else { "Skipped" }

        if ($webState -eq "Failed") {
            Write-Host "`n  ✗ Web server stopped unexpectedly!" -ForegroundColor Red
            Receive-Job -Name "ExactBetsWeb"
            break
        }
        if ($stripeState -eq "Failed") {
            Write-Host "`n  ✗ Stripe webhook stopped unexpectedly!" -ForegroundColor Red
            break
        }
    }
}
finally {
    Write-Host "`nShutting down..." -ForegroundColor Yellow
    $jobs = @()
    $webJob = Get-Job -Name "ExactBetsWeb" -ErrorAction SilentlyContinue
    $stripeJob = Get-Job -Name "StripeWebhook" -ErrorAction SilentlyContinue
    if ($webJob) { $jobs += $webJob }
    if ($stripeJob) { $jobs += $stripeJob }
    $jobs | Stop-Job -ErrorAction SilentlyContinue
    $jobs | Remove-Job -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
}
