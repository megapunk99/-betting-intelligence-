<#
.SYNOPSIS
    Forward Stripe webhook events to the local development server.
.DESCRIPTION
    Starts the Stripe CLI in listen mode, forwarding webhook events
    to the local Exact Bets dev server at localhost:8000/api/stripe/webhook.
.PARAMETER Port
    The local port where the web app is running (default: 8000).
.PARAMETER ApiVersion
    The Stripe API version to use (default: 2025-11-01).
.PARAMETER Events
    Comma-separated list of events to forward (default: all relevant subscription events).
.EXAMPLE
    .\start_stripe_webhook.ps1
    .\start_stripe_webhook.ps1 -Port 8080
#>

param(
    [int]$Port = 8000,
    [string]$ApiVersion = "2025-11-01",
    [string]$Events = "checkout.session.completed,customer.subscription.updated,customer.subscription.deleted,invoice.payment_succeeded,invoice.payment_failed"
)

$ProjectRoot = $PSScriptRoot
$LogFile = Join-Path $ProjectRoot "logs\stripe_webhook.log"

# Ensure logs directory exists
$LogDir = Split-Path $LogFile -Parent
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$ForwardUrl = "http://localhost:$Port/api/stripe/webhook"

Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Stripe Webhook Forwarding" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Forwarding to: $ForwardUrl" -ForegroundColor Cyan
Write-Host "  API Version:   $ApiVersion" -ForegroundColor Cyan
Write-Host "  Events:        $Events" -ForegroundColor Cyan
Write-Host "  Log file:      $LogFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "  First time? Run 'stripe login' to authenticate." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Press Ctrl+C to stop forwarding." -ForegroundColor Gray
Write-Host ""
Write-Host "──────────────────────────────────────────────" -ForegroundColor Gray
Write-Host ""

# Run stripe listen with forward-to and event filtering
# This stays running — press Ctrl+C to stop
# Check if authenticated first
$authCheck = stripe config --list 2>&1 | Select-String -Pattern "test_mode_key" -SimpleMatch
if (-not $authCheck) {
    Write-Host "`n⚠ You need to authenticate with Stripe first!`n" -ForegroundColor Yellow
    Write-Host "Run this command to log in:" -ForegroundColor Yellow
    Write-Host "  stripe login" -ForegroundColor White
    Write-Host "`nThen re-run this script.`n" -ForegroundColor Yellow
    exit 1
}

stripe listen `
    --forward-to $ForwardUrl `
    --api-version $ApiVersion `
    --events $Events `
    --log-level info 2>&1 | ForEach-Object {
        $_ | Out-File -FilePath $LogFile -Append -Encoding utf8
        Write-Host $_
    }
