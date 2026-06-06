@echo off
REM Stripe Webhook Forwarding — Local Dev
REM Usage: start_stripe_webhook.bat [port]
REM
REM This starts the Stripe CLI in listen mode, forwarding webhook events
REM to your local Exact Bets dev server.
REM
REM First time? Run: stripe login
REM Then run this batch file.
REM Press Ctrl+C to stop forwarding.

set PORT=%~1
if "%PORT%"=="" set PORT=8000

echo ==========================================
echo   Stripe Webhook Forwarding
echo ==========================================
echo.
echo   Forwarding to: http://localhost:%PORT%/api/stripe/webhook
echo.
echo   First time? Run: stripe login
echo.
echo   Press Ctrl+C to stop.
echo.
echo -----------------------------------------
echo.

stripe listen ^
    --forward-to http://localhost:%PORT%/api/stripe/webhook ^
    --events checkout.session.completed,customer.subscription.updated,customer.subscription.deleted,invoice.payment_succeeded,invoice.payment_failed ^
    --log-level info
