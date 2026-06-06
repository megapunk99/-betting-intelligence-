<#
.SYNOPSIS
    Create Stripe products and prices for Exact Bets subscription tiers.
.DESCRIPTION
    Creates 3 subscription products (Basic $49, Premium $199, Elite $999)
    with monthly recurring prices. Also creates annual prices (10 months
    for price of 12 = 2 months free).

    Authentication (pick one):
      1. Run 'stripe login' to authenticate via browser (one-time)
      2. Set STRIPE_API_KEY env var before running

    Run: .\setup_stripe_products.ps1

    After running, paste the output price IDs into your .env file.
.EXAMPLE
    .\setup_stripe_products.ps1
.EXAMPLE
    $env:STRIPE_API_KEY="sk_test_..."; .\setup_stripe_products.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Green
Write-Host "  Stripe Product & Price Setup" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

# -- Check prerequisites --
if (-not (Get-Command stripe -ErrorAction SilentlyContinue)) {
    Write-Host "Stripe CLI not found in PATH." -ForegroundColor Red
    Write-Host "  Install: npm install -g @stripe/cli" -ForegroundColor Yellow
    exit 1
}

# Check authentication: prefer STRIPE_API_KEY env var, fall back to stripe login
$hasApiKey = $false
if ($env:STRIPE_API_KEY) {
    $hasApiKey = $true
}

$hasLogin = $false
if (-not $hasApiKey) {
    $authCheck = stripe config --list 2>&1 | Select-String -Pattern "test_mode_key" -SimpleMatch
    if ($authCheck) {
        $hasLogin = $true
    }
}

if (-not $hasApiKey -and -not $hasLogin) {
    Write-Host "You need to authenticate with Stripe first!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Option 1: Set your API key as an env var:" -ForegroundColor White
    Write-Host '    $env:STRIPE_API_KEY="sk_test_..."' -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Option 2: Run browser login (one-time):" -ForegroundColor White
    Write-Host "    stripe login" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

if ($hasApiKey) {
    Write-Host "Authenticated via STRIPE_API_KEY env var" -ForegroundColor Green
} else {
    Write-Host "Authenticated via stripe login" -ForegroundColor Green
}
Write-Host ""

# -- Define products --
$products = @(
    @{
        Name          = "Exact Bets - Basic"
        Tier          = "basic"
        AmountMonthly = 4900
        AmountAnnual  = 49000
        Desc          = "Full daily betting card, all NBA games, Telegram + email delivery, Kelly stake sizing."
        Statement     = "EXACT BETS BASIC"
    }
    @{
        Name          = "Exact Bets - Premium"
        Tier          = "premium"
        AmountMonthly = 19900
        AmountAnnual  = 199000
        Desc          = "Live alerts, player props, CLV tracking, Discord access."
        Statement     = "EXACT BETS PREMIUM"
    }
    @{
        Name          = "Exact Bets - Elite"
        Tier          = "elite"
        AmountMonthly = 99900
        AmountAnnual  = 999000
        Desc          = "API access, custom model training, portfolio tracking, priority support."
        Statement     = "EXACT BETS ELITE"
    }
)

$priceIds   = @{}
$productIds = @{}

foreach ($p in $products) {
    Write-Host ("Creating " + $p.Tier + " product...") -ForegroundColor Cyan

    try {
        # Create product
        $prodResult = stripe products create `
            --name $p.Name `
            -d description=$p.Desc `
            -d statement_descriptor=$p.Statement `
            -d metadata[tier]=$p.Tier 2>&1

        $prodJson = $prodResult | ConvertFrom-Json
        $productIds[$p.Tier] = $prodJson.id
        Write-Host ("  Product: " + $prodJson.id) -ForegroundColor Green

        # Create monthly price
        $monthResult = stripe prices create `
            --product $prodJson.id `
            --unit-amount $p.AmountMonthly `
            --currency usd `
            --recurring "interval:month" `
            -d metadata[tier]=$p.Tier 2>&1

        $monthlyJson = $monthResult | ConvertFrom-Json
        $priceIds[$p.Tier] = $monthlyJson.id
        $monthlyDollars = $p.AmountMonthly / 100
        Write-Host ("  Monthly: " + $monthlyJson.id + " ($" + $monthlyDollars + "/mo)") -ForegroundColor Green

        # Create annual price (10 months for price of 12 = 2 months free)
        $annualResult = stripe prices create `
            --product $prodJson.id `
            --unit-amount $p.AmountAnnual `
            --currency usd `
            --recurring "interval:year" `
            -d metadata[tier]=$p.Tier `
            -d metadata[interval]=annual 2>&1

        $annualJson = $annualResult | ConvertFrom-Json
        $priceIds[$p.Tier + "_annual"] = $annualJson.id
        $annualDollars = $p.AmountAnnual / 100
        $monthlyTotal = $p.AmountMonthly * 12
        $savings = ($monthlyTotal - $p.AmountAnnual) / 100
        Write-Host ("  Annual:  " + $annualJson.id + " ($" + $annualDollars + "/yr, save $" + $savings + "/yr)") -ForegroundColor Green

    } catch {
        Write-Host ("  Failed: " + $_.Exception.Message) -ForegroundColor Red
    }
    Write-Host ""
}

# -- Summary Output --
Write-Host "================================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Paste these into your .env file:" -ForegroundColor Yellow
Write-Host ""

if ($priceIds["basic"]) {
    Write-Host ("STRIPE_PRICE_BASIC=" + $priceIds["basic"]) -ForegroundColor White
}
if ($priceIds["premium"]) {
    Write-Host ("STRIPE_PRICE_PREMIUM=" + $priceIds["premium"]) -ForegroundColor White
}
if ($priceIds["elite"]) {
    Write-Host ("STRIPE_PRICE_ELITE=" + $priceIds["elite"]) -ForegroundColor White
}
if ($priceIds["basic_annual"]) {
    Write-Host ("# STRIPE_PRICE_BASIC_ANNUAL=" + $priceIds["basic_annual"] + "  (optional)") -ForegroundColor Gray
}
if ($priceIds["premium_annual"]) {
    Write-Host ("# STRIPE_PRICE_PREMIUM_ANNUAL=" + $priceIds["premium_annual"] + "  (optional)") -ForegroundColor Gray
}
if ($priceIds["elite_annual"]) {
    Write-Host ("# STRIPE_PRICE_ELITE_ANNUAL=" + $priceIds["elite_annual"] + "  (optional)") -ForegroundColor Gray
}

Write-Host ""
Write-Host "Also set your API keys (get from https://dashboard.stripe.com/apikeys):" -ForegroundColor Yellow
Write-Host "  STRIPE_SECRET_KEY=sk_test_..." -ForegroundColor White
Write-Host "  STRIPE_PUBLISHABLE_KEY=pk_test_..." -ForegroundColor White
Write-Host ""
Write-Host "Then start the webhook listener:" -ForegroundColor Yellow
Write-Host "  .\start_stripe_webhook.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or set the API key directly and re-run this script:" -ForegroundColor Yellow
Write-Host '  $env:STRIPE_API_KEY="sk_test_..."; .\setup_stripe_products.ps1' -ForegroundColor White
