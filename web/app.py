"""
FastAPI web application for Betting Intelligence.
A proper web GUI replacing the Streamlit dashboard.

Run:
    betting-intel web start
    # or
    uvicorn web.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any
import asyncio
from typing import Any

from dotenv import load_dotenv

# Load .env into process environment BEFORE importing other modules
# This ensures os.getenv() works for Stripe keys in API routes
load_dotenv()

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Add src/ to path so betting_intel.* imports work when running as script
sys.path.insert(0, str(PROJECT_ROOT / "src"))

app = FastAPI(
    title="Betting Intelligence",
    description="AI-powered basketball betting recommendations",
    version="0.2.0",
)

# Mount static files
static_dir = HERE / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(HERE / "templates"))


# ── Engine Integration ─────────────────────────────────────────────────────

# Simple TTL cache: regenerates data every 30 seconds
_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
CACHE_TTL = 30.0  # seconds


def _now():
    import time
    return time.time()


def get_engine():
    """Lazy-load the recommendation engine."""
    from betting_intel.recommendations import RecommendationEngine
    return RecommendationEngine()


def load_dashboard_data(force_refresh: bool = False):
    """Load all data needed for the dashboard with 30s TTL caching."""
    now = _now()
    if not force_refresh and _cache["data"] is not None and (now - _cache["timestamp"]) < CACHE_TTL:
        return _cache["data"]

    engine = get_engine()
    all_bets = engine.generate_all_bets()
    clear_picks = engine.get_clear_picks()
    summary = engine.get_summary()
    todays = engine.get_todays_card()

    # Group today's bets by game
    games = {}
    for bet in todays:
        key = bet.matchup
        if key not in games:
            games[key] = {"league": bet.league, "bets": []}
        games[key]["bets"].append(bet)

    data = {
        "all_bets": all_bets,
        "clear_picks": clear_picks,
        "summary": summary,
        "todays_bets": todays,
        "games": games,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _cache["data"] = data
    _cache["timestamp"] = now
    return data


def bet_to_dict(bet) -> dict:
    """Convert a BetSuggestion to a plain dict for templates."""
    return {
        "game_id": bet.game_id,
        "game_date": bet.game_date,
        "matchup": bet.matchup,
        "league": bet.league,
        "bet_type": bet.bet_type.value,
        "bet_type_display": bet.bet_type.display_name(),
        "bet_type_icon": bet.bet_type.icon(),
        "bet_side": bet.bet_side,
        "action": bet.action,
        "edge_pct": round(bet.edge_pct * 100, 1),  # percentage
        "stake_dollars": round(bet.stake_dollars, 0),
        "kelly_fraction": round(bet.kelly_fraction, 4),
        "win_probability": round(bet.win_probability, 3),
        "confidence": bet.confidence.value,
        "is_clear_pick": bet.is_clear_pick,
        "reasoning": bet.reasoning,
        "model_name": bet.model_name,
        "tags": bet.tags,
        "market_line": bet.market_line,
        "predicted_value": round(bet.predicted_value, 1),
    }


# ── Routes ─────────────────────────────────────────────────────────────────



async def _safe_load_dashboard() -> dict:
    """Load dashboard data with a 30-second timeout. Returns empty defaults on failure."""
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(load_dashboard_data),
            timeout=15.0
        )
        if isinstance(data, dict):
            return data
    except asyncio.TimeoutError:
        print("  [WARN] Dashboard data load timed out (30s)")
    except Exception as e:
        print(f"  [WARN] Dashboard data load failed: {e}")
    # Complete fallback with all fields expected by templates
    return {
        "summary": {
            "total_bets": 0, "total_stake": 0.0, "total_collected": 0.0,
            "clear_picks": 0, "games_available": 0, "freshness_seconds": 0,
            "active_signals": 0, "by_type": {}, "nitter_available": False,
        },
        "clear_picks": [],
        "todays_bets": [],
        "games": {},
        "generated_at": "",
    }

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing / marketing page with pricing tiers."""
    return templates.TemplateResponse(
        request,
        "landing.html",
        {},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    data = await _safe_load_dashboard()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "summary": data["summary"],
            "clear_picks": data["clear_picks"],
            "todays_bets": data["todays_bets"],
            "games": data["games"],
            "generated_at": data["generated_at"],
            "today": date.today().isoformat(),
        },
    )


@app.get("/clear-picks", response_class=HTMLResponse)
async def clear_picks_page(
    request: Request,
    league: str = "all",
    bet_type: str = "all",
    min_edge: float = 0.0,
):
    """Clear picks page with HTMX filtering."""
    data = await _safe_load_dashboard()
    picks = data["clear_picks"]

    # Filter
    filtered = []
    for cp in picks:
        bet = cp.bet
        if league != "all" and bet.league.lower() != league.lower():
            continue
        if bet_type != "all" and bet.bet_type.value != bet_type:
            continue
        if bet.edge_pct < min_edge:
            continue
        filtered.append(cp)

    leagues = sorted(set(cp.bet.league for cp in picks))
    types = sorted(set(cp.bet.bet_type.value for cp in picks))

    # Check if HTMX request (partial update)
    is_htmx = request.headers.get("HX-Request") == "true"

    template = "partials/clear_picks_list.html" if is_htmx else "clear_picks.html"
    return templates.TemplateResponse(
        request,
        template,
        {
            "clear_picks": filtered,
            "leagues": leagues,
            "types": types,
            "selected_league": league,
            "selected_type": bet_type,
            "min_edge": min_edge,
            "summary": data["summary"],
            "generated_at": data["generated_at"],
        },
    )


@app.get("/all-bets", response_class=HTMLResponse)
async def all_bets_page(
    request: Request,
    league: str = "all",
    bet_type: str = "all",
    min_edge: float = 0.0,
    sort_by: str = "edge",
):
    """All bets page with sorting and filtering."""
    data = await _safe_load_dashboard()
    bets = data["all_bets"]

    # Filter
    if league != "all":
        bets = [b for b in bets if b.league.lower() == league.lower()]
    if bet_type != "all":
        bets = [b for b in bets if b.bet_type.value == bet_type]
    if min_edge > 0:
        bets = [b for b in bets if b.edge_pct >= min_edge]

    # Sort
    if sort_by == "edge":
        bets.sort(key=lambda b: b.edge_pct, reverse=True)
    elif sort_by == "stake":
        bets.sort(key=lambda b: b.stake_dollars, reverse=True)
    elif sort_by == "confidence":
        bets.sort(key=lambda b: b.confidence.numeric(), reverse=True)

    leagues = sorted(set(b.league for b in data["all_bets"]))
    types = sorted(set(b.bet_type.value for b in data["all_bets"]))

    is_htmx = request.headers.get("HX-Request") == "true"
    template = "partials/bets_table.html" if is_htmx else "all_bets.html"

    return templates.TemplateResponse(
        request,
        template,
        {
            "bets": [bet_to_dict(b) for b in bets],
            "leagues": leagues,
            "types": types,
            "selected_league": league,
            "selected_type": bet_type,
            "min_edge": min_edge,
            "sort_by": sort_by,
            "total": len(bets),
            "summary": data["summary"],
            "generated_at": data["generated_at"],
        },
    )


@app.get("/todays-card", response_class=HTMLResponse)
async def todays_card_page(request: Request):
    """Today's betting card."""
    data = await _safe_load_dashboard()
    return templates.TemplateResponse(
        request,
        "todays_card.html",
        {
            "games": data["games"],
            "todays_bets": data["todays_bets"],
            "summary": data["summary"],
            "generated_at": data["generated_at"],
            "today": date.today().isoformat(),
        },
    )


@app.get("/tomorrow", response_class=HTMLResponse)
async def tomorrow_page(request: Request):
    """Tomorrow's betting card — one-day-ahead predictions."""
    engine = get_engine()
    tomorrow_bets = engine.get_tomorrows_card()
    summary = engine.get_summary()

    # Group by game
    games = {}
    for bet in tomorrow_bets:
        key = bet.matchup
        if key not in games:
            games[key] = {"league": bet.league, "series": "", "bets": []}
        games[key]["bets"].append(bet)

    tomorrow_date = (date.today() + timedelta(days=1)).isoformat()

    return templates.TemplateResponse(
        request,
        "tomorrow.html",
        {
            "games": games,
            "tomorrow_bets": tomorrow_bets,
            "summary": summary,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tomorrow": tomorrow_date,
            "today": date.today().isoformat(),
        },
    )


@app.get("/player-props", response_class=HTMLResponse)
async def player_props_page(
    request: Request,
    home: str = "Spurs",
    away: str = "Thunder",
    league: str = "NBA",
):
    """Player props page."""
    from betting_intel.recommendations.player_props import PlayerPropEngine

    engine = PlayerPropEngine()
    props = engine.predict_for_game(home=home, away=away, league=league)
    props_dicts = [bet_to_dict(p) for p in props]

    # Group by team
    home_props = [p for p in props_dicts if home in p["bet_side"]]
    away_props = [p for p in props_dicts if away in p["bet_side"]]

    return templates.TemplateResponse(
        request,
        "player_props.html",
        {
            "home_team": home,
            "away_team": away,
            "league": league,
            "home_props": home_props,
            "away_props": away_props,
            "total": len(props),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


@app.get("/api/refresh")
async def api_refresh():
    """Refresh cached data and return summary JSON."""
    data = load_dashboard_data(force_refresh=True)
    return JSONResponse(content={
        "total_bets": data["summary"]["total_bets"],
        "clear_picks": data["summary"]["clear_picks"],
        "games_available": data["summary"]["games_available"],
        "avg_edge": round(data["summary"]["avg_edge"] * 100, 1),
        "total_stake": round(data["summary"]["total_stake"], 0),
        "generated_at": data["generated_at"],
    })


@app.get("/api/bets")
async def api_bets(
    league: str = "all",
    bet_type: str = "all",
    min_edge: float = 0.0,
    limit: int = 50,
):
    """JSON API for bets."""
    data = await _safe_load_dashboard()
    bets = data["all_bets"]
    if league != "all":
        bets = [b for b in bets if b.league.lower() == league.lower()]
    if bet_type != "all":
        bets = [b for b in bets if b.bet_type.value == bet_type]
    if min_edge > 0:
        bets = [b for b in bets if b.edge_pct >= min_edge]
    bets.sort(key=lambda b: b.edge_pct, reverse=True)
    return JSONResponse(content=[bet_to_dict(b) for b in bets[:limit]])


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request):
    """X/Twitter intelligence signals page."""
    from betting_intel.data.x_signals import TwitterSignalCollector

    collector = TwitterSignalCollector()
    collector.collect_all()
    signals = collector.get_recent_signals(limit=40)
    summary = collector.get_summary_stats()
    team_alerts = collector.get_team_alerts()
    most_impactful = [s.to_dict() for s in collector.get_most_impactful_signals(limit=10)]

    # Get account counts
    from betting_intel.data.nba_accounts import (
        get_all_accounts, get_accounts_by_role, AccountRole
    )
    all_accounts = get_all_accounts()
    insider_count = len(get_accounts_by_role(AccountRole.INSIDER))
    beat_count = len(get_accounts_by_role(AccountRole.BEAT_REPORTER))
    tracker_count = len(get_accounts_by_role(AccountRole.INJURY_TRACKER))

    return templates.TemplateResponse(
        request,
        "signals.html",
        {
            "signals": signals,
            "summary": summary,
            "team_alerts": team_alerts,
            "most_impactful": most_impactful,
            "account_count": len(all_accounts),
            "insider_count": insider_count,
            "beat_count": beat_count,
            "tracker_count": tracker_count,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


@app.get("/api/signals")
async def api_signals():
    """JSON API for X/Twitter signals."""
    from betting_intel.data.x_signals import TwitterSignalCollector
    collector = TwitterSignalCollector()
    collector.collect_all()
    return JSONResponse(content={
        "signals": collector.get_recent_signals(limit=20),
        "summary": collector.get_summary_stats(),
        "team_alerts": {
            team: [s.to_dict() for s in sigs]
            for team, sigs in collector.get_team_alerts().items()
        },
    })


@app.get("/api/clear-picks")
async def api_clear_picks():
    """JSON API for clear picks."""
    data = await _safe_load_dashboard()
    return JSONResponse(content=[
        {
            "bet": bet_to_dict(cp.bet),
            "clear_score": round(cp.clear_score, 1),
            "risk_level": cp.risk_level,
            "reasons": cp.reasons,
        }
        for cp in data["clear_picks"]
    ])


# ── Stripe Integration ────────────────────────────────────────────────────

_stripe_manager = None


def _get_stripe_manager() -> StripeIntegration:
    """Lazy-load the Stripe integration."""
    global _stripe_manager
    if _stripe_manager is None:
        from betting_intel.business.subscriptions import StripeIntegration
        _stripe_manager = StripeIntegration()
    return _stripe_manager


def _get_sub_manager() -> "SubscriptionManager":
    """Lazy-load the subscription manager."""
    from betting_intel.business.subscriptions import SubscriptionManager
    sm = SubscriptionManager(str(PROJECT_ROOT / "data" / "subscribers.json"))
    return sm


@app.get("/api/stripe/config")
async def stripe_config():
    """Return Stripe publishable key for the frontend."""
    import os
    pub_key = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    return JSONResponse({
        "publishableKey": pub_key,
        "enabled": bool(pub_key) and pub_key != "your-stripe-key-here",
    })


@app.post("/api/stripe/create-checkout-session")
async def create_checkout_session(request: Request):
    """
    Create a Stripe Checkout Session for a subscription.

    Expects JSON: { "tier": "basic", "interval": "month", "user_id": "..." }
      - interval: "month" or "year"
    Returns: { "url": "https://checkout.stripe.com/..." }
    """
    import os

    body = await request.json()
    tier = body.get("tier", "basic")
    interval = body.get("interval", "month")
    user_id = body.get("user_id", f"user_{os.urandom(4).hex()}")

    domain = os.getenv("SITE_DOMAIN", "http://localhost:8000")
    success_url = f"{domain}/subscribe/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{domain}/subscribe/cancel"

    stripe_mgr = _get_stripe_manager()

    if not stripe_mgr.is_enabled():
        # Demo mode: simulate checkout for development
        sm = _get_sub_manager()
        months = 12 if interval == "year" else 1
        sm.add_subscriber(
            user_id=user_id,
            tier=tier,
            email="demo@example.com",
            months=months,
        )
        return JSONResponse({
            "url": f"/subscribe/success?demo=1&tier={tier}&interval={interval}",
            "demo": True,
            "user_id": user_id,
        })

    checkout_url = stripe_mgr.create_checkout_session(
        user_id=user_id,
        tier=tier,
        interval=interval,
        success_url=success_url,
        cancel_url=cancel_url,
    )

    if not checkout_url:
        return JSONResponse(
            {"error": "Failed to create checkout session. Check Stripe configuration."},
            status_code=400,
        )

    return JSONResponse({"url": checkout_url, "demo": False})


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Handle incoming Stripe webhook events.

    This endpoint receives events from Stripe:
      - checkout.session.completed → activate subscription
      - customer.subscription.updated → tier changes
      - customer.subscription.deleted → cancellations
      - invoice.payment_succeeded → renewal confirmations
    """
    import stripe
    import os

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # If no webhook secret configured, skip verification for development
    if webhook_secret:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
        except (ValueError, stripe.error.SignatureVerificationError) as e:
            logger.warning(f"Stripe webhook signature verification failed: {e}")
            return JSONResponse({"error": "Invalid signature"}, status_code=400)
    else:
        # Dev mode: parse payload manually
        import json
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    stripe_mgr = _get_stripe_manager()
    result = stripe_mgr.process_webhook(event)

    if result:
        sm = _get_sub_manager()
        action = result.get("action")
        user_id = result.get("user_id", "")
        tier = result.get("tier", "basic")
        status = result.get("status", "")

        if action == "subscribe" and status == "active":
            months = result.get("months", 1)
            sm.add_subscriber(
                user_id=user_id,
                tier=tier,
                email=result.get("email", ""),
                stripe_customer_id=result.get("stripe_customer_id", ""),
                stripe_subscription_id=result.get("stripe_subscription_id", ""),
                months=months,
            )
        elif action == "update":
            sm.update_tier(user_id, tier)
        elif action == "cancel":
            sm.cancel_subscription(user_id)

    return JSONResponse({"received": True})


@app.get("/subscribe/success", response_class=HTMLResponse)
async def subscribe_success(request: Request, session_id: str = "", demo: str = "0", tier: str = "basic"):
    """Show a success page after a subscription is activated."""
    is_demo = demo == "1"
    tier_name = tier.capitalize()
    return templates.TemplateResponse(
        request,
        "subscribe_success.html",
        {
            "session_id": session_id,
            "demo": is_demo,
            "tier": tier_name,
            "dashboard_url": "/dashboard",
        },
    )


@app.get("/subscribe/cancel", response_class=HTMLResponse)
async def subscribe_cancel(request: Request):
    """Show a cancellation page if the user leaves the checkout."""
    return templates.TemplateResponse(
        request,
        "subscribe_cancel.html",
        {
            "pricing_url": "/#pricing",
        },
    )


@app.get("/subscribe/manage", response_class=HTMLResponse)
async def subscribe_manage(request: Request, user_id: str = ""):
    """Subscription management page — view/edit current plan."""
    sm = _get_sub_manager()
    stats = sm.get_stats()

    # Look up stripe_customer_id for the portal link
    stripe_customer_id = ""
    if user_id:
        sub = sm.get_subscriber(user_id)
        if sub and sub.stripe_customer_id:
            stripe_customer_id = sub.stripe_customer_id

    return templates.TemplateResponse(
        request,
        "subscribe_manage.html",
        {
            "stats": stats,
            "dashboard_url": "/dashboard",
            "stripe_customer_id": stripe_customer_id,
        },
    )


@app.post("/api/stripe/create-portal-session")
async def create_portal_session(request: Request):
    """
    Create a Stripe Customer Portal session for subscription management.

    Expects JSON: { "customer_id": "cus_...", "return_url": "..." }
    Returns: { "url": "https://billing.stripe.com/..." }
    """
    import os

    body = await request.json()
    customer_id = body.get("customer_id", "")
    return_url = body.get("return_url", os.getenv("SITE_DOMAIN", "http://localhost:8000") + "/subscribe/manage")

    if not customer_id:
        return JSONResponse(
            {"error": "customer_id is required"},
            status_code=400,
        )

    stripe_mgr = _get_stripe_manager()
    portal_url = stripe_mgr.create_customer_portal_session(
        customer_id=customer_id,
        return_url=return_url,
    )

    if not portal_url:
        return JSONResponse(
            {"error": "Failed to create portal session. Stripe may not be configured."},
            status_code=400,
        )

    return JSONResponse({"url": portal_url})


# ── Legal Pages ───────────────────────────────────────────────────────────


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """Privacy Policy page."""
    return templates.TemplateResponse(request, "privacy.html", {})


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    """Terms of Service page."""
    return templates.TemplateResponse(request, "terms.html", {})


# ── SEO Routes ──────────────────────────────────────────────────────────────


@app.get("/robots.txt", response_class=HTMLResponse)
async def robots_txt():
    """Serve robots.txt for search engines."""
    return HTMLResponse(
        content="""# https://www.robotstxt.org/robotstxt.html
User-agent: *
Allow: /
Allow: /dashboard
Allow: /static/
Disallow: /api/
Disallow: /signals

Sitemap: https://exactbets.com/sitemap.xml
""",
        media_type="text/plain",
    )


@app.get("/sitemap.xml", response_class=HTMLResponse)
async def sitemap_xml():
    """Serve sitemap.xml for search engines."""
    today = date.today().isoformat()
    return HTMLResponse(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://exactbets.com/</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://exactbets.com/dashboard</loc>
    <lastmod>{today}</lastmod>
    <changefreq>hourly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://exactbets.com/todays-card</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.7</priority>
  </url>
  <url>
    <loc>https://exactbets.com/all-bets</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.6</priority>
  </url>
  <url>
    <loc>https://exactbets.com/player-props</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>0.5</priority>
  </url>
  <url>
    <loc>https://exactbets.com/privacy</loc>
    <lastmod>{today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.3</priority>
  </url>
  <url>
    <loc>https://exactbets.com/terms</loc>
    <lastmod>{today}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.3</priority>
  </url>
</urlset>
""",
        media_type="application/xml",
    )


# ── Run ────────────────────────────────────────────────────────────────────


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the web server via uvicorn."""
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run()
