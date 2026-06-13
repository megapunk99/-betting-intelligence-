#!/usr/bin/env python3
"""
Backfill the market_odds table with historical NBA data from TheOddsAPI.

Two modes:
───────
  scores (free tier)   — Fetch completed game metadata + scores from
                         /v4/sports/{sport}/scores/. This does NOT include
                         odds data (free tier limitation), but it populates
                         game_id, team names, dates, and final scores so
                         the MarketInefficiencySystem can at least build
                         the game schedule mapping.

  historical (paid)    — Fetch full historical odds snapshots from
                         /v4/historical/sports/{sport}/odds. Requires a
                         paid TheOddsAPI subscription. Returns closing
                         lines for every game on the requested date
                         going back to ~June 2020.

Usage:
    # Free tier: backfill scores for the last 3 days
    python tools/backfill_market_odds.py --mode scores

    # Free tier: backfill scores with a custom date range (limited to 3 days)
    python tools/backfill_market_odds.py --mode scores --days-back 3

    # Paid tier: backfill historical odds for a specific date range
    python tools/backfill_market_odds.py --mode historical \\
        --start-date 2024-10-01 --end-date 2025-04-15

    # Paid tier: backfill with daily snapshots (captures line movement)
    python tools/backfill_market_odds.py --mode historical \\
        --start-date 2024-10-01 --snapshot-interval daily

    # Check what's already stored
    python tools/backfill_market_odds.py --mode stats
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Team name mapping — reused across modes
_ODDS_SHORT: dict[str, str] = {}
def _get_odds_short() -> dict[str, str]:
    """Lazy-load the ODDS_TO_SHORT_NAME mapping."""
    global _ODDS_SHORT
    if not _ODDS_SHORT:
        from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME
        _ODDS_SHORT = ODDS_TO_SHORT_NAME
    return _ODDS_SHORT

def _short_name(full_name: str) -> str:
    """Convert full team name to short name using the shared mapping."""
    mapping = _get_odds_short()
    if full_name in mapping:
        return mapping[full_name]
    if " " in full_name:
        return full_name.split()[-1]
    return full_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill")


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Rate limiting: TheOddsAPI free tier allows 500 requests/month
# Paid tier allows more. We'll throttle to 1 req/sec to be safe.
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Default sport
SPORT_KEY = "basketball_nba"
SPORT_TITLE = "NBA"

# US regions + international for max book coverage
REGIONS = "us,us2,eu,uk,au"
MARKETS = "h2h,spreads,totals"


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════


def get_api_key() -> str:
    """Get the API key from environment or settings."""
    # Try direct env var first
    api_key = os.getenv("ODDS_API_KEY", "")
    if api_key and api_key not in ("your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"):
        return api_key

    # Fall back to settings
    try:
        from betting_intel.config import settings as cfg
        if cfg.odds_api_key and cfg.odds_api_key not in ("your-api-key-here", ""):
            return cfg.odds_api_key
    except Exception:
        pass

    return ""


def has_valid_api_key(api_key: str) -> bool:
    """Check if the API key is configured and non-default."""
    return bool(api_key) and api_key not in (
        "your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"
    )


def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def _remove_vig(home_p: float, away_p: float) -> float:
    """Remove vig from two implied probabilities, return home vig-free prob."""
    total = home_p + away_p
    if total > 0:
        return home_p / total
    return home_p


def _fetch_json(url: str, api_key: str, retries: int = MAX_RETRIES) -> Optional[list]:
    """
    Fetch JSON from TheOddsAPI with retry logic.

    Returns the parsed JSON list on success, None on failure.
    Handles 429 (rate limit), 401 (bad key), and network errors.
    """
    headers = {"User-Agent": "betting-intel-backfill/1.0"}

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)

            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")

            if isinstance(data, list):
                logger.info(f"  → {len(data)} events returned (quota: {remaining} remaining, {used} used)")
                return data
            elif isinstance(data, dict) and "message" in data:
                logger.warning(f"  → API message: {data['message']}")
                return None
            else:
                logger.warning(f"  → Unexpected response format")
                return None

        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"  ⚠  Rate limited (429) — waiting {RETRY_DELAY_SECONDS}s before retry {attempt}/{retries}")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            elif e.code == 401:
                logger.error("  ✗  Invalid API key (401). Check your ODDS_API_KEY.")
                return None
            elif e.code == 404:
                logger.warning(f"  → No data for this endpoint/sport (404)")
                return None
            else:
                logger.warning(f"  ⚠  HTTP {e.code} on attempt {attempt}/{retries}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

        except urllib.error.URLError as e:
            logger.warning(f"  ⚠  Connection error on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  ⚠  Response error on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

    logger.error(f"  ✗  Failed after {retries} retries")
    return None


# ═══════════════════════════════════════════════════════════════════════
#  MODE 1: SCORES (Free Tier)
# ═══════════════════════════════════════════════════════════════════════

# TheOddsAPI sport key → our internal odds_fetcher mapping
# We need this because the scores endpoint might return team names slightly
# differently than the odds endpoint for some sports.
SPORT_KEY_TO_RESPONSE_TEAM = {
    "basketball_nba": "nba",
    "basketball_ncaab": "ncaab",
}


def _extract_scores_from_event(event: dict) -> dict:
    """Extract score info from a /scores/ API event."""
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    commence_time = event.get("commence_time", "")
    game_date = commence_time[:10] if commence_time else ""

    # Scores: array of {name, score}
    scores = event.get("scores", [])
    home_score = None
    away_score = None
    for s in scores:
        if s.get("name") == home_team:
            home_score = s.get("score")
        elif s.get("name") == away_team:
            away_score = s.get("score")

    home_score = float(home_score) if home_score is not None else None
    away_score = float(away_score) if away_score is not None else None
    total_points = (home_score + away_score) if home_score is not None and away_score is not None else None

    # Bookmakers — usually empty for completed games on free tier,
    # but TheOddsAPI sometimes includes them for recently completed games.
    bookmakers = event.get("bookmakers", [])

    return {
        "game_id": event.get("id", ""),
        "sport_key": event.get("sport_key", SPORT_KEY),
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
        "game_date": game_date,
        "completed": event.get("completed", False),
        "home_score": home_score,
        "away_score": away_score,
        "total_points": total_points,
        "bookmakers": bookmakers,
    }


def _extract_odds_from_bookmakers(
    game_info: dict,
    home_team: str,
    away_team: str,
) -> dict:
    """
    Extract consensus moneyline odds from bookmaker data.

    Some /scores/ events include bookmaker odds for recently completed
    games (closing lines). This handles that rare case.
    """
    home_ml_values: list[float] = []
    away_ml_values: list[float] = []
    spread_values: list[float] = []
    total_values: list[float] = []

    bookmakers = game_info.get("bookmakers") or []
    for book in bookmakers:
        for market in book.get("markets", []):
            key = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    name = o.get("name", "")
                    price = o.get("price")
                    if price is not None:
                        if name == home_team:
                            home_ml_values.append(float(price))
                        elif name == away_team:
                            away_ml_values.append(float(price))
            elif key == "spreads":
                for o in outcomes:
                    if o.get("name", "") == home_team and o.get("point") is not None:
                        spread_values.append(float(o["point"]))
            elif key == "totals":
                for o in outcomes:
                    if o.get("point") is not None:
                        total_values.append(float(o["point"]))

    def median_or_none(values: list) -> Optional[float]:
        if not values:
            return None
        sorted_vals = sorted(values)
        return sorted_vals[len(sorted_vals) // 2]

    return {
        "home_ml": median_or_none(home_ml_values),
        "away_ml": median_or_none(away_ml_values),
        "spread": median_or_none(spread_values),
        "market_total": median_or_none(total_values),
        "n_books_ml": len(home_ml_values),
    }


def run_scores_mode(args) -> int:
    """
    Fetch completed game scores from TheOddsAPI and store them.

    Uses the /v4/sports/{sport}/scores endpoint with daysFrom parameter.
    Stores game metadata (game_id, teams, date, scores) in the market_odds table.

    The scores endpoint does NOT return odds data on the free tier, but
    it gives us the game schedule mapping that the training pipeline needs.
    """
    api_key = get_api_key()
    if not has_valid_api_key(api_key):
        logger.error("No valid ODDS_API_KEY configured. Set ODDS_API_KEY in .env or environment.")
        logger.error("Get a free key at: https://the-odds-api.com/")
        return 1

    days_back = min(max(1, args.days_back), 3)  # Free tier max is 3 days
    logger.info("═" * 60)
    logger.info(f"SCORES MODE — fetching completed {SPORT_TITLE} games (last {days_back} day(s))")
    logger.info(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
    logger.info("═" * 60)

    # ── Build URL ────────────────────────────────────────────────────
    url = (
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores"
        f"?apiKey={api_key}"
        f"&daysFrom={days_back}"
        f"&dateFormat=iso"
    )

    data = _fetch_json(url, api_key)
    if data is None:
        return 1

    if not data:
        logger.info("No completed games found in the requested window.")
        return 0

    # ── Parse and store ──────────────────────────────────────────────
    from betting_intel.db.market_odds_store import MarketOddsStore

    store = MarketOddsStore()
    stored_count = 0
    skipped_count = 0
    odds_count = 0

    for event in data:
        game_info = _extract_scores_from_event(event)
        if not game_info["game_id"] or not game_info["home_team"] or not game_info["away_team"]:
            skipped_count += 1
            continue

        home_team = game_info["home_team"]
        away_team = game_info["away_team"]
        game_date = game_info["game_date"]
        game_id = game_info["game_id"]

        # Extract short names using same mapping as the engine
        home_short = _short_name(home_team)
        away_short = _short_name(away_team)

        # Check if we already have this game_id in the store
        existing = store.get_odds_for_date(game_date)
        already_stored = False
        if not existing.empty and game_id in existing["game_id"].values:
            already_stored = True

        if already_stored and not args.force:
            skipped_count += 1
            continue

        # Try to extract odds from bookmakers (if this game has them)
        odds_data = _extract_odds_from_bookmakers(game_info, home_team, away_team)
        has_odds = odds_data["home_ml"] is not None

        source_label = "historical_scores" if not has_odds else "historical_closing"

        ok = store.log_snapshot(
            game_id=game_id,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            home_team_short=home_short,
            away_team_short=away_short,
            home_ml=odds_data["home_ml"],
            away_ml=odds_data["away_ml"],
            spread=odds_data["spread"],
            market_total=odds_data["market_total"],
            n_books_ml=odds_data["n_books_ml"],
            source=source_label,
            sport_key=SPORT_KEY,
        )

        if ok:
            stored_count += 1
            if has_odds:
                odds_count += 1

            summary = (
                f"  {'✓' if has_odds else '·'} "
                f"{game_date}  {home_short} vs {away_short}"
            )
            if has_odds:
                ml_home = odds_data["home_ml"]
                ml_away = odds_data["away_ml"]
                summary += f"  [ML: {ml_home:+.0f}/{ml_away:+.0f}]"
            if game_info["total_points"] is not None:
                summary += f"  (final: {game_info['home_score']:.0f}-{game_info['away_score']:.0f}={game_info['total_points']:.0f})"
            logger.info(summary)
        else:
            skipped_count += 1

        time.sleep(REQUEST_DELAY_SECONDS)  # Be kind to the API

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info(f"Stored: {stored_count} games ({odds_count} with odds, {stored_count - odds_count} score-only)")
    logger.info(f"Skipped: {skipped_count} (already in store)")
    logger.info(f"Total API events: {len(data)}")

    stats = store.get_stats()
    logger.info(f"Market odds table now has {stats['total_snapshots']} snapshots across {stats['unique_games']} unique games")

    return 0


# ═══════════════════════════════════════════════════════════════════════
#  MODE 2: HISTORICAL ODDS (Paid Tier)
# ═══════════════════════════════════════════════════════════════════════


def _is_paid_tier(api_key: str) -> bool:
    """
    Quick check if this API key has access to the historical endpoint.

    Makes a lightweight HEAD-style request to the historical endpoint.
    Paid tier returns data; free tier returns 403/401.
    """
    test_url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
        f"?apiKey={api_key}"
        f"&date=2024-01-15T00:00:00Z"
        f"&regions={REGIONS}"
        f"&markets={MARKETS}"
        f"&oddsFormat=american"
        f"&dateFormat=iso"
    )

    try:
        req = urllib.request.Request(test_url, headers={"User-Agent": "betting-intel-backfill/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            logger.info(f"  Historical endpoint accessible (quota: {remaining} remaining, {used} used)")
            return True
    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            logger.warning("  Historical endpoint returned 403/401 — paid subscription required.")
            logger.warning("  See: https://the-odds-api.com/#pricing")
            return False
        elif e.code == 404:
            # 404 could mean no data for that date, which is different from no access
            # The actual response body would tell us
            logger.info("  Historical endpoint returned 404 (likely means no data for that date, or free tier)")
            try:
                body = e.read().decode("utf-8")
                if "method not allowed" in body.lower() or "access" in body.lower() or "permission" in body.lower():
                    return False
            except Exception:
                pass
            return True  # 404 could also mean no games on that date
        else:
            logger.warning(f"  Historical endpoint check failed with HTTP {e.code}")
            return False
    except Exception as e:
        logger.warning(f"  Historical endpoint check failed: {e}")
        return False


def _fetch_historical_snapshot(api_key: str, snapshot_date: str) -> Optional[list]:
    """
    Fetch a single historical odds snapshot for a specific date.

    The Odds API returns the closest available snapshot at or before
    the requested date. Each snapshot contains all games that had odds
    available at that point in time, with full bookmaker data.
    """
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
        f"?apiKey={api_key}"
        f"&date={snapshot_date}T00:00:00Z"
        f"&regions={REGIONS}"
        f"&markets={MARKETS}"
        f"&oddsFormat=american"
        f"&dateFormat=iso"
    )

    data = _fetch_json(url, api_key)
    return data


def _extract_historical_games(event: dict) -> list[dict]:
    """
    Extract game data from a historical snapshot event.

    The historical endpoint returns the same structure as the odds
    endpoint — each event has home_team, away_team, commence_time,
    and bookmakers with markets and outcomes.
    """
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    commence_time = event.get("commence_time", "")
    game_date = commence_time[:10] if commence_time else ""
    event_id = event.get("id", "")

    # Short names (uses lazy-loaded module-level mapping)
    home_short = _short_name(home_team)
    away_short = _short_name(away_team)

    games = []  # Usually one, but a historical snapshot could have multiple if snapshot covers multiple days

    # Extract consensus odds from all bookmakers
    home_ml_values: list[float] = []
    away_ml_values: list[float] = []
    spread_values: list[float] = []
    total_values: list[float] = []

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                for o in outcomes:
                    name = o.get("name", "")
                    price = o.get("price")
                    if price is not None:
                        if name == home_team:
                            home_ml_values.append(float(price))
                        elif name == away_team:
                            away_ml_values.append(float(price))
            elif key == "spreads":
                for o in outcomes:
                    point = o.get("point")
                    if point is not None and o.get("name", "") == home_team:
                        spread_values.append(float(point))
            elif key == "totals":
                for o in outcomes:
                    point = o.get("point")
                    if point is not None:
                        total_values.append(float(point))

    def median_or_none(values: list) -> Optional[float]:
        if not values:
            return None
        sorted_vals = sorted(values)
        return sorted_vals[len(sorted_vals) // 2]

    games.append({
        "game_id": event_id,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_team_short": home_short,
        "away_team_short": away_short,
        "commence_time": commence_time,
        "home_ml": median_or_none(home_ml_values),
        "away_ml": median_or_none(away_ml_values),
        "spread": median_or_none(spread_values),
        "market_total": median_or_none(total_values),
        "n_books_ml": len(home_ml_values),
        "source": "historical_odds",
    })

    return games


def run_historical_mode(args) -> int:
    """
    Fetch historical odds snapshots from TheOddsAPI and store them.

    Uses the /v4/historical/sports/{sport}/odds endpoint which requires
    a paid subscription. Iterates through dates from start_date to end_date,
    fetching the closest available snapshot for each period.

    Each snapshot contains all games that had odds available at that time,
    with full bookmaker data (moneyline, spread, totals).
    """
    api_key = get_api_key()
    if not has_valid_api_key(api_key):
        logger.error("No valid ODDS_API_KEY configured. Set ODDS_API_KEY in .env or environment.")
        return 1

    # Parse date range
    try:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1

    if start >= end:
        logger.error("start_date must be before end_date")
        return 1

    # Determine snapshot interval
    snapshot_interval = args.snapshot_interval
    if snapshot_interval == "daily":
        step = timedelta(days=1)
        snapshot_label = "daily snapshots"
    elif snapshot_interval == "weekly":
        step = timedelta(weeks=1)
        snapshot_label = "weekly snapshots"
    elif snapshot_interval == "monthly":
        step = timedelta(days=30)
        snapshot_label = "monthly snapshots"
    else:
        step = timedelta(days=max(1, int(snapshot_interval)))
        snapshot_label = f"{int(step.days)}-day snapshots"

    logger.info("═" * 60)
    logger.info(f"HISTORICAL ODDS MODE — {snapshot_label}")
    logger.info(f"Range: {args.start_date} to {args.end_date}")
    logger.info("═" * 60)
    logger.info(f"Attempting to access /v4/historical/sports/{SPORT_KEY}/odds...")
    logger.info("(Requires a paid TheOddsAPI subscription. Free tier will fail gracefully.)")

    from betting_intel.db.market_odds_store import MarketOddsStore

    store = MarketOddsStore()

    total_stored = 0
    total_skipped = 0
    total_api_calls = 0
    dates_with_data = 0
    dates_empty = 0

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Snapshot: {date_str}")

        data = _fetch_historical_snapshot(api_key, date_str)
        total_api_calls += 1

        if data is None:
            logger.warning(f"  ⚠  Skipping {date_str} (API error)")
            current += step
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        if not data:
            dates_empty += 1
            logger.info(f"  No games found for {date_str}")
            current += step
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        dates_with_data += 1
        games_stored = 0
        games_skipped = 0

        for event in data:
            extracted = _extract_historical_games(event)
            for game_data in extracted:
                if not game_data["game_id"]:
                    games_skipped += 1
                    continue

                # Check if already stored (avoid duplicates across snapshot dates)
                existing = store.get_odds_for_date(game_data["game_date"])
                already = False
                if not existing.empty and game_data["game_id"] in existing["game_id"].values:
                    already = True

                if already and not args.force:
                    games_skipped += 1
                    continue

                ok = store.log_snapshot(
                    game_id=game_data["game_id"],
                    game_date=game_data["game_date"],
                    home_team=game_data["home_team"],
                    away_team=game_data["away_team"],
                    home_team_short=game_data["home_team_short"],
                    away_team_short=game_data["away_team_short"],
                    home_ml=game_data["home_ml"],
                    away_ml=game_data["away_ml"],
                    spread=game_data["spread"],
                    market_total=game_data["market_total"],
                    n_books_ml=game_data["n_books_ml"],
                    source=game_data["source"],
                    sport_key=SPORT_KEY,
                )

                if ok:
                    games_stored += 1
                    ml_str = ""
                    if game_data["home_ml"] and game_data["away_ml"]:
                        ml_str = f"  ML: {game_data['home_ml']:+.0f}/{game_data['away_ml']:+.0f}"
                    logger.info(
                        f"  ✓ {game_data['game_date']}  "
                        f"{game_data['home_team_short']} vs {game_data['away_team_short']}"
                        f"{ml_str}"
                    )
                else:
                    games_skipped += 1

        total_stored += games_stored
        total_skipped += games_skipped
        logger.info(f"  → Stored: {games_stored}, Skipped: {games_skipped}")

        current += step
        time.sleep(REQUEST_DELAY_SECONDS)  # Rate limiting

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("SUMMARY")
    logger.info("═" * 60)
    logger.info(f"API calls made: {total_api_calls}")
    logger.info(f"Dates with data: {dates_with_data}")
    logger.info(f"Dates empty: {dates_empty}")
    logger.info(f"Games stored: {total_stored}")
    logger.info(f"Games skipped (duplicates): {total_skipped}")

    stats = store.get_stats()
    logger.info(f"Market odds table now has {stats['total_snapshots']} snapshots across {stats['unique_games']} unique games")

    return 0


# ═══════════════════════════════════════════════════════════════════════
#  MODE 3: STATS
# ═══════════════════════════════════════════════════════════════════════


def run_stats_mode(args) -> int:
    """Display current state of the market_odds table."""
    from betting_intel.db.market_odds_store import MarketOddsStore

    store = MarketOddsStore()
    stats = store.get_stats()

    logger.info("═" * 60)
    logger.info("MARKET ODDS TABLE STATS")
    logger.info("═" * 60)
    logger.info(f"Total snapshots:  {stats['total_snapshots']}")
    logger.info(f"Unique games:     {stats['unique_games']}")

    # Get date range
    from betting_intel.db.schema import MarketOdds
    session = store._db.get_session()
    try:
        first = session.query(MarketOdds.game_date).order_by(MarketOdds.game_date.asc()).first()
        last = session.query(MarketOdds.game_date).order_by(MarketOdds.game_date.desc()).first()
        if first and last:
            logger.info(f"Date range:       {first[0]} to {last[0]}")
        else:
            logger.info("Date range:       (empty)")
    except Exception:
        pass
    finally:
        session.close()

    # Count by source
    session = store._db.get_session()
    try:
        from sqlalchemy import func
        source_counts = (
            session.query(MarketOdds.source, func.count(MarketOdds.id))
            .group_by(MarketOdds.source)
            .all()
        )
        if source_counts:
            logger.info("Source breakdown:")
            for source, count in source_counts:
                logger.info(f"  {source:25s}  {count}")
    except Exception:
        pass
    finally:
        session.close()

    return 0


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Backfill market_odds table with historical NBA data from TheOddsAPI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Free tier: backfill game scores for last 3 days
  python tools/backfill_market_odds.py --mode scores

  # Paid tier: full historical odds backfill
  python tools/backfill_market_odds.py --mode historical \\
      --start-date 2024-10-01 --end-date 2024-11-01

  # Paid tier: weekly snapshots (less API calls)
  python tools/backfill_market_odds.py --mode historical \\
      --start-date 2024-10-01 --end-date 2025-04-15 --snapshot-interval weekly

  # Check what's stored
  python tools/backfill_market_odds.py --mode stats
        """,
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["scores", "historical", "stats"],
        help="scores: free-tier game metadata | historical: paid-tier full odds | stats: check DB",
    )

    # Scores mode options
    parser.add_argument(
        "--days-back", type=int, default=3,
        help="How many days back to query (free tier max: 3, default: 3)",
    )

    # Historical mode options
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD) for historical mode")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD) for historical mode")
    parser.add_argument(
        "--snapshot-interval", type=str, default="daily",
        choices=["daily", "weekly", "monthly"],
        help="How often to take snapshots (daily | weekly | monthly, default: daily)",
    )

    # Global options
    parser.add_argument("--force", action="store_true", help="Overwrite existing records")

    args = parser.parse_args()

    if args.mode == "scores":
        return run_scores_mode(args)
    elif args.mode == "historical":
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required for --mode historical")
        return run_historical_mode(args)
    elif args.mode == "stats":
        return run_stats_mode(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
