"""
StealthBrowser — Chrome-based web scraper for NBA odds.

DESIGN:
  Two-tier approach:
    Tier 1 (fast path): Direct HTTP call to ESPN's public JSON API.
      No browser needed. Returns data in seconds.
    Tier 2 (stealth path): Launches headless Chrome via Playwright with
      the `playwright-stealth` plugin to evade bot detection. Used when
      the direct HTTP call fails (blocked, changed, etc.).

  ESPN API endpoints used:
    - GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
        → Returns the daily schedule with event IDs.
    - GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eventId}/competitions/{competitionId}/odds
        → Returns detailed odds (spread, moneyline, totals) for each game.

  Output: List of dicts in TheOddsAPI-compatible format so the rest of the
    engine (LivePredictionEngine._parse_games, OddsAPIClient) works unchanged.

USAGE:
    from betting_intel.data.stealth_scraper import StealthBrowser
    games = StealthBrowser.sync_scrape_live_odds()
    # -> [{"id": "...", "home_team": "...", "away_team": "...", "bookmakers": [...]}, ...]

REQUIRES:
    playwright==1.54.0+    (pip install playwright)
    playwright-stealth     (pip install playwright-stealth)
    Chromium installed via: playwright install chromium
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import shared team name mappings from odds_fetcher
from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME, SHORT_TO_ODDS_NAME

# Alias for code clarity: ESPN team names -> short names
ESPN_TEAM_SHORT: dict[str, str] = ODDS_TO_SHORT_NAME

# Reverse mapping: short names -> canonical full names
TEAM_TO_ESPN: dict[str, str] = SHORT_TO_ODDS_NAME

# How long to cache the odds in memory (seconds)
ODDS_CACHE_TTL = 120  # 2 minutes
# Cooldown between Playwright launches (to avoid detection)
PLAYWRIGHT_COOLDOWN = 30  # 30 seconds between browser launches

# ScraperCoordinator integration removed during cleanup — no longer used

_GLOBAL_SCRAPER_MONITOR = None


class StealthBrowser:
    """
    Chrome-based stealth scraper for NBA odds from ESPN.

    Provides TWO methods for fetching odds:
      1. sync_scrape_live_odds() — top-level entry point. Tries fast HTTP
         first, falls back to Playwright stealth if needed.
      2. _scrape_via_http() — direct ESPN API call (fast path).
      3. _scrape_via_playwright() — headless Chrome + stealth (fallback).

    Output format matches TheOddsAPI so the existing parsing code in
    LivePredictionEngine._parse_games() works without modification.
    """

    # Class-level cache to share across engine instances
    _cache: Optional[list[dict]] = None
    _cache_time: float = 0.0
    _last_playwright_launch: float = 0.0

    @classmethod
    def sync_scrape_live_odds(
        cls,
        odds_api_key: Optional[str] = None,
        timeout: int = 25,
    ) -> list[dict]:
        """
        Top-level entry point. Returns live NBA odds in TheOddsAPI format.

        Strategy:
          1. Check in-memory cache (fresh within ODDS_CACHE_TTL)
          2. Try direct ESPN API HTTP call (Tier 1 — fast path)
          3. If that fails, launch Playwright + stealth (Tier 2 — fallback)
          4. Cache and return results

        Args:
            odds_api_key: Ignored (ESPN doesn't need a key), kept for
                          compatibility with the engine's _fetch_stealth_fallback.
            timeout: Max seconds for the entire operation.

        Returns:
            List of game dicts in TheOddsAPI format, or empty list on failure.
        """
        # Check memory cache
        now = time.time()
        if cls._cache is not None and (now - cls._cache_time) < ODDS_CACHE_TTL:
            logger.debug("StealthScraper: Returning cached odds")
            return cls._cache

        # Tier 1: Direct ESPN API (fast path)
        games = cls._scrape_via_http(timeout=timeout)
        if games:
            logger.info(f"StealthScraper (HTTP): {len(games)} games from ESPN API")
            cls._cache = games
            cls._cache_time = now
            return games

        # Tier 2: Playwright + stealth (fallback)
        logger.info("StealthScraper: HTTP path failed, trying Playwright stealth...")

        # Cooldown: don't launch browser too often
        if (now - cls._last_playwright_launch) < PLAYWRIGHT_COOLDOWN:
            logger.warning("StealthScraper: Playwright cooldown active, returning empty")
            return cls._cache or []

        games = cls._scrape_via_playwright(timeout=timeout)
        cls._last_playwright_launch = time.time()

        if games:
            logger.info(f"StealthScraper (Playwright): {len(games)} games scraped")
            cls._cache = games
            cls._cache_time = time.time()
            return games

        logger.warning("StealthScraper: All methods failed — no odds available")
        cls._cache = []
        cls._cache_time = time.time()
        return []

    # ── Tier 1: Direct ESPN HTTP API ─────────────────────────────────────

    @classmethod
    def _scrape_via_http(cls, timeout: int = 15) -> list[dict]:
        """
        Fetch NBA odds directly from ESPN's public JSON API.

        Uses two endpoints:
          1. Scoreboard: lists today's games with event IDs
          2. Core odds API: per-game spread, moneyline, totals

        This is the preferred fast path — no browser needed.
        """
        import urllib.request

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        try:
            # Step 1: Get today's schedule
            scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            req = urllib.request.Request(scoreboard_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                scoreboard = json.loads(raw)

            events = scoreboard.get("events", [])
            if not events:
                logger.info("ESPN scoreboard: no events today")
                return []

            parsed_games: list[dict] = []
            now_utc = datetime.now(timezone.utc)

            for event in events:
                try:
                    game = cls._parse_espn_event(event, now_utc)
                    if game:
                        parsed_games.append(game)
                except Exception as e:
                    logger.debug(f"Skipping malformed ESPN event: {e}")
                    continue

            # Step 2: For each event, fetch detailed odds from core API
            for game in parsed_games:
                event_id = game.get("_espn_event_id")
                competition_id = game.get("_espn_comp_id")
                if event_id and competition_id:
                    try:
                        odds = cls._fetch_espn_detail_odds(event_id, competition_id, timeout)
                        if odds:
                            # Merge odds into the game dict (TheOddsAPI format)
                            cls._merge_odds_into_game(game, odds)
                    except Exception as e:
                        logger.debug(f"Detail odds for event {event_id}: {e}")

            # Clean up internal fields
            for game in parsed_games:
                game.pop("_espn_event_id", None)
                game.pop("_espn_comp_id", None)

            # Filter: only include games with at least some odds data
            valid_games = [
                g for g in parsed_games
                if g.get("bookmakers") and len(g["bookmakers"]) > 0
            ]

            return valid_games

        except urllib.error.HTTPError as e:
            logger.warning(f"ESPN HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}")
            return []
        except urllib.error.URLError as e:
            logger.warning(f"ESPN connection failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"ESPN JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"ESPN scrape failed: {e}")
            return []

    @classmethod
    def _parse_espn_event(cls, event: dict, now_utc: datetime) -> Optional[dict]:
        """Parse an ESPN event into TheOddsAPI format (partial — odds added later)."""
        event_id = event.get("id", "")
        name = event.get("name", "")
        competitions = event.get("competitions", [])
        if not competitions:
            return None

        comp = competitions[0]
        competition_id = comp.get("id", "")
        date_str = comp.get("date", "")

        # Teams
        competitors = comp.get("competitors", [])
        home_team = ""
        away_team = ""
        for c in competitors:
            team_name = c.get("team", {}).get("displayName", "")
            is_home = c.get("homeAway") == "home"
            if is_home:
                home_team = team_name
            else:
                away_team = team_name

        if not home_team or not away_team:
            return None

        # Use canonical full names from the shared mapping
        # ESPN may return "LA Clippers" or "Los Angeles Clippers" — normalize
        short = ESPN_TEAM_SHORT.get(home_team)
        home_full = TEAM_TO_ESPN.get(short, home_team) if short else home_team
        short = ESPN_TEAM_SHORT.get(away_team)
        away_full = TEAM_TO_ESPN.get(short, away_team) if short else away_team

        return {
            "_espn_event_id": event_id,
            "_espn_comp_id": competition_id,
            "id": f"espn_nba_{event_id}",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": date_str,
            "home_team": home_full,
            "away_team": away_full,
            "bookmakers": [],
        }

    @classmethod
    def _fetch_espn_detail_odds(
        cls, event_id: str, competition_id: str, timeout: int
    ) -> Optional[list[dict]]:
        """Fetch detailed betting odds from ESPN's core API for a single game."""
        import urllib.request

        url = (
            f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/"
            f"{event_id}/competitions/{competition_id}/odds"
        )
        headers = {"User-Agent": "Mozilla/5.0"}

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            odds_data = json.loads(raw)

        items = odds_data.get("items", [])
        if not items:
            return None

        parsed_odds = []
        for item in items:
            provider = item.get("provider", {}).get("name", "ESPN")
            provider_id = item.get("provider", {}).get("id", "0")
            spread = item.get("spread")
            over_under = item.get("overUnder")
            over_odds_val = item.get("overOdds")
            under_odds_val = item.get("underOdds")
            away_ml = item.get("awayTeamOdds", {}).get("moneyLine")
            home_ml = item.get("homeTeamOdds", {}).get("moneyLine")
            away_spread_odds = item.get("awayTeamOdds", {}).get("spreadOdds")
            home_spread_odds = item.get("homeTeamOdds", {}).get("spreadOdds")

            parsed_odds.append({
                "key": f"espn_{provider_id}",
                "title": provider,
                "last_update": datetime.now().isoformat(),
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "", "price": home_ml},
                            {"name": "", "price": away_ml},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "", "point": spread, "price": home_spread_odds},
                            {"name": "", "point": -spread if spread else None, "price": away_spread_odds},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": over_under, "price": over_odds_val},
                            {"name": "Under", "point": over_under, "price": under_odds_val},
                        ],
                    },
                ],
            })

        return parsed_odds

    @classmethod
    def _merge_odds_into_game(cls, game: dict, odds_list: list[dict]):
        """Merge ESPN odds into a game dict, matching team names to outcome names."""
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_short = ESPN_TEAM_SHORT.get(home_full, home_full.split()[-1] if " " in home_full else home_full)
        away_short = ESPN_TEAM_SHORT.get(away_full, away_full.split()[-1] if " " in away_full else away_full)

        for bookmaker in odds_list:
            for market in bookmaker.get("markets", []):
                key = market.get("key", "")
                outcomes = market.get("outcomes", [])
                if key in ("h2h", "spreads"):
                    # First outcome = home team, second outcome = away team
                    if len(outcomes) >= 1 and outcomes[0].get("name", "") == "":
                        outcomes[0]["name"] = home_full
                    if len(outcomes) >= 2 and outcomes[1].get("name", "") == "":
                        outcomes[1]["name"] = away_full

            game["bookmakers"].append(bookmaker)

    # ── Tier 2: Playwright Stealth ───────────────────────────────────────

    @classmethod
    def _scrape_via_playwright(cls, timeout: int = 25) -> list[dict]:
        """
        Fallback: Launch headless Chrome via Playwright with stealth plugin.

        Navigates to ESPN's scoreboard, intercepts the JSON API response,
        and parses the data. This approach:
          - Looks like a real Chrome browser to ESPN
          - Cannot be detected as a bot (stealth plugin hides WebDriver flags)
          - Returns the same JSON structure as the HTTP path

        Used only when the direct HTTP call fails.
        """
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                    timezone_id="America/New_York",
                )

                page = context.new_page()

                # Apply stealth if available
                try:
                    from playwright_stealth import Stealth
                    Stealth().apply_stealth_sync(context)
                    logger.debug("Stealth plugin applied to Playwright page")
                except ImportError:
                    logger.debug("playwright-stealth not available, launching without stealth")

                # Intercept JSON API responses
                api_responses: list[dict] = []

                def handle_response(response):
                    url = response.url
                    if "scoreboard" in url or "odds" in url:
                        try:
                            data = response.json()
                            api_responses.append(data)
                        except Exception:
                            pass

                page.on("response", handle_response)

                # Navigate to ESPN NBA scoreboard
                logger.debug("Navigating to ESPN NBA scoreboard...")
                page.goto(
                    "https://www.espn.com/nba/scoreboard",
                    wait_until="domcontentloaded",
                    timeout=timeout * 1000,
                )

                # Wait for odds data to load
                page.wait_for_timeout(5000)

                # Also try the scoreboard API via page.evaluate with fetch
                # Use an async IIFE so fetch() is properly awaited by Playwright
                scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
                try:
                    resp_json = page.evaluate(
                        f"(async () => {{ const r = await fetch('{scoreboard_url}'); return await r.json(); }})()"
                    )
                    if resp_json and isinstance(resp_json, dict):
                        api_responses.append(resp_json)
                except Exception:
                    pass

                browser.close()

                # Parse collected API responses
                for data in api_responses:
                    if isinstance(data, dict) and "events" in data:
                        return cls._parse_api_response(data)

                logger.warning("Playwright: No valid API responses captured")
                return []

        except ImportError as e:
            logger.warning(f"Playwright not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"Playwright scrape failed: {e}")
            return []

    @classmethod
    def _parse_api_response(cls, data: dict) -> list[dict]:
        """
        Parse ESPN API response into TheOddsAPI format.

        Handles both:
          - The main scoreboard response (with events + embedded odds)
          - Individual event odds from the core API
        """
        events = data.get("events", [])
        if not events:
            return []

        now_utc = datetime.now(timezone.utc)
        parsed: list[dict] = []

        for event in events:
            try:
                game = cls._parse_espn_event(event, now_utc)
                if not game:
                    continue

                event_id = game.pop("_espn_event_id", "")
                comp_id = game.pop("_espn_comp_id", "")

                # Check if odds are embedded in the competition data
                competitions = event.get("competitions", [])
                if competitions:
                    comp = competitions[0]
                    embedded_odds = comp.get("odds", [])
                    for odds_item in embedded_odds:
                        odds_list = cls._parse_embedded_odds(odds_item, game)
                        if odds_list:
                            game["bookmakers"].extend(odds_list)

                    # If no embedded odds, try fetching from core API
                    if not game["bookmakers"] and event_id and comp_id:
                        try:
                            odds_list = cls._fetch_espn_detail_odds(event_id, comp_id, 10)
                            if odds_list:
                                cls._merge_odds_into_game(game, odds_list)
                        except Exception:
                            pass

                if game.get("bookmakers"):
                    parsed.append(game)

            except Exception as e:
                logger.debug(f"Parse error for event: {e}")
                continue

        return parsed

    @classmethod
    def _parse_embedded_odds(cls, odds_item: dict, game: dict) -> list[dict]:
        """
        Parse odds embedded in the ESPN competition object.
        These are simpler than the core API odds.
        """
        provider = odds_item.get("provider", {}).get("name", "ESPN")
        provider_id = odds_item.get("provider", {}).get("id", "0")
        spread = odds_item.get("spread")
        over_under = odds_item.get("overUnder")
        over_odds_val = odds_item.get("overOdds")
        under_odds_val = odds_item.get("underOdds")
        away_ml = odds_item.get("awayTeamOdds", {}).get("moneyLine")
        home_ml = odds_item.get("homeTeamOdds", {}).get("moneyLine")
        away_spread_odds = odds_item.get("awayTeamOdds", {}).get("spreadOdds")
        home_spread_odds = odds_item.get("homeTeamOdds", {}).get("spreadOdds")

        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")

        bookmaker = {
            "key": f"espn_{provider_id}",
            "title": provider,
            "last_update": datetime.now().isoformat(),
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": home_full, "price": home_ml},
                        {"name": away_full, "price": away_ml},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": home_full, "point": spread, "price": home_spread_odds},
                        {"name": away_full, "point": -spread if spread else None, "price": away_spread_odds},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": over_under, "price": over_odds_val},
                        {"name": "Under", "point": over_under, "price": under_odds_val},
                    ],
                },
            ],
        }

        return [bookmaker]

    @classmethod
    def clear_cache(cls):
        """Clear the in-memory cache. Call after manual refresh."""
        cls._cache = None
        cls._cache_time = 0.0


def scrape_espn_odds_direct() -> list[dict]:
    """
    Convenience function: scrape NBA odds from ESPN in TheOddsAPI format.

    Faster than StealthBrowser.sync_scrape_live_odds() because it only
    uses the direct HTTP path without Playwright fallback.

    Returns:
        List of game dicts with bookmaker data.
    """
    return StealthBrowser._scrape_via_http(timeout=15)