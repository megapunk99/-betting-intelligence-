"""
DraftKings Sportsbook Scraper — free second odds source alongside ESPN.

DESIGN:
  Since DraftKings blocks direct HTTP requests (Cloudflare + regional blocks),
  this scraper uses Playwright to launch a real Chromium browser that:
    1. Navigates to sportsbook.draftkings.com/basketball/nba
    2. Intercepts the internal JSON API responses
    3. Parses the DraftKings format into TheOddsAPI-compatible format

  The output format matches TheOddsAPI so the engine's _parse_games() method
  works without modification. This means DraftKings odds get merged alongside
  ESPN odds, giving better consensus lines across multiple books.

  DraftKings internal API structure (reverse-engineered from their web app):
    - Event groups:  /sites/US-DK/api/v5/eventgroups/{eventGroupId}?format=json
      Returns nested JSON with events, markets, and outcomes.
    - Team names:  "New York Knicks", "Los Angeles Lakers" (full names)
    - Markets:  "Moneyline" → h2h, "Point Spread" → spreads, "Total Points" → totals

USAGE:
    from betting_intel.data.draftkings_scraper import DraftKingsScraper
    games = DraftKingsScraper.scrape()

REQUIRES:
    playwright==1.54.0+  (pip install playwright)
    Chromium: playwright install chromium
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import shared team name mappings
from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME, SHORT_TO_ODDS_NAME

# How long to cache results (seconds)
ODDS_CACHE_TTL = 120  # 2 minutes

# Known DraftKings event group IDs for NBA
# These can change between seasons; we try multiple known IDs
NBA_EVENT_GROUP_IDS = [42648, 89226889, 89226336, 42647, 88822667]
NBA_SPORT_URL = "https://sportsbook.draftkings.com/basketball/nba"
SPORTSBOOK_BASE = "https://sportsbook.draftkings.com"

# Mapping DraftKings market names to standard keys
DK_MARKET_MAP = {
    "Moneyline": "h2h",
    "Point Spread": "spreads",
    "Total Points": "totals",
    "Total Points Over/Under": "totals",
}

# American odds have a standard lower bound check
MIN_REASONABLE_ODDS = -50000
MAX_REASONABLE_ODDS = 50000


class DraftKingsScraper:
    """
    Playwright-based scraper for DraftKings NBA odds.

    Uses a real Chromium browser to bypass Cloudflare and regional blocks.
    Outputs odds in TheOddsAPI-compatible format for integration with the
    LivePredictionEngine.

    Thread-safe: class-level cache avoids redundant browser launches.
    """

    # Class-level cache to share across engine instances
    _cache: Optional[list[dict]] = None
    _cache_time: float = 0.0
    _last_browser_launch: float = 0.0
    _browser_cooldown: float = 15.0  # Min seconds between launches

    @classmethod
    def scrape(cls, timeout: int = 30) -> list[dict]:
        """
        Top-level entry point. Returns NBA odds in TheOddsAPI format.

        Strategy:
          1. Check in-memory cache (fresh within ODDS_CACHE_TTL)
          2. Try direct HTTP call (fails on most networks — Cloudflare)
          3. Fall back to Playwright headless browser (more reliable)
          4. Cache and return results

        Args:
            timeout: Max seconds for the entire operation

        Returns:
            List of game dicts in TheOddsAPI format, or empty list if unavailable
        """
        now = time.time()

        # Check cache
        if cls._cache is not None and (now - cls._cache_time) < ODDS_CACHE_TTL:
            logger.debug("DraftKingsScraper: Returning cached odds")
            return cls._cache

        # Tier 1: Try direct HTTP (fast, usually blocked)
        games = cls._scrape_via_http(timeout=timeout)
        if games:
            logger.info(f"DraftKingsScraper (HTTP): {len(games)} games")
            cls._cache = games
            cls._cache_time = now
            return games

        # Cooldown check for browser launches
        if (now - cls._last_browser_launch) < cls._browser_cooldown:
            logger.debug("DraftKingsScraper: Browser cooldown active")
            return cls._cache or []

        # Tier 2: Playwright headless browser
        logger.info("DraftKingsScraper: Trying Playwright...")
        games = cls._scrape_via_playwright(timeout=timeout)
        cls._last_browser_launch = time.time()

        if games:
            logger.info(f"DraftKingsScraper (Playwright): {len(games)} games")
            cls._cache = games
            cls._cache_time = time.time()
            return games

        logger.warning("DraftKingsScraper: All methods failed")
        cls._cache = []
        cls._cache_time = time.time()
        return []

    # ── Tier 1: Direct HTTP ───────────────────────────────────────────────

    @classmethod
    def _scrape_via_http(cls, timeout: int = 15) -> list[dict]:
        """
        Try direct HTTP call to DraftKings API.

        Usually blocked by Cloudflare. Included as a fast-path attempt
        in case the network allows it.
        """
        import urllib.request
        import urllib.error

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Referer": "https://sportsbook.draftkings.com/basketball/nba",
            "Origin": "https://sportsbook.draftkings.com",
            "Accept": "application/json",
        }

        for event_id in NBA_EVENT_GROUP_IDS:
            try:
                url = (
                    f"{SPORTSBOOK_BASE}/sites/US-DK/api/v5/eventgroups/"
                    f"{event_id}?format=json"
                )
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    games = cls._parse_dk_response(data)
                    if games:
                        return games
            except (urllib.error.HTTPError, urllib.error.URLError,
                    json.JSONDecodeError, TimeoutError) as e:
                logger.debug(f"DraftKings HTTP {event_id}: {e}")
                continue
            except Exception as e:
                logger.debug(f"DraftKings HTTP {event_id}: unexpected {e}")
                continue

        return []

    # ── Tier 2: Playwright ─────────────────────────────────────────────────

    @classmethod
    def _scrape_via_playwright(cls, timeout: int = 30) -> list[dict]:
        """
        Launch headless Chromium and intercept DraftKings API responses.

        This is the primary method. Playwright with stealth should bypass
        Cloudflare and regional blocks by looking like a real browser.
        """
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",  # Needed for CORS on intercepted calls
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
                    geolocation={"latitude": 40.7128, "longitude": -74.0060},
                    permissions=["geolocation"],
                    storage_state=None,
                )

                page = context.new_page()

                # Apply stealth to hide automation
                try:
                    from playwright_stealth import Stealth
                    Stealth(page)
                    logger.debug("Stealth applied to Playwright page")
                except ImportError:
                    logger.debug("playwright-stealth not available")

                # Collect API responses we care about
                api_responses: list[dict] = []

                def handle_response(response):
                    url = response.url
                    if "eventgroups" in url and "format=json" in url:
                        try:
                            data = response.json()
                            if isinstance(data, dict) and cls._is_valid_dk_response(data):
                                api_responses.append(data)
                                logger.debug(f"Captured DK API response: {url}")
                        except Exception:
                            pass

                page.on("response", handle_response)

                # Navigate to DraftKings NBA page
                logger.debug("Navigating to DraftKings NBA...")
                try:
                    page.goto(
                        NBA_SPORT_URL,
                        wait_until="domcontentloaded",
                        timeout=timeout * 1000,
                    )
                except Exception as e:
                    logger.debug(f"DraftKings page load failed: {e}")
                    # Try direct API fetch via page.evaluate as fallback
                    # Use async IIFE so the fetch Promise is properly awaited by Playwright
                    try:
                        for event_id in NBA_EVENT_GROUP_IDS:
                            api_url = (
                                f"{SPORTSBOOK_BASE}/sites/US-DK/api/v5/eventgroups/"
                                f"{event_id}?format=json"
                            )
                            js = (
                                f"(async () => {{"
                                f"const r = await fetch('{api_url}', "
                                f"{{headers: {{'Accept': 'application/json'}}}}); "
                                f"return await r.json(); "
                                f"}})()"
                            )
                            result = page.evaluate(js)
                            if result and cls._is_valid_dk_response(result):
                                api_responses.append(result)
                                break
                    except Exception as e2:
                        logger.debug(f"DraftKings fetch fallback failed: {e2}")

                # Wait for API responses to arrive
                page.wait_for_timeout(8000)

                browser.close()

                # Parse all collected API responses
                for data in api_responses:
                    games = cls._parse_dk_response(data)
                    if games:
                        return games

                logger.warning("DraftKings Playwright: No valid API responses captured")
                return []

        except ImportError as e:
            logger.warning(f"DraftKings Playwright not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"DraftKings Playwright scrape failed: {e}")
            return []

    # ── Response Parsing ───────────────────────────────────────────────────

    @classmethod
    def _is_valid_dk_response(cls, data: dict) -> bool:
        """Check if the response looks like a valid DraftKings event group response."""
        return (
            isinstance(data, dict)
            and "eventGroup" in data
            and isinstance(data["eventGroup"], dict)
        )

    @classmethod
    def _parse_dk_response(cls, data: dict) -> list[dict]:
        """
        Parse DraftKings event group response into TheOddsAPI format.

        DraftKings structure:
        {
          "eventGroup": {
            "eventGroupId": 42648,
            "name": "NBA",
            "events": [
              {
                "eventId": 12345,
                "name": "Team A @ Team B",
                "eventStartDate": "2026-06-11T00:30:00Z",
                "offerCategories": [
                  {
                    "offerCategoryId": ...,
                    "name": "...",
                    "offerSubcategoryDescriptors": [
                      {
                        "offerSubcategory": {
                          "offers": [
                            {
                              "offerId": ...,
                              "outcomes": [
                                {"label": "Team A", "oddsAmerican": -125, ...},
                                {"label": "Team B", "oddsAmerican": 105, ...}
                              ]
                            }
                          ]
                        }
                      }
                    ]
                  }
                ]
              }
            ]
          }
        }

        Returns:
            List of game dicts in TheOddsAPI format
        """
        event_group = data.get("eventGroup", {})
        events = event_group.get("events", [])
        if not events:
            return []

        parsed_games: list[dict] = []
        now_utc = datetime.now(timezone.utc)

        for event in events:
            try:
                game = cls._parse_dk_event(event)
                if game:
                    parsed_games.append(game)
            except Exception as e:
                logger.debug(f"Skipping malformed DK event: {e}")
                continue

        if parsed_games:
            logger.info(f"DraftKings parsed: {len(parsed_games)} games with odds")

        return parsed_games

    @classmethod
    def _parse_dk_event(cls, event: dict) -> Optional[dict]:
        """
        Parse a single DraftKings event into TheOddsAPI format.

        Extracts team names from the event name ("Team A @ Team B"),
        and parses all available markets (moneyline, spread, totals).
        """
        event_id = event.get("eventId", "")
        name = event.get("name", "")
        start_date = event.get("eventStartDate", "")

        if not name or " @" not in name:
            return None

        # Split team names: "New York Knicks @ San Antonio Spurs"
        parts = name.split(" @ ", 1)
        if len(parts) != 2:
            return None

        away_full = parts[0].strip()
        home_full = parts[1].strip()

        # Normalize team names using our shared mapping
        home_short = ODDS_TO_SHORT_NAME.get(home_full)
        away_short = ODDS_TO_SHORT_NAME.get(away_full)

        # If team names don't match our mapping, try partial match
        if not home_short:
            home_short = cls._fuzzy_team_match(home_full)
        if not away_short:
            away_short = cls._fuzzy_team_match(away_full)

        # Use canonical full names if we recognized them
        if home_short:
            home_full = SHORT_TO_ODDS_NAME.get(home_short, home_full)
        if away_short:
            away_full = SHORT_TO_ODDS_NAME.get(away_short, away_full)

        # Parse markets/odds from the offer categories
        bookmaker = cls._extract_dk_bookmaker(event, home_full, away_full)

        game = {
            "id": f"dk_nba_{event_id}",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": start_date,
            "home_team": home_full,
            "away_team": away_full,
            "bookmakers": [bookmaker] if bookmaker else [],
        }

        return game if bookmaker else None

    @classmethod
    def _extract_dk_bookmaker(
        cls, event: dict, home_full: str, away_full: str
    ) -> Optional[dict]:
        """
        Extract odds from DraftKings event structure.

        Navigates the nested DraftKings structure to find:
          - Moneyline (h2h)
          - Point Spread (spreads)
          - Total Points (totals)
        """
        offer_categories = event.get("offerCategories", [])
        if not offer_categories:
            return None

        h2h_outcomes: list[dict] = []
        spread_outcomes: list[dict] = []
        totals_outcomes: list[dict] = []

        for category in offer_categories:
            subcategory_descriptors = category.get(
                "offerSubcategoryDescriptors", []
            )
            for descriptor in subcategory_descriptors:
                subcategory = descriptor.get("offerSubcategory", {})
                offers = cls._extract_offers(subcategory)
                market_name = category.get("name", "")
                market_key = DK_MARKET_MAP.get(market_name, "")

                for offer in offers:
                    outcomes = offer.get("outcomes", [])

                    if market_key == "h2h":
                        parsed = cls._parse_h2h_outcomes(
                            outcomes, home_full, away_full
                        )
                        if parsed:
                            h2h_outcomes.extend(parsed)

                    elif market_key == "spreads":
                        parsed = cls._parse_spread_outcomes(
                            outcomes, home_full, away_full
                        )
                        if parsed:
                            spread_outcomes.extend(parsed)

                    elif market_key == "totals":
                        parsed = cls._parse_totals_outcomes(outcomes)
                        if parsed:
                            totals_outcomes.extend(parsed)

        # Deduplicate outcomes (DraftKings sometimes returns duplicates)
        h2h_outcomes = cls._deduplicate_outcomes(h2h_outcomes)
        spread_outcomes = cls._deduplicate_outcomes(spread_outcomes)
        totals_outcomes = cls._deduplicate_outcomes(totals_outcomes)

        markets = []
        if h2h_outcomes:
            markets.append({"key": "h2h", "outcomes": h2h_outcomes})
        if spread_outcomes:
            markets.append({"key": "spreads", "outcomes": spread_outcomes})
        if totals_outcomes:
            markets.append({"key": "totals", "outcomes": totals_outcomes})

        if not markets:
            return None

        return {
            "key": "draftkings",
            "title": "DraftKings",
            "last_update": datetime.now().isoformat(),
            "markets": markets,
        }

    @classmethod
    def _extract_offers(cls, subcategory: dict) -> list[dict]:
        """Extract offers from a DraftKings subcategory, handling nesting."""
        offers = subcategory.get("offers", [])
        if not offers:
            return []

        # DraftKings nests offers: list of lists
        all_offers: list[dict] = []
        for grouping in offers:
            if isinstance(grouping, list):
                all_offers.extend(grouping)
            elif isinstance(grouping, dict):
                all_offers.append(grouping)
        return all_offers

    @classmethod
    def _parse_h2h_outcomes(
        cls, outcomes: list[dict], home_full: str, away_full: str
    ) -> list[dict]:
        """Parse DraftKings moneyline outcomes into TheOddsAPI h2h format."""
        parsed = []
        for o in outcomes:
            label = o.get("label", "")
            odds = o.get("oddsAmerican") or o.get("oddsDecimal")
            if not odds:
                continue
            price = cls._to_american_odds(odds)
            if not cls._is_reasonable_odds(price):
                continue

            # Match to home or away team
            team_name = cls._match_team_name(label, home_full, away_full)
            if team_name:
                parsed.append({"name": team_name, "price": price})
        return parsed

    @classmethod
    def _parse_spread_outcomes(
        cls, outcomes: list[dict], home_full: str, away_full: str
    ) -> list[dict]:
        """Parse DraftKings spread outcomes into TheOddsAPI spreads format."""
        parsed = []
        for o in outcomes:
            label = o.get("label", "")
            # DraftKings spread label: "Team -3.5" or "Team +3.5"
            point = o.get("point") or o.get("spread")

            # If point not directly available, parse from label
            if point is None:
                import re
                # Use split-based approach for team name extraction instead of
                # regex sub, to handle edge cases like "76ers -3.5" where the
                # team name contains digits
                match = re.search(r"([+-]?\d+\.?\d*)$", label.strip())
                if match:
                    try:
                        point = float(match.group(1))
                    except ValueError:
                        continue
                    # Get team name by removing the trailing spread value
                    # Use the stripped version for position matching
                    stripped_label = label.strip()
                    team_label = stripped_label[:match.start()].strip().rstrip(",;")
                else:
                    team_label = label.strip()
            else:
                team_label = label.strip()

            odds = o.get("oddsAmerican") or o.get("oddsDecimal")
            if point is None or not odds:
                continue

            price = cls._to_american_odds(odds)
            team_name = cls._match_team_name(team_label, home_full, away_full)
            if team_name and cls._is_reasonable_odds(price):
                parsed.append({
                    "name": team_name,
                    "point": float(point),
                    "price": price,
                })
        return parsed

    @classmethod
    def _parse_totals_outcomes(cls, outcomes: list[dict]) -> list[dict]:
        """Parse DraftKings totals outcomes into TheOddsAPI format.

        DraftKings puts the point total in the label ("Over 224.5")
        rather than in a separate field. This method extracts it from
        the label if not provided directly.
        """
        import re

        over = None
        under = None

        for o in outcomes:
            label = o.get("label", "").lower()
            point = o.get("point") or o.get("line")
            odds = o.get("oddsAmerican") or o.get("oddsDecimal")
            if not odds:
                continue

            price = cls._to_american_odds(odds)
            if not cls._is_reasonable_odds(price):
                continue

            # Extract point from label if not directly provided
            # Label format: "Over 224.5", "Under 218.5", "O 224.5", "U 218.5"
            if point is None:
                match = re.search(r"(\d+\.?\d*)\s*$", label)
                if match:
                    try:
                        point = float(match.group(1))
                    except ValueError:
                        continue

            if point is None:
                continue

            if label.startswith("over") or label.startswith("o"):
                over = {"name": "Over", "point": float(point), "price": price}
            elif label.startswith("under") or label.startswith("u"):
                under = {"name": "Under", "point": float(point), "price": price}

        result = []
        if over:
            result.append(over)
        if under:
            result.append(under)
        return result

    @classmethod
    def _match_team_name(
        cls, label: str, home_full: str, away_full: str
    ) -> Optional[str]:
        """
        Match a DraftKings team label to canonical team name.

        DraftKings displays team names like "Knicks" or "Spurs" (short names),
        but we need to map them to full names for the engine's team matching.
        """
        label_clean = label.strip().lower()

        # Direct check against full names
        if home_full.lower() == label_clean:
            return home_full
        if away_full.lower() == label_clean:
            return away_full

        # Check against short names
        home_short = ODDS_TO_SHORT_NAME.get(home_full, "").lower()
        away_short = ODDS_TO_SHORT_NAME.get(away_full, "").lower()

        if home_short and home_short == label_clean:
            return home_full
        if away_short and away_short == label_clean:
            return away_full

        # Check if label contains the team's city name
        for full_name in [home_full, away_full]:
            city = full_name.split()[-1].lower()  # Last word (usually city)
            if city and (city == label_clean or label_clean.endswith(city)):
                return full_name

        # Fuzzy: check if the team's short name is contained in the label
        for short_name, full_name in [
            (ODDS_TO_SHORT_NAME.get(home_full, "").lower(), home_full),
            (ODDS_TO_SHORT_NAME.get(away_full, "").lower(), away_full),
        ]:
            if short_name and short_name in label_clean:
                return full_name

        logger.debug(f"DraftKings: Could not match team label '{label}'")
        return None

    @classmethod
    def _fuzzy_team_match(cls, team_name: str) -> Optional[str]:
        """
        Try to match an unknown team name to our known team short names.

        Used as fallback when DraftKings uses non-standard team names.
        """
        name_lower = team_name.lower()

        # CheckODDS_TO_SHORT_NAME for partial matches
        for full_name, short_name in ODDS_TO_SHORT_NAME.items():
            if full_name.lower() in name_lower or name_lower in full_name.lower():
                return short_name

        return None

    @classmethod
    def _to_american_odds(cls, odds: Any) -> Optional[int]:
        """
        Convert various odds formats to American odds.

        DraftKings may return odds as:
          - American: -110, +150 (integer)
          - Decimal: 1.91, 2.20 (float, between 1.01 and ~10)

        Detection logic:
          - Values between 1.01 and 10.0 → decimal odds → convert to American
          - Everything else → already American odds → return as-is
        """
        if odds is None:
            return None
        try:
            val = float(odds)

            # Decimal odds are always between 1.01 and ~10.0
            if 1.01 <= val <= 10.0:
                # Decimal format: convert to American
                if val >= 2.0:
                    # Underdog: (decimal - 1) * 100 = +American
                    return int(round((val - 1) * 100))
                else:
                    # Favorite: -100 / (decimal - 1) = -American
                    return int(round(-100 / (val - 1)))

            # Already American format (or 0, which is invalid but pass through)
            return int(round(val))
        except (ValueError, TypeError):
            return None

    @classmethod
    def _is_reasonable_odds(cls, odds: Optional[int]) -> bool:
        """
        Validate odds are within a reasonable range.
        Catches data errors where odds are 0 or absurdly high.
        """
        if odds is None:
            return False
        return MIN_REASONABLE_ODDS <= odds <= MAX_REASONABLE_ODDS

    @staticmethod
    def _deduplicate_outcomes(outcomes: list[dict]) -> list[dict]:
        """Remove duplicate outcomes with the same name."""
        seen: set[str] = set()
        unique: list[dict] = []
        for o in outcomes:
            name = o.get("name", "")
            if name and name not in seen:
                seen.add(name)
                unique.append(o)
        return unique

    @classmethod
    def clear_cache(cls):
        """Clear the in-memory cache."""
        cls._cache = None
        cls._cache_time = 0.0
