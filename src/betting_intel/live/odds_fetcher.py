"""
OddsFetcher — fetches real-time odds from multiple sources.

Handles:
  - TheOddsAPI (primary, requires API key)
  - ESPN stealth scraper (free fallback)
  - DraftKings scraper (free fallback)
  - Parallel execution of scrapers with configurable timeout
  - Merging odds from multiple sources

Thread-safety note: the caller (LivePredictionEngine) manages the shared
odds cache lock. This class is stateless with respect to the cache.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from threading import Lock
from typing import Any, Optional

from betting_intel.live.models import ODDS_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


# ── Invalid key sentinels ────────────────────────────────────────────────

_INVALID_KEYS = frozenset({"your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"})


# ── Fetcher ───────────────────────────────────────────────────────────────

class OddsFetcher:
    """Fetches and merges real-time odds from TheOddsAPI and free scrapers."""

    def __init__(
        self,
        odds_api_key: str,
        odds_api_key_fallback: str,
        scraper_timeout: float = 15.0,
    ):
        self._odds_api_key = odds_api_key
        self._odds_api_key_fallback = odds_api_key_fallback
        self._scraper_timeout = scraper_timeout

    # ── Public API ────────────────────────────────────────────────────────

    def has_valid_api_key(self) -> bool:
        primary_valid = (
            bool(self._odds_api_key)
            and self._odds_api_key not in _INVALID_KEYS
        )
        fallback_valid = (
            bool(self._odds_api_key_fallback)
            and self._odds_api_key_fallback not in _INVALID_KEYS
        )
        return primary_valid or fallback_valid

    def fetch(
        self,
        cached_odds_raw: Optional[list[dict]],
        last_odds_fetch: float,
        cache_lock: Lock,
        now: float,
    ) -> list[dict]:
        """
        Fetch fresh odds, respecting the cache TTL.

        Thread-safe: reads/writes the cache under *cache_lock*.

        Returns:
            List of raw odds dicts (may be empty).
        """
        # Fast-path: return cached odds if still fresh
        with cache_lock:
            if cached_odds_raw is not None and (now - last_odds_fetch) < ODDS_CACHE_TTL_SECONDS:
                return cached_odds_raw

        if self.has_valid_api_key():
            logger.info("Valid ODDS_API_KEY found — trying TheOddsAPI first")
            theodds_data = self._fetch_via_theoddsapi()
            if theodds_data:
                return theodds_data
            logger.info("TheOddsAPI returned no data — trying free scrapers")
        else:
            logger.info("No valid ODDS_API_KEY — using free scrapers")

        # Free scrapers run in parallel
        merged = self._fetch_from_scrapers()

        with cache_lock:
            return merged

    # ── TheOddsAPI ────────────────────────────────────────────────────────

    def _fetch_via_theoddsapi(self) -> list[dict]:
        """Try primary key, then fallback key."""
        if not self.has_valid_api_key():
            return []

        from betting_intel.live.sport_configs import get_active_sports, SPORT_KEY_TO_CONFIG, SportConfig

        active_sports: list[SportConfig] = get_active_sports()
        if not active_sports:
            nba_config = SPORT_KEY_TO_CONFIG.get("basketball_nba")
            if nba_config is None:
                return []
            active_sports = [nba_config]

        # Primary key
        result = self._fetch_via_theoddsapi_with_key(self._odds_api_key, active_sports)

        # Fallback key if primary returned nothing
        if not result and self._odds_api_key_fallback:
            logger.warning("Primary ODDS_API_KEY returned no data — trying fallback key")
            key_label = self._odds_api_key_fallback[:8] + "..."
            logger.info(f"Using fallback key: {key_label}")
            result = self._fetch_via_theoddsapi_with_key(self._odds_api_key_fallback, active_sports)

        return result

    def _fetch_via_theoddsapi_with_key(self, api_key: str, active_sports: list) -> list[dict]:
        """Fetch odds from TheOddsAPI using a specific API key."""
        import urllib.request
        import urllib.error

        all_games: list[dict] = []
        total_quota = "?"
        key_label = api_key[:8] + "..." if len(api_key) > 8 else api_key

        for sport in active_sports:
            try:
                markets_str = ",".join(sport.markets_to_fetch)
                url = (
                    f"https://api.the-odds-api.com/v4/sports/{sport.sport_key}/odds"
                    f"?apiKey={api_key}"
                    f"&regions=us,us2,eu,uk,au"
                    f"&markets={markets_str}"
                    f"&oddsFormat=american"
                    f"&dateFormat=iso"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "betting-intel-live/3.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                if isinstance(data, list) and len(data) > 0:
                    for game in data:
                        game["_sport_config_key"] = sport.sport_key
                    all_games.extend(data)
                    remaining = resp.headers.get("x-requests-remaining", "?")
                    if remaining != "?":
                        total_quota = remaining
                    logger.info(f"{sport.display_name}: {len(data)} games (quota: {remaining}, key: {key_label})")
                else:
                    logger.info(f"{sport.display_name}: no games available")

            except urllib.error.HTTPError as e:
                if e.code == 401:
                    logger.warning(f"ODDS_API_KEY ({key_label}) returned 401 — invalid key")
                    return []
                if e.code == 429:
                    logger.warning(f"TheOddsAPI quota exceeded (429) — key: {key_label}")
                    break
                logger.debug(f"{sport.display_name}: HTTP {e.code} (skipping)")
                continue
            except urllib.error.URLError as e:
                logger.debug(f"{sport.display_name}: connection failed ({e})")
                continue
            except Exception as e:
                logger.debug(f"{sport.display_name}: error ({e})")
                continue

        logger.info(f"TheOddsAPI ({key_label}) total: {len(all_games)} games across {len(active_sports)} sports")
        return all_games

    # ── Free Scrapers ─────────────────────────────────────────────────────

    def _fetch_from_scrapers(self) -> list[dict]:
        """
        Run ESPN and DraftKings scrapers CONCURRENTLY with a single combined timeout.

        Previously they ran sequentially: ESPN (25s) then DraftKings (30s) = up to 55s
        blocking the entire engine. Now both run in parallel within ~15s total.
        """
        scraper_timeout = self._scraper_timeout

        scraper_results: dict[str, list[dict]] = {}
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            espn_future = pool.submit(self._fetch_stealth_scraper)
            dk_future = pool.submit(self._fetch_draftkings_odds)
            processed: set[str] = set()

            try:
                for future in as_completed([espn_future, dk_future], timeout=scraper_timeout):
                    try:
                        data = future.result()
                    except Exception as e:
                        logger.debug(f"Scraper result error: {e}")
                        continue
                    key = "espn" if future == espn_future else "dk"
                    scraper_results[key] = data if data else []
                    processed.add(key)
                    logger.info(f"{'ESPN' if key == 'espn' else 'DraftKings'} scraper: {len(data) if data else 0} games")
            except FuturesTimeoutError:
                logger.warning(f"Scrapers timed out after {scraper_timeout}s — using partial results")

            # Collect late-finishing futures
            for future, key, name in [
                (espn_future, "espn", "ESPN"),
                (dk_future, "dk", "DraftKings"),
            ]:
                if key in processed:
                    continue
                if future.done():
                    try:
                        data = future.result()
                        scraper_results[key] = data if data else []
                        logger.info(f"{name} scraper (late): {len(data) if data else 0} games")
                    except Exception as e:
                        logger.debug(f"{name} scraper late result error: {e}")
                        scraper_results.setdefault(key, [])
                else:
                    logger.warning(f"{name} scraper did not finish — results discarded")
                    scraper_results.setdefault(key, [])
        finally:
            pool.shutdown(wait=False)

        stealth_data = scraper_results.get("espn", [])
        dk_data = scraper_results.get("dk", [])

        return self.merge_odds_sources(stealth_data, dk_data)

    def _fetch_stealth_scraper(self) -> list[dict]:
        """Fetch odds from ESPN via Playwright-based stealth scraper."""
        logger.info("Attempting ESPN stealth scraper...")
        try:
            from betting_intel.data.stealth_scraper import StealthBrowser
            scraped = StealthBrowser.sync_scrape_live_odds(
                odds_api_key=self._odds_api_key,
                timeout=12,
            )
            if scraped:
                logger.info(f"ESPN stealth scraper: {len(scraped)} games")
                return scraped
            logger.info("ESPN stealth scraper: returned no data")
            return []
        except ImportError as e:
            logger.debug(f"ESPN stealth scraper: not available ({e})")
            return []
        except Exception as e:
            logger.warning(f"ESPN stealth scraper failed: {e}")
            return []

    @staticmethod
    def _fetch_draftkings_odds() -> list[dict]:
        """Fetch odds from DraftKings via API scraper."""
        logger.info("Attempting DraftKings scraper...")
        try:
            from betting_intel.data.draftkings_scraper import DraftKingsScraper
            scraped = DraftKingsScraper.scrape(timeout=12)
            if scraped:
                logger.info(f"DraftKings scraper: {len(scraped)} games")
                return scraped
            logger.info("DraftKings scraper: returned no data")
            return []
        except ImportError as e:
            logger.debug(f"DraftKings scraper: not available ({e})")
            return []
        except Exception as e:
            logger.warning(f"DraftKings scraper failed: {e}")
            return []

    # ── Merging ───────────────────────────────────────────────────────────

    @staticmethod
    def merge_odds_sources(espn_data: list[dict], dk_data: list[dict]) -> list[dict]:
        """Merge odds from ESPN and DraftKings, combining bookmakers for same matchups."""
        if not espn_data and not dk_data:
            return []
        if not dk_data:
            return espn_data
        if not espn_data:
            return dk_data

        dk_by_matchup: dict[str, list[dict]] = {}
        for dk_game in dk_data:
            home = dk_game.get("home_team", "")
            away = dk_game.get("away_team", "")
            if home and away:
                key = f"{home}|{away}"
                if key not in dk_by_matchup:
                    dk_by_matchup[key] = []
                dk_by_matchup[key].append(dk_game)

        merged: list[dict] = []
        seen_matchups: set[str] = set()

        for espn_game in espn_data:
            home = espn_game.get("home_team", "")
            away = espn_game.get("away_team", "")
            key = f"{home}|{away}"
            seen_matchups.add(key)
            merged_game = dict(espn_game)
            merged_books = list(merged_game.get("bookmakers", []))
            dk_games = dk_by_matchup.get(key, [])
            for dk_game in dk_games:
                dk_books = dk_game.get("bookmakers", [])
                merged_books.extend(dk_books)
            merged_game["bookmakers"] = merged_books
            merged.append(merged_game)

        for dk_game in dk_data:
            home = dk_game.get("home_team", "")
            away = dk_game.get("away_team", "")
            key = f"{home}|{away}"
            if key not in seen_matchups:
                merged.append(dk_game)
                seen_matchups.add(key)

        logger.info(
            f"Merged odds: {len(merged)} total games "
            f"({len(espn_data)} from ESPN, {len(dk_data)} from DraftKings)"
        )
        return merged
