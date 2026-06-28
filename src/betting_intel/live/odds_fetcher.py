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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime
from threading import Lock
from typing import Any, Optional

from betting_intel.live.models import ODDS_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


# ── Invalid key sentinels ────────────────────────────────────────────────

_INVALID_KEYS = frozenset({"your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"})


# ── Quota warning thresholds ────────────────────────────────────────────
# TheOddsAPI free tier gives 500 credits/month. Each API call costs
# markets × regions credits per sport. With regions=us and 3 markets,
# each sport call costs 3 credits. Track remaining and warn at thresholds.
_QUOTA_WARN_LEVELS = [100, 50, 25, 10, 5, 1]


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
        self._last_theoddsapi_fetch: float = 0.0  # When TheOddsAPI was last called
        self._last_quota_remaining: Optional[str] = None  # x-requests-remaining from last call
        self._last_quota_used: Optional[str] = None       # x-requests-used from last call
        self._last_quota_credits_cost: Optional[int] = None  # Estimated credits spent on last call
        self._quota_has_warned: set[int] = set()  # Track which threshold levels we've warned at

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

    def _check_quota_warnings(self, remaining: int):
        """Log warnings when quota drops below thresholds."""
        if not isinstance(remaining, (int, float)) or remaining < 0:
            return
        for level in _QUOTA_WARN_LEVELS:
            if remaining <= level and level not in self._quota_has_warned:
                self._quota_has_warned.add(level)
                if level <= 10:
                    logger.warning(
                        f"TheOddsAPI quota critically low: {remaining} credits remaining! "
                        f"Consider reducing refresh frequency or upgrading plan."
                    )
                elif level <= 50:
                    logger.warning(
                        f"TheOddsAPI quota running low: {remaining} credits remaining. "
                        f"Check usage at https://the-odds-api.com/manage"
                    )
                else:
                    logger.info(
                        f"TheOddsAPI quota: {remaining} credits remaining "
                        f"({(remaining / 500.0) * 100:.0f}% of monthly free tier)"
                    )

    @property
    def quota_summary(self) -> dict:
        """Return a summary of TheOddsAPI quota status for the dashboard."""
        remaining = self._last_quota_remaining
        remaining_int = None
        try:
            if remaining and remaining != "?":
                remaining_int = int(remaining)
        except (ValueError, TypeError):
            remaining_int = None

        used = self._last_quota_used
        used_int = None
        try:
            if used and used != "?":
                used_int = int(used)
        except (ValueError, TypeError):
            used_int = None

        return {
            "remaining": remaining_int,
            "used": used_int,
            "credits_cost": self._last_quota_credits_cost,
            "has_valid_key": self.has_valid_api_key(),
            "last_fetch": self._last_theoddsapi_fetch,
        }

    # ── Daily Schedule ────────────────────────────────────────────────

    @staticmethod
    def _get_daily_fetch_today() -> float:
        """
        Get the scheduled fetch time for today (in epoch seconds).

        Uses DAILY_FETCH_HOUR from config. The daily fetch triggers at or
        after this hour. For example, if hour=6, the scheduler considers
        any time after 6:00 AM as "today's fetch window".
        """
        from betting_intel.config import DAILY_FETCH_HOUR
        now = datetime.now()
        # Scheduled time: today at DAILY_FETCH_HOUR:00
        scheduled = now.replace(hour=DAILY_FETCH_HOUR, minute=0, second=0, microsecond=0)
        return scheduled.timestamp()

    @staticmethod
    def _is_time_for_daily_fetch(last_fetch: float) -> bool:
        """
        Check if it's time for the daily morning fetch.

        Rules:
          1. Last fetch was before today's scheduled hour (e.g. 6:00 AM)
             → yes, we need today's fetch.
          2. Last fetch was AFTER today's scheduled hour
             → already fetched today, skip.
          3. No previous fetch (last_fetch == 0)
             → yes, we need the first fetch.
          4. Last fetch was yesterday or earlier AND it's past today's hour
             → yes, time for today's fetch.
        """
        from betting_intel.config import DAILY_FETCH_ENABLED

        if not DAILY_FETCH_ENABLED:
            # Schedule disabled: only manual/force_refresh will trigger API calls
            return False

        now = time.time()
        if last_fetch == 0.0:
            # Never fetched before — first fetch is allowed anytime
            return True

        # Get today's scheduled morning time
        scheduled_today = self._get_daily_fetch_today()

        # If we last fetched before today's scheduled time, and it's now
        # past that time, we should fetch.
        return last_fetch < scheduled_today and now >= scheduled_today

    @staticmethod
    def _schedule_summary(last_fetch: float) -> dict:
        """
        Return a human-readable summary of the fetch schedule.

        Returns dict with:
          - next_fetch_at: epoch seconds when next daily fetch will trigger
          - next_fetch_in_seconds: seconds until next fetch
          - next_fetch_display: human-readable "today at 6:00 AM" etc.
          - last_fetch_display: human-readable "today at 6:02 AM" etc.
          - daily_fetch_hour: the configured hour
          - daily_fetch_enabled: whether the schedule is active
        """
        from betting_intel.config import DAILY_FETCH_HOUR, DAILY_FETCH_ENABLED

        today_scheduled = datetime.fromtimestamp(
            OddsFetcher._get_daily_fetch_today()
        )
        now = datetime.now()

        # If already past today's schedule, next is tomorrow
        if now >= today_scheduled:
            from datetime import timedelta
            tomorrow = now + timedelta(days=1)
            next_fetch = tomorrow.replace(hour=DAILY_FETCH_HOUR, minute=0, second=0, microsecond=0)
        else:
            next_fetch = today_scheduled

        # Build last fetch display
        if last_fetch == 0.0:
            last_display = "never"
        else:
            last_dt = datetime.fromtimestamp(last_fetch)
            hours_ago = (now - last_dt).total_seconds() / 3600
            if hours_ago < 1:
                mins = int((now - last_dt).total_seconds() / 60)
                last_display = f"{mins}m ago" if mins > 0 else "just now"
            elif hours_ago < 24:
                last_display = f"{int(hours_ago)}h ago at {last_dt.strftime('%H:%M')}"
            else:
                days = int(hours_ago / 24)
                last_display = f"{days}d ago on {last_dt.strftime('%m/%d')}"

        next_display = next_fetch.strftime("today at %H:%M") if next_fetch.date() == now.date() else next_fetch.strftime("tomorrow at %H:%M")
        seconds_until = (next_fetch - now).total_seconds()

        return {
            "daily_fetch_enabled": DAILY_FETCH_ENABLED,
            "daily_fetch_hour": DAILY_FETCH_HOUR,
            "last_fetch": last_fetch,
            "last_fetch_display": last_display,
            "next_fetch_at": next_fetch.timestamp(),
            "next_fetch_in_seconds": max(0.0, seconds_until),
            "next_fetch_display": next_display,
        }

    # ── Quota Warnings ─────────────────────────────────────────────────

    @staticmethod
    def _get_schedule_status_dict(last_fetch: float) -> dict:
        """Get schedule status as a dict for the dashboard."""
        s = OddsFetcher._schedule_summary(last_fetch)
        now_time = datetime.now()

        # Determine if we SHOULD fetch right now
        should_fetch = s["daily_fetch_enabled"] and (
            last_fetch == 0.0
            or last_fetch < datetime.now().replace(
                hour=s["daily_fetch_hour"], minute=0, second=0, microsecond=0
            ).timestamp()
        ) and now_time.hour >= s["daily_fetch_hour"]

        return {
            "enabled": s["daily_fetch_enabled"],
            "hour": s["daily_fetch_hour"],
            "last_fetch": s["last_fetch_display"],
            "next_fetch": s["next_fetch_display"],
            "next_fetch_in_seconds": s["next_fetch_in_seconds"],
            "should_fetch_now": should_fetch,
        }

    def fetch(
        self,
        cached_odds_raw: Optional[list[dict]],
        last_odds_fetch: float,
        cache_lock: Lock,
        now: float,
        force_theoddsapi: bool = False,
    ) -> list[dict]:
        """
        Fetch fresh odds, respecting the cache TTL and daily API schedule.

        Thread-safe: reads/writes the cache under *cache_lock*.

        API key usage is controlled by the daily morning schedule:
          - By default, TheOddsAPI is called ONCE per day (at 6:00 AM)
          - Use force_theoddsapi=True to bypass the schedule (manual refresh)
          - The 5-minute odds cache prevents repeated calls on page loads
          - Free scrapers (ESPN, DraftKings) ALWAYS run as fallback

        Args:
            cached_odds_raw: Previously cached odds (None if no cache)
            last_odds_fetch: Timestamp of last odds fetch
            cache_lock: Lock for thread-safe cache access
            now: Current timestamp
            force_theoddsapi: If True, bypass the daily schedule and call API

        Returns:
            List of raw odds dicts (may be empty).
        """
        # Fast-path: return cached odds if still fresh
        with cache_lock:
            if cached_odds_raw is not None and (now - last_odds_fetch) < ODDS_CACHE_TTL_SECONDS:
                return cached_odds_raw

        # Call TheOddsAPI if:
        #   a) force_theoddsapi is True (manual refresh / force_refresh), OR
        #   b) it's time for the daily morning fetch
        should_call_api = self.has_valid_api_key() and (
            force_theoddsapi or self._is_time_for_daily_fetch(self._last_theoddsapi_fetch)
        )

        if should_call_api:
            schedule_info = self._schedule_summary(self._last_theoddsapi_fetch)
            since_last = schedule_info["last_fetch_display"]
            logger.info(
                f"Daily scheduled fetch: calling TheOddsAPI "
                f"(last call: {since_last}, hour: {schedule_info['daily_fetch_hour']}:00)"
            )
            theodds_data = self._fetch_via_theoddsapi()
            # Always mark as called — even an empty 200 response consumed quota
            self._last_theoddsapi_fetch = now
            if theodds_data:
                return theodds_data
            logger.info("TheOddsAPI returned no data — trying free scrapers")
        elif self.has_valid_api_key():
            # Schedule says don't fetch yet — log the reason
            schedule_info = self._schedule_summary(self._last_theoddsapi_fetch)
            if schedule_info["daily_fetch_enabled"]:
                logger.info(
                    f"Skipping TheOddsAPI — next daily fetch at {schedule_info['next_fetch_display']} "
                    f"(last: {schedule_info['last_fetch_display']}). "
                    f"Use force_refresh or wait for the scheduled hour."
                )
            else:
                logger.info(
                    "Skipping TheOddsAPI — daily fetch is disabled. "
                    "Use force_refresh to call manually."
                )
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

        from betting_intel.live.sport_configs import get_active_sports, ALL_SPORTS, SportConfig

        active_sports: list[SportConfig] = get_active_sports()
        # When no sports are in-season, try ALL supported sports anyway
        # (TheOddsAPI may still return data for offseason leagues)
        if not active_sports:
            logger.info("No in-season sports — trying all supported sports on TheOddsAPI")
            active_sports = list(ALL_SPORTS)

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
        total_used = "?"
        key_label = api_key[:8] + "..." if len(api_key) > 8 else api_key

        for sport in active_sports:
            try:
                markets_str = ",".join(sport.markets_to_fetch)
                url = (
                    f"https://api.the-odds-api.com/v4/sports/{sport.sport_key}/odds"
                    f"?apiKey={api_key}"
                    f"&regions=us"
                    f"&markets={markets_str}"
                    f"&oddsFormat=american"
                    f"&dateFormat=iso"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "betting-intel-live/3.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                remaining = resp.headers.get("x-requests-remaining", "?")
                used = resp.headers.get("x-requests-used", "?")

                if isinstance(data, list) and len(data) > 0:
                    for game in data:
                        game["_sport_config_key"] = sport.sport_key
                    all_games.extend(data)
                    if remaining != "?":
                        total_quota = remaining
                    logger.info(f"{sport.display_name}: {len(data)} games (quota: {remaining}, key: {key_label})")
                else:
                    logger.info(f"{sport.display_name}: no games available")

                # Track quota after each sport call
                if used is not None and used != "?":
                    total_used = used
                self._last_quota_remaining = total_quota if total_quota and total_quota != "?" else self._last_quota_remaining
                if remaining is not None and remaining != "?":
                    try:
                        remaining_int = int(remaining)
                        self._check_quota_warnings(remaining_int)
                    except (ValueError, TypeError):
                        pass

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

        # Store quota info for dashboard display
        self._last_quota_remaining = total_quota if total_quota != "?" else None
        self._last_quota_used = total_used if total_used != "?" else None

        # Estimate credits consumed: regions × markets per sport
        # regions=1 (us), markets=3 (h2h,spreads,totals) = 3 credits/sport
        n_sports_called = len(active_sports)
        # Get markets_str from last sport or default to 3
        first_sport_markets = ",".join(active_sports[0].markets_to_fetch) if active_sports else "h2h,spreads,totals"
        credits_per_sport = 1 * len(first_sport_markets.split(",")) if first_sport_markets else 3
        self._last_quota_credits_cost = n_sports_called * credits_per_sport

        logger.info(f"TheOddsAPI ({key_label}) total: {len(all_games)} games across {len(active_sports)} sports "
                    f"(~{self._last_quota_credits_cost} credits consumed, ~{total_quota} remaining)")
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


# ── Pre-import scrapers on the main thread to avoid import deadlocks ──
# The scrapers are imported lazily inside thread worker functions. If a
# worker thread imports a module that itself imports from odds_fetcher
# (creating a circular import), Python's module lock can deadlock.
#
# Pre-importing here (at module load time, after all symbols are defined)
# ensures the modules are cached in sys.modules before any thread runs.
# The lazy imports in the methods above then just hit the cache.
try:
    from betting_intel.data.stealth_scraper import StealthBrowser  # noqa: F401
except ImportError:
    pass
try:
    from betting_intel.data.draftkings_scraper import DraftKingsScraper  # noqa: F401
except ImportError:
    pass
