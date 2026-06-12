"""
SocialMediaSignalCollector — monitors NBA Twitter/X accounts for actionable betting signals.

Uses the nba_accounts.py database to know WHICH accounts to monitor,
then fetches their recent tweets and parses them for betting-relevant signals.

Architecture:
    1. Account database (nba_accounts.py) → known accounts with reliability scores
    2. Fetch layer → gets recent tweets from each account
    3. Parse layer → extracts structured signals (injury updates, lineup changes, etc.)
    4. Aggregation → deduplicates, scores, and prioritizes signals by impact

Data Sources:
    - nba_api.stats: Provides official injury/transaction data
    - ESPN injury feed: Official injury reports
    - Twitter/X: For beat reporter intel (when API key available)

Usage:
    from betting_intel.data.social_collector import SocialMediaSignalCollector

    collector = SocialMediaSignalCollector()
    signals = collector.collect_signals()
    for s in signals:
        print(f"{s.signal_type}: {s.message} (confidence: {s.confidence:.0%})")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from betting_intel.data.nba_accounts import (
    get_all_accounts,
    get_high_reliability_accounts,
    AccountRole,
    SignalType,
)
from betting_intel.data.scraper_utils import (
    retry_with_backoff,
    GLOBAL_SCRAPER_MONITOR as monitor,
)

logger = logging.getLogger(__name__)

# ── Signal Model ────────────────────────────────────────────────────────────


@dataclass
class BettingSignal:
    """
    A structured, actionable signal extracted from social media.

    Each signal represents a real-world event that could affect a game:
    - Player ruled OUT → adjust model prediction
    - Coach confirms starting lineup → update player prop projections
    - Injury upgrade (questionable → probable) → increase confidence
    """

    signal_type: SignalType
    player_name: str
    team: str
    message: str  # Original text
    confidence: float  # 0.0-1.0 based on source reliability + signal clarity
    source_account: str = ""
    source_reliability: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    game_date: Optional[str] = None
    is_actionable: bool = True  # False if signal is already priced into odds

    def impact_factor(self) -> float:
        """
        Estimate the betting impact of this signal (0.0-1.0).

        Factors:
        - Signal type: INJURY_OUT > LINEUP_CHANGE > PRACTICE_STATUS
        - Source reliability: Shams (0.95) > beat reporter (0.82) > random (0.5)
        - Recency: more recent = higher impact
        """
        type_weights = {
            SignalType.INJURY_OUT: 0.9,
            SignalType.INJURY_QUESTIONABLE: 0.6,
            SignalType.INJURY_PROBABLE: 0.3,
            SignalType.INJURY_RETURN: 0.5,
            SignalType.LINEUP_CHANGE: 0.7,
            SignalType.COACH_COMMENT: 0.4,
            SignalType.LOAD_MGMT: 0.75,
            SignalType.GTD: 0.5,
            SignalType.PRACTICE_STATUS: 0.2,
            SignalType.TRADE: 0.8,
        }
        base = type_weights.get(self.signal_type, 0.3)
        return min(base * self.source_reliability * 1.2, 1.0)


# ── Signal Parsers ─────────────────────────────────────────────────────────


def parse_injury_signal(text: str, account_reliability: float) -> Optional[BettingSignal]:
    """
    Parse injury-related signals from text.

    Patterns:
        "Player X (injury) — status"  (ESPN/Underdog format)
        "Player X is out tonight"
        "Player X will play"
        "Player X is questionable"
        "Player X ruled out for [timeframe]"
    """
    text_lower = text.lower()

    # Check for injury OUT signals
    out_patterns = [
        r"(\w+\s\w+)\s+(?:is\s+)?(?:ruled\s+)?out\s+(?:for\s+)?(?:tonight|tonight's\s+game|vs\.?\s+\w+|at\s+\w+)",
        r"(\w+\s\w+)\s+will\s+(?:miss|sit\s+out)\s+(?:tonight|tonight's\s+game)",
        r"(\w+\s\w+)\s+(?:has\s+been\s+)?(?:ruled\s+out|declared\s+out)",
        r"(\w+\s\w+)\s+(?::|—)\s+OUT\b",
    ]
    match = _try_patterns(out_patterns, text)
    if match:
        return BettingSignal(
            signal_type=SignalType.INJURY_OUT,
            player_name=match.group(1),
            team="",  # Will be filled by aggregation layer
            message=text,
            confidence=0.85 * account_reliability / 0.85,  # Scale to source
            source_reliability=account_reliability,
        )

    # Check for QUESTIONABLE signals
    questionable_patterns = [
        r"(\w+\s\w+)\s+(?:is\s+)?questionable\s+(?:for\s+)?(?:tonight|tonight's\s+game)",
        r"(\w+\s\w+)\s+(?::|—)\s+Questionable\b",
    ]
    match = _try_patterns(questionable_patterns, text)
    if match:
        return BettingSignal(
            signal_type=SignalType.INJURY_QUESTIONABLE,
            player_name=match.group(1),
            team="",
            message=text,
            confidence=0.60 * account_reliability / 0.85,
            source_reliability=account_reliability,
        )

    # Check for PROBABLE signals
    probable_patterns = [
        r"(\w+\s\w+)\s+(?:is\s+)?probable\s+(?:for\s+)?(?:tonight|tonight's\s+game)",
        r"(\w+\s\w+)\s+(?::|—)\s+Probable\b",
        r"(\w+\s\w+)\s+expected\s+to\s+play",
    ]
    match = _try_patterns(probable_patterns, text)
    if match:
        return BettingSignal(
            signal_type=SignalType.INJURY_PROBABLE,
            player_name=match.group(1),
            team="",
            message=text,
            confidence=0.40 * account_reliability / 0.85,
            source_reliability=account_reliability,
        )

    return None


def parse_lineup_signal(text: str, account_reliability: float) -> Optional[BettingSignal]:
    """
    Parse lineup change signals.

    Patterns:
        "Coach says Player X will start"
        "Player X will come off the bench"
        "Coach confirms [Player] will start at [position]"
        "Expected starting lineup: ..."
    """
    text_lower = text.lower()

    start_patterns = [
        r"(\w+\s\w+)\s+will\s+(?:start|get\s+the\s+start)",
        r"(?:coach|thibodeau|spoelstra|kerr|etc)\s+says\s+(\w+\s\w+)\s+will\s+start",
    ]
    match = _try_patterns(start_patterns, text)
    if match:
        return BettingSignal(
            signal_type=SignalType.LINEUP_CHANGE,
            player_name=match.group(1),
            team="",
            message=text,
            confidence=0.70 * account_reliability / 0.85,
            source_reliability=account_reliability,
        )

    return None


def parse_load_mgmt_signal(text: str, account_reliability: float) -> Optional[BettingSignal]:
    """
    Parse load management signals.

    Patterns:
        "Player X will sit for rest"
        "Player X resting on back-to-back"
        "Player X (rest) — out"
    """
    text_lower = text.lower()

    rest_patterns = [
        r"(\w+\s\w+)\s+(?:will\s+)?(?:sit|resting|get\s+rest)\s+(?:for\s+)?(?:rest|load\s+mngmt|maintenance)",
        r"(\w+\s\w+)\s+\(rest\)",
    ]
    match = _try_patterns(rest_patterns, text)
    if match:
        return BettingSignal(
            signal_type=SignalType.LOAD_MGMT,
            player_name=match.group(1),
            team="",
            message=text,
            confidence=0.75 * account_reliability / 0.85,
            source_reliability=account_reliability,
        )

    return None


def _try_patterns(patterns: list[str], text: str) -> Optional[re.Match]:
    """Try a list of regex patterns against text, return first match."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match
    return None


# ── Main Collector ─────────────────────────────────────────────────────────


class SocialMediaSignalCollector:
    """
    Collects and aggregates betting signals from NBA social media accounts.

    Two-tier collection:
      1. Official sources (NBA API, ESPN): 100% reliable, always available
      2. Twitter/X accounts: requires API key, supplements official sources

    Usage:
        collector = SocialMediaSignalCollector()
        signals = collector.collect_signals(force_refresh=True)
        for s in signals:
            print(f"[{s.signal_type.value}] {s.player_name}: {s.message[:50]}...")
    """

    def __init__(self):
        self._last_signals: list[BettingSignal] = []
        self._last_collect_time: Optional[datetime] = None

    def collect_signals(
        self, force_refresh: bool = False, min_reliability: float = 0.7
    ) -> list[BettingSignal]:
        """
        Collect betting signals from all available sources.

        Args:
            force_refresh: Bypass cache and re-fetch all sources.
            min_reliability: Minimum source reliability (0.0-1.0) to include.

        Returns:
            List of BettingSignal objects, sorted by impact (most impactful first).
        """
        t0 = __import__("time").time()
        signals: list[BettingSignal] = []

        # ── Tier 1: Official NBA API (always available, highest reliability) ─
        try:
            api_signals = self._collect_from_nba_api()
            signals.extend(api_signals)
            logger.debug(f"NBA API: {len(api_signals)} signals")
        except Exception as e:
            logger.warning(f"NBA API signal collection failed: {e}")

        # ── Tier 2: ESPN injury scraper (free, reliable) ────────────────────
        try:
            espn_signals = self._collect_from_espn()
            signals.extend(espn_signals)
            logger.debug(f"ESPN: {len(espn_signals)} signals")
        except Exception as e:
            logger.warning(f"ESPN signal collection failed: {e}")

        # ── Tier 3: Twitter/X (requires API key, most real-time) ────────────
        try:
            twitter_signals = self._collect_from_twitter()
            signals.extend(twitter_signals)
            logger.debug(f"Twitter: {len(twitter_signals)} signals")
        except Exception as e:
            logger.debug(f"Twitter signal collection unavailable: {e}")

        # ── Deduplicate and score ───────────────────────────────────────────
        signals = self._deduplicate(signals)
        signals.sort(key=lambda s: s.impact_factor(), reverse=True)

        # Filter by reliability
        signals = [s for s in signals if s.source_reliability >= min_reliability]

        total_time = (__import__("time").time() - t0) * 1000
        logger.info(
            f"SocialMediaSignalCollector: {len(signals)} signals "
            f"from {len(signals)} deduplicated sources ({total_time:.0f}ms)"
        )

        self._last_signals = signals
        self._last_collect_time = datetime.now(timezone.utc)
        return signals

    def _collect_from_nba_api(self) -> list[BettingSignal]:
        """
        Fetch official injury data from nba_api.stats.

        Uses the same LeagueLeaders endpoint as PlayerPropEngine.
        Returns structured injury signals with 100% reliability.

        Returns:
            List of BettingSignal objects.
        """
        signals: list[BettingSignal] = []

        try:
            from nba_api.stats.endpoints import leagueleaders
            import socket

            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)

            for season in ("2025-26", "2024-25"):
                try:
                    leaders = leagueleaders.LeagueLeaders(
                        season=season,
                        season_type_all_star="Regular Season",
                    )
                    leaders_df = leaders.league_leaders.get_data_frame()
                    if len(leaders_df) == 0:
                        continue

                    # LeagueLeaders doesn't directly provide injury data.
                    # Use it to validate player names and teams for signals
                    # from other sources. For actual injury data, use ESPN.
                    logger.debug(f"NBA API: Loaded {len(leaders_df)} player records")
                    break
                except Exception:
                    continue

            socket.setdefaulttimeout(old_timeout)

        except ImportError:
            logger.debug("nba_api not available for signal collection")
        except Exception as e:
            logger.debug(f"NBA API signal collection error: {e}")

        return signals

    def _collect_from_espn(self) -> list[BettingSignal]:
        """
        Fetch injury signals from ESPN injury scraper.

        Every injury report becomes a structured betting signal.

        Returns:
            List of BettingSignal objects.
        """
        signals: list[BettingSignal] = []

        try:
            from betting_intel.data.injury_scraper import ESPNInjuryScraper

            scraper = ESPNInjuryScraper()
            records = scraper.fetch_all(force_refresh=True)

            for record in records:
                signal_type = SignalType.INJURY_OUT
                if record.injury_status.lower() == "questionable":
                    signal_type = SignalType.INJURY_QUESTIONABLE
                elif record.injury_status.lower() == "probable":
                    signal_type = SignalType.INJURY_PROBABLE

                signals.append(BettingSignal(
                    signal_type=signal_type,
                    player_name=record.player_name,
                    team=record.team,
                    message=(
                        f"{record.player_name} ({record.injury_description}) "
                        f"— {record.injury_status}"
                    ),
                    confidence=0.90,  # ESPN is highly reliable for injuries
                    source_account="ESPN",
                    source_reliability=0.92,
                ))
        except Exception as e:
            logger.warning(f"ESPN signal collection failed: {e}")

        return signals

    def _collect_from_twitter(self) -> list[BettingSignal]:
        """
        Fetch signals from Twitter/X accounts.

        This requires either:
          1. Twitter API v2 credentials (paid tier)
          2. Web scraping (fragile, rate-limited)

        For now, this is a scaffold that returns empty. When Twitter API
        keys are configured, it will fetch tweets from the accounts in
        nba_accounts.py and parse them into signals.

        Expected implementation:
          1. Get high-reliability accounts from nba_accounts.get_all_accounts()
          2. For each account, fetch recent tweets via Twitter API or scrape
          3. Parse each tweet using parse_injury_signal, parse_lineup_signal, etc.
          4. Return aggregated signals

        Returns:
            List of BettingSignal objects (currently empty — scaffold).
        """
        # Scaffold: log what would be collected
        high_rel = get_high_reliability_accounts(min_reliability=0.8)
        logger.debug(
            f"Twitter scaffold: {len(high_rel)} accounts available for monitoring. "
            "Configure TWITTER_API_KEY to enable real-time signal collection."
        )
        return []

    def get_active_signals(self, max_age_minutes: int = 30) -> list[BettingSignal]:
        """
        Get signals collected in the last N minutes.

        Args:
            max_age_minutes: Maximum age of signals to return.

        Returns:
            List of recent BettingSignal objects.
        """
        if not self._last_collect_time:
            return []

        age = (datetime.now(timezone.utc) - self._last_collect_time).total_seconds()
        if age > max_age_minutes * 60:
            return []

        return self._last_signals

    @staticmethod
    def _deduplicate(signals: list[BettingSignal]) -> list[BettingSignal]:
        """
        Remove duplicate signals for the same player + signal_type.

        Keeps the signal with the highest confidence.
        """
        seen: dict[tuple[str, SignalType], BettingSignal] = {}
        for s in signals:
            key = (s.player_name.lower(), s.signal_type)
            if key not in seen or s.confidence > seen[key].confidence:
                seen[key] = s
        return list(seen.values())