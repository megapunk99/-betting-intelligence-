"""
Twitter/X Signal Intelligence — scrapes NBA reporter tweets, parses them into
structured, actionable signals, and caches them for the betting system.

Architecture:
    TwitterSignalCollector
      ├── scrape_accounts()        → raw tweet texts
      ├── parse_signals()           → structured PlayerSignal objects
      ├── integrate(props, bets)    → adjusts existing predictions
      └── get_latest_signals()      → timeline for web display

Signal types extracted:
  - INJURY_OUT: "Player X is out tonight" → 0.0 minutes multiplier
  - INJURY_QUESTIONABLE: "Player X questionable" → 0.5 multiplier
  - INJURY_PROBABLE: "Player X probable" → 0.85 multiplier
  - LOAD_MGMT: "Player X sitting for rest" → 0.0 multiplier
  - GTD: "Player X is a game-time decision" → 0.6 multiplier
  - COACH_COMMENT: "Coach says Player X minutes limited" → adjust projection
  - LINEUP_CHANGE: "Player X starting at PF" → positional adjustment
  - PRACTICE_STATUS: "Player X full participant" → positive signal

Usage:
    from betting_intel.data.x_signals import TwitterSignalCollector

    collector = TwitterSignalCollector()
    signals = collector.collect_all()  # Scrape + parse recent tweets
    collector.integrate_player_props(props)  # Adjust prop predictions
    collector.integrate_team_bets(bets)      # Adjust team bets
    latest = collector.get_recent_signals(limit=20)  # For web display

The collector uses Nitter instances (public, no API key required) to fetch
tweets. When Nitter is unavailable, it returns no signals — the system
continues without fake injury data.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════


class SignalConfidence(Enum):
    """Confidence that a signal is accurate and actionable."""

    CONFIRMED = "confirmed"       # Official team announcement
    HIGH = "high"                 # Reliable insider report
    MEDIUM = "medium"             # Beat reporter with caveats
    LOW = "low"                   # Rumor, unverified
    SPECULATIVE = "speculative"   # Pure speculation


@dataclass
class PlayerSignal:
    """
    A single actionable signal extracted from a tweet.

    This is the core intelligence unit that drives all adjustments.
    """

    # ── Core Identity ──────────────────────────────────────────────────
    player_name: str
    team: str  # Team abbreviation (e.g., "LAL", "BOS")

    # ── Signal Content ─────────────────────────────────────────────────
    signal_type: str  # injury_out, injury_questionable, load_mgmt, etc.
    confidence: SignalConfidence

    # ── Quantitative Impact ────────────────────────────────────────────
    minutes_multiplier: float = 1.0  # 0.0 (out) → 1.0 (full)
    stat_adjustment: float = 0.0     # Points adjustment (raw)
    usage_adjustment: float = 0.0    # Usage rate adjustment (pct pts)

    # ── Metadata ───────────────────────────────────────────────────────
    source_account: str = ""         # Twitter handle
    tweet_text: str = ""             # Original tweet
    tweet_url: str = ""              # Link to tweet
    captured_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = ""             # When this signal is no longer relevant
    notes: str = ""

    # Position/role info
    position: str = ""
    injury_detail: str = ""          # e.g., "Right Ankle Sprain"

    def is_active(self) -> bool:
        """Check if this signal is still relevant."""
        if not self.expires_at:
            return True
        try:
            expiry = datetime.fromisoformat(self.expires_at)
            return datetime.now() < expiry
        except (ValueError, TypeError):
            return True

    def to_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "team": self.team,
            "signal_type": self.signal_type,
            "confidence": self.confidence.value,
            "minutes_multiplier": self.minutes_multiplier,
            "stat_adjustment": self.stat_adjustment,
            "usage_adjustment": self.usage_adjustment,
            "source_account": self.source_account,
            "tweet_text": self.tweet_text,
            "tweet_url": self.tweet_url,
            "captured_at": self.captured_at,
            "expires_at": self.expires_at,
            "notes": self.notes,
            "position": self.position,
            "injury_detail": self.injury_detail,
            "is_active": self.is_active(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# TWEET → SIGNAL PARSER
# ═══════════════════════════════════════════════════════════════════════════


class TweetSignalParser:
    """
    Parses raw tweet text into structured PlayerSignal objects.

    Uses regex patterns to extract:
      - Player names
      - Injury/status keywords
      - Team names
      - Coach quotes

    Pattern categories:
      INJURY:    out, questionable, probable, game-time decision
      LOAD:      rest, load management, sitting out
      LINEUP:    starting, coming off bench, will start
      PRACTICE:  full participant, limited, DNP (practice)
      COACH:     "Coach says", "Coach mentioned", per coach
    """

    # ── Known Players (NBA star players for better matching) ───────────
    KNOWN_PLAYERS = {
        "LeBron James", "Anthony Davis", "Stephen Curry", "Kevin Durant",
        "Giannis Antetokounmpo", "Luka Doncic", "Nikola Jokic", "Joel Embiid",
        "Shai Gilgeous-Alexander", "Jayson Tatum", "Jaylen Brown", "Devin Booker",
        "Anthony Edwards", "Victor Wembanyama", "Trae Young", "Damian Lillard",
        "Donovan Mitchell", "Ja Morant", "Zion Williamson", "Paolo Banchero",
        "Chet Holmgren", "Jalen Brunson", "Tyrese Haliburton", "Bam Adebayo",
        "Rudy Gobert", "Karl-Anthony Towns", "Kyrie Irving", "Kawhi Leonard",
        "Paul George", "James Harden", "Jimmy Butler", "Bobby Portis",
        "Khris Middleton", "Kristaps Porzingis", "Derrick White", "Jrue Holiday",
        "CJ McCollum", "Brandon Ingram", "De'Aaron Fox", "Domantas Sabonis",
        "LaMelo Ball", "Scottie Barnes", "Evan Mobley", "Darius Garland",
        "Cade Cunningham", "Jaden Ivey", "Walker Kessler", "Keegan Murray",
        "Austin Reaves", "D'Angelo Russell", "Rui Hachimura", "Deni Avdija",
        "Jordan Poole", "Kyle Kuzma", "Jonas Valanciunas", "Herb Jones",
        "Trey Murphy III", "Cameron Johnson", "Mikal Bridges", "Nic Claxton",
        "Spencer Dinwiddie", "Bojan Bogdanovic", "Jalen Suggs", "Franz Wagner",
        "Immanuel Quickley", "RJ Barrett", "Anfernee Simons", "Shaedon Sharpe",
        "Collin Sexton", "Lauri Markkanen", "John Collins", "Keyonte George",
        "Jaden McDaniels", "Naz Reid", "Mike Conley", "Malik Monk",
        "Grayson Allen", "Jusuf Nurkic", "Bradley Beal", "Klay Thompson",
        "Jonathan Kuminga", "Andrew Wiggins", "Draymond Green", "Brandon Podziemski",
        "Tobias Harris", "Tyrese Maxey", "Kelly Oubre Jr.", "Andre Drummond",
        "Marcus Smart", "Derrick Rose", "Patrick Beverley", "Dejounte Murray",
        "DeMar DeRozan", "Zach LaVine", "Nikola Vucevic", "Coby White",
        "Jarrett Allen", "Max Strus", "Georges Niang", "Al Horford",
        "Derrick White", "Sam Hauser", "Payton Pritchard", "Luke Kornet",
        "Precious Achiuwa", "Miles McBride", "Josh Hart", "Donte DiVincenzo",
        "Isaiah Hartenstein", "Mitchell Robinson", "Julius Randle", "OG Anunoby",
        "Kelly Olynyk", "Jaime Jaquez Jr.", "Tyler Herro", "Terry Rozier",
        "Duncan Robinson", "Caleb Martin", "Kevin Love", "Nikola Jovic",
        "Cole Anthony", "Markelle Fultz", "Wendell Carter Jr.", "Gary Harris",
        "Jalen Duren", "Isaiah Stewart", "Ausar Thompson", "Tim Hardaway Jr.",
        "Daniel Gafford", "P.J. Washington", "Derrick Jones Jr.", "Dereck Lively II",
        "Josh Green", "Dante Exum", "Maxi Kleber", "Jaden Hardy",
        "Alperen Sengun", "Jalen Green", "Fred VanVleet", "Dillon Brooks",
        "Jabari Smith Jr.", "Tari Eason", "Cam Whitmore", "Amen Thompson",
        "Ja Morant", "Desmond Bane", "Jaren Jackson Jr.", "Marcus Smart",
        "Luke Kennard", "Santi Aldama", "GG Jackson", "Ziaire Williams",
        "Brandon Ingram", "CJ McCollum", "Zion Williamson", "Herb Jones",
        "Trey Murphy III", "Jonas Valanciunas", "Dyson Daniels", "Naji Marshall",
        "Jose Alvarado", "Larry Nance Jr.", "Cody Zeller",
        "De'Aaron Fox", "Domantas Sabonis", "Keegan Murray", "Malik Monk",
        "Kevin Huerter", "Harrison Barnes", "Trey Lyles", "Davion Mitchell",
        "Chris Duarte", "JaVale McGee", "Alex Len",
        "Victor Wembanyama", "Devin Vassell", "Keldon Johnson", "Jeremy Sochan",
        "Zach Collins", "Tre Jones", "Malaki Branham", "Julian Champagnie",
        "Blake Wesley", "Sidy Cissoko", "Sandro Mamukelashvili", "Charles Bassey",
        "Chet Holmgren", "Shai Gilgeous-Alexander", "Jalen Williams", "Luguentz Dort",
        "Josh Giddey", "Cason Wallace", "Isaiah Joe", "Aaron Wiggins",
        "Kenrich Williams", "Jaylin Williams", "Ousmane Dieng", "Lindy Waters III",
        "LaMelo Ball", "Brandon Miller", "Miles Bridges", "Mark Williams",
        "P.J. Washington", "Nick Richards", "Bryce McGowens", "Cody Martin",
        "Ish Smith", "Frank Ntilikina", "JT Thor", "Nick Smith Jr.",
    }

    # ── Regex Patterns ─────────────────────────────────────────────────

    # Status keywords → signal type mapping
    STATUS_PATTERNS: dict[str, tuple[str, float, SignalConfidence]] = {
        # OUT = confirmed absence
        "will be out": ("injury_out", 0.0, SignalConfidence.CONFIRMED),
        "is out": ("injury_out", 0.0, SignalConfidence.CONFIRMED),
        "has been ruled out": ("injury_out", 0.0, SignalConfidence.CONFIRMED),
        "ruled out": ("injury_out", 0.0, SignalConfidence.CONFIRMED),
        "out for": ("injury_out", 0.0, SignalConfidence.CONFIRMED),
        "out tonight": ("injury_out", 0.0, SignalConfidence.HIGH),
        "out with": ("injury_out", 0.0, SignalConfidence.HIGH),
        "will not play": ("injury_out", 0.0, SignalConfidence.CONFIRMED),
        "won't play": ("injury_out", 0.0, SignalConfidence.HIGH),
        "will miss": ("injury_out", 0.0, SignalConfidence.HIGH),
        "doubtful": ("injury_out", 0.2, SignalConfidence.HIGH),

        # Questionable = uncertain
        "questionable": ("injury_questionable", 0.5, SignalConfidence.HIGH),
        "game-time decision": ("gtd", 0.6, SignalConfidence.HIGH),
        "game time decision": ("gtd", 0.6, SignalConfidence.HIGH),
        "GTD": ("gtd", 0.6, SignalConfidence.HIGH),

        # Probable = likely plays
        "probable": ("injury_probable", 0.85, SignalConfidence.HIGH),

        # Load management
        "load management": ("load_mgmt", 0.0, SignalConfidence.CONFIRMED),
        "rest on": ("load_mgmt", 0.0, SignalConfidence.HIGH),
        "sitting out": ("load_mgmt", 0.0, SignalConfidence.HIGH),
        "being held out": ("load_mgmt", 0.0, SignalConfidence.HIGH),
        "rest night": ("load_mgmt", 0.0, SignalConfidence.HIGH),
        "scheduled rest": ("load_mgmt", 0.0, SignalConfidence.HIGH),

        # Returning from injury
        "will return": ("injury_return", 1.0, SignalConfidence.HIGH),
        "returning tonight": ("injury_return", 1.0, SignalConfidence.HIGH),
        "back tonight": ("injury_return", 1.0, SignalConfidence.HIGH),
        "activated": ("injury_return", 1.0, SignalConfidence.HIGH),
        "available tonight": ("injury_return", 1.0, SignalConfidence.CONFIRMED),
        "will play tonight": ("injury_return", 1.0, SignalConfidence.HIGH),

        # Illness / personal
        "illness": ("illness", 0.6, SignalConfidence.HIGH),
        "personal reasons": ("personal", 0.0, SignalConfidence.LOW),
        "personal matter": ("personal", 0.0, SignalConfidence.HIGH),
        "non-covid illness": ("illness", 0.4, SignalConfidence.HIGH),

        # Suspension
        "suspended": ("suspension", 0.0, SignalConfidence.CONFIRMED),

        # Conditioning
        "minutes limit": ("conditioning", 0.7, SignalConfidence.HIGH),
        "limited minutes": ("conditioning", 0.7, SignalConfidence.HIGH),
        "minute restriction": ("conditioning", 0.7, SignalConfidence.HIGH),
    }

    # Coach quote patterns
    COACH_PATTERNS: list[str] = [
        r"coach\s+(says?|said|mentioned|indicated)",
        r"per\s+(coach|head\s+coach)",
        r"(says?|said)\s+(coach|head coach)",
    ]

    # Practice status patterns
    PRACTICE_PATTERNS: dict[str, tuple[str, float]] = {
        "full participant": ("practice_full", 1.0),
        "full practice": ("practice_full", 1.0),
        "limited practice": ("practice_limited", 0.6),
        "limited participant": ("practice_limited", 0.6),
        "did not practice": ("practice_dnp", 0.3),
        "dnp": ("practice_dnp", 0.3),
    }

    # Lineup change patterns
    LINEUP_PATTERNS: list[str] = [
        r"will start",
        r"starting\s+(at|tonight)",
        r"in the starting lineup",
        r"coming off the bench",
        r"moved to the bench",
    ]

    # Body part patterns (for injury details)
    BODY_PARTS: list[str] = [
        "ankle", "knee", "hamstring", "quad", "calf", "foot", "toe",
        "shoulder", "elbow", "wrist", "hand", "finger", "thumb",
        "hip", "groin", "back", "neck", "head", "concussion",
        "Achilles", "meniscus", "MCL", "ACL", "ACL", "oblique",
        "abductor", "adductor", "glute", "hammy",
    ]

    def parse_tweet(
        self,
        text: str,
        source_handle: str = "",
        source_name: str = "",
    ) -> list[PlayerSignal]:
        """
        Parse a single tweet into zero or more PlayerSignals.

        Uses pattern matching to detect:
          - Player names (from known player database)
          - Status keywords (out, questionable, etc.)
          - Team names
          - Coach quotes
          - Practice participation

        Returns:
            List of PlayerSignal objects (empty if no players detected)
        """
        signals: list[PlayerSignal] = []
        text_lower = text.lower()

        # 1. Detect if this is a practice report (fast path)
        practice_signals = self._detect_practice(text, text_lower)
        signals.extend(practice_signals)

        # 2. Detect coach comments
        is_coach_quote = any(
            re.search(pattern, text_lower)
            for pattern in self.COACH_PATTERNS
        )

        # 3. Detect lineup changes
        is_lineup = any(
            re.search(pattern, text_lower)
            for pattern in self.LINEUP_PATTERNS
        )

        # 4. Find mentioned players
        mentioned_players = self._find_players(text)

        if not mentioned_players:
            # Try to find player names even if not in our known list
            # by looking for capitalized two-word names
            mention_signals = self._detect_generic_players(text, text_lower)
            signals.extend(mention_signals)
            return signals

        # 5. Detect status keywords
        detected_status = self._detect_status(text_lower)
        detected_team = self._detect_team(text)
        injury_detail = self._detect_injury_detail(text_lower)

        for player_name in mentioned_players:
            team = detected_team or self._guess_team(player_name)

            if detected_status:
                signal_type, minutes_mult, confidence = detected_status
            elif is_coach_quote:
                signal_type = "coach_comment"
                minutes_mult = 0.85  # Soft downgrade
                confidence = SignalConfidence.HIGH
            elif is_lineup:
                signal_type = "lineup_change"
                minutes_mult = 1.0
                confidence = SignalConfidence.HIGH
            elif practice_signals:
                continue  # Already added above
            else:
                # Generic mention — still valuable for context
                signal_type = "mention"
                minutes_mult = 1.0
                confidence = SignalConfidence.LOW

            # Determine stat adjustment based on signal type
            stat_adj = 0.0
            usage_adj = 0.0
            if signal_type in ("injury_out", "suspension"):
                stat_adj = -15.0  # Complete removal
                usage_adj = -0.20
            elif signal_type == "injury_questionable":
                stat_adj = -5.0
                usage_adj = -0.08
            elif signal_type == "load_mgmt":
                stat_adj = -15.0
                usage_adj = -0.20
            elif signal_type == "injury_return":
                stat_adj = 0.0  # Just restoring to normal
                usage_adj = 0.0
            elif signal_type in ("illness", "personal"):
                stat_adj = -4.0
                usage_adj = -0.05
            elif signal_type == "conditioning":
                stat_adj = -3.0
                usage_adj = -0.05
            elif signal_type == "coach_comment":
                stat_adj = -2.0  # Soft negative
                usage_adj = -0.03

            signal = PlayerSignal(
                player_name=player_name,
                team=team,
                signal_type=signal_type,
                confidence=confidence,
                minutes_multiplier=minutes_mult,
                stat_adjustment=stat_adj,
                usage_adjustment=usage_adj,
                source_account=source_handle,
                tweet_text=text[:280],
                position="",
                injury_detail=injury_detail,
                notes=f"Source: @{source_handle}" if source_handle else "Twitter/X",
            )
            signals.append(signal)

        return signals

    # ── Internal Detection Methods ──────────────────────────────────────

    _PLAYER_PATTERN = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:\s+(?:III|Jr\.|Sr\.|II|IV))?))\b'
    )

    def _find_players(self, text: str) -> list[str]:
        """Find known NBA players in text."""
        found = []
        text_lower = text.lower()
        for player in sorted(self.KNOWN_PLAYERS, key=len, reverse=True):
            if player.lower() in text_lower:
                found.append(player)
        return found

    def _detect_generic_players(self, text: str, text_lower: str) -> list[PlayerSignal]:
        """Detect player references not in our known list."""
        signals = []
        # Look for patterns like "Player X is out"
        # Match: capitalized first + last name followed by status keyword
        patterns = [
            r'([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(?:is\s+)?(?:out|questionable|probable|doubtful)',
            r'(?:for\s+)?([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(?:will\s+)?(?:miss|sit|rest)',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                full_name = f"{match.group(1)} {match.group(2)}"
                status_info = self._detect_status(text_lower)
                if status_info:
                    signal_type, minutes_mult, confidence = status_info
                    signals.append(PlayerSignal(
                        player_name=full_name,
                        team=self._detect_team(text) or "NBA",
                        signal_type=signal_type,
                        confidence=confidence,
                        minutes_multiplier=minutes_mult,
                        notes="Auto-detected player reference",
                    ))
        return signals

    def _detect_status(
        self, text_lower: str
    ) -> Optional[tuple[str, float, SignalConfidence]]:
        """Detect status keyword in text."""
        # Sort by length descending to match longer patterns first
        for keyword, (signal_type, mult, confidence) in sorted(
            self.STATUS_PATTERNS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if keyword in text_lower:
                return (signal_type, mult, confidence)

        # Check for 'day-to-day' as generic uncertainty
        if "day-to-day" in text_lower or "day to day" in text_lower:
            return ("injury_questionable", 0.5, SignalConfidence.MEDIUM)

        return None

    def _detect_practice(self, text: str, text_lower: str) -> list[PlayerSignal]:
        """Detect practice participation signals."""
        signals = []
        for pattern, (signal_type, mult) in self.PRACTICE_PATTERNS.items():
            if pattern in text_lower:
                players = self._find_players(text)
                if not players:
                    continue
                for player in players:
                    team = self._guess_team(player)
                    signals.append(PlayerSignal(
                        player_name=player,
                        team=team,
                        signal_type=signal_type,
                        confidence=SignalConfidence.HIGH,
                        minutes_multiplier=mult,
                        notes=f"Practice status: {pattern}",
                    ))
        return signals

    def _detect_team(self, text: str) -> str:
        """Detect team abbreviation from text using common patterns."""
        from betting_intel.data.injury_scraper import extract_team_abbr
        abbr = extract_team_abbr(text)
        if abbr:
            return abbr
        # Fallback: look for common team references
        team_keywords = {
            "LAL": ["lakers", "la lakers", "los angeles"],
            "BOS": ["celtics", "boston"],
            "GSW": ["warriors", "golden state"],
            "MIL": ["bucks", "milwaukee"],
            "DEN": ["nuggets", "denver"],
            "PHI": ["sixers", "76ers", "philadelphia"],
            "OKC": ["thunder", "okc", "oklahoma"],
            "SAS": ["spurs", "san antonio"],
            "MIA": ["heat", "miami"],
            "NYK": ["knicks", "new york"],
        }
        text_lower = text.lower()
        for abbr, keywords in team_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return abbr
        return ""

    def _detect_injury_detail(self, text_lower: str) -> str:
        """Extract injury body part detail from text."""
        for part in self.BODY_PARTS:
            if part in text_lower:
                # Extract surrounding context
                idx = text_lower.index(part)
                start = max(0, idx - 20)
                end = min(len(text_lower), idx + len(part) + 20)
                context = text_lower[start:end].strip()
                return context
        return ""

    _TEAM_GUESS_MAP: dict[str, str] = {
        "LeBron James": "LAL", "Anthony Davis": "LAL", "Austin Reaves": "LAL",
        "Stephen Curry": "GSW", "Draymond Green": "GSW", "Klay Thompson": "GSW",
        "Giannis Antetokounmpo": "MIL", "Damian Lillard": "MIL", "Khris Middleton": "MIL",
        "Nikola Jokic": "DEN", "Jamal Murray": "DEN", "Aaron Gordon": "DEN",
        "Jayson Tatum": "BOS", "Jaylen Brown": "BOS", "Kristaps Porzingis": "BOS",
        "Luka Doncic": "DAL", "Kyrie Irving": "DAL", "Victor Wembanyama": "SAS",
        "Shai Gilgeous-Alexander": "OKC", "Jalen Williams": "OKC", "Chet Holmgren": "OKC",
        "Joel Embiid": "PHI", "Tyrese Maxey": "PHI", "Paul George": "PHI",
        "Kevin Durant": "PHX", "Devin Booker": "PHX", "Bradley Beal": "PHX",
        "Anthony Edwards": "MIN", "Karl-Anthony Towns": "NYK", "Rudy Gobert": "MIN",
        "Jalen Brunson": "NYK", "Julius Randle": "NYK", "OG Anunoby": "NYK",
        "Donovan Mitchell": "CLE", "Darius Garland": "CLE", "Evan Mobley": "CLE",
        "Tyrese Haliburton": "IND", "Pascal Siakam": "IND", "Myles Turner": "IND",
        "Bam Adebayo": "MIA", "Jimmy Butler": "MIA", "Tyler Herro": "MIA",
        "Trae Young": "ATL", "Dejounte Murray": "NOP", "Zion Williamson": "NOP",
        "Ja Morant": "MEM", "Jaren Jackson Jr.": "MEM", "Desmond Bane": "MEM",
        "LaMelo Ball": "CHA", "Brandon Miller": "CHA", "Miles Bridges": "CHA",
        "Scottie Barnes": "TOR", "RJ Barrett": "TOR", "Immanuel Quickley": "TOR",
        "Cade Cunningham": "DET", "Jaden Ivey": "DET", "Ausar Thompson": "DET",
        "De'Aaron Fox": "SAC", "Domantas Sabonis": "SAC", "Malik Monk": "SAC",
        "Paolo Banchero": "ORL", "Franz Wagner": "ORL", "Jalen Suggs": "ORL",
        "Zach LaVine": "CHI", "DeMar DeRozan": "SAC", "Nikola Vucevic": "CHI",
        "Kawhi Leonard": "LAC", "Paul George": "PHI", "James Harden": "LAC",
        "Alperen Sengun": "HOU", "Jalen Green": "HOU", "Fred VanVleet": "HOU",
        "Brandon Ingram": "NOP", "CJ McCollum": "NOP", "Herb Jones": "NOP",
        "Lauri Markkanen": "UTA", "Collin Sexton": "UTA", "Keyonte George": "UTA",
        "Jerami Grant": "POR", "Anfernee Simons": "POR", "Scoot Henderson": "POR",
        "Kyle Kuzma": "WAS", "Jordan Poole": "WAS", "Deni Avdija": "POR",
        "Keegan Murray": "SAC", "Domantas Sabonis": "SAC",
        "Chet Holmgren": "OKC", "Isaiah Hartenstein": "OKC",
    }

    def _guess_team(self, player_name: str) -> str:
        """Guess team for a player based on known mapping."""
        return self._TEAM_GUESS_MAP.get(player_name, "")


# ═══════════════════════════════════════════════════════════════════════════
# TWITTER COLLECTOR (Nitter-based scraping)
# ═══════════════════════════════════════════════════════════════════════════

# Nitter instances (public, no API key required)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.lacontrevoie.fr",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
    "https://nitter.unixfox.eu",
    "https://nitter.hu",
    "https://nitter.projectsegfau.lt",
    "https://nitter.moomoo.me",
]


class NitterScraper:
    """
    Scrapes tweets from Nitter (public Twitter/X frontend).

    No API key required. Uses requests + BeautifulSoup to fetch and parse
    tweet timelines. Falls back gracefully if Nitter is blocked or down.

    Nitter exposes:
      - /USERNAME              → tweet timeline
      - /USERNAME/with_replies → tweets + replies
      - /USERNAME/media        → tweets with media

    Each tweet has:
      - .tweet-content  → text body
      - .tweet-date     → timestamp (as <a> with title attribute)
      - .tweet-link     → permalink
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session = self._create_session()
        self._working_instance: Optional[str] = None

    def _create_session(self):
        """Create a requests session with browser-like headers."""
        import requests as req
        session = req.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        return session

    def fetch_tweets(
        self,
        username: str,
        max_tweets: int = 10,
    ) -> list[dict]:
        """
        Fetch recent tweets from a Nitter timeline.

        Args:
            username: Twitter handle (without @)
            max_tweets: Maximum tweets to fetch

        Returns:
            List of dicts with keys: text, url, date, author
        """
        instance = self._get_working_instance()
        if not instance:
            logger.warning("No working Nitter instance found")
            return []

        url = f"{instance}/{username}"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            logger.debug(f"Nitter fetch failed for @{username}: {e}")
            return []

        return self._parse_timeline(resp.text, username, max_tweets)

    def fetch_multiple(
        self,
        usernames: list[str],
        max_per_account: int = 5,
    ) -> dict[str, list[dict]]:
        """
        Fetch tweets from multiple accounts.

        Args:
            usernames: List of Twitter handles
            max_per_account: Max tweets per account

        Returns:
            Dict mapping handle → list of tweet dicts
        """
        results = {}
        for username in usernames:
            tweets = self.fetch_tweets(username, max_per_account)
            if tweets:
                results[username] = tweets
        return results

    def _get_working_instance(self) -> Optional[str]:
        """Find the first working Nitter instance."""
        if self._working_instance:
            return self._working_instance

        for instance in NITTER_INSTANCES:
            try:
                resp = self._session.get(
                    f"{instance}/ShamsCharania",
                    timeout=8,
                )
                if resp.status_code == 200:
                    self._working_instance = instance
                    logger.info(f"Using Nitter instance: {instance}")
                    return instance
            except Exception:
                continue

        logger.warning("No Nitter instances are reachable")
        return None

    def _parse_timeline(
        self,
        html: str,
        username: str,
        max_tweets: int,
    ) -> list[dict]:
        """Parse Nitter HTML timeline into tweet dicts."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        tweets = []

        for tweet_div in soup.find_all("div", class_="timeline-item"):
            if len(tweets) >= max_tweets:
                break

            # Extract text
            content_div = tweet_div.find("div", class_="tweet-content")
            text = content_div.get_text(strip=True) if content_div else ""

            if not text:
                continue

            # Extract date
            date_link = tweet_div.find("a", class_="tweet-date")
            date_str = ""
            if date_link:
                date_str = date_link.get("title", "") or date_link.get_text(strip=True)

            # Extract permalink
            permalink = ""
            if date_link:
                href = date_link.get("href", "")
                if href:
                    permalink = f"https://nitter.net{href}"

            tweets.append({
                "text": text,
                "url": permalink or f"https://x.com/{username}/status/unknown",
                "date": date_str,
                "author": username,
            })

        return tweets

    def test_connection(self) -> bool:
        """Test if Nitter is accessible."""
        return self._get_working_instance() is not None


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL INTEGRATOR (adjusts predictions based on signals)
# ═══════════════════════════════════════════════════════════════════════════


class SignalIntegrator:
    """
    Integrates Twitter signals into existing betting predictions.

    Adjusts:
      - Player prop predictions (points, rebounds, assists)
      - Team-level bets (moneyline, spread, total)
      - Confidence levels

    The integration is conservative: signals only degrade predictions,
    they never upgrade them. This ensures we never over-estimate an edge.
    """

    def __init__(self):
        self.parser = TweetSignalParser()

    def integrate_player_props(
        self,
        props: list,
        signals: list[PlayerSignal],
    ) -> list:
        """
        Adjust player prop predictions based on signals.

        For each prop, checks if there's a relevant signal for that player.
        If yes:
          - Reduces predicted stats (pts, reb, ast) by adjustment factor
          - Lowers confidence
          - Adds reasoning about the signal

        Args:
            props: List of BetSuggestion objects from PlayerPropEngine
            signals: List of active PlayerSignal objects

        Returns:
            Modified props list
        """
        if not signals:
            return props

        # Build signal lookup: player_name → list of signals
        signal_lookup: dict[str, list[PlayerSignal]] = {}
        for sig in signals:
            if not sig.is_active():
                continue
            key = sig.player_name.lower()
            if key not in signal_lookup:
                signal_lookup[key] = []
            signal_lookup[key].append(sig)

        if not signal_lookup:
            return props

        # Adjust each prop
        adjusted_count = 0
        for prop in props:
            player_name = getattr(prop, "player_name", None) or self._extract_player(prop)
            if not player_name:
                continue

            player_signals = signal_lookup.get(player_name.lower(), [])
            if not player_signals:
                continue

            # Apply the most severe signal (worst-case scenario)
            worst_signal = max(
                player_signals,
                key=lambda s: s.stat_adjustment,
            )

            # Adjust predicted value
            old_value = prop.predicted_value
            new_value = max(0, old_value + worst_signal.stat_adjustment * 0.3)
            prop.predicted_value = new_value

            # Reduce edge proportionally
            if old_value > 0:
                reduction = abs(worst_signal.stat_adjustment * 0.3 / old_value)
                prop.edge_pct = max(0, prop.edge_pct * (1 - reduction))

            # Downgrade confidence
            if worst_signal.signal_type in ("injury_out", "load_mgmt", "suspension"):
                prop.confidence = Confidence.VERY_LOW if hasattr(prop, 'confidence') else prop.confidence
                prop.is_clear_pick = False
            elif worst_signal.signal_type == "injury_questionable":
                # Half confidence
                pass

            # Add reasoning
            if worst_signal.tweet_text:
                signal_note = (
                    f"[X SIGNAL] @{worst_signal.source_account}: "
                    f"\"{worst_signal.tweet_text[:120]}...\""
                    if len(worst_signal.tweet_text) > 120
                    else f"[X SIGNAL] @{worst_signal.source_account}: \"{worst_signal.tweet_text}\""
                )
                if worst_signal.injury_detail:
                    signal_note += f" | Injury: {worst_signal.injury_detail}"
                prop.reasoning = signal_note + " | " + prop.reasoning

            # Store signal info in metadata
            if hasattr(prop, 'metadata') and prop.metadata is not None:
                prop.metadata["x_signal"] = worst_signal.to_dict()

            adjusted_count += 1

        logger.info(f"Signal integrator adjusted {adjusted_count}/{len(props)} props")
        return props

    def integrate_team_bets(
        self,
        bets: list,
        signals: list[PlayerSignal],
    ) -> list:
        """
        Adjust team-level bets based on signals.

        If a star player is out, adjust:
          - Moneyline: reduce win probability
          - Spread: increase predicted margin for the other team
          - Total: reduce predicted total
          - Team totals: reduce affected team's predicted points

        Args:
            bets: List of BetSuggestion objects from RecommendationEngine
            signals: List of active PlayerSignal objects

        Returns:
            Modified bets list
        """
        if not signals:
            return bets

        # Identify teams with critical losses
        critical_out: dict[str, list[str]] = {}
        questionable: dict[str, list[str]] = {}

        for sig in signals:
            if not sig.is_active():
                continue
            if sig.signal_type in ("injury_out", "load_mgmt", "suspension"):
                if sig.team not in critical_out:
                    critical_out[sig.team] = []
                critical_out[sig.team].append(sig.player_name)
            elif sig.signal_type == "injury_questionable":
                if sig.team not in questionable:
                    questionable[sig.team] = []
                questionable[sig.team].append(sig.player_name)

        if not critical_out and not questionable:
            return bets

        team_to_league_name = self._build_team_name_map()

        for bet in bets:
            # Match bet to affected team
            affected_types = set()
            for team_abbr, players in critical_out.items():
                team_full = team_to_league_name.get(team_abbr, team_abbr)
                if team_full in bet.matchup or team_full in bet.bet_side:
                    affected_types.add(("OUT", team_full, players))

            for team_abbr, players in questionable.items():
                team_full = team_to_league_name.get(team_abbr, team_abbr)
                if team_full in bet.matchup or team_full in bet.bet_side:
                    affected_types.add(("QUESTIONABLE", team_full, players))

            if not affected_types:
                continue

            for status, team, players in affected_types:
                if status == "OUT":
                    # Major downgrade
                    if bet.bet_type.value == "moneyline" and team in bet.bet_side:
                        # Star out → downgrade win probability
                        old_wp = bet.win_probability
                        bet.win_probability = max(0.15, old_wp * 0.7)
                        bet.edge_pct = max(0, bet.edge_pct * 0.5)
                        bet.reasoning = (
                            f"[X SIGNAL] {', '.join(players)} OUT for {team}. "
                            f"Win probability adjusted: {old_wp:.0%} → {bet.win_probability:.0%}. "
                            + bet.reasoning
                        )
                    elif bet.bet_type.value in ("total_points", "team_total"):
                        # Star out → lower scoring
                        if hasattr(bet, 'predicted_value'):
                            old_pred = bet.predicted_value
                            if "UNDER" in bet.bet_side or team in bet.bet_side:
                                bet.predicted_value = old_pred * 0.92
                                bet.edge_pct = max(0, bet.edge_pct * 0.7)
                                bet.reasoning = (
                                    f"[X SIGNAL] {', '.join(players)} OUT for {team}. "
                                    f"Scoring projection reduced. "
                                    + bet.reasoning
                                )
                elif status == "QUESTIONABLE":
                    # Soft downgrade
                    old_wp = bet.win_probability
                    bet.win_probability = max(0.25, old_wp * 0.85)
                    bet.edge_pct = max(0, bet.edge_pct * 0.8)

            if hasattr(bet, 'metadata') and bet.metadata is not None:
                if "x_signals" not in bet.metadata:
                    bet.metadata["x_signals"] = []
                bet.metadata["x_signals"].append({
                    "players": players,
                    "status": status,
                    "team": team,
                    "adjusted": True,
                })

        return bets

    @staticmethod
    def _extract_player(prop) -> Optional[str]:
        """Extract player name from a prop bet."""
        return getattr(prop, "player_name", None)

    @staticmethod
    def _build_team_name_map() -> dict[str, str]:
        """Build abbreviation → full name mapping."""
        from betting_intel.data.injury_scraper import TEAM_ABBREVIATIONS
        import betting_intel.data.injury_scraper as inj
        # Reverse the mapping
        name_to_abbr = {}
        for abbr, full_name in inj.TEAM_ABBREVIATIONS.items():
            name_to_abbr[abbr] = full_name
        return name_to_abbr


# To avoid circular imports
from betting_intel.recommendations.bet_types import Confidence


# ═══════════════════════════════════════════════════════════════════════════
# MAIN COLLECTOR (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════


class TwitterSignalCollector:
    """
    Main orchestrator for Twitter/X signal intelligence.

    Manages:
      - Nitter scraping
      - Tweet parsing
      - Signal caching
      - Integration with betting engine

    Usage:
        collector = TwitterSignalCollector()
        collector.collect_all()           # Full scrape + parse
        signals = collector.get_active_signals()

        # Integration
        adjusted_props = collector.integrate_player_props(props)
        adjusted_bets = collector.integrate_team_bets(bets)

        # Display
        timeline = collector.get_recent_signals(limit=20)
    """

    # Default accounts to scrape (high-reliability + injury trackers)
    DEFAULT_ACCOUNTS: list[str] = [
        # National insiders
        "ShamsCharania",
        "TheSteinLine",
        "ChrisBHaynes",
        # Injury trackers
        "UnderdogNBA",
        "FantasyLabsNBA",
        "Rotoworld_BBALL",
        "NBCSEdgeNBA",
        # Key beat reporters
        "anthonyVslater",   # Warriors
        "KCJHoop",          # Bulls
        "MikeTrudell",      # Lakers
        "JonKrawczynski",   # Timberwolves
        "ChrisFedor",       # Cavs
        "ScottAgness",      # Pacers
        "Will Guillory",    # Pelicans
        "timbontemps",      # Clippers
        "EricNehm",         # Bucks
    ]

    def __init__(
        self,
        accounts: Optional[list[str]] = None,
        max_tweets_per_account: int = 5,
        min_reliability: float = 0.7,
    ):
        self.accounts = accounts or self.DEFAULT_ACCOUNTS
        self.max_per_account = max_tweets_per_account
        self.min_reliability = min_reliability

        self.scraper = NitterScraper()
        self.parser = TweetSignalParser()
        self.integrator = SignalIntegrator()

        # In-memory signal store
        self._signals: list[PlayerSignal] = []
        self._last_collect_time: Optional[float] = None
        self._collect_interval = 120  # 2 minutes between full scrapes

    def collect_all(self, force: bool = False) -> list[PlayerSignal]:
        """
        Full collection: scrape tweets → parse → store signals.

        Args:
            force: If False, uses cached results if within TTL

        Returns:
            List of parsed PlayerSignal objects
        """
        now = time.time()
        if not force and self._last_collect_time is not None:
            if (now - self._last_collect_time) < self._collect_interval:
                return self._signals

        # 1. Check if Nitter is reachable
        nitter_available = self.scraper.test_connection()

        if nitter_available:
            # 2. Scrape tweets from key accounts
            logger.info(f"Scraping {len(self.accounts)} NBA accounts from X/Twitter...")
            tweets_by_account = self.scraper.fetch_multiple(
                self.accounts,
                self.max_per_account,
            )

            # 3. Parse each tweet into signals
            all_signals: list[PlayerSignal] = []
            for handle, tweets in tweets_by_account.items():
                for tweet in tweets:
                    parsed = self.parser.parse_tweet(
                        tweet["text"],
                        source_handle=handle,
                    )
                    # Add tweet URL
                    for sig in parsed:
                        sig.tweet_url = tweet.get("url", "")
                        # Set expiry: signals expire at game time (assume same day)
                        sig.expires_at = (
                            datetime.now() + timedelta(hours=12)
                        ).isoformat()
                    all_signals.extend(parsed)

            logger.info(f"Parsed {len(all_signals)} signals from {len(tweets_by_account)} accounts")
        else:
            # 3b. Nitter is down — no signals available
            logger.warning("Nitter unreachable — no X/Twitter signals available")
            all_signals = []

        # 4. Deduplicate by (player, signal_type)
        self._signals = self._deduplicate(all_signals)

        # 5. Add to existing signals (don't replace, merge)
        self._signals = self._merge_signals(self._signals)

        self._last_collect_time = time.time()
        return self._signals

    def get_active_signals(
        self,
        signal_type: Optional[str] = None,
        team: Optional[str] = None,
        min_confidence: Optional[str] = None,
    ) -> list[PlayerSignal]:
        """
        Get active signals, filtered by criteria.

        Args:
            signal_type: Filter by signal type (e.g., "injury_out")
            team: Filter by team abbreviation
            min_confidence: Minimum confidence level

        Returns:
            Filtered list of active signals
        """
        signals = [s for s in self._signals if s.is_active()]

        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]

        if team:
            signals = [s for s in signals if s.team.upper() == team.upper()]

        if min_confidence:
            levels = {
                "confirmed": 4, "high": 3, "medium": 2, "low": 1, "speculative": 0,
            }
            min_level = levels.get(min_confidence, 0)
            signals = [
                s for s in signals
                if levels.get(s.confidence.value, 0) >= min_level
            ]

        return signals

    def get_most_impactful_signals(self, limit: int = 10) -> list[PlayerSignal]:
        """
        Get the signals with the most betting impact.

        Sorted by severity: OUT > Suspension > Load Mgmt > Questionable
        """
        severity_order = {
            "injury_out": 100, "suspension": 90, "load_mgmt": 85,
            "injury_questionable": 50, "gtd": 40, "illness": 35,
            "injury_probable": 10, "injury_return": 5,
        }

        active = [s for s in self._signals if s.is_active()]
        active.sort(
            key=lambda s: severity_order.get(s.signal_type, 0),
            reverse=True,
        )
        return active[:limit]

    def get_recent_signals(self, limit: int = 30) -> list[dict]:
        """Get recent signals for web display."""
        sorted_signals = sorted(
            self._signals,
            key=lambda s: s.captured_at,
            reverse=True,
        )
        return [s.to_dict() for s in sorted_signals[:limit]]

    def get_team_alerts(self) -> dict[str, list[PlayerSignal]]:
        """
        Get signals grouped by team for alert display.

        Returns:
            Dict mapping team → list of impactful signals
        """
        alerts: dict[str, list[PlayerSignal]] = {}
        for sig in self.get_most_impactful_signals():
            if not sig.team:
                continue
            if sig.team not in alerts:
                alerts[sig.team] = []
            alerts[sig.team].append(sig)
        return alerts

    def get_summary_stats(self) -> dict:
        """Get summary statistics about collected signals."""
        active = [s for s in self._signals if s.is_active()]
        by_type: dict[str, int] = {}
        by_team: dict[str, int] = {}
        by_confidence: dict[str, int] = {}

        for sig in active:
            by_type[sig.signal_type] = by_type.get(sig.signal_type, 0) + 1
            by_team[sig.team] = by_team.get(sig.team, 0) + 1
            by_confidence[sig.confidence.value] = by_confidence.get(sig.confidence.value, 0) + 1

        return {
            "total_collected": len(self._signals),
            "active_signals": len(active),
            "by_type": by_type,
            "by_team": by_team,
            "by_confidence": by_confidence,
            "freshness_seconds": (
                int(time.time() - self._last_collect_time)
                if self._last_collect_time else None
            ),
            "nitter_available": self.scraper._working_instance is not None,
        }

    def integrate_player_props(self, props: list) -> list:
        """Convenience: collect then integrate."""
        self.collect_all()
        return self.integrator.integrate_player_props(props, self._signals)

    def integrate_team_bets(self, bets: list) -> list:
        """Convenience: collect then integrate."""
        self.collect_all()
        return self.integrator.integrate_team_bets(bets, self._signals)

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(signals: list[PlayerSignal]) -> list[PlayerSignal]:
        """Deduplicate signals by (player, signal_type), keeping most severe."""
        seen: dict[tuple[str, str], PlayerSignal] = {}
        for sig in signals:
            key = (sig.player_name.lower(), sig.signal_type)
            if key not in seen:
                seen[key] = sig
            else:
                # Keep the more severe signal
                existing = seen[key]
                if sig.minutes_multiplier < existing.minutes_multiplier:
                    seen[key] = sig
        return list(seen.values())

    def _merge_signals(self, new_signals: list[PlayerSignal]) -> list[PlayerSignal]:
        """Merge new signals with existing ones, keeping newer ones."""
        merged: dict[tuple[str, str], PlayerSignal] = {}

        # Add existing signals first
        for sig in self._signals:
            key = (sig.player_name.lower(), sig.signal_type)
            merged[key] = sig

        # Override with newer signals
        for sig in new_signals:
            key = (sig.player_name.lower(), sig.signal_type)
            merged[key] = sig

        return list(merged.values())


