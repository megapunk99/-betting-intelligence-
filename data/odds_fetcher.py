"""
Live odds integration: fetches upcoming NBA games and market odds from TheOddsAPI.
Provides team name normalization and game-level data for the prediction engine.

KEY IMPROVEMENT: Stores odds from ALL sportsbooks, computes consensus (median)
lines across books, tracks per-book line dispersion, and identifies the best
available line for each side. This replaces the old 'first book wins' approach.

Usage:
    client = OddsAPIClient(api_key="your_key")
    games = client.get_upcoming_games_with_odds()
    for g in games:
        print(g.matchup, g.consensus.total_over_under)
        print(f"  Books offering ML: {g.consensus.home_ml_n_books}")
        print(f"  ML range: {g.consensus.home_ml_low} to {g.consensus.home_ml_high}")
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
import warnings
import statistics
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config import ODDS_API_KEY, ODDS_API_BASE_URL, CACHE_DIR


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BookOdds:
    """Odds from a single sportsbook for a single game."""
    book_key: str
    book_title: str
    last_update: str
    home_moneyline: Optional[float] = None
    away_moneyline: Optional[float] = None
    home_spread: Optional[float] = None
    home_spread_odds: Optional[float] = None
    away_spread: Optional[float] = None
    away_spread_odds: Optional[float] = None
    total_over: Optional[float] = None
    total_over_odds: Optional[float] = None
    total_under: Optional[float] = None
    total_under_odds: Optional[float] = None


@dataclass
class ConsensusOdds:
    """
    Consensus market data aggregated across ALL sportsbooks.

    For each market (ML, totals, spread), we compute:
      - consensus / sharp: median line value
      - n_books: how many books offer this market
      - low / high: min / max line value across books (the range)
      - std: standard deviation of line values (low = high agreement)
    """

    # ── Moneyline Consensus ─────────────────────────────────────────
    home_ml_consensus: Optional[float] = None   # median home moneyline
    away_ml_consensus: Optional[float] = None   # median away moneyline
    home_ml_n_books: int = 0
    away_ml_n_books: int = 0
    home_ml_low: Optional[float] = None
    home_ml_high: Optional[float] = None
    away_ml_low: Optional[float] = None
    away_ml_high: Optional[float] = None
    home_ml_std: Optional[float] = None
    away_ml_std: Optional[float] = None

    # No-vig implied probabilities from consensus lines
    consensus_home_win_prob: Optional[float] = None
    consensus_away_win_prob: Optional[float] = None

    # Best lines available (for finding +EV against consensus)
    best_home_ml: Optional[float] = None
    best_away_ml: Optional[float] = None
    best_home_ml_book: str = ""
    best_away_ml_book: str = ""

    # ── Totals Consensus ───────────────────────────────────────────
    total_consensus: Optional[float] = None     # median total line
    total_n_books: int = 0
    total_low: Optional[float] = None
    total_high: Optional[float] = None
    total_std: Optional[float] = None
    total_over_odds_consensus: Optional[float] = None
    total_under_odds_consensus: Optional[float] = None

    best_total_over: Optional[float] = None
    best_total_under: Optional[float] = None
    best_total_book: str = ""

    # ── Spread Consensus ───────────────────────────────────────────
    spread_consensus: Optional[float] = None    # median home spread
    spread_n_books: int = 0
    spread_low: Optional[float] = None
    spread_high: Optional[float] = None
    spread_std: Optional[float] = None
    spread_home_odds_consensus: Optional[float] = None

    best_spread: Optional[float] = None
    best_spread_book: str = ""


def _median_ignore_none(values: list) -> Optional[float]:
    """Median of a list that may contain None values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def _mean_ignore_none(values: list) -> Optional[float]:
    """Mean of a list that may contain None values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _std_ignore_none(values: list) -> Optional[float]:
    """Std dev of a list that may contain None values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return float(np.std(clean, ddof=1))


def _min_ignore_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return min(clean) if clean else None


def _max_ignore_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return max(clean) if clean else None


def _best_ml(home_or_away: str, values: list, books: list) -> Tuple[Optional[float], str]:
    """
    Find the best moneyline price for a side.
    For the team we're betting ON, we want the highest (least negative / most positive) price.
    The best book is the one offering that price.
    """
    best_price = None
    best_book = ""
    for price, book in zip(values, books):
        if price is None:
            continue
        if best_price is None or price > best_price:
            best_price = price
            best_book = book
    return (best_price, best_book)


def _compute_moneyline_consensus(books: List[BookOdds]) -> Dict:
    """Aggregate moneyline odds across all books into consensus stats."""
    home_mls = [b.home_moneyline for b in books]
    away_mls = [b.away_moneyline for b in books]
    titles = [b.book_title for b in books]

    consensus = {
        "home_ml_consensus": _median_ignore_none(home_mls),
        "away_ml_consensus": _median_ignore_none(away_mls),
        "home_ml_n_books": sum(1 for v in home_mls if v is not None),
        "away_ml_n_books": sum(1 for v in away_mls if v is not None),
        "home_ml_low": _min_ignore_none(home_mls),
        "home_ml_high": _max_ignore_none(home_mls),
        "away_ml_low": _min_ignore_none(away_mls),
        "away_ml_high": _max_ignore_none(away_mls),
        "home_ml_std": _std_ignore_none(home_mls),
        "away_ml_std": _std_ignore_none(away_mls),
    }

    # Best line for each side
    best_home_ml, best_home_book = _best_ml("home", home_mls, titles)
    best_away_ml, best_away_book = _best_ml("away", away_mls, titles)
    consensus["best_home_ml"] = best_home_ml
    consensus["best_home_ml_book"] = best_home_book
    consensus["best_away_ml"] = best_away_ml
    consensus["best_away_ml_book"] = best_away_book

    # No-vig probabilities from consensus lines
    con_home = consensus["home_ml_consensus"]
    con_away = consensus["away_ml_consensus"]
    if con_home is not None and con_away is not None:
        def ml_to_prob(odds):
            if odds > 0:
                return 100.0 / (odds + 100.0)
            else:
                return abs(odds) / (abs(odds) + 100.0)
        home_p = ml_to_prob(con_home)
        away_p = ml_to_prob(con_away)
        total_p = home_p + away_p
        if total_p > 0:
            consensus["consensus_home_win_prob"] = home_p / total_p
            consensus["consensus_away_win_prob"] = away_p / total_p

    return consensus


def _compute_totals_consensus(books: List[BookOdds]) -> Dict:
    """Aggregate totals across all books."""
    overs = [b.total_over for b in books]
    unders = [b.total_under for b in books]
    over_odds = [b.total_over_odds for b in books]
    under_odds = [b.total_under_odds for b in books]
    titles = [b.book_title for b in books]

    # The total line per book: average of over and under (should be same point)
    per_book_totals = []
    valid_books = []
    for i, b in enumerate(books):
        if b.total_over is not None and b.total_under is not None and b.total_over == b.total_under:
            per_book_totals.append(float(b.total_over))
            valid_books.append(b.book_title)

    results = {
        "total_consensus": _median_ignore_none(per_book_totals),
        "total_n_books": len(per_book_totals),
        "total_low": _min_ignore_none(per_book_totals),
        "total_high": _max_ignore_none(per_book_totals),
        "total_std": _std_ignore_none(per_book_totals),
        "total_over_odds_consensus": _median_ignore_none(over_odds),
        "total_under_odds_consensus": _median_ignore_none(under_odds),
    }

    # Best total (highest over = best for OVER bettor, lowest under = best for UNDER)
    # For OVER bet: higher total is better (more points needed to hit, but better odds)
    # For UNDER bet: lower total is better
    if per_book_totals and valid_books:
        best_idx = per_book_totals.index(max(per_book_totals))
        results["best_total_over"] = per_book_totals[best_idx]
        results["best_total_book"] = valid_books[best_idx]
        best_idx_under = per_book_totals.index(min(per_book_totals))
        results["best_total_under"] = per_book_totals[best_idx_under]
    else:
        results["best_total_over"] = None
        results["best_total_under"] = None
        results["best_total_book"] = ""

    return results


def _compute_spread_consensus(books: List[BookOdds]) -> Dict:
    """Aggregate spreads across all books."""
    spreads = [b.home_spread for b in books]
    spread_odds = [b.home_spread_odds for b in books]

    results = {
        "spread_consensus": _median_ignore_none(spreads),
        "spread_n_books": sum(1 for v in spreads if v is not None),
        "spread_low": _min_ignore_none(spreads),
        "spread_high": _max_ignore_none(spreads),
        "spread_std": _std_ignore_none(spreads),
        "spread_home_odds_consensus": _median_ignore_none(spread_odds),
    }

    # Best spread: highest (most favorable) for each side
    titles = [b.book_title for b in books]
    clean_spreads = [(s, titles[i]) for i, s in enumerate(spreads) if s is not None]
    if clean_spreads:
        best_spread, best_book = max(clean_spreads, key=lambda x: x[0])
        results["best_spread"] = best_spread
        results["best_spread_book"] = best_book
    else:
        results["best_spread"] = None
        results["best_spread_book"] = ""

    return results


@dataclass
class OddsGame:
    """
    A single upcoming NBA game with FULL multi-sportsbook market odds.

    KEY CHANGE from v1: Stores odds from ALL books in `all_books`, and
    computes a `consensus` field that aggregates across all books.
    The legacy fields (home_moneyline, market_total, etc.) now come from
    the consensus (median) line, not from whichever book appeared first.
    """

    id: str
    sport_key: str
    sport_title: str
    commence_time: str  # ISO 8601
    home_team: str
    away_team: str
    home_team_short: str = ""
    away_team_short: str = ""

    # ── Legacy fields: now populated from consensus median ──────────
    home_moneyline: Optional[float] = None
    away_moneyline: Optional[float] = None
    home_spread: Optional[float] = None
    home_spread_odds: Optional[float] = None
    away_spread: Optional[float] = None
    away_spread_odds: Optional[float] = None
    total_over: Optional[float] = None
    total_under: Optional[float] = None
    total_over_odds: Optional[float] = None
    total_under_odds: Optional[float] = None

    implied_home_win_prob: Optional[float] = None
    market_total: Optional[float] = None
    vig_free_total: Optional[float] = None

    # ── NEW: Per-book and consensus data ────────────────────────────
    all_books: List[BookOdds] = field(default_factory=list)
    consensus: Optional[ConsensusOdds] = None

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    @property
    def commence_datetime(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.commence_time.replace("Z", "+00:00"))
        except Exception:
            return None

    @property
    def is_today(self) -> bool:
        dt = self.commence_datetime
        if dt:
            return dt.date() == datetime.now(timezone.utc).date()
        return False

    @property
    def is_tomorrow(self) -> bool:
        dt = self.commence_datetime
        if dt:
            return dt.date() == (datetime.now(timezone.utc) + timedelta(days=1)).date()
        return False

    def compute_implied_probs(self):
        """Compute vig-free implied win probabilities from CONSENSUS lines."""
        if self.home_moneyline and self.away_moneyline:
            def moneyline_to_prob(odds):
                if odds > 0:
                    return 100 / (odds + 100)
                else:
                    return abs(odds) / (abs(odds) + 100)

            home_imp = moneyline_to_prob(self.home_moneyline)
            away_imp = moneyline_to_prob(self.away_moneyline)
            total_imp = home_imp + away_imp
            self.implied_home_win_prob = home_imp / total_imp if total_imp > 0 else 0.5

        if self.total_over and self.total_under:
            self.market_total = (self.total_over + self.total_under) / 2

    def to_dict(self) -> dict:
        """Serialise to JSON, handling nested dataclasses."""
        d = asdict(self)
        # Ensure nested dataclasses are also serialized
        if self.consensus:
            d["consensus"] = asdict(self.consensus)
        else:
            d["consensus"] = None
        d["all_books"] = [asdict(b) for b in self.all_books]
        return d

    def get_book_summary(self) -> str:
        """Return a human-readable summary of how many books offer each market."""
        if not self.consensus:
            return "No consensus data"
        c = self.consensus
        lines = []
        lines.append(f"ML: {c.home_ml_n_books} books (range {c.home_ml_low} to {c.home_ml_high})")
        if c.total_n_books:
            lines.append(f"Total: {c.total_n_books} books (range {c.total_low} to {c.total_high})")
        if c.spread_n_books:
            lines.append(f"Spread: {c.spread_n_books} books (range {c.spread_low} to {c.spread_high})")
        return " | ".join(lines)

    def __str__(self) -> str:
        dt = self.commence_datetime
        time_str = dt.strftime("%a %I:%M %p ET") if dt else self.commence_time
        total_str = f"O/U {self.market_total:.1f}" if self.market_total else "No total"
        n_books = self.consensus.home_ml_n_books if self.consensus else 0
        spread_str = f"Spread: {self.home_team_short} {self.home_spread:+.0f}" if self.home_spread is not None else ""
        return f"{self.matchup:45s} | {time_str:20s} | {total_str:12s} | {spread_str} | {n_books} books"


# ── Team Name Mapping ───────────────────────────────────────────────────────

ODDS_TO_SHORT_NAME: Dict[str, str] = {
    "Atlanta Hawks": "Hawks",
    "Boston Celtics": "Celtics",
    "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets",
    "Chicago Bulls": "Bulls",
    "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks",
    "Denver Nuggets": "Nuggets",
    "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors",
    "Houston Rockets": "Rockets",
    "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers",
    "Los Angeles Clippers": "Clippers",
    "LA Lakers": "Lakers",
    "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder",
    "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers",
    "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers",
    "Portland Trailblazers": "Trail Blazers",
    "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs",
    "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz",
    "Washington Wizards": "Wizards",
}

SHORT_TO_ODDS_NAME: Dict[str, str] = {short: full for full, short in ODDS_TO_SHORT_NAME.items()}

SHORT_NAME_TO_TEAM_ID: Dict[str, int] = {
    "Hawks": 1610612737, "Celtics": 1610612738, "Nets": 1610612751,
    "Hornets": 1610612766, "Bulls": 1610612741, "Cavaliers": 1610612739,
    "Mavericks": 1610612742, "Nuggets": 1610612743, "Pistons": 1610612765,
    "Warriors": 1610612744, "Rockets": 1610612745, "Pacers": 1610612754,
    "Clippers": 1610612746, "Lakers": 1610612747, "Grizzlies": 1610612763,
    "Heat": 1610612748, "Bucks": 1610612749, "Timberwolves": 1610612750,
    "Pelicans": 1610612740, "Knicks": 1610612752, "Thunder": 1610612760,
    "Magic": 1610612753, "76ers": 1610612755, "Suns": 1610612756,
    "Trail Blazers": 1610612757, "Kings": 1610612758, "Spurs": 1610612759,
    "Raptors": 1610612761, "Jazz": 1610612762, "Wizards": 1610612764,
}


# ═══════════════════════════════════════════════════════════════════════════
#  ODDS API CLIENT
# ═══════════════════════════════════════════════════════════════════════════


class OddsAPIClient:
    """
    Client for TheOddsAPI v4.
    Fetches upcoming NBA games with odds from ALL available sportsbooks,
    computes consensus lines, and tracks best lines per side.

    Usage:
        client = OddsAPIClient(api_key="your_key_here")
        games = client.get_upcoming_games_with_odds()
        for g in games:
            print(g)
            print(f"  (aggregated from {g.consensus.home_ml_n_books} books)")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = ODDS_API_BASE_URL,
        cache_dir: Optional[Path] = None,
        cache_ttl_minutes: int = 15,
    ):
        self.api_key = api_key or ODDS_API_KEY
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else (CACHE_DIR / "odds")
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key or self.api_key == "" or self.api_key == "your-api-key-here":
            print("  [!] No OddsAPI key set. Set ODDS_API_KEY in config.py or env.")
            print(f"     Get a free key at: https://the-odds-api.com/\n")
            self._configured = False
        else:
            self._configured = True

    def get_upcoming_games_with_odds(
        self,
        sport: str = "basketball_nba",
        regions: str = "us,us2,eu,uk,au",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        ignore_live: bool = True,
        use_cache: bool = True,
    ) -> List[OddsGame]:
        """
        Fetch upcoming NBA games with odds from ALL sportsbooks.

        Args:
            sport: Sport key (default: basketball_nba)
            regions: Bookmaker regions. Default queries MULTIPLE regions
                     to capture as many books as possible.
            markets: Markets to fetch (default: h2h,spreads,totals)
            odds_format: american or decimal
            date_format: iso or unix
            ignore_live: If True, filter out games that have already started
            use_cache: Use cached results if fresh enough

        Returns:
            List of OddsGame objects with consensus and per-book odds
        """
        if not self._configured:
            return []

        cache_key = f"odds_{sport}_{markets.replace(',', '_')}_{regions.replace(',', '_')}"
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                print(f"  [OddsAPI] Using cached odds from {len(cached)} games (TTL: {self.cache_ttl})")
                return cached

        url = f"{self.base_url}/v4/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }

        try:
            print(f"  [OddsAPI] Fetching {sport.upper()} odds from regions: {regions}...")
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            print(f"  [OddsAPI] Quota: {remaining} remaining, {used} used")

            data = resp.json()
            games = self._parse_odds_response(data, ignore_live=ignore_live)
            print(f"  [OddsAPI] {len(games)} games with odds from multiple sportsbooks")

            self._save_cache(cache_key, games)
            return games

        except ImportError:
            print("  [OddsAPI] 'requests' library not installed. Install with: pip install requests")
            return []
        except requests.exceptions.RequestException as e:
            print(f"  [OddsAPI] API request failed: {e}")
            stale = self._load_cache(cache_key, ignore_ttl=True)
            if stale:
                print(f"  [OddsAPI] Using stale cache ({len(stale)} games)")
                return stale
            return []

    def get_upcoming_games(self, sport: str = "basketball_nba", **kwargs) -> List[OddsGame]:
        """Alias for get_upcoming_games_with_odds."""
        return self.get_upcoming_games_with_odds(sport=sport, **kwargs)

    def get_game_by_matchup(self, home_team: str, away_team: str, **kwargs) -> Optional[OddsGame]:
        """Find odds for a specific matchup."""
        games = self.get_upcoming_games_with_odds(**kwargs)
        for g in games:
            if home_team.lower() in g.home_team.lower() and away_team.lower() in g.away_team.lower():
                return g
            if home_team.lower() in g.away_team.lower() and away_team.lower() in g.home_team.lower():
                return g
        return None

    def get_team_id_map(self, db_teams_df) -> Dict[str, int]:
        """Build a mapping from short team names to database team IDs."""
        team_map = {}
        if "TEAM_NAME" in db_teams_df.columns:
            unique_teams = db_teams_df[["TEAM_NAME", "TEAM_ID"]].drop_duplicates()
            for _, row in unique_teams.iterrows():
                short = str(row["TEAM_NAME"]).strip()
                team_map[short] = int(row["TEAM_ID"])
        for short, tid in SHORT_NAME_TO_TEAM_ID.items():
            if short not in team_map:
                team_map[short] = tid
        return team_map

    # ═══════════════════════════════════════════════════════════════════
    #  INTERNAL: Parse & Normalize
    # ═══════════════════════════════════════════════════════════════════

    def _parse_odds_response(self, data: list, ignore_live: bool = True) -> List[OddsGame]:
        """Parse the raw JSON response into OddsGame objects with multi-book data."""
        games = []
        now_utc = datetime.now(timezone.utc)

        for event in data:
            if ignore_live:
                commence_str = event.get("commence_time", "")
                try:
                    commence = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                    if commence < now_utc:
                        continue
                except Exception:
                    pass

            home = event.get("home_team", "?")
            away = event.get("away_team", "?")
            home_short = ODDS_TO_SHORT_NAME.get(home, home.split()[-1] if " " in home else home)
            away_short = ODDS_TO_SHORT_NAME.get(away, away.split()[-1] if " " in away else away)

            game = OddsGame(
                id=event.get("id", ""),
                sport_key=event.get("sport_key", ""),
                sport_title=event.get("sport_title", ""),
                commence_time=event.get("commence_time", ""),
                home_team=home,
                away_team=away,
                home_team_short=home_short,
                away_team_short=away_short,
            )

            # Parse ALL bookmakers into BookOdds, then compute consensus
            self._extract_multi_book_odds(game, event.get("bookmakers", []))
            self._compute_consensus(game)

            # Legacy fields populated from consensus
            if game.consensus:
                c = game.consensus
                game.home_moneyline = c.home_ml_consensus
                game.away_moneyline = c.away_ml_consensus
                game.home_spread = c.spread_consensus
                if c.spread_home_odds_consensus is not None:
                    game.home_spread_odds = c.spread_home_odds_consensus
                game.total_over = c.total_consensus
                game.total_under = c.total_consensus
                if c.total_over_odds_consensus is not None:
                    game.total_over_odds = c.total_over_odds_consensus
                if c.total_under_odds_consensus is not None:
                    game.total_under_odds = c.total_under_odds_consensus

            game.compute_implied_probs()
            games.append(game)

        return games

    def _extract_multi_book_odds(self, game: OddsGame, bookmakers: list):
        """
        Extract odds from ALL bookmakers and store them as BookOdds objects.
        Unlike the old v1 `_extract_odds` which only kept the first book's line,
        this stores every book's data so we can compute proper consensus.
        """
        per_book_data = []

        for book in bookmakers:
            bk = book.get("key", "")
            title = book.get("title", bk)
            last_update = book.get("last_update", "")

            book_odds = BookOdds(
                book_key=bk,
                book_title=title,
                last_update=last_update,
            )

            for market in book.get("markets", []):
                key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if key == "h2h":
                    for o in outcomes:
                        name = o.get("name", "")
                        price = o.get("price")
                        if name == game.home_team:
                            book_odds.home_moneyline = price
                        elif name == game.away_team:
                            book_odds.away_moneyline = price

                elif key == "spreads":
                    for o in outcomes:
                        name = o.get("name", "")
                        point = o.get("point")
                        price = o.get("price")
                        if name == game.home_team:
                            book_odds.home_spread = point
                            book_odds.home_spread_odds = price
                        elif name == game.away_team:
                            book_odds.away_spread = point
                            book_odds.away_spread_odds = price

                elif key == "totals":
                    for o in outcomes:
                        name = o.get("name", "")
                        point = o.get("point")
                        price = o.get("price")
                        if name == "Over":
                            book_odds.total_over = point
                            book_odds.total_over_odds = price
                        elif name == "Under":
                            book_odds.total_under = point
                            book_odds.total_under_odds = price

            # Only add if this book actually has odds
            has_any = any([
                book_odds.home_moneyline is not None,
                book_odds.away_moneyline is not None,
                book_odds.home_spread is not None,
                book_odds.away_spread is not None,
                book_odds.total_over is not None,
            ])
            if has_any:
                per_book_data.append(book_odds)

        game.all_books = per_book_data

    def _compute_consensus(self, game: OddsGame):
        """Compute consensus odds across all stored books."""
        if not game.all_books:
            return

        ml_consensus = _compute_moneyline_consensus(game.all_books)
        totals_consensus = _compute_totals_consensus(game.all_books)
        spread_consensus = _compute_spread_consensus(game.all_books)

        game.consensus = ConsensusOdds(
            **ml_consensus,
            **totals_consensus,
            **spread_consensus,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  CACHING
    # ═══════════════════════════════════════════════════════════════════

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str, ignore_ttl: bool = False) -> Optional[List[OddsGame]]:
        """Load cached odds if not expired."""
        cache_path = self._cache_path(key)
        if not cache_path.exists():
            return None

        if not ignore_ttl:
            mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mod_time > self.cache_ttl:
                return None

        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            games = []
            for g in raw:
                # Hydrate nested BookOdds and ConsensusOdds
                books_raw = g.pop("all_books", [])
                consensus_raw = g.pop("consensus", None)
                game = OddsGame(**g)
                game.all_books = [BookOdds(**b) for b in books_raw]
                if consensus_raw:
                    game.consensus = ConsensusOdds(**consensus_raw)
                game.compute_implied_probs()
                games.append(game)
            return games
        except Exception:
            return None

    def _save_cache(self, key: str, games: List[OddsGame]):
        """Save odds to cache."""
        cache_path = self._cache_path(key)
        try:
            with open(cache_path, "w") as f:
                json.dump([g.to_dict() for g in games], f, indent=2, default=str)
        except Exception as e:
            print(f"  [OddsAPI] Cache write failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  UTILITY: Feature Engineering
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def build_feature_row_for_game(
        game: OddsGame,
        historical_df,
        feature_cols: list,
    ) -> Optional[Dict]:
        """Build a feature vector for an upcoming game using historical data."""
        if historical_df is None or historical_df.empty:
            return None

        home_short = game.home_team_short
        away_short = game.away_team_short
        home_id = SHORT_NAME_TO_TEAM_ID.get(home_short)
        away_id = SHORT_NAME_TO_TEAM_ID.get(away_short)

        if not home_id or not away_id:
            print(f"  [OddsAPI] Unknown team IDs for {home_short} vs {away_short}")
            return None

        home_games = historical_df[
            (historical_df.get("TEAM_ID_home") == home_id) |
            (historical_df.get("TEAM_ID_away") == home_id)
        ].sort_values("GAME_DATE").tail(20)

        away_games = historical_df[
            (historical_df.get("TEAM_ID_home") == away_id) |
            (historical_df.get("TEAM_ID_away") == away_id)
        ].sort_values("GAME_DATE").tail(20)

        if len(home_games) < 5 or len(away_games) < 5:
            return None

        last_home = home_games.iloc[-1]
        last_away = away_games.iloc[-1]

        recent_home_home = historical_df[
            (historical_df["TEAM_ID_home"] == home_id)
        ].sort_values("GAME_DATE").tail(10)

        recent_away_away = historical_df[
            (historical_df["TEAM_ID_away"] == away_id)
        ].sort_values("GAME_DATE").tail(10)

        if len(recent_home_home) < 3 or len(recent_away_away) < 3:
            recent_home_home = home_games.tail(5)
            recent_away_away = away_games.tail(5)

        best_match = None
        for idx in range(len(historical_df) - 1, max(0, len(historical_df) - 500), -1):
            row = historical_df.iloc[idx]
            if row.get("TEAM_ID_home") == home_id and row.get("TEAM_ID_away") == away_id:
                best_match = row
                break

        if best_match is not None:
            feature_row = {col: best_match.get(col, 0) for col in feature_cols}
        else:
            feature_row = {}
            for col in feature_cols:
                feature_row[col] = 0
            for col in feature_cols:
                if col.endswith("_home") and not col.startswith("TEAM_"):
                    if not recent_home_home.empty and col in recent_home_home.columns:
                        feature_row[col] = recent_home_home[col].iloc[-1]
                elif col.endswith("_away") and not col.startswith("TEAM_"):
                    if not recent_away_away.empty and col in recent_away_away.columns:
                        feature_row[col] = recent_away_away[col].iloc[-1]
            feature_series = pd.Series(feature_row)
            feature_series = feature_series.fillna(0)
            feature_row = feature_series.to_dict()

        return feature_row


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def format_american_odds(odds: Optional[float]) -> str:
    if odds is None:
        return "N/A"
    if odds > 0:
        return f"+{odds:.0f}"
    return f"{odds:.0f}"


def format_implied_prob(moneyline_odds: Optional[float]) -> Optional[float]:
    if moneyline_odds is None:
        return None
    if moneyline_odds > 0:
        return 100 / (moneyline_odds + 100)
    return abs(moneyline_odds) / (abs(moneyline_odds) + 100)


def display_odds_card(games: List[OddsGame], title: str = "UPCOMING NBA GAMES"):
    """Pretty-print odds for upcoming games with multi-book info."""
    if not games:
        print("No upcoming games found.")
        return

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"  Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}")

    for i, g in enumerate(games, 1):
        dt_str = g.commence_datetime.strftime("%a %b %d, %I:%M %p ET") if g.commence_datetime else "TBD"
        total_str = f"{g.market_total:.1f}" if g.market_total else "N/A"
        spread_str = f"{g.home_spread:+.0f}" if g.home_spread is not None else "N/A"
        ml_str = f"{format_american_odds(g.home_moneyline)} / {format_american_odds(g.away_moneyline)}"
        book_summary = g.get_book_summary() if g.consensus else ""

        print(f"\n  [{i}] {g.away_team} @ {g.home_team}")
        print(f"       {dt_str}")
        print(f"       Moneyline: {ml_str:30s} | Spread: {spread_str:>4s} | Total: {total_str}")
        if g.implied_home_win_prob:
            print(f"       Implied Home Win: {g.implied_home_win_prob:.1%}")
        if book_summary:
            print(f"       {book_summary}")

    print(f"\n  Total: {len(games)} games")
    print(f"{'=' * 100}\n")
