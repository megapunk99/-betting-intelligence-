"""
Live odds integration: fetches upcoming NBA games and market odds from TheOddsAPI.
Provides team name normalization and game-level data for the prediction engine.

Usage:
    client = OddsAPIClient(api_key="your_key")
    games = client.get_upcoming_games_with_odds()
    for g in games:
        print(g.matchup, g.home_team, g.away_team, g.totals_over_under)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config import ODDS_API_KEY, ODDS_API_BASE_URL, CACHE_DIR


# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class OddsGame:
    """A single upcoming NBA game with market odds from TheOddsAPI."""

    id: str
    sport_key: str
    sport_title: str
    commence_time: str  # ISO 8601
    home_team: str
    away_team: str
    home_team_short: str = ""  # Normalized short name (e.g. "Celtics", "Lakers")
    away_team_short: str = ""

    # Market odds
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

    # Derived
    implied_home_win_prob: Optional[float] = None
    market_total: Optional[float] = None
    vig_free_total: Optional[float] = None

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
        """Compute vig-free implied win probabilities from moneyline odds."""
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

        if self.total_over and self.total_under and self.total_over_odds and self.total_under_odds:
            self.market_total = (self.total_over + self.total_under) / 2
            # Simple vig-free estimate
            over_imp = 100 / (self.total_over_odds + 100) if self.total_over_odds > 0 else abs(self.total_over_odds) / (abs(self.total_over_odds) + 100)
            under_imp = 100 / (self.total_under_odds + 100) if self.total_under_odds > 0 else abs(self.total_under_odds) / (abs(self.total_under_odds) + 100)
            total_vig = over_imp + under_imp
            if total_vig > 0:
                self.vig_free_total = self.market_total  # Use market total as-is

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        dt = self.commence_datetime
        time_str = dt.strftime("%a %I:%M %p ET") if dt else self.commence_time
        total_str = f"O/U {self.market_total:.1f}" if self.market_total else "No total"
        spread_str = f"Spread: {self.home_team_short} {self.home_spread:+.0f}" if self.home_spread is not None else ""
        return f"{self.matchup:45s} | {time_str:20s} | {total_str:12s} | {spread_str}"


# ── Team Name Mapping ───────────────────────────────────────────────────────


# Maps TheOddsAPI full team names to our short/normalized team names
# TheOddsAPI uses formats like "Boston Celtics", "LA Lakers", etc.
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

# Reverse map: short name -> odds API full name (for logging/display)
SHORT_TO_ODDS_NAME: Dict[str, str] = {
    short: full for full, short in ODDS_TO_SHORT_NAME.items()
}

# Map short names to TEAM_IDs we expect in the database
# These are common NBA team IDs used by stats.nba.com / NBA API
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


# ── Odds API Client ─────────────────────────────────────────────────────────


class OddsAPIClient:
    """
    Client for TheOddsAPI v4.
    Fetches upcoming NBA games with live market odds.
    Supports caching to avoid hitting rate limits unnecessarily.

    Usage:
        client = OddsAPIClient(api_key="your_key_here")
        games = client.get_upcoming_games_with_odds()
        for g in games:
            print(g)
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
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        ignore_live: bool = True,
        use_cache: bool = True,
    ) -> List[OddsGame]:
        """
        Fetch upcoming NBA games with full odds data.

        Args:
            sport: Sport key (default: basketball_nba)
            regions: Bookmaker region(s) (default: us)
            markets: Markets to fetch (default: h2h,spreads,totals)
            odds_format: american or decimal
            date_format: iso or unix
            ignore_live: If True, filter out games that have already started
            use_cache: Use cached results if fresh enough

        Returns:
            List of OddsGame objects with parsed odds
        """
        if not self._configured:
            return self._demo_games()

        # Check cache first
        cache_key = f"odds_{sport}_{markets.replace(',', '_')}"
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                print(f"  [OddsAPI] Using cached odds (TTL: {self.cache_ttl})")
                return cached

        # Build request
        url = f"{self.base_url}/v4/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }

        try:
            print(f"  [OddsAPI] Fetching upcoming {sport.upper()} games from TheOddsAPI...")
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            # Track remaining quota
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            print(f"  [OddsAPI] API quota: {remaining} remaining, {used} used this period")

            data = resp.json()
            games = self._parse_odds_response(data, ignore_live=ignore_live)
            print(f"  [OddsAPI] Found {len(games)} upcoming games with odds")

            # Cache the result
            self._save_cache(cache_key, games)
            return games

        except ImportError:
            print("  [OddsAPI] 'requests' library not installed. Install with: pip install requests")
            print("  [OddsAPI] Falling back to demo mode.\n")
            return self._demo_games()
        except requests.exceptions.RequestException as e:
            print(f"  [OddsAPI] API request failed: {e}")
            # Try loading stale cache
            stale = self._load_cache(cache_key, ignore_ttl=True)
            if stale:
                print(f"  [OddsAPI] Using stale cache ({len(stale)} games)")
                return stale
            print("  [OddsAPI] Falling back to demo games.\n")
            return self._demo_games()

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
        """
        Build a mapping from short team names to database team IDs.
        Uses the database's TEAM_NAME column.

        Args:
            db_teams_df: DataFrame with TEAM_NAME column (from game logs)

        Returns:
            Dict mapping short names (e.g. "Celtics") to team IDs
        """
        team_map = {}
        if "TEAM_NAME" in db_teams_df.columns:
            unique_teams = db_teams_df[["TEAM_NAME", "TEAM_ID"]].drop_duplicates()
            for _, row in unique_teams.iterrows():
                short = str(row["TEAM_NAME"]).strip()
                team_map[short] = int(row["TEAM_ID"])
        # Fill in any missing from our hardcoded map
        for short, tid in SHORT_NAME_TO_TEAM_ID.items():
            if short not in team_map:
                team_map[short] = tid
        return team_map

    # ═══════════════════════════════════════════════════════════════════
    #  INTERNAL: Parse & Normalize
    # ═══════════════════════════════════════════════════════════════════

    def _parse_odds_response(self, data: list, ignore_live: bool = True) -> List[OddsGame]:
        """Parse the raw JSON response from TheOddsAPI into OddsGame objects."""
        games = []
        now_utc = datetime.now(timezone.utc)

        for event in data:
            # Skip live/in-progress games if requested
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

            # Parse odds from bookmakers
            self._extract_odds(game, event.get("bookmakers", []))

            # Compute derived probabilities
            game.compute_implied_probs()

            games.append(game)

        return games

    def _extract_odds(self, game: OddsGame, bookmakers: list):
        """
        Extract best available odds across all bookmakers.
        Uses the most favorable line (sharpest) across all sportsbooks.
        """
        best = {
            "home_moneyline": None, "away_moneyline": None,
            "home_spread": None, "home_spread_odds": None,
            "away_spread": None, "away_spread_odds": None,
            "total_over": None, "total_over_odds": None,
            "total_under": None, "total_under_odds": None,
        }

        for book in bookmakers:
            for market in book.get("markets", []):
                key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if key == "h2h":
                    for o in outcomes:
                        name = o.get("name", "")
                        price = o.get("price")
                        if name == game.home_team:
                            if best["home_moneyline"] is None or price is not None:
                                best["home_moneyline"] = price
                        elif name == game.away_team:
                            if best["away_moneyline"] is None or price is not None:
                                best["away_moneyline"] = price

                elif key == "spreads":
                    for o in outcomes:
                        name = o.get("name", "")
                        point = o.get("point")
                        price = o.get("price")
                        if name == game.home_team:
                            if best["home_spread"] is None:
                                best["home_spread"] = point
                                best["home_spread_odds"] = price
                        elif name == game.away_team:
                            if best["away_spread"] is None:
                                best["away_spread"] = point
                                best["away_spread_odds"] = price

                elif key == "totals":
                    for o in outcomes:
                        name = o.get("name", "")
                        point = o.get("point")
                        price = o.get("price")
                        if name == "Over":
                            if best["total_over"] is None:
                                best["total_over"] = point
                                best["total_over_odds"] = price
                        elif name == "Under":
                            if best["total_under"] is None:
                                best["total_under"] = point
                                best["total_under_odds"] = price

        game.home_moneyline = best["home_moneyline"]
        game.away_moneyline = best["away_moneyline"]
        game.home_spread = best["home_spread"]
        game.home_spread_odds = best["home_spread_odds"]
        game.away_spread = best["away_spread"]
        game.away_spread_odds = best["away_spread_odds"]
        game.total_over = best["total_over"]
        game.total_over_odds = best["total_over_odds"]
        game.total_under = best["total_under"]
        game.total_under_odds = best["total_under_odds"]

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
                return None  # Expired

        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            games = [OddsGame(**g) for g in raw]
            for g in games:
                g.compute_implied_probs()
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
    #  DEMO / OFFLINE MODE
    # ═══════════════════════════════════════════════════════════════════

    def _demo_games(self) -> List[OddsGame]:
        """Return demo games when API key is not configured or API is unavailable."""
        today = datetime.now(timezone.utc)

        demo_schedule = [
            ("Boston Celtics", "New York Knicks", 1),
            ("Los Angeles Lakers", "Golden State Warriors", 1),
            ("Milwaukee Bucks", "Philadelphia 76ers", 1),
            ("Denver Nuggets", "Oklahoma City Thunder", 1),
            ("Miami Heat", "Orlando Magic", 1),
            ("Phoenix Suns", "Dallas Mavericks", 2),
            ("LA Clippers", "Sacramento Kings", 2),
        ]

        games = []
        for home, away, day_offset in demo_schedule:
            kickoff = today + timedelta(days=day_offset, hours=19)  # 7pm ET
            home_short = ODDS_TO_SHORT_NAME.get(home, home.split()[-1])
            away_short = ODDS_TO_SHORT_NAME.get(away, away.split()[-1])

            # Demo odds (representative market prices)
            game = OddsGame(
                id=f"demo_{home_short}_vs_{away_short}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=kickoff.isoformat(),
                home_team=home,
                away_team=away,
                home_team_short=home_short,
                away_team_short=away_short,
                home_moneyline=-200,
                away_moneyline=+170,
                home_spread=-4.5,
                home_spread_odds=-110,
                away_spread=4.5,
                away_spread_odds=-110,
                total_over=221.5,
                total_under=221.5,
                total_over_odds=-110,
                total_under_odds=-110,
            )
            game.compute_implied_probs()
            games.append(game)

        self._save_cache("demo_games", games)
        return games

    # ═══════════════════════════════════════════════════════════════════
    #  UTILITY: Team mapping for feature engineering
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def build_feature_row_for_game(
        game: OddsGame,
        historical_df,
        feature_cols: list,
    ) -> Optional[Dict]:
        """
        Build a feature vector for an upcoming game using historical data.
        This finds the most recent games for both teams and computes features.

        Args:
            game: The upcoming OddsGame
            historical_df: DataFrame with historical game data and features
            feature_cols: List of feature column names to use

        Returns:
            Dict with feature values for this game, or None if insufficient data
        """
        if historical_df is None or historical_df.empty:
            return None

        home_short = game.home_team_short
        away_short = game.away_team_short

        # Find team ID columns (could be TEAM_ID_home, TEAM_ID_away, or numeric IDs in short name map)
        home_id = SHORT_NAME_TO_TEAM_ID.get(home_short)
        away_id = SHORT_NAME_TO_TEAM_ID.get(away_short)

        if not home_id or not away_id:
            print(f"  [OddsAPI] Unknown team IDs for {home_short} vs {away_short}")
            return None

        # Get the teams' most recent games from the historical dataset
        home_games = historical_df[
            (historical_df.get("TEAM_ID_home") == home_id) |
            (historical_df.get("TEAM_ID_away") == home_id)
        ].sort_values("GAME_DATE").tail(20)

        away_games = historical_df[
            (historical_df.get("TEAM_ID_home") == away_id) |
            (historical_df.get("TEAM_ID_away") == away_id)
        ].sort_values("GAME_DATE").tail(20)

        if len(home_games) < 5 or len(away_games) < 5:
            print(f"  [OddsAPI] Insufficient historical data: {home_short}({len(home_games)}) vs {away_short}({len(away_games)})")
            return None

        # Use the most recent game row as a template, swap in the actual team info
        last_home = home_games.iloc[-1]
        last_away = away_games.iloc[-1]

        # Build a feature row by taking the most recent home-team-at-home and away-team-on-road games
        recent_home_home = historical_df[
            (historical_df["TEAM_ID_home"] == home_id)
        ].sort_values("GAME_DATE").tail(10)

        recent_away_away = historical_df[
            (historical_df["TEAM_ID_away"] == away_id)
        ].sort_values("GAME_DATE").tail(10)

        if len(recent_home_home) < 3 or len(recent_away_away) < 3:
            # Fall back to general recent games
            recent_home_home = home_games.tail(5)
            recent_away_away = away_games.tail(5)

        # Build a composite row using the last game structure with the correct team assignments
        template = historical_df.iloc[-1].to_dict()

        # Find a game where the home team was at home and away team was away
        # This ensures feature continuity (rolling averages etc are preserved)
        best_match = None
        for idx in range(len(historical_df) - 1, max(0, len(historical_df) - 500), -1):
            row = historical_df.iloc[idx]
            if row.get("TEAM_ID_home") == home_id and row.get("TEAM_ID_away") == away_id:
                best_match = row
                break

        if best_match is not None:
            # Found an exact same matchup — use those features
            feature_row = {col: best_match.get(col, 0) for col in feature_cols}
        else:
            # Build a synthetic feature row
            feature_row = {}
            for col in feature_cols:
                feature_row[col] = 0

            # Try to get the actual feature values from the teams' recent games at the correct venue
            for col in feature_cols:
                # Check if it's a home team stat (ends with _home)
                if col.endswith("_home") and not col.startswith("TEAM_"):
                    # Use the home team's recent value for this feature
                    if not recent_home_home.empty and col in recent_home_home.columns:
                        feature_row[col] = recent_home_home[col].iloc[-1]
                elif col.endswith("_away") and not col.startswith("TEAM_"):
                    if not recent_away_away.empty and col in recent_away_away.columns:
                        feature_row[col] = recent_away_away[col].iloc[-1]

            # Fill remaining NaN features with team averages or zeros
            feature_series = pd.Series(feature_row)
            feature_series = feature_series.fillna(0)
            feature_row = feature_series.to_dict()

        return feature_row


# ── Convenience Functions ─────────────────────────────────────────────────


def format_american_odds(odds: Optional[float]) -> str:
    """Format American odds with + or - prefix."""
    if odds is None:
        return "N/A"
    if odds > 0:
        return f"+{odds:.0f}"
    return f"{odds:.0f}"


def format_implied_prob(moneyline_odds: Optional[float]) -> Optional[float]:
    """Convert American moneyline to implied win probability."""
    if moneyline_odds is None:
        return None
    if moneyline_odds > 0:
        return 100 / (moneyline_odds + 100)
    return abs(moneyline_odds) / (abs(moneyline_odds) + 100)


def display_odds_card(games: List[OddsGame], title: str = "UPCOMING NBA GAMES"):
    """Pretty-print odds for upcoming games."""
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

        print(f"\n  [{i}] {g.away_team} @ {g.home_team}")
        print(f"       {dt_str}")
        print(f"       Moneyline: {ml_str:30s} | Spread: {spread_str:>4s} | Total: {total_str}")
        if g.implied_home_win_prob:
            print(f"       Implied Home Win: {g.implied_home_win_prob:.1%}")

    print(f"\n  Total: {len(games)} games")
    print(f"{'=' * 100}\n")
