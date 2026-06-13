"""
MLB Data Source — fetches historical game data, starting pitcher info,
and team performance metrics from ESPN's free public API.

No API key required. Rate-limited to 2 requests/second.

Data sources:
  1. ESPN Scoreboard: https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard
  2. ESPN Athlete API: https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/athletes/{id}/statistics

Typical usage (training):
    from betting_intel.data.mlb_data import MLBDataSource
    source = MLBDataSource()
    df = source.load_historical_games(days_back=365)
    # df has: game_id, date, home_team, away_team, home_score, away_score,
    #         home_pitcher, away_pitcher, home_era, away_era,
    #         home_win_pct, away_win_pct, ... etc.

Typical usage (prediction):
    games = source.load_upcoming_games_with_pitchers()
    # each game: home_team, away_team, home_pitcher, away_pitcher,
    #            home_pitcher_era, away_pitcher_era, etc.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, date as dt_date
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ESPN API base URLs
ESPN_MLB_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
ESPN_ATHLETE_STATS = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/athletes/{athlete_id}/statistics"

# MLB team name normalization (TheOddsAPI full name → ESPN short name)
MLB_TEAM_MAP: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
}

# Reverse: ESPN short abbreviation → TheOddsAPI full name
ABBREV_TO_FULL: dict[str, str] = {v: k for k, v in MLB_TEAM_MAP.items()}

# ── Park Factors (3-year rolling, indexed to 100 = neutral) ──────────────
# Higher = more hitter-friendly (more runs scored)
# Sources: Baseball Savant statcast-park-factors, FanGraphs GABF
# These are for RUN scoring environment (not just HR)
# ── Bullpen Quality Proxy (based on team-season bullpen ERA vs league avg) ──
# Approximate bullpen quality rating: higher = better bullpen
# Calculated from 2024 season bullpen ERA rankings
# These serve as prior estimates; actual values are refined during training
BULLPEN_QUALITY: dict[str, float] = {
    "ARI": 0.52, "ATL": 0.60, "BAL": 0.65, "BOS": 0.48, "CHC": 0.55,
    "CHW": 0.35, "CIN": 0.45, "CLE": 0.62, "COL": 0.30, "DET": 0.50,
    "HOU": 0.58, "KC": 0.42, "LAA": 0.38, "LAD": 0.63, "MIA": 0.40,
    "MIL": 0.56, "MIN": 0.53, "NYM": 0.51, "NYY": 0.61, "OAK": 0.36,
    "PHI": 0.59, "PIT": 0.44, "SD": 0.57, "SEA": 0.64, "SF": 0.49,
    "STL": 0.54, "TB": 0.60, "TEX": 0.55, "TOR": 0.47, "WAS": 0.39,
}

PARK_FACTORS: dict[str, float] = {
    "ARI": 103.0,  # Chase Field (retractable roof, slight hitter)
    "ATL": 100.0,  # Truist Park (neutral)
    "BAL": 103.0,  # Oriole Park at Camden Yards (hitter-friendly)
    "BOS": 103.0,  # Fenway Park (unique dimensions, hitter-friendly)
    "CHC": 102.0,  # Wrigley Field (slight hitter, day games)
    "CHW": 98.0,   # Guaranteed Rate Field (slight pitcher)
    "CIN": 102.0,  # Great American Ball Park (hitter-friendly)
    "CLE": 95.0,   # Progressive Field (pitcher-friendly)
    "COL": 116.0,  # Coors Field (extreme hitter, thin air)
    "DET": 105.0,  # Comerica Park (hitter-friendly, moved fences in)
    "HOU": 97.0,   # Daikin Park (pitcher-friendly)
    "KC": 97.0,    # Kauffman Stadium (pitcher-friendly)
    "LAA": 101.0,  # Angel Stadium (slight hitter)
    "LAD": 104.0,  # Dodger Stadium (hitter-friendly)
    "MIA": 97.0,   # LoanDepot Park (pitcher-friendly)
    "MIL": 100.0,  # American Family Field (neutral)
    "MIN": 100.0,  # Target Field (neutral)
    "NYM": 100.0,  # Citi Field (neutral-to-pitcher)
    "NYY": 101.0,  # Yankee Stadium (slight hitter, short porch)
    "OAK": 108.0,  # Sutter Health Park (temporary, hitter-friendly)
    "PHI": 100.0,  # Citizens Bank Park (neutral-to-slight hitter)
    "PIT": 96.0,   # PNC Park (pitcher-friendly)
    "SD": 95.0,    # Petco Park (pitcher-friendly)
    "SEA": 91.0,   # T-Mobile Park (extreme pitcher)
    "SF": 97.0,    # Oracle Park (pitcher-friendly, marine layer)
    "STL": 97.0,   # Busch Stadium (pitcher-friendly)
    "TB": 97.0,    # Tropicana Field (pitcher-friendly, dome)
    "TEX": 91.0,   # Globe Life Field (pitcher-friendly, covered)
    "TOR": 103.0,  # Rogers Centre (hitter-friendly, dome)
    "WAS": 100.0,  # Nationals Park (neutral)
}


class MLBDataSource:
    """Fetch MLB game data, pitcher stats, and team metrics from ESPN's free API."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
        self._last_request = 0.0
        # Cache: team abbreviation -> {stat_name: value}
        # Cache: athlete_id -> {stat_name: value}
        self._pitcher_stats_cache: dict[str, dict[str, float]] = {}

    # ── Rate limiting ────────────────────────────────────────────────

    def _rate_limit(self):
        """Respect ESPN's servers: max 2 requests/second."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.time()

    # ── Team name helpers ────────────────────────────────────────────

    def _to_short(self, display_name: str) -> str:
        """Convert ESPN display name to short abbreviation."""
        return MLB_TEAM_MAP.get(display_name, display_name)

    def _to_odds_name(self, team: str) -> str:
        """Convert any team identifier to TheOddsAPI full name."""
        if team in ABBREV_TO_FULL:
            return ABBREV_TO_FULL[team]
        if team in MLB_TEAM_MAP:
            return team  # already full name
        return team

    @staticmethod
    def get_park_factor(team_abbrev: str) -> float:
        """Get the park factor for a team's home ballpark.

        Returns the factor as a decimal (e.g., 1.03 for a 3% increase in
        run scoring). Neutral is 1.00.
        """
        pf = PARK_FACTORS.get(team_abbrev.upper(), 100.0)
        return pf / 100.0

    # ── Historical game data ─────────────────────────────────────────

    def load_historical_games(self, days_back: int = 365) -> pd.DataFrame:
        """Load completed MLB games from the ESPN scoreboard API.

        Args:
            days_back: How many days of history to fetch. Each day makes
                       one API call. Default 365 (full season).

        Returns:
            DataFrame with columns: game_id, date, home_team, away_team,
            home_score, away_score, home_pitcher, away_pitcher,
            home_pitcher_era, away_pitcher_era, total_runs, home_win
        """
        all_games: list[dict] = []
        today = dt_date.today()
        errors = 0

        for day_offset in range(days_back, -1, -1):
            if errors > 10:
                logger.warning("Too many consecutive ESPN API errors — stopping fetch")
                break

            query_date = today - timedelta(days=day_offset)
            date_str = query_date.strftime("%Y%m%d")

            games = self._fetch_date_games(date_str)
            if games is None:
                errors += 1
                continue
            errors = 0

            for game in games:
                if game.get("status") == "completed":
                    all_games.append(game)

            if day_offset % 30 == 0:
                logger.info(f"MLB history: {len(all_games)} games so far (at {query_date})")

        if not all_games:
            logger.warning("No MLB historical games fetched")
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df = df.sort_values("date").reset_index(drop=True)

        # Engineer additional features
        df = self._engineer_team_stats(df)
        logger.info(f"MLB historical data: {len(df)} completed games, {len(df.columns)} columns")
        return df

    def _fetch_date_games(self, date_str: str) -> Optional[list[dict]]:
        """Fetch all games for a specific date from ESPN scoreboard.

        Returns None on error, list of game dicts on success.
        """
        self._rate_limit()
        try:
            resp = self._session.get(
                ESPN_MLB_SCOREBOARD,
                params={"dates": date_str, "limit": 300},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug(f"ESPN MLB fetch failed for {date_str}: {e}")
            return None

        events = data.get("events", [])
        if not events:
            return []

        games = []
        for event in events:
            game = self._parse_event(event)
            if game:
                games.append(game)
        return games

    def _parse_event(self, event: dict) -> Optional[dict]:
        """Parse a single ESPN event into a game record."""
        try:
            comps = event.get("competitions", [])
            if not comps:
                return None
            comp = comps[0]

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None

            home = None
            away = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home = c
                else:
                    away = c

            if not home or not away:
                return None

            # Status
            status = comp.get("status", {}).get("type", {}).get("name", "")
            is_completed = status in {
                "STATUS_FINAL", "STATUS_FINAL_ALT", "STATUS_FINAL_ALT_2",
                "STATUS_OFFICIAL_FINAL", "STATUS_COMPLETE",
            }

            home_team_full = home.get("team", {}).get("displayName", "")
            away_team_full = away.get("team", {}).get("displayName", "")
            home_short = self._to_short(home_team_full)
            away_short = self._to_short(away_team_full)

            event_id = event.get("id", "")
            event_date = event.get("date", "")[:10]

            home_score = home.get("score")
            away_score = away.get("score")

            home_score_int = int(home_score) if home_score is not None else None
            away_score_int = int(away_score) if away_score is not None else None

            # Probable pitchers
            home_pitcher, home_pitcher_stats = self._get_pitcher(home)
            away_pitcher, away_pitcher_stats = self._get_pitcher(away)

            game = {
                "game_id": f"mlb_{event_id}",
                "date": event_date,
                "home_team": home_short,
                "away_team": away_short,
                "home_team_full": home_team_full,
                "away_team_full": away_team_full,
                "status": "completed" if is_completed else ("scheduled" if status == "STATUS_SCHEDULED" else "live"),
                "home_score": home_score_int,
                "away_score": away_score_int,
                "total_runs": (home_score_int + away_score_int) if home_score_int is not None and away_score_int is not None else None,
                "home_win": 1 if home_score_int is not None and away_score_int is not None and home_score_int > away_score_int else (0 if is_completed else None),
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
                "home_pitcher_id": home.get("probables", [{}])[0].get("id", "") if home.get("probables") else "",
                "away_pitcher_id": away.get("probables", [{}])[0].get("id", "") if away.get("probables") else "",
            }

            # Add pitcher season stats
            if home_pitcher_stats:
                for k, v in home_pitcher_stats.items():
                    game[f"home_pitcher_{k}"] = v
            if away_pitcher_stats:
                for k, v in away_pitcher_stats.items():
                    game[f"away_pitcher_{k}"] = v

            # Team records
            game["home_record_wins"] = self._parse_record(home, "wins")
            game["home_record_losses"] = self._parse_record(home, "losses")
            game["away_record_wins"] = self._parse_record(away, "wins")
            game["away_record_losses"] = self._parse_record(away, "losses")

            return game

        except Exception as e:
            logger.debug(f"Skipping malformed MLB event: {e}")
            return None

    def _get_pitcher(self, competitor: dict) -> tuple[str, dict[str, float]]:
        """Extract probable pitcher name and season stats from a competitor."""
        probables = competitor.get("probables", [])
        if not probables:
            return "", {}

        pitcher = probables[0]
        name = pitcher.get("displayName", pitcher.get("fullName", ""))
        athlete_id = pitcher.get("id", "")

        # Stats may be embedded directly in the probables object
        stats = {}
        for stat_key in ["era", "wins", "losses", "strikeouts", "inningsPitched",
                         "whip", "kPer9", "bbPer9", "hrPer9", "gamesPlayed"]:
            val = pitcher.get(stat_key)
            if val is not None:
                try:
                    stats[stat_key] = float(val)
                except (ValueError, TypeError):
                    pass

        # If stats are sparse, try fetching from athlete API
        if athlete_id and len(stats) < 3:
            deeper_stats = self._fetch_pitcher_stats(athlete_id)
            stats.update(deeper_stats)

        return name, stats

    def _fetch_pitcher_stats(self, athlete_id: str) -> dict[str, float]:
        """Fetch detailed pitcher stats from ESPN's athlete statistics API."""
        if athlete_id in self._pitcher_stats_cache:
            return self._pitcher_stats_cache[athlete_id]

        self._rate_limit()
        try:
            url = ESPN_ATHLETE_STATS.format(athlete_id=athlete_id)
            resp = self._session.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return {}

        stats = {}
        try:
            # Navigate the nested stats structure
            categories = data.get("categories", [])
            for cat in categories:
                if cat.get("name") == "pitching":
                    for stat_group in cat.get("groups", []):
                        for display_name, val in stat_group.get("stats", {}).items():
                            key = display_name.lower().replace(" ", "_").replace(".", "")
                            try:
                                stats[key] = float(val.get("value", 0))
                            except (ValueError, TypeError, AttributeError):
                                pass
        except Exception:
            pass

        self._pitcher_stats_cache[athlete_id] = stats
        return stats

    @staticmethod
    def _parse_record(competitor: dict, key: str) -> Optional[int]:
        """Parse wins or losses from a competitor's record."""
        records = competitor.get("records", [])
        for rec in records:
            val = rec.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        return None

    # ── Team-level feature engineering ───────────────────────────────

    def _engineer_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling team performance metrics to historical game data.

        Computes per-team rolling averages for:
          - Win rate (last 10 games)
          - Runs scored and allowed (last 5/10 games)
          - Park-factor adjusted run differential
          - Run differential (runs scored - runs allowed)
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values("date")

        if "home_win" not in df.columns:
            return df

        # ── Win rates (rolling 10 games) ──────────────────────────────
        try:
            df["home_win_pct_10"] = df.groupby("home_team")["home_win"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )
        except Exception:
            df["home_win_pct_10"] = 0.500

        try:
            df["away_win_pct_10"] = df.groupby("away_team")["home_win"].transform(
                lambda x: (1 - x).rolling(10, min_periods=1).mean()
            )
        except Exception:
            df["away_win_pct_10"] = 0.500

        # ── Runs scored (rolling 5 games) ─────────────────────────────
        if "home_score" in df.columns:
            try:
                df["home_runs_avg_5"] = df.groupby("home_team")["home_score"].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
            except Exception:
                df["home_runs_avg_5"] = 4.5
        if "away_score" in df.columns:
            try:
                df["away_runs_avg_5"] = df.groupby("away_team")["away_score"].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
            except Exception:
                df["away_runs_avg_5"] = 4.5

        # ── Runs allowed (rolling 5 games) ────────────────────────────
        # Runs allowed by home team = away_score, and vice versa
        if "away_score" in df.columns:
            try:
                df["home_runs_allowed_avg_5"] = df.groupby("home_team")["away_score"].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
            except Exception:
                df["home_runs_allowed_avg_5"] = 4.5
        if "home_score" in df.columns:
            try:
                df["away_runs_allowed_avg_5"] = df.groupby("away_team")["home_score"].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
            except Exception:
                df["away_runs_allowed_avg_5"] = 4.5

        # ── Run differential ──────────────────────────────────────────
        if "home_runs_avg_5" in df.columns and "home_runs_allowed_avg_5" in df.columns:
            df["home_run_diff_5"] = df["home_runs_avg_5"] - df["home_runs_allowed_avg_5"]
        if "away_runs_avg_5" in df.columns and "away_runs_allowed_avg_5" in df.columns:
            df["away_run_diff_5"] = df["away_runs_avg_5"] - df["away_runs_allowed_avg_5"]

        # ── Park factor ───────────────────────────────────────────────
        df["home_park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(100.0) / 100.0
        df["away_park_factor"] = df["away_team"].map(PARK_FACTORS).fillna(100.0) / 100.0
        df["park_factor_diff"] = df["home_park_factor"] - df["away_park_factor"]

        # ── Bullpen quality ────────────────────────────────────────────
        df["home_bullpen_quality"] = df["home_team"].map(BULLPEN_QUALITY).fillna(0.50)
        df["away_bullpen_quality"] = df["away_team"].map(BULLPEN_QUALITY).fillna(0.50)
        df["bullpen_quality_diff"] = df["home_bullpen_quality"] - df["away_bullpen_quality"]

        return df

    # ── Upcoming games with pitcher data ─────────────────────────────

    def load_upcoming_games_with_pitchers(
        self, days_ahead: int = 3
    ) -> pd.DataFrame:
        """Fetch upcoming MLB games with probable starting pitchers.

        This is used at prediction time to build feature vectors for
        games that haven't happened yet.

        Args:
            days_ahead: How many days ahead to scan. Default 3.

        Returns:
            DataFrame with: game_id, date, home_team, away_team,
            home_pitcher, away_pitcher, home_pitcher_era, away_pitcher_era,
            home_record_wins, home_record_losses, etc.
        """
        today = dt_date.today()
        all_games: list[dict] = []

        for day_offset in range(days_ahead + 1):
            query_date = today + timedelta(days=day_offset)
            date_str = query_date.strftime("%Y%m%d")
            games = self._fetch_date_games(date_str)
            if games:
                all_games.extend(games)

        if not all_games:
            logger.info("No upcoming MLB games found from ESPN")
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df = self._engineer_team_stats(df)
        logger.info(f"MLB upcoming: {len(df)} games with pitcher data")
        return df

    # ── Training data builder ────────────────────────────────────────

    def build_training_dataset(self, days_back: int = 365) -> pd.DataFrame:
        """Build a complete training dataset with features for ML model.

        Loads historical games and engineers features suitable for a
        binary classifier (home win / home loss).

        Feature categories:
          1. HISTORICAL PITCHING STATS:
             - Pitcher ERA, WHIP, K/9 (from ESPN probables data)
             - Pitcher ERA/WHIP differential (home - away)
             - Both pitchers known flag

          2. PARK FACTORS:
             - Home/away park factors (3-year rolling, indexed to 1.00)
             - Park factor differential (home - away)
             - Park-adjusted runs

          3. TEAM STRENGTH:
             - Team win percentage (last 10 games)
             - Team runs scored (last 5 games)
             - Team runs allowed (last 5 games)
             - Team run differential
             - Team record entering game
             - Win percentage differential

        Returns:
            DataFrame ready for model training with target column 'home_win'.
        """
        df = self.load_historical_games(days_back=days_back)
        if df.empty:
            return df

        if "home_win" not in df.columns:
            return df

        df = df.dropna(subset=["home_win"]).copy()

        # ═══════════════════════════════════════════════════════════════
        # 1. PITCHER STATS & MATCHUP FEATURES
        # ═══════════════════════════════════════════════════════════════

        # Raw pitcher stats (with fillna for missing data)
        for col in ["home_pitcher_era", "away_pitcher_era",
                     "home_pitcher_whip", "away_pitcher_whip",
                     "home_pitcher_k_per_9", "away_pitcher_k_per_9",
                     "home_pitcher_bb_per_9", "away_pitcher_bb_per_9",
                     "home_pitcher_hr_per_9", "away_pitcher_hr_per_9",
                     "home_pitcher_inningspitched", "away_pitcher_inningspitched"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0.0

        # Pitcher matchup differentials
        df["pitcher_era_diff"] = df["home_pitcher_era"] - df["away_pitcher_era"]
        df["pitcher_whip_diff"] = df["home_pitcher_whip"] - df["away_pitcher_whip"]
        df["pitcher_k9_diff"] = df["home_pitcher_k_per_9"] - df["away_pitcher_k_per_9"]

        # Both pitchers known = information advantage over market
        df["both_pitchers_known"] = ((df["home_pitcher"] != "") & (df["away_pitcher"] != "")).astype(int)

        # ═══════════════════════════════════════════════════════════════
        # 2. PARK FACTORS
        # ═══════════════════════════════════════════════════════════════

        df["home_park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(100.0) / 100.0
        df["away_park_factor"] = df["away_team"].map(PARK_FACTORS).fillna(100.0) / 100.0
        df["park_factor_diff"] = df["home_park_factor"] - df["away_park_factor"]

        # Park-adjusted run scoring estimates
        # Multiply raw runs by park factor to normalize
        if "home_runs_avg_5" in df.columns:
            df["home_runs_park_adj"] = df["home_runs_avg_5"] / df["home_park_factor"]
        if "away_runs_avg_5" in df.columns:
            df["away_runs_park_adj"] = df["away_runs_avg_5"] / df["away_park_factor"]

        # ═══════════════════════════════════════════════════════════════
        # 3. TEAM STRENGTH
        # ═══════════════════════════════════════════════════════════════

        # Team record entering game
        for side in ["home", "away"]:
            win_col = f"{side}_record_wins"
            loss_col = f"{side}_record_losses"
            if win_col in df.columns and loss_col in df.columns:
                df[f"{side}_win_pct"] = df[win_col].fillna(0) / (
                    df[win_col].fillna(0) + df[loss_col].fillna(0) + 1
                )

        # Win percentage differential
        if "home_win_pct" in df.columns and "away_win_pct" in df.columns:
            df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]

        # Rolling win rate (already computed in _engineer_team_stats)
        # Rolling run differential
        if "home_run_diff_5" in df.columns:
            df["home_run_diff_5"] = df["home_run_diff_5"].fillna(0)
        if "away_run_diff_5" in df.columns:
            df["away_run_diff_5"] = df["away_run_diff_5"].fillna(0)
            df["run_diff_5_diff"] = df["home_run_diff_5"].fillna(0) - df["away_run_diff_5"].fillna(0)

        # ═══════════════════════════════════════════════════════════════
        # ENSURE ALL NUMERIC FEATURES ARE CLEAN
        # ═══════════════════════════════════════════════════════════════

        # All potential feature columns
        all_feature_cols = {
            # Pitcher stats
            "home_pitcher_era", "away_pitcher_era",
            "home_pitcher_whip", "away_pitcher_whip",
            "home_pitcher_k_per_9", "away_pitcher_k_per_9",
            "home_pitcher_bb_per_9", "away_pitcher_bb_per_9",
            "home_pitcher_hr_per_9", "away_pitcher_hr_per_9",
            "home_pitcher_inningspitched", "away_pitcher_inningspitched",
            "pitcher_era_diff", "pitcher_whip_diff", "pitcher_k9_diff",
            "both_pitchers_known",
            # Park factors
            "home_park_factor", "away_park_factor", "park_factor_diff",
            "home_runs_park_adj", "away_runs_park_adj",
            # Team strength
            "home_win_pct_10", "away_win_pct_10",
            "home_runs_avg_5", "away_runs_avg_5",
            "home_runs_allowed_avg_5", "away_runs_allowed_avg_5",
            "home_run_diff_5", "away_run_diff_5", "run_diff_5_diff",
            "home_win_pct", "away_win_pct", "win_pct_diff",
            "home_bullpen_quality", "away_bullpen_quality", "bullpen_quality_diff",
            "home_win",
        }

        available = [c for c in all_feature_cols if c in df.columns]
        for c in available:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        logger.info(
            f"MLB training dataset: {len(df)} games, "
            f"{len([c for c in available if c != 'home_win'])} features"
        )
        return df


def get_mlb_team_abbrev(full_name: str) -> str:
    """Get the ESPN short abbreviation for an MLB team."""
    return MLB_TEAM_MAP.get(full_name, full_name)


def get_mlb_full_name(abbrev: str) -> str:
    """Get the TheOddsAPI full name for an MLB team abbreviation."""
    return ABBREV_TO_FULL.get(abbrev.upper(), abbrev)


__all__ = [
    "MLBDataSource",
    "MLB_TEAM_MAP",
    "ABBREV_TO_FULL",
    "get_mlb_team_abbrev",
    "get_mlb_full_name",
]
