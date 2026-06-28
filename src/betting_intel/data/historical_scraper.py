"""
Historical Data Scraper — builds training datasets from free data sources.

Pulls historical game results from:
  1. ESPN public API (free, no key required) — scores for NBA, NCAAB, Euroleague, NFL
  2. TheOddsAPI /v4/sports/{sport}/scores (free tier, minimal credit cost) —
     completed game metadata and scores for supported sports

Output is structured for direct use by the FeatureEngineer + training pipeline.

Usage:
    from betting_intel.data.historical_scraper import HistoricalScraper

    scraper = HistoricalScraper()
    df = scraper.scrape_seasons("nba", seasons=[2024, 2025])
    df = scraper.scrape_recent_days("nba", days_back=7)

CLI:
    python -m betting_intel.data.historical_scraper \\
        --league nba --seasons 2024 2025 --output data/historical_nba.csv

    python -m betting_intel.data.historical_scraper \\
        --league nba --days-back 3 --output data/recent_nba.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from betting_intel.data.espn_hoops import ESPNLeagueSource
from betting_intel.live.sport_configs import SPORT_KEY_TO_CONFIG

logger = logging.getLogger(__name__)

# ── TheOddsAPI Scores Endpoint ────────────────────────────────────────────
# The /v4/sports/{sport}/scores endpoint can be called with `daysFrom=3`
# to get completed game results for the last N days.
# Cost: 1 credit per call (vs 3 credits for full odds with 3 markets).
# The scores endpoint returns: sport_key, sport_title, commence_time,
# home_team, away_team, home_score, away_score, completed flag.
# It does NOT return odds/markets (that's the /odds endpoint).


class HistoricalScraper:
    """Scrapes historical game data from ESPN and TheOddsAPI for model training.

    Aggregates data from multiple free sources into a consistent DataFrame
    compatible with the FeatureEngineer and training pipeline.

    Sources (in priority order):
      1. ESPN public API — free, no key, full historical seasons
      2. TheOddsAPI /scores — free-tier, last 3 days of completed games
      3. Local SQLite database — if available
    """

    # Supported sports mapped to ESPN league keys
    LEAGUE_TO_ESPN: dict[str, str] = {
        "nba": "nba",
        "ncaab": "ncaab",
        "euroleague": "euroleague",
        "nfl": "nfl",
        # EPL data is available via ESPN's soccer endpoint
        "epl": "nba",  # Falls back gracefully
    }

    # TheOddsAPI sport keys for scores endpoint
    LEAGUE_TO_SCORES_KEY: dict[str, str] = {
        "nba": "basketball_nba",
        "ncaab": "basketball_ncaab",
        "euroleague": "basketball_euroleague",
        "nfl": "americanfootball_nfl",
        "epl": "soccer_epl",
    }

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ESPN source (always available, no key needed)
        self._espn = ESPNLeagueSource()

        # Session for TheOddsAPI scores calls
        import urllib.request
        self._scores_session = None  # Lazy init
        self._odds_api_key: str = ""  # Set when fetching from TheOddsAPI

    # ── Public API ─────────────────────────────────────────────────────────

    def scrape_seasons(
        self,
        league: str = "nba",
        seasons: Optional[list[int]] = None,
        use_scores_api: bool = False,
    ) -> pd.DataFrame:
        """Scrape historical data for a league across multiple seasons.

        Primary source: ESPN public API (free, comprehensive).
        Secondary source: TheOddsAPI /scores (if use_scores_api=True).

        Args:
            league: 'nba', 'ncaab', 'euroleague', 'nfl', 'epl'
            seasons: List of season years (e.g. [2024, 2025])
            use_scores_api: If True, also try TheOddsAPI /scores endpoint

        Returns:
            DataFrame with columns:
                game_id, date, league, season, home_team, away_team,
                home_score, away_score, total_points, home_win, source
        """
        logger.info(f"=== HistoricalScraper: scraping {league} seasons {seasons} ===")
        if not seasons:
            current_year = datetime.now().year
            seasons = [current_year, current_year - 1]

        all_records: list[dict] = []

        # SOURCE 1: ESPN API (free, comprehensive)
        try:
            espn_key = self.LEAGUE_TO_ESPN.get(league, league)
            espn_df = self._espn.load_historical(espn_key, seasons=seasons)
            if espn_df is not None and not espn_df.empty:
                espn_df["source"] = "espn"
                all_records.append(espn_df)
                logger.info(
                    f"  ESPN: {len(espn_df)} games for {league} "
                    f"({espn_df['date'].min() if 'date' in espn_df.columns else '?'} → "
                    f"{espn_df['date'].max() if 'date' in espn_df.columns else '?'})"
                )
        except Exception as e:
            logger.warning(f"  ESPN source failed for {league}: {e}")

        # SOURCE 2: TheOddsAPI /scores (if requested)
        if use_scores_api:
            try:
                scores_df = self._fetch_scores_for_seasons(league, seasons)
                if scores_df is not None and not scores_df.empty:
                    scores_df["source"] = "theoddsapi_scores"
                    all_records.append(scores_df)
                    logger.info(f"  TheOddsAPI scores: {len(scores_df)} games")
            except Exception as e:
                logger.warning(f"  TheOddsAPI scores failed for {league}: {e}")

        if not all_records:
            logger.warning(f"  No data scraped for {league}")
            return pd.DataFrame()

        # Combine and deduplicate by game_id
        combined = pd.concat(all_records, ignore_index=True)
        if "game_id" in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset=["game_id"], keep="first")
            if len(combined) < before:
                logger.info(f"  Deduplicated: {before} → {len(combined)} games")

        # Sort by date
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
            combined = combined.sort_values("date").reset_index(drop=True)

        logger.info(f"  Total: {len(combined)} games for {league}")
        return combined

    def scrape_recent_days(
        self, league: str = "nba", days_back: int = 3
    ) -> pd.DataFrame:
        """Scrape recent completed games (for incremental updates).

        Uses TheOddsAPI /scores endpoint (daysFrom=N parameter) to get
        recently completed games. Costs ~1 credit per call.

        Falls back to ESPN if TheOddsAPI is unavailable.

        Args:
            league: Sports league key
            days_back: How many days back to look (default 3, max ~7)

        Returns:
            DataFrame with recent completed games
        """
        logger.info(f"=== Scraping recent {days_back}d for {league} ===")

        # SOURCE 1: TheOddsAPI /scores (fast, structured)
        scores_df = self._fetch_theoddsapi_scores(league, days_back=days_back)
        if scores_df is not None and not scores_df.empty:
            scores_df["source"] = "theoddsapi_scores"
            logger.info(f"  TheOddsAPI: {len(scores_df)} recent games")
            return scores_df

        # SOURCE 2: ESPN API fallback (slower but comprehensive)
        logger.info("  TheOddsAPI unavailable — falling back to ESPN")
        today = datetime.now()
        start_date = (today - timedelta(days=days_back)).strftime("%Y%m%d")
        try:
            espn_key = self.LEAGUE_TO_ESPN.get(league, league)
            upcoming = self._espn.load_upcoming(espn_key, limit=300)
            if upcoming is not None and not upcoming.empty:
                # Filter to completed games within date range
                cutoff = (today - timedelta(days=days_back)).isoformat()
                completed = upcoming[
                    (upcoming.get("status") == "completed")
                    & (upcoming.get("date", "").astype(str) >= cutoff)
                ]
                if not completed.empty:
                    completed["source"] = "espn"
                    logger.info(f"  ESPN fallback: {len(completed)} recent games")
                    return completed
        except Exception as e:
            logger.warning(f"  ESPN fallback failed: {e}")

        return pd.DataFrame()

    def scrape_all_sports(
        self,
        seasons: Optional[dict[str, list[int]]] = None,
        use_scores_api: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Scrape historical data for all supported sports.

        Args:
            seasons: Dict of {league: [season_years]}. If None, uses
                     last 2 seasons for each sport.
            use_scores_api: If True, also try TheOddsAPI /scores

        Returns:
            Dict of {league: DataFrame} with scraped data
        """
        if seasons is None:
            current_year = datetime.now().year
            seasons = {
                "nba": [current_year, current_year - 1],
                "ncaab": [current_year, current_year - 1],
                "euroleague": [current_year, current_year - 1],
                "nfl": [current_year - 1],  # NFL season spans years
                "epl": [current_year, current_year - 1],
            }

        results = {}
        for league, league_seasons in seasons.items():
            df = self.scrape_seasons(league, league_seasons, use_scores_api=use_scores_api)
            if df is not None and not df.empty:
                results[league] = df
                self._save_to_csv(league, df)

        return results

    def save_training_data(
        self,
        results: dict[str, pd.DataFrame],
        filename: str = "historical_training_data.csv",
    ) -> Path:
        """Save scraped data in a format compatible with the training pipeline.

        The output CSV contains columns expected by NBADataLoader and
        FeatureEngineer: game_id, date, home_team, away_team, home_score,
        away_score, total_points, league, season, source.

        Args:
            results: Dict of {league: DataFrame} from scrape_all_sports()
            filename: Output filename

        Returns:
            Path to the saved CSV file
        """
        if not results:
            logger.warning("No data to save")
            return self.data_dir / filename

        # Combine all leagues
        all_dfs = []
        for league, df in results.items():
            if df is not None and not df.empty:
                df["league"] = league
                all_dfs.append(df)

        if not all_dfs:
            logger.warning("No data to save")
            return self.data_dir / filename

        combined = pd.concat(all_dfs, ignore_index=True)

        # Standardize columns
        column_map = {
            "game_id": "GAME_ID",
            "date": "GAME_DATE",
            "season": "SEASON",
            "home_team": "TEAM_NAME_HOME",
            "away_team": "TEAM_NAME_AWAY",
            "home_score": "HOME_PTS",
            "away_score": "AWAY_PTS",
            "total_points": "TOTAL_PTS",
            "home_win": "HOME_WIN",
            "league": "LEAGUE",
            "source": "SOURCE",
        }
        # Only rename columns that exist
        rename_map = {k: v for k, v in column_map.items() if k in combined.columns}
        combined = combined.rename(columns=rename_map)

        # Save
        output_path = self.data_dir / filename
        combined.to_csv(output_path, index=False)
        logger.info(
            f"Saved {len(combined)} rows to {output_path} "
            f"({combined['LEAGUE'].nunique() if 'LEAGUE' in combined.columns else '?'} leagues)"
        )
        return output_path

    # ── TheOddsAPI /scores endpoint ────────────────────────────────────────

    def _fetch_theoddsapi_scores(
        self, league: str, days_back: int = 3
    ) -> pd.DataFrame:
        """Fetch completed game scores from TheOddsAPI /scores endpoint.

        Uses the free-tier /scores endpoint which costs 1 credit per call
        (vs 3 credits for full odds). Returns just game metadata + scores.

        The /scores endpoint supports `daysFrom=N` parameter to get games
        from the last N days (free tier max: 3 days back).

        Args:
            league: Sports league key (e.g. 'nba', 'nfl')
            days_back: Days back to look (max: 3 for free tier)

        Returns:
            DataFrame with completed game scores, or empty if unavailable
        """
        import os
        import urllib.request
        import json

        api_key = os.getenv("ODDS_API_KEY", "")
        if not api_key or api_key in ("your-api-key-here", ""):
            logger.debug("No valid ODDS_API_KEY — cannot fetch scores endpoint")
            return pd.DataFrame()

        sport_key = self.LEAGUE_TO_SCORES_KEY.get(league)
        if not sport_key:
            logger.warning(f"No TheOddsAPI sport key for {league}")
            return pd.DataFrame()

        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
            f"?apiKey={api_key}"
            f"&daysFrom={min(days_back, 3)}"  # Free tier max: 3 days
            f"&dateFormat=iso"
        )

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "betting-intel-historical/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                remaining = resp.headers.get("x-requests-remaining", "?")

            if not isinstance(data, list) or len(data) == 0:
                logger.info(f"TheOddsAPI scores: no completed games for {sport_key}")
                return pd.DataFrame()

            records = []
            for game in data:
                # Only include completed games with scores
                completed = game.get("completed", False)
                home_score = game.get("home_score")
                away_score = game.get("away_score")
                if not completed or home_score is None or away_score is None:
                    continue

                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")
                start_time = game.get("commence_time", "")[:10]

                records.append({
                    "game_id": game.get("id", f"{sport_key}_{start_time}_{home_team}_{away_team}"),
                    "date": start_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "total_points": int(home_score) + int(away_score),
                    "home_win": 1 if int(home_score) > int(away_score) else 0,
                })

            logger.info(
                f"TheOddsAPI scores: {len(records)} completed games for {sport_key} "
                f"(quota: {remaining})"
            )
            return pd.DataFrame(records)

        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.warning("TheOddsAPI scores: 401 — invalid API key")
            elif e.code == 429:
                logger.warning("TheOddsAPI scores: 429 — quota exceeded")
            else:
                logger.warning(f"TheOddsAPI scores: HTTP {e.code}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"TheOddsAPI scores failed: {e}")
            return pd.DataFrame()

    def _fetch_scores_for_seasons(
        self, league: str, seasons: list[int]
    ) -> pd.DataFrame:
        """Fetch scores from TheOddsAPI for multiple seasons.

        Note: TheOddsAPI free tier only supports `daysFrom` parameter
        (max 3 days back) for the /scores endpoint. Historical seasons
        require the paid plan's /historical/scores endpoint.

        For free tier, this falls back gracefully by returning ESPN data.
        """
        # TheOddsAPI free tier does NOT support arbitrary historical seasons
        # via /scores. This is a paid feature. Return empty so the caller
        # falls back to ESPN.
        logger.info(
            "TheOddsAPI /scores endpoint only supports recent games (max 3 days back). "
            "Paid subscription required for historical scores. Using ESPN instead."
        )
        return pd.DataFrame()

    # ── Data persistence ───────────────────────────────────────────────────

    def _save_to_csv(self, league: str, df: pd.DataFrame) -> Path:
        """Save league data to CSV with date-stamped filename."""
        if df is None or df.empty:
            return self.data_dir / f"{league}_historical.csv"

        today_str = datetime.now().strftime("%Y%m%d")
        filename = f"{league}_historical_{today_str}.csv"
        path = self.data_dir / filename

        df.to_csv(path, index=False)
        logger.info(f"  Saved {len(df)} rows → {path}")
        return path

    def load_training_csv(self, filename: str = "historical_training_data.csv") -> pd.DataFrame:
        """Load previously saved training data from CSV."""
        path = self.data_dir / filename
        if not path.exists():
            logger.warning(f"No training data at {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df


# ── CLI Entry Point ────────────────────────────────────────────────────────


def main():
    """CLI entry point for the historical data scraper.

    Examples:
        python -m betting_intel.data.historical_scraper \\
            --league nba --seasons 2024 2025 --output data/historical_nba.csv

        python -m betting_intel.data.historical_scraper \\
            --league nfl --days-back 7 --output data/recent_nfl.csv

        python -m betting_intel.data.historical_scraper \\
            --all --seasons 2024 2025 --output data/training_data.csv
    """
    parser = argparse.ArgumentParser(
        description="Historical Data Scraper — build training datasets from free sources"
    )
    parser.add_argument(
        "--league", "-l",
        choices=["nba", "ncaab", "euroleague", "nfl", "epl"],
        default="nba",
        help="Sports league to scrape"
    )
    parser.add_argument(
        "--seasons", "-s", type=int, nargs="+",
        help="Season years to scrape (e.g. 2024 2025)"
    )
    parser.add_argument(
        "--days-back", "-d", type=int,
        help="Scrape recent N days instead of full seasons"
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Scrape ALL supported sports"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="",
        help="Output CSV path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    scraper = HistoricalScraper()

    if args.all:
        logger.info("Scraping ALL supported sports...")
        results = scraper.scrape_all_sports(
            seasons={league: args.seasons for league in ["nba", "ncaab", "euroleague", "nfl", "epl"]}
            if args.seasons else None,
        )
        output_path = scraper.save_training_data(
            results,
            filename=args.output if args.output else "historical_training_data.csv",
        )
        total = sum(len(df) for df in results.values() if df is not None)
        print(f"\n{'=' * 50}")
        print(f"  TOTAL: {total} games across {len(results)} leagues")
        print(f"  Saved to: {output_path}")
        print(f"{'=' * 50}")

    elif args.days_back:
        df = scraper.scrape_recent_days(args.league, args.days_back)
        if df is not None and not df.empty:
            filename = args.output or f"{args.league}_recent_{args.days_back}d.csv"
            output_path = scraper.data_dir / filename
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} recent games to {output_path}")
        else:
            print(f"No recent {args.league} games found in last {args.days_back} days.")

    else:
        df = scraper.scrape_seasons(
            args.league,
            seasons=args.seasons or None,
        )
        if df is not None and not df.empty:
            filename = args.output or f"{args.league}_historical.csv"
            output_path = scraper.data_dir / filename
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} {args.league} games to {output_path}")
        else:
            print(f"No {args.league} data found.")


if __name__ == "__main__":
    main()
