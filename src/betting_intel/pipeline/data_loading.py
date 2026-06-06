"""
Data loading mixin — loads game data from various sources.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from betting_intel.pipeline.bootstrap import (
    PROJECT_ROOT, ODDS_API_KEY, logger,
)


class DataLoadingMixin:
    """Mixin providing data-loading methods for PredictionPipeline."""

    def load_data(self) -> pd.DataFrame:
        """Load game data from the best available source."""
        print("\n" + "=" * 70)
        print("  📊  STAGE 1: DATA LOADING")
        print("=" * 70)

        if self.args.live:
            print("  🌐  Attempting live odds fetch from TheOddsAPI...")
            df = self._load_live_data()
            if df is not None and not df.empty:
                return df
            print("  ⚠  Live data unavailable, falling back to historical.")
            self.args.live = False

        # Historical loading
        df = self._load_historical_data()
        return df

    def _load_live_data(self) -> Optional[pd.DataFrame]:
        """Fetch live upcoming games via TheOddsAPI.

        Distinguishes between:
        - HTTP 429 (quota exceeded): raises a clear error, no fallback to historical
        - HTTP 401 (bad key / no key): returns None so nba_api fallback can generate schedule
        - Other errors: falls back to historical data with a warning
        """
        # Skip if API key is a placeholder — avoids stale cache with wrong team name format
        if not ODDS_API_KEY or ODDS_API_KEY in ("your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"):
            print("  ⚠  No valid ODDS_API_KEY configured. Skipping live odds fetch.")
            print("  ℹ  The pipeline will generate upcoming games from NBA static data instead.")
            return None

        try:
            from betting_intel.data.live_gateway import LiveDataGateway
            gateway = LiveDataGateway(odds_api_key=ODDS_API_KEY)
            odds_data = gateway.get_live_odds(force_refresh=True)
            if odds_data and len(odds_data) > 0:
                df = pd.DataFrame(odds_data)
                print(f"  ✅  Fetched {len(df)} games from LiveDataGateway")
                self.results["metadata"]["data_source"] = "live_gateway"
                return df
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg:
                print(f"  ❌  API QUOTA EXCEEDED (HTTP 429): {e}")
                print("  ❌  Cannot continue in --live mode without a valid TheOddsAPI quota.")
                print("  ❌  Set ODDS_API_KEY in your .env file or wait for quota to reset.")
                raise RuntimeError(f"TheOddsAPI quota exceeded: {e}") from e
            if "401" in err_msg or "unauthorized" in err_msg:
                print("  ⚠  Invalid ODDS_API_KEY (HTTP 401). Using NBA static data fallback.")
                return None
            print(f"  ⚠  LiveDataGateway failed: {e}")

        return None

    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """Load historical game data from CSV, SQLite, or data loader."""
        days = self.args.days_history

        # Try CSV path
        if self.args.csv_path:
            path = Path(self.args.csv_path)
            if path.exists():
                df = pd.read_csv(path)
                print(f"  ✅  Loaded {len(df)} games from CSV: {path.name}")
                self.results["metadata"]["data_source"] = "csv"
                return df

        # Try NBADataLoader
        try:
            from betting_intel.data.loader import NBADataLoader
            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is not None and not raw_df.empty:
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])
                recent = raw_df[raw_df["GAME_DATE"] >= cutoff]
                if recent.empty:
                    print(f"  ⚠  No games found in the last {days} days (data may be from an older season)")
                    print(f"  ℹ  Falling back to all {len(raw_df)} available historical games")
                    df = loader.build_game_dataset(raw_df)
                else:
                    df = loader.build_game_dataset(recent)
                if df is not None and not df.empty:
                    print(f"  ✅  Loaded {len(df)} games from NBADataLoader")
                    self.results["metadata"]["data_source"] = "nba_dataloader"
                    return df
        except Exception as e:
            print(f"  ⚠  NBADataLoader failed: {e}")

        # Try scripts
        try:
            from scripts.fetch_real_nba_data import NBAStatsFetcher
            fetcher = NBAStatsFetcher()
            df = fetcher.fetch_game_logs(days=days)
            if df is not None and not df.empty:
                print(f"  ✅  Loaded {len(df)} games from NBAStatsFetcher")
                self.results["metadata"]["data_source"] = "nba_stats"
                return df
        except Exception as e:
            print(f"  ⚠  NBAStatsFetcher failed: {e}")

        return None
