"""
QuarterHalfProjector — projects predicted total game scores into quarter-by-quarter
and half-by-half splits using team-specific scoring ratios from real NBA data.

Data Source: ESPN summary API (free, no key required)
  Endpoint: /summary?event=<event_id>
  Returns: linescores array with per-quarter home/away scores

Strategy:
  1. Fetch recent completed NBA games from ESPN scoreboard
  2. For each game, fetch summary to get per-quarter linescores
  3. Compute per-team ratios: what % of a team's total points come in each quarter
  4. To project: predicted_total → estimated home/away split → apply quarter ratios
"""

from __future__ import annotations

import json
import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from betting_intel.pipeline.bootstrap import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ESPN API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# ── NBA League-Average Quarter/Half Ratios ──────────────────────────────
# From 2023-24 NBA season data:
# Q1: ~24.2%, Q2: ~25.1%, Q3: ~25.3%, Q4: ~25.4%
# 1H: ~49.3%, 2H: ~50.7%
# These are used as defaults when team-specific data is unavailable.
_LEAGUE_AVG_RATIOS = {
    "q1": 0.242, "q2": 0.251, "q3": 0.253, "q4": 0.254,
    "h1": 0.493, "h2": 0.507,
}

# Default home team share of total points
_HOME_SHARE = 0.51


class QuarterHalfProjector:
    """Projects total game scores into quarter/half splits using team-specific ratios.

    Usage:
        projector = QuarterHalfProjector()
        projector.compute_ratios()  # Fetches data from ESPN (once)
        result = projector.project(225.0, "Celtics", "Lakers")
        # Returns:
        #   { "q1_home": 28.1, "q1_away": 27.0, "q1_total": 55.1,
        #     "q2_home": 29.0, "q2_away": 28.1, "q2_total": 57.1,
        #     "h1_total": 112.2, "h2_total": 112.8, ... }
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self._ratios: Dict[str, Dict[str, float]] = {}  # team_name -> {q1, q2, q3, q4, h1, h2}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
        self._last_request = 0.0
        self._loaded = False

        if cache_path is None:
            cache_path = PROJECT_ROOT / "data" / "quarter_ratios_cache.json"
        self._cache_path = cache_path

    # ── Public API ───────────────────────────────────────────────────────

    def compute_ratios(self, league_key: str = "nba", max_games: int = 300) -> int:
        """Fetch historical games from ESPN and compute per-team quarter ratios.

        Args:
            league_key: League identifier (only 'nba' supported for now)
            max_games: Max number of games to process

        Returns:
            Number of teams with computed ratios
        """
        if self._loaded:
            return len(self._ratios)

        # Try loading from cache first
        if self._load_cache():
            self._loaded = True
            logger.info(f"Loaded quarter ratios for {len(self._ratios)} teams from cache")
            return len(self._ratios)

        # Fetch from ESPN
        if league_key == "nba":
            self._compute_nba_ratios(max_games)

        if self._ratios:
            self._save_cache()
            self._loaded = True
        return len(self._ratios)

    def project(
        self,
        predicted_total: float,
        home_team: str,
        away_team: str,
    ) -> Dict[str, float]:
        """Project total game score into quarter/half splits.

        Args:
            predicted_total: Model's predicted total game points
            home_team: Home team name (e.g., "Celtics")
            away_team: Away team name

        Returns:
            Dict with keys: q1_home, q1_away, q1_total, q2_home, q2_away, q2_total,
                           q3_home, q3_away, q3_total, q4_home, q4_away, q4_total,
                           h1_home, h1_away, h1_total, h2_home, h2_away, h2_total
        """
        # Estimate home/away score from total
        home_score = predicted_total * _HOME_SHARE
        away_score = predicted_total * (1.0 - _HOME_SHARE)

        # Get ratios (team-specific or league-average)
        home_r = self._ratios.get(self._normalize_team(home_team), _LEAGUE_AVG_RATIOS)
        away_r = self._ratios.get(self._normalize_team(away_team), _LEAGUE_AVG_RATIOS)

        result = {}
        for q in ["q1", "q2", "q3", "q4"]:
            h_q = home_score * home_r[q]
            a_q = away_score * away_r[q]
            result[f"{q}_home"] = round(h_q, 1)
            result[f"{q}_away"] = round(a_q, 1)
            result[f"{q}_total"] = round(h_q + a_q, 1)

        for h in ["h1", "h2"]:
            h_h = home_score * home_r[h]
            a_h = away_score * away_r[h]
            result[f"{h}_home"] = round(h_h, 1)
            result[f"{h}_away"] = round(a_h, 1)
            result[f"{h}_total"] = round(h_h + a_h, 1)

        # Add team totals per half
        result["home_score"] = round(home_score, 1)
        result["away_score"] = round(away_score, 1)
        result["predicted_total"] = round(predicted_total, 1)

        return result

    def get_quarter_market(self, predicted_total: float, quarter: int) -> float:
        """Estimate the market line for a single quarter total."""
        factor = _LEAGUE_AVG_RATIOS.get(f"q{quarter}", 0.25)
        return round(predicted_total * factor, 1)

    def get_half_market(self, predicted_total: float, half: int = 1) -> float:
        """Estimate the market line for a half total."""
        key = f"h{half}"
        factor = _LEAGUE_AVG_RATIOS.get(key, 0.5)
        return round(predicted_total * factor, 1)

    # ── NBA Data Loading ─────────────────────────────────────────────────

    def _compute_nba_ratios(self, max_games: int = 300):
        """Fetch recent NBA games and compute per-team quarter ratios."""
        try:
            # Step 1: Get recent completed games from scoreboard
            game_ids = self._fetch_recent_game_ids(max_games // 2)
            if not game_ids:
                logger.warning("No recent NBA games found from ESPN")
                return

            # Step 2: Fetch summary for each game to get linescores
            team_quarters: Dict[str, List[Dict[str, float]]] = {}
            for i, gid in enumerate(game_ids):
                if i >= max_games:
                    break
                ls = self._fetch_linescores(gid)
                if ls:
                    for team_name, scores in ls:
                        normalized = self._normalize_team(team_name)
                        if normalized not in team_quarters:
                            team_quarters[normalized] = []
                        team_quarters[normalized].append(scores)

            # Step 3: Compute ratios per team
            for team_name, games_list in team_quarters.items():
                if len(games_list) < 3:
                    continue
                # Average per-quarter points
                avg_q = {f"q{i+1}": float(np.mean([g[f"q{i+1}"] for g in games_list]))
                         for i in range(4)}
                avg_total = sum(avg_q.values())
                if avg_total <= 0:
                    continue

                self._ratios[team_name] = {
                    "q1": avg_q["q1"] / avg_total,
                    "q2": avg_q["q2"] / avg_total,
                    "q3": avg_q["q3"] / avg_total,
                    "q4": avg_q["q4"] / avg_total,
                    "h1": (avg_q["q1"] + avg_q["q2"]) / avg_total,
                    "h2": (avg_q["q3"] + avg_q["q4"]) / avg_total,
                }

            logger.info(
                f"Computed quarter ratios for {len(self._ratios)} NBA teams "
                f"from {len(game_ids)} games"
            )

        except Exception as e:
            logger.warning(f"Failed to compute NBA quarter ratios: {e}")

    def _fetch_recent_game_ids(self, limit: int = 150) -> List[str]:
        """Fetch recent completed game IDs from ESPN scoreboard."""
        game_ids: List[str] = []

        # Query multiple dates in the past to find completed games
        from datetime import datetime, timedelta
        today = datetime.now()

        for days_ago in range(1, min(30, limit + 1)):
            date_str = (today - timedelta(days=days_ago)).strftime("%Y%m%d")
            self._rate_limit()

            try:
                url = ESPN_SCOREBOARD_URL
                resp = self._session.get(
                    url, params={"dates": date_str, "limit": 300}, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
                events = data.get("events", [])

                for event in events:
                    status = self._get_status(event)
                    if status in ("STATUS_FINAL", "STATUS_FINAL_ALT", "STATUS_FINAL_ALT_2"):
                        eid = event.get("id")
                        if eid:
                            game_ids.append(eid)

                if len(game_ids) >= limit:
                    break

            except Exception:
                continue

        return game_ids[:limit]

    def _fetch_linescores(
        self, event_id: str
    ) -> Optional[List[Tuple[str, Dict[str, float]]]]:
        """Fetch per-quarter scores for a game from ESPN summary endpoint."""
        self._rate_limit()

        try:
            url = f"{ESPN_SUMMARY_URL}?event={event_id}"
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        try:
            competitions = data.get("header", {}).get("competitions", [])
            if not competitions:
                competitions = data.get("gameInfo", {}).get("competitions", [])
            if not competitions:
                return None

            competitors = competitions[0].get("competitors", [])
            results = []
            for comp in competitors:
                team_info = comp.get("team", {})
                team_name = team_info.get("displayName", team_info.get("name", ""))
                if not team_name:
                    continue

                linescores = comp.get("linescores", [])
                if len(linescores) >= 4:  # At least 4 quarters
                    scores = {
                        "q1": float(linescores[0].get("value", 0)),
                        "q2": float(linescores[1].get("value", 0)),
                        "q3": float(linescores[2].get("value", 0)),
                        "q4": float(linescores[3].get("value", 0)),
                    }
                    results.append((team_name, scores))

            return results if len(results) == 2 else None

        except (KeyError, IndexError, ValueError, TypeError):
            return None

    @staticmethod
    def _get_status(event: dict) -> str:
        """Extract game status from an ESPN event."""
        try:
            comps = event.get("competitions", [])
            if comps:
                return comps[0].get("status", {}).get("type", {}).get("name", "")
        except Exception:
            pass
        return ""

    # ── Helpers ───────────────────────────────────────────────────────────

    def _rate_limit(self):
        """Rate limit API requests (free API, be respectful)."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.3:
            time.sleep(0.3 - elapsed)
        self._last_request = time.time()

    @staticmethod
    def _normalize_team(name: str) -> str:
        """Normalize team name for consistent matching."""
        return name.strip().lower()

    # ── Cache ─────────────────────────────────────────────────────────────

    def _load_cache(self) -> bool:
        """Load pre-computed ratios from cache file."""
        try:
            if self._cache_path and self._cache_path.exists():
                with open(self._cache_path, "r") as f:
                    data = json.load(f)
                self._ratios = {
                    k: {sk: sv for sk, sv in v.items()}
                    for k, v in data.items()
                }
                return True
        except Exception as e:
            logger.debug(f"Failed to load quarter ratios cache: {e}")
        return False

    def _save_cache(self):
        """Save computed ratios to cache file."""
        try:
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "w") as f:
                    json.dump(self._ratios, f, indent=2)
                logger.info(f"Saved quarter ratios for {len(self._ratios)} teams to {self._cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save quarter ratios cache: {e}")
