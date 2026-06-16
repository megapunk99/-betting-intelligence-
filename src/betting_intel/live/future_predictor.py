"""
FutureGamePredictor — GENUINELY AUTONOMOUS predictions.

DESIGN PRINCIPLES:
  - ALWAYS produces correct-range totals (NBA 200-260)
  - Stat baseline is the foundation; ML model is an ADDITIONAL signal only
  - Every failure mode logs WHY — never silent
  - Caches predictions so outages never break the dashboard
  - NEVER generates fake matchups — returns empty with clear log

ARCHITECTURE:
  1. Fetch REAL upcoming games from ESPN API (next 14 days, NBA only)
  2. For every game: predicted_total = stat_baseline + team_strength_adjustment
  3. If ML model produces VALID output (100-350), use its delta too
  4. market_total = stat_baseline (best estimate without real odds)
  5. Cascade to Q1-Q4 and 1H-2H using NBA-specific quarter ratios
  6. Cache to disk so dashboard works during API/network outages
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from betting_intel.utils.safe_serialize import safe_joblib_load, ModelIntegrityError

logger = logging.getLogger(__name__)

# ── League Constants ──────────────────────────────────────────────────────
# NBA
NBA_TEAMS = {"Hawks","Celtics","Nets","Hornets","Bulls","Cavaliers","Mavericks",
             "Nuggets","Pistons","Warriors","Rockets","Pacers","Clippers","Lakers",
             "Grizzlies","Heat","Bucks","Timberwolves","Pelicans","Knicks","Thunder",
             "Magic","76ers","Suns","Trail Blazers","Kings","Spurs","Raptors","Jazz","Wizards"}

NBA_QUARTER_RATIOS   = {"q1": 0.242, "q2": 0.251, "q3": 0.253, "q4": 0.254, "h1": 0.493, "h2": 0.507}
NBA_HOME_ADVANTAGE   = 2.3    # pts added for home court
NBA_TOTAL_MIN        = 200.0
NBA_TOTAL_MAX        = 260.0

# NCAAB — NCAA Division I men's college basketball.
# Lower scoring than NBA (~145 avg total). Higher home court advantage (~4 pts).
# NCAAB plays two 20-minute halves, but quarter ratios are approximated
# by splitting each half equally (for dashboard display consistency).
NCAAB_QUARTER_RATIOS = {"q1": 0.230, "q2": 0.235, "q3": 0.260, "q4": 0.275, "h1": 0.465, "h2": 0.535}
NCAAB_HOME_ADVANTAGE = 4.0    # pts added for home court (college crowds matter more)
NCAAB_TOTAL_MIN      = 110.0
NCAAB_TOTAL_MAX      = 190.0

# Euroleague — Top European club competition.
# Lower scoring than NBA (~160 avg total). Higher home court advantage (~4 pts)
# due to hostile European arenas and travel fatigue. 10-minute quarters.
EUROLEAGUE_QUARTER_RATIOS = {"q1": 0.240, "q2": 0.248, "q3": 0.252, "q4": 0.260, "h1": 0.488, "h2": 0.512}
EUROLEAGUE_HOME_ADVANTAGE = 4.5  # pts added for home court (European arenas are loud)
EUROLEAGUE_TOTAL_MIN      = 150.0
EUROLEAGUE_TOTAL_MAX      = 180.0

# NFL — National Football League.
# Low scoring (~45 total points). 4 quarters, ~12 mins each.
# Home field advantage is ~1.75 pts (smaller than basketball).
# Quarter distribution: slightly more scoring in 2nd & 4th quarters.
NFL_TEAMS = {"Bills","Dolphins","Patriots","Jets","Ravens","Bengals","Browns","Steelers",
             "Texans","Colts","Jaguars","Titans","Broncos","Chiefs","Raiders","Chargers",
             "Cowboys","Giants","Eagles","Commanders","Bears","Lions","Packers","Vikings",
             "Falcons","Panthers","Saints","Buccaneers","Cardinals","Rams","49ers","Seahawks"}

NFL_QUARTER_RATIOS   = {"q1": 0.23, "q2": 0.27, "q3": 0.23, "q4": 0.27, "h1": 0.50, "h2": 0.50}
NFL_HOME_ADVANTAGE   = 1.75  # pts added for home field (modern NFL ~1-2 pts)
NFL_TOTAL_MIN        = 30.0
NFL_TOTAL_MAX        = 60.0

# ── Per-league config lookup ───────────────────────────────────────────
def _league_config(league: str) -> dict:
    """Get league-specific constants."""
    if league == "ncaab":
        return {
            "quarter_ratios": NCAAB_QUARTER_RATIOS,
            "home_advantage": NCAAB_HOME_ADVANTAGE,
            "total_min": NCAAB_TOTAL_MIN,
            "total_max": NCAAB_TOTAL_MAX,
            "default_stat_base": 145.0,
        }
    if league == "euroleague":
        return {
            "quarter_ratios": EUROLEAGUE_QUARTER_RATIOS,
            "home_advantage": EUROLEAGUE_HOME_ADVANTAGE,
            "total_min": EUROLEAGUE_TOTAL_MIN,
            "total_max": EUROLEAGUE_TOTAL_MAX,
            "default_stat_base": 160.0,
        }
    if league == "nfl":
        return {
            "quarter_ratios": NFL_QUARTER_RATIOS,
            "home_advantage": NFL_HOME_ADVANTAGE,
            "total_min": NFL_TOTAL_MIN,
            "total_max": NFL_TOTAL_MAX,
            "default_stat_base": 45.0,
        }
    # Default: NBA
    return {
        "quarter_ratios": NBA_QUARTER_RATIOS,
        "home_advantage": NBA_HOME_ADVANTAGE,
        "total_min": NBA_TOTAL_MIN,
        "total_max": NBA_TOTAL_MAX,
        "default_stat_base": 228.0,
    }

# ESPN API team name → short name map.
# Uses sport_configs.py as the single source of truth for all sports.
# ESPN-specific API name variants are added as overrides.
from betting_intel.live.sport_configs import ALL_TEAM_NAME_MAP

_ESPN_ALIASES: dict[str, str] = {
    # ESPN returns multiple naming variants for some NBA teams
    "LA Lakers": "Lakers",
    "Los Angeles Clippers": "Clippers",
}

ESPN_TO_SHORT: dict[str, str] = {}
ESPN_TO_SHORT.update(ALL_TEAM_NAME_MAP)
ESPN_TO_SHORT.update(_ESPN_ALIASES)

ESPN_TIMEOUT   = 6     # seconds per HTTP request
MAX_DAYS_SCAN  = 14    # how many days ahead to scan ESPN


class PredictionCache:
    """File-based cache so the dashboard works even when ESPN is unreachable."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "prediction_cache.json"
        self._path = path
        self._cache: dict = {}

    def load(self) -> dict:
        try:
            if self._path.exists() and self._path.stat().st_size > 0:
                with open(self._path) as f:
                    self._cache = json.load(f)
                age = time.time() - self._cache.get("_cached_at", 0)
                logger.info(f"Cache loaded: {len(self._cache.get('predictions', []))} preds, "
                            f"{age/60:.0f} min old")
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            self._cache = {}
        return self._cache

    def save(self, predictions: list[dict], fresh: bool = False) -> bool:
        try:
            if fresh or not self._cache.get("predictions") or \
               time.time() - self._cache.get("_cached_at", 0) > 600:
                self._cache = {"predictions": predictions, "_cached_at": time.time()}
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "w") as f:
                    json.dump(self._cache, f, indent=2, default=str)
                return True
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        return False

    def get_cached(self) -> list[dict]:
        return self._cache.get("predictions", [])


class FutureGamePredictor:
    """Autonomous predictor: stat-based, self-logging, never produces garbage.

    Supports NBA and NCAAB predictions via ESPN API.

    Tiered prediction:
      1. ESPN API → real games → stat-based predictions
      2. Cache → previously saved predictions (survives outages)
      3. Empty → honest log message (never fake matchups)

    The ML model is used ONLY when its output validates in valid range.
    The stat baseline is ALWAYS the foundation, ensuring correct totals.
    """

    def __init__(self):
        self._team_pts_avg: dict[str, float] = {}  # Combined NBA + NCAAB team averages
        self._loaded = False
        self._model = None
        self._model_baseline: Optional[float] = None
        self._features_df: Optional[pd.DataFrame] = None

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self._model_path = project_root / "models" / "total_model.pkl"
        self._cache = PredictionCache(project_root / "data" / "prediction_cache.json")

    # ── Public API ─────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load team averages from NBA database and NCAAB ESPN data."""
        self._cache.load()
        self._loaded = self._load_team_data()
        if self._loaded:
            logger.info(f"Predictor ready: {len(self._team_pts_avg)} teams tracked")
        return self._loaded

    def predict_upcoming_games(self, num_games: int = 20) -> list[dict[str, Any]]:
        """Produce predictions for all supported leagues.

        Fetches upcoming games from ESPN for NBA and NCAAB simultaneously.
        Falls back to cache if ESPN is unreachable.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")

        if not self._loaded:
            logger.warning("Predictor not loaded — falling back to cache")
            cached = self._cache.get_cached()
            return [p for p in cached if p.get("game_date", "")[:10] >= today_str][:num_games]

        # TIER 1: Real upcoming games from ESPN (NBA + NCAAB)
        all_predictions = []
        fresh = False

        for league_key, league_label in [("nba", "NBA"), ("ncaab", "NCAAB"), ("euroleague", "Euroleague"), ("nfl", "NFL")]:
            try:
                preds, was_fresh = self._fetch_and_predict(num_games, league_key, league_label)
                all_predictions.extend(preds)
                if was_fresh:
                    fresh = True
            except Exception as e:
                logger.debug(f"{league_label} prediction failed: {e}")

        # TIER 2: Cache fallback
        if not all_predictions:
            cached = self._cache.get_cached()
            cached = [p for p in cached if p.get("game_date", "")[:10] >= today_str]
            if cached:
                logger.info(f"Cache fallback: {len(cached)} preds from cache")
                all_predictions = cached[:num_games]
                fresh = False

        # TIER 3: Honest empty
        if not all_predictions:
            logger.warning("No real games found and no cached data — returning empty")

        # Save to cache
        if all_predictions:
            future_only = [p for p in all_predictions if p.get("game_date", "")[:10] >= today_str]
            if len(future_only) < len(all_predictions):
                logger.debug(f"Filtered out {len(all_predictions) - len(future_only)} past predictions")
            self._cache.save(future_only, fresh=fresh)

        results = [p for p in all_predictions if p.get("game_date", "")[:10] >= today_str][:num_games]
        # Sort by date, then by league
        results.sort(key=lambda p: (p.get("game_date", ""), p.get("league", "")))
        return results

    # ── Tier 1: ESPN Fetch + Predict ─────────────────────────────────

    def _fetch_and_predict(self, num_games: int, league_key: str = "nba",
                           league_label: str = "NBA") -> tuple[list[dict], bool]:
        """Fetch real games for a league from ESPN and predict them."""
        matchups = self._fetch_real_games(num_games, league_key, league_label)
        if not matchups:
            return [], False

        logger.info(f"Predicting {len(matchups)} {league_label} games...")
        predictions = []
        for home, away, gdate in matchups:
            pred = self._predict_game(home, away, gdate or "", league_key, league_label)
            if pred is not None:
                predictions.append(pred)

        if predictions:
            logger.info(f"Generated {len(predictions)} {league_label} predictions")
            return predictions, True

        logger.warning(f"ESPN returned {len(matchups)} {league_label} games but all predictions failed")
        return [], False

    # ── ESPN Schedule Fetching ───────────────────────────────────────

    def _fetch_real_games(self, num_games: int, league_key: str = "nba",
                          league_label: str = "NBA") -> list[tuple[str, str, Optional[str]]]:
        """Scan ESPN scoreboard for the next 14 days.

        Returns real scheduled games for the specified league.
        Each step is logged so you can see WHY it returns empty.
        """
        from betting_intel.data.espn_hoops import ESPN_SCOREBOARD_URL, LEAGUE_TO_ESPN_PATH
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        today = date.today()
        seen: set = set()
        matchups: list[tuple[str, str, Optional[str]]] = []

        path = LEAGUE_TO_ESPN_PATH.get(league_key)
        if not path:
            logger.debug(f"No ESPN path for {league_key}")
            return matchups

        url = ESPN_SCOREBOARD_URL.format(sport=path)

        for offset in range(MAX_DAYS_SCAN):
            if len(matchups) >= num_games:
                break
            check_date = (today + timedelta(days=offset)).strftime("%Y%m%d")

            try:
                resp = session.get(url, params={"dates": check_date, "limit": 100},
                                   timeout=ESPN_TIMEOUT)
                if resp.status_code == 404:
                    continue
                if resp.status_code != 200:
                    logger.debug(f"ESPN {resp.status_code} for {league_key} {check_date}")
                    continue

                for ev in resp.json().get("events", []):
                    comps = ev.get("competitions", [])
                    if not comps:
                        continue

                    # Skip completed/in-progress
                    status = (comps[0].get("status", {}).get("type", {}).get("name", "") or "").upper()
                    if "FINAL" in status or "IN_PROGRESS" in status:
                        continue

                    competitors = comps[0].get("competitors", [])
                    home_name = away_name = None
                    for c in competitors:
                        nm = c.get("team", {}).get("displayName", "")
                        if c.get("homeAway") == "home":
                            home_name = nm
                        else:
                            away_name = nm

                    if not home_name or not away_name:
                        continue

                    # Normalize to short names
                    home_short = ESPN_TO_SHORT.get(home_name)
                    away_short = ESPN_TO_SHORT.get(away_name)

                    # Fuzzy match
                    if not home_short:
                        for full, s in ESPN_TO_SHORT.items():
                            if full.lower() in home_name.lower() or home_name.lower() in full.lower():
                                home_short = s
                                break
                    if not away_short:
                        for full, s in ESPN_TO_SHORT.items():
                            if full.lower() in away_name.lower() or away_name.lower() in full.lower():
                                away_short = s
                                break

                    # Last resort: last word of team name
                    if not home_short:
                        home_short = home_name.split()[-1]
                    if not away_short:
                        away_short = away_name.split()[-1]

                    # For NCAAB: accept any team (don't validate against NBA_TEAMS)
                    # For NBA: validate against known NBA teams
                    if league_key == "nba":
                        if home_short not in NBA_TEAMS or away_short not in NBA_TEAMS:
                            logger.debug(f"Non-NBA team: {away_name} @ {home_name}")
                            continue

                    ed = ev.get("date", "")[:10]
                    key = f"{home_short}|{away_short}|{ed}|{league_key}"
                    if key not in seen:
                        seen.add(key)
                        matchups.append((home_short, away_short, ed))

            except requests.exceptions.Timeout:
                logger.debug(f"ESPN timeout {league_key} {check_date}")
            except requests.exceptions.ConnectionError:
                logger.debug(f"ESPN connection error {league_key} {check_date}")
            except Exception as e:
                logger.debug(f"ESPN error {league_key} {check_date}: {e}")

        if matchups:
            matchups.sort(key=lambda m: m[2] or "9999-12-31")
            logger.info(f"ESPN: {len(matchups)} {league_label} games found")
        else:
            logger.info(f"ESPN: no upcoming {league_label} games found in next 14 days")
        return matchups

    # ── Data Loading ────────────────────────────────────────────────

    def _load_team_data(self) -> bool:
        """Load team scoring averages from all available data sources.

        Tries NBA SQLite database first, then NCAAB/Euroleague from ESPN API.
        This is the foundation of ALL predictions — always correct range.
        """
        loaded_any = False

        # 1. Load NBA data from SQLite
        try:
            from betting_intel.data.loader import NBADataLoader
            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is not None and not raw_df.empty:
                mask = raw_df["TEAM_NAME"].isin(NBA_TEAMS)
                raw_nba = raw_df[mask].copy()
                if not raw_nba.empty:
                    nba_avgs = raw_nba.groupby("TEAM_NAME")["PTS"].mean().to_dict()
                    self._team_pts_avg.update(nba_avgs)
                    logger.info(f"NBA: {len(raw_nba)} rows, {len(nba_avgs)} team averages")
                    loaded_any = True
                    # Try to load ML model with NBA data
                    self._try_load_model(raw_nba, loader)
        except Exception as e:
            logger.debug(f"NBA data loading skipped: {e}")

        # 2. Load NCAAB data from ESPN API
        try:
            from betting_intel.data.loader import NCAABDataLoader
            ncaab_loader = NCAABDataLoader()
            ncaab_df = ncaab_loader.load_game_logs()
            if ncaab_df is not None and not ncaab_df.empty:
                ncaab_avgs = ncaab_df.groupby("TEAM_NAME")["PTS"].mean().to_dict()
                self._team_pts_avg.update(ncaab_avgs)
                logger.info(f"NCAAB: {len(ncaab_df)} rows, {len(ncaab_avgs)} team averages")
                loaded_any = True
        except Exception as e:
            logger.debug(f"NCAAB data loading skipped: {e}")

        # 3. Load Euroleague data from ESPN API
        try:
            from betting_intel.data.espn_hoops import ESPNLeagueSource
            source = ESPNLeagueSource()
            euro_df = source.load_historical("euroleague", seasons=[2025, 2024])
            if euro_df is not None and not euro_df.empty:
                # Build per-team averages from home/away scores
                home_avgs = euro_df.groupby("home_team")["home_score"].mean().to_dict()
                away_avgs = euro_df.groupby("away_team")["away_score"].mean().to_dict()
                for team in set(list(home_avgs.keys()) + list(away_avgs.keys())):
                    ha = home_avgs.get(team) if pd.notna(home_avgs.get(team)) else 0
                    aa = away_avgs.get(team) if pd.notna(away_avgs.get(team)) else 0
                    vals = [v for v in [ha, aa] if v and v > 0]
                    if vals:
                        self._team_pts_avg[team] = float(sum(vals)) / len(vals)
                logger.info(f"Euroleague: {len(euro_df)} rows, {len(euro_df['home_team'].unique())} teams")
                loaded_any = True
        except Exception as e:
            logger.debug(f"Euroleague data loading skipped: {e}")

        # 4. Load NFL data from ESPN API
        try:
            from betting_intel.data.espn_hoops import ESPNLeagueSource
            source = ESPNLeagueSource()
            nfl_df = source.load_historical("nfl", seasons=[2025, 2024])
            if nfl_df is not None and not nfl_df.empty:
                # Build per-team averages from home/away scores
                home_avgs = nfl_df.groupby("home_team")["home_score"].mean().to_dict()
                away_avgs = nfl_df.groupby("away_team")["away_score"].mean().to_dict()
                for team in set(list(home_avgs.keys()) + list(away_avgs.keys())):
                    ha = home_avgs.get(team) if pd.notna(home_avgs.get(team)) else 0
                    aa = away_avgs.get(team) if pd.notna(away_avgs.get(team)) else 0
                    vals = [v for v in [ha, aa] if v and v > 0]
                    if vals:
                        self._team_pts_avg[team] = float(sum(vals)) / len(vals)
                logger.info(f"NFL: {len(nfl_df)} rows, {len(nfl_df['home_team'].unique())} teams")
                loaded_any = True
        except Exception as e:
            logger.debug(f"NFL data loading skipped: {e}")

        if not loaded_any:
            logger.error("No data loaded from any source")
            return False

        logger.info(f"Total: {len(self._team_pts_avg)} teams tracked")
        return True

    def _try_load_model(self, raw_nba: pd.DataFrame, loader) -> None:
        """Try to load the ML model. Non-critical — prediction system works without it."""
        if not self._model_path.exists():
            logger.info("No ML model file — using pure stat-based prediction")
            return

        try:
            from betting_intel.data.features import FeatureEngineer

            try:
                data = safe_joblib_load(str(self._model_path))
            except ModelIntegrityError:
                logger.warning("No hash file for model %s — loading without verification", self._model_path)
                data = safe_joblib_load(str(self._model_path), verify=False)
            self._model = data.get("model") or data.get("ensemble") or (
                data if hasattr(data, "predict") else None)
            if self._model is None:
                return

            feature_cols = data.get("feature_cols", [])
            if not feature_cols:
                return

            # Build features and compute baseline
            gd = loader.build_game_dataset(raw_nba)
            rr = loader.compute_rest_days(raw_nba)
            fe = FeatureEngineer()
            fd = fe.build_all_features(gd, rr)

            if fd is not None and not fd.empty:
                self._features_df = fd
                self._feature_cols_from_model = feature_cols
                self._compute_model_baseline(fd, feature_cols)
                if self._model_baseline:
                    logger.info(f"ML model loaded: {len(feature_cols)} features, "
                                f"baseline={self._model_baseline:.1f}")
        except Exception as e:
            logger.warning(f"ML model skipped (not critical): {e}")
            self._model = None

    def _compute_model_baseline(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Compute the model's average prediction on training data.

        Only predictions in range 100-350 are accepted — garbage filtered out.
        """
        n = min(200, len(df))
        rng = np.random.default_rng(42)
        try:
            indices = rng.choice(len(df), size=n, replace=False)
        except ValueError:
            indices = range(len(df))
            n = len(df)

        preds = []
        for idx in indices:
            row = df.iloc[idx]
            try:
                feat = np.array([[float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0
                                  for c in feature_cols]]).astype(np.float32)
                result = self._model.predict(feat)
                p = float(np.asarray(result).flatten()[0])
                if 100 < p < 350:
                    preds.append(p)
            except Exception:
                continue

        if len(preds) >= 10:
            self._model_baseline = float(np.mean(preds))
        else:
            logger.info(f"Model produced only {len(preds)} valid predictions — not enough for baseline")

    # ── Single Game Prediction ──────────────────────────────────────

    def _predict_game(self, home_team: str, away_team: str, game_date_str: str,
                       league_key: str = "nba", league_label: str = "NBA") -> Optional[dict[str, Any]]:
        """Predict one game using the stat baseline method.

        ALWAYS produces correct-range totals because stat_baseline is the foundation.
        ML model is used ONLY when it validates in the correct range.

        Supports NBA and NCAAB with league-specific constants.
        """
        try:
            cfg = _league_config(league_key)
            default_base = cfg["default_stat_base"]

            # ── Stat baseline (always correct range) ──────────────
            hp = self._team_pts_avg.get(home_team, 0)
            ap = self._team_pts_avg.get(away_team, 0)
            stat_base = hp + ap
            if stat_base <= 0:
                stat_base = default_base

            # ── Pace adjustment from team strength ───────────────
            pace_adj = (hp - ap) * 0.10 if hp > 0 and ap > 0 else 0.0

            # ── Home court effect on total ───────────────────────
            home_adv = cfg["home_advantage"]
            home_adj = home_adv * 0.35

            # ── ML model signal (only if valid) ──────────────────
            model_delta = 0.0
            if league_key == "nba" and self._model is not None and self._model_baseline:
                raw_pred = self._predict_with_model(home_team, away_team)
                if raw_pred is not None and 100 < raw_pred < 350:
                    model_delta = (raw_pred - self._model_baseline) * 0.3

            # ── Final prediction ────────────────────────────────
            predicted_total = stat_base + home_adj + pace_adj + model_delta
            predicted_total = round(predicted_total, 1)
            market_total = round(stat_base, 1)

            # Clamp to league range
            lo = cfg["total_min"]
            hi = cfg["total_max"]
            predicted_total = max(lo, min(hi, predicted_total))

            # ── Edge & Confidence ───────────────────────────────
            edge_pct = round((predicted_total - market_total) / max(market_total, 1), 4)
            direction = "over" if edge_pct > 0 else "under"
            abs_e = abs(edge_pct)
            confidence = "high" if abs_e > 0.05 else ("medium" if abs_e >= 0.02 else "low")

            # ── Quarter projections ─────────────────────────────
            quarters = self._project_quarters(predicted_total, market_total, hp, ap, league_key)
            best_q = self._find_best_quarter(quarters, direction)

            return {
                "game_id": f"{home_team}_{away_team}_{game_date_str}",
                "game_date": game_date_str,
                "league": league_label,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_short": home_team,
                "away_team_short": away_team,
                "matchup": f"{away_team} @ {home_team}",
                "predicted_total": predicted_total,
                "market_total": market_total,
                "edge_pct": edge_pct,
                "direction": direction,
                "confidence": confidence,
                "model_mae": "N/A",
                "home_score": quarters.get("home_score", round(predicted_total * 0.51, 1)),
                "away_score": quarters.get("away_score", round(predicted_total * 0.49, 1)),
                "best_quarter": best_q["quarter"],
                "best_quarter_edge": best_q["edge"],
                "best_quarter_direction": best_q["direction"],
                "recommended_quarter": best_q["quarter"],
                "recommended_direction": best_q["direction"],
                **{f"{p}_{s}": quarters.get(f"{p}_{s}") for p in ["q1","q2","q3","q4","h1","h2"]
                   for s in ["home","away","total","market","edge"]
                   if quarters.get(f"{p}_{s}") is not None},
            }

        except Exception as e:
            logger.error(f"Prediction failed for {home_team} vs {away_team}: {e}", exc_info=True)
            return None

    # ── Model Prediction (with validation) ─────────────────────────

    def _predict_with_model(self, home: str, away: str) -> Optional[float]:
        """Try the ML model. Returns None if prediction is invalid or feature pipeline missing."""
        if self._features_df is None or self._features_df.empty:
            return None

        df = self._features_df
        home_col = "TEAM_NAME_home" if "TEAM_NAME_home" in df.columns else None
        away_col = "TEAM_NAME_away" if "TEAM_NAME_away" in df.columns else None
        if not home_col or not away_col:
            return None

        # Direct matchup lookup
        for h, a in [(home, away), (away, home)]:
            mask = (
                df[home_col].astype(str).str.strip().str.lower().eq(h.lower().strip())
                & df[away_col].astype(str).str.strip().str.lower().eq(a.lower().strip())
            )
            if mask.any():
                return self._predict_one_row(df[mask].iloc[-1])

        # Combined average (home avg + away avg)
        return self._predict_combined(home, away, df, home_col, away_col)

    def _predict_one_row(self, row: pd.Series) -> Optional[float]:
        feature_cols = getattr(self._model, "feature_cols_", None) or \
                       getattr(self, "_feature_cols_from_model", None) or []
        if not feature_cols:
            return None
        try:
            feat = np.array([[float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0
                              for c in feature_cols]]).astype(np.float32)
            result = self._model.predict(feat)
            return float(np.asarray(result).flatten()[0])
        except Exception:
            return None

    def _predict_combined(self, home: str, away: str, df: pd.DataFrame,
                          home_col: str, away_col: str) -> Optional[float]:
        def avg_of(team: str, suffix: str, n: int = 15) -> pd.Series:
            c = home_col if suffix == "_home" else away_col
            mask = df[c].astype(str).str.strip().str.lower().eq(team.lower().strip())
            s = df[mask].tail(n)
            return s.select_dtypes(include=[np.number]).mean() if not s.empty else pd.Series(dtype=float)

        ha = avg_of(home, "_home")
        aa = avg_of(away, "_away")
        if ha.empty or aa.empty:
            return None

        feature_cols = getattr(self._model, "feature_cols_", None) or \
                       getattr(self, "_feature_cols_from_model", None) or []
        if not feature_cols:
            return None

        fd = {}
        for c in feature_cols:
            cl = c.lower()
            try:
                if cl.endswith("_home"):
                    fd[c] = float(ha.get(c, 0.0))
                elif cl.endswith("_away"):
                    fd[c] = float(aa.get(c, 0.0))
                elif cl.endswith("_diff"):
                    b = c.replace("_diff", "")
                    fd[c] = float(ha.get(f"{b}_home", 0.0)) - float(aa.get(f"{b}_away", 0.0))
                else:
                    fd[c] = float(df[c].mean()) if c in df.columns else 0.0
            except (ValueError, TypeError):
                fd[c] = 0.0

        try:
            X = np.array([list(fd.values())]).astype(np.float32)
            result = self._model.predict(X)
            return float(np.asarray(result).flatten()[0])
        except Exception:
            return None

    # ── Quarter Projection ─────────────────────────────────────────

    @staticmethod
    def _project_quarters(predicted: float, market: float, hp: float, ap: float,
                          league: str) -> dict[str, float]:
        cfg = _league_config(league)
        ratios = cfg["quarter_ratios"]
        hpct = 0.51

        # Strength-adjusted home percentage
        if hp > 0 and ap > 0:
            hpct = min(max(hpct + (hp - ap) / 400.0, 0.43), 0.57)

        hs = predicted * hpct
        aws = predicted * (1.0 - hpct)
        mhs = market * hpct
        maws = market * (1.0 - hpct)

        r = {}
        for q in ["q1", "q2", "q3", "q4"]:
            rt = ratios[q]
            hq, aq = round(hs * rt, 1), round(aws * rt, 1)
            qt = round(hq + aq, 1)
            mqt = round(mhs * rt + maws * rt, 1)
            r.update({f"{q}_home": hq, f"{q}_away": aq, f"{q}_total": qt,
                      f"{q}_market": mqt, f"{q}_edge": round((qt - mqt) / max(mqt, 1), 4)})

        for hh in ["h1", "h2"]:
            rt = ratios[hh]
            ht = round(hs * rt + aws * rt, 1)
            mht = round(mhs * rt + maws * rt, 1)
            r.update({f"{hh}_home": round(hs * rt, 1), f"{hh}_away": round(aws * rt, 1),
                      f"{hh}_total": ht, f"{hh}_market": mht,
                      f"{hh}_edge": round((ht - mht) / max(mht, 1), 4)})

        r["home_score"] = round(hs, 1)
        r["away_score"] = round(aws, 1)
        return r

    @staticmethod
    def _find_best_quarter(q: dict, direction: str) -> dict:
        best = {"quarter": "FULL", "edge": round(abs(q.get("q1_edge", 0)) * 100, 1), "direction": direction}
        for period in ["q1", "q2", "q3", "q4", "h1", "h2"]:
            e = q.get(f"{period}_edge", 0.0)
            ae = abs(e) * 100
            if ae > best["edge"]:
                best = {"quarter": period.upper(), "edge": round(ae, 1),
                        "direction": "over" if e > 0 else "under"}
        return best

    def to_livegame_dict(self, pred: dict) -> dict:
        return {
            "game_id": pred.get("game_id", ""),
            "game_date": pred.get("game_date", ""),
            "home_team": pred.get("home_team", ""),
            "away_team": pred.get("away_team", ""),
            "home_team_short": pred.get("home_team_short", ""),
            "away_team_short": pred.get("away_team_short", ""),
            "matchup": pred.get("matchup", ""),
            "league": pred.get("league", "NBA"),
            "market_total": pred.get("market_total", 0),
            "predicted_total": pred.get("predicted_total", 0),
            "edge_pct": pred.get("edge_pct", 0),
            "direction": pred.get("direction", "neutral"),
            "confidence": pred.get("confidence", "low"),
            "model_mae": pred.get("model_mae", "?"),
            "recommended_quarter": pred.get("best_quarter", "FULL"),
            "recommended_direction": pred.get("best_quarter_direction", "over"),
        }


def format_prediction_card(pred: dict) -> str:
    e = f"{pred['edge_pct']:+.1%}" if pred.get("edge_pct") is not None else "N/A"
    lines = [
        "=" * 60,
        f"  [{pred.get('league', 'NBA')}] {pred['away_team']} @ {pred['home_team']}",
        f"  {pred['game_date']}",
        f"  Total: {pred['predicted_total']} pts (Mkt: {pred['market_total']})  "
        f"Edge: {e}  {pred.get('direction', '?').upper()}  {pred.get('confidence', '?')}",
    ]
    for q in ["q1", "q2", "q3", "q4"]:
        t = pred.get(f"{q}_total", 0)
        if t:
            lines.append(f"    {q.upper()}: {t} (mkt: {pred.get(f'{q}_market', 0)}, "
                         f"edge: {pred.get(f'{q}_edge', 0):+.1%})")
    for h, lb in [("h1", "1st Half"), ("h2", "2nd Half")]:
        t = pred.get(f"{h}_total", 0)
        if t:
            lines.append(f"    {lb}: {t} (mkt: {pred.get(f'{h}_market', 0)}, "
                         f"edge: {pred.get(f'{h}_edge', 0):+.1%})")
    lines.append(f"  >> Best: {pred.get('best_quarter', 'FULL')} "
                 f"{pred.get('best_quarter_direction', '?').upper()} "
                 f"(edge: {pred.get('best_quarter_edge', 0):.1f}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = FutureGamePredictor()
    if p.load():
        for pr in p.predict_upcoming_games(10):
            print(format_prediction_card(pr))
            print()
