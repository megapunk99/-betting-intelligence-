"""
FutureGamePredictor — GENUINELY AUTONOMOUS predictions.

DESIGN PRINCIPLES:
  - ALWAYS produces correct-range totals (NBA 200-260, WNBA 140-190)
  - Stat baseline is the foundation; ML model is an ADDITIONAL signal only
  - Every failure mode logs WHY — never silent
  - Caches predictions so outages never break the dashboard
  - NEVER generates fake matchups — returns empty with clear log

ARCHITECTURE:
  1. Fetch REAL upcoming games from ESPN API (next 14 days, NBA + WNBA)
  2. For every game: predicted_total = stat_baseline + team_strength_adjustment
  3. If ML model produces VALID output (100-350), use its delta too
  4. market_total = stat_baseline (best estimate without real odds)
  5. Cascade to Q1-Q4 and 1H-2H using league-specific quarter ratios
  6. Cache to disk so dashboard works during API/network outages
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── NBA Constants ─────────────────────────────────────────────────────────
NBA_TEAMS = {"Hawks","Celtics","Nets","Hornets","Bulls","Cavaliers","Mavericks",
             "Nuggets","Pistons","Warriors","Rockets","Pacers","Clippers","Lakers",
             "Grizzlies","Heat","Bucks","Timberwolves","Pelicans","Knicks","Thunder",
             "Magic","76ers","Suns","Trail Blazers","Kings","Spurs","Raptors","Jazz","Wizards"}

NBA_QUARTER_RATIOS   = {"q1": 0.242, "q2": 0.251, "q3": 0.253, "q4": 0.254, "h1": 0.493, "h2": 0.507}
NBA_HOME_ADVANTAGE   = 2.3    # pts added for home court
NBA_TOTAL_MIN        = 200.0
NBA_TOTAL_MAX        = 260.0

ESPN_TO_SHORT = {
    "Atlanta Hawks":"Hawks","Boston Celtics":"Celtics","Brooklyn Nets":"Nets",
    "Charlotte Hornets":"Hornets","Chicago Bulls":"Bulls","Cleveland Cavaliers":"Cavaliers",
    "Dallas Mavericks":"Mavericks","Denver Nuggets":"Nuggets","Detroit Pistons":"Pistons",
    "Golden State Warriors":"Warriors","Houston Rockets":"Rockets","Indiana Pacers":"Pacers",
    "LA Clippers":"Clippers","Los Angeles Clippers":"Clippers","Los Angeles Lakers":"Lakers",
    "LA Lakers":"Lakers","Memphis Grizzlies":"Grizzlies","Miami Heat":"Heat",
    "Milwaukee Bucks":"Bucks","Minnesota Timberwolves":"Timberwolves",
    "New Orleans Pelicans":"Pelicans","New York Knicks":"Knicks",
    "Oklahoma City Thunder":"Thunder","Orlando Magic":"Magic","Philadelphia 76ers":"76ers",
    "Phoenix Suns":"Suns","Portland Trail Blazers":"Trail Blazers",
    "Sacramento Kings":"Kings","San Antonio Spurs":"Spurs","Toronto Raptors":"Raptors",
    "Utah Jazz":"Jazz","Washington Wizards":"Wizards",
}

# ── WNBA Constants ────────────────────────────────────────────────────────
WNBA_TEAMS = {"Dream","Sky","Sun","Wings","Fever","Aces","Sparks","Lynx",
              "Liberty","Mercury","Storm","Mystics"}

WNBA_QUARTER_RATIOS  = {"q1": 0.240, "q2": 0.250, "q3": 0.253, "q4": 0.257, "h1": 0.490, "h2": 0.510}
WNBA_HOME_ADVANTAGE  = 1.5
WNBA_TOTAL_MIN       = 140.0
WNBA_TOTAL_MAX       = 190.0

WNBA_ESPN_TO_SHORT = {
    "Atlanta Dream":"Dream","Chicago Sky":"Sky","Connecticut Sun":"Sun",
    "Dallas Wings":"Wings","Indiana Fever":"Fever","Las Vegas Aces":"Aces",
    "Los Angeles Sparks":"Sparks","Minnesota Lynx":"Lynx","New York Liberty":"Liberty",
    "Phoenix Mercury":"Mercury","Seattle Storm":"Storm","Washington Mystics":"Mystics",
}
WNBA_TEAM_PTS = {"Aces":87.0,"Liberty":86.0,"Sun":83.0,"Fever":82.0,"Lynx":82.0,
                 "Mercury":81.0,"Storm":81.0,"Dream":80.0,"Sky":79.0,"Mystics":78.0,
                 "Sparks":78.0,"Wings":77.0}

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

    Tiered prediction:
      1. ESPN API → real games → stat-based predictions
      2. Cache → previously saved predictions (survives outages)
      3. Empty → honest log message (never fake matchups)

    The ML model is used ONLY when its output validates (100-350 range).
    The stat baseline is ALWAYS the foundation, ensuring correct totals.
    """

    def __init__(self):
        self._team_pts_avg: dict[str, float] = {}
        self._loaded = False
        self._model = None
        self._model_baseline: Optional[float] = None
        self._features_df: Optional[pd.DataFrame] = None

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self._model_path = project_root / "models" / "total_model.pkl"
        self._cache = PredictionCache(project_root / "data" / "prediction_cache.json")

    # ── Public API ─────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load team averages from the NBA database and try the ML model."""
        self._cache.load()
        self._loaded = self._load_team_data()
        if self._loaded:
            logger.info(f"Predictor ready: {len(self._team_pts_avg)} teams tracked")
        return self._loaded

    def predict_upcoming_games(self, num_games: int = 20) -> list[dict[str, Any]]:
        """Produce predictions. Never silently returns empty. Never fake matchups."""
        if not self._loaded:
            logger.warning("Predictor not loaded — falling back to cache")
            return self._cache.get_cached()[:num_games]

        # TIER 1: Real upcoming games from ESPN
        predictions, fresh = self._fetch_and_predict(num_games)

        # TIER 2: Cache fallback
        if not predictions:
            cached = self._cache.get_cached()
            if cached:
                logger.info(f"Cache fallback: {len(cached)} preds from cache")
                predictions = cached[:num_games]
                fresh = False

        # TIER 3: Honest empty
        if not predictions:
            logger.warning("No real games found and no cached data — returning empty")

        # Save to cache if we got predictions
        if predictions:
            self._cache.save(predictions, fresh=fresh)

        return predictions[:num_games]

    # ── Tier 1: ESPN Fetch + Predict ─────────────────────────────────

    def _fetch_and_predict(self, num_games: int) -> tuple[list[dict], bool]:
        """Fetch real games from ESPN and predict them using stat baselines."""
        matchups = self._fetch_real_games(num_games)
        if not matchups:
            return [], False

        logger.info(f"Predicting {len(matchups)} real games...")
        predictions = []
        for home, away, gdate in matchups:
            pred = self._predict_game(home, away, gdate or "")
            if pred is not None:
                predictions.append(pred)

        if predictions:
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions, True

        logger.warning(f"ESPN returned {len(matchups)} games but all predictions failed")
        return [], False

    # ── ESPN Schedule Fetching ───────────────────────────────────────

    def _fetch_real_games(self, num_games: int) -> list[tuple[str, str, Optional[str]]]:
        """Scan ESPN scoreboard for the next 14 days. Returns real scheduled games.

        Each step is logged so you can see WHY it returns empty.
        """
        from betting_intel.data.espn_hoops import ESPN_SCOREBOARD_URL, LEAGUE_TO_ESPN_PATH
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        today = date.today()
        seen: set = set()
        matchups: list[tuple[str, str, Optional[str]]] = []

        for league_key in ["nba", "wnba"]:
            path = LEAGUE_TO_ESPN_PATH.get(league_key)
            if not path:
                logger.debug(f"No ESPN path for {league_key}")
                continue

            url = ESPN_SCOREBOARD_URL.format(sport=path)
            name_map = ESPN_TO_SHORT if league_key == "nba" else WNBA_ESPN_TO_SHORT
            team_set = NBA_TEAMS if league_key == "nba" else WNBA_TEAMS
            games_found = 0

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
                        home_short = name_map.get(home_name)
                        away_short = name_map.get(away_name)

                        # Fuzzy match for NBA (ESPN sometimes uses different formatting)
                        if league_key == "nba":
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

                        # Validate against known teams
                        if home_short not in team_set or away_short not in team_set:
                            logger.debug(f"Non-{league_key} team: {away_name} @ {home_name}")
                            continue

                        ed = ev.get("date", "")[:10]
                        key = f"{home_short}|{away_short}|{ed}"
                        if key not in seen:
                            seen.add(key)
                            matchups.append((home_short, away_short, ed))
                            games_found += 1

                except requests.exceptions.Timeout:
                    logger.debug(f"ESPN timeout {league_key} {check_date}")
                except requests.exceptions.ConnectionError:
                    logger.debug(f"ESPN connection error {league_key} {check_date}")
                except Exception as e:
                    logger.debug(f"ESPN error {league_key} {check_date}: {e}")

            if games_found:
                logger.info(f"{league_key.upper()}: {games_found} upcoming games")

        if matchups:
            matchups.sort(key=lambda m: m[2] or "9999-12-31")
            logger.info(f"ESPN: {len(matchups)} total games found")
        else:
            logger.info("ESPN: no upcoming games found in next 14 days")
        return matchups

    # ── Data Loading ────────────────────────────────────────────────

    def _load_team_data(self) -> bool:
        """Load team scoring averages from the NBA database.

        This is the foundation of ALL predictions — always correct range.
        """
        try:
            from betting_intel.data.loader import NBADataLoader
            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.error("NBA database empty")
                return False

            # Filter to known NBA teams
            mask = raw_df["TEAM_NAME"].isin(NBA_TEAMS)
            raw_nba = raw_df[mask].copy()
            if not raw_nba.empty:
                team_pts = raw_nba.groupby("TEAM_NAME")["PTS"].mean().to_dict()
                for t, v in WNBA_TEAM_PTS.items():
                    team_pts.setdefault(t, v)
                self._team_pts_avg = team_pts
                logger.info(f"NBA: {len(raw_nba)} rows, {len(team_pts)} team averages")

                # Try to build feature pipeline for ML model (non-critical)
                self._try_load_model(raw_nba, loader)
            else:
                # Use WNBA defaults for everything
                self._team_pts_avg = dict(WNBA_TEAM_PTS)
                logger.info("No NBA data — using WNBA defaults only")

            return True
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False

    def _try_load_model(self, raw_nba: pd.DataFrame, loader) -> None:
        """Try to load the ML model. Non-critical — prediction system works without it."""
        if not self._model_path.exists():
            logger.info("No ML model file — using pure stat-based prediction")
            return

        try:
            import joblib
            from betting_intel.data.features import FeatureEngineer

            data = joblib.load(str(self._model_path))
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

    def _predict_game(self, home_team: str, away_team: str, game_date_str: str) -> Optional[dict[str, Any]]:
        """Predict one game using the stat baseline method.

        ALWAYS produces correct-range totals because stat_baseline is the foundation.
        ML model is used ONLY when it validates (predicts 100-350).
        """
        # Determine league
        if home_team in NBA_TEAMS and away_team in NBA_TEAMS:
            league = "nba"
            league_label = "NBA"
        elif home_team in WNBA_TEAMS and away_team in WNBA_TEAMS:
            league = "wnba"
            league_label = "WNBA"
        else:
            logger.debug(f"Unknown league: {home_team} vs {away_team}")
            return None

        try:
            # ── Stat baseline (always correct range) ──────────────
            hp = self._team_pts_avg.get(home_team, 0)
            ap = self._team_pts_avg.get(away_team, 0)
            stat_base = hp + ap
            if stat_base <= 0:
                stat_base = 228.0 if league == "nba" else 165.0

            # ── Pace adjustment from team strength ───────────────
            # Stronger home team = faster pace = more total points
            # Weaker home team = slower pace = fewer total points
            # Max realistic NBA team diff is ~15 pts → pace adjustment ±1.5
            pace_adj = (hp - ap) * 0.10 if hp > 0 and ap > 0 else 0.0

            # ── Home court effect on total ───────────────────────
            # Home court adds ~2.3 pts to home score, slightly reduces away score
            # Net effect on total is about +0.8 pts
            home_adv = NBA_HOME_ADVANTAGE if league == "nba" else WNBA_HOME_ADVANTAGE
            home_adj = home_adv * 0.35

            # ── ML model signal (only if valid) ──────────────────
            model_delta = 0.0
            if league == "nba" and self._model is not None and self._model_baseline:
                raw_pred = self._predict_with_model(home_team, away_team)
                if raw_pred is not None and 100 < raw_pred < 350:
                    model_delta = (raw_pred - self._model_baseline) * 0.3  # diluted signal
                    logger.debug(f"Model signal: {home_team} vs {away_team}: "
                                 f"pred={raw_pred:.1f}, baseline={self._model_baseline:.1f}, "
                                 f"delta={model_delta:.1f}")

            # ── Final prediction ────────────────────────────────
            predicted_total = stat_base + home_adj + pace_adj + model_delta
            predicted_total = round(predicted_total, 1)
            market_total = round(stat_base, 1)

            # Clamp to league range
            lo = NBA_TOTAL_MIN if league == "nba" else WNBA_TOTAL_MIN
            hi = NBA_TOTAL_MAX if league == "nba" else WNBA_TOTAL_MAX
            predicted_total = max(lo, min(hi, predicted_total))

            # ── Edge & Confidence ───────────────────────────────
            edge_pct = round((predicted_total - market_total) / max(market_total, 1), 4)
            direction = "over" if edge_pct > 0 else "under"
            abs_e = abs(edge_pct)
            confidence = "high" if abs_e > 0.05 else ("medium" if abs_e >= 0.02 else "low")

            # ── Quarter projections ─────────────────────────────
            quarters = self._project_quarters(predicted_total, market_total, hp, ap, league)
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
        ratios = WNBA_QUARTER_RATIOS if league == "wnba" else NBA_QUARTER_RATIOS
        hpct = 0.52 if league == "wnba" else 0.51

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
