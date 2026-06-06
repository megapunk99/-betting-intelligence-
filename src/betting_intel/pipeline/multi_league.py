"""
Multi-League Prediction Module — trains and predicts for all basketball leagues.

This module bridges the gap between the ESPN API data source and the
LightGBM model training pipeline. It:

1. Fetches historical data from ESPN for each league (NBA, WNBA, NCAAB)
2. Engineers features using MultiLeagueFeatureEngineer
3. Trains per-league LightGBM models with team stats caching
4. Makes predictions on upcoming games using trained models + cached team stats
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from betting_intel.data.basketball_leagues import (
    LEAGUES_WITH_ESPN_API,
    LEAGUE_BY_KEY,
    BasketballLeague,
)
from betting_intel.data.espn_hoops import ESPNLeagueSource
from betting_intel.data.multi_league_features import (
    MultiLeagueFeatureEngineer,
    train_league_model,
    load_league_model,
)

logger = logging.getLogger(__name__)

# ── Team stats cache (populated during training) ──────────────────────

_team_stats_cache: dict[str, dict[str, dict[str, float]]] = {}

def _get_team_recent_stats(league_key: str, team: str) -> Optional[dict[str, float]]:
    """Get recent rolling stats for a specific team from cache."""
    return _team_stats_cache.get(league_key, {}).get(team)


def cache_team_stats(
    league_key: str,
    team_logs: pd.DataFrame,
):
    """Store per-team rolling averages for prediction-time feature building.

    Takes the team_logs DataFrame (produced by MultiLeagueFeatureEngineer._build_team_logs)
    and caches the most recent rolling stats for each team. These cached stats
    are used at prediction time to build feature vectors for upcoming games.
    """
    if league_key not in _team_stats_cache:
        _team_stats_cache[league_key] = {}

    for _, row in team_logs.iterrows():
        team = row.get("team", "")
        if not team:
            continue
        if team not in _team_stats_cache[league_key]:
            _team_stats_cache[league_key][team] = {}
        team_data = _team_stats_cache[league_key][team]
        for w in [5, 10]:
            col = f"avg_pts_{w}g"
            if col in row.index and pd.notna(row[col]):
                team_data[col] = float(row[col])
            col_opp = f"avg_pts_allowed_{w}g"
            if col_opp in row.index and pd.notna(row[col_opp]):
                team_data[col_opp] = float(row[col_opp])
            col_margin = f"avg_margin_{w}g"
            if col_margin in row.index and pd.notna(row[col_margin]):
                team_data[col_margin] = float(row[col_margin])
        team_data["win_rate_10g"] = float(row.get("win_rate_10g", 0.5))
        team_data["last_3_margin"] = float(row.get("last_3_margin", 0.0))


def get_cached_leagues() -> list[str]:
    """Return list of league keys that have cached team stats."""
    return list(_team_stats_cache.keys())


# ── Training ───────────────────────────────────────────────────────────

def train_all_basketball_models(
    leagues: Optional[list[BasketballLeague]] = None,
    seasons_per_league: Optional[dict[str, list]] = None,
    feature_version: str = "v3",
    output_dir: str = "models/saved",
) -> dict[str, dict[str, Any]]:
    """
    Train LightGBM models for ALL basketball leagues with ESPN data.

    Available leagues with ESPN API: NBA, WNBA, NCAAB.
    Euroleague, NBL, etc. require a different data source (TheOddsAPI
    for odds only, no historical game data).

    Args:
        leagues: Leagues to train. Defaults to all leagues with ESPN API.
        seasons_per_league: Override seasons per league key.
        output_dir: Where to save models

    Returns:
        Dict[league_key, {status, n_games, ...}]
    """
    if leagues is None:
        leagues = [lg for lg in LEAGUES_WITH_ESPN_API if lg.train_model]

    print("\n" + "█" * 70)
    print("  🏀  MULTI-LEAGUE MODEL TRAINING")
    print("█" * 70)
    print(f"  Leagues to train: {', '.join(lg.key for lg in leagues)}")

    source = ESPNLeagueSource()
    engineer = MultiLeagueFeatureEngineer()

    results: dict[str, dict[str, Any]] = {}

    for league in leagues:
        print(f"\n  {'=' * 60}")
        print(f"  Training {league.name} ({league.key})")
        print(f"  {'=' * 60}")

        # Determine seasons
        if seasons_per_league and league.key in seasons_per_league:
            seasons = seasons_per_league[league.key]
        else:
            seasons = _default_seasons(league)

        # Fetch historical data
        print(f"  Fetching {league.key} data from ESPN: seasons {seasons}...")
        df = source.load_historical(league.key, seasons=seasons)
        if df is None or df.empty:
            print(f"  No data for {league.key} — skipping")
            results[league.key] = {"status": "skipped", "reason": "no_data", "n_games": 0}
            continue

        if len(df) < league.min_games_for_model:
            print(f"  Only {len(df)} games for {league.key} (need {league.min_games_for_model}) — skipping")
            results[league.key] = {"status": "skipped", "reason": "insufficient_data", "n_games": len(df)}
            continue

        print(f"  Fetched {len(df)} games for {league.key}")

        # Engineer features
        print(f"  Engineering features...")
        features_df = engineer.build_features(df)
        if features_df is None or features_df.empty:
            print(f"  Feature engineering failed for {league.key}")
            results[league.key] = {"status": "failed", "reason": "feature_engineering", "n_games": len(df)}
            continue

        feature_cols = engineer.auto_select_features(features_df)
        if len(feature_cols) < 3:
            print(f"  Too few features ({len(feature_cols)}) for {league.key}")
            results[league.key] = {"status": "failed", "reason": "too_few_features", "n_games": len(df)}
            continue

        print(f"  Engineered {len(feature_cols)} features from {len(features_df)} rows")

        # ── CRITICAL: Cache team stats for prediction-time feature building ──
        team_logs = engineer.team_logs
        if team_logs is not None:
            # Build the rolling stat columns on team logs before caching
            team_logs_rolled = team_logs.copy()
            team_logs_rolled = team_logs_rolled.sort_values(["team", "date"])
            for w in [5, 10]:
                team_logs_rolled[f"avg_pts_{w}g"] = (
                    team_logs_rolled.groupby("team")["pts"]
                    .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
                )
                team_logs_rolled[f"avg_pts_allowed_{w}g"] = (
                    team_logs_rolled.groupby("team")["opp_pts"]
                    .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
                )
                team_logs_rolled["margin"] = team_logs_rolled["pts"] - team_logs_rolled["opp_pts"]
                team_logs_rolled[f"avg_margin_{w}g"] = (
                    team_logs_rolled.groupby("team")["margin"]
                    .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
                )
            team_logs_rolled["win_rate_10g"] = (
                team_logs_rolled.groupby("team")["win"]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )
            team_logs_rolled["last_3_margin"] = (
                team_logs_rolled.groupby("team")["margin"]
                .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            )
            cache_team_stats(league.key, team_logs_rolled)
            n_cached = len(_team_stats_cache.get(league.key, {}))
            print(f"  Cached stats for {n_cached} teams")

        # Train model
        model = train_league_model(
            league_key=league.key,
            features_df=features_df,
            feature_cols=feature_cols,
            model_dir=output_dir,
        )

        if model is not None:
            results[league.key] = {
                "status": "trained",
                "n_games": len(df),
                "n_features": len(feature_cols),
                "model_type": "LightGBM",
            }
        else:
            results[league.key] = {
                "status": "failed",
                "reason": "training_failed",
                "n_games": len(df),
            }

    # Print summary
    print(f"\n  {'=' * 60}")
    print(f"  MULTI-LEAGUE TRAINING SUMMARY")
    print(f"  {'=' * 60}")
    trained = [k for k, v in results.items() if v.get("status") == "trained"]
    skipped = [k for k, v in results.items() if v.get("status") != "trained"]
    print(f"  Trained: {', '.join(trained) if trained else 'none'}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    print(f"  {'=' * 60}")

    return results


# ── Prediction ─────────────────────────────────────────────────────────

def predict_league_games(
    league_key: str,
    upcoming_games: pd.DataFrame,
    model: Optional[object] = None,
    feature_cols: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Predict upcoming games for a specific league using its trained model.

    Args:
        league_key: 'wnba', 'ncaab', 'nbl', etc.
        upcoming_games: DataFrame with home_team, away_team, game_date, market_total columns
        model: Pre-loaded model (optional — loads from disk if not provided)
        feature_cols: Feature columns matching the model (required if model is pre-loaded)

    Returns:
        List of prediction dicts with edge_pct, direction, etc.
    """
    if model is None:
        model, feature_cols = load_league_model(league_key)
    if model is None or not feature_cols:
        return []

    league = LEAGUE_BY_KEY.get(league_key)
    if league is None:
        return []

    print(f"\n  {'=' * 60}")
    print(f"  Predicting {league_key.upper()} games ({len(upcoming_games)} games)")
    print(f"  {'=' * 60}")

    predictions: list[dict[str, Any]] = []

    for idx, row in upcoming_games.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        game_date = row.get("game_date", str(date.today()))
        game_id = row.get("game_id", f"{league_key}_{idx}")

        if not home or not away:
            continue

        # Build feature vector from cached team stats
        feat = _build_prediction_vector(league_key, home, away, feature_cols, league)
        if feat is None:
            predicted_total = league.avg_total
        else:
            try:
                X_pred = feat.values.reshape(1, -1)
                raw_pred = model.predict(X_pred)
                predicted_total = float(np.asarray(raw_pred).flatten()[0])
            except Exception as e:
                logger.debug(f"Predict failed for {home} vs {away}: {e}")
                predicted_total = league.avg_total

        # Get market total from odds if available
        market_total = row.get("market_total", 0)
        if market_total is None or market_total <= 0:
            market_total = league.avg_total

        # Validate — filter garbage predictions
        max_sane = league.avg_total * 2.0
        min_sane = league.avg_total * 0.3
        if predicted_total > max_sane or predicted_total < min_sane:
            print(f"  Unreasonable {league_key} prediction: {predicted_total:.0f} pts — using league avg")
            predicted_total = league.avg_total

        edge = (predicted_total - market_total) / max(market_total, 1)
        direction = "over" if edge > 0 else "under"
        abs_edge = abs(edge)

        game_pred = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "game_date": game_date,
            "league": league_key,
            "predicted_total": round(predicted_total, 1),
            "market_total": market_total,
            "edge_pct": round(edge, 4),
            "direction": direction,
            "confidence": "high" if abs_edge > 0.05 else ("medium" if abs_edge > 0.02 else "low"),
            "implied_odds": {
                "home_moneyline": row.get("home_ml_odds", -110),
                "away_moneyline": row.get("away_ml_odds", -110),
            },
        }
        predictions.append(game_pred)

        arrow = "🟢" if abs_edge > 0.03 else ("🔵" if abs_edge > 0.01 else "⚪")
        print(f"  {arrow}  {home:20s} vs {away:<20s}  "
              f"pred={predicted_total:.1f}  mkt={market_total}  edge={edge:+.2%}  {direction}")

    if predictions:
        print(f"  {league_key.upper()}: {len(predictions)} predictions generated")
    else:
        print(f"  {league_key.upper()}: No valid predictions")

    return predictions


# ── Helpers ────────────────────────────────────────────────────────────

def _default_seasons(league: BasketballLeague) -> list:
    """Determine default seasons to fetch based on current date."""
    now = datetime.now()
    if league.key in ("nba", "ncaab", "wncaab", "euroleague", "euroleague_women"):
        if now.month >= 10:
            return [now.year, now.year - 1, now.year - 2]
        else:
            return [now.year - 1, now.year - 2, now.year - 3]
    elif league.key == "wnba":
        return [now.year, now.year - 1, now.year - 2]
    elif league.key == "nbl":
        return [now.year - 1, now.year - 2]
    else:
        return [now.year - 1, now.year - 2, now.year - 3]


def _build_prediction_vector(
    league_key: str,
    home_team: str,
    away_team: str,
    feature_cols: list[str],
    league: Optional[BasketballLeague] = None,
) -> Optional[pd.Series]:
    """
    Build a feature vector for a specific matchup from cached team stats.

    Uses the per-team rolling averages cached during training. If a team
    has no cached stats, falls back to league-average default values so
    the model still produces reasonable predictions.
    """
    team_stats = _get_team_recent_stats(league_key, home_team)
    team_stats_away = _get_team_recent_stats(league_key, away_team)

    # Default values from league config
    default_pts = league.avg_total / 2 if league else 0.0
    default_pts_allowed = league.avg_total / 2 if league else 0.0
    default_margin = league.avg_margin if league else 0.0

    def _get_stat(team_dict: Optional[dict], key: str, default: float = 0.0) -> float:
        """Get a stat from team cache with optional fallback."""
        if team_dict:
            return team_dict.get(key, default)
        return default

    feature_dict: dict[str, float] = {}
    for col in feature_cols:
        # Rolling averages per side
        if col.startswith("avg_pts_") and col.endswith("_home"):
            parts = col.replace("avg_pts_", "").replace("g_home", "")
            val = _get_stat(team_stats, f"avg_pts_{parts}g", default_pts)
        elif col.startswith("avg_pts_") and col.endswith("_away"):
            parts = col.replace("avg_pts_", "").replace("g_away", "")
            val = _get_stat(team_stats_away, f"avg_pts_{parts}g", default_pts)
        elif col.startswith("avg_pts_allowed_") and col.endswith("_home"):
            parts = col.replace("avg_pts_allowed_", "").replace("g_home", "")
            val = _get_stat(team_stats, f"avg_pts_allowed_{parts}g", default_pts_allowed)
        elif col.startswith("avg_pts_allowed_") and col.endswith("_away"):
            parts = col.replace("avg_pts_allowed_", "").replace("g_away", "")
            val = _get_stat(team_stats_away, f"avg_pts_allowed_{parts}g", default_pts_allowed)
        elif col.startswith("avg_margin_") and col.endswith("_home"):
            parts = col.replace("avg_margin_", "").replace("g_home", "")
            val = _get_stat(team_stats, f"avg_margin_{parts}g", default_margin)
        elif col.startswith("avg_margin_") and col.endswith("_away"):
            parts = col.replace("avg_margin_", "").replace("g_away", "")
            val = _get_stat(team_stats_away, f"avg_margin_{parts}g", -default_margin)
        elif col.startswith("ema_pts_") and col.endswith("_home"):
            parts = col.replace("ema_pts_", "").replace("g_home", "")
            val = _get_stat(team_stats, f"avg_pts_{parts}g", default_pts)
        elif col.startswith("ema_pts_") and col.endswith("_away"):
            parts = col.replace("ema_pts_", "").replace("g_away", "")
            val = _get_stat(team_stats_away, f"avg_pts_{parts}g", default_pts)
        elif col == "win_rate_10g_home":
            val = _get_stat(team_stats, "win_rate_10g", 0.5)
        elif col == "win_rate_10g_away":
            val = _get_stat(team_stats_away, "win_rate_10g", 0.5)
        elif col == "last_3_margin_home":
            val = _get_stat(team_stats, "last_3_margin", default_margin)
        elif col == "last_3_margin_away":
            val = _get_stat(team_stats_away, "last_3_margin", -default_margin)
        elif col == "win_streak_home":
            val = _get_stat(team_stats, "win_streak", 0)
        elif col == "win_streak_away":
            val = _get_stat(team_stats_away, "win_streak", 0)
        elif col.endswith("_diff"):
            # Differential: home - away on same-metric
            base = col.replace("_diff", "")
            home_col = [c for c in feature_cols if c.startswith(base) and c.endswith("_home")]
            away_col = [c for c in feature_cols if c.startswith(base) and c.endswith("_away")]
            if home_col and away_col:
                h_val = feature_dict.get(home_col[0], 0.0)
                a_val = feature_dict.get(away_col[0], 0.0)
                val = h_val - a_val
            else:
                val = 0.0
        elif col == "pace_estimate":
            val = 100.0
        elif col == "pace_deviation":
            val = 0.0
        elif col == "momentum_diff":
            h_mom = feature_dict.get("momentum_composite_home", 50.0)
            a_mom = feature_dict.get("momentum_composite_away", 50.0)
            val = h_mom - a_mom
        elif col.startswith("momentum_composite"):
            # Composite = win_rate*100 + margin*2
            if col.endswith("_home"):
                wr = _get_stat(team_stats, "win_rate_10g", 0.5)
                mg = _get_stat(team_stats, "avg_margin_10g", default_margin)
            else:
                wr = _get_stat(team_stats_away, "win_rate_10g", 0.5)
                mg = _get_stat(team_stats_away, "avg_margin_10g", -default_margin)
            val = wr * 100 + mg * 2
        elif col.startswith("pts_pace_interact"):
            # Interaction: pace * point differential
            val = 0.0
        elif col == "win_rate_diff":
            h_wr = _get_stat(team_stats, "win_rate_10g", 0.5)
            a_wr = _get_stat(team_stats_away, "win_rate_10g", 0.5)
            val = h_wr - a_wr
        elif col == "margin_diff_3g":
            h_m3 = _get_stat(team_stats, "last_3_margin", default_margin)
            a_m3 = _get_stat(team_stats_away, "last_3_margin", -default_margin)
            val = h_m3 - a_m3
        else:
            val = 0.0

        feature_dict[col] = float(val) if pd.notna(val) else 0.0

    result = pd.Series(feature_dict)
    return result if not result.isnull().any() else None


def get_upcoming_league_games(league: BasketballLeague) -> Optional[pd.DataFrame]:
    """Fetch upcoming games for a league. Tries ESPN API, falls back to synthetic."""
    if league.has_espn_api:
        try:
            source = ESPNLeagueSource()
            upcoming = source.load_upcoming(league.key, limit=30)
            if upcoming is not None and not upcoming.empty:
                upcoming["league"] = league.key
                if "market_total" not in upcoming.columns:
                    upcoming["market_total"] = league.avg_total
                if "home_ml_odds" not in upcoming.columns:
                    upcoming["home_ml_odds"] = -110
                if "away_ml_odds" not in upcoming.columns:
                    upcoming["away_ml_odds"] = -110
                return upcoming
        except Exception as e:
            logger.debug(f"ESPN upcoming for {league.key} failed: {e}")

    return None
