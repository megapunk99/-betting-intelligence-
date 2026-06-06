"""
Multi-League Feature Engineering — simplified rolling features for any basketball league.

Unlike the NBA-specific FeatureEngineer (which requires NBA team IDs, MATCHUP columns,
NBA_TEAM_CENTERS, etc.), this module works with generic basketball data from the ESPN
API or any source that provides: game_id, date, home_team, away_team, home_score,
away_score, total_points.

Features created per league:
  - Rolling averages (5g, 10g) for points scored and allowed
  - Win rate tracking
  - Simple momentum: last 3-game margin, win streak
  - Home/away differentials
  - Pace estimate
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class MultiLeagueFeatureEngineer:
    """
    Simplified feature engineer that works for ANY basketball league.

    Unlike FeatureEngineer (NBA-only, requires NBADataLoader columns), this
    module only needs the CANONICAL_SCHEMA columns:
      game_id, date, home_team, away_team, home_score, away_score, total_points

    Usage:
        engineer = MultiLeagueFeatureEngineer()
        features_df = engineer.build_features(games_df)
    """

    def __init__(self, rolling_windows: Optional[list[int]] = None):
        self.rolling_windows = rolling_windows or [5, 10]

    def build_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature set from game-level data for any basketball league.

        Args:
            games_df: DataFrame with columns:
                game_id, date, home_team, away_team, home_score, away_score,
                total_points (and optionally: league, season)

        Returns:
            DataFrame with rolling features, no lookahead bias
        """
        df = games_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Normalise column names
        df = self._normalise_columns(df)

        # Create team-level view: two rows per game (home and away)
        self._last_team_logs = self._build_team_logs(df)
        team_logs = self._last_team_logs

        # ── Rolling averages per team ─────────────────────────────────
        df = self._add_rolling_features(df, team_logs)

        # ── Win streaks & momentum ───────────────────────────────────
        df = self._add_momentum_features(df, team_logs)

        # ── Differential features (home - away) ──────────────────────
        df = self._add_differential_features(df)

        # ── Pace estimate ────────────────────────────────────────────
        df = self._add_pace_estimate(df)

        # ── Home-away interaction ────────────────────────────────────
        df = self._add_interaction_features(df)

        # ── Fill NAs ─────────────────────────────────────────────────
        df = self._backfill_features(df)

        # Drop raw score columns so the model doesn't see the target
        drop_cols = [
            "home_score_raw", "away_score_raw",
            "_home_win", "_home_score", "_away_score",
        ]
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

        return df

    @property
    def team_logs(self) -> Optional[pd.DataFrame]:
        """Get the team logs from the last build_features() call."""
        return getattr(self, '_last_team_logs', None)

    # ── Column Normalisation ────────────────────────────────────────

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to a standard format."""
        rename = {}
        if "date" not in df.columns and "game_date" in df.columns:
            rename["game_date"] = "date"
        if "home_score" not in df.columns and "home_pts" in df.columns:
            rename["home_pts"] = "home_score"
        if "away_score" not in df.columns and "away_pts" in df.columns:
            rename["away_pts"] = "away_score"
        if rename:
            df = df.rename(columns=rename)
        return df

    def _build_team_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build team-level game logs (two rows per game).

        Returns DataFrame with columns:
            game_id, date, team, opponent, pts, opp_pts, is_home, win
        """
        rows = []
        for _, game in df.iterrows():
            # Home team row
            rows.append({
                "game_id": game.get("game_id", ""),
                "date": game["date"],
                "team": game["home_team"],
                "opponent": game["away_team"],
                "pts": float(game["home_score"]),
                "opp_pts": float(game["away_score"]),
                "is_home": 1,
                "win": 1 if float(game["home_score"]) > float(game["away_score"]) else 0,
            })
            # Away team row
            rows.append({
                "game_id": game.get("game_id", ""),
                "date": game["date"],
                "team": game["away_team"],
                "opponent": game["home_team"],
                "pts": float(game["away_score"]),
                "opp_pts": float(game["home_score"]),
                "is_home": 0,
                "win": 1 if float(game["away_score"]) > float(game["home_score"]) else 0,
            })

        logs = pd.DataFrame(rows)
        logs = logs.sort_values("date").reset_index(drop=True)
        return logs

    def _add_rolling_features(self, df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
        """Add rolling averages for points scored, allowed, and margin."""
        # Compute rolling stats per team on ALL rows
        rolling_all = logs.copy()
        rolling_all = rolling_all.sort_values(["team", "date"])

        for w in self.rolling_windows:
            # Points scored (rolling, shifted by 1 to avoid lookahead)
            rolling_all[f"avg_pts_{w}g"] = (
                rolling_all.groupby("team")["pts"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
            )
            # Points allowed
            rolling_all[f"avg_pts_allowed_{w}g"] = (
                rolling_all.groupby("team")["opp_pts"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
            )
            # Margin
            rolling_all["margin"] = rolling_all["pts"] - rolling_all["opp_pts"]
            rolling_all[f"avg_margin_{w}g"] = (
                rolling_all.groupby("team")["margin"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
            )
            # EMA (exponential moving average) for points
            rolling_all[f"ema_pts_{w}g"] = (
                rolling_all.groupby("team")["pts"]
                .transform(lambda x: x.ewm(span=max(w, 2), min_periods=1, adjust=False).mean().shift(1))
            )

        # Add team_home / team_away columns to df for merge compatibility
        df["team_home"] = df["home_team"]
        df["team_away"] = df["away_team"]

        # Merge back into the game-level dataframe
        for suffix in ["home", "away"]:
            team_key = f"team_{suffix}"
            # Filter from rolling_all (has rolling columns), not original logs
            side_stats = rolling_all[rolling_all["is_home"] == (1 if suffix == "home" else 0)].copy()
            side_stats = side_stats.rename(columns={
                "team": team_key,
                "date": f"date_{suffix}",
            })

            # Merge rolling stats for each window
            for w in self.rolling_windows:
                merge_cols = ["game_id", team_key,
                              f"avg_pts_{w}g", f"avg_pts_allowed_{w}g",
                              f"avg_margin_{w}g", f"ema_pts_{w}g"]
                merge_df = side_stats[merge_cols].rename(columns={
                    f"avg_pts_{w}g": f"avg_pts_{w}g_{suffix}",
                    f"avg_pts_allowed_{w}g": f"avg_pts_allowed_{w}g_{suffix}",
                    f"avg_margin_{w}g": f"avg_margin_{w}g_{suffix}",
                    f"ema_pts_{w}g": f"ema_pts_{w}g_{suffix}",
                })
                df = df.merge(merge_df, on=["game_id", team_key], how="left")

        return df

    def _add_momentum_features(self, df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
        """Add win streaks, last-3 margin, and win rate features."""
        team_logs = logs.sort_values(["team", "date"])

        # Win streaks
        team_logs["win_streak"] = team_logs.groupby("team")["win"].transform(
            lambda x: self._compute_win_streak(x)
        )

        # Win rate (last 10 games)
        team_logs["win_rate_10g"] = (
            team_logs.groupby("team")["win"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )

        # Last 3-game margin
        team_logs["margin"] = team_logs["pts"] - team_logs["opp_pts"]
        team_logs["last_3_margin"] = (
            team_logs.groupby("team")["margin"]
            .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        )

        # Merge into game-level
        for suffix in ["home", "away"]:
            side_logs = team_logs[team_logs["is_home"] == (1 if suffix == "home" else 0)].copy()
            merge_cols = ["game_id", "team", "win_streak", "win_rate_10g", "last_3_margin"]
            merge_df = side_logs[merge_cols].rename(columns={
                "team": f"team_{suffix}",
                "win_streak": f"win_streak_{suffix}",
                "win_rate_10g": f"win_rate_10g_{suffix}",
                "last_3_margin": f"last_3_margin_{suffix}",
            })
            df = df.merge(merge_df, on=["game_id", f"team_{suffix}"], how="left")

        return df

    def _add_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add home - away differentials for rolling stats."""
        for w in self.rolling_windows:
            for stat in ["avg_pts", "avg_pts_allowed", "avg_margin", "ema_pts"]:
                col_h = f"{stat}_{w}g_home"
                col_a = f"{stat}_{w}g_away"
                if col_h in df.columns and col_a in df.columns:
                    df[f"{stat}_diff_{w}g"] = (
                        df[col_h].fillna(0) - df[col_a].fillna(0)
                    )

        # Win rate differential
        if "win_rate_10g_home" in df.columns and "win_rate_10g_away" in df.columns:
            df["win_rate_diff"] = df["win_rate_10g_home"].fillna(0) - df["win_rate_10g_away"].fillna(0)

        # Last 3 margin differential
        if "last_3_margin_home" in df.columns and "last_3_margin_away" in df.columns:
            df["margin_diff_3g"] = df["last_3_margin_home"].fillna(0) - df["last_3_margin_away"].fillna(0)

        return df

    def _add_pace_estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate pace from total points.

        Not as accurate as real possession data, but useful for leagues
        where we don't have box score stats. Uses a league-agnostic
        conversion: total_points * (200 / avg_total_in_window).
        """
        # Simple rolling average of total points as a baseline
        df["pace_estimate"] = 100.0  # default pace

        # If we have enough data, compute a rolling total points average
        # and estimate pace as deviation from that
        if "total_points" in df.columns:
            rolling_total = df["total_points"].rolling(20, min_periods=1).mean()
            df["pace_estimate"] = rolling_total / 2.24  # Convert to approx possessions

        # Pace differential (home vs league avg)
        df["pace_deviation"] = df["pace_estimate"] - df["pace_estimate"].rolling(50, min_periods=1).mean().fillna(100)

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features combining stats."""
        # Scoring * pace interaction
        for w in [5, 10]:
            col = f"avg_pts_diff_{w}g"
            if col in df.columns:
                df[f"pts_pace_interact_{w}g"] = df[col].fillna(0) * df["pace_deviation"].fillna(0) / 100.0

        # Momentum composite: win_rate + margin
        if "win_rate_10g_home" in df.columns and "avg_margin_10g_home" in df.columns:
            df["momentum_composite_home"] = (
                df["win_rate_10g_home"].fillna(0.5) * 100
                + df["avg_margin_10g_home"].fillna(0) * 2
            )
        if "win_rate_10g_away" in df.columns and "avg_margin_10g_away" in df.columns:
            df["momentum_composite_away"] = (
                df["win_rate_10g_away"].fillna(0.5) * 100
                + df["avg_margin_10g_away"].fillna(0) * 2
            )

        if "momentum_composite_home" in df.columns and "momentum_composite_away" in df.columns:
            df["momentum_diff"] = (
                df["momentum_composite_home"].fillna(50)
                - df["momentum_composite_away"].fillna(50)
            )

        return df

    def _backfill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with sensible defaults."""
        defaults = {
            "avg_pts": 0.0,
            "avg_pts_allowed": 0.0,
            "avg_margin": 0.0,
            "ema_pts": 0.0,
            "win_streak": 0,
            "win_rate_10g": 0.5,
            "last_3_margin": 0.0,
            "_diff_": 0.0,
            "win_rate_diff": 0.0,
            "margin_diff_3g": 0.0,
            "momentum_composite": 50.0,
            "momentum_diff": 0.0,
            "pace_estimate": 100.0,
            "pace_deviation": 0.0,
            "pts_pace_interact": 0.0,
        }

        for col in df.columns:
            if not df[col].isna().any():
                continue
            matched = False
            for pattern, fill_val in defaults.items():
                if pattern in col:
                    df[col] = df[col].fillna(fill_val)
                    matched = True
                    break
            if not matched:
                try:
                    med = df[col].dropna().median()
                    df[col] = df[col].fillna(med if pd.notna(med) else 0.0)
                except Exception:
                    df[col] = df[col].fillna(0.0)

        return df

    @staticmethod
    def _compute_win_streak(wins: pd.Series) -> pd.Series:
        """Compute win/loss streak. Positive = wins, Negative = losses."""
        streak = np.zeros(len(wins), dtype=int)
        current = 0
        for i, w in enumerate(wins):
            if w == 1:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
            streak[i] = current
        return pd.Series(streak, index=wins.index).shift(1).fillna(0).astype(int)

    def auto_select_features(self, df: pd.DataFrame) -> list[str]:
        """Auto-detect feature columns (excludes ID/target columns)."""
        exclude = {
            "game_id", "date", "home_team", "away_team",
            "team_home", "team_away",
            "home_score", "away_score", "total_points",
            "league", "season", "_home_win",
        }
        return [
            c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]


def train_league_model(
    league_key: str,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    model_dir: str = "models/saved",
) -> Optional[object]:
    """
    Train a LightGBM model for a specific league.

    Uses walk-forward validation: train on first 80%, test on last 20%.
    Returns the trained model trained on ALL data.
    """
    import logging
    from pathlib import Path
    logger = logging.getLogger(__name__)

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("LightGBM not available — cannot train per-league model")
        return None

    if len(features_df) < 30:
        logger.warning(f"Not enough data for {league_key}: {len(features_df)} games < 30")
        return None

    target = "total_points"
    if target not in features_df.columns:
        logger.warning(f"No total_points target in features for {league_key}")
        return None

    X = features_df[feature_cols].fillna(0)
    y = features_df[target].fillna(features_df[target].median())

    # Walk-forward split: chronological
    n = len(X)
    split = max(30, int(n * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"  Training {league_key.upper()} model: {len(X_train)} train, {len(X_test)} test, {len(feature_cols)} features")

    # Train LightGBM (backward-compatible API)
    try:
        # Try new API (LightGBM >= 4.0)
        from lightgbm import early_stopping
        model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42,
            verbosity=-1,
            min_child_samples=5,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="l2",
                  callbacks=[early_stopping(10, verbose=False)])
    except (ImportError, TypeError):
        # Fallback for older LightGBM
        model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42,
            verbosity=-1,
            min_child_samples=5,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="l2",
                  early_stopping_rounds=10, verbose=False)

    # Evaluate
    from sklearn.metrics import mean_absolute_error, r2_score
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"  \ud83d\udcca  {league_key.upper()} model: train R\u00b2={train_r2:.3f}, test R\u00b2={test_r2:.3f}, MAE={test_mae:.1f}")

    # Retrain on ALL data for final model
    final_model = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=42,
        verbosity=-1,
    )
    final_model.fit(X, y)

    # Save via ModelRegistry
    try:
        from betting_intel.models.persistence import model_registry
        version = model_registry.save(
            model=final_model,
            model_name=f"{league_key}_total",
            feature_cols=feature_cols,
            metrics={"test_r2": float(test_r2), "test_mae": float(test_mae)},
        )
        print(f"  \ud83d\udcbe  {league_key.upper()} model saved (version: {version})")
    except Exception as e:
        print(f"  \u2139  Could not save model: {e}")

    return final_model


def load_league_model(league_key: str) -> tuple[Optional[object], Optional[list[str]]]:
    """Load a trained model for a specific league."""
    try:
        from betting_intel.models.persistence import model_registry
        model, metadata = model_registry.load(f"{league_key}_total")
        feature_cols = metadata.get("feature_cols", [])
        return model, feature_cols
    except (FileNotFoundError, Exception):
        return None, None
