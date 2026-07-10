"""
GamePredictor — ML model building and inference for the LivePredictionEngine.

Two prediction tiers:
  1. MarketInefficiencySystem (moneyline/home_win prediction)
  2. TotalsRegressor (over/under total points prediction)

Both share the same feature pipeline (FeatureEngineer + NBADataLoader).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from betting_intel.live.models import LiveGame, MIN_EDGE_THRESHOLD

logger = logging.getLogger(__name__)


class GamePredictor:
    """
    Builds and runs ML predictions for NBA games.

    Holds model state (robust system + totals regressor) but no engine-level
    state (locks, caches). The engine manages thread-safety externally.
    """

    def __init__(
        self,
        kelly_staker: Any,
        market_odds_store: Any,
        model_dir: Optional[Path] = None,
    ):
        self._kelly_staker = kelly_staker
        self._market_odds_store = market_odds_store
        self._model_dir = model_dir

        # Robust system (MarketInefficiencySystem)
        self._robust_system: Any = None
        self._robust_system_fitted: bool = False

        # Totals model (TotalsRegressor)
        self._totals_model: Any = None
        self._totals_fitted: bool = False
        self._totals_mae: float = 12.0
        self._totals_std: float = 15.0

        # Cached feature columns (so _predict_games can pass them)
        self._robust_feature_cols: list[str] = []
        self._totals_feature_cols: list[str] = []

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def robust_system(self) -> Any:
        return self._robust_system

    @property
    def robust_system_fitted(self) -> bool:
        return self._robust_system_fitted

    @property
    def totals_fitted(self) -> bool:
        return self._totals_fitted

    def robust_system_summary(self) -> dict:
        if self._robust_system is None:
            return {"fitted": False, "status": "not_initialized"}
        if not self._robust_system_fitted:
            return {"fitted": False, "status": "not_fitted"}
        try:
            return self._robust_system.get_summary()
        except Exception:
            return {"fitted": True, "status": "error_reading_summary"}

    def clear(self):
        self._robust_system = None
        self._robust_system_fitted = False
        self._robust_feature_cols = []
        self._totals_model = None
        self._totals_fitted = False
        self._totals_feature_cols = []

    # ── Robust System ─────────────────────────────────────────────────────

    def _build_robust_system(self) -> bool:
        """
        Build and train the MarketInefficiencySystem on historical NBA data.

        Trains the model to predict MARKET ERROR — the difference between
        actual outcomes and market-implied probabilities (ELO proxy or real odds).
        """
        try:
            from betting_intel.models.robust_ensemble import (
                MarketInefficiencySystem,
            )
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer
            from betting_intel.features.market_inefficiency import (
                compute_market_inefficiency_targets,
            )

            logger.info(
                "Building MarketInefficiencySystem on historical NBA data "
                "(v5.0 — market inefficiency training)..."
            )

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.warning("No historical NBA data available for robust system")
                return False

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                logger.warning("Feature engineering produced no data")
                return False

            # Query MarketOddsStore for real historical market data
            try:
                store_start = features_df["GAME_DATE"].min()
                store_end = features_df["GAME_DATE"].max()
                if hasattr(store_start, "strftime"):
                    store_start = store_start.strftime("%Y-%m-%d")
                    store_end = store_end.strftime("%Y-%m-%d")
                else:
                    store_start = str(store_start)[:10]
                    store_end = str(store_end)[:10]

                raw_overrides = self._market_odds_store.get_market_probs_for_date_range(
                    start_date=store_start,
                    end_date=store_end,
                )

                from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME

                market_prob_overrides: dict[tuple[str, str, str], float] = {}
                for (home_full, away_full, game_date), prob in raw_overrides.items():
                    home_short = ODDS_TO_SHORT_NAME.get(
                        home_full,
                        home_full.split()[-1] if " " in home_full else home_full,
                    )
                    away_short = ODDS_TO_SHORT_NAME.get(
                        away_full,
                        away_full.split()[-1] if " " in away_full else away_full,
                    )
                    market_prob_overrides[(home_short, away_short, game_date)] = prob

                logger.info(
                    f"Loaded {len(market_prob_overrides)} real market probs from store "
                    f"({store_start} to {store_end})"
                )
            except Exception:
                logger.debug(
                    "Failed to query MarketOddsStore — using ELO proxy only",
                    exc_info=True,
                )
                market_prob_overrides = None

            # Derive home_win target
            if "home_win" not in features_df.columns:
                if "point_diff" in features_df.columns:
                    features_df["home_win"] = (features_df["point_diff"] > 0).astype(
                        int
                    )
                elif "WL_home" in features_df.columns:
                    features_df["home_win"] = (features_df["WL_home"] == "W").astype(
                        int
                    )
                else:
                    logger.warning(
                        "Cannot derive home_win — no point_diff or WL_home column"
                    )
                    return False

            # Compute market inefficiency targets
            features_df = compute_market_inefficiency_targets(
                features_df,
                market_prob_overrides=market_prob_overrides,
            )

            # Log proxy source distribution
            if "market_proxy_source" in features_df.columns:
                source_counts = (
                    features_df["market_proxy_source"].value_counts().to_dict()
                )
                total_games = sum(source_counts.values())
                real_odds_count = source_counts.get("real_odds", 0)
                logger.info(
                    f"Market proxy source distribution: {source_counts} "
                    f"(real_odds={real_odds_count}/{total_games} = "
                    f"{real_odds_count / max(total_games, 1) * 100:.1f}%)"
                )
                if real_odds_count == 0:
                    logger.warning(
                        "ZERO games have real market odds. "
                        "Run the live engine refresh cycle several times to "
                        "accumulate odds data, or backfill via the CLI."
                    )

            # Build clean feature matrix
            clean_feature_cols = fe.select_features(features_df)
            _market_target_cols = {
                "market_implied_home_prob",
                "market_error",
                "abs_market_error",
                "market_error_clipped",
                "market_error_binary",
                "total_market_error",
                "weighted_market_error",
                "elo_error",
                "market_error_ma_5g",
                "market_error_ma_10g",
                "market_error_trend_home",
                "recent_edge_streak",
            }
            feature_cols = [
                c for c in clean_feature_cols if c not in _market_target_cols
            ]

            if len(feature_cols) < 3:
                logger.warning(
                    f"Only {len(feature_cols)} feature cols — too few for robust system"
                )
                return False

            X = features_df[feature_cols].fillna(0).values
            n_samples = len(X)
            if n_samples < 200:
                logger.warning(
                    f"Only {n_samples} samples — need at least 200 for robust system"
                )
                return False

            y_binary = features_df["home_win"].values.astype(int)
            market_probs = features_df["market_implied_home_prob"].values.astype(float)

            # v6.0 — Mutual information feature selection
            # Removes noise features with near-zero predictive power
            try:
                from sklearn.feature_selection import mutual_info_classif

                mi = mutual_info_classif(X, y_binary, random_state=42)
                top_n = min(60, len(feature_cols))
                if len(mi) > top_n:
                    top_idx = np.argsort(mi)[-top_n:]
                    feature_cols = [feature_cols[i] for i in top_idx]
                    X = X[:, top_idx]
                    logger.info(
                        f"Mutual info selection: {len(mi)} features → "
                        f"{top_n} retained (removed {len(mi) - top_n} noise cols)"
                    )
            except Exception:
                logger.debug("Mutual info feature selection skipped (non-critical)")

            logger.info(
                f"Training MarketInefficiencySystem on {n_samples} samples "
                f"with {len(feature_cols)} features..."
            )

            system = MarketInefficiencySystem(
                calibrate=True,
                n_folds=5,
                min_train_samples=50,
                random_state=42,
            )
            system.fit(
                X,
                y_binary,
                market_probs=market_probs,
                feature_names=feature_cols,
                verbose=True,
            )

            self._robust_system = system
            self._robust_system_fitted = True
            self._robust_feature_cols = feature_cols

            summary = system.get_summary()
            logger.info(
                f"MarketInefficiencySystem built: "
                f"{summary.get('n_models', '?')} classifier models, "
                f"{summary.get('n_error_models', 0)} error regressors, "
                f"Brier={summary.get('calibrated_brier', 'N/A')}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to build MarketInefficiencySystem: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def _predict_with_robust_system(self, games: list[LiveGame]) -> list[LiveGame]:
        """Predict games using the MarketInefficiencySystem with full feature pipeline."""
        if not games:
            return games

        # Guard: if kelly staker failed to init, skip stake computations
        if self._kelly_staker is None:
            logger.warning(
                "Kelly staker not available — predictions will have zero stakes"
            )

        from betting_intel.recommendations.staking import american_to_decimal
        from betting_intel.features.market_inefficiency import (
            american_to_implied_prob,
            remove_vig,
        )

        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                return games

            games_df = loader.build_game_dataset(raw_df)
            if games_df is None or games_df.empty:
                return games

            raw_df = loader.compute_rest_days(raw_df)
            if raw_df is None or raw_df.empty:
                return games

            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)
            if features_df is None or features_df.empty:
                return games

            system = self._robust_system
            if system is None:
                logger.warning("Robust system not initialized — skipping predictions")
                return games

            system_feature_cols = self._robust_feature_cols or []
            n_expected = len(system_feature_cols) if system_feature_cols else 0

            for game in games:
                try:
                    # Guard: skip if no matchup data
                    if not game.matchup:
                        continue

                    # Market-implied probability
                    market_prob = None
                    if game.home_ml is not None and game.away_ml is not None:
                        try:
                            home_implied = american_to_implied_prob(game.home_ml)
                            away_implied = american_to_implied_prob(game.away_ml)
                            market_prob, _ = remove_vig(home_implied, away_implied)
                        except Exception:
                            logger.debug(
                                f"Failed to compute market prob for {game.matchup}"
                            )
                            market_prob = None

                    # Build feature vector
                    if not game.home_team_short or not game.away_team_short:
                        continue
                    feat = self._build_feature_vector(
                        game.home_team_short,
                        game.away_team_short,
                        features_df,
                        feature_cols=system_feature_cols or None,
                    )
                    if feat is None:
                        continue

                    if n_expected > 0 and len(feat) != n_expected:
                        logger.warning(
                            f"Feature count mismatch for {game.matchup}: "
                            f"got {len(feat)}, expected {n_expected}. Skipping."
                        )
                        continue

                    X_pred = feat.values.reshape(1, -1)
                    result = system.predict_with_details(
                        X_pred, market_prob=market_prob
                    )

                    if result is None:
                        continue

                    # Guard: check calibration_failed attr exists
                    if getattr(result, "calibration_failed", False) and getattr(
                        result, "calibration_warning", None
                    ):
                        logger.warning(
                            f"Calibration warning for {game.matchup}: {result.calibration_warning}"
                        )

                    home_win_prob = getattr(result, "home_win_prob", 0.5)

                    # Feature importance
                    feature_importance = getattr(result, "feature_importance", None)
                    if feature_importance:
                        top_features = dict(
                            sorted(
                                feature_importance.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:8]
                        )
                        game.feature_importance = top_features

                    # Apply predictions
                    if market_prob is not None:
                        predicted_error = (
                            result.edge_pct
                            if getattr(result, "edge_pct", None) is not None
                            else 0.0
                        )

                        if abs(predicted_error) < MIN_EDGE_THRESHOLD:
                            game.edge_pct = 0.0
                            game.direction = "neutral"
                            game.confidence = "low"
                            game.stake_dollars = 0.0
                        else:
                            game.edge_pct = predicted_error
                            game.direction = "home" if predicted_error > 0 else "away"
                            game.confidence = (
                                getattr(result, "confidence_label", None) or "low"
                            ).lower()

                        if predicted_error >= 0:
                            decimal_odds = (
                                american_to_decimal(game.home_ml)
                                if game.home_ml
                                else 2.0
                            )
                            team_for_kelly = game.home_team_short
                            win_prob_for_kelly = home_win_prob
                        else:
                            decimal_odds = (
                                american_to_decimal(game.away_ml)
                                if game.away_ml
                                else 2.0
                            )
                            team_for_kelly = game.away_team_short
                            win_prob_for_kelly = 1.0 - home_win_prob

                        stake_result = self._kelly_staker.compute_stake(
                            win_probability=max(win_prob_for_kelly, 0.01),
                            decimal_odds=decimal_odds,
                            confidence_score=getattr(result, "confidence_score", 0.5),
                            confidence_label=getattr(result, "confidence_label", "low"),
                            edge_pct=abs(predicted_error),
                            league=game.league,
                            team=team_for_kelly,
                            game_id=game.game_id,
                        )
                        game.stake_dollars = (
                            stake_result.stake_dollars if stake_result else 0.0
                        )
                    else:
                        game.edge_pct = 0.0
                        game.direction = "neutral"
                        game.confidence = "low"

                    game.predicted_total = round(home_win_prob, 3)
                    game.predicted_at = datetime.now().isoformat()

                except Exception as e:
                    logger.debug(f"Robust prediction failed for {game.matchup}: {e}")
                    continue

            return games

        except Exception as e:
            logger.warning(f"Robust prediction pipeline failed: {e}")
            return games

    # ── Totals Model ──────────────────────────────────────────────────────

    def _build_totals_model(self) -> bool:
        """Build and train a totals regression model on historical NBA data."""
        try:
            from betting_intel.live.totals_model import TotalsRegressor
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            logger.info("Building TotalsRegressor on historical NBA data...")

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.warning("No historical NBA data for totals model")
                return False

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                logger.warning("Feature engineering produced no data for totals model")
                return False

            clean_feature_cols = fe.select_features(features_df)
            _exclude_totals = {
                "total_points",
                "point_diff",
                "market_implied_home_prob",
                "market_error",
                "abs_market_error",
                "market_error_clipped",
                "market_error_binary",
                "total_market_error",
                "weighted_market_error",
                "elo_error",
                "market_error_ma_5g",
                "market_error_ma_10g",
                "market_error_trend_home",
                "recent_edge_streak",
            }
            feature_cols = [c for c in clean_feature_cols if c not in _exclude_totals]

            if len(feature_cols) < 3:
                logger.warning(
                    f"Only {len(feature_cols)} feature cols for totals model"
                )
                return False

            X = features_df[feature_cols].fillna(0).values
            n_samples = len(X)
            if n_samples < 200:
                logger.warning(f"Only {n_samples} samples for totals model")
                return False

            if "total_points" not in features_df.columns:
                logger.warning("`total_points` column not found for totals model")
                return False

            y_total = features_df["total_points"].values.astype(float)

            # v6.0 — Mutual information feature selection for regression
            # Removes noise features with near-zero predictive power for totals
            try:
                from sklearn.feature_selection import mutual_info_regression

                mi = mutual_info_regression(X, y_total, random_state=42)
                top_n = min(60, len(feature_cols))
                if len(mi) > top_n:
                    top_idx = np.argsort(mi)[-top_n:]
                    feature_cols = [feature_cols[i] for i in top_idx]
                    X = X[:, top_idx]
                    logger.info(
                        f"Mutual info selection (totals): {len(mi)} features → "
                        f"{top_n} retained (removed {len(mi) - top_n} noise cols)"
                    )
            except Exception:
                logger.debug(
                    "Mutual info feature selection for totals skipped (non-critical)"
                )

            logger.info(
                f"Training TotalsRegressor on {n_samples} samples with {len(feature_cols)} features..."
            )

            regressor = TotalsRegressor(random_state=42)
            regressor.fit(X, y_total, feature_names=feature_cols, verbose=True)

            self._totals_model = regressor
            self._totals_fitted = True
            self._totals_mae = regressor.mae or 12.0
            self._totals_feature_cols = feature_cols

            logger.info(
                f"TotalsRegressor built: {len(regressor._models)} models, "
                f"MAE={regressor.mae:.1f}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to build totals model: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def _predict_totals(self, games: list[LiveGame]) -> list[LiveGame]:
        """Predict total points for each game using the totals regression model."""
        # Safety: don't attempt prediction if the model was never built
        if self._totals_model is None:
            return games
        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                return games

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)
            if features_df is None or features_df.empty:
                return games

            nba_games = [
                g
                for g in games
                if g.sport_group == "Basketball"
                and g.market_total
                and g.market_total > 0
            ]
            if not nba_games:
                return games

            system_feature_cols = self._totals_feature_cols or []
            predicted_count = 0

            for game in nba_games:
                try:
                    feat = self._build_feature_vector(
                        game.home_team_short,
                        game.away_team_short,
                        features_df,
                        feature_cols=system_feature_cols or None,
                    )
                    if feat is None:
                        continue

                    X_pred = feat.values.reshape(1, -1)
                    result = self._totals_model.predict_single(
                        X_pred,
                        market_total=game.market_total,
                    )

                    game.total_prediction = result.predicted_total
                    game.total_edge_pct = result.edge_pct
                    game.total_direction = result.direction
                    game.total_confidence = result.confidence
                    predicted_count += 1

                except Exception as e:
                    logger.debug(f"Totals prediction failed for {game.matchup}: {e}")
                    continue

            if predicted_count > 0:
                logger.info(f"Totals model: predicted {predicted_count} NBA games")

            return games

        except Exception as e:
            logger.warning(f"Totals prediction pipeline failed: {e}")
            return games

    # ── Feature Vector Builder ────────────────────────────────────────────

    def _build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        features_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> Optional[pd.Series]:
        """
        Build a full feature vector for an upcoming game using the FeatureEngineer pipeline.

        v6.0 — MULTI-GAME WEIGHTED AVERAGE (replaces single-most-recent-row).

        Uses the last N=3 games for each team with recency weighting [0.5, 0.3, 0.2]
        to reduce noise from outlier games while remaining responsive to recent form.

        For *_home columns: weighted average of home team's last 3 HOME games.
        For *_away columns: weighted average of away team's last 3 AWAY games.
        For *_diff columns: home_weighted_avg - away_weighted_avg.
        For global features (rest, travel, fatigue): uses most recent direct matchup.
        For H2H features: uses most recent direct matchup.
        """
        import pandas as pd

        if features_df is None or features_df.empty:
            return None

        home_col = (
            "TEAM_NAME_home" if "TEAM_NAME_home" in features_df.columns else "home_team"
        )
        away_col = (
            "TEAM_NAME_away" if "TEAM_NAME_away" in features_df.columns else "away_team"
        )

        if feature_cols is not None:
            cols = feature_cols
        else:
            from betting_intel.data.features import FeatureEngineer

            _fe = FeatureEngineer()
            cols = _fe.select_features(features_df)
            cols = [
                c
                for c in cols
                if c
                not in {
                    "total_points",
                    "point_diff",
                    "home_score",
                    "away_score",
                    "spread",
                    "label",
                    "home_win",
                    "market_implied_home_prob",
                    "market_error",
                    "abs_market_error",
                    "market_error_clipped",
                    "market_error_binary",
                    "total_market_error",
                    "weighted_market_error",
                    "elo_error",
                    "market_error_ma_5g",
                    "market_error_ma_10g",
                    "market_error_trend_home",
                    "recent_edge_streak",
                }
            ]

        def _norm(name: str) -> str:
            return str(name).strip().lower()

        target_home_norm = _norm(home_team)
        target_away_norm = _norm(away_team)
        home_norm = features_df[home_col].astype(str).str.strip().str.lower()
        away_norm = features_df[away_col].astype(str).str.strip().str.lower()

        # ── Multi-game indices (last 3 games per team) ────────────────
        # Recency weights: most recent game gets 50%, second 30%, third 20%
        _RECENCY_WEIGHTS = np.array([0.50, 0.30, 0.20])
        _N_GAMES = 3

        home_mask = home_norm == target_home_norm
        home_indices = (
            features_df.index[home_mask][-_N_GAMES:]
            if home_mask.sum() >= 1
            else pd.Index([])
        )
        away_mask = away_norm == target_away_norm
        away_indices = (
            features_df.index[away_mask][-_N_GAMES:]
            if away_mask.sum() >= 1
            else pd.Index([])
        )

        # Most recent direct matchup (for H2H, rest, travel, fatigue)
        direct_mask = (
            (home_norm == target_home_norm) & (away_norm == target_away_norm)
        ) | ((home_norm == target_away_norm) & (away_norm == target_home_norm))
        direct_idx = features_df.index[direct_mask][-1] if direct_mask.any() else None
        direct_row = features_df.loc[direct_idx] if direct_idx is not None else None

        # ── Helper: compute recency-weighted average for a set of columns ──
        def _weighted_avg(indices: pd.Index, col_suffix: str) -> dict[str, float]:
            """Compute recency-weighted average of *_<col_suffix> columns.

            CRITICAL: indices are in CHRONOLOGICAL order (oldest first), but the
            recency weights [0.5, 0.3, 0.2] are in REVERSE-chronological order
            (most recent = highest weight). We reverse the values so the most
            recent game gets weight 0.5.
            """
            result: dict[str, float] = {}
            n = len(indices)
            if n == 0:
                return result
            w = _RECENCY_WEIGHTS[:_N_GAMES] if n >= _N_GAMES else _RECENCY_WEIGHTS[:n]
            w = w / w.sum()  # renormalize
            for c in cols:
                if not c.endswith(f"_{col_suffix}"):
                    continue
                if c not in features_df.columns:
                    continue
                # Reverse values: indices are chronological (oldest first),
                # weights are recency-order (most recent first)
                vals = features_df.loc[indices, c].values[::-1]
                vals_float = pd.to_numeric(vals, errors="coerce")
                vals_float = np.nan_to_num(vals_float, nan=0.0)
                result[c] = float(np.average(vals_float, weights=w[: len(vals_float)]))
            return result

        # Weighted averages for home (*_home) and away (*_away) columns
        home_team_features = _weighted_avg(home_indices, "home")
        away_team_features = _weighted_avg(away_indices, "away")

        # ── Assemble feature dict ────────────────────────────────────────
        feature_dict: dict[str, float] = {}
        _rest_cols = {
            "rest_home_days",
            "rest_advantage",
            "is_b2b_home",
            "fatigue_home",
            "rest_3in4_home",
            "both_b2b",
            "fatigue_diff",
            "rest_home_sq",
            "rest_adv_sq",
            "fatigue_rest_interact",
        }
        _away_rest_cols = {
            "rest_away_days",
            "is_b2b_away",
            "fatigue_away",
            "rest_3in4_away",
            "rest_away_sq",
        }
        _travel_cols = {
            "travel_distance",
            "travel_distance_norm",
            "tz_diff",
            "cum_travel_diff",
        }
        _h2h_cols = {"h2h_win_rate", "h2h_avg_margin"}

        for col in cols:
            val: Optional[float] = None

            if col.endswith("_home"):
                val = home_team_features.get(col)
            elif col.endswith("_away"):
                val = away_team_features.get(col)
            elif col.endswith("_diff"):
                base = col[:-5]
                home_base = home_team_features.get(f"{base}_home")
                away_base = away_team_features.get(f"{base}_away")
                if home_base is not None and away_base is not None:
                    val = home_base - away_base
                elif direct_row is not None and col in direct_row.index:
                    val = float(direct_row[col]) if pd.notna(direct_row[col]) else 0.0
            elif (
                col.startswith("TEAM_")
                or "rest_key" in col
                or col in ("home_team_name", "away_team_name")
            ):
                val = 0.0
            elif col in _rest_cols:
                if direct_row is not None and col in direct_row.index:
                    val = float(direct_row[col]) if pd.notna(direct_row[col]) else 0.0
                elif len(home_indices) > 0 and col in features_df.columns:
                    # Use most recent game for game-specific features
                    val = (
                        float(features_df.loc[home_indices[-1], col])
                        if pd.notna(features_df.loc[home_indices[-1], col])
                        else 0.0
                    )
            elif col in _away_rest_cols:
                if direct_row is not None and col in direct_row.index:
                    val = float(direct_row[col]) if pd.notna(direct_row[col]) else 0.0
                elif len(away_indices) > 0 and col in features_df.columns:
                    val = (
                        float(features_df.loc[away_indices[-1], col])
                        if pd.notna(features_df.loc[away_indices[-1], col])
                        else 0.0
                    )
            elif col in _travel_cols:
                if direct_row is not None and col in direct_row.index:
                    val = float(direct_row[col]) if pd.notna(direct_row[col]) else 0.0
                elif len(home_indices) > 0 and col in features_df.columns:
                    val = (
                        float(features_df.loc[home_indices[-1], col])
                        if pd.notna(features_df.loc[home_indices[-1], col])
                        else 0.0
                    )
            elif col in _h2h_cols:
                if direct_row is not None and col in direct_row.index:
                    val = float(direct_row[col]) if pd.notna(direct_row[col]) else 0.0
            else:
                if direct_row is not None and col in direct_row.index:
                    val = float(direct_row[col]) if pd.notna(direct_row[col]) else None
                if val is None and len(home_indices) > 0 and col in features_df.columns:
                    # Use most recent home game as fallback
                    val = (
                        float(features_df.loc[home_indices[-1], col])
                        if pd.notna(features_df.loc[home_indices[-1], col])
                        else None
                    )
                if val is None and col in features_df.columns:
                    val = (
                        float(features_df[col].iloc[-1])
                        if pd.notna(features_df[col].iloc[-1])
                        else 0.0
                    )

            feature_dict[col] = val if val is not None else 0.0

        result = pd.Series(feature_dict)
        if result.isnull().any():
            result = result.fillna(0.0)
        return result
