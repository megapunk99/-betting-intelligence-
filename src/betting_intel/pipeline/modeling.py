"""
Modeling mixin — feature engineering, model training, tuning, and tomorrow predictions.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from betting_intel.pipeline.bootstrap import (
    PROJECT_ROOT, logger, FeatureEngineer, TotalPointsPredictor,
    SpreadPredictor, StackingEnsemblePredictor,
)

HAS_MLP = False
try:
    from betting_intel.models.mlp_predictor import (
        MLPPredictor, EnhancedEnsemble, SpreadPredictorWithUncertainty,
    )
    HAS_MLP = True
except ImportError:
    pass


# ── Feature Selection ───────────────────────────────────────────────────


def _select_top_features(
    X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
    n_select: int = 50,
) -> List[str]:
    """Select top N features using mutual information.

    Mutual information captures non-linear relationships and is robust
    to outliers. Unlike LASSO (which assumes linearity), MI measures
    how much knowing X reduces uncertainty about y.

    Args:
        X: (n_samples, n_features) training features
        y: (n_samples,) target values
        feature_names: List of feature column names
        n_select: Number of top features to keep

    Returns:
        List of selected feature names (subset of feature_names)
    """
    if X.shape[1] <= n_select:
        return feature_names

    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        top_indices = np.argsort(mi)[-n_select:][::-1]
        selected = [feature_names[i] for i in top_indices]
        # Log which features were kept
        top5 = selected[:5]
        print(f"  🎯  Feature selection: {len(feature_names)} → {n_select} "
              f"(top 5: {', '.join(top5)}...)")
        return selected
    except ImportError:
        print("  ℹ  sklearn.feature_selection unavailable, using all features")
        return feature_names
    except Exception as e:
        print(f"  ℹ  Feature selection failed ({e}), using all features")
        return feature_names


def _detect_overfitting(fold_metrics: List[Dict]) -> Dict[str, Any]:
    """Analyze per-fold metrics to detect overfitting.

    Looks for:
      - Large train/test R² gap (train >> test = overfitting)
      - Negative test R² (model is worse than mean prediction)
      - High variance in test metrics across folds
      - Train R² suspiciously close to 1.0

    Returns:
        Dict with overfitting flags and diagnostics
    """
    if not fold_metrics:
        return {"overfit": False, "reason": "no_folds"}

    train_r2s = [f.get("train_r2", 0) for f in fold_metrics]
    test_r2s = [f.get("test_r2", 0) for f in fold_metrics]
    train_maes = [f.get("train_mae", 0) for f in fold_metrics]
    test_maes = [f.get("test_mae", 0) for f in fold_metrics]

    avg_train_r2 = float(np.mean(train_r2s))
    avg_test_r2 = float(np.mean(test_r2s))
    r2_gap = avg_train_r2 - avg_test_r2

    flags = []
    if avg_train_r2 > 0.9 and r2_gap > 0.3:
        flags.append("HIGH_GAP")
    if avg_test_r2 < 0.0:
        flags.append("NEGATIVE_TEST_R2")
    if avg_train_r2 > 0.95:
        flags.append("SUSPICIOUSLY_HIGH_TRAIN_R2")
    if np.std(test_r2s) > 0.3:
        flags.append("HIGH_FOLD_VARIANCE")

    is_overfit = len(flags) >= 2 or "NEGATIVE_TEST_R2" in flags

    if is_overfit or flags:
        print(f"  ⚠  Overfitting analysis:")
        print(f"       Train R²: {avg_train_r2:.3f}, Test R²: {avg_test_r2:.3f}, Gap: {r2_gap:.3f}")
        for flag in flags:
            print(f"       ⚠  Flag: {flag}")

    return {
        "overfit": is_overfit,
        "flags": flags,
        "avg_train_r2": avg_train_r2,
        "avg_test_r2": avg_test_r2,
        "r2_gap": r2_gap,
        "avg_train_mae": float(np.mean(train_maes)),
        "avg_test_mae": float(np.mean(test_maes)),
        "n_folds": len(fold_metrics),
    }


class ModelingMixin:
    """Mixin providing feature engineering and model training methods."""

    # Columns to exclude when selecting feature columns for models
    EXCLUDE_COLS = {
        "game_id", "game_date", "home_team", "away_team",
        "total_points", "spread", "label", "home_win",
        "home_score", "away_score",
    }

    # ── Feature Engineering ─────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features for model input."""
        print("\n" + "=" * 70)
        print("  🔧  STAGE 2: FEATURE ENGINEERING")
        print("=" * 70)

        try:
            engineer = FeatureEngineer()
            if hasattr(engineer, 'build_all_features'):
                features_df = self._build_features_via_pipeline(engineer, df)
            elif hasattr(engineer, 'create_features'):
                features_df = engineer.create_features(df)
            else:
                features_df = None

            if features_df is not None and not features_df.empty:
                print(f"  ✅  Engineered {len(features_df.columns)} features from {len(features_df)} rows")
                return features_df
        except Exception as e:
            print(f"  ⚠  FeatureEngineer failed: {e}")

        # Manual feature engineering fallback
        return self._manual_feature_engineering(df)

    def _build_features_via_pipeline(self, engineer, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Run the FeatureEngineer.build_all_features pipeline with proper data preparation."""
        try:
            from betting_intel.data.loader import NBADataLoader
            loader = NBADataLoader()

            if "IS_HOME" not in df.columns:
                if "TEAM_NAME_home" in df.columns:
                    raw_df = loader.load_game_logs()
                    raw_df["IS_HOME"] = raw_df["MATCHUP"].fillna("").str.contains("vs.").astype(int)
                    raw_df = loader.compute_rest_days(raw_df)
                    return engineer.build_all_features(df, raw_df)
                else:
                    df["IS_HOME"] = df["MATCHUP"].fillna("").str.contains("vs.").astype(int)
                    games_df = loader.build_game_dataset(df)
                    raw_df = loader.compute_rest_days(df)
                    result = engineer.build_all_features(games_df, raw_df)
                    df = games_df
                    self.df = games_df
                    return result
            else:
                games_df = loader.build_game_dataset(df)
                raw_df = loader.compute_rest_days(df)
                result = engineer.build_all_features(games_df, raw_df)
                df = games_df
                self.df = games_df
                return result
        except Exception as e:
            print(f"  ⚠  Full feature pipeline failed: {e}")
            return None

    def _manual_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic manual feature engineering fallback."""
        print("  ℹ  Using manual feature engineering...")
        df_feat = df.copy()

        if "game_date" in df_feat.columns:
            df_feat["game_date"] = pd.to_datetime(df_feat["game_date"])
            df_feat = df_feat.sort_values("game_date")

        for col in df_feat.select_dtypes(include=[np.number]).columns:
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

        interaction_pairs = [
            ("home_fg_pct", "away_fg_pct", "fg_pct_diff"),
            ("home_rebounds", "away_rebounds", "rebound_diff"),
            ("home_turnovers", "away_turnovers", "turnover_diff"),
            ("home_elo", "away_elo", "elo_diff"),
            ("home_pace", "away_pace", "pace_diff"),
        ]
        for c1, c2, out in interaction_pairs:
            if all(c in df_feat.columns for c in [c1, c2]):
                df_feat[out] = df_feat[c1] - df_feat[c2]

        print(f"  ✅  Engineered {len(df_feat.columns)} total columns")
        return df_feat

    # ── Training & Prediction ───────────────────────────────────────

    def train_and_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Run the multi-strategy prediction engine."""
        print("\n" + "=" * 70)
        print("  🤖  STAGE 3: MODEL TRAINING & PREDICTION")
        print("=" * 70)

        tune = not self.args.no_tune
        strategy = self.args.strategy
        model_dir = Path(self.args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        has_total = "total_points" in features_df.columns
        has_spread = "spread" in features_df.columns

        feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]
        if len(feature_cols) < 3:
            feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                           if c not in self.EXCLUDE_COLS]

        print(f"  📐  Using {len(feature_cols)} feature columns")

        # Try StackingEnsemblePredictor first
        result_df = self._try_stacking_ensemble(features_df, feature_cols, has_total)
        if result_df is not None:
            return result_df

        # Fallback to manual prediction loop
        return self._manual_train_predict(features_df, feature_cols, has_total, has_spread, model_dir, tune)

    def _try_stacking_ensemble(self, features_df: pd.DataFrame, feature_cols: list,
                                has_total: bool) -> Optional[pd.DataFrame]:
        """Use StackingEnsemble with walk-forward validation & feature selection.

        Key improvements over single 80/20 split:
          1. Feature selection: mutual information → top 50 features
          2. Walk-forward: multiple train/test windows over time
          3. Per-fold metrics: tracks train/test R² per fold
          4. Overfitting detection: flags suspicious patterns
          5. Only out-of-sample predictions are returned for metrics
        """
        from betting_intel.pipeline.bootstrap import HAS_ROOT_PREDICTORS
        if not HAS_ROOT_PREDICTORS or not has_total:
            return None

        target_total = "total_points" if has_total else None
        try:
            # ── 1. Feature selection: reduce from 353 → top 50 ──────
            X_all = features_df[feature_cols].fillna(0)
            y_all = features_df[target_total].fillna(features_df[target_total].median())

            selected_cols = _select_top_features(
                X_all.values, y_all.values, feature_cols, n_select=50,
            )
            self.results["metadata"]["n_features_raw"] = len(feature_cols)
            self.results["metadata"]["n_features_selected"] = len(selected_cols)

            # ── 2. Walk-forward windows (pure chronological) ────────────
            n = len(features_df)
            # Use chronological walk-forward: train on past, test on future.
            # Each fold trains on the first K% of data and tests on the next chunk.
            # This simulates real-world: model trained on data up to date T,
            # then predicts games after date T.
            n_folds = 5
            fold_size = n // n_folds
            min_train = 50  # Minimum training samples per fold

            X = X_all.values
            y = y_all.values

            all_preds = pd.Series(index=features_df.index, dtype=float)
            all_is_oos = pd.Series(False, index=features_df.index)
            fold_metrics = []

            for fold in range(n_folds):
                test_start = fold * fold_size
                test_end = test_start + fold_size

                # Ensure training set has enough data
                if test_start < min_train or test_end > n:
                    continue
                # Ensure test set is large enough
                if test_end - test_start < 10:
                    continue

                X_train, y_train = X[:test_start], y[:test_start]
                X_test_slice = X[test_start:test_end]
                y_test_slice = y[test_start:test_end]

                # Apply feature selection to TRAINING data only (no lookahead)
                train_selected = _select_top_features(
                    X_train, y_train, feature_cols,
                    n_select=min(50, len(feature_cols)),
                )
                train_col_indices = [feature_cols.index(c) for c in train_selected]

                # Build ensemble on selected features
                from sklearn.linear_model import Ridge
                sub_ridge = Ridge(alpha=1.0, random_state=42)
                sub_ridge.fit(X_train[:, train_col_indices], y_train)

                models_trained = [("ridge", sub_ridge)]

                try:
                    from lightgbm import LGBMRegressor
                    sub_lgb = LGBMRegressor(
                        n_estimators=150, learning_rate=0.05, max_depth=4,
                        num_leaves=24, random_state=42, verbosity=-1,
                        reg_alpha=0.1, reg_lambda=0.3,
                    )
                    sub_lgb.fit(X_train[:, train_col_indices], y_train)
                    models_trained.append(("lightgbm", sub_lgb))
                except ImportError:
                    pass

                if len(models_trained) >= 1:
                    from betting_intel.models.mlp_predictor import EnhancedEnsemble
                    fold_ensemble = EnhancedEnsemble(
                        log_odds_averaging=False, weight_decay=0.95
                    )
                    for name, model in models_trained:
                        fold_ensemble.add_model(name, model, "regression")

                    # Predict on test window
                    fold_preds = fold_ensemble.predict(
                        X_test_slice[:, train_col_indices]
                    )

                    # Store predictions
                    test_indices = features_df.index[test_start:test_end]
                    all_preds.loc[test_indices] = fold_preds
                    all_is_oos.loc[test_indices] = True

                    # Per-fold metrics
                    from sklearn.metrics import r2_score, mean_absolute_error
                    train_preds = fold_ensemble.predict(X_train[:, train_col_indices])
                    fold_metrics.append({
                        "train_r2": float(r2_score(y_train, train_preds)),
                        "train_mae": float(mean_absolute_error(y_train, train_preds)),
                        "test_r2": float(r2_score(y_test_slice, fold_preds)),
                        "test_mae": float(mean_absolute_error(y_test_slice, fold_preds)),
                        "n_train": len(X_train),
                        "n_test": len(X_test_slice),
                        "n_features": len(train_selected),
                        "fold": fold,
                    })

            # ── 3. Diagnostic: overfitting analysis ──────────────────
            oos_preds = all_preds[all_is_oos].dropna()
            if len(oos_preds) == 0:
                print("  ⚠  No OOS predictions generated — falling back to simple split")
                return None

            diag = _detect_overfitting(fold_metrics)
            self.results["metadata"]["overfitting_diag"] = diag
            self.results["metadata"]["n_folds"] = len(fold_metrics)
            self.results["metadata"]["n_oos_predictions"] = len(oos_preds)
            self.results["metadata"]["model"] = "stacking_ensemble_walkforward"

            if diag.get("overfit", False):
                print(f"  ⚠  ⚠  OVERFITTING DETECTED ⚠  ⚠")
                print(f"       Train R²: {diag['avg_train_r2']:.3f}, Test R²: {diag['avg_test_r2']:.3f}")
                print(f"       Flags: {diag.get('flags', [])}")

            # ── 4. Build result DataFrame with OOS flag ──────────────
            result_df = features_df.copy()
            result_df["predicted_total"] = all_preds
            result_df["_is_oos"] = all_is_oos

            n_oos = all_is_oos.sum()
            n_with_pred = all_preds.notna().sum()
            print(f"  ✅  Walk-forward: {len(fold_metrics)} folds, "
                  f"{n_oos} OOS predictions (of {n_with_pred} total)")
            if fold_metrics:
                avg_test_r2 = np.mean([f["test_r2"] for f in fold_metrics])
                avg_test_mae = np.mean([f["test_mae"] for f in fold_metrics])
                print(f"       Avg fold test R²: {avg_test_r2:.3f}, MAE: {avg_test_mae:.2f}")

            return result_df

        except Exception as e:
            print(f"  ⚠  Walk-forward ensemble failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _manual_train_predict(self, features_df: pd.DataFrame, feature_cols: list,
                               has_total: bool, has_spread: bool,
                               model_dir: Path, tune: bool) -> pd.DataFrame:
        """Fallback manual training and prediction loop."""
        print("  ℹ  Using manual prediction pipeline...")
        n = len(features_df)
        split = max(1, int(n * 0.8))
        train_df = features_df.iloc[:split]
        test_df = features_df.iloc[split:]

        for target_name in [t for t in
                            ["total_points" if has_total else None,
                             "spread" if has_spread else None]
                            if t]:
            try:
                print(f"  🎯  Training predictor for: {target_name}")
                X_train = train_df[feature_cols].fillna(0)
                y_train = train_df[target_name].fillna(train_df[target_name].median())
                X_test = test_df[feature_cols].fillna(0)

                try:
                    from lightgbm import LGBMRegressor
                    model_class = LGBMRegressor
                except ImportError:
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        model_class = RandomForestRegressor
                        print("  ℹ  lightgbm not available, using RandomForestRegressor")
                    except ImportError:
                        print("  ❌  No regression library available (need lightgbm or sklearn)")
                        continue

                model = model_class(
                    n_estimators=200 if tune else 100,
                    learning_rate=0.05,
                    max_depth=5,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                test_df = test_df.copy()
                test_df[f"predicted_{target_name}"] = preds
                print(f"  ✅  {target_name}: trained on {len(train_df)} rows, predicted {len(test_df)}")

                try:
                    if hasattr(model, 'booster_') and hasattr(model.booster_, 'save_model'):
                        model_path = model_dir / f"{target_name}_model.txt"
                        model.booster_.save_model(str(model_path))
                        print(f"  💾  Model saved to {model_path}")
                except Exception:
                    pass
            except Exception as e:
                print(f"  ⚠  Failed to train {target_name}: {e}")

        if "predicted_total" not in test_df.columns and has_total:
            print("  ℹ  No models trained. Using naive historical averages.")
            avg_total = features_df["total_points"].mean()
            test_df["predicted_total"] = avg_total
        if "predicted_spread" not in test_df.columns and has_spread:
            avg_spread = features_df["spread"].mean()
            test_df["predicted_spread"] = avg_spread

        return test_df

    # ── Hyperparameter Tuning ───────────────────────────────────────

    def tune_hyperparameters(self, features_df: pd.DataFrame):
        """Optional hyperparameter tuning with cross-validation."""
        if self.args.no_tune:
            print("\n  ⏩  Hyperparameter tuning skipped (--no-tune)")
            return

        print("\n" + "=" * 70)
        print("  🎛   STAGE 3b: HYPERPARAMETER TUNING")
        print("=" * 70)

        from betting_intel.pipeline.bootstrap import HAS_VALIDATION
        if not HAS_VALIDATION:
            print("  ⚠  Cross-validation module unavailable, skipping tuning")
            return

        feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]
        if not feature_cols:
            print("  ⚠  No feature columns for tuning")

    # ── Full-Data Model (for tomorrow predictions) ───────────────────

    def _train_all_data_model(self, features_df: pd.DataFrame):
        """Train model on ALL historical data and save to self.model.

        Now uses EnhancedEnsemble combining:
          - LightGBM (gradient boosted trees)
          - MLP Neural Network (256→128→64, PyTorch)
          - Ridge Regression (interpretable baseline)

        Like NBA_AI's ensemble approach, weights are adaptive:
        models with lower recent error get higher weight.
        """
        print("  🏋  Training model on FULL dataset for tomorrow predictions...")

        feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]
        if len(feature_cols) < 3:
            print("  ⚠  Not enough feature columns for full-data model")
            return

        target_total = "total_points" if "total_points" in features_df.columns else None
        if not target_total:
            print("  ⚠  No total_points target for full-data model")
            return

        X = features_df[feature_cols].fillna(0)
        y = features_df[target_total].fillna(features_df[target_total].median())

        # ── Train multiple models ────────────────────────────────
        models_trained = []

        # 1. LightGBM (primary model)
        try:
            from lightgbm import LGBMRegressor
            lgb = LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                num_leaves=31, random_state=42, verbosity=-1,
            )
            lgb.fit(X, y)
            models_trained.append(("lightgbm", lgb))
            print(f"  ✅  LightGBM trained on {len(X)} rows")
        except ImportError:
            print("  ℹ  LightGBM unavailable, skipping")

        # 2. Ridge Regression (interpretable baseline)
        try:
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X, y)
            models_trained.append(("ridge", ridge))
            print(f"  ✅  Ridge trained on {len(X)} rows")
        except Exception as e:
            print(f"  ℹ  Ridge failed: {e}")

        # 3. MLP Neural Network (NBA_AI-inspired: 256→128→64)
        if HAS_MLP:
            try:
                mlp = MLPPredictor(
                    input_dim=len(feature_cols),
                    prediction_type="regression",
                    hidden_dims=[256, 128, 64],
                    dropout=0.2,
                    max_epochs=50,  # Quick train for full dataset
                    patience=10,
                    batch_size=64,
                )
                # Use 10% holdout for early stopping
                n = len(X)
                split = int(n * 0.9)
                mlp.fit(
                    X.values[:split], y.values[:split],
                    X_val=X.values[split:], y_val=y.values[split:],
                )
                models_trained.append(("mlp_256", mlp))
                print(f"  ✅  MLP Neural Net trained on {split} rows (val={n - split})")
            except Exception as e:
                print(f"  ℹ  MLP failed (non-fatal): {e}")
        else:
            print("  ℹ  MLP module unavailable (HAS_MLP=False), skipping")

        if not models_trained:
            print("  ❌  No models trained successfully.")
            return

        # ── Build EnhancedEnsemble ─────────────────────────────────
        if HAS_MLP and len(models_trained) >= 2:
            ensemble = EnhancedEnsemble(log_odds_averaging=False, weight_decay=0.95)
            for name, model in models_trained:
                ensemble.add_model(name, model, model_type="regression")
            self.model = ensemble
            print(f"  🧩  EnhancedEnsemble created with {len(models_trained)} models")
            print(f"      Models: {', '.join(n for n, _ in models_trained)}")
        elif models_trained:
            # Fallback: use the first model directly
            self.model = models_trained[0][1]
            print(f"  ℹ  Single model fallback: {models_trained[0][0]}")

        self.model_feature_cols = feature_cols

        # Store feature pipeline for consistent inference
        try:
            self._feature_pipeline = FeatureEngineer()
            if hasattr(self, 'df') and self.df is not None:
                self._pipeline_raw_df = self.df.copy()
        except Exception as e:
            print(f"  ℹ  Feature pipeline storage failed (non-fatal): {e}")

    # ── Tomorrow Feature Vector ─────────────────────────────────────

    def _build_tomorrow_feature_vector(self, home_team: str, away_team: str) -> Optional[pd.Series]:
        """Build a model-compatible feature vector for a specific tomorrow matchup.

        Uses the HISTORICAL features_df to compute team-specific rolling averages
        for each feature column. For a matchup between teams that have played
        before (in the historical data), it extracts the most recent game's
        features. For new matchups, it uses per-team rolling averages.
        """
        if self.model is None or not self.model_feature_cols or self.features_df is None:
            return None

        df = self.features_df

        # Step 1: Try to find a DIRECT matchup in the historical data
        # (teams that have played each other before)
        direct_match = self._find_direct_matchup_features(home_team, away_team, df)
        if direct_match is not None:
            return direct_match

        # Step 2: If no direct matchup, try pipeline-based extraction
        pipeline_result = self._pipeline_based_feature_vector(home_team, away_team, df)
        if pipeline_result is not None:
            return pipeline_result

        # Step 3: Fallback — prefix-based per-team averages
        return self._team_avg_feature_vector(home_team, away_team, df)

    def _find_direct_matchup_features(self, home_team: str, away_team: str,
                                       df: pd.DataFrame) -> Optional[pd.Series]:
        """Find a direct matchup in historical features and return its feature row."""
        try:
            home_cols = [c for c in df.columns if c.startswith("TEAM_NAME_home") or c == "home_team"]
            away_cols = [c for c in df.columns if c.startswith("TEAM_NAME_away") or c == "away_team"]
            if not home_cols or not away_cols:
                return None
            home_col = home_cols[0]
            away_col = away_cols[0]

            # Find the most recent game where these two teams faced each other
            mask = (
                (df[home_col].str.contains(home_team, case=False, na=False))
                & (df[away_col].str.contains(away_team, case=False, na=False))
            ) | (
                (df[home_col].str.contains(away_team, case=False, na=False))
                & (df[away_col].str.contains(home_team, case=False, na=False))
            )
            matched = df[mask]
            if not matched.empty:
                # Use the most recent matchup's feature row
                last_row = matched.iloc[-1]
                feature_dict: Dict[str, float] = {}
                for col in self.model_feature_cols:
                    if col in last_row.index:
                        val = last_row[col]
                        feature_dict[col] = float(val) if pd.notna(val) else 0.0
                    else:
                        feature_dict[col] = 0.0
                result = pd.Series(feature_dict)
                if not result.isnull().any():
                    logger.debug(f"Direct matchup found for {home_team} vs {away_team}")
                    return result
        except Exception as e:
            logger.debug(f"Direct matchup lookup failed: {e}")
        return None

    def _pipeline_based_feature_vector(self, home_team: str, away_team: str,
                                        df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract feature vector using stored FeatureEngineer pipeline."""
        if (self._feature_pipeline is None
                or not hasattr(self._feature_pipeline, 'build_all_features')):
            return None

        try:
            team_mask = (
                (df.get("home_team", "") == home_team)
                | (df.get("away_team", "") == away_team)
                | (df.get("home_team", "") == away_team)
                | (df.get("away_team", "") == home_team)
            )
            team_games = df[team_mask].tail(40).copy()

            if len(team_games) >= 5 and self._pipeline_raw_df is not None:
                pipeline_features = self._feature_pipeline.build_all_features(
                    team_games, self._pipeline_raw_df
                )
                if pipeline_features is not None and len(pipeline_features) > 0:
                    last_row = pipeline_features.iloc[-1]
                    feature_dict = {}
                    for col in self.model_feature_cols:
                        if col in last_row.index:
                            val = last_row[col]
                            feature_dict[col] = float(val) if pd.notna(val) else 0.0
                        else:
                            feature_dict[col] = 0.0
                    result = pd.Series(feature_dict)
                    if not result.isnull().any():
                        logger.debug(f"Pipeline features extracted for {home_team} vs {away_team}")
                        return result
        except Exception as e:
            logger.debug(f"Pipeline feature extraction failed, falling back: {e}")

        return None

    def _team_avg_feature_vector(self, home_team: str, away_team: str,
                                  df: pd.DataFrame) -> pd.Series:
        """Build feature vector using per-team rolling averages for each feature column."""

        # Determine which team name columns exist in df
        # FeatureEngineer produces TEAM_NAME_home/TEAM_NAME_away
        # Some pipelines remap to home_team/away_team
        if "home_team" in df.columns:
            home_team_col = "home_team"
        elif "TEAM_NAME_home" in df.columns:
            home_team_col = "TEAM_NAME_home"
        else:
            home_team_col = None

        if "away_team" in df.columns:
            away_team_col = "away_team"
        elif "TEAM_NAME_away" in df.columns:
            away_team_col = "TEAM_NAME_away"
        else:
            away_team_col = None

        def _team_avg(team: str, base_stat: str, n: int = 10) -> float:
            """Compute rolling average for a stat for a given team.

            Looks for columns matching home_{base_stat} and away_{base_stat}
            patterns in the historical feature set."""
            home_col = f"home_{base_stat}"
            away_col = f"away_{base_stat}"

            # Safely get home values: filter df where home team name matches
            home_vals = pd.Series(dtype=float)
            if home_col in df.columns and home_team_col is not None:
                try:
                    home_mask = df[home_team_col].astype(str).str.contains(team, case=False, na=False)
                    home_vals = df.loc[home_mask, home_col]
                except Exception:
                    pass

            # Safely get away values
            away_vals = pd.Series(dtype=float)
            if away_col in df.columns and away_team_col is not None:
                try:
                    away_mask = df[away_team_col].astype(str).str.contains(team, case=False, na=False)
                    away_vals = df.loc[away_mask, away_col]
                except Exception:
                    pass

            combined = pd.concat([home_vals, away_vals]).tail(n)
            return float(combined.mean()) if len(combined) > 0 else 0.0

        feature_dict: Dict[str, float] = {}
        for col in self.model_feature_cols:
            if col.startswith("home_"):
                base = col[5:]
                feature_dict[col] = _team_avg(home_team, base)
            elif col.startswith("away_"):
                base = col[5:]
                feature_dict[col] = _team_avg(away_team, base)
            elif col.endswith("_diff"):
                base = col.replace("_diff", "")
                feature_dict[col] = _team_avg(home_team, base) - _team_avg(away_team, base)
            elif col.startswith("TEAM_"):
                # Team ID columns: use the team's value from historical data
                feature_dict[col] = 0.0
            else:
                feature_dict[col] = float(df[col].mean()) if col in df.columns else 0.0

        return pd.Series(feature_dict)

    # ── Predict Tomorrow's Games ─────────────────────────────────────

    def predict_tomorrow_games(self) -> List[Dict[str, Any]]:
        """Predict tomorrow's games using the full-data model + market odds.

        Uses self._upcoming_games_df (live odds) as the game list, and
        builds feature vectors from the HISTORICAL features_df and trained
        model. Each upcoming game gets a feature vector by looking at past
        performances of the two teams (rolling averages from the historical
        feature set).
        """
        print("\n" + "=" * 70)
        print("  🔮  STAGE: TOMORROW PREDICTIONS (Full-Data Model)")
        print("=" * 70)

        tomorrow_preds: List[Dict[str, Any]] = []

        if self.model is None:
            print("  ⚠  No full-data model available. Skipping tomorrow predictions.")
            return tomorrow_preds

        # Use upcoming games (live odds) as the source of games to predict
        upcoming_df = getattr(self, '_upcoming_games_df', None)
        if upcoming_df is None or upcoming_df.empty:
            print("  ⚠  No upcoming games data. Skipping tomorrow predictions.")
            return tomorrow_preds

        from betting_intel.pipeline.bootstrap import BetJournal

        for idx, row in upcoming_df.iterrows():
            home = row.get("home_team", row.get("team", ""))
            away = row.get("away_team", row.get("opponent", ""))
            game_id = row.get("game_id", f"upcoming_{idx}")
            if not home or not away:
                continue

            # Build feature vector for this matchup from historical data
            feat = self._build_tomorrow_feature_vector(home, away)
            if feat is None or feat.isnull().any():
                print(f"  ⚠  Could not build features for {home} vs {away} — skipping")
                continue

            try:
                X_pred = feat.values.reshape(1, -1)
                # Use EnhancedEnsemble.predict() if available, else single model
                raw_pred = self.model.predict(X_pred)
                if isinstance(raw_pred, (list, tuple)):
                    # Some ensemble wrappers return tuple
                    raw_pred = raw_pred[0]
                if hasattr(raw_pred, '__len__') and not isinstance(raw_pred, (str, bytes)):
                    predicted_total = float(np.asarray(raw_pred).flatten()[0])
                else:
                    predicted_total = float(raw_pred)
            except Exception as e:
                print(f"  ⚠  Model predict failed for {home} vs {away}: {e}")
                continue

            # Get market odds from the live odds DataFrame
            market_total = row.get("market_total")
            if market_total is None or market_total <= 0:
                print(f"  ⚠  No market line for {home} vs {away} — using model-only estimate")
                market_total = predicted_total  # Can't compute edge without market

            home_ml = row.get("home_ml_odds", -110)
            away_ml = row.get("away_ml_odds", -110)

            edge = (predicted_total - market_total) / max(market_total, 1)
            direction = "over" if edge > 0 else "under"
            abs_edge = abs(edge)
            conf = "high" if abs_edge > 0.05 else ("medium" if abs_edge >= self.args.min_edge else "low")

            game_pred = {
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "game_date": str(row.get("game_date", "")),
                "predicted_total": round(predicted_total, 1),
                "market_total": market_total,
                "edge_pct": round(edge, 4),
                "direction": direction,
                "confidence": conf,
                "implied_odds": {
                    "home_moneyline": home_ml,
                    "away_moneyline": away_ml,
                },
            }
            tomorrow_preds.append(game_pred)

            arrow = "🟢" if abs_edge > 0.03 else ("🔵" if abs_edge > 0.01 else "⚪")
            print(f"  {arrow}  {home:20s} vs {away:<20s}  "
                  f"pred={predicted_total:.1f}  mkt={market_total}  edge={edge:+.2%}  {direction}")

        if tomorrow_preds:
            print(f"  ✅  Predicted {len(tomorrow_preds)} tomorrow games with real model")
            self.results["tomorrow_predictions"] = tomorrow_preds
            self.tomorrow_recommendations_final = tomorrow_preds

            try:
                journal = BetJournal(db_path=str(PROJECT_ROOT / "data" / "bets_journal.db"))
                journal_bets = []
                for tp in tomorrow_preds:
                    d = tp.get("direction", "over")
                    journal_bets.append({
                        "game_date": tp.get("game_date", ""),
                        "game_id": tp.get("game_id", ""),
                        "home_team": tp.get("home_team", ""),
                        "away_team": tp.get("away_team", ""),
                        "bet_type": f"total_{d}",
                        "side": d.upper(),
                        "model_prediction": tp.get("predicted_total"),
                        "market_line": tp.get("market_total"),
                        "edge_pct": tp.get("edge_pct"),
                        "strategy": "ml_pipeline",
                        "model_version": "3.0",
                        "confidence": tp.get("confidence", ""),
                        "league": "NBA",
                    })
                if journal_bets:
                    journal.record_bets(journal_bets)
            except Exception as j_e:
                logger.debug(f"Journal recording failed (non-fatal): {j_e}")
        else:
            print("  ⚠  No valid tomorrow predictions generated.")

        return tomorrow_preds
