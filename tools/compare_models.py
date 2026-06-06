#!/usr/bin/env python3
"""
Model comparison: LightGBM vs Ridge vs MLP vs EnhancedEnsemble.

Runs TimeSeriesSplit cross-validation (5 folds) on historical NBA data.
Reports per-fold and aggregate MAE + direction accuracy for each model.

Usage:
    python tools/compare_models.py
    python tools/compare_models.py --features 50   # Use top 50 features via MI
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# -- Model imports --------------------------------------------------------
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from betting_intel.models.mlp_predictor import (
        MLPPredictor, EnhancedEnsemble,
    )
    HAS_MLP = True
except ImportError:
    HAS_MLP = False

try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_MI = True
except ImportError:
    HAS_MI = False

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer


# -- Data loading ---------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and engineer features from the NBA database."""
    print("Loading data...")
    loader = NBADataLoader()
    raw_df = loader.load_game_logs()
    if raw_df is None or raw_df.empty:
        print("[FAIL]  No data loaded. Check database connection.")
        sys.exit(1)

    print(f"  Loaded {len(raw_df)} game logs")
    raw_df["IS_HOME"] = raw_df["MATCHUP"].fillna("").str.contains("vs.").astype(int)
    raw_df = loader.compute_rest_days(raw_df)
    games_df = loader.build_game_dataset(raw_df)
    print(f"  Built {len(games_df)} game records")

    engineer = FeatureEngineer()
    features_df = engineer.build_all_features(games_df, raw_df)
    print(f"  Engineered {len(features_df.columns)} columns, {len(features_df)} rows")
    return features_df


def select_features(features_df: pd.DataFrame, n_select: int = 0) -> list:
    """Get feature columns, optionally with MI selection."""
    exclude = {
        "game_id", "game_date", "home_team", "away_team",
        "total_points", "spread", "label", "home_win",
        "home_score", "away_score", "GAME_ID", "SEASON_ID",
        "TEAM_ID_home", "TEAM_ID_away",
        "TEAM_ABBREVIATION_home", "TEAM_ABBREVIATION_away",
        "TEAM_NAME_home", "TEAM_NAME_away", "GAME_DATE",
        "MATCHUP_home", "MATCHUP_away",
        "WL_home", "WL_away",
        "team_pts_home", "team_pts_away",
        "point_diff", "team_fgm_home", "team_fgm_away",
        "team_fga_home", "team_fga_away",
        "team_fg3m_home", "team_fg3m_away",
        "team_fg3a_home", "team_fg3a_away",
        "team_ftm_home", "team_ftm_away",
        "team_fta_home", "team_fta_away",
        "team_oreb_home", "team_oreb_away",
        "team_dreb_home", "team_dreb_away",
        "team_reb_home", "team_reb_away",
        "team_ast_home", "team_ast_away",
        "team_stl_home", "team_stl_away",
        "team_blk_home", "team_blk_away",
        "team_tov_home", "team_tov_away",
        "team_pf_home", "team_pf_away",
        "team_plus_minus_home", "team_plus_minus_away",
        "MIN_home", "MIN_away",
        "home_team_name", "away_team_name",
        "rest_home_key", "rest_away_key",
        "market_line_baseline", "market_line_pace_adj",
        "trailing_avg_total_10g",
        "IS_HOME_home", "IS_HOME_away",
        "OPPONENT_home", "OPPONENT_away",
        "three_pt_rate_home", "three_pt_rate_away",
        "ft_rate_home", "ft_rate_away",
        "ast_ratio_home", "ast_ratio_away",
        "ts_pct_home", "ts_pct_away",
        "reb_pct_home", "reb_pct_away",
        "home_tz", "away_tz",
        "WL_num_home", "WL_num_away",
    }

    feature_cols = [
        c for c in features_df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    # If n_select > 0 and less than total, run MI selection on all data
    # (This is a global selection for comparison purposes -- each model
    # sees the same features within each fold)
    if n_select > 0 and n_select < len(feature_cols) and HAS_MI:
        X_all = features_df[feature_cols].fillna(0).values
        y_all = features_df["total_points"].fillna(
            features_df["total_points"].median()
        ).values
        mi = mutual_info_regression(X_all, y_all, random_state=42, n_neighbors=5)
        top_indices = np.argsort(mi)[-n_select:][::-1]
        feature_cols = [feature_cols[i] for i in top_indices]
        print(f"  Selected top {n_select} features via mutual information")

    print(f"  Using {len(feature_cols)} features")
    return feature_cols


# -- Model builders -------------------------------------------------------


def build_ridge(**kwargs):
    return Ridge(alpha=1.0, random_state=42)


def build_lightgbm(**kwargs):
    return LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4,
        num_leaves=24, random_state=42, verbosity=-1,
        reg_alpha=0.1, reg_lambda=0.3,
    )


def build_mlp(n_features: int):
    return MLPPredictor(
        input_dim=n_features,
        prediction_type="regression",
        hidden_dims=[64, 32],       # smaller to prevent divergence with 361 features
        dropout=0.3,                 # increased dropout for regularization
        max_epochs=100,              # smaller network trains fast; more epochs for convergence
        patience=15,
        batch_size=64,
    )


def build_ensemble(ridge, lgb, mlp):
    ensemble = EnhancedEnsemble(log_odds_averaging=False, weight_decay=0.95)
    ensemble.add_model("ridge", ridge, model_type="regression")
    if lgb is not None:
        ensemble.add_model("lightgbm", lgb, model_type="regression")
    if mlp is not None:
        ensemble.add_model("mlp_64_32", mlp, model_type="regression")
    return ensemble


# -- TimeSeriesSplit ------------------------------------------------------


def run_comparison(
    features_df: pd.DataFrame,
    feature_cols: list,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run TimeSeriesSplit comparison across all models.

    Each fold: train on [0:test_start], test on [test_start:test_end].
    Records per-fold MAE and direction accuracy for each model.
    """
    n = len(features_df)
    fold_size = n // n_splits
    min_train = 50

    results = []

    X_all = features_df[feature_cols].fillna(0).values
    y_all = features_df["total_points"].fillna(
        features_df["total_points"].median()
    ).values

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        if test_start < min_train or test_end > n:
            print(f"  Fold {fold + 1}: skipped (insufficient training data)")
            continue
        if test_end - test_start < 10:
            continue

        X_train, y_train = X_all[:test_start], y_all[:test_start]
        X_test, y_test = X_all[test_start:test_end], y_all[test_start:test_end]

        fold_results = {"fold": fold + 1, "n_train": len(X_train), "n_test": len(X_test)}

        # -- 1. Ridge ----------------------------------------------------
        ridge = build_ridge()
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)
        fold_results["ridge_mae"] = float(mean_absolute_error(y_test, ridge_preds))
        fold_results["ridge_dir_acc"] = float(np.mean(
            (ridge_preds > np.mean(y_train)) == (y_test > np.mean(y_train))
        ))

        # -- 2. LightGBM -------------------------------------------------
        if HAS_LGB:
            lgb = build_lightgbm()
            lgb.fit(X_train, y_train)
            lgb_preds = lgb.predict(X_test)
            fold_results["lgb_mae"] = float(mean_absolute_error(y_test, lgb_preds))
            fold_results["lgb_dir_acc"] = float(np.mean(
                (lgb_preds > np.mean(y_train)) == (y_test > np.mean(y_train))
            ))
        else:
            fold_results["lgb_mae"] = float("nan")
            fold_results["lgb_dir_acc"] = float("nan")

        # -- 3. MLP ------------------------------------------------------
        if HAS_MLP:
            try:
                mlp = build_mlp(n_features=len(feature_cols))
                # Use 10% of training data as validation
                val_split = int(len(X_train) * 0.9)
                mlp.fit(
                    X_train[:val_split], y_train[:val_split],
                    X_val=X_train[val_split:], y_val=y_train[val_split:],
                )
                mlp_preds = mlp.predict(X_test)
                fold_results["mlp_mae"] = float(mean_absolute_error(y_test, mlp_preds))
                fold_results["mlp_dir_acc"] = float(np.mean(
                    (mlp_preds > np.mean(y_train)) == (y_test > np.mean(y_train))
                ))
            except Exception as e:
                print(f"  [WARN] MLP fold {fold + 1} failed: {e}")
                fold_results["mlp_mae"] = float("nan")
                fold_results["mlp_dir_acc"] = float("nan")
        else:
            fold_results["mlp_mae"] = float("nan")
            fold_results["mlp_dir_acc"] = float("nan")

        # -- 4. EnhancedEnsemble -----------------------------------------
        if HAS_MLP and HAS_LGB:
            try:
                mlp_ok = not np.isnan(fold_results.get("mlp_mae", np.nan))
                ensemble = build_ensemble(ridge, lgb, mlp if mlp_ok else None)
                ens_preds = ensemble.predict(X_test)
                fold_results["ensemble_mae"] = float(mean_absolute_error(y_test, ens_preds))
                fold_results["ensemble_dir_acc"] = float(np.mean(
                    (ens_preds > np.mean(y_train)) == (y_test > np.mean(y_train))
                ))
            except Exception as e:
                print(f"  [WARN] Ensemble fold {fold + 1} failed: {e}")
                fold_results["ensemble_mae"] = float("nan")
                fold_results["ensemble_dir_acc"] = float("nan")
        else:
            # Partial ensemble (just Ridge + available)
            try:
                parts = [("ridge", ridge)]
                if HAS_LGB:
                    parts.append(("lightgbm", lgb))
                ensemble = EnhancedEnsemble(log_odds_averaging=False, weight_decay=0.95)
                for name, model in parts:
                    ensemble.add_model(name, model, "regression")
                ens_preds = ensemble.predict(X_test)
                fold_results["ensemble_mae"] = float(mean_absolute_error(y_test, ens_preds))
                fold_results["ensemble_dir_acc"] = float(np.mean(
                    (ens_preds > np.mean(y_train)) == (y_test > np.mean(y_train))
                ))
            except Exception as e:
                print(f"  [WARN] Ensemble fold {fold + 1} failed: {e}")
                fold_results["ensemble_mae"] = float("nan")
                fold_results["ensemble_dir_acc"] = float("nan")

        results.append(fold_results)

        mean_total = float(np.mean(y_test))
        print(f"  Fold {fold + 1}: train={len(X_train)}, test={len(X_test)}, "
              f"mean_total={mean_total:.1f}")

    return pd.DataFrame(results)


# -- Main / Output formatting ---------------------------------------------


def print_results(df: pd.DataFrame):
    """Print formatted comparison results."""
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON -- TimeSeriesSplit Cross-Validation")
    print("=" * 72)

    models = [
        ("ridge", "Ridge"),
        ("lgb", "LightGBM"),
        ("mlp", "MLP"),
        ("ensemble", "EnhancedEnsemble"),
    ]

    header = f"{'Model':<20} {'MAE':<10} {'Dir Acc':<10} {'Folds':<8}"
    print(header)
    print("-" * 60)

    for key, label in models:
        mae_col = f"{key}_mae"
        dir_col = f"{key}_dir_acc"
        if mae_col not in df.columns:
            continue

        mae_vals = df[mae_col].dropna()
        dir_vals = df[dir_col].dropna()

        if len(mae_vals) == 0:
            continue

        avg_mae = float(mae_vals.mean())
        avg_dir = float(dir_vals.mean())

        print(f"  {label:<18} {avg_mae:<10.2f} {avg_dir:<10.1%} "
              f"{len(mae_vals):<8} ")

    # Compute improvement over baseline (Ridge)
    ens_mae_vals = df["ensemble_mae"].dropna()
    if len(ens_mae_vals) > 0:
        ens_mae = float(ens_mae_vals.mean())
        print("-" * 60)
        for key, label in models:
            mae_vals = df[f"{key}_mae"].dropna()
            if key == "ensemble" or len(mae_vals) == 0:
                continue
            avg_mae = float(mae_vals.mean())
            improvement = (avg_mae - ens_mae) / avg_mae * 100
            print(f"  {label} vs Ensemble: {improvement:+.1f}% MAE change")

    print("-" * 60)

    # Per-fold breakdown
    print("\n  Per-fold MAE:")
    print(f"  {'Fold':<6} {'Ridge':<10} {'LGB':<10} {'MLP':<10} {'Ensemble':<10}")
    for _, row in df.iterrows():
        fold = int(row["fold"])
        r = f"{row.get('ridge_mae', float('nan')):<10.2f}"
        l = f"{row.get('lgb_mae', float('nan')):<10.2f}" if HAS_LGB else "N/A       "
        m = f"{row.get('mlp_mae', float('nan')):<10.2f}" if HAS_MLP else "N/A       "
        e = f"{row.get('ensemble_mae', float('nan')):<10.2f}"
        print(f"  {fold:<6} {r} {l} {m} {e}")

    print("\n  Per-fold Direction Accuracy:")
    print(f"  {'Fold':<6} {'Ridge':<10} {'LGB':<10} {'MLP':<10} {'Ensemble':<10}")
    for _, row in df.iterrows():
        fold = int(row["fold"])
        r = f"{row.get('ridge_dir_acc', float('nan')):<10.1%}"
        l = f"{row.get('lgb_dir_acc', float('nan')):<10.1%}" if HAS_LGB else "N/A       "
        m = f"{row.get('mlp_dir_acc', float('nan')):<10.1%}" if HAS_MLP else "N/A       "
        e = f"{row.get('ensemble_dir_acc', float('nan')):<10.1%}"
        print(f"  {fold:<6} {r} {l} {m} {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare ML models for NBA total prediction")
    parser.add_argument("--features", type=int, default=0,
                        help="Number of top features to select (0 = all)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of TimeSeriesSplit folds")
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    features_df = load_data()
    feature_cols = select_features(features_df, n_select=args.features)

    lgb_status = "[OK]" if HAS_LGB else "[NO]"
    mlp_status = "[OK]" if HAS_MLP else "[NO]"
    ens_status = "[OK]" if HAS_MLP else "[PARTIAL]"
    print(f"\nModels available: Ridge [OK]  LightGBM {lgb_status}  "
          f"MLP {mlp_status}  Ensemble {ens_status}")
    print(f"Running {args.folds}-fold TimeSeriesSplit...\n")

    # Run comparison
    results_df = run_comparison(
        features_df, feature_cols, n_splits=args.folds,
    )

    if results_df.empty:
        print("[FAIL]  No folds completed. Not enough data.")
        return

    print_results(results_df)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
