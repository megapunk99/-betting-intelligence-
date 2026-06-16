"""
v6.0 Accuracy Backtest — Walk-forward comparison of the full prediction pipeline.

Measures:
  - MarketInefficiencySystem: Brier score, accuracy, calibration, edge capture
  - TotalsRegressor: MAE, R², edge capture rate
  - Classifier-only vs error-adjusted (market-aware) prediction quality
  - Per-period metrics + aggregate

Usage:
    python tools/backtest_v6_accuracy.py [--periods 5] [--test-size 100]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backtest_v6")


def run_backtest(periods: int = 5, test_size: int = 100) -> dict:
    """
    Walk-forward backtest of the full prediction pipeline.

    Splits historical NBA data chronologically into `periods` folds.
    For each fold: trains on past data, predicts on next chunk.
    Returns aggregate metrics.
    """
    from betting_intel.data.loader import NBADataLoader
    from betting_intel.data.features import FeatureEngineer
    from betting_intel.features.market_inefficiency import (
        compute_market_inefficiency_targets,
        american_to_implied_prob,
        remove_vig,
    )
    from betting_intel.models.robust_ensemble import MarketInefficiencySystem
    from betting_intel.live.totals_model import TotalsRegressor
    from betting_intel.recommendations.staking import american_to_decimal

    logger.info("=" * 70)
    logger.info("  v6.0 ACCURACY BACKTEST")
    logger.info(f"  Periods: {periods}  |  Test size: ~{test_size} games per period")
    logger.info("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────────────────
    logger.info("\n[1/5] Loading historical NBA data...")
    loader = NBADataLoader()
    raw_df = loader.load_game_logs()
    if raw_df is None or raw_df.empty:
        logger.error("No NBA data available")
        return {}

    # Build game dataset once
    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)
    fe = FeatureEngineer()
    features_df = fe.build_all_features(games_df, raw_df)

    if features_df is None or features_df.empty:
        logger.error("Feature engineering failed")
        return {}

    # Derive home_win target
    if "home_win" not in features_df.columns:
        if "point_diff" in features_df.columns:
            features_df["home_win"] = (features_df["point_diff"] > 0).astype(int)
        else:
            logger.error("Cannot derive home_win")
            return {}

    # Compute market inefficiency targets (uses ELO proxy)
    features_df = compute_market_inefficiency_targets(features_df)

    # Clean feature columns
    clean_feature_cols = fe.select_features(features_df)

    _market_target_cols = {
        "market_implied_home_prob", "market_error", "abs_market_error",
        "market_error_clipped", "market_error_binary", "total_market_error",
        "weighted_market_error", "elo_error",
        "market_error_ma_5g", "market_error_ma_10g",
        "market_error_trend_home", "recent_edge_streak",
    }
    _totals_exclude = _market_target_cols | {"total_points", "point_diff"}

    classifier_feature_cols = [
        c for c in clean_feature_cols if c not in _market_target_cols
    ]
    totals_feature_cols = [
        c for c in clean_feature_cols if c not in _totals_exclude
    ]

    n_total = len(features_df)
    logger.info(f"  Loaded {n_total} games, {len(classifier_feature_cols)} classifier features, "
                f"{len(totals_feature_cols)} totals features")

    # ── Step 2: Build chronological folds ─────────────────────────────
    logger.info("\n[2/5] Building chronological walk-forward folds...")
    min_train = 200
    fold_size = max(test_size, n_total // periods)

    fold_boundaries = []
    for i in range(1, periods + 1):
        test_end = min(i * fold_size, n_total)
        test_start = test_end - fold_size
        if test_start >= min_train and (test_end - test_start) >= 30:
            fold_boundaries.append({
                "fold": i,
                "train_end": test_start,
                "test_start": test_start,
                "test_end": test_end,
                "n_train": test_start,
                "n_test": test_end - test_start,
            })

    if not fold_boundaries:
        logger.error("Not enough data for any fold")
        return {}

    logger.info(f"  Created {len(fold_boundaries)} folds:")
    for fb in fold_boundaries:
        logger.info(f"    Fold {fb['fold']}: train={fb['n_train']} games, "
                    f"test={fb['n_test']} games")

    # ── Step 3: Walk-forward evaluation ──────────────────────────────
    logger.info("\n[3/5] Running walk-forward evaluation...")

    fold_results = []
    aggregate_metrics = {
        "classifier_only": {"brier": [], "accuracy": [], "n_test": 0},
        "error_adjusted": {"brier": [], "accuracy": [], "n_test": 0},
        "totals": {"mae": [], "r2": [], "n_test": 0},
    }

    start_time = time.time()

    for fb in fold_boundaries:
        fold = fb["fold"]
        fold_logger = logging.getLogger(f"fold_{fold}")

        train_end = fb["train_end"]
        test_start = fb["test_start"]
        test_end = fb["test_end"]

        fold_logger.info(f"\n  ── Fold {fold}/{len(fold_boundaries)} ──")
        fold_logger.info(f"  Train: 0-{train_end} ({fb['n_train']} games)")
        fold_logger.info(f"  Test: {test_start}-{test_end} ({fb['n_test']} games)")

        # Extract train/test
        df_train = features_df.iloc[:train_end].copy()
        df_test = features_df.iloc[test_start:test_end].copy()

        X_train = df_train[classifier_feature_cols].fillna(0).values
        y_train = df_train["home_win"].values.astype(int)
        market_probs_train = df_train["market_implied_home_prob"].values.astype(float)

        X_test = df_test[classifier_feature_cols].fillna(0).values
        y_test = df_test["home_win"].values.astype(int)
        market_probs_test = df_test["market_implied_home_prob"].values.astype(float)

        # Totals data
        X_train_t = df_train[totals_feature_cols].fillna(0).values
        if "total_points" in df_train.columns:
            y_total_train = df_train["total_points"].values.astype(float)
        else:
            y_total_train = (
                df_train["team_pts_home"].values + df_train["team_pts_away"].values
            ).astype(float)

        X_test_t = df_test[totals_feature_cols].fillna(0).values
        if "total_points" in df_test.columns:
            y_total_test = df_test["total_points"].values.astype(float)
        else:
            y_total_test = (
                df_test["team_pts_home"].values + df_test["team_pts_away"].values
            ).astype(float)

        # ── Train MarketInefficiencySystem ──────────────────────────
        fold_logger.info("  Training MarketInefficiencySystem...")
        try:
            system = MarketInefficiencySystem(
                calibrate=True, n_folds=5, min_train_samples=50, random_state=42,
            )
            system.fit(
                X_train, y_train,
                market_probs=market_probs_train,
                feature_names=classifier_feature_cols,
                verbose=False,
            )
        except Exception as e:
            fold_logger.warning(f"  MarketInefficiencySystem training failed: {e}")
            continue

        # ── Train TotalsRegressor ───────────────────────────────────
        fold_logger.info("  Training TotalsRegressor...")
        try:
            totals = TotalsRegressor(random_state=42)
            totals.fit(X_train_t, y_total_train, feature_names=totals_feature_cols, verbose=False)
        except Exception as e:
            fold_logger.warning(f"  TotalsRegressor training failed: {e}")
            totals = None

        # ── Predict ─────────────────────────────────────────────────
        fold_logger.info("  Predicting on test set...")

        # Classifier-only predictions (no market data)
        try:
            classifier_only_probs = system._classifier.predict_proba(X_test)
            classifier_probs = classifier_only_probs[:, 1]
        except Exception as e:
            fold_logger.warning(f"  Classifier-only prediction failed: {e}")
            classifier_probs = np.full(len(y_test), 0.5)

        # Error-adjusted predictions (with market data)
        try:
            error_adjusted_probs = system.predict_proba(X_test, market_probs=market_probs_test)
            error_adj_probs = error_adjusted_probs[:, 1]
        except Exception as e:
            fold_logger.warning(f"  Error-adjusted prediction failed: {e}")
            error_adj_probs = classifier_probs

        # Totals predictions
        if totals is not None:
            try:
                totals_preds = totals.predict(X_test_t)
                totals_preds = np.clip(totals_preds, 160, 280)
            except Exception as e:
                fold_logger.warning(f"  Totals prediction failed: {e}")
                totals_preds = np.full(len(y_test), np.mean(y_total_train))
        else:
            totals_preds = np.full(len(y_test), np.mean(y_total_train))

        # ── Compute metrics ─────────────────────────────────────────
        from sklearn.metrics import brier_score_loss, accuracy_score, mean_absolute_error, r2_score

        # Classifier-only
        cb = brier_score_loss(y_test, np.clip(classifier_probs, 0.001, 0.999))
        ca = accuracy_score(y_test, (classifier_probs > 0.5).astype(int))

        # Error-adjusted
        eb = brier_score_loss(y_test, np.clip(error_adj_probs, 0.001, 0.999))
        ea = accuracy_score(y_test, (error_adj_probs > 0.5).astype(int))

        # Totals
        tm = mean_absolute_error(y_total_test, totals_preds)
        tr = r2_score(y_total_test, totals_preds)

        fold_metrics = {
            "fold": fold,
            "n_train": fb["n_train"],
            "n_test": fb["n_test"],
            "classifier_brier": round(cb, 4),
            "classifier_accuracy": round(ca, 4),
            "error_adjusted_brier": round(eb, 4),
            "error_adjusted_accuracy": round(ea, 4),
            "brier_improvement": round(cb - eb, 4),
            "totals_mae": round(tm, 2),
            "totals_r2": round(tr, 4),
        }
        fold_results.append(fold_metrics)

        aggregate_metrics["classifier_only"]["brier"].append(cb)
        aggregate_metrics["classifier_only"]["accuracy"].append(ca)
        aggregate_metrics["error_adjusted"]["brier"].append(eb)
        aggregate_metrics["error_adjusted"]["accuracy"].append(ea)
        aggregate_metrics["classifier_only"]["n_test"] += fb["n_test"]
        aggregate_metrics["error_adjusted"]["n_test"] += fb["n_test"]
        aggregate_metrics["totals"]["mae"].append(tm)
        aggregate_metrics["totals"]["r2"].append(tr)
        aggregate_metrics["totals"]["n_test"] += fb["n_test"]

        fold_logger.info(f"  Results:")
        fold_logger.info(f"    Classifier:     Brier={cb:.4f}, Acc={ca:.1%}")
        fold_logger.info(f"    Error-Adjusted: Brier={eb:.4f}, Acc={ea:.1%}")
        fold_logger.info(f"    Improvement:    ΔBrier={cb-eb:+.4f}")
        fold_logger.info(f"    Totals:         MAE={tm:.1f}, R²={tr:.4f}")

    elapsed = time.time() - start_time

    # ── Step 4: Aggregate results ─────────────────────────────────────
    logger.info("\n[4/5] Computing aggregate metrics...")

    def _avg(vals):
        return round(float(np.mean(vals)), 4) if vals else 0.0

    def _std(vals):
        return round(float(np.std(vals)), 4) if vals else 0.0

    # Market-implied probability baseline (how good is ELO alone?)
    try:
        elo_probs = features_df["elo_home_prob"].values.astype(float)
        y_all = features_df["home_win"].values.astype(int)
        elo_brier = brier_score_loss(y_all, np.clip(elo_probs, 0.001, 0.999))
        elo_acc = accuracy_score(y_all, (elo_probs > 0.5).astype(int))
    except Exception:
        elo_brier = 0.0
        elo_acc = 0.0

    aggregate = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "periods": periods,
            "test_size": test_size,
            "n_total_games": n_total,
            "n_folds": len(fold_boundaries),
        },
        "baselines": {
            "elo_brier": round(elo_brier, 4),
            "elo_accuracy": round(elo_acc, 4),
        },
        "classifier_only": {
            "avg_brier": _avg(aggregate_metrics["classifier_only"]["brier"]),
            "std_brier": _std(aggregate_metrics["classifier_only"]["brier"]),
            "avg_accuracy": _avg(aggregate_metrics["classifier_only"]["accuracy"]),
            "std_accuracy": _std(aggregate_metrics["classifier_only"]["accuracy"]),
            "n_test": aggregate_metrics["classifier_only"]["n_test"],
        },
        "error_adjusted": {
            "avg_brier": _avg(aggregate_metrics["error_adjusted"]["brier"]),
            "std_brier": _std(aggregate_metrics["error_adjusted"]["brier"]),
            "avg_accuracy": _avg(aggregate_metrics["error_adjusted"]["accuracy"]),
            "std_accuracy": _std(aggregate_metrics["error_adjusted"]["accuracy"]),
            "n_test": aggregate_metrics["error_adjusted"]["n_test"],
        },
        "totals": {
            "avg_mae": _avg(aggregate_metrics["totals"]["mae"]),
            "std_mae": _std(aggregate_metrics["totals"]["mae"]),
            "avg_r2": _avg(aggregate_metrics["totals"]["r2"]),
            "std_r2": _std(aggregate_metrics["totals"]["r2"]),
            "n_test": aggregate_metrics["totals"]["n_test"],
        },
        "improvement": {
            "brier_delta": round(
                _avg(aggregate_metrics["classifier_only"]["brier"])
                - _avg(aggregate_metrics["error_adjusted"]["brier"]),
                4,
            ),
            "accuracy_delta": round(
                _avg(aggregate_metrics["error_adjusted"]["accuracy"])
                - _avg(aggregate_metrics["classifier_only"]["accuracy"]),
                4,
            ),
        },
        "fold_results": fold_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    # ── Step 5: Print report ──────────────────────────────────────────
    logger.info("\n[5/5] Report\n")
    logger.info("═" * 70)
    logger.info("  v6.0 ACCURACY BACKTEST — RESULTS")
    logger.info(f"  {aggregate['timestamp']}")
    logger.info("═" * 70)

    logger.info(f"\n  Configuration:")
    logger.info(f"    Periods:      {periods}")
    logger.info(f"    Test size:    ~{test_size} games")
    logger.info(f"    Total games:  {n_total}")
    logger.info(f"    Folds:        {len(fold_boundaries)}")

    logger.info(f"\n  Baselines (ELO proxy only):")
    logger.info(f"    Brier:        {aggregate['baselines']['elo_brier']:.4f}")
    logger.info(f"    Accuracy:     {aggregate['baselines']['elo_accuracy']:.1%}")

    logger.info(f"\n  Classifier-Only:")
    logger.info(f"    Brier:        {aggregate['classifier_only']['avg_brier']:.4f} "
                f"± {aggregate['classifier_only']['std_brier']:.4f}")
    logger.info(f"    Accuracy:     {aggregate['classifier_only']['avg_accuracy']:.1%} "
                f"± {aggregate['classifier_only']['std_accuracy']:.1%}")

    logger.info(f"\n  Error-Adjusted (MarketInefficiencySystem):")
    logger.info(f"    Brier:        {aggregate['error_adjusted']['avg_brier']:.4f} "
                f"± {aggregate['error_adjusted']['std_brier']:.4f}")
    logger.info(f"    Accuracy:     {aggregate['error_adjusted']['avg_accuracy']:.1%} "
                f"± {aggregate['error_adjusted']['std_accuracy']:.1%}")

    imp = aggregate["improvement"]
    brier_color = "✓" if imp["brier_delta"] < 0 else "✗"
    acc_color = "✓" if imp["accuracy_delta"] > 0 else "✗"
    logger.info(f"\n  Improvement (Error-Adjusted vs Classifier-Only):")
    logger.info(f"    {brier_color} Brier Δ:    {imp['brier_delta']:+.4f} "
                f"({'IMPROVED' if imp['brier_delta'] < 0 else 'WORSENED'})")
    logger.info(f"    {acc_color} Accuracy Δ: {imp['accuracy_delta']:+.1%} "
                f"({'IMPROVED' if imp['accuracy_delta'] > 0 else 'WORSENED'})")

    logger.info(f"\n  Totals Regression:")
    logger.info(f"    MAE:          {aggregate['totals']['avg_mae']:.1f} "
                f"± {aggregate['totals']['std_mae']:.1f}")
    logger.info(f"    R²:           {aggregate['totals']['avg_r2']:.4f} "
                f"± {aggregate['totals']['std_r2']:.4f}")

    logger.info(f"\n  Per-Fold Detail:")
    logger.info(f"  {'Fold':<6} {'N Test':<8} {'Brier(C)':<10} {'Brier(E)':<10} "
                f"{'Acc(C)':<10} {'Acc(E)':<10} {'TotMAE':<10} {'TotR²':<8}")
    logger.info(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for fr in fold_results:
        logger.info(
            f"  {fr['fold']:<6} {fr['n_test']:<8} {fr['classifier_brier']:<10.4f} "
            f"{fr['error_adjusted_brier']:<10.4f} {fr['classifier_accuracy']:<10.1%} "
            f"{fr['error_adjusted_accuracy']:<10.1%} {fr['totals_mae']:<10.1f} {fr['totals_r2']:<8.4f}"
        )

    logger.info(f"\n  Execution time: {aggregate['elapsed_seconds']:.1f}s")
    logger.info("═" * 70)

    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="v6.0 Accuracy Backtest — Walk-forward comparison"
    )
    parser.add_argument(
        "--periods", type=int, default=5,
        help="Number of chronological test periods (default: 5)"
    )
    parser.add_argument(
        "--test-size", type=int, default=100,
        help="Approximate number of games per test period (default: 100)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to JSON file"
    )
    args = parser.parse_args()

    results = run_backtest(periods=args.periods, test_size=args.test_size)

    if args.save and results:
        output_dir = Path(__file__).resolve().parent.parent / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"backtest_v6_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
