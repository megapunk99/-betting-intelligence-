#!/usr/bin/env python3
"""
Monthly Retraining Pipeline — Model Refresh & Drift Tracking.

Trains all models on the latest NBA data, saves them to the model registry
with versioned metadata, and compares performance against the previous month's
versions. Detects feature importance drift and logs everything for monitoring.

Usage:
    python tools/monthly_retrain.py                              # Normal run
    python tools/monthly_retrain.py --force                       # Re-train even if same month
    python tools/monthly_retrain.py --dry-run                     # Show what would happen
    python tools/monthly_retrain.py --output-dir ./output/retrain # Custom output
    python tools/monthly_retrain.py --compare-only                # Just compare, don't train

Scheduling (Windows):
    python tools/schedule_monthly_retrain.py  # Interactive setup
"""

import sys
import json
import warnings
import argparse
import textwrap
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Project imports ───────────────────────────────────────────────────
from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import (
    TotalPointsPredictor, MomentumModel,
)
from betting_intel.models.persistence import model_registry
from betting_intel.backtesting.engine import WalkForwardEngine
from betting_intel.backtesting.metrics import BacktestMetrics
from betting_intel.config import (
    WALK_FORWARD_WINDOW, WALK_FORWARD_STEP, MIN_TRAIN_SAMPLES,
)

# ── ANSI colors ───────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


# ======================================================================
#  Phase 1: Data Loading
# ======================================================================

def load_and_prepare_data(verbose: bool = True) -> tuple[pd.DataFrame, list[str], FeatureEngineer]:
    """Load all historical NBA data and engineer features."""
    if verbose:
        print("  Loading NBA data from database...")

    loader = NBADataLoader()
    fe = FeatureEngineer()

    raw_df = loader.load_game_logs()
    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)

    if verbose:
        print(f"    Raw game logs: {len(raw_df):,} rows")
        print(f"    Merged games:  {len(games_df):,} rows")
        date_range = f"{games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}"
        print(f"    Date range:    {date_range}")

    feature_df = fe.build_all_features(games_df, raw_df)
    feature_cols = fe.select_features(feature_df)

    if verbose:
        print(f"    Features:      {len(feature_cols)}")

    clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
    clean_df = clean_df.reset_index(drop=True)
    clean_df["home_win"] = (clean_df["point_diff"] > 0).astype(int)

    if verbose:
        print(f"    Clean samples: {len(clean_df):,} (dropped {len(feature_df) - len(clean_df)})")

    return clean_df, feature_cols, fe


# ======================================================================
#  Phase 2: Feature Importance Extraction
# ======================================================================

def _classify_feature_theme(name: str) -> str:
    """Classify a feature into an interpretable theme."""
    n = name.lower()
    themes = [
        ("momentum/form",   any(k in n for k in ["streak", "momentum", "weighted", "form_score", "win_streak", "last_3_margin"])),
        ("rest/fatigue",    any(k in n for k in ["rest", "fatigue", "b2b", "3in4"])),
        ("travel/schedule", any(k in n for k in ["travel", "tz_diff", "consec_road", "road_trip", "cum_trav"])),
        ("pace/speed",      any(k in n for k in ["pace"])),
        ("scoring avg",     any(k in n for k in ["avg_pts", "ema_pts", "pts_zscore"])),
        ("scoring eff",     any(k in n for k in ["efg", "ts_pct", "three_pt_rate", "ft_rate"])),
        ("rebounding",      any(k in n for k in ["reb_pct", "oreb", "dreb"])),
        ("margin/dom",      any(k in n for k in ["pm_", "margin", "avg_pm", "ema_pm"])),
        ("opponent qual",   any(k in n for k in ["sos_", "opp_avg", "opp_trailing", "offense_vs_defense", "defense_vs_offense"])),
        ("consistency",     any(k in n for k in ["volatility", "zscore"])),
        ("trend",           any(k in n for k in ["trend_", "ema_"])),
        ("misc/interact",   any(k in n for k in ["interact", "diff", "sq", "adv"])),
    ]
    for theme, matched in themes:
        if matched:
            return theme
    return "other"


def extract_feature_importance(
    model: MomentumModel,
    feature_cols: list[str],
) -> dict:
    """Extract and classify feature importance from a trained model."""
    coefs = model.feature_importance
    if coefs is None and hasattr(model.model, "coef_"):
        coefs = model.model.coef_[0]
    if coefs is None:
        return {"top_features": [], "theme_breakdown": {}}

    feature_map = sorted(
        zip(feature_cols, coefs),
        key=lambda x: abs(x[1]), reverse=True,
    )

    top_features = [
        {"name": name, "coefficient": float(c), "abs_coef": round(abs(c), 4), "theme": _classify_feature_theme(name)}
        for name, c in feature_map[:30]
    ]

    theme_breakdown = defaultdict(float)
    for name, c in feature_map:
        theme_breakdown[_classify_feature_theme(name)] += abs(c)
    total = sum(theme_breakdown.values())
    theme_breakdown = {t: round(v / total, 4) if total > 0 else 0 for t, v in sorted(theme_breakdown.items(), key=lambda x: x[1], reverse=True)}

    return {"top_features": top_features, "theme_breakdown": theme_breakdown}


# ======================================================================
#  Phase 3: Model Training + Walk-Forward
# ======================================================================

MODEL_DEFINITIONS = {
    "totals_ridge": {
        "label": "Total Points (Ridge)",
        "builder": lambda: TotalPointsPredictor("ridge"),
        "target": "total_points",
        "type": "regression",
    },
    "totals_xgboost": {
        "label": "Total Points (GBM)",
        "builder": lambda: TotalPointsPredictor("xgboost"),
        "target": "total_points",
        "type": "regression",
    },
    "momentum_uncalibrated": {
        "label": "Momentum (LogisticRegression)",
        "builder": lambda: MomentumModel("logistic", calibrate=False),
        "target": "home_win",
        "type": "classification",
    },
    "momentum_calibrated": {
        "label": "Momentum (Platt Calibrated)",
        "builder": lambda: MomentumModel("logistic", calibrate=True),
        "target": "home_win",
        "type": "classification",
    },
}


def _select_momentum_features(feature_cols: list[str]) -> list[str]:
    """Select momentum/reversion-related features."""
    momentum_kw = [
        "streak", "momentum", "win_pct", "margin_volatility",
        "form_", "weighted_", "rest_", "fatigue", "travel",
        "net_rating", "sos_", "avg_pm_", "avg_pts_",
        "avg_ts_", "avg_efg_", "tz_", "pace_",
    ]
    selected = [c for c in feature_cols if any(k in c for k in momentum_kw)]
    return selected if len(selected) >= 10 else feature_cols


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_key: str,
    model_def: dict,
    engine: WalkForwardEngine,
    verbose: bool = True,
) -> dict:
    """Train a model on full data AND run walk-forward backtest."""
    if verbose:
        print(f"\n  {CYAN}>> {model_def['label']}{RESET}")

    builder = model_def["builder"]
    target = model_def["target"]
    pred_type = model_def["type"]

    # Select features
    if model_key.startswith("momentum"):
        use_features = _select_momentum_features(feature_cols)
    else:
        use_features = feature_cols

    # Create target if needed
    if target == "home_win" and "home_win" not in df.columns:
        df = df.copy()
        df["home_win"] = (df["point_diff"] > 0).astype(int)

    X = df[use_features].dropna()
    y = df.loc[X.index, target]

    if verbose:
        print(f"    Samples: {len(X):,}  Features: {len(use_features)}")

    # ── Train on full dataset ──────────────────────────────────────
    model = builder()
    model.fit(X.values, y.values)

    # ── Extract feature importance for momentum models ──────────────
    feature_importance = {}
    if model_key.startswith("momentum"):
        feature_importance = extract_feature_importance(model, use_features)

    # Compute training accuracy
    if pred_type == "classification":
        acc = float(np.mean(model.predict(X.values) == y.values))
        if verbose:
            print(f"    Train Acc: {acc:.1%}")
    else:
        preds = model.predict(X.values)
        mae = float(np.mean(np.abs(preds - y.values)))
        if verbose:
            print(f"    Train MAE:  {mae:.1f}")

    # ── Walk-forward backtest ──────────────────────────────────────
    if verbose:
        print(f"    Walk-forward: window={engine.train_window}, step={engine.step}")

    result = engine.run_walk_forward(
        df=df,
        feature_cols=use_features,
        target_col=target,
        model_builder=builder,
        strategy_name="monthly_retrain",
        model_name=model_def["label"],
        prediction_type=pred_type,
        make_bets=True,
    )

    if verbose and result.total_bets > 0:
        print(f"    Walk-forward: {result.total_bets} bets, {result.win_rate:.1%} win rate, {result.total_profit:+.1f}u profit")

    # ── Collect metadata ───────────────────────────────────────────
    model_params = {"model_type": pred_type, "features": len(use_features)}

    metrics = {}
    if result.total_bets > 0:
        detailed = BacktestMetrics.compute_all(result.bets_df)
        if "error" not in detailed:
            metrics = {k: v for k, v in detailed.items() if isinstance(v, (int, float))}

    # Add walk-forward core stats
    metrics["walk_forward_bets"] = result.total_bets
    metrics["walk_forward_win_rate"] = result.win_rate
    metrics["walk_forward_profit"] = result.total_profit
    metrics["walk_forward_roi"] = result.roi
    metrics["walk_forward_sharpe"] = result.sharpe_ratio
    metrics["walk_forward_max_drawdown"] = result.max_drawdown

    # Training set metrics
    if pred_type == "classification":
        metrics["train_accuracy"] = float(acc)
    else:
        metrics["train_mae"] = float(mae)

    metrics["n_samples"] = len(X)
    metrics["n_features"] = len(use_features)

    return {
        "model": model,
        "model_key": model_key,
        "model_label": model_def["label"],
        "feature_cols": use_features,
        "metrics": metrics,
        "params": model_params,
        "feature_importance": feature_importance,
        "backtest_result": result,
    }


# ======================================================================
#  Phase 4: Registry & Versioning
# ======================================================================

def save_to_registry(train_results: list[dict], run_id: str) -> list[dict]:
    """Save all trained models to the model registry with metadata."""
    saved_versions = []

    for res in train_results:
        if res["model"] is None:
            continue

        version = model_registry.save(
            model=res["model"],
            model_name=res["model_label"],
            feature_cols=res["feature_cols"],
            metrics=res["metrics"],
            parameters=res["params"],
        )
        saved_versions.append({
            "model_name": res["model_label"],
            "version": version,
            "metrics": res["metrics"],
        })

        print(f"    {GREEN}Saved{RESET} {res['model_label']} -> version {YELLOW}{version}{RESET}")

    # Save an aggregate run manifest
    manifest_path = model_registry.models_dir / f"retrain_run_{run_id}.json"
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "models": saved_versions,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"    {DIM}Manifest saved: {manifest_path}{RESET}")

    return saved_versions


def load_previous_versions(all_configs: dict) -> dict:
    """Load the most recent previous version of each model from the registry."""
    previous = {}
    for key, config in all_configs.items():
        try:
            model, metadata = model_registry.load(config["label"])
            previous[config["label"]] = {"model": model, "metadata": metadata}
        except (FileNotFoundError, KeyError):
            previous[config["label"]] = None
    return previous


# ======================================================================
#  Phase 5: Comparison & Drift Detection
# ======================================================================

def compare_performance(
    current_versions: list[dict],
    previous_versions: dict,
    all_configs: dict,
) -> dict:
    """Compare current vs previous month performance."""
    comparison = {}

    for saved in current_versions:
        name = saved["model_name"]
        current_metrics = saved.get("metrics", {})
        prev = previous_versions.get(name)

        entry = {
            "model_name": name,
            "current_version": saved["version"],
            "current": current_metrics,
            "previous": None,
            "delta": {},
        }

        if prev is not None:
            prev_meta = prev.get("metadata", {})
            prev_metrics = prev_meta.get("metrics", {})
            entry["previous_version"] = prev_meta.get("version", "unknown")
            entry["previous"] = prev_metrics

            # Compute deltas for key metrics
            for metric in ["walk_forward_win_rate", "walk_forward_profit",
                           "walk_forward_sharpe", "walk_forward_roi",
                           "walk_forward_max_drawdown"]:
                curr_val = current_metrics.get(metric, 0)
                prev_val = prev_metrics.get(metric, 0)
                if prev_val != 0:
                    entry["delta"][metric] = curr_val - prev_val
        else:
            entry["previous_version"] = None

        comparison[name] = entry

    return comparison


def detect_feature_drift(
    current_results: list[dict],
    previous_versions: dict,
) -> list[dict]:
    """Detect drift in feature importance rankings vs previous month."""
    drift_report = []

    for res in current_results:
        if not res["model_key"].startswith("momentum"):
            continue

        name = res["model_label"]
        curr_imp = res.get("feature_importance", {}).get("top_features", [])
        curr_theme = res.get("feature_importance", {}).get("theme_breakdown", {})
        curr_coefs = {f["name"]: f["coefficient"] for f in curr_imp}

        prev = previous_versions.get(name)
        if prev is None:
            drift_report.append({
                "model_name": name,
                "has_drift": False,
                "note": "First version — no baseline to compare",
            })
            continue

        prev_model = prev.get("model")
        if prev_model is None or not hasattr(prev_model, "feature_importance"):
            drift_report.append({
                "model_name": name,
                "has_drift": False,
                "note": "Previous model has no feature importance data",
            })
            continue

        prev_coef = prev_model.feature_importance
        prev_feature_cols = prev.get("metadata", {}).get("feature_cols", [])

        if prev_coef is None or len(prev_coef) == 0 or len(prev_feature_cols) == 0:
            drift_report.append({
                "model_name": name,
                "has_drift": False,
                "note": "Previous model has no coefficient data",
            })
            continue

        # Align features between old and new
        common_features = [f for f in prev_feature_cols if f in curr_coefs]
        if len(common_features) < 5:
            drift_report.append({
                "model_name": name,
                "has_drift": False,
                "note": f"Too few common features ({len(common_features)}) for drift comparison",
            })
            continue

        prev_aligned = np.array([prev_coef[prev_feature_cols.index(f)] for f in common_features])
        curr_aligned = np.array([curr_coefs[f] for f in common_features])

        corr = np.corrcoef(prev_aligned.flatten(), curr_aligned.flatten())[0, 1]
        top_drift = []
        for f in common_features[:10]:
            prev_c = prev_coef[prev_feature_cols.index(f)]
            curr_c = curr_coefs.get(f, 0)
            change = abs(curr_c - prev_c)
            if change > 0.2:
                top_drift.append({"feature": f, "prev": float(prev_c), "curr": float(curr_c), "change": float(change)})

        drift_report.append({
            "model_name": name,
            "has_drift": corr < 0.80,
            "correlation": round(corr, 3),
            "common_features": len(common_features),
            "top_drifters": sorted(top_drift, key=lambda x: x["change"], reverse=True)[:10],
            "note": f"Correlation: r={corr:.3f} ({'DRIFT DETECTED' if corr < 0.80 else 'stable'})",
        })

    return drift_report


# ======================================================================
#  Phase 6: Reporting
# ======================================================================

def print_header():
    print()
    print(CYAN + BOLD + ("=" * 70) + RESET)
    print(CYAN + BOLD + "  MONTHLY MODEL RETRAINING PIPELINE" + RESET)
    print(CYAN + BOLD + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + RESET)
    print(CYAN + BOLD + ("=" * 70) + RESET)


def print_performance_table(comparison: dict):
    """Print a nicely formatted table comparing current vs previous."""
    print(f"\n  {BOLD}PERFORMANCE COMPARISON (Current vs Previous){RESET}")
    print(f"  " + ("-" * 80))

    headers = ["Model", "Metric", "Current", "Previous", "Delta", "Trend"]
    print(f"  {headers[0]:<30s} {headers[1]:<22s} {headers[2]:>8s} {headers[3]:>8s} {headers[4]:>8s} {headers[5]:>6s}")
    print(f"  " + ("-" * 80))

    for name, entry in comparison.items():
        c = entry["current"]
        p = entry["previous"] or {}
        d = entry["delta"]

        metrics = [
            ("Win Rate", c.get("walk_forward_win_rate", 0), p.get("walk_forward_win_rate", 0)),
            ("Profit (u)", c.get("walk_forward_profit", 0), p.get("walk_forward_profit", 0)),
            ("ROI", c.get("walk_forward_roi", 0), p.get("walk_forward_roi", 0)),
            ("Sharpe", c.get("walk_forward_sharpe", 0), p.get("walk_forward_sharpe", 0)),
            ("Max DD", c.get("walk_forward_max_drawdown", 0), p.get("walk_forward_max_drawdown", 0)),
        ]

        first = True
        for label, curr, prev in metrics:
            delta_val = curr - prev if prev != 0 else 0

            if label == "Win Rate":
                curr_s = f"{curr:.1%}"
                prev_s = f"{prev:.1%}" if prev else "N/A"
                delta_s = f"{delta_val:+.1%}" if prev else "NEW"
            elif label == "Sharpe":
                curr_s = f"{curr:.2f}"
                prev_s = f"{prev:.2f}" if prev else "N/A"
                delta_s = f"{delta_val:+.2f}" if prev else "NEW"
            else:
                curr_s = f"{curr:+.1f}" if curr != 0 else "0.0"
                prev_s = f"{prev:+.1f}" if prev else "N/A"
                delta_s = f"{delta_val:+.1f}" if prev else "NEW"

            # Trend indicator
            if prev == 0:
                trend = "NEW"
            elif label == "Max DD":
                trend = f"{GREEN}UP{RESET}" if delta_val < 0 else (f"{RED}DOWN{RESET}" if delta_val > 0 else "FLAT")
            else:
                trend = f"{GREEN}UP{RESET}" if delta_val > 0 else (f"{RED}DOWN{RESET}" if delta_val < 0 else "FLAT")

            name_col = name if first else ""
            first = False

            delta_col = f"{GREEN}{delta_s}{RESET}" if "+" in str(delta_s) and prev else (f"{RED}{delta_s}{RESET}" if "-" in str(delta_s) and prev else delta_s)

            print(f"  {name_col:<30s} {label:<22s} {curr_s:>8s} {prev_s:>8s} {delta_col:>8s} {trend:>6s}")

        print(f"  " + ("-" * 80))

    # Best and worst
    print(f"\n  {BOLD}Summary{RESET}")
    for name, entry in comparison.items():
        c = entry["current"]
        p = entry["previous"] or {}
        curr_wr = c.get("walk_forward_win_rate", 0)
        prev_wr = p.get("walk_forward_win_rate", 0)
        curr_profit = c.get("walk_forward_profit", 0)
        prev_profit = p.get("walk_forward_profit", 0)

        wr_delta = curr_wr - prev_wr if prev_wr else 0
        profit_delta = curr_profit - prev_profit if prev_profit else 0

        verdict_parts = []
        if curr_wr > 0.524:
            verdict_parts.append(f"{GREEN}beating market{RESET}")
        else:
            verdict_parts.append(f"{RED}below breakeven{RESET}")

        if prev_wr:
            if wr_delta > 0.01:
                verdict_parts.append(f"{GREEN}improving{RESET}")
            elif wr_delta < -0.01:
                verdict_parts.append(f"{RED}declining{RESET}")
            else:
                verdict_parts.append("stable")

        print(f"  {name:<34s} WR={curr_wr:.1%} ({wr_delta:+.1%}) Profit={curr_profit:+.1f}u ({profit_delta:+.1f}u) — {' | '.join(verdict_parts)}")


def print_drift_report(drift_report: list[dict]):
    """Print feature importance drift analysis."""
    print(f"\n  {BOLD}FEATURE IMPORTANCE DRIFT{RESET}")
    print(f"  " + ("-" * 70))

    for report in drift_report:
        print(f"\n  {report['model_name']}")
        if report["has_drift"]:
            print(f"    {RED}{BOLD}DRIFT DETECTED{RESET} {report['note']}")
        else:
            print(f"    {GREEN}Stable{RESET} {report['note']}")

        if report.get("top_drifters"):
            print(f"    Top features that changed:")
            for d in report["top_drifters"][:5]:
                print(f"      {d['feature']:<40s} {d['prev']:>+8.4f} -> {d['curr']:>+8.4f} (delta: {d['change']:.4f})")


def print_data_summary(df: pd.DataFrame, feature_cols: list[str]):
    """Print a summary of the data used for training."""
    print(f"\n  {BOLD}DATA SUMMARY{RESET}")
    print(f"  " + ("-" * 50))
    print(f"  Total games:      {len(df):,}")
    print(f"  Date range:       {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
    print(f"  Total features:   {len(feature_cols)}")
    print(f"  Home win rate:    {(df['point_diff'] > 0).mean():.1%}")
    print(f"  Avg total points: {df['total_points'].mean():.1f}")


def generate_report_json(
    comparison: dict,
    drift_report: list[dict],
    train_results: list[dict],
    run_id: str,
    output_dir: Path,
) -> Path:
    """Generate a structured JSON report for easier consumption."""
    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "data_date_range": None,
        "models": {},
        "drift": drift_report,
        "comparison": {},
    }

    for name, entry in comparison.items():
        report["comparison"][name] = {
            "current_version": entry.get("current_version"),
            "previous_version": entry.get("previous_version"),
            "current_metrics": entry.get("current", {}),
            "previous_metrics": entry.get("previous", {}),
            "deltas": entry.get("delta", {}),
        }

    for res in train_results:
        if res["model"] is not None:
            report["models"][res["model_label"]] = {
                "metrics": res.get("metrics", {}),
                "features": len(res.get("feature_cols", [])),
                "feature_importance_top10": res.get("feature_importance", {}).get("top_features", [])[:10],
            }

    report_path = output_dir / f"retrain_report_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monthly Model Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python tools/monthly_retrain.py                        # Normal monthly run
              python tools/monthly_retrain.py --force                 # Re-train even if already done
              python tools/monthly_retrain.py --dry-run               # Show what would happen
              python tools/monthly_retrain.py --compare-only          # Just compare, don't train
        """),
    )
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if already done this month")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without actually training")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training, just compare existing versions")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for reports (default: ./output/retrain)")
    parser.add_argument("--window", type=int, default=WALK_FORWARD_WINDOW,
                        help=f"Walk-forward window (default: {WALK_FORWARD_WINDOW})")
    parser.add_argument("--step", type=int, default=WALK_FORWARD_STEP,
                        help=f"Walk-forward step (default: {WALK_FORWARD_STEP})")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save models to registry (test only)")

    args = parser.parse_args()

    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "output" / "retrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already run this month ────────────────────────────
    this_month = start_time.strftime("%Y-%m")
    manifests = sorted(output_dir.glob("retrain_run_*.json"), reverse=True)
    if manifests and not args.force and not args.compare_only:
        last_manifest = manifests[0]
        try:
            with open(last_manifest) as f:
                last_run = json.load(f)
            last_month = last_run.get("created_at", "")[:7]
            if last_month == this_month:
                print(f"\n  {YELLOW}Already ran this month ({this_month}). Use --force to re-run.{RESET}")
                print(f"  Last run: {last_run.get('run_id', '?')} at {last_run.get('created_at', '?')}")
                print(f"  Use --compare-only to see performance comparison without re-training.\n")
                return 0
        except Exception:
            pass  # Ignore corrupt manifest

    print_header()

    # ── Phase 1: Load previous versions ────────────────────────────
    print(f"\n  {CYAN}{BOLD}[Phase 0/5] Loading Previous Models{RESET}")
    previous = load_previous_versions(MODEL_DEFINITIONS)
    n_prev = sum(1 for v in previous.values() if v is not None)
    print(f"    Found {n_prev} previous model version(s) in registry")

    # ── Compare-only mode ──────────────────────────────────────────
    if args.compare_only:
        print(f"\n  {YELLOW}Compare-only mode: no training, just showing comparison{RESET}")
        # Load the most recent versions for comparison
        current_versions = []
        for key, config in MODEL_DEFINITIONS.items():
            try:
                _, metadata = model_registry.load(config["label"])
                current_versions.append({
                    "model_name": config["label"],
                    "version": metadata.get("version", "?"),
                    "metrics": metadata.get("metrics", {}),
                })
            except FileNotFoundError:
                pass

        if not current_versions:
            print(f"  {RED}No saved models found. Run normally first.{RESET}")
            return 1

        comparison = compare_performance(current_versions, previous, MODEL_DEFINITIONS)
        print_performance_table(comparison)
        print(f"\n  {CYAN}Run with --dry-run to see what would be trained, or without flags to train.{RESET}\n")
        return 0

    # ── Dry run ─────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n  {YELLOW}[DRY RUN] Would perform the following:{RESET}")
        print(f"   1. Load data from database")
        print(f"   2. Train {len(MODEL_DEFINITIONS)} models:")
        for key, config in MODEL_DEFINITIONS.items():
            print(f"      - {config['label']}")
        print(f"   3. Run walk-forward backtest (window={args.window}, step={args.step})")
        print(f"   4. Save to model registry")
        print(f"   5. Compare with {n_prev} previous version(s)")
        print(f"   6. Save report to {output_dir}")
        print(f"\n  Use without --dry-run to execute.\n")
        return 0

    # ── Phase 1: Data Loading ──────────────────────────────────────
    print(f"\n  {CYAN}{BOLD}[Phase 1/5] Loading Data & Engineering Features{RESET}")
    df, feature_cols, fe = load_and_prepare_data()
    print_data_summary(df, feature_cols)

    # ── Phase 2: Walk-Forward Engine ───────────────────────────────
    print(f"\n  {CYAN}{BOLD}[Phase 2/5] Walk-Forward Backtesting{RESET}")
    print(f"    Window: {args.window} games  Step: {args.step} games")
    engine = WalkForwardEngine(train_window=args.window, step=args.step, min_train=MIN_TRAIN_SAMPLES)

    # ── Phase 3: Train All Models ──────────────────────────────────
    print(f"\n  {CYAN}{BOLD}[Phase 3/5] Training {len(MODEL_DEFINITIONS)} Models{RESET}")

    train_results = []
    for key, config in MODEL_DEFINITIONS.items():
        res = train_and_evaluate(df, feature_cols, key, config, engine)
        train_results.append(res)

    # ── Phase 4: Save to Registry ──────────────────────────────────
    print(f"\n  {CYAN}{BOLD}[Phase 4/5] Saving to Model Registry{RESET}")

    if args.no_save:
        print(f"  {YELLOW}--no-save: Skipping registry save{RESET}")
        saved_versions = [
            {"model_name": r["model_label"], "version": "dry-run", "metrics": r["metrics"]}
            for r in train_results
        ]
    else:
        saved_versions = save_to_registry(train_results, run_id)

    # ── Phase 5: Comparison & Drift ────────────────────────────────
    print(f"\n  {CYAN}{BOLD}[Phase 5/5] Performance Comparison & Drift Analysis{RESET}")

    comparison = compare_performance(saved_versions, previous, MODEL_DEFINITIONS)
    print_performance_table(comparison)

    drift_report = detect_feature_drift(train_results, previous)
    print_drift_report(drift_report)

    # ── Save report ────────────────────────────────────────────────
    report_path = generate_report_json(comparison, drift_report, train_results, run_id, output_dir)
    print(f"\n  {GREEN}Report saved: {report_path}{RESET}")

    # ── Summary line for logging ───────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  {BOLD}Run Complete{RESET} ({elapsed:.1f}s)")
    print(f"  {GREEN}" + ("=" * 70) + RESET)

    return 0


if __name__ == "__main__":
    sys.exit(main())
