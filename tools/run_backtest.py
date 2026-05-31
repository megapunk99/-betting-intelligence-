#!/usr/bin/env python3
"""
Walk-Forward Backtest - Momentum Model Validation on Real NBA Data

Loads 695 real games from nba_data.db, builds features, and runs walk-forward
validation on multiple model configurations. Reports comprehensive metrics
including win rate, ROI, Sharpe ratio, drawdown, and statistical significance.

Usage:
    python tools/run_backtest.py                          # Quick run: Ridge + Momentum only
    python tools/run_backtest.py --full                   # All models
    python tools/run_backtest.py --model ridge            # Just Ridge
    python tools/run_backtest.py --model momentum         # Just MomentumModel
    python tools/run_backtest.py --model xgboost          # Just XGBoost
    python tools/run_backtest.py --window 150 --step 15   # Custom walk-forward params
    python tools/run_backtest.py --no-bets                # Model metrics only (no betting sim)
"""

import sys
import warnings
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import (
    TotalPointsPredictor, MomentumModel,
)
from betting_intel.backtesting.engine import WalkForwardEngine, BacktestResult
from betting_intel.backtesting.metrics import BacktestMetrics
from betting_intel.config import (
    WALK_FORWARD_WINDOW, WALK_FORWARD_STEP, MIN_TRAIN_SAMPLES,
    MIN_EDGE_THRESHOLD,
)

# -- ANSI Colors --
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(label: str, value: str) -> str:
    return f"{GREEN}{label}:{RESET} {value}"


def warn(label: str, value: str) -> str:
    return f"{YELLOW}{label}:{RESET} {value}"


def bad(label: str, value: str) -> str:
    return f"{RED}{label}:{RESET} {value}"


def header(text: str) -> str:
    return "\n" + CYAN + BOLD + ("=" * 65) + "\n  " + text + "\n" + ("=" * 65) + RESET


# ============================================================================
#  Backtest Runner
# ============================================================================


def load_and_prepare_data():
    """Load 695 real games and engineer features. Returns (clean_df, feature_cols)."""
    print("  Loading NBA data from database...")
    loader = NBADataLoader()
    fe = FeatureEngineer()

    raw_df = loader.load_game_logs()
    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)

    print(f"    Raw game logs: {len(raw_df):,} rows")
    print(f"    Merged games:  {len(games_df):,} rows")
    print(f"    Date range:    {games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}")
    print(f"    Unique teams:  {games_df['TEAM_NAME_home'].nunique()}")

    print("  Engineering features...")
    feature_df = fe.build_all_features(games_df, raw_df)
    feature_cols = fe.select_features(feature_df)

    print(f"    Total features: {len(feature_cols)}")
    if feature_cols:
        print(f"    Sample: {feature_cols[:5]}...")

    clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
    clean_df = clean_df.reset_index(drop=True)
    print(f"    Clean samples: {len(clean_df):,} (dropped {len(feature_df) - len(clean_df)})")

    return clean_df, feature_cols


def build_model_configs(calibrated: bool = False):
    """Define model configurations to test."""
    configs = {
        "Momentum (LogisticRegression)": {
            "builder": lambda: MomentumModel("logistic", calibrate=False),
            "target": "home_win",
            "type": "classification",
            "features": None,
        },
        "Total Points (Ridge)": {
            "builder": lambda: TotalPointsPredictor("ridge"),
            "target": "total_points",
            "type": "regression",
            "features": None,
        },
        "Total Points (GBM)": {
            "builder": lambda: TotalPointsPredictor("xgboost"),
            "target": "total_points",
            "type": "regression",
            "features": None,
        },
    }
    if calibrated:
        configs["Momentum (Platt Calibrated)"] = {
            "builder": lambda: MomentumModel("logistic", calibrate=True),
            "target": "home_win",
            "type": "classification",
            "features": None,
        }
    return configs


def select_momentum_features(feature_cols):
    """Feature subset for momentum/reversion models."""
    momentum_kw = [
        "streak", "momentum", "win_pct", "margin_volatility",
        "form_", "weighted_", "rest_", "fatigue", "travel",
        "net_rating", "sos_", "avg_pm_", "avg_pts_",
        "avg_ts_", "avg_efg_", "tz_", "pace_",
    ]
    selected = [c for c in feature_cols if any(kw in c for kw in momentum_kw)]
    return selected if len(selected) >= 10 else feature_cols


def run_backtest(df, feature_cols, model_name, model_config, make_bets, engine):
    """Run a single walk-forward backtest."""
    target = model_config["target"]
    pred_type = model_config["type"]
    model_builder = model_config["builder"]

    # Select features
    if model_name.startswith("Momentum"):
        use_features = select_momentum_features(feature_cols)
    else:
        use_features = feature_cols

    # For classification, create the target column
    if target == "home_win" and "home_win" not in df.columns:
        df = df.copy()
        df["home_win"] = (df["point_diff"] > 0).astype(int)

    return engine.run_walk_forward(
        df=df,
        feature_cols=use_features,
        target_col=target,
        model_builder=model_builder,
        strategy_name="backtest",
        model_name=model_name,
        prediction_type=pred_type,
        make_bets=make_bets,
    )


def print_model_summary(name, result):
    """Print one model's backtest results."""
    print(f"\n  {BOLD}{name}{RESET}")
    if result.errors:
        for e in result.errors[:3]:
            print(f"    {YELLOW}[!]{RESET} {e[:120]}")

    if result.total_bets > 0:
        print(f"    {ok('Bets', str(result.total_bets))}")
        wr_str = f"{result.win_rate:.1%}"
        if result.win_rate > 0.524:
            wr_display = ok("Win Rate", wr_str)
        elif result.win_rate > 0.50:
            wr_display = warn("Win Rate", wr_str)
        else:
            wr_display = bad("Win Rate", wr_str)
        print(f"    {wr_display}")

        roi_str = f"{result.roi:+.1f}%"
        if result.roi > 5:
            roi_display = ok("ROI", roi_str)
        elif result.roi > 0:
            roi_display = warn("ROI", roi_str)
        else:
            roi_display = bad("ROI", roi_str)
        print(f"    {roi_display}")

        profit_str = f"{result.total_profit:+.1f}u"
        if result.total_profit > 10:
            profit_display = ok("Profit", profit_str)
        elif result.total_profit > 0:
            profit_display = warn("Profit", profit_str)
        else:
            profit_display = bad("Profit", profit_str)
        print(f"    {profit_display}")

        sharpe_str = f"{result.sharpe_ratio:.2f}"
        if result.sharpe_ratio > 1.0:
            sharpe_display = ok("Sharpe", sharpe_str)
        elif result.sharpe_ratio > 0.5:
            sharpe_display = warn("Sharpe", sharpe_str)
        else:
            sharpe_display = bad("Sharpe", sharpe_str)
        print(f"    {sharpe_display}")

        print(f"    {ok('Max DD', f'{result.max_drawdown:.1f}u')}")
        print(f"    Wins: {result.wins}  Losses: {result.losses}  Pushes: {result.pushes}")

        # Historical win rate needed to break even at -110 odds
        be = 52.38 / 100
        if result.win_rate > be:
            edge_over_vig = result.win_rate - be
            print(f"    {ok('Edge over vig', f'{edge_over_vig:+.2%}')}")
        else:
            edge_over_vig = result.win_rate - be
            print(f"    {bad('Edge over vig', f'{edge_over_vig:+.2%}')}")

    elif result.model_metrics:
        m = result.model_metrics
        print(f"    {ok('Predictions', str(m.get('n_predictions', 0)))}")
        print(f"    MAE: {m.get('mae', 0):.1f}  RMSE: {m.get('rmse', 0):.1f}  R2: {m.get('r2', 0):.3f}")

        # Probability calibration metrics (for classification models)
        if "brier_score" in m:
            brier = m["brier_score"]
            brier_str = f"{brier:.4f}"
            brier_color = GREEN if brier < 0.20 else (YELLOW if brier < 0.22 else RED)
            print(f"    Brier: {brier_color}{brier_str}{RESET}  "
                  f"LogLoss: {m.get('log_loss', 0):.4f}  "
                  f"ECE: {m.get('calibration_error_ece', 0):.3f}")
            if "brier_skill_score" in m:
                print(f"    Brier Skill Score: {m['brier_skill_score']:+.2%}  "
                      f"Max Cal Error: {m.get('calibration_error_max', 0):.3f}")
    else:
        print(f"    {YELLOW}[!]{RESET} No predictions or bets generated")


def print_detailed_metrics(result):
    """Print comprehensive BacktestMetrics from a result."""
    if result.bets_df.empty:
        return

    metrics = BacktestMetrics.compute_all(result.bets_df)
    if "error" in metrics:
        print(f"    {YELLOW}[!]{RESET} Metrics error: {metrics['error']}")
        return

    print(f"\n  {BOLD}Detailed Performance{RESET}")
    print(f"  {'-'*50}")

    # Statistical significance
    wr = metrics.get("win_rate", 0)
    pval = metrics.get("p_value_gt_50pct", 1.0)
    ci_low = metrics.get("win_rate_ci_lower", 0)
    ci_high = metrics.get("win_rate_ci_upper", 0)
    significant = metrics.get("is_significant", False)

    sig_str = f"  Win Rate:  {wr:.1%}  (95% CI: {ci_low:.1%} - {ci_high:.1%})"
    if significant:
        sig_str += f"  {GREEN}Statistically significant (p={pval:.4f}){RESET}"
    else:
        sig_str += f"  {YELLOW}Not significant (p={pval:.4f}){RESET}"
    print(sig_str)

    # Risk metrics
    print(f"  Sharpe:    {metrics.get('sharpe_ratio', 0):.2f}  "
          f"Sortino: {metrics.get('sortino_ratio', 0):.2f}  "
          f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Max DD:    {metrics.get('max_drawdown_units', 0):.1f}u  "
          f"Recovery: {metrics.get('recovery_factor', 0):.2f}")

    # Streaks
    print(f"  Streaks:   Win {metrics.get('longest_win_streak', 0)}  "
          f"Loss {metrics.get('longest_loss_streak', 0)}")

    # Monthly breakdown
    profitable = metrics.get("profitable_months", 0)
    total_months = metrics.get("total_months", 1)
    best_m = metrics.get("best_month", 0)
    worst_m = metrics.get("worst_month", 0)
    print(f"  Months:    {profitable}/{total_months} profitable  "
          f"Best: {best_m:+.1f}u  Worst: {worst_m:+.1f}u")

    # Edge analysis
    if "avg_edge_pct" in metrics:
        print(f"  Avg Edge:  {metrics['avg_edge_pct']:+.2%}  "
              f"Edge-Outcome Corr: {metrics.get('edge_outcome_corr', 0):.2f}")

    # Probability calibration metrics (for classification models with predicted probs)
    if "brier_score" in metrics:
        print(f"  Brier:     {metrics['brier_score']:.4f}  "
              f"LogLoss: {metrics.get('log_loss', 0):.4f}  "
              f"ECE: {metrics.get('calibration_error_ece', 0):.3f}")
        if "brier_skill_score" in metrics:
            print(f"  Brier SS:  {metrics['brier_skill_score']:+.2%}  "
                  f"Max Cal Error: {metrics.get('calibration_error_max', 0):.3f}")

        # Store calibration metrics back on result for later use by comparison table
        for key in ["brier_score", "brier_skill_score", "log_loss",
                     "calibration_error_ece", "calibration_error_max"]:
            if key in metrics:
                result.model_metrics[key] = metrics[key]


def print_comparison_table(all_results):
    """Print a side-by-side comparison of all models."""
    print(f"\n{BOLD}{'-'*65}")
    print(f"  MODEL COMPARISON")
    print(f"{'-'*65}{RESET}")

    # Check if any model has probability metrics
    has_brier = any(
        "brier_score" in r.model_metrics for r in all_results.values()
    )

    if has_brier:
        # Extended table with calibration columns
        print(f"  {'Model':<34s} {'Bets':>5s} {'Win%':>7s} {'Profit':>8s} {'Sharpe':>7s} {'Brier':>7s} {'ECE':>6s}")
        print(f"  {'-'*34} {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")

        for name, result in all_results.items():
            if result.total_bets > 0:
                brier_str = ""
                ece_str = ""
                if "brier_score" in result.model_metrics:
                    brier_str = f"{result.model_metrics['brier_score']:.4f}"
                    ece_str = f"{result.model_metrics.get('calibration_error_ece', 0):.3f}"
                print(f"  {name:<34s} {result.total_bets:>5d} "
                      f"{result.win_rate:>6.1%} "
                      f"{result.total_profit:>+7.1f}u "
                      f"{result.sharpe_ratio:>6.2f} "
                      f"{brier_str:>7s} {ece_str:>6s}")
    else:
        # Standard table
        print(f"  {'Model':<34s} {'Bets':>5s} {'Win%':>7s} {'Profit':>8s} {'ROI':>7s} {'Sharpe':>7s} {'MaxDD':>7s}")
        print(f"  {'-'*34} {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

        for name, result in all_results.items():
            if result.total_bets > 0:
                print(f"  {name:<34s} {result.total_bets:>5d} "
                      f"{result.win_rate:>6.1%} "
                      f"{result.total_profit:>+7.1f}u "
                      f"{result.roi:>+6.1f}% "
                      f"{result.sharpe_ratio:>6.2f} "
                      f"{result.max_drawdown:>6.1f}u")

    print(f"  {'-'*65}")
    print()

    # Best performer
    best = max(all_results.items(), key=lambda x: x[1].sharpe_ratio if x[1].sharpe_ratio > 0 else -999)
    print(f"  {GREEN}Best: {best[0]} (Sharpe {best[1].sharpe_ratio:.2f}){RESET}")

    # Check significance
    for name, result in all_results.items():
        if result.total_bets > 0:
            total = result.wins + result.losses
            if total > 0:
                pval = float(stats.binomtest(result.wins, total, p=0.5).pvalue)
                if pval < 0.05:
                    print(f"  {GREEN}OK{RESET} {name}: statistically significant (p={pval:.4f})")
                else:
                    print(f"  {YELLOW}--{RESET} {name}: NOT significant (p={pval:.4f})")


def print_sample_bets(result, n=8):
    """Print a sample of individual bet records."""
    if result.bets_df.empty:
        return

    df = result.bets_df.sort_values("game_date").head(n)
    print(f"\n  {BOLD}Sample Bets (first {n}){RESET}")
    print(f"  {'-'*75}")
    print(f"  {'Date':<12s} {'Matchup':<30s} {'Pred':>6s} {'Market':>6s} {'Actual':>6s} {'Edge':>7s} {'Result':>6s}")
    print(f"  {'-'*75}")

    for _, bet in df.iterrows():
        matchup = str(bet.get("matchup", ""))[:28]
        edge = bet.get("edge_pct", 0)
        edge_str = f"{edge:+.1%}"
        outcome = bet.get("outcome", "?")
        outcome_str = f"{GREEN}{outcome}{RESET}" if outcome == "WIN" else f"{RED}{outcome}{RESET}"

        print(f"  {bet.get('game_date', '?'):<12s} {matchup:<30s} "
              f"{bet.get('predicted_total', 0):>6.1f} "
              f"{bet.get('market_line', 0):>6.1f} "
              f"{bet.get('actual_total', 0):>6.1f} "
              f"{edge_str:>7s} {outcome_str:>6s}")

    print(f"  {'-'*75}")


def print_monthly_summary(result):
    """Print monthly profit breakdown."""
    if result.bets_df.empty:
        return

    df = result.bets_df.copy()
    df["month"] = pd.to_datetime(df["game_date"]).dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        bets=("profit_units", "count"),
        profit=("profit_units", "sum"),
        wins=("outcome", lambda x: (x == "WIN").sum()),
    ).reset_index()
    monthly["win_rate"] = monthly["wins"] / monthly["bets"]

    print(f"\n  {BOLD}Monthly Performance{RESET}")
    print(f"  {'-'*57}")
    print(f"  {'Month':<10s} {'Bets':>5s} {'Wins':>5s} {'Win%':>7s} {'Profit':>10s} {'Cumulative':>10s}")
    print(f"  {'-'*57}")

    cumulative = 0
    for _, row in monthly.iterrows():
        cumulative += row["profit"]
        profit_str = f"{GREEN}{row['profit']:+.1f}u{RESET}" if row["profit"] > 0 else f"{RED}{row['profit']:+.1f}u{RESET}"
        cum_str = f"{GREEN}{cumulative:+.1f}u{RESET}" if cumulative > 0 else f"{RED}{cumulative:+.1f}u{RESET}"
        print(f"  {row['month']:<10s} {row['bets']:>5d} {row['wins']:>5d} "
              f"{row['win_rate']:>6.1%} {profit_str:>10s} {cum_str:>10s}")

    print(f"  {'-'*57}")


def print_execution_info(total_time, params):
    """Print execution metadata."""
    print(f"\n  {BOLD}Run Configuration{RESET}")
    print(f"  {'-'*40}")
    for k, v in params.items():
        print(f"  {k:<18s}: {v}")
    print(f"  {'Elapsed':<18s}: {total_time:.1f}s")
    print(f"  {'-'*40}")


# ============================================================================
#  Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Backtest - Momentum Model Validation on Real NBA Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/run_backtest.py                          # Quick run
  python tools/run_backtest.py --full                   # All models
  python tools/run_backtest.py --model ridge            # Just Ridge
  python tools/run_backtest.py --model momentum         # Just MomentumModel
  python tools/run_backtest.py --window 150 --step 15   # Custom window params
  python tools/run_backtest.py --no-bets                # Model metrics only
        """
    )
    parser.add_argument("--full", action="store_true", help="Run all models (default: Ridge + Momentum)")
    parser.add_argument("--model", type=str, default=None,
                        choices=["ridge", "momentum", "xgboost"],
                        help="Run a single model by name")
    parser.add_argument("--window", type=int, default=WALK_FORWARD_WINDOW,
                        help=f"Walk-forward train window (default: {WALK_FORWARD_WINDOW})")
    parser.add_argument("--step", type=int, default=WALK_FORWARD_STEP,
                        help=f"Walk-forward step size (default: {WALK_FORWARD_STEP})")
    parser.add_argument("--no-bets", action="store_true",
                        help="Skip betting simulation, show model metrics only")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results CSV to this path")
    parser.add_argument("--calibrated", action="store_true",
                        help="Apply Platt scaling to MomentumModel probability estimates")
    args = parser.parse_args()

    start_time = datetime.now()

    # -- Header --
    print()
    print(CYAN + BOLD + ("=" * 65) + RESET)
    print(CYAN + BOLD + "  WALK-FORWARD BACKTEST - MOMENTUM MODEL VALIDATION" + RESET)
    print(CYAN + BOLD + ("  " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))) + RESET)
    print(CYAN + BOLD + ("=" * 65) + RESET)

    # -- Load Data --
    print(header("Phase 1: Data Loading & Feature Engineering"))
    df, feature_cols = load_and_prepare_data()

    # -- Configure Models --
    print(header("Phase 2: Walk-Forward Backtesting"))

    all_configs = build_model_configs(calibrated=args.calibrated)

    # Filter models
    if args.model:
        model_map = {
            "ridge": "Total Points (Ridge)",
            "momentum": "Momentum (LogisticRegression)",
            "xgboost": "Total Points (GBM)",
        }
        selected = model_map.get(args.model)
        if selected and selected in all_configs:
            all_configs = {selected: all_configs[selected]}
        else:
            print(f"  {YELLOW}[!] Unknown model '{args.model}', running Ridge only{RESET}")
            all_configs = {"Total Points (Ridge)": all_configs["Total Points (Ridge)"]}
    elif not args.full:
        # Default: Ridge + Momentum (and calibrated if --calibrated)
        default_keys = ["Total Points (Ridge)", "Momentum (LogisticRegression)"]
        if args.calibrated:
            default_keys.append("Momentum (Platt Calibrated)")
        all_configs = {
            k: v for k, v in all_configs.items()
            if k in default_keys
        }

    # -- Walk-Forward Engine --
    engine = WalkForwardEngine(
        train_window=args.window,
        step=args.step,
        min_train=MIN_TRAIN_SAMPLES,
    )

    params = {
        "Train Window": f"{args.window} games",
        "Step Size": f"{args.step} games",
        "Min Train": f"{MIN_TRAIN_SAMPLES} games",
        "Min Edge": f"{MIN_EDGE_THRESHOLD:.0%}",
        "Models": ", ".join(all_configs.keys()),
        "Make Bets": str(not args.no_bets),
    }
    print(f"  Running {len(all_configs)} model(s)...")

    # -- Run All Backtests --
    results = {}
    for model_name, config in all_configs.items():
        print(f"\n  {CYAN}>> {model_name}{RESET}")
        result = run_backtest(
            df=df,
            feature_cols=feature_cols,
            model_name=model_name,
            model_config=config,
            make_bets=not args.no_bets,
            engine=engine,
        )
        results[model_name] = result
        print_model_summary(model_name, result)

    # -- Detailed Metrics --
    if not args.no_bets:
        print(header("Phase 3: Detailed Performance Analysis"))
        for name, result in results.items():
            if result.total_bets > 0:
                print_detailed_metrics(result)

        # -- Sample Bets --
        print(header("Phase 4: Bet-Level Inspection"))
        for name, result in results.items():
            if result.total_bets > 0:
                print(f"\n  {BOLD}{name}{RESET}")
                print_sample_bets(result, n=6)

        # -- Monthly Breakdown --
        print(header("Phase 5: Monthly Performance"))
        for name, result in results.items():
            if result.total_bets > 0:
                print(f"\n  {BOLD}{name}{RESET}")
                print_monthly_summary(result)

    # -- Model Comparison --
    if len(results) > 1:
        print(header("Phase 6: Model Comparison"))
        print_comparison_table(results)

    # -- Save Output --
    if args.output and not args.no_bets:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = []
        for name, result in results.items():
            if not result.bets_df.empty:
                bet_df = result.bets_df.copy()
                bet_df["model"] = name
                combined.append(bet_df)
        if combined:
            full_df = pd.concat(combined, ignore_index=True)
            full_df.to_csv(output_path, index=False)
            print(f"\n  {ok('Saved', str(output_path))}")

    # -- Execution Info --
    elapsed = (datetime.now() - start_time).total_seconds()
    print(header("Run Complete"))
    print_execution_info(elapsed, params)

    # -- Summary Verdict --
    print()
    print(f"  {BOLD}SUMMARY VERDICT{RESET}")
    print(f"  {'-'*50}")

    for name, result in results.items():
        if result.total_bets > 0:
            wr = result.win_rate
            total_decided = result.wins + result.losses
            pval = float(stats.binomtest(result.wins, total_decided, p=0.5).pvalue)
            be = 52.38 / 100

            if wr > be and pval < 0.05:
                verdict = f"{GREEN}BEATING THE MARKET - significant edge detected{RESET}"
            elif wr > 0.50 and pval < 0.10:
                verdict = f"{YELLOW}Encouraging but not conclusive - keep testing{RESET}"
            elif wr > 0.50:
                verdict = f"{YELLOW}Above 50% but not statistically significant{RESET}"
            else:
                verdict = f"{RED}No edge detected - model is not predictive{RESET}"

            print(f"\n  {BOLD}{name}{RESET}")
            print(f"  {verdict}")
            print(f"  Win Rate: {wr:.1%}  (n={result.total_bets}, p={pval:.4f})")

    print()
    print(CYAN + BOLD + ("=" * 65) + RESET)
    print(f"  Done. Run 'python tools/run_backtest.py --full' for all models." + RESET)
    print(CYAN + BOLD + ("=" * 65) + RESET)
    print()


if __name__ == "__main__":
    main()
