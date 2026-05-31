#!/usr/bin/env python3
"""
Deep analysis of the Momentum Model.

Shows:
  1. TOP FEATURES — which stats drive the model (positive & negative)
  2. CONFIDENCE BUCKETS — performance at different confidence levels
  3. CALIBRATION — predicted probability vs actual win rate
  4. HIGH-CONFIDENCE BETS — best picks the model has made
  5. MODEL STABILITY — how features vary across walk-forward windows

Usage:
    python tools/analyze_momentum.py                         # Full analysis
    python tools/analyze_momentum.py --top-features 30       # Show 30 features
    python tools/analyze_momentum.py --output analysis.csv   # Save results
"""

import sys
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import MomentumModel
from betting_intel.backtesting.engine import WalkForwardEngine
from betting_intel.config import WALK_FORWARD_WINDOW, WALK_FORWARD_STEP

# -- ANSI Colors --
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ============================================================================
#  Phase 1: Load Data
# ============================================================================


def load_and_prepare_data():
    """Load historical NBA data and engineer features."""
    print(CYAN + BOLD + "\n  " + ("=" * 75) + RESET)
    print(CYAN + BOLD + "  MOMENTUM MODEL - DEEP ANALYSIS" + RESET)
    print(CYAN + BOLD + "  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + RESET)
    print(CYAN + BOLD + "  " + ("=" * 75) + RESET)

    print("\n  Loading NBA data...")
    loader = NBADataLoader()
    fe = FeatureEngineer()

    raw_df = loader.load_game_logs()
    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)

    print(f"    Games loaded: {len(games_df):,}")
    print(f"    Date range:   {games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}")

    print("  Engineering features...")
    feature_df = fe.build_all_features(games_df, raw_df)
    feature_cols = fe.select_features(feature_df)
    print(f"    Features: {len(feature_cols)}")

    clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
    clean_df = clean_df.reset_index(drop=True)
    clean_df["home_win"] = (clean_df["point_diff"] > 0).astype(int)
    print(f"    Clean samples: {len(clean_df):,}\n")

    return clean_df, feature_cols


# ============================================================================
#  Phase 2: Feature Importance (Full Dataset)
# ============================================================================


def classify_feature_theme(feature_name: str) -> str:
    """Classify a feature into an interpretable theme."""
    name = feature_name.lower()

    themes = [
        ("momentum/recent form", any(k in name for k in ["streak", "momentum", "weighted", "form_score", "win_streak", "last_3_margin"])),
        ("rest/fatigue", any(k in name for k in ["rest", "fatigue", "b2b", "3in4"])),
        ("travel/schedule", any(k in name for k in ["travel", "tz_diff", "consec_road", "road_trip", "cum_trav"])),
        ("pace/speed", any(k in name for k in ["pace"])),
        ("scoring avg", any(k in name for k in ["avg_pts", "ema_pts", "pts_zscore"])),
        ("scoring efficiency", any(k in name for k in ["efg", "ts_pct", "three_pt_rate", "ft_rate"])),
        ("rebounding", any(k in name for k in ["reb_pct", "oreb", "dreb"])),
        ("margin/dominance", any(k in name for k in ["pm_", "margin", "avg_pm", "ema_pm"])),
        ("opponent quality", any(k in name for k in ["sos_", "opp_avg", "opp_trailing", "offense_vs_defense", "defense_vs_offense"])),
        ("consistency", any(k in name for k in ["volatility", "zscore"])),
        ("trend", any(k in name for k in ["trend_", "ema_"])),
        ("misc/interaction", any(k in name for k in ["interact", "diff", "sq", "adv"])),
    ]
    for theme, matched in themes:
        if matched:
            return theme
    return "other"


def analyze_feature_importance(df, feature_cols, top_n=20, calibrated=False):
    """Train on full dataset and extract feature importances."""
    print(CYAN + BOLD + "  " + ("=" * 75) + RESET)
    print(f"  FEATURE IMPORTANCE - Top {top_n}" + RESET)
    print("  " + ("=" * 75))

    X = df[feature_cols].dropna()
    y = df.loc[X.index, "home_win"]
    print(f"    Training on {len(X):,} samples with {len(feature_cols)} features\n")

    model = MomentumModel("logistic", calibrate=calibrated)
    model.fit(X.values, y.values)

    coefs = model.feature_importance
    if coefs is None:
        coefs = model.model.coef_[0]

    # Sort by absolute magnitude
    feature_map = list(zip(feature_cols, coefs))
    feature_map.sort(key=lambda x: abs(x[1]), reverse=True)

    # Top features
    print(f"  {BOLD}Top Drivers (by absolute coefficient){RESET}")
    print(f"  {'#':<4s} {'Feature':<42s} {'Coef':>9s} {'Direction':>10s}  {'Theme'}")
    print(f"  {'-'*4:<4s} {'-'*42:<42s} {'-'*9:>9s} {'-'*10:>10s}  {'-'*30}")

    for i, (name, coef) in enumerate(feature_map[:top_n], 1):
        direction = GREEN + "HOME WIN" + RESET if coef > 0 else RED + "AWAY WIN" + RESET
        theme = classify_feature_theme(name)
        print(f"  {i:<4d} {name:<42s} {coef:>+9.4f} {direction:>10s}  {theme}")

    # Theme aggregation
    print(f"\n  {BOLD}Feature Groups (aggregated importance){RESET}")
    theme_importance = defaultdict(float)
    theme_count = defaultdict(int)
    for name, coef in feature_map:
        theme = classify_feature_theme(name)
        theme_importance[theme] += abs(coef)
        theme_count[theme] += 1

    total_importance = sum(theme_importance.values())
    print(f"  {'Theme':<30s} {'Features':>8s} {'Importance':>10s} {'Share':>8s}")
    print(f"  {'-'*30:<30s} {'-'*8:>8s} {'-'*10:>10s} {'-'*8:>8s}")
    for theme, importance in sorted(theme_importance.items(), key=lambda x: x[1], reverse=True):
        share = importance / total_importance if total_importance > 0 else 0
        print(f"  {theme:<30s} {theme_count[theme]:>8d} {importance:>10.4f} {share:>7.1%}")

    # Top positive and negative
    positive = [(n, c) for n, c in feature_map if c > 0][:10]
    negative = [(n, c) for n, c in feature_map if c < 0][-10:][::-1]

    print(f"\n  {BOLD}Top Positive (favor HOME win){RESET}")
    for name, coef in positive:
        print(f"    {GREEN}+{RESET} {name:<45s} {coef:>+8.4f}")

    print(f"\n  {BOLD}Top Negative (favor AWAY win){RESET}")
    for name, coef in negative:
        print(f"    {RED} {RESET} {name:<45s} {coef:>+8.4f}")

    return model, feature_map


# ============================================================================
#  Phase 3: Walk-Forward with Confidence Bucketing
# ============================================================================


def select_momentum_features(feature_cols):
    """Select momentum/reversion-related features."""
    momentum_kw = [
        "streak", "momentum", "win_pct", "margin_volatility",
        "form_", "weighted_", "rest_", "fatigue", "travel",
        "net_rating", "sos_", "avg_pm_", "avg_pts_",
        "avg_ts_", "avg_efg_", "tz_", "pace_",
    ]
    selected = [c for c in feature_cols if any(k in c for k in momentum_kw)]
    return selected if len(selected) >= 10 else feature_cols


def run_walk_forward_with_probs(df, feature_cols, window=200, step=20, calibrated=False):
    """Run walk-forward validation capturing prediction probabilities."""
    print(CYAN + BOLD + "\n  " + ("=" * 75) + RESET)
    print(f"  WALK-FORWARD - Confidence Analysis" + RESET)
    print("  " + ("=" * 75))
    print(f"    Window: {window} games  Step: {step} games")

    use_features = select_momentum_features(feature_cols)
    print(f"    Features: {len(use_features)} momentum-related")
    if calibrated:
        print(f"    {GREEN}Platt scaling enabled{RESET}")

    engine = WalkForwardEngine(train_window=window, step=step, min_train=50)
    result = engine.run_walk_forward(
        df=df,
        feature_cols=use_features,
        target_col="home_win",
        model_builder=lambda: MomentumModel("logistic", calibrate=calibrated),
        strategy_name="momentum_analysis",
        model_name="Momentum (Analysis)",
        prediction_type="classification",
        make_bets=True,
    )

    if result.bets_df.empty:
        print(f"  {RED}No bets generated.{RESET}")
        return result, use_features

    print(f"    Total bets: {result.total_bets}")
    print(f"    Win rate:   {result.win_rate:.1%}")
    print(f"    Profit:     {result.total_profit:+.1f}u")
    print(f"    Sharpe:     {result.sharpe_ratio:.2f}")

    return result, use_features


def analyze_confidence_buckets(bets_df):
    """Group bets by home_win_prob and show performance per bucket."""
    if bets_df.empty:
        return

    if "home_win_prob" not in bets_df.columns:
        print(f"\n  {YELLOW}[!] No home_win_prob column found. Columns: {list(bets_df.columns)}{RESET}")
        return

    probs = bets_df["home_win_prob"].values
    outcomes = bets_df["outcome"].map({"WIN": 1, "LOSS": 0, "PUSH": None}).values

    print(f"\n  {BOLD}Confidence Buckets{RESET}")
    print(f"  Predicted home win probability vs actual results (flat -110 juice)")
    print(f"  " + ("-" * 80))
    print(f"  {'Bucket':<12s} {'Bets':>6s} {'Wins':>5s} {'Losses':>7s} {'Win%':>8s} {'Profit':>10s} {'ROI':>8s} {'Avg Prob':>10s}")
    print(f"  " + ("-" * 80))

    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.0)]
    bucket_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

    for (lo, hi), label in zip(buckets, bucket_labels):
        mask = (probs >= lo) & (probs < hi)
        n_bets = int(mask.sum())
        if n_bets == 0:
            continue
        wins = int((outcomes[mask] == 1).sum())
        losses = int((outcomes[mask] == 0).sum())
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        avg_prob = float(probs[mask].mean())

        profit = wins * 1.0 - losses * 1.0
        roi = profit / n_bets if n_bets > 0 else 0

        color = GREEN if win_rate > 0.524 else (YELLOW if win_rate > 0.50 else RED)
        print(f"  {label:<12s} {n_bets:>6d} {wins:>5d} {losses:>7d} {color}{win_rate:>7.1%}{RESET} {profit:>+9.1f}u {roi:>+7.1%} {avg_prob:>9.1%}")

    # High-confidence cumulative buckets
    print(f"\n  {BOLD}High-Confidence Cumulative{RESET}")
    print(f"  {'Threshold':<12s} {'Bets':>6s} {'Wins':>5s} {'Losses':>7s} {'Win%':>8s} {'Profit':>10s} {'ROI':>8s}")
    print(f"  " + ("-" * 60))

    for threshold, label in [(0.60, ">=60%"), (0.65, ">=65%"), (0.70, ">=70%")]:
        mask = probs >= threshold
        n = int(mask.sum())
        if n == 0:
            continue
        w = int((outcomes[mask] == 1).sum())
        l = int((outcomes[mask] == 0).sum())
        wr = w / (w + l) if (w + l) > 0 else 0
        p = w * 1.0 - l * 1.0
        roi = p / n if n > 0 else 0
        color = GREEN if wr > 0.524 else (YELLOW if wr > 0.50 else RED)
        print(f"  {label:<12s} {n:>6d} {w:>5d} {l:>7d} {color}{wr:>7.1%}{RESET} {p:>+9.1f}u {roi:>+7.1%}")


def analyze_calibration(bets_df):
    """Calibration curve: predicted probability vs actual win rate."""
    if bets_df.empty or "home_win_prob" not in bets_df.columns:
        return

    probs = bets_df["home_win_prob"].values
    outcomes = bets_df["outcome"].map({"WIN": 1, "LOSS": 0, "PUSH": None}).values

    print(f"\n  {BOLD}Calibration{RESET}")
    print(f"  How well do predicted probabilities match actual outcomes?")
    print(f"  " + ("-" * 60))
    print(f"  {'Prob Range':<12s} {'Bets':>6s} {'Actual Win%':>12s} {'Expected':>10s} {'Delta':>8s}")
    print(f"  " + ("-" * 60))

    boundaries = np.linspace(0.50, 1.0, 11)
    total_delta = 0
    n_buckets = 0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n < 10:
            continue
        actual = float(outcomes[mask].mean())
        expected = float(probs[mask].mean())
        delta = actual - expected
        total_delta += abs(delta)
        n_buckets += 1

        color = GREEN if abs(delta) < 0.02 else (YELLOW if abs(delta) < 0.05 else RED)
        print(f"  {lo:.0%}-{hi:.0%}:{'':<6s} {n:>6d} {color}{actual:>10.1%}{RESET}  {expected:>9.1%} {delta:>+7.1%}")

    if n_buckets > 0:
        mce = total_delta / n_buckets
        print(f"\n  Mean Calibration Error: {mce:.2%}")
        if mce < 0.02:
            print(f"  {GREEN}Well-calibrated within {mce:.1%}{RESET}")
        elif mce < 0.05:
            print(f"  {YELLOW}Moderately calibrated ({mce:.1%} avg error){RESET}")
        else:
            print(f"  {RED}Poorly calibrated ({mce:.1%} avg error) - consider Platt scaling{RESET}")


def analyze_high_confidence_bets(bets_df, n=15):
    """List the top high-confidence bets with details."""
    if bets_df.empty or "home_win_prob" not in bets_df.columns:
        return

    df = bets_df.copy()
    df = df.sort_values("home_win_prob", ascending=False).head(n)

    print(f"\n  {BOLD}Top {n} High-Confidence Bets{RESET}")
    print(f"  " + ("-" * 85))
    print(f"  {'Date':<12s} {'Matchup':<30s} {'Home Prob':>10s} {'Result':>8s} {'Profit':>8s}")
    print(f"  " + ("-" * 85))

    for _, bet in df.iterrows():
        matchup = str(bet.get("matchup", ""))[:28]
        prob = bet.get("home_win_prob", 0.5)
        outcome = bet.get("outcome", "?")
        profit = bet.get("profit_units", 0)

        oc = GREEN + "WIN" + RESET if outcome == "WIN" else (RED + "LOSS" + RESET if outcome == "LOSS" else outcome)
        pc = GREEN if profit > 0 else RED
        print(f"  {bet.get('game_date', '?'):<12s} {matchup:<30s} {prob:>9.1%}  {oc:>8s} {pc}{profit:>+7.1f}u{RESET}")


def analyze_bet_type_performance(bets_df):
    """Analyze performance when model favors home vs away."""
    if bets_df.empty or "home_win_prob" not in bets_df.columns:
        return

    df = bets_df.copy()
    outcomes = df["outcome"].map({"WIN": 1, "LOSS": 0, "PUSH": None})

    print(f"\n  {BOLD}Bet Type Analysis{RESET}")
    print(f"  " + ("-" * 60))

    # Home strong favorite (prob >= 0.65)
    home_fav = df[df["home_win_prob"] >= 0.65]
    if len(home_fav) > 0:
        wr = outcomes[home_fav.index].mean()
        print(f"  Home strong favorite (p>=65%):   {len(home_fav):>4d} bets, {wr:.1%} win rate")

    # Home slight favorite (0.55 - 0.65)
    home_slight = df[(df["home_win_prob"] >= 0.55) & (df["home_win_prob"] < 0.65)]
    if len(home_slight) > 0:
        wr = outcomes[home_slight.index].mean()
        print(f"  Home slight favorite (55-65%):  {len(home_slight):>4d} bets, {wr:.1%} win rate")

    # Near coin flip (0.50 - 0.55)
    toss_up = df[(df["home_win_prob"] >= 0.50) & (df["home_win_prob"] < 0.55)]
    if len(toss_up) > 0:
        wr = outcomes[toss_up.index].mean()
        print(f"  Near coin flip (50-55%):         {len(toss_up):>4d} bets, {wr:.1%} win rate")

    # Model picks away (p < 0.50)
    away_pick = df[df["home_win_prob"] < 0.50]
    if len(away_pick) > 0:
        wr = outcomes[away_pick.index].mean()
        print(f"  Model picks away (p<50%):        {len(away_pick):>4d} bets, {wr:.1%} win rate")


# ============================================================================
#  Phase 4: Model Stability Across Slices
# ============================================================================


def analyze_model_stability(df, feature_cols, use_features, calibrated=False):
    """Train on different data slices and compare coefficient rankings."""
    print(CYAN + BOLD + "\n  " + ("=" * 75) + RESET)
    print(f"  MODEL STABILITY - Coefficient Consistency" + RESET)
    print("  " + ("=" * 75))

    X = df[use_features].dropna().values
    y = df.loc[df[use_features].dropna().index, "home_win"].values
    total = len(X)

    splits = [
        ("First 50%", X[:total // 2], y[:total // 2]),
        ("First 75%", X[:3 * total // 4], y[:3 * total // 4]),
        ("Full", X, y),
        ("Last 75%", X[total // 4:], y[total // 4:]),
        ("Last 50%", X[total // 2:], y[total // 2:]),
    ]

    all_coefs = {}
    for label, X_sub, y_sub in splits:
        if len(X_sub) < 50:
            continue
        m = MomentumModel("logistic", calibrate=calibrated)
        m.fit(X_sub, y_sub)
        all_coefs[label] = m.feature_importance

    # Correlation between coefficient vectors
    print(f"\n  {BOLD}Coefficient Stability (Pearson correlation){RESET}")
    print(f"  " + ("-" * 50))
    labels = list(all_coefs.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            c = np.corrcoef(all_coefs[labels[i]], all_coefs[labels[j]])[0, 1]
            color = GREEN if abs(c) > 0.95 else (YELLOW if abs(c) > 0.85 else RED)
            print(f"  {labels[i]:<18s} vs {labels[j]:<18s}  {color}r = {c:.3f}{RESET}")

    # Top features per slice
    print(f"\n  {BOLD}Top 5 Features Per Slice{RESET}")
    for label, coefs in all_coefs.items():
        feat_imp = sorted(zip(use_features, coefs), key=lambda x: abs(x[1]), reverse=True)
        top5 = [f"{name}={coef:+.3f}" for name, coef in feat_imp[:5]]
        print(f"  {label:<18s}: {', '.join(top5)}")


# ============================================================================
#  Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Deep analysis of the Momentum Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top-features", type=int, default=20, help="Number of top features to show (default: 20)")
    parser.add_argument("--output", type=str, default=None, help="Save bets to CSV")
    parser.add_argument("--window", type=int, default=WALK_FORWARD_WINDOW, help=f"Walk-forward window (default: {WALK_FORWARD_WINDOW})")
    parser.add_argument("--step", type=int, default=WALK_FORWARD_STEP, help=f"Walk-forward step (default: {WALK_FORWARD_STEP})")
    parser.add_argument("--calibrated", action="store_true",
                        help="Apply Platt scaling to probability estimates")
    args = parser.parse_args()

    # Phase 1: Load Data
    df, feature_cols = load_and_prepare_data()

    # Phase 2: Feature Importance
    model, feature_map = analyze_feature_importance(df, feature_cols, top_n=args.top_features, calibrated=args.calibrated)

    # Phase 3: Walk-forward with Confidence
    result, use_features = run_walk_forward_with_probs(df, feature_cols, window=args.window, step=args.step, calibrated=args.calibrated)

    if result.bets_df.empty:
        print(f"\n  {RED}Could not run confidence analysis - no bets generated.{RESET}")
        print(f"  Try a larger walk-forward window (--window {args.window + 50}).\n")
        return 1

    # Quick sanity check on probability data
    if "home_win_prob" in result.bets_df.columns:
        prob_range = (result.bets_df["home_win_prob"].min(), result.bets_df["home_win_prob"].max())
        print(f"    Probability range: {prob_range[0]:.1%} to {prob_range[1]:.1%} (mean: {result.bets_df['home_win_prob'].mean():.1%})")
    else:
        print(f"  {YELLOW}  [!] home_win_prob column NOT found in bets_df - check engine.py{RESET}")

    # Confidence bucketing
    analyze_confidence_buckets(result.bets_df)

    # Calibration
    analyze_calibration(result.bets_df)

    # High-confidence bets
    analyze_high_confidence_bets(result.bets_df, n=15)

    # Bet type analysis
    analyze_bet_type_performance(result.bets_df)

    # Model stability
    analyze_model_stability(df, feature_cols, use_features, calibrated=args.calibrated)

    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        result.bets_df.to_csv(output_path, index=False)
        print(f"\n  {GREEN}Saved bets to {output_path}{RESET}")

    print(f"\n  {GREEN}{BOLD}Done.{RESET}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
