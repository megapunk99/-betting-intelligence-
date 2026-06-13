#!/usr/bin/env python3
"""
Retrain the MarketInefficiencySystem using real market odds from MarketOddsStore.

Usage:
    python tools/retrain_model.py
"""
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Force UTF-8 for stdout on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain")


def hr(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> int:
    # Step 1: Check MarketOddsStore stats
    hr("Market Odds Store Stats")
    from betting_intel.db.market_odds_store import MarketOddsStore
    store = MarketOddsStore()
    stats = store.get_stats()
    print(f"  Total snapshots:  {stats['total_snapshots']}")
    print(f"  Unique games:     {stats['unique_games']}")

    from betting_intel.db.schema import MarketOdds
    session = store._db.get_session()
    try:
        first = session.query(MarketOdds.game_date).order_by(MarketOdds.game_date.asc()).first()
        last = session.query(MarketOdds.game_date).order_by(MarketOdds.game_date.desc()).first()
        if first and last:
            print(f"  Date range:       {first[0]} to {last[0]}")
        from sqlalchemy import func
        source_counts = (
            session.query(MarketOdds.source, func.count(MarketOdds.id))
            .group_by(MarketOdds.source)
            .all()
        )
        if source_counts:
            print(f"  Source breakdown:")
            for source, count in source_counts:
                print(f"    {source:30s}  {count}")
    finally:
        session.close()

    # Step 2: Clear engine cache to force rebuild
    hr("Clearing Engine Cache")
    from betting_intel.live.engine import LivePredictionEngine
    engine = LivePredictionEngine()
    engine.clear_cache()
    print("  [OK] Engine cache cleared")

    # Step 3: Force refresh (triggers _build_robust_system)
    hr("Training MarketInefficiencySystem")
    print("  Refreshing engine (this will build the model from scratch)...")
    sys.stdout.flush()

    try:
        snapshot = engine.refresh_now()
    except Exception as e:
        logger.error(f"Engine refresh failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Report training results
    summary = engine.robust_system_summary
    print()
    hr("Training Results")

    if summary.get("fitted"):
        print(f"  Status:              FITTED")
        print(f"  Classifier models:   {summary.get('n_models', '?')}")
        print(f"  Error regressors:    {summary.get('n_error_models', 0)}")
        print(f"  Training samples:    {summary.get('n_train_samples', '?')}")
        print(f"  Features:            {summary.get('n_features', '?')}")
        print(f"  Target mean:         {summary.get('target_mean', '?')}")
        print(f"  Brier (raw):         {summary.get('brier_score', 'N/A')}")
        print(f"  Brier (calibrated):  {summary.get('calibrated_brier', 'N/A')}")
        print(f"  Market error std:    {summary.get('error_std', 'N/A')}")

        if summary.get("model_weights"):
            print(f"\n  Model weights:")
            for name, weight in sorted(summary["model_weights"].items(), key=lambda x: x[1], reverse=True):
                print(f"    {name:25s}  {weight:.4f}")

        if summary.get("model_diagnostics"):
            print(f"\n  Model diagnostics:")
            for name, diag in summary["model_diagnostics"].items():
                print(f"    {name:25s}  Brier={diag.get('oos_brier', '?'):.4f}  Acc={diag.get('oos_accuracy', '?'):.2%}  n={diag.get('n_oos', '?')}  status={diag.get('status', '?')}")

        if summary.get("overfitting"):
            of = summary["overfitting"]
            print(f"\n  Overfitting check:")
            print(f"    Is overfit:      {of.get('is_overfit', '?')}")
            print(f"    Train R2:        {of.get('avg_train_r2', '?')}")
            print(f"    Test R2:         {of.get('avg_test_r2', '?')}")
            print(f"    R2 gap:          {of.get('r2_gap', '?')}")

        print(f"\n  Market error trained: {summary.get('market_error_trained', False)}")
        if summary.get("error_weights"):
            print(f"  Error regressor weights:")
            for name, weight in summary["error_weights"].items():
                print(f"    {name:25s}  {weight:.4f}")
    else:
        print(f"  Status: FAILED")
        print(f"  Reason: {summary.get('status', 'unknown')}")

    # Step 5: Update running server engine
    hr("Updating Server Engine")
    try:
        import web.app as webapp
        server_engine = webapp.get_live_engine()
        server_engine.clear_cache()
        print("  [OK] Server engine cache cleared")
        try:
            server_snapshot = server_engine.refresh_now()
            print(f"  [OK] Server refreshed: {server_snapshot.n_total} games in window")
        except Exception as e:
            print(f"  [..] Server refresh skipped (will rebuild on next user refresh): {e}")
    except Exception:
        print("  [..] Web server not running -- standalone retrain only")

    # Step 6: Summary
    hr("Next Steps")
    print(f"  Model trained with {stats['unique_games']} games of real odds data.")
    print(f"  Each engine refresh accumulates more real odds in MarketOddsStore.")
    print(f"  Re-run this script after several refresh cycles to improve coverage.")
    print(f"  For large backfill, get a valid API key at: https://the-odds-api.com/")
    print(f"  Then: python tools/backfill_market_odds.py --mode scores")

    return 0


if __name__ == "__main__":
    sys.exit(main())
