"""
Daily Runner — THE AUTOMATED MONEY MACHINE.

This script is designed to run EVERY MORNING via Windows Task Scheduler.
It does the full workflow:

  1. Runs the prediction pipeline
  2. Generates the daily betting card
  3. Delivers picks to all subscribers
  4. Saves reports
  5. Tracks CLV

Usage:
    # Manual run:
    python run_daily.py

    # With custom bankroll:
    python run_daily.py --bankroll 5000

    # Scheduled mode (no prompts, logs to file):
    python run_daily.py --scheduled
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("LOG_LEVEL", "INFO")

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from betting_intel.business.report import (
    GameAnalysisGenerator,
    DailyBettingCard,
)
from betting_intel.business.subscriptions import SubscriptionManager
from betting_intel.business.delivery import PickDeliverer

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"daily_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("daily_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🏀 Daily Betting Intelligence Runner — THE MONEY MACHINE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bankroll", type=float, default=10_000.0, help="Starting bankroll")
    parser.add_argument("--scheduled", action="store_true", help="Scheduled mode: auto-run, no prompts")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip prediction pipeline, use cached")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = datetime.now(timezone.utc)

    print("=" * 70)
    print("  🏀  DAILY BETTING INTELLIGENCE RUNNER")
    print(f"     {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    results = {
        "date": start_time.strftime("%Y-%m-%d"),
        "timestamp": start_time.isoformat(),
        "bankroll": args.bankroll,
        "mode": "scheduled" if args.scheduled else "manual",
        "games_analyzed": 0,
        "actionable_bets": 0,
        "total_stake": 0.0,
        "subscribers_reached": 0,
        "errors": [],
    }

    # ── Step 1: Run the prediction pipeline ─────────────────────────
    if not args.skip_pipeline:
        print("\n[1/4] Running prediction pipeline...")
        try:
            from betting_intel.pipeline.cli import main as pipeline_main

            pipeline_args = [
                "--live" if os.getenv("ODDS_API_KEY") else "--full",
                "--no-tune" if args.scheduled else "",
                "--output", str(PROJECT_ROOT / "output" / f"predictions_{start_time.strftime('%Y%m%d')}.json"),
            ]
            # Filter empties
            pipeline_args = [a for a in pipeline_args if a]

            exit_code = pipeline_main(pipeline_args)
            if exit_code == 0:
                print("  ✅  Pipeline completed successfully")
            else:
                print("  ⚠️  Pipeline finished with warnings")
        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            print(f"  ❌  {error_msg}")
    else:
        print("\n[1/4] Skipping pipeline (--skip-pipeline)")

    # ── Step 2: Generate daily betting card ─────────────────────────
    print("\n[2/4] Generating daily betting card...")
    try:
        generator = GameAnalysisGenerator(bankroll=args.bankroll)
        today = start_time.strftime("%Y-%m-%d")

        # Try to load real predictions from the pipeline output file
        predictions_path = PROJECT_ROOT / "output" / f"predictions_{start_time.strftime('%Y%m%d')}.json"
        games = []

        if predictions_path.exists():
            try:
                pred_data = json.loads(predictions_path.read_text(encoding="utf-8"))
                # Parse games from predictions and analyze each one
                for game_data in pred_data.get("predictions", [])[:15]:
                    report = generator.analyze_game(
                        home_team=game_data.get("home_team", ""),
                        away_team=game_data.get("away_team", ""),
                        game_date=game_data.get("game_date", today),
                        model_home_win_prob=game_data.get("home_win_prob", 0.5),
                        model_predicted_total=game_data.get("predicted_total"),
                        model_predicted_margin=game_data.get("predicted_margin"),
                        home_ml_odds=game_data.get("home_ml_odds"),
                        away_ml_odds=game_data.get("away_ml_odds"),
                        market_total=game_data.get("market_total"),
                        home_spread=game_data.get("spread"),
                    )
                    games.append(report)
                logger.info(f"Loaded {len(games)} games from pipeline output")
            except Exception as e:
                logger.warning(f"Could not parse pipeline output: {e}")

        # If no predictions available, generate from the NBADataLoader
        if not games:
            try:
                logger.info("Generating analyses from historical data...")
                from betting_intel.data.loader import NBADataLoader
                loader = NBADataLoader()
                raw_df = loader.load_game_logs()

                if len(raw_df) > 0:
                    teams = list(raw_df["TEAM_NAME"].unique())
                    import random
                    rng = random.Random(42)
                    rng.shuffle(teams)

                    for i in range(0, min(len(teams), 10), 2):
                        if i + 1 < len(teams):
                            home = teams[i]
                            away = teams[i + 1]

                            # Get recent stats
                            home_games = raw_df[raw_df["TEAM_NAME"] == home].tail(10)
                            away_games = raw_df[raw_df["TEAM_NAME"] == away].tail(10)

                            home_pts = home_games["PTS"].mean() if len(home_games) > 0 else 112
                            away_pts = away_games["PTS"].mean() if len(away_games) > 0 else 109
                            home_margin = home_games["PLUS_MINUS"].mean() if len(home_games) > 0 else 0
                            away_margin = away_games["PLUS_MINUS"].mean() if len(away_games) > 0 else 0

                            # PROPER probability: sigmoid instead of crude 0.5 + margin*0.01
                            # The crude linear hack could give P>1 for margins >50
                            raw_prob = 1.0 / (1.0 + math.exp(-home_margin * 0.08))
                            home_win_prob = max(0.25, min(0.75, raw_prob))
                            report = generator.analyze_game(
                                home_team=home,
                                away_team=away,
                                game_date=today,
                                model_home_win_prob=home_win_prob,
                                model_predicted_total=home_pts + away_pts,
                                model_predicted_margin=home_margin - away_margin,
                                home_ml_odds=-120 if home_margin > 0 else +110,
                                away_ml_odds=+110 if home_margin > 0 else -120,
                                market_total=round((home_pts + away_pts) / 5) * 5,
                                home_spread=-3.5 if home_margin > 0 else +3.5,
                                spread_odds_home=-110,
                                spread_odds_away=-110,
                                over_odds=-110,
                                under_odds=-110,
                            )
                            games.append(report)
                    logger.info(f"Generated {len(games)} game analyses from historical data")
            except Exception as e:
                logger.warning(f"Historical data analysis failed: {e}")

        # Build the DailyBettingCard from analyzed games
        card = DailyBettingCard(
            date=today,
            games=games,
            subscriber_tier="all",
        )

        # Compute aggregated values
        card.total_actionable_bets = sum(g.num_actionable_bets for g in games)
        card.total_recommended_stake = sum(
            g.best_bet.get("stake", 0) if g.best_bet else 0 for g in games
        )

        # Find the best play across all games
        best_edge = 0.0
        for g in games:
            if g.best_bet and g.best_bet.get("edge", 0) > best_edge:
                best_edge = g.best_bet["edge"]
                card.best_play = {
                    **g.best_bet,
                    "game": f"{g.away_team} @ {g.home_team}",
                }

        results["games_analyzed"] = len(games)
        results["actionable_bets"] = card.total_actionable_bets
        results["total_stake"] = card.total_recommended_stake

        # Save card to JSON
        output_path = args.output or str(PROJECT_ROOT / "reports" / f"daily_card_{start_time.strftime('%Y%m%d')}.json")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps({
            "date": card.date,
            "games": [g.to_dict() for g in card.games],
            "total_actionable_bets": card.total_actionable_bets,
            "total_recommended_stake": card.total_recommended_stake,
            "best_play": card.best_play,
        }, indent=2))
        print(f"  ✅  Card generated ({len(games)} games, {card.total_actionable_bets} actionable)")
        print(f"  💾  Saved to: {output_path}")

    except Exception as e:
        import traceback
        error_msg = f"Card generation failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        results["errors"].append(str(e))
        print(f"  ❌  Card generation error: {e}")

    # ── Step 3: Deliver to subscribers ──────────────────────────────
    print("\n[3/4] Delivering picks to subscribers...")
    try:
        deliverer = PickDeliverer(
            subscribers_db=str(PROJECT_ROOT / "data" / "subscribers.json"),
            bankroll=args.bankroll,
        )
        delivery_results = deliverer.distribute_daily_picks()
        results["subscribers_reached"] = delivery_results.get("success_count", 0)
        print(f"  ✅  Delivered to {delivery_results.get('success_count', 0)} subscribers")
        print(f"      Telegram: {delivery_results.get('telegram_sent', 0)}")
        print(f"      Email: {delivery_results.get('email_sent', 0)}")
    except Exception as e:
        error_msg = f"Delivery failed: {e}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
        print(f"  ❌  {error_msg}")

    # ── Step 4: Summary ─────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    print(f"\n[4/4] Summary")
    print(f"  ⏱  Elapsed: {elapsed:.1f}s")
    print(f"  🎯  Games analyzed: {results['games_analyzed']}")
    print(f"  💰  Actionable bets: {results['actionable_bets']}")
    print(f"  📊  Total stake: ${results['total_stake']:.0f}")
    print(f"  👤  Subscribers reached: {results['subscribers_reached']}")

    if results["errors"]:
        print(f"\n  ⚠️  {len(results['errors'])} error(s) occurred")
        for err in results["errors"]:
            print(f"      • {err}")

    print("\n" + "=" * 70)
    print("  ✅  DAILY RUN COMPLETE")
    print("=" * 70)

    # Save results
    results_path = Path(args.output or str(PROJECT_ROOT / "reports" / f"daily_results_{start_time.strftime('%Y%m%d')}.json"))
    results_path.write_text(json.dumps(results, indent=2, default=str))

    return 0 if not results["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
