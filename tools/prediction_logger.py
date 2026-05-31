#!/usr/bin/env python3
"""
Prediction Logger — tracks forward-test predictions vs actual game outcomes.

Saves every prediction as a JSONL entry with timestamp, game info, model
outputs, market data, and edge details. Later, when game results are known,
run build_report() to compare predictions vs actuals and produce:

  - Calibration curve (predicted prob vs actual win rate)
  - Edge accuracy tracking (did flagged edges actually hit?)
  - Brier score / LogLoss on held-out predictions
  - Sharpe / ROI simulation for logged bets

Usage:
    # In forward_test.py:
    from tools.prediction_logger import PredictionLogger
    logger = PredictionLogger()
    logger.log_predictions(predictions)

    # Later, after games have been played:
    report = logger.build_report()
    print(report)

Storage:
    data/predictions/YYYY-MM-DD_HH-MM-SS.jsonl  — one file per run
    data/predictions/_master.csv                  — merged summary all-time
"""

import os
import json
import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "predictions"


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGER
# ═══════════════════════════════════════════════════════════════════════════


class PredictionLogger:
    """
    Saves each prediction as a JSONL entry to data/predictions/.

    Each entry captures:
      - timestamp, game info, matchup
      - model predictions (total, ML probabilities)
      - market consensus (no-vig probabilities, consensus total)
      - multi-book context (n_books, std, range)
      - edge decision (verdict, confidence, stake)
      - unique run_id for grouping predictions by session

    After the games are played, load past predictions and compare vs
    actual outcomes to build a validation report.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{self.run_id}.jsonl"
        self._entries: list[dict] = []

    def log_predictions(self, predictions: list, source: str = "forward_test") -> str:
        """
        Log a list of ForwardPrediction objects to JSONL.

        Args:
            predictions: List of ForwardPrediction objects
            source: Label for this run (e.g., "forward_test", "daily_run")

        Returns:
            Path to log file as string
        """
        n_logged = 0
        with open(self.log_path, "w", encoding="utf-8") as f:
            for p in predictions:
                entry = self._prediction_to_entry(p, source)
                f.write(json.dumps(entry, default=str) + "\n")
                self._entries.append(entry)
                n_logged += 1

        # Append to master CSV for easy lookup
        self._append_to_master()

        print(f"  [Logger] Saved {n_logged} predictions to {self.log_path}")
        return str(self.log_path)

    def _prediction_to_entry(self, p: Any, source: str) -> dict:
        """Convert a ForwardPrediction object to a JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "game_date": p.game_date,
            "matchup": p.matchup,
            "home_team": p.home_team,
            "away_team": p.away_team,

            # Model predictions
            "model_total": p.model_total,
            "model_home_win_prob": p.home_win_prob,
            "model_away_win_prob": p.away_win_prob,

            # Market consensus (no-vig)
            "market_total": p.market_total,
            "market_home_implied": p.market_home_implied,
            "market_away_implied": p.market_away_implied,

            # Raw market lines
            "home_ml_raw": p.home_ml_raw,
            "away_ml_raw": p.away_ml_raw,

            # Multi-book context
            "n_books_ml": p.n_books_ml,
            "n_books_total": p.n_books_total,
            "n_books_spread": p.n_books_spread,
            "total_range_low": p.total_range[0] if p.total_range else None,
            "total_range_high": p.total_range[1] if p.total_range else None,
            "ml_range_home_low": p.ml_range_home[0] if p.ml_range_home else None,
            "ml_range_home_high": p.ml_range_home[1] if p.ml_range_home else None,
            "ml_range_away_low": p.ml_range_away[0] if p.ml_range_away else None,
            "ml_range_away_high": p.ml_range_away[1] if p.ml_range_away else None,
            "ml_std": p.ml_std,
            "total_std": p.total_std,

            # Edge decision
            "total_edge_pct": p.total_edge_pct,
            "total_verdict": p.total_verdict,
            "home_ml_edge": p.home_ml_edge,
            "away_ml_edge": p.away_ml_edge,
            "ml_verdict": p.ml_verdict,
            "edge_confidence": p.edge_confidence,
            "kelly_fraction": p.kelly_fraction,
            "recommended_stake": p.recommended_stake,

            # Bet type: "total", "moneyline", or "both"
            "bet_type": self._determine_bet_type(p),

            # Placeholder for actual results (filled later by build_report)
            "actual_home_score": None,
            "actual_away_score": None,
            "actual_total": None,
            "actual_home_win": None,
            "total_result": None,      # "WIN", "LOSS", "PUSH", or None for totals bet
            "ml_result": None,         # "WIN", "LOSS", "PUSH", or None for ML bet
            "total_profit": None,      # Profit from totals bet
            "ml_profit": None,         # Profit from ML bet
            "actual_result": None,     # Legacy: primary result (ML wins over totals)
            "actual_profit": None,     # Legacy: combined profit
        }

    @staticmethod
    def _determine_bet_type(p: Any) -> str:
        """Determine the bet type from a ForwardPrediction."""
        has_total = bool(p.total_verdict) and (p.recommended_stake or 0) > 0
        has_ml = bool(p.ml_verdict) and (p.recommended_stake or 0) > 0
        if has_total and has_ml:
            return "both"
        elif has_total:
            return "total"
        elif has_ml:
            return "moneyline"
        return "none"

    def _append_to_master(self):
        """Append all entries to a master CSV file for easy/all-time queries."""
        master_path = self.log_dir / "_master.csv"
        is_new = not master_path.exists()

        with open(master_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._entries[0].keys()) if self._entries else [])
            if is_new:
                writer.writeheader()
            for entry in self._entries:
                writer.writerow(entry)

    # ═══════════════════════════════════════════════════════════════════
    #  REPORTING — Compare Predictions vs Actual Outcomes
    # ═══════════════════════════════════════════════════════════════════

    def resolve_from_db(
        self,
        db_path: Optional[Path] = None,
        game_results: Optional[dict[str, dict]] = None,
    ) -> dict[str, Any]:
        """
        Resolve logged predictions against actual game results.

        Two modes:
          1. Provide game_results dict: {matchup_key: {"home_score": ..., "away_score": ...}}
          2. Read from database (uses NBADataLoader)

        Returns a dict with resolution statistics, and writes actual results
        back to the JSONL files.
        """
        if db_path is None and game_results is None:
            print("  [Logger] No data source provided for resolution")
            return {"resolved": 0, "errors": ["no data source"]}

        resolved = 0
        errors = 0

        # Load all JSONL files and resolve each entry
        for jsonl_file in sorted(self.log_dir.glob("*.jsonl")):
            resolved_lines = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    key = entry.get("matchup", "")
                    gd = entry.get("game_date", "")

                    result = None
                    home_score = None
                    away_score = None

                    if game_results and key in game_results:
                        res = game_results[key]
                        home_score = res.get("home_score")
                        away_score = res.get("away_score")
                    elif game_results:
                        # Try matching by team names
                        home = entry.get("home_team", "")
                        away = entry.get("away_team", "")
                        for gk, gv in game_results.items():
                            if (home in gk or home in gk.replace("@", "")) and \
                               (away in gk or away in gk.replace("@", "")):
                                home_score = gv.get("home_score")
                                away_score = gv.get("away_score")
                                key = gk
                                break

                    if home_score is not None and away_score is not None:
                        total = home_score + away_score
                        home_win = 1 if home_score > away_score else 0

                        entry["actual_home_score"] = home_score
                        entry["actual_away_score"] = away_score
                        entry["actual_total"] = total
                        entry["actual_home_win"] = home_win
                        stake = entry.get("recommended_stake", 0) or 0

                        # ── Totals resolution ──────────────────────────────
                        if entry.get("total_verdict") == "OVER":
                            entry["total_result"] = "WIN" if total > entry.get("market_total", 0) else "LOSS"
                        elif entry.get("total_verdict") == "UNDER":
                            entry["total_result"] = "WIN" if total < entry.get("market_total", 0) else "LOSS"

                        if entry["total_result"] == "WIN":
                            # Standard -110 odds for totals: profit = stake * (100/110)
                            entry["total_profit"] = round(stake * 0.909, 2)
                        elif entry["total_result"] == "LOSS":
                            entry["total_profit"] = -stake
                        else:
                            entry["total_profit"] = 0

                        # ── ML resolution ──────────────────────────────────
                        if entry.get("ml_verdict"):
                            verdict_team = entry["ml_verdict"]
                            if verdict_team == entry.get("home_team"):
                                entry["ml_result"] = "WIN" if home_win == 1 else "LOSS"
                            elif verdict_team == entry.get("away_team"):
                                entry["ml_result"] = "WIN" if home_win == 0 else "LOSS"

                            if entry["ml_result"] == "WIN":
                                ml_raw = entry.get("home_ml_raw") if entry.get("ml_verdict") == entry.get("home_team") else entry.get("away_ml_raw")
                                if ml_raw:
                                    dec = 1 + ml_raw / 100 if ml_raw > 0 else 1 + 100 / abs(ml_raw)
                                    entry["ml_profit"] = round(stake * (dec - 1), 2)
                                else:
                                    entry["ml_profit"] = 0
                            elif entry["ml_result"] == "LOSS":
                                entry["ml_profit"] = -stake
                            else:
                                entry["ml_profit"] = 0

                        # ── Legacy single-result fields ────────────────────
                        # For backwards compat: actual_result picks the active bet type
                        # ML takes priority over totals if both exist
                        bt = entry.get("bet_type", "none")
                        if bt == "both":
                            entry["actual_result"] = entry.get("ml_result")
                            entry["actual_profit"] = (entry.get("total_profit", 0) or 0) + (entry.get("ml_profit", 0) or 0)
                        elif bt == "total":
                            entry["actual_result"] = entry.get("total_result")
                            entry["actual_profit"] = entry.get("total_profit", 0)
                        elif bt == "moneyline":
                            entry["actual_result"] = entry.get("ml_result")
                            entry["actual_profit"] = entry.get("ml_profit", 0)

                        resolved += 1

                    resolved_lines.append(entry)

            # Rewrite the file with resolved data
            if resolved > 0:
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for entry in resolved_lines:
                        f.write(json.dumps(entry, default=str) + "\n")

        print(f"  [Logger] Resolved {resolved} predictions against actual results")
        return {"resolved": resolved, "errors": errors}

    def build_report(self) -> dict[str, Any]:
        """
        Build a validation report from all logged (and resolved) predictions.

        Returns a dict with:
          - total_predictions: int
          - bets_placed: int (predictions that triggered stake > 0)
          - wins / losses / push: int
          - win_rate: float
          - roi: float (return on investment)
          - total_profit: float
          - total_stake: float
          - avg_edge: float
          - edge_accuracy: dict (how often edges at different thresholds hit)
          - calibration: dict (predicted prob bins vs actual win rate)
          - brier_score: float (for ML predictions)
        """
        report = {
            "total_predictions": 0,
            "bets_placed": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "total_profit": 0.0,
            "total_stake": 0.0,
            "avg_edge": 0.0,
            "edges_used": [],
            "edge_accuracy": {},
            "calibration": [],
            "brier_score": None,
        }

        all_ml_probs = []
        all_ml_actuals = []

        # Read all JSONL files
        for jsonl_file in sorted(self.log_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    report["total_predictions"] += 1

                    # Track ML probs for Brier score
                    if entry.get("model_home_win_prob") is not None and entry.get("actual_home_win") is not None:
                        all_ml_probs.append(entry["model_home_win_prob"])
                        all_ml_actuals.append(entry["actual_home_win"])

                    # Only count bets that were placed (had stake > 0 and result known)
                    stake = entry.get("recommended_stake", 0) or 0
                    result = entry.get("actual_result")
                    if stake > 0 and result is not None:
                        report["bets_placed"] += 1
                        report["total_stake"] += stake

                        if result == "WIN":
                            report["wins"] += 1
                        elif result == "LOSS":
                            report["losses"] += 1
                        else:
                            report["pushes"] += 1

                        profit = entry.get("actual_profit", 0) or 0
                        report["total_profit"] += profit

                        # Track edge accuracy
                        edge_pct = entry.get("total_edge_pct") or entry.get("home_ml_edge") or entry.get("away_ml_edge") or 0
                        report["edges_used"].append({
                            "edge": edge_pct,
                            "won": result == "WIN",
                            "confidence": entry.get("edge_confidence", "unknown"),
                        })

        # Compute stats
        if report["bets_placed"] > 0:
            report["win_rate"] = report["wins"] / report["bets_placed"]
            report["roi"] = (report["total_profit"] / report["total_stake"]) if report["total_stake"] > 0 else 0.0

        # Edge accuracy by threshold
        if report["edges_used"]:
            for threshold in [0.01, 0.02, 0.03, 0.05, 0.10]:
                subset = [e for e in report["edges_used"] if abs(e["edge"]) >= threshold]
                if subset:
                    wins = sum(1 for e in subset if e["won"])
                    report["edge_accuracy"][f"{threshold:.0%}+"] = {
                        "n_bets": len(subset),
                        "win_rate": wins / len(subset) if subset else 0,
                    }

        # Calibration curve
        if len(all_ml_probs) >= 20:
            probs = np.array(all_ml_probs)
            actuals = np.array(all_ml_actuals)
            bins = np.linspace(0, 1, 11)
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                in_bin = (probs >= lo) & (probs < hi)
                n = in_bin.sum()
                if n >= 5:
                    report["calibration"].append({
                        "bin": f"{lo:.0%}-{hi:.0%}",
                        "n": int(n),
                        "mean_pred": float(probs[in_bin].mean()),
                        "actual_rate": float(actuals[in_bin].mean()),
                    })

            # Brier score
            report["brier_score"] = float(np.mean((probs - actuals) ** 2))

        return report


# ═══════════════════════════════════════════════════════════════════════════
#  CLI — Build report from existing logs
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """CLI: Build a prediction validation report from logged data."""
    import argparse

    parser = argparse.ArgumentParser(description="Prediction Logger — Validate predictions vs actuals")
    parser.add_argument("--resolve", action="store_true",
                        help="Load actual game results from DB and resolve predictions")
    parser.add_argument("--report", action="store_true",
                        help="Build and print validation report from resolved predictions")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite DB (default: auto-detect from config)")

    args = parser.parse_args()
    logger = PredictionLogger()

    if args.resolve:
        # Try to load game results from the database
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.config import settings

            db_path = Path(args.db) if args.db else settings.resolved_nba_db_path
            loader = NBADataLoader(db_path=db_path)
            raw_df = loader.load_game_logs()
            games_df = loader.build_game_dataset(raw_df)

            # Build results dict: matchup -> {home_score, away_score}
            results = {}
            for _, row in games_df.iterrows():
                matchup = f"{row.get('TEAM_NAME_away', '?')} @ {row.get('TEAM_NAME_home', '?')}"
                results[matchup] = {
                    "home_score": row.get("PTS_home", 0),
                    "away_score": row.get("PTS_away", 0),
                }
            logger.resolve_from_db(game_results=results)
        except Exception as e:
            print(f"  [Logger] Resolution failed: {e}")
            return 1

    if args.report:
        report = logger.build_report()
        print(f"\n{'=' * 60}")
        print(f"  PREDICTION VALIDATION REPORT")
        print(f"{'=' * 60}")
        print(f"  Total logged:  {report['total_predictions']}")
        print(f"  Bets placed:   {report['bets_placed']}")
        if report['bets_placed'] > 0:
            print(f"  Wins:          {report['wins']} / {report['bets_placed']} ({report['win_rate']:.1%})")
            print(f"  Losses:        {report['losses']}")
            print(f"  Pushes:        {report['pushes']}")
            print(f"  Total stake:   ${report['total_stake']:.2f}")
            print(f"  Total profit:  ${report['total_profit']:.2f}")
            print(f"  ROI:           {report['roi']:.2%}")
        else:
            print(f"  (no bets resolved yet — run --resolve first)")
        print(f"  Avg edge:      {report['avg_edge']:.1%}" if report['avg_edge'] else "")
        if report["brier_score"] is not None:
            print(f"  ML Brier:      {report['brier_score']:.4f}")
        if report["calibration"]:
            print(f"\n  Calibration:")
            for c in report["calibration"]:
                print(f"    {c['bin']:>10s}: n={c['n']:>4d}  "
                      f"pred={c['mean_pred']:.1%}  actual={c['actual_rate']:.1%}"
                      f"  {'✓' if abs(c['mean_pred'] - c['actual_rate']) < 0.05 else '✗'}")
        print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
