"""
ResultsTracker — The feedback loop your betting system has been missing.

Loads logged predictions, fetches actual game results from the local NBA
database, matches each prediction against what actually happened, and
computes:

  - P&L by model, league, bet type, and date
  - Win rate, ROI, Sharpe ratio
  - Trailing 30-day performance per strategy
  - Alerts when any strategy drops below -5% ROI threshold

Usage:
    from betting_intel.analytics.tracker import ResultsTracker

    tracker = ResultsTracker()
    tracker.resolve_all()                # Fetch results & match
    report = tracker.generate_report()    # Full report
    tracker.check_alerts()               # Check for underperformance
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FORWARD_TEST_JSON = PROJECT_ROOT / "data" / "forward_test_results.json"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
ALERTS_LOG = PROJECT_ROOT / "data" / "analytics_alerts.jsonl"

# ── Thresholds ─────────────────────────────────────────────────────────────

DEFAULT_BANKROLL = 10_000.0
TRAILING_WINDOW_DAYS = 30
ALERT_ROI_THRESHOLD = -0.05  # -5% ROI triggers alert
MIN_BETS_FOR_ALERT = 5       # Minimum bets before we sound the alarm


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ResolvedBet:
    """A single prediction that has been matched against an actual result."""

    # ── Identity ─────────────────────────────────────────────────────
    prediction_id: str = ""          # Unique ID from the log entry
    run_id: str = ""                 # Run ID (can group by prediction run)
    source: str = "forward_test"      # "forward_test", "daily_run", etc.
    source_file: str = ""             # Path to the source file

    # ── Game info ────────────────────────────────────────────────────
    game_date: str = ""              # YYYY-MM-DD
    matchup: str = ""                # "Away @ Home"
    home_team: str = ""
    away_team: str = ""
    league: str = "NBA"

    # ── Prediction details ───────────────────────────────────────────
    bet_type: str = ""               # "total", "moneyline", "both"
    bet_side: str = ""               # e.g., "Total OVER 224.5", "ML Celtics"
    market_line: float = 0.0         # Market line at prediction time
    predicted_value: float = 0.0     # What the model predicted
    edge_pct: float = 0.0            # Predicted edge percentage
    stake_dollars: float = 0.0       # Recommended stake
    kelly_fraction: float = 0.0
    confidence: str = "low"          # "high", "medium", "low"

    # ── Model info ───────────────────────────────────────────────────
    model_name: str = "ensemble"     # Which model generated this
    predicted_total: Optional[float] = None
    home_win_prob: Optional[float] = None

    # ── Actual results (filled after resolution) ─────────────────────
    actual_home_score: Optional[float] = None
    actual_away_score: Optional[float] = None
    actual_total: Optional[float] = None
    actual_home_win: Optional[bool] = None

    # ── Resolution ───────────────────────────────────────────────────
    result: Optional[str] = None     # "WIN", "LOSS", "PUSH"
    profit_dollars: float = 0.0      # Actual profit/loss in dollars
    resolved_at: Optional[str] = None

    # ── Derived (computed after resolution) ──────────────────────────
    roi: float = 0.0                 # profit / stake
    is_clear_pick: bool = False

    # ── Closing Line Value (filled by compute_clv) ───────────────────
    closing_implied_prob: Optional[float] = None  # Closing market-implied home win prob
    predicted_implied_prob: Optional[float] = None  # Our predicted home win prob
    clv: Optional[float] = None  # Closing Line Value: positive = we beat the market


@dataclass
class StrategyPerformance:
    """Performance of a single strategy (model/league/bet_type combo) over a window."""

    strategy_name: str = ""           # e.g., "ensemble/NBA/total"
    model: str = "ensemble"
    league: str = "NBA"
    bet_type: str = "total"

    n_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    win_rate: float = 0.0

    total_stake: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0

    avg_edge: float = 0.0
    avg_odds: float = 0.0           # Average decimal odds (-110 = 1.909)
    sharpe: float = 0.0             # Risk-adjusted return

    trailing_profits: list[float] = field(default_factory=list)  # Daily P&L
    is_alerted: bool = False         # True if ROI < threshold
    last_bet_date: str = ""


@dataclass
class PerformanceReport:
    """Complete performance report across all strategies."""

    generated_at: str = ""
    n_resolved: int = 0
    n_unresolved: int = 0
    total_bets: int = 0
    total_stake: float = 0.0
    total_profit: float = 0.0
    overall_roi: float = 0.0
    overall_win_rate: float = 0.0

    strategies: list[StrategyPerformance] = field(default_factory=list)
    confident_strategies: list[StrategyPerformance] = field(default_factory=list)
    alerted_strategies: list[StrategyPerformance] = field(default_factory=list)
    recent_bets: list[ResolvedBet] = field(default_factory=list)

    daily_pnl: list[dict] = field(default_factory=list)
    model_comparison: dict[str, dict] = field(default_factory=dict)
    league_comparison: dict[str, dict] = field(default_factory=dict)

    # Closing Line Value metrics
    avg_clv: Optional[float] = None  # Average CLV across all resolved bets
    clv_wins: int = 0  # Number of bets where we beat the closing line
    clv_losses: int = 0  # Number of bets where the closing line beat us
    clv_win_rate: Optional[float] = None  # % of bets with positive CLV


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS TRACKER
# ═══════════════════════════════════════════════════════════════════════════


class ResultsTracker:
    """
    Tracks every prediction, resolves against actual results, computes P&L.

    Data pipeline:
      1. Read predictions from JSONL files and forward_test_results.json
      2. Fetch actual game results from ESPN (free, no key needed)
      3. Match predictions to results by (team names, date)
      4. Compute P&L for each bet
      5. Aggregate by strategy (model × league × bet_type)
      6. Check trailing performance for underperformance alerts
      7. Generate report

    Usage:
        tracker = ResultsTracker()
        tracker.resolve_all()
        report = tracker.generate_report(window_days=30)
        alerts = tracker.check_alerts(report)
        tracker.save_report(report)
    """

    def __init__(
        self,
        predictions_dir: Optional[Path] = None,
        bankroll: float = DEFAULT_BANKROLL,
    ):
        self.predictions_dir = Path(predictions_dir) if predictions_dir else PREDICTIONS_DIR
        self.bankroll = bankroll
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._raw_predictions: list[dict] = []
        self._resolved_bets: list[ResolvedBet] = []
        self._unresolved_predictions: list[dict] = []

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def resolve_all(self) -> int:
        """
        Resolve ALL logged predictions against actual game results.

        Flow:
          1. Load all predictions from disk (JSONL files + forward_test_results.json)
          2. Fetch actual results from the local NBA database
          3. Match each prediction to its outcome
          4. Compute P&L per bet

        Returns:
            Number of predictions that were newly resolved
        """
        # Step 1: Load all predictions
        self._load_predictions()
        if not self._raw_predictions:
            logger.warning("No predictions found to resolve")
            return 0

        # Step 2: Separate resolved vs unresolved
        resolved_entries = [p for p in self._raw_predictions if p.get("actual_result") is not None]
        unresolved = [p for p in self._raw_predictions if p.get("actual_result") is None]

        logger.info(
            f"Loaded {len(self._raw_predictions)} predictions "
            f"({len(resolved_entries)} already resolved, {len(unresolved)} pending)"
        )

        if not unresolved:
            # Already all resolved — just load them
            self._resolved_bets = [self._entry_to_resolved_bet(e) for e in resolved_entries]
            logger.info("All predictions already resolved")
            return 0

        # Step 3: Fetch actual results for unresolved games
        results_map = self._fetch_results(unresolved)

        if not results_map:
            logger.warning("No actual results fetched — predictions remain unresolved")
            return 0

        # Step 4: Match and resolve
        newly_resolved = 0
        for entry in unresolved:
            result = self._match_entry_to_result(entry, results_map)
            if result:
                entry.update({
                    "actual_home_score": result["home_score"],
                    "actual_away_score": result["away_score"],
                    "actual_total": result["home_score"] + result["away_score"],
                    "actual_home_win": 1 if result["home_score"] > result["away_score"] else 0,
                    "actual_result": result["verdict"],
                    "actual_profit": result["profit"],
                    "resolved_at": datetime.now().isoformat(),
                })
                newly_resolved += 1

        if newly_resolved > 0:
            # Step 5: Save resolved data back to source files
            self._save_resolved_predictions()

        # Step 6: Build ResolvedBet objects from ALL entries
        all_entries = [p for p in self._raw_predictions if p.get("actual_result") is not None]
        self._resolved_bets = [self._entry_to_resolved_bet(e) for e in all_entries]
        self._unresolved_predictions = [p for p in self._raw_predictions if p.get("actual_result") is None]

        logger.info(f"Resolved {newly_resolved} predictions against actual results")
        return newly_resolved

    def generate_report(self, window_days: int = TRAILING_WINDOW_DAYS) -> PerformanceReport:
        """
        Generate a comprehensive performance report from resolved bets.

        Aggregates by strategy (model × league × bet_type), computes
        trailing window performance, and identifies underperforming strategies.

        Args:
            window_days: Number of days for trailing performance window

        Returns:
            PerformanceReport with all metrics
        """
        report = PerformanceReport(
            generated_at=datetime.now().isoformat(),
            n_resolved=len(self._resolved_bets),
            n_unresolved=len(self._unresolved_predictions),
        )

        if not self._resolved_bets:
            return report

        # ── Overall stats ──────────────────────────────────────────
        report.total_bets = len(self._resolved_bets)
        report.total_stake = sum(b.stake_dollars for b in self._resolved_bets)
        report.total_profit = sum(b.profit_dollars for b in self._resolved_bets)
        wins = sum(1 for b in self._resolved_bets if b.result == "WIN")
        losses = sum(1 for b in self._resolved_bets if b.result == "LOSS")
        report.overall_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        report.overall_roi = report.total_profit / report.total_stake if report.total_stake > 0 else 0.0

        # ── Daily P&L ──────────────────────────────────────────────
        daily: dict[str, list[float]] = {}
        for bet in self._resolved_bets:
            day = bet.game_date[:10]
            if day not in daily:
                daily[day] = []
            daily[day].append(bet.profit_dollars)

        report.daily_pnl = [
            {"date": day, "profit": round(sum(profits), 2), "n_bets": len(profits)}
            for day, profits in sorted(daily.items())
        ]

        # ── Strategy breakdown ──────────────────────────────────────
        strategy_map: dict[str, list[ResolvedBet]] = {}
        for bet in self._resolved_bets:
            key = f"{bet.model_name}/{bet.league}/{bet.bet_type}"
            if key not in strategy_map:
                strategy_map[key] = []
            strategy_map[key].append(bet)

        for key, bets in strategy_map.items():
            parts = key.split("/")
            perf = self._compute_strategy_performance(bets, parts[0], parts[1], parts[2])
            report.strategies.append(perf)

            # Confident strategies: at least 5 bets
            if perf.n_bets >= MIN_BETS_FOR_ALERT:
                report.confident_strategies.append(perf)

            # Check for alerted strategies
            if perf.is_alerted:
                report.alerted_strategies.append(perf)

        report.strategies.sort(key=lambda s: s.roi, reverse=True)

        # ── Model comparison ────────────────────────────────────────
        model_map: dict[str, list[ResolvedBet]] = {}
        for bet in self._resolved_bets:
            model_map.setdefault(bet.model_name, []).append(bet)
        for model_name, bets in model_map.items():
            perf = self._compute_strategy_performance(bets, model_name, "ALL", "ALL")
            report.model_comparison[model_name] = {
                "n_bets": perf.n_bets,
                "wins": perf.wins,
                "losses": perf.losses,
                "win_rate": round(perf.win_rate, 4),
                "total_stake": round(perf.total_stake, 2),
                "total_profit": round(perf.total_profit, 2),
                "roi": round(perf.roi, 4),
                "avg_edge": round(perf.avg_edge, 4),
            }

        # ── League comparison ───────────────────────────────────────
        league_map: dict[str, list[ResolvedBet]] = {}
        for bet in self._resolved_bets:
            league_map.setdefault(bet.league, []).append(bet)
        for league_name, bets in league_map.items():
            perf = self._compute_strategy_performance(bets, "ALL", league_name, "ALL")
            report.league_comparison[league_name] = {
                "n_bets": perf.n_bets,
                "wins": perf.wins,
                "losses": perf.losses,
                "win_rate": round(perf.win_rate, 4),
                "total_stake": round(perf.total_stake, 2),
                "total_profit": round(perf.total_profit, 2),
                "roi": round(perf.roi, 4),
            }

        # ── CLV metrics ─────────────────────────────────────────────
        self.compute_clv()
        clv_values = [b.clv for b in self._resolved_bets if b.clv is not None]
        if clv_values:
            report.avg_clv = round(sum(clv_values) / len(clv_values), 4)
            report.clv_wins = sum(1 for c in clv_values if c > 0)
            report.clv_losses = sum(1 for c in clv_values if c < 0)
            total_clv_bets = report.clv_wins + report.clv_losses
            report.clv_win_rate = round(report.clv_wins / total_clv_bets, 4) if total_clv_bets > 0 else None

        # ── Recent bets (last 50) ───────────────────────────────────
        report.recent_bets = sorted(
            self._resolved_bets,
            key=lambda b: b.game_date,
            reverse=True,
        )[:50]

        return report

    # ── CLV Computation ───────────────────────────────────────────────────

    def compute_clv(self) -> None:
        """
        Compute Closing Line Value (CLV) for all resolved bets.

        CLV measures whether our predicted line was better than the closing
        market line. Positive CLV means we identified market inefficiency.
        This is the single most important metric in sports betting analytics.

        For each resolved bet with real market odds data:
          1. Query MarketOddsStore for opening AND closing vig-free home probs
          2. Our predicted prob = opening_prob + edge_pct (our edge prediction)
          3. CLV = our_predicted_prob - closing_prob
             - Positive: we predicted a line BETTER than where the market settled
             - Negative: the market moved against our prediction
             - Zero: our edge was exactly the market movement (neutral)
          4. Store clv, closing_implied_prob, predicted_implied_prob

        NOTE: Requires odds data in MarketOddsStore. Bets from before the store
        was set up will have no CLV data until the store accumulates history.
        """
        if not self._resolved_bets:
            return

        try:
            from betting_intel.db.market_odds_store import MarketOddsStore
            store = MarketOddsStore()
        except Exception:
            logger.debug("MarketOddsStore not available — cannot compute CLV")
            return

        clv_count = 0
        for bet in self._resolved_bets:
            if not bet.home_team or not bet.away_team or not bet.game_date:
                continue

            try:
                # Get BOTH opening and closing market probabilities
                opening_prob, closing_prob = store.get_closing_vs_opening_prob(
                    home_team=bet.home_team,
                    away_team=bet.away_team,
                    game_date=bet.game_date[:10],
                )

                if opening_prob is None or closing_prob is None:
                    continue

                bet.closing_implied_prob = closing_prob

                # Our predicted probability at prediction time:
                # We predicted the market was wrong by edge_pct.
                # So our true belief = opening_market_prob + edge_pct
                our_prob = opening_prob + bet.edge_pct
                bet.predicted_implied_prob = round(our_prob, 4)

                # True CLV = our_prob - closing_prob
                # Positive = we saw value that the market didn't fully price in
                # (the closing line didn't fully move against our prediction)
                bet.clv = round(our_prob - closing_prob, 4)

                clv_count += 1

            except Exception as e:
                logger.debug(f"CLV computation failed for {bet.matchup}: {e}")
                continue

        if clv_count > 0:
            logger.info(f"Computed CLV for {clv_count}/{len(self._resolved_bets)} resolved bets")

    def check_alerts(self, report: Optional[PerformanceReport] = None) -> list[StrategyPerformance]:
        """
        Check all strategies for underperformance and log alerts.

        Args:
            report: If provided, checks strategies in the report.
                    If None, generates a fresh report first.

        Returns:
            List of strategies that triggered alerts
        """
        if report is None:
            self.resolve_all()
            report = self.generate_report()

        alerts_triggered = []
        for strategy in report.confident_strategies:
            if strategy.is_alerted:
                alerts_triggered.append(strategy)
                self._log_alert(strategy)

        if alerts_triggered:
            logger.warning(
                f"  {len(alerts_triggered)} strategy(ies) below -5% ROI threshold: "
                + ", ".join(s.strategy_name for s in alerts_triggered)
            )
        else:
            logger.info("No strategy alerts triggered — all strategies above threshold")

        return alerts_triggered

    def save_report(self, report: PerformanceReport) -> str:
        """
        Save the performance report to disk as JSON.

        Returns:
            Path to the saved report file
        """
        report_path = REPORTS_DIR / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "generated_at": report.generated_at,
            "overall_roi": report.overall_roi,
            "overall_win_rate": report.overall_win_rate,
            "total_bets": report.total_bets,
            "total_stake": round(report.total_stake, 2),
            "total_profit": round(report.total_profit, 2),
            "strategies": [
                {
                    "strategy_name": s.strategy_name,
                    "n_bets": s.n_bets,
                    "wins": s.wins,
                    "losses": s.losses,
                    "win_rate": round(s.win_rate, 4),
                    "total_stake": round(s.total_stake, 2),
                    "total_profit": round(s.total_profit, 2),
                    "roi": round(s.roi, 4),
                    "avg_edge": round(s.avg_edge, 4),
                    "is_alerted": s.is_alerted,
                }
                for s in report.strategies
            ],
            "daily_pnl": report.daily_pnl,
            "model_comparison": report.model_comparison,
            "league_comparison": report.league_comparison,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Performance report saved to {report_path}")
        return str(report_path)

    def get_dashboard_data(self) -> dict:
        """
        Get a JSON-friendly dict for the P&L web dashboard.

        Returns a dict that can be directly returned from a FastAPI route
        or template context.
        """
        self.resolve_all()
        report = self.generate_report()

        # Check for alerts
        alerted = self.check_alerts(report)

        return {
            "generated_at": report.generated_at,
            "overall": {
                "total_bets": report.total_bets,
                "total_stake": round(report.total_stake, 2),
                "total_profit": round(report.total_profit, 2),
                "overall_roi": round(report.overall_roi, 4),
                "overall_win_rate": round(report.overall_win_rate, 4),
                "n_resolved": report.n_resolved,
                "n_unresolved": report.n_unresolved,
            },
            "strategies": [
                {
                    "strategy_name": s.strategy_name,
                    "model": s.model,
                    "league": s.league,
                    "bet_type": s.bet_type,
                    "n_bets": s.n_bets,
                    "wins": s.wins,
                    "losses": s.losses,
                    "win_rate": round(s.win_rate, 4),
                    "total_stake": round(s.total_stake, 2),
                    "total_profit": round(s.total_profit, 2),
                    "roi": round(s.roi, 4),
                    "avg_edge": round(s.avg_edge, 4),
                    "is_alerted": s.is_alerted,
                }
                for s in report.strategies
            ],
            "alerted_strategies": [
                {
                    "strategy_name": s.strategy_name,
                    "roi": round(s.roi, 4),
                    "n_bets": s.n_bets,
                    "total_profit": round(s.total_profit, 2),
                }
                for s in alerted
            ],
            "daily_pnl": report.daily_pnl,
            "model_comparison": report.model_comparison,
            "league_comparison": report.league_comparison,
            "recent_bets": [
                {
                    "game_date": b.game_date,
                    "matchup": b.matchup,
                    "bet_side": b.bet_side,
                    "bet_type": b.bet_type,
                    "stake_dollars": round(b.stake_dollars, 2),
                    "profit_dollars": round(b.profit_dollars, 2),
                    "result": b.result,
                    "roi": round(b.roi, 4),
                    "edge_pct": round(b.edge_pct, 4),
                    "model_name": b.model_name,
                    "league": b.league,
                    "confidence": b.confidence,
                    "is_clear_pick": b.is_clear_pick,
                }
                for b in report.recent_bets
            ],
        }

    # ── INTERNAL: Data Loading ─────────────────────────────────────────────

    def _load_predictions(self):
        """Load all predictions from disk (JSONL files + forward_test_results.json)."""
        self._raw_predictions = []
        seen_ids: set[str] = set()

        # 1. Load from JSONL files (prediction_logger format)
        if self.predictions_dir.exists():
            for jsonl_file in sorted(self.predictions_dir.glob("*.jsonl")):
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            entry = json.loads(line)
                            gid = f"{entry.get('matchup', '')}_{entry.get('game_date', '')}"
                            if gid not in seen_ids:
                                seen_ids.add(gid)
                                self._raw_predictions.append(entry)
                except Exception as e:
                    logger.debug(f"Failed to load {jsonl_file}: {e}")

        # 2. Load from forward_test_results.json
        if FORWARD_TEST_JSON.exists():
            try:
                with open(FORWARD_TEST_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for bet in data.get("all_bets", []):
                    gid = f"{bet.get('matchup', '')}_{bet.get('game_date', '')}"
                    if gid not in seen_ids:
                        seen_ids.add(gid)
                        self._raw_predictions.append({
                            "game_date": bet.get("game_date", ""),
                            "matchup": bet.get("matchup", ""),
                            "home_team": bet.get("home_team", ""),
                            "away_team": bet.get("away_team", ""),
                            "bet_type": bet.get("bet_type", "total"),
                            "bet_side": bet.get("bet_side", ""),
                            "market_line": bet.get("market_line", 0),
                            "model_line": bet.get("model_line", 0),
                            "edge_pct": bet.get("edge_pct", 0),
                            "stake_dollars": bet.get("stake_dollars", 0),
                            "kelly_fraction": bet.get("kelly_fraction", 0),
                            "edge_confidence": bet.get("edge_confidence", "low"),
                            "model_name": "forward_test_ensemble",
                            "predicted_total": bet.get("model_line"),
                            "home_win_prob": bet.get("model_line"),
                            "actual_home_score": bet.get("actual_home_score"),
                            "actual_away_score": bet.get("actual_away_score"),
                            "actual_result": bet.get("actual_result"),
                            "actual_profit": bet.get("actual_profit"),
                            "is_clear_pick": bet.get("is_clear_pick", False),
                            "source": "forward_test_results.json",
                        })
            except Exception as e:
                logger.debug(f"Failed to load forward_test_results.json: {e}")

        # 3. Load from _master.csv
        master_csv = self.predictions_dir / "_master.csv"
        if master_csv.exists():
            try:
                with open(master_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        gid = f"{row.get('matchup', '')}_{row.get('game_date', '')}"
                        if gid not in seen_ids:
                            seen_ids.add(gid)
                            self._raw_predictions.append({
                                "game_date": row.get("game_date", ""),
                                "matchup": row.get("matchup", ""),
                                "home_team": row.get("home_team", ""),
                                "away_team": row.get("away_team", ""),
                                "bet_type": row.get("bet_type", "total"),
                                "bet_side": row.get("bet_side", ""),
                                "market_line": float(row.get("market_total", 0) or 0),
                                "model_line": float(row.get("model_total", 0) or 0),
                                "edge_pct": float(row.get("total_edge_pct", 0) or 0),
                                "stake_dollars": float(row.get("recommended_stake", 0) or 0),
                                "kelly_fraction": float(row.get("kelly_fraction", 0) or 0),
                                "edge_confidence": row.get("edge_confidence", "low"),
                                "model_name": "csv_log",
                                "predicted_total": float(row.get("model_total", 0) or 0),
                                "actual_home_score": row.get("actual_home_score"),
                                "actual_away_score": row.get("actual_away_score"),
                                "actual_result": row.get("actual_result"),
                                "actual_profit": row.get("actual_profit"),
                                "is_clear_pick": False,
                                "source": "_master.csv",
                            })
            except Exception as e:
                logger.debug(f"Failed to load _master.csv: {e}")

        logger.info(f"Loaded {len(self._raw_predictions)} unique predictions from {len(seen_ids)} games")

    # ── INTERNAL: Result Fetching ───────────────────────────────────────────

    def _fetch_results(
        self,
        unresolved: list[dict],
    ) -> dict[str, dict]:
        """
        Fetch actual game results from the local NBA database.

        Returns dict of {matchup_key: {home_score, away_score}} for each
        completed game matching the unresolved predictions' dates.
        """
        dates_needed: set[str] = set()
        for entry in unresolved:
            gd = entry.get("game_date", "")
            if gd and gd[:10]:
                dates_needed.add(gd[:10])

        if not dates_needed:
            return {}

        results_map: dict[str, dict] = {}

        # Try local NBA database first (fastest path)
        try:
            from betting_intel.data.loader import NBADataLoader
            loader = NBADataLoader()
            import pandas as pd
            raw_df = loader.load_game_logs()
            if raw_df is not None and not raw_df.empty:
                games_df = loader.build_game_dataset(raw_df)
                for _, row in games_df.iterrows():
                    gdate = str(pd.to_datetime(row["GAME_DATE"]).date())
                    home = str(row.get("TEAM_NAME_home", ""))
                    away = str(row.get("TEAM_NAME_away", ""))
                    if home and away:
                        for key in [f"{away} @ {home}^{gdate}", f"{home} @ {away}^{gdate}"]:
                            if key not in results_map:
                                results_map[key] = {
                                    "home_score": float(row.get("team_pts_home", 0)),
                                    "away_score": float(row.get("team_pts_away", 0)),
                                }
                logger.info(f"Loaded {len(results_map)} result entries from database")
                return results_map
        except ImportError:
            logger.debug("NBADataLoader not available")
        except Exception as e:
            logger.debug(f"Database result fetch failed: {e}")

        logger.warning(f"No results found for {len(dates_needed)} date(s): {dates_needed}")
        return results_map

    # ── INTERNAL: Matching ─────────────────────────────────────────────────

    def _match_entry_to_result(
        self,
        entry: dict,
        results_map: dict[str, dict],
    ) -> Optional[dict]:
        """
        Match a prediction entry to an actual result.

        Tries exact date match first, then ±1 day for timezone-offset games
        (NBA games tip late US ET, so UTC commence_time can be +1 day).

        Tries two key formats: "Away @ Home^{date}" and fuzzy team-name lookup.
        """
        game_date = entry.get("game_date", "")[:10]
        home = entry.get("home_team", "")
        away = entry.get("away_team", "")
        matchup = entry.get("matchup", "")

        # Generate candidate dates: exact, day before, day after
        dates_to_try = [game_date]
        try:
            dt = datetime.strptime(game_date, "%Y-%m-%d")
            dates_to_try.append((dt - timedelta(days=1)).strftime("%Y-%m-%d"))
            dates_to_try.append((dt + timedelta(days=1)).strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            pass  # If date can't be parsed, just use exact

        for candidate_date in dates_to_try:
            result = self._try_match_date(entry, results_map, candidate_date, home, away, matchup)
            if result is not None:
                return result

        return None

    def _try_match_date(
        self,
        entry: dict,
        results_map: dict[str, dict],
        game_date: str,
        home: str,
        away: str,
        matchup: str,
    ) -> Optional[dict]:
        """Try matching a prediction against results for a specific date."""
        # 1. Direct lookup: "Away @ Home^{date}"
        if matchup and " @ " in matchup:
            key = f"{matchup}^{game_date}"
            if key in results_map:
                res = results_map[key]
                return self._resolve_bet(entry, res["home_score"], res["away_score"])

        # 2. Build key from team names
        if home and away:
            key = f"{away} @ {home}^{game_date}"
            if key in results_map:
                res = results_map[key]
                return self._resolve_bet(entry, res["home_score"], res["away_score"])

        # 3. Fuzzy: team names appear anywhere in key with matching date
        if home and away:
            for key, res in results_map.items():
                if game_date in key and home.lower() in key.lower() and away.lower() in key.lower():
                    return self._resolve_bet(entry, res["home_score"], res["away_score"])

        return None

    def _resolve_bet(self, entry: dict, home_score: float, away_score: float) -> dict:
        """
        Resolve a bet against actual scores.

        Handles two prediction formats:
          - Original forward_test_results format using `bet_side` (e.g., "ML Spurs", "Total OVER 224.5")
          - JSONL format using `ml_verdict`/`total_verdict` (e.g., ml_verdict="Spurs", total_verdict="OVER")

        For "both" bet_type, totals are checked first; if unresolved, moneyline is checked.
        Uses -110 odds (1.909 decimal) for profit calculation.
        """
        total_score = home_score + away_score
        home_win = home_score > away_score
        bet_type = entry.get("bet_type", "total")
        bet_side = entry.get("bet_side", "")
        market_line = float(entry.get("market_line", 0))

        # Handle both stake_dollars (forward_test format) and recommended_stake (JSONL format)
        stake = float(entry.get("stake_dollars", 0) or 0)
        if stake == 0:
            stake = float(entry.get("recommended_stake", 0) or 0)

        # JSONL format uses ml_verdict / total_verdict instead of bet_side
        ml_verdict = entry.get("ml_verdict", "")
        total_verdict = entry.get("total_verdict", "")

        verdict = "PUSH"
        profit = 0.0

        is_total = bet_type in ("total", "both")
        is_ml = bet_type in ("moneyline", "both")

        # ── Totals (OVER/UNDER) ────────────────────────────────────────────
        if is_total:
            if total_verdict:
                # JSONL format: total_verdict is "OVER" or "UNDER"
                tv = str(total_verdict).upper()
                mt = market_line
                if tv == "OVER":
                    verdict = "WIN" if total_score > mt else ("LOSS" if total_score < mt else "PUSH")
                elif tv == "UNDER":
                    verdict = "WIN" if total_score < mt else ("LOSS" if total_score > mt else "PUSH")
            elif bet_side and market_line > 0:
                # Original format: bet_side like "Total OVER 224.5"
                bs_upper = str(bet_side).upper()
                mt = market_line
                if "OVER" in bs_upper:
                    verdict = "WIN" if total_score > mt else ("LOSS" if total_score < mt else "PUSH")
                elif "UNDER" in bs_upper:
                    verdict = "WIN" if total_score < mt else ("LOSS" if total_score > mt else "PUSH")

        # ── Moneyline ──────────────────────────────────────────────────────
        # If totals already resolved (e.g., "both" type where totals hit),
        # don't double-resolve. But if still PUSH, try moneyline.
        if is_ml and verdict == "PUSH":
            home_name = str(entry.get("home_team", ""))
            away_name = str(entry.get("away_team", ""))

            if ml_verdict:
                # JSONL format: ml_verdict is team name like "Spurs"
                mv = str(ml_verdict).lower()
                if mv == home_name.lower():
                    verdict = "WIN" if home_win else "LOSS"
                elif mv == away_name.lower():
                    verdict = "WIN" if not home_win else "LOSS"
                else:
                    # Fuzzy match
                    if home_name.lower() in mv:
                        verdict = "WIN" if home_win else "LOSS"
                    elif away_name.lower() in mv:
                        verdict = "WIN" if not home_win else "LOSS"
            elif bet_side:
                # Original format: bet_side like "ML Spurs"
                bs_lower = str(bet_side).lower()
                if home_name.lower() in bs_lower:
                    verdict = "WIN" if home_win else "LOSS"
                elif away_name.lower() in bs_lower:
                    verdict = "WIN" if not home_win else "LOSS"
                elif "ML" in bs_lower:
                    verdict = "WIN" if home_win else "LOSS"

        # ── Profit calculation (standard -110 odds = 1.909 decimal) ────────
        if verdict == "WIN":
            profit = round(stake * 0.909, 2)
        elif verdict == "LOSS":
            profit = -stake
        # else profit stays 0.0 for PUSH

        return {
            "home_score": home_score,
            "away_score": away_score,
            "verdict": verdict,
            "profit": profit,
        }

    # ── INTERNAL: Saving ───────────────────────────────────────────────────

    def _save_resolved_predictions(self):
        """Save resolved predictions back to their source files."""
        # Save back to JSONL files (group by source file)
        jsonl_files: dict[str, list[dict]] = {}
        for entry in self._raw_predictions:
            src = entry.get("source_file", "")
            if src:
                jsonl_files.setdefault(src, []).append(entry)

        for filepath, entries in jsonl_files.items():
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, default=str) + "\n")
            except Exception as e:
                logger.debug(f"Failed to save resolved predictions to {filepath}: {e}")

        # Also update forward_test_results.json if applicable
        ft_entries = [p for p in self._raw_predictions if p.get("source") == "forward_test_results.json"]
        if ft_entries and FORWARD_TEST_JSON.exists():
            try:
                with open(FORWARD_TEST_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Update all_bets with actual results
                bet_map: dict[str, dict] = {}
                for entry in ft_entries:
                    gid = f"{entry.get('matchup', '')}_{entry.get('game_date', '')}"
                    bet_map[gid] = entry

                for bet in data.get("all_bets", []):
                    gid = f"{bet.get('matchup', '')}_{bet.get('game_date', '')}"
                    if gid in bet_map:
                        resolved = bet_map[gid]
                        bet["actual_home_score"] = resolved.get("actual_home_score")
                        bet["actual_away_score"] = resolved.get("actual_away_score")
                        bet["actual_result"] = resolved.get("actual_result")
                        bet["actual_profit"] = resolved.get("actual_profit")

                with open(FORWARD_TEST_JSON, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

                logger.info(f"Updated {len(ft_entries)} entries in forward_test_results.json")
            except Exception as e:
                logger.debug(f"Failed to update forward_test_results.json: {e}")

    # ── INTERNAL: Conversion ───────────────────────────────────────────────

    def _entry_to_resolved_bet(self, entry: dict) -> ResolvedBet:
        """Convert a raw prediction dict to a ResolvedBet dataclass."""
        stake = float(entry.get("stake_dollars", 0.0) or 0)
        if stake == 0:
            stake = float(entry.get("recommended_stake", 0.0) or 0)
        profit = float(entry.get("actual_profit", 0.0) or 0)
        _pid = entry.get("prediction_id") or f"{entry.get('matchup', '?')}_{entry.get('game_date', '?')}_{datetime.now().timestamp()}"

        return ResolvedBet(
            prediction_id=_pid,
            run_id=entry.get("run_id", ""),
            source=entry.get("source", "unknown"),
            source_file=entry.get("source_file", ""),
            game_date=str(entry.get("game_date", ""))[:10],
            matchup=str(entry.get("matchup", "")),
            home_team=str(entry.get("home_team", "")),
            away_team=str(entry.get("away_team", "")),
            league=entry.get("league", "NBA"),
            bet_type=str(entry.get("bet_type", "")),
            bet_side=str(entry.get("bet_side", "")),
            market_line=float(entry.get("market_line", 0) or 0),
            predicted_value=float(entry.get("model_line", 0) or 0),
            edge_pct=float(entry.get("edge_pct", 0) or 0),
            stake_dollars=stake,
            kelly_fraction=float(entry.get("kelly_fraction", 0) or 0),
            confidence=str(entry.get("edge_confidence", "low")),
            model_name=str(entry.get("model_name", "ensemble")),
            predicted_total=entry.get("predicted_total"),
            home_win_prob=entry.get("home_win_prob"),
            actual_home_score=entry.get("actual_home_score"),
            actual_away_score=entry.get("actual_away_score"),
            actual_total=entry.get("actual_total"),
            actual_home_win=bool(entry.get("actual_home_win")) if entry.get("actual_home_win") is not None else None,
            result=str(entry.get("actual_result")) if entry.get("actual_result") else None,
            profit_dollars=profit,
            roi=profit / stake if stake > 0 else 0.0,
            is_clear_pick=bool(entry.get("is_clear_pick", False)),
            resolved_at=str(entry.get("resolved_at", "")),
        )

    # ── INTERNAL: Performance Computation ──────────────────────────────────

    def _compute_strategy_performance(
        self,
        bets: list[ResolvedBet],
        model: str,
        league: str,
        bet_type: str,
    ) -> StrategyPerformance:
        """Compute performance metrics for a group of bets."""
        wins = sum(1 for b in bets if b.result == "WIN")
        losses = sum(1 for b in bets if b.result == "LOSS")
        pushes = sum(1 for b in bets if b.result == "PUSH")
        total_stake = sum(b.stake_dollars for b in bets)
        total_profit = sum(b.profit_dollars for b in bets)
        total_bets = wins + losses + pushes

        edges = [b.edge_pct for b in bets if b.edge_pct != 0]
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        # Date range
        dates = sorted(set(b.game_date for b in bets if b.game_date))
        last_date = dates[-1] if dates else ""

        # Win rate
        decision_bets = wins + losses
        win_rate = wins / decision_bets if decision_bets > 0 else 0.0

        # ROI
        roi = total_profit / total_stake if total_stake > 0 else 0.0

        # Sharpe-like ratio: profit / (std of profits * sqrt(n))
        profits = [b.profit_dollars for b in bets]
        avg_profit = sum(profits) / len(profits) if profits else 0.0
        variance = sum((p - avg_profit) ** 2 for p in profits) / len(profits) if len(profits) > 1 else 0.0
        std_profit = math.sqrt(variance) if variance > 0 else 1.0
        sharpe = (avg_profit / std_profit) * math.sqrt(len(profits)) if std_profit > 0 else 0.0

        # Trailing profits (only last TRAILING_WINDOW_DAYS)
        cutoff = (datetime.now() - timedelta(days=TRAILING_WINDOW_DAYS)).strftime("%Y-%m-%d")
        trailing_bets = [b for b in bets if b.game_date >= cutoff]
        trailing_profits = [b.profit_dollars for b in trailing_bets]
        trailing_stake = sum(b.stake_dollars for b in trailing_bets)
        trailing_profit = sum(trailing_profits)
        trailing_roi = trailing_profit / trailing_stake if trailing_stake > 0 else 0.0

        # Alert if trailing ROI < threshold and enough bets
        is_alerted = (
            total_bets >= MIN_BETS_FOR_ALERT
            and trailing_roi < ALERT_ROI_THRESHOLD
        )

        name = f"{model}/{league}/{bet_type}"
        return StrategyPerformance(
            strategy_name=name,
            model=model,
            league=league,
            bet_type=bet_type,
            n_bets=total_bets,
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=win_rate,
            total_stake=total_stake,
            total_profit=total_profit,
            roi=roi,
            avg_edge=avg_edge,
            sharpe=round(sharpe, 2),
            trailing_profits=trailing_profits,
            is_alerted=is_alerted,
            last_bet_date=last_date,
        )

    # ── INTERNAL: Alert Logging ────────────────────────────────────────────

    def _log_alert(self, strategy: StrategyPerformance):
        """Log an alert to the alerts JSONL file."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "strategy_name": strategy.strategy_name,
            "roi": round(strategy.roi, 4),
            "n_bets": strategy.n_bets,
            "wins": strategy.wins,
            "losses": strategy.losses,
            "total_profit": round(strategy.total_profit, 2),
            "trailing_profit": round(sum(strategy.trailing_profits), 2),
            "alert_type": "underperformance",
            "threshold": ALERT_ROI_THRESHOLD,
            "model": strategy.model,
            "league": strategy.league,
            "bet_type": strategy.bet_type,
        }

        ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(ALERTS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log alert: {e}")

        logger.warning(
            f"\n  STRATEGY ALERT: {strategy.strategy_name}\n"
            f"   ROI: {strategy.roi:.1%} (threshold: {ALERT_ROI_THRESHOLD:.0%})\n"
            f"   Bets: {strategy.n_bets} | Profit: ${strategy.total_profit:.0f}\n"
            f"   Trailing 30d Profit: ${sum(strategy.trailing_profits):.0f}\n"
            f"   Consider pausing or re-tuning this strategy.\n"
        )

