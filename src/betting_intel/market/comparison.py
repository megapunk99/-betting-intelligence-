"""
Model vs Market Comparison — Phase 3.12 of the Professional Betting Intelligence Platform.

For every game, calculate:
    model_probability — what our model thinks
    market_probability — what the market implies (vig-free)
    edge — difference = model - market

Only recommend bets exceeding minimum edge threshold.

Also tracks:
    - How often model agrees with market (consensus rate)
    - When model disagrees significantly (contrarian signals)
    - Historical model vs market accuracy
"""

import json
import sqlite3
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Model vs market comparison for a single side/market of a game."""
    game_id: str
    home_team: str
    away_team: str
    market_type: str           # 'moneyline', 'spread', 'total'
    bet_side: str              # 'home', 'away', 'over', 'under'

    model_probability: float = 0.0
    market_probability: float = 0.0
    vig_free_probability: Optional[float] = None

    edge: float = 0.0          # model - market (positive = model thinks it's undervalued)
    edge_pct: float = 0.0

    # Comparison
    model_favors: str = ""     # 'home', 'away', 'over', 'under', or 'market' if aligned
    market_favors: str = ""
    agreement: str = "agree"   # 'agree', 'disagree', 'strongly_disagree'

    model_confidence: float = 0.0
    consensus_level: float = 0.0

    # Metadata
    model_name: str = ""
    timestamp: str = ""

    def __post_init__(self):
        self.edge = self.model_probability - self.market_probability
        self.edge_pct = self.edge

        # Determine agreement
        model_fav = "home" if self.model_probability > 0.5 else "away"
        market_fav = "home" if self.market_probability > 0.5 else "away"

        if self.bet_side == "over":
            model_fav = "over" if self.model_probability > 0.5 else "under"
            market_fav = "over" if self.market_probability > 0.5 else "under"
        elif self.bet_side == "under":
            model_fav = "under" if self.model_probability > 0.5 else "over"
            market_fav = "under" if self.market_probability > 0.5 else "over"

        self.model_favors = model_fav
        self.market_favors = market_fav

        edge_abs = abs(self.edge)
        if edge_abs < 0.02:
            self.agreement = "agree"
        elif edge_abs < 0.05:
            self.agreement = "disagree"
        else:
            self.agreement = "strongly_disagree"

    @property
    def is_contrarian(self) -> bool:
        """Model disagrees with market consensus."""
        return self.agreement in ("disagree", "strongly_disagree")


@dataclass
class ComparisonAggregate:
    """Aggregated model vs market stats across all tracked games."""
    total_comparisons: int = 0
    agreement_rate: float = 0.0
    disagreement_rate: float = 0.0
    strong_disagreement_rate: float = 0.0

    avg_edge: float = 0.0
    avg_abs_edge: float = 0.0
    max_edge: float = 0.0
    min_edge: float = 0.0

    contrarian_count: int = 0
    market_aligned_count: int = 0

    total_correct: int = 0      # When model and market agreed, and market was right
    model_correct: int = 0       # When model disagreed with market and was right

    # Per market
    ml_avg_edge: float = 0.0
    spread_avg_edge: float = 0.0
    total_avg_edge: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL vs MARKET COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ModelMarketComparison:
    """
    Compares model predictions against market-implied probabilities.

    For every game, computes:
        model_probability vs market_probability (vig-free)
        edge = model - market
        agreement level

    Usage:
        comparison = ModelMarketComparison(DB_PATH)

        # Compare a single game
        result = comparison.compare_game(
            game_id="game_123",
            home_team="Spurs", away_team="Knicks",
            model_home_prob=0.62,
            home_ml_odds=-110, away_ml_odds=-110,
        )

        # Get aggregate stats
        agg = comparison.get_aggregate()
        print(f"Agreement rate: {agg.agreement_rate:.1%}")
        print(f"Contrarian count: {agg.contrarian_count}")
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    # ═══════════════════════════════════════════════════════════════════
    #  SINGLE GAME COMPARISON
    # ═══════════════════════════════════════════════════════════════════

    def compare_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        model_home_prob: float,
        home_ml_odds: Optional[float] = None,
        away_ml_odds: Optional[float] = None,
        model_total_over_prob: Optional[float] = None,
        over_odds: Optional[float] = None,
        under_odds: Optional[float] = None,
        model_home_cover_prob: Optional[float] = None,
        home_spread_odds: Optional[float] = None,
        away_spread_odds: Optional[float] = None,
        model_name: str = "",
    ) -> List[ComparisonResult]:
        """
        Compare model predictions against market-implied probabilities.

        Args:
            game_id: Game identifier
            home_team, away_team: Team names
            model_home_prob: Model's estimated home win probability (0-1)
            home_ml_odds, away_ml_odds: Moneyline odds in American format
            model_total_over_prob: Model's estimated over probability (0-1)
            over_odds, under_odds: Totals odds
            model_home_cover_prob: Model's estimated home cover probability
            home_spread_odds, away_spread_odds: Spread odds
            model_name: Model identifier

        Returns:
            List of ComparisonResult (one per side/market)
        """
        results = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Moneyline comparison
        if home_ml_odds and away_ml_odds:
            home_market_prob = self._american_to_implied(home_ml_odds)
            away_market_prob = self._american_to_implied(away_ml_odds)

            vig_free_home, vig_free_away = self._vig_free_probs(home_ml_odds, away_ml_odds)

            results.append(ComparisonResult(
                game_id=game_id, home_team=home_team, away_team=away_team,
                market_type="moneyline", bet_side="home",
                model_probability=model_home_prob,
                market_probability=home_market_prob,
                vig_free_probability=vig_free_home,
                model_name=model_name, timestamp=timestamp,
            ))

            results.append(ComparisonResult(
                game_id=game_id, home_team=home_team, away_team=away_team,
                market_type="moneyline", bet_side="away",
                model_probability=1.0 - model_home_prob,
                market_probability=away_market_prob,
                vig_free_probability=vig_free_away,
                model_name=model_name, timestamp=timestamp,
            ))

        # 2. Totals comparison
        if model_total_over_prob is not None and over_odds and under_odds:
            over_market_prob = self._american_to_implied(over_odds)
            under_market_prob = self._american_to_implied(under_odds)
            vf_over, vf_under = self._vig_free_probs(over_odds, under_odds)

            results.append(ComparisonResult(
                game_id=game_id, home_team=home_team, away_team=away_team,
                market_type="total", bet_side="over",
                model_probability=model_total_over_prob,
                market_probability=over_market_prob,
                vig_free_probability=vf_over,
                model_name=model_name, timestamp=timestamp,
            ))

            results.append(ComparisonResult(
                game_id=game_id, home_team=home_team, away_team=away_team,
                market_type="total", bet_side="under",
                model_probability=1.0 - model_total_over_prob,
                market_probability=under_market_prob,
                vig_free_probability=vf_under,
                model_name=model_name, timestamp=timestamp,
            ))

        # 3. Spread comparison
        if model_home_cover_prob is not None and home_spread_odds and away_spread_odds:
            home_market_cover = self._american_to_implied(home_spread_odds)
            away_market_cover = self._american_to_implied(away_spread_odds)
            vf_home, vf_away = self._vig_free_probs(home_spread_odds, away_spread_odds)

            results.append(ComparisonResult(
                game_id=game_id, home_team=home_team, away_team=away_team,
                market_type="spread", bet_side="home",
                model_probability=model_home_cover_prob,
                market_probability=home_market_cover,
                vig_free_probability=vf_home,
                model_name=model_name, timestamp=timestamp,
            ))

            results.append(ComparisonResult(
                game_id=game_id, home_team=home_team, away_team=away_team,
                market_type="spread", bet_side="away",
                model_probability=1.0 - model_home_cover_prob,
                market_probability=away_market_cover,
                vig_free_probability=vf_away,
                model_name=model_name, timestamp=timestamp,
            ))

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  BATCH COMPARISON
    # ═══════════════════════════════════════════════════════════════════

    def compare_batch(self, predictions: List[Dict]) -> List[ComparisonResult]:
        """
        Compare batch of model predictions against market odds.

        Args:
            predictions: List of dicts with game_id, home_team, away_team,
                        model_home_prob, home_ml_odds, away_ml_odds, etc.

        Returns:
            Flat list of ComparisonResult objects
        """
        all_results = []
        for pred in predictions:
            results = self.compare_game(
                game_id=pred.get("game_id", ""),
                home_team=pred.get("home_team", ""),
                away_team=pred.get("away_team", ""),
                model_home_prob=pred.get("model_home_prob", 0.5),
                home_ml_odds=pred.get("home_ml_odds"),
                away_ml_odds=pred.get("away_ml_odds"),
                model_total_over_prob=pred.get("model_total_over_prob"),
                over_odds=pred.get("over_odds"),
                under_odds=pred.get("under_odds"),
                model_home_cover_prob=pred.get("model_home_cover_prob"),
                home_spread_odds=pred.get("home_spread_odds"),
                away_spread_odds=pred.get("away_spread_odds"),
                model_name=pred.get("model_name", "ensemble"),
            )
            all_results.extend(results)
        return all_results

    # ═══════════════════════════════════════════════════════════════════
    #  AGGREGATE STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def get_aggregate(self, results: Optional[List[ComparisonResult]] = None) -> ComparisonAggregate:
        """
        Get aggregate comparison statistics.

        Args:
            results: List of ComparisonResult. If None, loads from database.

        Returns:
            ComparisonAggregate
        """
        if results is None:
            results = self._load_from_db()

        if not results:
            return ComparisonAggregate()

        n = len(results)
        agree_count = sum(1 for r in results if r.agreement == "agree")
        disagree_count = sum(1 for r in results if r.agreement == "disagree")
        strong_disagree = sum(1 for r in results if r.agreement == "strongly_disagree")

        edges = [r.edge for r in results]
        abs_edges = [abs(r.edge) for r in results]

        contrarian = [r for r in results if r.is_contrarian]
        aligned = [r for r in results if not r.is_contrarian]

        # Per market
        ml_edges = [r.edge for r in results if r.market_type == "moneyline"]
        spread_edges = [r.edge for r in results if r.market_type == "spread"]
        total_edges = [r.edge for r in results if r.market_type == "total"]

        return ComparisonAggregate(
            total_comparisons=n,
            agreement_rate=agree_count / n if n > 0 else 0,
            disagreement_rate=disagree_count / n if n > 0 else 0,
            strong_disagreement_rate=strong_disagree / n if n > 0 else 0,
            avg_edge=np.mean(edges) if edges else 0,
            avg_abs_edge=np.mean(abs_edges) if abs_edges else 0,
            max_edge=max(edges) if edges else 0,
            min_edge=min(edges) if edges else 0,
            contrarian_count=len(contrarian),
            market_aligned_count=len(aligned),
            ml_avg_edge=np.mean(ml_edges) if ml_edges else 0,
            spread_avg_edge=np.mean(spread_edges) if spread_edges else 0,
            total_avg_edge=np.mean(total_edges) if total_edges else 0,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════

    def save_results(self, results: List[ComparisonResult]):
        """Save comparison results to database."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_vs_market (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id             TEXT NOT NULL,
                    home_team           TEXT NOT NULL,
                    away_team           TEXT NOT NULL,
                    market_type         TEXT NOT NULL,
                    bet_side            TEXT NOT NULL,
                    model_probability   REAL NOT NULL,
                    market_probability  REAL NOT NULL,
                    vig_free_probability REAL,
                    edge                REAL NOT NULL,
                    agreement           TEXT NOT NULL,
                    model_name          TEXT DEFAULT '',
                    timestamp           TEXT NOT NULL,
                    created_at          TEXT DEFAULT (datetime('now')),
                    UNIQUE(game_id, market_type, bet_side, model_name)
                )
            """)
            for r in results:
                conn.execute(
                    """INSERT OR REPLACE INTO model_vs_market
                       (game_id, home_team, away_team, market_type, bet_side,
                        model_probability, market_probability, vig_free_probability,
                        edge, agreement, model_name, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r.game_id, r.home_team, r.away_team, r.market_type, r.bet_side,
                     r.model_probability, r.market_probability, r.vig_free_probability,
                     r.edge, r.agreement, r.model_name, r.timestamp)
                )
            conn.commit()

    def _load_from_db(self) -> List[ComparisonResult]:
        """Load comparison results from database."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_vs_market ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def _row_to_result(self, row) -> ComparisonResult:
        return ComparisonResult(
            game_id=row["game_id"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            market_type=row["market_type"],
            bet_side=row["bet_side"],
            model_probability=row["model_probability"],
            market_probability=row["market_probability"],
            vig_free_probability=row["vig_free_probability"],
            model_name=row["model_name"],
            timestamp=row["timestamp"],
        )

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _american_to_implied(self, american_odds: float) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100.0 / (american_odds + 100.0)
        return abs(american_odds) / (abs(american_odds) + 100.0)

    def _vig_free_probs(self, odds_a: float, odds_b: float) -> Tuple[float, float]:
        """Remove vig to get true probabilities."""
        p_a = self._american_to_implied(odds_a)
        p_b = self._american_to_implied(odds_b)
        total = p_a + p_b
        if total <= 0:
            return (0.5, 0.5)
        return (p_a / total, p_b / total)

    def format_comparison(self, result: ComparisonResult) -> str:
        """Format a single comparison result for display."""
        icon = "✅" if result.agreement == "agree" else "⚠️" if result.agreement == "disagree" else "🔴"
        return (
            f"{icon} {result.bet_side.upper()} ({result.market_type})\n"
            f"   Model: {result.model_probability:.1%} | "
            f"Market: {result.market_probability:.1%} | "
            f"Edge: {result.edge:+.2%}\n"
            f"   Agreement: {result.agreement}"
        )

    def format_aggregate(self, agg: ComparisonAggregate) -> str:
        """Format aggregate comparison stats for display."""
        return (
            f"📊 MODEL vs MARKET COMPARISON\n"
            f"{'─' * 45}\n"
            f"Total comparisons: {agg.total_comparisons}\n"
            f"Agreement rate:    {agg.agreement_rate:.1%}\n"
            f"Disagreement rate: {agg.disagreement_rate:.1%}\n"
            f"Strong disagree:   {agg.strong_disagreement_rate:.1%}\n"
            f"\n"
            f"Avg edge:    {agg.avg_edge:+.2%}\n"
            f"Avg abs edge: {agg.avg_abs_edge:.2%}\n"
            f"Max edge:    {agg.max_edge:+.2%}\n"
            f"Min edge:    {agg.min_edge:+.2%}\n"
            f"\n"
            f"Contrarian signals: {agg.contrarian_count}\n"
            f"Market aligned:     {agg.market_aligned_count}\n"
            f"\n"
            f"By Market:\n"
            f"  Moneyline: {agg.ml_avg_edge:+.2%}\n"
            f"  Spread:    {agg.spread_avg_edge:+.2%}\n"
            f"  Total:     {agg.total_avg_edge:+.2%}\n"
        )
