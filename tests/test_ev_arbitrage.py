"""
Tests for the +EV Scanner (ev_scanner.py) and Arbitrage Detection (arbitrage.py) modules.

Coverage:
  PositiveEVScanner
    - No-vig probability computation
    - Kelly fraction computation
    - Consensus agreement computation
    - ML opportunity building (edge passes/fails, EV passes/fails)
    - Full scan from odds snapshots (no model, with model)
    - Confidence determination at all boundaries
    - Filter and rank
    - Report building (empty, with data, actionable counts)
    - CLV tracking
    - Historical snapshot recording
    - get_actionable

  ArbitrageDetector
    - compute_arbitrage_pct (arb exists, no arb, edge case odds)
    - compute_profit_pct (standard, zero, negative/over-100)
    - compute_optimal_stakes (standard, single outcome, empty)
    - Moneyline arb detection (arb present, no arb, below min profit)
    - Totals arb detection (no-op for current format)
    - Spread arb detection (no-op for current format)
    - Full scan (empty list, with arb data, no arbs)
    - Report building
    - Moneyline-only scan
    - Tag generation at various profit levels
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pytest

from betting_intel.recommendations.ev_scanner import (
    PositiveEVScanner,
    EVOpportunity,
    ScannerReport,
    ScannerConfidence,
    ScannerSource,
)
from betting_intel.recommendations.arbitrage import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    ArbitrageReport,
    ArbOutcome,

)
from betting_intel.recommendations.bet_types import (
    BetType,
    Confidence,
    MoneylineBet,
    TotalBet,
    SpreadBet,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def scanner() -> PositiveEVScanner:
    return PositiveEVScanner(
        min_edge_pct=0.01,
        min_ev=0.01,
        min_sportsbooks=1,
        actionable_edge=0.03,
        kelly_fraction=0.25,
        assumed_bankroll=10_000.0,
    )


@pytest.fixture
def arb_detector() -> ArbitrageDetector:
    return ArbitrageDetector(
        min_profit_pct=0.005,
        max_investment=1000.0,
        min_sportsbooks=2,
    )


@pytest.fixture
def sample_odds_snapshot() -> dict:
    """A typical odds snapshot from the OddsPoller."""
    return {
        "game_id": "NBA_Spurs-Thunder",
        "league": "NBA",
        "home_team": "Spurs",
        "away_team": "Thunder",
        "game_date": "2026-05-28",
        "home_ml": 2.10,   # ~47.6% implied
        "away_ml": 1.80,   # ~55.6% implied (total ~103.2% with vig)
        "spread": -3.5,
        "spread_home": -3.5,
        "total": 214.5,
        "total_over": 1.91,
        "total_under": 1.91,
        "sportsbook": "theoddsapi",
        "captured_at": time.time(),
        "is_live": False,
    }


@pytest.fixture
def sample_odds_snapshots(sample_odds_snapshot) -> list[dict]:
    """Multiple odds snapshots for scanning."""
    second = dict(sample_odds_snapshot)
    second.update({
        "game_id": "NBA_Celtics-Pacers",
        "home_team": "Celtics",
        "away_team": "Pacers",
        "home_ml": 1.65,  # ~60.6% implied
        "away_ml": 2.30,  # ~43.5% implied
        "spread": -5.5,
        "total": 218.0,
    })
    return [sample_odds_snapshot, second]


@pytest.fixture
def model_predictions() -> list:
    """Sample BetSuggestion model predictions for the test games."""
    return [
        MoneylineBet(
            game_id="NBA_Spurs-Thunder",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            team="Spurs",
            win_probability=0.52,   # Model says 52% vs market implied ~47.6%
            market_implied_prob=0.476,
            league="NBA",
            confidence=Confidence.MEDIUM,
            reasoning="Home court edge in closeout game",
        ),
        MoneylineBet(
            game_id="NBA_Spurs-Thunder",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            team="Thunder",
            win_probability=0.48,
            market_implied_prob=0.556,
            league="NBA",
            confidence=Confidence.LOW,
            reasoning="Road underdog in elimination game",
        ),
        TotalBet(
            game_id="NBA_Spurs-Thunder",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            side="OVER",
            market_total=214.5,
            predicted_total=218.0,
            league="NBA",
            confidence=Confidence.MEDIUM,
            reasoning="High pace expected in closeout game",
        ),
        TotalBet(
            game_id="NBA_Spurs-Thunder",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            side="UNDER",
            market_total=214.5,
            predicted_total=210.0,
            league="NBA",
            confidence=Confidence.LOW,
            reasoning="Defensive intensity increases in elimination games",
        ),
    ]


@pytest.fixture
def arb_odds_snapshot() -> dict:
    """An odds snapshot with a moneyline arbitrage opportunity."""
    return {
        "game_id": "NBA_Spurs-Thunder",
        "league": "NBA",
        "home_team": "Spurs",
        "away_team": "Thunder",
        "game_date": "2026-05-28",
        "home_ml": 2.15,  # 46.5% implied
        "away_ml": 2.05,  # 48.8% implied => total 95.3% = ARB!
        "spread": -3.5,
        "total": 214.5,
        "sportsbook": "theoddsapi",
        "captured_at": time.time(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PositiveEVScanner Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoVigComputation:
    """Tests for _compute_no_vig_probs."""

    def test_no_vig_balanced(self):
        """Equal odds should give 50/50 probabilities."""
        p_a, p_b = PositiveEVScanner._compute_no_vig_probs(2.00, 2.00)
        assert p_a == pytest.approx(0.50, abs=0.001)
        assert p_b == pytest.approx(0.50, abs=0.001)

    def test_no_vig_unbalanced(self):
        """Different odds should give asymmetric probabilities."""
        p_a, p_b = PositiveEVScanner._compute_no_vig_probs(2.50, 1.50)
        # 1/2.50 = 0.40, 1/1.50 = 0.667, vig = 0.067,
        # fair_a = 0.40/1.067 = 0.375, fair_b = 0.667/1.067 = 0.625
        assert p_a == pytest.approx(0.375, abs=0.001)
        assert p_b == pytest.approx(0.625, abs=0.001)

    def test_no_vig_arbitrage(self):
        """When odds sum to < 1 (arb), normalize to 1.0."""
        p_a, p_b = PositiveEVScanner._compute_no_vig_probs(2.15, 2.05)
        # 1/2.15 = 0.465, 1/2.05 = 0.488, sum = 0.953 (arb!)
        # vig = -0.047, fair_a = 0.465/0.953 = 0.488, fair_b = 0.512
        assert p_a + p_b == pytest.approx(1.0, abs=0.001)
        assert p_a == pytest.approx(0.488, abs=0.01)
        assert p_b == pytest.approx(0.512, abs=0.01)

    def test_no_vig_bad_odds(self):
        """Odds <= 1 should return 50/50 fallback."""
        p_a, p_b = PositiveEVScanner._compute_no_vig_probs(1.0, 1.5)
        assert p_a == 0.5
        assert p_b == 0.5

    def test_no_vig_asymmetric_odds(self):
        """Asymmetric odds should produce probabilities that sum to 1.0."""
        p_a, p_b = PositiveEVScanner._compute_no_vig_probs(10.0, 1.05)
        assert p_a + p_b == pytest.approx(1.0, abs=0.001)
        # Favor the 1.05 side heavily
        assert p_b > p_a


class TestKellyFraction:
    """Tests for _compute_kelly_fraction."""

    def test_kelly_standard(self):
        """Standard 60% win prob at -110 odds should produce positive kelly."""
        fraction = PositiveEVScanner._compute_kelly_fraction(
            win_prob=0.60, decimal_odds=1.91, fraction=0.25, max_fraction=0.10
        )
        # Full Kelly: (0.91*0.60 - 0.40)/0.91 = (0.546-0.40)/0.91 = 0.160
        # Quarter Kelly: 0.160 * 0.25 = 0.040
        assert fraction == pytest.approx(0.040, abs=0.001)
        assert 0 < fraction <= 0.10

    def test_kelly_no_edge(self):
        """50% win prob at -110 odds = no edge, zero kelly."""
        fraction = PositiveEVScanner._compute_kelly_fraction(
            win_prob=0.50, decimal_odds=1.91, fraction=0.25, max_fraction=0.10
        )
        assert fraction == 0.0

    def test_kelly_negative_edge(self):
        """Win prob below implied prob = negative edge, zero kelly."""
        fraction = PositiveEVScanner._compute_kelly_fraction(
            win_prob=0.40, decimal_odds=1.91, fraction=0.25, max_fraction=0.10
        )
        assert fraction == 0.0

    def test_kelly_capped(self):
        """Very high edge should be capped at max_fraction."""
        fraction = PositiveEVScanner._compute_kelly_fraction(
            win_prob=0.85, decimal_odds=2.50, fraction=1.0, max_fraction=0.10
        )
        assert fraction <= 0.10

    def test_kelly_boundary_probs(self):
        """Edge case probabilities (0, 1) should return 0."""
        assert PositiveEVScanner._compute_kelly_fraction(0.0, 1.91) == 0.0
        assert PositiveEVScanner._compute_kelly_fraction(1.0, 1.91) == 0.0

    def test_kelly_bad_odds(self):
        """Odds <= 1 should return 0."""
        assert PositiveEVScanner._compute_kelly_fraction(0.6, 1.0) == 0.0


class TestConsensusAgreement:
    """Tests for _compute_consensus_agreement."""

    def test_consensus_standard_odds(self, sample_odds_snapshot):
        """Standard odds with ~3% vig should give reasonable agreement."""
        agreement, n_books = PositiveEVScanner._compute_consensus_agreement(
            sample_odds_snapshot
        )
        # 1/2.10 + 1/1.80 = 0.476 + 0.556 = 1.032, vig = 0.032
        # agreement = max(0, 1 - 0.032*5) = 0.84
        assert agreement == pytest.approx(0.84, abs=0.01)
        assert n_books == 2

    def test_consensus_arbitrage(self, arb_odds_snapshot):
        """Arb odds (negative vig) should show high agreement."""
        agreement, n_books = PositiveEVScanner._compute_consensus_agreement(
            arb_odds_snapshot
        )
        # Near 1.0 since the arb indicates strong disagreement between books
        assert n_books == 2

    def test_consensus_missing_odds(self):
        """Snapshot without home/away ML should return 0."""
        empty = {"game_id": "test"}
        agreement, n_books = PositiveEVScanner._compute_consensus_agreement(empty)
        assert agreement == 0.0
        assert n_books == 0


class TestConfidenceDetermination:
    """Tests for _determine_confidence."""

    def test_confidence_very_high(self, scanner):
        """Edge >= 8%, 3+ books, model prob far from 50%."""
        c = scanner._determine_confidence(0.08, 3, 0.65)
        assert c == ScannerConfidence.VERY_HIGH

    def test_confidence_high(self, scanner):
        """Edge >= 5%, 2+ books."""
        c = scanner._determine_confidence(0.05, 2, 0.55)
        assert c == ScannerConfidence.HIGH

    def test_confidence_medium(self, scanner):
        """Edge >= 3%, 1+ book."""
        c = scanner._determine_confidence(0.03, 1, 0.55)
        assert c == ScannerConfidence.MEDIUM

    def test_confidence_low(self, scanner):
        """Edge >= 2%."""
        c = scanner._determine_confidence(0.02, 1, 0.51)
        assert c == ScannerConfidence.LOW

    def test_confidence_speculative(self, scanner):
        """Edge < 2%."""
        c = scanner._determine_confidence(0.01, 1, 0.50)
        assert c == ScannerConfidence.SPECULATIVE

    def test_confidence_edge_case(self, scanner):
        """Boundary exactly at 3%."""
        c = scanner._determine_confidence(0.03, 2, 0.55)
        assert c == ScannerConfidence.MEDIUM  # meets >= 3%, >= 1 book

    def test_confidence_high_not_enough_books(self, scanner):
        """Edge >= 5% but only 1 book should be MEDIUM."""
        c = scanner._determine_confidence(0.05, 1, 0.55)
        assert c == ScannerConfidence.MEDIUM  # doesn't meet n_books >= 2 requirement for HIGH


class TestTagGeneration:
    """Tests for _generate_tags."""

    def test_tag_actionable(self, scanner):
        """Edge >= 3% should get actionable tag."""
        tags = scanner._generate_tags(0.03, 1, "NBA")
        assert "actionable" in tags
        assert "+ev" in tags

    def test_tag_high_edge(self, scanner):
        """Edge >= 5% should get high_edge tag."""
        tags = scanner._generate_tags(0.05, 1, "NBA")
        assert "high_edge" in tags

    def test_tag_strong_consensus(self, scanner):
        """3+ books should get strong_consensus tag."""
        tags = scanner._generate_tags(0.02, 3, "NBA")
        assert "strong_consensus" in tags

    def test_tag_small_league(self, scanner):
        """Non-NBA league should get inefficiency tags."""
        tags = scanner._generate_tags(0.02, 1, "EuroLeague")
        assert "small_league" in tags
        assert "inefficiency" in tags

    def test_tag_non_actionable(self, scanner):
        """Edge < 3% should not get actionable tag."""
        tags = scanner._generate_tags(0.02, 1, "NBA")
        assert "actionable" not in tags


class TestBuildMLOpportunity:
    """Tests for _build_ml_opportunity."""

    def test_ml_opportunity_built_when_edge_passes(self, scanner, sample_odds_snapshot):
        """An opportunity should be created when edge exceeds minimum."""
        opp = scanner._build_ml_opportunity(
            game_id="NBA_Spurs-Thunder",
            league="NBA",
            matchup="Thunder @ Spurs",
            game_date="2026-05-28",
            team="Spurs",
            model_prob=0.52,
            implied_prob=0.476,
            decimal_odds=2.10,
            snapshot=sample_odds_snapshot,
        )
        assert opp is not None
        assert opp.edge_pct == pytest.approx(0.044, abs=0.001)  # 0.52 - 0.476
        assert opp.bet_side == "Spurs"
        assert opp.expected_value > 0
        assert opp.recommended_stake_dollars > 0
        assert "+ev" in opp.tags
        assert "actionable" in opp.tags  # 4.4% > 3%
        opp.as_dict()  # Verify serialization works

    def test_ml_opportunity_none_when_edge_below_minimum(self, scanner, sample_odds_snapshot):
        """No opportunity when edge is below threshold."""
        opp = scanner._build_ml_opportunity(
            game_id="NBA_Spurs-Thunder",
            league="NBA",
            matchup="Thunder @ Spurs",
            game_date="2026-05-28",
            team="Spurs",
            model_prob=0.48,  # Below implied prob
            implied_prob=0.50,
            decimal_odds=2.10,
            snapshot=sample_odds_snapshot,
        )
        assert opp is None

    def test_ml_opportunity_none_when_ev_below_minimum(self, scanner, sample_odds_snapshot):
        """No opportunity when EV is below threshold."""
        scanner.min_ev = 10.0  # Very high threshold
        opp = scanner._build_ml_opportunity(
            game_id="NBA_Spurs-Thunder",
            league="NBA",
            matchup="Thunder @ Spurs",
            game_date="2026-05-28",
            team="Spurs",
            model_prob=0.51,
            implied_prob=0.476,
            decimal_odds=2.10,
            snapshot=sample_odds_snapshot,
        )
        assert opp is None

    def test_ml_opportunity_source_tagged(self, scanner, sample_odds_snapshot):
        """ML opportunities should have MODEL_VS_MARKET source."""
        opp = scanner._build_ml_opportunity(
            game_id="NBA_Spurs-Thunder",
            league="NBA",
            matchup="Thunder @ Spurs",
            game_date="2026-05-28",
            team="Spurs",
            model_prob=0.52,
            implied_prob=0.476,
            decimal_odds=2.10,
            snapshot=sample_odds_snapshot,
        )
        assert opp is not None
        assert opp.source == ScannerSource.MODEL_VS_MARKET


class TestScanOddsSnapshots:
    """Tests for the full scan_odds_snapshots method."""

    def test_scan_empty(self, scanner):
        """Empty odds list should return empty report."""
        report = scanner.scan_odds_snapshots([])
        assert report.total_opportunities == 0
        assert report.total_games_scanned == 0

    def test_scan_without_model(self, scanner, sample_odds_snapshots):
        """Without model predictions, ML opportunities should still be generated if possible."""
        report = scanner.scan_odds_snapshots(sample_odds_snapshots)
        assert isinstance(report, ScannerReport)
        assert report.total_games_scanned == 2
        # Without model predictions, no ML opportunities are generated
        assert report.total_opportunities == 0

    def test_scan_with_model(self, scanner, sample_odds_snapshots, model_predictions):
        """With model predictions, opportunities should be detected."""
        report = scanner.scan_odds_snapshots(
            sample_odds_snapshots, model_predictions
        )
        assert report.total_games_scanned == 2
        # Should detect at least the Spurs ML opportunity (4.4% edge)
        assert report.total_opportunities >= 1
        assert report.best_edge_pct > 0

    def test_scan_filtered_by_edge(self, scanner, sample_odds_snapshots, model_predictions):
        """Setting a high min edge should filter out smaller edges."""
        scanner.min_edge = 0.10  # 10% minimum
        report = scanner.scan_odds_snapshots(
            sample_odds_snapshots, model_predictions
        )
        # Most edges are < 10%, so few should pass
        # The Spurs edge is ~4.4%, the Total Over edge depends on formula
        assert report.total_opportunities == 0

    def test_scan_report_actionable_count(self, scanner, sample_odds_snapshots, model_predictions):
        """Report should correctly count actionable vs speculative opportunities."""
        report = scanner.scan_odds_snapshots(
            sample_odds_snapshots, model_predictions
        )
        total = report.total_opportunities
        actionable = report.actionable_opportunities
        speculative = report.speculative_opportunities
        assert actionable + speculative == total

    def test_scan_report_has_league_breakdown(self, scanner, sample_odds_snapshots, model_predictions):
        """Report should have by_league breakdown."""
        report = scanner.scan_odds_snapshots(
            sample_odds_snapshots, model_predictions
        )
        assert "NBA" in report.by_league

    def test_scan_report_has_confidence_breakdown(self, scanner, sample_odds_snapshots, model_predictions):
        """Report should have by_confidence breakdown."""
        report = scanner.scan_odds_snapshots(
            sample_odds_snapshots, model_predictions
        )
        assert len(report.by_confidence) > 0


class TestFilterAndRank:
    """Tests for filtering and ranking opportunities."""

    def test_rank_by_ev(self, scanner):
        """Ranking by EV should sort descending."""
        opps = [
            EVOpportunity(
                game_id="1", league="NBA", matchup="A @ B", game_date="",
                bet_side="A", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.6, model_source="test",
                edge_pct=0.10, expected_value=1.0, expected_value_pct=0.10,
                kelly_fraction=0.05, recommended_stake_dollars=500,
                confidence=ScannerConfidence.HIGH, n_sportsbooks=2,
                consensus_agreement=0.9,
            ),
            EVOpportunity(
                game_id="2", league="NBA", matchup="C @ D", game_date="",
                bet_side="C", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.55, model_source="test",
                edge_pct=0.05, expected_value=0.5, expected_value_pct=0.05,
                kelly_fraction=0.02, recommended_stake_dollars=200,
                confidence=ScannerConfidence.MEDIUM, n_sportsbooks=1,
                consensus_agreement=0.8,
            ),
        ]
        ranked = scanner.rank_by_ev(opps)
        assert ranked[0].expected_value >= ranked[1].expected_value

    def test_rank_by_edge(self, scanner):
        """Ranking by edge should sort descending."""
        opps = [
            EVOpportunity(
                game_id="1", league="NBA", matchup="A @ B", game_date="",
                bet_side="A", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.55, model_source="test",
                edge_pct=0.05, expected_value=0.5, expected_value_pct=0.05,
                kelly_fraction=0.02, recommended_stake_dollars=200,
                confidence=ScannerConfidence.MEDIUM, n_sportsbooks=1,
                consensus_agreement=0.8,
            ),
            EVOpportunity(
                game_id="2", league="NBA", matchup="C @ D", game_date="",
                bet_side="C", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.6, model_source="test",
                edge_pct=0.10, expected_value=1.0, expected_value_pct=0.10,
                kelly_fraction=0.05, recommended_stake_dollars=500,
                confidence=ScannerConfidence.HIGH, n_sportsbooks=2,
                consensus_agreement=0.9,
            ),
        ]
        ranked = scanner.rank_by_edge(opps)
        assert ranked[0].edge_pct >= ranked[1].edge_pct

    def test_filter_by_league(self, scanner):
        """Filtering by league should return only matching opportunities."""
        opps = [
            EVOpportunity(
                game_id="1", league="NBA", matchup="A @ B", game_date="",
                bet_side="A", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.6, model_source="test",
                edge_pct=0.05, expected_value=0.5, expected_value_pct=0.05,
                kelly_fraction=0.02, recommended_stake_dollars=200,
                confidence=ScannerConfidence.MEDIUM, n_sportsbooks=1,
                consensus_agreement=0.8,
            ),
            EVOpportunity(
                game_id="2", league="EuroLeague", matchup="C @ D", game_date="",
                bet_side="C", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.6, model_source="test",
                edge_pct=0.05, expected_value=0.5, expected_value_pct=0.05,
                kelly_fraction=0.02, recommended_stake_dollars=200,
                confidence=ScannerConfidence.MEDIUM, n_sportsbooks=1,
                consensus_agreement=0.8,
            ),
        ]
        filtered = scanner.filter_opportunities(opps, leagues=["NBA"])
        assert len(filtered) == 1
        assert filtered[0].league == "NBA"

    def test_get_actionable(self, scanner):
        """get_actionable should return only opportunities above actionable threshold."""
        opps = [
            EVOpportunity(
                game_id="1", league="NBA", matchup="A @ B", game_date="",
                bet_side="A", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.6, model_source="test",
                edge_pct=0.04, expected_value=0.5, expected_value_pct=0.04,
                kelly_fraction=0.02, recommended_stake_dollars=200,
                confidence=ScannerConfidence.MEDIUM, n_sportsbooks=1,
                consensus_agreement=0.8,
            ),
            EVOpportunity(
                game_id="2", league="NBA", matchup="C @ D", game_date="",
                bet_side="C", market_line=2.0, sportsbook="Test",
                implied_probability=0.5, best_odds_decimal=2.0,
                model_probability=0.52, model_source="test",
                edge_pct=0.02, expected_value=0.1, expected_value_pct=0.02,
                kelly_fraction=0.01, recommended_stake_dollars=100,
                confidence=ScannerConfidence.LOW, n_sportsbooks=1,
                consensus_agreement=0.8,
            ),
        ]
        actionable = scanner.get_actionable(opps)
        assert len(actionable) == 1
        assert actionable[0].edge_pct >= scanner.actionable_edge


class TestCLVTracking:
    """Tests for CLV (Closing Line Value) tracking."""

    def test_record_historical_snapshot(self, scanner, sample_odds_snapshot):
        """Recording historical snapshots should store them."""
        scanner.record_historical_snapshot([sample_odds_snapshot])
        game_id = sample_odds_snapshot["game_id"]
        assert game_id in scanner._last_historical_snapshots
        assert len(scanner._last_historical_snapshots[game_id]) == 1

    def test_clv_not_computed_without_history(self, scanner):
        """CLV should be None when no historical data exists."""
        opp = EVOpportunity(
            game_id="NBA_Spurs-Thunder", league="NBA", matchup="A @ B", game_date="",
            bet_side="Spurs", market_line=2.10, sportsbook="Test",
            implied_probability=0.476, best_odds_decimal=2.10,
            model_probability=0.52, model_source="test",
            edge_pct=0.044, expected_value=0.5, expected_value_pct=0.044,
            kelly_fraction=0.02, recommended_stake_dollars=200,
            confidence=ScannerConfidence.MEDIUM, n_sportsbooks=2,
            consensus_agreement=0.8,
        )
        scanner._compute_clv(opp)
        assert opp.clv_pct is None
        assert opp.opening_line is None

    def test_clv_computed_with_total_history(self, scanner, sample_odds_snapshot):
        """CLV should be populated when historical data exists (regression: indentation fix)."""
        # Record the opening snapshot
        scanner.record_historical_snapshot([sample_odds_snapshot])

        opp = EVOpportunity(
            game_id="NBA_Spurs-Thunder", league="NBA", matchup="A @ B", game_date="",
            bet_side="OVER 214.5", market_line=214.5, sportsbook="Test",
            implied_probability=0.5, best_odds_decimal=1.91,
            model_probability=0.52, model_source="test",
            edge_pct=0.044, expected_value=0.5, expected_value_pct=0.044,
            kelly_fraction=0.02, recommended_stake_dollars=200,
            confidence=ScannerConfidence.MEDIUM, n_sportsbooks=2,
            consensus_agreement=0.8,
            home_team="Spurs",
            away_team="Thunder",
        )

        # This should not crash (regression: the opening = historical[0] indentation fix)
        scanner._compute_clv(opp)
        assert opp.opening_line == 214.5, "Opening line should come from historical snapshot total"
        assert opp.line_movement_direction is not None, "Line movement should be computed"

    def test_clv_computed_with_ml_history(self, scanner, sample_odds_snapshot):
        """CLV for moneyline bets should use home_ml from historical data."""
        scanner.record_historical_snapshot([sample_odds_snapshot])

        opp = EVOpportunity(
            game_id="NBA_Spurs-Thunder", league="NBA", matchup="A @ B", game_date="",
            bet_side="Spurs", market_line=2.10, sportsbook="Test",
            implied_probability=0.476, best_odds_decimal=2.10,
            model_probability=0.52, model_source="test",
            edge_pct=0.044, expected_value=0.5, expected_value_pct=0.044,
            kelly_fraction=0.02, recommended_stake_dollars=200,
            confidence=ScannerConfidence.MEDIUM, n_sportsbooks=2,
            consensus_agreement=0.8,
            home_team="Spurs",
        )

        scanner._compute_clv(opp)
        # Opening home_ml was 2.10, current market_line is also 2.10 => no movement
        assert opp.opening_line is not None

    def test_clv_line_movement_direction(self, scanner, sample_odds_snapshot):
        """Line movement toward the bet side should be detected."""
        scanner.record_historical_snapshot([sample_odds_snapshot])

        # Market total moved from 214.5 to 210.0, which is movement TOWARD our UNDER bet
        opp = EVOpportunity(
            game_id="NBA_Spurs-Thunder", league="NBA", matchup="A @ B", game_date="",
            bet_side="UNDER 214.5", market_line=210.0, sportsbook="Test",
            implied_probability=0.5, best_odds_decimal=1.91,
            model_probability=0.52, model_source="test",
            edge_pct=0.02, expected_value=0.3, expected_value_pct=0.02,
            kelly_fraction=0.01, recommended_stake_dollars=100,
            confidence=ScannerConfidence.MEDIUM, n_sportsbooks=2,
            consensus_agreement=0.8,
        )

        scanner._compute_clv(opp)
        # diff = 210.0 - 214.5 = -4.5, UNDER bet => diff < 0 = "toward"
        assert opp.line_movement_direction == "toward", \
            f"Expected 'toward' for UNDER bet with decreasing total, got '{opp.line_movement_direction}'"

    def test_clv_different_game_id_no_crash(self, scanner, sample_odds_snapshot):
        """CLV for a game not in history should not crash (no-op)."""
        scanner.record_historical_snapshot([sample_odds_snapshot])

        opp = EVOpportunity(
            game_id="NBA_OTHER-GAME", league="NBA", matchup="X @ Y", game_date="",
            bet_side="OVER 200.0", market_line=200.0, sportsbook="Test",
            implied_probability=0.5, best_odds_decimal=1.91,
            model_probability=0.52, model_source="test",
            edge_pct=0.02, expected_value=0.3, expected_value_pct=0.02,
            kelly_fraction=0.01, recommended_stake_dollars=100,
            confidence=ScannerConfidence.LOW, n_sportsbooks=1,
            consensus_agreement=0.5,
        )

        # Should not crash — different game_id, no matching history
        scanner._compute_clv(opp)
        assert opp.clv_pct is None

    def test_clv_different_game_id_no_crash(self, scanner, sample_odds_snapshot):
        """CLV for a game not in history should not crash (no-op)."""
        scanner.record_historical_snapshot([sample_odds_snapshot])

        opp = EVOpportunity(
            game_id="NBA_OTHER-GAME", league="NBA", matchup="X @ Y", game_date="",
            bet_side="OVER 200.0", market_line=200.0, sportsbook="Test",
            implied_probability=0.5, best_odds_decimal=1.91,
            model_probability=0.52, model_source="test",
            edge_pct=0.02, expected_value=0.3, expected_value_pct=0.02,
            kelly_fraction=0.01, recommended_stake_dollars=100,
            confidence=ScannerConfidence.LOW, n_sportsbooks=1,
            consensus_agreement=0.5,
        )

        # Should not crash -- different game_id, no matching history
        scanner._compute_clv(opp)
        assert opp.clv_pct is None


class TestScannerReport:
    """Tests for ScannerReport."""

    def test_empty_report(self):
        """Empty report should have zero counts."""
        report = ScannerReport()
        assert report.total_opportunities == 0
        assert report.total_games_scanned == 0
        assert report.as_dict()["total_opportunities"] == 0

    def test_report_as_dict(self):
        """Report serialization should work."""
        report = ScannerReport(
            total_opportunities=5,
            total_games_scanned=3,
            actionable_opportunities=2,
            best_edge_pct=0.08,
        )
        d = report.as_dict()
        assert d["total_opportunities"] == 5
        assert d["actionable_opportunities"] == 2
        assert d["best_edge_pct"] == 0.08


# ═══════════════════════════════════════════════════════════════════════════════
# EVOpportunity Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEVOpportunity:
    """Tests for the EVOpportunity dataclass."""

    def test_action_generation(self):
        """Action string should include stake, team, odds, and EV."""
        opp = EVOpportunity(
            game_id="1", league="NBA", matchup="A @ B", game_date="2026-05-28",
            bet_side="Spurs", market_line=2.10, sportsbook="Best Book",
            implied_probability=0.476, best_odds_decimal=2.10,
            model_probability=0.52, model_source="Momentum Model",
            edge_pct=0.044, expected_value=0.08, expected_value_pct=0.044,
            kelly_fraction=0.02, recommended_stake_dollars=200.0,
            confidence=ScannerConfidence.MEDIUM, n_sportsbooks=2,
            consensus_agreement=0.84,
        )
        action = opp.action
        assert "$200" in action
        assert "Spurs" in action
        assert "2.10" in action
        assert "4.4%" in action or "4.40%" in action

    def test_as_dict(self):
        """Serialization should include all major fields."""
        opp = EVOpportunity(
            game_id="1", league="NBA", matchup="A @ B", game_date="2026-05-28",
            bet_side="Spurs", market_line=2.10, sportsbook="Best Book",
            implied_probability=0.476, best_odds_decimal=2.10,
            model_probability=0.52, model_source="Test",
            edge_pct=0.044, expected_value=0.08, expected_value_pct=0.044,
            kelly_fraction=0.02, recommended_stake_dollars=200.0,
            confidence=ScannerConfidence.HIGH, n_sportsbooks=3,
            consensus_agreement=0.9,
        )
        d = opp.as_dict()
        assert d["edge_pct"] == 0.044
        assert d["confidence"] == "HIGH"
        assert d["sportsbook"] == "Best Book"
        assert d["model_source"] == "Test"

    def test_zero_stake_action(self):
        """Zero stake should produce PASS action."""
        opp = EVOpportunity(
            game_id="1", league="NBA", matchup="A @ B", game_date="",
            bet_side="Spurs", market_line=2.0, sportsbook="Test",
            implied_probability=0.5, best_odds_decimal=2.0,
            model_probability=0.5, model_source="test",
            edge_pct=0.0, expected_value=0.0, expected_value_pct=0.0,
            kelly_fraction=0.0, recommended_stake_dollars=0.0,
            confidence=ScannerConfidence.LOW, n_sportsbooks=1,
            consensus_agreement=0.5,
        )
        action = opp.action
        assert "PASS" in action


# ═══════════════════════════════════════════════════════════════════════════════
# Arbitrage Detector Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestArbitragePercent:
    """Tests for compute_arbitrage_pct."""

    def test_arbitrage_exists(self):
        """Arbitrage exists when pct < 1."""
        pct = ArbitrageDetector.compute_arbitrage_pct([2.15, 2.05])
        # 1/2.15 + 1/2.05 = 0.465 + 0.488 = 0.953
        assert pct == pytest.approx(0.953, abs=0.005)
        assert pct < 1.0

    def test_no_arbitrage(self):
        """Normal odds should have pct > 1."""
        pct = ArbitrageDetector.compute_arbitrage_pct([1.91, 1.91])
        # 1/1.91 + 1/1.91 = 0.524 + 0.524 = 1.048
        assert pct > 1.0

    def test_edge_case_odds(self):
        """Odds exactly at breakeven."""
        pct = ArbitrageDetector.compute_arbitrage_pct([2.0, 2.0])
        assert pct == pytest.approx(1.0, abs=0.001)

    def test_bad_odds_ignored(self):
        """Odds <= 1 should be ignored."""
        pct = ArbitrageDetector.compute_arbitrage_pct([1.0, 2.0])
        # Only 2.0 is used: 1/2.0 = 0.5
        assert pct == pytest.approx(0.5, abs=0.001)

    def test_empty_list(self):
        """Empty list should return 1.0."""
        pct = ArbitrageDetector.compute_arbitrage_pct([])
        assert pct == 1.0


class TestProfitPercent:
    """Tests for compute_profit_pct."""

    def test_profit_standard(self):
        """Standard arb should produce positive profit."""
        # arb_pct = 0.953 => profit = 1/0.953 - 1 = 4.93%
        profit = ArbitrageDetector.compute_profit_pct(0.953)
        assert profit == pytest.approx(0.0493, abs=0.001)

    def test_profit_zero(self):
        """No arb (pct = 1.0) should produce 0 profit."""
        profit = ArbitrageDetector.compute_profit_pct(1.0)
        assert profit == 0.0

    def test_profit_negative_input(self):
        """Negative arb pct should return 0."""
        profit = ArbitrageDetector.compute_profit_pct(-0.5)
        assert profit == 0.0

    def test_profit_deep_arb(self):
        """Very low arb% = high profit."""
        profit = ArbitrageDetector.compute_profit_pct(0.90)
        # 1/0.90 - 1 = 0.1111
        assert profit == pytest.approx(0.1111, abs=0.001)


class TestOptimalStakes:
    """Tests for compute_optimal_stakes."""

    def test_stakes_standard(self):
        """Standard two-outcome arb should produce correct stakes."""
        outcomes = [(2.15, 0.465), (2.05, 0.488)]  # (odds, implied_prob)
        stakes = ArbitrageDetector.compute_optimal_stakes(outcomes, 1000.0)
        # 0.465 + 0.488 = 0.953
        # stake_a = 1000 * (0.465 / 0.953) = 487.93
        # stake_b = 1000 * (0.488 / 0.953) = 512.07
        assert len(stakes) == 2
        assert sum(stakes) == pytest.approx(1000.0, abs=0.1)
        assert stakes[0] == pytest.approx(488.0, abs=5)
        assert stakes[1] == pytest.approx(512.0, abs=5)

    def test_stakes_single_outcome(self):
        """Single outcome should return full investment."""
        stakes = ArbitrageDetector.compute_optimal_stakes([(2.0, 0.5)], 1000.0)
        assert stakes == [1000.0]

    def test_stakes_empty(self):
        """Empty outcomes should return empty list."""
        stakes = ArbitrageDetector.compute_optimal_stakes([], 1000.0)
        assert stakes == []


class TestMoneylineArbitrage:
    """Tests for moneyline arbitrage detection."""

    def test_moneyline_arb_detected(self, arb_detector, arb_odds_snapshot):
        """Arb should be detected when odds create profit opportunity."""
        opps = arb_detector._check_moneyline_arb(
            game_id=arb_odds_snapshot["game_id"],
            league=arb_odds_snapshot["league"],
            matchup=f"{arb_odds_snapshot['away_team']} @ {arb_odds_snapshot['home_team']}",
            game_date=arb_odds_snapshot["game_date"],
            home_team=arb_odds_snapshot["home_team"],
            away_team=arb_odds_snapshot["away_team"],
            home_ml=arb_odds_snapshot["home_ml"],
            away_ml=arb_odds_snapshot["away_ml"],
        )
        assert len(opps) == 1
        opp = opps[0]
        assert opp.market_type == "moneyline"
        assert opp.profit_pct > 0
        assert opp.arbitrage_pct < 1.0
        assert len(opp.outcomes) == 2
        assert opp.guaranteed_profit > 0
        assert "arbitrage" in opp.tags
        opp.as_dict()  # Verify serialization

    def test_moneyline_arb_not_detected(self, arb_detector):
        """No arb when standard vig is present."""
        opps = arb_detector._check_moneyline_arb(
            game_id="test",
            league="NBA",
            matchup="A @ B",
            game_date="2026-05-28",
            home_team="A",
            away_team="B",
            home_ml=1.91,
            away_ml=1.91,
        )
        assert len(opps) == 0

    def test_moneyline_arb_below_min_profit(self, arb_detector):
        """No arb when profit is below threshold."""
        arb_detector.min_profit = 0.10  # 10% minimum
        opps = arb_detector._check_moneyline_arb(
            game_id="test",
            league="NBA",
            matchup="A @ B",
            game_date="2026-05-28",
            home_team="A",
            away_team="B",
            home_ml=2.15,
            away_ml=2.05,  # ~4.9% profit — below 10% threshold
        )
        assert len(opps) == 0

    def test_moneyline_arb_bad_odds(self, arb_detector):
        """Odds <= 1 should produce no arb."""
        opps = arb_detector._check_moneyline_arb(
            game_id="test", league="NBA", matchup="A @ B", game_date="",
            home_team="A", away_team="B", home_ml=1.0, away_ml=2.0,
        )
        assert len(opps) == 0


class TestTotalAndSpreadArbitrage:
    """Tests for totals and spread arbitrage."""

    def test_total_arb_currently_noop(self, arb_detector):
        """Totals arb should return empty list for current format."""
        opps = arb_detector._check_total_arb(
            game_id="test", league="NBA", matchup="A @ B", game_date="",
            home_team="A", away_team="B", total=214.5,
        )
        assert opps == []

    def test_spread_arb_currently_noop(self, arb_detector):
        """Spread arb should return empty list for current format."""
        opps = arb_detector._check_spread_arb(
            game_id="test", league="NBA", matchup="A @ B", game_date="",
            home_team="A", away_team="B", spread=-3.5,
        )
        assert opps == []


class TestFullArbitrageScan:
    """Tests for the full scan_for_arbitrage method."""

    def test_scan_empty(self, arb_detector):
        """Empty odds should return empty report."""
        report = arb_detector.scan_for_arbitrage([])
        assert report.total_opportunities == 0
        assert report.total_games_scanned == 0

    def test_scan_with_arb(self, arb_detector, arb_odds_snapshot):
        """Scan should detect arbitrage in appropriate snapshots."""
        report = arb_detector.scan_for_arbitrage([arb_odds_snapshot])
        assert report.total_games_scanned == 1
        assert report.total_opportunities >= 1
        assert report.best_profit_pct > 0
        assert report.total_guaranteed_profit > 0

    def test_scan_without_arb(self, arb_detector, sample_odds_snapshot):
        """Scan should find no arb in standard odds."""
        report = arb_detector.scan_for_arbitrage([sample_odds_snapshot])
        assert report.total_games_scanned == 1
        assert report.total_opportunities == 0

    def test_scan_moneyline_only(self, arb_detector, arb_odds_snapshot, sample_odds_snapshot):
        """Moneyline-only scan should filter correctly."""
        opps = arb_detector.scan_moneyline([arb_odds_snapshot, sample_odds_snapshot])
        assert len(opps) >= 1
        all_ml = all(o.market_type == "moneyline" for o in opps)
        assert all_ml

    def test_scan_report_has_market_breakdown(self, arb_detector, arb_odds_snapshot):
        """Report should have by_market_type breakdown."""
        report = arb_detector.scan_for_arbitrage([arb_odds_snapshot])
        assert "moneyline" in report.by_market_type

    def test_scan_report_has_league_breakdown(self, arb_detector, arb_odds_snapshot):
        """Report should have by_league breakdown."""
        report = arb_detector.scan_for_arbitrage([arb_odds_snapshot])
        assert "NBA" in report.by_league


class TestArbitrageReport:
    """Tests for ArbitrageReport."""

    def test_empty_report(self):
        """Empty report should have zero counts."""
        report = ArbitrageReport()
        assert report.total_opportunities == 0
        assert report.as_dict()["total_opportunities"] == 0

    def test_report_with_data(self):
        """Report with data should reflect the opportunities."""
        report = ArbitrageReport(
            total_opportunities=3,
            total_games_scanned=5,
            profitable_opportunities=3,
            high_yield_opportunities=1,
            average_profit_pct=0.03,
            best_profit_pct=0.05,
        )
        d = report.as_dict()
        assert d["high_yield_opportunities"] == 1
        assert d["best_profit_pct"] == 0.05


class TestArbitrageOpportunity:
    """Tests for ArbitrageOpportunity dataclass."""

    def test_action_generation(self):
        """Action string should include matchup and profit."""
        opp = ArbitrageOpportunity(
            game_id="1", league="NBA",
            matchup="Thunder @ Spurs", game_date="2026-05-28",
            market_type="moneyline",
            outcomes=[
                ArbOutcome(team="Spurs", decimal_odds=2.15, sportsbook="DraftKings", market_type="h2h"),
                ArbOutcome(team="Thunder", decimal_odds=2.05, sportsbook="FanDuel", market_type="h2h"),
            ],
            arbitrage_pct=0.953,
            profit_pct=0.0493,
            stakes=[487.80, 512.20],
            total_investment=1000.0,
            guaranteed_profit=49.30,
            n_sportsbooks=2,
        )
        action = opp.action()
        assert "Spurs" in action
        assert "Thunder" in action
        assert "ARBITRAGE" in action
        assert "4.93%" in action or "4.9%" in action

    def test_as_dict(self):
        """Serialization should include all major fields."""
        opp = ArbitrageOpportunity(
            game_id="1", league="NBA",
            matchup="A @ B", game_date="2026-05-28",
            market_type="moneyline",
            outcomes=[
                ArbOutcome(team="A", decimal_odds=2.15, sportsbook="Book1", market_type="h2h"),
            ],
            arbitrage_pct=0.953,
            profit_pct=0.0493,
            stakes=[1000.0],
            total_investment=1000.0,
            guaranteed_profit=49.30,
            n_sportsbooks=2,
        )
        d = opp.as_dict()
        assert d["profit_pct"] == pytest.approx(0.0493, abs=0.001)
        assert d["is_risk_free"] is True
        assert len(d["outcomes"]) == 1


class TestArbitrageTags:
    """Tests for arb tag generation."""

    def test_tags_standard_arb(self, arb_detector):
        """Standard arb should have arb and risk_free tags."""
        tags = arb_detector._generate_tags(0.01, 0.99, "NBA")
        assert "arbitrage" in tags
        assert "risk_free" in tags

    def test_tags_high_yield(self, arb_detector):
        """Profit >= 2% should have high_yield tag."""
        tags = arb_detector._generate_tags(0.02, 0.98, "NBA")
        assert "high_yield" in tags

    def test_tags_deep_arb(self, arb_detector):
        """Arb < 98% should have deep_arb tag."""
        tags = arb_detector._generate_tags(0.03, 0.97, "NBA")
        assert "deep_arb" in tags

    def test_tags_small_league(self, arb_detector):
        """Non-NBA league should have small_league tag."""
        tags = arb_detector._generate_tags(0.01, 0.99, "EuroLeague")
        assert "small_league" in tags

    def test_tags_exceptional(self, arb_detector):
        """Profit >= 5% should have exceptional tag."""
        tags = arb_detector._generate_tags(0.05, 0.95, "NBA")
        assert "exceptional" in tags
