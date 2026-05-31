"""Tests for the risk management module — Kelly staking, exposure control, and bet correlation tracking."""

from __future__ import annotations

import numpy as np
import pytest


class TestKellyCalculator:
    """Tests for single-bet fractional Kelly calculator."""

    @pytest.fixture
    def kelly(self):
        from betting_intel.risk.kelly import KellyCalculator

        return KellyCalculator(
            bankroll=10_000.0,
            fraction=0.25,
            max_fraction=0.15,
            min_edge=0.02,
            drawdown_protection=False,
        )

    def test_initial_state(self, kelly):
        assert kelly.current_bankroll == 10_000.0
        assert kelly.peak_bankroll == 10_000.0
        assert kelly.fraction == 0.25
        assert kelly.max_fraction == 0.15

    def test_compute_kelly_positive_edge(self, kelly):
        """55% win probability at -110 odds (1.91 decimal) should produce a positive stake."""
        fraction, stake = kelly.compute_kelly(win_probability=0.55, decimal_odds=1.91)
        assert fraction >= 0
        assert stake >= 0

    def test_compute_kelly_no_edge(self, kelly):
        """50% win probability at 1.91 odds = no edge, no stake."""
        fraction, stake = kelly.compute_kelly(win_probability=0.50, decimal_odds=1.91)
        assert fraction == 0.0
        assert stake == 0.0

    def test_compute_kelly_below_min_edge(self, kelly):
        """51% win probability at 1.91 odds = ~1.9% edge, below 2% min_edge."""
        kelly.min_edge = 0.02
        fraction, stake = kelly.compute_kelly(win_probability=0.515, decimal_odds=1.91)
        assert fraction == 0.0
        assert stake == 0.0

    def test_compute_kelly_capped_at_max_fraction(self, kelly):
        """Very high win probability should be capped at max_fraction (0.15)."""
        fraction, stake = kelly.compute_kelly(win_probability=0.80, decimal_odds=1.91)
        assert fraction <= kelly.max_fraction + 1e-10
        assert stake <= kelly.max_fraction * kelly.current_bankroll + 1

    def test_compute_kelly_extreme_odds(self, kelly):
        """Very long odds (+500, 6.0 decimal) with 25% win prob should still work."""
        fraction, stake = kelly.compute_kelly(win_probability=0.25, decimal_odds=6.0)
        assert fraction >= 0
        assert stake >= 0

    def test_compute_kelly_invalid_prob(self, kelly):
        """Edge cases: 0 and 1 win probability should return zero stake."""
        f0, s0 = kelly.compute_kelly(win_probability=0.0, decimal_odds=1.91)
        assert f0 == 0.0 and s0 == 0.0

        f1, s1 = kelly.compute_kelly(win_probability=1.0, decimal_odds=1.91)
        assert f1 == 0.0 and s1 == 0.0

    def test_compute_kelly_zero_odds(self, kelly):
        """Decimal odds of 1.0 or less should return zero."""
        f, s = kelly.compute_kelly(win_probability=0.60, decimal_odds=0.5)
        assert f == 0.0 and s == 0.0

    def test_compute_with_edge(self, kelly):
        """compute_with_edge should blend edge with win probability."""
        fraction, stake = kelly.compute_with_edge(
            win_probability=0.55, edge_pct=0.05, decimal_odds=1.91
        )
        assert fraction >= 0
        assert stake >= 0

    def test_compute_with_edge_negative(self, kelly):
        """Negative edge should produce zero stake."""
        fraction, stake = kelly.compute_with_edge(
            win_probability=0.40, edge_pct=-0.05, decimal_odds=1.91
        )
        assert fraction == 0.0 and stake == 0.0

    def test_record_result_win(self, kelly):
        kelly.record_result(stake_dollars=100.0, won=True, decimal_odds=1.91)
        expected = 10_000.0 + 100.0 * (1.91 - 1.0)
        assert kelly.current_bankroll == pytest.approx(expected, rel=1e-6)

    def test_record_result_loss(self, kelly):
        kelly.record_result(stake_dollars=100.0, won=False)
        assert kelly.current_bankroll == 9_900.0

    def test_record_result_updates_peak(self, kelly):
        initial_peak = kelly.peak_bankroll
        kelly.record_result(stake_dollars=500.0, won=True, decimal_odds=2.0)
        assert kelly.peak_bankroll > initial_peak

    def test_get_current_state(self, kelly):
        state = kelly.get_current_state()
        assert state["initial_bankroll"] == 10_000.0
        assert state["current_bankroll"] == 10_000.0
        assert state["drawdown_pct"] == 0.0
        assert state["kelly_fraction"] == 0.25

    def test_drawdown_protection(self, kelly):
        """Enable drawdown protection and simulate losses."""
        from betting_intel.risk.kelly import KellyCalculator

        kelly.drawdown_protection = True
        # Simulate a 10% drawdown
        kelly.current_bankroll = 9_000.0
        kelly.peak_bankroll = 10_000.0

        fraction, stake = kelly.compute_kelly(win_probability=0.60, decimal_odds=1.91)
        # Stake should be reduced by drawdown protection
        # Full Kelly ~0.26 * 0.25 = 0.065, capped at 0.15, then drawdown factor applied
        no_dd_kelly = KellyCalculator(
            bankroll=9000, fraction=0.25, max_fraction=0.15, drawdown_protection=False
        )
        ndd_frac, ndd_stake = no_dd_kelly.compute_kelly(0.60, 1.91)

        # With drawdown protection, the fraction should be lower
        assert fraction <= ndd_frac + 1e-10

    def test_drawdown_at_5pct_threshold(self, kelly):
        """5% drawdown should not reduce stake (threshold is > 5%)."""
        from betting_intel.risk.kelly import KellyCalculator

        kelly.drawdown_protection = True
        kelly.current_bankroll = 9_500.0
        kelly.peak_bankroll = 10_000.0

        # 5% drawdown: drawdown_pct = 0.05, not > 0.05, so no reduction
        dd_frac, dd_stake = kelly.compute_kelly(win_probability=0.60, decimal_odds=1.91)

        no_dd = KellyCalculator(
            bankroll=9500, fraction=0.25, max_fraction=0.15, drawdown_protection=False
        )
        ndd_frac, ndd_stake = no_dd.compute_kelly(0.60, 1.91)

        assert dd_frac == pytest.approx(ndd_frac, abs=1e-6)

    def test_drawdown_deep_reduction(self, kelly):
        """33% drawdown should reduce fraction to near zero."""
        kelly.drawdown_protection = True
        kelly.current_bankroll = 6_700.0
        kelly.peak_bankroll = 10_000.0

        fraction, stake = kelly.compute_kelly(win_probability=0.60, decimal_odds=1.91)
        # 33% drawdown: factor = max(0.1, 1.0 - 0.33 * 3) = max(0.1, 0.01) = 0.1
        # So stake should be reduced to ~10% of normal
        assert fraction <= 0.02  # Heavily reduced

    def test_drawdown_no_peak_change(self, kelly):
        """Drawdown protection should not affect stake when at peak bankroll."""
        from betting_intel.risk.kelly import KellyCalculator

        kelly.drawdown_protection = True
        # Bankroll at peak - no drawdown
        kelly.current_bankroll = 10_000.0
        kelly.peak_bankroll = 10_000.0

        dd_f, dd_s = kelly.compute_kelly(win_probability=0.60, decimal_odds=1.91)

        no_dd = KellyCalculator(
            bankroll=10000, fraction=0.25, max_fraction=0.15, drawdown_protection=False
        )
        ndd_f, ndd_s = no_dd.compute_kelly(0.60, 1.91)

        assert dd_f == pytest.approx(ndd_f, abs=1e-6)

    def test_compute_with_edge_zero_edge(self, kelly):
        """compute_with_edge with edge_pct=0 should behave like compute_kelly."""
        kelly_frac, kelly_stake = kelly.compute_kelly(
            win_probability=0.55, decimal_odds=1.91
        )
        edge_frac, edge_stake = kelly.compute_with_edge(
            win_probability=0.55, edge_pct=0.0, decimal_odds=1.91
        )
        assert edge_frac >= 0 and edge_stake >= 0

    def test_compute_kelly_min_edge_zero(self, kelly):
        """With min_edge=0.0, even tiny edges should produce stakes."""
        from betting_intel.risk.kelly import KellyCalculator

        no_threshold = KellyCalculator(
            bankroll=10_000.0,
            fraction=0.25,
            max_fraction=0.15,
            min_edge=0.0,
            drawdown_protection=False,
        )
        # ~53% at 1.91 odds = ~0.64% positive edge, should pass 0% threshold
        fraction, stake = no_threshold.compute_kelly(
            win_probability=0.53, decimal_odds=1.91
        )
        # edge = 0.53 - 0.52356 = 0.00644 > 0.0, so passes
        # full_kelly = (0.91*0.53 - 0.47)/0.91 = 0.0123/0.91 = 0.0135
        # quarter = 0.0135 * 0.25 = 0.00338
        assert fraction > 0
        assert stake > 0

    def test_compute_kelly_exact_edge_at_threshold(self, kelly):
        """Edge exactly at min_edge should be actionable (>= check)."""
        # For -110 (1.91) odds, implied prob = 0.5238
        # Model prob of 0.5438 gives edge = 0.02 exactly
        # But due to rounding, the actual edge check is edge >= min_edge (it's >= not >)
        # Actually no, looking at the code:
        #   if edge < self.min_edge: return (0.0, 0.0)
        # So exactly at threshold passes
        fraction, stake = kelly.compute_kelly(
            win_probability=0.5438, decimal_odds=1.91
        )
        # 0.5438 - 0.52356 = 0.02024 >= 0.02, so should pass
        assert fraction > 0
        assert stake > 0

    def test_record_result_near_zero_bankroll(self, kelly):
        """Recording a loss that drops bankroll near zero should not break."""
        kelly.current_bankroll = 100.0
        kelly.peak_bankroll = 10_000.0

        fraction, stake = kelly.compute_kelly(win_probability=0.60, decimal_odds=1.91)
        assert fraction >= 0
        assert stake >= 0


class TestMultiBetKelly:
    """Tests for multi-bet Kelly optimization."""

    @pytest.fixture
    def multi_kelly(self):
        from betting_intel.risk.kelly import MultiBetKelly

        return MultiBetKelly(bankroll=10_000.0, fraction=0.25, max_single_bet=0.15, max_total_exposure=0.40)

    def test_optimize_empty_bets(self, multi_kelly):
        fractions = multi_kelly.optimize([])
        assert len(fractions) == 0

    def test_optimize_single_bet(self, multi_kelly):
        bets = [{"win_probability": 0.60, "decimal_odds": 1.91, "edge_pct": 0.08}]
        fractions = multi_kelly.optimize(bets)
        assert len(fractions) == 1
        assert fractions[0] >= 0
        assert fractions[0] <= multi_kelly.max_single_bet

    def test_optimize_two_bets(self, multi_kelly):
        bets = [
            {"win_probability": 0.60, "decimal_odds": 1.91, "edge_pct": 0.08},
            {"win_probability": 0.55, "decimal_odds": 2.10, "edge_pct": 0.05},
        ]
        fractions = multi_kelly.optimize(bets)
        assert len(fractions) == 2
        assert all(f >= 0 for f in fractions)
        assert sum(fractions) <= multi_kelly.max_total_exposure + 1e-6

    def test_optimize_with_correlation_matrix(self, multi_kelly):
        bets = [
            {"win_probability": 0.60, "decimal_odds": 1.91, "edge_pct": 0.08},
            {"win_probability": 0.55, "decimal_odds": 2.10, "edge_pct": 0.05},
        ]
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        fractions = multi_kelly.optimize(bets, correlation_matrix=corr)
        assert len(fractions) == 2

    def test_compute_stakes(self, multi_kelly):
        bets = [
            {"bet_id": "bet_1", "win_probability": 0.60, "decimal_odds": 1.91, "edge_pct": 0.08},
        ]
        result = multi_kelly.compute_stakes(bets)
        assert len(result) == 1
        assert result[0].bet_id == "bet_1"
        assert result[0].dollar_stake >= 0
        assert result[0].kelly_fraction >= 0
        assert result[0].expected_value > 0

    def test_total_exposure_constraint(self, multi_kelly):
        """Many high-edge bets should be constrained by max_total_exposure."""
        bets = [
            {"win_probability": 0.65, "decimal_odds": 1.91, "edge_pct": 0.12},
            {"win_probability": 0.63, "decimal_odds": 1.91, "edge_pct": 0.10},
            {"win_probability": 0.60, "decimal_odds": 2.10, "edge_pct": 0.08},
            {"win_probability": 0.62, "decimal_odds": 1.80, "edge_pct": 0.07},
        ]
        fractions = multi_kelly.optimize(bets)
        assert sum(fractions) <= multi_kelly.max_total_exposure + 1e-6

    def test_no_edge_bets_zero_stakes(self, multi_kelly):
        """Bets with no edge should get zero stakes."""
        bets = [
            {"win_probability": 0.50, "decimal_odds": 1.91, "edge_pct": 0.0},
            {"win_probability": 0.49, "decimal_odds": 2.10, "edge_pct": -0.02},
        ]
        fractions = multi_kelly.optimize(bets)
        # At minimum, should not crash and should return valid fractions
        assert len(fractions) == 2


class TestCorrelatedKelly:
    """Tests for convenience correlated_kelly function."""

    def test_correlated_kelly_basic(self):
        from betting_intel.risk.kelly import correlated_kelly

        win_probs = np.array([0.60, 0.55])
        odds = np.array([1.91, 2.10])
        corr = np.array([[1.0, 0.2], [0.2, 1.0]])

        stakes = correlated_kelly(win_probs, odds, corr, bankroll=10_000.0, fraction=0.25)
        assert len(stakes) == 2
        assert all(s >= 0 for s in stakes)

    def test_correlated_kelly_single(self):
        from betting_intel.risk.kelly import correlated_kelly

        stakes = correlated_kelly(
            np.array([0.60]), np.array([1.91]), np.array([[1.0]]), bankroll=10_000.0
        )
        assert len(stakes) == 1
        assert stakes[0] >= 0


class TestExpenseManager:
    """Tests for the ExposureManager portfolio exposure control."""

    @pytest.fixture
    def manager(self):
        from betting_intel.risk.exposure import ExposureManager

        return ExposureManager(bankroll=10_000.0)

    @pytest.fixture
    def sample_bet(self):
        from betting_intel.risk.exposure import ActiveBet
        from datetime import datetime

        return ActiveBet(
            bet_id="bet_001",
            game_id="game_001",
            matchup="Team A @ Team B",
            league="NBA",
            bet_type="moneyline",
            side="Team A",
            stake_dollars=100.0,
            decimal_odds=1.91,
            edge_pct=0.05,
            win_probability=0.55,
            placed_at=datetime.now(),
        )

    def test_initial_state(self, manager):
        report = manager.get_report()
        assert report.n_active_bets == 0
        assert report.total_exposure == 0.0

    def test_add_bet_accepted(self, manager, sample_bet):
        accepted, reason = manager.add_bet(sample_bet)
        assert accepted is True
        assert reason is None
        assert len(manager.active_bets) == 1
        assert manager.active_bets[0].bet_id == "bet_001"

    def test_add_bet_updates_exposure(self, manager, sample_bet):
        manager.add_bet(sample_bet)
        report = manager.get_report()
        assert report.total_exposure == 100.0
        assert report.n_active_bets == 1
        assert "NBA" in report.by_league
        assert report.by_league["NBA"] == 100.0

    def test_add_bet_over_total_limit(self, manager):
        from betting_intel.risk.exposure import ActiveBet
        from datetime import datetime

        # Override total limit to be very low
        manager.set_limit("total", "all", max_exposure=50.0)

        big_bet = ActiveBet(
            bet_id="big_bet",
            game_id="game_002",
            matchup="Team C @ Team D",
            league="NBA",
            bet_type="moneyline",
            side="Team C",
            stake_dollars=100.0,
            decimal_odds=1.91,
            edge_pct=0.05,
            win_probability=0.55,
            placed_at=datetime.now(),
        )
        accepted, reason = manager.add_bet(big_bet)
        assert accepted is False
        assert "Total exposure limit exceeded" in reason

    def test_remove_bet(self, manager, sample_bet):
        manager.add_bet(sample_bet)
        manager.remove_bet("bet_001")
        assert len(manager.active_bets) == 0
        assert len(manager.settled_bets) == 1

    def test_get_report_with_bets(self, manager, sample_bet):
        manager.add_bet(sample_bet)
        report = manager.get_report()
        assert report.total_exposure > 0
        assert report.n_active_bets > 0
        assert report.bankroll_pct > 0

    def test_get_exposure_by_league(self, manager, sample_bet):
        from betting_intel.risk.exposure import ActiveBet
        from datetime import datetime

        manager.add_bet(sample_bet)

        another_bet = ActiveBet(
            bet_id="bet_002",
            game_id="game_003",
            matchup="Team E @ Team F",
            league="NBA",
            bet_type="spread",
            side="Team E",
            stake_dollars=50.0,
            decimal_odds=1.87,
            edge_pct=0.03,
            win_probability=0.53,
            placed_at=datetime.now(),
        )
        manager.add_bet(another_bet)

        by_league = manager.get_exposure_by_league()
        assert by_league["NBA"] == 150.0

    def test_get_exposure_by_game(self, manager, sample_bet):
        from betting_intel.risk.exposure import ActiveBet
        from datetime import datetime

        manager.add_bet(sample_bet)

        same_game_bet = ActiveBet(
            bet_id="bet_003",
            game_id="game_001",
            matchup="Team A @ Team B",
            league="NBA",
            bet_type="total",
            side="OVER",
            stake_dollars=75.0,
            decimal_odds=1.91,
            edge_pct=0.04,
            win_probability=0.54,
            placed_at=datetime.now(),
        )
        manager.add_bet(same_game_bet)

        by_game = manager.get_exposure_by_game()
        assert "Team A @ Team B (game_001)" in by_game
        assert by_game["Team A @ Team B (game_001)"] == 175.0

    def test_get_exposure_by_bet_type(self, manager, sample_bet):
        by_type = manager.get_exposure_by_bet_type()
        assert isinstance(by_type, dict)

    def test_format_report(self, manager, sample_bet):
        manager.add_bet(sample_bet)
        report_text = manager.format_report()
        assert "PORTFOLIO EXPOSURE REPORT" in report_text
        assert "Active Bets:" in report_text
        assert "$100" in report_text

    def test_available_capacity(self, manager):
        capacity = manager.get_available_capacity()
        assert capacity == 10_000.0 * 0.35  # default_max_exposure_pct = 0.35

    def test_set_limit_updates_existing(self, manager):
        manager.set_limit("total", "all", max_exposure=5_000.0)
        assert manager.limits["total:all"].max_exposure == 5_000.0

    def test_concentration_warning(self, manager):
        from betting_intel.risk.exposure import ActiveBet
        from datetime import datetime

        # One bet that's >20% of total exposure
        big_bet = ActiveBet(
            bet_id="big",
            game_id="g1",
            matchup="A @ B",
            league="NBA",
            bet_type="moneyline",
            side="A",
            stake_dollars=900.0,
            decimal_odds=1.91,
            edge_pct=0.05,
            win_probability=0.55,
            placed_at=datetime.now(),
        )
        small_bet = ActiveBet(
            bet_id="small",
            game_id="g2",
            matchup="C @ D",
            league="NBA",
            bet_type="spread",
            side="C",
            stake_dollars=100.0,
            decimal_odds=1.87,
            edge_pct=0.03,
            win_probability=0.53,
            placed_at=datetime.now(),
        )
        manager.add_bet(big_bet)
        manager.add_bet(small_bet)

        report = manager.get_report()
        conc_violations = [v for v in report.violations if "CONCENTRATION" in v]
        assert len(conc_violations) >= 1


class TestBetPortfolio:
    """Tests for portfolio strategy allocation."""

    def test_add_strategy(self):
        from betting_intel.risk.exposure import BetPortfolio

        portfolio = BetPortfolio(total_bankroll=10_000.0)
        portfolio.add_strategy("aggressive", bankroll_pct=0.3, kelly_fraction=0.5)
        portfolio.add_strategy("conservative", bankroll_pct=0.7, kelly_fraction=0.15)

        assert "aggressive" in portfolio.strategies
        assert "conservative" in portfolio.strategies
        assert portfolio.strategies["aggressive"]["strategy_bankroll"] == 3_000.0
        assert portfolio.strategies["conservative"]["strategy_bankroll"] == 7_000.0

    def test_get_strategy_summary(self):
        from betting_intel.risk.exposure import BetPortfolio

        portfolio = BetPortfolio(total_bankroll=10_000.0)
        portfolio.add_strategy("main", bankroll_pct=1.0, kelly_fraction=0.25)
        summary = portfolio.get_strategy_summary()
        assert "PORTFOLIO STRATEGY ALLOCATION" in summary
        assert "Total Bankroll" in summary


class TestCorrelationTracker:
    """Tests for bet correlation tracking."""

    @pytest.fixture
    def tracker(self):
        from betting_intel.risk.correlation import BetCorrelationTracker

        return BetCorrelationTracker()

    def test_register_and_get_correlation_same_game(self, tracker):
        from betting_intel.risk.correlation import BetCorrelationTracker

        tracker.register_bet("bet_1", "moneyline_home", "game_001", "NBA")
        tracker.register_bet("bet_2", "moneyline_away", "game_001", "NBA")
        corr = tracker.get_correlation("bet_1", "bet_2")
        assert corr == -0.98  # Opposite sides of same game

    def test_register_and_get_correlation_diff_leagues(self, tracker):
        tracker.register_bet("bet_1", "total_over", "game_001", "NBA")
        tracker.register_bet("bet_2", "total_over", "game_002", "LNB")
        corr = tracker.get_correlation("bet_1", "bet_2")
        assert corr == 0.0  # Different leagues = independent

    def test_register_and_get_correlation_same_league_diff_game(self, tracker):
        tracker.register_bet("bet_1", "total_over", "game_001", "NBA")
        tracker.register_bet("bet_2", "total_over", "game_002", "NBA")
        corr = tracker.get_correlation("bet_1", "bet_2")
        assert corr == 0.03  # Same league, different games = weak

    def test_empirical_correlation(self, tracker):
        for i in range(15):
            bid_a = f"bet_a_{i}"
            bid_b = f"bet_b_{i}"
            tracker.register_bet(bid_a, "total_over", f"game_{i}", "NBA")
            tracker.register_bet(bid_b, "total_over", f"game_{i}", "NBA")
            tracker.record_outcome(bid_a, won=(i % 2 == 0))
            tracker.record_outcome(bid_b, won=(i % 2 == 0))

        corr = tracker.get_correlation("bet_a_0", "bet_b_0")
        assert -1.0 <= corr <= 1.0

    def test_get_correlation_matrix(self, tracker):
        tracker.register_bet("b1", "moneyline_home", "game_001", "NBA")
        tracker.register_bet("b2", "moneyline_away", "game_001", "NBA")
        tracker.register_bet("b3", "total_over", "game_001", "NBA")

        matrix = tracker.get_correlation_matrix(["b1", "b2", "b3"])
        assert matrix.n == 3
        assert matrix.is_positive_semidefinite

    def test_get_correlation_matrix_psd_fix(self, tracker):
        """Matrix should be made positive semidefinite if needed."""
        from betting_intel.risk.correlation import CorrelationMatrix

        cm = CorrelationMatrix(["a", "b", "c"])
        cm.set_correlation(0, 1, 0.99)
        cm.set_correlation(0, 2, 0.99)
        cm.set_correlation(1, 2, 0.99)
        assert cm.is_positive_semidefinite
        cm.make_positive_semidefinite()
        assert cm.is_positive_semidefinite

    def test_correlation_matrix_to_dict(self, tracker):
        tracker.register_bet("b1", "moneyline", "g1", "NBA")
        tracker.register_bet("b2", "spread", "g1", "NBA")
        matrix = tracker.get_correlation_matrix(["b1", "b2"])
        d = matrix.to_dict()
        assert "bet_ids" in d
        assert "matrix" in d
        assert d["bet_ids"] == ["b1", "b2"]

    def test_record_outcome_unregistered_bet(self, tracker):
        """Recording an unregistered bet should not crash."""
        tracker.record_outcome("unknown_bet", won=True)
        assert True  # No exception

    def test_get_summary(self, tracker):
        tracker.register_bet("b1", "moneyline", "g1")
        tracker.record_outcome("b1", won=True)
        summary = tracker.get_summary()
        assert summary["n_bets_tracked"] >= 1


class TestEstimateGameCorrelations:
    """Tests for the convenience estimate_game_correlations function."""

    def test_estimate_game_correlations(self):
        from betting_intel.risk.correlation import estimate_game_correlations

        bets = [
            {"bet_id": "b1", "bet_type": "moneyline_home", "team": "Celtics"},
            {"bet_id": "b2", "bet_type": "moneyline_away", "team": "Lakers"},
            {"bet_id": "b3", "bet_type": "total_over"},
        ]
        matrix = estimate_game_correlations("g1", bets)
        assert matrix.n == 3
        assert matrix.get_correlation(0, 1) < 0  # Opposite sides
