"""Tests for bankroll management module."""

from __future__ import annotations

import pytest

from betting_intel.betting.bankroll import BankrollManager, BankrollSnapshot


class TestBankrollManager:
    """Tests for BankrollManager with Kelly staking."""

    @pytest.fixture
    def manager(self):
        return BankrollManager(
            initial_bankroll=10_000.0,
            base_kelly_fraction=0.25,
            max_kelly_fraction=0.5,
        )

    def test_initial_state(self, manager):
        """Manager should start with correct initial state."""
        assert manager.current_bankroll == 10_000.0
        assert manager.peak_bankroll == 10_000.0
        assert manager.total_bets == 0
        assert manager.winning_bets == 0
        assert manager.consecutive_losses == 0

    def test_compute_kelly_stake_positive_edge(self, manager):
        """Kelly stake should be positive when edge exists."""
        fraction, stake = manager.compute_kelly_stake(
            win_probability=0.55,
            decimal_odds=1.91,
            edge_pct=0.05,
        )
        assert fraction >= 0, "Kelly fraction should be non-negative"
        assert stake >= 0, "Stake should be non-negative"

    def test_compute_kelly_stake_no_edge(self, manager):
        """Kelly stake should be zero when no edge exists."""
        fraction, stake = manager.compute_kelly_stake(
            win_probability=0.50,
            decimal_odds=1.91,
            edge_pct=0.0,
        )
        assert fraction >= 0, "Kelly fraction should be non-negative"

    def test_place_and_record_bet_win(self, manager):
        """Placing a winning bet should increase bankroll."""
        bet = manager.place_bet(
            game_id="GAME_001",
            strategy="momentum",
            win_probability=0.60,
            decimal_odds=1.91,
            edge_pct=0.10,
        )
        assert bet is not None, "Bet should be placed with positive edge"

        manager.record_result(bet, won=True)
        assert manager.current_bankroll > 10_000.0
        assert manager.winning_bets == 1
        assert manager.total_bets == 1

    def test_place_and_record_bet_loss(self, manager):
        """Placing a losing bet should decrease bankroll."""
        initial = manager.current_bankroll
        bet = manager.place_bet(
            game_id="GAME_002",
            strategy="spread_model",
            win_probability=0.55,
            decimal_odds=1.91,
            edge_pct=0.05,
        )
        assert bet is not None

        manager.record_result(bet, won=False)
        assert manager.current_bankroll < initial
        assert manager.consecutive_losses == 1

    def test_consecutive_losses_reduce_stakes(self, manager):
        """Multiple consecutive losses should reduce stake size."""
        stakes = []
        for i in range(5):
            bet = manager.place_bet(
                game_id=f"GAME_{i:04d}",
                strategy="momentum",
                win_probability=0.55,
                decimal_odds=1.91,
                edge_pct=0.05,
            )
            if bet:
                stakes.append(bet.stake_dollars)
                manager.record_result(bet, won=False)

        # Stakes should decrease after consecutive losses
        assert stakes[-1] < stakes[0], "Late stakes should be smaller"

    def test_take_snapshot(self, manager):
        """Snapshot should capture bankroll state."""
        snapshot = manager.take_snapshot("2023-01-01")
        assert isinstance(snapshot, BankrollSnapshot)
        assert snapshot.bankroll == 10_000.0
        assert snapshot.total_bets == 0

    def test_get_metrics(self, manager):
        """Metrics should return all expected fields."""
        metrics = manager.get_metrics()
        assert "initial_bankroll" in metrics
        assert "current_bankroll" in metrics
        assert "total_return_pct" in metrics
        assert "drawdown_pct" in metrics
        assert metrics["initial_bankroll"] == 10_000.0

    def test_drawdown_reduces_exposure(self, manager):
        """Large drawdowns should reduce stake size."""
        # Simulate losses to create drawdown
        for i in range(3):
            bet = manager.place_bet(
                game_id=f"GAME_{i:04d}",
                strategy="momentum",
                win_probability=0.55,
                decimal_odds=1.91,
                edge_pct=0.05,
            )
            if bet:
                manager.record_result(bet, won=False)

        drawdown_factor = manager._get_drawdown_factor()
        assert drawdown_factor <= 1.0, "Drawdown factor should reduce exposure"

    def test_max_drawdown_stops_betting(self, manager):
        """Extreme drawdowns should stop all betting."""
        # Simulate major losses to reach >30% drawdown
        manager.current_bankroll = 5_000.0  # 50% drawdown from 10k
        manager.peak_bankroll = 10_000.0

        factor = manager._get_drawdown_factor()
        assert factor == 0.0, "Should stop betting at >30% drawdown"
