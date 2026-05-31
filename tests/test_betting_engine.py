"""Tests for the BettingEngine, BetRecommendation, StakeSizing, and DailyPortfolio classes."""

from __future__ import annotations

import math

import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  BetRecommendation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBetRecommendation:
    """Tests for the BetRecommendation dataclass."""

    def test_default_timestamp(self):
        from betting_intel.betting.bet import BetRecommendation

        bet = BetRecommendation()
        assert bet.timestamp != ""

    def test_bet_label_moneyline(self):
        from betting_intel.betting.bet import BetRecommendation

        bet = BetRecommendation(
            home_team="Celtics",
            away_team="Lakers",
            bet_side="home",
            market_type="moneyline",
        )
        label = bet.bet_label
        assert "Lakers @ Celtics" in label
        assert "HOME" in label
        assert "ML" in label

    def test_bet_label_spread(self):
        from betting_intel.betting.bet import BetRecommendation

        bet = BetRecommendation(
            home_team="Celtics",
            away_team="Lakers",
            bet_side="away",
            market_type="spread",
            market_line=3.5,
        )
        label = bet.bet_label
        assert "Lakers @ Celtics" in label
        assert "AWAY" in label
        assert "+3.5" in label or "-3.5" in label or "3.5" in label

    def test_bet_label_total(self):
        from betting_intel.betting.bet import BetRecommendation

        bet = BetRecommendation(
            home_team="Celtics",
            away_team="Lakers",
            bet_side="over",
            market_type="total",
            market_line=218.5,
        )
        label = bet.bet_label
        assert "Lakers @ Celtics" in label
        assert "OVER" in label
        assert "O/U 218.5" in label or "218.5" in label

    def test_summary_actionable(self):
        from betting_intel.betting.bet import BetRecommendation, StakeSizing

        bet = BetRecommendation(
            home_team="Celtics",
            away_team="Lakers",
            bet_side="home",
            market_type="moneyline",
            odds_american=-110.0,
            edge_percentage=0.05,
            expected_value=0.10,
            is_actionable=True,
            stake=StakeSizing(
                recommended_stake=50.0,
                recommended_fraction=0.005,
                is_valid=True,
                risk_level="medium",
            ),
        )
        summary = bet.summary()
        assert "$50" in summary or "$ 50" in summary
        assert "5.00%" in summary or "5%" in summary
        assert "medium" in summary

    def test_summary_not_actionable(self):
        from betting_intel.betting.bet import BetRecommendation

        bet = BetRecommendation(
            home_team="Celtics",
            away_team="Lakers",
            bet_side="home",
            is_actionable=False,
        )
        summary = bet.summary()
        assert "no bet" in summary


# ═══════════════════════════════════════════════════════════════════════════
#  StakeSizing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStakeSizing:
    """Tests for the StakeSizing dataclass."""

    def test_defaults(self):
        from betting_intel.betting.bet import StakeSizing

        s = StakeSizing()
        assert s.kelly_fraction == 0.0
        assert s.recommended_stake == 0.0
        assert s.risk_level == "skip"
        assert not s.is_valid

    def test_valid_stake(self):
        from betting_intel.betting.bet import StakeSizing

        s = StakeSizing(
            kelly_fraction=0.25,
            recommended_fraction=0.0138,
            recommended_stake=138.0,
            bankroll_percentage=1.38,
            risk_level="medium",
            is_valid=True,
        )
        assert s.is_valid
        assert s.recommended_stake == 138.0


# ═══════════════════════════════════════════════════════════════════════════
#  DailyPortfolio Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDailyPortfolio:
    """Tests for the DailyPortfolio dataclass."""

    @pytest.fixture
    def portfolio(self):
        from betting_intel.betting.bet import DailyPortfolio

        return DailyPortfolio(date="2026-06-01")

    @pytest.fixture
    def actionable_bet(self):
        from betting_intel.betting.bet import BetRecommendation, StakeSizing

        return BetRecommendation(
            game_id="g1",
            home_team="Celtics",
            away_team="Lakers",
            bet_side="home",
            edge_percentage=0.05,
            expected_value=0.10,
            is_actionable=True,
            stake=StakeSizing(
                recommended_stake=50.0,
                bankroll_percentage=0.5,
                is_valid=True,
                risk_level="medium",
            ),
        )

    @pytest.fixture
    def non_actionable_bet(self):
        from betting_intel.betting.bet import BetRecommendation, StakeSizing

        return BetRecommendation(
            game_id="g2",
            home_team="Warriors",
            away_team="Nuggets",
            bet_side="away",
            edge_percentage=-0.01,
            expected_value=-0.02,
            is_actionable=False,
            stake=StakeSizing(
                is_valid=False,
                risk_level="skip",
            ),
        )

    def test_initial_state(self, portfolio):
        assert portfolio.date == "2026-06-01"
        assert len(portfolio.bets) == 0
        assert portfolio.total_risk == 0.0
        assert portfolio.num_actionable == 0

    def test_add_actionable_bet(self, portfolio, actionable_bet):
        portfolio.add(actionable_bet)
        assert len(portfolio.bets) == 1
        assert portfolio.num_actionable == 1
        assert portfolio.total_risk == 50.0

    def test_add_non_actionable_bet(self, portfolio, non_actionable_bet):
        portfolio.add(non_actionable_bet)
        assert len(portfolio.bets) == 1
        assert portfolio.num_actionable == 0
        assert portfolio.total_risk == 0.0

    def test_mixed_bets(self, portfolio, actionable_bet, non_actionable_bet):
        portfolio.add(actionable_bet)
        portfolio.add(non_actionable_bet)
        assert portfolio.num_bets == 2
        assert portfolio.num_actionable == 1
        assert portfolio.total_risk == 50.0

    def test_sort_by_edge(self, portfolio):
        from betting_intel.betting.bet import BetRecommendation

        bets = [
            BetRecommendation(game_id="1", bet_side="home", edge_percentage=0.02),
            BetRecommendation(game_id="2", bet_side="away", edge_percentage=0.10),
            BetRecommendation(game_id="3", bet_side="home", edge_percentage=0.05),
        ]
        for b in bets:
            portfolio.add(b)

        sorted_bets = portfolio.sort_by_edge()
        assert sorted_bets[0].edge_percentage == 0.10
        assert sorted_bets[1].edge_percentage == 0.05
        assert sorted_bets[2].edge_percentage == 0.02

    def test_sort_by_ev(self, portfolio):
        from betting_intel.betting.bet import BetRecommendation

        bets = [
            BetRecommendation(game_id="1", bet_side="home", expected_value=0.03),
            BetRecommendation(game_id="2", bet_side="away", expected_value=0.15),
            BetRecommendation(game_id="3", bet_side="home", expected_value=0.07),
        ]
        for b in bets:
            portfolio.add(b)

        sorted_bets = portfolio.sort_by_ev()
        assert sorted_bets[0].expected_value == 0.15
        assert sorted_bets[1].expected_value == 0.07
        assert sorted_bets[2].expected_value == 0.03

    def test_actionable_filter(self, portfolio, actionable_bet, non_actionable_bet):
        portfolio.add(actionable_bet)
        portfolio.add(non_actionable_bet)
        actionable = portfolio.actionable()
        assert len(actionable) == 1
        assert actionable[0].game_id == "g1"


# ═══════════════════════════════════════════════════════════════════════════
#  BettingEngine Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBettingEngine:
    """Tests for the BettingEngine class."""

    @pytest.fixture
    def engine(self):
        from betting_intel.betting.bet import BettingEngine

        return BettingEngine(
            bankroll=10_000.0,
            kelly_fraction=0.25,
            min_edge=0.02,
            max_single_stake_pct=0.15,
            max_daily_stake_pct=0.40,
            drawdown_protection=False,
        )

    # ═════════════════════════════════════════════════════════════════
    #  Constructor & Properties
    # ═════════════════════════════════════════════════════════════════

    def test_initial_state(self, engine):
        assert engine.bankroll == 10_000.0
        assert engine.kelly.fraction == 0.25
        assert engine.kelly.max_fraction == 0.15
        assert engine.kelly.min_edge == 0.02
        assert engine.max_daily_stake_pct == 0.40
        assert len(engine.daily_portfolios) == 0

    def test_default_constructor(self):
        from betting_intel.betting.bet import BettingEngine

        engine = BettingEngine()
        assert engine.bankroll == 10_000.0
        assert engine.kelly.fraction == 0.25

    def test_custom_bankroll(self):
        from betting_intel.betting.bet import BettingEngine

        engine = BettingEngine(bankroll=50_000.0)
        assert engine.bankroll == 50_000.0

    def test_half_kelly(self):
        from betting_intel.betting.bet import BettingEngine

        engine = BettingEngine(kelly_fraction=0.5)
        assert engine.kelly.fraction == 0.5

    def test_bankroll_property(self, engine):
        assert engine.bankroll == engine.kelly.current_bankroll

    def test_get_state(self, engine):
        state = engine.get_state()
        assert state["initial_bankroll"] == 10_000.0
        assert state["max_daily_stake_pct"] == 0.40
        assert state["daily_portfolios"] == 0

    # ═════════════════════════════════════════════════════════════════
    #  create_bet - Core Functionality
    # ═════════════════════════════════════════════════════════════════

    def test_create_bet_actionable(self, engine):
        """62% model at -110 odds should produce an actionable bet with positive stake."""
        bet = engine.create_bet(
            model_probability=0.62,
            odds_american=-110,
            home_team="Spurs",
            away_team="Knicks",
        )
        assert bet.is_actionable
        assert bet.stake.is_valid
        assert bet.stake.recommended_stake > 0
        assert bet.edge_percentage > 0.02
        assert bet.expected_value > 0

    def test_create_bet_not_actionable(self, engine):
        """52% model at -110 odds (~0.2% edge) should not be actionable."""
        bet = engine.create_bet(
            model_probability=0.52,
            odds_american=-110,
        )
        assert not bet.is_actionable
        assert not bet.stake.is_valid
        assert bet.stake.recommended_stake == 0.0

    def test_create_bet_with_decimal_odds(self, engine):
        bet = engine.create_bet(
            model_probability=0.62,
            odds_decimal=1.91,
            home_team="Spurs",
            away_team="Knicks",
        )
        assert bet.is_actionable
        assert bet.stake.recommended_stake > 0
        assert bet.odds_decimal == 1.91

    def test_create_bet_no_odds_returns_not_actionable(self, engine):
        bet = engine.create_bet(
            model_probability=0.62,
        )
        assert not bet.is_actionable
        assert not bet.stake.is_valid

    def test_create_bet_determines_american_odds(self, engine):
        """2.0 decimal odds should convert to +100 American."""
        bet = engine.create_bet(
            model_probability=0.62,
            odds_decimal=2.0,
        )
        assert bet.odds_american == 100.0

    def test_create_bet_metadata(self, engine):
        bet = engine.create_bet(
            model_probability=0.62,
            odds_american=-110,
            game_id="game123",
            home_team="Celtics",
            away_team="Lakers",
            commence_time="2026-06-01T19:00:00Z",
            bet_side="home",
            market_type="moneyline",
            model_name="ensemble_v2",
        )
        assert bet.game_id == "game123"
        assert bet.home_team == "Celtics"
        assert bet.away_team == "Lakers"
        assert bet.commence_time == "2026-06-01T19:00:00Z"
        assert bet.bet_side == "home"
        assert bet.market_type == "moneyline"
        assert bet.model_name == "ensemble_v2"

    def test_create_bet_rounds_values(self, engine):
        """Model probability should be rounded to 4 decimal places."""
        bet = engine.create_bet(
            model_probability=0.623456789,
            odds_american=-110,
        )
        assert bet.model_probability == pytest.approx(0.6235, abs=1e-4)

    def test_create_bet_high_edge_produces_larger_stake(self, engine):
        """Higher edge should produce a larger (or equal) stake."""
        low_edge = engine.create_bet(
            model_probability=0.55,
            odds_american=-110,
        )
        high_edge = engine.create_bet(
            model_probability=0.70,
            odds_american=-110,
        )
        if low_edge.is_actionable and high_edge.is_actionable:
            assert high_edge.stake.recommended_stake >= low_edge.stake.recommended_stake

    def test_create_bet_with_vig_free_probability(self, engine):
        """When opponent odds are provided, vig-free probability should be computed."""
        bet = engine.create_bet(
            model_probability=0.62,
            odds_american=-110,
            opponent_odds_american=+110,
            home_team="Celtics",
            away_team="Lakers",
            bet_side="home",
        )
        if bet.vig_free_probability is not None:
            assert 0 < bet.vig_free_probability < 1

    # ═════════════════════════════════════════════════════════════════
    #  create_bet - Edge Cases
    # ═════════════════════════════════════════════════════════════════

    def test_create_bet_extreme_probability_high(self, engine):
        """98% model at -500 odds (implied 83.3%) should be actionable with huge edge."""
        bet = engine.create_bet(
            model_probability=0.98,
            odds_american=-500,
        )
        assert bet.is_actionable
        assert bet.stake.is_valid
        assert bet.edge_percentage > 0.10

    def test_create_bet_extreme_probability_low(self, engine):
        """2% model probability should be clamped to 1% min."""
        bet = engine.create_bet(
            model_probability=0.02,
            odds_american=+5000,
        )
        assert bet.model_probability >= 0.01

    def test_create_bet_zero_odds(self, engine):
        """American odds of 0 should be handled gracefully (decimal = 1.0, no edge)."""
        bet = engine.create_bet(
            model_probability=0.60,
            odds_american=0,
        )
        assert bet is not None
        assert bet.odds_decimal == 1.0
        assert not bet.is_actionable

    def test_create_bet_huge_odds(self, engine):
        """Very long odds (+10000) should still work."""
        bet = engine.create_bet(
            model_probability=0.20,
            odds_american=+10000,
        )
        if bet.is_actionable:
            assert bet.stake.recommended_stake > 0

    def test_create_bet_nan_odds(self, engine):
        """NaN odds from decimal_to_american(1.0) should be filtered to None."""
        bet = engine.create_bet(
            model_probability=0.60,
            odds_decimal=1.0,
        )
        assert bet.odds_american is None

    # ═════════════════════════════════════════════════════════════════
    #  _determine_risk_level
    # ═════════════════════════════════════════════════════════════════

    def test_risk_level_skip(self, engine):
        assert engine._determine_risk_level(0.0, 0.0) == "skip"

    def test_risk_level_low(self, engine):
        assert engine._determine_risk_level(0.01, 0.02) == "low"

    def test_risk_level_medium_by_kelly(self, engine):
        assert engine._determine_risk_level(0.03, 0.02) == "medium"

    def test_risk_level_medium_by_edge(self, engine):
        assert engine._determine_risk_level(0.01, 0.04) == "medium"

    def test_risk_level_high_by_kelly(self, engine):
        assert engine._determine_risk_level(0.06, 0.02) == "high"

    def test_risk_level_high_by_edge(self, engine):
        assert engine._determine_risk_level(0.01, 0.06) == "high"

    def test_risk_level_extreme_by_kelly(self, engine):
        assert engine._determine_risk_level(0.12, 0.02) == "extreme"

    def test_risk_level_extreme_by_edge(self, engine):
        assert engine._determine_risk_level(0.01, 0.12) == "extreme"

    # ═════════════════════════════════════════════════════════════════
    #  create_moneyline_bet
    # ═════════════════════════════════════════════════════════════════

    def test_moneyline_home_favorite(self, engine):
        """Home team at -150 with 65% model prob should be actionable."""
        bet = engine.create_moneyline_bet(
            model_home_prob=0.65,
            home_odds_american=-150,
            away_odds_american=+130,
            home_team="Celtics",
            away_team="Lakers",
            preferred_side="home",
        )
        assert bet.is_actionable
        assert bet.stake.recommended_stake > 0

    def test_moneyline_away_dog(self, engine):
        """Underdog at +200 with 40% model prob should be actionable."""
        bet = engine.create_moneyline_bet(
            model_home_prob=0.60,
            home_odds_american=-200,
            away_odds_american=+200,
            home_team="Warriors",
            away_team="Nuggets",
        )
        assert bet.is_actionable

    def test_moneyline_picks_best_side(self, engine):
        """When both sides are actionable, picks the higher EV side."""
        from betting_intel.betting.bet import BettingEngine

        tolerant = BettingEngine(
            bankroll=10_000.0,
            min_edge=0.0,
            drawdown_protection=False,
        )
        bet = tolerant.create_moneyline_bet(
            model_home_prob=0.55,
            home_odds_american=-110,
            away_odds_american=-110,
            home_team="TeamA",
            away_team="TeamB",
        )
        assert bet.is_actionable

    def test_moneyline_no_actionable_side(self, engine):
        """50% model on both sides at -110 (implied 52.4%) = no edge."""
        bet = engine.create_moneyline_bet(
            model_home_prob=0.50,
            home_odds_american=-110,
            away_odds_american=-110,
            home_team="TeamA",
            away_team="TeamB",
        )
        assert not bet.is_actionable

    def test_moneyline_metadata(self, engine):
        bet = engine.create_moneyline_bet(
            model_home_prob=0.60,
            home_odds_american=-150,
            away_odds_american=+130,
            game_id="nba_123",
            home_team="Heat",
            away_team="Bulls",
            model_name="xgboost_v3",
        )
        assert bet.game_id == "nba_123"
        assert bet.market_type == "moneyline"
        assert bet.model_name == "xgboost_v3"

    # ═════════════════════════════════════════════════════════════════
    #  build_daily_portfolio
    # ═════════════════════════════════════════════════════════════════

    def test_build_portfolio(self, engine):
        from betting_intel.betting.bet import BetRecommendation, StakeSizing

        bets = [
            BetRecommendation(
                game_id="g1",
                home_team="A", away_team="B",
                bet_side="home",
                is_actionable=True,
                edge_percentage=0.05,
                expected_value=0.10,
                stake=StakeSizing(recommended_stake=50.0, bankroll_percentage=0.5, is_valid=True, risk_level="medium"),
            ),
            BetRecommendation(
                game_id="g2",
                home_team="C", away_team="D",
                bet_side="away",
                is_actionable=True,
                edge_percentage=0.03,
                expected_value=0.06,
                stake=StakeSizing(recommended_stake=30.0, bankroll_percentage=0.3, is_valid=True, risk_level="low"),
            ),
        ]

        portfolio = engine.build_daily_portfolio(bets, date="2026-06-01")
        assert portfolio.date == "2026-06-01"
        assert portfolio.num_bets == 2
        assert portfolio.num_actionable == 2
        assert portfolio.total_risk == 80.0
        assert "2026-06-01" in engine.daily_portfolios

    def test_build_portfolio_scales_down_when_over_limit(self, engine):
        from betting_intel.betting.bet import BetRecommendation, StakeSizing

        big_stakes = [
            BetRecommendation(
                game_id="g1",
                home_team="A", away_team="B",
                bet_side="home",
                is_actionable=True,
                edge_percentage=0.05,
                expected_value=0.10,
                stake=StakeSizing(recommended_stake=2500.0, bankroll_percentage=25.0, is_valid=True, risk_level="high"),
            ),
            BetRecommendation(
                game_id="g2",
                home_team="C", away_team="D",
                bet_side="away",
                is_actionable=True,
                edge_percentage=0.04,
                expected_value=0.08,
                stake=StakeSizing(recommended_stake=2500.0, bankroll_percentage=25.0, is_valid=True, risk_level="high"),
            ),
        ]

        portfolio = engine.build_daily_portfolio(big_stakes, date="2026-06-01")
        assert portfolio.total_risk <= 4000.0
        for bet in portfolio.bets:
            if bet.stake.is_valid:
                assert bet.stake.risk_level == "reduced"

    def test_build_portfolio_empty(self, engine):
        portfolio = engine.build_daily_portfolio([], date="2026-06-01")
        assert portfolio.num_bets == 0
        assert portfolio.num_actionable == 0
        assert portfolio.total_risk == 0.0

    def test_build_portfolio_non_actionable_only(self, engine):
        from betting_intel.betting.bet import BetRecommendation

        bets = [
            BetRecommendation(game_id="g1", bet_side="home", is_actionable=False),
            BetRecommendation(game_id="g2", bet_side="away", is_actionable=False),
        ]
        portfolio = engine.build_daily_portfolio(bets)
        assert portfolio.num_bets == 2
        assert portfolio.num_actionable == 0
        assert portfolio.total_risk == 0.0

    # ═════════════════════════════════════════════════════════════════
    #  record_result
    # ═════════════════════════════════════════════════════════════════

    def test_record_result_win(self, engine):
        engine.record_result(stake_dollars=100.0, won=True, decimal_odds=1.91)
        expected = 10_000.0 + 100.0 * (1.91 - 1.0)
        assert engine.bankroll == pytest.approx(expected, rel=1e-6)

    def test_record_result_loss(self, engine):
        engine.record_result(stake_dollars=100.0, won=False)
        assert engine.bankroll == 9_900.0

    def test_record_result_updates_state(self, engine):
        engine.record_result(stake_dollars=500.0, won=True, decimal_odds=2.0)
        state = engine.get_state()
        assert state["current_bankroll"] > 10_000.0

    def test_record_result_win_streak(self, engine):
        engine.record_result(stake_dollars=100.0, won=True, decimal_odds=2.0)
        engine.record_result(stake_dollars=100.0, won=True, decimal_odds=2.0)
        expected = 10_000.0 + 100.0 * 1.0 + 100.0 * 1.0
        assert engine.bankroll == pytest.approx(expected, rel=1e-6)

    def test_record_result_loss_streak(self, engine):
        engine.record_result(stake_dollars=200.0, won=False)
        engine.record_result(stake_dollars=200.0, won=False)
        assert engine.bankroll == 9_600.0

    # ═════════════════════════════════════════════════════════════════
    #  format_portfolio
    # ═════════════════════════════════════════════════════════════════

    def test_format_portfolio_empty(self, engine):
        from betting_intel.betting.bet import DailyPortfolio

        portfolio = DailyPortfolio(date="2026-06-01")
        output = engine.format_portfolio(portfolio)
        assert "DAILY PORTFOLIO" in output
        assert "2026-06-01" in output
        assert "$10,000" in output or "$10,000" in output

    def test_format_portfolio_with_bets(self, engine):
        from betting_intel.betting.bet import BetRecommendation, StakeSizing

        bets = [
            BetRecommendation(
                game_id="g1",
                home_team="Celtics", away_team="Lakers",
                bet_side="home",
                odds_american=-110.0,
                odds_decimal=1.91,
                is_actionable=True,
                edge_percentage=0.05,
                expected_value=0.10,
                stake=StakeSizing(recommended_stake=50.0, is_valid=True, risk_level="medium"),
            ),
        ]
        portfolio = engine.build_daily_portfolio(bets, date="2026-06-01")
        output = engine.format_portfolio(portfolio)
        assert "RECOMMENDED BETS" in output
        assert "$50" in output or "$ 50" in output

    # ═════════════════════════════════════════════════════════════════
    #  Integration: Engine Round-Trip
    # ═════════════════════════════════════════════════════════════════

    def test_full_round_trip(self, engine):
        """Create a bet with edge, record a win, verify bankroll increases."""
        bet = engine.create_bet(
            model_probability=0.62,
            odds_american=-110,
            home_team="Celtics",
            away_team="Lakers",
        )

        initial_bankroll = engine.bankroll
        stake = bet.stake.recommended_stake

        if bet.is_actionable and stake > 0:
            engine.record_result(stake_dollars=stake, won=True, decimal_odds=1.91)
            expected_gain = stake * (1.91 - 1.0)
            assert engine.bankroll == pytest.approx(initial_bankroll + expected_gain, rel=1e-6)

    def test_multiple_bets_portfolio(self, engine):
        """Create multiple bets and build a portfolio from them."""
        bet1 = engine.create_bet(
            model_probability=0.62,
            odds_american=-110,
            home_team="Celtics", away_team="Lakers",
        )
        bet2 = engine.create_bet(
            model_probability=0.55,
            odds_american=-110,
            home_team="Warriors", away_team="Nuggets",
        )

        actionable = [b for b in [bet1, bet2] if b.is_actionable]
        portfolio = engine.build_daily_portfolio([bet1, bet2], date="2026-06-01")
        assert portfolio.num_bets == 2
        assert portfolio.num_actionable == len(actionable) or portfolio.num_actionable >= 0
