"""
Unit tests for the EV formula in betting_intel/betting/ev.py.

Covers:
  - American ↔ Decimal ↔ Implied Probability conversions (all edge cases)
  - Vig-free probability computation (balanced, unbalanced, arb markets)
  - ExpectedValueEngine.calculate() with zero odds, extreme probabilities,
    missing odds, boundary thresholds
  - calculate_moneyline, calculate_total, analyze_game
  - Convenience functions (calculate_expected_value, edge_from_probabilities)
"""

from __future__ import annotations

import math

import pytest

from betting_intel.betting.ev import (
    # Engine
    ExpectedValueEngine,
    EVResult,
    BetSide,
    MarketType,
    # Conversions
    american_to_decimal,
    decimal_to_american,
    american_to_implied_prob,
    decimal_to_implied_prob,
    compute_vig_free_prob,
    compute_vig_free_total,
    # Convenience
    calculate_expected_value,
    edge_from_probabilities,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONVERSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestAmericanToDecimal:
    """american_to_decimal(american_odds: float) -> float"""

    def test_positive_odds(self):
        """+150 American → 2.50 decimal."""
        assert american_to_decimal(150) == pytest.approx(2.50, abs=0.001)

    def test_negative_odds(self):
        """-110 American → 1.9091 decimal."""
        result = american_to_decimal(-110)
        assert result == pytest.approx(1.90909, abs=0.001)

    def test_even_money_positive(self):
        """+100 American → 2.00 decimal."""
        assert american_to_decimal(100) == pytest.approx(2.00, abs=0.001)

    def test_even_money_negative(self):
        """-100 American → 2.00 decimal."""
        assert american_to_decimal(-100) == pytest.approx(2.00, abs=0.001)

    def test_zero_odds(self):
        """0 American → 1.00 decimal (degenerate, but should not crash)."""
        assert american_to_decimal(0) == pytest.approx(1.00, abs=0.001)

    def test_extreme_favorite(self):
        """-10000 American → 1.01 decimal."""
        result = american_to_decimal(-10000)
        assert result == pytest.approx(1.01, abs=0.001)

    def test_extreme_underdog(self):
        """+10000 American → 101.00 decimal."""
        result = american_to_decimal(10000)
        assert result == pytest.approx(101.00, abs=0.001)

    def test_negative_odds_boundary(self):
        """-101 → ~1.9901. Very edge of negative branch."""
        result = american_to_decimal(-101)
        assert result == pytest.approx(1.9901, abs=0.001)

    def test_positive_odds_boundary(self):
        """+101 → 2.01."""
        result = american_to_decimal(101)
        assert result == pytest.approx(2.01, abs=0.001)

    def test_small_positive_odds(self):
        """+1 → 1.01."""
        assert american_to_decimal(1) == pytest.approx(1.01, abs=0.001)

    def test_small_negative_odds(self):
        """-1 → 101.00 (1 + 100/1 = 101)."""
        result = american_to_decimal(-1)
        assert result == pytest.approx(101.00, abs=0.001)


class TestDecimalToAmerican:
    """decimal_to_american(decimal_odds: float) -> float"""

    def test_decimal_above_two(self):
        """2.50 decimal → +150 American."""
        assert decimal_to_american(2.50) == pytest.approx(150, abs=0.1)

    def test_decimal_below_two(self):
        """1.91 decimal → -110 American."""
        result = decimal_to_american(1.91)
        assert result == pytest.approx(-110, abs=1)

    def test_exactly_two(self):
        """2.00 decimal → +100 American."""
        assert decimal_to_american(2.00) == pytest.approx(100, abs=0.1)

    def test_near_one(self):
        """1.01 decimal → -10000 American."""
        result = decimal_to_american(1.01)
        assert result == pytest.approx(-10000, abs=10)

    def test_very_high_decimal(self):
        """101.00 decimal → +10000 American."""
        result = decimal_to_american(101.00)
        assert result == pytest.approx(10000, abs=10)

    def test_decimal_one_or_less(self):
        """1.0 → nan (division by zero protection)."""
        result = decimal_to_american(1.0)
        assert math.isnan(result)

    def test_sub_one_decimal(self):
        """0.5 → nan."""
        result = decimal_to_american(0.5)
        assert math.isnan(result)

    def test_round_trip_positive(self):
        """+150 → decimal → +150."""
        decimal = american_to_decimal(150)
        back = decimal_to_american(decimal)
        assert back == pytest.approx(150, abs=0.1)

    def test_round_trip_negative(self):
        """-110 → decimal → -110."""
        decimal = american_to_decimal(-110)
        back = decimal_to_american(decimal)
        assert back == pytest.approx(-110, abs=1)


class TestAmericanToImpliedProb:
    """american_to_implied_prob(american_odds: float) -> float"""

    def test_even_money(self):
        """+100 → 50% implied."""
        assert american_to_implied_prob(100) == pytest.approx(0.50, abs=0.001)

    def test_negative_even_money(self):
        """-100 → 50% implied."""
        assert american_to_implied_prob(-100) == pytest.approx(0.50, abs=0.001)

    def test_standard_favorite(self):
        """-110 → ~52.38% implied."""
        result = american_to_implied_prob(-110)
        assert result == pytest.approx(0.5238, abs=0.001)

    def test_standard_underdog(self):
        """+150 → 40% implied."""
        result = american_to_implied_prob(150)
        assert result == pytest.approx(0.40, abs=0.001)

    def test_zero_odds(self):
        """0 → 100% implied (degenerate but should not crash)."""
        assert american_to_implied_prob(0) == pytest.approx(1.0, abs=0.001)

    def test_extreme_favorite(self):
        """-10000 → ~99.01% implied."""
        result = american_to_implied_prob(-10000)
        assert result == pytest.approx(0.9901, abs=0.001)

    def test_extreme_underdog(self):
        """+10000 → ~0.99% implied."""
        result = american_to_implied_prob(10000)
        assert result == pytest.approx(0.0099, abs=0.0001)

    def test_negative_just_above_zero(self):
        """-1 → ~0.99% implied (| -1| / (| -1| + 100) = 1/101 ≈ 0.0099)."""
        result = american_to_implied_prob(-1)
        assert result == pytest.approx(0.0099, abs=0.001)

    def test_positive_just_above_zero(self):
        """+1 → ~99.01% implied (100/(1+100) = ~0.9901)."""
        result = american_to_implied_prob(1)
        assert result == pytest.approx(0.9901, abs=0.001)


class TestDecimalToImpliedProb:
    """decimal_to_implied_prob(decimal_odds: float) -> float"""

    def test_even_money(self):
        """2.00 → 50% implied."""
        assert decimal_to_implied_prob(2.00) == pytest.approx(0.50, abs=0.001)

    def test_favorite(self):
        """1.50 → 66.67% implied."""
        result = decimal_to_implied_prob(1.50)
        assert result == pytest.approx(0.6667, abs=0.001)

    def test_underdog(self):
        """3.00 → 33.33% implied."""
        result = decimal_to_implied_prob(3.00)
        assert result == pytest.approx(0.3333, abs=0.001)

    def test_decimal_one(self):
        """1.00 → 1.0 (100% implied — no division by zero)."""
        assert decimal_to_implied_prob(1.0) == pytest.approx(1.0, abs=0.001)

    def test_sub_one_decimal(self):
        """0.5 → 1.0 (clamped)."""
        assert decimal_to_implied_prob(0.5) == pytest.approx(1.0, abs=0.001)

    def test_very_high_decimal(self):
        """100.0 → 1% implied."""
        result = decimal_to_implied_prob(100.0)
        assert result == pytest.approx(0.01, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIG-FREE PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════


class TestVigFreeProbability:
    """compute_vig_free_prob(home_odds, away_odds) -> (home_prob, away_prob)"""

    def test_balanced_market(self):
        """-110 / -110 → 50/50 vig-free."""
        home, away = compute_vig_free_prob(-110, -110)
        assert home == pytest.approx(0.50, abs=0.001)
        assert away == pytest.approx(0.50, abs=0.001)

    def test_unbalanced_market(self):
        """-150 / +130 → vig-free probabilities that sum to 1.0."""
        home, away = compute_vig_free_prob(-150, 130)
        # Implied: home = 150/250 = 0.60, away = 100/230 = 0.4348
        # Total vig: 1.0348, vig-free: home = 0.60/1.0348 = 0.5798, away = 0.4348/1.0348 = 0.4202
        assert home + away == pytest.approx(1.0, abs=0.001)
        assert home > away  # Favorite gets higher prob

    def test_extreme_market(self):
        """-10000 / +2000 → nearly 1.0 / small."""
        home, away = compute_vig_free_prob(-10000, 2000)
        assert home + away == pytest.approx(1.0, abs=0.001)
        assert home > 0.95
        assert away < 0.05

    def test_zero_odds(self):
        """0 / -110 → should not crash, return something sensible."""
        home, away = compute_vig_free_prob(0, -110)
        # 0 → implied = 100/(0+100) = 1.0
        # -110 → implied = 110/210 = 0.5238
        # Total = 1.5238
        # Vig-free: home = 1.0/1.5238 = 0.656, away = 0.5238/1.5238 = 0.344
        assert home + away == pytest.approx(1.0, abs=0.001)
        assert 0 < home < 1 and 0 < away < 1

    def test_same_side_both_positive(self):
        """+150 / +150 → 50/50 with vig."""
        home, away = compute_vig_free_prob(150, 150)
        assert home == pytest.approx(0.50, abs=0.001)
        assert away == pytest.approx(0.50, abs=0.001)

    def test_total_vig_removed(self):
        """Verify the sum of implied probs > 1.0, but sum of vig-free == 1.0."""
        home_imp = american_to_implied_prob(-110)
        away_imp = american_to_implied_prob(-110)
        assert home_imp + away_imp > 1.0  # Vig present

        home_vf, away_vf = compute_vig_free_prob(-110, -110)
        assert home_vf + away_vf == pytest.approx(1.0, abs=0.001)  # Vig removed


class TestVigFreeTotal:
    """compute_vig_free_total(over_odds, under_odds) -> (over_prob, under_prob)"""

    def test_balanced_total(self):
        """-110 / -110 → 50/50."""
        over, under = compute_vig_free_total(-110, -110)
        assert over == pytest.approx(0.50, abs=0.001)
        assert under == pytest.approx(0.50, abs=0.001)

    def test_unbalanced_total(self):
        """-120 / +100 → vig-free that sum to 1.0."""
        over, under = compute_vig_free_total(-120, 100)
        assert over + under == pytest.approx(1.0, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════
# 3. EXPECTED VALUE ENGINE — CORE calculate()
# ═══════════════════════════════════════════════════════════════════════════


class TestEVEngineConstruction:
    """ExpectedValueEngine construction with various thresholds."""

    def test_default_thresholds(self):
        """Default min_edge = 2%, min_ev = 0, clamp [1%, 99%]."""
        engine = ExpectedValueEngine()
        assert engine.min_edge_threshold == pytest.approx(0.02)
        assert engine.min_ev_threshold == pytest.approx(0.0)
        assert engine.min_probability == pytest.approx(0.01)
        assert engine.max_probability == pytest.approx(0.99)
        assert engine.require_positive_ev is True

    def test_custom_thresholds(self):
        """All thresholds configurable."""
        engine = ExpectedValueEngine(
            min_edge_threshold=0.05,
            min_ev_threshold=0.01,
            min_probability=0.001,
            max_probability=0.999,
            require_positive_ev=False,
        )
        assert engine.min_edge_threshold == pytest.approx(0.05)
        assert engine.min_ev_threshold == pytest.approx(0.01)
        assert engine.min_probability == pytest.approx(0.001)
        assert engine.max_probability == pytest.approx(0.999)
        assert engine.require_positive_ev is False

    def test_zero_threshold(self):
        """Zero min_edge means every positive-edge bet is actionable."""
        engine = ExpectedValueEngine(min_edge_threshold=0.0, min_ev_threshold=-float("inf"))
        # 53% at -110 gives edge = 0.53 - 0.5238 = 0.0062 (> 0)
        result = engine.calculate(
            model_probability=0.53,
            market_odds_american=-110,
        )
        assert result.is_actionable is True


class TestEVCalculation:
    """ExpectedValueEngine.calculate() — core EV formula."""

    def test_positive_ev_bet(self):
        """Model 62% at -110 odds should give positive EV ~18.4%."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=0.62,
            market_odds_american=-110,
            game_id="test_1",
            home_team="Lakers",
            away_team="Celtics",
        )
        # Implied prob = 110/210 = 0.5238
        assert result.implied_probability == pytest.approx(0.5238, abs=0.001)
        # Edge = 0.62 - 0.5238 = 0.0962
        assert result.edge_percentage == pytest.approx(0.0962, abs=0.001)
        # Decimal = 1.9091
        assert result.market_odds_decimal == pytest.approx(1.9091, abs=0.001)
        # EV = 0.62 * (1.9091 - 1) - 0.38 * 1 = 0.62 * 0.9091 - 0.38 = 0.5636 - 0.38 = 0.1836
        assert result.expected_value == pytest.approx(0.1836, abs=0.001)
        assert result.is_actionable is True
        assert result.recommendation == "RECOMMEND"
        assert result.game_id == "test_1"
        assert result.home_team == "Lakers"

    def test_negative_ev_bet(self):
        """Model 45% at -110 odds should give negative EV."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.45,
            market_odds_american=-110,
        )
        # Edge = 0.45 - 0.5238 = -0.0738
        assert result.edge_percentage < 0
        assert result.expected_value < 0
        assert result.is_actionable is False

    def test_small_edge_below_threshold(self):
        """Model 52% at -110 odds gives ~ -0.38% edge → not actionable at 2% threshold."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=0.52,
            market_odds_american=-110,
        )
        # Edge = 0.52 - 0.5238 = -0.0038 (below 2% and negative)
        assert result.edge_percentage == pytest.approx(-0.0038, abs=0.001)
        assert result.is_actionable is False

    def test_edge_above_threshold(self):
        """Model 60% at +200 odds gives 20% edge — clearly actionable."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=0.60,
            market_odds_american=200,
        )
        # Implied = 100/300 = 0.3333
        assert result.implied_probability == pytest.approx(0.3333, abs=0.001)
        # Edge = 0.60 - 0.3333 = 0.2667
        assert result.edge_percentage == pytest.approx(0.2667, abs=0.001)
        assert result.is_actionable is True

    def test_edge_exactly_at_threshold(self):
        """Edge at/above threshold (2%) should be actionable (>=)."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        # Edge = 0.5439 - 0.5238095... = 0.02009 (unrounded > 2%)
        result = engine.calculate(
            model_probability=0.5439,
            market_odds_american=-110,
        )
        assert result.edge_percentage == pytest.approx(0.0201, abs=0.001)
        assert result.is_actionable is True

    def test_edge_barely_below_threshold(self):
        """Edge just below threshold (1.99%) should not be actionable."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=0.5436,
            market_odds_american=-110,
        )
        assert result.edge_percentage == pytest.approx(0.0198, abs=0.001)
        assert result.is_actionable is False

    def test_model_probability_zero(self):
        """Model prob = 0 should be clamped to min_probability (1%)."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.0,
            market_odds_american=-110,
        )
        assert result.model_probability >= engine.min_probability
        assert result.is_actionable is False  # Edge will be negative

    def test_model_probability_one(self):
        """Model prob = 1.0 should be clamped to max_probability (99%)."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=1.0,
            market_odds_american=-110,
        )
        assert result.model_probability <= engine.max_probability
        assert result.model_probability == pytest.approx(0.99)

    def test_probability_out_of_range_not_actionable(self):
        """If clamping results in prob at boundary, edge check still applies."""
        engine = ExpectedValueEngine(min_edge_threshold=0.0, min_ev_threshold=-float("inf"))
        result = engine.calculate(
            model_probability=0.001,
            market_odds_american=-110,
        )
        assert result.model_probability == pytest.approx(0.01)

    def test_extreme_underdog_model(self):
        """Model says 90% at +1000 odds → huge edge."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=0.90,
            market_odds_american=1000,
        )
        # Implied = 100/1100 = 0.0909
        # Edge = 0.90 - 0.0909 = 0.8091
        assert result.edge_percentage == pytest.approx(0.8091, abs=0.001)
        assert result.is_actionable is True
        # EV = 0.90 * (11 - 1) - 0.10 * 1 = 0.90 * 10 - 0.10 = 9.0 - 0.10 = 8.90
        assert result.expected_value == pytest.approx(8.90, abs=0.01)

    def test_extreme_favorite_model(self):
        """Model says 99% at -10000 odds → very small edge."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        # Model = 0.99 (clamped from 1.0)
        result = engine.calculate(
            model_probability=0.999,
            market_odds_american=-10000,
        )
        # Implied = 10000/10100 = 0.9901
        # Edge = 0.99 - 0.9901 = -0.0001
        assert result.edge_percentage == pytest.approx(-0.0001, abs=0.001)
        assert result.is_actionable is False

    def test_no_odds_provided(self):
        """No odds at all → ERROR recommendation."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.60,
        )
        assert result.is_actionable is False
        assert "ERROR" in result.recommendation
        assert result.market_odds_decimal is None

    def test_decimal_odds_provided(self):
        """Decimal odds input works correctly."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=0.62,
            market_odds_decimal=1.9091,
        )
        assert result.expected_value == pytest.approx(0.1836, abs=0.01)
        assert result.is_actionable is True

    def test_american_odds_takes_precedence(self):
        """When both american and decimal provided, american is used."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.60,
            market_odds_american=-110,
            market_odds_decimal=10.0,
        )
        # Should use -110, not 10.0
        assert result.market_odds_decimal == pytest.approx(1.9091, abs=0.001)
        assert result.implied_probability == pytest.approx(0.5238, abs=0.001)

    def test_vig_free_with_opponent_odds(self):
        """When opponent_odds_american is provided, vig_free_probability is computed."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.62,
            market_odds_american=-110,
            opponent_odds_american=-110,
        )
        assert result.vig_free_probability is not None
        assert result.vig_free_probability == pytest.approx(0.50, abs=0.001)

    def test_vig_free_home_side(self):
        """Home side vig-free calculation with asymmetric odds."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.65,
            market_odds_american=-150,
            opponent_odds_american=130,
            bet_side="home",
        )
        # Vig-free home = (150/250) / (150/250 + 100/230) = 0.60 / 1.0348 = 0.5798
        assert result.vig_free_probability is not None
        assert result.vig_free_probability == pytest.approx(0.5798, abs=0.001)

    def test_vig_free_away_side(self):
        """Away side vig-free picks the other odds."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.35,
            market_odds_american=130,
            opponent_odds_american=-150,
            bet_side="away",
        )
        # Vig-free away = (100/230) / (150/250 + 100/230) = 0.4348 / 1.0348 = 0.4202
        assert result.vig_free_probability is not None
        assert result.vig_free_probability == pytest.approx(0.4202, abs=0.001)


class TestEVWithDifferentOddsFormats:
    """EV calculation with various American odds values."""

    @pytest.mark.parametrize(
        "american_odds, model_prob, expected_edge, expected_ev, actionable",
        [
            # (odds, model_prob, expected_edge, expected_ev, actionable)
            (-110, 0.55, 0.0262, 0.05, True),    # Slight edge
            (-110, 0.50, -0.0238, -0.0455, False), # No edge
            (+150, 0.50, 0.10, 0.25, True),       # Decent edge on dog
            (+150, 0.35, -0.05, -0.125, False),    # Below implied
            (-200, 0.70, 0.0333, 0.05, True),     # Favorite with edge
            (-200, 0.60, -0.0667, -0.10, False),   # Favorite no edge
            (+500, 0.25, 0.0833, 0.50, True),     # Big dog with edge
            (+500, 0.12, -0.0467, -0.28, False),   # Big dog no edge
        ],
    )
    def test_parametrized_ev(
        self, american_odds, model_prob, expected_edge, expected_ev, actionable,
    ):
        """Parametrized test for various American odds scenarios."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.calculate(
            model_probability=model_prob,
            market_odds_american=american_odds,
        )
        assert result.edge_percentage == pytest.approx(expected_edge, abs=0.01)
        assert result.expected_value == pytest.approx(expected_ev, abs=0.01)
        assert result.is_actionable == actionable

    def test_ev_symmetry(self):
        """If edge is positive on one side, it's negative on the other (vig aside)."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        home_result = engine.calculate(
            model_probability=0.60,
            market_odds_american=-110,
        )
        away_result = engine.calculate(
            model_probability=0.40,
            market_odds_american=-110,
        )
        # If home has positive edge, away should have negative edge
        assert home_result.is_actionable != away_result.is_actionable


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATE MONEYLINE AND TOTAL METHODS
# ═══════════════════════════════════════════════════════════════════════════


class TestCalculateMoneyline:
    """ExpectedValueEngine.calculate_moneyline()"""

    def test_moneyline_both_sides(self):
        """Both sides of moneyline returned correctly."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        home_ev, away_ev = engine.calculate_moneyline(
            model_home_prob=0.62,
            home_odds_american=-110,
            away_odds_american=-110,
            game_id="game_1",
            home_team="Lakers",
            away_team="Celtics",
        )
        assert home_ev.bet_side == "home"
        assert away_ev.bet_side == "away"
        assert home_ev.home_team == "Lakers"
        assert away_ev.away_team == "Celtics"
        # Home prob = 0.62, implied 0.5238 → positive edge
        assert home_ev.is_actionable is True
        # Away prob = 0.38, implied 0.5238 → negative edge
        assert away_ev.is_actionable is False

    def test_moneyline_vig_free(self):
        """Moneyline calculation passes opponent odds for vig-free calc."""
        engine = ExpectedValueEngine()
        home_ev, away_ev = engine.calculate_moneyline(
            model_home_prob=0.62,
            home_odds_american=-110,
            away_odds_american=-110,
        )
        assert home_ev.vig_free_probability is not None
        assert away_ev.vig_free_probability is not None
        assert home_ev.vig_free_probability == pytest.approx(0.50, abs=0.001)
        assert away_ev.vig_free_probability == pytest.approx(0.50, abs=0.001)


class TestCalculateTotal:
    """ExpectedValueEngine.calculate_total()"""

    def test_total_both_sides(self):
        """Both sides of totals market returned correctly."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        over_ev, under_ev = engine.calculate_total(
            model_over_prob=0.62,
            over_odds_american=-110,
            under_odds_american=-110,
            game_id="game_1",
            home_team="Lakers",
            away_team="Celtics",
        )
        assert over_ev.bet_side == "over"
        assert under_ev.bet_side == "under"
        assert over_ev.is_actionable is True
        assert under_ev.is_actionable is False

    def test_total_market_type(self):
        """Totals calculation uses market_type='total'."""
        engine = ExpectedValueEngine()
        over_ev, _ = engine.calculate_total(
            model_over_prob=0.55,
            over_odds_american=-110,
            under_odds_american=-110,
        )
        assert over_ev.market_type == "total"


# ═══════════════════════════════════════════════════════════════════════════
# 5. ANALYZE GAME
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalyzeGame:
    """ExpectedValueEngine.analyze_game() — full game analysis."""

    def test_full_analysis(self):
        """All markets analyzed. Best bet is the one with highest EV."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.analyze_game(
            game_id="game_1",
            home_team="Lakers",
            away_team="Celtics",
            commence_time="2026-06-01T19:00:00Z",
            model_home_prob=0.62,
            model_total_over_prob=0.55,
            model_home_cover_prob=0.58,
            home_ml_odds=-110,
            away_ml_odds=-110,
            total_over_odds=-110,
            total_under_odds=-110,
            home_spread_odds=-110,
            away_spread_odds=-110,
        )
        assert result.game_id == "game_1"
        assert result.home_team == "Lakers"
        assert result.away_team == "Celtics"
        # Home ML has highest edge (9.62%) → should be best bet
        assert result.home_ml is not None
        assert result.away_ml is not None
        assert result.over_total is not None
        assert result.under_total is not None
        assert result.home_spread is not None
        assert result.away_spread is not None
        assert result.best_edge is not None
        assert result.best_edge.bet_side == "home"
        assert result.num_actionable >= 1

    def test_no_market_data(self):
        """Only moneyline data provided."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.analyze_game(
            game_id="game_2",
            home_team="Lakers",
            away_team="Celtics",
            commence_time="2026-06-01T19:00:00Z",
            model_home_prob=0.62,
            home_ml_odds=-110,
            away_ml_odds=-110,
        )
        assert result.home_ml is not None
        assert result.away_ml is not None
        assert result.over_total is None
        assert result.under_total is None
        assert result.home_spread is None
        assert result.away_spread is None

    def test_no_actionable_bets(self):
        """All edges below threshold — num_actionable = 0, best_edge = None."""
        engine = ExpectedValueEngine(min_edge_threshold=0.10)
        result = engine.analyze_game(
            game_id="game_3",
            home_team="Lakers",
            away_team="Celtics",
            commence_time="2026-06-01T19:00:00Z",
            model_home_prob=0.55,
            model_total_over_prob=0.52,
            home_ml_odds=-110,
            away_ml_odds=-110,
            total_over_odds=-110,
            total_under_odds=-110,
        )
        # Home ML edge = 0.55 - 0.5238 = 0.0262 → below 10%
        # Over edge = 0.52 - 0.5238 = -0.0038 → negative
        assert result.num_actionable == 0
        # best_edge could be set even if not actionable — it tracks highest EV
        # Actually looking at the code: best_edge is only set if is_actionable
        assert result.best_edge is None

    def test_away_side_better(self):
        """When away side has better edge, it should be best_edge."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.analyze_game(
            game_id="game_4",
            home_team="Lakers",
            away_team="Celtics",
            commence_time="2026-06-01T19:00:00Z",
            model_home_prob=0.38,  # Model thinks home loses
            home_ml_odds=-110,
            away_ml_odds=-110,
        )
        # Home ML edge = 0.38 - 0.5238 = -0.1438 → negative
        # Away ML edge = 0.62 - 0.5238 = 0.0962 → positive
        assert result.away_ml.is_actionable is True
        assert result.home_ml.is_actionable is False
        assert result.best_edge is not None
        assert result.best_edge.bet_side == "away"


# ═══════════════════════════════════════════════════════════════════════════
# 6. FORMATTING
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatting:
    """EV result formatting for display."""

    def test_format_ev_result_actionable(self):
        """Format actionable EV result."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.60,
            market_odds_american=-110,
            bet_side="home",
            market_type="moneyline",
        )
        formatted = engine.format_ev_result(result)
        assert "✅" in formatted
        assert "HOME" in formatted
        assert "moneyline" in formatted
        assert "RECOMMEND" in formatted

    def test_format_ev_result_not_actionable(self):
        """Format non-actionable EV result."""
        engine = ExpectedValueEngine()
        result = engine.calculate(
            model_probability=0.40,
            market_odds_american=-110,
        )
        formatted = engine.format_ev_result(result)
        assert "⛔" in formatted

    def test_format_ev_result_none(self):
        """Format None EV result."""
        engine = ExpectedValueEngine()
        formatted = engine.format_ev_result(None)  # type: ignore
        assert formatted == "No EV data"

    def test_format_game_analysis(self):
        """Full game analysis formatting."""
        engine = ExpectedValueEngine(min_edge_threshold=0.02)
        result = engine.analyze_game(
            game_id="game_1",
            home_team="Lakers",
            away_team="Celtics",
            commence_time="2026-06-01T19:00:00Z",
            model_home_prob=0.62,
            home_ml_odds=-110,
            away_ml_odds=-110,
        )
        formatted = engine.format_game_analysis(result)
        assert "Celtics" in formatted
        assert "Lakers" in formatted
        assert "BEST BET" in formatted
        assert "Actionable bets" in formatted


# ═══════════════════════════════════════════════════════════════════════════
# 7. CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestCalculateExpectedValue:
    """calculate_expected_value() convenience function."""

    def test_basic_calculation(self):
        """Standard EV calc via convenience function."""
        result = calculate_expected_value(
            model_probability=0.62,
            market_odds_american=-110,
        )
        assert result["edge_percentage"] == pytest.approx(0.0962, abs=0.001)
        assert result["expected_value"] == pytest.approx(0.1836, abs=0.001)
        assert result["is_actionable"] is True

    def test_custom_min_edge(self):
        """Custom min_edge passed through."""
        result = calculate_expected_value(
            model_probability=0.55,
            market_odds_american=-110,
            min_edge=0.05,
        )
        # Edge = 0.55 - 0.5238 = 0.0262 < 0.05 → not actionable
        assert result["is_actionable"] is False

    def test_decimal_odds_input(self):
        """Decimal odds with convenience function."""
        result = calculate_expected_value(
            model_probability=0.62,
            market_odds_decimal=1.9091,
        )
        assert result["expected_value"] == pytest.approx(0.1836, abs=0.01)

    def test_no_odds(self):
        """No odds returns error."""
        result = calculate_expected_value(model_probability=0.60)
        assert result["is_actionable"] is False


class TestEdgeFromProbabilities:
    """edge_from_probabilities() simple helper."""

    def test_positive_edge(self):
        """Model above market → positive edge."""
        assert edge_from_probabilities(0.60, 0.50) == pytest.approx(0.10)

    def test_negative_edge(self):
        """Model below market → negative edge."""
        assert edge_from_probabilities(0.40, 0.50) == pytest.approx(-0.10)

    def test_zero_edge(self):
        """Model equals market → zero edge."""
        assert edge_from_probabilities(0.50, 0.50) == pytest.approx(0.0)

    def test_extreme_values(self):
        """Edge with extreme probabilities still works."""
        assert edge_from_probabilities(0.99, 0.01) == pytest.approx(0.98)


# ═══════════════════════════════════════════════════════════════════════════
# 8. EVResult DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════


class TestEVResult:
    """EVResult dataclass integrity."""

    def test_default_values(self):
        """Default EVResult has reasonable values."""
        result = EVResult()
        assert result.model_probability == 0.0
        assert result.is_actionable is False
        assert result.recommendation == ""

    def test_actionable_recommendation(self):
        """Actionable result has RECOMMEND."""
        result = EVResult(
            model_probability=0.60,
            edge_percentage=0.10,
            expected_value=0.20,
            is_actionable=True,
            recommendation="RECOMMEND",
        )
        assert result.is_actionable is True
        assert "RECOMMEND" in result.recommendation
