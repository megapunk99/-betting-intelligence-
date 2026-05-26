"""Tests for the betting recommendation engine — bet types, engine, ranker, and player props."""

from __future__ import annotations

import pytest


class TestBetTypes:
    """Test all bet type constructors."""

    def test_moneyline_bet(self):
        from betting_intel.recommendations import MoneylineBet, BetType, Confidence

        bet = MoneylineBet(
            game_id="test_1",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            team="Spurs",
            win_probability=0.58,
            market_implied_prob=0.55,
            confidence=Confidence.HIGH,
            reasoning="Home court",
        )

        assert bet.bet_type == BetType.MONEYLINE
        assert bet.edge_pct == pytest.approx(0.03, abs=1e-10)
        assert bet.win_probability == 0.58
        assert bet.bet_side == "Spurs"
        assert bet.action  # Has an action string
        assert bet.as_dict()["bet_type"] == "moneyline"

    def test_spread_bet(self):
        from betting_intel.recommendations import SpreadBet, BetType

        bet = SpreadBet(
            game_id="test_2",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            team="Spurs",
            spread_line=-3.5,
            predicted_margin=5.0,
        )

        assert bet.bet_type == BetType.SPREAD
        assert bet.edge_pct > 0  # Should have positive edge
        assert "-3.5" in bet.bet_side
        assert bet.action  # Has an actionable instruction

    def test_total_bet_over(self):
        from betting_intel.recommendations import TotalBet, BetType

        bet = TotalBet(
            game_id="test_3",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            side="OVER",
            market_total=214.0,
            predicted_total=218.0,
        )

        assert bet.bet_type == BetType.TOTAL_POINTS
        assert bet.edge_pct > 0
        assert "OVER" in bet.bet_side

    def test_total_bet_under(self):
        from betting_intel.recommendations import TotalBet

        bet = TotalBet(
            game_id="test_4",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            side="UNDER",
            market_total=214.0,
            predicted_total=210.0,
        )

        assert bet.edge_pct > 0
        assert "UNDER" in bet.bet_side

    def test_player_prop_bet(self):
        from betting_intel.recommendations import PlayerPropBet, BetType

        bet = PlayerPropBet(
            game_id="test_5",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            player_name="Wemby",
            prop_type=BetType.PLAYER_POINTS,
            market_line=24.5,
            predicted_value=27.0,
            side="OVER",
        )

        assert bet.bet_type == BetType.PLAYER_POINTS
        assert bet.edge_pct > 0
        assert "Wemby" in bet.bet_side
        assert "player_prop" in bet.tags

    def test_quarter_bet(self):
        from betting_intel.recommendations import QuarterBet, BetType

        bet = QuarterBet(
            game_id="test_6",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            quarter=1,
            team="Spurs",
            win_probability=0.60,
        )

        assert bet.bet_type == BetType.FIRST_QUARTER_WINNER
        assert "1st Quarter" in bet.bet_side

    def test_half_total_bet(self):
        from betting_intel.recommendations import HalfTotalBet, BetType

        bet = HalfTotalBet(
            game_id="test_7",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            side="OVER",
            market_half_total=108.0,
            predicted_half_total=112.0,
        )

        assert bet.bet_type == BetType.FIRST_HALF_TOTAL
        assert "1st Half" in bet.bet_side

    def test_team_total_bet(self):
        from betting_intel.recommendations import TeamTotalBet, BetType

        bet = TeamTotalBet(
            game_id="test_8",
            game_date="2026-05-28",
            matchup="Thunder @ Spurs",
            team="Spurs",
            side="OVER",
            market_team_total=107.5,
            predicted_team_total=110.0,
        )

        assert bet.bet_type == BetType.TEAM_TOTAL
        assert "Spurs" in bet.bet_side

    def test_parlay_suggestion(self):
        from betting_intel.recommendations import MoneylineBet, ParlaySuggestion, BetType, Confidence

        leg1 = MoneylineBet(
            game_id="g1", game_date="2026-05-28",
            matchup="A @ B", team="B",
            win_probability=0.60,
            confidence=Confidence.HIGH,
            reasoning="",
        )
        leg2 = MoneylineBet(
            game_id="g2", game_date="2026-05-28",
            matchup="C @ D", team="C",
            win_probability=0.55,
            confidence=Confidence.MEDIUM,
            reasoning="",
        )

        parlay = ParlaySuggestion(legs=[leg1, leg2])
        assert parlay.bet_type == BetType.PARLAY
        assert parlay.is_parlay
        assert len(parlay.legs) == 2
        assert "Parlay" in parlay.bet_side

    def test_parlay_requires_legs(self):
        from betting_intel.recommendations import ParlaySuggestion

        with pytest.raises(ValueError, match="at least one leg"):
            ParlaySuggestion(legs=[])

    def test_negative_edge_bet(self):
        """Negative edge bets should still be created but flagged."""
        from betting_intel.recommendations import MoneylineBet

        bet = MoneylineBet(
            game_id="test", game_date="2026-05-28",
            matchup="A @ B", team="A",
            win_probability=0.48,
            market_implied_prob=0.55,
        )
        assert bet.edge_pct < 0
        assert not bet.is_clear_pick

    def test_as_dict_serialization(self):
        from betting_intel.recommendations import MoneylineBet, Confidence

        bet = MoneylineBet(
            game_id="test", game_date="2026-05-28",
            matchup="A @ B", team="A",
            win_probability=0.60,
            confidence=Confidence.HIGH,
            reasoning="Test",
            model_name="TestModel",
        )

        d = bet.as_dict()
        assert d["bet_type"] == "moneyline"
        assert d["edge_pct"] >= -1
        assert d["confidence"] == "HIGH"
        assert d["model_name"] == "TestModel"
        assert "action" in d
        assert "stake_dollars" in d


class TestBetTypeEnum:
    """Test the BetType enum utilities."""

    def test_display_names(self):
        from betting_intel.recommendations import BetType

        assert BetType.MONEYLINE.display_name() == "Moneyline"
        assert BetType.TOTAL_POINTS.display_name() == "Total Points O/U"
        assert BetType.PLAYER_PRA.display_name() == "Player Pts + Reb + Ast"

    def test_icons_are_strings(self):
        """Icons should return a non-empty string for each bet type."""
        from betting_intel.recommendations import BetType

        for bt in BetType:
            icon = bt.icon()
            assert isinstance(icon, str), f"Icon for {bt.value} is not a string: {type(icon)}"
            assert len(icon) > 0, f"Empty icon for {bt.value}. Icons must be non-empty."


class TestConfidence:
    """Test the Confidence enum."""

    def test_numeric_values(self):
        from betting_intel.recommendations import Confidence

        assert Confidence.VERY_HIGH.numeric() == 0.9
        assert Confidence.LOW.numeric() == 0.25

    def test_is_clear(self):
        from betting_intel.recommendations import Confidence

        assert Confidence.VERY_HIGH.is_clear()
        assert Confidence.HIGH.is_clear()
        assert not Confidence.MEDIUM.is_clear()
        assert not Confidence.LOW.is_clear()


class TestBetRanker:
    """Test the bet ranking and clear pick detection."""

    @pytest.fixture
    def sample_bets(self):
        from betting_intel.recommendations import (
            MoneylineBet, SpreadBet, TotalBet,
            Confidence,
        )

        good_bets = [
            MoneylineBet(
                game_id="g1", game_date="2026-05-28",
                matchup="A @ B", team="B",
                win_probability=0.65,
                market_implied_prob=0.50,
                confidence=Confidence.VERY_HIGH,
                reasoning="Strong edge",
                model_name="TestModel",
            ),
            SpreadBet(
                game_id="g2", game_date="2026-05-28",
                matchup="C @ D", team="C",
                spread_line=-3.5, predicted_margin=8.0,
                confidence=Confidence.HIGH,
                reasoning="Good spread edge",
            ),
        ]

        bad_bets = [
            MoneylineBet(
                game_id="g3", game_date="2026-05-28",
                matchup="E @ F", team="E",
                win_probability=0.51,
                market_implied_prob=0.50,
                confidence=Confidence.LOW,
                reasoning="No edge",
            ),
            TotalBet(
                game_id="g4", game_date="2026-05-28",
                matchup="G @ H", side="OVER",
                market_total=210.0, predicted_total=211.0,
                confidence=Confidence.VERY_LOW,
            ),
        ]

        return good_bets + bad_bets

    def test_rank_bets(self, sample_bets):
        from betting_intel.recommendations import BetRanker

        ranker = BetRanker()
        ranked = ranker.rank_bets(sample_bets)

        assert len(ranked) == 4
        scores = [b.metadata.get("composite_score", 0) for b in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_clear_picks_detection(self, sample_bets):
        """Clear picks need stake_dollars >= $10, so set that."""
        from betting_intel.recommendations import BetRanker

        # Give all bets a stake so they pass the clear-pick filter
        for bet in sample_bets:
            bet.stake_dollars = 50.0

        ranker = BetRanker(min_edge=0.03)
        clear = ranker.get_clear_picks(sample_bets)

        assert len(clear) >= 1
        for cp in clear:
            assert cp.bet.edge_pct >= 0.03
            assert cp.clear_score > 0
            assert len(cp.reasons) >= 1
            assert cp.risk_level in ("CONSERVATIVE", "MODERATE", "AGGRESSIVE")

    def test_clear_pick_as_dict(self, sample_bets):
        from betting_intel.recommendations import BetRanker

        for bet in sample_bets:
            bet.stake_dollars = 50.0

        ranker = BetRanker()
        clear = ranker.get_clear_picks(sample_bets)
        if clear:
            d = clear[0].as_dict()
            assert "clear_score" in d
            assert "risk_level" in d
            assert "reasons" in d

    def test_empty_bets(self):
        from betting_intel.recommendations import BetRanker

        ranker = BetRanker()
        assert ranker.rank_bets([]) == []
        assert ranker.get_clear_picks([]) == []

    def test_get_summary(self, sample_bets):
        from betting_intel.recommendations import BetRanker

        ranker = BetRanker()
        summary = ranker.get_summary(sample_bets)

        assert summary["total"] == 4
        assert summary["total_stake"] >= 0
        assert "clear_picks" in summary
        assert "by_type" in summary
        assert "clear_picks_detail" in summary


class TestRecommendationEngine:
    """Test the core recommendation engine."""

    def test_engine_creation(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine(bankroll=10_000.0)
        assert engine.bankroll == 10_000.0
        assert engine.min_edge == 0.01

    def test_generate_all_bets(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        bets = engine.generate_all_bets()

        assert len(bets) > 0
        for bet in bets:
            assert bet.game_id
            assert bet.game_date
            assert bet.matchup
            assert bet.bet_type is not None
            assert bet.bet_side
            assert bet.action

    def test_todays_card(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        bets = engine.get_todays_card()
        assert isinstance(bets, list)

    def test_clear_picks(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        clear = engine.get_clear_picks(threshold=0.03)

        assert isinstance(clear, list)
        if clear:
            for cp in clear:
                assert cp.bet.edge_pct >= 0.03
                assert cp.clear_score >= 0
                assert cp.reasons

    def test_get_summary(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        summary = engine.get_summary()

        assert summary["total_bets"] > 0
        assert summary["games_available"] >= 0
        assert summary["avg_edge"] >= 0
        assert summary["max_edge"] >= 0
        assert summary["bankroll"] == 10_000.0

    def test_bet_types_coverage(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        bets = engine.generate_all_bets()
        types_present = set(b.bet_type.value for b in bets)
        assert len(types_present) >= 3, f"Only found types: {types_present}"

    def test_deterministic_output(self):
        from betting_intel.recommendations import RecommendationEngine

        engine1 = RecommendationEngine()
        engine2 = RecommendationEngine()

        bets1 = engine1.generate_all_bets()
        bets2 = engine2.generate_all_bets()

        sides1 = [(b.bet_side, round(b.edge_pct, 4), round(b.stake_dollars, 2)) for b in bets1[:10]]
        sides2 = [(b.bet_side, round(b.edge_pct, 4), round(b.stake_dollars, 2)) for b in bets2[:10]]

        assert sides1 == sides2, "Engine is not deterministic!"

    def test_staking_values(self):
        """Bets with edge high enough to overcome vig should have stakes."""
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine(bankroll=10_000.0)
        bets = engine.generate_all_bets()

        # Find bets with significant edge (≥5%) — these should definitely have stakes
        high_edge_bets = [b for b in bets if b.edge_pct >= 0.05]
        if high_edge_bets:
            for bet in high_edge_bets:
                assert bet.stake_dollars > 0, (
                    f"Bet '{bet.bet_side}' with edge {bet.edge_pct:.4f} has no stake!"
                )
                assert bet.kelly_fraction > 0
        else:
            # At minimum, no crash — edge computation works
            all_positive = [b for b in bets if b.edge_pct > 0]
            assert len(all_positive) > 0

    def test_filter_by_type(self):
        from betting_intel.recommendations import RecommendationEngine, BetType

        engine = RecommendationEngine()
        moneylines = engine.get_bets_by_type(BetType.MONEYLINE)

        for bet in moneylines:
            assert bet.bet_type == BetType.MONEYLINE

    def test_filter_by_league(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        nba_bets = engine.get_bets_by_league("NBA")

        for bet in nba_bets:
            assert bet.league == "NBA"

    def test_rank_by_edge(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine()
        ranked = engine.rank_by_edge()

        assert len(ranked) > 0
        edges = [b.edge_pct for b in ranked]
        assert edges == sorted(edges, reverse=True)


class TestPlayerPropEngine:
    """Test the player prop prediction engine."""

    def test_predict_for_game(self):
        from betting_intel.recommendations.player_props import PlayerPropEngine

        engine = PlayerPropEngine()
        props = engine.predict_for_game(home="Spurs", away="Thunder")

        assert len(props) > 0
        for prop in props:
            assert prop.bet_type.value.startswith("player_")
            assert prop.edge_pct != 0
            assert prop.action

    def test_deterministic_props(self):
        from betting_intel.recommendations.player_props import PlayerPropEngine

        props1 = PlayerPropEngine(seed=42).predict_for_game(home="Celtics", away="Lakers")
        props2 = PlayerPropEngine(seed=42).predict_for_game(home="Celtics", away="Lakers")

        sides1 = [(p.bet_side, round(p.predicted_value, 2)) for p in props1]
        sides2 = [(p.bet_side, round(p.predicted_value, 2)) for p in props2]

        assert sides1 == sides2, "Player props are not deterministic!"

    def test_different_seed_different_props(self):
        from betting_intel.recommendations.player_props import PlayerPropEngine

        props1 = PlayerPropEngine(seed=42).predict_for_game(home="Celtics", away="Lakers")
        props2 = PlayerPropEngine(seed=99).predict_for_game(home="Celtics", away="Lakers")

        sides1 = [(p.bet_side, round(p.predicted_value, 2)) for p in props1]
        sides2 = [(p.bet_side, round(p.predicted_value, 2)) for p in props2]

        assert sides1 != sides2 or len(props1) != len(props2)

    def test_no_players_found(self):
        from betting_intel.recommendations.player_props import PlayerPropEngine

        engine = PlayerPropEngine()
        props = engine.predict_for_game(home="UnknownTeam", away="OtherTeam", league="unknown")
        assert len(props) > 0

    def test_game_id_propagation(self):
        from betting_intel.recommendations.player_props import PlayerPropEngine

        props = PlayerPropEngine().predict_for_game(
            home="Lakers", away="Warriors",
            game_id="lal_vs_gsw", game_date="2026-05-28",
        )

        for prop in props:
            assert prop.game_id == "lal_vs_gsw"
            assert prop.game_date == "2026-05-28"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_engine(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine(include_small_leagues=False)
        bets = engine.generate_all_bets()
        assert len(bets) > 0

    def test_zero_bankroll(self):
        from betting_intel.recommendations import RecommendationEngine

        engine = RecommendationEngine(bankroll=0)
        bets = engine.generate_all_bets()

        for bet in bets:
            assert bet.stake_dollars == 0.0

    def test_negative_edge_not_clear(self):
        from betting_intel.recommendations import BetRanker, MoneylineBet, Confidence

        bet = MoneylineBet(
            game_id="test", game_date="2026-05-28",
            matchup="A @ B", team="A",
            win_probability=0.40,
            market_implied_prob=0.55,
            confidence=Confidence.HIGH,
            reasoning="Bad bet",
        )

        ranker = BetRanker()
        ranked = ranker.rank_bets([bet])
        clear = ranker.get_clear_picks(ranked)

        assert len(clear) == 0
        assert not ranked[0].is_clear_pick

    def test_recommendation_imports(self):
        from betting_intel.recommendations import (
            RecommendationEngine,
            BetType,
            BetSuggestion,
            MoneylineBet,
            SpreadBet,
            TotalBet,
            TeamTotalBet,
            QuarterBet,
            HalfTotalBet,
            PlayerPropBet,
            ParlaySuggestion,
            BetRanker,
            ClearPick,
            Confidence,
        )
        assert RecommendationEngine is not None

    def test_bet_suggestion_action_property(self):
        from betting_intel.recommendations import MoneylineBet, Confidence

        bet = MoneylineBet(
            game_id="test", game_date="2026-05-28",
            matchup="A @ B", team="Home",
            win_probability=0.60,
            confidence=Confidence.HIGH,
            reasoning="Good matchup",
        )

        action = bet.action
        assert "on" in action or "$" in action
        assert isinstance(action, str)
        assert len(action) > 5
