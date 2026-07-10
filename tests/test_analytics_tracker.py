"""
Unit tests for ResultsTracker — bet resolution, P&L computation, trailing window alerts.
All tests mock the database layer so they run fast with zero external dependencies.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from betting_intel.analytics.tracker import (
    ResultsTracker,
    ResolvedBet,
    StrategyPerformance,
    TRAILING_WINDOW_DAYS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_tracker(tmp_path: Path) -> ResultsTracker:
    """ResultsTracker that writes predictions to a temp dir (no real disk I/O)."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    return ResultsTracker(predictions_dir=str(pred_dir))


@pytest.fixture
def sample_entry_over() -> dict:
    """A total OVER bet prediction entry."""
    return {
        "game_date": "2026-01-15",
        "matchup": "Celtics @ Lakers",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "bet_type": "total",
        "bet_side": "Total OVER 224.5",
        "market_line": 224.5,
        "model_line": 230.0,
        "edge_pct": 0.025,
        "stake_dollars": 100.0,
        "kelly_fraction": 0.05,
        "edge_confidence": "medium",
        "model_name": "test_ensemble",
        "league": "NBA",
    }


@pytest.fixture
def sample_entry_under() -> dict:
    """A total UNDER bet prediction entry."""
    return {
        "game_date": "2026-01-15",
        "matchup": "Celtics @ Lakers",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "bet_type": "total",
        "bet_side": "Total UNDER 224.5",
        "market_line": 224.5,
        "model_line": 215.0,
        "edge_pct": 0.025,
        "stake_dollars": 100.0,
        "kelly_fraction": 0.05,
        "edge_confidence": "medium",
        "model_name": "test_ensemble",
        "league": "NBA",
    }


@pytest.fixture
def sample_entry_moneyline_home() -> dict:
    """A home-team moneyline bet prediction entry."""
    return {
        "game_date": "2026-01-15",
        "matchup": "Celtics @ Lakers",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "bet_type": "moneyline",
        "bet_side": "ML Lakers",
        "market_line": -150,
        "model_line": 0.62,
        "edge_pct": 0.04,
        "stake_dollars": 200.0,
        "kelly_fraction": 0.08,
        "edge_confidence": "high",
        "model_name": "test_mlp",
        "league": "NBA",
    }


@pytest.fixture
def results_map() -> dict[str, dict]:
    """A map of matchup keys to actual game results."""
    return {
        "Celtics @ Lakers^2026-01-15": {"home_score": 115.0, "away_score": 108.0},
        "Heat @ Bucks^2026-01-15": {"home_score": 120.0, "away_score": 100.0},
        "Warriors @ Suns^2026-01-14": {"home_score": 112.0, "away_score": 105.0},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BET RESOLUTION — WIN / LOSS / PUSH
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveBet:
    """Unit tests for the _resolve_bet internal method."""

    def test_total_over_win(
        self, temp_tracker: ResultsTracker, sample_entry_over: dict
    ):
        """Total OVER 224.5, actual total = 230 → OVER wins (230 > 224.5)."""
        result = temp_tracker._resolve_bet(
            sample_entry_over, home_score=120.0, away_score=110.0
        )
        assert result["verdict"] == "WIN"
        assert result["profit"] == round(100.0 * 0.909, 2)

    def test_total_over_loss(
        self, temp_tracker: ResultsTracker, sample_entry_over: dict
    ):
        """Total OVER 224.5, actual total = 220 → LOSS."""
        result = temp_tracker._resolve_bet(
            sample_entry_over, home_score=110.0, away_score=110.0
        )
        assert result["verdict"] == "LOSS"
        assert result["profit"] == -100.0

    def test_total_over_push(
        self, temp_tracker: ResultsTracker, sample_entry_over: dict
    ):
        """Total OVER 224.5, actual total = 224.5 exactly → PUSH."""
        result = temp_tracker._resolve_bet(
            sample_entry_over, home_score=112.25, away_score=112.25
        )
        assert result["verdict"] == "PUSH"
        assert result["profit"] == 0.0

    def test_total_under_win(
        self, temp_tracker: ResultsTracker, sample_entry_under: dict
    ):
        """Total UNDER 224.5, actual total = 210 → WIN."""
        result = temp_tracker._resolve_bet(
            sample_entry_under, home_score=105.0, away_score=105.0
        )
        assert result["verdict"] == "WIN"
        assert result["profit"] == round(100.0 * 0.909, 2)

    def test_total_under_loss(
        self, temp_tracker: ResultsTracker, sample_entry_under: dict
    ):
        """Total UNDER 224.5, actual total = 230 → LOSS."""
        result = temp_tracker._resolve_bet(
            sample_entry_under, home_score=120.0, away_score=110.0
        )
        assert result["verdict"] == "LOSS"
        assert result["profit"] == -100.0

    def test_total_under_push(
        self, temp_tracker: ResultsTracker, sample_entry_under: dict
    ):
        """Total UNDER 224.5, actual total = 224.5 exactly → PUSH."""
        result = temp_tracker._resolve_bet(
            sample_entry_under, home_score=112.25, away_score=112.25
        )
        assert result["verdict"] == "PUSH"
        assert result["profit"] == 0.0

    def test_moneyline_home_win(
        self, temp_tracker: ResultsTracker, sample_entry_moneyline_home: dict
    ):
        """ML Lakers, Lakers (home) win → WIN."""
        result = temp_tracker._resolve_bet(
            sample_entry_moneyline_home, home_score=115.0, away_score=108.0
        )
        assert result["verdict"] == "WIN"
        assert result["profit"] == round(200.0 * 0.909, 2)

    def test_moneyline_home_loss(
        self, temp_tracker: ResultsTracker, sample_entry_moneyline_home: dict
    ):
        """ML Lakers, Lakers (home) lose → LOSS."""
        result = temp_tracker._resolve_bet(
            sample_entry_moneyline_home, home_score=100.0, away_score=115.0
        )
        assert result["verdict"] == "LOSS"
        assert result["profit"] == -200.0

    def test_moneyline_away_team_name_in_bet_side(self, temp_tracker: ResultsTracker):
        """ML bet on away team by name."""
        entry = {
            "game_date": "2026-01-15",
            "matchup": "Celtics @ Lakers",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "bet_type": "moneyline",
            "bet_side": "Celtics ML",
            "market_line": 130,
            "stake_dollars": 100.0,
        }
        # Celtics (away) win
        result = temp_tracker._resolve_bet(entry, home_score=105.0, away_score=115.0)
        assert result["verdict"] == "WIN"

        # Celtics (away) lose
        result = temp_tracker._resolve_bet(entry, home_score=115.0, away_score=105.0)
        assert result["verdict"] == "LOSS"

    def test_no_stake_no_profit(
        self, temp_tracker: ResultsTracker, sample_entry_over: dict
    ):
        """Bet with $0 stake → $0 profit regardless of outcome."""
        entry = {**sample_entry_over, "stake_dollars": 0.0}
        result = temp_tracker._resolve_bet(entry, home_score=120.0, away_score=110.0)
        assert result["verdict"] == "WIN"
        assert result["profit"] == 0.0

    def test_market_line_zero_defaults_to_push(self, temp_tracker: ResultsTracker):
        """Bet with market_line = 0 should not resolve as total (no comparison possible)."""
        entry = {
            "bet_type": "total",
            "bet_side": "OVER 0.0",
            "market_line": 0,
            "stake_dollars": 100.0,
        }
        result = temp_tracker._resolve_bet(entry, home_score=100, away_score=90)
        # market_line=0 means the condition bet_type in ("total", "both") and market_line > 0 is False
        # So it falls through to elif (moneyline path), which also doesn't match
        # So verdict stays PUSH
        assert result["verdict"] == "PUSH"

    def test_both_type_over_side(self, temp_tracker: ResultsTracker):
        """'both' bet type with OVER side → resolves as total."""
        entry = {
            "bet_type": "both",
            "bet_side": "Total OVER 220.5",
            "market_line": 220.5,
            "stake_dollars": 100.0,
        }
        # total = 230 > 220.5 → WIN
        result = temp_tracker._resolve_bet(entry, home_score=120, away_score=110)
        assert result["verdict"] == "WIN"


# ═══════════════════════════════════════════════════════════════════════════
#  MATCH ENTRY TO RESULT
# ═══════════════════════════════════════════════════════════════════════════


class TestMatchEntryToResult:
    """Unit tests for _match_entry_to_result."""

    def test_direct_matchup_key(
        self, temp_tracker: ResultsTracker, sample_entry_over: dict, results_map: dict
    ):
        """Match by standard 'Away @ Home^{date}' key."""
        result = temp_tracker._match_entry_to_result(sample_entry_over, results_map)
        assert result is not None
        assert result["home_score"] == 115.0
        assert result["away_score"] == 108.0
        # total=223 < market=224.5, OVER → LOSS

    def test_no_match_returns_none(
        self, temp_tracker: ResultsTracker, sample_entry_over: dict
    ):
        """Entry with date that has no results → None."""
        entry = {**sample_entry_over, "game_date": "2099-12-31", "matchup": "No @ Game"}
        result = temp_tracker._match_entry_to_result(entry, {})
        assert result is None

    def test_fuzzy_team_name_match(
        self, temp_tracker: ResultsTracker, results_map: dict
    ):
        """Fuzzy match when exact key not found but team names and date are present."""
        entry = {
            "game_date": "2026-01-15",
            "matchup": "BOS @ LAL",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "bet_type": "total",
            "bet_side": "Total OVER 224.5",
            "market_line": 224.5,
            "stake_dollars": 100.0,
        }
        result = temp_tracker._match_entry_to_result(entry, results_map)
        assert result is not None
        assert result["home_score"] == 115.0

    def test_match_by_team_name_key(
        self, temp_tracker: ResultsTracker, results_map: dict
    ):
        """Match by {away} @ {home}^{date} key when matchup format differs."""
        entry = {
            "game_date": "2026-01-14",
            "matchup": "GS @ PHX",
            "home_team": "Suns",
            "away_team": "Warriors",
            "bet_type": "total",
            "bet_side": "Total OVER 220.0",
            "market_line": 220.0,
            "stake_dollars": 100.0,
        }
        result = temp_tracker._match_entry_to_result(entry, results_map)
        assert result is not None
        # Warriors @ Suns^2026-01-14 -> home_score=112, away_score=105
        assert result["home_score"] == 112.0


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY PERFORMANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeStrategyPerformance:
    """Unit tests for _compute_strategy_performance."""

    def test_all_wins(self, temp_tracker: ResultsTracker):
        """All bets WIN → 100% win rate, positive ROI."""
        bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
            ),
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-11",
            ),
        ]
        perf = temp_tracker._compute_strategy_performance(
            bets, "model1", "NBA", "total"
        )
        assert perf.n_bets == 2
        assert perf.wins == 2
        assert perf.losses == 0
        assert perf.win_rate == 1.0
        assert perf.roi == pytest.approx(180.0 / 200.0)

    def test_all_losses(self, temp_tracker: ResultsTracker):
        """All bets LOSS → 0% win rate, negative ROI."""
        bets = [
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-11",
            ),
        ]
        perf = temp_tracker._compute_strategy_performance(
            bets, "model1", "NBA", "total"
        )
        assert perf.wins == 0
        assert perf.losses == 2
        assert perf.win_rate == 0.0
        assert perf.roi == -1.0

    def test_mixed_results(self, temp_tracker: ResultsTracker):
        """Mixed WIN/LOSS → 50% win rate, ROI should reflect net."""
        bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-11",
            ),
            ResolvedBet(
                result="PUSH",
                profit_dollars=0.0,
                stake_dollars=100.0,
                game_date="2026-01-12",
            ),
        ]
        perf = temp_tracker._compute_strategy_performance(
            bets, "model1", "NBA", "total"
        )
        assert perf.n_bets == 3
        assert perf.wins == 1
        assert perf.losses == 1
        assert perf.pushes == 1
        assert perf.win_rate == 0.5
        assert perf.roi == pytest.approx(-10.0 / 300.0)  # -10 / 300

    def test_strategy_name_format(self, temp_tracker: ResultsTracker):
        """Strategy name is '{model}/{league}/{bet_type}'."""
        bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
            )
        ]
        perf = temp_tracker._compute_strategy_performance(
            bets, "ridge", "WNBA", "spread"
        )
        assert perf.strategy_name == "ridge/WNBA/spread"
        assert perf.model == "ridge"
        assert perf.league == "WNBA"
        assert perf.bet_type == "spread"

    def test_sharpe_ratio_computed(self, temp_tracker: ResultsTracker):
        """Sharpe-like ratio is computed from profit distribution."""
        bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-11",
            ),
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-12",
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-13",
            ),
        ]
        perf = temp_tracker._compute_strategy_performance(bets, "e", "NBA", "t")
        # Sharpe = (avg_profit / std_profit) * sqrt(n)
        profits = [90.0, -100.0, 90.0, -100.0]
        avg = sum(profits) / 4
        var = sum((p - avg) ** 2 for p in profits) / 4
        std = math.sqrt(var)
        expected_sharpe = (avg / std) * math.sqrt(4) if std > 0 else 0.0
        assert perf.sharpe == pytest.approx(expected_sharpe, abs=0.01)

    def test_no_decision_bets_zero_win_rate(self, temp_tracker: ResultsTracker):
        """Only pushes → win_rate = 0.0 (no decisions)."""
        bets = [
            ResolvedBet(
                result="PUSH",
                profit_dollars=0.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
            )
        ]
        perf = temp_tracker._compute_strategy_performance(bets, "m", "NBA", "t")
        assert perf.win_rate == 0.0
        assert perf.roi == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  TRAILING WINDOW ALERT THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertThreshold:
    """Tests for the trailing 30-day ROI alert threshold logic."""

    def test_below_threshold_triggers_alert(self, temp_tracker: ResultsTracker):
        """Strategy with -10% trailing ROI and 5+ bets → is_alerted=True."""
        today = datetime.now().strftime("%Y-%m-%d")
        bets = []
        for i in range(5):
            bets.append(
                ResolvedBet(
                    result="LOSS",
                    profit_dollars=-100.0,
                    stake_dollars=100.0,
                    game_date=today,
                )
            )
        perf = temp_tracker._compute_strategy_performance(bets, "m", "NBA", "t")
        assert perf.is_alerted is True
        assert perf.roi == -1.0
        # ALERT_ROI_THRESHOLD = -0.05, trailing ROI = -1.0 < -0.05 → alerted

    def test_above_threshold_no_alert(self, temp_tracker: ResultsTracker):
        """Strategy with +5% trailing ROI and 5+ bets → is_alerted=False."""
        today = datetime.now().strftime("%Y-%m-%d")
        bets = []
        for i in range(5):
            bets.append(
                ResolvedBet(
                    result="WIN",
                    profit_dollars=90.0,
                    stake_dollars=100.0,
                    game_date=today,
                )
            )
        perf = temp_tracker._compute_strategy_performance(bets, "m", "NBA", "t")
        assert perf.is_alerted is False
        # ROI = 450/500 = 0.9 >> -0.05 → not alerted

    def test_not_enough_bets_no_alert_even_if_bad(self, temp_tracker: ResultsTracker):
        """Strategy with only 2 bets at -10% → is_alerted=False (< MIN_BETS_FOR_ALERT)."""
        today = datetime.now().strftime("%Y-%m-%d")
        bets = [
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
        ]
        perf = temp_tracker._compute_strategy_performance(bets, "m", "NBA", "t")
        assert perf.n_bets == 2
        assert perf.roi == -1.0
        # -100% < -5%, but only 2 bets < MIN_BETS_FOR_ALERT (5)
        assert perf.is_alerted is False

    def test_trailing_window_only_recent_bets(self, temp_tracker: ResultsTracker):
        """Trailing ROI only considers bets within TRAILING_WINDOW_DAYS."""
        long_ago = (
            datetime.now() - timedelta(days=TRAILING_WINDOW_DAYS + 10)
        ).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        bets = [
            # Old bets (outside window — ignored for trailing)
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date=long_ago,
            ),
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date=long_ago,
            ),
            # Recent bets (inside window — counted)
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date=today,
            ),
        ]
        perf = temp_tracker._compute_strategy_performance(bets, "m", "NBA", "t")
        # Total: 7 bets, 2 wins, 5 losses, total_stake=700, total_profit=-320
        # Trailing: 5 bets, 5 losses, trailing_stake=500, trailing_profit=-500 → trailing_roi=-1.0
        assert perf.n_bets == 7
        assert perf.roi == pytest.approx(-320.0 / 700.0)
        assert len(perf.trailing_profits) == 5
        assert perf.is_alerted is True  # 5 losses, trailing_roi=-1.0 < -0.05

    def test_check_alerts_returns_alerted_strategies(
        self, temp_tracker: ResultsTracker
    ):
        """check_alerts() returns only strategies below threshold with enough bets."""
        today = datetime.now().strftime("%Y-%m-%d")
        # Add 5 losing bets directly to resolved state
        for _ in range(5):
            temp_tracker._resolved_bets.append(
                ResolvedBet(
                    result="LOSS",
                    profit_dollars=-100.0,
                    stake_dollars=100.0,
                    game_date=today,
                    model_name="bad_model",
                    league="NBA",
                    bet_type="total",
                )
            )
        report = temp_tracker.generate_report()
        alerts = temp_tracker.check_alerts(report)
        assert len(alerts) == 1
        assert alerts[0].strategy_name == "bad_model/NBA/total"

    def test_check_alerts_no_alerts_when_profitable(self, temp_tracker: ResultsTracker):
        """check_alerts() returns empty list when all strategies are above threshold."""
        today = datetime.now().strftime("%Y-%m-%d")
        for _ in range(5):
            temp_tracker._resolved_bets.append(
                ResolvedBet(
                    result="WIN",
                    profit_dollars=90.0,
                    stake_dollars=100.0,
                    game_date=today,
                    model_name="good_model",
                    league="NBA",
                    bet_type="total",
                )
            )
        report = temp_tracker.generate_report()
        alerts = temp_tracker.check_alerts(report)
        assert len(alerts) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION & EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestReportGeneration:
    """Tests for generate_report() and edge cases."""

    def test_empty_report(self, temp_tracker: ResultsTracker):
        """No resolved bets → report with zeros."""
        report = temp_tracker.generate_report()
        assert report.total_bets == 0
        assert report.total_stake == 0.0
        assert report.total_profit == 0.0
        assert report.overall_roi == 0.0
        assert report.overall_win_rate == 0.0
        assert report.n_resolved == 0
        assert report.strategies == []

    def test_report_with_multiple_strategies(self, temp_tracker: ResultsTracker):
        """Multiple strategies in the same tracker → separate breakdowns."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
                model_name="m1",
                league="NBA",
                bet_type="total",
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
                model_name="m1",
                league="NBA",
                bet_type="total",
            ),
            ResolvedBet(
                result="WIN",
                profit_dollars=45.0,
                stake_dollars=50.0,
                game_date="2026-01-10",
                model_name="m2",
                league="WNBA",
                bet_type="spread",
            ),
        ]
        report = temp_tracker.generate_report()
        assert report.total_bets == 3
        assert len(report.strategies) == 2
        # Sorted by ROI descending
        assert (
            report.strategies[0].strategy_name == "m2/WNBA/spread"
        )  # ROI = 45/50 = 0.9
        assert (
            report.strategies[1].strategy_name == "m1/NBA/total"
        )  # ROI = -10/200 = -0.05

    def test_report_daily_pnl(self, temp_tracker: ResultsTracker):
        """Daily P&L groups bets by game_date."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
                model_name="m1",
                league="NBA",
                bet_type="total",
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
                model_name="m1",
                league="NBA",
                bet_type="total",
            ),
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-11",
                model_name="m1",
                league="NBA",
                bet_type="total",
            ),
        ]
        report = temp_tracker.generate_report()
        assert len(report.daily_pnl) == 2
        day1 = [d for d in report.daily_pnl if d["date"] == "2026-01-10"][0]
        day2 = [d for d in report.daily_pnl if d["date"] == "2026-01-11"][0]
        assert day1["profit"] == -10.0  # 90 + (-100)
        assert day1["n_bets"] == 2
        assert day2["profit"] == 90.0
        assert day2["n_bets"] == 1

    def test_model_and_league_comparison(self, temp_tracker: ResultsTracker):
        """Model and league comparison dicts are populated."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
                model_name="ridge",
                league="NBA",
                bet_type="total",
            ),
            ResolvedBet(
                result="WIN",
                profit_dollars=45.0,
                stake_dollars=50.0,
                game_date="2026-01-10",
                model_name="xgboost",
                league="NBA",
                bet_type="total",
            ),
        ]
        report = temp_tracker.generate_report()
        assert "ridge" in report.model_comparison
        assert "xgboost" in report.model_comparison
        assert "NBA" in report.league_comparison
        assert report.model_comparison["ridge"]["roi"] == pytest.approx(0.9)

    def test_save_report_creates_file(self, temp_tracker: ResultsTracker):
        """save_report() writes JSON to disk."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-10",
                model_name="m1",
                league="NBA",
                bet_type="total",
            ),
        ]
        report = temp_tracker.generate_report()
        path = temp_tracker.save_report(report)
        saved = Path(path)
        assert saved.exists()
        data = json.loads(saved.read_text())
        assert data["total_bets"] == 1
        assert data["overall_roi"] == 0.9


# ═══════════════════════════════════════════════════════════════════════════
#  INTEGRATION-SMOKE: RESOLVE FROM PREDICTIONS FILE
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveFromDisk:
    """Test resolve_all with synthetic prediction files (mocking the DB fetch)."""

    def test_resolve_all_with_forward_test_json(
        self, temp_tracker: ResultsTracker, tmp_path: Path
    ):
        """resolve_all loads predictions from forward_test_results.json, matches results."""
        # Write a forward_test_results.json with one unresolved bet
        ft_file = tmp_path / "data" / "forward_test_results.json"
        ft_file.parent.mkdir(parents=True)
        ft_data = {
            "all_bets": [
                {
                    "game_date": "2026-01-15",
                    "matchup": "Celtics @ Lakers",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "bet_type": "total",
                    "bet_side": "Total OVER 224.5",
                    "market_line": 224.5,
                    "edge_pct": 0.025,
                    "stake_dollars": 100.0,
                    "is_clear_pick": True,
                }
            ]
        }
        with open(ft_file, "w") as f:
            json.dump(ft_data, f)

        # Patch FORWARD_TEST_JSON path and _fetch_results to return synthetic results
        with patch("betting_intel.analytics.tracker.FORWARD_TEST_JSON", ft_file):
            with patch.object(
                temp_tracker,
                "_fetch_results",
                return_value={
                    "Celtics @ Lakers^2026-01-15": {
                        "home_score": 120.0,
                        "away_score": 110.0,
                    },
                },
            ):
                n = temp_tracker.resolve_all()
                assert n == 1  # 1 newly resolved

                report = temp_tracker.generate_report()
                assert report.total_bets == 1
                assert report.total_profit == pytest.approx(
                    90.9, abs=0.01
                )  # 100 * 0.909
                assert report.overall_roi == pytest.approx(0.909, abs=0.001)

    def test_resolve_all_empty_predictions_dir(self, temp_tracker: ResultsTracker):
        """resolve_all with no prediction files → 0 resolved."""
        # Override FORWARD_TEST_JSON to non-existent path
        with patch(
            "betting_intel.analytics.tracker.FORWARD_TEST_JSON",
            Path("/nonexistent/path.json"),
        ):
            n = temp_tracker.resolve_all()
            assert n == 0

    def test_get_dashboard_data_smoke(self, temp_tracker: ResultsTracker):
        """get_dashboard_data() returns correctly shaped dict even with empty state."""
        with patch(
            "betting_intel.analytics.tracker.FORWARD_TEST_JSON",
            Path("/nonexistent/path.json"),
        ):
            with patch.object(temp_tracker, "_fetch_results", return_value={}):
                data = temp_tracker.get_dashboard_data()
        assert "generated_at" in data
        assert "overall" in data
        assert data["overall"]["total_bets"] == 0
        assert data["overall"]["total_profit"] == 0.0
        assert "strategies" in data
        assert "alerted_strategies" in data
        assert "daily_pnl" in data
        assert "model_comparison" in data
        assert "league_comparison" in data
        assert "recent_bets" in data


# ═══════════════════════════════════════════════════════════════════════════
#  _ENTRY_TO_RESOLVED_BET — FIELD MAPPING
# ═══════════════════════════════════════════════════════════════════════════


class TestEntryToResolvedBet:
    """Unit tests for _entry_to_resolved_bet field mapping."""

    def test_all_fields_mapped_correctly(self, temp_tracker: ResultsTracker):
        """Every field in the input dict is mapped to the correct ResolvedBet field."""
        entry = {
            "game_date": "2026-01-15",
            "matchup": "Celtics @ Lakers",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "league": "NBA",
            "bet_type": "total",
            "bet_side": "Total OVER 224.5",
            "market_line": 224.5,
            "model_line": 230.0,
            "edge_pct": 0.035,
            "stake_dollars": 200.0,
            "kelly_fraction": 0.08,
            "edge_confidence": "high",
            "model_name": "xgboost_xval",
            "predicted_total": 228.5,
            "home_win_prob": 0.65,
            "actual_home_score": 120.0,
            "actual_away_score": 110.0,
            "actual_total": 230.0,
            "actual_home_win": 1,
            "actual_result": "WIN",
            "actual_profit": 181.8,
            "run_id": "run_20260114_060000",
            "is_clear_pick": True,
            "resolved_at": "2026-01-16T06:00:00",
        }
        bet = temp_tracker._entry_to_resolved_bet(entry)
        assert "Celtics @ Lakers_2026-01-15" in bet.prediction_id
        assert bet.game_date == "2026-01-15"
        assert bet.matchup == "Celtics @ Lakers"
        assert bet.home_team == "Lakers"
        assert bet.away_team == "Celtics"
        assert bet.league == "NBA"
        assert bet.bet_type == "total"
        assert bet.bet_side == "Total OVER 224.5"
        assert bet.market_line == 224.5
        assert bet.predicted_value == 230.0
        assert bet.edge_pct == 0.035
        assert bet.stake_dollars == 200.0
        assert bet.kelly_fraction == 0.08
        assert bet.confidence == "high"
        assert bet.model_name == "xgboost_xval"
        assert bet.predicted_total == 228.5
        assert bet.home_win_prob == 0.65
        assert bet.actual_home_score == 120.0
        assert bet.actual_away_score == 110.0
        assert bet.actual_total == 230.0
        assert bet.actual_home_win is True
        assert bet.result == "WIN"
        assert bet.profit_dollars == 181.8
        assert bet.roi == pytest.approx(181.8 / 200.0)
        assert bet.run_id == "run_20260114_060000"
        assert bet.is_clear_pick is True
        assert bet.resolved_at == "2026-01-16T06:00:00"

    def test_missing_fields_default_correctly(self, temp_tracker: ResultsTracker):
        """Fields not present in the entry default to sensible empty/zero values."""
        entry = {"game_date": "2026-01-15"}
        bet = temp_tracker._entry_to_resolved_bet(entry)
        assert bet.game_date == "2026-01-15"
        assert bet.matchup == ""
        assert bet.home_team == ""
        assert bet.away_team == ""
        assert bet.league == "NBA"
        assert bet.bet_type == ""
        assert bet.bet_side == ""
        assert bet.market_line == 0.0
        assert bet.predicted_value == 0.0
        assert bet.edge_pct == 0.0
        assert bet.stake_dollars == 0.0
        assert bet.kelly_fraction == 0.0
        assert bet.confidence == "low"
        assert bet.model_name == "ensemble"
        assert bet.predicted_total is None
        assert bet.home_win_prob is None
        assert bet.actual_home_score is None
        assert bet.actual_away_score is None
        assert bet.actual_total is None
        assert bet.actual_home_win is None
        assert bet.result is None
        assert bet.profit_dollars == 0.0
        assert bet.roi == 0.0
        assert bet.run_id == ""
        assert bet.is_clear_pick is False
        assert bet.resolved_at == ""

    def test_zero_stake_roi_is_zero(self, temp_tracker: ResultsTracker):
        """stake=0 → roi=0 regardless of profit (prevents division by zero)."""
        entry = {"stake_dollars": 0.0, "actual_profit": 50.0}
        bet = temp_tracker._entry_to_resolved_bet(entry)
        assert bet.stake_dollars == 0.0
        assert bet.profit_dollars == 50.0
        assert bet.roi == 0.0

    def test_actual_home_win_false_from_zero(self, temp_tracker: ResultsTracker):
        """actual_home_win=0 → actual_home_win is False (not None)."""
        entry = {"actual_home_win": 0}
        bet = temp_tracker._entry_to_resolved_bet(entry)
        assert bet.actual_home_win is False

    def test_actual_home_win_none_if_missing(self, temp_tracker: ResultsTracker):
        """actual_home_win not in entry → None."""
        bet = temp_tracker._entry_to_resolved_bet({})
        assert bet.actual_home_win is None

    def test_result_none_if_empty_string(self, temp_tracker: ResultsTracker):
        """Empty string actual_result → None (not "")."""
        entry = {"actual_result": ""}
        bet = temp_tracker._entry_to_resolved_bet(entry)
        assert bet.result is None

    def test_result_preserved_if_valid(self, temp_tracker: ResultsTracker):
        """Valid result string is preserved."""
        entry = {"actual_result": "PUSH"}
        bet = temp_tracker._entry_to_resolved_bet(entry)
        assert bet.result == "PUSH"


# ═══════════════════════════════════════════════════════════════════════════
#  _LOG_ALERT — WRITES TO DISK
# ═══════════════════════════════════════════════════════════════════════════


class TestLogAlert:
    """Tests for _log_alert writing alerts to JSONL disk storage."""

    def test_log_alert_writes_to_disk(
        self, temp_tracker: ResultsTracker, tmp_path: Path
    ):
        """_log_alert writes a JSONL entry to the alerts log file."""
        strategy = StrategyPerformance(
            strategy_name="test_model/NBA/total",
            model="test_model",
            league="NBA",
            bet_type="total",
            n_bets=10,
            wins=2,
            losses=8,
            roi=-0.25,
            total_profit=-500.0,
            trailing_profits=[-100.0] * 10,
            is_alerted=True,
        )
        alert_file = tmp_path / "analytics_alerts.jsonl"
        with patch("betting_intel.analytics.tracker.ALERTS_LOG", alert_file):
            temp_tracker._log_alert(strategy)

        assert alert_file.exists()
        lines = alert_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["strategy_name"] == "test_model/NBA/total"
        assert entry["alert_type"] == "underperformance"
        assert entry["roi"] == -0.25
        assert entry["n_bets"] == 10
        assert entry["wins"] == 2
        assert entry["losses"] == 8
        assert entry["total_profit"] == -500.0
        assert entry["threshold"] == -0.05
        assert "timestamp" in entry

    def test_log_alert_appends_multiple_alerts(
        self, temp_tracker: ResultsTracker, tmp_path: Path
    ):
        """Multiple _log_alert calls append to the same file."""
        alert_file = tmp_path / "analytics_alerts.jsonl"
        with patch("betting_intel.analytics.tracker.ALERTS_LOG", alert_file):
            temp_tracker._log_alert(
                StrategyPerformance(
                    strategy_name="s1",
                    n_bets=5,
                    losses=5,
                    roi=-1.0,
                    total_profit=-500.0,
                )
            )
            temp_tracker._log_alert(
                StrategyPerformance(
                    strategy_name="s2",
                    n_bets=6,
                    losses=6,
                    roi=-0.8,
                    total_profit=-480.0,
                )
            )

        lines = alert_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["strategy_name"] == "s1"
        assert json.loads(lines[1])["strategy_name"] == "s2"

    def test_log_alert_handles_write_error_gracefully(
        self, temp_tracker: ResultsTracker, caplog
    ):
        """Write failure does not raise — just logs a warning."""
        with patch(
            "betting_intel.analytics.tracker.ALERTS_LOG",
            Path("/nonexistent_dir/alert.jsonl"),
        ):
            temp_tracker._log_alert(
                StrategyPerformance(
                    strategy_name="s1",
                    n_bets=5,
                    losses=5,
                    roi=-1.0,
                    total_profit=-500.0,
                )
            )
        # Should not raise — gracefully handled
        assert True


# ═══════════════════════════════════════════════════════════════════════════
#  RESOLVE_ALL — PRE-RESOLVED ENTRIES
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveAllPreResolved:
    """Tests for resolve_all behavior with already-resolved prediction entries."""

    def test_all_entries_already_resolved(self, temp_tracker: ResultsTracker):
        """All predictions have actual_result → no fetch, 0 newly resolved, bets loaded."""
        with patch.object(
            temp_tracker,
            "_raw_predictions",
            [
                {
                    "game_date": "2026-01-15",
                    "matchup": "Celtics @ Lakers",
                    "bet_type": "total",
                    "actual_result": "WIN",
                    "actual_profit": 90.9,
                    "stake_dollars": 100.0,
                    "model_name": "m1",
                    "league": "NBA",
                },
                {
                    "game_date": "2026-01-15",
                    "matchup": "Heat @ Bucks",
                    "bet_type": "total",
                    "actual_result": "LOSS",
                    "actual_profit": -100.0,
                    "stake_dollars": 100.0,
                    "model_name": "m1",
                    "league": "NBA",
                },
            ],
        ):
            with patch.object(temp_tracker, "_load_predictions", return_value=None):
                with patch.object(temp_tracker, "_fetch_results") as mock_fetch:
                    n = temp_tracker.resolve_all()

        assert n == 0  # nothing new resolved
        mock_fetch.assert_not_called()  # no fetch needed
        assert len(temp_tracker._resolved_bets) == 2
        assert temp_tracker._resolved_bets[0].result == "WIN"
        assert temp_tracker._resolved_bets[1].result == "LOSS"

    def test_mixed_resolved_and_unresolved(self, temp_tracker: ResultsTracker):
        """Some pre-resolved, some pending — only pending are fetched and matched."""
        with patch.object(
            temp_tracker,
            "_raw_predictions",
            [
                {
                    "game_date": "2026-01-15",
                    "matchup": "Celtics @ Lakers",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "bet_type": "total",
                    "bet_side": "Total OVER 224.5",
                    "market_line": 224.5,
                    "stake_dollars": 100.0,
                    "actual_result": "WIN",
                    "actual_profit": 90.9,
                },  # pre-resolved
                {
                    "game_date": "2026-01-15",
                    "matchup": "Heat @ Bucks",
                    "home_team": "Bucks",
                    "away_team": "Heat",
                    "bet_type": "total",
                    "bet_side": "Total OVER 220.0",
                    "market_line": 220.0,
                    "stake_dollars": 100.0,
                },  # pending
            ],
        ):
            with patch.object(temp_tracker, "_load_predictions", return_value=None):
                with patch.object(
                    temp_tracker,
                    "_fetch_results",
                    return_value={
                        "Heat @ Bucks^2026-01-15": {
                            "home_score": 120.0,
                            "away_score": 100.0,
                        },
                    },
                ):
                    n = temp_tracker.resolve_all()

        assert n == 1  # 1 newly resolved
        assert len(temp_tracker._resolved_bets) == 2
        # First entry already resolved
        assert temp_tracker._resolved_bets[0].result == "WIN"
        # Second entry newly resolved: total=220, market=220.0 -> PUSH
        assert temp_tracker._resolved_bets[1].result == "PUSH"

    def test_resolve_all_saves_to_forward_test_json(
        self, temp_tracker: ResultsTracker, tmp_path: Path
    ):
        """Resolved predictions are written back to forward_test_results.json."""
        ft_file = tmp_path / "forward_test_results.json"
        ft_data = {
            "all_bets": [
                {
                    "game_date": "2026-01-15",
                    "matchup": "Celtics @ Lakers",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                    "bet_type": "total",
                    "bet_side": "Total OVER 224.5",
                    "market_line": 224.5,
                    "stake_dollars": 100.0,
                    "is_clear_pick": True,
                }
            ]
        }
        with open(ft_file, "w") as f:
            json.dump(ft_data, f)

        with patch("betting_intel.analytics.tracker.FORWARD_TEST_JSON", ft_file):
            with patch.object(
                temp_tracker,
                "_fetch_results",
                return_value={
                    "Celtics @ Lakers^2026-01-15": {
                        "home_score": 120.0,
                        "away_score": 110.0,
                    },
                },
            ):
                n = temp_tracker.resolve_all()

        assert n == 1
        # Verify the file was updated with actual results
        saved = json.loads(ft_file.read_text())
        assert saved["all_bets"][0]["actual_home_score"] == 120.0
        assert saved["all_bets"][0]["actual_away_score"] == 110.0
        assert saved["all_bets"][0]["actual_result"] == "WIN"
        assert saved["all_bets"][0]["actual_profit"] == pytest.approx(90.9, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
#  JSONL FILE LOADING
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadPredictionsFromJsonl:
    """Tests for _load_predictions reading from JSONL files."""

    def test_load_from_single_jsonl(self, temp_tracker: ResultsTracker):
        """A single .jsonl file with 3 predictions loads all entries."""
        jsonl = temp_tracker.predictions_dir / "predictions_20260115.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(
                    json.dumps(
                        {
                            "game_date": f"2026-01-1{i}",
                            "matchup": "TeamA @ TeamB",
                            "bet_type": "total",
                            "bet_side": "OVER 220.0",
                            "stake_dollars": 100.0,
                        }
                    )
                    + "\n"
                )

        with patch(
            "betting_intel.analytics.tracker.FORWARD_TEST_JSON", Path("/nonexistent")
        ):
            temp_tracker._load_predictions()

        assert len(temp_tracker._raw_predictions) == 3

    def test_load_from_multiple_jsonl_files(self, temp_tracker: ResultsTracker):
        """Multiple .jsonl files are all loaded and deduplicated by game ID."""
        for fname in ["batch1.jsonl", "batch2.jsonl"]:
            fpath = temp_tracker.predictions_dir / fname
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "game_date": "2026-01-15",
                            "matchup": "Celtics @ Lakers",
                            "bet_type": "total",
                            "stake_dollars": 100.0,
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "game_date": "2026-01-16",
                            "matchup": "Heat @ Bucks"
                            if fname == "batch1.jsonl"
                            else "Warriors @ Suns",
                            "bet_type": "total",
                            "stake_dollars": 50.0,
                        }
                    )
                    + "\n"
                )

        with patch(
            "betting_intel.analytics.tracker.FORWARD_TEST_JSON", Path("/nonexistent")
        ):
            temp_tracker._load_predictions()

        # 3 unique games: Celtics@Lakers (batch1+batch2 deduped) + Heat@Bucks (batch1) + Warriors@Suns (batch2)
        assert len(temp_tracker._raw_predictions) == 3

    def test_empty_jsonl_file_skipped_gracefully(self, temp_tracker: ResultsTracker):
        """Empty .jsonl files do not cause errors."""
        empty = temp_tracker.predictions_dir / "empty.jsonl"
        empty.touch()

        with patch(
            "betting_intel.analytics.tracker.FORWARD_TEST_JSON", Path("/nonexistent")
        ):
            temp_tracker._load_predictions()

        assert len(temp_tracker._raw_predictions) == 0

    def test_invalid_json_line_aborts_file(self, temp_tracker: ResultsTracker):
        """Invalid JSON line causes file read to abort (current behavior)."""
        bad = temp_tracker.predictions_dir / "bad.jsonl"
        with open(bad, "w", encoding="utf-8") as f:
            f.write('{"game_date": "2026-01-15", "matchup": "Good"}\n')
            f.write("not valid json\n")
            f.write('{"game_date": "2026-01-16", "matchup": "Also Good"}\n')

        with patch(
            "betting_intel.analytics.tracker.FORWARD_TEST_JSON", Path("/nonexistent")
        ):
            temp_tracker._load_predictions()

        # The try/except wraps the entire file, so a bad line aborts the read
        assert len(temp_tracker._raw_predictions) == 1
        assert temp_tracker._raw_predictions[0]["matchup"] == "Good"

    def test_jsonl_and_forward_test_dedup_same_game(
        self, temp_tracker: ResultsTracker, tmp_path: Path
    ):
        """Same game in both JSONL and forward_test_results.json — deduped, first source wins."""
        # Write a jsonl file
        jsonl = temp_tracker.predictions_dir / "daily.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "game_date": "2026-01-15",
                        "matchup": "Celtics @ Lakers",
                        "bet_type": "total",
                        "stake_dollars": 100.0,
                    }
                )
                + "\n"
            )

        # Write a forward_test_results.json with the same game
        ft_file = tmp_path / "forward_test_results.json"
        with open(ft_file, "w") as f:
            json.dump(
                {
                    "all_bets": [
                        {
                            "game_date": "2026-01-15",
                            "matchup": "Celtics @ Lakers",
                            "bet_type": "total",
                            "stake_dollars": 200.0,
                            "is_clear_pick": True,
                        }
                    ]
                },
                f,
            )

        with patch("betting_intel.analytics.tracker.FORWARD_TEST_JSON", ft_file):
            temp_tracker._load_predictions()

        # Only 1 entry (deduped), and the jsonl source wins (listed first in _load_predictions)
        assert len(temp_tracker._raw_predictions) == 1
        # JSONL stake=100.0 should be preserved (first source wins)
        assert temp_tracker._raw_predictions[0]["stake_dollars"] == 100.0

    def test_only_forward_test_json_no_jsonl(
        self, temp_tracker: ResultsTracker, tmp_path: Path
    ):
        """No JSONL files, only forward_test_results.json — loads from JSON."""
        ft_file = tmp_path / "forward_test_results.json"
        with open(ft_file, "w") as f:
            json.dump(
                {
                    "all_bets": [
                        {
                            "game_date": "2026-01-15",
                            "matchup": "Game1",
                            "bet_type": "moneyline",
                            "stake_dollars": 100.0,
                            "is_clear_pick": False,
                        },
                        {
                            "game_date": "2026-01-16",
                            "matchup": "Game2",
                            "bet_type": "spread",
                            "stake_dollars": 50.0,
                            "is_clear_pick": True,
                        },
                    ]
                },
                f,
            )

        with patch("betting_intel.analytics.tracker.FORWARD_TEST_JSON", ft_file):
            temp_tracker._load_predictions()

        assert len(temp_tracker._raw_predictions) == 2
        # Check fields were correctly mapped
        assert temp_tracker._raw_predictions[0]["model_name"] == "forward_test_ensemble"
        assert temp_tracker._raw_predictions[0]["source"] == "forward_test_results.json"
        assert temp_tracker._raw_predictions[1]["is_clear_pick"] is True


# ═══════════════════════════════════════════════════════════════════════════
#  CLV COMPUTATION — compute_clv
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeCLV:
    """Tests for compute_clv() — Closing Line Value computation.

    CLV = (opening_market_prob + edge_pct) - closing_market_prob
    Positive = our predicted line beat the closing market.
    """

    def test_positive_clv_beats_market(self, temp_tracker: ResultsTracker):
        """We predicted 3% edge and closing line only moved 1% → CLV = +0.02."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=0.03,  # We predicted 3% edge
            )
        ]
        # Mock store: opening=0.50, closing=0.52 (market moved 2% toward our prediction but we predicted 3%)
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.return_value = (0.50, 0.52)

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        bet = temp_tracker._resolved_bets[0]
        assert bet.closing_implied_prob == 0.52
        assert bet.predicted_implied_prob == 0.53  # 0.50 + 0.03
        assert bet.clv == pytest.approx(0.01, abs=0.001)  # 0.53 - 0.52 = +0.01

    def test_negative_clv_market_moved_against(self, temp_tracker: ResultsTracker):
        """We predicted 2% edge but closing line moved 5% against us → CLV = -0.03."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=0.02,
            )
        ]
        # Opening=0.50, closing=0.55 (market moved to favor our team by 5%,
        # but we only predicted 2% → the market moved MORE than we predicted)
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.return_value = (0.50, 0.55)

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        bet = temp_tracker._resolved_bets[0]
        assert bet.closing_implied_prob == 0.55
        assert bet.predicted_implied_prob == 0.52  # 0.50 + 0.02
        assert bet.clv == pytest.approx(-0.03, abs=0.001)  # 0.52 - 0.55 = -0.03

    def test_zero_edge_clv_is_market_movement(self, temp_tracker: ResultsTracker):
        """Edge=0, market moved 2% → CLV = -0.02 (we just tracked the market, didn't beat it)."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=0.0,
            )
        ]
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.return_value = (0.48, 0.50)

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        bet = temp_tracker._resolved_bets[0]
        assert bet.clv == pytest.approx(-0.02, abs=0.001)  # 0.48 - 0.50 = -0.02

    def test_negative_edge_still_proper_clv(self, temp_tracker: ResultsTracker):
        """Negative edge (we predicted against the market) with closing line movement."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=-0.02,  # We bet against the home team
            )
        ]
        # Opening=0.55, closing=0.50 (market moved 5% against home team)
        # Our prediction = 0.55 + (-0.02) = 0.53 → CLV = 0.53 - 0.50 = +0.03 (we correctly called the fade)
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.return_value = (0.55, 0.50)

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        bet = temp_tracker._resolved_bets[0]
        assert bet.clv == pytest.approx(0.03, abs=0.001)

    def test_no_market_data_clv_none(self, temp_tracker: ResultsTracker):
        """No odds data available → CLV stays None."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=0.03,
            )
        ]
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.return_value = (None, None)

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        bet = temp_tracker._resolved_bets[0]
        assert bet.clv is None
        assert bet.closing_implied_prob is None
        assert bet.predicted_implied_prob is None

    def test_no_resolved_bets_is_noop(self, temp_tracker: ResultsTracker):
        """No resolved bets → compute_clv does nothing."""
        # No bets added
        mock_store = MagicMock()
        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()
        mock_store.get_closing_vs_opening_prob.assert_not_called()

    def test_missing_team_fields_skipped(self, temp_tracker: ResultsTracker):
        """Bets without home/away team or game_date are skipped gracefully."""
        temp_tracker._resolved_bets = [
            ResolvedBet(home_team="", away_team="", game_date="", edge_pct=0.03),
            ResolvedBet(
                home_team="Celtics", away_team="", game_date="2026-01-15", edge_pct=0.03
            ),
        ]
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.return_value = (0.50, 0.52)

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        # Both bets should be skipped (empty home/away or missing team)
        assert temp_tracker._resolved_bets[0].clv is None
        assert temp_tracker._resolved_bets[1].clv is None
        mock_store.get_closing_vs_opening_prob.assert_not_called()

    def test_multiple_bets_all_computed(self, temp_tracker: ResultsTracker):
        """Multiple bets: each gets CLV computed independently."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=0.03,
            ),
            ResolvedBet(
                home_team="Heat",
                away_team="Knicks",
                game_date="2026-01-15",
                edge_pct=0.01,
            ),
        ]

        mock_store = MagicMock()

        def side_effect(home_team, away_team, game_date):
            if home_team == "Celtics":
                return (0.50, 0.52)
            elif home_team == "Heat":
                return (0.48, 0.49)
            return (None, None)

        mock_store.get_closing_vs_opening_prob.side_effect = side_effect

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()

        # Celtics: 0.50 + 0.03 = 0.53, clv = 0.53 - 0.52 = +0.01
        assert temp_tracker._resolved_bets[0].clv == pytest.approx(0.01, abs=0.001)
        # Heat: 0.48 + 0.01 = 0.49, clv = 0.49 - 0.49 = 0.0
        assert temp_tracker._resolved_bets[1].clv == pytest.approx(0.0, abs=0.001)

    def test_exception_in_store_handled_gracefully(self, temp_tracker: ResultsTracker):
        """Store raises exception → bet is skipped (no crash)."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                home_team="Celtics",
                away_team="Lakers",
                game_date="2026-01-15",
                edge_pct=0.03,
            ),
        ]
        mock_store = MagicMock()
        mock_store.get_closing_vs_opening_prob.side_effect = ValueError("DB error")

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            temp_tracker.compute_clv()  # Should not raise

        assert temp_tracker._resolved_bets[0].clv is None

    def test_clv_in_report_with_data(self, temp_tracker: ResultsTracker):
        """generate_report() populates CLV metrics when compute_clv succeeds."""
        temp_tracker._resolved_bets = [
            ResolvedBet(
                result="WIN",
                profit_dollars=90.0,
                stake_dollars=100.0,
                game_date="2026-01-15",
                model_name="m1",
                league="NBA",
                bet_type="total",
                home_team="Celtics",
                away_team="Lakers",
                edge_pct=0.03,
            ),
            ResolvedBet(
                result="LOSS",
                profit_dollars=-100.0,
                stake_dollars=100.0,
                game_date="2026-01-15",
                model_name="m1",
                league="NBA",
                bet_type="total",
                home_team="Heat",
                away_team="Knicks",
                edge_pct=0.01,
            ),
        ]

        mock_store = MagicMock()

        def side_effect(home_team, away_team, game_date):
            if home_team == "Celtics":
                return (0.50, 0.52)
            return (0.48, 0.49)

        mock_store.get_closing_vs_opening_prob.side_effect = side_effect

        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            report = temp_tracker.generate_report()

        # CLV metrics should be populated
        assert report.avg_clv is not None
        # Celtics: 0.01, Heat: 0.0 → avg = 0.005
        assert report.avg_clv == pytest.approx(0.005, abs=0.001)
        assert report.clv_wins == 1  # Celtics: +0.01 > 0
        assert report.clv_losses == 0  # Heat: 0.0 is not < 0
        assert report.clv_win_rate == 1.0  # 1/1

    def test_clv_in_report_empty_bets(self, temp_tracker: ResultsTracker):
        """generate_report() with no resolved bets → CLV metrics are None."""
        mock_store = MagicMock()
        with patch(
            "betting_intel.db.market_odds_store.MarketOddsStore",
            return_value=mock_store,
        ):
            report = temp_tracker.generate_report()

        assert report.avg_clv is None
        assert report.clv_wins == 0
        assert report.clv_losses == 0
        assert report.clv_win_rate is None
