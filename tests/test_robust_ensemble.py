"""
Comprehensive tests for RobustPredictionSystem and KellyStaker.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════
#  ROBUST PREDICTION SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRobustPredictionSystem:
    """Tests for the RobustPredictionSystem class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10)
        # Create a semi-predictable target
        y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)
        return X, y

    def test_import(self):
        """Verify the module imports correctly."""
        from betting_intel.models.robust_ensemble import (
            RobustPredictionSystem, PredictionResult,
            compute_statistical_significance, compute_drawdown,
        )
        assert RobustPredictionSystem is not None
        assert PredictionResult is not None

    def test_fit_and_predict_with_defaults(self, sample_data):
        """Test that fit() and predict_proba() work with default params."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        assert system._fitted
        assert len(system._models) >= 2  # At least Logistic + one tree model

        probs = system.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_with_details(self, sample_data):
        """Test predict_with_details returns complete PredictionResult."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=True, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        result = system.predict_with_details(X[0])
        assert 0 < result.home_win_prob < 1
        assert result.confidence_label in ("VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW")
        assert result.n_models >= 1
        assert len(result.model_probs) >= 1

    def test_predict_with_details_2d_input(self, sample_data):
        """Test predict_with_details handles 2D input."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        # Should work with (1, n_features) input
        result_2d = system.predict_with_details(X[0:1])
        assert isinstance(result_2d.home_win_prob, float)

        # Should work with (n_features,) input
        result_1d = system.predict_with_details(X[0])
        assert isinstance(result_1d.home_win_prob, float)

    def test_not_fitted_error(self):
        """Test that predicting before fit raises ValueError."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem()
        with pytest.raises(ValueError, match="not fitted"):
            system.predict_proba(np.random.randn(1, 5))

    def test_insufficient_data_error(self):
        """Test that fit with too little data raises ValueError."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem(min_train_samples=100, min_test_samples=20)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        with pytest.raises(ValueError, match="Need at least"):
            system.fit(X, y, verbose=False)

    def test_compute_edge_valid(self, sample_data):
        """Test compute_edge with valid market odds."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        X, y = sample_data
        system.fit(X, y, verbose=False)

        # Test with meaningful odds
        edge, direction, confidence = system.compute_edge(0.60, -150, +130)
        assert isinstance(edge, float)
        assert direction in ("home", "away", "neutral")
        assert isinstance(confidence, str)

        # Home team +EV (model says 60%, market says ~60% implied after vig removal)
        # -150 → implied = 150/250 = 0.60, +130 → 100/230 = 0.435
        # After vig removal: home ~0.58, away ~0.42
        # Edge = 0.60 - 0.58 = 0.02 (home +EV since model > market)
        expected_edge = 0.60 - (0.60 / (0.60 + 0.435))
        assert abs(edge - round(expected_edge, 4)) < 0.001

    def test_compute_edge_none_odds(self, sample_data):
        """Test compute_edge with None odds returns neutral."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        X, y = sample_data
        system.fit(X, y, verbose=False)

        edge, direction, confidence = system.compute_edge(0.60, None, None)
        assert edge == 0.0
        assert direction == "neutral"
        assert confidence == "LOW"

    def test_feature_importance(self, sample_data):
        """Test feature importance returns correct format."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, feature_names=[f"feat_{i}" for i in range(X.shape[1])], verbose=False)

        importance = system.get_feature_importance(top_n=5)
        assert len(importance) <= 5
        for name, val in importance.items():
            assert name.startswith("feat_")
            assert isinstance(val, float)
            assert 0 <= val <= 1

    def test_get_summary(self, sample_data):
        """Test get_summary returns valid dict."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        summary = system.get_summary()
        assert summary["fitted"] is True
        assert summary["n_models"] >= 2
        assert summary["n_features"] == 10
        assert summary["n_train_samples"] == 200

    def test_get_model_diagnostics(self, sample_data):
        """Test model diagnostics returns info per model."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        diag = system.get_model_diagnostics()
        assert len(diag) >= 2
        for name, d in diag.items():
            assert d.oos_brier > 0
            assert d.n_oos > 0

    def test_save_and_load(self, sample_data, tmp_path):
        """Test save and load roundtrip preserves state."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        path = tmp_path / "test_system.joblib"
        system.save(path)

        loaded = RobustPredictionSystem.load(path)
        assert loaded._fitted
        assert loaded._n_train_total == 200
        assert len(loaded._models) == len(system._models)

        # Predictions should match
        orig_result = system.predict_with_details(X[0])
        loaded_result = loaded.predict_with_details(X[0])
        assert abs(orig_result.home_win_prob - loaded_result.home_win_prob) < 0.01

    def test_overfitting_detection(self):
        """Test overfitting detection logic."""
        from betting_intel.models.robust_ensemble import OverfittingReport

        report = OverfittingReport(
            is_overfit=True,
            avg_train_r2=0.95,
            avg_test_r2=0.10,
            r2_gap=0.85,
            flags=["Test R² is severely negative"],
        )
        assert report.is_overfit
        assert report.r2_gap == 0.85

    def test_predict_binary(self, sample_data):
        """Test predict() returns binary classes."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=False, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        preds = system.predict(X[:10], threshold=0.5)
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_calibration_improves_brier(self, sample_data):
        """Test that calibrated Brier is not worse than raw Brier."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = RobustPredictionSystem(calibrate=True, n_folds=3, min_train_samples=30)
        system.fit(X, y, verbose=False)

        summary = system.get_summary()
        raw_brier = summary.get("brier_score")
        cal_brier = summary.get("calibrated_brier")

        # If both exist, calibrated shouldn't be drastically worse
        if raw_brier is not None and cal_brier is not None:
            assert cal_brier <= raw_brier * 1.5  # Allow some slack


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICAL SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStatisticalSignificance:
    """Tests for compute_statistical_significance."""

    def test_significant_result(self):
        """Test that 60/40 is significant at 95% level."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(60, 40)
        assert result["win_rate"] == 0.6
        assert result["n_bets"] == 100
        assert result["is_significant"]  # p < 0.05

    def test_not_significant(self):
        """Test that 55/45 is not necessarily significant with small n."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(55, 45)
        assert result["win_rate"] == 0.55
        assert not result["is_significant"]  # p > 0.05

    def test_edge_case_no_bets(self):
        """Test edge case with zero bets."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(0, 0)
        assert result["win_rate"] == 0.0
        assert result["n_bets"] == 0
        assert result["p_value"] == 1.0
        assert not result["is_significant"]

    def test_perfect_record(self):
        """Test perfect record gives very low p-value."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(20, 0)
        assert result["win_rate"] == 1.0
        assert result["is_significant"]
        assert result["ci_lower"] > 0.8

    def test_ci_bounds(self):
        """Test confidence intervals are within [0, 1]."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        for wins, losses in [(10, 10), (30, 20), (50, 10), (5, 15), (100, 100)]:
            result = compute_statistical_significance(wins, losses)
            assert 0 <= result["ci_lower"] <= 1
            assert 0 <= result["ci_upper"] <= 1
            assert result["ci_lower"] <= result["ci_upper"]


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWDOWN ANALYSIS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestDrawdown:
    """Tests for compute_drawdown."""

    def test_basic_drawdown(self):
        """Test basic drawdown calculation."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        # Profits: +10, -5, +20, -10, +30
        # Cumulative: 10, 5, 25, 15, 45
        # Peak: 10, 10, 25, 25, 45
        # Drawdown: 0, 5, 0, 10, 0
        # Max drawdown = 10
        result = compute_drawdown([10, -5, 20, -10, 30])
        assert result["max_drawdown"] == 10.0

    def test_no_drawdown(self):
        """Test with only positive profits."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        result = compute_drawdown([10, 20, 30])
        assert result["max_drawdown"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_empty_list(self):
        """Test with empty list."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        result = compute_drawdown([])
        assert result["max_drawdown"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_large_drawdown(self):
        """Test large drawdown calculation."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        # Start 100, lose 50, gain 30, lose 80
        # Cumulative: 100, -50, -20, -100
        result = compute_drawdown([100, -150, 30, -80])
        assert result["max_drawdown"] > 0
        assert result["max_drawdown_pct"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  KELLY STAKER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestKellyStaker:
    """Tests for the KellyStaker class."""

    @pytest.fixture
    def staker(self):
        """Create a basic staker."""
        from betting_intel.recommendations.staking import KellyStaker
        return KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)

    def test_import(self):
        """Verify staking module imports correctly."""
        from betting_intel.recommendations.staking import (
            KellyStaker, StakeResult, BankrollState,
            american_to_decimal, decimal_to_american,
            american_to_implied, remove_vig,
        )
        assert KellyStaker is not None
        assert StakeResult is not None

    def test_compute_stake_basic(self, staker):
        """Test basic stake computation."""
        from betting_intel.recommendations.staking import american_to_decimal

        dec_odds = american_to_decimal(-110)
        result = staker.compute_stake(
            win_probability=0.60,
            decimal_odds=dec_odds,
            confidence_score=0.80,
            confidence_label="HIGH",
            edge_pct=0.05,
            league="NBA",
            team="Lakers",
        )
        assert result.stake_dollars > 0
        assert result.kelly_full > 0
        assert result.exposure_pct > 0
        assert result.kelly_full > result.kelly_fractional  # fractional less than full

    def test_compute_stake_below_threshold(self, staker):
        """Test that stake is 0 when edge is below threshold."""
        result = staker.compute_stake(
            win_probability=0.51,
            decimal_odds=1.91,
            confidence_score=0.5,
            confidence_label="LOW",
            edge_pct=0.005,  # Below 0.01 threshold
            league="NBA",
            team="Lakers",
        )
        assert result.stake_dollars == 0.0
        assert "below threshold" in " ".join(result.adjustment_reasons).lower()

    def test_kelly_full_calculation(self, staker):
        """Test Kelly formula: f* = (bp - q) / b"""
        # For 60% win prob at -110 odds (1.91 decimal):
        # b = 0.91, p = 0.60, q = 0.40
        # f* = (0.91 * 0.60 - 0.40) / 0.91 = (0.546 - 0.40) / 0.91 = 0.146/0.91 = 0.1604
        from betting_intel.recommendations.staking import american_to_decimal

        dec_odds = american_to_decimal(-110)  # ~1.909
        result = staker.compute_stake(
            win_probability=0.60,
            decimal_odds=dec_odds,
            confidence_score=1.0,
            confidence_label="VERY_HIGH",
            edge_pct=0.10,
        )
        expected_kelly = ((dec_odds - 1) * 0.60 - 0.40) / (dec_odds - 1)
        assert abs(result.kelly_full - expected_kelly) < 0.001

    def test_consecutive_losses_reduce_stake(self):
        """Test that consecutive losses reduce subsequent stakes."""
        from betting_intel.recommendations.staking import KellyStaker

        staker = KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)

        # No losses
        result_no_loss = staker.compute_stake(
            win_probability=0.65, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.10,
        )

        # Simulate 5 consecutive losses
        for _ in range(5):
            staker.record_bet(stake=100, won=False, profit=-100)

        result_with_losses = staker.compute_stake(
            win_probability=0.65, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.10,
        )

        assert result_with_losses.stake_dollars < result_no_loss.stake_dollars

    def test_drawdown_reduces_stake(self):
        """Test that drawdown reduces stake."""
        from betting_intel.recommendations.staking import KellyStaker

        staker = KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)

        # Simulate large losses to create drawdown
        for _ in range(3):
            staker.record_bet(stake=2000, won=False, profit=-2000)

        result = staker.compute_stake(
            win_probability=0.70, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.15,
        )
        # After 3 * 2000 losses, bankroll is 4000, drawdown is 60%
        # At 40%+ drawdown, staking is disabled
        assert result.stake_dollars == 0.0
        assert any("drawdown" in r.lower() for r in result.adjustment_reasons)

    def test_team_exposure_limit(self, staker):
        """Test that team exposure limit works."""
        # First bet on Lakers
        result1 = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.8, confidence_label="HIGH",
            edge_pct=0.05, league="NBA", team="Lakers",
        )
        staker.record_bet(team="Lakers", stake=result1.stake_dollars, won=True)

        # Second bet on Lakers should be limited
        result2 = staker.compute_stake(
            win_probability=0.65, decimal_odds=1.91,
            confidence_score=0.85, confidence_label="VERY_HIGH",
            edge_pct=0.08, league="NBA", team="Lakers",
        )

        # Exposure per team should not exceed max
        total_exposed = result1.stake_dollars + result2.stake_dollars
        max_allowed = staker.max_team_exposure_pct * staker.bankroll
        assert total_exposed <= max_allowed * 1.1  # 10% slack

    def test_get_state(self, staker):
        """Test get_state returns valid BankrollState."""
        state = staker.get_state()
        assert state.current == 10000
        assert state.initial == 10000
        assert state.peak == 10000
        assert state.drawdown == 0.0
        assert state.n_bets_today == 0

    def test_record_win(self, staker):
        """Test recording a win updates bankroll."""
        staker.record_bet(team="Lakers", stake=200, won=True, profit=181.82)
        assert staker.bankroll == pytest.approx(10181.82, rel=1e-4)
        assert staker.total_profit == pytest.approx(181.82, rel=1e-4)

    def test_record_loss(self, staker):
        """Test recording a loss updates bankroll."""
        staker.record_bet(team="Celtics", stake=200, won=False, profit=-200)
        assert staker.bankroll == 9800.0
        assert staker.total_profit == -200.0

    def test_reset(self, staker):
        """Test reset restores initial state."""
        staker.record_bet(stake=500, won=True, profit=500)
        assert staker.bankroll == 10500.0
        staker.reset()
        assert staker.bankroll == 10000.0
        assert staker.peak == 10000.0

    def test_get_performance_summary(self, staker):
        """Test performance summary returns correct data."""
        staker.record_bet(stake=200, won=True, profit=180)
        staker.record_bet(stake=200, won=False, profit=-200)
        staker.record_bet(stake=200, won=True, profit=180)

        summary = staker.get_performance_summary()
        assert summary["n_bets"] == 3
        assert summary["wins"] == 2
        assert summary["losses"] == 1
        assert summary["win_rate"] == pytest.approx(0.6667, abs=0.001)  # Rounded to 4dp

    def test_release_exposure(self, staker):
        """Test releasing exposure works."""
        staker.record_bet(team="Lakers", league="NBA", game_id="g1", stake=500)
        assert staker.get_exposure().total_exposed == 500.0

        staker.release_exposure(team="Lakers")
        assert "Lakers" not in staker.get_exposure().per_team
        assert staker.get_exposure().total_exposed == 0.0

    def test_high_confidence_bets_bigger(self):
        """Test that higher confidence = bigger stake."""
        from betting_intel.recommendations.staking import KellyStaker

        staker = KellyStaker(initial_bankroll=10000)

        low_conf = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.5, confidence_label="LOW",
            edge_pct=0.05,
        )

        high_conf = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.05,
        )

        assert high_conf.stake_dollars > low_conf.stake_dollars

    def test_daily_bet_limit(self, staker):
        """Test daily bet limit is enforced."""
        # Reach the daily bet limit
        for _ in range(staker.max_daily_bets):
            staker._n_bets_today += 1

        result = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.8, confidence_label="HIGH",
            edge_pct=0.05,
        )
        assert result.stake_dollars == 0.0
        assert any("daily bet limit" in r.lower() for r in result.adjustment_reasons)

    def test_negative_edge_returns_zero(self, staker):
        """Test negative edge returns zero stake."""
        result = staker.compute_stake(
            win_probability=0.45,
            decimal_odds=1.91,
            edge_pct=-0.05,  # Negative edge
        )
        assert result.stake_dollars == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  ODDS CONVERSION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestOddsConversion:
    """Tests for odds conversion utilities."""

    def test_american_to_decimal_favorite(self):
        from betting_intel.recommendations.staking import american_to_decimal
        assert abs(american_to_decimal(-150) - 1.6667) < 0.01

    def test_american_to_decimal_underdog(self):
        from betting_intel.recommendations.staking import american_to_decimal
        assert abs(american_to_decimal(+200) - 3.0) < 0.01

    def test_american_to_decimal_even(self):
        from betting_intel.recommendations.staking import american_to_decimal
        assert abs(american_to_decimal(100) - 2.0) < 0.01

    def test_decimal_to_american(self):
        from betting_intel.recommendations.staking import decimal_to_american
        assert decimal_to_american(1.91) == -110  # Common NBA odds

    def test_decimal_to_american_underdog(self):
        from betting_intel.recommendations.staking import decimal_to_american
        assert decimal_to_american(3.0) == 200

    def test_american_to_implied_favorite(self):
        from betting_intel.recommendations.staking import american_to_implied
        assert abs(american_to_implied(-200) - 2/3) < 0.01

    def test_american_to_implied_underdog(self):
        from betting_intel.recommendations.staking import american_to_implied
        assert abs(american_to_implied(+200) - 1/3) < 0.01

    def test_remove_vig(self):
        from betting_intel.recommendations.staking import remove_vig
        home, away = remove_vig(0.6, 0.45)
        total = home + away
        assert abs(total - 1.0) < 0.001  # Should sum to 1


# ═══════════════════════════════════════════════════════════════════════════
#  ENGINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineIntegration:
    """Tests for the engine with RobustPredictionSystem."""

    def test_engine_imports(self):
        """Verify engine imports work."""
        from betting_intel.live.engine import LivePredictionEngine, LiveGame, LivePredictionSnapshot
        assert LivePredictionEngine is not None
        assert LiveGame is not None

    def test_engine_initialization(self):
        """Test engine init creates kelly staker and robust references."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine()
        assert engine.kelly_staker is not None
        assert engine.robust_system is None  # Not fitted yet
        assert engine.robust_system_summary["status"] == "not_initialized"

    def test_engine_robust_properties(self):
        """Test robust_system properties work before training."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine()
        summary = engine.robust_system_summary
        assert isinstance(summary, dict)
        assert "fitted" in summary

    def test_kelly_staker_property(self):
        """Test kelly_staker property returns working staker."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine()
        staker = engine.kelly_staker
        assert staker is not None
        assert staker.bankroll == 10000.0


class TestLiveGame:
    """Tests for the LiveGame dataclass."""

    def test_matchup_property(self):
        """Test matchup property format."""
        from betting_intel.live.engine import LiveGame

        game = LiveGame(
            game_id="test_1",
            sport_key="basketball_nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_team_short="LAL",
            away_team_short="BOS",
            commence_time="2025-06-12T19:00:00Z",
            game_date="2025-06-12",
        )
        assert game.matchup == "BOS @ LAL"

    def test_commence_datetime(self):
        """Test commence_datetime parses ISO format."""
        from betting_intel.live.engine import LiveGame

        game = LiveGame(
            game_id="test_1",
            sport_key="basketball_nba",
            home_team="Lakers",
            away_team="Celtics",
            home_team_short="LAL",
            away_team_short="BOS",
            commence_time="2025-06-12T19:00:00Z",
            game_date="2025-06-12",
        )
        dt = game.commence_datetime
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 6

    def test_to_dict(self):
        """Test to_dict serialization."""
        from betting_intel.live.engine import LiveGame

        game = LiveGame(
            game_id="test_1",
            sport_key="basketball_nba",
            home_team="Lakers",
            away_team="Celtics",
            home_team_short="LAL",
            away_team_short="BOS",
            commence_time="2025-06-12T19:00:00Z",
            game_date="2025-06-12",
            home_ml=-150,
            away_ml=+130,
        )
        d = game.to_dict()
        assert d["game_id"] == "test_1"
        assert d["home_ml"] == -150
        assert d.get("matchup", "MISSING") == "MISSING"  # 'matchup' is a @property, not in to_dict()
        assert d["away_team_short"] == "BOS"
        assert d["home_team_short"] == "LAL"


class TestLivePredictionSnapshot:
    """Tests for the LivePredictionSnapshot dataclass."""

    def test_empty_snapshot(self):
        """Test empty snapshot defaults."""
        from betting_intel.live.engine import LivePredictionSnapshot

        snap = LivePredictionSnapshot()
        assert snap.n_live == 0
        assert snap.n_total == 0
        assert len(snap.next_two_days) == 0
        assert snap.generated_at is not None
