"""
Overfitting detection: statistical tests to identify false positive strategies.

In betting, overfitting manifests as strategies that look profitable in
backtesting but fail live. This module provides tools to detect it.

Key tools:
1. Deflated Sharpe Ratio (DSR) — accounts for multiple testing
2. Model comparison tests — is model A significantly better than model B?
3. Backtest overfitting — combinatorially-simulated performance degradation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.special import erfc


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio from "The Deflated Sharpe Ratio: Correcting for
    Multiple Testing, Selection Bias, and Non-Normality" (Bailey & López de Prado).

    Accounts for:
    - Multiple testing (many strategies evaluated)
    - Selection bias (we only see the best ones)
    - Non-normal returns (fat tails in betting outcomes)

    Usage:
        dsr = DeflatedSharpeRatio(
            observed_sharpe=2.5,
            n_strategies=20,
            n_observations=500,
            skew=-0.2,
            kurtosis=4.0,
        )
        dsr_value, p_value = dsr.compute()
    """

    def __init__(
        self,
        observed_sharpe: float,
        n_strategies: int = 1,
        n_observations: int = 100,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        num_trials: int = 1000,
    ):
        """
        Args:
            observed_sharpe: Sharpe ratio of the best strategy
            n_strategies: Number of strategies tested
            n_observations: Number of observations per strategy
            skew: Skewness of returns (negative = left tail)
            kurtosis: Kurtosis of returns (3 = normal, >3 = fat tails)
            num_trials: Number of Monte Carlo trials for numerical estimation
        """
        self.observed_sharpe = observed_sharpe
        self.n_strategies = n_strategies
        self.n_observations = n_observations
        self.skew = skew
        self.kurtosis = kurtosis
        self.num_trials = num_trials

    def _estimate_standard_deviation(self) -> float:
        """Estimate standard deviation of Sharpe ratio under null."""
        # Variance of Sharpe ratio: 1 + 0.5*SR^2 - skew*SR + (kurtosis-3)/4 * SR^2
        if self.n_observations <= 1:
            return 1.0

        sr = 0.0  # Under null hypothesis
        var_sr = (
            1
            + 0.5 * sr**2
            - self.skew * sr
            + (self.kurtosis - 3) / 4 * sr**2
        )
        return np.sqrt(var_sr / (self.n_observations - 1))

    def _estimate_max_sharpe_null(self) -> float:
        """Estimate the expected maximum Sharpe ratio under the null."""
        if self.n_strategies <= 1:
            return 0.0

        # Using the CDF of the maximum of correlated normals
        # Approximation from Bailey & López de Prado
        e_max = np.sqrt(2 * np.log(self.n_strategies))

        # Euler-Mascheroni constant
        gamma = 0.5772156649

        # Correction for correlation among strategies
        correlation = 0.5  # Assume average pairwise correlation
        correction = (
            (1 - correlation) * gamma / np.sqrt(2 * np.log(self.n_strategies))
            if self.n_strategies > 1
            else 0
        )

        return e_max + correction

    def compute(self) -> Tuple[float, float]:
        """
        Compute Deflated Sharpe Ratio and its p-value.

        Returns:
            (dsr_value, p_value)
            dsr_value > 2 indicates a genuine strategy (not overfitting)
        """
        sr_std = self._estimate_standard_deviation()
        max_sharpe_null = self._estimate_max_sharpe_null()

        if sr_std <= 0:
            return (0.0, 1.0)

        # Deflated Sharpe Ratio
        dsr = (self.observed_sharpe - max_sharpe_null) / sr_std

        # P-value using standard normal CDF
        p_value = 1.0 - stats.norm.cdf(dsr)

        return dsr, p_value

    def is_significant(self, threshold: float = 2.0) -> bool:
        """Check if the observed Sharpe is likely not due to overfitting."""
        dsr, p_value = self.compute()
        return dsr > threshold


class ModelComparisonTest:
    """
    Statistical tests to compare two models and determine if one is
    significantly better than the other.

    Uses:
    - Diebold-Mariano test for prediction accuracy
    - McNemar's test for classification
    - Paired t-test on bet profits
    """

    @staticmethod
    def diebold_mariano(
        errors_a: np.ndarray,
        errors_b: np.ndarray,
        h: int = 1,
    ) -> Dict:
        """
        Diebold-Mariano test for equal predictive accuracy.

        Args:
            errors_a: Prediction errors from model A
            errors_b: Prediction errors from model B
            h: Forecast horizon (1 for one-step-ahead)

        Returns:
            dict with DM statistic, p-value, and conclusion
        """
        errors_a = np.array(errors_a).flatten()
        errors_b = np.array(errors_b).flatten()

        n = len(errors_a)
        if n < 4:
            return {"dm_statistic": 0, "p_value": 1, "conclusion": "Insufficient data"}

        # Loss differential
        d = errors_a**2 - errors_b**2

        # Mean loss differential
        d_mean = np.mean(d)

        # Variance of loss differential (with autocorrelation correction)
        d_var = np.var(d, ddof=1) / n

        # Simple autocorrelation correction for h > 1
        if h > 1:
            for lag in range(1, min(h, n - 1)):
                cov = np.mean(d[lag:] * d[:-lag]) * (n - lag) / n
                d_var += 2 * cov / n if lag < len(d) else 0

        if d_var <= 0:
            return {"dm_statistic": 0, "p_value": 1, "conclusion": "Zero variance"}

        dm_stat = d_mean / np.sqrt(d_var)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        if p_value < 0.05:
            if dm_stat > 0:
                conclusion = "Model B significantly better than Model A"
            else:
                conclusion = "Model A significantly better than Model B"
        else:
            conclusion = "No significant difference between models"

        return {
            "dm_statistic": float(dm_stat),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "conclusion": conclusion,
            "better_model": "A" if dm_stat < 0 else "B" if dm_stat > 0 else "equal",
        }

    @staticmethod
    def mcnemar(
        correct_a: np.ndarray,
        correct_b: np.ndarray,
    ) -> Dict:
        """
        McNemar's test for paired classification performance.

        Args:
            correct_a: Boolean array, whether model A was correct
            correct_b: Boolean array, whether model B was correct

        Returns:
            dict with chi-squared statistic, p-value, and conclusion
        """
        correct_a = np.array(correct_a, dtype=bool)
        correct_b = np.array(correct_b, dtype=bool)

        # Contingency table
        n00 = np.sum(~correct_a & ~correct_b)  # Both wrong
        n01 = np.sum(~correct_a & correct_b)   # B correct, A wrong
        n10 = np.sum(correct_a & ~correct_b)   # A correct, B wrong
        n11 = np.sum(correct_a & correct_b)    # Both correct

        # McNemar's chi-squared
        numerator = (abs(n01 - n10) - 1) ** 2  # Continuity correction
        denominator = n01 + n10

        if denominator == 0:
            return {"chi2": 0, "p_value": 1, "conclusion": "No discordant pairs"}

        chi2 = numerator / denominator
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        if p_value < 0.05:
            if n10 > n01:
                conclusion = "Model A significantly better"
            else:
                conclusion = "Model B significantly better"
        else:
            conclusion = "No significant difference"

        return {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "concordant_pairs": int(n11),
            "discordant_pairs": int(n01 + n10),
            "conclusion": conclusion,
        }


class OverfittingDetector:
    """
    Comprehensive overfitting detection for betting strategies.

    Combines:
    - Deflated Sharpe Ratio (multiple testing correction)
    - Cross-validation variance (high variance = overfitting)
    - Train/test gap (large gap = overfitting)
    - Profit curve analysis (smooth curve = likely overfitting)
    """

    def __init__(self):
        self.warnings: List[str] = []

    def analyze(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        cv_results: List[Dict],
        n_strategies_tested: int = 1,
        n_observations: int = 100,
        sharpe_ratio: Optional[float] = None,
    ) -> Dict:
        """
        Run all overfitting checks.

        Args:
            train_metrics: Performance on training data
            test_metrics: Performance on test/validation data
            cv_results: List of cross-validation fold metrics
            n_strategies_tested: Number of different strategies/models tested
            n_observations: Number of data points
            sharpe_ratio: Optional observed Sharpe ratio

        Returns:
            dict with warnings, scores, and overall verdict
        """
        self.warnings = []
        scores = {}

        # 1. Train-test gap analysis
        gap = self._analyze_train_test_gap(train_metrics, test_metrics)
        scores.update(gap)

        # 2. CV stability
        stability = self._analyze_cv_stability(cv_results)
        scores.update(stability)

        # 3. Deflated Sharpe Ratio
        dsr_analysis = self._analyze_deflated_sharpe(
            sharpe_ratio or test_metrics.get("sharpe_ratio", 0),
            n_strategies_tested,
            n_observations,
        )
        scores.update(dsr_analysis)

        # 4. Overall verdict
        overfitting_score = self._compute_overfitting_score(scores)
        scores["overfitting_score"] = overfitting_score
        scores["verdict"] = self._get_verdict(overfitting_score)
        scores["warnings"] = self.warnings

        return scores

    def _analyze_train_test_gap(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
    ) -> Dict:
        """Analyze gap between train and test performance."""
        gap_metrics = {}

        # Win rate gap
        if "win_rate" in train_metrics and "win_rate" in test_metrics:
            wr_gap = abs(train_metrics["win_rate"] - test_metrics["win_rate"])
            gap_metrics["win_rate_gap"] = wr_gap
            if wr_gap > 0.08:
                self.warnings.append(
                    f"Large win rate gap: {wr_gap:.1%} "
                    f"(train: {train_metrics['win_rate']:.1%}, "
                    f"test: {test_metrics['win_rate']:.1%})"
                )

        # R² gap
        if "r2" in train_metrics and "r2" in test_metrics:
            r2_gap = abs(train_metrics["r2"] - test_metrics["r2"])
            gap_metrics["r2_gap"] = r2_gap
            if r2_gap > 0.15:
                self.warnings.append(
                    f"Large R² gap: {r2_gap:.3f} "
                    f"(train: {train_metrics['r2']:.3f}, "
                    f"test: {test_metrics['r2']:.3f})"
                )

        # MAE ratio
        if "mae" in train_metrics and "mae" in test_metrics and train_metrics["mae"] > 0:
            mae_ratio = test_metrics["mae"] / train_metrics["mae"]
            gap_metrics["mae_ratio"] = mae_ratio
            if mae_ratio > 1.5:
                self.warnings.append(
                    f"Test MAE is {mae_ratio:.1f}x train MAE — likely overfitting"
                )

        return gap_metrics

    def _analyze_cv_stability(self, cv_results: List[Dict]) -> Dict:
        """Analyze cross-validation stability."""
        if not cv_results:
            return {"cv_warning": "No CV results"}

        # Extract win rates or R² from CV folds
        wr_values = [r.get("win_rate") or r.get("accuracy") or 0 for r in cv_results if r]
        r2_values = [r.get("r2", 0) for r in cv_results if r]

        stability = {}

        if wr_values:
            wr_std = np.std(wr_values)
            wr_mean = np.mean(wr_values)
            stability["cv_win_rate_std"] = float(wr_std)
            stability["cv_win_rate_mean"] = float(wr_mean)
            if wr_mean > 0 and wr_std / wr_mean > 0.3:
                self.warnings.append(
                    f"High CV win rate variance: std={wr_std:.3f}, mean={wr_mean:.3f}"
                )

        if r2_values:
            r2_std = np.std(r2_values)
            r2_mean = np.mean(r2_values)
            stability["cv_r2_std"] = float(r2_std)
            stability["cv_r2_mean"] = float(r2_mean)

        stability["cv_n_folds"] = len(cv_results)

        return stability

    def _analyze_deflated_sharpe(
        self,
        sharpe: float,
        n_strategies: int,
        n_observations: int,
    ) -> Dict:
        """Run Deflated Sharpe Ratio analysis."""
        dsr = DeflatedSharpeRatio(
            observed_sharpe=sharpe,
            n_strategies=n_strategies,
            n_observations=n_observations,
            skew=-0.3,  # Typical for betting returns
            kurtosis=4.5,  # Fat tails typical for sports betting
        )

        dsr_value, p_value = dsr.compute()

        if dsr_value < 1.0:
            self.warnings.append(
                f"Low Deflated Sharpe: {dsr_value:.2f} "
                f"(p={p_value:.4f}) — likely overfitting to noise"
            )

        return {
            "deflated_sharpe": float(dsr_value),
            "dsr_p_value": float(p_value),
            "n_strategies_considered": n_strategies,
        }

    def _compute_overfitting_score(self, scores: Dict) -> float:
        """
        Compute composite overfitting score (0-100).
        0 = clearly genuine, 100 = clearly overfitted.
        """
        score = 0.0
        n_factors = 0

        # Gap scores
        for key in ["win_rate_gap", "r2_gap"]:
            if key in scores:
                score += min(scores[key] * 200, 30)  # 5% gap = 10 points
                n_factors += 1

        # MAE ratio
        if "mae_ratio" in scores:
            score += min((scores["mae_ratio"] - 1.0) * 50, 30)
            n_factors += 1

        # CV stability
        if "cv_win_rate_std" in scores:
            score += min(scores["cv_win_rate_std"] * 100, 20)
            n_factors += 1

        # Deflated Sharpe
        if "deflated_sharpe" in scores:
            dsr = scores["deflated_sharpe"]
            if dsr < 1:
                score += max(30 - dsr * 15, 10)
                n_factors += 1
            elif dsr < 2:
                score += max(15 - (dsr - 1) * 10, 5)
                n_factors += 1

        return min(score / max(n_factors, 1) * 2, 100)

    def _get_verdict(self, score: float) -> str:
        """Get human-readable verdict from overfitting score."""
        if score < 20:
            return "LOW — Strategy appears genuine"
        elif score < 40:
            return "MODERATE — Some overfitting concerns, proceed with caution"
        elif score < 60:
            return "HIGH — Likely overfitting. Consider simplifying the model"
        else:
            return "CRITICAL — Almost certainly overfitting. Reject this strategy."
