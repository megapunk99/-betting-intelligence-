"""
Probability calibration: Platt scaling, isotonic regression, and beta calibration
for classification models used in betting predictions.

Well-calibrated probabilities are critical for Kelly staking.
Without calibration, edge calculations are meaningless.
"""

import numpy as np
from typing import Optional, Tuple, List
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


class ProbabilityCalibrator:
    """
    Base calibrator for transforming model scores into well-calibrated probabilities.

    Usage:
        calibrator = ProbabilityCalibrator(method='platt')
        calibrator.fit(train_scores, train_labels)
        calibrated = calibrator.calibrate(test_scores)
    """

    def __init__(self, method: str = "platt"):
        """
        Args:
            method: 'platt' (LogisticRegression), 'isotonic' (IsotonicRegression),
                    'beta' (Beta calibration), or 'none' (identity)
        """
        self.method = method
        self._calibrator = None
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit the calibrator to model scores vs actual outcomes."""
        scores = np.array(scores).flatten()
        labels = np.array(labels).flatten()

        if self.method == "platt":
            # Platt scaling: fit logistic regression on model scores
            X = scores.reshape(-1, 1)
            self._calibrator = LogisticRegression(C=1.0, class_weight="balanced")
            self._calibrator.fit(X, labels)

        elif self.method == "isotonic":
            # Isotonic regression: non-parametric calibration
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(scores, labels)

        elif self.method == "beta":
            # Beta calibration: 3-parameter beta distribution transformation
            # Better for probability-like inputs (0-1 range)
            from scipy.optimize import minimize

            def beta_loss(params, s, y):
                a, b, c = params
                # Beta calibration: p' = 1 / (1 + exp(-(a * logit(s) + b)))
                eps = 1e-10
                s = np.clip(s, eps, 1 - eps)
                logit_s = np.log(s / (1 - s))
                logit_cal = a * logit_s + b
                p_cal = 1 / (1 + np.exp(-logit_cal))
                p_cal = np.clip(p_cal, eps, 1 - eps)
                # Binary cross-entropy
                return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))

            result = minimize(
                beta_loss,
                x0=[1.0, 0.0, 0.0],
                args=(scores, labels),
                method="Nelder-Mead",
            )
            self._beta_params = result.x

        else:
            # Identity calibration (no-op)
            self._calibrator = None

        self.is_fitted = True

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Transform model scores into calibrated probabilities."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet. Call .fit() first.")

        scores = np.array(scores).flatten()

        if self.method == "platt":
            X = scores.reshape(-1, 1)
            return self._calibrator.predict_proba(X)[:, 1]

        elif self.method == "isotonic":
            return self._calibrator.transform(scores)

        elif self.method == "beta":
            eps = 1e-10
            s = np.clip(scores, eps, 1 - eps)
            a, b, _ = self._beta_params
            logit_s = np.log(s / (1 - s))
            logit_cal = a * logit_s + b
            return 1 / (1 + np.exp(-logit_cal))

        else:
            return scores

    def evaluate(self, scores: np.ndarray, labels: np.ndarray) -> dict:
        """Evaluate calibration quality."""
        calibrated = self.calibrate(scores)

        return {
            "brier_score": float(brier_score_loss(labels, calibrated)),
            "log_loss": float(log_loss(labels, calibrated)),
            "mean_pred": float(np.mean(calibrated)),
            "mean_actual": float(np.mean(labels)),
            "calibration_error": float(np.abs(np.mean(calibrated) - np.mean(labels))),
        }


class PlattCalibrator(ProbabilityCalibrator):
    """Convenience wrapper for Platt scaling."""

    def __init__(self):
        super().__init__(method="platt")


class IsotonicCalibrator(ProbabilityCalibrator):
    """Convenience wrapper for isotonic regression calibration."""

    def __init__(self):
        super().__init__(method="isotonic")


class BetaCalibrator(ProbabilityCalibrator):
    """Convenience wrapper for beta calibration."""

    def __init__(self):
        super().__init__(method="beta")


def evaluate_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Evaluate calibration quality using reliability diagrams.

    Returns:
        dict with ECE (Expected Calibration Error), MCE (Max Calibration Error),
        and per-bin statistics
    """
    prob_true, prob_pred = calibration_curve(labels, scores, n_bins=n_bins, strategy="uniform")

    ece = np.mean(np.abs(prob_true - prob_pred))
    mce = np.max(np.abs(prob_true - prob_pred))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_probs_true": prob_true.tolist(),
        "bin_probs_pred": prob_pred.tolist(),
    }


def find_best_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    methods: List[str] = None,
) -> Tuple[ProbabilityCalibrator, dict]:
    """
    Try multiple calibration methods and return the best one (lowest Brier score).

    Args:
        scores: Model prediction scores
        labels: Actual binary outcomes
        methods: List of methods to try (default: ['platt', 'isotonic', 'beta', 'none'])

    Returns:
        (best_calibrator, evaluation_metrics)
    """
    if methods is None:
        methods = ["platt", "isotonic", "beta", "none"]

    best_brier = float("inf")
    best_calibrator = None
    best_metrics = {}

    for method in methods:
        try:
            calibrator = ProbabilityCalibrator(method=method)
            calibrator.fit(scores, labels)
            metrics = calibrator.evaluate(scores, labels)

            if metrics["brier_score"] < best_brier:
                best_brier = metrics["brier_score"]
                best_calibrator = calibrator
                best_metrics = {**metrics, "method": method}
        except Exception:
            continue

    return best_calibrator, best_metrics
