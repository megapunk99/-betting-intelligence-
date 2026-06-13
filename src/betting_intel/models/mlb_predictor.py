"""
MLB Moneyline Predictor — predicts home win probability for MLB games
using historical pitching stats, park factors, and team strength metrics.

KEY INSIGHTS (market inefficiencies):
  1. The public overvalues "name brand" pitchers (deGrom, Ohtani, etc.),
     creating value on the opposing team
  2. Park factors are systematically underpriced by casual bettors who
     don't adjust for Coors Field, Petco Park, etc.
  3. Bullpen quality is undervalued — late-inning relievers have a
     disproportionate impact on win probability
  4. Small market teams are chronically undervalued by public money
  5. Run differential in recent games is a stronger predictor than
     simple win-loss record

FEATURE CATEGORIES:
  1. Historical Pitching Stats:
     - ERA, WHIP, K/9, BB/9, HR/9, innings pitched (season-to-date)
     - Pitcher matchup differentials (home ERA - away ERA)
     - Both pitchers known flag (information advantage)

  2. Park Factors:
     - Home/away park factor (3-year rolling, normalized to 1.00)
     - Park factor differential
     - Park-adjusted run scoring estimates
     - Bullpen quality rating by team

  3. Team Strength:
     - Win rate last 10 games (both home and away)
     - Runs scored and allowed last 5 games
     - Run differential (rolling 5 games)
     - Team record entering game
     - Win percentage differential

Usage:
    from betting_intel.models.mlb_predictor import MLBMoneylinePredictor

    predictor = MLBMoneylinePredictor()
    predictor.fit(X_train, y_train)
    probs = predictor.predict_proba(X_test)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

HAS_LIGHTGBM = False
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    pass


class MLBMoneylinePredictor:
    """
    MLB-specific moneyline predictor using pitcher + team features.

    Trains a gradient-boosted classifier to predict home win probability,
    then calculates edge against market-implied probability from TheOddsAPI lines.

    Parameters
    ----------
    calibrate : bool
        Apply Platt scaling for calibrated probabilities. Default: True.
    random_state : int
        Random seed. Default: 42.

    Attributes
    ----------
    model_ : object
        Trained classifier (LightGBM or logistic fallback).
    is_fitted : bool
        Whether the model has been fitted.
    feature_names_ : list of str
        Feature names used during training.
    metrics_ : dict
        Latest evaluation metrics.
    """

    def __init__(self, calibrate: bool = True, random_state: int = 42):
        self.calibrate = calibrate
        self.random_state = random_state
        self.model_: Any = None
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.metrics_: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MLBMoneylinePredictor":
        """
        Fit the MLB moneyline predictor.

        Args:
            X: (n_samples, n_features) training features.
            y: (n_samples,) binary labels (1 = home win, 0 = home loss).
            feature_names: Optional list of feature names.

        Returns:
            self
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) length mismatch")

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Build model — prefer LightGBM, fallback to logistic
        if HAS_LIGHTGBM:
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=24,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=3.0,
                class_weight="balanced",
                random_state=self.random_state,
                verbosity=-1,
            )
        else:
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )

        if self.calibrate:
            try:
                calibrated = CalibratedClassifierCV(
                    estimator=model,
                    method="sigmoid",
                    cv=3,
                )
                calibrated.fit(X, y)
                self.model_ = calibrated
            except Exception:
                model.fit(X, y)
                self.model_ = model
        else:
            model.fit(X, y)
            self.model_ = model

        self.is_fitted = True
        logger.info(f"MLB predictor fitted: {len(X)} samples, {X.shape[1]} features")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary outcomes (0/1)."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated home win probabilities."""
        if not self.is_fitted:
            raise ValueError("Not fitted yet. Call fit() first.")

        try:
            proba = self.model_.predict_proba(X)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        except Exception as e:
            logger.warning(f"MLB predict_proba failed: {e}")
            return np.full(X.shape[0], 0.5)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictor performance.

        Returns dict with brier, log_loss, accuracy, auc_roc,
        calibration_error.
        """
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        metrics: Dict[str, float] = {
            "brier": float(brier_score_loss(y, probs)),
            "accuracy": float(accuracy_score(y, preds)),
            "n_samples": len(y),
        }

        try:
            metrics["log_loss"] = float(log_loss(y, probs))
        except Exception:
            metrics["log_loss"] = 1.0

        if len(np.unique(y)) == 2:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y, probs))
            except Exception:
                metrics["auc_roc"] = 0.5
        else:
            metrics["auc_roc"] = 0.5

        # Calibration error
        try:
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, 9)
            cal_error = 0.0
            n_bins_used = 0
            for b in range(10):
                mask = bin_indices == b
                if mask.sum() > 0:
                    cal_error += abs(probs[mask].mean() - y[mask].mean()) * mask.sum()
                    n_bins_used += mask.sum()
            metrics["calibration_error"] = float(cal_error / max(n_bins_used, 1))
        except Exception:
            metrics["calibration_error"] = 1.0

        self.metrics_ = metrics
        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Walk-forward cross-validation (no lookahead bias).

        Returns dict with avg metrics and per-fold results.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oos_probs = np.full(len(y), np.nan)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                fold_model = MLBMoneylinePredictor(
                    calibrate=self.calibrate,
                    random_state=self.random_state,
                )
                fold_model.fit(X_train, y_train, feature_names=feature_names)
                fold_probs = fold_model.predict_proba(X_test)
                oos_probs[test_idx] = fold_probs

                fm = fold_model.evaluate(X_test, y_test)
                fm["fold"] = fold
                fm["n_train"] = len(X_train)
                fm["n_test"] = len(X_test)
                fold_metrics.append(fm)
            except Exception as e:
                logger.debug(f"Fold {fold} failed: {e}")
                continue

        if not fold_metrics:
            return {"avg_brier": 1.0, "avg_log_loss": 1.0, "avg_accuracy": 0.5, "n_folds": 0}

        valid_oos = oos_probs[~np.isnan(oos_probs)]
        valid_y = y[~np.isnan(oos_probs)]

        avg = {
            "avg_brier": float(np.mean([m["brier"] for m in fold_metrics])),
            "avg_log_loss": float(np.mean([m["log_loss"] for m in fold_metrics])),
            "avg_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "avg_auc_roc": float(np.mean([m.get("auc_roc", 0.5) for m in fold_metrics])),
            "n_folds": len(fold_metrics),
            "n_oos": len(valid_oos),
            "fold_metrics": fold_metrics,
        }

        if len(valid_oos) > 0:
            avg["oos_brier"] = float(brier_score_loss(valid_y, valid_oos))
            avg["oos_accuracy"] = float(accuracy_score(
                (valid_oos >= 0.5).astype(int), valid_y
            ))
            if len(np.unique(valid_y)) == 2:
                try:
                    avg["oos_auc_roc"] = float(roc_auc_score(valid_y, valid_oos))
                except Exception:
                    avg["oos_auc_roc"] = 0.5

        logger.info(
            f"MLB CV: {avg['n_folds']} folds, "
            f"Brier={avg['avg_brier']:.4f}, "
            f"AUC={avg['avg_auc_roc']:.3f}"
        )

        # Refit on full data
        self.fit(X, y, feature_names=feature_names)

        return avg

    def compute_edge(
        self,
        home_prob: float,
        home_ml_odds: Optional[float],
        away_ml_odds: Optional[float],
    ) -> Tuple[float, str, str]:
        """
        Compute edge against market-implied probability.

        Args:
            home_prob: Model's predicted home win probability (0-1).
            home_ml_odds: Home moneyline odds (American format, e.g. -150).
            away_ml_odds: Away moneyline odds (American format, e.g. +130).

        Returns:
            Tuple of (edge_pct, direction, confidence).
            edge_pct > 0 means the model favors the home team.
            direction is "home" or "away" indicating which side has edge.
            confidence is "high", "medium", or "low".
        """
        if home_ml_odds is None or away_ml_odds is None:
            return 0.0, "neutral", "low"

        # Convert American odds to implied probability
        def american_to_prob(odds: float) -> float:
            if odds > 0:
                return 100.0 / (odds + 100.0)
            return abs(odds) / (abs(odds) + 100.0)

        home_implied = american_to_prob(home_ml_odds)
        away_implied = american_to_prob(away_ml_odds)

        # Remove vig by normalizing
        total_implied = home_implied + away_implied
        if total_implied <= 0:
            return 0.0, "neutral", "low"

        home_market = home_implied / total_implied

        # Edge: positive means model thinks home is more likely than market
        edge = home_prob - home_market

        abs_edge = abs(edge)
        if abs_edge > 0.05:
            confidence = "high"
        elif abs_edge >= 0.02:
            confidence = "medium"
        else:
            confidence = "low"

        direction = "home" if edge > 0 else "away"
        return round(edge, 4), direction, confidence


def train_mlb_model(
    days_back: int = 365,
    calibrate: bool = True,
    cv: bool = True,
) -> Tuple[MLBMoneylinePredictor, Dict[str, Any], pd.DataFrame]:
    """
    Convenience function: fetch MLB data, train model, return predictor.

    Args:
        days_back: How many days of history to train on.
        calibrate: Apply Platt scaling.
        cv: Run walk-forward cross-validation.

    Returns:
        Tuple of (trained predictor, metrics dict, feature DataFrame).
    """
    from betting_intel.data.mlb_data import MLBDataSource

    source = MLBDataSource()
    df = source.build_training_dataset(days_back=days_back)

    if df.empty:
        raise ValueError("No MLB training data available")

    # Determine feature columns (exclude target and identifiers)
    exclude = {"home_win", "game_id", "date", "home_team", "away_team",
               "home_team_full", "away_team_full", "status",
               "home_pitcher_id", "away_pitcher_id",
               "home_pitcher", "away_pitcher"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    if not feature_cols:
        raise ValueError("No feature columns available for training")

    X = df[feature_cols].fillna(0).values
    y = df["home_win"].values.astype(int)

    predictor = MLBMoneylinePredictor(calibrate=calibrate)
    metrics: Dict[str, Any] = {}

    if cv:
        cv_results = predictor.cross_validate(X, y, feature_names=feature_cols)
        metrics["cv"] = cv_results
        logger.info(f"MLB CV done: AUC={cv_results.get('avg_auc_roc', '?'):.3f}")
    else:
        predictor.fit(X, y, feature_names=feature_cols)
        probs = predictor.predict_proba(X)
        train_metrics = predictor.evaluate(X, y)
        metrics["train"] = train_metrics

    return predictor, metrics, df


__all__ = [
    "MLBMoneylinePredictor",
    "train_mlb_model",
]
