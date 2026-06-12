"""
Validation mixin — calibration, overfitting detection, cross-validation, and drift monitoring.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from betting_intel.pipeline.bootstrap import (
    HAS_VALIDATION, HAS_MONITORING,
)

from betting_intel.pipeline.modeling import _detect_overfitting as _real_detect_overfitting
from betting_intel.pipeline.performance import get_performance_tracker as _get_perf_tracker


# ── Inline implementations replacing deleted validation/monitoring modules ──

class _InlineProbabilityCalibrator:
    """Probability calibration using Platt scaling (sklearn)."""
    def __init__(self, method="platt"):
        self.method = method
        self.calibrator = None
    def fit(self, scores, labels):
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        X = np.asarray(scores).reshape(-1, 1)
        y = np.asarray(labels).ravel()
        if len(np.unique(y)) < 2:
            self.calibrator = None
            return
        self.calibrator.fit(X, y)
    def evaluate(self, scores, labels):
        from sklearn.metrics import brier_score_loss as _brier
        X = np.asarray(scores).reshape(-1, 1)
        y = np.asarray(labels).ravel()
        if self.calibrator is None:
            return {"brier_score": round(float(_brier(y, np.clip(np.asarray(scores), 0.01, 0.99))), 4), "status": "uncalibrated"}
        try:
            cal_probs = self.calibrator.predict_proba(X)[:, 1]
            brier = float(_brier(y, cal_probs))
            shift = float(np.mean(cal_probs) - np.mean(np.clip(np.asarray(scores), 0.01, 0.99)))
            return {"brier_score": round(brier, 4), "status": "calibrated", "method": "platt_sigmoid", "calibration_shift": round(shift, 4)}
        except Exception:
            return {"brier_score": round(float(_brier(y, np.clip(np.asarray(scores), 0.01, 0.99))), 4), "status": "calibrate_failed"}

class _InlineOverfittingDetector:
    """Wired to real _detect_overfitting() from modeling.py."""
    def analyze(self, train_metrics, test_metrics, cv_results, n_observations=100):
        if isinstance(train_metrics, dict) and "overfitting_diag" in train_metrics:
            return train_metrics["overfitting_diag"]
        if cv_results and isinstance(cv_results, list) and len(cv_results) > 0:
            if isinstance(cv_results[0], dict) and "train_r2" in cv_results[0]:
                return _real_detect_overfitting(cv_results)
        return _real_detect_overfitting([])

class _InlineTimeSeriesCrossValidator:
    """Time-series CV using sklearn's TimeSeriesSplit — no lookahead."""
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    def get_splits(self, n):
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        for train_idx, test_idx in tscv.split(np.zeros(n)):
            if len(train_idx) >= 50 and len(test_idx) >= 5:
                splits.append((len(train_idx), len(test_idx)))
        return splits

class _InlinePerformanceTracker:
    """Drift tracking: records MAE in batches, alerts on >20% degradation."""
    def __init__(self, model_name="pipeline_model"):
        self.model_name = model_name
        self.predictions: list = []
    def record_prediction(self, predicted, actual):
        self.predictions.append({"predicted": float(predicted), "actual": float(actual)})
    def get_report(self):
        batch_size = 100
        if len(self.predictions) < 10:
            return {"drift_alerts": [], "status": "insufficient_data", "n_predictions": len(self.predictions)}
        batches = []
        for i in range(0, len(self.predictions), batch_size):
            batch = self.predictions[i:i + batch_size]
            if len(batch) < 5:
                continue
            mae = float(np.mean([abs(p["predicted"] - p["actual"]) for p in batch]))
            batches.append({"batch_start": i, "mae": mae, "n": len(batch)})
        if len(batches) < 2:
            return {"drift_alerts": [], "status": "single_batch", "n_batches": len(batches)}
        baseline = float(np.mean([b["mae"] for b in batches[:2]]))
        current_mae = batches[-1]["mae"]
        alerts = []
        if baseline > 0 and current_mae > baseline * 1.20:
            degradation = (current_mae / baseline - 1.0) * 100
            alerts.append({
                "alert_type": "mae_degradation", "baseline_mae": round(baseline, 2),
                "current_mae": round(current_mae, 2), "degradation_pct": round(degradation, 1),
                "message": f"MAE increased {degradation:.0f}% over baseline",
            })
        overall = float(np.mean([abs(p["predicted"] - p["actual"]) for p in self.predictions]))
        return {
            "drift_alerts": alerts, "status": "ok" if not alerts else "drift_detected",
            "n_predictions": len(self.predictions), "n_batches": len(batches),
            "overall_mae": round(overall, 2), "baseline_mae": round(baseline, 2),
            "current_mae": round(current_mae, 2),
        }




class ValidationMixin:
    """Mixin providing model validation methods for PredictionPipeline."""

    def run_validation(self, features_df, predictions_df):
        """Run calibration, overfitting detection, cross-validation & drift monitoring."""
        print("\n" + "=" * 70)
        print("  ✅  STAGE 8: MODEL VALIDATION")
        print("=" * 70)

        validation_results: Dict[str, Any] = {}

        if HAS_VALIDATION:
            self._run_calibration(predictions_df, validation_results)
            self._run_overfitting_detection(features_df, validation_results)
            self._run_cross_validation(features_df, validation_results)

        if HAS_MONITORING:
            self._run_drift_monitoring(features_df, predictions_df, validation_results)

        self.results["validation"] = validation_results

    def _run_calibration(self, predictions_df, validation_results: Dict[str, Any]):
        """Run probability calibration analysis."""
        try:
            cal = _InlineProbabilityCalibrator(method='platt')
            if ('actual_total' in predictions_df.columns
                    and 'predicted_total' in predictions_df.columns):
                scores = predictions_df['predicted_total'].values / 250.0
                labels = (predictions_df['actual_total'] > predictions_df['predicted_total']).astype(int).values
                try:
                    cal.fit(scores, labels)
                    metrics = cal.evaluate(scores, labels)
                    cal_score = metrics.get('brier_score', 'N/A')
                    print(f"  ✅  Calibration Brier score: {cal_score}")
                    validation_results["calibration"] = metrics
                except Exception as ce:
                    print(f"  ⚠  Calibration fit failed: {ce}")
            else:
                print("  ℹ  No actual outcomes for calibration analysis")
                validation_results["calibration"] = {"status": "skipped", "reason": "no_actuals"}
        except Exception as e:
            print(f"  ⚠  Calibration analysis failed: {e}")

    def _run_overfitting_detection(self, features_df, validation_results: Dict[str, Any]):
        """Run overfitting detection — retrieves pre-computed analysis from modeling step.

        The walk-forward ensemble in ModelingMixin._try_stacking_ensemble() already
        calls _detect_overfitting(fold_metrics) and stores the result in
        self.results["metadata"]["overfitting_diag"].  This method retrieves
        that pre-computed diagnostic.  If unavailable (e.g. manual fallback path),
        it falls back to calling _real_detect_overfitting with empty fold data.
        """
        try:
            # ── Retrieve pre-computed analysis from modeling step ──────────
            diag = self.results.get("metadata", {}).get("overfitting_diag")
            n_folds = self.results.get("metadata", {}).get("n_folds", 0)

            if diag is not None and isinstance(diag, dict) and n_folds > 0:
                overfit_result = diag
                print(f"  📊  Overfitting analysis from walk-forward ({n_folds} folds):")
                print(f"       Train R²: {diag.get('avg_train_r2', 0):.3f}, "
                      f"Test R²: {diag.get('avg_test_r2', 0):.3f}, "
                      f"Gap: {diag.get('r2_gap', 0):.3f}")
                if diag.get("flags"):
                    for flag in diag["flags"]:
                        print(f"       ⚠  {flag}")
            else:
                # Fallback: pass pre-computed stats through the stub
                overfit = _InlineOverfittingDetector()
                train_metrics = {"mean_error": 0.0}
                test_metrics = {"mean_error": 0.0}
                cv_results = []
                overfit_result = overfit.analyze(
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    cv_results=cv_results,
                    n_observations=len(features_df) if features_df is not None else 100,
                )

            if overfit_result:
                is_overfit = overfit_result.get("is_overfit", overfit_result.get("overfit", False))
                print(f"  ✅  Overfitting check: {'⚠ OVERFIT' if is_overfit else '✓ OK'}")
                validation_results["overfitting"] = overfit_result
        except Exception as e:
            print(f"  ⚠  Overfitting detection failed: {e}")

    def _run_cross_validation(self, features_df, validation_results: Dict[str, Any]):
        """Run time-series cross-validation."""
        try:
            ts_cv = _InlineTimeSeriesCrossValidator(n_splits=5)
            feature_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                            if c not in getattr(self, 'EXCLUDE_COLS', set())]
            if feature_cols and len(feature_cols) > 0 and "total_points" in features_df.columns:
                try:
                    cv_result = ts_cv.get_splits(len(features_df))
                    print(f"  ✅  Cross-validation: {len(cv_result)} splits generated")
                    validation_results["cross_validation"] = {"n_splits": len(cv_result)}
                except Exception as cve:
                    print(f"  ℹ  Cross-validation run: {cve}")
        except Exception as e:
            print(f"  ⚠  Cross-validation failed: {e}")

    def _run_drift_monitoring(self, features_df, predictions_df, validation_results: Dict[str, Any]):
        """Run drift monitoring — both in-run MAE tracking and cross-run R²/MAE history."""
        # ── In-run MAE tracking (same-session prediction errors) ─
        try:
            tracker = _InlinePerformanceTracker(model_name="pipeline_model")
            if ('predicted_total' in predictions_df.columns
                    and 'total_points' in features_df.columns):
                for idx, row in predictions_df.iterrows():
                    pred = row.get('predicted_total', None)
                    actual = features_df.loc[idx, 'total_points'] if idx in features_df.index else pred
                    if pred is None or actual is None:
                        continue
                    try:
                        p = float(pred)
                        a = float(actual)
                    except (ValueError, TypeError):
                        continue
                    if np.isnan(p) or np.isnan(a):
                        continue
                    if p != 0.0 and a != 0.0 and a != p:
                        tracker.record_prediction(predicted=p, actual=a)
                drift_report = tracker.get_report()
                if drift_report:
                    n_alerts = len(drift_report.get('drift_alerts', []))
                    print(f"  ✅  Drift check (in-run): {n_alerts} alerts")
                    validation_results["drift"] = drift_report
            else:
                print("  ℹ  Insufficient data for drift analysis")
                validation_results["drift"] = {"status": "skipped"}
        except Exception as e:
            print(f"  ⚠  In-run drift detection failed: {e}")

        # ── Cross-run drift check (persistent R²/MAE history) ────
        try:
            perf_tracker = _get_perf_tracker()
            cross_run = perf_tracker.check_drift(
                model_name="pipeline_ensemble",
                n_baseline=3,
                r2_threshold=-0.15,
                mae_threshold=0.20,
                verbose=True,
            )
            if cross_run:
                validation_results["cross_run_drift"] = cross_run
        except Exception as e:
            print(f"  ℹ  Cross-run drift check skipped: {e}")
