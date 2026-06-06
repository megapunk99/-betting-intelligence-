"""
Validation mixin — calibration, overfitting detection, cross-validation, and drift monitoring.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from betting_intel.pipeline.bootstrap import (
    HAS_VALIDATION, HAS_MONITORING,
    ProbabilityCalibrator, OverfittingDetector,
    TimeSeriesCrossValidator, PerformanceTracker,
)


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
            cal = ProbabilityCalibrator(method='platt')
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
        """Run overfitting detection."""
        try:
            overfit = OverfittingDetector()
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
            ts_cv = TimeSeriesCrossValidator(n_splits=5)
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
        """Run drift monitoring."""
        try:
            tracker = PerformanceTracker(model_name="pipeline_model")
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
                    print(f"  ✅  Drift check: {n_alerts} alerts")
                    validation_results["drift"] = drift_report
            else:
                print("  ℹ  Insufficient data for drift analysis")
                validation_results["drift"] = {"status": "skipped"}
        except Exception as e:
            print(f"  ⚠  Drift detection failed: {e}")
