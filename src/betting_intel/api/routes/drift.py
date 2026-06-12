"""
Drift monitoring API endpoints.

Reads the persistent model performance history (logs/model_performance.json)
and returns drift analysis: per-model R²/MAE trends, cross-run drift checks,
and summary statistics.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from betting_intel.pipeline.performance import (
    get_performance_tracker,
    DEFAULT_HISTORY_PATH,
)

router = APIRouter(prefix="/api", tags=["Drift"])


@router.get("/drift")
async def get_drift_report(
    model_name: Optional[str] = Query(None, description="Filter by model name (e.g. 'pipeline_ensemble', 'retrain_total')"),
    n_baseline: int = Query(3, description="Number of runs to use as drift baseline", ge=1, le=20),
    r2_threshold: float = Query(-0.15, description="Max acceptable R² drop before alerting"),
    mae_threshold: float = Query(0.20, description="Max acceptable MAE increase fraction before alerting"),
    verbose: bool = Query(False, description="Include raw history entries in response"),
) -> dict:
    """Read model performance history and return a comprehensive drift report.

    Returns:
        - summary: Total runs, models tracked, last update timestamp
        - per_model: For each tracked model:
            - status: Drift check result (ok / drift_detected / insufficient_history)
            - recent_r2, recent_mae, baseline_r2, baseline_mae, r2_change, mae_change_frac
            - alerts: Any R² or MAE degradation alerts
            - history: Raw run entries (only included when verbose=True)
    """
    tracker = get_performance_tracker()
    history = tracker.get_history()
    model_names = sorted(set(r.get("model_name", "unknown") for r in history))

    per_model = {}
    for mn in model_names:
        # If user filtered by model_name, skip others
        if model_name and mn != model_name:
            continue

        drift = tracker.check_drift(
            model_name=mn,
            n_baseline=n_baseline,
            r2_threshold=r2_threshold,
            mae_threshold=mae_threshold,
            verbose=False,
        )

        runs = tracker.get_history(model_name=mn, n_last=10)
        entry = {
            "drift": drift,
            "summary": {
                "n_runs": len(tracker.get_history(model_name=mn)),
                "last_run": runs[-1].get("timestamp", "N/A") if runs else None,
                "models_used": runs[-1].get("models_used", []) if runs else [],
            },
        }

        # Compute simple trend direction from last 3 runs with R²
        r2_vals = [r.get("test_r2") for r in runs if "test_r2" in r]
        if len(r2_vals) >= 2:
            entry["trend_r2"] = "improving" if r2_vals[-1] > r2_vals[0] else "degrading" if r2_vals[-1] < r2_vals[0] else "stable"

        mae_vals = [r.get("test_mae") for r in runs if "test_mae" in r]
        if len(mae_vals) >= 2:
            entry["trend_mae"] = "improving" if mae_vals[-1] < mae_vals[0] else "degrading" if mae_vals[-1] > mae_vals[0] else "stable"

        if verbose:
            entry["history"] = [
                {k: v for k, v in r.items() if k != "models_used"}
                for r in runs
            ]

        per_model[mn] = entry

    # Aggregate alerts across all models
    all_alerts = []
    for mn, data in per_model.items():
        for alert in data.get("drift", {}).get("alerts", []):
            all_alerts.append({"model_name": mn, **alert})

    return {
        "status": "ok" if not any(data["drift"]["drift_detected"] for data in per_model.values()) else "drift_detected",
        "n_models_tracked": len(per_model),
        "n_total_runs": len(history),
        "total_alerts": len(all_alerts),
        "alerts": all_alerts,
        "per_model": per_model,
        "history_path": str(DEFAULT_HISTORY_PATH),
    }


@router.get("/drift/summary")
async def get_drift_summary() -> dict:
    """Quick summary of all tracked performance data without full drift analysis."""
    tracker = get_performance_tracker()
    return tracker.get_summary()


@router.get("/drift/history")
async def get_performance_history(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    n_last: Optional[int] = Query(None, description="Only return last N entries"),
) -> dict:
    """Get raw performance history entries."""
    tracker = get_performance_tracker()
    history = tracker.get_history(model_name=model_name, n_last=n_last)
    return {
        "n_entries": len(history),
        "history": history,
        "history_path": str(DEFAULT_HISTORY_PATH),
    }
