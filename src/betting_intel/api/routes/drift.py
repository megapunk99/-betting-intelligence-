"""
Drift monitoring API endpoints.

Reads model version history from ModelRegistry and strategy performance from
ResultsTracker to answer: "Are my models/strategies getting worse over time?"

Provides:
  - Per-model version history (training dates, feature counts)
  - Strategy P&L trends over time
  - Performance degradation alerts
  - Summary stats

Usage:
    GET /api/drift                     — Comprehensive drift report
    GET /api/drift/summary             — Quick overview
    GET /api/drift/history             — Raw history entries

This replaced the original implementation which imported from the deleted
betting_intel.pipeline.performance module.
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query

from betting_intel.models.persistence import model_registry
from betting_intel.analytics.tracker import ResultsTracker, ALERTS_LOG

router = APIRouter(prefix="/api", tags=["Drift"])


def _get_tracker() -> ResultsTracker:
    """Get a ResultsTracker instance for querying P&L history."""
    return ResultsTracker()


@router.get("/drift")
async def get_drift_report(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    n_baseline: int = Query(
        3, description="Number of recent training runs to compare", ge=1, le=20
    ),
    window_days: int = Query(
        30, description="Performance window in days", ge=7, le=365
    ),
    verbose: bool = Query(
        False, description="Include extended daily P&L history (14 days vs 7)"
    ),
) -> dict:
    """Return a comprehensive drift report using model version history + P&L trends.

    Two dimensions of drift:
      1. Model version drift: how training metrics (features, samples, parameters)
         have changed across saved model versions.
      2. Strategy performance drift: how P&L, ROI, and win rate have changed
         over the trailing window.

    Returns:
        - status: ok / drift_detected
        - per_model: For each tracked model:
            - versions: list of saved versions with training date + feature count
            - version_count: total versions saved
            - latest_version: most recent version info
            - drift: trend analysis (improving/degrading/stable)
        - per_strategy: For each tracked strategy (from ResultsTracker):
            - n_bets, wins, losses, win_rate, total_profit, roi
            - trailing_profit, is_alerted
        - alerts: Any underperforming strategy alerts
        - summary: Aggregate stats (total models, total strategies, date range)
    """
    # ── Dimension 1: Model version drift from ModelRegistry ──────────
    all_models = model_registry.list_models()
    if model_name:
        all_models = [m for m in all_models if m["model_name"] == model_name]

    per_model: dict = {}
    for m in all_models:
        mname = m["model_name"]
        versions = m.get("versions", [])
        latest_ver = m.get("latest_version")

        # Get metadata for recent versions to track drift
        version_details = []
        for ver in versions[-n_baseline:]:
            try:
                _, metadata = model_registry.load(mname, version=ver)
                version_details.append(
                    {
                        "version": ver,
                        "created_at": metadata.get("created_at", ""),
                        "n_features": len(metadata.get("feature_cols", [])),
                        "n_metrics": len(metadata.get("metrics", {})),
                        "metrics": metadata.get("metrics", {}),
                    }
                )
            except Exception:
                version_details.append({"version": ver, "error": "could_not_load"})

        # Determine drift direction from training dates
        drift_direction = "stable"
        timestamps = [
            v.get("created_at", "") for v in version_details if v.get("created_at")
        ]
        if len(timestamps) >= 2:
            try:
                times = [datetime.fromisoformat(t) for t in timestamps]
                if times[-1] > times[0]:
                    drift_direction = "improving"  # actively being retrained
            except (ValueError, TypeError):
                pass

        per_model[mname] = {
            "version_count": m["total_versions"],
            "latest_version": latest_ver,
            "versions": version_details,
            "drift": drift_direction,
        }

    # ── Dimension 2: Strategy performance drift from ResultsTracker ──
    tracker = _get_tracker()
    tracker.resolve_all()
    report = tracker.generate_report(window_days=window_days)

    per_strategy: dict = {}
    for s in report.strategies:
        per_strategy[s.strategy_name] = {
            "n_bets": s.n_bets,
            "wins": s.wins,
            "losses": s.losses,
            "win_rate": round(s.win_rate, 4),
            "total_profit": round(s.total_profit, 2),
            "total_stake": round(s.total_stake, 2),
            "roi": round(s.roi, 4),
            "avg_edge": round(s.avg_edge, 4),
            "sharpe": s.sharpe,
            "trailing_profit": round(sum(s.trailing_profits), 2),
            "is_alerted": s.is_alerted,
            "last_bet_date": s.last_bet_date,
        }

    # Check for alerts
    alerts = tracker.check_alerts(report)
    strategy_alerts = [
        {
            "strategy_name": a.strategy_name,
            "roi": round(a.roi, 4),
            "n_bets": a.n_bets,
            "total_profit": round(a.total_profit, 2),
        }
        for a in alerts
    ]

    # ── Model comparison from report ─────────────────────────────────
    model_comparison = {}
    for mname, mdata in report.model_comparison.items():
        model_comparison[mname] = mdata

    # ── Daily P&L for trend analysis ─────────────────────────────────
    daily_pnl = report.daily_pnl

    # Simple trend: compare first half vs second half of window
    pnl_trend = "stable"
    if len(daily_pnl) >= 4:
        mid = len(daily_pnl) // 2
        first_half_avg = sum(d["profit"] for d in daily_pnl[:mid]) / max(mid, 1)
        second_half_avg = sum(d["profit"] for d in daily_pnl[mid:]) / max(
            len(daily_pnl) - mid, 1
        )
        if second_half_avg < first_half_avg * 0.5:
            pnl_trend = "degrading"
        elif second_half_avg > first_half_avg * 1.5:
            pnl_trend = "improving"

    # Determine overall status
    has_drift = len(strategy_alerts) > 0 or (
        per_model
        and not all(
            m.get("drift", "unknown") in ("improving", "stable")
            for m in per_model.values()
        )
    )

    return {
        "status": "drift_detected" if has_drift else "ok",
        "summary": {
            "n_models_tracked": len(per_model),
            "n_strategies_tracked": len(per_strategy),
            "n_total_versions": sum(m["version_count"] for m in per_model.values()),
            "total_bets_resolved": report.total_bets,
            "overall_roi": round(report.overall_roi, 4),
            "overall_win_rate": round(report.overall_win_rate, 4),
            "pnl_trend": pnl_trend,
            "n_alerts": len(strategy_alerts),
            "generated_at": report.generated_at,
        },
        "per_model": per_model,
        "per_strategy": per_strategy,
        "alerts": strategy_alerts,
        "model_comparison": model_comparison,
        "daily_pnl": daily_pnl[-14:] if verbose else daily_pnl[-7:],
        "alerts_log_path": str(ALERTS_LOG),
    }


@router.get("/drift/summary")
async def get_drift_summary() -> dict:
    """Quick overview of tracked models and strategy performance."""
    # Model summary
    all_models = model_registry.list_models()
    total_versions = sum(m["total_versions"] for m in all_models)

    # P&L summary
    tracker = _get_tracker()
    tracker.resolve_all()
    report = tracker.generate_report()

    return {
        "status": "ok",
        "n_models": len(all_models),
        "model_names": [m["model_name"] for m in all_models],
        "total_model_versions": total_versions,
        "n_strategies": len(report.strategies),
        "n_confident_strategies": len(report.confident_strategies),
        "n_alerts": len(report.alerted_strategies),
        "total_bets_resolved": report.total_bets,
        "overall_roi": round(report.overall_roi, 4),
        "overall_win_rate": round(report.overall_win_rate, 4),
        "total_profit": round(report.total_profit, 2),
        "generated_at": report.generated_at,
    }


@router.get("/drift/history")
async def get_performance_history(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    n_last: Optional[int] = Query(None, description="Only return last N entries"),
) -> dict:
    """Get raw model version history and recent bet resolution history."""
    result: dict[str, object] = {
        "n_entries": 0,
        "model_versions": [],
        "recent_bets": [],
        "history_path": "inline_data",
    }

    # Model version history
    all_models = model_registry.list_models()
    filtered = [
        m for m in all_models if model_name is None or m["model_name"] == model_name
    ]

    for m in filtered:
        mname = m["model_name"]
        versions = m.get("versions", [])
        if n_last is not None:
            versions = versions[-n_last:]
        for ver in versions:
            try:
                _, metadata = model_registry.load(mname, version=ver)
                result["model_versions"].append(
                    {
                        "model_name": mname,
                        "version": ver,
                        "created_at": metadata.get("created_at", ""),
                        "n_features": len(metadata.get("feature_cols", [])),
                        "metrics": metadata.get("metrics", {}),
                        "parameters": metadata.get("parameters", {}),
                    }
                )
            except Exception:
                result["model_versions"].append(
                    {
                        "model_name": mname,
                        "version": ver,
                        "error": "could_not_load_metadata",
                    }
                )

    # Recent resolved bets
    tracker = _get_tracker()
    tracker.resolve_all()
    report = tracker.generate_report()
    result["recent_bets"] = [
        {
            "game_date": b.game_date,
            "matchup": b.matchup,
            "bet_type": b.bet_type,
            "bet_side": b.bet_side,
            "result": b.result,
            "profit_dollars": round(b.profit_dollars, 2),
            "roi": round(b.roi, 4),
            "edge_pct": round(b.edge_pct, 4),
            "model_name": b.model_name,
            "confidence": b.confidence,
            "stake_dollars": round(b.stake_dollars, 2),
        }
        for b in report.recent_bets
    ]

    result["n_entries"] = len(result["model_versions"]) + len(result["recent_bets"])
    return result
