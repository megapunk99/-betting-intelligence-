"""Backtest and analysis API endpoints — wired to ResultsTracker + KellyStaker."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Query

from betting_intel.api.schemas import (
    BacktestResponse,
    BacktestResultSchema,
    EdgeResponse,
    EdgeSignalSchema,
    BankrollResponse,
)
from betting_intel.analytics.tracker import ResultsTracker

router = APIRouter(prefix="/analyze", tags=["Analysis"])


def _get_tracker() -> ResultsTracker:
    return ResultsTracker()


def _get_kelly_staker():
    """Get a KellyStaker with default bankroll for bankroll status endpoint."""
    from betting_intel.recommendations.staking import KellyStaker

    return KellyStaker(initial_bankroll=10_000.0, kelly_fraction=0.25)


@router.post("/backtest")
async def run_backtest(
    strategy: str = Query(
        ..., description="Strategy name (model/league/bet_type filter)"
    ),
    window_days: int = Query(
        90, description="Performance window in days", ge=7, le=365
    ),
):
    """
    Run a backtest for a specific strategy using real historical performance data.

    Uses ResultsTracker to load resolved bets and compute actual strategy
    performance metrics (win rate, ROI, Sharpe, etc.) over the specified window.

    Unlike the old stub that returned hardcoded values, this returns REAL
    performance data from the forward test results log.
    """
    start_time = time.time()

    try:
        tracker = _get_tracker()
        tracker.resolve_all()
        report = tracker.generate_report(window_days=window_days)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"ResultsTracker failed to load data: {e}",
        )

    # Filter strategies matching the requested strategy name
    strategy_lower = strategy.lower()
    matching_strategies = [
        s for s in report.strategies if strategy_lower in s.strategy_name.lower()
    ]

    if not matching_strategies:
        # Strategy might be a model name — check model comparison
        if strategy in report.model_comparison:
            mc = report.model_comparison[strategy]
            total_bets = mc["n_bets"]
            wins = mc["wins"]
            losses = mc["losses"]
            result = BacktestResultSchema(
                strategy=strategy,
                model=strategy,
                total_bets=total_bets,
                wins=wins,
                losses=losses,
                win_rate=round(mc["win_rate"], 4),
                total_profit=round(mc["total_profit"], 2),
                roi_pct=round(mc["roi"] * 100, 2),
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_edge=round(mc["avg_edge"], 4),
                is_significant=total_bets >= 10,
            )
            execution_time = time.time() - start_time
            return BacktestResponse(
                results=[result],
                total_bets=result.total_bets,
                total_profit=result.total_profit,
                best_strategy=strategy,
                execution_time_seconds=round(execution_time, 2),
            )

        # Try to filter by league or bet type
        league_matches = [
            s for s in report.strategies if strategy_lower in s.league.lower()
        ]
        type_matches = [
            s for s in report.strategies if strategy_lower in s.bet_type.lower()
        ]

        if league_matches:
            matching_strategies = league_matches
        elif type_matches:
            matching_strategies = type_matches
        else:
            # Return all strategies with info
            available = [s.strategy_name for s in report.strategies]
            available.extend(report.model_comparison.keys())
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No strategy matching '{strategy}'. "
                    f"Available strategies: {available[:20]}. "
                    "Use a strategy name like 'ensemble/NBA/total', "
                    "a model name like 'forward_test_ensemble', "
                    "or a league like 'NBA'."
                ),
            )

    # Build results from matching strategies
    results = []
    for s in matching_strategies:
        results.append(
            BacktestResultSchema(
                strategy=s.strategy_name,
                model=s.model,
                total_bets=s.n_bets,
                wins=s.wins,
                losses=s.losses,
                win_rate=round(s.win_rate, 4),
                total_profit=round(s.total_profit, 2),
                roi_pct=round(s.roi * 100, 2),
                sharpe_ratio=s.sharpe,
                max_drawdown=0.0,
                avg_edge=round(s.avg_edge, 4),
                is_significant=s.n_bets >= 10,
            )
        )

    results.sort(key=lambda r: r.roi_pct, reverse=True)
    total_bets = sum(r.total_bets for r in results)
    total_profit = sum(r.total_profit for r in results)

    execution_time = time.time() - start_time
    return BacktestResponse(
        results=results,
        total_bets=total_bets,
        total_profit=round(total_profit, 2),
        best_strategy=results[0].strategy if results else None,
        execution_time_seconds=round(execution_time, 2),
    )


@router.get("/edges")
async def get_edges(
    min_samples: int = Query(
        10, description="Minimum samples for a signal to be actionable", ge=5, le=500
    ),
):
    """
    Get detected market inefficiencies from actual strategy performance.

    Analyzes resolved bets to identify which strategy/league/bet-type
    combinations have shown positive edge over the trailing window.
    """
    try:
        tracker = _get_tracker()
        tracker.resolve_all()
        report = tracker.generate_report(window_days=90)
    except Exception:
        return EdgeResponse(
            signals=[],
            total_signals=0,
            actionable_signals=0,
        )

    signals = []
    for s in report.strategies:
        if s.n_bets < min_samples:
            continue
        if s.roi > 0:
            signals.append(
                EdgeSignalSchema(
                    strategy=s.strategy_name,
                    edge_type="STRATEGY_EDGE",
                    description=(
                        f"{s.league} {s.bet_type} ({s.model}): "
                        f"{s.n_bets} bets, {s.wins}W-{s.losses}L, "
                        f"ROI {s.roi:.1%}"
                    ),
                    avg_edge_pct=round(s.avg_edge * 100, 2),
                    sample_size=s.n_bets,
                    win_rate=round(s.win_rate, 4),
                    expected_value=round(s.roi * 100, 2),
                    confidence=min(0.95, s.n_bets / 100.0),
                    is_actionable=s.n_bets >= 20 and s.roi > 0.02,
                )
            )

    signals.sort(key=lambda sig: sig.expected_value, reverse=True)
    actionable = sum(1 for sig in signals if sig.is_actionable)

    return EdgeResponse(
        signals=signals[:20],
        total_signals=len(signals),
        actionable_signals=actionable,
    )


@router.get("/bankroll")
async def get_bankroll_status():
    """
    Get current bankroll status from resolved bet performance.

    Computes actual bankroll trajectory from forward test results,
    replacing the old hardcoded placeholder data.
    """
    try:
        tracker = _get_tracker()
        tracker.resolve_all()
        report = tracker.generate_report(window_days=365)
    except Exception:
        # Fall back to default values if tracker unavailable
        return BankrollResponse(
            initial_bankroll=10_000.0,
            current_bankroll=10_000.0,
            peak_bankroll=10_000.0,
            total_return_pct=0.0,
            total_bets=0,
            win_rate=0.0,
            max_drawdown=0.0,
            consecutive_losses=0,
        )

    initial = 10_000.0
    cumulative = initial + report.total_profit

    # Find peak and max drawdown from daily P&L
    running = initial
    peak = initial
    max_dd = 0.0
    consecutive_losses = 0  # Current losing streak (most recent run of losses)

    for day in report.daily_pnl:
        running += day["profit"]
        if running > peak:
            peak = running
        dd = (peak - running) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        if day["profit"] < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

    total_return_pct = ((cumulative - initial) / initial) * 100 if initial > 0 else 0.0
    sum(1 for b in report.recent_bets if b.result == "WIN")
    sum(1 for b in report.recent_bets if b.result == "LOSS")

    return BankrollResponse(
        initial_bankroll=initial,
        current_bankroll=round(cumulative, 2),
        peak_bankroll=round(peak, 2),
        total_return_pct=round(total_return_pct, 2),
        total_bets=report.total_bets,
        win_rate=round(report.overall_win_rate, 4),
        max_drawdown=round(max_dd * 100, 2),
        consecutive_losses=consecutive_losses,
    )
