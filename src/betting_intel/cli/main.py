"""
CLI tool — Click-based command-line interface for the betting intelligence system.
Provides commands for running pipeline, managing models, and querying results.

Usage:
    betting-intel run-pipeline
    betting-intel backtest --strategy momentum
    betting-intel models list
    betting-intel api start
    betting-intel db init
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import click

from betting_intel.config import settings
import logging
logger = logging.getLogger(__name__)


# ── Main CLI Group ─────────────────────────────────────────────────────────
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-file", type=click.Path(), help="Path to log file")
@click.version_option(version="0.2.0", prog_name="betting-intel")
def cli(verbose: bool, log_file: str | None):
    """Betting Intelligence System - Professional basketball betting analytics."""
    level = "DEBUG" if verbose else settings.log_level
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        filename=str(log_file) if log_file else None,
    )


# ── Pipeline Commands ─────────────────────────────────────────────────────
@cli.group()
def pipeline():
    """Pipeline management commands."""
    pass


@pipeline.command("run")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def pipeline_run(output: str | None):
    """Run the full data pipeline: load data, engineer features, train models, backtest."""
    logger.info("Starting pipeline run")

    from betting_intel.main import BettingIntelligenceSystem

    system = BettingIntelligenceSystem()
    results = system.run_full_pipeline()

    summary = results.get("summary", "Pipeline completed")
    click.echo(summary)


@pipeline.command("status")
@click.option("--last", default=5, help="Show last N runs")
def pipeline_status(last: int):
    """Show pipeline execution history."""
    click.echo(f"Last {last} pipeline runs:")
    click.echo("  (Database tracking coming soon)")


# ── Model Commands ────────────────────────────────────────────────────────
@cli.group()
def models():
    """Model management commands."""
    pass


@models.command("list")
def models_list():
    """List all trained models."""
    from betting_intel.models.persistence import model_registry

    all_models = model_registry.list_models()
    if not all_models:
        click.echo("No models found. Run 'betting-intel pipeline run' first.")
        return

    click.echo(f"{'Model Name':<20} {'Versions':<10} {'Latest':<25}")
    click.echo("-" * 55)
    for m in all_models:
        click.echo(f"{m['model_name']:<20} {m['total_versions']:<10} {m['latest_version'] or 'N/A':<25}")


@models.command("info")
@click.argument("model_name")
@click.option("--version", help="Model version (default: latest)")
def models_info(model_name: str, version: str | None):
    """Show details about a trained model."""
    from betting_intel.models.persistence import model_registry

    try:
        _, metadata = model_registry.load(model_name, version)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(json.dumps(metadata, indent=2, default=str))


# ── Backtest Commands ─────────────────────────────────────────────────────
@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command("run")
@click.option("--strategy", "-s", default="all",
              help="Strategy to backtest (total_ridge, total_xgboost, spread, momentum, ensemble, all)")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def backtest_run(strategy: str, output: str | None):
    """Run backtest for one or all strategies."""
    click.echo(f"Running backtest for strategy: {strategy}")
    click.echo("(Implementation in progress)")


@backtest.command("report")
@click.option("--latest", is_flag=True, help="Show latest backtest results")
def backtest_report(latest: bool):
    """Display backtest results."""
    output_dir = settings.resolved_output_dir
    summaries = sorted(output_dir.glob("summary_*.txt"))
    if not summaries:
        click.echo("No backtest results found. Run 'betting-intel backtest run' first.")
        return

    latest_summary = summaries[-1]
    click.echo(f"Loading: {latest_summary.name}")
    click.echo(latest_summary.read_text())


# ── API Commands ──────────────────────────────────────────────────────────
@cli.group()
def api():
    """API server commands."""
    pass


@api.command("start")
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to listen on")
@click.option("--workers", default=None, type=int, help="Number of workers")
def api_start(host: str | None, port: int | None, workers: int | None):
    """Start the REST API server."""
    from betting_intel.api.app import app
    import uvicorn

    uvicorn.run(
        "betting_intel.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        workers=workers or settings.api_workers,
        log_level=settings.api_log_level,
    )


# ── Database Commands ─────────────────────────────────────────────────────
@cli.group()
def db():
    """Database management commands."""
    pass


@db.command("init")
@click.option("--direct", is_flag=True, help="Use direct create_tables() instead of Alembic migrations")
def db_init(direct: bool):
    """Initialize the database using Alembic migrations.

    Runs ``alembic upgrade head`` to bring the schema to the latest version.
    Use ``--direct`` to fall back to the old ``create_tables()`` approach.
    """
    from betting_intel.db.connection import db_manager, run_migrations

    if direct:
        click.echo("Creating database tables (direct)...")
        db_manager.create_tables()
        click.echo("Database initialized successfully.")
        return

    click.echo("Running Alembic migrations...")
    success = run_migrations()
    if success:
        click.echo("Migrations applied. Database is up to date.")
    else:
        click.echo("Migrations failed. Try 'betting-intel db init --direct' as fallback.", err=True)
        raise click.Abort()


@db.command("check")
def db_check():
    """Check database connectivity."""
    from betting_intel.db.connection import db_manager

    if db_manager.health_check():
        click.echo("Database connection: OK")
    else:
        click.echo("Database connection: FAILED", err=True)
        sys.exit(1)


# ── Web Commands ─────────────────────────────────────────────────────────
@cli.group()
def web():
    """Web app (FastAPI) commands."""
    pass


@web.command("start")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes")
def web_start(host: str, port: int, reload: bool):
    """
    Start the FastAPI web GUI for betting recommendations.

    Features:
    - Dashboard with top clear picks and stats
    - Clear Picks page with filters
    - Today's Card grouped by game
    - All Bets with sortable table
    - Player Props generator
    - JSON API endpoints
    """
    import sys as _sys
    import subprocess

    web_dir = Path(__file__).resolve().parent.parent.parent.parent / "web"
    click.echo(f" Starting FastAPI Web App at http://localhost:{port}")
    click.echo(f" Dashboard:  http://localhost:{port}/")
    click.echo(f" Clear Picks: http://localhost:{port}/clear-picks")
    click.echo(f" API:         http://localhost:{port}/api/bets")
    click.echo()

    subprocess.run(
        [
            _sys.executable, "-m", "uvicorn", "web.app:app",
            "--host", host,
            "--port", str(port),
            "--reload" if reload else "--no-reload",
            "--log-level", "info",
        ],
        cwd=settings.project_root,
    )


# ── Small League Commands (stub — small_leagues package was deleted)
@cli.group()
def small_leagues():
    """Small-league data management commands."""
    pass


@small_leagues.command("list")
def small_leagues_list():
    """List available small leagues (unavailable — package deleted)."""
    click.echo("small_leagues package was deleted during cleanup. Re-create betting_intel/data/small_leagues/ to re-enable.")


@small_leagues.command("fetch")
def small_leagues_fetch():
    """Fetch small-league game data (unavailable — package deleted)."""
    click.echo("small_leagues package was deleted during cleanup. Re-create betting_intel/data/small_leagues/ to re-enable.")


@small_leagues.command("teams")
def small_leagues_teams(league: str = ""):
    """List teams for a small league (unavailable — package deleted)."""
    click.echo("small_leagues package was deleted during cleanup. Re-create betting_intel/data/small_leagues/ to re-enable.")


@small_leagues.command("bridge")
def small_leagues_bridge(league: str = ""):
    """Bridge small-league data to NBA-pipeline format (unavailable)."""
    click.echo("small_leagues package was deleted during cleanup. Re-create betting_intel/data/small_leagues/ to re-enable.")


# ── Recommendations Commands ───────────────────────────────────────────
@cli.group()
def recommendations():
    """Generate and view betting recommendations."""
    pass


def _format_action(bet) -> str:
    """Generate a clean, actionable bet instruction."""
    stake = f"${bet.stake_dollars:.0f}"
    if bet.bet_type.value == "moneyline":
        return f"  {stake} on {bet.bet_side} (Moneyline)"
    elif bet.bet_type.value == "spread":
        return f"  {stake} on {bet.bet_side}"
    elif bet.bet_type.value == "total_points":
        return f"  {stake} on {bet.bet_side}"
    elif bet.bet_type.value == "team_total":
        return f"  {stake} on {bet.bet_side}"
    elif bet.bet_type.value.startswith("player_"):
        return f"  {stake} on {bet.bet_side}"
    elif bet.bet_type.value == "first_quarter_winner":
        return f"  {stake} on {bet.bet_side}"
    elif bet.bet_type.value == "first_half_total":
        return f"  {stake} on {bet.bet_side}"
    return f"  {stake} on {bet.bet_side}"


@recommendations.command("list")
@click.option("--league", "-l", default="all", help="Filter by league (NBA, lnb_pro_b, cebl, bnxt, or all)")
@click.option("--type", "-t", "bet_type", default="all", help="Filter by bet type (moneyline, spread, total, props, or all)")
@click.option("--min-edge", default=0.01, type=float, help="Minimum edge threshold")
@click.option("--clear-only", is_flag=True, help="Show only clear picks")
@click.option("--limit", default=20, type=int, help="Max number of bets to show")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_list(league: str, bet_type: str, min_edge: float, clear_only: bool, limit: int, json_output: bool):
    """List betting recommendations across all markets."""
    click.echo("RecommendationEngine unavailable — recommendations/engine.py was deleted during cleanup.")
    return


@recommendations.command("todays-card")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_todays_card(json_output: bool):
    """Show today's betting card with exact bets to place."""
    click.echo("RecommendationEngine unavailable — recommendations/engine.py was deleted during cleanup.")


@recommendations.command("tomorrow")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_tomorrow(json_output: bool):
    """Show tomorrow's predictions with exact bets to place — one-day-ahead."""
    click.echo("RecommendationEngine unavailable — recommendations/engine.py was deleted during cleanup.")


@recommendations.command("clear-picks")
@click.option("--min-edge", default=0.03, type=float, help="Minimum edge threshold")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_clear_picks(min_edge: float, json_output: bool):
    """Show only the clear picks — the exact bets to place RIGHT NOW."""
    click.echo("RecommendationEngine unavailable — recommendations/engine.py was deleted during cleanup.")


# ── Analytics Commands ────────────────────────────────────────────────────
@cli.group()
def analytics():
    """Performance analytics, P&L tracking, and alerts."""
    pass


@analytics.command("resolve")
def analytics_resolve():
    """Resolve pending predictions against actual game results.

    Loads all logged predictions, matches them against actual scores
    from the NBA database, and computes W/L/PUSH with P&L.
    """
    from betting_intel.analytics.tracker import ResultsTracker
    tracker = ResultsTracker()
    n = tracker.resolve_all()
    click.echo(f"Resolved {n} predictions against actual results")


@analytics.command("report")
@click.option("--save", is_flag=True, help="Save report to disk")
def analytics_report(save: bool):
    """Generate and display the performance report.

    Shows total P&L, ROI, win rate, strategy breakdown, and
    alerts for underperforming strategies.
    """
    from betting_intel.analytics.tracker import ResultsTracker
    tracker = ResultsTracker()
    tracker.resolve_all()
    report = tracker.generate_report()

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  PERFORMANCE REPORT — {report.generated_at[:19]}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Total bets:    {report.total_bets}")
    click.echo(f"  Total stake:   ${report.total_stake:.2f}")
    click.echo(f"  Total profit:  ${report.total_profit:.2f}")
    click.echo(f"  Overall ROI:   {report.overall_roi:.2%}")
    click.echo(f"  Win rate:      {report.overall_win_rate:.2%}")
    click.echo(f"  Resolved:      {report.n_resolved}")
    click.echo(f"  Unresolved:    {report.n_unresolved}")
    click.echo()

    if report.strategies:
        click.echo(f"  {'Strategy':<40} {'Bets':<6} {'Wins':<6} {'P&L':<10} {'ROI':<8}")
        click.echo(f"  {'-' * 70}")
        for s in report.strategies:
            marker = "⚠ " if s.is_alerted else "  "
            roi_str = f"{s.roi:.1%}" if s.roi >= 0 else click.style(f"{s.roi:.1%}", fg="red")
            pnl_str = f"+${s.total_profit:.0f}" if s.total_profit >= 0 else click.style(f"-${abs(s.total_profit):.0f}", fg="red")
            click.echo(f"  {marker}{s.strategy_name:<38} {s.n_bets:<6} {s.wins:<6} {pnl_str:<10} {roi_str:<8}")

    if report.alerted_strategies:
        click.echo()
        click.echo(click.style(f"  ⚠  {len(report.alerted_strategies)} STRATEGY ALERT(S) BELOW -5% ROI:", fg="red"))
        for s in report.alerted_strategies:
            click.echo(f"      {s.strategy_name}: ROI={s.roi:.1%} ({s.n_bets} bets, P&L=${s.total_profit:.0f})")

    if save:
        path = tracker.save_report(report)
        click.echo(f"\n  Report saved to: {path}")

    click.echo(f"{'=' * 60}\n")


@analytics.command("check-alerts")
def analytics_check_alerts():
    """Check all strategies for underperformance and dispatch alerts.

    Sends Slack webhook and/or email alerts for any strategy that has
    fallen below the -5% ROI threshold over the trailing 30 days.
    """
    from betting_intel.analytics.tracker import ResultsTracker
    # Alert module was deleted during cleanup — no alerts dispatched

    tracker = ResultsTracker()
    tracker.resolve_all()
    report = tracker.generate_report()

    alerts = []
    click.echo("  Alerting module removed during cleanup. Configure channels to re-enable.")


@analytics.command("summary")
def analytics_summary():
    """Generate a concise Markdown summary for CI/CD job summaries (GitHub Actions).

    Outputs GitHub-flavored Markdown with key P&L metrics and strategy breakdown.
    Pipe to $GITHUB_STEP_SUMMARY in workflows.
    """
    from betting_intel.analytics.tracker import ResultsTracker

    tracker = ResultsTracker()
    tracker.resolve_all()
    report = tracker.generate_report()

    today = datetime.now().strftime("%Y-%m-%d")
    click.echo(f"## \uD83D\uDCCA Daily P&L Summary — {today}")
    click.echo()

    # Overall metrics table
    roi_color = "green" if report.overall_roi >= 0 else "red"
    pnl_color = "green" if report.total_profit >= 0 else "red"
    click.echo("| Metric | Value |")
    click.echo("|--------|-------|")
    click.echo(f"| Total Bets | {report.total_bets} |")
    click.echo(f"| Total Stake | ${report.total_stake:,.2f} |")
    click.echo(f"| **Total P&L** | **<span style='color:{pnl_color}'>${report.total_profit:+,.2f}</span>** |")
    click.echo(f"| **Overall ROI** | **<span style='color:{roi_color}'>{report.overall_roi:+.2%}</span>** |")
    click.echo(f"| Win Rate | {report.overall_win_rate:.1%} |")
    click.echo(f"| Resolved | {report.n_resolved} |")
    click.echo(f"| Unresolved | {report.n_unresolved} |")
    click.echo()

    # Strategy breakdown
    if report.strategies:
        click.echo("### Strategy Breakdown")
        click.echo()
        click.echo("| Strategy | Bets | Wins | P&L | ROI |")
        click.echo("|----------|------|------|-----|-----|")
        for s in report.strategies:
            pnl_color = "green" if s.total_profit >= 0 else "red"
            roi_color = "green" if s.roi >= 0 else "red"
            marker = "\u26a0\ufe0f " if s.is_alerted else ""
            click.echo(f"| {marker}{s.strategy_name} | {s.n_bets} | {s.wins} | <span style='color:{pnl_color}'>${s.total_profit:+,.0f}</span> | <span style='color:{roi_color}'>{s.roi:+.1%}</span> |")
        click.echo()

    # Alerts
    if report.alerted_strategies:
        click.echo("### \u26a0\ufe0f Underperforming Strategies")
        click.echo()
        for s in report.alerted_strategies:
            click.echo(f"- **{s.strategy_name}**: ROI={s.roi:.1%}, {s.n_bets} bets, P&L=${s.total_profit:+,.0f}")
        click.echo()

    # Daily P&L (last 7 days)
    if report.daily_pnl:
        click.echo("### Last 7 Days P&L")
        click.echo()
        last_7 = report.daily_pnl[-7:] if len(report.daily_pnl) > 7 else report.daily_pnl
        for day in last_7:
            emoji = "\U0001f7e2" if day["profit"] >= 0 else "\U0001f534"
            sign = "+" if day["profit"] >= 0 else ""
            click.echo(f"{emoji} {day['date']}: {sign}${day['profit']:,.0f} ({day['n_bets']} bet{'s' if day['n_bets'] != 1 else ''})")
        click.echo()

    click.echo(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*")


@analytics.command("dashboard")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def analytics_dashboard(json_output: bool):
    """Display the full P&L dashboard — summary, strategy breakdown, recent bets."""
    from betting_intel.analytics.tracker import ResultsTracker

    tracker = ResultsTracker()
    data = tracker.get_dashboard_data()

    if json_output:
        import json as _json
        click.echo(_json.dumps(data, indent=2, default=str))
        return

    overall = data.get("overall", {})
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  P&L DASHBOARD — {data.get('generated_at', '')[:19]}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Total Bets:   {overall.get('total_bets', 0)}")
    click.echo(f"  Total Stake:  ${overall.get('total_stake', 0):.2f}")
    click.echo(f"  Total P&L:    ${overall.get('total_profit', 0):.2f}")
    click.echo(f"  Overall ROI:  {overall.get('overall_roi', 0):.2%}")
    click.echo(f"  Win Rate:     {overall.get('overall_win_rate', 0):.2%}")
    click.echo(f"  Resolved:     {overall.get('n_resolved', 0)}")
    click.echo(f"  Unresolved:   {overall.get('n_unresolved', 0)}")

    alerted = data.get("alerted_strategies", [])
    if alerted:
        click.echo()
        click.echo(click.style(f"  ⚠  {len(alerted)} ALERT(S):", fg="red"))
        for a in alerted:
            click.echo(f"      {a['strategy_name']}: ROI={a['roi']:.1%} (${a['total_profit']:.0f})")

    click.echo(f"{'=' * 60}\n")


@recommendations.command("player-props")
@click.argument("home_team")
@click.argument("away_team")
@click.option("--league", default="NBA", help="League (NBA, lnb_pro_b, cebl, bnxt)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_player_props(home_team: str, away_team: str, league: str, json_output: bool):
    """Generate player prop predictions for a specific matchup."""
    from betting_intel.recommendations.player_props import PlayerPropEngine

    engine = PlayerPropEngine()
    props = engine.predict_for_game(
        home=home_team,
        away=away_team,
        league=league,
        game_id=f"{home_team}_vs_{away_team}_props",
    )

    if json_output:
        click.echo(json.dumps([p.as_dict() for p in props], indent=2))
        return

    if not props:
        click.echo(f"No player props generated for {home_team} vs {away_team}")
        return

    click.echo(f"\n{'=' * 80}")
    click.echo(f"  PLAYER PROPS — {away_team} @ {home_team}")
    click.echo(f"{'=' * 80}")

    for team in [home_team, away_team]:
        team_props = [p for p in props if team in p.bet_side]
        if team_props:
            click.echo(f"\n  {team}:")
            click.echo(f"  {'─' * 60}")
            for p in team_props:
                click.echo(f"  ${p.stake_dollars:.0f} on {p.bet_side}")


if __name__ == "__main__":
    cli()
