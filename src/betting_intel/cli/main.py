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
import os
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
        click.echo(
            f"{m['model_name']:<20} {m['total_versions']:<10} {m['latest_version'] or 'N/A':<25}"
        )


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


@backtest.command("report")
@click.option("--latest", is_flag=True, help="Show latest backtest results")
def backtest_report(latest: bool):
    """Display backtest results."""
    output_dir = settings.resolved_output_dir
    summaries = sorted(output_dir.glob("summary_*.txt"))
    if not summaries:
        click.echo("No backtest results found.")
        return

    latest_summary = summaries[-1]
    click.echo(f"Loading: {latest_summary.name}")
    click.echo(latest_summary.read_text())


# ── API Commands ──────────────────────────────────────────────────────────
@cli.group()
def api():
    """API server commands."""
    pass


@api.command("test-key")
def api_test_key():
    """Test the TheOddsAPI key — verify it works and check quota.

    Makes a single API call to /v4/sports (free endpoint, no credit cost)
    to list available sports, then attempts a single odds call (1 credit)
    to verify the key works end-to-end.
    """
    import urllib.request
    import json

    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key or api_key in ("your-api-key-here", ""):
        click.echo(
            click.style(" No valid ODDS_API_KEY found in .env or environment", fg="red")
        )
        click.echo(
            "  Set ODDS_API_KEY in your .env file or as an environment variable."
        )
        click.echo("  Get a free key at: https://the-odds-api.com/")
        raise click.Abort()

    click.echo(f"\n  Testing key: {api_key[:8]}...{api_key[-4:]}")

    # Test 1: /v4/sports (free endpoint)
    click.echo("\n  Step 1: Testing /v4/sports endpoint (free, no credit cost)...")
    try:
        url = f"https://api.the-odds-api.com/v4/sports?apiKey={api_key}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "betting-intel-cli/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        sports = [s for s in data if s.get("active", False)]
        in_season = [s for s in sports if s.get("has_odds", False)]
        click.echo(
            f"   Success! Found {len(data)} total sports, {len(sports)} active, {len(in_season)} with odds"
        )
        for s in in_season[:5]:
            click.echo(f"    - {s['group']:25s} {s['title']}")
        if len(in_season) > 5:
            click.echo(f"    ... and {len(in_season) - 5} more")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            click.echo(
                click.style(
                    f"   HTTP 401 — INVALID API KEY: {api_key[:8]}...", fg="red"
                )
            )
            click.echo("    Check that the key is correct and not expired.")
            click.echo("    Get a new key at: https://the-odds-api.com/")
        else:
            click.echo(click.style(f"   HTTP {e.code}: {e}", fg="red"))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"   Connection failed: {e}", fg="red"))
        raise click.Abort()

    # Test 2: Single odds call for the first active sport (costs 1 credit)
    if in_season:
        sport_key = in_season[0]["key"]
        click.echo(
            f"\n  Step 2: Testing /v4/sports/{sport_key}/odds (costs 1 credit)..."
        )
        try:
            url = (
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
                f"?apiKey={api_key}"
                f"&regions=us"
                f"&markets=h2h"
                f"&oddsFormat=american"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "betting-intel-cli/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                games = json.loads(resp.read().decode("utf-8"))
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            click.echo(f"   Odds fetched! {len(games)} games available")
            click.echo(f"    Quota remaining: {remaining}")
            click.echo(f"    Quota used: {used}")
            if games:
                g = games[0]
                click.echo(
                    f"  Sample game: {g.get('away_team', '?')} @ {g.get('home_team', '?')}"
                )
        except urllib.error.HTTPError as e:
            if e.code == 401:
                click.echo(
                    click.style("   HTTP 401 — key is invalid for odds calls", fg="red")
                )
            elif e.code == 429:
                click.echo(click.style("   HTTP 429 — quota exceeded", fg="red"))
            else:
                click.echo(click.style(f"   HTTP {e.code}", fg="red"))
        except Exception as e:
            click.echo(click.style(f"   Error: {e}", fg="red"))

    # Summary
    click.echo(f"\n  {'=' * 46}")
    click.echo(f"  Key: {api_key[:8]}...{api_key[-4:]} — appears valid")
    click.echo("  This key will cost ~3 credits/sport for full odds calls")
    click.echo("  -> regions=us, markets=h2h,spreads,totals (per sport)")
    click.echo("  With 500 free credits/month and 1 sport in-season:")
    click.echo("  -> ~166 refreshes/month, or ~5 per day")
    click.echo("  General: https://the-odds-api.com/docs")
    click.echo("  Billing:  https://the-odds-api.com/manage")
    click.echo(f"  {'=' * 46}\n")


@api.command("start")
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to listen on")
@click.option("--workers", default=None, type=int, help="Number of workers")
def api_start(host: str | None, port: int | None, workers: int | None):
    """Start the REST API server."""
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
@click.option(
    "--direct",
    is_flag=True,
    help="Use direct create_tables() instead of Alembic migrations",
)
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
        click.echo(
            "Migrations failed. Try 'betting-intel db init --direct' as fallback.",
            err=True,
        )
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

    Path(__file__).resolve().parent.parent.parent.parent / "web"
    click.echo(f" Starting FastAPI Web App at http://localhost:{port}")
    click.echo(f" Dashboard:  http://localhost:{port}/")
    click.echo(f" Clear Picks: http://localhost:{port}/clear-picks")
    click.echo(f" API:         http://localhost:{port}/api/bets")
    click.echo()

    subprocess.run(
        [
            _sys.executable,
            "-m",
            "uvicorn",
            "web.app:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload" if reload else "--no-reload",
            "--log-level",
            "info",
        ],
        cwd=settings.project_root,
    )


# ── Backfill Commands ────────────────────────────────────────────────────
@cli.group()
def backfill():
    """Backfill historical market data commands."""
    pass


@backfill.command("market-odds")
@click.option(
    "--mode",
    type=click.Choice(["scores", "historical", "stats"]),
    required=True,
    help="scores: free-tier game metadata | historical: paid-tier full odds | stats: check DB",
)
@click.option(
    "--days-back",
    type=int,
    default=3,
    help="Days back for scores mode (free tier max: 3)",
)
@click.option(
    "--start-date", type=str, help="Start date (YYYY-MM-DD) for historical mode"
)
@click.option("--end-date", type=str, help="End date (YYYY-MM-DD) for historical mode")
@click.option(
    "--snapshot-interval",
    type=click.Choice(["daily", "weekly", "monthly"]),
    default="daily",
    help="Snapshot frequency for historical mode",
)
@click.option("--force", is_flag=True, help="Overwrite existing records")
def backfill_market_odds(
    mode: str,
    days_back: int,
    start_date: str | None,
    end_date: str | None,
    snapshot_interval: str,
    force: bool,
):
    """
    Backfill the market_odds table with historical NBA data from TheOddsAPI.

    Scores mode (free tier): Fetches completed game metadata + scores from
    /v4/sports/{sport}/scores/. Does NOT include odds (free tier limitation)
    but builds the game schedule mapping for training.

    Historical mode (paid tier): Fetches historical odds snapshots from
    /v4/historical/sports/{sport}/odds. Requires paid subscription.

    Stats mode: Shows current state of the market_odds table.

    Examples:
        betting-intel backfill market-odds --mode stats
        betting-intel backfill market-odds --mode scores --days-back 3
        betting-intel backfill market-odds --mode historical \
            --start-date 2024-10-01 --end-date 2024-11-01
    """
    import sys as _sys
    import subprocess

    script_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "tools"
        / "backfill_market_odds.py"
    )
    if not script_path.exists():
        click.echo(f"Error: backfill script not found at {script_path}", err=True)
        raise click.Abort()

    cmd = [_sys.executable, str(script_path), "--mode", mode]

    if mode == "scores":
        cmd.extend(["--days-back", str(days_back)])
    elif mode == "historical":
        if not start_date or not end_date:
            click.echo(
                "Error: --start-date and --end-date required for historical mode",
                err=True,
            )
            raise click.Abort()
        cmd.extend(["--start-date", start_date, "--end-date", end_date])
        cmd.extend(["--snapshot-interval", snapshot_interval])

    if force:
        cmd.append("--force")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.Abort()


# ── Player Prop Commands ──────────────────────────────────────────────
@cli.group()
def recommendations():
    """Generate and view betting recommendations."""
    pass


@recommendations.command("player-props")
@click.argument("home_team")
@click.argument("away_team")
@click.option("--league", default="NBA", help="League (NBA only)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_player_props(
    home_team: str, away_team: str, league: str, json_output: bool
):
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
            marker = " " if s.is_alerted else "  "
            roi_str = (
                f"{s.roi:.1%}" if s.roi >= 0 else click.style(f"{s.roi:.1%}", fg="red")
            )
            pnl_str = (
                f"+${s.total_profit:.0f}"
                if s.total_profit >= 0
                else click.style(f"-${abs(s.total_profit):.0f}", fg="red")
            )
            click.echo(
                f"  {marker}{s.strategy_name:<38} {s.n_bets:<6} {s.wins:<6} {pnl_str:<10} {roi_str:<8}"
            )

    if report.alerted_strategies:
        click.echo()
        click.echo(
            click.style(
                f"    {len(report.alerted_strategies)} STRATEGY ALERT(S) BELOW -5% ROI:",
                fg="red",
            )
        )
        for s in report.alerted_strategies:
            click.echo(
                f"      {s.strategy_name}: ROI={s.roi:.1%} ({s.n_bets} bets, P&L=${s.total_profit:.0f})"
            )

    if save:
        path = tracker.save_report(report)
        click.echo(f"\n  Report saved to: {path}")

    click.echo(f"{'=' * 60}\n")


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
    click.echo(f"## Daily P&L Summary — {today}")
    click.echo()

    # Overall metrics table
    roi_color = "green" if report.overall_roi >= 0 else "red"
    pnl_color = "green" if report.total_profit >= 0 else "red"
    click.echo("| Metric | Value |")
    click.echo("|--------|-------|")
    click.echo(f"| Total Bets | {report.total_bets} |")
    click.echo(f"| Total Stake | ${report.total_stake:,.2f} |")
    click.echo(
        f"| **Total P&L** | **<span style='color:{pnl_color}'>${report.total_profit:+,.2f}</span>** |"
    )
    click.echo(
        f"| **Overall ROI** | **<span style='color:{roi_color}'>{report.overall_roi:+.2%}</span>** |"
    )
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
            marker = "[!]" if s.is_alerted else ""
            click.echo(
                f"| {marker}{s.strategy_name} | {s.n_bets} | {s.wins} | <span style='color:{pnl_color}'>${s.total_profit:+,.0f}</span> | <span style='color:{roi_color}'>{s.roi:+.1%}</span> |"
            )
        click.echo()

    # Alerts
    if report.alerted_strategies:
        click.echo("### Underperforming Strategies")
        click.echo()
        for s in report.alerted_strategies:
            click.echo(
                f"- **{s.strategy_name}**: ROI={s.roi:.1%}, {s.n_bets} bets, P&L=${s.total_profit:+,.0f}"
            )
        click.echo()

    # Daily P&L (last 7 days)
    if report.daily_pnl:
        click.echo("### Last 7 Days P&L")
        click.echo()
        last_7 = (
            report.daily_pnl[-7:] if len(report.daily_pnl) > 7 else report.daily_pnl
        )
        for day in last_7:
            sign = "+" if day["profit"] >= 0 else ""
            if day["profit"] >= 0:
                click.echo(
                    click.style(
                        f"[+] {day['date']}: {sign}${day['profit']:,.0f} ({day['n_bets']} bet{'s' if day['n_bets'] != 1 else ''})",
                        fg="green",
                    )
                )
            else:
                click.echo(
                    click.style(
                        f"[-] {day['date']}: {sign}${day['profit']:,.0f} ({day['n_bets']} bet{'s' if day['n_bets'] != 1 else ''})",
                        fg="red",
                    )
                )
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
        click.echo(click.style(f"    {len(alerted)} ALERT(S):", fg="red"))
        for a in alerted:
            click.echo(
                f"      {a['strategy_name']}: ROI={a['roi']:.1%} (${a['total_profit']:.0f})"
            )

    click.echo(f"{'=' * 60}\n")


# ── Train Commands ────────────────────────────────────────────────────────
@cli.group()
def train():
    """Train ML models on historical data."""
    pass


@train.command("pipeline")
@click.option(
    "--tune",
    is_flag=True,
    default=False,
    help="Run hyperparameter tuning before training (uses Optuna, costs extra time)",
)
@click.option(
    "--n-trials",
    default=30,
    type=int,
    help="Number of Optuna trials per model if --tune is set",
)
@click.option(
    "--model",
    "-m",
    default="all",
    type=click.Choice(
        ["all", "classifier", "totals", "adversarial", "permutation", "bootstrap"]
    ),
    help="Which model(s) to train. 'all' trains classifier + totals + all post-hoc analyses.",
)
@click.option(
    "--adversarial",
    is_flag=True,
    default=False,
    help="Enable adversarial validation (train/test distribution shift detection)",
)
@click.option(
    "--permutation",
    is_flag=True,
    default=False,
    help="Enable permutation feature importance computation",
)
@click.option(
    "--bootstrap",
    is_flag=True,
    default=False,
    help="Enable bootstrap uncertainty quantification",
)
@click.option(
    "--n-bootstrap",
    default=50,
    type=int,
    help="Number of bootstrap samples if --bootstrap is set",
)
@click.option(
    "--prune-top",
    default=0,
    type=int,
    help="Keep only top N models in ensemble (0 = keep all)",
)
@click.option(
    "--save/--no-save", default=True, help="Save trained models to the model registry"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print detailed training progress",
)
def train_pipeline(
    tune: bool,
    n_trials: int,
    model: str,
    adversarial: bool,
    permutation: bool,
    bootstrap: bool,
    n_bootstrap: int,
    prune_top: int,
    save: bool,
    verbose: bool,
):
    """
    Run the full training pipeline: load NBA data, engineer features,
    run hyperparameter tuning (optional), and train classifier + totals models.

    Examples:
        betting-intel train pipeline
        betting-intel train pipeline --tune --n-trials 20
        betting-intel train pipeline --model classifier --adversarial --permutation
        betting-intel train pipeline --model totals --save
    """
    from datetime import datetime as _dt

    start_time = _dt.now()

    click.echo(
        click.style(
            "\n  ═══════════════════════════════════════════", fg="cyan", bold=True
        )
    )
    click.echo(click.style("   TRAINING PIPELINE — v6.6", fg="cyan", bold=True))
    click.echo(
        click.style(
            "  ═══════════════════════════════════════════\n", fg="cyan", bold=True
        )
    )

    # ── Phase 1: Load NBA Data ───────────────────────────────────────
    click.echo(
        click.style(
            "   Phase 1/5: Loading NBA historical data...", fg="yellow", bold=True
        )
    )

    import numpy as np
    from betting_intel.data.loader import NBADataLoader
    from betting_intel.data.features import FeatureEngineer

    loader = NBADataLoader()
    raw_df = loader.load_game_logs()

    if raw_df is None or raw_df.empty:
        click.echo(
            click.style("   No NBA data found! Check your database path.", fg="red")
        )
        click.echo(f"    DB path: {loader.db_path}")
        raise click.Abort()

    n_raw = len(raw_df)
    n_games = raw_df["GAME_ID"].nunique() // 2
    click.echo(f"    Loaded {n_raw} rows ({n_games} games) from {loader.db_path}")

    # ── Phase 2: Engineer Features ───────────────────────────────────-
    click.echo(
        click.style("   Phase 2/5: Engineering features...", fg="yellow", bold=True)
    )

    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)

    fe = FeatureEngineer()
    features_df = fe.build_all_features(games_df, raw_df, league="NBA")

    if features_df is None or features_df.empty:
        click.echo(click.style("   Feature engineering produced no data!", fg="red"))
        raise click.Abort()

    clean_feature_cols = fe.select_features(features_df)
    click.echo(
        f"    {len(clean_feature_cols)} clean features from {features_df.shape[1]} total columns"
    )
    click.echo(
        f"    Date range: {features_df['GAME_DATE'].min():%Y-%m-%d} → {features_df['GAME_DATE'].max():%Y-%m-%d}"
    )

    # Build feature matrices
    _exclude_target_cols = {
        "total_points",
        "point_diff",
        "market_implied_home_prob",
        "market_error",
        "abs_market_error",
        "market_error_clipped",
        "market_error_binary",
        "total_market_error",
        "weighted_market_error",
        "elo_error",
        "market_error_ma_5g",
        "market_error_ma_10g",
        "market_error_trend_home",
        "recent_edge_streak",
    }

    classifier_cols = [c for c in clean_feature_cols if c not in _exclude_target_cols]

    if len(classifier_cols) < 3:
        click.echo(
            click.style(
                f"   Only {len(classifier_cols)} features — need at least 3!", fg="red"
            )
        )
        raise click.Abort()

    # Derive home_win target
    if "home_win" not in features_df.columns:
        if "point_diff" in features_df.columns:
            features_df["home_win"] = (features_df["point_diff"] > 0).astype(int)
        elif "WL_home" in features_df.columns:
            features_df["home_win"] = (features_df["WL_home"] == "W").astype(int)
        else:
            click.echo(click.style("   Cannot derive home_win target!", fg="red"))
            raise click.Abort()

    X_class = features_df[classifier_cols].fillna(0).values
    y_binary = features_df["home_win"].values.astype(int)

    n_samples = len(X_class)
    click.echo(f"    Feature matrix: {n_samples} samples × {X_class.shape[1]} features")
    click.echo(
        f"    Target distribution: home_win {y_binary.mean():.1%}, away_win {1 - y_binary.mean():.1%}"
    )

    results = {}
    totals_mae = None

    # ── Phase 3: Train Classifier (RobustPredictionSystem) ────────────
    if model in ("all", "classifier"):
        click.echo(
            click.style(
                f"   Phase 3/5: Training RobustPredictionSystem "
                f"({'with' if tune else 'without'} hyperparameter tuning)...",
                fg="yellow",
                bold=True,
            )
        )

        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem(
            calibrate=True,
            calibration_method="auto",
            n_folds=5,
            min_train_samples=100,
            use_stacking=True,
            use_hyperparameter_tuning=tune,
            use_adversarial_validation=adversarial or model == "adversarial",
            use_permutation_importance=permutation or model == "permutation",
            use_bootstrap_uncertainty=bootstrap or model == "bootstrap",
            n_bootstrap_samples=n_bootstrap,
            pruning_keep_top_n=prune_top,
            random_state=42,
        )

        t0 = _dt.now()
        system.fit(
            X_class,
            y_binary,
            feature_names=classifier_cols,
            verbose=verbose,
        )
        training_time = (_dt.now() - t0).total_seconds()

        summary = system.get_summary()
        results["classifier"] = summary

        click.echo(f"     Trained in {training_time:.1f}s")
        click.echo(
            f"      Models: {summary.get('n_models', '?')}  "
            f"Brier: {summary.get('calibrated_brier', 'N/A')}  "
            f"Features: {summary.get('n_features', '?')}"
        )

        # Overfitting check
        overfit = system.get_overfitting_report()
        if overfit and overfit.is_overfit:
            click.echo(
                click.style(
                    f"       Overfitting detected! Train R²={overfit.avg_train_r2:.3f} "
                    f"vs Test R²={overfit.avg_test_r2:.3f} (gap={overfit.r2_gap:.3f})",
                    fg="red",
                )
            )
        else:
            click.echo("       No overfitting detected")

        # Model diagnostics
        diags = system.get_model_diagnostics()
        if diags:
            click.echo("      Per-model OOS accuracy:")
            for name, d in sorted(
                diags.items(), key=lambda x: x[1].oos_accuracy, reverse=True
            ):
                click.echo(
                    f"        {name:22s}  acc={d.oos_accuracy:.3f}  Brier={d.oos_brier:.4f}  {d.status}"
                )

        if save:
            from betting_intel.models.persistence import model_registry

            version = model_registry.save(
                model=system,
                model_name="robust_prediction_system",
                feature_cols=classifier_cols,
                metrics={
                    "calibrated_brier": summary.get("calibrated_brier"),
                    "n_models": summary.get("n_models"),
                    "n_samples": n_samples,
                    "n_features": len(classifier_cols),
                    "training_time_seconds": round(training_time, 1),
                    "hyperparameter_tuning": tune,
                },
                parameters={
                    "n_folds": 5,
                    "use_stacking": True,
                    "calibration_method": "auto",
                    "use_adversarial_validation": adversarial,
                    "use_permutation_importance": permutation,
                    "use_bootstrap_uncertainty": bootstrap,
                    "pruning_keep_top_n": prune_top,
                },
            )
            click.echo(f"     Classifier saved as version: {version}")

    # ── Phase 4: Train Totals Model ──────────────────────────────────
    if model in ("all", "totals"):
        click.echo(
            click.style(
                "   Phase 4/5: Training TotalsRegressor...", fg="yellow", bold=True
            )
        )

        from betting_intel.live.totals_model import TotalsRegressor

        # Totals uses same features (exclude target cols)
        totals_cols = classifier_cols  # Same feature set

        if "total_points" not in features_df.columns:
            click.echo(click.style("   `total_points` column not found!", fg="red"))
        else:
            y_total = features_df["total_points"].values.astype(float)

            # Mutual information feature selection for regression
            from sklearn.feature_selection import mutual_info_regression

            mi_reg = mutual_info_regression(
                features_df[totals_cols].fillna(0).values,
                y_total,
                random_state=42,
            )
            top_n = min(60, len(totals_cols))
            if len(mi_reg) > top_n:
                top_idx = np.argsort(mi_reg)[-top_n:]
                totals_cols = [totals_cols[i] for i in top_idx]
                X_total = features_df[totals_cols].fillna(0).values
                click.echo(f"    Mutual info selection: {len(mi_reg)}→{top_n} features")
            else:
                X_total = features_df[totals_cols].fillna(0).values

            regressor = TotalsRegressor(random_state=42)
            t0 = _dt.now()
            regressor.fit(
                X_total,
                y_total,
                feature_names=totals_cols,
                verbose=verbose,
            )
            training_time = (_dt.now() - t0).total_seconds()

            totals_summary = regressor.get_summary()
            totals_mae = totals_summary.get("mae")
            results["totals"] = totals_summary

            click.echo(f"     Trained in {training_time:.1f}s")
            click.echo(
                f"      Models: {totals_summary.get('n_models', '?')}  "
                f"MAE: {totals_mae or '?'}  "
                f"Target mean: {totals_summary.get('target_mean', '?'):.1f}"
            )

            # Per-model diagnostics
            diags = totals_summary.get("model_diagnostics", {})
            if diags:
                sorted_models = sorted(diags.items(), key=lambda x: x[1]["mae"])
                click.echo("      Per-model MAE:")
                for name, d in sorted_models:
                    click.echo(
                        f"        {name:22s}  MAE={d['mae']:.1f}  R²={d.get('r2_score', 0):.3f}  {d.get('status', 'ok')}"
                    )

            if save:
                from betting_intel.models.persistence import model_registry

                version = model_registry.save(
                    model=regressor,
                    model_name="totals_regressor",
                    feature_cols=totals_cols,
                    metrics={
                        "mae": totals_mae,
                        "n_models": totals_summary.get("n_models"),
                        "n_samples": n_samples,
                        "n_features": len(totals_cols),
                        "training_time_seconds": round(training_time, 1),
                    },
                )
                click.echo(f"     Totals model saved as version: {version}")

    # ── Phase 5: Summary ─────────────────────────────────────────────
    elapsed = (_dt.now() - start_time).total_seconds()

    click.echo()
    click.echo(
        click.style(
            "  ═══════════════════════════════════════════", fg="cyan", bold=True
        )
    )
    click.echo(click.style("   TRAINING COMPLETE", fg="cyan", bold=True))
    click.echo(
        click.style(
            "  ═══════════════════════════════════════════", fg="cyan", bold=True
        )
    )
    click.echo(f"  Duration:         {elapsed:.0f}s")
    click.echo(f"  Dataset:          {n_samples} samples, {X_class.shape[1]} features")
    click.echo(
        f"  Date range:       {features_df['GAME_DATE'].min():%Y-%m-%d} → {features_df['GAME_DATE'].max():%Y-%m-%d}"
    )

    if "classifier" in results:
        s = results["classifier"]
        click.echo(f"  Classifier Brier: {s.get('calibrated_brier', 'N/A')}")
        click.echo(f"  Classifier models: {s.get('n_models', '?')}")

    if "totals" in results:
        s = results["totals"]
        click.echo(f"  Totals MAE:       {s.get('mae', 'N/A')}")
        click.echo(f"  Totals models:    {s.get('n_models', '?')}")

    click.echo()


# ── Telegram Commands ───────────────────────────────────────────────────
@cli.group()
def telegram():
    """Telegram notification commands."""
    pass


@telegram.command("test")
def telegram_test():
    """Send a test message to verify Telegram configuration.

    Checks TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or environment,
    then sends a test message to verify the setup is working.
    """
    from betting_intel.notifications.telegram_bot import TelegramNotifier

    notifier = TelegramNotifier()
    click.echo("Testing Telegram configuration...")

    if notifier.is_configured:
        token_masked = notifier.bot_token[:6] + "..." + notifier.bot_token[-4:]
        click.echo(f"  Bot token: {token_masked}")
        click.echo(f"  Chat ID:   {notifier.chat_id}")
    else:
        click.echo("\n  Telegram is not configured.")
        click.echo("  Set these in your .env file:")
        click.echo()
        click.echo(click.style("    TELEGRAM_BOT_TOKEN=your_bot_token_here", fg="cyan"))
        click.echo(click.style("    TELEGRAM_CHAT_ID=your_chat_id_here", fg="cyan"))
        click.echo()
        click.echo("  How to set up:")
        click.echo("  1. Open Telegram and search for @BotFather")
        click.echo("  2. Send /newbot and follow the prompts to create a bot")
        click.echo("  3. Copy the bot token")
        click.echo("  4. Start a chat with your new bot and send /start")
        click.echo("  5. Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
        click.echo("  6. Find your chat_id in the JSON response")
        click.echo()
        raise click.Abort()

    result = notifier.send_test_message_sync()
    click.echo(f"\n  Result: {result}")


@telegram.command("status")
def telegram_status():
    """Show Telegram notification configuration status."""
    from betting_intel.notifications.telegram_bot import TelegramNotifier

    notifier = TelegramNotifier()

    click.echo("\n  Telegram Notification Status")
    click.echo("  " + "-" * 30)

    if notifier.is_configured:
        token_masked = notifier.bot_token[:6] + "..." + notifier.bot_token[-4:]
        click.echo(f"  Status:     {click.style('CONFIGURED', fg='green')}")
        click.echo(f"  Bot token:  {token_masked}")
        click.echo(f"  Chat ID:    {notifier.chat_id}")
        click.echo(f"  Notified:   {len(notifier._notified_game_ids)} game(s) tracked")
    else:
        click.echo(f"  Status:     {click.style('NOT CONFIGURED', fg='red')}")
        click.echo()
        click.echo("  To configure, add to .env:")
        click.echo(f"    {click.style('TELEGRAM_BOT_TOKEN=xxx', fg='cyan')}")
        click.echo(f"    {click.style('TELEGRAM_CHAT_ID=xxx', fg='cyan')}")

    click.echo()


if __name__ == "__main__":
    cli()
