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
from betting_intel.services import logger, setup_logging


# ── Main CLI Group ─────────────────────────────────────────────────────────
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-file", type=click.Path(), help="Path to log file")
@click.version_option(version="0.2.0", prog_name="betting-intel")
def cli(verbose: bool, log_file: str | None):
    """Betting Intelligence System - Professional basketball betting analytics."""
    level = "DEBUG" if verbose else settings.log_level
    setup_logging(level=level, log_file=Path(log_file) if log_file else None)


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


# ── Dashboard Command ─────────────────────────────────────────────────────
@cli.command()
@click.option("--port", default=8501, help="Streamlit port")
def dashboard(port: int):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys as _sys

    dashboard_path = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "app.py"
    click.echo(f"Starting dashboard at http://localhost:{port}")
    subprocess.run(
        [_sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", str(port)],
        cwd=settings.project_root,
    )


# ── Web App (FastAPI) Commands ─────────────────────────────────────────────
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


# ── Small League Commands ──────────────────────────────────────────────
@cli.group()
def small_leagues():
    """Small-league data management commands."""
    pass


@small_leagues.command("list")
def small_leagues_list():
    """List available small leagues with metadata."""
    from betting_intel.data.small_leagues import SmallLeagueIngestion, LEAGUE_METADATA

    click.echo("\nAvailable Small Leagues:")
    click.echo("=" * 60)
    for key, meta in LEAGUE_METADATA.items():
        click.echo(f"\n  {meta['name']} ({key})")
        click.echo(f"    Country: {meta['country']}  |  Tier: {meta['tier']}")
        click.echo(f"    Teams: {meta['num_teams']}  |  Season: {meta['typical_season_months']}")
        click.echo(f"    Data: {meta['data_source']}")
        click.echo(f"    Note: {meta['market_notes']}")
    click.echo("")


@small_leagues.command("fetch")
@click.argument("league", required=False)
@click.option("--historical", is_flag=True, help="Fetch historical games")
@click.option("--upcoming", is_flag=True, help="Fetch upcoming games")
@click.option("--limit", default=20, help="Max upcoming games to fetch")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
def small_leagues_fetch(league: str | None, historical: bool, upcoming: bool, limit: int, output: str | None):
    """Fetch small-league game data.

    Examples:
        betting-intel small-leagues fetch lnb_pro_b --historical
        betting-intel small-leagues fetch cebl --upcoming --limit 10
        betting-intel small-leagues fetch --upcoming --output fixtures.json
    """
    from betting_intel.data.small_leagues import SmallLeagueIngestion

    ing = SmallLeagueIngestion()

    leagues_to_fetch = [league] if league else list(ing.SOURCES.keys())

    results = {}
    for lkey in leagues_to_fetch:
        click.echo(f"\nFetching {lkey}...")
        try:
            if historical:
                df = ing.load_historical(lkey)
                click.echo(f"  Historical: {len(df)} rows")
                if not df.empty:
                    click.echo(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                    click.echo(f"  Teams: {df['team_name'].nunique()}")
                results[f"{lkey}_historical"] = {
                    "count": len(df),
                    "columns": list(df.columns),
                    "sample": df.head(3).to_dict(orient="records") if not df.empty else [],
                }

            if upcoming:
                df = ing.load_upcoming(lkey, limit=limit)
                click.echo(f"  Upcoming: {len(df)} games")
                if not df.empty:
                    click.echo(f"  Games: {df['team_name'].value_counts().to_dict()}")
                results[f"{lkey}_upcoming"] = {
                    "count": len(df),
                    "columns": list(df.columns),
                    "games": df.to_dict(orient="records") if not df.empty else [],
                }
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            results[f"{lkey}_error"] = str(e)

    if output:
        Path(output).write_text(json.dumps(results, indent=2, default=str))
        click.echo(f"\nResults written to {output}")
    click.echo("\nDone.")


@small_leagues.command("teams")
@click.argument("league")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def small_leagues_teams(league: str, json_output: bool):
    """List teams for a small league."""
    from betting_intel.data.small_leagues import SmallLeagueIngestion

    ing = SmallLeagueIngestion()
    try:
        teams = ing.get_teams(league)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if teams.empty:
        click.echo(f"No teams found for '{league}'.")
        return

    if json_output:
        click.echo(teams.to_json(orient="records", indent=2))
    else:
        click.echo(f"\n{'Team Name':<30} {'ID':<15} {'Country':<12}")
        click.echo("-" * 60)
        for _, row in teams.iterrows():
            click.echo(f"{row.get('team_name', ''):<30} {str(row.get('team_id', '')):<15} {row.get('country', ''):<12}")
        click.echo(f"\nTotal: {len(teams)} teams")


@small_leagues.command("bridge")
@click.argument("league")
@click.option("--output", "-o", type=click.Path(), help="Output CSV file path")
def small_leagues_bridge(league: str, output: str | None):
    """Bridge small-league data to NBA-pipeline format and display summary."""
    from betting_intel.data.small_leagues import SmallLeagueIngestion
    from betting_intel.data.small_leagues.unified_bridge import SmallLeagueBridge

    ing = SmallLeagueIngestion()
    bridge = SmallLeagueBridge()

    try:
        df = ing.load_historical(league)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if df.empty:
        click.echo(f"No data for '{league}'. Try 'betting-intel small-leagues fetch {league} --historical' first.")
        return

    click.echo(f"\nBridging {league}: {len(df)} rows")
    bridged = bridge.bridge_dataframe(df)
    click.echo(f"Bridged: {len(bridged)} rows ({len(df) - len(bridged)} filtered)")

    total_games = bridged["game_id"].nunique()
    total_teams = bridged["team_name"].nunique()
    avg_pts = bridged["team_score"].mean()
    click.echo(f"  Games: {total_games}  |  Teams: {total_teams}  |  Avg PTS: {avg_pts:.1f}")

    if output:
        bridged.to_csv(output, index=False)
        click.echo(f"Written to {output}")


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
    from betting_intel.recommendations import RecommendationEngine

    engine = RecommendationEngine(min_edge_threshold=min_edge)

    with click.progressbar(length=1, label="Generating recommendations") as bar:
        all_bets = engine.generate_all_bets()
        bar.update(1)

    if clear_only:
        clear_picks = engine.get_clear_picks(threshold=min_edge)
        display_bets = [c.bet for c in clear_picks]
    else:
        display_bets = all_bets

    # Filter by league
    if league.lower() != "all":
        display_bets = [b for b in display_bets if b.league.lower() == league.lower()]

    # Filter by bet type
    if bet_type.lower() != "all":
        type_map = {"moneyline": "moneyline", "spread": "spread", "total": "total_points", "props": "player"}
        mapped = type_map.get(bet_type.lower(), bet_type.lower())
        display_bets = [b for b in display_bets if mapped in b.bet_type.value or mapped in b.bet_type.display_name().lower()]

    display_bets = display_bets[:limit]

    if json_output:
        click.echo(json.dumps([b.as_dict() for b in display_bets], indent=2))
        return

    if not display_bets:
        click.echo("No recommendations found matching your criteria.")
        return

    summary = engine.get_summary()
    click.echo(f"\n{'=' * 80}")
    click.echo(f"  BETTING RECOMMENDATIONS — EXACT BETS")
    click.echo(f"  {summary['total_bets']} bets evaluated, {summary['clear_picks']} clear picks, {summary['games_available']} games")
    click.echo(f"{'=' * 80}")

    for i, bet in enumerate(display_bets, 1):
        cp = "★" if bet.is_clear_pick else " "
        edge_str = f"edge: {bet.edge_pct:.1%}" if bet.edge_pct > 0 else ""
        action = _format_action(bet)
        click.echo(f"  {cp} #{i:<2} {action:<50} {edge_str}")

    total_stake = sum(b.stake_dollars for b in display_bets)
    click.echo(f"\n  Total exposure: ${total_stake:.0f} | Bankroll: ${summary['bankroll']:,.0f}")
    click.echo()


@recommendations.command("todays-card")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_todays_card(json_output: bool):
    """Show today's betting card with exact bets to place."""
    from betting_intel.recommendations import RecommendationEngine

    engine = RecommendationEngine()

    with click.progressbar(length=3, label="Loading today's card") as bar:
        all_bets = engine.generate_all_bets()
        bar.update(1)
        todays_bets = engine.get_todays_card()
        bar.update(1)
        summary = engine.get_summary()
        bar.update(1)

    if json_output:
        click.echo(json.dumps({
            "summary": summary,
            "bets": [b.as_dict() for b in todays_bets],
        }, indent=2))
        return

    click.echo(f"\n{'=' * 80}")
    click.echo(f"  TODAY'S BETTING CARD — PLACE THESE BETS")
    click.echo(f"  {datetime.now().strftime('%A, %B %d, %Y')}")
    click.echo(f"{'=' * 80}")

    # Group by game
    games = {}
    for bet in todays_bets:
        key = (bet.matchup, bet.league)
        if key not in games:
            games[key] = []
        games[key].append(bet)

    for (matchup, league), game_bets in games.items():
        click.echo(f"\n  {matchup}")
        click.echo(f"  {'─' * 40}")
        for bet in game_bets:
            cp = "★" if bet.is_clear_pick else " "
            action = _format_action(bet)
            click.echo(f"  {cp} {action}")


@recommendations.command("tomorrow")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_tomorrow(json_output: bool):
    """Show tomorrow's predictions with exact bets to place — one-day-ahead."""
    from betting_intel.recommendations import RecommendationEngine

    engine = RecommendationEngine()

    with click.progressbar(length=3, label="Loading tomorrow's card") as bar:
        all_bets = engine.generate_all_bets()
        bar.update(1)
        tomorrow_bets = engine.get_tomorrows_card()
        bar.update(1)
        summary = engine.get_summary()
        bar.update(1)

    if json_output:
        click.echo(json.dumps({
            "summary": summary,
            "bets": [b.as_dict() for b in tomorrow_bets],
        }, indent=2))
        return

    tomorrow_date = (datetime.now() + __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")

    click.echo(f"\n{'=' * 80}")
    click.echo(f"  TOMORROW'S BETTING CARD — {tomorrow_date}")
    click.echo(f"  One-day-ahead predictions — place these bets")
    click.echo(f"{'=' * 80}")

    # Group by game
    games = {}
    for bet in tomorrow_bets:
        key = (bet.matchup, bet.league)
        if key not in games:
            games[key] = []
        games[key].append(bet)

    if not games:
        click.echo("\n  No games scheduled for tomorrow.")
        click.echo()
        return

    total_stake = 0
    for (matchup, league), game_bets in games.items():
        best_bet = max(game_bets, key=lambda b: b.edge_pct)
        game_stake = sum(b.stake_dollars for b in game_bets)
        total_stake += game_stake

        click.echo(f"\n  {'─' * 60}")
        click.echo(f"  {matchup} ({league})")
        click.echo(f"  {'─' * 60}")

        # Prime play highlighted
        cp = "★" if best_bet.is_clear_pick else "▶"
        action = _format_action(best_bet)
        click.echo(f"  {cp} PRIME PLAY: {action}")
        click.echo(f"     Edge: {best_bet.edge_pct:.1%} | Confidence: {best_bet.confidence.value} | Stake: ${best_bet.stake_dollars:.0f}")

        # Other bets
        for bet in game_bets:
            if bet != best_bet:
                action = _format_action(bet)
                click.echo(f"     {action}")

    click.echo(f"\n  {'═' * 60}")
    click.echo(f"  TOMORROW'S TOTAL EXPOSURE: ${total_stake:,.0f}")
    click.echo(f"  {len(tomorrow_bets)} bets across {len(games)} games")
    click.echo()


@recommendations.command("clear-picks")
@click.option("--min-edge", default=0.03, type=float, help="Minimum edge threshold")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def recommendations_clear_picks(min_edge: float, json_output: bool):
    """Show only the clear picks — the exact bets to place RIGHT NOW."""
    from betting_intel.recommendations import RecommendationEngine

    engine = RecommendationEngine()
    clear_picks = engine.get_clear_picks(threshold=min_edge)

    if json_output:
        click.echo(json.dumps([c.as_dict() for c in clear_picks], indent=2))
        return

    if not clear_picks:
        click.echo("No clear picks found at this threshold.")
        return

    click.echo(f"\n{'★' * 80}")
    click.echo(f"  CLEAR PICKS — PLACE THESE BETS")
    click.echo(f"  {len(clear_picks)} high-confidence opportunities identified")
    click.echo(f"{'★' * 80}")

    for i, cp in enumerate(clear_picks, 1):
        bet = cp.bet
        risk_color = {"CONSERVATIVE": "safe", "MODERATE": "moderate", "AGGRESSIVE": "aggressive"}.get(cp.risk_level, "unknown")
        click.echo(f"\n  ★ PICK #{i} — {cp.risk_level} ({cp.clear_score:.0f}/100 confidence)")
        click.echo(f"  {'─' * 50}")
        click.echo(f"  \033[1mPLACE: ${bet.stake_dollars:.0f} on {bet.bet_side}\033[0m")
        click.echo(f"  Game:     {bet.matchup} ({bet.league})")
        click.echo(f"  Market:   {bet.bet_type.display_name()}")
        for reason in cp.reasons:
            click.echo(f"  → {reason}")

    total_stake = sum(c.bet.stake_dollars for c in clear_picks)
    avg_edge = sum(c.bet.edge_pct for c in clear_picks) / len(clear_picks)
    click.echo(f"\n  {'─' * 50}")
    click.echo(f"  TOTAL TO BET: ${total_stake:.0f} across {len(clear_picks)} bets")
    click.echo(f"  AVG EDGE:     {avg_edge:.1%}")
    click.echo()


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
