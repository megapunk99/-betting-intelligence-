"""
Reporting mixin — console, JSON, and HTML report generation.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from betting_intel.pipeline.bootstrap import PROJECT_ROOT


class ReportingMixin:
    """Mixin providing report generation methods for PredictionPipeline."""

    def generate_report(self):
        """Print a comprehensive summary and optionally save outputs."""
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("  📋  FINAL REPORT")
        print("=" * 70)
        print(f"  ⏱  Pipeline completed in {elapsed:.1f}s")
        print(f"  📊  Data source: {self.results.get('metadata', {}).get('data_source', 'N/A')}")
        print(f"  📅  Mode: {'LIVE' if self.args.live else 'HISTORICAL'}")

        n_games = len(self.results.get("predictions", []))
        n_recommendations = len(self.results.get("recommendations", []))
        n_clear = len(self.results.get("clear_picks", []))
        n_ev = len(self.results.get("ev_opportunities", []))
        n_arb = len(self.results.get("arbitrage_opportunities", []))
        n_props = len(self.results.get("player_props", []))

        print(f"  🎮  Games analyzed: {n_games}")
        print(f"  💰  Bet recommendations: {n_recommendations}")
        print(f"  🎯  Clear picks: {n_clear}")
        print(f"  🔬  +EV opportunities: {n_ev}")
        print(f"  ♻   Arbitrage opportunities: {n_arb}")
        print(f"  🏀  Player props: {n_props}")

        self._print_bankroll_summary()
        self._print_validation_summary()
        self._print_simulation_summary()
        self._print_top_picks()
        self._save_outputs(elapsed)

    def _print_bankroll_summary(self):
        """Print bankroll and staking summary."""
        risk = self.results.get("risk_assessment", {})
        if risk:
            bankroll = risk.get("bankroll", self.args.bankroll)
            total_staked = sum(b.get("stake", 0) for b in risk.get("bets", []))
            n_bets = len(risk.get("bets", []))
            pct = total_staked / bankroll if bankroll else 0
            print(f"  💵  Bankroll: ${bankroll:.2f} | Staked: ${total_staked:.2f} ({pct:.1%}) across {n_bets} bets")

    def _print_validation_summary(self):
        """Print validation results summary."""
        val = self.results.get("validation", {})
        if val:
            cal = val.get("calibration", {})
            overfit = val.get("overfitting", {})
            drift = val.get("drift", {})
            if cal:
                print(f"  📐  Calibration: checked")
            if overfit:
                status = 'DETECTED' if overfit.get('is_overfit', overfit.get('overfit', False)) else 'None'
                print(f"  ⚠  Overfitting: {status}")
            if drift:
                print(f"  🌊  Drift: {'DETECTED' if drift.get('drift_detected', False) else 'None'}")

    def _print_simulation_summary(self):
        """Print simulation results."""
        sim = self.results.get("simulation", {})
        if sim:
            med = sim.get("median_return", sim.get("median", 0))
            print(f"  🎲  Simulation (10k runs): median=${med:+.2f}")

    def _print_top_picks(self):
        """Print top clear picks, EV opportunities, and arbitrage."""
        clear_picks = self.results.get("clear_picks", [])
        if clear_picks:
            print(f"\n  ── TOP CLEAR PICKS ──")
            for i, pick in enumerate(clear_picks[:5]):
                print(f"   {i+1}. {pick.get('team', '?')} ({pick.get('bet_type', '?')}) "
                      f"edge={pick.get('edge', 0):.2%} "
                      f"conf={pick.get('confidence', 'N/A')}")

        ev_opps = self.results.get("ev_opportunities", [])
        if ev_opps:
            print(f"\n  ── TOP +EV OPPORTUNITIES ──")
            for i, opp in enumerate(ev_opps[:3]):
                print(f"   {i+1}. {opp.get('game', '?')} ({opp.get('bet_type', '?')}) "
                      f"EV={opp.get('expected_value', 0):.2%}")

        arb_opps = self.results.get("arbitrage_opportunities", [])
        if arb_opps:
            print(f"\n  ── ARBITRAGE OPPORTUNITIES ──")
            for i, arb in enumerate(arb_opps[:3]):
                print(f"   {i+1}. {arb.get('game', '?')} return={arb.get('return_pct', 0):.2%}")

    def _save_outputs(self, elapsed: float):
        """Save JSON output and generate HTML report if requested."""
        if self.args.output:
            output_path = Path(self.args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n  💾  Results saved to {output_path}")

        if self.args.html:
            self._generate_html_report()

        if self.args.scheduled:
            self._print_scheduled_summary(elapsed)

    def _print_scheduled_summary(self, elapsed: float):
        """Print JSON summary for scheduled mode (parsed by TaskScheduler)."""
        from datetime import datetime
        summary = {
            "status": "complete",
            "duration_seconds": round(elapsed, 1),
            "data_source": self.results.get("metadata", {}).get("data_source", "N/A"),
            "games": len(self.results.get("predictions", [])),
            "recommendations": len(self.results.get("recommendations", [])),
            "clear_picks": len(self.results.get("clear_picks", [])),
            "ev_opportunities": len(self.results.get("ev_opportunities", [])),
            "arbitrage": len(self.results.get("arbitrage_opportunities", [])),
            "player_props": len(self.results.get("player_props", [])),
            "bankroll": self.args.bankroll,
            "total_staked": sum(
                b.get("stake", 0)
                for b in self.results.get("risk_assessment", {}).get("bets", [])
            ),
            "risk_assessment": self.results.get("risk_assessment", {}),
            "clear_picks_detail": self.results.get("clear_picks", []),
            "ev_detail": self.results.get("ev_opportunities", []),
            "arbitrage_detail": self.results.get("arbitrage_opportunities", []),
            "timestamp": datetime.now().isoformat(),
        }
        print(f"##SCHEDULED_RESULT##{json.dumps(summary, default=str)}")

    # ── HTML Report ────────────────────────────────────────────────

    def _generate_html_report(self):
        """Generate a standalone HTML report with all results."""
        try:
            from jinja2 import Environment, FileSystemLoader
            templates_dir = PROJECT_ROOT / "web" / "templates"
            if templates_dir.exists():
                env = Environment(loader=FileSystemLoader(str(templates_dir)))
                template = env.get_template("tomorrow.html") if (templates_dir / "tomorrow.html").exists() else None
                if template:
                    html = template.render(
                        results=self.results,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        args=self.args,
                    )
                    report_path = PROJECT_ROOT / "reports" / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.html"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(html, encoding="utf-8")
                    print(f"  🌐  HTML report: {report_path}")
                    return
        except Exception as e:
            print(f"  ⚠  HTML report generation failed: {e}")

        # Fallback: inline HTML
        html = self._build_inline_html_report()
        report_path = PROJECT_ROOT / "reports" / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html, encoding="utf-8")
        print(f"  🌐  HTML report: {report_path}")

    def _build_inline_html_report(self) -> str:
        """Build a self-contained HTML report."""
        recs = self.results.get("recommendations", [])
        clear = self.results.get("clear_picks", [])
        ev = self.results.get("ev_opportunities", [])
        arb = self.results.get("arbitrage_opportunities", [])
        props = self.results.get("player_props", [])

        rows_html = ""
        for r in recs[:20]:
            rows_html += f"""
            <tr>
                <td>{r.get('team', '?')}</td>
                <td>{r.get('bet_type', '?')}</td>
                <td>{r.get('edge', 0):.2%}</td>
                <td>{r.get('confidence', 'N/A')}</td>
                <td>${r.get('stake', 0):.2f}</td>
                <td>{r.get('odds', 0)}</td>
            </tr>"""

        clear_html = ""
        for c in clear[:10]:
            clear_html += f"""
            <tr>
                <td>{c.get('team', '?')}</td>
                <td>{c.get('bet_type', '?')}</td>
                <td>{c.get('edge', 0):.2%}</td>
                <td>{c.get('confidence', 'N/A')}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Betting Intelligence — Prediction Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
          background:#0f1219; color:#e1e5ed; padding:2rem; }}
  .container {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:1.8rem; margin-bottom:0.5rem; background:linear-gradient(135deg,#6366f1,#a855f7);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .meta {{ color:#8892a4; font-size:0.9rem; margin-bottom:2rem; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1rem; margin-bottom:2rem; }}
  .stat-card {{ background:#1a1f2e; border-radius:12px; padding:1.2rem; text-align:center;
               border:1px solid #2a3040; }}
  .stat-card .num {{ font-size:1.8rem; font-weight:700; color:#6366f1; }}
  .stat-card .label {{ font-size:0.8rem; color:#8892a4; margin-top:0.3rem; }}
  table {{ width:100%; border-collapse:collapse; background:#1a1f2e; border-radius:12px;
          overflow:hidden; margin-bottom:2rem; }}
  th {{ background:#2a3040; padding:0.8rem 1rem; text-align:left; font-size:0.85rem;
        color:#8892a4; text-transform:uppercase; letter-spacing:0.05em; }}
  td {{ padding:0.7rem 1rem; border-top:1px solid #2a3040; font-size:0.9rem; }}
  tr:hover {{ background:#222838; }}
  .section-title {{ font-size:1.2rem; font-weight:600; margin:1.5rem 0 1rem;
                    color:#a5b4fc; }}
</style>
</head>
<body>
<div class="container">
  <h1>🏀 Betting Intelligence Report</h1>
  <p class="meta">Generated {datetime.now():%B %d, %Y at %H:%M:%S} · {'LIVE' if self.args.live else 'HISTORICAL'}</p>

  <div class="stats">
    <div class="stat-card"><div class="num">{len(recs)}</div><div class="label">Recommendations</div></div>
    <div class="stat-card"><div class="num">{len(clear)}</div><div class="label">Clear Picks</div></div>
    <div class="stat-card"><div class="num">{len(ev)}</div><div class="label">+EV Opportunities</div></div>
    <div class="stat-card"><div class="num">{len(arb)}</div><div class="label">Arbitrage</div></div>
    <div class="stat-card"><div class="num">{len(props)}</div><div class="label">Player Props</div></div>
  </div>

  <div class="section-title">🎯 Top Recommendations</div>
  <table>
    <thead><tr><th>Team</th><th>Type</th><th>Edge</th><th>Confidence</th><th>Stake</th><th>Odds</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="section-title">⭐ Clear Picks</div>
  <table>
    <thead><tr><th>Team</th><th>Type</th><th>Edge</th><th>Confidence</th></tr></thead>
    <tbody>{clear_html}</tbody>
  </table>

  <p class="meta" style="text-align:center;margin-top:3rem;">
    Powered by Betting Intelligence Engine v3.0
  </p>
</div>
</body>
</html>"""
