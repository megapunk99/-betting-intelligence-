"""
Staking mixin — recommendation generation, player props, +EV scanning, and arbitrage.

Extracted from the monolithic predict_tomorrow.py PredictionPipeline class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from betting_intel.pipeline.bootstrap import (
    logger,
    HAS_RECOMMENDATIONS,
    PlayerPropEngine,
)
from betting_intel.recommendations.engine import RecommendationEngine
from betting_intel.recommendations.ranker import BetRanker
from betting_intel.recommendations.ev_scanner import PositiveEVScanner
from betting_intel.recommendations.arbitrage import ArbitrageDetector


class StakingMixin:
    """Mixin providing recommendation, player props, and opportunity scanning methods."""

    # ── Recommendations ─────────────────────────────────────────────

    def generate_recommendations(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate all bet types, rank by edge, identify clear picks."""
        print("\n" + "=" * 70)
        print("  💰  STAGE 4: RECOMMENDATION ENGINE")
        print("=" * 70)

        recommendations: List[Dict[str, Any]] = []
        min_edge = self.args.min_edge

        if HAS_RECOMMENDATIONS:
            recommendations = self._run_recommendation_engine(predictions_df, min_edge)

        # Fallback: basic edge calculation
        if not recommendations and "predicted_total" in predictions_df.columns:
            recommendations = self._basic_edge_recommendations(predictions_df, min_edge)

        return recommendations

    def _run_recommendation_engine(self, predictions_df: pd.DataFrame,
                                    min_edge: float) -> List[Dict[str, Any]]:
        """Run the full recommendation engine pipeline."""
        recommendations: List[Dict[str, Any]] = []
        try:
            engine = RecommendationEngine()
            ranker = BetRanker()

            all_bets = engine.generate_all_bets(predictions=predictions_df)
            if all_bets:
                print(f"  ✅  Generated {len(all_bets)} total bet opportunities")

                ranked = ranker.rank_bets(all_bets) if hasattr(ranker, 'rank_bets') else all_bets

                print(f"  📊  Generated {len(all_bets)} total bets")
                for i, bet in enumerate(all_bets[:5]):
                    bd = bet.as_dict() if hasattr(bet, 'as_dict') else bet
                    team = self._extract_team_name(bet, bd)
                    edge = float(getattr(bet, 'edge_pct', bd.get('edge_pct', 0)))
                    conf = getattr(bet, 'confidence', bd.get('confidence', 'N/A'))
                    print(f"       {i+1}. {team}: edge={edge:.2%}, conf={conf}")

                # Identify clear picks
                self._identify_clear_picks(all_bets, ranker, min_edge)

                # Convert to dicts, filtering zero-edge bets
                recommendations = self._convert_bets_to_dicts(ranked, all_bets, min_edge)
                self.results["recommendations"] = recommendations

        except Exception as e:
            print(f"  ⚠  Recommendation engine failed: {e}")

        return recommendations

    def _extract_team_name(self, bet, bd: dict) -> str:
        """Extract team name from a bet suggestion object/dict."""
        team = getattr(bet, 'team', getattr(bet, 'home_team', bd.get('team', '')))
        if not team:
            matchup = getattr(bet, 'matchup', bd.get('matchup', ''))
            if matchup and ' @ ' in matchup:
                team = matchup.split(' @ ')[1]
            elif matchup and ' vs ' in matchup:
                team = matchup.split(' vs ')[0]
        return team or '?'

    def _identify_clear_picks(self, all_bets, ranker, min_edge: float):
        """Identify and store clear picks."""
        clear_picks = []
        try:
            if hasattr(ranker, 'get_clear_picks'):
                try:
                    clear_picks = ranker.get_clear_picks(all_bets, threshold=min_edge)
                except TypeError:
                    clear_picks = ranker.get_clear_picks(all_bets)
            elif hasattr(ranker, 'MIN_EDGE'):
                clear_picks = [b for b in all_bets if getattr(b, 'is_clear_pick', False)]
        except Exception:
            pass

        if clear_picks:
            print(f"  🎯  {len(clear_picks)} Clear Picks identified")
            self.results["clear_picks"] = [
                {
                    "team": getattr(p, 'team', getattr(p, 'home_team', '?')),
                    "edge": float(getattr(p, 'edge', 0)),
                    "confidence": str(getattr(p, 'confidence', '')),
                    "bet_type": str(getattr(p, 'bet_type', getattr(p, 'suggestion_type', ''))),
                    "odds": getattr(p, 'odds', 0),
                }
                for p in clear_picks[:10]
            ]

    def _convert_bets_to_dicts(self, ranked, all_bets, min_edge: float) -> List[Dict[str, Any]]:
        """Convert BetSuggestion objects to recommendation dicts."""
        recommendations = []
        for b in (ranked if isinstance(ranked, list) else all_bets):
            edge = float(getattr(b, 'edge_pct', getattr(b, 'edge', 0)))
            if abs(edge) < min_edge:
                continue
            team = self._extract_team_name(b, getattr(b, 'as_dict', lambda: {})() if hasattr(b, 'as_dict') else {})
            recommendations.append({
                "team": team,
                "bet_type": str(getattr(b, 'bet_type', getattr(b, 'suggestion_type', ''))),
                "edge": edge,
                "confidence": str(getattr(b, 'confidence', '')),
                "odds": getattr(b, 'odds', -110) or -110,
                "stake": getattr(b, 'stake_dollars', getattr(b, 'stake', 0)),
                "expected_value": float(getattr(b, 'expected_value', 0)),
            })
        return recommendations

    def _basic_edge_recommendations(self, predictions_df: pd.DataFrame,
                                     min_edge: float) -> List[Dict[str, Any]]:
        """Basic edge-based recommendations fallback."""
        print("  ℹ  Using basic edge calculation...")
        recommendations = []
        for _, row in predictions_df.iterrows():
            if "market_total" in predictions_df.columns:
                market_total = row.get("market_total", 0)
                predicted_total = row.get("predicted_total", 0)
                if market_total and predicted_total:
                    edge = (predicted_total - market_total) / market_total
                    if abs(edge) >= min_edge:
                        team = row.get("home_team", row.get("team", "?"))
                        recommendations.append({
                            "team": team,
                            "bet_type": "total_over" if edge > 0 else "total_under",
                            "edge": abs(edge),
                            "confidence": "high" if abs(edge) > 0.05 else "medium",
                            "odds": row.get("over_odds", -110) if edge > 0 else row.get("under_odds", -110),
                            "expected_value": abs(edge),
                        })
        return recommendations

    # ── Player Props ────────────────────────────────────────────────

    def generate_player_props(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate player prop bet recommendations."""
        print("\n" + "=" * 70)
        print("  🏀  STAGE 5: PLAYER PROPS")
        print("=" * 70)

        props: List[Dict[str, Any]] = []

        if HAS_RECOMMENDATIONS:
            try:
                generator = PlayerPropEngine()
                all_props_list = []
                home_col = 'home_team' if 'home_team' in predictions_df.columns else 'team'
                away_col = 'away_team' if 'away_team' in predictions_df.columns else 'opponent'

                for _, row in predictions_df.iterrows():
                    game_props = generator.predict_for_game(
                        home=str(row.get(home_col, 'Home')),
                        away=str(row.get(away_col, 'Away')),
                    )
                    if game_props:
                        all_props_list.extend(
                            game_props if isinstance(game_props, list) else [game_props]
                        )

                if all_props_list:
                    print(f"  ✅  Generated {len(all_props_list)} player props")
                    for prop in all_props_list[:5]:
                        prop_dict = prop.as_dict() if hasattr(prop, 'as_dict') else {}
                        bet_side = prop_dict.get('bet_side', getattr(prop, 'bet_side', ''))
                        player_name = bet_side.split(' o')[0].split(' u')[0] if bet_side else '?'
                        pt = prop_dict.get('bet_type_display', '?')
                        ml = prop_dict.get('market_line', 0)
                        ev = float(prop_dict.get('edge_pct', 0))
                        print(f"       {player_name}: {pt} → {ml} (edge: {ev:.2%})")

                    props = []
                    for p in all_props_list:
                        pdict = p.as_dict() if hasattr(p, 'as_dict') else {}
                        bet_side = pdict.get('bet_side', getattr(p, 'bet_side', ''))
                        player_name = bet_side.split(' o')[0].split(' u')[0] if bet_side else '?'
                        props.append({
                            "player": player_name,
                            "team": pdict.get('matchup', '?').split(' @ ')[-1] if pdict.get('matchup') else '?',
                            "prop_type": pdict.get('bet_type_display', '?'),
                            "line": pdict.get('market_line', 0),
                            "edge": float(pdict.get('edge_pct', 0)),
                            "confidence": str(pdict.get('confidence', '')),
                            "odds": pdict.get('odds', -110),
                        })
                    self.results["player_props"] = props
                else:
                    print("  ℹ  No player props generated")
            except Exception as e:
                print(f"  ⚠  PlayerPropGenerator failed: {e}")

        return props

    # ── +EV Scanning & Arbitrage ────────────────────────────────────

    def scan_opportunities(self, predictions_df: pd.DataFrame):
        """Scan for +EV opportunities and arbitrage across sportsbooks."""
        print("\n" + "=" * 70)
        print("  🔬  STAGE 6: +EV SCANNING & ARBITRAGE")
        print("=" * 70)

        self._scan_ev(predictions_df)
        self._scan_arbitrage(predictions_df)

    def _scan_ev(self, predictions_df: pd.DataFrame):
        """Scan for +EV opportunities."""
        if not HAS_RECOMMENDATIONS:
            print("  ℹ  +EV scanner not available")
            return
        try:
            scanner = PositiveEVScanner()
            ev_report = scanner.scan_odds_snapshots(predictions_df.to_dict("records"))
            if ev_report:
                opportunities = getattr(ev_report, "opportunities", []) or []
                if opportunities:
                    print(f"  ✅  Found {len(opportunities)} +EV opportunities")
                    for opp in opportunities[:5]:
                        print(f"       {getattr(opp, 'game', '?')}: "
                              f"EV={getattr(opp, 'expected_value', 0):.2%}, "
                              f"confidence={getattr(opp, 'confidence', 'N/A')}")
                    self.results["ev_opportunities"] = [
                        {
                            "game": getattr(o, "game", "?"),
                            "bet_type": getattr(o, "bet_type", "?"),
                            "expected_value": float(getattr(o, "expected_value", 0)),
                            "confidence": str(getattr(o, "confidence", "")),
                            "source": str(getattr(o, "source", "")),
                        }
                        for o in opportunities[:20]
                    ]
                else:
                    print("  ℹ  No +EV opportunities found")
            else:
                print("  ℹ  No EV report generated")
        except Exception as e:
            print(f"  ⚠  +EV scanning failed: {e}")

    def _scan_arbitrage(self, predictions_df: pd.DataFrame):
        """Scan for arbitrage opportunities."""
        if not HAS_RECOMMENDATIONS:
            print("  ℹ  Arbitrage detector not available")
            return
        try:
            detector = ArbitrageDetector()
            arb_report = detector.scan_for_arbitrage(predictions_df.to_dict("records"))
            if arb_report:
                opportunities = getattr(arb_report, "opportunities", []) or []
                if opportunities:
                    print(f"  ✅  Found {len(opportunities)} arbitrage opportunities!")
                    for arb in opportunities[:3]:
                        print(f"       {getattr(arb, 'game', '?')}: "
                              f"return={getattr(arb, 'return_pct', 0):.2%}")
                    self.results["arbitrage_opportunities"] = [
                        {
                            "game": getattr(a, "game", "?"),
                            "return_pct": float(getattr(a, "return_pct", 0)),
                            "outcomes": getattr(a, "outcomes", []),
                            "stakes": getattr(a, "stakes", {}),
                        }
                        for a in opportunities[:10]
                    ]
                else:
                    print("  ℹ  No arbitrage opportunities found")
        except Exception as e:
            print(f"  ⚠  Arbitrage detection failed: {e}")
