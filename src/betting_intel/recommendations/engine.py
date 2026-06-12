"""
RecommendationEngine — generates BetSuggestion objects for ALL bet types.

Uses model predictions (total points, spread, win prob) and market odds
to produce actionable betting recommendations with proper edge computation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import pandas as pd

from betting_intel.recommendations.bet_types import (
    BetType,
    BetSuggestion,
    Confidence,
    MoneylineBet,
    SpreadBet,
    TotalBet,
    TeamTotalBet,
    QuarterBet,
    QuarterTotalBet,
    HalfTotalBet,
)

logger = logging.getLogger(__name__)


# Lazy-loaded QuarterHalfProjector singleton
_quarter_projector = None


def _get_quarter_projector():
    """Get (or create) the quarter/half projector with cached ratios."""
    global _quarter_projector
    if _quarter_projector is None:
        from betting_intel.data.quarter_projector import QuarterHalfProjector
        _quarter_projector = QuarterHalfProjector()
        # Try to load cached ratios (non-blocking if no cache)
        _quarter_projector.compute_ratios(max_games=200)
    return _quarter_projector


def _confidence_from_edge(edge: float) -> Confidence:
    if edge >= 0.08:
        return Confidence.VERY_HIGH
    elif edge >= 0.05:
        return Confidence.HIGH
    elif edge >= 0.03:
        return Confidence.MEDIUM
    elif edge >= 0.01:
        return Confidence.LOW
    return Confidence.VERY_LOW


class RecommendationEngine:
    """Generates bet recommendations from model predictions."""

    def generate_all_bets(self, predictions: pd.DataFrame) -> List[BetSuggestion]:
        """Generate all bet types from a predictions DataFrame.

        Args:
            predictions: DataFrame with columns like home_team, away_team,
                        predicted_total, predicted_spread, market_total, etc.

        Returns:
            List of BetSuggestion objects for all eligible bet types.
        """
        bets: List[BetSuggestion] = []
        if predictions is None or predictions.empty:
            return bets

        for idx, row in predictions.iterrows():
            home = str(row.get("home_team", row.get("team", "")))
            away = str(row.get("away_team", row.get("opponent", "")))
            if not home or not away:
                continue

            game_id = str(row.get("game_id", f"g_{idx}"))
            game_date = str(row.get("game_date", ""))
            matchup = f"{away} @ {home}"
            league = str(row.get("league", "NBA"))

            # ── Total O/U bets ───────────────────────────────────────
            predicted_total = row.get("predicted_total", 0)
            market_total = row.get("market_total", 0) or row.get("market_line_baseline", 0)
            if predicted_total and market_total:
                edge = (predicted_total - market_total) / market_total
                if abs(edge) > 0:
                    side = "OVER" if edge > 0 else "UNDER"
                    conf = _confidence_from_edge(abs(edge))
                    bet = TotalBet(
                        game_id=game_id, game_date=game_date, matchup=matchup,
                        side=side, market_total=float(market_total),
                        predicted_total=float(predicted_total), league=league,
                        confidence=conf,
                        reasoning=f"Model predicts {predicted_total:.1f} vs market {market_total}",
                    )
                    bet.is_clear_pick = conf.is_clear() and abs(edge) >= 0.03
                    bet.model_name = "pipeline_ensemble"
                    bets.append(bet)

            # ── Spread bets ──────────────────────────────────────────
            predicted_spread = row.get("predicted_spread", None)
            market_spread = row.get("spread", None)
            if predicted_spread is not None and market_spread is not None:
                try:
                    ps = float(predicted_spread)
                    ms = float(market_spread)
                    if ps != 0 and ms != 0:
                        # Home team spread
                        home_edge = ps - ms
                        away_edge = -ps - (-ms)
                        for team, spread_line, pred_margin, edge_val in [
                            (home, ms, ps, home_edge),
                            (away, -ms, -ps, away_edge),
                        ]:
                            conf = _confidence_from_edge(abs(edge_val) / 10.0)
                            bet = SpreadBet(
                                game_id=game_id, game_date=game_date, matchup=matchup,
                                team=team, spread_line=float(spread_line),
                                predicted_margin=float(pred_margin), league=league,
                                confidence=conf,
                                reasoning=f"Model margin: {pred_margin:+.1f} vs spread {spread_line:+.1f}",
                            )
                            bet.is_clear_pick = conf.is_clear()
                            bet.model_name = "pipeline_ensemble"
                            bets.append(bet)
                except (ValueError, TypeError):
                    pass

            # ── Moneyline bets ───────────────────────────────────────
            # Derive win probability from predicted spread if available
            win_prob = row.get("win_probability", None)
            if win_prob is None and predicted_spread is not None:
                try:
                    import math
                    ps = float(predicted_spread)
                    # Sigmoid: spread of 0 → 50%, spread of +10 → ~90%
                    win_prob = 1.0 / (1.0 + math.exp(-abs(ps) * 0.08))
                    if ps < 0:
                        win_prob = 1.0 - win_prob
                except (ValueError, TypeError):
                    win_prob = None

            if win_prob is not None and 0 < win_prob < 1:
                for team, prob in [(home, win_prob), (away, 1.0 - win_prob)]:
                    implied = 0.5  # Default if no ML odds available
                    for col in ["home_ml_odds", "away_ml_odds"]:
                        if col in row:
                            ml_odds = row[col]
                            if ml_odds and ml_odds != -110:
                                implied = 1.0 / (1.0 + 100.0 / abs(ml_odds)) if ml_odds < 0 else 1.0 / (1.0 + ml_odds / 100.0)
                                break
                    edge = prob - implied
                    if abs(edge) > 0.01:
                        conf = _confidence_from_edge(abs(edge))
                        bet = MoneylineBet(
                            game_id=game_id, game_date=game_date, matchup=matchup,
                            team=team, win_probability=prob,
                            market_implied_prob=implied, league=league,
                            confidence=conf,
                            reasoning=f"Win prob: {prob:.1%} vs implied {implied:.1%}",
                        )
                        bet.is_clear_pick = conf.is_clear() and abs(edge) >= 0.03
                        bet.model_name = "pipeline_ensemble"
                        bets.append(bet)

            # ── Team Total bets (derived from predicted total) ───────
            if predicted_total and market_total:
                # Rough split: home teams score ~51% of total
                home_share = 0.51
                home_pred = predicted_total * home_share
                away_pred = predicted_total * (1 - home_share)
                home_mkt = market_total * home_share
                away_mkt = market_total * (1 - home_share)

                for team, pred_tt, mkt_tt in [
                    (home, home_pred, home_mkt),
                    (away, away_pred, away_mkt),
                ]:
                    edge_tt = (pred_tt - mkt_tt) / max(mkt_tt, 1)
                    if abs(edge_tt) > 0.01:
                        side_tt = "OVER" if edge_tt > 0 else "UNDER"
                        bet = TeamTotalBet(
                            game_id=game_id, game_date=game_date, matchup=matchup,
                            team=team, side=side_tt,
                            market_team_total=round(mkt_tt, 1),
                            predicted_team_total=round(pred_tt, 1),
                            league=league,
                            confidence=_confidence_from_edge(abs(edge_tt)),
                            reasoning=f"Derived from total: pred {pred_tt:.0f} vs mkt {mkt_tt:.0f}",
                        )
                        bet.model_name = "pipeline_ensemble"
                        bets.append(bet)

            # ── Quarter/Half Total bets (real projector) ────────────
            predicted_total = row.get("predicted_total", 0)
            proj_result = None
            if predicted_total:
                try:
                    proj = _get_quarter_projector()
                    proj_result = proj.project(float(predicted_total), home, away)
                    if proj_result and "q1_total" in proj_result:
                        # Quarter totals (Q1-Q4)
                        for q in range(1, 5):
                            q_key = f"q{q}_total"
                            pred_q = proj_result[q_key]
                            mkt_q = proj.get_quarter_market(float(predicted_total), q)
                            if pred_q and mkt_q and mkt_q > 0:
                                q_edge = (pred_q - mkt_q) / mkt_q
                                if abs(q_edge) > 0.005:
                                    side_q = "OVER" if q_edge > 0 else "UNDER"
                                    conf = _confidence_from_edge(abs(q_edge))
                                    bet = QuarterTotalBet(
                                        game_id=game_id, game_date=game_date, matchup=matchup,
                                        side=side_q, market_quarter_total=mkt_q,
                                        predicted_quarter_total=pred_q, quarter=q,
                                        league=league, confidence=conf,
                                        reasoning=f"Projected Q{q} total: {pred_q:.0f} (mkt: {mkt_q:.0f})",
                                    )
                                    bet.model_name = "pipeline_ensemble_with_projector"
                                    bets.append(bet)

                        # Half totals (1H and 2H) — inline to avoid HalfTotalBet param mismatch
                        for h in [1, 2]:
                            h_key = f"h{h}_total"
                            pred_h = proj_result[h_key]
                            mkt_h = proj.get_half_market(float(predicted_total), h)
                            if pred_h and mkt_h and mkt_h > 0:
                                h_edge = (pred_h - mkt_h) / mkt_h
                                if abs(h_edge) > 0.005:
                                    side_h = "OVER" if h_edge > 0 else "UNDER"
                                    conf = _confidence_from_edge(abs(h_edge))
                                    # Build HalfTotalBet inline to support both halves
                                    import math
                                    diff = abs(pred_h - mkt_h)
                                    win_prob = 1.0 / (1.0 + math.exp(-0.02 * diff)) if diff > 0 else 0.5
                                    win_prob = max(0.01, min(0.92, win_prob))
                                    edge_hw = win_prob - 0.5
                                    ev_h = (win_prob * 0.91) - ((1 - win_prob) * 1.0)
                                    ordinal = {1: "1st", 2: "2nd"}.get(h, f"{h}th")
                                    bt = BetType.FIRST_HALF_TOTAL if h == 1 else BetType.SECOND_HALF_TOTAL
                                    bet = BetSuggestion(
                                        game_id=game_id, game_date=game_date, matchup=matchup,
                                        league=league, bet_type=bt,
                                        bet_side=f"{ordinal} Half {side_h} {mkt_h:.0f}",
                                        market_line=mkt_h, predicted_value=pred_h,
                                        predicted_label=f"Pred: {pred_h:.0f}",
                                        edge_pct=edge_hw, expected_value=ev_h,
                                        win_probability=win_prob,
                                        confidence=conf,
                                        reasoning=f"Projected {'1st' if h == 1 else '2nd'} H total: {pred_h:.0f} (mkt: {mkt_h:.0f})",
                                        model_name="pipeline_ensemble_with_projector",
                                    )
                                    bets.append(bet)
                except Exception as e:
                    logger.debug(f"Quarter/half projection failed: {e}")

            # ── Quarter Winner bets (using real projected scores) ──
            if proj_result and "q1_home" in proj_result:
                try:
                    import math
                    for q in [1, 2, 3, 4]:
                        home_q = proj_result.get(f"q{q}_home", 0)
                        away_q = proj_result.get(f"q{q}_away", 0)
                        if home_q > 0 or away_q > 0:
                            home_margin = home_q - away_q
                            q_prob = 1.0 / (1.0 + math.exp(-home_margin * 0.08)) if home_margin != 0 else 0.5
                            q_prob = max(0.02, min(0.98, q_prob))
                            for team, qp in [(home, q_prob), (away, 1.0 - q_prob)]:
                                q_edge = qp - 0.5
                                if abs(q_edge) > 0.02:
                                    bet = QuarterBet(
                                        game_id=game_id, game_date=game_date, matchup=matchup,
                                        quarter=q, team=team, win_probability=qp,
                                        league=league,
                                        confidence=_confidence_from_edge(abs(q_edge)),
                                        reasoning=f"Projected Q{q} scores: {home_q:.0f}-{away_q:.0f} → {qp:.1%} win prob",
                                    )
                                    bet.model_name = "pipeline_ensemble_with_projector"
                                    bets.append(bet)
                except Exception as e:
                    logger.debug(f"Quarter winner projection failed: {e}")

        logger.info(f"RecommendationEngine: generated {len(bets)} bets")
        return bets
