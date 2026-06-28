"""
Arbitrage Detector — finds guaranteed-profit opportunities across sportsbooks.

Types of arbitrage detected:
  1. Standard 2-way (h2h): Back Team A at Book X, Back Team B at Book Y
  2. Three-way (soccer): Back Home, Draw, Away across 3 books
  3. Totals: Back Over at Book X, Back Under at Book Y

All detection operates on raw TheOddsAPI dicts (the same input as OddsParser).
No network calls — purely computational analysis of existing odds data.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from betting_intel.arbitrage.models import ArbLeg, ArbitrageOpportunity

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    elif american_odds < 0:
        return 1.0 + 100.0 / abs(american_odds)
    return 1.0


def american_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability (0-1)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    return 0.5


def _short_name(full_name: str) -> str:
    """Extract a short name from a full team name, or use full name as-is."""
    if not full_name:
        return ""
    parts = full_name.split()
    if len(parts) > 1:
        return parts[-1]
    return full_name


def _pct(val: float) -> str:
    """Format as percentage string."""
    return f"{val * 100:.2f}%"


def _compute_stakes(
    implied_probs: list[float],
    total_stake: float = 1000.0,
) -> list[float]:
    """Compute optimal stakes for each leg given implied probabilities.

    Args:
        implied_probs: 1/decimal_odds for each leg
        total_stake: Total dollar amount to stake

    Returns:
        Dollar stakes for each leg that guarantee equal payout.
    """
    total_implied = sum(implied_probs)
    if total_implied >= 1.0:
        return [0.0] * len(implied_probs)
    return [total_stake * (p / total_implied) for p in implied_probs]


# ── Main Detection ─────────────────────────────────────────────────────────

# Minimum profit % to report (avoid micro-arbs that aren't worth executing)
_MIN_PROFIT_PCT = 0.005  # 0.5%


def detect_arbitrage(raw_odds: list[dict]) -> list[ArbitrageOpportunity]:
    """Detect arbitrage opportunities across all sportsbooks.

    Args:
        raw_odds: List of raw TheOddsAPI event dicts (with bookmakers).

    Returns:
        List of ArbitrageOpportunity objects, sorted by profit_pct descending.
    """
    if not raw_odds:
        return []

    opportunities: list[ArbitrageOpportunity] = []
    seen_arb_ids: set[str] = set()

    for event in raw_odds:
        if not isinstance(event, dict):
            continue

        home_full = event.get("home_team", "") or ""
        away_full = event.get("away_team", "") or ""
        if not home_full or not away_full:
            continue

        sport_key = event.get("_sport_config_key", event.get("sport_key", "basketball_nba")) or ""
        game_id = event.get("id", "") or ""
        commence_time = event.get("commence_time", "") or ""
        game_date = commence_time[:10] if commence_time else ""
        league = _league_from_sport_key(sport_key)

        home_short = _short_name(home_full)
        away_short = _short_name(away_full)
        matchup = f"{away_short} @ {home_short}"

        bookmakers = event.get("bookmakers", [])
        if not isinstance(bookmakers, list) or len(bookmakers) < 2:
            # Need at least 2 books for any arbitrage
            continue

        # ── Extract per-book key prices ──────────────────────────────
        # h2h: {book_key: {home_price, away_price, draw_price}}
        # totals: {book_key: {over_price, under_price, point}}

        h2h_prices: dict[str, dict[str, Optional[int]]] = {}
        draw_prices: dict[str, Optional[int]] = {}
        total_prices: dict[str, dict[str, Optional[int]]] = {}
        total_points: dict[str, Optional[float]] = {}

        for book in bookmakers:
            if not isinstance(book, dict):
                continue
            book_key = book.get("key", "unknown")
            markets = book.get("markets", [])
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])
                if not isinstance(outcomes, list):
                    continue

                if market_key == "h2h":
                    home_price: Optional[int] = None
                    away_price: Optional[int] = None
                    draw_price: Optional[int] = None
                    for o in outcomes:
                        if not isinstance(o, dict):
                            continue
                        name = o.get("name", "")
                        price = o.get("price")
                        if price is not None:
                            try:
                                price_int = int(price)
                            except (ValueError, TypeError):
                                continue
                            if name == home_full:
                                home_price = price_int
                            elif name == away_full:
                                away_price = price_int
                            elif name and name.lower() == "draw":
                                draw_price = price_int
                    if home_price is not None and away_price is not None:
                        h2h_prices[book_key] = {
                            "home": home_price,
                            "away": away_price,
                        }
                        if draw_price is not None:
                            draw_prices[book_key] = draw_price

                elif market_key == "totals":
                    over_price: Optional[int] = None
                    under_price: Optional[int] = None
                    point: Optional[float] = None
                    for o in outcomes:
                        if not isinstance(o, dict):
                            continue
                        name = o.get("name", "")
                        price = o.get("price")
                        pt = o.get("point")
                        if pt is not None:
                            try:
                                point = float(pt)
                            except (ValueError, TypeError):
                                continue
                        if price is not None:
                            try:
                                price_int = int(price)
                            except (ValueError, TypeError):
                                continue
                            if name == "Over":
                                over_price = price_int
                            elif name == "Under":
                                under_price = price_int
                    if over_price is not None and under_price is not None and point is not None:
                        total_prices[book_key] = {
                            "over": over_price,
                            "under": under_price,
                        }
                        total_points[book_key] = point

        # ── 1. Standard 2-way arbitrage ──────────────────────────────
        if len(h2h_prices) >= 2:
            # Find best home price and best away price across all books
            best_home_book = ""
            best_home_price = 0
            best_away_book = ""
            best_away_price = 0

            for bk, prices in h2h_prices.items():
                hp = prices.get("home", 0) or 0
                ap = prices.get("away", 0) or 0
                if hp > best_home_price:
                    best_home_price = hp
                    best_home_book = bk
                if ap > best_away_price:
                    best_away_price = ap
                    best_away_book = bk

            if best_home_book and best_away_book and best_home_book != best_away_book:
                home_dec = american_to_decimal(best_home_price)
                away_dec = american_to_decimal(best_away_price)
                home_imp = 1.0 / home_dec
                away_imp = 1.0 / away_dec
                total_imp = home_imp + away_imp

                if total_imp < 1.0:
                    arb_id = f"h2h_{game_id}"
                    if arb_id not in seen_arb_ids:
                        seen_arb_ids.add(arb_id)
                        profit_pct = (1.0 / total_imp) - 1.0

                        if profit_pct >= _MIN_PROFIT_PCT:
                            stakes = _compute_stakes([home_imp, away_imp])
                            legs = [
                                ArbLeg(
                                    bookmaker=best_home_book,
                                    team=home_full,
                                    market="h2h",
                                    point=None,
                                    price=best_home_price,
                                    decimal_odds=round(home_dec, 2),
                                    stake_pct=round(stakes[0] / sum(stakes), 4) if sum(stakes) > 0 else 0.5,
                                    stake_dollars=round(stakes[0], 2),
                                ),
                                ArbLeg(
                                    bookmaker=best_away_book,
                                    team=away_full,
                                    market="h2h",
                                    point=None,
                                    price=best_away_price,
                                    decimal_odds=round(away_dec, 2),
                                    stake_pct=round(stakes[1] / sum(stakes), 4) if sum(stakes) > 0 else 0.5,
                                    stake_dollars=round(stakes[1], 2),
                                ),
                            ]

                            opportunities.append(ArbitrageOpportunity(
                                id=arb_id,
                                game_id=game_id,
                                matchup=matchup,
                                sport_key=sport_key,
                                league=league,
                                commence_time=commence_time,
                                game_date=game_date,
                                arb_type="standard_2way",
                                legs=legs,
                                total_implied_prob=round(total_imp, 4),
                                profit_pct=round(profit_pct, 4),
                                profit_per_1k=round(profit_pct * 1000, 2),
                                n_books=2,
                                depth=2,
                            ))

        # ── 2. Three-way arbitrage (soccer) ──────────────────────────
        if len(draw_prices) >= 1 and len(h2h_prices) >= 2:
            # Need at least 2 books for home/away + 1 book with draw
            best_home_book = ""
            best_home_price = 0
            best_away_book = ""
            best_away_price = 0
            best_draw_book = ""
            best_draw_price = 0

            for bk, prices in h2h_prices.items():
                hp = prices.get("home", 0) or 0
                ap = prices.get("away", 0) or 0
                if hp > best_home_price:
                    best_home_price = hp
                    best_home_book = bk
                if ap > best_away_price:
                    best_away_price = ap
                    best_away_book = bk

            for bk, dp in draw_prices.items():
                if (dp or 0) > best_draw_price:
                    best_draw_price = dp or 0
                    best_draw_book = bk

            # 3-way arb requires at least 2 distinct books
            distinct_books = len({best_home_book, best_away_book, best_draw_book})
            if distinct_books >= 2 and all([best_home_price, best_away_price, best_draw_price]):
                home_dec = american_to_decimal(best_home_price)
                away_dec = american_to_decimal(best_away_price)
                draw_dec = american_to_decimal(best_draw_price)
                home_imp = 1.0 / home_dec
                away_imp = 1.0 / away_dec
                draw_imp = 1.0 / draw_dec
                total_imp = home_imp + away_imp + draw_imp

                if total_imp < 1.0:
                    arb_id = f"threeway_{game_id}"
                    if arb_id not in seen_arb_ids:
                        seen_arb_ids.add(arb_id)
                        profit_pct = (1.0 / total_imp) - 1.0

                        if profit_pct >= _MIN_PROFIT_PCT:
                            stakes = _compute_stakes([home_imp, away_imp, draw_imp])
                            total_stake = sum(stakes) or 1.0
                            legs = [
                                ArbLeg(
                                    bookmaker=best_home_book,
                                    team=home_full,
                                    market="h2h",
                                    point=None,
                                    price=best_home_price,
                                    decimal_odds=round(home_dec, 2),
                                    stake_pct=round(stakes[0] / total_stake, 4),
                                    stake_dollars=round(stakes[0], 2),
                                ),
                                ArbLeg(
                                    bookmaker=best_away_book,
                                    team=away_full,
                                    market="h2h",
                                    point=None,
                                    price=best_away_price,
                                    decimal_odds=round(away_dec, 2),
                                    stake_pct=round(stakes[1] / total_stake, 4),
                                    stake_dollars=round(stakes[1], 2),
                                ),
                                ArbLeg(
                                    bookmaker=best_draw_book,
                                    team="Draw",
                                    market="h2h",
                                    point=None,
                                    price=best_draw_price,
                                    decimal_odds=round(draw_dec, 2),
                                    stake_pct=round(stakes[2] / total_stake, 4),
                                    stake_dollars=round(stakes[2], 2),
                                ),
                            ]

                            opportunities.append(ArbitrageOpportunity(
                                id=arb_id,
                                game_id=game_id,
                                matchup=matchup,
                                sport_key=sport_key,
                                league=league,
                                commence_time=commence_time,
                                game_date=game_date,
                                arb_type="three_way",
                                legs=legs,
                                total_implied_prob=round(total_imp, 4),
                                profit_pct=round(profit_pct, 4),
                                profit_per_1k=round(profit_pct * 1000, 2),
                                n_books=distinct_books,
                                depth=3,
                            ))

        # ── 3. Totals arbitrage (Over/Under) ─────────────────────────
        if len(total_prices) >= 2:
            # Find best Over and best Under across books (must be same/similar point)
            # Group by point first
            by_point: dict[float, list[tuple[str, int, int]]] = {}
            for bk, prices in total_prices.items():
                pt = total_points.get(bk)
                if pt is None:
                    continue
                over_pr = prices.get("over", 0) or 0
                under_pr = prices.get("under", 0) or 0
                by_point.setdefault(pt, []).append((bk, over_pr, under_pr))

            for pt, books_at_point in by_point.items():
                if len(books_at_point) < 2:
                    continue

                best_over_book = ""
                best_over_price = 0
                best_under_book = ""
                best_under_price = 0

                for bk, ov, un in books_at_point:
                    if ov > best_over_price:
                        best_over_price = ov
                        best_over_book = bk
                    if un > best_under_price:
                        best_under_price = un
                        best_under_book = bk

                if best_over_book and best_under_book and best_over_book != best_under_book:
                    over_dec = american_to_decimal(best_over_price)
                    under_dec = american_to_decimal(best_under_price)
                    over_imp = 1.0 / over_dec
                    under_imp = 1.0 / under_dec
                    total_imp = over_imp + under_imp

                    if total_imp < 1.0:
                        arb_id = f"total_{game_id}_{pt}"
                        if arb_id not in seen_arb_ids:
                            seen_arb_ids.add(arb_id)
                            profit_pct = (1.0 / total_imp) - 1.0

                            if profit_pct >= _MIN_PROFIT_PCT:
                                stakes = _compute_stakes([over_imp, under_imp])
                                total_stake = sum(stakes) or 1.0
                                legs = [
                                    ArbLeg(
                                        bookmaker=best_over_book,
                                        team=f"Over {pt}",
                                        market="total",
                                        point=pt,
                                        price=best_over_price,
                                        decimal_odds=round(over_dec, 2),
                                        stake_pct=round(stakes[0] / total_stake, 4),
                                        stake_dollars=round(stakes[0], 2),
                                    ),
                                    ArbLeg(
                                        bookmaker=best_under_book,
                                        team=f"Under {pt}",
                                        market="total",
                                        point=pt,
                                        price=best_under_price,
                                        decimal_odds=round(under_dec, 2),
                                        stake_pct=round(stakes[1] / total_stake, 4),
                                        stake_dollars=round(stakes[1], 2),
                                    ),
                                ]

                                opportunities.append(ArbitrageOpportunity(
                                    id=arb_id,
                                    game_id=game_id,
                                    matchup=matchup,
                                    sport_key=sport_key,
                                    league=league,
                                    commence_time=commence_time,
                                    game_date=game_date,
                                    arb_type="totals",
                                    legs=legs,
                                    total_implied_prob=round(total_imp, 4),
                                    profit_pct=round(profit_pct, 4),
                                    profit_per_1k=round(profit_pct * 1000, 2),
                                    n_books=2,
                                    depth=2,
                                ))

    # Sort by profit descending
    opportunities.sort(key=lambda o: o.profit_pct, reverse=True)
    logger.info(f"Arbitrage detection: {len(opportunities)} opportunities found across {len(raw_odds)} events")
    return opportunities


def _league_from_sport_key(sport_key: str) -> str:
    """Convert sport key to league display name."""
    from betting_intel.live.sport_configs import SPORT_KEY_TO_CONFIG
    config = SPORT_KEY_TO_CONFIG.get(sport_key)
    if config:
        return config.display_name
    parts = sport_key.split("_")
    if len(parts) >= 2:
        return parts[-1].upper()
    return sport_key.upper()
