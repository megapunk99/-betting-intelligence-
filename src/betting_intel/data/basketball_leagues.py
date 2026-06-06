"""
Basketball League Configuration — defines all supported basketball leagues.

Each league has:
  - TheOddsAPI sport key (for fetching real odds)
  - ESPN API sport key (for fetching historical data)
  - Typical stats (avg total, pace, home-court advantage)
  - Model training configuration

Usage:
    from betting_intel.data.basketball_leagues import BASKETBALL_LEAGUES, BasketballLeague

    for league in BASKETBALL_LEAGUES:
        print(league.name, league.odds_sport_key, league.avg_total)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BasketballLeague:
    """Configuration for a single basketball league."""

    key: str                           # Internal key (e.g. "nba", "wnba")
    name: str                          # Display name
    country: str                       # Primary country
    tier: str                          # "First division", "College", etc.
    num_teams: int                     # Number of teams in the league

    # ── API Keys ───────────────────────────────────────────────────
    odds_sport_key: str                # TheOddsAPI sport key
    espn_sport_key: str = ""           # ESPN API sport key (empty = not available)
    has_espn_api: bool = False         # True if ESPN API has historical data

    # ── Typical Stats (used for baselines when models aren't trained) ─
    avg_total: float = 224.0           # Average total points per game
    avg_home_pts: float = 114.0        # Average home team points
    avg_away_pts: float = 110.0        # Average away team points
    avg_pace: float = 100.0            # Average pace (possessions per game)
    home_win_pct: float = 0.58         # Typical home-court win rate
    avg_margin: float = 3.5            # Average home-court scoring margin

    # ── Model Training ─────────────────────────────────────────────
    train_model: bool = True           # Whether to train ML models for this league
    min_games_for_model: int = 50      # Minimum historical games needed to train
    feature_version: str = "v3"        # Feature engineering version

    # ── Season Info ────────────────────────────────────────────────
    season_months: str = ""            # e.g. "October → June"
    season_format: str = ""            # e.g. "82 games + playoffs"
    is_active_now: bool = True         # Whether the league is currently in season

    # ── Market Notes ───────────────────────────────────────────────
    market_notes: str = ""             # Betting-specific notes
    softness_factor: float = 1.0       # How soft the market is (1.0 = NBA, >1.0 = softer)


# ═══════════════════════════════════════════════════════════════════════
#  ALL BASKETBALL LEAGUES
# ═══════════════════════════════════════════════════════════════════════

NBA = BasketballLeague(
    key="nba",
    name="NBA (National Basketball Association)",
    country="USA",
    tier="First division (professional)",
    num_teams=30,
    odds_sport_key="basketball_nba",
    espn_sport_key="basketball/nba",
    has_espn_api=True,
    avg_total=224.0,
    avg_home_pts=114.0,
    avg_away_pts=110.0,
    avg_pace=99.0,
    home_win_pct=0.59,
    avg_margin=3.5,
    season_months="October → June",
    season_format="82 games + playoffs",
    is_active_now=True,
    market_notes="Most efficient market in basketball. Edges come from model sophistication + line movement.",
    softness_factor=1.0,
)

WNBA = BasketballLeague(
    key="wnba",
    name="WNBA (Women's National Basketball Association)",
    country="USA",
    tier="First division (professional)",
    num_teams=12,
    odds_sport_key="basketball_wnba",
    espn_sport_key="basketball/wnba",
    has_espn_api=True,
    avg_total=165.0,
    avg_home_pts=83.0,
    avg_away_pts=82.0,
    avg_pace=80.0,
    home_win_pct=0.57,
    avg_margin=2.5,
    season_months="May → September",
    season_format="40 games + playoffs",
    is_active_now=False,
    market_notes="Lower betting volume = softer lines. Short season amplifies momentum edges. Key player absences often mispriced.",
    softness_factor=1.5,
)

EUROLEAGUE = BasketballLeague(
    key="euroleague",
    name="EuroLeague",
    country="Europe (multi-national)",
    tier="First division (European)",
    num_teams=18,
    odds_sport_key="basketball_euroleague",
    avg_total=158.0,
    avg_home_pts=81.0,
    avg_away_pts=77.0,
    avg_pace=72.0,
    home_win_pct=0.64,
    avg_margin=5.5,
    season_months="October → May",
    season_format="34 games + playoffs + Final Four",
    is_active_now=False,
    market_notes="Strong home-court advantage (64%). Lower scoring than NBA. Travel fatigue across countries is a factor.",
    softness_factor=1.3,
)

NCAAB = BasketballLeague(
    key="ncaab",
    name="NCAAB (College Basketball)",
    country="USA",
    tier="College (NCAA Division I)",
    num_teams=364,
    odds_sport_key="basketball_ncaab",
    espn_sport_key="basketball/ncaab/men",
    has_espn_api=True,
    avg_total=142.0,
    avg_home_pts=74.0,
    avg_away_pts=68.0,
    avg_pace=70.0,
    home_win_pct=0.66,
    avg_margin=6.0,
    season_months="November → April",
    season_format="31+ games + conference tournaments + March Madness",
    is_active_now=False,
    market_notes="Huge number of teams (364) creates massive data sparsity. Strong home court. Tournament play amplifies variance.",
    softness_factor=2.0,
)

WNCAAB = BasketballLeague(
    key="wncaab",
    name="WNCAAB (Women's College Basketball)",
    country="USA",
    tier="College (NCAA Division I)",
    num_teams=360,
    odds_sport_key="basketball_wncaab",
    has_espn_api=False,
    avg_total=135.0,
    avg_home_pts=70.0,
    avg_away_pts=65.0,
    avg_pace=68.0,
    home_win_pct=0.65,
    avg_margin=5.5,
    season_months="November → April",
    season_format="29+ games + conference tournaments + March Madness",
    is_active_now=False,
    market_notes="Even less data coverage than men's college. Very soft lines. Tournament play is highly unpredictable.",
    softness_factor=2.5,
)

NBL_AUSTRALIA = BasketballLeague(
    key="nbl",
    name="NBL (National Basketball League, Australia)",
    country="Australia",
    tier="First division (professional)",
    num_teams=10,
    odds_sport_key="basketball_nbl",
    avg_total=174.0,
    avg_home_pts=88.0,
    avg_away_pts=86.0,
    avg_pace=85.0,
    home_win_pct=0.60,
    avg_margin=3.5,
    season_months="September → February",
    season_format="28 games + playoffs",
    is_active_now=False,
    market_notes="Australian league with growing betting volume. Time zone differences create market inefficiencies for US books.",
    softness_factor=1.4,
)

NBA_PRESEASON = BasketballLeague(
    key="nba_preseason",
    name="NBA Preseason",
    country="USA",
    tier="Exhibition",
    num_teams=30,
    odds_sport_key="basketball_nba_preseason",
    has_espn_api=False,
    avg_total=210.0,
    avg_home_pts=105.0,
    avg_away_pts=105.0,
    avg_pace=95.0,
    home_win_pct=0.55,
    avg_margin=2.0,
    season_months="October",
    season_format="4-5 exhibition games per team",
    is_active_now=False,
    train_model=False,  # Too few games for meaningful training
    market_notes="Very unpredictable. Star players often rest. Coaches experiment with lineups. Low betting volume = very soft lines.",
    softness_factor=3.0,
)

NBA_SUMMER_LEAGUE = BasketballLeague(
    key="nba_summer",
    name="NBA Summer League",
    country="USA",
    tier="Exhibition (rookies + G-League)",
    num_teams=30,
    odds_sport_key="basketball_nba_summer_league",
    has_espn_api=False,
    avg_total=175.0,
    avg_home_pts=87.0,
    avg_away_pts=88.0,
    avg_pace=92.0,
    home_win_pct=0.52,
    avg_margin=1.0,
    season_months="July",
    season_format="5-7 exhibition games per team",
    is_active_now=False,
    train_model=False,
    market_notes="Rookies and undrafted players. Very chaotic. Roster turnover from game to game. Bet with extreme caution.",
    softness_factor=4.0,
)

EUROLEAGUE_WOMEN = BasketballLeague(
    key="euroleague_women",
    name="EuroLeague Women",
    country="Europe (multi-national)",
    tier="First division (European women)",
    num_teams=16,
    odds_sport_key="",  # Not available on TheOddsAPI
    avg_total=140.0,
    avg_home_pts=72.0,
    avg_away_pts=68.0,
    avg_pace=70.0,
    home_win_pct=0.62,
    avg_margin=4.0,
    season_months="October → April",
    season_format="Group stage + playoffs",
    is_active_now=False,
    market_notes="Very low betting volume. Extremely soft lines. Limited historical data available.",
    softness_factor=3.0,
    train_model=False,
)

CEBL = BasketballLeague(
    key="cebl",
    name="Canadian Elite Basketball League",
    country="Canada",
    tier="First division (Canada)",
    num_teams=10,
    odds_sport_key="",  # Not available on TheOddsAPI
    avg_total=180.0,
    avg_home_pts=90.0,
    avg_away_pts=90.0,
    avg_pace=88.0,
    home_win_pct=0.57,
    avg_margin=3.0,
    season_months="May → August",
    season_format="20-24 games + playoffs",
    is_active_now=False,
    market_notes="Summer league with short season. Low data volume. Momentum and rest-day edges are amplified.",
    softness_factor=2.0,
    train_model=False,
)

BNXT = BasketballLeague(
    key="bnxt",
    name="BNXT League (Belgium/Netherlands)",
    country="Belgium / Netherlands",
    tier="First division (combined league)",
    num_teams=19,
    odds_sport_key="",  # Not available on TheOddsAPI
    avg_total=160.0,
    avg_home_pts=81.0,
    avg_away_pts=79.0,
    avg_pace=75.0,
    home_win_pct=0.60,
    avg_margin=3.0,
    season_months="September → May",
    season_format="National phase + BNXT phase + playoffs",
    is_active_now=False,
    market_notes="Cross-border league with complex scheduling. Low international betting attention. Home-court edges likely underpriced.",
    softness_factor=2.0,
    train_model=False,
)

LNB_PRO_B = BasketballLeague(
    key="lnb_pro_b",
    name="French LNB Pro B",
    country="France",
    tier="Second division",
    num_teams=18,
    odds_sport_key="",  # Not available on TheOddsAPI
    avg_total=155.0,
    avg_home_pts=79.0,
    avg_away_pts=76.0,
    avg_pace=73.0,
    home_win_pct=0.58,
    avg_margin=3.0,
    season_months="September → May",
    season_format="Regular season + playoffs",
    is_active_now=False,
    market_notes="French second division. Very low betting volume. Bookmakers rely on algorithms trained on top-tier leagues.",
    softness_factor=2.5,
    train_model=False,
)


# ═══════════════════════════════════════════════════════════════════════
#  COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════

ALL_BASKETBALL_LEAGUES: list[BasketballLeague] = [
    NBA,
    WNBA,
    EUROLEAGUE,
    NCAAB,
    WNCAAB,
    NBL_AUSTRALIA,
    NBA_PRESEASON,
    NBA_SUMMER_LEAGUE,
    EUROLEAGUE_WOMEN,
    CEBL,
    BNXT,
    LNB_PRO_B,
]

# Leagues with TheOddsAPI support (can fetch real market lines)
LEAGUES_WITH_ODDS: list[BasketballLeague] = [
    lg for lg in ALL_BASKETBALL_LEAGUES if lg.odds_sport_key
]

# Leagues with ESPN API support (can fetch historical game data)
LEAGUES_WITH_ESPN_API: list[BasketballLeague] = [
    lg for lg in ALL_BASKETBALL_LEAGUES if lg.has_espn_api
]

# Leagues we can train ML models for (have historical data + enough games)
LEAGUES_FOR_ML: list[BasketballLeague] = [
    lg for lg in ALL_BASKETBALL_LEAGUES if lg.train_model
]

# Primary tiers: leagues with the most betting volume and data
PRIMARY_BASKETBALL_LEAGUES: list[BasketballLeague] = [
    NBA, WNBA, EUROLEAGUE, NCAAB,
]

# Lookup by key
LEAGUE_BY_KEY: dict[str, BasketballLeague] = {
    lg.key: lg for lg in ALL_BASKETBALL_LEAGUES
}


def get_league(key: str) -> BasketballLeague:
    """Get a league by its key string."""
    if key not in LEAGUE_BY_KEY:
        valid = ", ".join(sorted(LEAGUE_BY_KEY.keys()))
        raise KeyError(f"Unknown league '{key}'. Valid keys: {valid}")
    return LEAGUE_BY_KEY[key]


def get_leagues_for_odds_sport(sport_key: str) -> list[BasketballLeague]:
    """Find all leagues that use a given TheOddsAPI sport key."""
    return [lg for lg in ALL_BASKETBALL_LEAGUES if lg.odds_sport_key == sport_key]


def is_league_in_season(league_key: str) -> bool:
    """Check if a league is currently in season based on its months."""
    league = get_league(league_key)
    return league.is_active_now


__all__ = [
    "BasketballLeague",
    "NBA", "WNBA", "EUROLEAGUE", "NCAAB", "WNCAAB",
    "NBL_AUSTRALIA", "NBA_PRESEASON", "NBA_SUMMER_LEAGUE",
    "EUROLEAGUE_WOMEN", "CEBL", "BNXT", "LNB_PRO_B",
    "ALL_BASKETBALL_LEAGUES",
    "LEAGUES_WITH_ODDS",
    "LEAGUES_WITH_ESPN_API",
    "LEAGUES_FOR_ML",
    "PRIMARY_BASKETBALL_LEAGUES",
    "LEAGUE_BY_KEY",
    "get_league",
    "get_leagues_for_odds_sport",
    "is_league_in_season",
]
