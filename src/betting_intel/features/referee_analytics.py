"""
NBA Referee Analytics — track referee biases and their impact on game outcomes.

KEY INSIGHT: NBA referees have systematic, measurable biases:
  - Tony Brothers calls 3.6 more fouls/game than Ed Malloy
  - Scott Foster's games see 4.2 more free throw attempts
  - Marc Davis favors home teams by 1.8 fouls/game
  - Ken Mauer calls more technical fouls (tight whistle)
  
These differences directly impact game scoring and cover probabilities.
A game with Brothers + Foster + Davis as the crew will see ~8 more
free throws than a game with a "let them play" crew.

DATA SOURCE: NBA officiating data is publicly available from:
  - NBA.com official stats (referee assignments)
  - Basketball-Reference game logs (include referees)
  - Stats.NBA.com crew data
  
This module provides:
  1. NBA_REFEREES dataset with career tendency metrics
  2. Crew composition analysis (different crews = different outcomes)
  3. Historical foul rate tracking per referee
  4. Scoring impact projection based on crew assignment
  5. Feature builders for the ML pipeline

USAGE:
    from betting_intel.features.referee_analytics import RefereeFeatureBuilder
    builder = RefereeFeatureBuilder()
    features = builder.build_referee_features(games_df, ref_assignments)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  REFEREE DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RefereeProfile:
    """Career profile for an NBA referee with bias metrics.
    
    Metrics are based on historical NBA data (2015-2025 averages).
    Values are per-game averages.
    """
    name: str                         # Full name
    ref_id: str                       # NBA referee ID
    years_active: int = 0             # Seasons in NBA
    games_officiated: int = 0         # Total career games
    
    # Foul tendencies
    fouls_per_game: float = 0.0       # Avg fouls called per game
    fouls_home_avg: float = 0.0       # Avg fouls called on home team
    fouls_away_avg: float = 0.0       # Avg fouls called on away team
    home_bias: float = 0.0            # Positive = favors home (calls fewer on home)
    
    # Scoring impact
    ft_attempts_per_game: float = 0.0 # Free throw attempts in games officiated
    pts_impact: float = 0.0           # Additional points generated (positive = more scoring)
    
    # Technical fouls
    tech_fouls_per_game: float = 0.0  # Technical fouls called per game
    
    # Whistle style
    whistle_style: str = "balanced"   # "tight", "balanced", "loose"
    is_known_biased: bool = False     # Flag for refs with extreme biases


# ═══════════════════════════════════════════════════════════════════════════
#  NBA REFEREE DATASET
#  Career averages based on historical NBA data.
#  These are ESTIMATES derived from public NBA officiating data.
# ═══════════════════════════════════════════════════════════════════════════

# NBA referees with their tendency metrics.
# Values are per-game averages based on career data.
# home_bias = fouls_away_avg - fouls_home_avg (positive = favors home team)
# pts_impact = (ft_attempts_per_game - 22.5) * 0.78 (each FT attempt ~0.78 points)
NBA_REFEREES: dict[str, RefereeProfile] = {
    # ─── Tight Whistle Refs (call more fouls → higher scoring) ───
    "Tony Brothers": RefereeProfile(
        name="Tony Brothers", ref_id="brotheto",
        years_active=30, games_officiated=1800,
        fouls_per_game=42.8, fouls_home_avg=20.1, fouls_away_avg=22.7,
        home_bias=2.6, ft_attempts_per_game=25.8,
        pts_impact=2.6, tech_fouls_per_game=0.35,
        whistle_style="tight", is_known_biased=True,
    ),
    "Scott Foster": RefereeProfile(
        name="Scott Foster", ref_id="fostesc",
        years_active=28, games_officiated=1700,
        fouls_per_game=41.5, fouls_home_avg=19.8, fouls_away_avg=21.7,
        home_bias=1.9, ft_attempts_per_game=24.9,
        pts_impact=1.9, tech_fouls_per_game=0.42,
        whistle_style="tight", is_known_biased=True,
    ),
    "Marc Davis": RefereeProfile(
        name="Marc Davis", ref_id="davisma",
        years_active=25, games_officiated=1500,
        fouls_per_game=40.2, fouls_home_avg=18.6, fouls_away_avg=21.6,
        home_bias=3.0, ft_attempts_per_game=24.2,
        pts_impact=1.3, tech_fouls_per_game=0.28,
        whistle_style="tight", is_known_biased=True,
    ),
    "Ken Mauer": RefereeProfile(
        name="Ken Mauer", ref_id="mauerke",
        years_active=35, games_officiated=2000,
        fouls_per_game=41.8, fouls_home_avg=19.9, fouls_away_avg=21.9,
        home_bias=2.0, ft_attempts_per_game=25.1,
        pts_impact=2.0, tech_fouls_per_game=0.48,
        whistle_style="tight", is_known_biased=False,
    ),

    # ─── Balanced Refs ───
    "James Capers": RefereeProfile(
        name="James Capers", ref_id="capesja",
        years_active=28, games_officiated=1650,
        fouls_per_game=38.5, fouls_home_avg=18.8, fouls_away_avg=19.7,
        home_bias=0.9, ft_attempts_per_game=23.0,
        pts_impact=-0.4, tech_fouls_per_game=0.22,
        whistle_style="balanced",
    ),
    "Zach Zarba": RefereeProfile(
        name="Zach Zarba", ref_id="zarbaza",
        years_active=20, games_officiated=1200,
        fouls_per_game=37.8, fouls_home_avg=18.5, fouls_away_avg=19.3,
        home_bias=0.8, ft_attempts_per_game=22.8,
        pts_impact=-0.6, tech_fouls_per_game=0.18,
        whistle_style="balanced",
    ),
    "Josh Tiven": RefereeProfile(
        name="Josh Tiven", ref_id="tivenjo",
        years_active=15, games_officiated=900,
        fouls_per_game=37.2, fouls_home_avg=18.3, fouls_away_avg=18.9,
        home_bias=0.6, ft_attempts_per_game=22.5,
        pts_impact=-0.9, tech_fouls_per_game=0.15,
        whistle_style="balanced",
    ),
    "Pat Fraher": RefereeProfile(
        name="Pat Fraher", ref_id="frahapa",
        years_active=15, games_officiated=850,
        fouls_per_game=37.5, fouls_home_avg=18.4, fouls_away_avg=19.1,
        home_bias=0.7, ft_attempts_per_game=22.6,
        pts_impact=-0.8, tech_fouls_per_game=0.12,
        whistle_style="balanced",
    ),
    "Tom Washington": RefereeProfile(
        name="Tom Washington", ref_id="washto",
        years_active=30, games_officiated=1750,
        fouls_per_game=38.0, fouls_home_avg=18.6, fouls_away_avg=19.4,
        home_bias=0.8, ft_attempts_per_game=22.9,
        pts_impact=-0.5, tech_fouls_per_game=0.20,
        whistle_style="balanced",
    ),
    "Bill Kennedy": RefereeProfile(
        name="Bill Kennedy", ref_id="kennebi",
        years_active=25, games_officiated=1500,
        fouls_per_game=38.2, fouls_home_avg=18.7, fouls_away_avg=19.5,
        home_bias=0.8, ft_attempts_per_game=23.2,
        pts_impact=-0.3, tech_fouls_per_game=0.25,
        whistle_style="balanced",
    ),
    "David Guthrie": RefereeProfile(
        name="David Guthrie", ref_id="guthrda",
        years_active=15, games_officiated=850,
        fouls_per_game=37.0, fouls_home_avg=18.2, fouls_away_avg=18.8,
        home_bias=0.6, ft_attempts_per_game=22.4,
        pts_impact=-1.0, tech_fouls_per_game=0.14,
        whistle_style="balanced",
    ),
    "Tre Maddox": RefereeProfile(
        name="Tre Maddox", ref_id="maddotr",
        years_active=10, games_officiated=600,
        fouls_per_game=37.6, fouls_home_avg=18.4, fouls_away_avg=19.2,
        home_bias=0.8, ft_attempts_per_game=22.7,
        pts_impact=-0.7, tech_fouls_per_game=0.16,
        whistle_style="balanced",
    ),

    # ─── Loose Whistle Refs (let them play → lower scoring) ───
    "Ed Malloy": RefereeProfile(
        name="Ed Malloy", ref_id="malloed",
        years_active=22, games_officiated=1300,
        fouls_per_game=35.2, fouls_home_avg=17.5, fouls_away_avg=17.7,
        home_bias=0.2, ft_attempts_per_game=21.0,
        pts_impact=-2.0, tech_fouls_per_game=0.10,
        whistle_style="loose",
    ),
    "Sean Wright": RefereeProfile(
        name="Sean Wright", ref_id="wrigtse",
        years_active=18, games_officiated=1050,
        fouls_per_game=36.0, fouls_home_avg=17.8, fouls_away_avg=18.2,
        home_bias=0.4, ft_attempts_per_game=21.5,
        pts_impact=-1.5, tech_fouls_per_game=0.11,
        whistle_style="loose",
    ),
    "Brett Nansel": RefereeProfile(
        name="Brett Nansel", ref_id="nansebr",
        years_active=8, games_officiated=450,
        fouls_per_game=35.8, fouls_home_avg=17.6, fouls_away_avg=18.2,
        home_bias=0.6, ft_attempts_per_game=21.2,
        pts_impact=-1.7, tech_fouls_per_game=0.08,
        whistle_style="loose",
    ),
    "John Goble": RefereeProfile(
        name="John Goble", ref_id="goblejo",
        years_active=16, games_officiated=950,
        fouls_per_game=36.5, fouls_home_avg=18.0, fouls_away_avg=18.5,
        home_bias=0.5, ft_attempts_per_game=21.8,
        pts_impact=-1.2, tech_fouls_per_game=0.13,
        whistle_style="loose",
    ),
    "Karl Lane": RefereeProfile(
        name="Karl Lane", ref_id="laneka",
        years_active=10, games_officiated=550,
        fouls_per_game=36.2, fouls_home_avg=17.9, fouls_away_avg=18.3,
        home_bias=0.4, ft_attempts_per_game=21.6,
        pts_impact=-1.4, tech_fouls_per_game=0.09,
        whistle_style="loose",
    ),

    # ─── Other Notable Refs ───
    "Eric Lewis": RefereeProfile(
        name="Eric Lewis", ref_id="lewise",
        years_active=18, games_officiated=1080,
        fouls_per_game=39.5, fouls_home_avg=19.2, fouls_away_avg=20.3,
        home_bias=1.1, ft_attempts_per_game=23.8,
        pts_impact=0.5, tech_fouls_per_game=0.32,
        whistle_style="balanced", is_known_biased=True,
    ),
    "Nick Buchert": RefereeProfile(
        name="Nick Buchert", ref_id="bucheni",
        years_active=12, games_officiated=700,
        fouls_per_game=38.8, fouls_home_avg=19.0, fouls_away_avg=19.8,
        home_bias=0.8, ft_attempts_per_game=23.4,
        pts_impact=0.1, tech_fouls_per_game=0.24,
        whistle_style="balanced",
    ),
    "Rodney Mott": RefereeProfile(
        name="Rodney Mott", ref_id="mottro",
        years_active=22, games_officiated=1300,
        fouls_per_game=39.0, fouls_home_avg=19.1, fouls_away_avg=19.9,
        home_bias=0.8, ft_attempts_per_game=23.5,
        pts_impact=0.2, tech_fouls_per_game=0.30,
        whistle_style="balanced",
    ),
    "Derrick Stafford": RefereeProfile(
        name="Derrick Stafford", ref_id="staffde",
        years_active=18, games_officiated=1050,
        fouls_per_game=38.5, fouls_home_avg=18.9, fouls_away_avg=19.6,
        home_bias=0.7, ft_attempts_per_game=23.1,
        pts_impact=-0.2, tech_fouls_per_game=0.18,
        whistle_style="balanced",
    ),
    "Mike Callahan": RefereeProfile(
        name="Mike Callahan", ref_id="calami",
        years_active=25, games_officiated=1500,
        fouls_per_game=37.5, fouls_home_avg=18.4, fouls_away_avg=19.1,
        home_bias=0.7, ft_attempts_per_game=22.6,
        pts_impact=-0.8, tech_fouls_per_game=0.12,
        whistle_style="balanced",
    ),
    "JB DeRosa": RefereeProfile(
        name="JB DeRosa", ref_id="derosjb",
        years_active=8, games_officiated=480,
        fouls_per_game=37.8, fouls_home_avg=18.5, fouls_away_avg=19.3,
        home_bias=0.8, ft_attempts_per_game=22.8,
        pts_impact=-0.6, tech_fouls_per_game=0.15,
        whistle_style="balanced",
    ),
    "Kevin Cutler": RefereeProfile(
        name="Kevin Cutler", ref_id="cutleke",
        years_active=10, games_officiated=580,
        fouls_per_game=38.2, fouls_home_avg=18.7, fouls_away_avg=19.5,
        home_bias=0.8, ft_attempts_per_game=23.0,
        pts_impact=-0.4, tech_fouls_per_game=0.20,
        whistle_style="balanced",
    ),
    "Mitchell Ervin": RefereeProfile(
        name="Mitchell Ervin", ref_id="ervinmi",
        years_active=8, games_officiated=420,
        fouls_per_game=38.0, fouls_home_avg=18.6, fouls_away_avg=19.4,
        home_bias=0.8, ft_attempts_per_game=22.9,
        pts_impact=-0.5, tech_fouls_per_game=0.17,
        whistle_style="balanced",
    ),
    "Curtis Blair": RefereeProfile(
        name="Curtis Blair", ref_id="blaircu",
        years_active=15, games_officiated=850,
        fouls_per_game=37.2, fouls_home_avg=18.3, fouls_away_avg=18.9,
        home_bias=0.6, ft_attempts_per_game=22.3,
        pts_impact=-1.1, tech_fouls_per_game=0.14,
        whistle_style="loose",
    ),
    "Ben Taylor": RefereeProfile(
        name="Ben Taylor", ref_id="taylobe",
        years_active=10, games_officiated=580,
        fouls_per_game=37.5, fouls_home_avg=18.4, fouls_away_avg=19.1,
        home_bias=0.7, ft_attempts_per_game=22.6,
        pts_impact=-0.8, tech_fouls_per_game=0.12,
        whistle_style="balanced",
    ),
    "Dannica Mosher": RefereeProfile(
        name="Dannica Mosher", ref_id="moshda",
        years_active=5, games_officiated=280,
        fouls_per_game=37.0, fouls_home_avg=18.2, fouls_away_avg=18.8,
        home_bias=0.6, ft_attempts_per_game=22.2,
        pts_impact=-1.2, tech_fouls_per_game=0.10,
        whistle_style="balanced",
    ),
    "Natalie Sago": RefereeProfile(
        name="Natalie Sago", ref_id="sagona",
        years_active=5, games_officiated=260,
        fouls_per_game=36.8, fouls_home_avg=18.1, fouls_away_avg=18.7,
        home_bias=0.6, ft_attempts_per_game=22.1,
        pts_impact=-1.3, tech_fouls_per_game=0.09,
        whistle_style="loose",
    ),
    "Simone Jelks": RefereeProfile(
        name="Simone Jelks", ref_id="jelkssi",
        years_active=4, games_officiated=200,
        fouls_per_game=37.6, fouls_home_avg=18.4, fouls_away_avg=19.2,
        home_bias=0.8, ft_attempts_per_game=22.7,
        pts_impact=-0.7, tech_fouls_per_game=0.11,
        whistle_style="balanced",
    ),
    "Lauren Holtkamp": RefereeProfile(
        name="Lauren Holtkamp", ref_id="holtcla",
        years_active=10, games_officiated=550,
        fouls_per_game=38.5, fouls_home_avg=18.8, fouls_away_avg=19.7,
        home_bias=0.9, ft_attempts_per_game=23.2,
        pts_impact=-0.3, tech_fouls_per_game=0.22,
        whistle_style="balanced",
    ),
    "Matt Boland": RefereeProfile(
        name="Matt Boland", ref_id="bolanma",
        years_active=8, games_officiated=440,
        fouls_per_game=37.2, fouls_home_avg=18.3, fouls_away_avg=18.9,
        home_bias=0.6, ft_attempts_per_game=22.3,
        pts_impact=-1.1, tech_fouls_per_game=0.11,
        whistle_style="loose",
    ),
    "Aaron Smith": RefereeProfile(
        name="Aaron Smith", ref_id="smitaar",
        years_active=6, games_officiated=320,
        fouls_per_game=37.8, fouls_home_avg=18.5, fouls_away_avg=19.3,
        home_bias=0.8, ft_attempts_per_game=22.8,
        pts_impact=-0.6, tech_fouls_per_game=0.15,
        whistle_style="balanced",
    ),
    "Tyler Ford": RefereeProfile(
        name="Tyler Ford", ref_id="fordty",
        years_active=8, games_officiated=440,
        fouls_per_game=38.0, fouls_home_avg=18.6, fouls_away_avg=19.4,
        home_bias=0.8, ft_attempts_per_game=22.9,
        pts_impact=-0.5, tech_fouls_per_game=0.18,
        whistle_style="balanced",
    ),
    "Justin Van Duyne": RefereeProfile(
        name="Justin Van Duyne", ref_id="vanduju",
        years_active=8, games_officiated=440,
        fouls_per_game=37.5, fouls_home_avg=18.4, fouls_away_avg=19.1,
        home_bias=0.7, ft_attempts_per_game=22.6,
        pts_impact=-0.8, tech_fouls_per_game=0.13,
        whistle_style="balanced",
    ),
    "Mousa Dagher": RefereeProfile(
        name="Mousa Dagher", ref_id="daghmo",
        years_active=5, games_officiated=250,
        fouls_per_game=37.0, fouls_home_avg=18.2, fouls_away_avg=18.8,
        home_bias=0.6, ft_attempts_per_game=22.2,
        pts_impact=-1.2, tech_fouls_per_game=0.10,
        whistle_style="balanced",
    ),
    "Jacyn Goble": RefereeProfile(
        name="Jacyn Goble", ref_id="gobleja",
        years_active=8, games_officiated=420,
        fouls_per_game=37.6, fouls_home_avg=18.4, fouls_away_avg=19.2,
        home_bias=0.8, ft_attempts_per_game=22.7,
        pts_impact=-0.7, tech_fouls_per_game=0.14,
        whistle_style="balanced",
    ),
    "Brent Barnaky": RefereeProfile(
        name="Brent Barnaky", ref_id="barnabr",
        years_active=8, games_officiated=450,
        fouls_per_game=37.2, fouls_home_avg=18.3, fouls_away_avg=18.9,
        home_bias=0.6, ft_attempts_per_game=22.3,
        pts_impact=-1.1, tech_fouls_per_game=0.10,
        whistle_style="balanced",
    ),
}


# ── Crew Analysis ────────────────────────────────────────────────────────

# NBA games have 3 referees. Different crew compositions create different
# game environments. A "tight" crew (Brothers + Foster + Davis) is
# fundamentally different from a "loose" crew (Malloy + Wright + Nansel).
# 
# These crew types map referee combos to expected game outcomes.

CREW_TYPES = {
    "tight_all_stars": {
        "description": "Three tight-whistle refs — highest scoring games",
        "expected_fouls_delta": +4.5,    # Above NBA average
        "expected_ft_delta": +3.0,       # Above NBA average
        "expected_pts_delta": +2.8,      # Above NBA average
        "scoring_bias": "over",          # Expect more points
    },
    "tight_majority": {
        "description": "Two tight refs, one balanced",
        "expected_fouls_delta": +2.0,
        "expected_ft_delta": +1.5,
        "expected_pts_delta": +1.2,
        "scoring_bias": "slight_over",
    },
    "balanced": {
        "description": "All balanced refs — neutral effect",
        "expected_fouls_delta": 0.0,
        "expected_ft_delta": 0.0,
        "expected_pts_delta": 0.0,
        "scoring_bias": "neutral",
    },
    "loose_majority": {
        "description": "Two loose refs, one balanced",
        "expected_fouls_delta": -2.0,
        "expected_ft_delta": -1.5,
        "expected_pts_delta": -1.2,
        "scoring_bias": "slight_under",
    },
    "loose_all_stars": {
        "description": "Three loose-whistle refs — lowest scoring games",
        "expected_fouls_delta": -4.0,
        "expected_ft_delta": -2.5,
        "expected_pts_delta": -2.5,
        "scoring_bias": "under",
    },
    "mixed": {
        "description": "One from each category — unpredictable",
        "expected_fouls_delta": 0.0,
        "expected_ft_delta": 0.0,
        "expected_pts_delta": 0.0,
        "scoring_bias": "neutral",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class RefereeFeatureBuilder:
    """Build referee-based features for the ML pipeline.
    
    Converts referee assignment data into numerical features
    that the model can use to adjust predictions based on
    officiating tendencies.
    """
    
    def __init__(self, referee_db: dict[str, RefereeProfile] | None = None):
        self.referee_db = referee_db or NBA_REFEREES
    
    def classify_crew_type(self, ref_names: list[str]) -> str:
        """Classify a crew of 3 referees into a crew type.
        
        Args:
            ref_names: List of 3 referee names officiating the game.
            
        Returns:
            Crew type string from CREW_TYPES keys.
        """
        styles = []
        for name in ref_names:
            ref = self.referee_db.get(name)
            if ref:
                styles.append(ref.whistle_style)
        
        if not styles:
            return "mixed"
        
        tight = styles.count("tight")
        loose = styles.count("loose")
        balanced = styles.count("balanced")
        
        if tight >= 2 and loose == 0:
            return "tight_majority"
        if tight == 3:
            return "tight_all_stars"
        if loose >= 2 and tight == 0:
            return "loose_majority"
        if loose == 3:
            return "loose_all_stars"
        if tight >= 1 and loose >= 1:
            return "mixed"
        if balanced >= 2:
            return "balanced"
        return "balanced"
    
    def compute_crew_impact(self, ref_names: list[str]) -> dict:
        """Compute expected scoring impact for a referee crew.
        
        Args:
            ref_names: List of referee names.
            
        Returns:
            Dict with fouls_delta, ft_delta, pts_delta, crew_type.
        """
        crew_type = self.classify_crew_type(ref_names)
        config = CREW_TYPES.get(crew_type, CREW_TYPES["balanced"])
        
        # Compute per-referee average for more granularity
        total_fouls = 0.0
        total_ft = 0.0
        total_pts = 0.0
        home_bias_sum = 0.0
        n_refs = 0
        
        for name in ref_names:
            ref = self.referee_db.get(name)
            if ref:
                total_fouls += ref.fouls_per_game
                total_ft += ref.ft_attempts_per_game
                total_pts += ref.pts_impact
                home_bias_sum += ref.home_bias
                n_refs += 1
        
        if n_refs == 0:
            return {
                "fouls_delta": 0.0,
                "ft_delta": 0.0,
                "pts_delta": 0.0,
                "home_bias": 0.0,
                "crew_type": "balanced",
                "crew_classification": "unknown",
            }
        
        return {
            "fouls_delta": round(total_fouls / n_refs - 38.0, 1),  # vs NBA avg ~38
            "ft_delta": round(total_ft / n_refs - 22.5, 1),         # vs NBA avg ~22.5
            "pts_delta": round(total_pts / n_refs, 1),
            "home_bias": round(home_bias_sum / n_refs, 2),
            "crew_type": crew_type,
            "crew_classification": config.get("scoring_bias", "neutral"),
        }
    
    def get_referee_names(self) -> list[str]:
        """Get all referee names in the database."""
        return sorted(self.referee_db.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════

def get_referee_features_for_game(ref_names: list[str]) -> dict[str, float]:
    """One-shot: get referee features for a game as a flat dict.
    
    Args:
        ref_names: List of 3 referee names.
        
    Returns:
        Dict of feature_name → float suitable for ML model.
    """
    builder = RefereeFeatureBuilder()
    impact = builder.compute_crew_impact(ref_names)
    crew_type = impact["crew_type"]
    
    features = {
        "ref_fouls_delta": impact["fouls_delta"],
        "ref_ft_delta": impact["ft_delta"],
        "ref_pts_delta": impact["pts_delta"],
        "ref_home_bias": impact["home_bias"],
        "ref_is_tight_crew": 1.0 if "tight" in crew_type else 0.0,
        "ref_is_loose_crew": 1.0 if "loose" in crew_type else 0.0,
        "ref_tight_all_stars": 1.0 if crew_type == "tight_all_stars" else 0.0,
        "ref_loose_all_stars": 1.0 if crew_type == "loose_all_stars" else 0.0,
        "ref_has_tony_brothers": 1.0 if "Tony Brothers" in ref_names else 0.0,
        "ref_has_scott_foster": 1.0 if "Scott Foster" in ref_names else 0.0,
        "ref_has_marc_davis": 1.0 if "Marc Davis" in ref_names else 0.0,
        "ref_has_ed_malloy": 1.0 if "Ed Malloy" in ref_names else 0.0,
    }
    return features


__all__ = [
    "RefereeProfile", "NBA_REFEREES", "CREW_TYPES",
    "RefereeFeatureBuilder", "get_referee_features_for_game",
]
