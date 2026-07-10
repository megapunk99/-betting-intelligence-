"""
Sport Configurations — Multi-league configuration for the LivePredictionEngine.

Currently supported sports:
  - NBA:       30 teams, Oct-Jun, 1,230 games/season
  - NCAAB:     ~100+ teams with betting lines, Nov-Apr, ~5,000 games/season
  - Euroleague: 20 teams, Sep-May, ~300 games/season

Strategy:
  Basketball leagues share the same feature pipeline (FeatureEngineer) and
  prediction models (MarketInefficiencySystem + TotalsRegressor) because they
  all produce the same box-score stats. Only team name maps, season dates,
  and total ranges differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ── Sport Config Data Class ───────────────────────────────────────────────


@dataclass
class SportConfig:
    """Configuration for the NBA prediction engine."""

    # TheOddsAPI sport key (e.g. "basketball_nba")
    sport_key: str

    # Display names
    display_name: str  # e.g. "NBA" (short badge text)
    full_name: str  # e.g. "National Basketball Association"

    # Market types this sport supports
    has_h2h: bool = True  # Moneyline (head-to-head)
    has_spreads: bool = False  # Point spreads
    has_totals: bool = False  # Over/under totals

    # Team name mapping: TheOddsAPI full name → short display name
    team_name_map: dict[str, str] = field(default_factory=dict)

    # Prediction strategy
    # "total" = over/under prediction (NBA)
    # "moneyline" = win probability
    prediction_strategy: str = "total"

    # Total range (NBA: 180-260)
    total_min: float = 180.0
    total_max: float = 260.0

    # Season months (Oct-Jun)
    season_start_month: int = 10
    season_end_month: int = 6

    # Markets to fetch from TheOddsAPI
    markets_to_fetch: list[str] = field(
        default_factory=lambda: ["h2h", "spreads", "totals"]
    )

    @property
    def is_in_season(self) -> bool:
        """Check if the sport is currently in season (Oct-Jun, spans year boundary)."""
        now = datetime.now()
        month = now.month
        if self.season_start_month <= self.season_end_month:
            return self.season_start_month <= month <= self.season_end_month
        return month >= self.season_start_month or month <= self.season_end_month

    def get_short_name(self, full_name: str) -> str:
        """Convert a TheOddsAPI full name to a short display name."""
        if not full_name:
            return ""
        short = self.team_name_map.get(full_name)
        if short:
            return short
        parts = full_name.split()
        if len(parts) > 1:
            return parts[-1]
        return full_name


# ── NBA Definition ────────────────────────────────────────────────────────

# NBA — Full season (Oct-Jun), 1,230 games, 30 teams.
# Team names use ESPN API displayName convention (shared by TheOddsAPI).
NBA = SportConfig(
    sport_key="basketball_nba",
    display_name="NBA",
    full_name="National Basketball Association",
    has_h2h=True,
    has_spreads=True,
    has_totals=True,
    team_name_map={
        "Atlanta Hawks": "Hawks",
        "Boston Celtics": "Celtics",
        "Brooklyn Nets": "Nets",
        "Charlotte Hornets": "Hornets",
        "Chicago Bulls": "Bulls",
        "Cleveland Cavaliers": "Cavaliers",
        "Dallas Mavericks": "Mavericks",
        "Denver Nuggets": "Nuggets",
        "Detroit Pistons": "Pistons",
        "Golden State Warriors": "Warriors",
        "Houston Rockets": "Rockets",
        "Indiana Pacers": "Pacers",
        "LA Clippers": "Clippers",
        "Los Angeles Lakers": "Lakers",
        "Memphis Grizzlies": "Grizzlies",
        "Miami Heat": "Heat",
        "Milwaukee Bucks": "Bucks",
        "Minnesota Timberwolves": "Timberwolves",
        "New Orleans Pelicans": "Pelicans",
        "New York Knicks": "Knicks",
        "Oklahoma City Thunder": "Thunder",
        "Orlando Magic": "Magic",
        "Philadelphia 76ers": "76ers",
        "Phoenix Suns": "Suns",
        "Portland Trail Blazers": "Trail Blazers",
        "Sacramento Kings": "Kings",
        "San Antonio Spurs": "Spurs",
        "Toronto Raptors": "Raptors",
        "Utah Jazz": "Jazz",
        "Washington Wizards": "Wizards",
    },
    prediction_strategy="total",
    total_min=180.0,
    total_max=260.0,
    season_start_month=10,
    season_end_month=6,
    markets_to_fetch=["h2h", "spreads", "totals"],
)


# ── NCAAB Definition ──────────────────────────────────────────────────────

# NCAAB — NCAA Division I men's college basketball.
# Season: Nov-Apr (~5,000 games). Lower scoring (avg ~145 total).
# ~100+ teams with active betting lines on TheOddsAPI.
# Higher home court advantage (~4 pts) than NBA (~2.3 pts).
NCAAB = SportConfig(
    sport_key="basketball_ncaab",
    display_name="NCAAB",
    full_name="NCAA Men's Division I Basketball",
    has_h2h=True,
    has_spreads=True,
    has_totals=True,
    team_name_map={
        # ACC
        "Duke Blue Devils": "Duke",
        "North Carolina Tar Heels": "UNC",
        "Virginia Cavaliers": "Virginia",
        "NC State Wolfpack": "NC State",
        "Clemson Tigers": "Clemson",
        "Miami Hurricanes": "Miami",
        "Florida State Seminoles": "Florida State",
        "Virginia Tech Hokies": "Virginia Tech",
        "Louisville Cardinals": "Louisville",
        "Syracuse Orange": "Syracuse",
        "Wake Forest Demon Deacons": "Wake Forest",
        "Georgia Tech Yellow Jackets": "Georgia Tech",
        "Pittsburgh Panthers": "Pittsburgh",
        "Boston College Eagles": "Boston College",
        "Notre Dame Fighting Irish": "Notre Dame",
        "California Golden Bears": "California",
        "SMU Mustangs": "SMU",
        "Stanford Cardinal": "Stanford",
        # Big Ten
        "Michigan State Spartans": "Michigan State",
        "Michigan Wolverines": "Michigan",
        "Purdue Boilermakers": "Purdue",
        "Indiana Hoosiers": "Indiana",
        "Illinois Fighting Illini": "Illinois",
        "Ohio State Buckeyes": "Ohio State",
        "Wisconsin Badgers": "Wisconsin",
        "Iowa Hawkeyes": "Iowa",
        "Maryland Terrapins": "Maryland",
        "Rutgers Scarlet Knights": "Rutgers",
        "Penn State Nittany Lions": "Penn State",
        "Minnesota Golden Gophers": "Minnesota",
        "Nebraska Cornhuskers": "Nebraska",
        "Northwestern Wildcats": "Northwestern",
        "UCLA Bruins": "UCLA",
        "USC Trojans": "USC",
        "Washington Huskies": "Washington",
        "Oregon Ducks": "Oregon",
        "Oregon State Beavers": "Oregon State",
        # SEC
        "Kentucky Wildcats": "Kentucky",
        "Tennessee Volunteers": "Tennessee",
        "Alabama Crimson Tide": "Alabama",
        "Auburn Tigers": "Auburn",
        "Florida Gators": "Florida",
        "Arkansas Razorbacks": "Arkansas",
        "LSU Tigers": "LSU",
        "Texas A&M Aggies": "Texas A&M",
        "Mississippi State Bulldogs": "Mississippi State",
        "Missouri Tigers": "Missouri",
        "South Carolina Gamecocks": "South Carolina",
        "Ole Miss Rebels": "Ole Miss",
        "Georgia Bulldogs": "Georgia",
        "Vanderbilt Commodores": "Vanderbilt",
        "Oklahoma Sooners": "Oklahoma",
        "Texas Longhorns": "Texas",
        # Big 12
        "Kansas Jayhawks": "Kansas",
        "Baylor Bears": "Baylor",
        "Houston Cougars": "Houston",
        "Texas Tech Red Raiders": "Texas Tech",
        "Iowa State Cyclones": "Iowa State",
        "TCU Horned Frogs": "TCU",
        "West Virginia Mountaineers": "West Virginia",
        "Kansas State Wildcats": "Kansas State",
        "Oklahoma State Cowboys": "Oklahoma State",
        "BYU Cougars": "BYU",
        "Cincinnati Bearcats": "Cincinnati",
        "UCF Knights": "UCF",
        "Arizona Wildcats": "Arizona",
        "Arizona State Sun Devils": "Arizona State",
        "Colorado Buffaloes": "Colorado",
        "Utah Utes": "Utah",
        # Big East
        "UConn Huskies": "UConn",
        "Marquette Golden Eagles": "Marquette",
        "Villanova Wildcats": "Villanova",
        "Creighton Bluejays": "Creighton",
        "Xavier Musketeers": "Xavier",
        "Providence Friars": "Providence",
        "St. John's Red Storm": "St. John's",
        "Butler Bulldogs": "Butler",
        "Seton Hall Pirates": "Seton Hall",
        "DePaul Blue Demons": "DePaul",
        "Georgetown Hoyas": "Georgetown",
        # West Coast
        "Gonzaga Bulldogs": "Gonzaga",
        "Saint Mary's Gaels": "Saint Mary's",
        "San Francisco Dons": "San Francisco",
        "Santa Clara Broncos": "Santa Clara",
        "Loyola Marymount Lions": "LMU",
        "Pepperdine Waves": "Pepperdine",
        # Mountain West
        "San Diego State Aztecs": "San Diego State",
        "Utah State Aggies": "Utah State",
        "Boise State Broncos": "Boise State",
        "Nevada Wolf Pack": "Nevada",
        "Colorado State Rams": "Colorado State",
        "UNLV Rebels": "UNLV",
        "New Mexico Lobos": "New Mexico",
        # American
        "Memphis Tigers": "Memphis",
        "FAU Owls": "FAU",
        "North Texas Mean Green": "North Texas",
        "Tulane Green Wave": "Tulane",
        "Wichita State Shockers": "Wichita State",
        "Charlotte 49ers": "Charlotte",
        # Atlantic 10
        "Dayton Flyers": "Dayton",
        "VCU Rams": "VCU",
        "St. Louis Billikens": "Saint Louis",
        "Loyola Chicago Ramblers": "Loyola Chicago",
        "Duquesne Dukes": "Duquesne",
        "Rhode Island Rams": "Rhode Island",
        "Davidson Wildcats": "Davidson",
        "George Mason Patriots": "George Mason",
        "Richmond Spiders": "Richmond",
        "Massachusetts Minutemen": "UMass",
        # Mid-major notables
        "Princeton Tigers": "Princeton",
        "Yale Bulldogs": "Yale",
        "Grand Canyon Antelopes": "Grand Canyon",
        "Drake Bulldogs": "Drake",
        "Indiana State Sycamores": "Indiana State",
        "McNeese Cowboys": "McNeese",
        "College of Charleston Cougars": "Charleston",
        "UNC Wilmington Seahawks": "UNC Wilmington",
        "James Madison Dukes": "James Madison",
        "Louisiana Ragin' Cajuns": "Louisiana",
        "Southern Miss Golden Eagles": "Southern Miss",
        "Western Kentucky Hilltoppers": "WKU",
        "Middle Tennessee Blue Raiders": "Middle Tennessee",
        "Liberty Flames": "Liberty",
        "Oakland Golden Grizzlies": "Oakland",
        "Akron Zips": "Akron",
        "Toledo Rockets": "Toledo",
        "Ohio Bobcats": "Ohio",
        "Colgate Raiders": "Colgate",
        "Saint Joseph's Hawks": "Saint Joseph's",
        "Samford Bulldogs": "Samford",
        "Furman Paladins": "Furman",
        "App State Mountaineers": "App State",
        "Louisiana Tech Bulldogs": "Louisiana Tech",
        "Texas State Bobcats": "Texas State",
        "South Alabama Jaguars": "South Alabama",
        "Utah Valley Wolverines": "Utah Valley",
        "Seattle Redhawks": "Seattle",
        "Navy Midshipmen": "Navy",
        "Army Black Knights": "Army",
        "Boston University Terriers": "Boston University",
    },
    prediction_strategy="total",
    total_min=110.0,
    total_max=190.0,
    season_start_month=11,
    season_end_month=4,
    markets_to_fetch=["h2h", "spreads", "totals"],
)


# ── Euroleague Definition ─────────────────────────────────────────────────

# Euroleague — Top European basketball competition.
# 20 teams, Sep-May (~300 games). Lower scoring than NBA (avg ~160 total).
# Higher home court advantage (~4 pts) due to hostile European arenas.
EUROLEAGUE = SportConfig(
    sport_key="basketball_euroleague",
    display_name="Euroleague",
    full_name="EuroLeague Basketball",
    has_h2h=True,
    has_spreads=True,
    has_totals=True,
    team_name_map={
        "Anadolu Efes Istanbul": "Anadolu Efes",
        "AS Monaco": "Monaco",
        "FC Barcelona": "Barcelona",
        "FC Bayern Munich": "Bayern Munich",
        "Crvena Zvezda Meridianbet Belgrade": "Crvena Zvezda",
        "Dubai Basketball": "Dubai",
        "EA7 Emporio Armani Milan": "Milan",
        "Fenerbahçe Beko": "Fenerbahçe",
        "Hapoel IBI Tel Aviv": "Hapoel Tel Aviv",
        "Kosner Baskonia Vitoria-Gasteiz": "Baskonia",
        "LDLC ASVEL": "ASVEL",
        "Maccabi Rapyd Tel Aviv": "Maccabi Tel Aviv",
        "Olympiacos Piraeus": "Olympiacos",
        "Panathinaikos AKTOR": "Panathinaikos",
        "Paris Basketball": "Paris",
        "Partizan Mozzart Bet Belgrade": "Partizan",
        "Real Madrid": "Real Madrid",
        "Valencia Basket": "Valencia",
        "Virtus Segafredo Bologna": "Virtus Bologna",
        "Žalgiris Kaunas": "Žalgiris",
    },
    prediction_strategy="total",
    total_min=150.0,
    total_max=180.0,
    season_start_month=9,
    season_end_month=5,
    markets_to_fetch=["h2h", "spreads", "totals"],
)


# ── EPL Definition ──────────────────────────────────────────────────────────

# EPL — English Premier League.
# 20 teams, Aug-May (380 games/season).
# Low scoring (~2.5 total goals). Home field advantage is ~0.38 expected goals.
# Prediction model: ELO-based with Poisson goals distribution.
# Market structure: 3-way h2h (home/draw/away) + over/under goals.
EPL = SportConfig(
    sport_key="soccer_epl",
    display_name="EPL",
    full_name="English Premier League",
    has_h2h=True,
    has_spreads=False,  # Asian handicaps exist but are separate
    has_totals=True,  # Over/under goals
    team_name_map={
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Bournemouth": "Bournemouth",
        "Brentford": "Brentford",
        "Brighton & Hove Albion": "Brighton",
        "Chelsea": "Chelsea",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Ipswich Town": "Ipswich",
        "Leicester City": "Leicester",
        "Liverpool": "Liverpool",
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Southampton": "Southampton",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        # Alternate name formats TheOddsAPI might use
        "Manchester City FC": "Man City",
        "Manchester United FC": "Man United",
        "Liverpool FC": "Liverpool",
        "Chelsea FC": "Chelsea",
        "Arsenal FC": "Arsenal",
        "Tottenham Hotspur FC": "Tottenham",
        "Newcastle United FC": "Newcastle",
        "West Ham United FC": "West Ham",
        "Aston Villa FC": "Aston Villa",
        "Everton FC": "Everton",
        "Wolverhampton Wanderers FC": "Wolves",
    },
    prediction_strategy="moneyline",  # Soccer: predict win/draw/loss
    total_min=0.5,
    total_max=5.5,
    season_start_month=8,
    season_end_month=5,
    markets_to_fetch=["h2h", "totals"],  # No spreads for EPL
)


# ── NFL Definition ─────────────────────────────────────────────────────────

# NFL — National Football League.
# 32 teams, Sep-Feb (~272 games per season).
# Low scoring (~45 total points). Home field advantage is lower (~1.75 pts).
# Prediction model: total points (over/under) using ESPN historical data.
NFL = SportConfig(
    sport_key="americanfootball_nfl",
    display_name="NFL",
    full_name="National Football League",
    has_h2h=True,
    has_spreads=True,
    has_totals=True,
    team_name_map={
        # AFC East
        "Buffalo Bills": "Bills",
        "Miami Dolphins": "Dolphins",
        "New England Patriots": "Patriots",
        "New York Jets": "Jets",
        # AFC North
        "Baltimore Ravens": "Ravens",
        "Cincinnati Bengals": "Bengals",
        "Cleveland Browns": "Browns",
        "Pittsburgh Steelers": "Steelers",
        # AFC South
        "Houston Texans": "Texans",
        "Indianapolis Colts": "Colts",
        "Jacksonville Jaguars": "Jaguars",
        "Tennessee Titans": "Titans",
        # AFC West
        "Denver Broncos": "Broncos",
        "Kansas City Chiefs": "Chiefs",
        "Las Vegas Raiders": "Raiders",
        "Los Angeles Chargers": "Chargers",
        # NFC East
        "Dallas Cowboys": "Cowboys",
        "New York Giants": "Giants",
        "Philadelphia Eagles": "Eagles",
        "Washington Commanders": "Commanders",
        # NFC North
        "Chicago Bears": "Bears",
        "Detroit Lions": "Lions",
        "Green Bay Packers": "Packers",
        "Minnesota Vikings": "Vikings",
        # NFC South
        "Atlanta Falcons": "Falcons",
        "Carolina Panthers": "Panthers",
        "New Orleans Saints": "Saints",
        "Tampa Bay Buccaneers": "Buccaneers",
        # NFC West
        "Arizona Cardinals": "Cardinals",
        "Los Angeles Rams": "Rams",
        "San Francisco 49ers": "49ers",
        "Seattle Seahawks": "Seahawks",
    },
    prediction_strategy="total",
    total_min=30.0,
    total_max=60.0,
    season_start_month=9,
    season_end_month=2,
    markets_to_fetch=["h2h", "spreads", "totals"],
)


# ── Master Lists ──────────────────────────────────────────────────────────

# All supported sports — NBA + NCAAB + Euroleague + EPL + NFL
ALL_SPORTS: list[SportConfig] = [NBA, NCAAB, EUROLEAGUE, EPL, NFL]


# Sports that are currently in season
def get_active_sports() -> list[SportConfig]:
    return [s for s in ALL_SPORTS if s.is_in_season]


# TheOddsAPI sport_key → SportConfig lookup
SPORT_KEY_TO_CONFIG: dict[str, SportConfig] = {s.sport_key: s for s in ALL_SPORTS}

# Display name → SportConfig lookup (for UI filtering)
DISPLAY_NAME_TO_CONFIG: dict[str, SportConfig] = {s.display_name: s for s in ALL_SPORTS}

# Combined team name map for all sports — single source of truth for ESPN/API lookups.
# Builds from each SportConfig's team_name_map so additions to a config
# are automatically reflected in ESPN and future_predictor modules.
ALL_TEAM_NAME_MAP: dict[str, str] = {}
for _sport in ALL_SPORTS:
    ALL_TEAM_NAME_MAP.update(_sport.team_name_map)


# ── Helper Functions ──────────────────────────────────────────────────────


def league_from_sport_key(sport_key: str) -> str:
    """Convert a TheOddsAPI sport key to a short league display name."""
    config = SPORT_KEY_TO_CONFIG.get(sport_key)
    if config:
        return config.display_name
    parts = sport_key.split("_")
    if len(parts) >= 2:
        return parts[-1].upper()
    return sport_key.upper()


def sport_key_to_group(sport_key: str) -> str:
    """Group sports by category for UI filtering. NBA/NCAAB → Basketball."""
    if "basketball" in sport_key:
        return "Basketball"
    if "football" in sport_key or "nfl" in sport_key:
        return "Football"
    if "hockey" in sport_key or "nhl" in sport_key:
        return "Hockey"
    if "baseball" in sport_key or "mlb" in sport_key:
        return "Baseball"
    if "soccer" in sport_key:
        return "Soccer"
    if "tennis" in sport_key:
        return "Tennis"
    return "Other"
