"""
ESPN Basketball Data Source — fetches historical game data for multiple basketball leagues.

Supports:
  - NBA (basketball/nba)
  - WNBA (basketball/wnba)
  - NCAAB (basketball/ncaab/men)

Uses ESPN's public API (no key required). Returns data in a format compatible
with the existing feature engineering pipeline.

Usage:
    from betting_intel.data.espn_hoops import ESPNLeagueSource

    source = ESPNLeagueSource()
    wnba_df = source.load_historical("wnba", seasons=[2024, 2025])
    ncaab_df = source.load_historical("ncaab", seasons=["2024-2025"])
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ESPN API base URLs
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/summary"

# Mapping from our league keys to ESPN sport paths
LEAGUE_TO_ESPN_PATH: dict[str, str] = {
    "nba": "basketball/nba",
    "wnba": "basketball/wnba",
    "ncaab": "basketball/mens-college-basketball",
    "wncaab": "basketball/womens-college-basketball",
}

# Mapping from ESPN team names to short names (for matching with DB)
ESPN_TEAM_TO_SHORT: dict[str, str] = {
    # NBA
    "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics", "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets", "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets", "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors", "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers", "Los Angeles Lakers": "Lakers", "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat", "Milwaukee Bucks": "Bucks", "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans", "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers", "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
    # WNBA
    "Atlanta Dream": "Dream", "Chicago Sky": "Sky", "Connecticut Sun": "Sun",
    "Dallas Wings": "Wings", "Indiana Fever": "Fever", "Las Vegas Aces": "Aces",
    "Los Angeles Sparks": "Sparks", "Minnesota Lynx": "Lynx", "New York Liberty": "Liberty",
    "Phoenix Mercury": "Mercury", "Seattle Storm": "Storm", "Washington Mystics": "Mystics",
    # NCAAB (362 teams from ESPN API)
    "Abilene Christian Wildcats": "Wildcats",
    "Air Force Falcons": "Falcons",
    "Akron Zips": "Zips",
    "Alabama A&M Bulldogs": "Bulldogs",
    "Alabama Crimson Tide": "Tide",
    "Alabama State Hornets": "Hornets",
    "Alcorn State Braves": "Braves",
    "American University Eagles": "Eagles",
    "App State Mountaineers": "Mountaineers",
    "Arizona State Sun Devils": "Devils",
    "Arizona Wildcats": "Wildcats",
    "Arkansas Razorbacks": "Razorbacks",
    "Arkansas State Red Wolves": "Wolves",
    "Arkansas-Pine Bluff Golden Lions": "Lions",
    "Army Black Knights": "Knights",
    "Auburn Tigers": "Tigers",
    "Austin Peay Governors": "Governors",
    "BYU Cougars": "Cougars",
    "Ball State Cardinals": "Cardinals",
    "Baylor Bears": "Bears",
    "Bellarmine Knights": "Knights",
    "Belmont Bruins": "Bruins",
    "Bethune-Cookman Wildcats": "Wildcats",
    "Binghamton Bearcats": "Bearcats",
    "Boise State Broncos": "Broncos",
    "Boston College Eagles": "Eagles",
    "Boston University Terriers": "Terriers",
    "Bowling Green Falcons": "Falcons",
    "Bradley Braves": "Braves",
    "Brown Bears": "Bears",
    "Bryant Bulldogs": "Bulldogs",
    "Bucknell Bison": "Bison",
    "Buffalo Bulls": "Bulls",
    "Butler Bulldogs": "Bulldogs",
    "Cal Poly Mustangs": "Mustangs",
    "Cal State Bakersfield Roadrunners": "Roadrunners",
    "Cal State Fullerton Titans": "Titans",
    "Cal State Northridge Matadors": "Matadors",
    "California Baptist Lancers": "Lancers",
    "California Golden Bears": "Bears",
    "Campbell Fighting Camels": "Camels",
    "Canisius Golden Griffins": "Griffins",
    "Central Arkansas Bears": "Bears",
    "Central Connecticut Blue Devils": "Devils",
    "Central Michigan Chippewas": "Chippewas",
    "Charleston Cougars": "Cougars",
    "Charleston Southern Buccaneers": "Buccaneers",
    "Charlotte 49ers": "49ers",
    "Chattanooga Mocs": "Mocs",
    "Chicago State Cougars": "Cougars",
    "Cincinnati Bearcats": "Bearcats",
    "Clemson Tigers": "Tigers",
    "Cleveland State Vikings": "Vikings",
    "Coastal Carolina Chanticleers": "Chanticleers",
    "Colgate Raiders": "Raiders",
    "Colorado Buffaloes": "Buffaloes",
    "Colorado State Rams": "Rams",
    "Columbia Lions": "Lions",
    "Coppin State Eagles": "Eagles",
    "Cornell Big Red": "Red",
    "Creighton Bluejays": "Bluejays",
    "Dartmouth Big Green": "Green",
    "Davidson Wildcats": "Wildcats",
    "Dayton Flyers": "Flyers",
    "DePaul Blue Demons": "Demons",
    "Delaware Blue Hens": "Hens",
    "Delaware State Hornets": "Hornets",
    "Denver Pioneers": "Pioneers",
    "Detroit Mercy Titans": "Titans",
    "Drake Bulldogs": "Bulldogs",
    "Drexel Dragons": "Dragons",
    "Duke Blue Devils": "Devils",
    "Duquesne Dukes": "Dukes",
    "East Carolina Pirates": "Pirates",
    "East Tennessee State Buccaneers": "Buccaneers",
    "East Texas A&M Lions": "Lions",
    "Eastern Illinois Panthers": "Panthers",
    "Eastern Kentucky Colonels": "Colonels",
    "Eastern Michigan Eagles": "Eagles",
    "Eastern Washington Eagles": "Eagles",
    "Elon Phoenix": "Phoenix",
    "Evansville Purple Aces": "Aces",
    "Fairfield Stags": "Stags",
    "Fairleigh Dickinson Knights": "Knights",
    "Florida A&M Rattlers": "Rattlers",
    "Florida Atlantic Owls": "Owls",
    "Florida Gators": "Gators",
    "Florida Gulf Coast Eagles": "Eagles",
    "Florida International Panthers": "Panthers",
    "Florida State Seminoles": "Seminoles",
    "Fordham Rams": "Rams",
    "Fresno State Bulldogs": "Bulldogs",
    "Furman Paladins": "Paladins",
    "Gardner-Webb Runnin' Bulldogs": "Bulldogs",
    "George Mason Patriots": "Patriots",
    "George Washington Revolutionaries": "Revolutionaries",
    "Georgetown Hoyas": "Hoyas",
    "Georgia Bulldogs": "Bulldogs",
    "Georgia Southern Eagles": "Eagles",
    "Georgia State Panthers": "Panthers",
    "Georgia Tech Yellow Jackets": "Jackets",
    "Gonzaga Bulldogs": "Bulldogs",
    "Grambling Tigers": "Tigers",
    "Grand Canyon Lopes": "Lopes",
    "Green Bay Phoenix": "Phoenix",
    "Hampton Pirates": "Pirates",
    "Harvard Crimson": "Crimson",
    "Hawai'i Rainbow Warriors": "Warriors",
    "High Point Panthers": "Panthers",
    "Hofstra Pride": "Pride",
    "Holy Cross Crusaders": "Crusaders",
    "Houston Christian Huskies": "Huskies",
    "Houston Cougars": "Cougars",
    "Howard Bison": "Bison",
    "IU Indianapolis Jaguars": "Jaguars",
    "Idaho State Bengals": "Bengals",
    "Idaho Vandals": "Vandals",
    "Illinois Fighting Illini": "Illini",
    "Illinois State Redbirds": "Redbirds",
    "Incarnate Word Cardinals": "Cardinals",
    "Indiana Hoosiers": "Hoosiers",
    "Indiana State Sycamores": "Sycamores",
    "Iona Gaels": "Gaels",
    "Iowa Hawkeyes": "Hawkeyes",
    "Iowa State Cyclones": "Cyclones",
    "Jackson State Tigers": "Tigers",
    "Jacksonville Dolphins": "Dolphins",
    "Jacksonville State Gamecocks": "Gamecocks",
    "James Madison Dukes": "Dukes",
    "Kansas City Roos": "Roos",
    "Kansas Jayhawks": "Jayhawks",
    "Kansas State Wildcats": "Wildcats",
    "Kennesaw State Owls": "Owls",
    "Kent State Golden Flashes": "Flashes",
    "Kentucky Wildcats": "Wildcats",
    "LSU Tigers": "Tigers",
    "La Salle Explorers": "Explorers",
    "Lafayette Leopards": "Leopards",
    "Lamar Cardinals": "Cardinals",
    "Le Moyne Dolphins": "Dolphins",
    "Lehigh Mountain Hawks": "Hawks",
    "Liberty Flames": "Flames",
    "Lipscomb Bisons": "Bisons",
    "Little Rock Trojans": "Trojans",
    "Long Beach State Beach": "Beach",
    "Long Island University Sharks": "Sharks",
    "Longwood Lancers": "Lancers",
    "Louisiana Ragin' Cajuns": "Cajuns",
    "Louisiana Tech Bulldogs": "Bulldogs",
    "Louisville Cardinals": "Cardinals",
    "Loyola Chicago Ramblers": "Ramblers",
    "Loyola Maryland Greyhounds": "Greyhounds",
    "Loyola Marymount Lions": "Lions",
    "Maine Black Bears": "Bears",
    "Manhattan Jaspers": "Jaspers",
    "Marist Red Foxes": "Foxes",
    "Marquette Golden Eagles": "Eagles",
    "Marshall Thundering Herd": "Herd",
    "Maryland Eastern Shore Hawks": "Hawks",
    "Maryland Terrapins": "Terrapins",
    "Massachusetts Minutemen": "Minutemen",
    "McNeese Cowboys": "Cowboys",
    "Memphis Tigers": "Tigers",
    "Mercer Bears": "Bears",
    "Mercyhurst Lakers": "Lakers",
    "Merrimack Warriors": "Warriors",
    "Miami (OH) RedHawks": "RedHawks",
    "Miami Hurricanes": "Hurricanes",
    "Michigan State Spartans": "Spartans",
    "Michigan Wolverines": "Wolverines",
    "Middle Tennessee Blue Raiders": "Raiders",
    "Milwaukee Panthers": "Panthers",
    "Minnesota Golden Gophers": "Gophers",
    "Mississippi State Bulldogs": "Bulldogs",
    "Mississippi Valley State Delta Devils": "Devils",
    "Missouri State Bears": "Bears",
    "Missouri Tigers": "Tigers",
    "Monmouth Hawks": "Hawks",
    "Montana Grizzlies": "Grizzlies",
    "Montana State Bobcats": "Bobcats",
    "Morehead State Eagles": "Eagles",
    "Morgan State Bears": "Bears",
    "Mount St. Mary's Mountaineers": "Mountaineers",
    "Murray State Racers": "Racers",
    "NC State Wolfpack": "Wolfpack",
    "NJIT Highlanders": "Highlanders",
    "Navy Midshipmen": "Midshipmen",
    "Nebraska Cornhuskers": "Cornhuskers",
    "Nevada Wolf Pack": "Pack",
    "New Hampshire Wildcats": "Wildcats",
    "New Haven Chargers": "Chargers",
    "New Mexico Lobos": "Lobos",
    "New Mexico State Aggies": "Aggies",
    "New Orleans Privateers": "Privateers",
    "Niagara Purple Eagles": "Eagles",
    "Nicholls Colonels": "Colonels",
    "Norfolk State Spartans": "Spartans",
    "North Alabama Lions": "Lions",
    "North Carolina A&T Aggies": "Aggies",
    "North Carolina Central Eagles": "Eagles",
    "North Carolina Tar Heels": "Heels",
    "North Dakota Fighting Hawks": "Hawks",
    "North Dakota State Bison": "Bison",
    "North Florida Ospreys": "Ospreys",
    "North Texas Mean Green": "Green",
    "Northeastern Huskies": "Huskies",
    "Northern Arizona Lumberjacks": "Lumberjacks",
    "Northern Colorado Bears": "Bears",
    "Northern Illinois Huskies": "Huskies",
    "Northern Iowa Panthers": "Panthers",
    "Northern Kentucky Norse": "Norse",
    "Northwestern State Demons": "Demons",
    "Northwestern Wildcats": "Wildcats",
    "Notre Dame Fighting Irish": "Irish",
    "Oakland Golden Grizzlies": "Grizzlies",
    "Ohio Bobcats": "Bobcats",
    "Ohio State Buckeyes": "Buckeyes",
    "Oklahoma Sooners": "Sooners",
    "Oklahoma State Cowboys": "Cowboys",
    "Old Dominion Monarchs": "Monarchs",
    "Ole Miss Rebels": "Rebels",
    "Omaha Mavericks": "Mavericks",
    "Oral Roberts Golden Eagles": "Eagles",
    "Oregon Ducks": "Ducks",
    "Oregon State Beavers": "Beavers",
    "Pacific Tigers": "Tigers",
    "Penn State Nittany Lions": "Lions",
    "Pennsylvania Quakers": "Quakers",
    "Pepperdine Waves": "Waves",
    "Pittsburgh Panthers": "Panthers",
    "Portland Pilots": "Pilots",
    "Portland State Vikings": "Vikings",
    "Prairie View A&M Panthers": "Panthers",
    "Presbyterian Blue Hose": "Hose",
    "Princeton Tigers": "Tigers",
    "Providence Friars": "Friars",
    "Purdue Boilermakers": "Boilermakers",
    "Purdue Fort Wayne Mastodons": "Mastodons",
    "Quinnipiac Bobcats": "Bobcats",
    "Radford Highlanders": "Highlanders",
    "Rhode Island Rams": "Rams",
    "Rice Owls": "Owls",
    "Richmond Spiders": "Spiders",
    "Rider Broncs": "Broncs",
    "Robert Morris Colonials": "Colonials",
    "Rutgers Scarlet Knights": "Knights",
    "SE Louisiana Lions": "Lions",
    "SIU Edwardsville Cougars": "Cougars",
    "SMU Mustangs": "Mustangs",
    "Sacramento State Hornets": "Hornets",
    "Sacred Heart Pioneers": "Pioneers",
    "Saint Francis Red Flash": "Flash",
    "Saint Joseph's Hawks": "Hawks",
    "Saint Louis Billikens": "Billikens",
    "Saint Mary's Gaels": "Gaels",
    "Saint Peter's Peacocks": "Peacocks",
    "Sam Houston Bearkats": "Bearkats",
    "Samford Bulldogs": "Bulldogs",
    "San Diego State Aztecs": "Aztecs",
    "San Diego Toreros": "Toreros",
    "San Francisco Dons": "Dons",
    "San Jose State Spartans": "Spartans",
    "Santa Clara Broncos": "Broncos",
    "Seattle U Redhawks": "Redhawks",
    "Seton Hall Pirates": "Pirates",
    "Siena Saints": "Saints",
    "South Alabama Jaguars": "Jaguars",
    "South Carolina Gamecocks": "Gamecocks",
    "South Carolina State Bulldogs": "Bulldogs",
    "South Carolina Upstate Spartans": "Spartans",
    "South Dakota Coyotes": "Coyotes",
    "South Dakota State Jackrabbits": "Jackrabbits",
    "South Florida Bulls": "Bulls",
    "Southeast Missouri State Redhawks": "Redhawks",
    "Southern Illinois Salukis": "Salukis",
    "Southern Jaguars": "Jaguars",
    "Southern Miss Golden Eagles": "Eagles",
    "Southern Utah Thunderbirds": "Thunderbirds",
    "St. Bonaventure Bonnies": "Bonnies",
    "St. John's Red Storm": "Storm",
    "St. Thomas-Minnesota Tommies": "Tommies",
    "Stanford Cardinal": "Cardinal",
    "Stephen F. Austin Lumberjacks": "Lumberjacks",
    "Stetson Hatters": "Hatters",
    "Stonehill Skyhawks": "Skyhawks",
    "Stony Brook Seawolves": "Seawolves",
    "Syracuse Orange": "Orange",
    "TCU Horned Frogs": "Frogs",
    "Tarleton State Texans": "Texans",
    "Temple Owls": "Owls",
    "Tennessee State Tigers": "Tigers",
    "Tennessee Tech Golden Eagles": "Eagles",
    "Tennessee Volunteers": "Volunteers",
    "Texas A&M Aggies": "Aggies",
    "Texas A&M-Corpus Christi Islanders": "Islanders",
    "Texas Longhorns": "Longhorns",
    "Texas Southern Tigers": "Tigers",
    "Texas State Bobcats": "Bobcats",
    "Texas Tech Red Raiders": "Raiders",
    "The Citadel Bulldogs": "Bulldogs",
    "Toledo Rockets": "Rockets",
    "Towson Tigers": "Tigers",
    "Troy Trojans": "Trojans",
    "Tulane Green Wave": "Wave",
    "Tulsa Golden Hurricane": "Hurricane",
    "UAB Blazers": "Blazers",
    "UAlbany Great Danes": "Danes",
    "UC Davis Aggies": "Aggies",
    "UC Irvine Anteaters": "Anteaters",
    "UC Riverside Highlanders": "Highlanders",
    "UC San Diego Tritons": "Tritons",
    "UC Santa Barbara Gauchos": "Gauchos",
    "UCF Knights": "Knights",
    "UCLA Bruins": "Bruins",
    "UConn Huskies": "Huskies",
    "UIC Flames": "Flames",
    "UL Monroe Warhawks": "Warhawks",
    "UMBC Retrievers": "Retrievers",
    "UMass Lowell River Hawks": "Hawks",
    "UNC Asheville Bulldogs": "Bulldogs",
    "UNC Greensboro Spartans": "Spartans",
    "UNC Wilmington Seahawks": "Seahawks",
    "UNLV Rebels": "Rebels",
    "USC Trojans": "Trojans",
    "UT Arlington Mavericks": "Mavericks",
    "UT Martin Skyhawks": "Skyhawks",
    "UT Rio Grande Valley Vaqueros": "Vaqueros",
    "UTEP Miners": "Miners",
    "UTSA Roadrunners": "Roadrunners",
    "Utah State Aggies": "Aggies",
    "Utah Tech Trailblazers": "Trailblazers",
    "Utah Utes": "Utes",
    "Utah Valley Wolverines": "Wolverines",
    "VCU Rams": "Rams",
    "VMI Keydets": "Keydets",
    "Valparaiso Beacons": "Beacons",
    "Vanderbilt Commodores": "Commodores",
    "Vermont Catamounts": "Catamounts",
    "Villanova Wildcats": "Wildcats",
    "Virginia Cavaliers": "Cavaliers",
    "Virginia Tech Hokies": "Hokies",
    "Wagner Seahawks": "Seahawks",
    "Wake Forest Demon Deacons": "Deacons",
    "Washington Huskies": "Huskies",
    "Washington State Cougars": "Cougars",
    "Weber State Wildcats": "Wildcats",
    "West Georgia Wolves": "Wolves",
    "West Virginia Mountaineers": "Mountaineers",
    "Western Carolina Catamounts": "Catamounts",
    "Western Illinois Leathernecks": "Leathernecks",
    "Western Kentucky Hilltoppers": "Hilltoppers",
    "Western Michigan Broncos": "Broncos",
    "Wichita State Shockers": "Shockers",
    "William & Mary Tribe": "Tribe",
    "Winthrop Eagles": "Eagles",
    "Wisconsin Badgers": "Badgers",
    "Wofford Terriers": "Terriers",
    "Wright State Raiders": "Raiders",
    "Wyoming Cowboys": "Cowboys",
    "Xavier Musketeers": "Musketeers",
    "Yale Bulldogs": "Bulldogs",
    "Youngstown State Penguins": "Penguins",
}

# Default season dates per league (used to find the right year to query)
LEAGUE_DEFAULT_MONTHS: dict[str, tuple[int, int]] = {
    "nba": (10, 6),      # Oct-Jun
    "wnba": (5, 9),      # May-Sep
    "ncaab": (11, 4),    # Nov-Apr
    "wncaab": (11, 4),   # Nov-Apr
}


def _team_short(name: str) -> str:
    """Convert an ESPN team name to a short name."""
    return ESPN_TEAM_TO_SHORT.get(name, name.split()[-1] if name else name)


class ESPNLeagueSource:
    """
    Fetch historical game data from ESPN's public API for supported leagues.

    The ESPN API provides scoreboard data with game results, team names,
    and scores. No API key is required.

    Usage:
        source = ESPNLeagueSource()
        df = source.load_historical("nba", seasons=[2024])
        upcoming = source.load_upcoming("nba", limit=10)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
        self._last_request = 0.0

    def _rate_limit(self):
        """Rate limit API requests (no key = be respectful)."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.time()

    def load_historical(
        self, league_key: str, seasons: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load historical game results for a league from ESPN.

        Args:
            league_key: 'nba', 'wnba', 'ncaab', or 'wncaab'
            seasons: List of season identifiers (years for pro, strings for college)

        Returns:
            DataFrame with columns: game_id, date, home_team, away_team,
            home_score, away_score, total_points, league, season
        """
        if league_key not in LEAGUE_TO_ESPN_PATH:
            logger.warning(f"League '{league_key}' not available on ESPN API")
            return pd.DataFrame()

        espn_path = LEAGUE_TO_ESPN_PATH[league_key]

        if seasons is None:
            # Default: current year and previous year
            current_year = datetime.now().year
            seasons = [current_year, current_year - 1]

        all_records = []
        for season in seasons:
            try:
                records = self._fetch_season(espn_path, league_key, season)
                all_records.extend(records)
                logger.info(f"ESPN {league_key} {season}: {len(records)} games")
            except Exception as e:
                logger.warning(f"ESPN {league_key} {season} failed: {e}")

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def load_upcoming(self, league_key: str, limit: int = 20) -> pd.DataFrame:
        """
        Load upcoming scheduled games from ESPN.

        Returns DataFrame with home_team, away_team, date, league columns.
        """
        if league_key not in LEAGUE_TO_ESPN_PATH:
            return pd.DataFrame()

        espn_path = LEAGUE_TO_ESPN_PATH[league_key]
        self._rate_limit()

        try:
            url = ESPN_SCOREBOARD_URL.format(sport=espn_path)
            params = {"dates": datetime.now().strftime("%Y%m%d"), "limit": 300}
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            records = []
            events = data.get("events", []) + data.get("upcoming", [])
            for event in events[:limit]:
                comp = self._get_competition(event)
                if not comp:
                    continue

                home, away = self._get_teams(comp)
                if not home or not away:
                    continue

                date_str = event.get("date", "")[:10]
                status = comp.get("status", {}).get("type", {}).get("name", "")
                is_completed = status in {"STATUS_FINAL", "STATUS_FINAL_ALT"}

                records.append({
                    "game_id": event.get("id", f"espn_{league_key}_{date_str}"),
                    "league": league_key,
                    "date": date_str,
                    "home_team": _team_short(home.get("displayName", "")),
                    "away_team": _team_short(away.get("displayName", "")),
                    "home_score": home.get("score", None) if is_completed else None,
                    "away_score": away.get("score", None) if is_completed else None,
                    "status": "completed" if is_completed else "scheduled",
                })

            return pd.DataFrame(records)

        except Exception as e:
            logger.warning(f"ESPN upcoming {league_key} failed: {e}")
            return pd.DataFrame()

    def _fetch_season(self, espn_path: str, league_key: str, season: Any) -> list[dict]:
        """
        Fetch all games for a season by iterating through dates.
        ESPN's API returns scoreboard data date-by-date.
        """
        # Determine the date range for this season
        start_month, end_month = LEAGUE_DEFAULT_MONTHS.get(league_key, (1, 12))
        season_year = int(season)

        # For seasons spanning two years (like NBA), the season label is the
        # starting year. So "2024" means Oct 2024 - Jun 2025.
        if start_month > end_month:
            # Season spans two years
            from datetime import date as dt_date, timedelta
            start_date = dt_date(season_year, start_month, 1)
            end_date = dt_date(season_year + 1, end_month + 1, 1) + timedelta(days=-1)
        else:
            from datetime import date as dt_date, timedelta
            start_date = dt_date(season_year, start_month, 1)
            end_date = dt_date(season_year, end_month + 1, 1) + timedelta(days=-1)

        all_games = {}
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y%m%d")
            records = self._fetch_date(espn_path, league_key, date_str)
            for rec in records:
                gid = rec["game_id"]
                if gid not in all_games:
                    all_games[gid] = rec
            current += timedelta(days=1)  # Step by day to capture ALL games

        return list(all_games.values())

    def _fetch_date(
        self, espn_path: str, league_key: str, date_str: str
    ) -> list[dict]:
        """Fetch all games for a specific date."""
        self._rate_limit()

        try:
            url = ESPN_SCOREBOARD_URL.format(sport=espn_path)
            resp = self._session.get(
                url, params={"dates": date_str, "limit": 300}, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        records = []
        events = data.get("events", []) + data.get("upcoming", [])
        for event in events:
            comp = self._get_competition(event)
            if not comp:
                continue

            home, away = self._get_teams(comp)
            if not home or not away:
                continue

            status = comp.get("status", {}).get("type", {}).get("name", "")
            is_completed = status in {"STATUS_FINAL", "STATUS_FINAL_ALT", "STATUS_FINAL_ALT_2"}
            if not is_completed:
                continue

            home_score = home.get("score")
            away_score = away.get("score")
            if home_score is None or away_score is None:
                continue

            home_short = _team_short(home.get("displayName", ""))
            away_short = _team_short(away.get("displayName", ""))
            event_date = event.get("date", "")[:10]

            # Two rows (one per team) to match CANONICAL_SCHEMA
            gid = event.get("id", f"espn_{league_key}_{event_date}_{home_short}_{away_short}")

            records.append({
                "game_id": gid,
                "league": league_key,
                "season": str(season_year_from_date(event_date)),
                "date": event_date,
                "home_team": home_short,
                "away_team": away_short,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "total_points": int(home_score) + int(away_score),
                "home_win": 1 if int(home_score) > int(away_score) else 0,
            })

        return records

    @staticmethod
    def _get_competition(event: dict) -> Optional[dict]:
        """Extract the competition object from an ESPN event."""
        competitions = event.get("competitions", [])
        return competitions[0] if competitions else None

    @staticmethod
    def _get_teams(competition: dict) -> tuple[Optional[dict], Optional[dict]]:
        """Extract home and away teams from a competition."""
        competitors = competition.get("competitors", [])
        home = None
        away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c.get("team", {})
            else:
                away = c.get("team", {})
        return home, away


def season_year_from_date(date_str: str) -> int:
    """Extract a season year from a date string."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # For NBA/college seasons that start in one year and end in the next,
        # the season label is the starting year
        if dt.month >= 10:
            return dt.year
        return dt.year
    except Exception:
        return datetime.now().year
