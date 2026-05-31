"""
NBA Player Injury & Importance Module.

Fetches player props from TheOddsAPI per-event endpoint to infer injury status
and compute "missing PPG weighted by importance" — an injury impact feature.

How it works:
  1. Fetch all upcoming NBA event IDs from TheOddsAPI
  2. For each event, fetch player props (player_points)
  3. For each team, identify which known roster players have prop lines
  4. Players without prop lines (who normally have them) are likely injured/out
  5. Compute injury_impact = sum of (missing_player_PPG × importance_weight)

Data sources:
  - TheOddsAPI v4 per-event endpoint for player props
  - Curated player importance database (150+ rotation players)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# ── Team Name Mapping ─────────────────────────────────────────────────────
# Maps TheOddsAPI full team names ("Boston Celtics") to abbreviations ("BOS").
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Team abbreviation → short name (for display)
TEAM_ABBR_TO_SHORT = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}

# ── Player Importance Database ────────────────────────────────────────────
# This is a curated list of NBA rotation players with their team, PPG, and
# role tier. Used to detect when a key player is missing from prop lines.
#
# Role tiers:
#   STAR     = All-Star / franchise player (weight: 1.0)
#   STARTER  = Rotation starter (weight: 0.7)
#   ROTATION = Bench / rotation player (weight: 0.3)
#   BENCH    = Deep bench / two-way (weight: 0.1)
#
# PPG values are season averages and will be updated as the season progresses.
# Each player should appear ONLY ONCE (with their current team).

PLAYER_DATABASE: dict[str, dict] = {
    # ── Atlanta Hawks ──
    "Trae Young": {"team": "ATL", "ppg": 25.5, "role": "STAR"},
    "Dejounte Murray": {"team": "ATL", "ppg": 22.0, "role": "STAR"},
    "Jalen Johnson": {"team": "ATL", "ppg": 16.5, "role": "STARTER"},
    "Clint Capela": {"team": "ATL", "ppg": 11.5, "role": "STARTER"},
    "Bogdan Bogdanovic": {"team": "ATL", "ppg": 15.0, "role": "STARTER"},
    "De'Andre Hunter": {"team": "ATL", "ppg": 14.0, "role": "STARTER"},
    "Onyeka Okongwu": {"team": "ATL", "ppg": 10.0, "role": "ROTATION"},
    "Saddiq Bey": {"team": "ATL", "ppg": 12.0, "role": "ROTATION"},
    "Garrison Mathews": {"team": "ATL", "ppg": 5.0, "role": "ROTATION"},
    "Kobe Bufkin": {"team": "ATL", "ppg": 7.0, "role": "ROTATION"},
    # ── Boston Celtics ──
    "Jayson Tatum": {"team": "BOS", "ppg": 27.5, "role": "STAR"},
    "Jaylen Brown": {"team": "BOS", "ppg": 23.0, "role": "STAR"},
    "Kristaps Porzingis": {"team": "BOS", "ppg": 19.0, "role": "STARTER"},
    "Derrick White": {"team": "BOS", "ppg": 14.5, "role": "STARTER"},
    "Jrue Holiday": {"team": "BOS", "ppg": 12.5, "role": "STARTER"},
    "Al Horford": {"team": "BOS", "ppg": 8.5, "role": "STARTER"},
    "Payton Pritchard": {"team": "BOS", "ppg": 12.0, "role": "ROTATION"},
    "Sam Hauser": {"team": "BOS", "ppg": 8.0, "role": "ROTATION"},
    "Luke Kornet": {"team": "BOS", "ppg": 4.5, "role": "ROTATION"},
    "Neemias Queta": {"team": "BOS", "ppg": 5.0, "role": "ROTATION"},
    # ── Brooklyn Nets ──
    "Cam Thomas": {"team": "BKN", "ppg": 22.0, "role": "STARTER"},
    "Ben Simmons": {"team": "BKN", "ppg": 7.0, "role": "STARTER"},
    "Nic Claxton": {"team": "BKN", "ppg": 12.0, "role": "STARTER"},
    "Cameron Johnson": {"team": "BKN", "ppg": 14.0, "role": "STARTER"},
    "Dorian Finney-Smith": {"team": "BKN", "ppg": 8.5, "role": "ROTATION"},
    "Dennis Schroder": {"team": "BKN", "ppg": 14.0, "role": "ROTATION"},
    "Day'Ron Sharpe": {"team": "BKN", "ppg": 7.0, "role": "ROTATION"},
    "Lonnie Walker IV": {"team": "BKN", "ppg": 9.0, "role": "ROTATION"},
    # ── Charlotte Hornets ──
    "LaMelo Ball": {"team": "CHA", "ppg": 24.0, "role": "STAR"},
    "Brandon Miller": {"team": "CHA", "ppg": 18.0, "role": "STARTER"},
    "Miles Bridges": {"team": "CHA", "ppg": 20.0, "role": "STARTER"},
    "Mark Williams": {"team": "CHA", "ppg": 12.0, "role": "STARTER"},
    "Tre Mann": {"team": "CHA", "ppg": 10.5, "role": "ROTATION"},
    "Nick Richards": {"team": "CHA", "ppg": 9.0, "role": "ROTATION"},
    "Grant Williams": {"team": "CHA", "ppg": 10.0, "role": "ROTATION"},
    "Vasilije Micic": {"team": "CHA", "ppg": 7.5, "role": "ROTATION"},
    # ── Chicago Bulls ──
    "DeMar DeRozan": {"team": "CHI", "ppg": 24.0, "role": "STAR"},
    "Zach LaVine": {"team": "CHI", "ppg": 22.0, "role": "STAR"},
    "Nikola Vucevic": {"team": "CHI", "ppg": 18.0, "role": "STARTER"},
    "Coby White": {"team": "CHI", "ppg": 18.5, "role": "STARTER"},
    "Alex Caruso": {"team": "CHI", "ppg": 8.0, "role": "STARTER"},
    "Patrick Williams": {"team": "CHI", "ppg": 10.0, "role": "STARTER"},
    "Ayo Dosunmu": {"team": "CHI", "ppg": 11.0, "role": "ROTATION"},
    # ── Cleveland Cavaliers ──
    "Donovan Mitchell": {"team": "CLE", "ppg": 27.0, "role": "STAR"},
    "Darius Garland": {"team": "CLE", "ppg": 20.0, "role": "STAR"},
    "Evan Mobley": {"team": "CLE", "ppg": 15.5, "role": "STARTER"},
    "Jarrett Allen": {"team": "CLE", "ppg": 14.0, "role": "STARTER"},
    "Max Strus": {"team": "CLE", "ppg": 12.0, "role": "STARTER"},
    "Caris LeVert": {"team": "CLE", "ppg": 12.0, "role": "ROTATION"},
    "Isaac Okoro": {"team": "CLE", "ppg": 8.5, "role": "ROTATION"},
    "Georges Niang": {"team": "CLE", "ppg": 8.0, "role": "ROTATION"},
    "Sam Merrill": {"team": "CLE", "ppg": 7.0, "role": "ROTATION"},
    # ── Dallas Mavericks ──
    "Luka Doncic": {"team": "DAL", "ppg": 33.0, "role": "STAR"},
    "Kyrie Irving": {"team": "DAL", "ppg": 25.0, "role": "STAR"},
    "Klay Thompson": {"team": "DAL", "ppg": 16.0, "role": "STARTER"},
    "P.J. Washington": {"team": "DAL", "ppg": 13.0, "role": "STARTER"},
    "Daniel Gafford": {"team": "DAL", "ppg": 12.0, "role": "STARTER"},
    "Dereck Lively II": {"team": "DAL", "ppg": 9.0, "role": "STARTER"},
    "Josh Green": {"team": "DAL", "ppg": 8.0, "role": "ROTATION"},
    "Jaden Hardy": {"team": "DAL", "ppg": 8.5, "role": "ROTATION"},
    "Dante Exum": {"team": "DAL", "ppg": 7.5, "role": "ROTATION"},
    "Maxi Kleber": {"team": "DAL", "ppg": 5.5, "role": "ROTATION"},
    # ── Denver Nuggets ──
    "Nikola Jokic": {"team": "DEN", "ppg": 28.5, "role": "STAR"},
    "Jamal Murray": {"team": "DEN", "ppg": 21.0, "role": "STAR"},
    "Aaron Gordon": {"team": "DEN", "ppg": 14.0, "role": "STARTER"},
    "Michael Porter Jr.": {"team": "DEN", "ppg": 17.0, "role": "STARTER"},
    "Kentavious Caldwell-Pope": {"team": "DEN", "ppg": 10.0, "role": "STARTER"},
    "Reggie Jackson": {"team": "DEN", "ppg": 10.0, "role": "ROTATION"},
    "Christian Braun": {"team": "DEN", "ppg": 8.0, "role": "ROTATION"},
    "Peyton Watson": {"team": "DEN", "ppg": 6.0, "role": "ROTATION"},
    "Julian Strawther": {"team": "DEN", "ppg": 6.5, "role": "ROTATION"},
    # ── Detroit Pistons ──
    "Cade Cunningham": {"team": "DET", "ppg": 23.0, "role": "STAR"},
    "Jaden Ivey": {"team": "DET", "ppg": 17.0, "role": "STARTER"},
    "Ausar Thompson": {"team": "DET", "ppg": 8.5, "role": "STARTER"},
    "Jalen Duren": {"team": "DET", "ppg": 12.0, "role": "STARTER"},
    "Isaiah Stewart": {"team": "DET", "ppg": 9.0, "role": "STARTER"},
    "Marcus Sasser": {"team": "DET", "ppg": 9.0, "role": "ROTATION"},
    "Simone Fontecchio": {"team": "DET", "ppg": 8.0, "role": "ROTATION"},
    "Tobias Harris": {"team": "DET", "ppg": 14.0, "role": "STARTER"},
    # ── Golden State Warriors ──
    "Stephen Curry": {"team": "GSW", "ppg": 26.0, "role": "STAR"},
    "Draymond Green": {"team": "GSW", "ppg": 8.5, "role": "STARTER"},
    "Andrew Wiggins": {"team": "GSW", "ppg": 14.0, "role": "STARTER"},
    "Jonathan Kuminga": {"team": "GSW", "ppg": 14.0, "role": "STARTER"},
    "Brandin Podziemski": {"team": "GSW", "ppg": 10.0, "role": "ROTATION"},
    "Trayce Jackson-Davis": {"team": "GSW", "ppg": 8.5, "role": "ROTATION"},
    "Moses Moody": {"team": "GSW", "ppg": 7.0, "role": "ROTATION"},
    "Gary Payton II": {"team": "GSW", "ppg": 5.5, "role": "ROTATION"},
    "Buddy Hield": {"team": "GSW", "ppg": 12.0, "role": "ROTATION"},
    # ── Houston Rockets ──
    "Jalen Green": {"team": "HOU", "ppg": 21.0, "role": "STARTER"},
    "Alperen Sengun": {"team": "HOU", "ppg": 19.0, "role": "STARTER"},
    "Fred VanVleet": {"team": "HOU", "ppg": 16.0, "role": "STARTER"},
    "Dillon Brooks": {"team": "HOU", "ppg": 13.0, "role": "STARTER"},
    "Jabari Smith Jr.": {"team": "HOU", "ppg": 12.0, "role": "STARTER"},
    "Amen Thompson": {"team": "HOU", "ppg": 10.0, "role": "ROTATION"},
    "Cam Whitmore": {"team": "HOU", "ppg": 11.0, "role": "ROTATION"},
    "Tari Eason": {"team": "HOU", "ppg": 9.5, "role": "ROTATION"},
    "Steven Adams": {"team": "HOU", "ppg": 5.0, "role": "ROTATION"},
    "Jae'Sean Tate": {"team": "HOU", "ppg": 6.0, "role": "ROTATION"},
    # ── Indiana Pacers ──
    "Tyrese Haliburton": {"team": "IND", "ppg": 21.0, "role": "STAR"},
    "Pascal Siakam": {"team": "IND", "ppg": 21.0, "role": "STAR"},
    "Myles Turner": {"team": "IND", "ppg": 17.0, "role": "STARTER"},
    "Bennedict Mathurin": {"team": "IND", "ppg": 16.0, "role": "STARTER"},
    "Aaron Nesmith": {"team": "IND", "ppg": 11.0, "role": "STARTER"},
    "Andrew Nembhard": {"team": "IND", "ppg": 9.5, "role": "ROTATION"},
    "T.J. McConnell": {"team": "IND", "ppg": 8.5, "role": "ROTATION"},
    "Obi Toppin": {"team": "IND", "ppg": 10.0, "role": "ROTATION"},
    "Isaiah Jackson": {"team": "IND", "ppg": 6.0, "role": "ROTATION"},
    # ── LA Clippers ──
    "Kawhi Leonard": {"team": "LAC", "ppg": 24.0, "role": "STAR"},
    "James Harden": {"team": "LAC", "ppg": 16.0, "role": "STARTER"},
    "Ivica Zubac": {"team": "LAC", "ppg": 12.0, "role": "STARTER"},
    "Norman Powell": {"team": "LAC", "ppg": 14.0, "role": "STARTER"},
    "Terance Mann": {"team": "LAC", "ppg": 8.0, "role": "ROTATION"},
    "Russell Westbrook": {"team": "LAC", "ppg": 10.0, "role": "ROTATION"},
    "Amir Coffey": {"team": "LAC", "ppg": 6.5, "role": "ROTATION"},
    "Daniel Theis": {"team": "LAC", "ppg": 5.5, "role": "ROTATION"},
    # ── Los Angeles Lakers ──
    "LeBron James": {"team": "LAL", "ppg": 25.0, "role": "STAR"},
    "Anthony Davis": {"team": "LAL", "ppg": 25.0, "role": "STAR"},
    "Austin Reaves": {"team": "LAL", "ppg": 16.0, "role": "STARTER"},
    "D'Angelo Russell": {"team": "LAL", "ppg": 14.0, "role": "STARTER"},
    "Rui Hachimura": {"team": "LAL", "ppg": 12.0, "role": "STARTER"},
    "Jarred Vanderbilt": {"team": "LAL", "ppg": 5.5, "role": "STARTER"},
    "Gabe Vincent": {"team": "LAL", "ppg": 6.0, "role": "ROTATION"},
    "Taurean Prince": {"team": "LAL", "ppg": 8.0, "role": "ROTATION"},
    "Max Christie": {"team": "LAL", "ppg": 5.0, "role": "ROTATION"},
    # ── Memphis Grizzlies ──
    "Ja Morant": {"team": "MEM", "ppg": 26.0, "role": "STAR"},
    "Jaren Jackson Jr.": {"team": "MEM", "ppg": 22.0, "role": "STAR"},
    "Desmond Bane": {"team": "MEM", "ppg": 19.0, "role": "STARTER"},
    "Marcus Smart": {"team": "MEM", "ppg": 11.0, "role": "STARTER"},
    "GG Jackson": {"team": "MEM", "ppg": 14.0, "role": "STARTER"},
    "Zach Edey": {"team": "MEM", "ppg": 10.0, "role": "STARTER"},
    "Brandon Clarke": {"team": "MEM", "ppg": 8.0, "role": "ROTATION"},
    "Luke Kennard": {"team": "MEM", "ppg": 8.5, "role": "ROTATION"},
    "Santi Aldama": {"team": "MEM", "ppg": 9.0, "role": "ROTATION"},
    "Derrick Rose": {"team": "MEM", "ppg": 6.0, "role": "ROTATION"},
    # ── Miami Heat ──
    "Jimmy Butler": {"team": "MIA", "ppg": 21.0, "role": "STAR"},
    "Bam Adebayo": {"team": "MIA", "ppg": 20.0, "role": "STAR"},
    "Tyler Herro": {"team": "MIA", "ppg": 20.5, "role": "STARTER"},
    "Jaime Jaquez Jr.": {"team": "MIA", "ppg": 12.0, "role": "STARTER"},
    "Terry Rozier": {"team": "MIA", "ppg": 15.0, "role": "STARTER"},
    "Caleb Martin": {"team": "MIA", "ppg": 10.0, "role": "ROTATION"},
    "Duncan Robinson": {"team": "MIA", "ppg": 11.0, "role": "ROTATION"},
    "Josh Richardson": {"team": "MIA", "ppg": 7.0, "role": "ROTATION"},
    "Kevin Love": {"team": "MIA", "ppg": 7.0, "role": "ROTATION"},
    # ── Milwaukee Bucks ──
    "Giannis Antetokounmpo": {"team": "MIL", "ppg": 31.0, "role": "STAR"},
    "Damian Lillard": {"team": "MIL", "ppg": 25.0, "role": "STAR"},
    "Khris Middleton": {"team": "MIL", "ppg": 15.0, "role": "STARTER"},
    "Brook Lopez": {"team": "MIL", "ppg": 13.0, "role": "STARTER"},
    "Bobby Portis": {"team": "MIL", "ppg": 13.0, "role": "STARTER"},
    "Pat Connaughton": {"team": "MIL", "ppg": 6.5, "role": "ROTATION"},
    "MarJon Beauchamp": {"team": "MIL", "ppg": 5.0, "role": "ROTATION"},
    "Jae Crowder": {"team": "MIL", "ppg": 6.0, "role": "ROTATION"},
    # ── Minnesota Timberwolves ──
    "Anthony Edwards": {"team": "MIN", "ppg": 26.0, "role": "STAR"},
    "Karl-Anthony Towns": {"team": "MIN", "ppg": 22.0, "role": "STAR"},
    "Rudy Gobert": {"team": "MIN", "ppg": 13.0, "role": "STARTER"},
    "Jaden McDaniels": {"team": "MIN", "ppg": 11.0, "role": "STARTER"},
    "Mike Conley": {"team": "MIN", "ppg": 10.5, "role": "STARTER"},
    "Naz Reid": {"team": "MIN", "ppg": 12.0, "role": "ROTATION"},
    "Nickeil Alexander-Walker": {"team": "MIN", "ppg": 7.5, "role": "ROTATION"},
    "Kyle Anderson": {"team": "MIN", "ppg": 6.0, "role": "ROTATION"},
    # ── New Orleans Pelicans ──
    "Zion Williamson": {"team": "NOP", "ppg": 24.0, "role": "STAR"},
    "Brandon Ingram": {"team": "NOP", "ppg": 22.0, "role": "STAR"},
    "CJ McCollum": {"team": "NOP", "ppg": 19.0, "role": "STARTER"},
    "Trey Murphy III": {"team": "NOP", "ppg": 15.0, "role": "STARTER"},
    "Herbert Jones": {"team": "NOP", "ppg": 11.0, "role": "STARTER"},
    "Naji Marshall": {"team": "NOP", "ppg": 8.0, "role": "ROTATION"},
    "Jose Alvarado": {"team": "NOP", "ppg": 7.0, "role": "ROTATION"},
    "Matt Ryan": {"team": "NOP", "ppg": 6.0, "role": "ROTATION"},
    "Dyson Daniels": {"team": "NOP", "ppg": 5.5, "role": "ROTATION"},
    # ── New York Knicks ──
    "Jalen Brunson": {"team": "NYK", "ppg": 27.0, "role": "STAR"},
    "Julius Randle": {"team": "NYK", "ppg": 22.0, "role": "STAR"},
    "OG Anunoby": {"team": "NYK", "ppg": 15.0, "role": "STARTER"},
    "Mikal Bridges": {"team": "NYK", "ppg": 19.0, "role": "STARTER"},
    "Josh Hart": {"team": "NYK", "ppg": 10.0, "role": "STARTER"},
    "Isaiah Hartenstein": {"team": "NYK", "ppg": 7.5, "role": "STARTER"},
    "Donte DiVincenzo": {"team": "NYK", "ppg": 12.0, "role": "ROTATION"},
    "Miles McBride": {"team": "NYK", "ppg": 7.0, "role": "ROTATION"},
    "Mitchell Robinson": {"team": "NYK", "ppg": 6.0, "role": "ROTATION"},
    "Precious Achiuwa": {"team": "NYK", "ppg": 7.0, "role": "ROTATION"},
    # ── Oklahoma City Thunder ──
    "Shai Gilgeous-Alexander": {"team": "OKC", "ppg": 31.0, "role": "STAR"},
    "Jalen Williams": {"team": "OKC", "ppg": 19.0, "role": "STARTER"},
    "Chet Holmgren": {"team": "OKC", "ppg": 17.0, "role": "STARTER"},
    "Josh Giddey": {"team": "OKC", "ppg": 12.0, "role": "STARTER"},
    "Luguentz Dort": {"team": "OKC", "ppg": 10.5, "role": "STARTER"},
    "Isaiah Joe": {"team": "OKC", "ppg": 9.0, "role": "ROTATION"},
    "Cason Wallace": {"team": "OKC", "ppg": 7.0, "role": "ROTATION"},
    "Aaron Wiggins": {"team": "OKC", "ppg": 6.0, "role": "ROTATION"},
    "Kenrich Williams": {"team": "OKC", "ppg": 5.0, "role": "ROTATION"},
    # ── Orlando Magic ──
    "Paolo Banchero": {"team": "ORL", "ppg": 23.0, "role": "STAR"},
    "Franz Wagner": {"team": "ORL", "ppg": 19.0, "role": "STARTER"},
    "Jalen Suggs": {"team": "ORL", "ppg": 12.0, "role": "STARTER"},
    "Wendell Carter Jr.": {"team": "ORL", "ppg": 11.0, "role": "STARTER"},
    "Cole Anthony": {"team": "ORL", "ppg": 11.5, "role": "ROTATION"},
    "Moritz Wagner": {"team": "ORL", "ppg": 10.0, "role": "ROTATION"},
    "Markelle Fultz": {"team": "ORL", "ppg": 8.0, "role": "STARTER"},
    "Gary Harris": {"team": "ORL", "ppg": 6.0, "role": "ROTATION"},
    "Goga Bitadze": {"team": "ORL", "ppg": 5.5, "role": "ROTATION"},
    "Anthony Black": {"team": "ORL", "ppg": 6.0, "role": "ROTATION"},
    # ── Philadelphia 76ers ──
    "Joel Embiid": {"team": "PHI", "ppg": 33.0, "role": "STAR"},
    "Tyrese Maxey": {"team": "PHI", "ppg": 24.0, "role": "STAR"},
    "Paul George": {"team": "PHI", "ppg": 22.0, "role": "STAR"},
    "Kelly Oubre Jr.": {"team": "PHI", "ppg": 12.0, "role": "STARTER"},
    "De'Anthony Melton": {"team": "PHI", "ppg": 10.0, "role": "ROTATION"},
    "Nicolas Batum": {"team": "PHI", "ppg": 5.0, "role": "ROTATION"},
    "Kyle Lowry": {"team": "PHI", "ppg": 7.0, "role": "ROTATION"},
    "Andre Drummond": {"team": "PHI", "ppg": 8.0, "role": "ROTATION"},
    # ── Phoenix Suns ──
    "Kevin Durant": {"team": "PHX", "ppg": 27.0, "role": "STAR"},
    "Devin Booker": {"team": "PHX", "ppg": 27.0, "role": "STAR"},
    "Bradley Beal": {"team": "PHX", "ppg": 20.0, "role": "STAR"},
    "Jusuf Nurkic": {"team": "PHX", "ppg": 11.0, "role": "STARTER"},
    "Grayson Allen": {"team": "PHX", "ppg": 12.0, "role": "STARTER"},
    "Eric Gordon": {"team": "PHX", "ppg": 10.0, "role": "ROTATION"},
    "Bol Bol": {"team": "PHX", "ppg": 5.0, "role": "ROTATION"},
    "Royce O'Neale": {"team": "PHX", "ppg": 7.0, "role": "ROTATION"},
    "Drew Eubanks": {"team": "PHX", "ppg": 5.5, "role": "ROTATION"},
    "Damion Lee": {"team": "PHX", "ppg": 4.0, "role": "ROTATION"},
    # ── Portland Trail Blazers ──
    "Anfernee Simons": {"team": "POR", "ppg": 22.0, "role": "STARTER"},
    "Shaedon Sharpe": {"team": "POR", "ppg": 17.0, "role": "STARTER"},
    "Jerami Grant": {"team": "POR", "ppg": 20.0, "role": "STARTER"},
    "Deandre Ayton": {"team": "POR", "ppg": 15.0, "role": "STARTER"},
    "Scoot Henderson": {"team": "POR", "ppg": 13.0, "role": "ROTATION"},
    "Malcolm Brogdon": {"team": "POR", "ppg": 14.0, "role": "ROTATION"},
    "Matisse Thybulle": {"team": "POR", "ppg": 5.5, "role": "ROTATION"},
    "Jabari Walker": {"team": "POR", "ppg": 7.0, "role": "ROTATION"},
    "Robert Williams III": {"team": "POR", "ppg": 6.0, "role": "ROTATION"},
    "Rayan Rupert": {"team": "POR", "ppg": 4.0, "role": "ROTATION"},
    # ── Sacramento Kings ──
    "De'Aaron Fox": {"team": "SAC", "ppg": 26.0, "role": "STAR"},
    "Domantas Sabonis": {"team": "SAC", "ppg": 20.0, "role": "STAR"},
    "Keegan Murray": {"team": "SAC", "ppg": 14.0, "role": "STARTER"},
    "Malik Monk": {"team": "SAC", "ppg": 15.0, "role": "STARTER"},
    "Kevin Huerter": {"team": "SAC", "ppg": 10.0, "role": "STARTER"},
    "Trey Lyles": {"team": "SAC", "ppg": 7.0, "role": "ROTATION"},
    "Kessler Edwards": {"team": "SAC", "ppg": 4.5, "role": "ROTATION"},
    "Davion Mitchell": {"team": "SAC", "ppg": 5.0, "role": "ROTATION"},
    "Keon Ellis": {"team": "SAC", "ppg": 5.0, "role": "ROTATION"},
    # ── San Antonio Spurs ──
    "Victor Wembanyama": {"team": "SAS", "ppg": 23.0, "role": "STAR"},
    "Devin Vassell": {"team": "SAS", "ppg": 18.0, "role": "STARTER"},
    "Keldon Johnson": {"team": "SAS", "ppg": 15.0, "role": "STARTER"},
    "Jeremy Sochan": {"team": "SAS", "ppg": 11.0, "role": "STARTER"},
    "Tre Jones": {"team": "SAS", "ppg": 9.0, "role": "STARTER"},
    "Zach Collins": {"team": "SAS", "ppg": 10.0, "role": "ROTATION"},
    "Malaki Branham": {"team": "SAS", "ppg": 8.0, "role": "ROTATION"},
    "Julian Champagnie": {"team": "SAS", "ppg": 7.0, "role": "ROTATION"},
    "Blake Wesley": {"team": "SAS", "ppg": 4.5, "role": "ROTATION"},
    "Harrison Barnes": {"team": "SAS", "ppg": 11.0, "role": "STARTER"},
    "Chris Paul": {"team": "SAS", "ppg": 9.0, "role": "STARTER"},
    # ── Toronto Raptors ──
    "Scottie Barnes": {"team": "TOR", "ppg": 20.0, "role": "STAR"},
    "RJ Barrett": {"team": "TOR", "ppg": 20.0, "role": "STARTER"},
    "Immanuel Quickley": {"team": "TOR", "ppg": 16.0, "role": "STARTER"},
    "Jakob Poeltl": {"team": "TOR", "ppg": 10.0, "role": "STARTER"},
    "Gary Trent Jr.": {"team": "TOR", "ppg": 12.0, "role": "STARTER"},
    "Ochai Agbaji": {"team": "TOR", "ppg": 7.0, "role": "ROTATION"},
    "Gradey Dick": {"team": "TOR", "ppg": 7.0, "role": "ROTATION"},
    "Chris Boucher": {"team": "TOR", "ppg": 8.0, "role": "ROTATION"},
    "Jontay Porter": {"team": "TOR", "ppg": 4.5, "role": "ROTATION"},
    "Bruce Brown": {"team": "TOR", "ppg": 8.0, "role": "ROTATION"},
    "Kelly Olynyk": {"team": "TOR", "ppg": 8.0, "role": "ROTATION"},
    # ── Utah Jazz ──
    "Lauri Markkanen": {"team": "UTA", "ppg": 23.0, "role": "STAR"},
    "Collin Sexton": {"team": "UTA", "ppg": 18.0, "role": "STARTER"},
    "Jordan Clarkson": {"team": "UTA", "ppg": 17.0, "role": "STARTER"},
    "Walker Kessler": {"team": "UTA", "ppg": 9.0, "role": "STARTER"},
    "Keyonte George": {"team": "UTA", "ppg": 14.0, "role": "STARTER"},
    "John Collins": {"team": "UTA", "ppg": 14.0, "role": "STARTER"},
    "Taylor Hendricks": {"team": "UTA", "ppg": 5.0, "role": "ROTATION"},
    "Brice Sensabaugh": {"team": "UTA", "ppg": 7.0, "role": "ROTATION"},
    "Kris Dunn": {"team": "UTA", "ppg": 6.0, "role": "ROTATION"},
    # ── Washington Wizards ──
    "Jordan Poole": {"team": "WAS", "ppg": 19.0, "role": "STARTER"},
    "Kyle Kuzma": {"team": "WAS", "ppg": 21.0, "role": "STARTER"},
    "Jonas Valanciunas": {"team": "WAS", "ppg": 12.0, "role": "STARTER"},
    "Deni Avdija": {"team": "WAS", "ppg": 13.0, "role": "STARTER"},
    "Tyus Jones": {"team": "WAS", "ppg": 10.0, "role": "STARTER"},
    "Bilal Coulibaly": {"team": "WAS", "ppg": 7.0, "role": "ROTATION"},
    "Corey Kispert": {"team": "WAS", "ppg": 12.0, "role": "ROTATION"},
    "Richaun Holmes": {"team": "WAS", "ppg": 5.0, "role": "ROTATION"},
    "Johnny Davis": {"team": "WAS", "ppg": 3.5, "role": "BENCH"},
    "Patrick Baldwin Jr.": {"team": "WAS", "ppg": 4.0, "role": "BENCH"},
}

# Role → importance weight mapping
ROLE_WEIGHT = {
    "STAR": 1.0,
    "STARTER": 0.7,
    "ROTATION": 0.3,
    "BENCH": 0.1,
}


@dataclass
class InjuryImpact:
    """Computed injury impact for a single team in a game."""

    team_abbr: str
    team_short: str
    total_player_props_pts: float = 0.0  # Sum of player_points prop lines
    num_players_with_props: int = 0
    missing_stars: list[str] = field(default_factory=list)  # Names of missing star/starter players
    missing_ppg_weighted: float = 0.0  # Sum of (missing_PPG × importance_weight)
    prop_players: list[dict] = field(default_factory=list)  # All players found in props


@dataclass
class GameInjuryData:
    """Injury data for a single game."""

    game_id: str
    home_team: str
    away_team: str
    home_impact: InjuryImpact | None = None
    away_impact: InjuryImpact | None = None
    total_prop_pts: float = 0.0  # Sum of both teams' prop points

    @property
    def has_injuries(self) -> bool:
        """Whether any significant injuries were detected."""
        if self.home_impact and self.home_impact.missing_ppg_weighted > 0:
            return True
        if self.away_impact and self.away_impact.missing_ppg_weighted > 0:
            return True
        return False

    @property
    def total_missing_ppg(self) -> float:
        """Total missing PPG across both teams (weighted by importance)."""
        total = 0.0
        if self.home_impact:
            total += self.home_impact.missing_ppg_weighted
        if self.away_impact:
            total += self.away_impact.missing_ppg_weighted
        return total


# ── Fetcher ───────────────────────────────────────────────────────────────


class PlayerInjuryFetcher:
    """
    Fetches player props from TheOddsAPI to compute injury impact.

    Uses the per-event endpoint to get player_points, player_rebounds, and
    player_assists markets. Cross-references against a known player database
    to identify missing/injured players.

    Usage:
        fetcher = PlayerInjuryFetcher(api_key="your_key")
        injury_data = fetcher.fetch_injury_impact_for_upcoming_games()
        for game_data in injury_data:
            print(game_data.home_impact.missing_stars)
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.the-odds-api.com",
        rate_limit_delay: float = 0.3,  # Be respectful of rate limits
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self._session = None

    def _get_session(self):
        """Lazy-init requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/120.0.0.0 Safari/537.36",
            })
        return self._session

    def fetch_upcoming_events(self) -> list[dict]:
        """Fetch all upcoming NBA events with their IDs."""
        session = self._get_session()
        url = f"{self.base_url}/v4/sports/basketball_nba/events"
        resp = session.get(url, params={"apiKey": self.api_key, "dateFormat": "iso"}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def fetch_player_props_for_event(self, event_id: str) -> list[dict] | None:
        """
        Fetch player props for a single event.

        Returns a list of player prop outcomes (name, market, point line).
        Returns None if props aren't available yet for this event.
        """
        session = self._get_session()
        url = f"{self.base_url}/v4/sports/basketball_nba/events/{event_id}/odds"
        resp = session.get(
            url,
            params={
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "player_points,player_rebounds,player_assists",
                "oddsFormat": "american",
            },
            timeout=15,
        )

        if resp.status_code != 200:
            return None

        data = resp.json()
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            return None

        # Collect all player prop outcomes from the first bookmaker
        # (all books will have similar lines; use the first one as canonical)
        all_outcomes = []
        for market in bookmakers[0].get("markets", []):
            market_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                outcome["_market"] = market_key
                all_outcomes.append(outcome)

        return all_outcomes

    def fetch_injury_impact_for_upcoming_games(self) -> list[GameInjuryData]:
        """
        Main method: fetch all upcoming games + player props → injury impacts.

        Returns:
            List of GameInjuryData, one per upcoming game with player props.
        """
        if not self.api_key:
            print(f"  [PlayerInjury] No API key provided")
            return []

        try:
            # Step 1: Get all upcoming events
            events = self.fetch_upcoming_events()
            events_with_props = [e for e in events if e.get("id")]

            if not events_with_props:
                print(f"  [PlayerInjury] No upcoming events found")
                return []

            print(f"  [PlayerInjury] Found {len(events_with_props)} upcoming events, "
                  f"fetching player props...")

            game_results = []

            for i, event in enumerate(events_with_props):
                event_id = event["id"]
                home_team = event.get("home_team", "?")
                away_team = event.get("away_team", "?")

                print(f"    [{i+1}/{len(events_with_props)}] {away_team} @ {home_team}...", end=" ")

                # Fetch props for this event
                props = self.fetch_player_props_for_event(event_id)

                if not props:
                    print("no props yet")
                    game_results.append(GameInjuryData(
                        game_id=event_id,
                        home_team=home_team,
                        away_team=away_team,
                    ))
                    time.sleep(self.rate_limit_delay)
                    continue

                # Map team names to abbreviations using direct mapping
                home_abbr = TEAM_NAME_TO_ABBR.get(home_team)
                away_abbr = TEAM_NAME_TO_ABBR.get(away_team)

                if not home_abbr or not away_abbr:
                    print(f"unknown teams ({home_team}, {away_team})")
                    game_results.append(GameInjuryData(
                        game_id=event_id,
                        home_team=home_team,
                        away_team=away_team,
                    ))
                    time.sleep(self.rate_limit_delay)
                    continue

                # Build per-team prop summaries
                home_prop_players = []
                away_prop_players = []

                for outcome in props:
                    description = outcome.get("description", "")  # Player name
                    point = outcome.get("point", 0)
                    market = outcome.get("_market", "")

                    # Determine which team this player belongs to
                    pinfo = PLAYER_DATABASE.get(description)
                    if pinfo:
                        if pinfo["team"] == home_abbr:
                            home_prop_players.append({
                                "name": description,
                                "point": point,
                                "market": market,
                                "ppg": pinfo["ppg"],
                                "role": pinfo["role"],
                            })
                        elif pinfo["team"] == away_abbr:
                            away_prop_players.append({
                                "name": description,
                                "point": point,
                                "market": market,
                                "ppg": pinfo["ppg"],
                                "role": pinfo["role"],
                            })

                # Compute home team injury impact
                home_impact = self._compute_team_impact(home_abbr, home_prop_players)
                away_impact = self._compute_team_impact(away_abbr, away_prop_players)

                total_pts = (home_impact.total_player_props_pts
                             + away_impact.total_player_props_pts)

                game_data = GameInjuryData(
                    game_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    home_impact=home_impact,
                    away_impact=away_impact,
                    total_prop_pts=total_pts,
                )
                game_results.append(game_data)

                # Print summary
                missing_home = ", ".join(home_impact.missing_stars[:3]) if home_impact.missing_stars else "none"
                missing_away = ", ".join(away_impact.missing_stars[:3]) if away_impact.missing_stars else "none"
                print(f"props: {home_impact.num_players_with_props}/{away_impact.num_players_with_props} players, "
                      f"missing: [{missing_home} | {missing_away}]")

                time.sleep(self.rate_limit_delay)

            print(f"  [PlayerInjury] Processed {len(game_results)} games")
            return game_results

        except Exception as e:
            print(f"  [PlayerInjury] Error: {e}")
            return []

    def _compute_team_impact(
        self,
        team_abbr: str,
        prop_players: list[dict],
    ) -> InjuryImpact:
        """
        Compute injury impact for a team by comparing players with props
        against the known roster.

        Players in the database but NOT in the props → likely injured/out.
        Their (PPG × importance_weight) is summed to produce the impact score.
        """
        impact = InjuryImpact(
            team_abbr=team_abbr,
            team_short=TEAM_ABBR_TO_SHORT.get(team_abbr, team_abbr),
        )

        # Sum up prop points
        for p in prop_players:
            impact.total_player_props_pts += p.get("point", 0)
            impact.prop_players.append(p)
        impact.num_players_with_props = len(prop_players)

        # Get all players on this team from the database
        prop_names = {p["name"] for p in prop_players}
        all_team_players = {
            name: info for name, info in PLAYER_DATABASE.items()
            if info["team"] == team_abbr
        }

        # Find missing players (in database, not in props)
        for name, info in all_team_players.items():
            if name not in prop_names:
                weight = ROLE_WEIGHT.get(info["role"], 0.3)
                weighted_ppg = info["ppg"] * weight
                impact.missing_ppg_weighted += weighted_ppg

                # Track missing star/starter players by name
                if info["role"] in ("STAR", "STARTER"):
                    impact.missing_stars.append(f"{name} ({info['ppg']:.0f} PPG, {info['role']})")

        return impact

    def get_injury_display_lines(
        self,
        game_data: GameInjuryData,
    ) -> list[str]:
        """
        Generate human-readable injury impact lines for display.

        Args:
            game_data: Injury data for a single game.

        Returns:
            List of display strings.
        """
        lines = []

        if not game_data.has_injuries:
            return lines

        lines.append(f"  Injury Impact:")

        if game_data.home_impact and game_data.home_impact.missing_stars:
            home_missing = "; ".join(game_data.home_impact.missing_stars)
            lines.append(f"    Home ({game_data.home_impact.team_short}) missing: {home_missing}")

        if game_data.away_impact and game_data.away_impact.missing_stars:
            away_missing = "; ".join(game_data.away_impact.missing_stars)
            lines.append(f"    Away ({game_data.away_impact.team_short}) missing: {away_missing}")

        # Total weighted impact estimate
        if game_data.total_missing_ppg > 0:
            lines.append(f"    Total weighted impact: ~{game_data.total_missing_ppg:.1f} PPG equivalent")

        # Team prop point totals (offensive firepower signal)
        if game_data.home_impact and game_data.away_impact:
            h_pts = game_data.home_impact.total_player_props_pts
            a_pts = game_data.away_impact.total_player_props_pts
            if h_pts > 0 or a_pts > 0:
                lines.append(f"    Player props total: {game_data.home_impact.team_short} {h_pts:.0f} pts, "
                            f"{game_data.away_impact.team_short} {a_pts:.0f} pts "
                            f"(sum: {game_data.total_prop_pts:.0f})")

        return lines
