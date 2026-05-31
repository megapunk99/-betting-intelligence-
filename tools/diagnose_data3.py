"""Check team name matching between odds API and database."""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import pandas as pd
from betting_intel.data.loader import NBADataLoader

loader = NBADataLoader()
raw_df = loader.load_game_logs()

print("=== ALL TEAM NAMES IN DATABASE ===")
names = sorted(raw_df['TEAM_NAME'].unique())
print(names)
print()

print("=== ALL TEAM ABBREVIATIONS ===")
abbrs = sorted(raw_df['TEAM_ABBREVIATION'].unique())
print(abbrs)
print()

# Check for whitespace issues
print("=== WHITESPACE CHECK ===")
for n in names:
    if n != n.strip():
        print(f"  WHITESPACE: '{repr(n)}'")
        
# Check for any non-standard characters
for n in names:
    clean = n.encode('ascii', 'ignore').decode()
    if clean != n:
        print(f"  NON-ASCII: '{repr(n)}'")

print()
print("=== MATCH AGAINST ODDS API NAMES ===")
odds_names = [
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
    'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
    'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
    'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
    'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans',
    'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers',
    'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
    'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'
]

for on in odds_names:
    found = on in names
    print(f"  {on:30s}: {'FOUND' if found else 'MISSING'}")

print()
print("=== CHECK FOR GAME ID PREFIX SPECIFICS ===")
# Check different game ID prefixes in 2025-26
s2 = raw_df[raw_df['SEASON'] == '2025-26']
for prefix in ['0012', '0022', '0042', '0052', '0062']:
    sub = s2[s2['GAME_ID'].str.startswith(prefix)]
    if len(sub) > 0:
        print(f"\n{prefix} ({len(sub)//2} games):")
        print(f"  Date range: {sub['GAME_DATE'].min()} to {sub['GAME_DATE'].max()}")
        print(f"  MIN mean: {sub['MIN'].mean():.1f}")
        print(f"  PTS mean: {sub['PTS'].mean():.1f}")
        print(f"  FTA mean: {sub['FTA'].mean():.1f}")
        # Sample matchups
        sample = sub['MATCHUP'].drop_duplicates().head(3).tolist()
        print(f"  Sample matchups: {sample}")
