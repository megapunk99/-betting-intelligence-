"""Deeper diagnostics - fix the season column issue and check more things."""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer

loader = NBADataLoader()
raw_df = loader.load_game_logs()

print("=" * 70)
print("DIAGNOSTIC A: Are 2025-26 games from preseason or regular season?")
print("=" * 70)
s2 = raw_df[raw_df['SEASON'] == '2025-26']
print(f"Game ID prefix distribution:")
prefixes = s2['GAME_ID'].str[:4].value_counts().sort_index()
print(prefixes.to_string())

# Check if 0012500008-style IDs are preseason
print(f"\nSample GAME_IDs: {list(s2['GAME_ID'].head(10))}")
print(f"Date range for 2025-26: {s2['GAME_DATE'].min()} to {s2['GAME_DATE'].max()}")

print("\n" + "=" * 70)
print("DIAGNOSTIC B: Is the FTA difference real or data source issue?")
print("=" * 70)
s1 = raw_df[raw_df['SEASON'] == '2024-25']

# Compare month-by-month for the overlapping periods
for season_name, df in [('2024-25', s1), ('2025-26', s2)]:
    df = df.copy()
    df['month'] = pd.to_datetime(df['GAME_DATE']).dt.month
    print(f"\n{season_name} FTA by month:")
    monthly = df.groupby('month')['FTA'].agg(['mean', 'std', 'count'])
    print(monthly.to_string())

print("\n" + "=" * 70)
print("DIAGNOSTIC C: Check for preseason data contamination")
print("=" * 70)
# NBA preseason typically has "PRESEASON" in matchup or is before October 22
preseason_25 = s2[pd.to_datetime(s2['GAME_DATE']).dt.month < 10]
print(f"2025-26 games before October: {len(preseason_25)}")
if len(preseason_25) > 0:
    print(preseason_25[['GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION', 'PTS', 'MIN']].head(10).to_string())
    print(f"\nFTA in preseason: {preseason_25['FTA'].mean():.2f}")
    regular_25 = s2[pd.to_datetime(s2['GAME_DATE']).dt.month >= 10]
    print(f"FTA in regular season: {regular_25['FTA'].mean():.2f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC D: What does the forward test actually predict?")
print("=" * 70)

# Now check the forward test flow
sys.path.insert(0, '.')
from data.odds_fetcher import OddsAPIClient
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.environ.get('ODDS_API_KEY', '')
client = OddsAPIClient(api_key=api_key)
games = client.get_upcoming_games_with_odds(sport='basketball_nba', markets='h2h,spreads,totals', use_cache=False)

print(f"\nOdds API games: {len(games)}")
for g in games:
    print(f"\n  GAME: {g.away_team} @ {g.home_team}")
    print(f"  Full names: away={g.away_team}, home={g.home_team}")
    print(f"  Short names: away={g.away_team_short}, home={g.home_team_short}")
    print(f"  Commence: {g.commence_time}")
    
    # Check that these teams exist in the database
    for team_short, team_full in [(g.home_team_short, g.home_team), (g.away_team_short, g.away_team)]:
        in_2024 = team_full in set(s1['TEAM_NAME'])
        in_2025 = team_full in set(s2['TEAM_NAME'])
        print(f"  {team_full} (short={team_short}): in 2024-25 DB={in_2024}, in 2025-26 DB={in_2025}")
    
    print(f"  Home ML: {g.home_moneyline}")
    print(f"  Away ML: {g.away_moneyline}")
    print(f"  Totals: {g.totals}")
    print(f"  Spreads: {g.spreads}")
    print(f"  Num books: {g.n_books}")

print("\n" + "=" * 70)
print("DIAGNOSTIC E: Check feature pipeline for upcoming game")
print("=" * 70)

fe = FeatureEngineer()
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

print(f"\nTotal features: {len(feature_cols)}")
print(f"Feature rows: {len(feature_df)}")

# Check the last few rows
last_rows = feature_df.sort_values('GAME_DATE_home').tail(5)
print(f"\nLast game dates in features: {list(last_rows['GAME_DATE_home'])}")
print(f"Last game matchups: {list(last_rows['MATCHUP_home'])}")

# Check if the SEASON column exists
if 'SEASON' in feature_df.columns:
    print(f"\nFeature SEASON distribution: {feature_df['SEASON'].value_counts().to_dict()}")
else:
    print(f"\nSEASON column not in features. Available: {[c for c in feature_df.columns if 'SEASON' in c]}")

print("\nDone.")
