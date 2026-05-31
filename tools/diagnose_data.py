"""Comprehensive data pipeline diagnostic.
Run this to find ALL issues with data quality.
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer

print("=" * 70)
print("DIAGNOSTIC 1: RAW DATA SOURCE COMPARISON")
print("=" * 70)

loader = NBADataLoader()
raw_df = loader.load_game_logs()

s1 = raw_df[raw_df['SEASON'] == '2024-25']
s2 = raw_df[raw_df['SEASON'] == '2025-26']

print(f"\n2024-25 (ESPN): {len(s1)} rows, {s1['GAME_ID'].nunique()} games")
print(f"2025-26 (NBA CDN): {len(s2)} rows, {s2['GAME_ID'].nunique()} games")

print("\n--- Stat Comparison ---")
for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FG3A', 'FTA', 'MIN', 'PLUS_MINUS']:
    v1 = s1[col]
    v2 = s2[col]
    diff_pct = ((v2.mean() - v1.mean()) / v1.mean() * 100) if v1.mean() != 0 else 0
    print(f"{col:15s}: 2024-25 mean={v1.mean():8.3f} std={v1.std():6.3f} | 2025-26 mean={v2.mean():8.3f} std={v2.std():6.3f} | diff={diff_pct:+.2f}%")

print("\n--- GAME_ID Format Check ---")
print(f"2024-25 first 5 IDs: {list(s1['GAME_ID'].unique()[:5])}")
print(f"2025-26 first 5 IDs: {list(s2['GAME_ID'].unique()[:5])}")

print("\n--- Team Abbreviation Consistency ---")
t1 = set(s1['TEAM_ABBREVIATION'].unique())
t2 = set(s2['TEAM_ABBREVIATION'].unique())
print(f"Only in 2024-25: {t1 - t2}")
print(f"Only in 2025-26: {t2 - t1}")

print("\n--- Team Name Consistency ---")
n1 = set(s1['TEAM_NAME'].unique())
n2 = set(s2['TEAM_NAME'].unique())
print(f"Only in 2024-25: {n1 - n2}")
print(f"Only in 2025-26: {n2 - n1}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: GAME DATASET INTEGRITY")
print("=" * 70)

games_df = loader.build_game_dataset(raw_df)
print(f"\nTotal merged games: {len(games_df)}")
print(f"Columns: {list(games_df.columns)}")

# Check if there are null values that indicate merge failures
null_counts = games_df.isna().sum()
null_cols = null_counts[null_counts > 0]
if len(null_cols) > 0:
    print(f"\nNULL VALUES in merged dataset:")
    for col, cnt in null_cols.items():
        print(f"  {col}: {cnt} nulls")
else:
    print("\nNo null values in merged dataset - good!")

# Per-season game stats
for season in ['2024-25', '2025-26']:
    sub = games_df[games_df['SEASON'] == season]
    print(f"\n{season} ({len(sub)} games):")
    print(f"  total_points: {sub['total_points'].mean():.1f} +/- {sub['total_points'].std():.1f}")
    print(f"  PTS_home: {sub['PTS_home'].mean():.1f} +/- {sub['PTS_home'].std():.1f}")
    print(f"  PTS_away: {sub['PTS_away'].mean():.1f} +/- {sub['PTS_away'].std():.1f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: FEATURE ENGINEERING CHECK")
print("=" * 70)

fe = FeatureEngineer()
raw_df = loader.compute_rest_days(raw_df)
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

print(f"\nTotal features: {len(feature_cols)}")
print(f"Feature rows: {len(feature_df)}")

# Check for NaN features
nan_counts = feature_df[feature_cols].isna().sum()
nan_features = nan_counts[nan_counts > 0]
if len(nan_features) > 0:
    print(f"\nWARNING: {len(nan_features)} features have NaN values:")
    for col, cnt in nan_features.items():
        print(f"  {col}: {cnt} NaN rows")
else:
    print("\nNo NaN values in features - good!")

# Per-season feature distributions 
print("\n--- Feature Distributions Per Season ---")
for season in ['2024-25', '2025-26']:
    sub = feature_df[feature_df['SEASON'] == season]
    print(f"\n{season} ({len(sub)} games):")
    for col in feature_cols[:5]:
        vals = sub[col].dropna()
        print(f"  {col:30s}: mean={vals.mean():10.4f}  std={vals.std():8.4f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC 4: LAST 10 GAMES RAW DATA")
print("=" * 70)

recent = raw_df.sort_values('GAME_DATE').tail(20)
print(recent[['GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION', 'PTS', 'WL', 'SEASON']].to_string(index=False))

print("\n" + "=" * 70)
print("DIAGNOSTIC 5: SEASON_ID VALUES")
print("=" * 70)

print(raw_df.groupby('SEASON')['SEASON_ID'].unique().to_string())

print("\n" + "=" * 70)
print("DIAGNOSTIC 6: TEAM NAME MAPPING FOR ODDS API")
print("=" * 70)

# Check team names in the database
team_names = sorted(raw_df['TEAM_NAME'].unique())
print(f"Database team names: {team_names}")

# Common odds API team names (full names)
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

print(f"\nOdds API expected names: {odds_names}")
mismatch = [n for n in team_names if n not in odds_names]
if mismatch:
    print(f"\nMISMATCH: Database names not matching odds API: {mismatch}")
else:
    print("\nAll database team names match odds API names - good!")
