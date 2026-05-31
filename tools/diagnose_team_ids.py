"""Check TEAM_ID consistency between data sources and standard NBA API IDs."""
import sys
sys.path.insert(0, 'src')
from betting_intel.data.loader import NBADataLoader

loader = NBADataLoader()
raw_df = loader.load_game_logs()

print('=== TEAM_ID PER TEAM PER SEASON ===')
teams = raw_df[['TEAM_NAME', 'TEAM_ID', 'SEASON']].drop_duplicates()
teams = teams.sort_values(['TEAM_NAME', 'SEASON'])
print(teams.to_string(index=False))

print()
print('=== DO TEAM_IDS MATCH STANDARD NBA API IDs? ===')
standard = {
    'Hawks': 1610612737, 'Celtics': 1610612738, 'Nets': 1610612751,
    'Hornets': 1610612766, 'Bulls': 1610612741, 'Cavaliers': 1610612739,
    'Mavericks': 1610612742, 'Nuggets': 1610612743, 'Pistons': 1610612765,
    'Warriors': 1610612744, 'Rockets': 1610612745, 'Pacers': 1610612754,
    'Clippers': 1610612746, 'Lakers': 1610612747, 'Grizzlies': 1610612763,
    'Heat': 1610612748, 'Bucks': 1610612749, 'Timberwolves': 1610612750,
    'Pelicans': 1610612740, 'Knicks': 1610612752, 'Thunder': 1610612760,
    'Magic': 1610612753, '76ers': 1610612755, 'Suns': 1610612756,
    'Trail Blazers': 1610612757, 'Kings': 1610612758, 'Spurs': 1610612759,
    'Raptors': 1610612761, 'Jazz': 1610612762, 'Wizards': 1610612764,
}

mismatches = []
for _, r in teams.iterrows():
    name = r['TEAM_NAME']
    tid = r['TEAM_ID']
    season = r['SEASON']
    expected = standard.get(name)
    if expected and tid != expected:
        mismatches.append(f'{name} ({season}): DB={tid}, expected={expected}')

if mismatches:
    print('MISMATCHES FOUND:')
    for m in mismatches:
        print(f'  {m}')
else:
    print('ALL MATCH - TEAM_IDs are consistent with standard NBA IDs')

print()
print('=== ARE TEAM_IDS CONSISTENT ACROSS SEASONS? ===')
s1_names = set(raw_df[raw_df['SEASON']=='2024-25']['TEAM_NAME'].unique())
s2_names = set(raw_df[raw_df['SEASON']=='2025-26']['TEAM_NAME'].unique())
common = s1_names & s2_names

cross_season_diff = []
for name in sorted(common):
    id1 = raw_df[(raw_df['SEASON']=='2024-25') & (raw_df['TEAM_NAME']==name)]['TEAM_ID'].iloc[0]
    id2 = raw_df[(raw_df['SEASON']=='2025-26') & (raw_df['TEAM_NAME']==name)]['TEAM_ID'].iloc[0]
    if id1 != id2:
        cross_season_diff.append(f'{name}: 2024-25={id1}, 2025-26={id2}')

if cross_season_diff:
    print('DIFFERENT TEAM_IDS ACROSS SEASONS:')
    for d in cross_season_diff:
        print(f'  {d}')
else:
    print('TEAM_IDS are consistent across both seasons - good!')

print()
print('=== GAME_ID PREFIX BREAKDOWN FOR 2025-26 ===')
s2 = raw_df[raw_df['SEASON'] == '2025-26']
for prefix in sorted(s2['GAME_ID'].str[:4].unique()):
    sub = s2[s2['GAME_ID'].str.startswith(prefix)]
    n_games = sub['GAME_ID'].nunique()
    print(f'  {prefix}: {n_games} games, dates {sub["GAME_DATE"].min()[:10]} to {sub["GAME_DATE"].max()[:10]}, MIN_avg={sub["MIN"].mean():.0f}, PTS_avg={sub["PTS"].mean():.1f}')
