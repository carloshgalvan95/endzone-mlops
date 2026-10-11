#!/usr/bin/env python3
"""
Investigate team franchise mapping issues in nflverse data.
"""

import duckdb
import pandas as pd

# Connect to DuckDB
con = duckdb.connect(':memory:')

# Load games and pbp data
print("Loading games.parquet...")
games = con.execute("SELECT * FROM read_parquet('data/raw/games.parquet')").df()
print(f"Total games: {len(games)}")

print("\nLoading play_by_play_1999.parquet...")
pbp_1999 = con.execute("SELECT * FROM read_parquet('data/raw/play_by_play_1999.parquet')").df()
print(f"Total plays 1999: {len(pbp_1999)}")

print("\nLoading play_by_play_2000.parquet...")
pbp_2000 = con.execute("SELECT * FROM read_parquet('data/raw/play_by_play_2000.parquet')").df()
print(f"Total plays 2000: {len(pbp_2000)}")

# Get all unique team abbreviations from games
print("\n" + "="*80)
print("TEAM ABBREVIATIONS IN GAMES")
print("="*80)
games_teams = set(games['away_team'].unique()) | set(games['home_team'].unique())
print(f"Unique teams in games: {sorted(games_teams)}")
print(f"Total: {len(games_teams)}")

# Get all unique team abbreviations from pbp
print("\n" + "="*80)
print("TEAM ABBREVIATIONS IN PLAY-BY-PLAY")
print("="*80)
pbp_1999_teams = set(pbp_1999['posteam'].dropna().unique()) | set(pbp_1999['defteam'].dropna().unique())
pbp_2000_teams = set(pbp_2000['posteam'].dropna().unique()) | set(pbp_2000['defteam'].dropna().unique())
print(f"Unique teams in 1999 pbp: {sorted(pbp_1999_teams)}")
print(f"Unique teams in 2000 pbp: {sorted(pbp_2000_teams)}")

# Identify mismatches
games_only = games_teams - pbp_1999_teams - pbp_2000_teams
pbp_only = (pbp_1999_teams | pbp_2000_teams) - games_teams
print(f"\n{'='*80}")
print("MISMATCHES")
print("="*80)
print(f"Teams in games but not in pbp: {sorted(games_only)}")
print(f"Teams in pbp but not in games: {sorted(pbp_only)}")

# Known franchise relocations
relocations = {
    'OAK': 'LV',  # Oakland Raiders -> Las Vegas Raiders (2020)
    'SD': 'LAC',  # San Diego Chargers -> Los Angeles Chargers (2017)
    'STL': 'LA',  # St. Louis Rams -> Los Angeles Rams (2016)
}

print(f"\n{'='*80}")
print("KNOWN FRANCHISE RELOCATIONS")
print("="*80)
for old, new in relocations.items():
    print(f"{old} -> {new}")

# Investigate the 4 specific rows: 1999 BAL, 2000 BUF, 2000 KC, 2000 MIA
print(f"\n{'='*80}")
print("INVESTIGATING 4 SPECIFIC ROWS")
print("="*80)

# 1999 BAL games
bal_1999_games = games[(games['season'] == 1999) &
                       ((games['home_team'] == 'BAL') | (games['away_team'] == 'BAL')) &
                       (games['home_score'].notna()) & (games['away_score'].notna())]
print(f"\n1999 BAL: {len(bal_1999_games)} completed games")
print(bal_1999_games[['game_id', 'season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].to_string())

# Check pbp for each game
for game_id in bal_1999_games['game_id']:
    pbp_plays = pbp_1999[(pbp_1999['game_id'] == game_id) &
                         ((pbp_1999['posteam'] == 'BAL') | (pbp_1999['defteam'] == 'BAL')) &
                         (pbp_1999['play'] == 1)]
    offense_plays = pbp_plays[pbp_plays['posteam'] == 'BAL']
    print(f"  {game_id}: {len(offense_plays)} offensive plays for BAL")

# 2000 games for BUF, KC, MIA
for team in ['BUF', 'KC', 'MIA']:
    team_2000_games = games[(games['season'] == 2000) &
                            ((games['home_team'] == team) | (games['away_team'] == team)) &
                            (games['home_score'].notna()) & (games['away_score'].notna())]
    print(f"\n2000 {team}: {len(team_2000_games)} completed games")
    print(team_2000_games[['game_id', 'season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].to_string())

    # Check pbp for each game
    for game_id in team_2000_games['game_id']:
        pbp_plays = pbp_2000[(pbp_2000['game_id'] == game_id) &
                             ((pbp_2000['posteam'] == team) | (pbp_2000['defteam'] == team)) &
                             (pbp_2000['play'] == 1)]
        offense_plays = pbp_plays[pbp_plays['posteam'] == team]
        if len(offense_plays) == 0:
            print(f"  {game_id}: {len(offense_plays)} offensive plays for {team} *** ZERO ***")
            # Show what teams are in this game
            game_teams = pbp_2000[pbp_2000['game_id'] == game_id][['posteam', 'defteam']].drop_duplicates()
            print(f"    Teams in pbp: {game_teams.to_dict('records')}")

# Build comprehensive franchise mapping
print(f"\n{'='*80}")
print("BUILDING COMPREHENSIVE FRANCHISE MAPPING")
print("="*80)

# Start with all teams that appear in games
franchise_map = {}
for team in sorted(games_teams):
    # Check if this is a relocated team
    if team in relocations:
        franchise_map[team] = relocations[team]
        print(f"{team} -> {relocations[team]} (relocation)")
    # Check if this team appears in pbp (means it's current)
    elif team in pbp_1999_teams or team in pbp_2000_teams:
        franchise_map[team] = team
        print(f"{team} -> {team} (identity)")
    else:
        franchise_map[team] = team
        print(f"{team} -> {team} (identity, not in 1999-2000 pbp)")

# Also include pbp-only teams (new abbreviations)
for team in sorted(pbp_1999_teams | pbp_2000_teams):
    if team not in franchise_map:
        franchise_map[team] = team
        print(f"{team} -> {team} (identity, pbp only)")

print(f"\nTotal mappings: {len(franchise_map)}")

# Save to CSV
print(f"\n{'='*80}")
print("SAVING FRANCHISE MAP TO CSV")
print("="*80)
df_map = pd.DataFrame(sorted(franchise_map.items()), columns=['team_abbr', 'franchise_abbr'])
df_map.to_csv('team_franchise_map.csv', index=False)
print(f"Saved {len(df_map)} rows to team_franchise_map.csv")
print(df_map.to_string())
