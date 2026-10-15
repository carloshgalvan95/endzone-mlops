{{
    config(
        materialized='table',
        tags=['marts', 'qb', 'starter']
    )
}}

/*
Expected starting quarterback for each completed game.

Grain: one row per completed game.
The starter is the offense QB listed at depth_team 1 on that week's chart.
The join is season, week, game_type, and club_code = home_team or away_team.

This model does not read play-by-play. Who threw in the game is not the starter.

A missing chart leaves the starter id null. Python then uses the previous
game's primary passer. A week with two QB1 rows keeps the lowest gsis_id.
Non-game chart rows (null week, including the conference bye) are dropped.
depth_team 2 is not promoted when QB1 is absent.

From 2025 the chart week is not on the nflverse file. Ingest assigns it.
The row is the latest snapshot with dt strictly before that game's kickoff,
so a Thursday game does not see a Friday scrape. CV folds are 2016-2023 and
do not use this mapping. 2025 is the held-out test. If that season was
skipped, both starter ids stay null and the feature code uses the
previous-game primary passer.
*/

with games as (

    select
        game_id,
        season,
        week,
        game_type,
        gameday,
        home_team,
        away_team,
        home_team_franchise,
        away_team_franchise
    from {{ ref('fct_games') }}

),

qb1 as (

    select
        cast(season as int) as season,
        cast(week as int) as week,
        game_type,
        club_code,
        min(gsis_id) as starter_player_id
    from {{ ref('stg_nflverse__depth_charts') }}
    where position = 'QB'
      and formation = 'Offense'
      and depth_team = '1'
      and week is not null
      and gsis_id is not null
    group by
        cast(season as int),
        cast(week as int),
        game_type,
        club_code

)

select
    games.game_id,
    games.season,
    games.week,
    games.game_type,
    games.gameday,
    games.home_team,
    games.away_team,
    games.home_team_franchise,
    games.away_team_franchise,
    home_qb.starter_player_id as home_starter_player_id,
    away_qb.starter_player_id as away_starter_player_id
from games
left join qb1 as home_qb
    on games.season = home_qb.season
    and cast(games.week as int) = home_qb.week
    and games.game_type = home_qb.game_type
    and games.home_team = home_qb.club_code
left join qb1 as away_qb
    on games.season = away_qb.season
    and cast(games.week as int) = away_qb.week
    and games.game_type = away_qb.game_type
    and games.away_team = away_qb.club_code
