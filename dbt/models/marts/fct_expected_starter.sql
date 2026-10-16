{{
    config(
        materialized='table',
        tags=['marts', 'qb', 'starter']
    )
}}

/*
Expected starting quarterback for each completed game.

Grain: one row per completed game.
The starter is the offense QB listed at depth 1 on that week's chart.
The join is season, week, game_type, and the schedule team abbreviation.

This model does not read play-by-play. Who threw in the game is not the starter.

Week is cast through double so an int 17 and a double 17.0 both match the
schedule. depth_team accepts 1, '1', and '1.0'. A legacy club code
(ARZ, BLT, CLV, HST, JAC, LAR, and a relocated code in the wrong years)
is translated with depth_chart_club_map. The current release already uses
the schedule codes, so that translation does not change those rows.

When several offense QB1 rows share a team-week, a depth_position of QB
wins over a blank or a different position (a tight end or an injured
player listed with an empty position). The remaining tie keeps the lowest
gsis_id. An exact club_code match beats a row that only matched through
the alias. depth 2 is not promoted when QB1 is absent.

A missing chart leaves the starter id null. Python then uses the previous
game's primary passer. Non-game chart rows (null week, including the
conference bye) are dropped.

From 2025 the chart week is not on the nflverse file. Ingest assigns it
and writes the same week type as the weekly files. The row is the latest
snapshot with dt strictly before that game's kickoff, so a Thursday game
does not see a Friday scrape. CV folds are 2016-2023 and do not use this
mapping. 2025 is the held-out test. If that season was skipped, both
starter ids stay null and the feature code uses the previous-game primary
passer.
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

chart as (

    select
        cast(season as int) as season,
        cast(round(try_cast(week as double)) as int) as week,
        trim(cast(game_type as string)) as game_type,
        trim(cast(club_code as string)) as club_code,
        nullif(trim(cast(gsis_id as string)), '') as gsis_id,
        trim(
            replace(coalesce(cast(depth_position as string), ''), chr(10), '')
        ) as depth_position_norm,
        cast(round(try_cast(depth_team as double)) as int) as depth_rank
    from {{ ref('stg_nflverse__depth_charts') }}
    where trim(cast(position as string)) = 'QB'
      and trim(cast(formation as string)) = 'Offense'
      and week is not null
      and gsis_id is not null

),

qb_rows as (

    select
        chart.season,
        chart.week,
        chart.game_type,
        chart.club_code,
        chart.gsis_id,
        coalesce(aliases.schedule_team, chart.club_code) as schedule_team,
        case
            when chart.depth_position_norm = 'QB' then 0
            when chart.depth_position_norm = '' then 1
            else 2
        end as position_rank
    from chart
    left join {{ ref('depth_chart_club_map') }} as aliases
        on chart.club_code = aliases.club_code
        and chart.season between aliases.season_start and aliases.season_end
    where chart.depth_rank = 1
      and chart.gsis_id is not null
      and chart.week is not null

),

qb1 as (

    select
        season,
        week,
        game_type,
        schedule_team,
        gsis_id as starter_player_id
    from (
        select
            season,
            week,
            game_type,
            club_code,
            schedule_team,
            gsis_id,
            row_number() over (
                partition by season, week, game_type, schedule_team
                order by
                    case when schedule_team = club_code then 0 else 1 end,
                    position_rank,
                    gsis_id
            ) as starter_rank
        from qb_rows
    ) as ranked
    where starter_rank = 1

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
    and games.home_team = home_qb.schedule_team
left join qb1 as away_qb
    on games.season = away_qb.season
    and cast(games.week as int) = away_qb.week
    and games.game_type = away_qb.game_type
    and games.away_team = away_qb.schedule_team
