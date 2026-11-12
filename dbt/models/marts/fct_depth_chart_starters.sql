{{
    config(
        materialized='table',
        tags=['marts', 'depth_charts', 'injuries']
    )
}}

/*
Depth-chart rank 1 for every listed position.

Grain: one row per season, week, game_type, schedule team, player, and
position. The injury feature keeps OL, skill (RB, WR, TE), front seven,
and DB, and drops QB. That filter is in the feature code, not here, so a
QB row can be shown to be excluded.

Week is cast through double so an int 17 and a double 17.0 both match.
depth_team accepts 1, '1', and '1.0'. A legacy club code is translated
with depth_chart_club_map. An exact club_code match beats an alias.
Non-game chart rows (null week) are dropped. Depth 2 is not promoted.
*/

with chart as (

    select
        cast(season as int) as season,
        cast(round(try_cast(week as double)) as int) as week,
        trim(cast(game_type as string)) as game_type,
        trim(cast(club_code as string)) as club_code,
        nullif(trim(cast(gsis_id as string)), '') as gsis_id,
        trim(cast(position as string)) as position,
        cast(round(try_cast(depth_team as double)) as int) as depth_rank
    from {{ ref('stg_nflverse__depth_charts') }}
    where week is not null
      and gsis_id is not null

),

ranked as (

    select
        chart.season,
        chart.week,
        chart.game_type,
        coalesce(aliases.schedule_team, chart.club_code) as team,
        chart.gsis_id,
        chart.position,
        row_number() over (
            partition by
                chart.season,
                chart.week,
                chart.game_type,
                coalesce(aliases.schedule_team, chart.club_code),
                chart.gsis_id,
                chart.position
            order by
                case
                    when coalesce(aliases.schedule_team, chart.club_code) = chart.club_code
                        then 0
                    else 1
                end,
                chart.gsis_id
        ) as starter_rank
    from chart
    left join {{ ref('depth_chart_club_map') }} as aliases
        on chart.club_code = aliases.club_code
        and chart.season between aliases.season_start and aliases.season_end
    where chart.depth_rank = 1
      and chart.gsis_id is not null
      and chart.week is not null
      and chart.position is not null
      and chart.position != ''

)

select
    season,
    week,
    game_type,
    team,
    gsis_id,
    position
from ranked
where starter_rank = 1
