{{
    config(
        materialized='table',
        tags=['marts', 'qb', 'dropbacks']
    )
}}

/*
Dropbacks by passer and completed game.

Grain: one row per game_id and passer_player_id.
This table records who played. It is not the starter.
The expected starter comes from the depth chart, in fct_expected_starter.

A dropback is qb_dropback = 1 with a passer and a possession team.
epa is summed with null treated as 0 so the dropback still counts.
cpoe_sum and cpoe_dropbacks are stored for a later rating. H5 does not
blend CPOE into EPA per dropback. The units differ, and a blend weight
would be a second hypothesis.

If one passer has dropbacks for two possession teams in the same game,
the row keeps the team with more dropbacks. The dropback and epa totals
still add every team. Ties keep the alphabetically first team.
*/

with games as (

    select
        game_id,
        season,
        week,
        gameday
    from {{ ref('fct_games') }}

),

plays as (

    select
        game_id,
        passer_player_id,
        possession_team as team,
        coalesce(epa, 0) as epa,
        cpoe
    from {{ ref('stg_nflverse__pbp') }}
    where qb_dropback = 1
      and passer_player_id is not null
      and possession_team is not null

),

by_team as (

    select
        game_id,
        passer_player_id,
        team,
        count(*) as dropbacks,
        sum(epa) as dropback_epa,
        sum(coalesce(cpoe, 0)) as cpoe_sum,
        sum(case when cpoe is not null then 1 else 0 end) as cpoe_dropbacks
    from plays
    group by game_id, passer_player_id, team

),

ranked as (

    select
        game_id,
        passer_player_id,
        team,
        dropbacks,
        dropback_epa,
        cpoe_sum,
        cpoe_dropbacks,
        row_number() over (
            partition by game_id, passer_player_id
            order by dropbacks desc, team asc
        ) as team_rank
    from by_team

),

collapsed as (

    select
        game_id,
        passer_player_id,
        max(case when team_rank = 1 then team end) as team,
        sum(dropbacks) as dropbacks,
        sum(dropback_epa) as dropback_epa,
        sum(cpoe_sum) as cpoe_sum,
        sum(cpoe_dropbacks) as cpoe_dropbacks
    from ranked
    group by game_id, passer_player_id

)

select
    collapsed.game_id,
    games.season,
    games.week,
    games.gameday,
    collapsed.team,
    franchise_map.franchise_abbr as team_franchise,
    collapsed.passer_player_id,
    collapsed.dropbacks,
    collapsed.dropback_epa,
    collapsed.cpoe_sum,
    collapsed.cpoe_dropbacks
from collapsed
inner join games
    on collapsed.game_id = games.game_id
left join {{ ref('team_franchise_map') }} as franchise_map
    on collapsed.team = franchise_map.team_abbr
