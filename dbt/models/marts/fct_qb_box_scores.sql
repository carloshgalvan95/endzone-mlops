{{
    config(
        materialized='table',
        tags=['marts', 'qb', 'box_scores']
    )
}}

/*
Pass and rush counting stats for one player in one completed game.

Grain: one row per game_id and player_id.
The QB VALUE formula is not applied here. Python applies it so there
is one implementation. This is who produced the counts. It is not the
expected starter. The starter comes from the depth chart.

A pass line is grouped by passer and possession team. It counts pass
attempts, completions, passing yards, passing touchdowns, interceptions,
and sacks. A sack keeps the passer even when pass_attempt is 0.
A rush line is grouped by rusher on rush_attempt = 1.
The two lines are joined on game, player, and team. A player with two
teams in one game keeps the team with more pass attempts, then more
rush attempts, then the alphabetically first team. The counts still add
every team.
*/

with games as (

    select
        game_id,
        season,
        week,
        gameday
    from {{ ref('fct_games') }}

),

passes as (

    select
        game_id,
        passer_player_id as player_id,
        possession_team as team,
        sum(case when pass_attempt = 1 then 1 else 0 end) as pass_attempts,
        sum(case when complete_pass = 1 then 1 else 0 end) as completions,
        sum(coalesce(passing_yards, 0)) as passing_yards,
        sum(case when pass_touchdown = 1 then 1 else 0 end) as passing_tds,
        sum(case when interception = 1 then 1 else 0 end) as interceptions,
        sum(case when sack = 1 then 1 else 0 end) as sacks
    from {{ ref('stg_nflverse__pbp') }}
    where passer_player_id is not null
      and possession_team is not null
      and (
          pass_attempt = 1
          or complete_pass = 1
          or pass_touchdown = 1
          or interception = 1
          or sack = 1
      )
    group by game_id, passer_player_id, possession_team

),

rushes as (

    select
        game_id,
        rusher_player_id as player_id,
        possession_team as team,
        sum(case when rush_attempt = 1 then 1 else 0 end) as rush_attempts,
        sum(coalesce(rushing_yards, 0)) as rushing_yards,
        sum(case when rush_touchdown = 1 then 1 else 0 end) as rushing_tds
    from {{ ref('stg_nflverse__pbp') }}
    where rush_attempt = 1
      and rusher_player_id is not null
      and possession_team is not null
    group by game_id, rusher_player_id, possession_team

),

combined as (

    select
        coalesce(passes.game_id, rushes.game_id) as game_id,
        coalesce(passes.player_id, rushes.player_id) as player_id,
        coalesce(passes.team, rushes.team) as team,
        coalesce(passes.pass_attempts, 0) as pass_attempts,
        coalesce(passes.completions, 0) as completions,
        coalesce(passes.passing_yards, 0) as passing_yards,
        coalesce(passes.passing_tds, 0) as passing_tds,
        coalesce(passes.interceptions, 0) as interceptions,
        coalesce(passes.sacks, 0) as sacks,
        coalesce(rushes.rush_attempts, 0) as rush_attempts,
        coalesce(rushes.rushing_yards, 0) as rushing_yards,
        coalesce(rushes.rushing_tds, 0) as rushing_tds
    from passes
    full outer join rushes
        on passes.game_id = rushes.game_id
        and passes.player_id = rushes.player_id
        and passes.team = rushes.team

),

ranked as (

    select
        game_id,
        player_id,
        team,
        pass_attempts,
        completions,
        passing_yards,
        passing_tds,
        interceptions,
        sacks,
        rush_attempts,
        rushing_yards,
        rushing_tds,
        row_number() over (
            partition by game_id, player_id
            order by pass_attempts desc, rush_attempts desc, team asc
        ) as team_rank
    from combined

),

collapsed as (

    select
        game_id,
        player_id,
        max(case when team_rank = 1 then team end) as team,
        sum(pass_attempts) as pass_attempts,
        sum(completions) as completions,
        sum(passing_yards) as passing_yards,
        sum(passing_tds) as passing_tds,
        sum(interceptions) as interceptions,
        sum(sacks) as sacks,
        sum(rush_attempts) as rush_attempts,
        sum(rushing_yards) as rushing_yards,
        sum(rushing_tds) as rushing_tds
    from ranked
    group by game_id, player_id

)

select
    collapsed.game_id,
    games.season,
    games.week,
    games.gameday,
    collapsed.team,
    franchise_map.franchise_abbr as team_franchise,
    collapsed.player_id,
    collapsed.pass_attempts,
    collapsed.completions,
    collapsed.passing_yards,
    collapsed.passing_tds,
    collapsed.interceptions,
    collapsed.sacks,
    collapsed.rush_attempts,
    collapsed.rushing_yards,
    collapsed.rushing_tds
from collapsed
inner join games
    on collapsed.game_id = games.game_id
left join {{ ref('team_franchise_map') }} as franchise_map
    on collapsed.team = franchise_map.team_abbr
