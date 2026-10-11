{{
    config(
        materialized='table',
        tags=['marts', 'games', 'upcoming']
    )
}}

/*
Fact table: upcoming games (scheduled but not yet completed).

Grain: One row per game (UPCOMING GAMES ONLY).
Purpose: Provide game-level features for pregame predictions (scoring input).

Filter: Only games where home_score and away_score are both null (not yet played).

Key fields:
- Game identifiers and context (season, week, teams)
- Schedule information (gameday, gametime)
- Betting lines (spread, total, moneylines)
- Venue and conditions (roof, surface, rest days)
- Franchise columns for grouping teams across relocations (OAK->LV, SD->LAC, STL->LA)

Useful for:
- Batch scoring input (games to predict)
- Weekly forecast refresh
*/

with games as (

    select
        game_id,
        season,
        week,
        game_type,
        away_team,
        home_team,
        gameday,
        weekday,
        gametime,
        spread_line,
        total_line,
        away_moneyline,
        home_moneyline,
        location,
        roof,
        surface,
        temp,
        wind,
        away_rest,
        home_rest
    from {{ ref('stg_nflverse__games') }}
    where home_score is null
      and away_score is null

),

franchise_map as (

    select
        team_abbr,
        franchise_abbr
    from {{ ref('team_franchise_map') }}

)

select
    g.game_id,
    g.season,
    g.week,
    g.game_type,
    g.away_team,
    g.home_team,
    
    -- Franchise columns for grouping across relocations
    af.franchise_abbr as away_team_franchise,
    hf.franchise_abbr as home_team_franchise,
    
    g.gameday,
    g.weekday,
    g.gametime,
    g.spread_line,
    g.total_line,
    g.away_moneyline,
    g.home_moneyline,
    g.location,
    g.roof,
    g.surface,
    g.temp,
    g.wind,
    g.away_rest,
    g.home_rest
from games g
left join franchise_map af
    on g.away_team = af.team_abbr
left join franchise_map hf
    on g.home_team = hf.team_abbr
