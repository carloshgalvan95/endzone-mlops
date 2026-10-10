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

)

select * from games
