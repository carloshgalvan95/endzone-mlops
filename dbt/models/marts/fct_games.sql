{{
    config(
        materialized='table',
        tags=['marts', 'games', 'results']
    )
}}

/*
Fact table: completed games with derived results and betting outcomes.

Grain: One row per game (COMPLETED GAMES ONLY).
Purpose: Provide game-level features and outcomes for model training and analysis.

Filter: Only games with both home_score and away_score not null (completed games).

Key fields:
- Game identifiers and context (season, week, teams)
- Final scores and derived outcomes (winner, margin, total points)
- Betting line results (spread covered, total over/under)
- Game metadata (date, venue conditions)

Useful for:
- Model training labels (did home team win, by how much)
- Feature engineering (recent game results, strength of schedule)
- Model evaluation and backtesting against betting lines
*/

with games as (

    select
        game_id,
        season,
        week,
        game_type,
        away_team,
        home_team,
        away_score,
        home_score,
        spread_line,
        total_line,
        away_moneyline,
        home_moneyline,
        gameday,
        weekday,
        gametime,
        location,
        roof,
        surface,
        temp,
        wind,
        away_rest,
        home_rest
    from {{ ref('stg_nflverse__games') }}
    where home_score is not null
      and away_score is not null

),

final as (

    select
        game_id,
        season,
        week,
        game_type,
        away_team,
        home_team,
        away_score,
        home_score,
        
        -- Derived outcomes
        case
            when home_score > away_score then 1
            else 0
        end as home_won,
        
        home_score - away_score as home_margin,
        
        home_score + away_score as total_points,
        
        -- Betting outcomes (null if lines missing)
        case
            when spread_line is null then null
            when (home_score - away_score) > spread_line then 1
            when (home_score - away_score) < spread_line then 0
            else null
        end as home_covered_spread,
        
        case
            when total_line is null then null
            when (home_score + away_score) > total_line then 1
            when (home_score + away_score) < total_line then 0
            else null
        end as total_over,
        
        -- Betting lines
        spread_line,
        total_line,
        away_moneyline,
        home_moneyline,
        
        -- Game context
        gameday,
        weekday,
        gametime,
        location,
        roof,
        surface,
        temp,
        wind,
        away_rest,
        home_rest

    from games

)

select * from final
