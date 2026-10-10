{{
    config(
        materialized='table',
        tags=['marts', 'games', 'results']
    )
}}

/*
Fact table: completed games with derived results and betting outcomes.

Grain: One row per game.
Purpose: Provide game-level features and outcomes for model training and analysis.

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
            when home_score is null or away_score is null then null
            when home_score > away_score then 1
            else 0
        end as home_won,
        
        case
            when home_score is null or away_score is null then null
            else home_score - away_score
        end as home_margin,
        
        case
            when home_score is null or away_score is null then null
            else home_score + away_score
        end as total_points,
        
        -- Betting outcomes (null if scores or lines missing)
        case
            when spread_line is null or home_score is null or away_score is null then null
            when (home_score - away_score) > spread_line then 1  -- Home covered
            when (home_score - away_score) < spread_line then 0  -- Home did not cover
            else null  -- Push
        end as home_covered_spread,
        
        case
            when total_line is null or home_score is null or away_score is null then null
            when (home_score + away_score) > total_line then 1  -- Over
            when (home_score + away_score) < total_line then 0  -- Under
            else null  -- Push
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
