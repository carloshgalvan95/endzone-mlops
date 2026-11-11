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
- Franchise columns for grouping teams across relocations (OAK->LV, SD->LAC, STL->LA)

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

franchise_map as (

    select
        team_abbr,
        franchise_abbr
    from {{ ref('team_franchise_map') }}

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
        
        -- Franchise columns for grouping across relocations
        af.franchise_abbr as away_team_franchise,
        hf.franchise_abbr as home_team_franchise,
        
        -- Derived outcomes. The casts keep the contract types stable:
        -- int for the 0/1 flags, double for the score arithmetic.
        cast(
            case
                when home_score > away_score then 1
                else 0
            end
            as int
        ) as home_won,
        
        cast(home_score - away_score as double) as home_margin,
        
        cast(home_score + away_score as double) as total_points,
        
        -- Betting outcomes (null if lines missing)
        cast(
            case
                when spread_line is null then null
                when (home_score - away_score) > spread_line then 1
                when (home_score - away_score) < spread_line then 0
                else null
            end
            as int
        ) as home_covered_spread,
        
        cast(
            case
                when total_line is null then null
                when (home_score + away_score) > total_line then 1
                when (home_score + away_score) < total_line then 0
                else null
            end
            as int
        ) as total_over,
        
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
    left join franchise_map af
        on away_team = af.team_abbr
    left join franchise_map hf
        on home_team = hf.team_abbr

)

select * from final
