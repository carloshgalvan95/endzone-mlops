{{
    config(
        materialized='view',
        tags=['staging', 'nflverse', 'games']
    )
}}

/*
Staging model for nflverse games (schedule) data.

Source: nflverse games.parquet
License: CC BY 4.0
Coverage: NFL games from 1999-present

This model provides a cleaned view of game-level data including:
- Game identifiers and matchup details
- Dates and times
- Final scores
- Betting lines (spread, total, moneyline)
- Weather and venue information

Day 1 Note: This model defines the schema for when raw data is loaded.
Currently serves as documentation and dbt compile test.
*/

with source as (

    -- Placeholder: Replace with actual source reference when data is loaded
    -- For Day 1, this compiles but won't run without data
    select
        game_id,
        season,
        week,
        game_type,
        away_team,
        home_team,
        away_score,
        home_score,
        gameday,
        weekday,
        gametime,
        
        -- Betting lines (closing)
        spread_line,
        total_line,
        away_moneyline,
        home_moneyline,
        
        -- Venue and conditions
        location,
        roof,
        surface,
        temp,
        wind,
        
        -- Rest days
        away_rest,
        home_rest,
        
        -- Cross-reference IDs (for future joins with other data sources)
        gsis_id,
        pfr_id,
        pff_id,
        espn_id,
        
        -- Metadata
        old_game_id
        
    from {{ source('nflverse', 'raw_games') }}
    
    -- Note: source('nflverse', 'raw_games') must be defined in sources.yml
    -- For Day 1 local compile: this model documents the expected schema

)

select * from source
