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

Raw data lands in workspace.nfl_raw.games via src/ingest/load_raw_nflverse.py.
Cross-reference IDs are renamed to *_id here (raw columns: gsis, pfr, pff, espn).
*/

with source as (

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
        gsis as gsis_id,
        pfr as pfr_id,
        pff as pff_id,
        espn as espn_id,
        
        -- Metadata
        old_game_id
        
    from {{ source('nflverse', 'raw_games') }}

)

select * from source
