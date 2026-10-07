{{
    config(
        materialized='view',
        tags=['staging', 'nflverse', 'pbp']
    )
}}

/*
Staging model for nflverse play-by-play data.

Source: nflverse play_by_play_{season}.parquet
License: CC BY 4.0
Coverage: NFL plays from 1999-present (~372 columns per play)

This model provides a focused view of play-level data for game modeling:
- Play and game identifiers
- Team and situation context
- Play outcomes and yards
- EPA (Expected Points Added) and WPA (Win Probability Added) metrics
- Drive and series information

Column selection focuses on features useful for pregame prediction and
in-game modeling, excluding granular player-level data and detailed
tackle/fumble tracking.

Raw data lands in workspace.nfl_raw.play_by_play via src/ingest/load_raw_nflverse.py.
*/

with source as (

    select
        -- Play identifiers
        play_id,
        game_id,
        old_game_id,
        
        -- Game context
        season,
        week,
        game_date,
        season_type,
        
        -- Teams
        posteam as possession_team,
        defteam as defensive_team,
        home_team,
        away_team,
        
        -- Play situation
        down,
        ydstogo as yards_to_go,
        yardline_100 as yards_to_endzone,
        qtr as quarter,
        quarter_seconds_remaining,
        half_seconds_remaining,
        game_seconds_remaining,
        game_half,
        
        -- Play type and outcome
        play_type,
        desc as play_description,
        yards_gained,
        first_down,
        touchdown,
        
        -- EPA and WPA metrics (core value)
        epa,
        wpa,
        success,
        ep as expected_points,
        wp as win_probability,
        
        -- Pass-specific
        pass_attempt,
        complete_pass,
        incomplete_pass,
        interception,
        air_yards,
        yards_after_catch,
        
        -- Rush-specific
        rush_attempt,
        
        -- Scoring plays
        field_goal_attempt,
        field_goal_result,
        extra_point_attempt,
        extra_point_result,
        two_point_attempt,
        two_point_conv_result,
        
        -- Penalties
        penalty,
        penalty_team,
        penalty_yards,
        
        -- Drive context
        drive,
        fixed_drive,
        series,
        series_success,
        
        -- Score state
        posteam_score as possession_team_score,
        defteam_score as defensive_team_score,
        score_differential,
        posteam_score_post as possession_team_score_post,
        defteam_score_post as defensive_team_score_post,
        
        -- Timeouts
        posteam_timeouts_remaining as possession_team_timeouts,
        defteam_timeouts_remaining as defensive_team_timeouts,
        
        -- Special teams
        punt_attempt,
        kickoff_attempt,
        
        -- Play flags
        play,
        aborted_play,
        
        -- Order
        order_sequence
        
    from {{ source('nflverse', 'raw_play_by_play') }}

)

select * from source
