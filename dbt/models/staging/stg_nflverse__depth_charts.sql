{{
    config(
        materialized='view',
        tags=['staging', 'nflverse', 'depth_charts']
    )
}}

/*
Staging model for nflverse weekly depth charts.

Source: nflverse depth_charts_{season}.parquet
License: CC BY 4.0
Coverage: 2001-present. 1999 and 2000 were not published.

The listed starter is position QB, formation Offense, depth_team 1.
This view does not decide that. The expected-starter mart does.
Week is null on non-game rows such as the conference bye.

Raw data lands in workspace.nfl_raw.depth_charts via
src/ingest/load_raw_nflverse.py --dataset depth_charts.
*/

with source as (

    select
        season,
        club_code,
        week,
        game_type,
        depth_team,
        last_name,
        first_name,
        football_name,
        formation,
        gsis_id,
        jersey_number,
        position,
        elias_id,
        depth_position,
        full_name

    from {{ source('nflverse', 'raw_depth_charts') }}

)

select * from source
