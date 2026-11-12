{{
    config(
        materialized='view',
        tags=['staging', 'nflverse', 'injuries']
    )
}}

/*
Staging model for nflverse weekly injury reports.

Source: nflverse injuries_{season}.parquet
License: CC BY 4.0
Coverage: 2009-present. Earlier seasons were not published.

Ingest writes one schema for every season. date_modified is the report
update time in UTC. The 2025 and 2026 releases do not publish that column,
so ingest stores it null. This view does not drop those rows and does not
compare the timestamp to kickoff. The feature code keeps the last row with
date_modified strictly before kickoff.

Raw data lands in workspace.nfl_raw.injuries via
endzone/ingest/load_raw_nflverse.py --dataset injuries.
*/

with source as (

    select
        season,
        game_type,
        team,
        week,
        gsis_id,
        position,
        full_name,
        first_name,
        last_name,
        report_primary_injury,
        report_secondary_injury,
        report_status,
        practice_primary_injury,
        practice_secondary_injury,
        practice_status,
        date_modified

    from {{ source('nflverse', 'raw_injuries') }}

)

select * from source
