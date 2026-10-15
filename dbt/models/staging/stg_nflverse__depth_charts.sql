{{
    config(
        materialized='view',
        tags=['staging', 'nflverse', 'depth_charts']
    )
}}

/*
Staging model for nflverse depth charts.

Source: nflverse depth_charts_{season}.parquet
License: CC BY 4.0
Coverage: 2001-present. 1999 and 2000 were not published.

2001-2024 files are already weekly. From 2025 the release is a timestamped
snapshot (dt, team, pos_grp, pos_abb, pos_rank) and has no week. Ingest maps
that file onto these columns before upload. Each team-game keeps the latest
snapshot with dt strictly before kickoff. Kickoff is gameday plus gametime
in Eastern time. team becomes club_code, pos_abb becomes position (QB stays
QB), pos_rank becomes depth_team, and pos_grp becomes formation (3WR 1TE is
Offense, Base 4-3 D and Base 3-4 D are Defense, Special Teams is unchanged).

The listed starter is position QB, formation Offense, depth_team 1.
This view does not decide that. The expected-starter mart does.
Week is null on non-game rows such as the conference bye. Mapped 2025 rows
are only team-games, so that season has no null week.

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
