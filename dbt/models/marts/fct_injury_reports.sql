{{
    config(
        materialized='table',
        tags=['marts', 'injuries']
    )
}}

/*
Weekly injury report rows for the pre-game absence features.

Grain: one row per player report. A player-week can have more than one
row when date_modified differs. Do not collapse to the latest row here.
A later timestamp can fall after kickoff, and the earlier row is then the
last report that was known before the game.

Null date_modified stays on the table. The feature code drops it, because
a missing clock is not evidence the row was known before kickoff.
*/

select
    cast(season as int) as season,
    cast(round(try_cast(week as double)) as int) as week,
    trim(cast(game_type as string)) as game_type,
    trim(cast(team as string)) as team,
    nullif(trim(replace(cast(gsis_id as string), chr(10), '')), '') as gsis_id,
    nullif(trim(replace(cast(report_status as string), chr(10), '')), '') as report_status,
    cast(date_modified as timestamp) as date_modified
from {{ ref('stg_nflverse__injuries') }}
where nullif(trim(replace(cast(gsis_id as string), chr(10), '')), '') is not null
  and week is not null
  and nullif(trim(cast(team as string)), '') is not null
