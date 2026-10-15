# EPA play filter check

Staging table: ``workspace`.`nfl_dev_staging`.`stg_nflverse__pbp``.
Rows with `play = 1`, grouped by `play_type`.
This is the filter `fct_team_game_epa` uses before it averages EPA.
No dbt model was changed.

## Play types

| play_type | n_plays | share |
| --- | --- | --- |
| pass | 528368 | 0.5362 |
| run | 388724 | 0.3945 |
| no_play | 68233 | 0.0692 |

`qb_kneel` and `qb_spike` are not both on ``workspace`.`nfl_dev_staging`.`stg_nflverse__pbp``. Counts come from ``workspace`.`nfl_raw`.`play_by_play``.

`n_flag_1` counts rows where the flag equals 1.
`n_flag_1_and_play_1` counts those rows that also have `play = 1`.

| column | n_flag_1 | n_flag_1_and_play_1 |
| --- | --- | --- |
| qb_kneel | 9965 | 36 |
| qb_spike | 1985 | 70 |
