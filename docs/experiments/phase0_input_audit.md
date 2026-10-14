# Phase 0 input audit

Diagnostic only. This note records what the current code does. It does not change features, and it does not report measurements from the warehouse. Each item is **verified** from the code in this repo, or **pending** a run of `make audit-features` on the marts (or a check this repo does not contain).

No counts, rates, or model metrics are stated here. Those belong in the report the audit writes when it is run.

The audit module is `src/analysis/feature_audit.py`. It calls `build_pregame_features`, `prepare_training_data`, `split_by_season`, and `train_logistic_regression`. Leakage tests live in `tests/test_feature_leakage.py` and use synthetic frames only.

`make audit-features` (or `.\scripts\dev.ps1 audit-features`) writes `reports/phase0/feature_audit.md` and `reports/phase0/feature_audit.csv`. It also logs both files to MLflow experiment `nfl_pregame/phase0_audit` on a run tagged `phase=0-audit`. 2025 is in the missingness and distribution tables only. It is not scored and it is not used to choose a model. Univariate CV defaults to seasons 2016 through 2023. Permutation importance fits the current logistic pipeline on seasons through 2023 and scores it on 2024.

## Checklist

### 1. games_played is a clock

**Verified (definition). Range shift on 2024 and 2025: pending.**

`home_games_played` is `len(home_prior)`. `home_prior` is every prior team-game with `gameday` strictly before the game. It is not filtered to the current season. `away_games_played` is the same count for the away team. The value grows with the length of the history in the frame. It is not "games played this season."

The audit flags `time_counter` when season medians are strictly monotone across at least three seasons and the 2024 or 2025 values sit entirely outside the training min/max. Whether `games_played` trips that flag on the marts is pending the audit run.

### 2. Cross-season rolling, no regression to the mean

**Verified.**

Rolling offensive EPA, defensive EPA, and win rate use `tail(lookback_games)` on all prior games for that franchise. The slice is not limited to the current season. Nothing in `build_pregame_features` shrinks a feature toward a league mean, and there is no exponential decay.

The only feature limited to the current `season` column is season-to-date offensive EPA. A week-1 game therefore has a missing season EPA and a rolling window that can be entirely prior-season games. The leakage tests lock that behavior on a synthetic frame.

### 3. No defensive season EPA

**Verified.**

Season features are `home_season_offensive_epa_per_play` and `away_season_offensive_epa_per_play` only. Defensive EPA is the rolling columns `home_recent_defensive_epa_per_play` and `away_recent_defensive_epa_per_play`, taken from `avg_defensive_epa_allowed`. There is no season-to-date defensive feature in `PREGAME_FEATURE_COLUMNS`.

### 4. No home-minus-away differentials

**Verified.**

Each model input is a separate home column or away column. `build_pregame_features` and `prepare_training_data` do not subtract away from home.

### 5. location and game_type are ignored

**Verified.**

`fct_games` selects `location` and `game_type` from staging and keeps them on the mart. `build_pregame_features` does not read either column. They are not in `PREGAME_FEATURE_COLUMNS`, so they are not in the training matrix. Neutral sites and playoff games are not marked on the model input.

### 6. Ties are coded as home losses

**Verified (label rule). How many ties are in the mart: pending.**

In `fct_games`, `home_won` is 1 only when `home_score > away_score`. Every other completed game, including a tie, is 0. The recent win rate then treats `home_won == 0` as a win for the away franchise, so a tie also counts as an away win when that game is inside the lookback window. This note does not count ties.

### 7. pass_epa and rush_epa are unused

**Verified.**

`fct_team_game_epa` computes `pass_epa` and `rush_epa` (sums of EPA on `pass_attempt = 1` and `rush_attempt = 1` inside the `play = 1` rows). `build_pregame_features` reads `avg_offensive_epa` and `avg_defensive_epa_allowed` only. The pass and rush totals never become model inputs.

### 8. What play = 1 includes

**Verified for the filter this repo applies. Pending for a warehouse check that loaded rows still match the upstream rule.**

`fct_team_game_epa` keeps play-by-play rows with `where play = 1`. It does not filter `play_type`, `aborted_play`, quarter, or score differential. Garbage time stays in the average. Staging describes `play` as "Play validity flag (1 = valid play, 0 = no play)." That sentence is coarser than the upstream definition.

Upstream, nflfastR sets `play` to 1 when EPA and posteam are present, the description is not a review and does not start with "Timeout ", and `play_type` is one of `no_play`, `pass`, or `run` (`clean_pbp` in `nflverse/nflfastR` `R/helper_additional_functions.R`). That includes penalty rows (`no_play`). Special-teams types (`kickoff`, `punt`, `field_goal`, `extra_point`) are outside that list. This repo does not recompute the flag. A check that the parquet in `workspace.nfl_raw.play_by_play` still matches that rule is pending. No row counts are stated here.

### 9. Week 1 rest

**Verified for the fill rule. Pending for the week-1 values stored by nflverse.**

`build_pregame_features` copies `home_rest` and `away_rest` from the current game. A null becomes 7, before the model imputer runs. A non-null value is kept, including whatever week 1 contains. The mart passes `home_rest` and `away_rest` through from `stg_nflverse__games` with no transform. Whether week 1 is an offseason gap, a null, or something else is not measured in this repo.

## What is still pending

- Run `make audit-features` on the marts and read the out-of-range and time-counter flags, univariate log loss and AUC, Spearman pairs, and 2024 permutation importance.
- Count ties in `fct_games` before deciding whether to drop them or score them as 0.5.
- Confirm `play = 1` on the loaded parquet against the nflfastR rule above.
- Inspect week-1 `home_rest` and `away_rest` on the games mart before changing the fill value of 7.
