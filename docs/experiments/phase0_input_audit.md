# Phase 0 input audit

Diagnostic only. This note records what the current code does. It does not change features, and it does not report measurements from the warehouse. Each item is **verified** from the code in this repo, or **pending** a run of `make audit-features` on the marts (or a check this repo does not contain).

No counts, rates, or model metrics are stated here. Those belong in the report the audit writes when it is run.

The audit module is `src/analysis/feature_audit.py`. It calls `build_pregame_features`, `prepare_training_data`, `split_by_season`, and `train_logistic_regression`. Leakage tests live in `tests/test_feature_leakage.py` and use synthetic frames only.

`make audit-features` (or `.\scripts\dev.ps1 audit-features`) writes `reports/phase0/feature_audit.md` and `reports/phase0/feature_audit.csv`. It also logs both files to MLflow experiment `nfl_pregame/phase0_audit` on a run tagged `phase=0-audit`. 2025 is in the missingness and distribution tables only. It is not scored and it is not used to choose a model. Univariate CV defaults to seasons 2016 through 2023. Permutation importance fits the current logistic pipeline on seasons through 2023 and scores it on 2024. The frozen copy of that report is `docs/experiments/reports/phase0_feature_audit.md`. This note does not restate it.

## Checklist

### 1. games_played is a clock

**Verified (definition). The mart measurement is in the frozen audit.**

`home_games_played` is `len(home_prior)`. `home_prior` is every prior team-game with `gameday` strictly before the game. It is not filtered to the current season. `away_games_played` is the same count for the away team. The value grows with the length of the history in the frame. It is not "games played this season."

The audit flags `time_counter` when season medians are strictly monotone across at least three seasons, or when more than 5 percent of the 2024 rows or of the 2025 rows sit outside the closed training min/max. An earlier rule required every later value to sit outside that range, which misses a column that only partly leaves it. Whether `games_played` trips the flag on the marts is in `docs/experiments/reports/phase0_feature_audit.md`. This note does not restate that report.

### 2. Cross-season rolling on the default matrix

**Verified.**

Rolling offensive EPA, defensive EPA, and win rate on feature set `current` use `tail(lookback_games)` on all prior games for that franchise. That slice is not limited to the current season, and those level columns are not shrunk toward a league mean.

The level season feature is still season-to-date offensive EPA. A week-1 game therefore has a missing season EPA on `current` and a rolling window that can be entirely prior-season games. The leakage tests lock that behavior on a synthetic frame.

`build_pregame_features` also appends regressed season and recent EPA columns. Those shrink the previous season's final value toward that season's league mean and blend in current-season games. They are not in `current`. The audit profiles `current`. See `docs/experiments/phase2_h2_season_def_regression.md`.

### 3. No defensive season EPA on the default matrix

**Verified.**

`PREGAME_FEATURE_COLUMNS` still has season offense only: `home_season_offensive_epa_per_play` and `away_season_offensive_epa_per_play`. Defensive EPA on that matrix is the rolling columns `home_recent_defensive_epa_per_play` and `away_recent_defensive_epa_per_play`, taken from `avg_defensive_epa_allowed`.

The builder also emits `home_season_defensive_epa_per_play` and `away_season_defensive_epa_per_play` for feature sets `h2_def` and `h2_full`. Those columns are not in the matrix this audit profiles.

### 4. The default matrix has no home-minus-away differentials

**Verified.**

`prepare_training_data` defaults to feature set `current`. Those columns are separate home and away levels. The audit profiles that matrix.

`build_pregame_features` also appends five home-minus-away columns. They are not in `current` or `with_games_played`. Feature sets `diff` and `diff_only` select them. See `docs/experiments/phase2_h1_diff_features.md`.

### 5. location and game_type are ignored

**Verified.**

`fct_games` selects `location` and `game_type` from staging and keeps them on the mart. `build_pregame_features` does not read either column. They are not in `PREGAME_FEATURE_COLUMNS`, so they are not in the training matrix. Neutral sites and playoff games are not marked on the model input.

### 6. Ties are coded as home losses

**Verified (label rule). How many ties are in the mart: pending.**

In `fct_games`, `home_won` is 1 only when `home_score > away_score`. Every other completed game, including a tie, is 0. The recent win rate then treats `home_won == 0` as a win for the away franchise, so a tie also counts as an away win when that game is inside the lookback window. This note does not count ties.

### 7. pass_epa and rush_epa are outside the default matrix

**Verified.**

`fct_team_game_epa` computes `pass_epa` and `rush_epa` (sums of EPA on `pass_attempt = 1` and `rush_attempt = 1` inside the `play = 1` rows) and the play counts `pass_plays` and `rush_plays`. Feature set `current` does not read them. The audit profiles `current`.

Feature sets `h3_split` and `h3_replace` turn those sums into pass and rush EPA per play, for offense and for defense. Defense uses the opponent's offensive rate on the same game, because the mart does not store a defensive pass or rush column. See `docs/experiments/phase2_h3_pass_rush_epa.md`.

### 8. What play = 1 includes

**Verified for the filter this repo applies. The warehouse counts are frozen elsewhere. This note does not restate them.**

`fct_team_game_epa` keeps play-by-play rows with `where play = 1` for the original EPA columns. Those columns do not filter `play_type`, `aborted_play`, quarter, or score differential. Garbage time stays in that average. Staging describes `play` as "Play validity flag (1 = valid play, 0 = no play)." That sentence is coarser than the upstream definition. Clean EPA columns use a separate filter and are not in the matrix this audit profiles. See `docs/experiments/phase2_h4_clean_epa.md`.

Upstream, nflfastR sets `play` to 1 when EPA and posteam are present, the description is not a review and does not start with "Timeout ", and `play_type` is one of `no_play`, `pass`, or `run` (`clean_pbp` in `nflverse/nflfastR` `R/helper_additional_functions.R`). That includes penalty rows (`no_play`). Special-teams types (`kickoff`, `punt`, `field_goal`, `extra_point`) are outside that list. This repo does not recompute the flag. The warehouse check is frozen in `docs/experiments/reports/epa_filter_check.md`. No row counts are stated in this note.

### 9. Week 1 rest

**Verified for the fill rule. Pending for the week-1 values stored by nflverse.**

`build_pregame_features` copies `home_rest` and `away_rest` from the current game. A null becomes 7, before the model imputer runs. A non-null value is kept, including whatever week 1 contains. The mart passes `home_rest` and `away_rest` through from `stg_nflverse__games` with no transform. Whether week 1 is an offseason gap, a null, or something else is not measured in this repo.

## What is still pending

- The audit report is frozen in `docs/experiments/reports/phase0_feature_audit.md`. This note does not restate those measurements.
- Count ties in `fct_games` before deciding whether to drop them or score them as 0.5.
- The `play = 1` warehouse check is frozen in `docs/experiments/reports/epa_filter_check.md`. The reading (no_play contamination, H4 next) is in `docs/experiments/phase2_ridge_reference.md`.
- Inspect week-1 `home_rest` and `away_rest` on the games mart before changing the fill value of 7.
