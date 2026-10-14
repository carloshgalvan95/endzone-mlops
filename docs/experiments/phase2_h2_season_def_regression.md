# Phase 2 H2: season defensive EPA and cross-season regression

Method note. This file does not report warehouse metrics. `make train-cv` (Windows: `.\scripts\dev.ps1 train-cv`) runs `python -m src.train.train --cv --phase 2 --hypothesis h2` and writes the measured comparison to `reports/phase2_h2/comparison.md` and `reports/phase2_h2/comparison.csv`. Those files are local outputs. Leave the result section empty until that run exists. Do not copy numbers from a test fixture.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here.

Hypothesis H1 stays reproducible with `python -m src.train.train --cv --phase 2 --hypothesis h1 --reference-model logistic --reference-feature-set current --output-dir reports/phase2`. See `docs/experiments/phase2_h1_diff_features.md`.

## Hypothesis

Season-to-date defensive EPA allowed, and a cross-season regression of season and recent EPA, improve pooled rolling-origin CV log loss versus the current logistic model (feature set `current`).

The level columns in `current` do not include a same-season defensive EPA, and week 1 season EPA is missing. A linear model cannot invent that history from the columns it already has. That is the new information this hypothesis adds. H1 showed that home-minus-away copies of the existing levels did not.

Phase 1 reference on real data, CV 2016-2023, 2,191 games, log loss: logistic `current` 0.6414, Elo 0.6529, market 0.6120. H1 on the same window rejected `diff` (0.6414) and `diff_only` (0.6416). Those numbers are the reference this hypothesis is measured against. They are not a result of the H2 run.

## Feature sets

`build_pregame_features` still emits the level columns and the H1 differences, then appends season defensive EPA and the regressed columns. `prepare_training_data` still defaults to `current`. The single-split trainer accepts every name through `--feature-set`.

Season defensive EPA is the mean of `avg_defensive_epa_allowed` on that team's prior games in the same `season`. It uses the season column, not the calendar year, so a January playoff stays in its season. Week 1 is missing. A missing value is not zero.

| Column | Definition |
| --- | --- |
| `home_season_defensive_epa_per_play` | Home season-to-date EPA allowed |
| `away_season_defensive_epa_per_play` | Away season-to-date EPA allowed |

Regressed columns apply to recent offensive EPA, recent defensive EPA, season offensive EPA, and season defensive EPA, for home and away. Eight columns, prefixed `home_regressed_` or `away_regressed_`.

For a game in season S, before any current-season games:

- Take that team's final value from season S-1 only. The recent final is the last `lookback_games` rows inside S-1, not a window that reaches into S-2. The season final is the mean of every S-1 game.
- Shrink it toward the S-1 league mean by `SEASON_SHRINK_FACTOR` (default 1/3, the same fraction Elo removes between seasons). One third of the gap is removed: `mean + (1 - 1/3) * (prior - mean)`.
- The league mean is the mean of that stat over all franchises' team-games in season S-1 with `gameday` strictly before this game.

As current-season games accumulate, blend toward the current-season value:

`weight = n / (n + k)`

`value = weight * current + (1 - weight) * shrunk_prior`

`n` is the number of prior games in season S. `k` is `SEASON_BLEND_K` (default 4). At 0 games the weight is 0 and the result is the shrunk prior. At 4 games the weight is one half. The current recent value is the last `lookback_games` rows inside season S. The current season value is the mean of all of them. `n` counts every current-season game, including ones outside the recent window.

A team with no completed season S-1 games has no shrunk prior. Before its first game the regressed value stays missing. After that it uses the current-season mean alone. The logistic pipeline still imputes a missing EPA column to 0.

Only team-games with `gameday` strictly before the game are eligible. A row labeled season S-1 but dated on or after this kickoff is not a completed prior game. Same-day games do not see each other.

| Name | Constant | Contents |
| --- | --- | --- |
| `current` | `PREGAME_FEATURE_COLUMNS` | Unchanged default matrix. |
| `with_games_played` | `PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED` | Unchanged previous matrix. Not scored in this CV. |
| `diff` | `PREGAME_FEATURE_COLUMNS_DIFF` | Unchanged H1 set. Not scored in this CV. |
| `diff_only` | `PREGAME_FEATURE_COLUMNS_DIFF_ONLY` | Unchanged H1 set. Not scored in this CV. |
| `h2_def` | `PREGAME_FEATURE_COLUMNS_H2_DEF` | `current` plus the two season defensive EPA columns. |
| `h2_full` | `PREGAME_FEATURE_COLUMNS_H2_FULL` | `h2_def` plus the eight regressed columns. |

## Method

Same rolling-origin window as Phase 1 and H1. For each season S from 2016 through 2023:

- Train on games with `season < S`.
- Score games with `season == S`.
- A fold with no rows, or with a single class in train or in the scored season, is skipped and recorded.

2024 is not a fold. 2025 is not a fold.

Scored models:

- logistic regression on `current`, `h2_def`, and `h2_full`
- home prior (`baseline-home`)
- Elo, with the Phase 1 settings
- no-vig closing market, skipped when `home_moneyline` and `away_moneyline` are absent

The default reference is logistic `current`. `log_loss_diff` is candidate log loss minus logistic `current` log loss on the games both models scored. A negative difference favors the candidate. The interval is the 2.5 and 97.5 percentiles of a paired bootstrap that draws the same game indexes for both models (1000 resamples, seed 42, unless the CLI overrides them).

Choose another reference with `--reference-model` and `--reference-feature-set`. Phase 1 (`python -m src.train.train --cv`, which is `--phase 1`) still uses `baseline-home`. H1 (`--phase 2 --hypothesis h1`) still scores `current`, `diff`, and `diff_only` into `reports/phase2/`.

MLflow experiment: `nfl_pregame/phase2_h2`. Each scored model is one run tagged `phase=2` and `hypothesis=h2-season-def-regression`, with `feature_set`, `cv_scheme=rolling_season_2016_2023`, `git_commit`, and the model params. H2 runs also log `shrink_factor` and `blend_k`. 2024 metrics are not logged.

## Acceptance rule

Accept a feature set only when both are true on the pooled out-of-fold games:

1. Pooled log loss improves on the reference by at least 0.003. That is `log_loss_diff` <= -0.003.
2. The 95% paired bootstrap interval excludes 0. Because the difference is candidate minus reference, the interval has to lie entirely below 0 (`log_loss_diff_ci_high` < 0). An interval that excludes 0 on the side that favors the reference is not an improvement.

If either condition fails, the set is rejected. If the interval is missing, the decision is inconclusive. The reference row is not a candidate. `h2_def` and `h2_full` are judged separately. One set can be accepted while the other is rejected. Baselines in the same report are labeled with the same rule. They are not the hypothesis.

## Result

Leave this table empty until `make train-cv` has been run locally. Do not fill it from a test fixture.

| feature_set | n | log_loss | log_loss_diff | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Decision

Pending. Carlos fills this after the local run.

`decision` is `accepted`, `rejected`, or `inconclusive` for each of `h2_def` and `h2_full`.
