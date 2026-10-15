# Phase 2 H4: clean EPA

Method note. Reproduce this comparison with `python -m src.train.train --cv --phase 2 --hypothesis h4 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h4`. It writes `reports/phase2_h4/comparison.md` and `reports/phase2_h4/comparison.csv`. Those files are local outputs. The result section records the local H4 run. The frozen copy is `docs/experiments/reports/phase2_h4_comparison.md`. Do not copy numbers from a test fixture. `make train-cv` (Windows: `.\scripts\dev.ps1 train-cv`) still runs this hypothesis.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here.

The ridge comparison stays reproducible with `python -m src.train.train --cv --phase 2 --hypothesis ridge --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_ridge`. It is no longer the `make train-cv` target. See `docs/experiments/phase2_ridge_reference.md`.

## Hypothesis

EPA computed only on competitive pass and run plays predicts the home win better than the current `play = 1` average. The comparison is pooled rolling-origin CV log loss versus logistic `h2_full`.

`h2_full` is the reference. H3 rejected both pass/rush splits. Fold-tuned L2 did not beat plain logistic `h2_full`, so the reference stays that model. Those decisions are recorded in `docs/experiments/phase2_h3_pass_rush_epa.md` and `docs/experiments/phase2_ridge_reference.md`. They are not a result of this run.

The EPA filter check (`docs/experiments/reports/epa_filter_check.md`) found that `play = 1` rows are pass 53.6%, run 39.5%, and `no_play` 6.9%. Kneels and spikes with `play = 1` are rare (36 and 70). The contamination is penalty-nullified `no_play` rows, plus garbage time, which the current average keeps. Greer (2018) and the Open Source Football EPA writeups report that a cleaner EPA, without garbage time and without non-competitive plays, predicts future margin better. This hypothesis tests that claim on home-win log loss. It does not retune the windows.

## Where the rates come from

`fct_team_game_epa` still stores `avg_offensive_epa` and `avg_defensive_epa_allowed` from every `play = 1` row. Those columns are unchanged, so `current`, `h2_full`, and the H3 sets stay reproducible.

New columns, from a subset of the same `play = 1` rows:

- `play_type` is `pass` or `run`. That drops `no_play` and every other play type.
- `qb_kneel` and `qb_spike` are not 1. A null flag is kept. Staging now selects both flags from `workspace.nfl_raw.play_by_play`. They were already on the bronze table.
- Garbage time is out when pre-snap `wp` (staged as `win_probability`, the possession team's win probability) is below 0.05 or above 0.95. A null `wp` is not garbage time. 0.05 and 0.95 stay in. Those cutoffs are fixed. They were not chosen on this comparison.

`wp` was already on the staging view as `win_probability`. This change does not require `make ingest`.

Per team-game:

- `avg_clean_offensive_epa` is the mean EPA of that team's clean offensive plays. Null when there are none.
- `avg_clean_defensive_epa_allowed` is the mean EPA of clean plays faced.
- `clean_success_rate` is the share of clean offensive plays with `success = 1`.
- `clean_defensive_success_rate` is the share of clean plays allowed with `success = 1`.
- `clean_offensive_plays` and `clean_defensive_plays` are the denominators. Zero stays a null rate, not a zero rate.

## Feature sets

`build_pregame_features` still emits the existing columns, then the clean EPA columns and the recent clean success rates. `prepare_training_data` still defaults to `current`.

`h4_clean` is `h2_full` with every EPA input replaced by the same window on the clean columns. Win rate and rest days stay. The replaced inputs are recent EPA (cross-season lookback), season-to-date EPA (same season only), regressed recent EPA (within-season lookback), and regressed season EPA, for offense and defense, home and away.

Shrinkage and the blend are the existing constants. They are not tuned for H4:

- `SEASON_SHRINK_FACTOR` is 1/3, the same fraction Elo uses between seasons.
- `SEASON_BLEND_K` is 4. Current-season weight is `n / (n + 4)`.
- The shrink target is the prior season's league mean of the clean column, not the original EPA column.
- The prior season is `season - 1` only. An earlier season is not carried forward.
- Week 1 of the first season in the frame stays missing.

`h4_clean_sr` is `h4_clean` plus the recent clean success rate, offense and defense, home and away. Recent means the same cross-season `lookback_games` window as `home_recent_offensive_epa_per_play`. It is not shrunk. A missing success rate is filled with 0.5 in the logistic pipeline, the same fill as a missing win rate. Clean EPA is still filled with 0.

| Name | Constant | Contents |
| --- | --- | --- |
| `h2_full` | `PREGAME_FEATURE_COLUMNS_H2_FULL` | Reference matrix. Unchanged. |
| `h4_clean` | `PREGAME_FEATURE_COLUMNS_H4_CLEAN` | `h2_full` with every EPA input taken from the clean columns. |
| `h4_clean_sr` | `PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR` | `h4_clean` plus four recent clean success rates. |

Names look like `home_recent_clean_offensive_epa_per_play`, `away_regressed_season_clean_defensive_epa_per_play`, and `home_recent_clean_offensive_success_rate`.

## Warehouse commands

Bronze already has `qb_kneel`, `qb_spike`, and `wp`. Do not run `make ingest` for this hypothesis.

```bash
make dbt-build-databricks
make train-cv
```

Windows:

```powershell
.\scripts\dev.ps1 dbt-build-databricks
.\scripts\dev.ps1 train-cv
```

`make dbt-build-databricks` rebuilds `nfl_dev_staging`, `nfl_dev_marts`, and `nfl_dev_test_results`, including the new staging columns and the clean mart columns. `make train-cv` then reads `fct_games` and `fct_team_game_epa` from `workspace.nfl_dev_marts`. Running `train-cv` before that rebuild leaves the clean columns absent. The features stay missing and the comparison is not this hypothesis.

## Method

Same rolling-origin window as Phase 1 and H1 through H3. For each season S from 2016 through 2023:

- Train on games with `season < S`.
- Score games with `season == S`.
- A fold with no rows, or with a single class in train or in the scored season, is skipped and recorded.

2024 is not a fold. 2025 is not a fold.

Scored models:

- logistic regression on `h2_full`, `h4_clean`, and `h4_clean_sr`
- home prior (`baseline-home`)
- Elo, with the Phase 1 settings
- no-vig closing market, skipped when `home_moneyline` and `away_moneyline` are absent

The default reference is logistic `h2_full`. `log_loss_diff` is candidate log loss minus logistic `h2_full` log loss on the games both models scored. A negative difference favors the candidate. The interval is the 2.5 and 97.5 percentiles of a paired bootstrap that draws the same game indexes for both models (1000 resamples, seed 42, unless the CLI overrides them).

MLflow experiment: `nfl_pregame/phase2_h4`. One invocation is a parent run named `phase2-h4-<timestamp>`, tagged `phase=2`, `hypothesis=h4-clean-epa`, `cv_scheme=rolling_season_2016_2023`, `git_commit`, `reference_model`, `reference_feature_set`, `best_feature_set`, and `data_digest`. The comparison report and CSV are artifacts on that parent. Each scored model is a nested child with the same `data_digest`, `feature_set`, the model params, `log_loss_diff`, `log_loss_diff_ci_low`, `log_loss_diff_ci_high`, and a `decision` tag (`reference`, `baseline`, `accepted`, `rejected`, or `inconclusive`). H4 children also log `shrink_factor` and `blend_k`. 2024 metrics are not logged. In the MLflow UI, filter child runs with `tags.decision` and compare them.

## Acceptance rule

Accept a feature set only when both are true on the pooled out-of-fold games, versus logistic `h2_full`:

1. Pooled log loss improves on the reference by at least 0.003. That is `log_loss_diff` <= -0.003.
2. The 95% paired bootstrap interval excludes 0. Because the difference is candidate minus reference, the interval has to lie entirely below 0 (`log_loss_diff_ci_high` < 0). An interval that excludes 0 on the side that favors the reference is not an improvement.

If either condition fails, the set is rejected. If the interval is missing, the decision is inconclusive. The reference row is not a candidate. `h4_clean` and `h4_clean_sr` are judged separately. One set can be accepted while the other is rejected. Baselines in the same report are labeled with the same rule. They are not the hypothesis.

## Result

Local rolling-origin run, CV 2016-2023, n=2191, reference logistic `h2_full` 0.6377. These rows are that run, after `make dbt-build-databricks` rebuilt the clean columns. They are not taken from a test fixture. The frozen copy is `docs/experiments/reports/phase2_h4_comparison.md`.

| feature_set | n | log_loss | log_loss_diff | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- |
| h2_full | 2191 | 0.6377 | 0.0000 | 0.0000 | 0.0000 | reference |
| h4_clean | 2191 | 0.6398 | +0.0021 | -0.0012 | 0.0053 | rejected |
| h4_clean_sr | 2191 | 0.6379 | +0.0002 | -0.0036 | 0.0039 | rejected |

`h2_full` is the reference row. Its difference is zero. `h4_clean` is worse by 0.0021. `h4_clean_sr` matches the reference within 0.0002. The home prior, Elo, and the market were scored by the same command. They are baselines, not this hypothesis, and this record does not restate them.

A first local run, before that dbt rebuild, printed the same log loss 0.6509 for the clean sets. The clean columns were absent from `fct_team_game_epa`, so the clean features were empty and the imputer filled them with constants. Those identical numbers are not this comparison. Training now refuses that case. Before CV, each requested feature column has to exist and stay under 50% null. The error names the feature set and the columns and says to run `make dbt-build-databricks`.

## Decision log

Rejected.

`h4_clean` and `h4_clean_sr` are both rejected. Neither improved pooled log loss by at least 0.003, and neither 95% interval lies entirely below 0. Removing `no_play`, kneels, spikes, and garbage time adds no signal. Averaging EPA over many plays already absorbs that noise. The feature sets stay in code so this comparison can be rerun. 2024 stays the final validation. 2025 stays unscored. No model is promoted from this note.
