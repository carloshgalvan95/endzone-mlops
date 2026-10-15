# Phase 2 H1: home-minus-away differentials

Method note. This file does not report warehouse metrics. Reproduce this comparison with `python -m src.train.train --cv --phase 2 --hypothesis h1 --reference-model logistic --reference-feature-set current --output-dir reports/phase2`. It writes `reports/phase2/comparison.md` and `reports/phase2/comparison.csv`. `make train-cv` runs the ridge reference comparison and writes `reports/phase2_ridge/`. Hypothesis H2 is `--phase 2 --hypothesis h2 --reference-model logistic --reference-feature-set current --output-dir reports/phase2_h2`. Hypothesis H3 is `--phase 2 --hypothesis h3` and writes `reports/phase2_h3/`. Those files are local outputs. The result section records the local H1 run. Do not copy numbers from a test fixture.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here.

## Hypothesis

Home-minus-away differential features improve pooled rolling-origin CV log loss versus the current logistic model (feature set `current`, without `games_played`).

Phase 1 reference on real data, CV 2016-2023, 2,191 games, log loss: logistic `current` 0.6414, Elo 0.6529, market 0.6120. Those three numbers are the Phase 1 reference this hypothesis is measured against. They are not a result of the H1 run.

## Feature sets

`build_pregame_features` still emits the level columns, then appends five differences. Each difference is the home value minus the away value. A missing side stays missing. The logistic pipeline imputes a missing difference to 0. Level win rates stay imputed at 0.5. A missing differential win rate is not treated as a home edge.

| Column | Definition |
| --- | --- |
| `diff_recent_offensive_epa_per_play` | Home recent offensive EPA minus away recent offensive EPA |
| `diff_recent_defensive_epa_per_play` | Home recent defensive EPA allowed minus away recent defensive EPA allowed |
| `diff_recent_win_rate` | Home recent win rate minus away recent win rate |
| `diff_season_offensive_epa_per_play` | Home season offensive EPA minus away season offensive EPA |
| `diff_rest_days` | Home rest days minus away rest days |

Defensive EPA is EPA allowed. The difference is not flipped. A negative defensive difference means the home defense allowed less EPA than the away defense.

| Name | Constant | Contents |
| --- | --- | --- |
| `current` | `PREGAME_FEATURE_COLUMNS` | Unchanged default matrix. No differences, no `games_played`. |
| `with_games_played` | `PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED` | Unchanged previous matrix. Not scored in this CV. |
| `diff` | `PREGAME_FEATURE_COLUMNS_DIFF` | `current` plus the five differences. |
| `diff_only` | `PREGAME_FEATURE_COLUMNS_DIFF_ONLY` | The five differences and nothing else. Parsimony check. |

`prepare_training_data` still defaults to `current`. The single-split trainer accepts all four names through `--feature-set`.

The differences use the same prior-game rows as the level columns (`gameday` strictly before the game). They do not read the current game or a later game.

## Method

Same rolling-origin window as Phase 1. For each season S from 2016 through 2023:

- Train on games with `season < S`.
- Score games with `season == S`.
- A fold with no rows, or with a single class in train or in the scored season, is skipped and recorded.

2024 is not a fold. 2025 is not a fold.

Scored models:

- logistic regression on `current`, `diff`, and `diff_only`
- home prior (`baseline-home`)
- Elo, with the Phase 1 settings
- no-vig closing market, skipped when `home_moneyline` and `away_moneyline` are absent

The default reference is logistic `current`. `log_loss_diff` is candidate log loss minus logistic `current` log loss on the games both models scored. A negative difference favors the candidate. The interval is the 2.5 and 97.5 percentiles of a paired bootstrap that draws the same game indexes for both models (1000 resamples, seed 42, unless the CLI overrides them).

Choose another reference with `--reference-model` and `--reference-feature-set`. Phase 1 (`python -m src.train.train --cv`, which is `--phase 1`) still uses `baseline-home` and still scores `current` and `with_games_played`. That default report is unchanged.

MLflow experiment: `nfl_pregame/phase2_h1`. One invocation is a parent run named `phase2-h1-<timestamp>`, tagged `phase=2`, `hypothesis=h1-diff`, `cv_scheme=rolling_season_2016_2023`, `git_commit`, `reference_model`, `reference_feature_set`, `best_feature_set`, and `data_digest`. The comparison report and CSV are artifacts on that parent. Each scored model is a nested child with the same `data_digest`, `feature_set`, the model params, `log_loss_diff`, `log_loss_diff_ci_low`, `log_loss_diff_ci_high`, and a `decision` tag (`reference`, `baseline`, `accepted`, `rejected`, or `inconclusive`). 2024 metrics are not logged. In the MLflow UI, filter child runs with `tags.decision` and compare them.

## Acceptance rule

Accept a feature set only when both are true on the pooled out-of-fold games:

1. Pooled log loss improves on the reference by at least 0.003. That is `log_loss_diff` <= -0.003.
2. The 95% paired bootstrap interval excludes 0. Because the difference is candidate minus reference, the interval has to lie entirely below 0 (`log_loss_diff_ci_high` < 0). An interval that excludes 0 on the side that favors the reference is not an improvement.

If either condition fails, the set is rejected. If the interval is missing, the decision is inconclusive. The reference row is not a candidate. `diff` and `diff_only` are judged separately. One set can be accepted while the other is rejected. Baselines in the same report are labeled with the same rule. They are not the hypothesis.

This rule is not applied to the Phase 1 decision log.

## Result

Local rolling-origin run, CV 2016-2023, reference logistic `current`. These rows are that run. They are not warehouse metrics and they are not taken from a test fixture.

| feature_set | n | log_loss | log_loss_diff | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- |
| current | 2191 | 0.6414 | 0.0000 | | | reference |
| diff | 2191 | 0.6414 | 0.0000 | -0.0000 | 0.0001 | rejected |
| diff_only | 2191 | 0.6416 | 0.0002 | -0.0011 | 0.0015 | rejected |
| elo | 2191 | 0.6529 | 0.0115 | | | inconclusive |
| market | 2191 | 0.6120 | -0.0296 | -0.0386 | -0.0204 | accepted |

`current` is the reference row. Its difference is zero. Elo's point estimate is worse by 0.0115. The local summary did not include Elo's interval, so that row stays inconclusive under the rule above. The market interval lies entirely below 0 and the gain is larger than 0.003, so the numeric rule labels it accepted. That label is the baseline comparison. It is not an acceptance of this hypothesis.

## Decision log

Rejected.

Neither `diff` nor `diff_only` improved pooled log loss by at least 0.003, and neither 95% interval lies entirely below 0. A linear model can already form the home-minus-away differences from the level columns, so the hypothesis added no new information. `diff_only` matches `current` with half the columns (0.6416 versus 0.6414). That is a parsimony option, not an accepted improvement. The feature sets stay in code so this comparison can be rerun.
