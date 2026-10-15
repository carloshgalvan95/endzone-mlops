# Phase 2 H3: pass and rush EPA

Method note. This file does not report warehouse metrics. Reproduce this comparison with `python -m src.train.train --cv --phase 2 --hypothesis h3 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h3`. It writes `reports/phase2_h3/comparison.md` and `reports/phase2_h3/comparison.csv`. Those files are local outputs. The result section records the local H3 run. The frozen copy is `docs/experiments/reports/phase2_h3_comparison.md`. Do not copy numbers from a test fixture. `make train-cv` runs H4 and writes `reports/phase2_h4/`. See `docs/experiments/phase2_h4_clean_epa.md`. The ridge comparison is `--phase 2 --hypothesis ridge` and writes `reports/phase2_ridge/`. See `docs/experiments/phase2_ridge_reference.md`.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here.

Hypothesis H1 stays reproducible with `python -m src.train.train --cv --phase 2 --hypothesis h1 --reference-model logistic --reference-feature-set current --output-dir reports/phase2`. Hypothesis H2 stays reproducible with `python -m src.train.train --cv --phase 2 --hypothesis h2 --reference-model logistic --reference-feature-set current --output-dir reports/phase2_h2`. See `docs/experiments/phase2_h1_diff_features.md` and `docs/experiments/phase2_h2_season_def_regression.md`.

## Hypothesis

Splitting offensive and defensive EPA into pass and rush improves pooled rolling-origin CV log loss versus logistic `h2_full`.

`h2_full` was accepted against logistic `current` on CV 2016-2023 (n=2191, log loss 0.6377 versus 0.6414). `h2_def` was rejected. The gain came from the regressed priors, not from season defensive EPA alone. The `h2_full` interval upper bound was close to 0, so that acceptance still has to be confirmed on the 2024 final validation. Those numbers are the H2 result this hypothesis starts from. They are not a result of the H3 run.

The combined EPA columns do not say whether the EPA came on a pass or a rush. A linear model cannot split a sum it was never given. That is the new information this hypothesis adds.

## Where the rates come from

`fct_team_game_epa` already stores offensive `pass_epa` and `rush_epa` (sums of EPA on `pass_attempt = 1` and `rush_attempt = 1` inside `play = 1`) and the counts `pass_plays` and `rush_plays`. Staging exposes `pass_attempt` and `rush_attempt`. The mart does not store a defensive pass or rush column. No dbt change was required.

Per play, on one team-game:

- offensive pass EPA per play is `pass_epa / pass_plays`
- offensive rush EPA per play is `rush_epa / rush_plays`
- defensive pass EPA allowed is the opponent's offensive pass EPA per play in that same game
- defensive rush EPA allowed is the opponent's offensive rush EPA per play in that same game

A play count of zero stays missing. It is not stored as zero. The rolling and season values are means of those game-level rates, the same game-weighted mean the combined EPA columns use. Only team-games with `gameday` strictly before the game are eligible. Same-day games do not see each other.

## Feature sets

`build_pregame_features` still emits the level columns, the H1 differences, the H2 season-defense columns, and the H2 regressed columns, then appends the pass/rush columns. `prepare_training_data` still defaults to `current`. The single-split trainer accepts every name through `--feature-set`.

Recent pass and rush use the same `lookback_games` window as `home_recent_offensive_epa_per_play`: the last N prior team-games, which can cross seasons. They are not shrunk.

Regressed-season pass and rush use the same shrinkage and blend as `home_regressed_season_offensive_epa_per_play`. For a game in season S:

- The current value is the mean of prior games in season S.
- The prior value is the mean of season S-1 only. An earlier season is not carried forward.
- Shrink toward the S-1 league mean by `SEASON_SHRINK_FACTOR` (default 1/3).
- Blend with weight `n / (n + k)`, where `n` is the number of prior games in season S and `k` is `SEASON_BLEND_K` (default 4).

The league mean is the mean of that pass or rush rate over all franchises' team-games in season S-1 with `gameday` strictly before this game. Week 1 of the first season in the frame stays missing.

| Family | Sides | Roles | Kinds |
| --- | --- | --- | --- |
| `recent` | home, away | offensive, defensive | pass, rush |
| `regressed_season` | home, away | offensive, defensive | pass, rush |

Sixteen columns. Names look like `home_recent_offensive_pass_epa_per_play` and `away_regressed_season_defensive_rush_epa_per_play`.

| Name | Constant | Contents |
| --- | --- | --- |
| `current` | `PREGAME_FEATURE_COLUMNS` | Unchanged default matrix. Not scored in this CV. |
| `with_games_played` | `PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED` | Unchanged previous matrix. Not scored in this CV. |
| `diff` | `PREGAME_FEATURE_COLUMNS_DIFF` | Unchanged H1 set. Not scored in this CV. |
| `diff_only` | `PREGAME_FEATURE_COLUMNS_DIFF_ONLY` | Unchanged H1 set. Not scored in this CV. |
| `h2_def` | `PREGAME_FEATURE_COLUMNS_H2_DEF` | Unchanged H2 set. Not scored in this CV. |
| `h2_full` | `PREGAME_FEATURE_COLUMNS_H2_FULL` | Reference matrix. Combined recent, season, and regressed EPA. |
| `h3_split` | `PREGAME_FEATURE_COLUMNS_H3_SPLIT` | `h2_full` plus the sixteen pass/rush columns. |
| `h3_replace` | `PREGAME_FEATURE_COLUMNS_H3_REPLACE` | `h2_full` with the combined recent EPA and the combined regressed-season EPA replaced by the pass/rush splits. Win rate, rest, season-to-date EPA, and the regressed recent block stay. |

`h3_replace` drops these combined columns:

- `home_recent_offensive_epa_per_play`, `home_recent_defensive_epa_per_play`
- `away_recent_offensive_epa_per_play`, `away_recent_defensive_epa_per_play`
- `home_regressed_season_offensive_epa_per_play`, `away_regressed_season_offensive_epa_per_play`
- `home_regressed_season_defensive_epa_per_play`, `away_regressed_season_defensive_epa_per_play`

## Method

Same rolling-origin window as Phase 1, H1, and H2. For each season S from 2016 through 2023:

- Train on games with `season < S`.
- Score games with `season == S`.
- A fold with no rows, or with a single class in train or in the scored season, is skipped and recorded.

2024 is not a fold. 2025 is not a fold.

Scored models:

- logistic regression on `h2_full`, `h3_split`, and `h3_replace`
- home prior (`baseline-home`)
- Elo, with the Phase 1 settings
- no-vig closing market, skipped when `home_moneyline` and `away_moneyline` are absent

The default reference is logistic `h2_full`. `log_loss_diff` is candidate log loss minus logistic `h2_full` log loss on the games both models scored. A negative difference favors the candidate. The interval is the 2.5 and 97.5 percentiles of a paired bootstrap that draws the same game indexes for both models (1000 resamples, seed 42, unless the CLI overrides them).

Choose another reference with `--reference-model` and `--reference-feature-set`. Phase 1 (`python -m src.train.train --cv`, which is `--phase 1`) still uses `baseline-home`. H1 (`--phase 2 --hypothesis h1`) still scores `current`, `diff`, and `diff_only` into `reports/phase2/` against logistic `current`. H2 (`--phase 2 --hypothesis h2`) still scores `current`, `h2_def`, and `h2_full` into `reports/phase2_h2/` against logistic `current`.

MLflow experiment: `nfl_pregame/phase2_h3`. One invocation is a parent run named `phase2-h3-<timestamp>`, tagged `phase=2`, `hypothesis=h3-pass-rush-epa`, `cv_scheme=rolling_season_2016_2023`, `git_commit`, `reference_model`, `reference_feature_set`, `best_feature_set`, and `data_digest`. The comparison report and CSV are artifacts on that parent. Each scored model is a nested child with the same `data_digest`, `feature_set`, the model params, `log_loss_diff`, `log_loss_diff_ci_low`, `log_loss_diff_ci_high`, and a `decision` tag (`reference`, `baseline`, `accepted`, `rejected`, or `inconclusive`). H3 children also log `shrink_factor` and `blend_k`. 2024 metrics are not logged. In the MLflow UI, filter child runs with `tags.decision` and compare them.

## Acceptance rule

Accept a feature set only when both are true on the pooled out-of-fold games, versus logistic `h2_full`:

1. Pooled log loss improves on the reference by at least 0.003. That is `log_loss_diff` <= -0.003.
2. The 95% paired bootstrap interval excludes 0. Because the difference is candidate minus reference, the interval has to lie entirely below 0 (`log_loss_diff_ci_high` < 0). An interval that excludes 0 on the side that favors the reference is not an improvement.

If either condition fails, the set is rejected. If the interval is missing, the decision is inconclusive. The reference row is not a candidate. `h3_split` and `h3_replace` are judged separately. One set can be accepted while the other is rejected. Baselines in the same report are labeled with the same rule. They are not the hypothesis.

## Result

Local rolling-origin run, CV 2016-2023, n=2191, reference logistic `h2_full` 0.6377. These rows are that run. They are not warehouse metrics and they are not taken from a test fixture.

| feature_set | n | log_loss | log_loss_diff | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- |
| h2_full | 2191 | 0.6377 | 0.0000 | | | reference |
| h3_split | 2191 | 0.6393 | 0.0016 | -0.0009 | 0.0039 | rejected |
| h3_replace | 2191 | 0.6395 | 0.0018 | -0.0008 | 0.0043 | rejected |

`h2_full` is the reference row. Its difference is zero. `h3_split` scored AUC 0.6801 and `h3_replace` scored AUC 0.6796. Both are lower than the reference AUC, and both have lower accuracy. The home prior, Elo, and the market were scored by the same command. They are baselines, not this hypothesis, and this record does not restate them.

## Decision log

Rejected.

`h3_split` and `h3_replace` are both rejected. Neither improved pooled log loss. The differences are +0.0016 and +0.0018. Both 95% intervals cross 0, so neither lies entirely below 0, and neither gain reaches 0.003. A season in this window is about 270 games. Splitting combined EPA into pass and rush adds columns, and those extra columns add noise. The combined EPA already carries the signal. The feature sets stay in code so this comparison can be rerun. 2024 stays the final validation. 2025 stays unscored. No model is promoted from this run.
