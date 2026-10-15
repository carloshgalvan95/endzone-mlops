# Phase 1 baselines

Method note. This file does not report warehouse metrics. Phase 1 is `python -m src.train.train --cv` (`--phase 1`), which writes `reports/phase1/comparison.md` and `reports/phase1/comparison.csv`. The frozen copy is `docs/experiments/reports/phase1_comparison.md`. `make train-cv` runs H4 and writes `reports/phase2_h4/`. Hypothesis H1 is `--phase 2 --hypothesis h1` and writes `reports/phase2/`. Hypothesis H2 is `--phase 2 --hypothesis h2` and writes `reports/phase2_h2/`. Hypothesis H3 is `--phase 2 --hypothesis h3` and writes `reports/phase2_h3/`. The ridge comparison is `--phase 2 --hypothesis ridge` and writes `reports/phase2_ridge/`. See `docs/experiments/phase2_h1_diff_features.md`, `docs/experiments/phase2_h2_season_def_regression.md`, `docs/experiments/phase2_h3_pass_rush_epa.md`, `docs/experiments/phase2_h4_clean_epa.md`, and `docs/experiments/phase2_ridge_reference.md`.

No model is promoted from a phase 1 run. 2024 stays the final validation. 2025 is not scored here.

## Question

How noisy is a single season, and where do the current logistic features sit against a home prior, a simple Elo, and the closing market?

## Rolling-origin CV

For each season S from 2016 through 2023:

- Train on games with `season < S`. Earlier seasons, including seasons before 2016, stay in that training block.
- Score games with `season == S`.
- A fold with no rows, or with a single class in train or in the scored season, is skipped and recorded.

Pooled log loss, Brier, AUC, and accuracy use the scored out-of-fold games only. Accuracy treats a probability of 0.5 or higher as a home win.

2024 is not a fold. 2025 is not a fold. Rows from those seasons can sit in the frame. They are not passed to a scored fold as training rows or as evaluation rows.

The reference model is `baseline-home`: on each fold, the probability is the training home win rate. `log_loss_diff` is candidate log loss minus reference log loss on the games both models scored. A negative difference favors the candidate. The interval is the 2.5 and 97.5 percentiles of a paired bootstrap that draws the same game indexes for both models.

MLflow experiment: `nfl_pregame/phase1_baselines`. One `--cv` invocation is a parent run named `phase1-<timestamp>`. It is tagged `phase=1`, `cv_scheme=rolling_season_2016_2023`, `git_commit`, `reference_model`, `reference_feature_set`, `best_feature_set`, and `data_digest`. The comparison report and CSV are artifacts on that parent. Each scored model is a nested child with the same `data_digest`, the model params, the pooled CV metrics, `log_loss_diff`, `log_loss_diff_ci_low`, `log_loss_diff_ci_high`, and a `decision` tag. The home prior is `reference`. Elo is `baseline`. A logistic set is `accepted`, `rejected`, or `inconclusive`. 2024 metrics are not logged. In the MLflow UI, filter child runs with `tags.decision` and compare them. The dataset input is named `nfl_pregame_cv`.

## Feature sets

`build_pregame_features` still computes `home_games_played` and `away_games_played`. They count every prior team-game in the frame. They are not "games played this season."

| Name | Constant | Contents |
| --- | --- | --- |
| `current` | `PREGAME_FEATURE_COLUMNS` | Default training matrix. Omits the two games-played columns. |
| `with_games_played` | `PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED` | Previous matrix, kept so that set can be rerun. |

`prepare_training_data(..., feature_set="current")` is the default. Pass `with_games_played` for the old matrix. Phase 1 logs a logistic model on `current` and `with_games_played` only. The single-split trainer also accepts `diff` and `diff_only`. Phase 2 defines those sets and scores them.

The games-played columns stay in the builder so leakage tests still cover them. They are not in the default model.

## Elo

`elo_home_probabilities` is a pregame baseline, not a feature in the logistic matrix.

- Every franchise starts at 1505.
- Update step K is 20.
- Home advantage is 65 Elo points, added only when the game is scored. It is not stored on the team rating.
- Expected home win rate is `1 / (1 + 10 ** (-(home + 65 - away) / 400))`.
- Games that share a `gameday` are scored from the ratings at the start of that date, then updated together. A game does not see a same-day result.
- The first time a later season appears, every stored rating moves one third of the way back to 1505. That uses only games from earlier dates.
- A tied score updates as 0.5. Without scores, the update uses `home_won`. A null label does not update.
- There is no margin-of-victory multiplier. This is the simple 538-style rule, not a copy of a published rating.

The CV fold passes only games from seasons before S plus games in S. A later season is not in that subset.

## Market

The market baseline is the no-vig closing moneyline for that game. It is a benchmark. It is not a training feature in this phase.

American odds `o`:

- `o > 0`: raw implied probability `100 / (o + 100)`
- `o < 0`: raw implied probability `-o / (-o + 100)`

Home no-vig probability: `home_raw / (home_raw + away_raw)`.

`fct_games` already selects `home_moneyline` and `away_moneyline` from `stg_nflverse__games`. Staging already has both columns. This phase does not change dbt. If a frame or a future extract drops the columns, `market_home_probabilities` returns status `skipped` and reason `moneyline_columns_absent`. A null, zero, or non-finite price is left out of the scored rows (`no_valid_moneylines` when none remain). To put the columns back on the mart, select `home_moneyline` and `away_moneyline` in `dbt/models/marts/fct_games.sql`. They are already on the staging view.

## Decision log

The measured comparison is frozen in `docs/experiments/reports/phase1_comparison.md`. This decision log stays empty. Do not fill this table from `make train-cv` or from the H1 command. Those differences are versus logistic `current`, not versus the home prior.

| id | date | run_id | model | feature_set | log_loss_diff_vs_home_prior | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |

`decision` is `accepted`, `rejected`, or `inconclusive`. Phase 1 records the baselines. It does not accept a champion.

Hypothesis H1 applies a stricter rule: pooled log loss must improve by at least 0.003 and the 95% paired interval must exclude 0. That rule is specified in `docs/experiments/phase2_h1_diff_features.md` and is not applied to this table.
