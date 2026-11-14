# Model card: endzone_winprob

This card follows Mitchell et al., "Model Cards for Model Reporting" (https://arxiv.org/abs/1810.03993). It is where I keep the intended use, the limits, and the metrics for the home-win model `endzone_winprob`.

The market has the lower log loss. Nothing in this card is a betting pick. This is not betting advice.

I copy every number from `docs/experiments/reference_metrics.json` or from a frozen report in `docs/experiments/reports/`. The table names the file. I do not recompute a figure. When a value is not in those files, I say so.

## Model details

- **Name.** Unity Catalog model `workspace.nfl_dev.endzone_winprob`. Catalog `workspace`, schema `nfl_dev`, model `endzone_winprob`. Unity Catalog does not allow a hyphen in the model name. A sqlite registry uses `endzone-winprob`. ADR-009.
- **Version and alias.** `prod` is an alias, not a version. The one-time promote in `docs/runbook.md` sets that alias on version 1: `make promote-model VERSION=1`. The Model registry workflow does the same step with action `promote` and version 1. I have not stored a live registry query in this repo, so this card does not confirm the warehouse alias by calling Databricks. A later register creates the next version and does not move the alias. Weekly scoring loads `models:/<name>@prod` and does not set the alias.
- **Registration.** `make register-model` fits logistic `h2_full` on completed games through season 2023 and writes one version. The version tags are the rolling-origin CV in `docs/experiments/reference_metrics.json`. They are not the in-sample loss of that fit. The command does not set an alias. It writes nothing when the reference check fails. `make reproduce` does not call it. Unity Catalog registration needs `MLFLOW_TRACKING_URI=databricks` and `MLFLOW_REGISTRY_URI=databricks-uc`. A sqlite tracking URI with `databricks-uc` is refused before a write. ADR-009.
- **Fresh window.** `make register-recent` fits the same logistic `h2_full` on completed games before the slate in a walk-forward report. The report is `docs/experiments/walk_forward.md`. That command is a refresh of the current recipe, not a new feature. It writes a version only when the recipe matches prod and the refresh gate passes. It does not set an alias. A different model, feature set, or hyperparameter is a challenger and is refused by this command. A `through_season` past 2023 without a walk-forward report is refused, so the golden CV tags cannot ride onto a later fit. Weekly automation of the refresh is future work.
- **Promotion.** `make promote-model` sets alias `prod` when the version tags match the golden logistic `h2_full` row within the tolerance in the reference file, or when a same-recipe walk-forward refresh passes its margin, or when a different recipe is strictly better on that backtest. Logistic `h2_full` does not have to beat the market to receive the alias. The 0.003 feature rule is a different gate. A refusal leaves every alias as it was.
- **Experiment path.** The logical name is `nfl_pregame/registry`. SQLite tracking, including the weekly page job and the tests, keeps that relative name. Databricks tracking stores `/Shared/endzone-mlops/nfl_pregame/registry` unless `ENDZONE_MLFLOW_EXPERIMENT_ROOT` is set. `databricks-uc` is a registry URI and does not rewrite the experiment name. ADR-009.
- **Type.** Logistic regression on feature set `h2_full`: imputation, standard scaling, then an L2 logistic classifier (`endzone/modeling/models.py`, called from `endzone/modeling/registry_command.py`). The penalty and the iteration cap are the command defaults. Those two defaults are not fields in the reference file or a frozen report, so I do not restate them.
- **Who and when.** I maintain this as a personal learning project. ADR-009 is the registry decision. A registration timestamp is not in the reference file or a frozen report.

## Intended use

**Primary use.** A learning and portfolio project. I show a batch, leakage-free home-win probability next to the market on a static weekly page. ADR-006.

**Primary users.** A reader of this repository or of the weekly page who wants to see the pipeline, the registry, and the evaluation together.

**Out of scope.**

- Betting, wagering, staking, or bankroll advice. The page label WINS is the rounded home probability. It is not a recommendation. Nothing here is a betting pick.
- A claim that the model beats the market. The market has the lower log loss.
- Live or in-game win probability. The weekly job is a batch.
- Player props, parlays, or any use of the probability as a price.
- Promoting another feature set without the rule in ADR-005.
- Treating a Tuesday fallback fit as alias `prod`. If the alias is missing, the scorer fits that week in the tracking store and does not write it to Unity Catalog. ADR-009.

## Factors

The scored unit is one game. The label is a home win. The reference file has no slice by team, quarterback, or stadium. The frozen H5 report does list logistic `h2_full` by season. I copy that fold table in the quantitative section.

`h2_full` has no injury column and no inactive column. Depth-chart starter ratings are not in this feature set. H5 added a quarterback rating in the same frozen report and was not promoted.

## Metrics

Log loss is the selection metric. Lower is closer to the outcomes. I also report Brier, AUC, and accuracy where the cited file has them. Accuracy treats a probability of 0.5 or higher as a home win (`docs/experiments/reports/phase2_h5_comparison.md`).

A new feature set is accepted only when pooled log loss improves by at least 0.003 and the 95% paired bootstrap interval lies entirely below 0 (ADR-005 and the same frozen report). The bootstrap in the reference file is 1000 resamples, seed 42. The market can receive the label `accepted` from that numeric rule. ADR-005 says that label is a baseline comparison, not a promotion. The frozen H5 report says no model is promoted from that report.

## Evaluation data

Rolling-origin CV, seasons 2016 through 2023. `docs/experiments/reference_metrics.json` records `cv_start_season` 2016 and `cv_end_season` 2023. For each season S, training uses `season < S` and scoring uses `season == S`. The frozen H5 report states that 2024 is the final validation and is not a CV fold, and that 2025 is not a CV fold. The one-shot holdout is a later frozen report, quoted below. The CV commands still do not score those seasons.

The market probability is the no-vig closing moneyline. It is skipped when those columns are absent (same frozen report). It is a benchmark. It is not a training feature.

Elo uses K 20, home advantage 65, mean rating 1505, and season reversion 0.333 on a scale of 400. There is no margin-of-victory multiplier (same frozen report).

The home baseline is `baseline-home`. On each fold the probability is the training home-win rate. Its locked log loss is in the reference file.

## Training data

Games and play-by-play come from nflverse. The credit is `docs/ATTRIBUTION.md`. A feature for a game uses only rows with `gameday` strictly before that game.

Feature set `h2_full` is the `current` pregame levels (recent offensive EPA, recent defensive EPA, recent win rate, rest days, and season offensive EPA, home and away), plus season-to-date defensive EPA allowed, plus the regressed recent and season EPA columns. The names are `PREGAME_FEATURE_COLUMNS_H2_FULL` in `endzone/features/pregame.py`. Lookback games: 10 (`docs/experiments/reports/phase2_h5_comparison.md`). Regressed EPA starts from the previous season's final value, shrunk by 0.333 toward that season's league mean, then blends in current-season games with weight n/(n+4) (`docs/experiments/reports/phase2_h2_comparison.md`). Week 1 of a season with no prior stays missing (same H2 report).

The CV out-of-fold count for logistic `h2_full` is n=2191 in the reference file. The registered artifact is a different object: one fit on completed games through season 2023. The row count of that fit, and its in-sample loss, are not in the reference file or a frozen report.

The 2016 fold in the frozen H5 report trains on seasons listed from 1999. That count is the fold. It is not the registered fit.

## Quantitative analyses

### Locked reference

Source: `docs/experiments/reference_metrics.json`. The tolerance in that file is 0.0001. `make verify-reference` fails when log loss, n, Elo, or the market differ by more than that tolerance.

The file stores Brier, AUC, and accuracy only on logistic `h2_full`. I leave the other cells blank here.

| Model | Feature set | n | Log loss | Brier | AUC | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | h2_full | 2191 | 0.6377 | 0.2235 | 0.6843 | 0.6426 |
| elo | elo | 2191 | 0.6529 |  |  |  |
| market | market | 2190 | 0.6120 |  |  |  |
| baseline-home | none | 2191 | 0.6888 |  |  |  |

The market has the lower log loss: market 0.6120, logistic `h2_full` 0.6377. Source: `docs/experiments/reference_metrics.json`.

### Paired comparison on the same window

Source: `docs/experiments/reports/phase2_h5_comparison.md`. `log_loss_diff` is the row log loss minus logistic `h2_full` log loss. A negative difference favors the row. The interval is the 2.5 and 97.5 percentiles of the paired bootstrap (1000 resamples, seed 42).

| Model | Feature set | n | Log loss | Brier | AUC | Accuracy | Diff | CI low | CI high | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-home | none | 2191 | 0.6888 | 0.2478 | 0.5006 | 0.5504 | 0.0511 | 0.0389 | 0.0630 | rejected |
| logistic | h2_full | 2191 | 0.6377 | 0.2235 | 0.6843 | 0.6426 | 0.0000 | 0.0000 | 0.0000 | reference |
| logistic | h5_qb | 2191 | 0.6355 | 0.2225 | 0.6869 | 0.6408 | -0.0022 | -0.0042 | -0.0003 | rejected |
| elo | elo | 2191 | 0.6529 | 0.2306 | 0.6655 | 0.6093 | 0.0152 | 0.0073 | 0.0235 | rejected |
| market | market | 2190 | 0.6120 | 0.2121 | 0.7182 | 0.6594 | -0.0259 | -0.0340 | -0.0174 | accepted |

The market row is the one with the lower log loss. The decision `accepted` on that row is the numeric rule against the reference. It is not a promotion and it is not a betting pick. The report says no model is promoted from that file.

`h5_qb` is the closest challenger in this file. The interval lies entirely below 0, and the gain is short of 0.003, so the decision is rejected. I did not move the 0.003 threshold after that result.

### Logistic h2_full by season

Source: `docs/experiments/reports/phase2_h5_comparison.md`, section `logistic (h2_full)`.

| Season | n train | n eval | Log loss |
| --- | --- | --- | --- |
| 2016 | 4515 | 267 | 0.6344 |
| 2017 | 4782 | 267 | 0.6204 |
| 2018 | 5049 | 267 | 0.6192 |
| 2019 | 5316 | 267 | 0.6472 |
| 2020 | 5583 | 269 | 0.6283 |
| 2021 | 5852 | 285 | 0.6669 |
| 2022 | 6137 | 284 | 0.6349 |
| 2023 | 6421 | 285 | 0.6480 |

The folds are not interchangeable. The file does not report a paired interval between seasons, so I do not add one.

### Why h2_full is the reference

Source: `docs/experiments/reports/phase2_h2_comparison.md`. That run's reference is logistic `current`, not the market. Logistic `h2_full` log loss is 0.6377, difference -0.0037, interval -0.0071 to -0.0003, decision accepted. I do not read that interval as a result against the market. The market comparison is the H5 table above.

### 2024 validation and 2025 test

Source: `docs/experiments/reports/holdout_2024_2025.json` and `docs/experiments/reports/holdout_2024_2025.md`. ADR-010. One run, 2026-11-13. Logistic `pkg_core_inj_l2` (C = 0.1, 47 columns) is the selected row. Logistic `h2_full` (C = 1.0) is the prior `prod` recipe. 2024 trains on seasons strictly before 2024. 2025 trains on seasons strictly before 2025. The pooled row concatenates those out-of-sample predictions. I round the file's log loss to four decimals. The file keeps the full precision.

| Split | n | pkg_core_inj_l2 | h2_full | market |
| --- | --- | --- | --- | --- |
| pooled 2024-2025 | 570 | 0.6171 | 0.6207 | 0.5976 |
| 2024 | 285 | 0.6016 | 0.6074 | 0.5892 |
| 2025 | 285 | 0.6326 | 0.6339 | 0.6059 |

The other pooled log losses in the same file are `elo_mov` 0.6249, Elo 0.6402, and the home baseline 0.6911.

Versus logistic `h2_full`, the pooled difference is -0.0035, interval -0.0115 to +0.0038 (1000 resamples, seed 42). The gain is past -0.003, and the interval includes 0, so ADR-005 is not met. I do not promote `pkg_core_inj_l2`. `prod` stays logistic `h2_full` under ADR-009. The frozen file records `promoted` false and `registered` false.

Versus the market, the pooled difference is +0.0196, interval +0.0049 to +0.0350. The interval is entirely above 0. The market is clearly better. This is not betting advice.

Pooled ECE for `pkg_core_inj_l2` is 0.0511 (10 equal-count bins, n=570). The same file has 0.0585 in 2024 and 0.0476 in 2025. The CV ECE 0.0242 is the figure I recorded for this fit on issue 61. It is not a field in the frozen holdout file. Calibration is worse out of time than that CV figure.

The market-information `model_coef` in the frozen file is 0.154 on the pool, 0.441 in 2024, and -0.152 in 2025. The 2025 coefficient is negative. I do not treat any of those coefficients as a blend or a publishing weight.

Holdout log loss 0.6171 is not worse than the CV log loss 0.6330 (the frozen file stores that CV figure as 0.633). This run does not show overfitting on the 2016-2023 window.

Limits of this split: each season is one 285-game sample, and the intervals are wide. The comparison with `h2_full` includes 0. A second season can move the gap, as 2024 and 2025 already do. I do not select a different model from this file.

## Ethical considerations

This is a learning project with production-shaped patterns. It is not betting advice, and it is not a production betting system.

The market has the lower log loss. A weekly probability printed next to the market can still be misread as a pick. The page says the numbers are not betting picks. This card says the same thing. I do not name a side to bet.

The market row is a closing price (`docs/experiments/reports/phase2_h5_comparison.md` calls it the no-vig closing moneyline). I publish the slate on the Tuesday Pages schedule, before later-week information, including injury reports. The line can move after that publish. I do not have a number, in the reference file or a frozen report, for how far the price moves between the publish and the close. The locked comparison still favors the market. It is not a same-time comparison, and it is not a betting pick.

Team names on the site belong to their owners. The site is not affiliated with the NFL. Data credits are in `docs/ATTRIBUTION.md`.

The published sample is small. ADR-008 says the sample is small, and the page labels a short record that way. A count of published games is not in the reference file or a frozen report, so I do not state one. A short public record does not replace the CV table.

## Caveats and recommendations

- Version 1 is one fit on games through 2023. A later refresh uses the same recipe on completed games before the slate. The warehouse pool from that backtest is in `docs/experiments/walk_forward.md`. It is not in the reference file. The bootstrap interval there is descriptive.
- I did not add a feature for this window. `h2_full` already has rolling EPA, win rate, rest, season EPA, and regressed EPA. Point differential stays out. Elo stays a baseline. Injury reports stay out because the slate is published before them.
- The slate is published on Tuesday, before injury reports. `h2_full` has no injury column and no inactive column. The injury-report calendar is not written down in this repo, so I do not state a clock time for those reports.
- If alias `prod` is missing, the Tuesday job fits that week in the tracking store. That fit is not version 1, and it is not written to Unity Catalog. ADR-009.
- The market row has n=2190. The other locked rows have n=2191. The frozen H5 report skips the market when the moneyline columns are absent. The identity of any omitted game is not in the reference file, so I do not name one.
- H5b is rejected in `docs/experiments/phase2_h5b_starter_coverage.md`. `docs/experiments/reports/README.md` says there is no frozen comparison file for that run. I do not copy a log loss from that note.
- A fold-tuned L2 fit did not replace plain logistic `h2_full`. The frozen file is `docs/experiments/reports/phase2_ridge_comparison.md`. I do not restate that table here.
- Margin and total are other models. ADR-007. I do not report them on this card.
- I recommend reading the card and the page as an open experiment. I do not recommend using either one as a betting input.

## Reproduce

The locked numbers are `docs/experiments/reference_metrics.json`. The frozen comparison I quote against the market, Elo, and the home baseline is `docs/experiments/reports/phase2_h5_comparison.md`. I do not hand-edit those files.

On a machine that has the warehouse credentials, which CI does not:

```bash
make verify-reference
```

That refits logistic `h2_full`, Elo, and the market, and it fails when they move from the reference file by more than its tolerance.

Registration is a separate step. I run it myself. It needs `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN` in `.env` (copy `.env.example`, do not commit it), plus:

```bash
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
```

Then, as the runbook records, the through-2023 path is:

```bash
make verify-reference
make register-model
make promote-model VERSION=1
```

A later prod candidate, still the same feature set, is a different sequence. I run it myself. The exact order, including the week 10 publish, is in `docs/runbook.md`. Short form:

```bash
make ingest-current
make dbt-build-databricks
make walk-forward SLATE_SEASON=<slate season> SLATE_WEEK=<slate week>
make verify-walk-forward
make register-recent
make promote-model VERSION=<printed version>
```

The runbook has the season and week for the next slate. I set both flags. I do not use the default slate.

`make verify-walk-forward` has to exit 0 before `make register-recent`. I use the version the register command prints. I do not assume it is 2. `make reproduce` installs the lock, builds the Databricks marts, and runs the reference check. It does not download nflverse. It does not register a model, and it does not set an alias. A push to `main` does not score the next week.
