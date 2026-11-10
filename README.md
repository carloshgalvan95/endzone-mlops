# endzone-mlops

[![CI](https://github.com/carloshgalvan95/endzone-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/carloshgalvan95/endzone-mlops/actions/workflows/ci.yml)

This is a learning project with production-shaped patterns. Batch only. It is not betting advice.

## What is live

The weekly page is [https://carloshgalvan95.github.io/endzone-mlops/](https://carloshgalvan95.github.io/endzone-mlops/). It shows one slate: the model's home win probability next to the market, an estimated score, and the final score once nflverse has it. The Track record on that page is the public count ([ADR-008](docs/adr/ADR-008-weekly-track-record.md)). Week-by-week rates change. They are not copied here.

Estimated points are shown because both margin and total passed the pre-registered test ([ADR-007](docs/adr/ADR-007-score-regression.md)). On the reference table below, market log loss is 0.6120 and the model is 0.6377. The market is closer. This does not beat the market. The model card is [docs/model_card.md](docs/model_card.md).

Published week files are not on `main`. The durable copy is the orphan branch `published-slates` ([ADR-006](docs/adr/ADR-006-weekly-predictions.md)). `predictions/` at the tip is the current slate, and `history/` keeps every earlier file. A published slate is frozen. A later deploy restores that branch and does not rescore it.

## How it runs weekly

Times are UTC, as the workflow crons set them.

[Pages](.github/workflows/pages.yml) scores on Tuesday at 12:00 (`0 12 * * 2`). That run refreshes the current season, builds the Databricks marts, restores `published-slates`, and writes the next unpublished week. It loads alias `prod` when the registry has it. If the alias is missing or the registry cannot be read, it logs that and fits logistic `h2_full` on completed games strictly before the scored week. It then overlays finals, writes `site/data/track_record.json`, archives the week files, and deploys `site/`. A push to `main` restores the branch, overlays finals, writes the track record, archives, and deploys. A push does not fit a model. If the archive write fails, the site is not deployed. A manual dispatch can score a chosen week. `replace_published` rebuilds every scheduled game for that season and week and trains only on games before it. `backfill_archive` seeds the branch from the live site and does not score or deploy.

[Results](.github/workflows/refresh-results.yml) adds finals to the slate that is already published. It does not use warehouse credentials and it does not rescore. It restores `published-slates`, overlays nflverse finals, writes the derived track record, archives the week files, and deploys. After the track record it opens or updates one issue when the published gap (model log loss minus market log loss) exceeds the reference CV gap plus a margin. A window that matches the reference table stays quiet. That step does not fail the deploy. See [ADR-009](docs/adr/ADR-009-model-registry.md). The crons are Friday 08:30 (`30 8 * * 5`, after Thursday night), every hour at minute 15 on Sunday (`15 * * * 0`), Monday 00:15 through 08:15 (`15 0-8 * * 1`, after the late Sunday kickoff), and Tuesday 08:30 (`30 8 * * 2`). The Tuesday run is the early pass for Monday Night Football, before the 12:00 scoring deploy.

Dropping a `game_id` that is already archived fails. Adding games is allowed. The intentional rebuild is `replace_published`, or `make rebuild-week` locally.

## Problem

Predict the home-team win before kickoff, and estimate the score from a separate margin and a separate total. The win-probability question is whether a leakage-free pregame feature set improves on a fixed reference on rolling-origin log loss. The score question is fixed in [ADR-007](docs/adr/ADR-007-score-regression.md). Weekly output is a batch page. See [ADR-006](docs/adr/ADR-006-weekly-predictions.md).

## Data

nflverse, CC BY 4.0. Games, play-by-play, and depth charts. Credits: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).

Bronze is `workspace.nfl_raw` (`games`, `play_by_play`, `depth_charts`). dbt staging views are `workspace.nfl_dev_staging`. Marts are `workspace.nfl_dev_marts`: `fct_games`, `fct_team_game_epa`, `fct_upcoming_games`, `fct_expected_starter`, `fct_qb_dropbacks`. The target schema is `nfl_dev`, and dbt appends the custom schema, so do not query staging or marts from unsuffixed `nfl_dev`.

Train through 2023, validation 2024, test 2025. A feature for a game uses only rows with `gameday` strictly before that game. Rolling-origin CV scores seasons 2016-2023. 2024 is not a fold. 2025 is not scored.

## Architecture

```text
endzone-mlops/
  MLproject  Makefile  pyproject.toml  uv.lock
  .github/workflows/   ci.yml, pages.yml, refresh-results.yml, pr-guard.yml, databricks-smoke.yml
  configs/experiments/ one YAML per hypothesis, plus s2.yaml for margin and total
  data/{raw,interim,processed,external}/   local cache, git keeps .gitkeep
  dbt/                 staging views, marts, seeds
  docs/                ADRs, model card, experiments, runbook
  endzone/             ingest, data, features, analysis, modeling, site, train, utils
  models/              gitignored serialized artifacts
  notebooks/           exploration only
  site/                static weekly page (index.html, assets, sample and team data)
  references/  reports/figures/  scripts/  tests/
```

Flow: nflverse -> bronze (`endzone/ingest`) -> dbt staging -> dbt marts -> `endzone/features` -> `endzone/modeling` -> local MLflow (`sqlite:///mlflow.db`, artifacts in `mlartifacts/`). `endzone/site` overlays finals, archives `published-slates`, and writes the track record. It does not fit a model.

`endzone/config.py` reads `.env`. `endzone/data/warehouse.py` is the SQL and Volume client. Hypothesis files are `configs/experiments/*.yaml`. The canonical command is `python -m endzone.modeling.train`. `python -m endzone.train.train` is the same flags. Weekly scoring is `python -m endzone.modeling.score`.

MLflow experiments: `nfl_pregame` (single split), `nfl_pregame/phase0_audit`, `nfl_pregame/phase1_baselines`, `nfl_pregame/phase2_h1` through `phase2_h5b` and `phase2_ridge`, `nfl_pregame/sprint2_score`, `nfl_pregame/weekly`, and `nfl_pregame/registry`. Those are the sqlite names. A Databricks tracking URI stores a relative name under `/Shared/endzone-mlops` unless `ENDZONE_MLFLOW_EXPERIMENT_ROOT` is set. A CV run logs one parent and one nested child per model. Child tag `decision` is `accepted`, `rejected`, `reference`, `baseline`, or `inconclusive`.

CI is [`.github/workflows/ci.yml`](.github/workflows/ci.yml), on push and pull request to `main`. Ubuntu runs ruff, the structure and import check, vulture, pytest, analysis coverage, and the local DuckDB dbt compile. Windows and macOS run ruff, the structure and import check, and pytest.

## Results

Rolling-origin CV, seasons 2016-2023. Source: [docs/experiments/reference_metrics.json](docs/experiments/reference_metrics.json). Logistic `h2_full` also has Brier 0.2235, AUC 0.6843, accuracy 0.6426.

| Model | Feature set | n | Log loss |
| --- | --- | --- | --- |
| logistic | h2_full | 2191 | 0.6377 |
| elo | elo | 2191 | 0.6529 |
| market | market | 2190 | 0.6120 |
| baseline-home | none | 2191 | 0.6888 |

The market is a baseline. It is not the model under test, and it has the lower log loss. The model does not beat the market. `make verify-reference` refits these rows and fails when log loss or n moves by more than 1e-4. Intended use, the registry name, and these limits are in the [model card](docs/model_card.md).

## Hypothesis log

Same window, n=2191, unless the note says otherwise. Detail is in `docs/experiments/`.

| Hypothesis | Decision | Against | What happened |
| --- | --- | --- | --- |
| H1 differentials | rejected | logistic `current` | `diff` 0.6414 (difference +0.0000), `diff_only` 0.6416 (+0.0002) |
| H2 season defense and regression | accepted | logistic `current` | `h2_full` 0.6377, difference -0.0037, interval -0.0071 to -0.0003. `h2_def` rejected |
| H3 pass and rush EPA | rejected | logistic `h2_full` | `h3_split` 0.6393 (+0.0016), `h3_replace` 0.6395 (+0.0018) |
| Ridge, fold-tuned L2 | did not help | logistic `h2_full` | `logistic_l2` on `h2_full` 0.6386 (+0.0009, interval -0.0008 to 0.0025) |
| H4 clean EPA | rejected | logistic `h2_full` | `h4_clean` 0.6398 (+0.0021), `h4_clean_sr` 0.6379 (+0.0002) |
| H5 starting QB | rejected | logistic `h2_full` | `h5_qb` 0.6355, difference -0.0022, interval -0.0042 to -0.0003 |
| H5b fixed starter | rejected | logistic `h2_full` | `h5_qb` 0.6356, difference -0.0021, interval -0.0039 to -0.0002 |
| Sprint 2 margin and total | accepted | Elo/25 and the league-mean total | Margin MAE 10.155811 versus Elo/25 10.412221 (8 of 8 seasons). Total MAE 10.925618 versus the league mean 11.130543 (5 of 8 seasons). n=2191. Market margin MAE 9.800548 and market total MAE 10.526928 are lower, and they are not a gate. [sprint2_cv_report.json](docs/experiments/reports/sprint2_cv_report.json) |

H2 is why logistic `h2_full` is the reference. H5 and H5b improve on it by less than 0.003. The intervals are entirely below 0, so the sign is a positive signal. The threshold was not moved. Notes: [H5](docs/experiments/phase2_h5_starting_qb.md), [H5b](docs/experiments/phase2_h5b_starter_coverage.md).

## Acceptance rule

On pooled out-of-fold games, accept a feature set only when `log_loss_diff` <= -0.003 and the 95% paired bootstrap interval lies entirely below 0 (1000 resamples, seed 42). Otherwise reject. The rule is [ADR-005](docs/adr/ADR-005-acceptance-rule.md).

Switch the reference to `logistic_l2` on `h2_full` only when that row's `log_loss_diff` is <= 0. The ridge run did not.

Sprint 2 is a different rule, for home margin and total points. Ridge on `h2_full` is accepted only when pooled CV MAE beats its baseline and the 95% paired bootstrap interval of that MAE difference lies entirely below 0, and the model is strictly better in at least 5 of 8 seasons (2016-2023). The margin baseline is the Elo point spread, rating gap including home field divided by 25. The total baseline is the training-fold mean total. Market MAE, RMSE, 80% interval coverage, and implied win log loss versus 0.638 are reported and are not gates. The thresholds are not moved after a result. Both targets were accepted. The frozen report is [sprint2_cv_report.json](docs/experiments/reports/sprint2_cv_report.json). See [ADR-007](docs/adr/ADR-007-score-regression.md).

## Roadmap

Planned, not done.

- Next hypothesis: an opponent-adjusted rating combined with Elo.

The registry and the published log-loss alert are in [ADR-009](docs/adr/ADR-009-model-registry.md). The alert fires when the published gap is worse than the reference CV gap plus a margin. It does not say the model wins. On the reference table the market is closer, and a window that looks like that table stays quiet. The issue is not a betting signal.

## Reproduce

Needs `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN` in `.env` (copy `.env.example`, do not commit it) and `dbt/profiles.yml` (copy `dbt/profiles.yml.example`, catalog `workspace`, target `databricks`). Python 3.12 from `uv.lock`.

`make reproduce` installs the lock, runs `dbt build` on Databricks, and refits the reference. It does not download nflverse. It does not register a model and it does not set an alias. On an empty catalog, load bronze first. `make register-model` and `make promote-model VERSION=<n>` are separate. The runbook has the one-time commands. `make train-cv` scores H5b. It loads bronze and builds marts when `.stamp/` says those steps are missing or bronze is newer. Delete `.stamp` to force them. `make ingest` and `make dbt-build-databricks` still run only their own step.

`scripts/train_sweep.sh` and `scripts/train_sweep.ps1` are the single-split five-config sweep (`make train-sweep`). They are not the reference comparison.

macOS or Linux:

```bash
uv sync --frozen
make data
make reproduce
mlflow run . -e verify_reference --env-manager local
```

`mlflow run . -e train_cv --env-manager local` is the H5b CV. Parameters: `hypothesis`, `reference_model`, `reference_feature_set`, `output_dir`.

## Score a week locally

`make score` loads alias `prod` when the registry has it. If the alias is missing or the registry cannot be read, it logs that and fits logistic `h2_full` on completed games strictly before the scored week, registers that fit in the tracking store, and does not set the alias. It writes the next unpublished week to `site/data/predictions/<season>-w<week>.json` and `site/data/latest.json`. If that week file is already on disk, it is kept. The fit does not use same-week finals as labels or as features. It also runs the Sprint 2 rule. Home points are `(total + margin) / 2` and away points are `(total - margin) / 2`, rounded on the page only. `endzone-margin` and `endzone-total` are registered only when that target passes. Both passed, so the command writes the estimate. `make score SEASON=<year> WEEK=<week>` scores that week when it is not already published. `make rebuild-week SEASON=<year> WEEK=<week>` rebuilds every scheduled game that week, played and unplayed, then overlays finals. A rebuild that would drop a `game_id` already in the week file fails. Windows: set `SEASON` and `WEEK`, then `.\scripts\dev.ps1 score` or `.\scripts\dev.ps1 rebuild-week`. The score command is also `mlflow run . -e score --env-manager local`.

`make train-s2` (Windows: `.\scripts\dev.ps1 train-s2`) writes `reports/sprint2/cv_report.json`. That directory is local scratch. The frozen warehouse copy is already `docs/experiments/reports/sprint2_cv_report.json`. Do not hand-edit the numbers. `mlflow run . -e score_cv --env-manager local` is the same comparison.

The page source is `site/index.html`. The How to read section explains each label on two frozen example cards from 2026 week 9. Those cards are not loaded from the weekly JSON. It loads `site/data/latest.json`. `site/data/sample.json` is an example slate with no season or week. It is shown only when `latest.json` is missing, and the EXAMPLE label stays visible. Generated prediction JSON stays off `main`. Team logos are loaded from nflverse URLs in `site/data/teams.json`. The GitHub Actions schedule is in [How it runs weekly](#how-it-runs-weekly).

`make refresh-results` overlays final nflverse scores onto week files already in `site/data`. It does not refit the model, run dbt, or read the warehouse. Predictions stay as published. Each game has `status` `scheduled` or `final`. Scores are integers and appear only when `status` is `final`. A tie is equal scores, including 0-0. `generated_at` stays the prediction time. `results_updated_at` is when the scores were checked. Windows: `.\scripts\dev.ps1 refresh-results`.

`make track-record` writes `site/data/track_record.json` from the archived slates. Probabilities come from the earliest archived copy of each game. Final scores come from the newest copy. The page shows that table. This README does not repeat the weekly rates. If the archive cannot be read, the file says the record is unavailable and the page does not invent one. The command does not edit a week file. Windows: `.\scripts\dev.ps1 track-record`. Pages and Results write the same file before they deploy. See [ADR-008](docs/adr/ADR-008-weekly-track-record.md).

Scoring needs the warehouse marts, including `fct_upcoming_games`, and the same `.env` as training. The results refresh does not.

Windows (PowerShell). `.\scripts\dev.ps1` matches the Make targets and also has `format`:

```powershell
uv sync --frozen
.\scripts\dev.ps1 data
.\scripts\dev.ps1 reproduce
mlflow run . -e verify_reference --env-manager local
```

Git Bash on Windows rewrites arguments that look like Unix paths. `make` exports `MSYS_NO_PATHCONV=1`. Set it yourself before a Python command outside `make`.

Local DuckDB compile, no warehouse: `make dbt-compile`.

## Quality gates

Listed in [docs/definition_of_done.md](docs/definition_of_done.md): ruff, `make lint-structure`, vulture at confidence 80, pytest, analysis coverage 100%, the golden CV test, `make dbt-compile`, `make verify-reference` on the warehouse, the PR template, Conventional Commits, no secrets. Modules group one responsibility. Size is a smell to review, not a gate.

## Docs

- [Model card](docs/model_card.md)
- [Architecture](docs/architecture.md)
- [ADR-001 Databricks](docs/adr/ADR-001-databricks.md), [ADR-002 CCDS](docs/adr/ADR-002-ccds-layout.md), [ADR-003 config and warehouse](docs/adr/ADR-003-config-and-warehouse.md), [ADR-004 modeling](docs/adr/ADR-004-modeling-and-hypotheses.md), [ADR-005 acceptance rule](docs/adr/ADR-005-acceptance-rule.md), [ADR-006 weekly predictions](docs/adr/ADR-006-weekly-predictions.md), [ADR-007 margin and total](docs/adr/ADR-007-score-regression.md), [ADR-008 track record](docs/adr/ADR-008-weekly-track-record.md), [ADR-009 model registry](docs/adr/ADR-009-model-registry.md)
- [Experiments](docs/experiments/reports/README.md)
- [Runbook](docs/runbook.md)
- [Site identity](docs/design/site_identity.md)
- [Attribution](docs/ATTRIBUTION.md)

## License

MIT. See [LICENSE](LICENSE).
