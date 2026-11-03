# endzone-mlops

[![CI](https://github.com/carloshgalvan95/endzone-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/carloshgalvan95/endzone-mlops/actions/workflows/ci.yml)

## Honesty

Public learning portfolio. Production analogues for CV/LinkedIn: Athena / Glue / Redshift Spectrum / Prefect (role-dependent).
Do not invent Databricks, dbt, MLflow, or Snowflake logos on PEMEX or Kavak bullets.

This is a learning project with production-shaped patterns, not production software. Batch only. No live feed in the repo.

## Problem

Predict the home-team win before kickoff. The question is whether a leakage-free pregame feature set beats a fixed reference on rolling-origin log loss. Margin and total are not modeled. There is no betting product and no scoring job yet. `score.py` is not in the repo.

## Data

nflverse, CC BY 4.0. Games, play-by-play, and depth charts. Credits: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).

Bronze is `workspace.nfl_raw` (`games`, `play_by_play`, `depth_charts`). dbt staging views are `workspace.nfl_dev_staging`. Marts are `workspace.nfl_dev_marts`: `fct_games`, `fct_team_game_epa`, `fct_upcoming_games`, `fct_expected_starter`, `fct_qb_dropbacks`. The target schema is `nfl_dev`, and dbt appends the custom schema, so do not query staging or marts from unsuffixed `nfl_dev`.

Train through 2023, validation 2024, test 2025. A feature for a game uses only rows with `gameday` strictly before that game. Rolling-origin CV scores seasons 2016-2023. 2024 is not a fold. 2025 is not scored.

## Architecture

```text
endzone-mlops/
  MLproject  Makefile  pyproject.toml  uv.lock
  configs/experiments/     one YAML file per hypothesis
  data/{raw,interim,processed,external}/   local cache, git keeps .gitkeep
  dbt/                     staging views, marts, seeds
  docs/                    ADRs, experiments, runbook
  endzone/                 Python package
  models/                  gitignored serialized artifacts
  notebooks/               exploration only
  references/  reports/figures/  scripts/  tests/
```

Flow: nflverse -> bronze (`endzone/ingest`) -> dbt staging -> dbt marts -> `endzone/features` -> `endzone/modeling` -> local MLflow (`sqlite:///mlflow.db`, artifacts in `mlartifacts/`).

`endzone/config.py` reads `.env`. `endzone/data/warehouse.py` is the SQL and Volume client. Hypothesis files are `configs/experiments/*.yaml`. The canonical command is `python -m endzone.modeling.train`. `python -m endzone.train.train` is the same flags.

MLflow experiments: `nfl_pregame` (single split), `nfl_pregame/phase1_baselines`, and `nfl_pregame/phase2_h1` through `phase2_h5b` and `phase2_ridge`. A CV run logs one parent and one nested child per model. Child tag `decision` is `accepted`, `rejected`, `reference`, `baseline`, or `inconclusive`.

## Results

Rolling-origin CV, seasons 2016-2023. Source: [docs/experiments/reference_metrics.json](docs/experiments/reference_metrics.json). Logistic `h2_full` also has Brier 0.2235, AUC 0.6843, accuracy 0.6426.

| Model | Feature set | n | Log loss |
| --- | --- | --- | --- |
| logistic | h2_full | 2191 | 0.6377 |
| elo | elo | 2191 | 0.6529 |
| market | market | 2190 | 0.6120 |
| baseline-home | none | 2191 | 0.6888 |

The market is a baseline. It is not the model under test. `make verify-reference` refits these rows and fails when log loss or n moves by more than 1e-4.

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

H2 is why logistic `h2_full` is the reference. H5 and H5b beat it by less than 0.003. The intervals are entirely below 0, so the sign is a positive signal. The threshold was not moved. Notes: [H5](docs/experiments/phase2_h5_starting_qb.md), [H5b](docs/experiments/phase2_h5b_starter_coverage.md).

## Acceptance rule

On pooled out-of-fold games, accept a feature set only when `log_loss_diff` <= -0.003 and the 95% paired bootstrap interval lies entirely below 0 (1000 resamples, seed 42). Otherwise reject. The rule is [ADR-005](docs/adr/ADR-005-acceptance-rule.md).

Switch the reference to `logistic_l2` on `h2_full` only when that row's `log_loss_diff` is <= 0. The ridge run did not.

## Reproduce

Needs `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN` in `.env` (copy `.env.example`, do not commit it) and `dbt/profiles.yml` (copy `dbt/profiles.yml.example`, catalog `workspace`, target `databricks`). Python 3.12 from `uv.lock`.

`make reproduce` installs the lock, runs `dbt build` on Databricks, and refits the reference. It does not download nflverse. It does not register a model. On an empty catalog, load bronze first. `make train-cv` scores H5b. It loads bronze and builds marts when `.stamp/` says those steps are missing or bronze is newer. Delete `.stamp` to force them. `make ingest` and `make dbt-build-databricks` still run only their own step.

`scripts/train_sweep.sh` and `scripts/train_sweep.ps1` are the single-split five-config sweep (`make train-sweep`). They are not the reference comparison.

macOS or Linux:

```bash
uv sync --frozen
make data
make reproduce
mlflow run . -e verify_reference --env-manager local
```

`mlflow run . -e train_cv --env-manager local` is the H5b CV. Parameters: `hypothesis`, `reference_model`, `reference_feature_set`, `output_dir`.

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

- [Architecture](docs/architecture.md)
- [ADR-001 Databricks](docs/adr/ADR-001-databricks.md), [ADR-002 CCDS](docs/adr/ADR-002-ccds-layout.md), [ADR-003 config and warehouse](docs/adr/ADR-003-config-and-warehouse.md), [ADR-004 modeling](docs/adr/ADR-004-modeling-and-hypotheses.md), [ADR-005 acceptance rule](docs/adr/ADR-005-acceptance-rule.md)
- [Experiments](docs/experiments/reports/README.md)
- [Runbook](docs/runbook.md)
- [Attribution](docs/ATTRIBUTION.md)

## License

MIT. See [LICENSE](LICENSE).
