# ADR-002: CCDS layout with a top-level dbt project

**Date**: 2026-11-03
**Status**: Accepted

## Context

Cookiecutter Data Science v2 puts a named Python package at the repository root and uses `data/`, `models/`, `notebooks/`, `references/`, and `reports/figures/` for local files. This repo already has a dbt project, Databricks marts, and a training entrypoint whose report paths are part of the recorded experiment notes.

The package used to be named `src`. Imports were `from src...`, and commands were `python -m src...`. That name is a layout convention, not the project name.

## Decision

The Python package is `endzone`. It is discovered by setuptools from the repository root (`uv sync --frozen`).

dbt stays a top-level project in `dbt/`. Staging and marts are the warehouse equivalent of interim and processed data. They are not copied into local `data/interim` or `data/processed`.

Local directories:

| Path | Role |
| --- | --- |
| `data/{raw,interim,processed,external}/` | Local cache. Git tracks `.gitkeep` only. |
| `models/` | Gitignored serialized artifacts. There is no `endzone/models` package. |
| `notebooks/` | Exploration only. Name a notebook `N.N-initials-description.ipynb`. |
| `references/` | External data notes. Attribution stays in `docs/ATTRIBUTION.md`. |
| `reports/figures/` | CCDS figures directory. Git tracks `.gitkeep` only. |

Training and audit commands keep their current output paths: `reports/phase0`, `reports/phase1`, `reports/phase2*`, and `reports/epa_filter_check.md`. Local MLflow artifacts stay in `mlartifacts/`. Those paths are gitignored. Moving them would change where a run writes its comparison, so this decision does not move them.

`scripts/check_structure.py` fails when a required directory or file is missing, when a top-level `src/` exists, when a non-`.gitkeep` file is tracked under `data/`, `models/`, or `reports/`, or when a Python file is tracked outside `endzone/`, `tests/`, and `scripts/`. `make lint-structure` runs that check and `lint-imports`. CI runs both. Windows uses `.\scripts\dev.ps1 lint-structure`.

Import-linter contracts match the packages as they are today:

1. Layers, higher to lower: `endzone.analysis`, then siblings `endzone.train` and `endzone.ingest`, then siblings `endzone.features` and `endzone.data`, then `endzone.utils`. Siblings do not import each other. A lower layer does not import a higher layer.
2. `endzone.features` does not import `endzone.train`.
3. `endzone.features` does not import `mlflow`.
4. `endzone.ingest` does not import `endzone.train`.
5. `endzone.analysis` does not import `endzone.train`, except `endzone.analysis.feature_audit -> endzone.train.train`. The phase 0 audit still calls the training pipeline. That ignore goes away when the audit no longer needs those functions.

`configs/` is not added. No experiment YAML is loaded yet.

## Consequences

`endzone/config.py` is the only module that reads `.env`, `DATABRICKS_*`, and `MLFLOW_TRACKING_URI`. `endzone/data/warehouse.py` is the SQL and Volume client. Ingest is split across `cli.py`, `nflverse.py`, `schema.py`, `depth_charts.py`, `snapshots.py`, and `volume.py`. The command stays `python -m endzone.ingest.load_raw_nflverse`. `endzone.config` is outside the import-linter layers because every package may import it, and it does not import analysis, train, ingest, features, or data.

Commands use `python -m endzone.train.train` and `python -m endzone.ingest.load_raw_nflverse`. Logger names follow `__name__`, so they are `endzone.*`. MLflow tags (`phase`, `hypothesis`, `git_commit`, `decision`, `feature_set`, `cv_scheme`, `reference_model`, `reference_feature_set`) do not embed the package name. The logged model artifact is still named `model`. The reviewed skops allowlist is still `numpy.dtype` and `sklearn.tree._tree.Tree`.

New code respects this layout and these contracts. `make lint-structure` is the check.
