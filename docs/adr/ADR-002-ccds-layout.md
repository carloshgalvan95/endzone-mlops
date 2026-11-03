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

1. Layers, higher to lower: `endzone.analysis`, then `endzone.train`, then siblings `endzone.modeling.train` and `endzone.ingest`, then siblings `endzone.modeling.pipeline` and `endzone.modeling.holdout`, then `endzone.modeling.scoring`, then siblings `endzone.modeling.cv`, `endzone.modeling.tracking`, and `endzone.modeling.report`, then siblings `endzone.modeling.evaluation`, `endzone.modeling.reference`, `endzone.modeling.models`, `endzone.modeling.hypotheses`, and `endzone.modeling.dataset`, then siblings `endzone.features` and `endzone.data`, then `endzone.utils`. Siblings do not import each other. A lower layer does not import a higher layer.
2. `endzone.features` does not import `endzone.train` or `endzone.modeling`.
3. `endzone.features` does not import `mlflow`.
4. `endzone.ingest` does not import `endzone.train` or `endzone.modeling`.
5. `endzone.analysis` does not import `endzone.modeling.train` or `endzone.train`, except `endzone.analysis.feature_audit -> endzone.modeling.train` and `endzone.analysis.feature_audit -> endzone.train.train`. The phase 0 audit calls dataset, model, and tracking helpers. Those ignores go away when the audit no longer needs the training entry point.

`configs/experiments/` holds one YAML file per hypothesis. `endzone.modeling.hypotheses` loads and validates that directory. Adding a hypothesis is a new file, not a new Python constant.

## Consequences

`endzone/config.py` is the only module that reads `.env`, `DATABRICKS_*`, and `MLFLOW_TRACKING_URI`. `endzone/data/warehouse.py` is the SQL and Volume client. Ingest is split across `cli.py`, `nflverse.py`, `schema.py`, `depth_charts.py`, `snapshots.py`, and `volume.py`. The command stays `python -m endzone.ingest.load_raw_nflverse`. `endzone.config` is outside the import-linter layers because every package may import it, and it does not import analysis, train, ingest, features, data, or modeling.

Training lives in `endzone/modeling/`. The canonical command is `python -m endzone.modeling.train`. `python -m endzone.train.train` is a shim with the same flags, and the Makefile, sweep scripts, and the commands below keep that path. Logger names follow `__name__`, so they are `endzone.*`. MLflow tags (`phase`, `hypothesis`, `git_commit`, `decision`, `feature_set`, `cv_scheme`, `reference_model`, `reference_feature_set`) do not embed the package name. The logged model artifact is still named `model`. The reviewed skops allowlist is still `numpy.dtype` and `sklearn.tree._tree.Tree`.

New code respects this layout and these contracts. `make lint-structure` is the check.
