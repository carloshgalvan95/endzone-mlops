# Infrastructure

Databricks Free Edition is the warehouse. The Day 1 spike passed on 2026-10-06. ADR-001 records the result. The procedure is `docs/runbook.md`.

## What is running

- Serverless SQL warehouse (pre-created). Free Edition does not create a custom cluster.
- Unity Catalog catalog `workspace`.
- Bronze schema `nfl_raw`: `games`, `play_by_play`, `depth_charts`, `injuries`.
- dbt target schema `nfl_dev`. Staging is `nfl_dev_staging`. Marts are `nfl_dev_marts`. Seeds stay in `nfl_dev`.
- MLflow on serverless compute, and locally at `sqlite:///mlflow.db`.

The host, HTTP path, and token stay in `.env` and in GitHub Actions secrets. This file does not store them.

## dbt

`dbt-databricks` is in `uv.lock`. Install with `uv sync --frozen` from the repo root.

Copy `dbt/profiles.yml.example` to `dbt/profiles.yml` and do not commit the copy. The example default target is `local` (DuckDB). The warehouse output is `databricks`, catalog `workspace`, schema `nfl_dev`. `make dbt-build-databricks` passes `--target databricks`.

The example file still has an unused `snowflake` output. `dbt-snowflake` is not installed. Do not select that target.

## GitHub Actions

`ci.yml` compiles dbt on DuckDB and does not use Databricks secrets.

`.github/workflows/databricks-smoke.yml` is manual. It uses `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN`.

## Snowflake

Not used. The spike passed, so the fallback was not activated. Do not add a second warehouse.
