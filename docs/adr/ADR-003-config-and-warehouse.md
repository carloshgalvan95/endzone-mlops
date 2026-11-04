# ADR-003: One settings module and one warehouse client

**Date**: 2026-11-03
**Status**: Accepted

## Context

Databricks credentials, the tracking URI, and `.env` were resolved in more than one place. Ingest and training each opened the warehouse their own way. A host or HTTP path copied into a second module is easy to commit.

## Decision

`endzone/config.py` is the only module that reads `.env`, `DATABRICKS_*`, and `MLFLOW_TRACKING_URI`. A variable already set in the process wins.

`endzone/data/warehouse.py` is the only SQL and Volume client. Ingest and training both use it. One statement per `cursor.execute`. Catalog and schema are connection parameters.

dbt does not use this loader. `make dbt-compile`, `make dbt-run`, and `make dbt-test` still source `.env`. `dbt/profiles.yml` stays gitignored. Compare it to `dbt/profiles.yml.example` after every pull.

## Consequences

Do not read those variables anywhere else, and do not put a warehouse host or HTTP path in the repo. CI tests the client with fixtures. A bad warehouse statement still has to be checked against Databricks.
