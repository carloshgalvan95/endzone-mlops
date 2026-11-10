# ADR-001: Databricks + dbt + MLflow as Primary Stack

**Date**: 2026-10-06
**Status**: Accepted
**Spike Status**: PASS (completed 2026-10-06)

## Context

Building a public MLOps portfolio to demonstrate lakehouse architecture, analytics engineering, and ML experiment tracking. The project needs:

1. A cloud data platform with a free tier for learning
2. SQL transformation layer with version control and testing
3. ML experiment tracking with model registry
4. CI/CD integration
5. A stack a reviewer can recognize as Databricks, dbt, and MLflow

Constraint: reproducible at zero cost for the initial 14-day development window (Day 1-14). This is a learning project with production-shaped patterns. Batch only. It is not betting advice. On the reference table the market has the lower log loss.

## Decision

**Primary stack**: Databricks Free Edition + dbt-databricks + MLflow + GitHub Actions

The Day 1 spike passed on 2026-10-06. The warehouse is the pre-created serverless SQL warehouse. Unity Catalog is on. The catalog in use is `workspace`.

**Snowflake**: A Snowflake trial was the written fallback only if this spike failed. The spike passed, so that fallback was not activated. `dbt-snowflake` is not a dependency. There is no ADR-001b. `dbt/profiles.yml.example` still contains an unused `snowflake` output from that plan. Nothing in this repo selects it. Do not run a second lake.

**Note**: Databricks Free Edition replaced Community Edition (retired in 2025). Free Edition is serverless-only. It does not offer a custom cluster. Unity Catalog is enabled by default.

## Rationale

### Why Databricks was chosen

| Criterion | Databricks Free Edition + dbt + MLflow | Snowflake + dbt + MLflow |
|-----------|----------------------------------------|--------------------------|
| MLOps signal | **High** (MLflow native, unified platform) | Medium (MLflow separate deployment) |
| dbt in free tier | **Validated** (serverless SQL warehouse, Unity Catalog) | Mature in trial, established patterns |
| FDE/DE signal | Medium-High (Spark SQL + Delta Lake) | High (SQL warehouse, established) |
| Post-14d continuity | **Free tier** with usage quotas | Trial expires in 30 days |
| Learning alignment | Matches the MLflow curriculum labs (patterns only, not copied deliverables) | Additional sidecar complexity |
| Industry adoption | Strong (Unity Catalog, lakehouse narrative) | Strong (enterprise warehouse) |

This table is the decision record. Snowflake is not used.

### Databricks advantages that held

1. **One platform**: Delta tables, SQL warehouse, and MLflow in one workspace
2. **MLOps native**: Experiment tracking and a model registry
3. **Curriculum alignment**: Reuses MLflow lab patterns (not copying deliverables)
4. **Readable stack**: Databricks + dbt + MLflow in the README
5. **Batch scope**: Weekly scoring writes JSON. It is not a betting product

### Known limits

| Limit | What the repo does |
|------|------------|
| Serverless-only (no custom clusters) | Accept. dbt and ingest use the serverless SQL warehouse |
| Serverless SQL warehouse auto-stops | Restart it in the workspace when a command needs it. The runbook says so |
| Usage quotas | `store_failures` stays off so `nfl_dev_test_results` does not fill the 100-relation quota |
| Unity Catalog | The spike validated catalog `workspace`, schema `nfl_raw`, and the dbt target `nfl_dev`. The runbook has the schema map |

## Day 1 spike checklist

Completed 2026-10-06. The click-by-click procedure is in `docs/runbook.md`. The host, HTTP path, and token stay in `.env` and in GitHub Actions secrets. They are not written here.

- [x] Create Databricks Free Edition account
- [x] Start Serverless Starter SQL Warehouse (pre-created, no cluster needed)
- [x] Generate personal access token for dbt and MLflow
- [x] Run `dbt compile` against the serverless SQL warehouse with Unity Catalog
- [x] Create a serverless notebook and log a test MLflow run (params and metrics)
- [x] Configure GitHub Actions secrets (`DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN`)

**Spike outcome**: PASS. Databricks Free Edition stays the primary stack.

## Snowflake

Not activated. Do not add a Snowflake target, a `dbt-snowflake` dependency, or ADR-001b.

## NFL dataset decision

**Historical source**: nflverse (CC BY 4.0)

### Why nflverse

1. **License**: Creative Commons Attribution 4.0, clear for a public portfolio
2. **Coverage**: Play-by-play from 1999 to the present
3. **Format**: Parquet files
4. **Quality**: Maintained. Play-by-play is conformed to 372 columns in `endzone/ingest/schemas/pbp_schema.json`
5. **Features**: Closing lines (spread, total, moneyline) are on the games file
6. **Community**: nflverse GitHub org, documented schemas

### Live path

A live feed is not in the repo. The unused feed stub was removed. BALLDONTLIE is a documented vendor for a later version, not a dependency, and it is not purchased. nflverse is not a live API. It publishes nightly updates and raw data after games.

### Alternatives considered

| Dataset | License | Coverage | Why not primary |
|---------|---------|----------|-----------------|
| Wyscout Figshare | CC BY 4.0 | Soccer 2017-18 | Domain preference: NFL |
| OpenF1 | Non-commercial/edu | F1 2023+ | Shorter history; non-commercial restriction |
| StatsBomb open-data | User Agreement | Soccer events | Redistribution restrictions |
| Sportmonks | Commercial | Soccer multi-league | Paid tier required |

## Consequences

### Positive

- One workspace for SQL and MLflow
- MLflow tracking is available on serverless compute and locally at `sqlite:///mlflow.db`
- The README can name Databricks, dbt, and MLflow without a second platform
- Bronze tables are Delta (`CREATE OR REPLACE TABLE ... USING DELTA`)
- The medallion in this repo is bronze (`nfl_raw`), staging views, and marts. See `docs/architecture.md`

### Negative

- No custom clusters. A Databricks Jobs cluster is not in the repo. The schedule is GitHub Actions
- The SQL warehouse auto-stops
- Free Edition usage quotas apply, including the 100-relation schema cap
- `dbt build` against the warehouse is a local or manual step (`make dbt-build-databricks`). CI compiles with DuckDB and does not build the warehouse

### Neutral

- The same dbt SQL compiles on DuckDB and builds on Databricks. There is no Snowflake project
- The MLflow tracking URI is configuration
- Training code reaches the warehouse through `endzone/data/warehouse.py`

## Decision owner

Carlos Galvan.

## Related documents

- [Runbook](../runbook.md): spike procedure and current operations
- [Architecture](../architecture.md): tables, columns, and data flow

## References

- Databricks Free Edition: https://www.databricks.com/try-databricks/signup-form
- dbt-databricks adapter: https://docs.getdbt.com/reference/warehouse-setups/databricks-setup
- nflverse: https://github.com/nflverse
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
