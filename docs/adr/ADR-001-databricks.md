# ADR-001: Databricks + dbt + MLflow as Primary Stack

**Date**: 2026-10-06  
**Status**: Accepted  
**Spike Status**: PENDING_OWNER (requires completion of Databricks Free Edition configuration)

## Context

Building a public MLOps portfolio to demonstrate lakehouse architecture, analytics engineering, and ML experiment tracking. The project needs:

1. A cloud data platform with free/community tier for learning
2. SQL transformation layer with version control and testing
3. ML experiment tracking with model registry
4. CI/CD integration
5. Resume-friendly technologies (hiring signal for DE/MLE/MLOps roles)

Constraint: Must be reproducible with zero cost for initial 14-day development (Day 1-14). Paid services acceptable post-Day 14 with explicit approval.

## Decision

**Primary stack**: Databricks Free Edition + dbt-databricks + MLflow + GitHub Actions

**Plan B**: Snowflake trial (30 days) + dbt-snowflake + MLflow sidecar, **only if Databricks Free Edition spike fails**. Never operate both in parallel.

**Note**: Databricks Free Edition replaced Community Edition (retired in 2025). Free Edition is serverless-only with Unity Catalog enabled by default.

## Rationale

### Why Databricks Primary

| Criterion | Databricks Free Edition + dbt + MLflow | Snowflake + dbt + MLflow |
|-----------|----------------------------------------|--------------------------|
| MLOps signal | **High** (MLflow native, unified platform) | Medium (MLflow separate deployment) |
| dbt in free tier | **Validated** (serverless SQL warehouse, Unity Catalog) | Mature in trial, established patterns |
| FDE/DE signal | Medium-High (Spark + Delta Lake) | High (SQL warehouse, established) |
| Post-14d continuity | **Forever free** with usage quotas | Trial expires in 30 days |
| Learning alignment | Matches TC5061 MLflow curriculum | Additional sidecar complexity |
| Industry adoption | Strong (Unity Catalog, lakehouse narrative) | Strong (enterprise warehouse) |

### Databricks Advantages

1. **Unified lakehouse**: Delta Lake, Spark SQL, and MLflow in one platform
2. **MLOps native**: Experiment tracking, model registry, and deployment built-in
3. **Curriculum alignment**: Reuses TC5061 MLflow labs patterns (NOT copying deliverables)
4. **Hiring signal**: Strong for ML Engineering and MLOps roles
5. **Real product path**: Can scale to real NFL predictions if project continues

### Known Risks

| Risk | Mitigation |
|------|------------|
| Serverless-only (no custom clusters) | Accept; serverless adequate for learning and small datasets |
| Serverless SQL warehouse auto-stops | Accept; document in runbook; restart when needed |
| Usage quotas and fair-use limits | Monitor usage; quotas sufficient for learning project |
| Unity Catalog learning curve | Spike validates catalog/schema setup; document in runbook |

## Day 1 Spike Checklist

Must validate within Day 1 (see `../runbook.md` for detailed status):

- [ ] Create Databricks Free Edition account
- [ ] Start Serverless Starter SQL Warehouse (pre-created, no cluster needed)
- [ ] Generate personal access token for dbt and MLflow
- [ ] Run `dbt compile` against serverless SQL warehouse with Unity Catalog
- [ ] Create serverless notebook and log test MLflow run (params + metrics)
- [ ] Configure GitHub Actions secrets (DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN)

**Spike outcome decision**:
- **PASS**: Continue with Databricks Free Edition as primary
- **FAIL**: Activate Plan B Snowflake same day; create ADR-001b with rationale

**Current status**: PENDING_OWNER (completion of Databricks Free Edition signup and configuration steps required)

## Plan B: Snowflake

If Databricks Free Edition spike fails, switch to:
- Snowflake 30-day trial (no credit card until trial ends)
- dbt-snowflake adapter (mature, well-documented)
- MLflow sidecar (separate tracking server or local ./mlruns)
- **Same repository layout** (only adapter and profiles.yml change)

ADR-001b will document the switch with specific failure reasons.

## NFL Dataset Decision

**Historical source**: nflverse (CC BY 4.0)

### Why nflverse

1. **License**: Creative Commons Attribution 4.0 - clear for public portfolio
2. **Coverage**: Play-by-play data from 1999-present (~27 seasons)
3. **Format**: Parquet files, CE/Spark-friendly
4. **Quality**: Clean, maintained, ~370 columns per play
5. **Features**: Includes closing lines (spread, total, moneyline) in games table
6. **Community**: Active nflverse GitHub org, documented schemas

### Live Path (Documented, Not Implemented v0.1)

- **Interface**: HistoricalFeed vs LiveFeed (same contract)
- **Future target**: BALLDONTLIE NFL GOAT API (~$39.99/month) for v0.2+
- **v0.1 status**: LiveFeed is stub (raises NotImplementedError)
- **Decision gate**: Live activation requires explicit owner approval, post-Day 14

**Important**: nflverse is NOT a live API. It provides clean nightly updates and raw data ~15 minutes post-game. For in-play predictions (future), a second vendor (BALLDONTLIE) is documented but not purchased in v0.1.

### Alternatives Considered

| Dataset | License | Coverage | Why Not Primary |
|---------|---------|----------|-----------------|
| Wyscout Figshare | CC BY 4.0 | Soccer 2017-18 | Domain preference: NFL for US hiring signal |
| OpenF1 | Non-commercial/edu | F1 2023+ | Shorter history; non-commercial restriction |
| StatsBomb open-data | User Agreement | Soccer events | Redistribution restrictions |
| Sportmonks | Commercial | Soccer multi-league | Paid tier required for Liga MX |

## Consequences

### Positive

- Unified platform reduces cognitive load (one authentication, one UI)
- MLflow tracking ready out-of-box
- Strong resume signal: "Databricks + dbt + MLflow"
- Delta Lake format aligns with modern data engineering practices
- Can demonstrate lakehouse medallion architecture (bronze/silver/gold)

### Negative

- CE limitations may force compromises (no Jobs = GitHub Actions schedule)
- Cluster auto-terminates (must restart for demos)
- If CE spike fails, lose Day 1 time pivoting to Plan B
- Free tier continuity uncertain beyond 14 days

### Neutral

- dbt models identical across Databricks/Snowflake (SQL + Jinja)
- MLflow tracking URI is configuration detail
- Python training code platform-agnostic

## Decision Owners

- **Primary**: Carlos Galvan (portfolio owner, must complete CE spike)
- **Advisor**: Idaho agent (scaffolds, documents spike checklist)
- **Reviewer**: Leto II (curriculum alignment, technical design review)

## Related Documents

- [Runbook](../runbook.md): CE spike step-by-step guide and status
- [Architecture](../architecture.md): Lakehouse design and data flow
- [Outline](../../uploads/outline_14d_2026-10-06.md): Full 14-day plan

## References

- Databricks Free Edition: https://www.databricks.com/try-databricks/signup-form
- dbt-databricks adapter: https://docs.getdbt.com/reference/warehouse-setups/databricks-setup
- nflverse: https://github.com/nflverse
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html

---

**Next action**: Complete Databricks Free Edition account creation and validation of spike checklist items (estimated 45-60 minutes).
