# Infrastructure Setup

This directory contains infrastructure configuration and setup instructions for Databricks Free Edition.

## Databricks Free Edition Setup

### Prerequisites

1. Create account at [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. Note workspace URL (e.g., `https://community.cloud.databricks.com/`)

### Cluster Configuration

Day 1 spike checklist (see `../docs/runbook.md` for detailed status):

- [ ] Create compute cluster
  - Runtime: DBR 13.3 LTS or later
  - Compute: Serverless (Free Edition is serverless-only, no cluster creation)
  - Terminate after: 120 minutes of inactivity

- [ ] SQL Warehouse (if available in CE)
  - Warehouse size: 2X-Small
  - Auto-stop: 10 minutes

### dbt Setup

Install dbt-databricks adapter:

```bash
pip install dbt-databricks
```

Configure `dbt/profiles.yml` (copy from `dbt/profiles.yml.example`):

```yaml
endzone_mlops:
  target: dev
  outputs:
    dev:
      type: databricks
      host: <your-workspace-url>
      http_path: <sql-warehouse-http-path>
      token: <personal-access-token>
      schema: nfl_dev
```

### MLflow Setup

Databricks Free Edition includes MLflow tracking with serverless compute. No additional setup required.

To log experiments:

```python
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/<your-email>/nfl_pregame")
```

### GitHub Actions Secrets

For CI/CD, add these secrets to your repository:

- `DATABRICKS_HOST`: Your workspace URL
- `DATABRICKS_TOKEN`: Personal access token with cluster access

### Plan B: Snowflake

If Databricks Free Edition spike fails, switch to Snowflake trial:
- 30-day trial at [signup.snowflake.com](https://signup.snowflake.com/)
- Same repository layout
- Use dbt-snowflake adapter instead

See ADR-001 for decision rationale.
