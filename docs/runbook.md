# Operational Runbook

This document contains operational procedures, setup checklists, and troubleshooting guides for the endzone-mlops project.

## Table of Contents

1. [Databricks Community Edition Spike](#databricks-community-edition-spike)
2. [Local Development Setup](#local-development-setup)
3. [GitHub Actions CI/CD](#github-actions-cicd)
4. [MLflow Tracking](#mlflow-tracking)
5. [Troubleshooting](#troubleshooting)

---

## Databricks Community Edition Spike

**Status**: PENDING_OWNER (requires Carlos to complete)

**Time estimate**: 60-90 minutes

**Decision criteria**: If ANY item fails, activate Plan B (Snowflake) same day.

### Spike Checklist

#### 1. Account Creation
- [ ] Navigate to [community.cloud.databricks.com](https://community.cloud.databricks.com/)
- [ ] Sign up with email (no credit card required)
- [ ] Verify email and complete onboarding
- [ ] Note workspace URL (e.g., `https://community.cloud.databricks.com/`)

#### 2. Compute Cluster
- [ ] Click "Compute" in left sidebar
- [ ] Create new cluster:
  - Name: `endzone-dev`
  - Runtime: DBR 13.3 LTS or later (includes Python 3.10+)
  - Node type: Single node (only option in CE)
  - Auto-termination: 120 minutes
- [ ] Start cluster and wait for "Running" status
- [ ] Note cluster ID from URL or details page

**Expected**: Cluster starts successfully. CE limit: auto-terminates after inactivity.

**Failure mode**: If cluster creation fails or times out repeatedly, proceed to Plan B.

#### 3. SQL Warehouse (Critical for dbt)
- [ ] Click "SQL Warehouses" in sidebar (or check if unavailable in CE)
- [ ] If available: Create warehouse (2X-Small, auto-stop 10 min)
- [ ] If unavailable: Note this limitation

**Expected**: SQL Warehouse available OR acceptable workaround exists (e.g., cluster with JDBC/ODBC endpoint).

**Failure mode**: If SQL Warehouse unavailable AND cluster JDBC endpoint doesn't support dbt, this blocks dbt-databricks. Proceed to Plan B.

**Note**: As of 2026, CE SQL Warehouse availability varies. Document actual finding.

#### 4. dbt Connection Test

Local machine setup:
```bash
# Install dbt-databricks
pip install dbt-databricks

# Copy profile template
cd dbt
cp profiles.yml.example profiles.yml

# Edit profiles.yml with your Databricks credentials
# For CE, typically:
#   host: community.cloud.databricks.com
#   http_path: /sql/1.0/warehouses/<warehouse-id> (or cluster path)
#   token: <personal-access-token>
```

Databricks token creation:
- [ ] In Databricks: User Settings → Access Tokens → Generate New Token
- [ ] Copy token to `profiles.yml` (NEVER commit this file)
- [ ] Set expiration: 90 days

dbt validation:
- [ ] Run `dbt debug --profiles-dir .` (should show "All checks passed!")
- [ ] Run `dbt compile --profiles-dir .` (compiles SQL, no data needed)
- [ ] Run `dbt deps --profiles-dir .` (installs dbt_utils package)

**Expected**: `dbt compile` succeeds, confirming dbt can talk to Databricks.

**Failure mode**: Connection errors, auth failures, or "feature not available in CE" → Plan B.

#### 5. MLflow Tracking Test

Databricks includes MLflow. Test via notebook or local Python:

Create notebook in Databricks:
```python
import mlflow

# Set experiment (creates if doesn't exist)
mlflow.set_experiment("/Users/<your-email>/nfl_spike_test")

# Log a test run
with mlflow.start_run():
    mlflow.log_param("test_param", 42)
    mlflow.log_metric("test_metric", 0.95)
    mlflow.log_artifact("test.txt", "Test artifact content")
    print("MLflow run logged successfully")
```

Verify:
- [ ] Run notebook or script
- [ ] Check "Experiments" in sidebar
- [ ] Confirm run appears with params/metrics

**Expected**: MLflow tracking works in CE workspace.

**Failure mode**: If MLflow unavailable (unlikely), document limitation. Can use local MLflow as fallback.

#### 6. GitHub Actions Integration

- [ ] In Databricks: Generate access token for CI
- [ ] In GitHub: Repository → Settings → Secrets → Actions
- [ ] Add secrets:
  - `DATABRICKS_HOST`: Your workspace URL
  - `DATABRICKS_TOKEN`: Access token (read/execute permissions)
  - (Optional) `DATABRICKS_HTTP_PATH`: SQL Warehouse or cluster path

**Expected**: Secrets stored securely; CI can authenticate.

**Failure mode**: Token permission issues → troubleshoot or use local runs only.

### Spike Outcome

Record result in ADR-001:

**PASS**: All 6 items validated → Continue with Databricks
- [ ] Update ADR-001 status to "PASS"
- [ ] Update this runbook with actual URLs/IDs
- [ ] Continue Day 1 work

**FAIL**: Any blocking issue → Activate Plan B
- [ ] Create `docs/adr/ADR-001b-snowflake.md` documenting failure reason
- [ ] Switch to Snowflake trial (see below)
- [ ] Update `dbt/profiles.yml.example` to use snowflake target
- [ ] Same Day 1 DoD, different adapter

### Plan B: Snowflake Activation

If Databricks spike fails:

1. Sign up at [signup.snowflake.com](https://signup.snowflake.com/) (30-day trial, no card upfront)
2. Select cloud provider and region
3. Note account identifier (e.g., `xy12345.us-east-1`)
4. Install `dbt-snowflake`: `pip install dbt-snowflake`
5. Configure `profiles.yml`:
   ```yaml
   endzone_mlops:
     target: snowflake
     outputs:
       snowflake:
         type: snowflake
         account: <account-id>
         user: <username>
         password: <password>
         database: NFL_DEV
         warehouse: COMPUTE_WH
         schema: PUBLIC
   ```
6. Run `dbt compile` to verify
7. For MLflow: Use local tracking (`./mlruns`) or deploy separate tracking server

---

## Local Development Setup

For Day 1 work without Databricks/Snowflake:

### DuckDB Path (Day 1 Default)

```bash
# Install dependencies
pip install -r requirements.txt

# dbt with DuckDB (in-process database)
cd dbt
cp profiles.yml.example profiles.yml
# Edit to use 'local' target (DuckDB)

# Compile models (no data load, validates SQL)
dbt compile --profiles-dir .

# When data is loaded: run models
dbt run --profiles-dir . --select stg_nflverse__games

# Run tests
dbt test --profiles-dir .
```

### Python Package Development

```bash
# Run tests
pytest tests/ -v

# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

### Data Loading (Future Step)

For actual data ingestion (post-Day 1):

```python
# Example: Download nflverse games
import pandas as pd

url = "https://github.com/nflverse/nflverse-data/releases/latest/download/games.parquet"
df = pd.read_parquet(url)

# Load to Databricks Delta table or DuckDB
# (Ingestion pipeline to be implemented)
```

---

## GitHub Actions CI/CD

Workflow: `.github/workflows/ci.yml`

### Required Secrets (Databricks Path)

If using Databricks in CI:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

### Workflow Steps

1. **Lint**: `ruff check` (fails on errors)
2. **Test**: `pytest` (fails if any test fails)
3. **dbt compile**: Validates SQL syntax
4. **(Future) dbt build**: Runs models and tests with data

### Local CI Simulation

```bash
make lint    # Run linting
make test    # Run tests
make dbt-compile  # Validate dbt models
```

All must pass before merging PR.

---

## MLflow Tracking

### Databricks Path

```python
import mlflow
import os

# Authenticate (uses DATABRICKS_TOKEN env var)
mlflow.set_tracking_uri("databricks")

# Set experiment
mlflow.set_experiment("/Users/<your-email>/nfl_pregame")

# Training script usage
with mlflow.start_run():
    mlflow.log_param("model_type", "logistic")
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", 0.72)
    mlflow.log_artifact("plots/confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")
```

### Local Path

```python
import mlflow

# Local tracking
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("nfl_pregame")

# Same API as Databricks
with mlflow.start_run():
    # ... log params, metrics, models
```

### View Experiments

- **Databricks**: Experiments sidebar in workspace
- **Local**: `mlflow ui` (opens http://localhost:5000)

---

## Troubleshooting

### Databricks

**Problem**: Cluster auto-terminates
- **Solution**: Expected CE behavior. Restart cluster before demos.

**Problem**: "SQL Warehouse not available"
- **Solution**: Use compute cluster with JDBC endpoint OR activate Plan B.

**Problem**: dbt connection timeout
- **Check**: Token valid? Cluster running? `http_path` correct?

**Problem**: MLflow experiment not found
- **Solution**: Experiment path must start with `/Users/<email>/` in CE.

### dbt

**Problem**: `dbt compile` fails with "relation not found"
- **Solution**: Day 1 models are schema-only. Data load comes later. Use `--select` to compile specific models.

**Problem**: dbt_utils package missing
- **Solution**: Run `dbt deps --profiles-dir .`

**Problem**: profiles.yml errors
- **Check**: Valid YAML? Correct indentation? No tabs?

### Python

**Problem**: Import errors
- **Solution**: Ensure PYTHONPATH includes project root: `export PYTHONPATH=.`

**Problem**: Tests fail in CI but pass locally
- **Check**: Dependencies in `requirements.txt`? Environment differences?

### GitHub Actions

**Problem**: CI workflow not triggering
- **Check**: Workflow file in `.github/workflows/`? Valid YAML?

**Problem**: "Secret not found"
- **Solution**: Add required secrets in repository settings.

---

## Next Steps After Spike

Once spike is PASS:

1. Load actual nflverse data (1-3 seasons for Day 1)
2. Run `dbt run` to materialize staging models
3. Create marts models (Day 7)
4. Implement `train.py` with MLflow (Day 7)
5. Implement `score.py` batch scoring (Day 14)

---

**Last updated**: 2026-10-06  
**Status**: Spike PENDING_OWNER
