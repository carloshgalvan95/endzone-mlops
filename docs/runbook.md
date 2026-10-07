# Operational Runbook

This document contains operational procedures, setup checklists, and troubleshooting guides for the endzone-mlops project.

## Table of Contents

1. [Databricks Free Edition Spike](#databricks-free-edition-spike)
2. [Local Development Setup](#local-development-setup)
3. [GitHub Actions CI/CD](#github-actions-cicd)
4. [MLflow Tracking](#mlflow-tracking)
5. [Troubleshooting](#troubleshooting)

---

## Databricks Free Edition Spike

**Status**: PENDING_OWNER (requires completion)

**Time estimate**: 45-60 minutes

**Decision criteria**: If ANY item fails, activate Plan B (Snowflake) same day.

**Note**: Databricks Free Edition replaced Community Edition (retired in 2025). Free Edition is serverless-only with Unity Catalog enabled by default.

### Spike Checklist

#### 1. Account Creation
- [ ] Navigate to [Databricks Free Edition signup](https://www.databricks.com/try-databricks/signup-form)
- [ ] Sign up with email (no credit card required)
- [ ] Verify email and complete onboarding
- [ ] Note workspace URL (assigned automatically)

#### 2. SQL Warehouse Setup (Critical for dbt)

Free Edition includes a pre-created serverless SQL warehouse. No cluster creation is needed or available.

- [ ] Click "SQL Warehouses" in left sidebar
- [ ] Locate "Serverless Starter Warehouse" (automatically provided)
- [ ] Click on the warehouse to open details
- [ ] Click "Start" if warehouse is stopped
- [ ] Click "Connection details" tab
- [ ] Copy the following values:
  - **Server hostname** (e.g., `adb-xxxxx.xx.azuredatabricks.net`)
  - **HTTP path** (e.g., `/sql/1.0/warehouses/xxxxx`)

**Expected**: Serverless Starter Warehouse starts successfully. This is the only SQL warehouse in Free Edition (2X-Small, serverless).

**Failure mode**: If warehouse fails to start or connection details are unavailable, proceed to Plan B.

**Note**: Free Edition does NOT allow creating compute clusters. All compute is serverless.

#### 3. Personal Access Token Creation

- [ ] Click on your profile icon (initials) in top-right corner
- [ ] Select "Settings"
- [ ] Click "Developer" in left sidebar
- [ ] Click "Access tokens" tab
- [ ] Click "Manage" (if available) or "Generate new token"
- [ ] Set:
  - Comment: "dbt development"
  - Lifetime: 90 days
- [ ] Click "Generate"
- [ ] Copy the token immediately (shown only once)
- [ ] Store token securely (NEVER commit to git)

**Expected**: Token generation succeeds.

**Failure mode**: If token generation is blocked, check account verification status.

#### 4. dbt Connection Test

Local machine setup:
```bash
# Install dbt-databricks (uncomment in requirements.txt if needed)
pip install dbt-databricks

# Copy profile template
cd dbt
cp profiles.yml.example profiles.yml
# profiles.yml uses env_var() to read credentials from environment

# Create .env file (never commit)
cd ..
cp .env.example .env
```

Edit `.env` with values from steps 2 and 3:
```bash
DATABRICKS_HOST=adb-xxxxx.xx.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxx
DATABRICKS_TOKEN=dapi...
```

**Important .env format rules**:
- No spaces around `=` (use `KEY=value`, not `KEY = value`)
- No `https://` prefix in `DATABRICKS_HOST` (hostname only)
- No quotes around values (unless value contains spaces)

**Load environment variables before running dbt**:

macOS / Linux (bash/zsh):
```bash
set -a; source .env; set +a
cd dbt && dbt debug --profiles-dir .
```

Windows PowerShell:
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
cd dbt; dbt debug --profiles-dir .
```

Windows Git Bash (MINGW):
```bash
# Git Bash mangles paths starting with / (e.g., /sql/1.0/warehouses/xxx becomes C:/Program Files/Git/...)
# This causes HTTP 404 errors. Fix with:
export MSYS_NO_PATHCONV=1  # Add to ~/.bashrc to persist
set -a; source ../.env; set +a
dbt debug --profiles-dir .
```

**Tip**: Use `make dbt-compile` or `.\scripts\dev.ps1 dbt-compile` which load `.env` automatically.

dbt validation:
- [ ] Run `dbt debug --profiles-dir .` (should show "All checks passed!")
- [ ] Run `dbt deps --profiles-dir .` (installs dbt_utils package)
- [ ] Run `dbt compile --profiles-dir . --target databricks` (compiles SQL, no data needed)

**Expected**: `dbt compile` succeeds, confirming dbt can talk to Databricks Free Edition.

**Failure mode**: Connection errors, auth failures, or Unity Catalog permission issues → Plan B.

#### 5. Serverless Compute for Notebooks

Free Edition uses serverless compute for notebooks (no cluster creation needed).

- [ ] Click "Workspace" in left sidebar
- [ ] Create a new notebook: Click "Create" → "Notebook"
- [ ] Name: "nfl_spike_test"
- [ ] Language: Python
- [ ] Click "Create"
- [ ] In top-right corner, click "Connect" dropdown
- [ ] Select "Serverless" (should be only option)
- [ ] Wait for serverless compute to start (first time may take 1-2 minutes)

**Expected**: Serverless compute connects successfully.

**Failure mode**: If serverless compute fails to start, wait and retry once. If persistent, proceed to Plan B.

#### 6. MLflow Tracking Test

Databricks Free Edition includes MLflow with serverless compute.

Test in the notebook created above:
```python
import mlflow

# Get current user (avoids hardcoding email)
user = spark.sql("select current_user()").first()[0]

# Set experiment (creates if doesn't exist)
mlflow.set_experiment(f"/Users/{user}/nfl_spike_test")

# Log a test run
with mlflow.start_run():
    mlflow.log_param("test_param", 42)
    mlflow.log_metric("test_metric", 0.95)
    print("MLflow run logged successfully")
```

Verify:
- [ ] Run the notebook cell
- [ ] Click "Experiments" in left sidebar (or top-right experiment icon)
- [ ] Confirm run appears with params/metrics
- [ ] Note experiment path format: `/Users/<your-email>/...`

**Expected**: MLflow tracking works with serverless compute.

**Failure mode**: If MLflow unavailable in Free Edition, document limitation. Can use local MLflow as fallback.

#### 7. GitHub Actions Integration

- [ ] In Databricks: Use access token from step 3 (or generate new one for CI)
- [ ] In GitHub: Repository → Settings → Secrets → Actions
- [ ] Add secrets:
  - `DATABRICKS_HOST`: Server hostname from step 2
  - `DATABRICKS_HTTP_PATH`: HTTP path from step 2
  - `DATABRICKS_TOKEN`: Access token from step 3

**Expected**: Secrets stored securely; CI can authenticate.

**Failure mode**: Token permission issues → troubleshoot or use local runs only.

### Spike Outcome

Record result in ADR-001:

**PASS**: All 7 items validated → Continue with Databricks Free Edition
- [ ] Update ADR-001 status to "PASS"
- [ ] Update this runbook with actual workspace URL
- [ ] Add credentials to local `.env` file (never commit)
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

### Databricks Free Edition

**Problem**: SQL Warehouse stops automatically
- **Solution**: Expected Free Edition behavior. Restart Serverless Starter Warehouse when needed (auto-stops after inactivity).

**Problem**: "Cannot create cluster"
- **Solution**: Free Edition is serverless-only. Use "Serverless" compute option for notebooks. No cluster creation available.

**Problem**: dbt connection timeout
- **Check**: Token valid? SQL Warehouse running? `DATABRICKS_HOST` and `DATABRICKS_HTTP_PATH` correct? Credentials in environment variables or `.env`?

**Problem**: MLflow experiment not found
- **Solution**: Experiment path must start with `/Users/<email>/` in Free Edition. Unity Catalog enabled by default.

**Problem**: "Unity Catalog permission denied"
- **Solution**: Free Edition uses Unity Catalog with workspace catalog. Ensure dbt profile specifies `catalog` (e.g., `workspace`) and `schema` (e.g., `default`).

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
