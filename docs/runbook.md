# Operational Runbook

This document contains operational procedures, setup checklists, and troubleshooting guides for the endzone-mlops project.

## Table of Contents

1. [PR Guard Workflow](#pr-guard-workflow)
2. [Databricks Free Edition Spike](#databricks-free-edition-spike)
3. [Local Development Setup](#local-development-setup)
4. [GitHub Actions CI/CD](#github-actions-cicd)
5. [MLflow Tracking](#mlflow-tracking)
6. [Metric locks](#metric-locks)
7. [Troubleshooting](#troubleshooting)
8. [Weekly page](#weekly-page)

---

## PR Guard Workflow

**Purpose**: Automated guardrails that validate every pull request against repository quality standards.

**Trigger**: Runs automatically on PR open, synchronize, reopen, and edit events.

**Job name**: `pr-guard` (stable name for branch protection rules)

### What It Checks

The PR Guard validates:

1. **Base branch freshness**: PR must contain the current base branch HEAD (not stale/behind)
2. **Pull request opener**: The account that opens the pull request is `carloshgalvan95`. The coding agent pushes the branch and does not open the pull request.
3. **Git authorship**: Author and committer email must be `carloshgalvan95@gmail.com`; GitHub login must be `carloshgalvan95`
4. **Co-authored-by trailers**: Only self Co-authored-by (owner's email) is allowed; blocks bots and other contributors
5. **Commit messages**: No AI agent mentions (cursor, claude, copilot, grok, etc.) except in code context like `db.cursor()`
6. **PR title**: Must follow Conventional Commit format (feat, fix, docs, chore, ci, test, refactor, perf, build)
7. **PR body**: No tool-generated footers or em dash character (U+2014); no AI agent mentions
8. **PR template**: Required sections (Summary, Changes, Verification, Risks) present; at least one verification item checked
9. **Changed files**: No em dash added; no data/secret files (*.parquet, *.csv, .env, *.pem) except in tests/fixtures
10. **File size**: No very large files (>5 MB heuristic)
11. **Head branch**: The branch must start with a conventional prefix (`feat/`, `fix/`, `docs/`, `chore/`, `ci/`, `test/`, `refactor/`, `perf/`, `build/`). A tool-generated prefix is rejected.

### How to Fix Failures

Close the pull request before pushing. A branch with an open pull request does not receive new commits. After the fix is on the branch, open a new pull request from the `carloshgalvan95` account.

The full local suite passes before that push: `ruff check endzone/ tests/`, `ruff format --check endzone/ tests/`, `make lint-structure` (Windows: `.\scripts\dev.ps1 lint-structure`), and `pytest tests/ -v`. Tests that read or write text pass `encoding="utf-8"`.

#### Base branch stale
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

The rebase rewrites the feature branch. `--force-with-lease` updates that branch after the pull request is closed. It is not a push to `main`.

#### Wrong author/committer
```bash
# Set git identity
git config user.name "Carlos Galván"
git config user.email "carloshgalvan95@gmail.com"

# Fix last commit
git commit --amend --reset-author --no-edit

# Fix multiple commits (interactive rebase)
git rebase -i origin/main
# Mark commits as 'edit', then for each:
git commit --amend --reset-author --no-edit
git rebase --continue
```

#### Co-authored-by trailers (non-owner)
Remove any Co-authored-by lines except `Co-authored-by: Carlos Galván <carloshgalvan95@gmail.com>` from commit messages:
```bash
git rebase -i origin/main
# Mark commits as 'reword' or 'edit', remove unwanted trailers
```

#### Non-conventional PR title
Edit the PR title to match format: `type(scope): description`

Valid types: feat, fix, docs, chore, refactor, test, ci, build, perf

Example: `feat(dbt): add fct_game_results mart`

#### Missing PR template sections or unchecked verification
Edit the PR body to include all required sections and check at least one verification item.

#### Em dash in PR body or files
Replace em dash (U+2014) with comma, period, colon, or regular hyphen.

#### Data/secret files
Remove data files from the commit. Use tests/fixtures for small test data only.

### Making PR Guard a Required Check

In GitHub repository settings:
1. Go to Settings > Branches
2. Add branch protection rule for `main`
3. Enable "Require status checks to pass before merging"
4. Search for and select: `pr-guard`
5. Save changes

---

## Databricks Free Edition Spike

**Status**: COMPLETED (2026-10-06)

**Outcome**: PASS - All validation items completed successfully.

**Note**: Databricks Free Edition replaced Community Edition (retired in 2025). Free Edition is serverless-only with Unity Catalog enabled by default.

### Spike Checklist

#### 1. Account Creation
- [x] Navigate to [Databricks Free Edition signup](https://www.databricks.com/try-databricks/signup-form)
- [x] Sign up with email (no credit card required)
- [x] Verify email and complete onboarding
- [x] Note workspace URL (assigned automatically)

#### 2. SQL Warehouse Setup (Critical for dbt)

Free Edition includes a pre-created serverless SQL warehouse. No cluster creation is needed or available.

- [x] Click "SQL Warehouses" in left sidebar
- [x] Locate "Serverless Starter Warehouse" (automatically provided)
- [x] Click on the warehouse to open details
- [x] Click "Start" if warehouse is stopped
- [x] Click "Connection details" tab
- [x] Copy the following values:
  - **Server hostname** (e.g., `adb-xxxxx.xx.azuredatabricks.net`)
  - **HTTP path** (e.g., `/sql/1.0/warehouses/xxxxx`)

**Expected**: Serverless Starter Warehouse starts successfully. This is the only SQL warehouse in Free Edition (2X-Small, serverless).

**Failure mode**: If the warehouse does not start, stop and retry it. The 2026-10-06 spike passed on this warehouse. Snowflake is not used.

**Note**: Free Edition does NOT allow creating compute clusters. All compute is serverless.

#### 3. Personal Access Token Creation

- [x] Click on your profile icon (initials) in top-right corner
- [x] Select "Settings"
- [x] Click "Developer" in left sidebar
- [x] Click "Access tokens" tab
- [x] Click "Manage" (if available) or "Generate new token"
- [x] Set:
  - Comment: "dbt development"
  - Lifetime: 90 days
- [x] Click "Generate"
- [x] Copy the token immediately (shown only once)
- [x] Store token securely (NEVER commit to git)

**Expected**: Token generation succeeds.

**Failure mode**: If token generation is blocked, check account verification status.

#### 4. dbt Connection Test

Local machine setup:
```bash
# dbt-databricks is in uv.lock. From the repo root, with uv on PATH:
uv sync --frozen

# Copy profile template
cd dbt
cp profiles.yml.example profiles.yml
# profiles.yml is gitignored. After a pull, compare it to profiles.yml.example.
# The warehouse build target must be databricks. That output's catalog must be workspace.
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
- [x] Run `dbt debug --profiles-dir .` (should show "All checks passed!")
- [x] Run `dbt deps --profiles-dir .` (installs dbt_utils package)
- [x] Run `dbt compile --profiles-dir . --target databricks` (compiles SQL, no data needed)

**Expected**: `dbt compile` succeeds, confirming dbt can talk to Databricks Free Edition.

**Failure mode**: Connection errors, auth failures, or Unity Catalog permission issues. Fix the host, HTTP path, token, and warehouse state. Snowflake is not used.

#### 5. Serverless Compute for Notebooks

Free Edition uses serverless compute for notebooks (no cluster creation needed).

- [x] Click "Workspace" in left sidebar
- [x] Create a new notebook: Click "Create" → "Notebook"
- [x] Name: "nfl_spike_test"
- [x] Language: Python
- [x] Click "Create"
- [x] In top-right corner, click "Connect" dropdown
- [x] Select "Serverless" (should be only option)
- [x] Wait for serverless compute to start (first time may take 1-2 minutes)

**Expected**: Serverless compute connects successfully.

**Failure mode**: If serverless compute fails to start, wait and retry once. Snowflake is not used.

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
- [x] Run the notebook cell
- [x] Click "Experiments" in left sidebar (or top-right experiment icon)
- [x] Confirm run appears with params/metrics
- [x] Note experiment path format: `/Users/<your-email>/...`

**Expected**: MLflow tracking works with serverless compute.

**Failure mode**: If MLflow unavailable in Free Edition, document limitation. Can use local MLflow as fallback.

#### 7. GitHub Actions Integration

- [x] In Databricks: Use access token from step 3 (or generate new one for CI)
- [x] In GitHub: Repository → Settings → Secrets → Actions
- [x] Add secrets:
  - `DATABRICKS_HOST`: Server hostname from step 2
  - `DATABRICKS_HTTP_PATH`: HTTP path from step 2
  - `DATABRICKS_TOKEN`: Access token from step 3

**Verify connection from GitHub Actions**:
- [x] Go to repository Actions tab
- [x] Select "Databricks Smoke Test" workflow (left sidebar)
- [x] Click "Run workflow" dropdown → "Run workflow" button
- [x] Wait for workflow to complete (~1-2 minutes)
- [x] Expect a green run. The job runs `dbt debug` and `dbt compile` on the Databricks target.

If the run fails, read the job logs and check the three secrets.

**Expected**: The secrets stay in GitHub Actions. The smoke test succeeds.

**Failure mode**: Token permission issues, incorrect secrets, or network errors → check workflow logs and troubleshoot.

### Spike outcome

**PASS** on 2026-10-06. ADR-001 records the same result. All 7 groups above were completed.

The workspace host, HTTP path, and token stay in `.env` and in GitHub Actions secrets. This file does not store them.

Snowflake was not activated. `dbt-snowflake` is not a dependency. There is no ADR-001b. `dbt/profiles.yml.example` still has an unused `snowflake` output. Do not select it. The warehouse target is `databricks`.

---

## Local Development Setup

For work that does not need the warehouse:

### Python 3.12 lock

`pyproject.toml` is the only list of direct dependencies. `uv.lock` pins the full graph, including hashes, for Python 3.12 (`requires-python` is `>=3.12,<3.13`, and `.python-version` is `3.12`).

uv is the lock tool. It reads `pyproject.toml` directly, installs the matching CPython, and creates `.venv` with `uv sync --frozen`. pip-tools would also emit a hashed lock, but it needs a second input file and does not install the interpreter. A hand-maintained `requirements.txt` had already drifted from `pyproject.toml` (scikit-learn, the SQL connector, and requests were missing from the package metadata, dbt floors disagreed, and ruff was `==0.17.0` in one file and `>=0.1.0` in the other). That file is gone so the two lists cannot diverge again. CI, `make setup`, and `.\scripts\dev.ps1 setup` all run `uv sync --frozen`.

Recreate `.venv` on a Mac:

```bash
brew install uv
rm -rf .venv
uv python install 3.12
uv sync --frozen
source .venv/bin/activate
```

The installer `curl -LsSf https://astral.sh/uv/install.sh | sh` works when Homebrew is not available. Then `source "$HOME/.local/bin/env"`.

Recreate `.venv` on Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
uv python install 3.12
uv sync --frozen
.venv\Scripts\Activate.ps1
```

Dev tools (pytest, pytest-cov, ruff `0.17.0`) are the `dev` group. `uv sync` installs that group by default (`tool.uv.default-groups`).

### DuckDB Path (Day 1 Default)

```bash
# Install the locked environment (includes dbt-duckdb)
uv sync --frozen
source .venv/bin/activate

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

Use Python 3.12. `.python-version` selects it, and CI tests only 3.12. Python 3.14 triggers pydantic v1 warnings inside dbt.

```bash
# Run tests
pytest tests/ -v

# Lint code
ruff check endzone/ tests/

# Format code
ruff format endzone/ tests/

# Layout and import contracts
make lint-structure
```

### Data loading

Ingest is `python -m endzone.ingest.load_raw_nflverse`. The commands are in [D2: Data Ingestion to Databricks Free Edition](#d2-data-ingestion-to-databricks-free-edition). A second loader is not in the repo.

---

## D2: Data Ingestion to Databricks Free Edition

This section covers loading nflverse games and play-by-play data into Databricks Free Edition.

**Prerequisites**:
- Databricks Free Edition account configured (see spike checklist above)
- Environment variables set in `.env` (DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN)
- Python dependencies installed: `uv sync --frozen`

### Data Ingestion Workflow

The ingestion script `endzone.ingest.load_raw_nflverse` supports two datasets and multi-season backfill:

**Games** (`--dataset games`):
1. Downloads nflverse games/schedules parquet from GitHub releases
2. Filters to specified seasons (default: 1999-2026)
3. Uploads to Unity Catalog Volume at `/Volumes/workspace/nfl_raw/landing/games/`
4. Creates bronze Delta table `workspace.nfl_raw.games` using `read_files()`

**Play-by-Play** (`--dataset pbp`):
1. Downloads nflverse play-by-play parquet per season from GitHub releases (one file per season)
2. Conforms each season to unified schema (handles column drift across 27 seasons with pyarrow)
3. Uploads to partitioned Unity Catalog Volume at `/Volumes/workspace/nfl_raw/landing/pbp/{season}/`
4. Creates bronze Delta table `workspace.nfl_raw.play_by_play` from entire pbp folder using `read_files()`

**Season Range Parsing**:
- Range: `--seasons 1999-2026` (expands to all years inclusive)
- List: `--seasons 2022,2023,2024` (comma-separated)
- Mixed: `--seasons 2020-2022,2024`
- Current season only: `--current-season` (for weekly refresh, ignores `--seasons`)

**Missing Seasons**: If a season's pbp file does not exist (e.g., future season 2027), it is logged as a warning and skipped, not an error.

**Note**: Databricks Free Edition does NOT have outbound internet from compute, so the script downloads data locally first, then uploads via the Files API. Python loads `.env` from the working directory. A variable already set in the shell wins, including after `set -a; source .env; set +a`. `make dbt-compile`, `make dbt-run`, and `make dbt-test` still source `.env` because dbt does not use that loader.

### Commands

#### macOS / Linux (bash)

```bash
# Load environment variables from .env
export MSYS_NO_PATHCONV=1  # Needed for Git Bash on Windows
set -a; source .env; set +a

# Full backfill (games and play-by-play 1999-2026, depth charts from 2001)
make ingest
# Or: python -m endzone.ingest.load_raw_nflverse --dataset all

# Weekly refresh (current season only)
make ingest-current
# Or: python -m endzone.ingest.load_raw_nflverse --current-season --dataset all

# Custom season range
python -m endzone.ingest.load_raw_nflverse --seasons 2020-2024 --dataset all

# Build dbt models on Databricks. This runs dbt deps, then dbt build.
# make train-cv is H5b and needs every depth-chart season reloaded.
# Chain with && so a failed
# ingest or build does not score.
python -m endzone.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && make train-cv
# Fresh clone, games and play-by-play included:
# make ingest && make dbt-build-databricks && make train-cv
# Or: cd dbt && dbt deps --profiles-dir . && dbt build --profiles-dir . --target databricks
```

#### Windows Git Bash

```bash
# Prevent path conversion issues with Git Bash
export MSYS_NO_PATHCONV=1

# Load environment variables from .env
set -a; source .env; set +a

# Full backfill (games and play-by-play 1999-2026, depth charts from 2001)
make ingest

# Weekly refresh (current season only)
make ingest-current

# Custom season range
python -m endzone.ingest.load_raw_nflverse --seasons 2020-2024 --dataset all

# Build dbt models on Databricks. This runs dbt deps, then dbt build.
# make train-cv is H5b and needs every depth-chart season reloaded.
# Chain with && so a failed
# ingest or build does not score.
python -m endzone.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && make train-cv
```

#### Windows PowerShell

```powershell
# PowerShell automatically loads .env when the script checks for it
# Alternatively, manually set environment variables:
# $env:DATABRICKS_HOST = "adb-xxxxx.xx.azuredatabricks.net"
# $env:DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/xxxxx"
# $env:DATABRICKS_TOKEN = "dapi..."

# Full backfill (games and play-by-play 1999-2026, depth charts from 2001)
.\scripts\dev.ps1 ingest

# Weekly refresh (current season only)
.\scripts\dev.ps1 ingest-current

# Custom season range (PowerShell does not support Make, use Python directly)
python -m endzone.ingest.load_raw_nflverse --seasons 2020-2024 --dataset all
.\scripts\dev.ps1 ingest

# Build dbt models on Databricks. This runs dbt deps, then dbt build.
# make train-cv is H5b and needs every depth-chart season reloaded.
# PowerShell 5.1 has no &&.
python -m endzone.ingest.load_raw_nflverse --dataset depth_charts
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 dbt-build-databricks }
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 train-cv }
```

### Unity Catalog schema map

Catalog `workspace`. The dbt Databricks target schema is `nfl_dev` (`dbt/profiles.yml.example`). This repo has no `generate_schema_name` macro, so dbt's default macro appends each custom schema from `dbt_project.yml`:

| Custom schema in dbt | Warehouse schema | Objects |
| --- | --- | --- |
| (none, seeds) | `nfl_dev` | `team_franchise_map`, `depth_chart_club_map`, `known_pbp_gaps` |
| `staging` | `nfl_dev_staging` | `stg_nflverse__games`, `stg_nflverse__pbp`, `stg_nflverse__depth_charts` |
| `marts` | `nfl_dev_marts` | `fct_games`, `fct_team_game_epa`, `fct_upcoming_games`, `fct_expected_starter`, `fct_qb_dropbacks` |
| `test_results` | `nfl_dev_test_results` | empty unless one test is run with `--store-failures` |

Bronze is not a dbt schema. Ingest writes `workspace.nfl_raw.games`, `workspace.nfl_raw.play_by_play`, and `workspace.nfl_raw.depth_charts`. Depth charts start in 2001. 1999 and 2000 were not published.

### Test result schema

`dbt build` and `dbt test` run the tests and do not store a failure relation for each one. Unity Catalog Free Edition allows 100 tables per schema and counts views toward that same quota. The project used to set `store_failures` on every test, so `<target_schema>_test_results` filled with one relation per test, including empty relations from tests that passed. The warning fires at 80 relations. `tests/test_dbt_store_failures.py` fails if a project or directory config turns `store_failures` back on, and the cap on named exceptions is 0.

The default profile uses catalog `workspace` and target schema `nfl_dev`, so the audit schema is `workspace.nfl_dev_test_results`. If `dbt/profiles.yml` uses another catalog or target schema, replace those two names: the schema is `<catalog>.<target_schema>_test_results`. Run each statement on its own. The warehouse rejects a script that starts with `USE`.

Count the relations that are still there:

```sql
SELECT table_type, COUNT(*) AS relation_count
FROM workspace.information_schema.tables
WHERE table_schema = 'nfl_dev_test_results'
GROUP BY table_type
ORDER BY table_type;
```

List them before dropping:

```sql
SELECT table_name, table_type
FROM workspace.information_schema.tables
WHERE table_schema = 'nfl_dev_test_results'
ORDER BY table_name;
```

Drop the schema and every table and view in it. `dbt build` does not recreate them. This removes whatever is in that schema, so read the list first.

```sql
DROP SCHEMA IF EXISTS workspace.nfl_dev_test_results CASCADE;
```

To inspect one failing test, store only that invocation, then drop the relation it created:

```bash
dbt test --profiles-dir . --target databricks --select assert_team_game_two_rows_per_game --store-failures
```

The list query above shows the new name. Drop that one relation when the debugging is done (`DROP TABLE` or `DROP VIEW` in `workspace.nfl_dev_test_results`). Leave `store_failures` off in `dbt/dbt_project.yml`.

### Depth charts from 2025

`depth_charts_2001.parquet` through `depth_charts_2024.parquet` are weekly. They already have `season`, `club_code`, `week`, `game_type`, `depth_team`, `formation`, `position`, and `gsis_id`.

`depth_charts_2025.parquet` is a different file. The real columns are `dt`, `team`, `player_name`, `espn_id`, `gsis_id`, `pos_grp_id`, `pos_grp`, `pos_id`, `pos_name`, `pos_abb`, `pos_slot`, and `pos_rank`. There is no `week`. `dt` is a UTC scrape time (`2025-09-04T07:21:47Z`). Each scrape covers all 32 teams. The same shape is expected for later seasons.

Ingest maps that file onto the weekly columns before upload:

- `season` is the file year, not the calendar year of `dt` (the 2025 file runs into the 2026 offseason).
- `club_code` is `team`.
- `position` is `pos_abb`. `QB` stays `QB`.
- `depth_team` is `pos_rank` as text. `1` is the listed starter.
- `formation` is `3WR 1TE` to Offense, `Base 4-3 D` and `Base 3-4 D` to Defense, and `Special Teams` unchanged. Any other `pos_grp` is dropped with a warning.
- `full_name` is `player_name`. Given name, jersey number, and `elias_id` are not on the file, so those columns stay null.
- `week` and `game_type` come from the games schedule. For each team-game, keep the latest snapshot with `dt` strictly before kickoff. Kickoff is `gameday` plus `gametime` interpreted as Eastern time (`America/New_York`), then compared with `dt` in UTC. A Thursday night game cannot see Friday's scrape. A Sunday game can. Scrapes after the last kickoff are not attached to that season.

The bronze table is still one schema. The starter join is still season, week, game_type, and the schedule team. Week and depth rank are cast through double, a legacy club code is translated with `depth_chart_club_map`, and a duplicate QB1 prefers `depth_position` QB. CV folds are 2016-2023, so they do not read the mapped 2025 rows. 2025 is the held-out test. If the mapping cannot be done, that season is skipped with a warning and the starter falls back to the previous-game primary passer.

A catalog that already has the 2001-2024 depth-chart files and only needs the 2025 snapshot mapping does not need those seasons again for that mapping. This uploads 2025 only, then rebuilds `workspace.nfl_raw.depth_charts` from every season file already in the volume folder. H5b is different: reload every depth-chart season so `week` is one int type. A 2025-only load leaves the int32 files for 2002-2006 and 2014 in the folder, and the bronze replace reads them.

```bash
python -m endzone.ingest.load_raw_nflverse --seasons 2025 --dataset depth_charts
```

Then rebuild the marts before training. Chain with `&&` so training does not start after a failed step:

```bash
python -m endzone.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && make train-cv
```

PowerShell 5.1 has no `&&`:

```powershell
python -m endzone.ingest.load_raw_nflverse --dataset depth_charts
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 dbt-build-databricks }
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 train-cv }
```

Do not select staging or marts from `workspace.nfl_dev`. That name is only the dbt target and the seed schema.

Python overrides, all optional: `DATABRICKS_CATALOG`, `DATABRICKS_STAGING_SCHEMA`, `DATABRICKS_RAW_SCHEMA`, `DATABRICKS_MARTS_SCHEMA`.

### Verification

After running the ingestion and dbt build, verify the data in Databricks SQL Editor:

```sql
-- Check bronze table row counts by season
SELECT season, COUNT(*) as game_count
FROM workspace.nfl_raw.games
GROUP BY season
ORDER BY season;

-- Expected output (approximate):
-- 2022: 285 games (272 regular + playoffs + preseason)
-- 2023: 285 games
-- 2024: 285 games

-- Check play-by-play bronze table
SELECT season, COUNT(*) as play_count
FROM workspace.nfl_raw.play_by_play
GROUP BY season
ORDER BY season;

-- Expected: ~45,000-50,000 plays per season

-- Check staging views
SELECT 'stg_nflverse__games' as model, COUNT(*) as row_count
FROM workspace.nfl_dev_staging.stg_nflverse__games
UNION ALL
SELECT 'stg_nflverse__pbp', COUNT(*)
FROM workspace.nfl_dev_staging.stg_nflverse__pbp;

-- Check pbp-to-games foreign key integrity
SELECT COUNT(DISTINCT pbp.game_id) as pbp_games,
       COUNT(DISTINCT g.game_id) as games_games
FROM workspace.nfl_dev_staging.stg_nflverse__pbp pbp
LEFT JOIN workspace.nfl_dev_staging.stg_nflverse__games g ON pbp.game_id = g.game_id;
-- Both counts should match

-- Sample play-by-play data
SELECT game_id, play_id, quarter, down, yards_to_go, play_type, 
       yards_gained, epa, wpa, success
FROM workspace.nfl_dev_staging.stg_nflverse__pbp
WHERE season = 2024
LIMIT 10;
```

### Troubleshooting Data Ingestion

**Problem**: `Missing required Databricks environment variables`
- **Solution**: Ensure `.env` file exists and contains DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
- **Check**: `cat .env` (bash) or `Get-Content .env` (PowerShell)

**Problem**: `Unauthorized` or `401` error
- **Solution**: Regenerate access token in Databricks (Settings > Developer > Access tokens)
- **Check**: Token has not expired (default 90 days)

**Problem**: `Volume does not exist` or `Schema does not exist`
- **Solution**: Script should create these automatically. If fails, manually create:
  ```sql
  CREATE SCHEMA IF NOT EXISTS workspace.nfl_raw;
  CREATE VOLUME IF NOT EXISTS workspace.nfl_raw.landing;
  ```

**Problem**: `dbt build --target databricks` fails with "relation not found"
- **Solution**: Ensure ingestion completed successfully first. Check bronze tables exist:
  ```sql
  SHOW TABLES IN workspace.nfl_raw;
  ```

**Problem**: Download fails with network error
- **Solution**: Check internet connection. nflverse data is downloaded from GitHub releases (public, no auth needed)

**Problem**: Unity Catalog warns that `nfl_dev_test_results` is near the 100-relation quota
- **Solution**: Run the count, list, and `DROP SCHEMA ... CASCADE` statements in [Test result schema](#test-result-schema). `dbt build` does not recreate those relations.

**Problem**: `dbt test` fails on pbp uniqueness test
- **Solution**: This indicates duplicate plays in source data. Check:
  ```sql
  SELECT game_id, play_id, COUNT(*) as cnt
  FROM workspace.nfl_raw.play_by_play
  GROUP BY game_id, play_id
  HAVING COUNT(*) > 1;
  ```

### Data Attribution

All data sourced from nflverse project:
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Source**: https://github.com/nflverse/nflverse-data
- **Coverage**: NFL games and play-by-play from 1999-present
- **Citation**: nflverse project contributors

When using this data in presentations or publications, credit the nflverse project as specified in `docs/ATTRIBUTION.md`.

---

## GitHub Actions CI/CD

Workflow: `.github/workflows/ci.yml`

CI runs on a push to `main` and on a pull request into `main`.

### Secrets

`ci.yml` does not read Databricks secrets. It compiles with DuckDB.

The manual workflow `.github/workflows/databricks-smoke.yml` uses `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN`.

### Workflow Steps

Ubuntu, in `.github/workflows/ci.yml`:

1. **Lint**: `ruff check` and `ruff format --check`
2. **Structure**: `scripts/check_structure.py` and `lint-imports`
3. **Unused code**: `vulture endzone scripts --min-confidence 80`
4. **Test**: `pytest`
5. **Analysis coverage**: 100% on `endzone/analysis`
6. **dbt compile**: `dbt deps`, `dbt seed`, and `dbt compile` on local DuckDB

Windows and macOS run the lint, format, structure, and pytest steps. They do not compile dbt.

`dbt build` on Databricks is `make dbt-build-databricks`. It is not a CI step. A CI job that builds warehouse tables is not in the repo.

### Local CI Simulation

```bash
make lint    # Run linting
make lint-structure  # CCDS layout and import contracts
make test    # Run tests
make dbt-compile  # Validate dbt models
```

The suite before any push is `ruff check endzone/ tests/`, `ruff format --check endzone/ tests/`, `make lint-structure` (Windows: `.\scripts\dev.ps1 lint-structure`), and `pytest tests/ -v`. Tests that read or write text pass `encoding="utf-8"`. The same commands pass again before merging. The coding agent pushes the branch and does not open the pull request. Carlos opens it from his account. A branch with an open pull request does not receive new commits.

---

## MLflow Tracking

### Databricks Path

```python
import mlflow

# Authenticate (uses DATABRICKS_TOKEN env var)
mlflow.set_tracking_uri("databricks")

# The Day 1 notebook spike passed an absolute path.
# Training code calls activate_tracking_experiment instead.
# A relative name such as nfl_pregame becomes
# /Shared/endzone-mlops/nfl_pregame unless
# ENDZONE_MLFLOW_EXPERIMENT_ROOT is set.
mlflow.set_experiment("/Users/<your-email>/nfl_pregame")
```

`endzone/modeling/tracking.py` logs params, `cv_*` metrics, `comparison.md`, `comparison.csv`, out-of-fold files, and the sklearn model inside `mlflow.start_run()`. It does not log a confusion-matrix plot.

### Local Path

```python
import mlflow

# Local tracking database in the repo root
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nfl_pregame")

# Same API as Databricks
with mlflow.start_run():
    # ... log params, metrics, models
```

### View Experiments

- **Databricks**: Experiments sidebar in workspace
- **Local**: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (opens http://localhost:5000)

---

## Metric locks

Two checks guard the rolling-origin report in `endzone/modeling/`. The offline check runs in CI. The warehouse check runs on a machine that can read the marts.

### Offline golden

`tests/test_golden_cv.py` builds the games and team-game tables in the test with numpy Generator seed 42. It calls `run_phase1_baselines` for phase 2 hypothesis `h3`: feature sets from `logistic_feature_sets_for` and `cv_model_specs_for`, the home prior, logistic regression, Elo, and the market. The window is rolling-origin 2016-2023. The paired bootstrap uses 1000 resamples and seed 42. Pooled metrics must match `tests/golden/cv_report.json` within 1e-9. Fold sizes and accept or reject labels must match exactly. Seasons 2024 and 2025 are in the slate. Scored folds stay inside 2016-2023.

Rewrite the snapshot only when the change is intentional:

```bash
UPDATE_GOLDEN=1 pytest tests/test_golden_cv.py
```

### Warehouse reference

`docs/experiments/reference_metrics.json` stores the H5 comparison for logistic `h2_full` (n=2191, log loss 0.6377, Brier 0.2235, AUC 0.6843, accuracy 0.6426), Elo 0.6529, the market 0.6120 (n=2190), and the home prior 0.6888. `make verify-reference` loads the marts and scores those rows through `run_phase1_baselines`. It exits non-zero when logistic `h2_full` log loss, n, Elo, or the market differ by more than 1e-4.

On a Mac, from the repo root:

```bash
source .venv/bin/activate
set -a; source .env; set +a
make verify-reference
```

Windows: `.\scripts\dev.ps1 verify-reference`.

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
- **Solution**: Free Edition uses Unity Catalog with the `workspace` catalog. The dbt profile `schema` is the target `nfl_dev`. Staging, marts, and test results are the suffixed schemas in the map above, not `default`.

### dbt

**Problem**: `dbt compile` fails with "relation not found"
- **Solution**: The local compile uses DuckDB after `dbt seed`. `make dbt-compile` does that. A missing seed or a profile whose target is not `local` is the usual cause. Warehouse rows are not required for the compile. The Databricks build is `make dbt-build-databricks` after ingest.

**Problem**: dbt_utils package missing
- **Solution**: Run `dbt deps --profiles-dir .`

**Problem**: profiles.yml errors
- **Check**: Valid YAML? Correct indentation? No tabs? `dbt/profiles.yml` is gitignored, so a pull does not update it. Compare it to `dbt/profiles.yml.example`. The warehouse build target must be `databricks`, and that output's catalog must be `workspace`.

**Problem**: `Season 2025 depth chart missing columns: season, club_code, week, ...`
- **Solution**: That message is the pre-2025 weekly check against the timestamped file. Pull the mapping fix, then load 2025 only. The 2001-2024 files already on the volume stay. `python -m endzone.ingest.load_raw_nflverse --seasons 2025 --dataset depth_charts`, then `make dbt-build-databricks`. If the log says the season was skipped, `fct_expected_starter` uses the previous-game primary passer for 2025. CV (2016-2023) does not use that season.

**Problem**: `dbt build` fails on a fresh clone with a missing package
- **Solution**: Run `dbt deps --profiles-dir .` first. `make dbt-build-databricks` and `.\scripts\dev.ps1 dbt-build-databricks` do that, then build. `make train-cv` is H5b. On a catalog that already has games and play-by-play, reload every depth-chart season: `python -m endzone.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && make train-cv`. A 2025-only load does not fix the int32 week files. A fresh clone uses `make ingest && make dbt-build-databricks && make train-cv`.

**Problem**: pydantic v1 warnings while running dbt
- **Check**: The local venv is Python 3.12 (`uv sync --frozen`). Python 3.14 triggers those warnings.

### Python

**Problem**: Import errors
- **Solution**: Ensure PYTHONPATH includes project root: `export PYTHONPATH=.`

**Problem**: Tests fail in CI but pass locally
- **Check**: Same lock (`uv sync --frozen`) and Python 3.12 locally and in CI?

### GitHub Actions

**Problem**: CI workflow not triggering
- **Check**: Workflow file in `.github/workflows/`? Valid YAML?

**Problem**: "Secret not found"
- **Solution**: Add required secrets in repository settings.

---

## After the spike

The spike passed on 2026-10-06. Ingest, the staging views, the marts, `endzone/modeling/train.py`, and the weekly page are in the repo. The public week file is frozen once it is published. See [Weekly page](#weekly-page). `make train-s2` writes the Sprint 2 CV report under `reports/sprint2/`. The frozen copy is `docs/experiments/reports/sprint2_cv_report.json`.

Still not in the repo: README screenshots, a short demo video, and a `v0.1.0` tag.

---

## Weekly page

The page is an open experiment. It is not betting picks. Generated week JSON stays off `main`. The durable copy is the orphan branch `published-slates`. The live site is what readers open. Losing either copy, or deploying a slate that was not archived, is a failure.

`predictions/<season>-w<week>.json` on that branch is the current file. `history/<season>-w<week>/` keeps every earlier copy, so a later overwrite does not throw the previous file away. Do not merge the branch into `main`, and do not force-push it.

**Push to `main`.** `.github/workflows/pages.yml` restores `published-slates` first. A week the branch does not have yet is copied from the live site. It then overlays nflverse finals, archives those exact files, and deploys `site/`. It does not fit a model and it does not rewrite probabilities. If the archive push fails, the site is not deployed. Use a push for UI and overlay changes.

**Tuesday 12:00 UTC.** The same workflow refreshes the current season, builds the Databricks marts, restores the archive first, and scores the next unpublished week. The score loads alias `prod` from Unity Catalog (`MLFLOW_REGISTRY_URI=databricks-uc`). Tracking stays `sqlite:///mlflow.db`. If the alias is missing or the registry cannot be read, the log says so and the job fits logistic `h2_full` on games before that week. That fit is registered in the sqlite file and does not get the alias. If that week file is already present after the restore, it is kept. A score or an archive write that would drop a `game_id` fails the job, and the previous deployment stays up. Adding games is allowed.

**Rebuild one week.** This is the only way to replace probabilities for a week that is already published. It predicts every scheduled game that week, played and unplayed, and trains only on completed games before that week. Same-week finals are not labels and are not features. The results overlay then sets `status` to `final` only when nflverse has both scores. A 0-0 final is a tie. A missing score stays `scheduled`. The archive is updated before the deploy, and it still refuses a dropped `game_id`.

Local:

```bash
make rebuild-week SEASON=2026 WEEK=9
```

Windows:

```powershell
$env:SEASON = "2026"
$env:WEEK = "9"
.\scripts\dev.ps1 rebuild-week
```

GitHub Actions: run Pages with `workflow_dispatch`, set `replace_published` to true, and set both season and week. Leaving the flag off scores the next unpublished week, or keeps it when the file is already there. The archive is updated before the deploy.

**Results.** `make score` is the unpublished-week command. `make refresh-results` overlays finals and does not refit. `.github/workflows/refresh-results.yml` restores the archive first, overlays, writes the track record, runs the model alert, archives, and deploys on game-day windows. It has no warehouse credentials. A pass that does not change the JSON does not add a commit. A new `results_updated_at` does.

The alert compares published log loss with the no-vig market on the last 64 decided games that have a price, and only after 48 of them. The reference gap is logistic `h2_full` minus the market row in `docs/experiments/reference_metrics.json` (0.6377 minus 0.6120 today). It trips when the window gap exceeds that reference gap by the margin, default 0.02, so the bar is about 0.046. A window that matches the reference table stays quiet. Trailing the market by the reference gap is the known state. The issue is not a betting signal. Override the margin with `ENDZONE_ALERT_GAP_MARGIN`. The other overrides are `ENDZONE_ALERT_WINDOW_GAMES` and `ENDZONE_ALERT_MIN_GAMES`. A trip opens or updates one issue labeled `model-behind-market`. A 403 or any other GitHub error is logged. The step does not fail the deploy. See ADR-009.

**Backfill, once.** Run this after the archive change is on `main`, and before the next Tuesday score, so 2026 week 9 is on the branch. That slate has 15 games and was regenerated at 2026-11-09T04:28:37Z. In Actions, open Pages, Run workflow, and set `backfill_archive` to true. Leave season and week empty. It copies the live site onto `published-slates`. It does not score and it does not deploy.

The same command locally, with the public Pages URL and a remote that accepts the push:

```bash
make backfill-archive PAGES_BASE=https://<owner>.github.io/<repo>
```

Windows:

```powershell
$env:PAGES_BASE = "https://<owner>.github.io/<repo>"
.\scripts\dev.ps1 backfill-archive
```

Use the dispatch. It commits as `github-actions[bot]`, and only that job has `contents: write`.

Confirm `predictions/2026-w9.json` is on `published-slates` and the file has 15 games. `history/2026-w9/` has that snapshot. If the push returns 403, the repository Actions setting is still read-only for the workflow token. Open Settings, Actions, General, Workflow permissions, and choose Read and write permissions. The other jobs do not request contents write.

Optional: protect `published-slates` against force-push and deletion. Do not set that branch as the Pages source.

Sprint 2 margin and total passed ADR-007, so the page shows those estimates. A target that fails its pre-registered test stays hidden.

**Track record.** `make track-record` (Windows: `.\scripts\dev.ps1 track-record`) writes `site/data/track_record.json`. It reads `published-slates` unless `ARCHIVE_DIR` points at a checkout of that branch. Probabilities come from the earliest archived copy of each game. Scores come from the newest copy, including the week files already in `site/data`. The command does not edit those files. Pages and Results run it after the overlay and before upload. They also add a content hash to the CSS and JavaScript URLs in the built `index.html`, because Pages caches files for 10 minutes.

If the branch cannot be read, the file says the record is unavailable and the page shows that sentence. It does not fill the table from the site files alone. If an archived week file cannot be parsed, the command fails, deletes a stale `track_record.json`, and the deploy does not go out. Definitions, the clip, ties, and the small-sample labels are in ADR-008.

The table shows one row per archived week and a season to date row. A week with no finals stays in the table with blank rates. A week that still has scheduled games is marked in progress and counts only finals. One finished week is labeled a small sample.

**First registration, once.** Run this after ADR-009 is on `main`, and before you expect Tuesday to load alias `prod`. Until the promote step finishes, Tuesday logs that the alias is missing and fits the week. These commands use the existing Databricks secrets. They create the model object `workspace.nfl_dev.endzone_winprob` in schema `nfl_dev`. They do not create a schema, and they do not alter or delete a table or a model version. Do not drop the model to retry. Register again only creates the next version.

Marts have to exist (`make dbt-build-databricks` if `make verify-reference` cannot read them). A local `.env` that is not committed:

```bash
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
```

Keep the existing `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, and `DATABRICKS_TOKEN`. Optional: `ENDZONE_MLFLOW_EXPERIMENT_ROOT=/Shared/endzone-mlops`. That is the default. Databricks rejects the relative name `nfl_pregame/registry`. The register command resolves it to `/Shared/endzone-mlops/nfl_pregame/registry` and creates the parent folder `/Shared/endzone-mlops/nfl_pregame` when it is missing. Creating the folder is idempotent: an existing folder is left as it is. The experiment path itself is not created as a folder. A name that already starts with `/` is not prefixed. sqlite, and the Pages workflow, keep `nfl_pregame/registry` and the other relative names. If `/Shared` is not writable, set the root to `/Users/<your-email>/endzone-mlops` before `make register-model`.

Then:

```bash
make verify-reference
make register-model
make promote-model VERSION=1
```

`make register-model` fits logistic `h2_full` through season 2023 and stores the rolling-origin CV tags from the reference check. It refuses, and writes nothing, when that check fails. It does not set the alias. `make promote-model VERSION=1` sets alias `prod` only when those tags match the golden row. A refusal leaves the alias unset.

Windows, from the repo root, with the same `.env`:

```powershell
.\scripts\dev.ps1 verify-reference
.\scripts\dev.ps1 register-model
$env:VERSION = "1"
.\scripts\dev.ps1 promote-model
```

The same two steps are the Model registry workflow (`workflow_dispatch` only). Run it once with action `register` and `through_season` `2023`. When that succeeds, run it again with action `promote` and `version` `1`. That workflow is not the Pages job. A failure there does not deploy.

`make reproduce` still does not register a model.

---

**Last updated**: 2026-11-09  
**Status**: Spike COMPLETED (PASS)
