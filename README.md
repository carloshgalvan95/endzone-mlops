# endzone-mlops

[![CI](https://github.com/carloshgalvan95/endzone-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/carloshgalvan95/endzone-mlops/actions/workflows/ci.yml)

NFL pregame prediction lakehouse demonstrating Databricks + dbt + MLflow + CI/CD patterns for MLOps portfolios.

## Problem Statement

Build a reproducible lakehouse pipeline for NFL pregame predictions (win probability, margin, total points) that demonstrates modern MLOps practices. Target audience: hiring managers evaluating Data Engineering, ML Engineering, or MLOps capabilities.

**This is a batch-first analytics platform**, not a sports betting application. The domain is NFL because the data story is rich and can become a real product later, but the primary goal is demonstrating: Databricks lakehouse architecture, dbt transformations, MLflow experiment tracking, GitHub Actions CI/CD, and modular Python design patterns.

## Current Status

**As of October 2026 (Day 7 of 14)**:

- **Databricks Free Edition**: Spike completed and passed. Serverless SQL warehouse validated with dbt-databricks locally and from GitHub Actions. MLflow tracking connectivity confirmed.
- **Bronze ingest**: Backfill complete (1999-2026, games and play-by-play). Unity Catalog tables: workspace.nfl_raw.games, workspace.nfl_raw.pbp (partitioned by season).
- **dbt**: 2 staging views (stg_nflverse__games, stg_nflverse__pbp), 3 marts tables (fct_games, fct_team_game_epa, fct_upcoming_games), 2 sources, 2 seeds. dbt build on Databricks: 64/64 nodes passing (57 data tests).
- **Training pipeline**: `src/train/train.py` with MLflow tracking. Supports baseline (home-always), logistic regression, and gradient boosting models. Features include rolling EPA metrics, win rates, rest days, games played counts (10-game lookback). Time-based split (train through 2023, validate 2024, test 2025). Logs params, metrics (log loss, Brier, accuracy, AUC), feature list, data window, git SHA. Can run against Databricks or local fixtures. Includes SimpleImputer for missing history (EPA imputed to 0, win_rate to 0.5). Tests include leakage protection and January playoff season handling. Local tracking uses `sqlite:///mlflow.db` in the repo root, with run artifacts under `mlartifacts/`.
- **CI**: GitHub Actions runs ruff lint + format check, pytest on Python 3.10/3.11, dbt compile against DuckDB, cross-platform matrix (Ubuntu, Windows, macOS). Tests verify column references against real nflverse schema (fixtures in tests/fixtures), schema conforming logic, feature leakage protection, and end-to-end training.
- **PR Guard**: Branch protection with freshness check, Conventional Commit titles, PR template verification, no secrets/data files, commit identity validation.

**Next**: First training runs on real Databricks data (real metrics pending), batch scoring (score.py).

## Architecture Overview

**Target architecture** (components marked [done] or [planned]):

```
nflverse (historical)     BALLDONTLIE (future live, documented only)
    |                            |
    v                            v
HistoricalFeed [done]       LiveFeed (stub) [planned]
    |                            |
    +----------------------------+
                |
                v
         Bronze (raw data) [done]
                |
                v
    dbt: Staging [done] -> Marts [done]
                |
                v
       Feature Engineering (Python) [planned]
                |
                v
    MLflow Tracking (train.py with hyperparameters) [planned]
                |
                v
       Model Registry + Artifacts [planned]
                |
                v
         Batch Scoring (score.py) [planned]
                |
                v
      Predictions Table + Monitoring [planned]
```

### Key Design Decisions

- **Batch-first**: v0.1 focuses on offline training and batch scoring. Real-time inference is documented as a future adapter pattern, not implemented.
- **Historical source**: nflverse (CC BY 4.0, play-by-play from 1999+, Parquet format)
- **Live path (documented, not active)**: HistoricalFeed vs LiveFeed interface. LiveFeed target is BALLDONTLIE GOAT API (~$40/month) for post-v0.1 with owner approval. No API keys in repo, no live calls in v0.1.
- **Primary stack**: Databricks Free Edition + dbt-databricks + MLflow + GitHub Actions
- **Plan B**: Snowflake trial (same repo layout) if Databricks Free Edition spike fails

## Product Framing: NFLow (Optional Nickname)

The optional product name "NFLow" may appear in documentation. Repository name remains `endzone-mlops`.

### Two ML Products (Separate Experiments)

1. **Pregame**: Predict win probability, margin, total before kickoff
2. **In-game** (future): Update probability after each play

Experiments are tracked separately in MLflow. Never mix game states between train and test splits.

## Data Sources and Attribution

### Historical Data: nflverse

- **License**: CC BY 4.0
- **Coverage**: NFL play-by-play data from 1999-present
- **Format**: Parquet files (~370 columns, ~19.5 MB per season)
- **Updates**: Clean nightly; raw available ~15 minutes after game ends
- **Citation**: nflverse project (https://github.com/nflverse)

### Future Live Data: BALLDONTLIE (Documented Only)

- **Status**: Stub interface in v0.1; no active API calls
- **Target plan**: BALLDONTLIE NFL GOAT tier (~$39.99/month) for v0.2+
- **Capabilities**: Play-by-play, stats, injuries, odds (near-real-time, no published SLA)
- **Decision gate**: Activation requires explicit owner approval and is post-Day 14

**No vendor API keys, no live API calls, no paid subscriptions in v0.1.**

## Repository Structure

```
endzone-mlops/
├── README.md                   # This file
├── LICENSE                     # MIT
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata
├── Makefile                    # Common tasks (lint, test, reproduce)
├── .env.example                # Environment template (no secrets)
├── .github/
│   └── workflows/
│       └── ci.yml             # Lint, pytest, dbt compile
├── infra/
│   └── README.md              # Databricks setup instructions
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml.example
│   ├── models/
│   │   ├── staging/           # stg_* models from raw nflverse
│   │   └── marts/             # Business logic transformations
│   ├── tests/
│   └── macros/
├── src/
│   ├── data/                  # Data ingestion, HistoricalFeed/LiveFeed
│   ├── features/              # Feature engineering
│   ├── models/                # train.py, score.py
│   └── utils/                 # Shared utilities
├── tests/                     # Pytest unit tests
├── notebooks/                 # Exploration only (not the product)
└── docs/
    ├── architecture.md        # C4 diagrams, detailed architecture
    ├── adr/                   # Architecture Decision Records
    │   └── ADR-001-databricks.md
    ├── runbook.md            # Operational procedures, CE spike status
    └── ATTRIBUTION.md        # Full data source credits
```

## How to Run

### Prerequisites

- Python 3.10+
- (Optional) Databricks Free Edition account
- (Optional) dbt installed locally or via Databricks SQL Warehouse

### Local Setup (Day 1 Path)

Cloud dev environment config (`.cursor/environment.json`) creates `.venv` and installs `requirements.txt`, with ruff pinned to 0.17.0 so it matches CI.

#### macOS / Linux

```bash
# Clone repository
git clone https://github.com/carloshgalvan95/endzone-mlops.git
cd endzone-mlops

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (or use: make setup)
pip install -r requirements.txt

# Copy config templates
cp .env.example .env
cp dbt/profiles.yml.example dbt/profiles.yml
# Edit .env and dbt/profiles.yml with your settings

# Run tests
make test

# Lint and format
make lint

# dbt: install packages and compile models (loads .env automatically)
make dbt-compile
```

**Tip**: Use `make setup` to automate virtualenv creation and dependency installation.

#### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/carloshgalvan95/endzone-mlops.git
cd endzone-mlops

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# If activation is blocked, enable scripts (one-time):
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Install dependencies (or use: .\scripts\dev.ps1 setup)
pip install -r requirements.txt

# Copy config templates
Copy-Item .env.example .env
Copy-Item dbt\profiles.yml.example dbt\profiles.yml
# Edit .env and dbt\profiles.yml with your settings

# Run tests
.\scripts\dev.ps1 test

# Lint and format
.\scripts\dev.ps1 lint

# dbt: install packages and compile models (loads .env automatically)
.\scripts\dev.ps1 dbt-compile
```

**Tip**: Use `.\scripts\dev.ps1 setup` to automate virtualenv creation and dependency installation.

### Training Models

After setting up the environment and building dbt models, you can train prediction models. Local runs record experiments in `sqlite:///mlflow.db` (repo root) and write model artifacts under `mlartifacts/`. Both paths are gitignored. If an existing `.env` still sets `MLFLOW_TRACKING_URI` to a filesystem path, remove that line so training uses the SQLite default.

#### macOS / Linux

```bash
# Train with default settings (logistic regression, lookback=10)
make train

# Or specify model and hyperparameters
python -m src.train.train --model logistic --lookback 10 --C 1.0 --seed 42
python -m src.train.train --model gbm --n-estimators 100 --max-depth 5 --learning-rate 0.1
python -m src.train.train --model baseline-home

# Run training sweep (5+ configs for MLflow comparison)
make train-sweep

# View MLflow UI to compare runs
make mlflow-ui
# or: mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in browser
```

#### Windows (PowerShell)

```powershell
# Train with default settings
.\scripts\dev.ps1 train

# Or specify model and hyperparameters
python -m src.train.train --model logistic --lookback 10 --C 1.0 --seed 42
python -m src.train.train --model gbm --n-estimators 100 --max-depth 5 --learning-rate 0.1
python -m src.train.train --model baseline-home

# Run training sweep (5+ configs for MLflow comparison)
.\scripts\dev.ps1 train-sweep

# View MLflow UI to compare runs
.\scripts\dev.ps1 mlflow-ui
# or: mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in browser
```

#### Windows (Git Bash)

Git Bash rewrites arguments that look like Unix paths. Set `MSYS_NO_PATHCONV=1` before training or opening the UI so `sqlite:///` is left intact.

```bash
export MSYS_NO_PATHCONV=1

# Train with default settings (logistic regression, lookback=10)
make train

# Or specify model and hyperparameters
python -m src.train.train --model logistic --lookback 10 --C 1.0 --seed 42
python -m src.train.train --model gbm --n-estimators 100 --max-depth 5 --learning-rate 0.1
python -m src.train.train --model baseline-home

# Run training sweep (5+ configs for MLflow comparison)
make train-sweep

# View MLflow UI to compare runs
make mlflow-ui
# or: mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in browser
```

**Note**: Training requires either Databricks credentials (in .env) or local fixture files. Metrics (accuracy, Brier score, log loss, ROC AUC) are logged to MLflow along with model artifacts. Real metrics pending first training run on Databricks data.

**Important .env notes**:
- `.env` is for environment variables (e.g., Databricks credentials), not the Python virtual environment
- Format: `KEY=value` with no spaces around `=`
- DATABRICKS_HOST should NOT include `https://` prefix
- Git Bash (MINGW) on Windows mangles paths starting with `/` - use PowerShell or add `export MSYS_NO_PATHCONV=1` to `~/.bashrc`
- Both `.venv/` and `dbt/profiles.yml` are gitignored

### Databricks Free Edition Path

**Spike completed**: Databricks Free Edition serverless SQL warehouse works with dbt-databricks. See `docs/runbook.md` for setup details and troubleshooting.

**To run ingest and dbt against Databricks**:

macOS / Linux (bash):
```bash
# Load environment variables from .env
set -a; source .env; set +a

# Ingest nflverse data to Unity Catalog
make ingest       # Load games + play-by-play

# Build dbt models on Databricks
make dbt-build-databricks
```

Windows Git Bash:
```bash
export MSYS_NO_PATHCONV=1  # Prevent path conversion issues
set -a; source .env; set +a
make ingest
make dbt-build-databricks
```

Windows PowerShell:
```powershell
# Load environment automatically via scripts
.\scripts\dev.ps1 ingest
.\scripts\dev.ps1 dbt-build-databricks
```

See `docs/runbook.md` for credential setup, Unity Catalog volume paths, and verification queries.

## CI/CD

### GitHub Actions Workflows

**CI** (`.github/workflows/ci.yml`): Runs on every push and pull request:
- **Lint**: ruff check + ruff format --check
- **Test**: pytest on Python 3.10 and 3.11 (Ubuntu)
- **dbt compile**: Validates SQL against DuckDB (no Databricks credentials in CI yet)
- **Cross-platform**: Lint + pytest on windows-latest and macos-latest (Python 3.11)
- Tests include verification that staging models reference real nflverse column names (fixtures in tests/fixtures)

**Databricks Smoke Test** (`.github/workflows/databricks-smoke.yml`): Manual workflow_dispatch, runs dbt debug and dbt compile against Databricks using GitHub secrets (DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN).

**PR Guard** (`.github/workflows/pr-guard.yml`): Automated PR validation:
- Branch must be up to date with main
- Conventional Commit PR titles (feat, fix, docs, chore, ci, test, refactor, perf, build)
- PR template with verification checklist
- No secrets or data files committed
- Commit identity checks (author/committer email validation)

**Branch protection**: main requires pr-guard + CI jobs to pass, branch must be up to date, no force pushes.

**Planned**: A Databricks integration job in CI that runs ingest for one season and dbt build in a separate CI schema on PRs that touch ingest or dbt.

## Roadmap

### v0.1 (Day 14 Target)
- [x] Repository scaffold
- [x] Bronze ingest to Unity Catalog Delta tables (games, play_by_play, 1999-2026 backfill complete)
- [x] Staging dbt models from nflverse (2 views)
- [x] Marts dbt models (fct_games, fct_team_game_epa, fct_upcoming_games)
- [x] dbt build on Databricks: 64/64 nodes passing (57 data tests)
- [x] CI with GitHub Actions (lint, tests, cross-platform, PR guard)
- [x] MLflow experiment tracking (train.py with baseline, logistic, gradient boosting)
- [x] Pregame feature engineering (rolling EPA, win rates, rest days, games played)
- [x] Training sweep (5+ configs, make train-sweep)
- [ ] First training runs on real Databricks data (real metrics pending)
- [ ] Databricks integration job in CI
- [ ] Batch scoring pipeline (score.py)
- [ ] Documentation and demo

### v0.2+ (Post-Day 14)
- [ ] Activate LiveFeed adapter with BALLDONTLIE (requires paid subscription and approval)
- [ ] In-game prediction experiment
- [ ] Monitoring and drift detection
- [ ] Databricks Jobs or GitHub Actions schedule

## Related Documentation

- [Architecture](docs/architecture.md): Detailed C4 diagrams and component descriptions
- [ADR-001](docs/adr/ADR-001-databricks.md): Why Databricks + dbt + MLflow
- [Runbook](docs/runbook.md): Operational procedures and spike status
- [Attribution](docs/ATTRIBUTION.md): Full data source credits and licenses

## License

MIT License - see [LICENSE](LICENSE) file.

## Project Context

This portfolio project was built as part of an MLOps learning path, synthesizing concepts from:
- SWEBOK software engineering practices
- Wilson's modularity patterns for ML
- Databricks lakehouse architecture
- dbt analytics engineering
- MLflow experiment tracking

Production analogues: AWS Glue + Athena + Redshift Spectrum, Prefect/Airflow orchestration, and similar cloud data platform patterns.

---

**Important**: This is a learning project with production-shaped patterns, not production software. Batch-first design. Live streaming adapter documented but not implemented in v0.1.
