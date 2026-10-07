# endzone-mlops

NFL pregame prediction lakehouse demonstrating Databricks + dbt + MLflow + CI/CD patterns for MLOps portfolios.

## Honesty

Public learning portfolio. Production analogues for CV/LinkedIn: Athena / Glue / Redshift Spectrum / Prefect (role-dependent).
Do not invent Databricks, dbt, MLflow, or Snowflake logos on PEMEX or Kavak bullets.

## Problem Statement

Build a reproducible lakehouse pipeline for NFL pregame predictions (win probability, margin, total points) that demonstrates modern MLOps practices. Target audience: hiring managers evaluating Data Engineering, ML Engineering, or MLOps capabilities.

**This is a batch-first analytics platform**, not a sports betting application. The domain is NFL because the data story is rich and can become a real product later, but the primary goal is demonstrating: Databricks lakehouse architecture, dbt transformations, MLflow experiment tracking, GitHub Actions CI/CD, and modular Python design patterns.

## Architecture Overview

```
nflverse (historical)     BALLDONTLIE (future live, documented only)
    |                            |
    v                            v
HistoricalFeed              LiveFeed (stub)
    |                            |
    +----------------------------+
                |
                v
         Bronze (raw data)
                |
                v
    dbt: Staging -> Marts (SQL transformations)
                |
                v
       Feature Engineering (Python)
                |
                v
    MLflow Tracking (train.py with hyperparameters)
                |
                v
       Model Registry + Artifacts
                |
                v
         Batch Scoring (score.py)
                |
                v
      Predictions Table + Monitoring
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

**Important .env notes**:
- `.env` is for environment variables (e.g., Databricks credentials), not the Python virtual environment
- Format: `KEY=value` with no spaces around `=`
- DATABRICKS_HOST should NOT include `https://` prefix
- Git Bash (MINGW) on Windows mangles paths starting with `/` - use PowerShell or add `export MSYS_NO_PATHCONV=1` to `~/.bashrc`
- Both `.venv/` and `dbt/profiles.yml` are gitignored

### Databricks Free Edition Path (Spike Pending)

See `docs/runbook.md` for Databricks Free Edition setup checklist:
- Serverless SQL Warehouse connection (pre-created)
- Personal access token generation
- dbt-databricks adapter configuration with Unity Catalog
- MLflow tracking with serverless compute
- GitHub Actions secrets for CI

**Spike status**: PENDING_OWNER (requires completion of Databricks Free Edition signup and configuration)

## CI/CD

GitHub Actions workflow runs on every push and pull request:
- **Lint**: ruff (or flake8)
- **Test**: pytest
- **dbt**: compile (build with secrets when available)

Badge: ![CI](https://github.com/carloshgalvan95/endzone-mlops/workflows/CI/badge.svg)

## Roadmap

### v0.1 (Day 14 Target)
- [x] Repository scaffold
- [x] Staging dbt models from nflverse
- [ ] Marts for pregame features
- [ ] MLflow experiment tracking (train.py)
- [ ] Batch scoring pipeline (score.py)
- [ ] CI/CD with GitHub Actions
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
