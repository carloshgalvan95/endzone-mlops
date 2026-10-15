#!/usr/bin/env pwsh
# Development automation script for Windows PowerShell
# Mirrors Makefile targets for cross-platform development

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

function Show-Help {
    Write-Host "Available targets:"
    Write-Host "  setup               - Create virtualenv and install dependencies"
    Write-Host "  install             - Install Python dependencies (in active environment)"
    Write-Host "  lint                - Run code quality checks (ruff)"
    Write-Host "  format              - Auto-format code with ruff"
    Write-Host "  test                - Run pytest unit tests"
    Write-Host "  dbt-compile         - Compile dbt models"
    Write-Host "  dbt-run             - Run dbt models (creates tables)"
    Write-Host "  dbt-test            - Run dbt tests"
    Write-Host "  ingest              - Load all seasons (games, pbp, depth charts from 2001)"
    Write-Host "  ingest-current      - Refresh current season only (for weekly update)"
    Write-Host "  dbt-build-databricks - dbt deps, then build models on the Databricks target"
    Write-Host "  train               - Train NFL pregame prediction models with MLflow"
    Write-Host "  train-sweep         - Run training sweep (5+ configs for MLflow comparison)"
    Write-Host "  train-cv            - Phase 2 H5 starting QB CV against logistic h2_full"
    Write-Host "  mlflow-ui           - Open MLflow UI against the local SQLite store"
    Write-Host "  audit-features      - Profile current pregame inputs (no feature changes)"
    Write-Host "  check-epa-filter    - Count play types in the EPA filter, plus kneel and spike rows"
    Write-Host "  clean               - Remove build artifacts"
    Write-Host "  reproduce           - Full reproduction pipeline"
    Write-Host ""
    Write-Host "Note: Most targets expect an activated virtualenv"
    Write-Host "      ingest* targets require DATABRICKS_* env vars (load from .env)"
    Write-Host "Usage: .\scripts\dev.ps1 <target>"
}

function Invoke-Setup {
    Write-Host "Creating virtual environment in .venv/"
    python -m venv .venv
    Write-Host "Installing dependencies..."
    .venv\Scripts\pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt
    Write-Host ""
    Write-Host "Setup complete! Activate the environment with:"
    Write-Host "  .venv\Scripts\Activate.ps1"
}

function Invoke-Install {
    pip install -r requirements.txt
}

function Invoke-Lint {
    ruff check src\ tests\
    ruff format --check src\ tests\
}

function Invoke-Format {
    ruff format src\ tests\
}

function Invoke-Test {
    pytest tests\ -v
}

function Load-Env {
    if (Test-Path .env) {
        Get-Content .env | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$' -and -not $_.StartsWith('#')) {
                [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
            }
        }
    }
}

function Invoke-DbtCompile {
    Load-Env
    Push-Location dbt
    try {
        dbt compile --profiles-dir .
    }
    finally {
        Pop-Location
    }
}

function Invoke-DbtRun {
    Load-Env
    Push-Location dbt
    try {
        dbt run --profiles-dir .
    }
    finally {
        Pop-Location
    }
}

function Invoke-DbtTest {
    Load-Env
    Push-Location dbt
    try {
        dbt test --profiles-dir .
    }
    finally {
        Pop-Location
    }
}

function Invoke-Clean {
    Write-Host "Removing build artifacts..."
    Get-ChildItem -Path . -Include __pycache__,.pytest_cache,.ruff_cache -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path dbt\target -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path dbt\logs -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path dbt\dbt_packages -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Clean complete"
}

function Invoke-Train {
    Load-Env
    Write-Host "Training NFL pregame prediction models with MLflow..."
    python -m src.train.train --model logistic --lookback 10 --seed 42
}

function Invoke-TrainSweep {
    Write-Host "Running training sweep..."
    & ".\scripts\train_sweep.ps1"
}

function Invoke-TrainCv {
    Load-Env
    Write-Host "Running phase 2 H5 starting QB CV..."
    python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5
}

function Invoke-MlflowUi {
    # One worker: uvicorn's default pool raises WinError 10022 on Windows.
    mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
}

function Invoke-AuditFeatures {
    Load-Env
    Write-Host "Auditing pregame inputs (features are not changed)..."
    python -m src.analysis.feature_audit --output-dir reports/phase0
}

function Invoke-CheckEpaFilter {
    Load-Env
    Write-Host "Checking play types inside the EPA play filter..."
    python -m src.analysis.epa_play_filter_check --output reports/epa_filter_check.md
}

function Invoke-Reproduce {
    Invoke-Install
    Invoke-DbtCompile
    Write-Host "Reproduction pipeline complete (dependencies installed, dbt compiled on DuckDB)."
}

function Invoke-Ingest {
    Load-Env
    Write-Host "Loading nflverse data (all seasons 1999-2026, games and pbp)..."
    python -m src.ingest.load_raw_nflverse --dataset all
}

function Invoke-IngestCurrent {
    Load-Env
    Write-Host "Refreshing current season only..."
    python -m src.ingest.load_raw_nflverse --current-season --dataset all
}

function Invoke-DbtBuildDatabricks {
    Write-Host "Installing dbt packages, then building on the Databricks target..."
    Push-Location dbt
    try {
        dbt deps --profiles-dir .
        if ($LASTEXITCODE -ne 0) { throw "dbt deps failed" }
        dbt build --profiles-dir . --target databricks
        if ($LASTEXITCODE -ne 0) { throw "dbt build failed" }
    }
    finally {
        Pop-Location
    }
}

# Main execution
switch ($Target.ToLower()) {
    "help" { Show-Help }
    "setup" { Invoke-Setup }
    "install" { Invoke-Install }
    "lint" { Invoke-Lint }
    "format" { Invoke-Format }
    "test" { Invoke-Test }
    "dbt-compile" { Invoke-DbtCompile }
    "dbt-run" { Invoke-DbtRun }
    "dbt-test" { Invoke-DbtTest }
    "ingest" { Invoke-Ingest }
    "ingest-current" { Invoke-IngestCurrent }
    "dbt-build-databricks" { Invoke-DbtBuildDatabricks }
    "train" { Invoke-Train }
    "train-sweep" { Invoke-TrainSweep }
    "train-cv" { Invoke-TrainCv }
    "mlflow-ui" { Invoke-MlflowUi }
    "audit-features" { Invoke-AuditFeatures }
    "check-epa-filter" { Invoke-CheckEpaFilter }
    "clean" { Invoke-Clean }
    "reproduce" { Invoke-Reproduce }
    default {
        Write-Host "Unknown target: $Target"
        Write-Host ""
        Show-Help
        exit 1
    }
}
