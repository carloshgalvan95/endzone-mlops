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
    Write-Host "  ingest              - Load both games and pbp datasets (2022-2024)"
    Write-Host "  ingest-games        - Load games dataset only (2022-2024)"
    Write-Host "  ingest-pbp          - Load play-by-play dataset only (2022-2024)"
    Write-Host "  dbt-build-databricks - Build dbt models on Databricks target"
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

function Invoke-Reproduce {
    Invoke-Install
    Invoke-DbtRun
    Write-Host "Reproduction pipeline complete. Next: run train.py when implemented."
}

function Invoke-Ingest {
    Load-Env
    Invoke-IngestGames
    Invoke-IngestPbp
}

function Invoke-IngestGames {
    Load-Env
    Write-Host "Loading nflverse games data (seasons 2022 2023 2024)..."
    python -m src.ingest.load_raw_nflverse --seasons 2022 2023 2024 --dataset games
}

function Invoke-IngestPbp {
    Load-Env
    Write-Host "Loading nflverse play-by-play data (seasons 2022 2023 2024)..."
    python -m src.ingest.load_raw_nflverse --seasons 2022 2023 2024 --dataset pbp
}

function Invoke-DbtBuildDatabricks {
    Write-Host "Building dbt models on Databricks target..."
    Push-Location dbt
    try {
        dbt build --profiles-dir . --target databricks
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
    "ingest-games" { Invoke-IngestGames }
    "ingest-pbp" { Invoke-IngestPbp }
    "dbt-build-databricks" { Invoke-DbtBuildDatabricks }
    "clean" { Invoke-Clean }
    "reproduce" { Invoke-Reproduce }
    default {
        Write-Host "Unknown target: $Target"
        Write-Host ""
        Show-Help
        exit 1
    }
}
