#!/usr/bin/env pwsh
# Development automation script for Windows PowerShell
# Mirrors Makefile targets for cross-platform development

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

function Show-Help {
    Write-Host "Available targets:"
    Write-Host "  setup               - Create the Python 3.12 venv from uv.lock"
    Write-Host "  install             - Install the locked environment (uv sync --frozen)"
    Write-Host "  lint                - Run code quality checks (ruff)"
    Write-Host "  lint-structure      - Check the CCDS layout and import-linter contracts"
    Write-Host "  format              - Auto-format code with ruff"
    Write-Host "  test                - Run pytest unit tests"
    Write-Host "  vulture             - Fail on high-confidence unused code"
    Write-Host "  dbt-compile         - Compile dbt models"
    Write-Host "  dbt-run             - Run dbt models (creates tables)"
    Write-Host "  dbt-test            - Run dbt tests"
    Write-Host "  data                - Load bronze (same as ingest) and record .stamp/data"
    Write-Host "  ingest              - Load all seasons (games, pbp, depth charts from 2001)"
    Write-Host "  ingest-current      - Refresh current season only (for weekly update)"
    Write-Host "  marts               - Build Databricks marts after bronze is loaded"
    Write-Host "  dbt-build-databricks - dbt deps, then build models on the Databricks target"
    Write-Host "  train               - Train one holdout model with MLflow"
    Write-Host "  train-sweep         - Single-split sweep (5 configs). Not the reference CV"
    Write-Host "  train-cv            - H5b CV. Builds data and marts when .stamp says they are stale"
    Write-Host "  mlflow-ui           - Open MLflow UI against the local SQLite store"
    Write-Host "  audit-features      - Profile current pregame inputs (no feature changes)"
    Write-Host "  check-epa-filter    - Count play types in the EPA filter, plus kneel and spike rows"
    Write-Host "  verify-reference    - Refit logistic h2_full and fail if it drifts from the reference file"
    Write-Host "  register-model      - Register logistic h2_full. Does not set the prod alias"
    Write-Host "  promote-model       - Set alias prod on VERSION when its CV tags pass the gate"
    Write-Host "  score               - Score the next unpublished week. Keeps a week file already on disk"
    Write-Host "  train-s2            - Sprint 2 margin and total CV. Registers a target only if it passes"
    Write-Host "  rebuild-week        - Rebuild one full week (SEASON and WEEK required), then overlay finals"
    Write-Host "  refresh-results     - Overlay final nflverse scores onto published week JSON"
    Write-Host "  track-record        - Write site/data/track_record.json from the archive. Does not edit week files"
    Write-Host "  backfill-archive    - Seed published-slates from PAGES_BASE. Does not score"
    Write-Host "  clean               - Remove build artifacts"
    Write-Host "  reproduce           - install, Databricks dbt build, verify-reference"
    Write-Host ""
    Write-Host "Note: Most targets expect an activated virtualenv"
    Write-Host "      Python targets load .env. A shell variable already set wins. dbt targets still source .env."
    Write-Host "Usage: .\scripts\dev.ps1 <target>"
}

function Invoke-Setup {
    Write-Host "Creating the Python 3.12 environment in .venv/ from uv.lock"
    uv python install 3.12
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host ""
    Write-Host "Setup complete. Activate the environment with:"
    Write-Host "  .venv\Scripts\Activate.ps1"
}

function Invoke-Install {
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Lint {
    ruff check endzone\ tests\
    ruff format --check endzone\ tests\
}

function Invoke-LintStructure {
    python scripts/check_structure.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    lint-imports
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Format {
    ruff format endzone\ tests\
}

function Invoke-Test {
    pytest tests\ -v
}

function Invoke-Vulture {
    vulture endzone scripts --min-confidence 80
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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
    Write-Host "Training NFL pregame prediction models with MLflow..."
    python -m endzone.train.train --model logistic --lookback 10 --seed 42
}

function Invoke-TrainSweep {
    Write-Host "Running training sweep..."
    & ".\scripts\train_sweep.ps1"
}

function Update-Stamp {
    param([string]$Name)
    New-Item -ItemType Directory -Force -Path .stamp | Out-Null
    $path = Join-Path .stamp $Name
    if (Test-Path $path) {
        (Get-Item $path).LastWriteTime = Get-Date
    }
    else {
        New-Item -ItemType File -Path $path | Out-Null
    }
}

function Test-TrainCvNeedsMarts {
    $marts = Join-Path .stamp "marts"
    $data = Join-Path .stamp "data"
    if (-not (Test-Path $marts)) { return $true }
    if ((Test-Path $data) -and ((Get-Item $data).LastWriteTimeUtc -gt (Get-Item $marts).LastWriteTimeUtc)) {
        return $true
    }
    return $false
}

function Invoke-TrainCv {
    if (Test-TrainCvNeedsMarts) { Invoke-Marts }
    Write-Host "Running phase 2 H5b starter coverage CV..."
    python -m endzone.train.train --cv --phase 2 --hypothesis h5b --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5b
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-MlflowUi {
    # One worker: uvicorn's default pool raises WinError 10022 on Windows.
    mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
}

function Invoke-AuditFeatures {
    Write-Host "Auditing pregame inputs (features are not changed)..."
    python -m endzone.analysis.feature_audit --output-dir reports/phase0
}

function Invoke-CheckEpaFilter {
    Write-Host "Checking play types inside the EPA play filter..."
    python -m endzone.analysis.epa_play_filter_check --output reports/epa_filter_check.md
}

function Invoke-VerifyReference {
    Write-Host "Checking logistic h2_full, Elo, and the market against the reference file..."
    python -m endzone.train.train --verify-reference
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-RegisterModel {
    Write-Host "Registering logistic h2_full. The prod alias is not set..."
    $registerArgs = @("-m", "endzone.modeling.registry_command", "register")
    if ($env:THROUGH_SEASON) { $registerArgs += @("--through-season", $env:THROUGH_SEASON) }
    if ($env:COMPARISON) { $registerArgs += @("--comparison-csv", $env:COMPARISON) }
    python @registerArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-PromoteModel {
    if (-not $env:VERSION) {
        Write-Host "promote-model needs VERSION"
        exit 1
    }
    Write-Host "Setting alias prod on version $env:VERSION when the gate passes..."
    python -m endzone.modeling.registry_command promote --version $env:VERSION
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-TrainS2 {
    Write-Host "Cross-validating ridge margin and total (ADR-007)..."
    python -m endzone.modeling.points
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Score {
    Write-Host "Scoring one unpublished week with logistic h2_full..."
    $scoreArgs = @("-m", "endzone.modeling.score")
    if ($env:SEASON) { $scoreArgs += @("--season", $env:SEASON) }
    if ($env:WEEK) { $scoreArgs += @("--week", $env:WEEK) }
    if ($env:REPLACE -eq "1") { $scoreArgs += "--replace-published" }
    python @scoreArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-RebuildWeek {
    if (-not $env:SEASON -or -not $env:WEEK) {
        Write-Host "rebuild-week needs SEASON and WEEK"
        exit 1
    }
    Write-Host "Rebuilding season $env:SEASON week $env:WEEK, then overlaying finals..."
    $scoreArgs = @("-m", "endzone.modeling.score", "--season", $env:SEASON, "--week", $env:WEEK, "--replace-published")
    if ($env:RESTORE_PUBLISHED) { $scoreArgs += @("--restore-published", $env:RESTORE_PUBLISHED) }
    python @scoreArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m endzone.site.results
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-RefreshResults {
    Write-Host "Overlaying final nflverse scores onto published week JSON..."
    $refreshArgs = @("-m", "endzone.site.results")
    if ($env:RESTORE_PUBLISHED) { $refreshArgs += @("--restore-published", $env:RESTORE_PUBLISHED) }
    python @refreshArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-TrackRecord {
    Write-Host "Writing the track record from archived slates..."
    $recordArgs = @(
        "-m", "endzone.site.track_record",
        "--repo", ".",
        "--site-data", "site/data",
        "--output", "site/data/track_record.json"
    )
    if ($env:ARCHIVE_DIR) { $recordArgs += @("--archive-dir", $env:ARCHIVE_DIR) }
    python @recordArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-BackfillArchive {
    if (-not $env:PAGES_BASE) {
        Write-Host "backfill-archive needs PAGES_BASE"
        exit 1
    }
    Write-Host "Seeding published-slates from the live site..."
    python -m endzone.site.archive backfill --pages-base $env:PAGES_BASE --remote origin
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Reproduce {
    Invoke-Install
    Invoke-DbtBuildDatabricks
    Invoke-VerifyReference
    Write-Host "Reference check finished. Marts were built on Databricks. No model was registered."
    Write-Host "reproduce does not download nflverse. Run .\scripts\dev.ps1 data first on an empty catalog."
}

function Invoke-Ingest {
    Write-Host "Loading nflverse data (all seasons 1999-2026, games, pbp, and depth charts)..."
    python -m endzone.ingest.load_raw_nflverse --dataset all
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Update-Stamp data
}

function Invoke-Data {
    Invoke-Ingest
}

function Invoke-IngestCurrent {
    Write-Host "Refreshing current season only..."
    python -m endzone.ingest.load_raw_nflverse --current-season --dataset all
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Update-Stamp data
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
    Update-Stamp marts
}

function Invoke-Marts {
    if (-not (Test-Path (Join-Path .stamp "data"))) { Invoke-Data }
    Invoke-DbtBuildDatabricks
}

# Main execution
switch ($Target.ToLower()) {
    "help" { Show-Help }
    "setup" { Invoke-Setup }
    "install" { Invoke-Install }
    "lint" { Invoke-Lint }
    "lint-structure" { Invoke-LintStructure }
    "format" { Invoke-Format }
    "test" { Invoke-Test }
    "vulture" { Invoke-Vulture }
    "dbt-compile" { Invoke-DbtCompile }
    "dbt-run" { Invoke-DbtRun }
    "dbt-test" { Invoke-DbtTest }
    "data" { Invoke-Data }
    "ingest" { Invoke-Ingest }
    "ingest-current" { Invoke-IngestCurrent }
    "marts" { Invoke-Marts }
    "dbt-build-databricks" { Invoke-DbtBuildDatabricks }
    "train" { Invoke-Train }
    "train-sweep" { Invoke-TrainSweep }
    "train-cv" { Invoke-TrainCv }
    "mlflow-ui" { Invoke-MlflowUi }
    "audit-features" { Invoke-AuditFeatures }
    "check-epa-filter" { Invoke-CheckEpaFilter }
    "verify-reference" { Invoke-VerifyReference }
    "register-model" { Invoke-RegisterModel }
    "promote-model" { Invoke-PromoteModel }
    "train-s2" { Invoke-TrainS2 }
    "score" { Invoke-Score }
    "rebuild-week" { Invoke-RebuildWeek }
    "refresh-results" { Invoke-RefreshResults }
    "track-record" { Invoke-TrackRecord }
    "backfill-archive" { Invoke-BackfillArchive }
    "clean" { Invoke-Clean }
    "reproduce" { Invoke-Reproduce }
    default {
        Write-Host "Unknown target: $Target"
        Write-Host ""
        Show-Help
        exit 1
    }
}
