#!/usr/bin/env pwsh
# Run training sweep: 5+ model configs for MLflow comparison

Write-Host "Running training sweep (5 configs)..."
Write-Host ""

# Load .env if it exists
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$' -and -not $_.StartsWith('#')) {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}

# 1. Baseline
Write-Host "1/5: Baseline (home-always)"
python -m src.train.train --model baseline-home --seed 42

# 2. Logistic C=0.1
Write-Host "2/5: Logistic Regression (C=0.1)"
python -m src.train.train --model logistic --C 0.1 --lookback 10 --seed 42

# 3. Logistic C=10.0
Write-Host "3/5: Logistic Regression (C=10.0)"
python -m src.train.train --model logistic --C 10.0 --lookback 10 --seed 42

# 4. GBM max_depth=3
Write-Host "4/5: Gradient Boosting (depth=3, n=50)"
python -m src.train.train --model gbm --max-depth 3 --n-estimators 50 --learning-rate 0.1 --seed 42

# 5. GBM max_depth=5
Write-Host "5/5: Gradient Boosting (depth=5, n=100)"
python -m src.train.train --model gbm --max-depth 5 --n-estimators 100 --learning-rate 0.1 --seed 42

Write-Host ""
Write-Host "Training sweep complete. View results: mlflow ui --backend-store-uri sqlite:///mlflow.db"
