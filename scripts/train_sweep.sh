#!/bin/bash
# Run training sweep: 5+ model configs for MLflow comparison

set -e

echo "Running training sweep (5 configs)..."
echo ""

# Load .env if it exists
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

# 1. Baseline
echo "1/5: Baseline (home-always)"
python -m src.train.train --model baseline-home --seed 42

# 2. Logistic C=0.1
echo "2/5: Logistic Regression (C=0.1)"
python -m src.train.train --model logistic --C 0.1 --lookback 10 --seed 42

# 3. Logistic C=10.0
echo "3/5: Logistic Regression (C=10.0)"
python -m src.train.train --model logistic --C 10.0 --lookback 10 --seed 42

# 4. GBM max_depth=3
echo "4/5: Gradient Boosting (depth=3, n=50)"
python -m src.train.train --model gbm --max-depth 3 --n-estimators 50 --learning-rate 0.1 --seed 42

# 5. GBM max_depth=5
echo "5/5: Gradient Boosting (depth=5, n=100)"
python -m src.train.train --model gbm --max-depth 5 --n-estimators 100 --learning-rate 0.1 --seed 42

echo ""
echo "Training sweep complete. View results: mlflow ui --backend-store-uri ./mlruns"
