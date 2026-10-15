.PHONY: help setup install lint test dbt-compile dbt-run dbt-test clean reproduce ingest ingest-current dbt-build-databricks train train-sweep train-cv mlflow-ui audit-features check-epa-filter

# Git Bash on Windows rewrites arguments that look like Unix paths.
# DATABRICKS_HTTP_PATH starts with /sql/1.0/... and would be mangled.
# Harmless on macOS and Linux.
export MSYS_NO_PATHCONV=1

help:
	@echo "Available targets:"
	@echo "  setup               - Create virtualenv and install dependencies"
	@echo "  install             - Install Python dependencies (in active environment)"
	@echo "  lint                - Run code quality checks (ruff)"
	@echo "  test                - Run pytest unit tests"
	@echo "  dbt-compile         - Compile dbt models"
	@echo "  dbt-run             - Run dbt models (creates tables)"
	@echo "  dbt-test            - Run dbt tests"
	@echo "  ingest              - Load nflverse games, play-by-play, and depth charts"
	@echo "  ingest-current      - Refresh current season only (for weekly update)"
	@echo "  dbt-build-databricks - dbt deps, then build models on the Databricks target"
	@echo "  train               - Train NFL pregame prediction models with MLflow"
	@echo "  train-sweep         - Run training sweep (5+ configs for MLflow comparison)"
	@echo "  train-cv            - Phase 2 H5 starting QB CV against logistic h2_full"
	@echo "  mlflow-ui           - Open MLflow UI against the local SQLite store"
	@echo "  audit-features      - Profile current pregame inputs (no feature changes)"
	@echo "  check-epa-filter    - Count play types in the EPA filter, plus kneel and spike rows"
	@echo "  reproduce           - Full reproduction pipeline (install -> dbt compile)"
	@echo "  clean               - Remove build artifacts"
	@echo ""
	@echo "Note: Most targets expect an activated virtualenv (source .venv/bin/activate)"

setup:
	@echo "Creating virtual environment in .venv/"
	python3 -m venv .venv
	@echo "Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Activate the environment with:"
	@echo "  source .venv/bin/activate"

install:
	pip install -r requirements.txt

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v

ingest:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	python -m src.ingest.load_raw_nflverse --dataset all

ingest-current:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	python -m src.ingest.load_raw_nflverse --current-season --dataset all

dbt-compile:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	cd dbt && dbt compile --profiles-dir .

dbt-run:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	cd dbt && dbt run --profiles-dir .

dbt-test:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	cd dbt && dbt test --profiles-dir .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dbt/target dbt/logs dbt/dbt_packages

dbt-build-databricks:
	@echo "Installing dbt packages, then building on the Databricks target..."
	cd dbt && dbt deps --profiles-dir . && dbt build --profiles-dir . --target databricks

train:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	python -m src.train.train --model logistic --lookback 10 --seed 42

train-sweep:
	bash scripts/train_sweep.sh

train-cv:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

audit-features:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	python -m src.analysis.feature_audit --output-dir reports/phase0 $(AUDIT_ARGS)

check-epa-filter:
	@if [ -f .env ]; then set -a && . ./.env && set +a; fi; \
	python -m src.analysis.epa_play_filter_check --output reports/epa_filter_check.md

reproduce: install dbt-compile
	@echo "Reproduction pipeline complete (dependencies installed, dbt compiled on DuckDB)."
