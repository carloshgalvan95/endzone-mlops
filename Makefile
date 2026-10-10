.PHONY: help setup install lint test dbt-compile dbt-run dbt-test clean reproduce ingest ingest-current dbt-build-databricks

help:
	@echo "Available targets:"
	@echo "  setup               - Create virtualenv and install dependencies"
	@echo "  install             - Install Python dependencies (in active environment)"
	@echo "  lint                - Run code quality checks (ruff)"
	@echo "  test                - Run pytest unit tests"
	@echo "  dbt-compile         - Compile dbt models"
	@echo "  dbt-run             - Run dbt models (creates tables)"
	@echo "  dbt-test            - Run dbt tests"
	@echo "  ingest              - Load nflverse data (all seasons 1999-2026, games and pbp)"
	@echo "  ingest-current      - Refresh current season only (for weekly update)"
	@echo "  dbt-build-databricks - Build dbt models on Databricks target"
	@echo "  reproduce           - Full reproduction pipeline (install -> dbt -> train)"
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
	@echo "Building dbt models on Databricks target..."
	cd dbt && dbt build --profiles-dir . --target databricks

reproduce: install dbt-run
	@echo "Reproduction pipeline complete. Next: run train.py when implemented."
