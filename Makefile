.PHONY: help setup install lint test dbt-compile dbt-run dbt-test clean reproduce

help:
	@echo "Available targets:"
	@echo "  setup        - Create virtualenv and install dependencies"
	@echo "  install      - Install Python dependencies (in active environment)"
	@echo "  lint         - Run code quality checks (ruff)"
	@echo "  test         - Run pytest unit tests"
	@echo "  dbt-compile  - Compile dbt models"
	@echo "  dbt-run      - Run dbt models (creates tables)"
	@echo "  dbt-test     - Run dbt tests"
	@echo "  reproduce    - Full reproduction pipeline (install -> dbt -> train)"
	@echo "  clean        - Remove build artifacts"
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

dbt-compile:
	cd dbt && dbt compile --profiles-dir .

dbt-run:
	cd dbt && dbt run --profiles-dir .

dbt-test:
	cd dbt && dbt test --profiles-dir .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dbt/target dbt/logs dbt/dbt_packages

reproduce: install dbt-run
	@echo "Reproduction pipeline complete. Next: run train.py when implemented."
