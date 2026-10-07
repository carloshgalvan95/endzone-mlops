.PHONY: help install lint test dbt-compile dbt-run dbt-test clean reproduce

help:
	@echo "Available targets:"
	@echo "  install      - Install Python dependencies"
	@echo "  lint         - Run code quality checks (ruff)"
	@echo "  test         - Run pytest unit tests"
	@echo "  dbt-compile  - Compile dbt models"
	@echo "  dbt-run      - Run dbt models (creates tables)"
	@echo "  dbt-test     - Run dbt tests"
	@echo "  reproduce    - Full reproduction pipeline (install -> dbt -> train)"
	@echo "  clean        - Remove build artifacts"

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
