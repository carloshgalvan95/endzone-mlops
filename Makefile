.PHONY: help setup install lint lint-structure test vulture dbt-compile dbt-run dbt-test clean reproduce data ingest ingest-current marts dbt-build-databricks train train-sweep train-cv train-recency train-h6 train-pkg-core train-pkg-core-inj train-ensembles train-s2 mlflow-ui audit-features check-epa-filter verify-reference walk-forward verify-walk-forward score rebuild-week refresh-results track-record backfill-archive register-model register-recent promote-model

# Finished bronze and mart steps. Gitignored. A second train-cv does not
# reload bronze when these stamps are current.
STAMP_DIR := .stamp

# Git Bash on Windows rewrites arguments that look like Unix paths.
# DATABRICKS_HTTP_PATH starts with /sql/1.0/... and would be mangled.
# Harmless on macOS and Linux.
export MSYS_NO_PATHCONV=1

help:
	@echo "Available targets:"
	@echo "  setup               - Create the Python 3.12 venv from uv.lock"
	@echo "  install             - Install the locked environment (uv sync --frozen)"
	@echo "  lint                - Run code quality checks (ruff)"
	@echo "  lint-structure      - Check the CCDS layout and import-linter contracts"
	@echo "  test                - Run pytest unit tests"
	@echo "  vulture             - Fail on high-confidence unused code"
	@echo "  dbt-compile         - Compile dbt models"
	@echo "  dbt-run             - Run dbt models (creates tables)"
	@echo "  dbt-test            - Run dbt tests"
	@echo "  data                - Load bronze (same as ingest) and record .stamp/data"
	@echo "  ingest              - Load nflverse games, play-by-play, and depth charts"
	@echo "  ingest-current      - Refresh current season only (for weekly update)"
	@echo "  marts               - Build Databricks marts after bronze is loaded"
	@echo "  dbt-build-databricks - dbt deps, then build models on the Databricks target"
	@echo "  train               - Train one holdout model with MLflow"
	@echo "  train-sweep         - Single-split sweep (5 configs). Not the reference CV"
	@echo "  train-cv            - H5b CV. Builds data and marts when .stamp says they are stale"
	@echo "  train-recency       - Preregistered recency CV, seasons 2016-2023. Does not register"
	@echo "  train-h6            - Preregistered H6 CV, seasons 2016-2023. Does not register"
	@echo "  train-pkg-core      - Preregistered pkg_core CV, seasons 2016-2023. Does not register"
	@echo "  train-pkg-core-inj  - Preregistered injury CV, seasons 2016-2023. Does not register"
	@echo "  train-ensembles     - Preregistered booster and mean, seasons 2016-2023. Does not register"
	@echo "  mlflow-ui           - Open MLflow UI against the local SQLite store"
	@echo "  audit-features      - Profile current pregame inputs (no feature changes)"
	@echo "  check-epa-filter    - Count play types in the EPA filter, plus kneel and spike rows"
	@echo "  verify-reference    - Refit logistic h2_full and fail if it drifts from the reference file"
	@echo "  walk-forward        - Week backtest of a fresh h2_full fit against the through-2023 fit"
	@echo "  verify-walk-forward - Fail unless the same-recipe refresh is inside the margin"
	@echo "  register-model      - Register logistic h2_full through 2023. Does not set the prod alias"
	@echo "  register-recent     - Register the same-recipe refresh when the gate passes"
	@echo "  promote-model       - Set alias prod on VERSION when its tags pass the gate"
	@echo "  score               - Score the next unpublished week. Keeps a week file already on disk"
	@echo "  train-s2            - Sprint 2 margin and total CV. Registers a target only if it passes"
	@echo "  rebuild-week        - Rebuild one full week (SEASON and WEEK required), then overlay finals"
	@echo "  refresh-results     - Overlay final nflverse scores onto published week JSON"
	@echo "  track-record        - Write site/data/track_record.json from the archive. Does not edit week files"
	@echo "  backfill-archive    - Seed published-slates from PAGES_BASE. Does not score"
	@echo "  reproduce           - install, Databricks dbt build, verify-reference"
	@echo "  clean               - Remove build artifacts"
	@echo ""
	@echo "Note: Most targets expect an activated virtualenv (source .venv/bin/activate)"
	@echo "      Python targets load .env. dbt-compile, dbt-run, and dbt-test still source it."

setup:
	@echo "Creating the Python 3.12 environment in .venv/ from uv.lock"
	uv python install 3.12
	uv sync --frozen
	@echo ""
	@echo "Setup complete. Activate the environment with:"
	@echo "  source .venv/bin/activate"

install:
	uv sync --frozen

lint:
	ruff check endzone/ tests/
	ruff format --check endzone/ tests/

lint-structure:
	python scripts/check_structure.py
	lint-imports

test:
	pytest tests/ -v

vulture:
	vulture endzone scripts --min-confidence 80

# data and ingest are the bronze load. The stamp lets marts and train-cv
# skip a second download.
data: ingest

ingest:
	python -m endzone.ingest.load_raw_nflverse --dataset all
	mkdir -p $(STAMP_DIR)
	touch $(STAMP_DIR)/data

ingest-current:
	python -m endzone.ingest.load_raw_nflverse --current-season --dataset all
	mkdir -p $(STAMP_DIR)
	touch $(STAMP_DIR)/data

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
	mkdir -p $(STAMP_DIR)
	touch $(STAMP_DIR)/marts

# marts depends on bronze. A data stamp already on disk is not reloaded.
marts:
	@if [ ! -f $(STAMP_DIR)/data ]; then $(MAKE) data; fi
	$(MAKE) dbt-build-databricks

train:
	python -m endzone.train.train --model logistic --lookback 10 --seed 42

train-sweep:
	bash scripts/train_sweep.sh

# data -> marts -> train-cv. A missing marts stamp builds both. A data stamp
# newer than the marts stamp rebuilds marts only. A marts stamp with no data
# stamp (depth charts loaded outside make data) does not reload bronze.
train-cv:
	@if [ ! -f $(STAMP_DIR)/marts ] || { [ -f $(STAMP_DIR)/data ] && [ $(STAMP_DIR)/data -nt $(STAMP_DIR)/marts ]; }; then \
		$(MAKE) marts; \
	fi
	python -m endzone.train.train --cv --phase 2 --hypothesis h5b --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5b

# Preregistered recency challengers. Does not build marts, walk forward,
# register, or promote. The script fixes the window at 2016-2023.
train-recency:
	python scripts/train_recency.py

# Preregistered H6 challengers. Does not build marts, walk forward,
# register, or promote. The script fixes the window at 2016-2023.
train-h6:
	python scripts/train_h6.py

# Preregistered pkg_core candidate. Does not build marts, walk forward,
# register, or promote. The script fixes the window at 2016-2023.
train-pkg-core:
	python scripts/train_pkg_core.py

# Preregistered injury candidates. Does not build marts, walk forward,
# register, or promote. The script fixes the window at 2016-2023.
train-pkg-core-inj:
	python scripts/train_pkg_core_inj.py

# Preregistered booster and mean ensemble. Does not build marts, walk
# forward, register, or promote. The script fixes the window at 2016-2023.
train-ensembles:
	python scripts/train_ensembles.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

audit-features:
	python -m endzone.analysis.feature_audit --output-dir reports/phase0 $(AUDIT_ARGS)

check-epa-filter:
	python -m endzone.analysis.epa_play_filter_check --output reports/epa_filter_check.md

verify-reference:
	python -m endzone.train.train --verify-reference

# Week walk-forward. Default slate is the week after the last completed week.
# Set both for a publish, for example SLATE_SEASON=2026 SLATE_WEEK=10.
# Writes reports/walk_forward/. That directory is local scratch.
walk-forward:
	python -m endzone.modeling.walk_forward $(if $(SLATE_SEASON),--slate-season $(SLATE_SEASON),) $(if $(SLATE_WEEK),--slate-week $(SLATE_WEEK),) $(if $(OUTPUT),--output-dir $(OUTPUT),)

# Exit 1 unless the report passes. A same-recipe refresh uses the 0.002 margin.
# A different recipe has to be strictly lower. An existing JSON is enough.
verify-walk-forward:
	python -m endzone.modeling.walk_forward --verify --report $(if $(REPORT),$(REPORT),reports/walk_forward/walk_forward.json)

# Register logistic h2_full through THROUGH_SEASON (default 2023).
# Does not set the prod alias. COMPARISON points at a comparison CSV and
# skips the reference check. The gate still applies.
# Unity Catalog needs MLFLOW_TRACKING_URI=databricks and
# MLFLOW_REGISTRY_URI=databricks-uc. See the runbook.
register-model:
	python -m endzone.modeling.registry_command register $(if $(THROUGH_SEASON),--through-season $(THROUGH_SEASON),) $(if $(COMPARISON),--comparison-csv $(COMPARISON),)

# Fit games before the slate in the walk-forward report and register that version.
# Does not set the alias. The recipe has to match prod, and the refresh gate
# has to pass. A different recipe is refused before a write.
# Unity Catalog needs the same URIs as register-model.
register-recent:
	python -m endzone.modeling.registry_command register-recent --report $(if $(REPORT),$(REPORT),reports/walk_forward/walk_forward.json)

# Set alias prod on one existing version. Example: make promote-model VERSION=1
promote-model:
	@test -n "$(VERSION)" || (echo "promote-model needs VERSION" >&2; exit 1)
	python -m endzone.modeling.registry_command promote --version $(VERSION)

# Score the next unpublished week. A file already on disk is kept.
# make score SEASON=2026 WEEK=10
# A week file already in site/data is kept. REPLACE=1 rebuilds that week.
# Generated JSON under site/data is not committed. See ADR-006.
# Margin and total stay null unless both pass ADR-007.
score:
	python -m endzone.modeling.score $(if $(SEASON),--season $(SEASON),) $(if $(WEEK),--week $(WEEK),) $(if $(REPLACE),--replace-published,)

# Rebuild every scheduled game for one week, then overlay finals.
# Fits only on completed games before that week. Example:
# make rebuild-week SEASON=2026 WEEK=9
rebuild-week:
	@test -n "$(SEASON)" && test -n "$(WEEK)" || (echo "rebuild-week needs SEASON and WEEK" >&2; exit 1)
	python -m endzone.modeling.score --season $(SEASON) --week $(WEEK) --replace-published $(if $(RESTORE_PUBLISHED),--restore-published $(RESTORE_PUBLISHED),)
	python -m endzone.site.results

# Sprint 2 margin and total CV. Writes reports/sprint2/cv_report.json.
# Registers endzone-margin and endzone-total only when each target passes.
train-s2:
	python -m endzone.modeling.points

# Overlay final nflverse scores onto week JSON already in site/data.
# This does not refit the model, run dbt, or read the warehouse.
refresh-results:
	python -m endzone.site.results $(if $(RESTORE_PUBLISHED),--restore-published $(RESTORE_PUBLISHED),)

# Write site/data/track_record.json from published-slates and site/data.
# Does not modify week files. ARCHIVE_DIR skips the git checkout.
track-record:
	python -m endzone.site.track_record --repo . --site-data site/data --output site/data/track_record.json $(if $(ARCHIVE_DIR),--archive-dir $(ARCHIVE_DIR),)

# One-time seed of the published-slates branch from the live site.
# Does not score and does not deploy. Example:
# make backfill-archive PAGES_BASE=https://<owner>.github.io/<repo>
backfill-archive:
	@test -n "$(PAGES_BASE)" || (echo "backfill-archive needs PAGES_BASE" >&2; exit 1)
	python -m endzone.site.archive backfill --pages-base "$(PAGES_BASE)" --remote origin

# Installs the lock, builds Databricks marts from bronze already in the
# warehouse, and refits the reference. It does not download nflverse and it
# does not register a model and does not set an alias. Run `make data`
# first on an empty catalog. Register and promote are separate targets.
# Local DuckDB compile stays `make dbt-compile`.
reproduce: install
	$(MAKE) dbt-build-databricks
	$(MAKE) verify-reference
	@echo "Reference check finished. Marts were built on Databricks. No model was registered."
