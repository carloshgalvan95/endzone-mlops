# Definition of done

A change is done when the gates below are green. Modules group one responsibility. Size is a smell to review, not a gate. There is no line-count check in CI.

## Gates

- `ruff check endzone/ tests/` and `ruff format --check endzone/ tests/`
- `make lint-structure` (`scripts/check_structure.py` and `lint-imports`)
- `vulture endzone scripts --min-confidence 80`
- `pytest tests/ -v`
- Analysis coverage stays at 100% when `endzone/analysis` changes: `pytest tests/test_feature_leakage.py tests/test_feature_audit.py tests/test_epa_play_filter_check.py --cov=endzone/analysis --cov-report=term-missing --cov-fail-under=100`
- Golden test: `pytest tests/test_golden_cv.py`. Metrics match `tests/golden/cv_report.json`. Do not edit that file to make a refactor pass.
- `make dbt-compile` (local DuckDB). Warehouse `dbt build` is not a CI job.
- `make verify-reference` on a machine with warehouse credentials. CI does not run it. It refits logistic `h2_full`, Elo, and the market and fails when they differ from `docs/experiments/reference_metrics.json` by more than 1e-4.
- Pull request template filled in (`.github/pull_request_template.md`)
- Conventional Commits (`type(scope): subject`). Author and committer are Carlos Galván. No `Co-authored-by` trailer.
- No secrets, hosts, HTTP paths, `.env`, `profiles.yml`, or data files in the diff.

Windows runs the same gates through `.\scripts\dev.ps1` (`lint`, `lint-structure`, `test`, `vulture`, `dbt-compile`, `verify-reference`). `dev.ps1` also has `format`. The Makefile does not.

## What done is not

`make reproduce` installs the lock, builds Databricks marts, and runs `verify-reference`. It does not download nflverse and it does not register a model. `score.py` is still not in the repo.
