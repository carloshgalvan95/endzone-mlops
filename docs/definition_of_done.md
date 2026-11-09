# Definition of done

A change is done when the gates below are green. Modules group one responsibility. Size is a smell to review, not a gate. There is no line-count check in CI.

## Gates

- `ruff check endzone/ tests/` and `ruff format --check endzone/ tests/`
- `make lint-structure` (`scripts/check_structure.py` and `lint-imports`)
- `vulture endzone scripts --min-confidence 80`
- `pytest tests/ -v`
- Analysis coverage stays at 100% when `endzone/analysis` changes: `pytest tests/test_feature_leakage.py tests/test_feature_audit.py tests/test_epa_play_filter_check.py --cov=endzone/analysis --cov-report=term-missing --cov-fail-under=100`
- Golden test: `pytest tests/test_golden_cv.py`. Metrics match `tests/golden/cv_report.json`. Do not edit that file to make a refactor pass.
- `make dbt-compile` (local DuckDB). The test workflow does not run warehouse `dbt build`. `pages.yml` builds marts and scores only on the Tuesday cron or a manual dispatch. A push restores `published-slates` first, overlays finals, archives the files it will deploy, and does not fit a model. A failed archive does not deploy. See ADR-006.
- `make verify-reference` on a machine with warehouse credentials. CI does not run it. It refits logistic `h2_full`, Elo, and the market and fails when they differ from `docs/experiments/reference_metrics.json` by more than 1e-4.
- Pull request template filled in (`.github/pull_request_template.md`)
- Conventional Commits (`type(scope): subject`). Author and committer are Carlos Galván. No `Co-authored-by` trailer.
- The local suite above passes before any push. Tests that read or write text pass `encoding="utf-8"`.
- The coding agent pushes a `feat/`, `fix/`, or other conventional branch and does not open the pull request. Carlos opens it from his account. A branch with an open pull request does not get new commits: close the pull request, push, then open a new one from Carlos's account.
- No secrets, hosts, HTTP paths, `.env`, `profiles.yml`, or data files in the diff.

Windows runs the same gates through `.\scripts\dev.ps1` (`lint`, `lint-structure`, `test`, `vulture`, `dbt-compile`, `verify-reference`). `dev.ps1` also has `format`. The Makefile does not.

## What done is not

`make reproduce` installs the lock, builds Databricks marts, and runs `verify-reference`. It does not download nflverse and it does not register a model. `make score` fits logistic `h2_full` on completed games strictly before the scored week, registers `endzone-winprob`, evaluates the Sprint 2 rule, and writes `site/data` only when that week file is not already there. A published week is kept. `make rebuild-week SEASON=2026 WEEK=9` is the explicit full-week rebuild, then the results overlay. `endzone-margin` and `endzone-total` are registered only when each target passes ADR-007. Point estimates stay null until both pass. That JSON is not committed on `main`. The orphan branch `published-slates` stores it, and `history/` keeps every earlier file. A push to `main` restores that branch first, overlays finals, and archives the files before deploy. It does not rescore. The Tuesday Pages cron scores the next unpublished week. `make refresh-results` overlays final nflverse scores onto those week files. It does not refit the model. `.github/workflows/refresh-results.yml` restores, archives, and deploys that JSON without Databricks. `make backfill-archive PAGES_BASE=<pages url>` seeds the branch once and does not score. `make track-record` writes `site/data/track_record.json` from that archive and does not edit a week file. Pages and Results do the same write before deploy. If the archive cannot be read, the file says the record is unavailable. See ADR-006, ADR-007, and ADR-008.
