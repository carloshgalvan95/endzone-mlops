## Definition of done

Gates: `docs/definition_of_done.md`. Modules group one responsibility. Size is a smell to review, not a gate.

- [ ] Before the push: `ruff check endzone/ tests/`, `ruff format --check endzone/ tests/`, `make lint-structure` (or `.\scripts\dev.ps1 lint-structure`), and `pytest tests/ -v`. Tests that read or write text pass `encoding="utf-8"`.
- [ ] `vulture endzone scripts --min-confidence 80`
- [ ] Analysis coverage at 100% when `endzone/analysis` changes
- [ ] Golden test (`tests/test_golden_cv.py`) still matches `tests/golden/cv_report.json`
- [ ] `make dbt-compile`
- [ ] Conventional Commits, Carlos as author and committer, no Co-authored-by
- [ ] Branch name uses a conventional prefix (`feat/`, `fix/`, ...), never `cursor/`
- [ ] I opened this pull request from my account. This branch had no open pull request when I pushed. A fix to an open pull request starts by closing it.
- [ ] This change does not overwrite or shrink a published weekly prediction file unless the PR is an explicit, documented regenerate
- [ ] No secrets, hosts, HTTP paths, `.env`, or data files
- [ ] `make verify-reference` is for a warehouse machine, not CI

## Summary

Briefly describe the change and the problem it solves.

## Changes

List the main technical changes:

- 
- 

## Verification

Checklist (check at least one):

- [ ] Rebased on latest main (`git fetch && git rebase origin/main`)
- [ ] Full local suite passed before the push (`ruff check`, `ruff format --check`, `make lint-structure` or `.\scripts\dev.ps1 lint-structure`, `pytest`). Text reads and writes use `encoding="utf-8"`.
- [ ] dbt compiles successfully (`cd dbt && dbt compile --target local`)
- [ ] For ingest/dbt changes: source column names verified against real data file
- [ ] For ingest/dbt changes: Databricks run evidence pasted below, or marked N/A with reason
- [ ] N/A (explain why verification was not needed)

Evidence:

```
(paste test output, dbt logs, or Databricks job run URL here)
```

## Risks

Describe potential issues, rollback plan, or mark as low risk with justification:

- 
