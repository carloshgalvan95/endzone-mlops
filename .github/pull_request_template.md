## Summary

Briefly describe the change and the problem it solves.

## Changes

List the main technical changes:

- 
- 

## Verification

Checklist (check at least one):

- [ ] Rebased on latest main (`git fetch && git rebase origin/main`)
- [ ] Lint and tests pass locally (`ruff check`, `ruff format --check`, `pytest`)
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
