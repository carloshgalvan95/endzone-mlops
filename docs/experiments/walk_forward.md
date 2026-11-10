# Walk-forward week backtest

Method note. This is not a new feature phase. The feature set stays logistic `h2_full`. The 0.003 rule in ADR-005 is not this gate and is not moved. Phases 3 and 4 are still not started.

There is no log loss in this file. `reports/walk_forward/` is gitignored scratch. I paste the table here only after I run the command on the warehouse. I do not hand-edit `walk_forward.json`. A hand edit of one field fails the gate, because the aggregate has to match the week rows.

The market has the lower log loss on the locked 2016-2023 table in `docs/experiments/reference_metrics.json`. This backtest does not replace that table. Nothing here is a betting pick.

## What is compared

Each scored week is fit only on completed games before that week.

- Candidate: logistic `h2_full` on every completed game before the week.
- v1: the same recipe, capped at seasons through `cv_end_season` in the reference file (2023). That training set does not grow.
- Elo: ratings at the start of the week. Games inside the week do not update each other.
- Home baseline: the home win rate on completed games before the week. It is not capped at 2023.
- Market: the no-vig closing moneyline on the row. A game with no line is left out of the market row only. Market `n` can be smaller.

The aggregate is the pooled log loss, not the mean of the weekly means. The paired bootstrap of candidate minus v1 uses `n_bootstrap` and `bootstrap_seed` from the reference file unless the command overrides them. The interval is descriptive. Promotion uses the point estimate: the candidate log loss has to be strictly lower than v1. A tie does not pass. The 0.003 margin is not applied.

Weeks inside the v1 window are not scored. There the two training sets are the same fit. The default window starts the season after 2023 and stops before the slate.

## Leakage

Eval features, labels, and the Elo update drop the whole scored week. The slate is published on Tuesday, before any game that week, and before injury reports. A Thursday result is not a feature for Sunday.

Training rows for earlier games still use `gameday` strictly before kickoff. That is the same rule `build_pregame_features` already uses. The outer feature build stops at the slate, so a game on or after the slate cannot enter a training row.

The model I would register is a different object. It is one fit on every completed game before the slate, including the last scored week. The walk-forward log loss is not the in-sample loss of that fit.

## Features I did not add

`h2_full` already has recent offensive and defensive EPA (lookback 10, cross-season), recent win rate, rest days, season offensive EPA, season defensive EPA, and the regressed recent and season EPA columns (prior-season shrink 0.333, blend n/(n+4)).

Point differential is leak-free at kickoff and is not a column. I am not adding it. A new column would mix a feature change into the window test, and H1 differentials were already rejected. Elo stays a baseline. It is not an input. Injury and inactive lists are not inputs: Tuesday publish is before those reports.

## Command

Pass the slate. The default is the week after the last completed week in the latest season, and a later week in the marts would move that default.

```bash
make walk-forward SLATE_SEASON=2026 SLATE_WEEK=10
```

Windows, from the repo root:

```powershell
$env:SLATE_SEASON = "2026"
$env:SLATE_WEEK = "10"
.\scripts\dev.ps1 walk-forward
```

That writes `reports/walk_forward/walk_forward.json` and `reports/walk_forward/walk_forward.md`. A candidate that loses to v1 still exits 0. The run is the result.

```bash
make verify-walk-forward
```

Windows: `.\scripts\dev.ps1 verify-walk-forward`. Exit 1 unless the report beats v1. Exit 1 is also a bad or edited file.

`make register-model` stays the through-2023 golden path. Do not pass `THROUGH_SEASON=2026` and expect a promote. A `through_season` tag past 2023 without `evaluation=walk_forward` is refused. The Model registry workflow is that same golden path and was not changed.

The order for a fresh prod candidate, including the week 10 publish, is in `docs/runbook.md`.
