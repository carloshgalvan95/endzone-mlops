# ADR-007: Margin and total ridge regressions

**Date**: 2026-11-04
**Status**: Accepted

## Context

The page shows a home win probability. Sprint 2 adds an estimated score: home points = (total + margin) / 2 and away points = (total - margin) / 2. Margin is home score minus away score. The two targets are fit separately.

The win-probability reference stays logistic `h2_full`. Its rolling-origin log loss on seasons 2016-2023 is 0.6377. This ADR does not move the 0.003 rule in ADR-005. The comparison value reported for an implied win probability is 0.638, the one-decimal figure named before the fit. It is a report, not a gate.

Closing lines are a hard benchmark. This model does not use `spread_line` or `total_line` as features. nflverse stores `spread_line` as the home line, and a negative number means the home team is favored (`dbt/models/staging/sources.yml`). The market margin used for MAE is therefore `-spread_line`. `total_line` is the market total.

## Decision

The hypothesis file is `configs/experiments/s2.yaml`. The classification catalog does not score it. Thresholds below are fixed before any fit. They are not moved after a result.

### Model

- One ridge regression per target, on the `h2_full` pregame columns only.
- `Ridge(alpha=1.0)` after the same imputation groups as the logistic pipeline, then `StandardScaler`.
- `random_state` 42. The solver is `cholesky`.
- Features for a game use only rows with `gameday` strictly before that game. The same rule as the win-probability model.

### Cross-validation

Rolling-origin seasons 2016 through 2023. Train on `season < S` and score `season == S`. Eight seasons. 2024 is not a fold. 2025 is not scored. Bootstrap: 1000 resamples, seed 42, the same game indexes for the model and the baseline.

### Baselines the model must beat

- Margin: Elo point spread. Pregame Elo, including the stored home-field rating, divided by 25. That is the FiveThirtyEight mapping from a rating gap to a spread. The Elo settings stay the ones already used for win probability.
- Total: the mean total points on the training rows of that fold.

### Acceptance

Each target is accepted only when both are true:

1. Pooled out-of-fold MAE beats its baseline, and the 95% paired bootstrap interval of (model MAE minus baseline MAE) lies entirely below 0.
2. The model MAE is strictly lower than the baseline MAE in at least 5 of the 8 seasons.

Anything else is rejected. A missing interval, or fewer than 8 scored seasons, is inconclusive. An inconclusive target is not accepted. The displayed score requires both targets accepted. A registered model is that target's ridge pipeline refit on every completed game. `endzone-margin` and `endzone-total` are registered only for an accepted target. A rejected or inconclusive target emits null point estimates.

### Always reported, never required

- Market MAE against `-spread_line` and `total_line`. Rows with a missing line are left out of the market MAE only.
- RMSE.
- 80% interval coverage. For each fold, sigma is the training residual standard deviation (`ddof=1`). The interval is the prediction plus or minus `z * sigma`, where `z` is the standard normal quantile for a central 80% interval. Coverage is the share of eval games inside that interval.
- Implied home win probability `Phi(mu / sigma)` from the margin model, and its log loss next to 0.638. Sigma is the same per-fold training residual sigma.

## Consequences

`python -m endzone.modeling.points` runs the comparison and writes `reports/sprint2/cv_report.json`. That directory is local scratch. Copy the JSON into `docs/experiments/reports/` only after the warehouse run, and do not hand-edit the numbers. `make score` uses the same rule. Until both targets pass, the page hides the estimate and says it did not pass its pre-registered test.

The win-probability golden file is unchanged. This run does not promote a new win-probability reference.
