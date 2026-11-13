# Margin model converted to win probability

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_margin/comparison.md` here only after the warehouse run, and only if I decide a candidate.

## Goal

Round 5 converts a point-margin model into a home win probability. A margin ridge already exists for ADR-007, on `h2_full`, and its implied win log loss tied logistic `h2_full` on those same columns. On the defender's columns the margin model alone is likely to land close to the defender. The research Page already says the most likely use is as the second member of an average. The Page's expected effect is about -0.001 to -0.004, and a result near 0 is likely.

Preseason win totals are not used. The ADR-007 score models are not changed.

## Reference

Logistic `pkg_core_inj_l2`, C fixed at 0.1. Rolling-origin CV, seasons 2016 through 2023. The published defender count on that window is 2191 games.

`h2_full`, Elo, and `elo_mov` are scored in the same run. They are context. They are not candidates. `elo_mov` is the home win probability from `elo_mov_diff` plus home advantage 65 on a 400 scale. Its decision is `baseline`.

The market is a benchmark. It is not a candidate.

## Candidates

Two candidates. Logistic `pkg_core_inj_l2` stays the defender. No search.

1. `margin_ridge_pkg_core_inj`. Ridge regression of the home margin, home score minus away score, on exactly the defender's columns. The imputation and the `StandardScaler` are the defender's. `Ridge(alpha=1.0)` is the ADR-007 value. The home win probability is `p = Phi(mu / sigma)`. Sigma is the standard deviation of that fold's training residuals, `ddof=1`. Probabilities are clipped to `[1e-6, 1 - 1e-6]`.
2. `margin_mean`. The logit average of candidate 1 and the defender, with equal weights 0.5 and 0.5. No weight is fit. Both probabilities are clipped to `[1e-6, 1 - 1e-6]` before the logit.

Alpha, the weights, and sigma are fixed. Sigma depends only on the training seasons of each fold. Win labels and ties are the defender's: `home_won` is 1 only when the home score is greater, and a tie is 0. Log loss uses those same rows and labels. Nothing from 2024, 2025, or later enters any fold.

## Acceptance

ADR-005, unchanged. Each candidate is judged alone against logistic `pkg_core_inj_l2`. Accept it only when the log loss difference is at most -0.003 and the 95 percent paired bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. If both pass, the one with the lower log loss is the comparison's best row. That does not register or promote it.

Per-season results, calibration, and the market-information regression are descriptive. Margin MAE, the per-fold sigma, and the correlation between candidate 1 and the defender in logit space are descriptive. They are not gates.

Do not score 2024, 2025, or the held-out split. No tuning after the results. No threshold changes. No extra candidates. Nothing is registered or promoted.

## Reproduce

No dbt build is needed before training. The candidates reuse the defender's columns. Only the target and the probability mapping are new.

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_margin.py
```

`make train-margin` is the train step alone. Windows: `.\scripts\dev.ps1 train-margin`.

`train_margin.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` keeps C at 0.1. Ridge alpha stays 1.0. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`. `elo_mov` is `baseline`.

The local files are:

- `reports/phase2_margin/comparison.md`
- `reports/phase2_margin/comparison.csv`
- `reports/phase2_margin/acceptance.csv`
- `reports/phase2_margin/calibration.csv`
- `reports/phase2_margin/market_information.csv`
- `reports/phase2_margin/margin_diagnostics.csv`
- `reports/phase2_margin/oof_logistic_h2_full.csv`
- `reports/phase2_margin/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_margin/oof_margin_ridge_pkg_core_inj_pkg_core_inj_l2.csv`
- `reports/phase2_margin/oof_margin_mean_pkg_core_inj_l2.csv`
- `reports/phase2_margin/oof_elo_elo.csv`
- `reports/phase2_margin/oof_elo_mov_elo_mov.csv`

The home prior and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_margin`.

Read `reports/phase2_margin/comparison.md` after the run.

No model is promoted from this run.
