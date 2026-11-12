# Prior strength and season weights

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_prior_shrink/comparison.md` here only after the warehouse run, and only if I decide a candidate.

## Goal

The defender already shrinks early-season EPA. Every regressed adjusted EPA column blends the current season with the prior-season final value moved 1/3 toward the prior-season league mean. The current weight is `n / (n + k)`, and one shared `k` of 4 is used for offense and defense.

This comparison changes two things, and nothing else. Offense and defense use different blend constants, taken from FTN's published DAVE weights. The second candidate also down-weights older training seasons. The research Page expected a gain of about 0.002 to 0.006 when none of the existing shrinkage was in place, so a smaller effect is likely.

Preseason win totals are not used. The prior is the previous season only, as it is today.

## Reference

Logistic `pkg_core_inj_l2`, C fixed at 0.1. Rolling-origin CV, seasons 2016 through 2023. The published defender count on that window is 2191 games.

`h2_full`, Elo, and `elo_mov` are scored in the same run. They are context. They are not candidates. `elo_mov` is the home win probability from `elo_mov_diff` plus home advantage 65 on a 400 scale. Its decision is `baseline`.

The market is a benchmark. It is not a candidate.

## Candidates

Two candidates. Both keep the defender's column names and order. Logistic regression, C fixed at 0.1. No search.

1. `pkg_core_inj_l2_k39`. The blend constant in every regressed season and within-season recent EPA column is `k = 3` for offensive columns and `k = 9` for defensive columns. The shrink factor stays 1/3.
2. `pkg_core_inj_l2_k39_decay`. The same columns and the same blend. Training rows are weighted by `w = 0.5 ** ((last_train_season - season) / 5)`. `last_train_season` is the latest season in that fold's training rows. Scoring is unweighted.

The constants are not tuned. `k` of 3 and 9 come from FTN's DAVE: about 30 percent preseason weight on offense and 56 percent on defense after 8 games, which implies `k` of about 3.4 and 10. Both are rounded down, as written in the research Page before this round. The half-life of 5 seasons is the research Page value.

Windows, the cross-season raw recent columns, Elo, QB, recency, and the injury columns are untouched. Nothing from 2024, 2025, or later enters any fold.

## Acceptance

ADR-005, unchanged. Each candidate is judged alone against logistic `pkg_core_inj_l2`. Accept it only when the log loss difference is at most -0.003 and the 95 percent paired bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. If both pass, the one with the lower log loss is the comparison's best row. That does not register or promote it.

Per-season results, calibration, and the market-information regression are descriptive. They are not gates.

Do not score 2024, 2025, or the held-out split. No tuning after the results. No threshold changes. No extra candidates. Nothing is registered or promoted.

## Reproduce

No dbt build is needed before training. The candidates reuse columns that are already on the pregame frame. Only the blend constant and, for the decay candidate, the training weights change.

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_prior_shrink.py
```

`make train-prior-shrink` is the train step alone. Windows: `.\scripts\dev.ps1 train-prior-shrink`.

`train_prior_shrink.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` and both candidates keep C at 0.1. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`. `elo_mov` is `baseline`.

The local files are:

- `reports/phase2_prior_shrink/comparison.md`
- `reports/phase2_prior_shrink/comparison.csv`
- `reports/phase2_prior_shrink/acceptance.csv`
- `reports/phase2_prior_shrink/calibration.csv`
- `reports/phase2_prior_shrink/market_information.csv`
- `reports/phase2_prior_shrink/oof_logistic_h2_full.csv`
- `reports/phase2_prior_shrink/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_prior_shrink/oof_logistic_pkg_core_inj_l2_k39.csv`
- `reports/phase2_prior_shrink/oof_logistic_pkg_core_inj_l2_k39_decay.csv`
- `reports/phase2_prior_shrink/oof_elo_elo.csv`
- `reports/phase2_prior_shrink/oof_elo_mov_elo_mov.csv`

The home prior and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_prior_shrink`.

Read `reports/phase2_prior_shrink/comparison.md` after the run.

No model is promoted from this run.
