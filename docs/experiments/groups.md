# Group logistics without bagging

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_groups/comparison.md` here only after the warehouse run, and only if I decide a candidate.

## Goal

Round 6 fits four logistic models on a fixed partition of the defender columns and averages them in logit space. There is no bagging and no search. Preseason win totals are not used.

## Reference

Logistic `pkg_core_inj_l2`, C fixed at 0.1, 47 columns. Rolling-origin CV, seasons 2016 through 2023. The published defender count on that window is 2191 games.

`h2_full`, Elo, and `elo_mov` are scored in the same run. They are context. They are not candidates. `elo_mov` is the home win probability from `elo_mov_diff` plus home advantage 65 on a 400 scale. Its decision is `baseline`.

The market is a benchmark. It is not a candidate.

## Column partition

The four groups are pairwise disjoint. Their union is `PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ`.

1. `epa_form`, 36 columns. The 16 opponent-adjusted EPA columns, the 16 unadjusted EPA columns from `h2_full`, `home_recent_win_rate`, `away_recent_win_rate`, `home_recent_point_diff`, and `away_recent_point_diff`.
2. `qb`, 4 columns. `home_qb_rating`, `away_qb_rating`, `home_qb_rating_minus_team_pass_epa`, and `away_qb_rating_minus_team_pass_epa`.
3. `ratings`, 1 column. `elo_mov_diff`.
4. `injuries_rest`, 6 columns. The four home-minus-away starter absence counts, `home_rest_days`, and `away_rest_days`.

Each group uses the defender's imputation groups and `StandardScaler`, and fits `LogisticRegression` with C fixed at 0.1. The fit uses only seasons before the scored season. These four models are reported. They are not candidates.

## Candidates

Two candidates. Logistic `pkg_core_inj_l2` stays the defender. No search.

1. `group_mean`. The logit average of the four group logistics, with equal weights 0.25. Each probability is clipped to `[1e-6, 1 - 1e-6]` before the logit.
2. `group_mean_def`. The logit average of `group_mean` and the defender, with equal weights 0.5 and 0.5. The same clip applies before the logit. No weight is fit.

C, the groups, and the weights are fixed. There is no C search, no weight search, and no bagging. Nothing from 2024, 2025, or later enters any fold.

## Acceptance

ADR-005, unchanged. Each candidate is judged alone against logistic `pkg_core_inj_l2`. Accept it only when the log loss difference is at most -0.003 and the 95 percent paired bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. If both pass, the one with the lower log loss is the comparison's best row. That does not register or promote it.

Per-season results, calibration, and the market-information regression are descriptive. The out-of-fold log loss of each group and the correlation between `group_mean` and the defender in logit space are descriptive. They are not gates.

Do not score 2024, 2025, or the held-out split. The holdout command refuses the new names. No tuning after the results. No threshold changes. No extra candidates. Nothing is registered or promoted.

## Reproduce

No dbt build is needed before training. The candidates reuse the defender's columns. Only the partition and the averages are new.

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_groups.py
```

`make train-groups` is the train step alone. Windows: `.\scripts\dev.ps1 train-groups`.

`train_groups.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` keeps C at 0.1. Each group logistic uses C 0.1. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`. `elo_mov` is `baseline`.

The local files are:

- `reports/phase2_groups/comparison.md`
- `reports/phase2_groups/comparison.csv`
- `reports/phase2_groups/acceptance.csv`
- `reports/phase2_groups/calibration.csv`
- `reports/phase2_groups/market_information.csv`
- `reports/phase2_groups/group_diagnostics.csv`
- `reports/phase2_groups/oof_logistic_h2_full.csv`
- `reports/phase2_groups/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_groups/oof_group_mean_pkg_core_inj_l2.csv`
- `reports/phase2_groups/oof_group_mean_def_pkg_core_inj_l2.csv`
- `reports/phase2_groups/oof_elo_elo.csv`
- `reports/phase2_groups/oof_elo_mov_elo_mov.csv`

The home prior and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_groups`.

Read `reports/phase2_groups/comparison.md` after the run.

No model is promoted from this run.
