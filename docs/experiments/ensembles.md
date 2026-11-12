# Gradient boosting and a mean ensemble

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_ensembles/comparison.md` here only after the warehouse run, and only if I decide a candidate.

## Goal

Test whether a non-linear model or an ensemble beats the current challenger `pkg_core_inj_l2` (accepted in #95). Preregistered before any code or result.

## Reference

`pkg_core_inj_l2` (logistic, C 0.1, pkg_core + 4 injury columns), log loss 0.6330 on rolling-origin CV 2016-2023, n 2191. `h2_full` stays in the report for context.

## Candidates (exactly two, fixed before any result)

1. `gbm_pkg_core_inj`: scikit-learn `HistGradientBoostingClassifier` on the same columns as `pkg_core_inj_l2`. Fixed hyperparameters, no search: `learning_rate=0.05`, `max_iter=300`, `max_leaf_nodes=15`, `min_samples_leaf=50`, `l2_regularization=1.0`, `random_state=42`, no early stopping. `max_features` stays 1.0, the library default, so a column is not dropped at random. Missing values stay missing. The booster does not impute them and does not scale them. The fit uses one thread so the histogram build stays deterministic.
2. `ens_mean`: unweighted mean of probabilities from `pkg_core_inj_l2` and `gbm_pkg_core_inj`, each trained per fold exactly as above. No learned weights and no stacking, so nothing is fit on out-of-fold predictions.

## Acceptance (ADR-005, unchanged)

Log loss diff vs `pkg_core_inj_l2` of at most -0.003, and the 95% paired bootstrap CI upper end below 0 (1000 resamples, seed 42). Calibration, per-season results and the market-information regression are descriptive only.

## Rules

- Same folds and training seasons as #95. 2024, 2025 and the #61 holdout are not scored.
- No tuning after seeing results, no threshold changes, no extra candidates added after the run.
- Nothing is registered or promoted from this experiment.
- Outputs under `reports/phase2_ensembles/`, MLflow experiment `nfl_pregame/phase2_ensembles`.

## Reproduce

On a catalog that already has the injury marts:

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_ensembles.py
```

```bash
make train-ensembles
```

```powershell
.\scripts\dev.ps1 train-ensembles
```

`train_ensembles.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` keeps C at 0.1. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`.

Outputs:

- `reports/phase2_ensembles/comparison.md`
- `reports/phase2_ensembles/comparison.csv`
- `reports/phase2_ensembles/acceptance.csv`
- `reports/phase2_ensembles/calibration.csv`
- `reports/phase2_ensembles/market_information.csv`
- `reports/phase2_ensembles/oof_logistic_h2_full.csv`
- `reports/phase2_ensembles/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_ensembles/oof_gbm_pkg_core_inj_pkg_core_inj_l2.csv`
- `reports/phase2_ensembles/oof_ens_mean_pkg_core_inj_l2.csv`

The home prior, Elo, and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_ensembles`.

The single-split trainer refuses `pkg_core_inj_l2` before it reads the warehouse. That split scores 2024 and 2025. Both candidates use those columns, so this command is the only scoring path.

Read `reports/phase2_ensembles/comparison.md` after the run.
