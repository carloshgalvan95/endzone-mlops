# Mean and stack of the defender and Elo

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_ensembles_r2/comparison.md` here only after the warehouse run, and only if I decide a candidate.

## Goal

Round 2 of the ensemble plan. Test whether combining the defender `pkg_core_inj_l2` with Elo beats the defender alone. Preregistered before any code or result. The round 1 booster is not a candidate.

## Defender

Logistic `pkg_core_inj_l2` (C 0.1), log loss 0.6330 on rolling-origin CV 2016-2023, n 2191. `h2_full`, Elo, and the market stay in the report for context. They are not candidates.

## Candidates (exactly two, fixed before any result)

1. `ens_mean_elo`: unweighted mean of probabilities from `pkg_core_inj_l2` and Elo. Elo keeps the current settings: K 20, home advantage 65, mean 1505, season reversion 1/3 (the preregistered 0.333), scale 400. Each probability is the one already scored for that outer fold. No weight is learned.
2. `stack_elo`: logistic regression on `logit(p_pkg_core_inj_l2)` and `logit(p_elo)` with an intercept. C is 1.0 and is not searched. Probabilities are clipped at 1e-6 before the logit. For each scored season S, the stacker is fit only on inner out-of-fold predictions from seasons strictly before S. For each inner season T from 2010 through S-1, the members are trained on seasons before T and predict T. A season already published by the outer fold is reused, because that fold trained on seasons before T. The stacker never sees season S or any later season.

## Acceptance (ADR-005, unchanged)

Log loss diff vs `pkg_core_inj_l2` of at most -0.003, and the 95% paired bootstrap CI upper end below 0 (1000 resamples, seed 42). Calibration, per-season results, stacker coefficients per fold, and the market-information regression are descriptive only. The market-information table names the model as well as the feature set, so two rows on `pkg_core_inj_l2` stay distinct.

## Rules

- Same outer folds as #95 and #98. 2024, 2025, and the #61 holdout are not scored.
- No tuning after seeing results, no threshold changes, no extra candidates added after the run.
- Nothing is registered or promoted from this experiment.
- Outputs under `reports/phase2_ensembles_r2/`, MLflow experiment `nfl_pregame/phase2_ensembles_r2`.

## Reproduce

On a catalog that already has the injury marts:

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_ensembles_r2.py
```

```bash
make train-ensembles-r2
```

```powershell
.\scripts\dev.ps1 train-ensembles-r2
```

`train_ensembles_r2.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` keeps C at 0.1. The stacker C stays 1.0. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`.

Outputs:

- `reports/phase2_ensembles_r2/comparison.md`
- `reports/phase2_ensembles_r2/comparison.csv`
- `reports/phase2_ensembles_r2/acceptance.csv`
- `reports/phase2_ensembles_r2/calibration.csv`
- `reports/phase2_ensembles_r2/market_information.csv`
- `reports/phase2_ensembles_r2/stacker_coefficients.csv`
- `reports/phase2_ensembles_r2/oof_logistic_h2_full.csv`
- `reports/phase2_ensembles_r2/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_ensembles_r2/oof_elo_elo.csv`
- `reports/phase2_ensembles_r2/oof_ens_mean_elo_pkg_core_inj_l2.csv`
- `reports/phase2_ensembles_r2/oof_stack_elo_pkg_core_inj_l2.csv`

The home prior and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_ensembles_r2`.

The single-split trainer refuses `pkg_core_inj_l2` before it reads the warehouse. That split scores 2024 and 2025. Both candidates use those columns, so this command is the only scoring path.

Read `reports/phase2_ensembles_r2/comparison.md` after the run.
