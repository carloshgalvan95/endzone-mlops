# Injury coverage and home-field context

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_closing/comparison.md` here only after the warehouse run, and only if I decide a candidate.

## Goal

Round 7 adds two logistic candidates on the defender. One replaces the four Out or Doubtful counts with a weighted absence and a coverage indicator. The other keeps those counts and adds three pregame home-field columns. There is no search. Preseason win totals are not used.

## Reference

Logistic `pkg_core_inj_l2`, C fixed at 0.1, 47 columns. Rolling-origin CV, seasons 2016 through 2023. The published defender count on that window is 2191 games. `h2_full`, Elo, and `elo_mov` stay unchanged.

`h2_full`, Elo, and `elo_mov` are scored in the same run. They are context. They are not candidates. `elo_mov` is the home win probability from `elo_mov_diff` plus home advantage 65 on a 400 scale. Its decision is `baseline`.

The market is a benchmark. It is not a candidate.

## Candidates

Two candidates. Logistic `pkg_core_inj_l2` stays the defender. No search.

1. `pkg_core_inj_l2_inj_cov_q`, 48 columns, logistic C fixed at 0.1. The 43 `pkg_core` columns, then four weighted starter-absence differences, then `inj_report_covered`. The four Out or Doubtful columns are not in this set, and their formulas are unchanged.
   - `diff_ol_starters_absent_w`, `diff_skill_starters_absent_w`, `diff_front7_starters_absent_w`, and `diff_db_starters_absent_w` are home minus away. Out weighs 1.0. Doubtful weighs 1.0. Questionable weighs 0.27. The weight is the last report with `date_modified` strictly before kickoff, and only a depth-1 starter counts. QB is excluded.
   - `inj_report_covered` is 1.0 when the season is 2009 or later and both injury tables are provided. It is 0.0 before 2009, and 0.0 when the tables are omitted. It is never missing.
2. `pkg_core_inj_l2_hfa_ctx`, 50 columns, logistic C fixed at 0.1. The 47 defender columns, then three indicators. Each indicator is 0.0 or 1.0 and is never missing.
   - `is_divisional_game` is 1.0 when `home_team_franchise` and `away_team_franchise` share a division in that season. 1999 through 2001 uses the six-division alignment. 2002 and later uses the eight-division alignment. OAK, SD, and STL map onto LV, LAC, and LA. An unknown franchise is 0.0.
   - `away_surface_mismatch` is 1.0 when the current surface category differs from the away team's most recent prior home surface. Home means `location` is Home, the surface is a known grass or turf value, and `gameday` is strictly earlier. Grass is `grass` or `dessograss`. Turf is `a_turf`, `astroplay`, `astroturf`, `fieldturf`, `matrixturf`, or `sportturf`. A blank or unknown current surface, or no prior home surface, is 0.0.
   - `west_coast_early_kickoff` is 1.0 when `location` is Home, the away franchise is one of ARI, DEN, LA, LAC, LV, SEA, or SF, the home franchise is not in that set, and `gametime` in Eastern time is strictly before 14:00. A neutral site, a blank clock, or a malformed clock is 0.0.

C, the weights, the division maps, the surface categories, the western set, and the kickoff cutoff are fixed. There is no C search. Nothing from 2024, 2025, or later enters any fold.

## Acceptance

ADR-005, unchanged. Each candidate is judged alone against logistic `pkg_core_inj_l2`. Accept it only when the log loss difference is at most -0.003 and the 95 percent paired bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. If both pass, the one with the lower log loss is the comparison's best row. That does not register or promote it.

Per-season results, calibration, ECE, Brier, ROC AUC, and the market-information regression are descriptive. They are not gates.

Do not score 2024, 2025, or the held-out split. The holdout command refuses the new feature sets. No tuning after the results. No threshold changes. No extra candidates. Nothing is registered or promoted.

## Reproduce

No dbt build is needed before training. The columns are computed from the games mart and the injury tables that are already loaded.

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_closing.py
```

`make train-closing` is the train step alone. Windows: `.\scripts\dev.ps1 train-closing`.

`train_closing.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` and both candidates keep C at 0.1. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`. `elo_mov` is `baseline`.

The local files are:

- `reports/phase2_closing/comparison.md`
- `reports/phase2_closing/comparison.json`
- `reports/phase2_closing/comparison.csv`
- `reports/phase2_closing/acceptance.csv`
- `reports/phase2_closing/calibration.csv`
- `reports/phase2_closing/market_information.csv`
- `reports/phase2_closing/oof_logistic_h2_full.csv`
- `reports/phase2_closing/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_closing/oof_logistic_pkg_core_inj_l2_inj_cov_q.csv`
- `reports/phase2_closing/oof_logistic_pkg_core_inj_l2_hfa_ctx.csv`
- `reports/phase2_closing/oof_elo_elo.csv`
- `reports/phase2_closing/oof_elo_mov_elo_mov.csv`

The home prior and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_closing`.

Read `reports/phase2_closing/comparison.md` after the run.

No model is promoted from this run.
