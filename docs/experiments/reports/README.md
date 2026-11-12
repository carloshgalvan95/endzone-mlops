# Frozen experiment snapshots

`reports/` at the repo root is local scratch. It is gitignored. `make audit-features`, `make train-cv`, `make check-epa-filter`, and the other `--cv` commands write there and can be rerun. `make train-cv` writes `reports/phase2_h5b/`. H5 is decided. The frozen copy is `phase2_h5_comparison.md`. H5b is rejected. The decision numbers are in `docs/experiments/phase2_h5b_starter_coverage.md`. This folder has no frozen H5b comparison file. Recency is preregistered in `docs/experiments/recency_streak.md` and has no frozen snapshot, because that comparison has not been run. H6 is preregistered in `docs/experiments/h6_opponent_adjusted.md` and has no frozen snapshot, because that comparison has not been run. `pkg_core` is preregistered in `docs/experiments/pkg_core.md` and has no frozen snapshot, because that comparison has not been run. `pkg_core_inj` is preregistered in `docs/experiments/pkg_core_inj.md` and has no frozen snapshot, because that comparison has not been run. Ensembles are preregistered in `docs/experiments/ensembles.md` and have no frozen snapshot, because that comparison has not been run. Ensembles round 2 is preregistered in `docs/experiments/ensembles_r2.md` and has no frozen snapshot, because that comparison has not been run.

`docs/experiments/reports/` is different. Each file here is a frozen copy of one local report, kept as evidence for a decision. Copy a report here only when that hypothesis is decided. Do not edit the numbers afterward. Replace the snapshot only when the hypothesis is rerun and the decision is recorded again.

| Snapshot | Experiment note | What it freezes |
| --- | --- | --- |
| [phase0_feature_audit.md](phase0_feature_audit.md) | [phase0_input_audit.md](../phase0_input_audit.md) | Phase 0 input audit |
| [phase1_comparison.md](phase1_comparison.md) | [phase1_baselines.md](../phase1_baselines.md) | Phase 1 baselines |
| [phase2_h1_comparison.md](phase2_h1_comparison.md) | [phase2_h1_diff_features.md](../phase2_h1_diff_features.md) | H1 differentials, rejected |
| [phase2_h2_comparison.md](phase2_h2_comparison.md) | [phase2_h2_season_def_regression.md](../phase2_h2_season_def_regression.md) | H2 season defense and regression, `h2_full` accepted |
| [phase2_h3_comparison.md](phase2_h3_comparison.md) | [phase2_h3_pass_rush_epa.md](../phase2_h3_pass_rush_epa.md) | H3 pass and rush EPA, rejected |
| [phase2_ridge_comparison.md](phase2_ridge_comparison.md) | [phase2_ridge_reference.md](../phase2_ridge_reference.md) | Fold-tuned L2 versus logistic `h2_full` |
| [phase2_h4_comparison.md](phase2_h4_comparison.md) | [phase2_h4_clean_epa.md](../phase2_h4_clean_epa.md) | H4 clean EPA, rejected |
| [phase2_h5_comparison.md](phase2_h5_comparison.md) | [phase2_h5_starting_qb.md](../phase2_h5_starting_qb.md) | H5 starting QB, rejected, positive signal |
| [epa_filter_check.md](epa_filter_check.md) | [phase0_input_audit.md](../phase0_input_audit.md) | `play = 1` play types, plus kneel and spike counts |
| [sprint2_cv_report.json](sprint2_cv_report.json) | [ADR-007](../../adr/ADR-007-score-regression.md) | Margin and total, both accepted |

The copies are verbatim local reports. A Databricks hostname, if one were present, is stripped before commit. These files do not contain one.

Sprint 2 is frozen in [sprint2_cv_report.json](sprint2_cv_report.json). `make train-s2` still writes a local copy at `reports/sprint2/cv_report.json`. Do not hand-edit the numbers. ADR-007. Both targets accepted, n=2191, seasons 2016-2023. Margin MAE 10.155811 versus Elo/25 10.412221 (8 of 8 seasons). Total MAE 10.925618 versus the league mean 11.130543 (5 of 8 seasons). Market margin MAE is 9.800548 and market total MAE is 10.526928. Both are lower than the model, and neither is a gate.
