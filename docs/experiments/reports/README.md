# Frozen experiment snapshots

`reports/` at the repo root is local scratch. It is gitignored. `make audit-features`, `make train-cv`, `make check-epa-filter`, and the other `--cv` commands write there and can be rerun. `make train-cv` writes `reports/phase2_h5/`. H5 is decided. The frozen copy is `phase2_h5_comparison.md`.

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

The copies are verbatim local reports. A Databricks hostname, if one were present, is stripped before commit. These files do not contain one.
