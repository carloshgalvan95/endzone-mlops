# pkg_core

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_pkg_core/comparison.md` here only after the warehouse run, and only if I decide the candidate.

## Question

Do the three short Phase 2 signals, stacked as one logistic feature set, beat logistic `h2_full` on rolling-origin seasons 2016-2023?

Each piece is already scored and stays exactly as defined. I am not adding a parameter, dropping a column, or moving the 0.003 threshold after I see this result.

Recorded warehouse results, not this run. H5b is in `docs/experiments/phase2_h5b_starter_coverage.md`. Recency and H6 are recorded on issues 83 and 59 and are not frozen snapshots in this repo:

| Piece | Recorded difference versus logistic `h2_full` |
| --- | --- |
| `h5_qb` (H5b) | -0.0021, interval -0.0039 to -0.0002 |
| `recency_margin` | -0.0019, interval -0.0032 to -0.0003 |
| `h6_adj_elo` | -0.0016, interval -0.0049 to 0.0017 |

## Candidate

`pkg_core` is the ordered union of three existing column lists:

1. Every `h6_adj_elo` column, in that set's order.
2. The four `h5_qb` rating columns: `home_qb_rating`, `away_qb_rating`, `home_qb_rating_minus_team_pass_epa`, `away_qb_rating_minus_team_pass_epa`.
3. Every `recency_margin` column that is not already in the list. That set is `h2_full` plus `home_recent_point_diff` and `away_recent_point_diff`. Win rate and rest are already on `h6_adj_elo`, so they are not repeated. The raw EPA columns stay. I do not drop them in favor of the adjusted names.

The locked names, in model order:

`home_recent_adj_offensive_epa_per_play`, `home_recent_adj_defensive_epa_per_play`, `away_recent_adj_offensive_epa_per_play`, `away_recent_adj_defensive_epa_per_play`, `home_recent_win_rate`, `away_recent_win_rate`, `home_rest_days`, `away_rest_days`, `home_season_adj_offensive_epa_per_play`, `away_season_adj_offensive_epa_per_play`, `home_season_adj_defensive_epa_per_play`, `away_season_adj_defensive_epa_per_play`, `home_regressed_recent_adj_offensive_epa_per_play`, `home_regressed_recent_adj_defensive_epa_per_play`, `away_regressed_recent_adj_offensive_epa_per_play`, `away_regressed_recent_adj_defensive_epa_per_play`, `home_regressed_season_adj_offensive_epa_per_play`, `away_regressed_season_adj_offensive_epa_per_play`, `home_regressed_season_adj_defensive_epa_per_play`, `away_regressed_season_adj_defensive_epa_per_play`, `elo_mov_diff`, `home_qb_rating`, `away_qb_rating`, `home_qb_rating_minus_team_pass_epa`, `away_qb_rating_minus_team_pass_epa`, `home_recent_offensive_epa_per_play`, `home_recent_defensive_epa_per_play`, `away_recent_offensive_epa_per_play`, `away_recent_defensive_epa_per_play`, `home_season_offensive_epa_per_play`, `away_season_offensive_epa_per_play`, `home_season_defensive_epa_per_play`, `away_season_defensive_epa_per_play`, `home_regressed_recent_offensive_epa_per_play`, `home_regressed_recent_defensive_epa_per_play`, `away_regressed_recent_offensive_epa_per_play`, `away_regressed_recent_defensive_epa_per_play`, `home_regressed_season_offensive_epa_per_play`, `away_regressed_season_offensive_epa_per_play`, `home_regressed_season_defensive_epa_per_play`, `away_regressed_season_defensive_epa_per_play`, `home_recent_point_diff`, `away_recent_point_diff`.

Lookback stays 10. Logistic C stays 1.0, max_iter 1000, seed 42. QB shrinkage stays 100 prior dropbacks and replacement EPA per dropback stays 0. Opponent adjustment, the Elo margin multiplier, and the 5-game point differential stay on the recipes already scored. Injuries are out of scope.

The same run also scores logistic `h5_qb`, `recency_margin`, and `h6_adj_elo`. Those rows are context. Each one is accepted or rejected by the same numeric rule. They are not a second search, and I will not adopt one of them from this run. `pkg_core` is the only preregistered candidate.

The reference row is logistic `h2_full`. The home prior, the baseline Elo probability, and the market are scored with the same code as the other phase 2 runs.

## Gate

Accept `pkg_core` only when `log_loss_diff` <= -0.003 and the paired 95% bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. That is ADR-005. The market row is labeled `benchmark`. It is not a candidate and it is not labeled `accepted`.

The locked logistic `h2_full` count in `docs/experiments/reference_metrics.json` is n=2191. The command checks that each challenger OOF file matches `oof_logistic_h2_full.csv` on `game_id` order, `y_true`, and seasons inside 2016-2023. I do not drop games to force 2191. If this run's count differs, I will say so and I will not call it that table. Market `n` can be smaller.

Nothing from this run is registered or promoted. 2024 and 2025 are not scored.

## Calibration and market information

On the out-of-fold games, the report has a reliability table by equal-count deciles, ECE, and the bin reliability, resolution, and uncertainty, for the logistic rows, Elo, and the market. `brier` in that table is the score of the game probabilities. The bin components are not a substitute for that score when probabilities vary inside a bin. This block is informative. It is not a gate.

The report also fits a logistic regression of the result on logit(market) and logit(model), with an intercept, C 1.0, and a bootstrap interval. That regression is descriptive. It is not a blend. No blend is written to an OOF file, the site, or the registry.

## Command

From the repo root, after the marts exist. Git Bash on Windows:

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_pkg_core.py
```

The same environment variables, then:

```bash
make train-pkg-core
```

Windows PowerShell, with the virtualenv active and the same variables set:

```powershell
.\scripts\dev.ps1 train-pkg-core
```

`train_pkg_core.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not touch the model registry or the `prod` alias. It does not score 2024 or 2025. It does load the starter tables, because `pkg_core` and `h5_qb` use the existing QB rating columns.

Local outputs, gitignored:

- `reports/phase2_pkg_core/comparison.md`
- `reports/phase2_pkg_core/comparison.csv`
- `reports/phase2_pkg_core/acceptance.csv`
- `reports/phase2_pkg_core/calibration.csv`
- `reports/phase2_pkg_core/market_information.csv`
- `reports/phase2_pkg_core/oof_logistic_h2_full.csv`
- `reports/phase2_pkg_core/oof_logistic_pkg_core.csv`
- `reports/phase2_pkg_core/oof_logistic_h5_qb.csv`
- `reports/phase2_pkg_core/oof_logistic_recency_margin.csv`
- `reports/phase2_pkg_core/oof_logistic_h6_adj_elo.csv`

The home prior, Elo, and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_pkg_core`. Calibration and the market-information table are logged as artifacts with the comparison.

The single-split trainer refuses `pkg_core` before it reads the warehouse. That split scores 2024 and 2025. `h5_qb` stays available on that split. `recency_margin` and `h6_adj_elo` stay refused there, as they already were.

If a requested column is missing or at least half null, the existing check names the feature set and points at `make dbt-build-databricks`. That rebuild is a separate command.

Read `reports/phase2_pkg_core/comparison.md` after the run.
