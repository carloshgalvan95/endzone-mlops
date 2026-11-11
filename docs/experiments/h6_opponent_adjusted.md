# H6 opponent-adjusted rating

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_h6/comparison.md` here only after the warehouse run, and only if I decide the hypothesis.

## Question

Does replacing raw EPA in logistic `h2_full` with opponent-adjusted EPA, and then adding an Elo difference that uses margin of victory, beat logistic `h2_full` on rolling-origin seasons 2016-2023?

Two feature sets, one comparison:

| Feature set | What changes |
| --- | --- |
| `h6_adj` | Every EPA input in `h2_full` comes from opponent-adjusted EPA. Win rate and rest stay. Lookback, shrinkage, and the blend stay on the frozen recipe. |
| `h6_adj_elo` | `h6_adj` plus `elo_mov_diff`. Pre-game home Elo minus pre-game away Elo. Home advantage is not in this column. |

The reference row is logistic `h2_full`, lookback 10, C 1.0, max_iter 1000, seed 42. The home prior, the baseline Elo probability, and the market are scored with the same code as the other phase 2 runs. The baseline Elo probability does not use a margin-of-victory multiplier. `elo_mov_diff` does.

## Adjustment

For each gameday, ridge regression fits team-games with `gameday` strictly before that date. Same-gameday games share the fit and do not enter it. The target is offensive EPA per play. The design is an offense dummy, a defense dummy, and a home sign (+1 at home, -1 away). There is no neutral-site flag, so venue is home or away. Ridge uses penalty 1.0, an unpenalized intercept, and the cholesky solver. Teams are sorted so the column order is stable. I am not using the play-level penalty grid of 75 to 325. Each row here is already a game mean, and that penalty would crush the coefficients. I am not using `RidgeCV`. Choosing the penalty on the same window would be a search, and this penalty is locked before the predictive fit.

A window with fewer than 4 team-games or fewer than 2 teams leaves the adjusted values missing. An opponent missing from the fit uses coefficient 0.

Adjusted offense is offensive EPA minus the opponent defense coefficient minus the home coefficient times the home sign. Adjusted EPA allowed is defensive EPA allowed minus the opponent offense coefficient minus the home coefficient times the sign from the opponent's side. Those two series then go through the same recent, season, and cross-season regression windows as raw EPA in `h2_full`.

## Elo

`elo_mov_diff` uses K 20, home advantage 65, mean rating 1505, and a one-third season reversion, on the scale of 400. When both scores are present, K is multiplied by the FiveThirtyEight margin-of-victory factor: `ln(max(|margin|, 1) + 1) * 2.2 / denominator`. The denominator is 1 for a tie. Otherwise it is the winner's pre-game rating gap, including home advantage, times 0.001, plus 2.2. A missing score keeps the plain K step. The multiplier is not clipped. The column stores home rating minus away rating before the game.

## Gate

Accept `h6_adj` or `h6_adj_elo` only when `log_loss_diff` <= -0.003 and the paired 95% bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. That is ADR-005. The market row is labeled `benchmark`. It is not a candidate and it is not labeled `accepted`.

The locked logistic `h2_full` count in `docs/experiments/reference_metrics.json` is n=2191. The command checks that each challenger OOF file matches `oof_logistic_h2_full.csv` on `game_id` order, `y_true`, and seasons inside 2016-2023. I do not drop games to force 2191. If this run's count differs, I will say so and I will not call it that table. Market `n` can be smaller.

## Calibration and market information

On the out-of-fold games, the report has a reliability table by equal-count deciles, ECE, and the bin reliability, resolution, and uncertainty, for the logistic rows, Elo, and the market. `brier` in that table is the score of the game probabilities. The bin components are not a substitute for that score when probabilities vary inside a bin. This block is informative. It is not a gate.

The report also fits a logistic regression of the result on logit(market) and logit(model), with an intercept, C 1.0, and a bootstrap interval. That regression is descriptive. It is not a blend. No blend is written to an OOF file, the site, or the registry.

## Command

From the repo root, after the marts exist. Git Bash on Windows:

```bash
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_h6.py
```

The same environment variables, then:

```bash
make train-h6
```

Windows PowerShell, with the virtualenv active and the same variables set:

```powershell
.\scripts\dev.ps1 train-h6
```

`train_h6.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not touch the model registry or the `prod` alias. It does not score 2024 or 2025.

Local outputs, gitignored:

- `reports/phase2_h6/comparison.md`
- `reports/phase2_h6/comparison.csv`
- `reports/phase2_h6/acceptance.csv`
- `reports/phase2_h6/calibration.csv`
- `reports/phase2_h6/market_information.csv`
- `reports/phase2_h6/oof_logistic_h2_full.csv`
- `reports/phase2_h6/oof_logistic_h6_adj.csv`
- `reports/phase2_h6/oof_logistic_h6_adj_elo.csv`

The home prior, Elo, and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_h6`. Calibration and the market-information table are logged as artifacts with the comparison.

The single-split trainer refuses `h6_adj` and `h6_adj_elo` before it reads the warehouse. That split scores 2024 and 2025.

If a requested column is missing or at least half null, the existing check names the feature set and points at `make dbt-build-databricks`. That rebuild is a separate command.

Read `reports/phase2_h6/comparison.md` after the run.
