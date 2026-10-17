# Phase 2 H5: starting quarterback

Method note. Reproduce this comparison with `python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5`. It writes `reports/phase2_h5/comparison.md` and `reports/phase2_h5/comparison.csv`. Those files are local outputs. The result section records the local H5 run. The frozen copy is `docs/experiments/reports/phase2_h5_comparison.md`. Do not copy numbers from a test fixture. `make train-cv` (Windows: `.\scripts\dev.ps1 train-cv`) now runs H5b and writes `reports/phase2_h5b/`. See `docs/experiments/phase2_h5b_starter_coverage.md`. After the depth-chart reload in that note, this same `--hypothesis h5` command reads the fixed starter. The frozen snapshot is the original decision.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here.

H4 is rejected. The reference stays logistic `h2_full`. See `docs/experiments/phase2_h4_clean_epa.md`.

## Hypothesis

A pregame starting-quarterback rating predicts the home win better than logistic `h2_full` alone. FiveThirtyEight's QB-adjusted Elo treats the quarterback as the main information the market has that a team-only rating lacks. This test asks whether a leakage-free starter rating, shrunk toward replacement, moves home-win log loss by the phase 2 rule (at least 0.003, with the paired interval entirely below 0).

## Starter

The expected starter is the offense quarterback listed at `depth_team` 1 on that week's nflverse depth chart. The join is `season`, `week`, `game_type`, and `club_code` equal to `home_team` or `away_team`. `gsis_id` matches `passer_player_id`. Depth charts exist from 2001. 1999 and 2000 were not published.

`fct_expected_starter` does not read play-by-play. Who threw in the game is not the starter.

When the chart has no QB1, the starter is the previous game's primary passer for that franchise: the passer with the most dropbacks, and the lower id on a tie. The source is `depth_chart` or `previous_passer`. The change flag is 1 only when the chart and that previous passer both exist and differ, and 0 when they match. A missing chart cannot confirm a change, so the flag stays missing. Some weeks have no QB1 (2024 Jacksonville weeks 15 and 17, and 2001 week 1 Baltimore and Buffalo). Those rows use the fallback.

From 2025 the nflverse file is a timestamped snapshot and has no `week`. Ingest keeps, for each team-game, the latest snapshot with `dt` strictly before kickoff (`gameday` plus `gametime` in Eastern time). `pos_abb` QB and `pos_grp` `3WR 1TE` are the offense QB, and `pos_rank` 1 is `depth_team` 1. A Thursday game does not see a Friday scrape. CV folds are 2016-2023, so they do not use this mapping. 2025 is the held-out test. If that season is skipped, the starter id stays null and the feature code uses the previous-game primary passer.

## Rating

For the starter, sum dropbacks and EPA on plays with `gameday` strictly before the game. History follows the player across teams and seasons. Same-day games are excluded.

`weight = n / (n + 100)`. The rating is `weight * (epa / n) + (1 - weight) * 0`. One hundred prior dropbacks is half weight. Replacement EPA per dropback is 0. Both numbers are fixed. They are not tuned on this run. A known starter with no prior dropbacks is rated at replacement, not left missing. An unknown starter stays missing.

Team recent passing EPA is that franchise's EPA per dropback over the last `lookback_games` team-games strictly before the game, counting every passer. The gap is the QB rating minus that team rate. It is missing when either side is missing.

CPOE is summed on `fct_qb_dropbacks` (`cpoe_sum`, `cpoe_dropbacks`) and is not in the rating. EPA per dropback and CPOE are different units. A blend weight would be a second hypothesis.

## Feature set

`h5_qb` is `h2_full` plus:

- `home_qb_rating`
- `away_qb_rating`
- `home_qb_rating_minus_team_pass_epa`
- `away_qb_rating_minus_team_pass_epa`

Starter ids, sources, and the change flags are emitted and are not model inputs. A missing rating is filled with 0, the same replacement level.

The report also describes games where either change flag is 1: paired log loss of `h5_qb` versus logistic `h2_full` on that out-of-fold subset, plus counts of depth-chart starters and previous-passer fallbacks. That section is not an acceptance decision. It is skipped when the subset has fewer than two games or a single class.

## Warehouse commands

H5 adds `workspace.nfl_raw.depth_charts`. Games and play-by-play can stay. On a catalog that already has those two tables:

```bash
python -m src.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5
```

`make dbt-build-databricks` runs `dbt deps`, then `dbt build --target databricks`. The H5 command reads `fct_games`, `fct_team_game_epa`, `fct_expected_starter`, and `fct_qb_dropbacks` from `workspace.nfl_dev_marts`. Chain the commands with `&&` so training does not start after a failed ingest or build. `make train-cv` is H5b, not this hypothesis. The reload also rewrites every depth-chart season so `week` is one int type. See `docs/experiments/phase2_h5b_starter_coverage.md`.

A fresh clone loads every bronze table, including depth charts:

```bash
make ingest && make dbt-build-databricks && python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5
```

Windows PowerShell 5.1 has no `&&`. After games and play-by-play are already loaded:

```powershell
python -m src.ingest.load_raw_nflverse --dataset depth_charts
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 dbt-build-databricks }
if ($LASTEXITCODE -eq 0) { python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5 }
```

Before CV, each requested feature column has to exist and stay under 50% null. A failure names the feature set and says to run `make dbt-build-databricks`. If the depth-chart bronze table was never loaded, the warehouse read fails first. Load depth charts, then rebuild.

`dbt/profiles.yml` is gitignored. After a pull, compare it to `dbt/profiles.yml.example`. The warehouse target must be `databricks` and the catalog must be `workspace`. The recommended venv is Python 3.11. Python 3.14 triggers pydantic v1 warnings inside dbt.

## Result

Local rolling-origin run, CV 2016-2023, n=2191, reference logistic `h2_full` 0.6377. These rows are that run. They are not taken from a test fixture. The frozen copy is `docs/experiments/reports/phase2_h5_comparison.md`.

| feature_set | n | log_loss | auc | log_loss_diff | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h2_full | 2191 | 0.6377 | 0.6843 | 0.0000 | 0.0000 | 0.0000 | reference |
| h5_qb | 2191 | 0.6355 | 0.6869 | -0.0022 | -0.0042 | -0.0003 | rejected |

`h5_qb` improves log loss by 0.0022. The pre-registered rule needs a gain of at least 0.003 (`log_loss_diff` <= -0.003) and an interval entirely below 0. The interval does lie entirely below 0, so the sign is a positive signal, but 0.0022 misses the 0.003 bar. The threshold is not changed after seeing this result. The home prior, Elo, and the market were scored by the same command. They are baselines, not this hypothesis.

On 383 out-of-fold games with a QB change, `h5_qb` log loss is 0.6402 and the reference is 0.6428. The difference is -0.0026 (interval -0.0075 to 0.0023). That slice is descriptive. It is not an acceptance decision. The same report counted 10,249 depth-chart starter sides and 4,399 previous-passer fallback sides, about 30% fallback, and 972 games flagged as a QB change. Those coverage counts are the reason for the next starter-join check. The counts are in the frozen snapshot. The starter-join check is H5b. Its score is pending. See `docs/experiments/phase2_h5b_starter_coverage.md`.
