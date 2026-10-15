# Phase 2 H5: starting quarterback

Method note. Result pending. Reproduce this comparison with `make train-cv` (Windows: `.\scripts\dev.ps1 train-cv`), which is `python -m src.train.train --cv --phase 2 --hypothesis h5 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5`. It writes `reports/phase2_h5/comparison.md` and `reports/phase2_h5/comparison.csv`. Those files are local outputs. Do not copy numbers from a test fixture. There is no frozen snapshot until this hypothesis is decided.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here.

H4 is rejected. The reference stays logistic `h2_full`. See `docs/experiments/phase2_h4_clean_epa.md`.

## Hypothesis

A pregame starting-quarterback rating predicts the home win better than logistic `h2_full` alone. FiveThirtyEight's QB-adjusted Elo treats the quarterback as the main information the market has that a team-only rating lacks. This test asks whether a leakage-free starter rating, shrunk toward replacement, moves home-win log loss by the phase 2 rule (at least 0.003, with the paired interval entirely below 0).

## Starter

The expected starter is the offense quarterback listed at `depth_team` 1 on that week's nflverse depth chart. The join is `season`, `week`, `game_type`, and `club_code` equal to `home_team` or `away_team`. `gsis_id` matches `passer_player_id`. Depth charts exist from 2001. 1999 and 2000 were not published.

`fct_expected_starter` does not read play-by-play. Who threw in the game is not the starter.

When the chart has no QB1, the starter is the previous game's primary passer for that franchise: the passer with the most dropbacks, and the lower id on a tie. The source is `depth_chart` or `previous_passer`. The change flag is 1 only when the chart and that previous passer both exist and differ, and 0 when they match. A missing chart cannot confirm a change, so the flag stays missing. Some weeks have no QB1 (2024 Jacksonville weeks 15 and 17, and 2001 week 1 Baltimore and Buffalo). Those rows use the fallback.

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
python -m src.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && make train-cv
```

`make dbt-build-databricks` runs `dbt deps`, then `dbt build --target databricks`. `make train-cv` reads `fct_games`, `fct_team_game_epa`, `fct_expected_starter`, and `fct_qb_dropbacks` from `workspace.nfl_dev_marts`. Chain the commands with `&&` so training does not start after a failed ingest or build.

A fresh clone loads every bronze table, including depth charts:

```bash
make ingest && make dbt-build-databricks && make train-cv
```

Windows PowerShell 5.1 has no `&&`. After games and play-by-play are already loaded:

```powershell
python -m src.ingest.load_raw_nflverse --dataset depth_charts
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 dbt-build-databricks }
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 train-cv }
```

Before CV, each requested feature column has to exist and stay under 50% null. A failure names the feature set and says to run `make dbt-build-databricks`. If the depth-chart bronze table was never loaded, the warehouse read fails first. Load depth charts, then rebuild.

`dbt/profiles.yml` is gitignored. After a pull, compare it to `dbt/profiles.yml.example`. The warehouse target must be `databricks` and the catalog must be `workspace`. The recommended venv is Python 3.11. Python 3.14 triggers pydantic v1 warnings inside dbt.

## Result

Pending. Do not fill this section from a test fixture.
