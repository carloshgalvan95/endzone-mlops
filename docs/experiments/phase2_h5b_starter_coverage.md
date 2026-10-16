# Phase 2 H5b: starter coverage

Method note. The warehouse score is pending. Do not copy a log loss into this file until `reports/phase2_h5b/comparison.md` exists from a local run.

`make train-cv` (Windows: `.\scripts\dev.ps1 train-cv`) is:

```bash
python -m src.train.train --cv --phase 2 --hypothesis h5b --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h5b
```

It writes `reports/phase2_h5b/comparison.md` and `reports/phase2_h5b/comparison.csv`. Those files are local outputs. Runs go to MLflow experiment `nfl_pregame/phase2_h5b` with tags `phase=2` and `hypothesis=h5b-starter-coverage`. The parent run name is `phase2-h5b-<timestamp>`.

H5 stays available as `--hypothesis h5` and still writes `reports/phase2_h5/`. That command scores the same `h5_qb` columns. After the depth-chart reload and the mart rebuild below, it reads the fixed starter too. The original H5 decision stays in `docs/experiments/reports/phase2_h5_comparison.md`. This note does not replace that snapshot.

No model is promoted from a phase 2 run. 2024 stays the final validation. 2025 is not scored here. The acceptance rule is unchanged: pooled `log_loss_diff` <= -0.003 and the 95% paired bootstrap interval entirely below 0 (1000 resamples, seed 42). The reference stays logistic `h2_full`.

## Hypothesis

H5b is H5 with a fixed depth-chart starter and no other change. The feature set is still `h5_qb`: `h2_full` plus `home_qb_rating`, `away_qb_rating`, `home_qb_rating_minus_team_pass_epa`, and `away_qb_rating_minus_team_pass_epa`. Shrinkage stays `n / (n + 100)`. Replacement EPA per dropback stays 0. CPOE stays stored and stays out of the rating. The QB-change flag stays 1 only when the chart starter and the previous game's primary passer both exist and differ. Week 1 still compares with the prior season's passer. That flag is descriptive and is not an acceptance decision.

## Why about 30% of sides missed a chart QB1

The H5 report counted 10,249 depth-chart starter sides and 4,399 previous-passer fallbacks, and flagged 972 games as a QB change. Those counts are the warehouse feature frame. The diagnosis below uses the nflverse release downloaded on 2026-10-16 (`depth_charts_2001.parquet` through `depth_charts_2025.parquet`, plus `games.parquet`). It does not use a warehouse.

### Mixed week types dropped six seasons

`read_files` on `workspace.nfl_raw.depth_charts` sees every season file in one folder. Pandas writes `week` as float64 when the file contains a null (the Super Bowl bye) and as int32 when every week is filled. The int32 seasons are 2002, 2003, 2004, 2005, 2006, and 2014. The warehouse schema merge drops those seasons. The float-week files from 2001-2024 contribute 9,679 chart sides. The 2025 snapshot maps onto 570 of 570 team-games. 9,679 + 570 = 10,249, the H5 depth-chart count.

A local pyarrow dataset does not reproduce the drop. It upcasts `week` to double. CI fixtures will not catch this. The warehouse reader is the one that fails. dbt cannot put those weeks back. The files on the volume have to be rewritten as one nullable int32 `week`, and `read_files` now passes `schemaHints` of `season int, week int, depth_team string, club_code string`.

1999-2025 completed games are 7,276 (14,552 sides). Chart sides 10,249 and fallback sides 4,303 partition that set. The report's 4,399 fallbacks are 96 sides more, which is 48 extra 2026 games in the feature frame. 4,303 + 96 = 4,399.

### What is not the 30%

The current release already uses the schedule's historical abbreviations. There is no ARZ, BLT, CLV, HST, or JAC in these files. LA stays LA (not LAR). OAK runs through 2019 and LV starts in 2020. SD runs through 2016 and LAC starts in 2017. STL runs through 2015. A direct join on club code, season, week, and game type hits 12,855 of 12,946 sides in 2001-2024 (99.3%). The true misses are 91 sides. 2005 has no postseason chart (22 sides, regular season weeks 1-17 only). 2011 misses 2 postseason sides and 2019 misses 1. There is no `POST` label and no systematic week offset. An extra regular-season week 18 (before 2021) or 19 (from 2021) is unused, not a missed game. Shifting the chart week by one would raise agreement and would also read the postgame chart, so that shift is not used.

`depth_team` on the 2001-2024 files is already the string `1`. The 2025 `pos_rank` is an int. That type is not the coverage hole. Ingest and the mart still coerce `1`, `1.0`, and `'1'` to the same rank.

`depth_chart_club_map` covers older scraps (ARZ, BLT, CLV, HST, JAC, and a relocated code used in the wrong years). Applying it to this release changes nothing, and applying it twice does not either. A code that already matches the schedule is not in the seed.

### Duplicate QB1 rows

Thirty team-weeks list two offense QB1 ids. `min(gsis_id)` keeps the lower id, which is often the wrong person. The mart now prefers `depth_position` QB, then a blank position, then any other position, then the lowest `gsis_id`. An exact `club_code` beats a row that only matched through the alias. Depth 2 is not promoted: among the 91 true misses, promoting depth 2 recovers 23 sides and attaches the wrong backup on 37.

Agreement with `games.home_qb_id` / `away_qb_id` on 2001-2024 chart sides moves from 0.8967 (1,328 disagreements out of 12,855) to 0.8968 (1,326). The identity fixes are real even though the rate barely moves:

- 2015 TB week 14: the lower id is Ryan Griffin, a tight end. The preferred row is Jameis Winston, who played.
- 2023 NYJ weeks 17 and 18: the lower id is Aaron Rodgers with a blank `depth_position`. The preferred row is Zach Wilson.
- 2005 CHI: both Rex Grossman and Kyle Orton have `depth_position` QB. The lowest id still wins. Who played is not used to break that tie.

### Why so many games were flagged as a QB change

Listed-starter changes on 1999-2025, using `games.home_qb_id` and `away_qb_id`, are 1,528 games out of 7,261 with a prior listed starter (1,674 sides, 11.5%). Week-1 change sides are 290. Cross-season change sides are 292. In-season change sides are 1,382 (1,291 games). On the 2016-2023 out-of-fold window the same listed-starter definition flags 476 games, 396 of them in-season. The report's 972 flags are below that turnover, not above it.

The missing int-week seasons contributed zero change flags, because a missing chart cannot confirm a change. Restoring them raises the listed-starter change count from 801 games to 1,193 games on 1999-2025. 2004 is 145 of 267 games (54%) because that chart is stale: Baltimore lists Anthony Wright against Kyle Boller, Cleveland lists Tim Couch against Jeff Garcia, Dallas lists Quincy Carter against Vinny Testaverde, the Giants list Kerry Collins against Kurt Warner, Oakland lists Rick Mirer against Rich Gannon, and Cincinnati lists Jon Kitna against Carson Palmer. Agreement with who played in 2004 is 0.631, and the played quarterback is often absent from the chart (180 of 197 disagreements). The file cannot be repaired from itself.

Chart lag is a separate miss. 2023 NYJ week 1 lists Aaron Rodgers and he played. Week 2 still lists Rodgers while Zach Wilson played. Week 3 has no QB1 (Wilson is only at depth 2, and depth 2 is not promoted). Week 4 lists Wilson. 2017 SF week 13 lists C.J. Beathard while Jimmy Garoppolo played. 2022 SF week 13 lists Jimmy Garoppolo and he played. The flag definition is not changed to hide week 1 or a prior-season passer.

On the current files, before the six dropped seasons are restored, chart-versus-previous-listed-starter changes are 1,123 games out of 6,447 (17%). Of those change sides, 247 are week 1 and 248 cross a season. Franchise mapping breaks none of them. Non-week-1 changes where the chart disagrees with who played: 594. Where the chart matches who played: 404. Week-1 changes where the chart matches who played: 240. Where it does not: 7.

The gap from 801 listed-starter change games to the report's 972 is the primary-passer rule. A backup who throws more dropbacks than the listed starter makes the next game look like a change even when the chart starter is unchanged. Out-of-fold 2016-2023 chart-change games on the current join are 306 of 2,191 (chart sides 4,367 of 4,382). Those seasons were not the int-week bug, so the out-of-fold change count stays 306 after the fix. Scored-season starter ids in 2016-2023 move only where the tie-break changes (2023 NYJ weeks 17 and 18). Training rows for 2002-2006 and 2014 do change, and those seasons sit inside every later fold's training set.

## Coverage by season

Change rate here is games where the chart QB differs from the previous listed starter (`games.home_qb_id` / `away_qb_id`). It is not the report's primary-passer flag. "Before" is the warehouse behavior: int-week seasons 2002-2006 and 2014 have no chart, and duplicate QB1 rows keep `min(gsis_id)`. "After" rewrites `week` as one int type and prefers a real QB `depth_position`. Seasons that already joined are unchanged except 2015, where the tie-break removes one change game.

| season | games | before chart | before fallback | before changes | after chart | after fallback | after changes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1999 | 259 | 0.000 | 1.000 | 0 | 0.000 | 1.000 | 0 |
| 2000 | 259 | 0.000 | 1.000 | 0 | 0.000 | 1.000 | 0 |
| 2001 | 259 | 0.992 | 0.008 | 35 | 0.992 | 0.008 | 35 |
| 2002 | 267 | 0.000 | 1.000 | 0 | 1.000 | 0.000 | 51 |
| 2003 | 267 | 0.000 | 1.000 | 0 | 1.000 | 0.000 | 62 |
| 2004 | 267 | 0.000 | 1.000 | 0 | 1.000 | 0.000 | 145 |
| 2005 | 267 | 0.000 | 1.000 | 0 | 0.955 | 0.045 | 53 |
| 2006 | 267 | 0.000 | 1.000 | 0 | 0.994 | 0.006 | 30 |
| 2007 | 267 | 0.998 | 0.002 | 73 | 0.998 | 0.002 | 73 |
| 2008 | 267 | 1.000 | 0.000 | 41 | 1.000 | 0.000 | 41 |
| 2009 | 267 | 0.996 | 0.004 | 40 | 0.996 | 0.004 | 40 |
| 2010 | 267 | 0.998 | 0.002 | 47 | 0.998 | 0.002 | 47 |
| 2011 | 267 | 0.972 | 0.028 | 38 | 0.972 | 0.028 | 38 |
| 2012 | 267 | 0.996 | 0.004 | 23 | 0.996 | 0.004 | 23 |
| 2013 | 267 | 0.974 | 0.026 | 31 | 0.974 | 0.026 | 31 |
| 2014 | 267 | 0.000 | 1.000 | 0 | 0.998 | 0.002 | 52 |
| 2015 | 267 | 0.994 | 0.006 | 39 | 0.994 | 0.006 | 38 |
| 2016 | 267 | 1.000 | 0.000 | 33 | 1.000 | 0.000 | 33 |
| 2017 | 267 | 1.000 | 0.000 | 31 | 1.000 | 0.000 | 31 |
| 2018 | 267 | 0.996 | 0.004 | 36 | 0.996 | 0.004 | 36 |
| 2019 | 267 | 0.996 | 0.004 | 39 | 0.996 | 0.004 | 39 |
| 2020 | 269 | 0.998 | 0.002 | 26 | 0.998 | 0.002 | 26 |
| 2021 | 285 | 0.996 | 0.004 | 47 | 0.996 | 0.004 | 47 |
| 2022 | 284 | 0.996 | 0.004 | 46 | 0.996 | 0.004 | 46 |
| 2023 | 285 | 0.989 | 0.011 | 48 | 0.989 | 0.011 | 48 |
| 2024 | 285 | 0.989 | 0.011 | 64 | 0.989 | 0.011 | 64 |
| 2025 | 285 | 1.000 | 0.000 | 64 | 1.000 | 0.000 | 64 |

1999-2025 totals. Before: 10,249 chart sides, 4,303 fallback sides, 801 change games. After: 13,425 chart sides, 1,127 fallback sides, 1,193 change games. The remaining fallback is 1999-2000 (no charts were published) plus the 91 true misses and the 2005 postseason.

Spot checks of who played, then `min(gsis_id)`, then the preferred `depth_position`:

- 2020 NE week 1: Cam Newton, Cam Newton, Cam Newton.
- 2008 GB weeks 1 and 2: Aaron Rodgers, Aaron Rodgers, Aaron Rodgers.
- 2016 DEN week 1: Trevor Siemian, Trevor Siemian, Trevor Siemian.
- 2023 NYJ week 1: Aaron Rodgers on all three. Week 2: Zach Wilson played, both pickers still list Aaron Rodgers. Week 3: no QB1. Week 4: Zach Wilson. Week 17: Trevor Siemian played, `min(gsis_id)` is Aaron Rodgers, the preferred row is Zach Wilson.
- 2015 TB week 14: Jameis Winston played, `min(gsis_id)` is Ryan Griffin, the preferred row is Jameis Winston.
- 2017 SF week 13: Jimmy Garoppolo played, both pickers list C.J. Beathard.
- 2022 SF week 13: Jimmy Garoppolo on all three.

## Warehouse commands

Games and play-by-play can stay. Depth charts have to be reloaded. A 2025-only load leaves the int32 week files for 2002-2006 and 2014 in the volume, and the bronze replace reads that whole folder.

On a catalog that already has games and play-by-play:

```bash
python -m src.ingest.load_raw_nflverse --dataset depth_charts && make dbt-build-databricks && make train-cv
```

`make dbt-build-databricks` runs `dbt deps`, then `dbt build --target databricks`. `make train-cv` then scores H5b. Chain the commands with `&&` so training does not start after a failed ingest or build.

A fresh clone:

```bash
make ingest && make dbt-build-databricks && make train-cv
```

Windows PowerShell 5.1 has no `&&`:

```powershell
python -m src.ingest.load_raw_nflverse --dataset depth_charts
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 dbt-build-databricks }
if ($LASTEXITCODE -eq 0) { .\scripts\dev.ps1 train-cv }
```

## Result

Pending. This environment did not score H5b against the warehouse. The acceptance rule is not changed in advance of that run.
