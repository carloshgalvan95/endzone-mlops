# pkg_core injury reports

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_pkg_core_inj/comparison.md` here only after the warehouse run, and only if I decide a candidate.

Pre-results bug fix: a game with no injury rows does not need a kickoff, and a game whose kickoff cannot be built stays at 0. The 2016-2023 rule is unchanged.

The question is whether four pre-game starter-absence counts improve logistic `pkg_core` on the same rolling-origin window as ADR-005. The reference is logistic `h2_full`. `pkg_core` is scored in the same run for context. It is not a second candidate.

## Coverage gate

I checked the public nflverse injury releases for seasons 2016-2023 against `games.parquet`. The window is regular season plus postseason (`REG`, `WC`, `DIV`, `CON`, `SB`). There are 2,191 games and 4,382 team-games. `gameday` and `gametime` are present on every game. No team-week has more than one game, so season, week, game type, and team identify the kickoff.

Kickoff is `gameday` plus `gametime` in `America/New_York`, converted to UTC. `date_modified` is already UTC. A row is pre-game only when `date_modified` is non-null and strictly earlier than that kickoff. Week number is not a clock. A Friday update is after a Thursday kickoff in the same week.

| Check | Count | Share |
| --- | ---: | ---: |
| Team-games with any injury row | 4,361 / 4,382 | 99.52% |
| Team-games with a row strictly before kickoff | 4,360 / 4,382 | 99.50% |
| Matched report rows strictly before kickoff | 43,233 / 43,256 | 99.95% |

Both team-game shares are above 95%, so the gate passes. The 17 unmatched report rows are 2022 week 17 BUF (11) and CIN (6), the cancelled Bills-Bengals game, which is not in the 2,191-game slate.

By season, team-games with any row: 2016 534/534, 2017 533/534, 2018 533/534, 2019 534/534, 2020 537/538, 2021 570/570, 2022 565/568, 2023 555/570. By game type: REG 4,184/4,190, WC 80/80, DIV 56/64, CON 27/32, SB 14/16. The pooled share is the gate. Postseason coverage is thinner and still inside that pool.

The 21 team-games with no injury row are 2017 week 1 LA, 2018 week 20 LA, 2020 week 1 MIN, 2022 weeks 2 and 12 JAX, 2022 week 7 CHI, 2023 week 5 ATL, and the 2023 divisional, conference, and Super Bowl team-games (BAL, BUF, DET, GB, HOU, KC, SF, TB in the rounds they played). A missing report is a count of 0, not a null, once both marts are present.

`date_modified` is non-null on all 43,273 rows in 2016-2023. Twenty-three matched rows are on or after kickoff and are excluded:

- 2018 week 1 GB, Trevor Davis, WR, Doubtful, modified 2018-09-11 09:54:45Z, kickoff 2018-09-10 00:20Z.
- 2018 week 4 SF, Antone Exum, S, Questionable, modified 2018-09-30 20:42:59Z, kickoff 2018-09-30 20:25Z.
- 2019 week 13 ATL, five rows (McKinley, Hooper, Jones, Stocker, Neasman), modified 2019-11-29 13:25Z to 13:28Z, kickoff 2019-11-29 01:20Z (Thursday).
- 2020 week 6 KC, thirteen rows, modified 2020-10-21 about 11:15Z to 11:19Z, kickoff 2020-10-19 21:00Z (Monday).
- 2021 week 1 TB, Gronkowski, McLendon, and Pierre-Paul, modified 2021-09-10 09:27:20Z, kickoff 2021-09-10 00:20Z (Thursday).

Among matched rows, the modal calendar lag in Eastern time is two days before gameday (37,596 rows). The median pre-kickoff row is about 53.4 hours before kickoff. `report_status` is null on 22,077 of 43,273 rows (51.0%). The rest are Questionable 12,079, Out 7,759, and Doubtful 1,358. Of the 21,194 rows with a status, 21,185 are before kickoff. 4,252 team-games have a status row before kickoff. 9,113 Out or Doubtful player-rows are before kickoff, on 3,661 team-games.

Outside the gate, 2009 has `date_modified` on 17 of 4,821 rows. 2010 is about 1.4% null. 2011-2024 are complete on the files I downloaded. The 2025 and 2026 releases have no `date_modified` column. Ingest stores that column null, and the feature drops a null clock. 2024 has two player-weeks with two rows and different timestamps. The mart keeps both rows. Collapsing to the latest timestamp in SQL would hide a pre-kickoff status when a later update falls after kickoff.

## Columns

Four columns, fixed. For each group, the value is the home count minus the away count of that team's depth-chart starters listed Out or Doubtful. Positive means the home side has more listed absences.

| Column | Depth-chart `position` |
| --- | --- |
| `diff_ol_starters_out_or_doubtful` | C, G, T |
| `diff_skill_starters_out_or_doubtful` | RB, WR, TE |
| `diff_front_seven_starters_out_or_doubtful` | DE, DT, NT, LB, ILB, OLB, MLB |
| `diff_db_starters_out_or_doubtful` | CB, FS, SS, DB, S |

QB is excluded. `h5_qb` already covers the starter. FB, K, P, and LS are not in a group. The group comes from the depth chart, not the injury-report position. Depth 1 is the starter (`depth_team` 1, `1`, or `1.0`). A duplicate player inside a group counts once.

The report is the last row for that player with `date_modified` strictly before that team-game's kickoff. Out and Doubtful count. Questionable does not. A null timestamp does not count. The same timestamp keeps Out over Doubtful over any other status.

`pkg_core_inj` is `pkg_core` plus these four columns. Logistic settings match `pkg_core`: L2, `StandardScaler` after imputation, C 1.0, `max_iter` 1000, solver `lbfgs`, seed 42. `pkg_core_inj_l2` uses the same columns and the same pipeline. The only change is C fixed at 0.1. That is not the inner-fold C search used by `logistic_l2`.

I will not add a column, drop a column, or change C after I see the log loss.

## Decision

Accept a candidate only when `log_loss_diff` <= -0.003 and the paired 95% bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42. That is ADR-005. Each injury row is judged on its own against logistic `h2_full`. The market row is labeled `benchmark`. It is not a candidate and it is not labeled `accepted`.

Calibration and the market-information regression are descriptive. They are not acceptance decisions. No blend is published.

2024 and 2025 are not scored. Nothing is registered or promoted.

## Reproduce

On a catalog that already has games, play-by-play, and depth charts:

```bash
python -m endzone.ingest.load_raw_nflverse --dataset injuries && make dbt-build-databricks && PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_pkg_core_inj.py
```

```bash
make train-pkg-core-inj
```

```powershell
.\scripts\dev.ps1 train-pkg-core-inj
```

`train_pkg_core_inj.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. It loads `fct_depth_chart_starters` and `fct_injury_reports` because the four columns are computed at train time from those marts. The SQL marts keep every report row. The kickoff filter is in `endzone/features/injuries.py`.

Outputs:

- `reports/phase2_pkg_core_inj/comparison.md`
- `reports/phase2_pkg_core_inj/comparison.csv`
- `reports/phase2_pkg_core_inj/acceptance.csv`
- `reports/phase2_pkg_core_inj/calibration.csv`
- `reports/phase2_pkg_core_inj/market_information.csv`
- `reports/phase2_pkg_core_inj/oof_logistic_h2_full.csv`
- `reports/phase2_pkg_core_inj/oof_logistic_pkg_core.csv`
- `reports/phase2_pkg_core_inj/oof_logistic_pkg_core_inj.csv`
- `reports/phase2_pkg_core_inj/oof_logistic_pkg_core_inj_l2.csv`

The home prior, Elo, and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_pkg_core_inj`.

The single-split trainer refuses `pkg_core_inj` and `pkg_core_inj_l2` before it reads the warehouse. That split scores 2024 and 2025.

Read `reports/phase2_pkg_core_inj/comparison.md` after the run.
