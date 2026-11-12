# FiveThirtyEight QB adjustment on Elo

Preregistered before any fit. This note does not report a log loss. I will copy `reports/phase2_qbelo538/comparison.md` here only after the warehouse run, and only if I decide the candidate.

## Goal

Test whether FiveThirtyEight's starting-QB adjustment on top of our margin-of-victory Elo beats logistic `pkg_core_inj_l2`. Round 2 was rejected. The defender stays logistic `pkg_core_inj_l2` with C fixed at 0.1. Preseason win totals are not used. Travel and bye stay out of this round.

## Reference

Logistic `pkg_core_inj_l2`, C 0.1. The same rolling-origin window as the earlier phase 2 notes: train on seasons before S, score season S, for S from 2016 through 2023. Do not score 2024, 2025, or the holdout split.

## Candidate

One candidate. `pkg_core_inj_l2_qbelo538` is the defender columns with `elo_mov_diff` replaced by `qbelo538_diff`. Logistic regression, C fixed at 0.1. No search. The column count is unchanged.

`qbelo538_diff = elo_mov_diff + 3.3 * (home_qb_adj - away_qb_adj)`.

`qb_adj = starter_qb_rating - team_qb_rating`.

The starter is the pre-game depth-chart QB1 from `fct_depth_chart_starters`, with the same previous-passer fallback H5 uses when the chart is missing. It is never the quarterback who played the most snaps in the game being scored.

VALUE uses only games strictly before the scored game's date. A same-day game is not history. A quarterback with no prior game has adjustment 0, including when the team already has a rating. There is no draft-position prior and no between-season reversion of the QB ratings.

## Constants

Source: [How Our NFL Predictions Work](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/). The live page no longer serves that methodology. I checked the archived text last edited 15 Oct 2020 and captured 15 Dec 2020: [archive](https://web.archive.org/web/20201215073842/https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).

| Constant | Value in this round | Page |
| --- | --- | --- |
| Pass attempt | -2.2 | same |
| Completion | 3.7 | same |
| Passing yards | 1/5 | same |
| Passing touchdown | 11.3 | same |
| Interception | -14.1 | same |
| Sack | -8 | same |
| Rush attempt | -1.1 | same |
| Rushing yards | 0.6 | same |
| Rushing touchdown | 15.9 | same |
| Quarterback rating | 0.9 old + 0.1 new | "updated every 10 games" |
| Team rating | 0.95 old + 0.05 new | "updated every 20 games" |
| Elo per point of starter minus team | 3.3 | same |
| Win probability scale | 400 | same |
| Context home advantage | 65 | page base is 55 plus travel |

None of the VALUE weights, the 0.9 and 0.1 update, the 3.3 factor, or the 400 scale differs from the issue. The page does not print 0.95. It says the team rolling rating is updated every 20 games, in the same construction as the individual rating updated every 10 games (0.9 old + 0.1 new). One over twenty is 0.05, so I kept 0.95 and 0.05.

The page's base home-field adjustment is 55 Elo points plus 4 points per 1,000 miles traveled, and 33 points when few fans attend. The context probabilities in this round stay at home advantage 65, which is the number already used for our Elo and the number this comparison preregistered. Travel is a later round. I did not replace 65 with 55.

The page also adjusts a game's VALUE for the opponent's rolling VALUE allowed. It does not publish a separate decay for that defensive rating. The preregistered formula has no defensive term, so this round does not apply one.

The page reverts veteran QB ratings one-fourth between seasons and starts drafted rookies away from zero. This round does neither. A new quarterback starts at 0, and the adjustment stays 0 until that quarterback has a prior game.

`elo_mov_diff` already uses the page's margin-of-victory multiplier, K of 20, and reversion of one-third toward 1505. Those stay in the existing Elo. This round does not retune them.

## Context rows

`elo_mov` is the home win probability from `elo_mov_diff` plus home advantage 65 on a 400 scale: `1 / (10 ** (-(diff + 65) / 400) + 1)`.

`qbelo538` is the same formula with `qbelo538_diff`.

Both are context. They are not candidates. `h2_full` is also scored for context. The market is a benchmark only.

## Acceptance

ADR-005, unchanged. Accept only when the log loss difference versus logistic `pkg_core_inj_l2` is at most -0.003 and the 95% paired bootstrap interval lies entirely below 0. The bootstrap is 1000 resamples, seed 42.

Calibration, per-season results, QB coverage, and the market-information regression are descriptive. QB coverage is one row per scored season: the share of games with a resolved starter for both teams, and the share with a prior VALUE game for both starters.

## Rules

- Same outer folds as the injury note and both ensemble notes.
- Do not score 2024, 2025, or the holdout split.
- No tuning after results, no threshold changes, no extra candidates.
- Nothing is registered or promoted.

## Reproduce

Rebuild the marts so `fct_qb_box_scores` exists. Do not add an ingest. The counting stats are already on the play-by-play source.

```bash
make dbt-build-databricks
PYTHONUTF8=1 MLFLOW_TRACKING_URI=databricks MLFLOW_REGISTRY_URI=databricks-uc uv run python scripts/train_qbelo538.py
```

`make train-qbelo538` is the train step alone. Windows: `.\scripts\dev.ps1 train-qbelo538`.

`train_qbelo538.py` takes no arguments. It does not ingest, build marts, walk forward, register, or promote. It does not score 2024 or 2025. Logistic `h2_full` uses C 1.0. Logistic `pkg_core_inj_l2` and `pkg_core_inj_l2_qbelo538` keep C at 0.1. The home prior, Elo, and the market are scored with the other rows. The market decision is `benchmark`. `elo_mov` and `qbelo538` are `baseline`.

## Output

- `reports/phase2_qbelo538/comparison.md`
- `reports/phase2_qbelo538/comparison.csv`
- `reports/phase2_qbelo538/acceptance.csv`
- `reports/phase2_qbelo538/calibration.csv`
- `reports/phase2_qbelo538/market_information.csv`
- `reports/phase2_qbelo538/qb_coverage.csv`
- `reports/phase2_qbelo538/oof_logistic_h2_full.csv`
- `reports/phase2_qbelo538/oof_logistic_pkg_core_inj_l2.csv`
- `reports/phase2_qbelo538/oof_logistic_pkg_core_inj_l2_qbelo538.csv`
- `reports/phase2_qbelo538/oof_elo_elo.csv`
- `reports/phase2_qbelo538/oof_elo_mov_elo_mov.csv`
- `reports/phase2_qbelo538/oof_qbelo538_qbelo538.csv`

The home prior and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_qbelo538`.

Read `reports/phase2_qbelo538/comparison.md` after the run.
