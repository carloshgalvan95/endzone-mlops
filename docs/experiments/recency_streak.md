# Recency and streak challengers

Method note. I have not fit these models. This file has no log loss. Issue 83 stays open until the warehouse run is recorded. There is no snapshot under `docs/experiments/reports/`.

`h2_full` already averages recent EPA and recent win rate over 10 prior games, across seasons. A short streak can disappear in that mean. I preregister four logistic variants. Each one changes a single block. I do not cross the blocks, and I do not search window lengths.

## Variants

| Feature set | Change from logistic `h2_full` |
| --- | --- |
| `recency_epa5` | Replace the four cross-season recent EPA columns with a 5-game mean. Regressed EPA, season EPA, win rate, and rest stay on the frozen recipe. |
| `recency_margin` | Add `home_recent_point_diff` and `away_recent_point_diff`. Mean of team points minus opponent points over the same 5 games. |
| `recency_streak` | Add `home_signed_streak` and `away_signed_streak`. Signed length of the current win or loss run. This is the full prior sequence, not a 5-game count. |
| `recency_elo` | Add `elo_rating_diff`. Pre-game home Elo minus pre-game away Elo. Home advantage is not in this column. |

`recency_epa5` uses new names (`home_recent5_*` and `away_recent5_*`). The 10-game columns stay on the other three sets.

## Choices I locked before any fit

- Window: 5 games. I considered a 3-game mean and an exponential mean. I am not registering either. Trying one after I see these results is a new preregistration.
- Warmup: the mean uses the prior games that exist. Zero prior games stay missing. The row stays in the cohort.
- Season carry: EPA5, point differential, and the streak include every earlier season. Week 1 does not reset them. This is wider than the regressed EPA prior, which uses only the previous season.
- Same week: an earlier gameday counts. Thursday is history for Sunday. The same gameday does not. A later gameday does not. Kickoff time is not the cutoff. The cutoff is `gameday`, strictly before, which is the rule `h2_full` and Elo already use.
- Ties: the sign comes from the final scores. A tie is margin 0, fills one slot in the point window, and breaks the streak. A trailing tie is 0. An empty history is missing, not 0. I do not use `home_won`. The games mart stores a tie as 0, which would look like a loss.
- `lookback_games` still controls the frozen `h2_full` recent EPA and win rate. It does not change the four new blocks. The command keeps lookback 10.
- Elo: new teams start at 1505. A difference of 0 is that rating, not an imputed missing. The baseline Elo probability and the point spread still add home advantage (65). The feature does not.
- Imputer: a name containing `epa` uses the existing EPA fill of 0, so the 5-game EPA columns do. Point differential, signed streak, and Elo difference use a form fill of 0. `h2_full` has no form columns, and its transformer list stays the EPA imputer, the win-rate imputer, and the rest passthrough.
- Scaler: `StandardScaler` is fit inside each fold, on that fold's training rows, after imputation. The classifier stays logistic, C 1.0, `max_iter` 1000, seed 42.

A score row with no EPA, or an EPA row with no score, can make the windows differ. I leave that gap. I do not invent the missing side.

## What this command is for

Rolling-origin CV, seasons 2016 through 2023. Train on `season < S` and score `season == S`. The candidate and logistic `h2_full` use the same games and the same folds. Each fold refits the frozen recipe. I do not score 2024 or 2025 here.

After the run, a set is eligible for adoption only under ADR-005: pooled log loss at least 0.003 lower than logistic `h2_full`, and the paired 95% bootstrap interval of the difference entirely below 0. The bootstrap is 1000 resamples, seed 42. The 0.002 refresh margin is the walk-forward gate for an unchanged recipe. It is not this gate.

`select_best_feature_set` writes one parent tag: the accepted logistic set with the lowest pooled log loss, or `h2_full` when none is accepted. That tag is a development label. It is not a promotion, and it does not open issue 61. I will report every variant, accepted and rejected. Stacking two passing blocks is a new preregistration. I will not use the 2024 validation or the 2025 test to choose among these four.

The command then checks the written OOF files. Each challenger file has to match `oof_logistic_h2_full.csv` on `game_id` order, `y_true`, and seasons inside 2016-2023. A shorter file fails the command. I do not drop games to force the published reference count. The locked table in `docs/experiments/reference_metrics.json` is n=2191. If this run's `h2_full` count is different, I will say so and I will not call it that table. Market `n` can be smaller. The logistic challenger count has to match logistic `h2_full`.

## Conflicts with the existing gates

Walk-forward issue 80, recorded in `docs/experiments/walk_forward.md`, scores logistic `h2_full` on 2024 week 1 through 2026 week 9. The pool is n=706 and includes 2025. The refresh passes the 0.002 margin. A different feature set on that command would be strict superiority with margin 0, which is still not ADR-005. The prod refresh was fit on completed games before 2026 week 10 (`docs/runbook.md`). I do not score historical folds with that pretrained model.

I do not claim 2025 is untouched. That walk-forward already scored 2025 for logistic `h2_full`. Phase 2 H1 through H5b did not use 2025 to choose features. Phase 0 used 2025 for missingness and range. H5b counted 2025 depth-chart coverage, not log loss. Issue 61, the once-only 2024 validation and 2025 test, is still open. H6 is preregistered in `docs/experiments/h6_opponent_adjusted.md` and has not been fit. The stack of `h6_adj_elo`, the `h5_qb` ratings, and `recency_margin` is preregistered in `docs/experiments/pkg_core.md` and has not been fit. I do not run issue 61 from this comparison.

## Command

From the repo root, after the marts exist:

```bash
make train-recency
```

Windows:

```powershell
.\scripts\dev.ps1 train-recency
```

Both call `python scripts/train_recency.py` and pass no season flags. The script refuses extra arguments. It does not ingest, build marts, walk forward, register, or promote. It does not touch published slates or the `prod` alias.

Local outputs, gitignored:

- `reports/phase2_recency/comparison.md`
- `reports/phase2_recency/comparison.csv`
- `reports/phase2_recency/oof_logistic_h2_full.csv`
- `reports/phase2_recency/oof_logistic_recency_epa5.csv`
- `reports/phase2_recency/oof_logistic_recency_margin.csv`
- `reports/phase2_recency/oof_logistic_recency_streak.csv`
- `reports/phase2_recency/oof_logistic_recency_elo.csv`

The home prior, Elo, and the market write their own OOF files when they score. The MLflow experiment is `nfl_pregame/phase2_recency`.

The single-split trainer refuses these four feature sets before it reads the warehouse. That split scores 2024 and 2025.

If a requested column is missing or at least half null, the existing check names the feature set and points at `make dbt-build-databricks`. That rebuild is a separate command.
