# Phase 2 ridge reference

Method note. This file does not report warehouse metrics. `make train-cv` (Windows: `.\scripts\dev.ps1 train-cv`) runs `python -m src.train.train --cv --phase 2 --hypothesis ridge --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_ridge` and writes the measured comparison to `reports/phase2_ridge/comparison.md` and `reports/phase2_ridge/comparison.csv`. Those files are local outputs. Leave the result section empty until that run exists. Do not copy numbers from a test fixture.

No model is promoted from this run. 2024 stays the final validation. 2025 is not scored here.

This is setup work, not a new signal hypothesis. Fixed-C logistic regression on `h2_full` and the H3 splits fits correlated EPA columns with one C. A later rejection can then mean the columns were noisy, or it can mean the un-tuned penalty let those columns move the fit. Tuning L2 `C` inside each training fold separates those two stories. If `logistic_l2` on `h2_full` is not worse than logistic `h2_full`, it becomes the reference for later hypotheses. A rejection after that change is a lack of signal, not noise from correlated columns.

Hypothesis H3 stays reproducible with `python -m src.train.train --cv --phase 2 --hypothesis h3 --reference-model logistic --reference-feature-set h2_full --output-dir reports/phase2_h3`. H3 rejected both `h3_split` and `h3_replace` against logistic `h2_full` (0.6393 and 0.6395 versus 0.6377, n=2191). See `docs/experiments/phase2_h3_pass_rush_epa.md`.

## What is scored

Same rolling-origin window as Phase 1 and the earlier Phase 2 hypotheses. For each season S from 2016 through 2023, train on `season < S` and score `season == S`. 2024 is not a fold. 2025 is not a fold.

| Model | Feature set | Role |
| --- | --- | --- |
| `logistic` | `h2_full` | Reference. Fixed C, default 1.0. |
| `logistic_l2` | `h2_full` | Same columns. C is chosen inside the training seasons. |
| `logistic_l2` | `h3_split` | Pass and rush columns added to `h2_full`, with the same inner C search. |
| `logistic_l2` | `h3_replace` | Combined recent and regressed-season EPA replaced by the splits, with the same inner C search. |
| `baseline-home` | none | Home prior. |
| `elo` | `elo` | Phase 1 Elo settings. |
| `market` | `market` | No-vig closing moneyline, skipped when the lines are absent. |

`logistic` is already L2. `LogisticRegression` keeps its default L2 penalty. `logistic_l2` does not change the penalty. It changes how `C` is chosen.

## How C is chosen

For a fold that scores season S, the search sees only games with `season < S`.

- Sort the training seasons.
- For each later training season T, fit on seasons strictly before T and score log loss on T. The earliest training season has no prior rows, so it is not an inner validation season. An inner season with a single class in the prior rows or in T is skipped.
- Try C in `0.01, 0.1, 1.0, 10.0, 100.0`. The value with the lowest mean inner log loss wins. A tie keeps the smaller C.
- Refit on every training season with that C, then score S.

S is not an inner season. Flipping the labels of S does not change the chosen C or the probabilities for S. When no inner season can be scored, the fold keeps the fallback C (the `--C` value, default 1.0) and records `c_selection=fallback`.

Each scored `logistic_l2` child run logs `chosen_c_by_fold` and `chosen_c_<season>`. The fold table in the report has `chosen_c` and `c_selection`.

`--model logistic_l2` on a single split uses the same search on the training seasons only. 2024 is scored after C is fixed. 2025 is not used to pick C.

## Acceptance rule

The comparison table uses the same rule as H1, H2, and H3, versus logistic `h2_full`:

1. Pooled log loss improves by at least 0.003 (`log_loss_diff` <= -0.003).
2. The 95% paired bootstrap interval lies entirely below 0 (`log_loss_diff_ci_high` < 0).

If either condition fails, the row is rejected. A missing interval is inconclusive. The reference row is not a candidate. `logistic_l2` on `h2_full`, `h3_split`, and `h3_replace` are judged separately. Baselines in the same report are labeled with the same rule. They are not the setup decision.

## Setup decision

Apply this only to `logistic_l2` on `h2_full`, after the local run:

- Not worse: `log_loss_diff` <= 0. That row becomes the reference for the next hypothesis, even when the 0.003 rule rejects it. Matching the reference, or beating it by less than 0.003, is enough. The point of the row is a penalty that does not add noise, not a new feature set.
- Worse: `log_loss_diff` > 0. Keep logistic `h2_full` as the reference.
- Inconclusive: the interval is missing. Do not switch the reference.

`h3_split` and `h3_replace` stay candidates. They answer whether L2 changes the H3 rejection. They do not become the reference in this run.

## Result

Leave this table empty until `make train-cv` has been run locally. Do not fill it from a test fixture.

| model | feature_set | n | log_loss | log_loss_diff | ci_low | ci_high | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## Decision

Pending. Carlos fills this after the local run.

`decision` on the comparison table is `accepted`, `rejected`, or `inconclusive`. The setup decision for the next reference is separate, and it is recorded here after the run.
