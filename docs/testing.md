# Testing

## When a test is worth adding

Add a test when a failure would hide a real bug:

- The scored week leaks into its own features, labels, or Elo update.
- A published slate is overwritten or shrunk, including `published-slates` and `replace_published`.
- The promotion gate or the refresh non-inferiority margin (`REFRESH_LOG_LOSS_MARGIN`) moves.
- Log loss, the golden CV file, or a data contract changes by accident.
- The PR guard misses an attribution trailer, a non-Conventional Commit, or a bad branch name.
- `store_failures` starts writing failure relations.

Path, encoding, line-ending, CLI, script (`scripts/dev.ps1`), and file I/O tests take the `cross_platform` marker. Linux runs every test on a pull request. Windows and macOS run that marker on a pull request, and the full suite on push to `main`, the Monday schedule, and `workflow_dispatch`.

Do not add a test that only checks a document contains a sentence, bans a year or another number in prose, or locks an implementation detail such as a CSS declaration. Parametrize a near-duplicate instead of copying it. The model-card scan that rejected any number missing from a cited file is the example to avoid: it forced the year 2026 out of the card.

## How CI runs pytest

`.github/workflows/ci.yml` keeps the check names `test (3.12)`, `cross-platform (windows-latest, 3.12)`, and `cross-platform (macos-latest, 3.12)`.

| Event | Linux `test` | Windows and macOS `cross-platform` |
| --- | --- | --- |
| `pull_request` | full suite | `pytest -m cross_platform` |
| push to `main` | full suite | full suite |
| Monday 12:00 UTC | full suite | full suite |
| `workflow_dispatch` | full suite | full suite |

The Windows and macOS jobs are not skipped on a pull request. Filtering the pytest command keeps the required check green. `pytest-xdist` is pinned in `uv.lock` and `addopts` includes `-n auto`. Tests keep files under `tmp_path` and do not share one sqlite database, so workers do not depend on order.

## Before

Local, Python 3.12, `main` at `42542f9`, 10 Nov 2026. `pytest --durations=30` with no xdist:

- 485 passed
- 102.70 s

CI wall clock for six recent runs of `ci.yml`. The clock includes checkout and install.

| Run | Event | Linux | Windows | macOS |
| --- | --- | --- | --- | --- |
| 42846472477 | push | 3 min 35 s | 8 min 32 s | 2 min 58 s |
| 42844548422 | pull_request | 3 min 28 s | 9 min 51 s | 3 min 33 s |
| 42843974119 | pull_request | 3 min 59 s | 11 min 7 s | 3 min 13 s |
| 42839375564 | pull_request | 3 min 25 s | 12 min 25 s | 2 min 12 s |
| 42751247376 | push | 3 min 22 s | 7 min 33 s | 1 min 49 s |
| 42749662547 | pull_request | 2 min 12 s | 6 min 57 s | 2 min 37 s |

On run 42846472477 the Windows pytest step alone was 8 min 4 s. Checkout through the structure check was about 22 s. The Linux pytest step on that run was 2 min 24 s, then 43 s of analysis coverage and about 16 s of dbt.

## After

Same machine, this branch, `pytest-xdist` with 4 workers (`-n auto`):

- 475 collected
- 473 passed, 2 skipped, 36.04 s for `PYTHONUTF8=1 pytest tests/ -v`
- 87 passed, 2 skipped, 6.78 s for `pytest -m cross_platform`

The two skips are `scripts/dev.ps1` and `scripts/train_sweep.ps1`. This Linux image has no PowerShell, so the parser is skipped after the UTF-8 read. Windows CI has `pwsh` and parses both files.

The ten collected tests that left are doc and CSS snapshots, two fixture-size checks, and two logger duplicates that are now one test. The season parser and the staging column contract are parametrized, so those cases are still collected. The slow tests (golden CV, hypothesis CLIs, walk-forward, promotion) are unchanged.

Expected CI time, using the step split above:

- A Windows pull request replaces the 8 min pytest step with the OS subset (6.78 s here, well under a minute even if Windows is several times slower) plus about 22 s of checkout, install, and lint. Expected job about 1 to 2 min, under 3 min.
- A macOS pull request uses the same filter. Expected under 2 min.
- Linux still runs the full suite, coverage, and dbt. Local pytest fell from 103 s to 36 s, so the Linux job should drop by about a minute and stay near 2 to 3 min.
- Push to `main`, Monday 12:00 UTC, and `workflow_dispatch` still run the full suite on Windows and macOS. xdist shortens that pytest step. It does not remove it.

## What stayed

Leakage in `tests/test_feature_leakage.py` and `tests/test_walk_forward.py`. Published slates in `tests/test_published_archive.py` and `tests/test_weekly_predictions.py` (`replace_published`, no shrink, push does not score). Promotion and the refresh margin in `tests/test_registry.py` and `tests/test_walk_forward.py`. Log loss and the golden file in `tests/test_golden_cv.py`, `tests/test_reference_metrics.py`, and the metrics tests. Data contracts in the staging, clean-EPA, and starter-mart tests. The PR guard, including cursor attribution and Conventional Commit and branch rules, in `tests/test_pr_guard.py`. `store_failures` in `tests/test_dbt_store_failures.py`. The model card still checks that the locked reference numbers are quoted and that the market has the lower log loss.
