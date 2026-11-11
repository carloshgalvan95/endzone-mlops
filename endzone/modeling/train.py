"""Command line for pregame training.

Canonical command: ``python -m endzone.modeling.train``.
``python -m endzone.train.train`` calls this same ``main``.
"""

from __future__ import annotations

import argparse

from endzone.features.pregame import FEATURE_SET_CURRENT, FEATURE_SET_NAMES
from endzone.modeling.dataset import DEFAULT_MAX_NULL_FRACTION
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import load_catalog
from endzone.modeling.pipeline import run_cv_cli, run_reference_check
from endzone.modeling.reference import format_reference_check
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the training parser. Hypothesis choices come from the experiment files."""
    parser = argparse.ArgumentParser(description="Train NFL pregame prediction models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline-home", "logistic", "logistic_l2", "gbm"],
        default="logistic",
        help=(
            "Model type to train. logistic_l2 is L2 logistic regression "
            "with C tuned on the training seasons only."
        ),
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Number of recent games for rolling features",
    )
    parser.add_argument(
        "--train-end-season",
        type=int,
        default=2023,
        help="Last season to include in training",
    )
    parser.add_argument("--val-season", type=int, default=2024, help="Validation season")
    parser.add_argument("--test-season", type=int, default=2025, help="Test season")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=list(FEATURE_SET_NAMES),
        default=FEATURE_SET_CURRENT,
        help="Columns for a single split. --cv selects the logistic sets for --phase.",
    )
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help=(
            "Refit logistic h2_full, the home prior, Elo, and the market "
            "on the window in docs/experiments/reference_metrics.json. "
            "Exit non-zero when log loss, n, Elo, or the market differ "
            "by more than the file tolerance (1e-4)."
        ),
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help=(
            "Run rolling-origin CV on 2016-2023. "
            "--phase 1 is the baseline report. "
            "--phase 2 --hypothesis h1 scores diff feature sets. "
            "--phase 2 --hypothesis h2 scores season defense and regressed EPA. "
            "--phase 2 --hypothesis h3 scores pass and rush splits against h2_full. "
            "--phase 2 --hypothesis h4 scores clean EPA against h2_full. "
            "--phase 2 --hypothesis h5 scores starting QB ratings against h2_full. "
            "--phase 2 --hypothesis h5b scores the same ratings after the starter fix. "
            "--phase 2 --hypothesis ridge scores logistic_l2 against logistic h2_full. "
            "python scripts/train_recency.py scores the preregistered recency challengers. "
            "python scripts/train_h6.py scores the preregistered H6 challengers. "
            "Does not score 2024 or 2025."
        ),
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "1 keeps the phase 1 report (home prior, current, with_games_played). "
            "2 scores the hypothesis from --hypothesis. H1 and H2 use logistic "
            "current. H3, H4, H5, H5b, and ridge use logistic h2_full."
        ),
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        choices=list(load_catalog().alias_names()),
        default=None,
        help=(
            "Phase 2 hypothesis. Default h1 scores diff and diff_only "
            "and writes reports/phase2. h2 scores h2_def and h2_full "
            "and writes reports/phase2_h2. h3 scores h3_split and h3_replace "
            "against h2_full and writes reports/phase2_h3. h4 scores h4_clean "
            "and h4_clean_sr against h2_full and writes reports/phase2_h4. "
            "h5 scores h5_qb against h2_full and writes reports/phase2_h5. "
            "h5b scores the same h5_qb set against h2_full and writes "
            "reports/phase2_h5b. "
            "ridge scores "
            "logistic_l2 against logistic h2_full and writes reports/phase2_ridge. "
            "Phase 1 rejects this flag."
        ),
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help=(
            "Model for the paired log loss reference. "
            "Default is baseline-home for phase 1 and logistic for phase 2."
        ),
    )
    parser.add_argument(
        "--reference-feature-set",
        type=str,
        default=None,
        help=(
            "Feature set of the reference model. "
            "Default is none for the home prior, current for logistic on H1 and H2, "
            "and h2_full for logistic on H3, H4, H5, H5b, and ridge."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/phase1",
        help=(
            "Directory for the comparison report. "
            "Phase 2 h1 with this default writes reports/phase2. "
            "Phase 2 h2 with this default writes reports/phase2_h2. "
            "Phase 2 h3 with this default writes reports/phase2_h3. "
            "Phase 2 h4 with this default writes reports/phase2_h4. "
            "Phase 2 h5 with this default writes reports/phase2_h5. "
            "Phase 2 ridge with this default writes reports/phase2_ridge."
        ),
    )
    parser.add_argument(
        "--max-null-fraction",
        type=float,
        default=DEFAULT_MAX_NULL_FRACTION,
        help=(
            "Refuse a requested CV feature column whose null share is at "
            "least this value. Default is 0.5. The error names the feature "
            "set and says to run make dbt-build-databricks."
        ),
    )
    parser.add_argument("--cv-start-season", type=int, default=2016)
    parser.add_argument("--cv-end-season", type=int, default=2023)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--fixture-path",
        type=str,
        help="Path to fixture directory (for testing without Databricks)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (logistic)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations (logistic)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting stages (gbm)",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum tree depth (gbm)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (gbm)")
    return parser


def recency_challenger_argv() -> list[str]:
    """Argv for the preregistered recency CV.

    The window is seasons 2016-2023. Logistic ``h2_full`` stays lookback 10,
    C 1, max_iter 1000, and seed 42. The bootstrap is 1000 resamples, seed 42.
    The command does not register a model and does not score 2024 or 2025.

    Returns:
        Arguments for ``endzone.modeling.train.main``.
    """
    return [
        "--cv",
        "--phase",
        "2",
        "--hypothesis",
        "recency",
        "--reference-model",
        "logistic",
        "--reference-feature-set",
        "h2_full",
        "--lookback",
        "10",
        "--C",
        "1.0",
        "--max-iter",
        "1000",
        "--seed",
        "42",
        "--cv-start-season",
        "2016",
        "--cv-end-season",
        "2023",
        "--n-bootstrap",
        "1000",
        "--bootstrap-seed",
        "42",
        "--output-dir",
        "reports/phase2_recency",
    ]


def h6_challenger_argv() -> list[str]:
    """Argv for the preregistered H6 CV.

    The window is seasons 2016-2023. Logistic ``h2_full`` stays lookback 10,
    C 1, max_iter 1000, and seed 42. The bootstrap is 1000 resamples, seed 42.
    The command does not register a model and does not score 2024 or 2025.

    Returns:
        Arguments for ``endzone.modeling.train.main``.
    """
    return [
        "--cv",
        "--phase",
        "2",
        "--hypothesis",
        "h6",
        "--reference-model",
        "logistic",
        "--reference-feature-set",
        "h2_full",
        "--lookback",
        "10",
        "--C",
        "1.0",
        "--max-iter",
        "1000",
        "--seed",
        "42",
        "--cv-start-season",
        "2016",
        "--cv-end-season",
        "2023",
        "--n-bootstrap",
        "1000",
        "--bootstrap-seed",
        "42",
        "--output-dir",
        "reports/phase2_h6",
    ]


def main(argv: list[str] | None = None) -> None:
    """Train one holdout model, run rolling-origin CV, or refit the reference metrics."""
    args = build_parser().parse_args(argv)
    if args.verify_reference:
        fixture_path: str | None = args.fixture_path if args.fixture_path else None
        reference_result = run_reference_check(fixture_path=fixture_path)
        reference_message: str = format_reference_check(reference_result)
        if reference_result["ok"]:
            logger.info(reference_message)
            return
        logger.error(reference_message)
        raise SystemExit(1)
    if args.cv:
        run_cv_cli(args)
        return
    run_holdout(args)


if __name__ == "__main__":
    main()
