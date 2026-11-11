#!/usr/bin/env python3
"""Score the preregistered recency challengers.

The arguments are fixed in ``recency_challenger_argv``. This command reads
the marts through the existing train loader. It does not ingest, build
marts, walk forward, register, or promote. 2024 and 2025 are not scored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from endzone.features.recency import RECENCY_CHALLENGER_FEATURE_SETS
from endzone.modeling.evaluation import require_aligned_oof
from endzone.modeling.train import main as train_main
from endzone.modeling.train import recency_challenger_argv

DEVELOPMENT_SEASONS: tuple[int, ...] = tuple(range(2016, 2024))
REFERENCE_OOF_NAME: str = "oof_logistic_h2_full.csv"


def _output_dir(argv: list[str]) -> Path:
    """Return the output directory declared in ``argv``."""
    if "--output-dir" not in argv:
        raise ValueError("recency argv is missing --output-dir")
    return Path(argv[argv.index("--output-dir") + 1])


def _read_oof(path: Path) -> pd.DataFrame:
    """Read one OOF csv as UTF-8."""
    if not path.is_file():
        raise ValueError(f"missing OOF file: {path.as_posix()}")
    return pd.read_csv(path, encoding="utf-8")


def require_written_cohort(output_dir: Path) -> dict[str, int]:
    """Fail unless every challenger OOF file matches logistic h2_full.

    Args:
        output_dir: Directory written by the recency CV.

    Returns:
        Dict with ``n`` and ``variants``.

    Raises:
        ValueError: A file is missing or a challenger does not match the
            reference games on seasons 2016-2023.
    """
    reference: pd.DataFrame = _read_oof(output_dir / REFERENCE_OOF_NAME)
    n_rows: int | None = None
    for feature_set in RECENCY_CHALLENGER_FEATURE_SETS:
        candidate: pd.DataFrame = _read_oof(output_dir / f"oof_logistic_{feature_set}.csv")
        aligned: dict[str, int] = require_aligned_oof(
            reference,
            candidate,
            candidate_name=feature_set,
            allowed_seasons=DEVELOPMENT_SEASONS,
        )
        n_rows = aligned["n"]
    if n_rows is None:
        raise ValueError("recency challenger list is empty")
    return {"n": n_rows, "variants": len(RECENCY_CHALLENGER_FEATURE_SETS)}


def main() -> None:
    """Run the fixed CV, then refuse a challenger that drops games."""
    if sys.argv[1:]:
        raise SystemExit(
            "train_recency.py does not take arguments. "
            "The window, the recipe, and the output directory are fixed."
        )
    argv: list[str] = recency_challenger_argv()
    train_main(argv)
    try:
        summary: dict[str, int] = require_written_cohort(_output_dir(argv))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"recency cohort n={summary['n']} variants={summary['variants']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
