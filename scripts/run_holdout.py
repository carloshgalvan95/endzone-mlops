"""Write the one-shot 2024 and 2025 holdout report.

The recipe is fixed in ``endzone.modeling.holdout_eval``. This command
reads the warehouse only after the frozen JSON is confirmed absent.
It logs the run and does not register or promote a model.

Usage:
    python scripts/run_holdout.py
    python scripts/run_holdout.py --allow-overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from endzone.modeling.holdout_eval import (
    default_report_dir,
    log_holdout_run,
    refuse_existing_holdout,
    run_holdout_from_warehouse,
    write_holdout_reports,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser. The recipe is not a flag."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Score the locked 2024 and 2025 holdout once."
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace docs/experiments/reports/holdout_2024_2025.json if it already exists.",
    )
    return parser


def main(argv: list[str] | None = None, *, report_dir: Path | None = None) -> None:
    """Refuse an existing report, then score, write, and log.

    ``report_dir`` is for tests. The command-line path stays the frozen directory.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    directory: Path = default_report_dir() if report_dir is None else report_dir
    refuse_existing_holdout(directory, allow_overwrite=args.allow_overwrite)
    report: dict[str, object] = run_holdout_from_warehouse()
    paths: dict[str, Path] = write_holdout_reports(report, directory)
    log_holdout_run(report, artifact_path=paths["json"])
    print(f"Wrote {paths['json']}")


if __name__ == "__main__":
    main(sys.argv[1:])
