#!/usr/bin/env python3
"""Score the preregistered mean and stack of the defender and Elo.

The arguments are fixed in ``ensembles_r2_challenger_argv``. This command
reads the marts through the existing train loader. It does not ingest,
build marts, walk forward, register, or promote. 2024 and 2025 are not scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from endzone.modeling.ensembles import MODEL_ENS_MEAN_ELO, MODEL_STACK_ELO
from endzone.modeling.evaluation import require_aligned_oof
from endzone.modeling.train import ensembles_r2_challenger_argv
from endzone.modeling.train import main as train_main

DEVELOPMENT_SEASONS: tuple[int, ...] = tuple(range(2016, 2024))
REFERENCE_OOF_NAME: str = "oof_logistic_pkg_core_inj_l2.csv"
SCORED_OOF_NAMES: tuple[tuple[str, str], ...] = (
    ("h2_full", "oof_logistic_h2_full.csv"),
    ("elo", "oof_elo_elo.csv"),
    (MODEL_ENS_MEAN_ELO, f"oof_{MODEL_ENS_MEAN_ELO}_pkg_core_inj_l2.csv"),
    (MODEL_STACK_ELO, f"oof_{MODEL_STACK_ELO}_pkg_core_inj_l2.csv"),
)


def _output_dir(argv: list[str]) -> Path:
    """Return the output directory declared in ``argv``."""
    if "--output-dir" not in argv:
        raise ValueError("ensembles r2 argv is missing --output-dir")
    return Path(argv[argv.index("--output-dir") + 1])


def locked_logistic_h2_full_n(reference_path: Path | None = None) -> int:
    """Return the published logistic ``h2_full`` count.

    Args:
        reference_path: Metrics file. The default is the repo file, resolved
            from this script so the path is not machine-specific.

    Returns:
        The locked ``n`` for logistic ``h2_full``.

    Raises:
        ValueError: The file has no logistic ``h2_full`` row.
    """
    path: Path = (
        Path(__file__).resolve().parents[1] / "docs" / "experiments" / "reference_metrics.json"
        if reference_path is None
        else reference_path
    )
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    models: object = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{path.as_posix()} has no models list")
    for row in models:
        if not isinstance(row, dict):
            continue
        if row.get("model") == "logistic" and row.get("feature_set") == "h2_full":
            return int(row["n"])
    raise ValueError(f"{path.as_posix()} is missing logistic h2_full")


def _read_oof(path: Path) -> pd.DataFrame:
    """Read one OOF csv as UTF-8."""
    if not path.is_file():
        raise ValueError(f"missing OOF file: {path.as_posix()}")
    return pd.read_csv(path, encoding="utf-8")


def require_written_cohort(output_dir: Path) -> dict[str, int]:
    """Fail unless every scored OOF file matches logistic ``pkg_core_inj_l2``.

    Args:
        output_dir: Directory written by the round 2 ensemble CV.

    Returns:
        Dict with ``n`` and ``variants``.

    Raises:
        ValueError: A file is missing or a row does not match the reference
            games on seasons 2016-2023.
    """
    reference: pd.DataFrame = _read_oof(output_dir / REFERENCE_OOF_NAME)
    n_rows: int | None = None
    for candidate_name, file_name in SCORED_OOF_NAMES:
        candidate: pd.DataFrame = _read_oof(output_dir / file_name)
        aligned: dict[str, int] = require_aligned_oof(
            reference,
            candidate,
            candidate_name=candidate_name,
            allowed_seasons=DEVELOPMENT_SEASONS,
        )
        n_rows = aligned["n"]
    if n_rows is None:
        raise ValueError("ensembles r2 scored list is empty")
    return {"n": n_rows, "variants": len(SCORED_OOF_NAMES)}


def require_market_benchmark(output_dir: Path) -> None:
    """Fail unless the written market decision is ``benchmark``.

    Args:
        output_dir: Directory written by the round 2 ensemble CV.

    Raises:
        ValueError: The acceptance file or the market row is missing, or the
            market decision is not ``benchmark``.
    """
    path: Path = output_dir / "acceptance.csv"
    if not path.is_file():
        raise ValueError(f"missing acceptance file: {path.as_posix()}")
    frame: pd.DataFrame = pd.read_csv(path, encoding="utf-8")
    market: pd.DataFrame = frame.loc[
        (frame["model"] == "market") & (frame["feature_set"] == "market")
    ]
    if len(market) != 1:
        raise ValueError("acceptance.csv is missing the market row")
    decision: str = str(market.iloc[0]["decision"])
    if decision != "benchmark":
        raise ValueError(f"market decision is {decision}, expected benchmark")


def cohort_message(n_rows: int, locked_n: int) -> str:
    """Describe the written cohort without forcing the locked count.

    Args:
        n_rows: Rows in the written logistic ``pkg_core_inj_l2`` OOF file.
        locked_n: Published logistic ``h2_full`` count.

    Returns:
        One line for the command log.
    """
    variants: int = len(SCORED_OOF_NAMES)
    if n_rows == locked_n:
        return (
            f"ensembles r2 cohort n={n_rows} variants={variants} "
            "matches the locked logistic h2_full count"
        )
    return (
        f"ensembles r2 cohort n={n_rows} variants={variants}. "
        f"The locked table in docs/experiments/reference_metrics.json is n={locked_n}. "
        "This run is not that table."
    )


def main() -> None:
    """Run the fixed CV, then refuse a misaligned cohort or an accepted market."""
    if sys.argv[1:]:
        raise SystemExit(
            "train_ensembles_r2.py does not take arguments. "
            "The window, the recipe, and the output directory are fixed."
        )
    argv: list[str] = ensembles_r2_challenger_argv()
    train_main(argv)
    output_dir: Path = _output_dir(argv)
    try:
        summary: dict[str, int] = require_written_cohort(output_dir)
        require_market_benchmark(output_dir)
        locked_n: int = locked_logistic_h2_full_n()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(cohort_message(summary["n"], locked_n), flush=True)


if __name__ == "__main__":
    main()
