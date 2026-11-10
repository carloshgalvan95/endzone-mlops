"""Decide whether a Pages run should fit a model.

A scheduled run and a dispatch without ``replace_published`` score the
next unplayed week. When ``predictions/<season>-w<week>.json`` is already
on disk, that file came from ``published-slates`` (the restore step copies
it first). Scoring is skipped. The restore, the finals overlay, and the
deploy still run. ``replace_published`` is never skipped: it stays a
manual rebuild.

The week is ``select_slate`` on games with both scores missing, the same
picker ``python -m endzone.modeling.score`` uses. This module does not fit
a model and does not write a slate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from endzone.ingest.nflverse import NFLVERSE_GAMES_URL, download_parquet_with_validation
from endzone.modeling.weekly import (
    parse_optional_int,
    published_week_is_present,
    select_slate,
    unplayed_games,
)
from endzone.site.results import read_games_file
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)


def score_gate_decision(
    output_dir: Path,
    schedule: pd.DataFrame,
    season: int | None,
    week: int | None,
    *,
    replace_published: bool,
) -> dict[str, Any]:
    """Return whether scoring should be skipped for this restored directory.

    ``replace_published`` always returns ``skip`` false. Otherwise the
    target week is ``select_slate`` on unplayed games. ``skip`` is true
    only when that week's file is already a real slate.

    Returns:
        Dict with ``skip``, ``reason`` (``replace_published``,
        ``published``, ``missing``, or ``no_unplayed_games``), ``season``,
        and ``week``.

    Raises:
        ValueError: The schedule cannot be read, or the week file on disk
            is a real slate that does not match its name.
    """
    if replace_published:
        return {
            "skip": False,
            "reason": "replace_published",
            "season": season,
            "week": week,
        }
    upcoming: pd.DataFrame = unplayed_games(schedule)
    try:
        chosen: dict[str, Any] = select_slate(upcoming, season, week)
    except ValueError as exc:
        logger.info("No unplayed week to score: %s", exc)
        return {
            "skip": False,
            "reason": "no_unplayed_games",
            "season": season,
            "week": week,
        }
    target_season: int = int(chosen["season"])
    target_week: int = int(chosen["week"])
    if published_week_is_present(output_dir, target_season, target_week):
        logger.info(
            "predictions/%s-w%s.json is already published. Skipping the score.",
            target_season,
            target_week,
        )
        return {
            "skip": True,
            "reason": "published",
            "season": target_season,
            "week": target_week,
        }
    return {
        "skip": False,
        "reason": "missing",
        "season": target_season,
        "week": target_week,
    }


def format_gate_output(decision: dict[str, Any]) -> str:
    """Return GitHub Actions output lines for one gate decision."""
    skip: str = "true" if decision["skip"] else "false"
    lines: list[str] = [
        f"skip={skip}",
        f"reason={decision['reason']}",
    ]
    season: object = decision.get("season")
    week: object = decision.get("week")
    if isinstance(season, int) and not isinstance(season, bool):
        lines.append(f"season={season}")
    if isinstance(week, int) and not isinstance(week, bool):
        lines.append(f"week={week}")
    return "\n".join(lines) + "\n"


def write_gate_output(decision: dict[str, Any], destination: Path | None) -> None:
    """Write the gate outputs. ``None`` prints them for a local run.

    ``GITHUB_OUTPUT`` is a file. Logging stays on the process stdout, so
    this function does not mix log lines into that file.
    """
    payload: str = format_gate_output(decision)
    if destination is None:
        sys.stdout.write(payload)
        return
    path: Path = Path(destination)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def build_parser() -> argparse.ArgumentParser:
    """Build the score-gate parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Skip scoring when the target week file is already published. Does not fit a model."
        )
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--season", type=str, default="")
    parser.add_argument("--week", type=str, default="")
    parser.add_argument(
        "--replace-published",
        action="store_true",
        help="Manual rebuild. Never skipped.",
    )
    parser.add_argument(
        "--games-path",
        type=str,
        default=None,
        help="Local games parquet or CSV. Omit it to download the nflverse schedules file.",
    )
    parser.add_argument(
        "--github-output",
        type=str,
        default=None,
        help="File to append skip= and reason= lines to. Defaults to GITHUB_OUTPUT.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the gate decision and return 0.

    A bad schedule or a corrupt week file returns 1. The score step must
    not run over a file it cannot trust.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    try:
        decision: dict[str, Any] = _decision(args)
    except (OSError, ValueError) as exc:
        logger.error("Score gate failed: %s", exc)
        return 1
    destination: str | None = args.github_output or os.environ.get("GITHUB_OUTPUT")
    write_gate_output(decision, Path(destination) if destination else None)
    return 0


def _decision(args: argparse.Namespace) -> dict[str, Any]:
    """Load the schedule unless this is an explicit rebuild."""
    season: int | None = parse_optional_int(args.season)
    week: int | None = parse_optional_int(args.week)
    if args.replace_published:
        return score_gate_decision(
            Path(args.output_dir),
            pd.DataFrame(),
            season,
            week,
            replace_published=True,
        )
    schedule: pd.DataFrame = _load_schedule(args.games_path)
    return score_gate_decision(
        Path(args.output_dir),
        schedule,
        season,
        week,
        replace_published=False,
    )


def _load_schedule(games_path: str | None) -> pd.DataFrame:
    """Read a local games file, or download the nflverse schedules release."""
    if games_path:
        return read_games_file(Path(games_path))
    logger.info("Downloading the nflverse schedule from %s", NFLVERSE_GAMES_URL)
    return download_parquet_with_validation(NFLVERSE_GAMES_URL)


if __name__ == "__main__":
    raise SystemExit(main())
