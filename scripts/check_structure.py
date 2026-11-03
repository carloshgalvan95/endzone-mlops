#!/usr/bin/env python3
"""Check the Cookiecutter Data Science layout adapted for this repo.

The checker fails when a required directory or file is missing, when a
top-level ``src/`` package is present, when data or model artifacts are
tracked, or when a Python module sits outside ``endzone/``, ``tests/``,
and ``scripts/``. ``configs/experiments`` holds one YAML file per hypothesis.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "models",
    "notebooks",
    "references",
    "reports/figures",
    "endzone",
    "dbt",
    "docs",
    "docs/adr",
    "scripts",
    "tests",
    "configs",
    "configs/experiments",
)

REQUIRED_FILES: tuple[str, ...] = (
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "models/.gitkeep",
    "notebooks/README.md",
    "references/README.md",
    "reports/figures/.gitkeep",
    "endzone/__init__.py",
)

ALLOWED_PYTHON_ROOTS: frozenset[str] = frozenset({"endzone", "tests", "scripts"})

TRACKED_REPORT: str = "reports/figures/.gitkeep"
TRACKED_MODEL: str = "models/.gitkeep"


def _normalize(path: str) -> str:
    """Return a git path with forward slashes and no leading ./."""
    normalized: str = path.replace("\\", "/").removeprefix("./")
    return normalized


def list_tracked_paths(root: Path) -> list[str]:
    """Return git index paths relative to ``root``.

    Args:
        root: Repository root that contains the ``.git`` directory.

    Returns:
        Forward-slash paths from ``git ls-files``. Empty names are dropped.
    """
    completed: subprocess.CompletedProcess[bytes] = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    raw_output: str = completed.stdout.decode("utf-8")
    return [_normalize(item) for item in raw_output.split("\0") if item]


def find_structure_violations(root: Path, tracked_paths: list[str]) -> list[str]:
    """Return layout violations for ``root``. An empty list means the tree is valid.

    Args:
        root: Repository root. Required paths are read from this directory.
        tracked_paths: Paths recorded in git, relative to ``root``.

    Returns:
        Human-readable violation messages, in check order.
    """
    violations: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        if not (root / relative).is_dir():
            violations.append(f"missing directory: {relative}")
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            violations.append(f"missing file: {relative}")

    if (root / "src").exists():
        violations.append("forbidden top-level src/; the Python package is endzone/")

    for path in (_normalize(item) for item in tracked_paths):
        if path == "src" or path.startswith("src/"):
            violations.append(f"forbidden tracked path under src/: {path}")
        if path.startswith("data/") and not path.endswith(".gitkeep"):
            violations.append(f"forbidden tracked data file: {path}")
        if path.startswith("models/") and path != TRACKED_MODEL:
            violations.append(f"forbidden tracked model artifact: {path}")
        if path.startswith("reports/") and path != TRACKED_REPORT:
            violations.append(f"forbidden tracked report output: {path}")
        if path.endswith(".py"):
            top_level: str = path.split("/", 1)[0]
            if top_level not in ALLOWED_PYTHON_ROOTS:
                violations.append(f"Python file outside endzone/, tests/, and scripts/: {path}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """Check the repository layout and return a process status.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv``.

    Returns:
        ``0`` when the layout matches the contract, ``1`` otherwise.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Check the CCDS repository layout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    root: Path = args.root.resolve()
    try:
        tracked: list[str] = list_tracked_paths(root)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"git ls-files failed: {exc}", file=sys.stderr)
        return 1
    violations: list[str] = find_structure_violations(root, tracked)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print("repository layout ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
