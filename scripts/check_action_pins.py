#!/usr/bin/env python3
"""Fail when a GitHub Action ref is not pinned to a full commit SHA.

Every ``uses:`` in ``.github/workflows`` and in a composite ``action.yml``
must be ``owner/repo@`` plus 40 lowercase hex characters, with a trailing
``# vX.Y.Z`` comment. A local action whose ref starts with ``./`` is exempt.
The check reads files only, so it stays fast.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

_SHA: re.Pattern[str] = re.compile(r"[0-9a-f]{40}")
_VERSION: re.Pattern[str] = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
_USES_KEY: re.Pattern[str] = re.compile(r"^[ ]*(?:- )?uses:[ ]*")
_USES_LINE: re.Pattern[str] = re.compile(
    r"""^[ ]*(?:- )?uses:[ ]*(?P<value>"[^"]+"|'[^']+'|[^"'#\s]+)[ ]*(?:#[ ]*(?P<comment>\S+))?[ ]*$"""
)
_BLOCK_HEADER: re.Pattern[str] = re.compile(r":[ ]*[|>][-+]?[ ]*(?:#.*)?$")
_SKIPPED_PARTS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".stamp",
        "__pycache__",
        "dbt_packages",
        "mlartifacts",
        "mlruns",
        "node_modules",
        "target",
    }
)
_ACTION_FILENAMES: frozenset[str] = frozenset({"action.yml", "action.yaml"})


class _LineScan(NamedTuple):
    """Uses lines from one file, and line numbers that did not parse."""

    rows: tuple[tuple[int, str, str | None], ...]
    unreadable: tuple[int, ...]


def _leading_spaces(line: str) -> int:
    """Return the number of leading space characters on ``line``.

    Args:
        line: One physical line from a workflow file.

    Returns:
        The count of leading spaces. A tab does not count.
    """
    count: int = 0
    for char in line:
        if char != " ":
            break
        count += 1
    return count


def _unquote(value: str) -> str:
    """Return ``value`` without one matching pair of quotes.

    Args:
        value: The raw token after ``uses:``.

    Returns:
        The token with a surrounding ``'`` or ``"`` pair removed.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _pin_error(value: str, comment: str | None) -> str | None:
    """Return an error for one ``uses`` value, or ``None`` when it is valid.

    Args:
        value: The action reference, without quotes.
        comment: The trailing comment token, when the line has one.

    Returns:
        A short error, or ``None`` for a local action or a pinned SHA.
    """
    if value.startswith("./"):
        return None
    repo, separator, ref = value.rpartition("@")
    if separator != "@" or repo == "" or _SHA.fullmatch(ref) is None:
        return "uses is not a 40-character commit SHA"
    if comment is None or _VERSION.fullmatch(comment) is None:
        return "uses is missing a # vX.Y.Z comment"
    return None


def _scan_uses_lines(text: str) -> _LineScan:
    """Read ``uses:`` keys, skipping folded and literal block bodies.

    Args:
        text: Full workflow or composite-action source.

    Returns:
        Parsed uses rows, plus line numbers whose ``uses:`` key did not match.
    """
    rows: list[tuple[int, str, str | None]] = []
    unreadable: list[int] = []
    block_indent: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if block_indent is not None:
            if line.strip() == "" or _leading_spaces(line) > block_indent:
                continue
            block_indent = None
        if line.lstrip(" ").startswith("#"):
            continue
        if _USES_KEY.match(line) is None:
            if _BLOCK_HEADER.search(line) is not None:
                block_indent = _leading_spaces(line)
            continue
        if _BLOCK_HEADER.search(line) is not None:
            unreadable.append(line_number)
            block_indent = _leading_spaces(line)
            continue
        match: re.Match[str] | None = _USES_LINE.match(line)
        if match is None:
            unreadable.append(line_number)
            continue
        rows.append((line_number, _unquote(match.group("value")), match.group("comment")))
    return _LineScan(rows=tuple(rows), unreadable=tuple(unreadable))


def _collect_uses(node: object, found: list[object]) -> None:
    """Append every ``uses`` value under ``node``, in document order.

    Args:
        node: A YAML node from a workflow or composite action.
        found: Destination list. Nested values are appended in walk order.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses":
                found.append(value)
            else:
                _collect_uses(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_uses(item, found)


def _action_files(root: Path) -> list[Path]:
    """Return workflow files and composite actions under ``root``.

    Args:
        root: Repository root.

    Returns:
        Absolute paths, sorted by POSIX relative form. Heavy caches are skipped.
    """
    found: dict[Path, None] = {}
    workflows: Path = root / ".github" / "workflows"
    if workflows.is_dir():
        for pattern in ("*.yml", "*.yaml"):
            for path in workflows.glob(pattern):
                found[path.resolve()] = None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIPPED_PARTS]
        for filename in filenames:
            if filename in _ACTION_FILENAMES:
                found[(Path(dirpath) / filename).resolve()] = None
    return sorted(found, key=lambda path: path.as_posix())


def _relative(root: Path, path: Path) -> str:
    """Return ``path`` relative to ``root`` with forward slashes.

    Args:
        root: Repository root.
        path: A file under ``root``.

    Returns:
        A POSIX relative path.
    """
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_violations(root: Path, path: Path) -> list[str]:
    """Return pin violations for one workflow or composite action.

    Args:
        root: Repository root, used for the message path.
        path: File to read as UTF-8.

    Returns:
        Human-readable violations. An empty list means the file is pinned.
    """
    relative: str = _relative(root, path)
    try:
        text: str = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        return [f"{relative}: not utf-8: {exc}"]
    try:
        documents: list[object] = [item for item in yaml.safe_load_all(text) if item is not None]
    except yaml.YAMLError as exc:
        return [f"{relative}: invalid YAML: {exc}"]

    parsed: list[object] = []
    for document in documents:
        _collect_uses(document, parsed)
    violations: list[str] = []
    parsed_values: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or item.strip() == "":
            violations.append(f"{relative}: uses value is not a string")
        else:
            parsed_values.append(item)
    scan: _LineScan = _scan_uses_lines(text)
    for line_number in scan.unreadable:
        violations.append(f"{relative}:{line_number}: could not read uses:")
    if violations:
        return violations
    scanned: list[str] = [value for _, value, _ in scan.rows]
    if parsed_values != scanned:
        return [f"{relative}: parsed uses keys do not match uses: lines"]
    for line_number, value, comment in scan.rows:
        error: str | None = _pin_error(value, comment)
        if error is not None:
            violations.append(f"{relative}:{line_number}: {error}: {value}")
    return violations


def find_pin_violations(root: Path) -> list[str]:
    """Return pin violations for workflows and composite actions under ``root``.

    Args:
        root: Repository root that contains ``.github/workflows``.

    Returns:
        Violation messages. An empty list means every third-party ``uses:``
        is a 40-character SHA with a ``# vX.Y.Z`` comment.
    """
    files: list[Path] = _action_files(root)
    if not files:
        return ["no GitHub Actions workflows or composite actions found"]
    violations: list[str] = []
    for path in files:
        violations.extend(_file_violations(root, path))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Check action pins and return a process status.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv``.

    Returns:
        ``0`` when every third-party ``uses:`` is pinned, ``1`` otherwise.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Check that workflow actions are pinned to commit SHAs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1
    violations: list[str] = find_pin_violations(root)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print("action pins ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
