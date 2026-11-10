"""Key=value status files for a workflow step.

Pages and Results write one small file per step. The run summary reads
those files. Values are single-line text. This module does not read
``.env`` and it does not call Databricks.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

_STATUS_ENV: str = "ENDZONE_RUN_STATUS_DIR"
_KEY: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_MAX_VALUE: int = 200


def status_directory() -> Path | None:
    """Return the status directory when the workflow set it.

    An empty variable means the command is local and should not write a file.
    """
    raw: str | None = os.environ.get(_STATUS_ENV)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw.strip())


def write_step_status(directory: Path, name: str, fields: Mapping[str, object]) -> None:
    """Write ``name``.txt as ``key=value`` lines.

    Empty values are omitted. A value with a newline is dropped so one
    field cannot inject another line. Keys must be short identifiers.

    Raises:
        OSError: The directory cannot be created or the file cannot be written.
    """
    if not _KEY.fullmatch(name):
        raise ValueError(f"Status name {name!r} is not a short identifier.")
    lines: list[str] = []
    for key, value in fields.items():
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            continue
        text: str = _one_line(value)
        if text == "":
            continue
        lines.append(f"{key}={text}")
    folder: Path = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    payload: str = "\n".join(lines)
    if payload != "":
        payload = payload + "\n"
    (folder / f"{name}.txt").write_text(payload, encoding="utf-8", newline="\n")


def append_github_output(fields: Mapping[str, object]) -> None:
    """Append ``key=value`` lines to ``GITHUB_OUTPUT`` when that file is set.

    A local run leaves the variable unset and writes nothing. Multiline
    values are skipped.
    """
    raw: str | None = os.environ.get("GITHUB_OUTPUT")
    if raw is None or raw.strip() == "":
        return
    lines: list[str] = []
    for key, value in fields.items():
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            continue
        text: str = _one_line(value)
        lines.append(f"{key}={text}")
    if not lines:
        return
    path: Path = Path(raw)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def read_status_directory(directory: Path) -> dict[str, dict[str, str]]:
    """Read every ``*.txt`` file in ``directory``.

    The result is keyed by the file stem. A missing directory is an empty
    mapping. A line that is not ``key=value`` is ignored. Later lines with
    the same key replace earlier ones.

    Returns:
        ``{stem: {key: value}}``.
    """
    folder: Path = Path(directory)
    found: dict[str, dict[str, str]] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.txt")):
        if not path.is_file():
            continue
        stem: str = path.stem
        if _KEY.fullmatch(stem) is None:
            continue
        found[stem] = _read_status_file(path)
    return found


def _read_status_file(path: Path) -> dict[str, str]:
    """Parse one status file. Unreadable text becomes an empty mapping."""
    try:
        raw: str = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if _KEY.fullmatch(key) is None:
            continue
        fields[key] = value.strip()
    return fields


def _one_line(value: object) -> str:
    """Return a single-line value, or an empty string when it should be omitted."""
    if value is None or isinstance(value, bool):
        if value is True:
            return "true"
        if value is False:
            return "false"
        return ""
    text: str = str(value).replace("\r", " ").replace("\n", " ").strip()
    if text == "":
        return ""
    return text[:_MAX_VALUE]
