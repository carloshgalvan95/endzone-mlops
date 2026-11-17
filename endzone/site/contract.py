"""Validate published site JSON against the v1 contract.

The schemas live in ``contracts/site-data/v1/``. This module only reads.
It does not score, overlay finals, archive a slate, or edit a week file.
See ADR-011.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

CONTRACT_VERSION: str = "1.0.0"
SCHEMA_DIR: Path = Path(__file__).resolve().parents[2] / "contracts" / "site-data" / "v1"
SLATE_SCHEMA: str = "slate.schema.json"
TEAMS_SCHEMA: str = "teams.schema.json"
TRACK_RECORD_SCHEMA: str = "track_record.schema.json"


class ContractError(ValueError):
    """A site data file is missing or does not match its schema."""


def load_schema(name: str) -> dict[str, Any]:
    """Read one draft 2020-12 schema from ``contracts/site-data/v1/``.

    Raises:
        ContractError: The file is missing or is not a JSON object.
    """
    path: Path = SCHEMA_DIR / name
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read schema {name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"Schema {name} is not a JSON object.")
    return payload


def validate_instance(instance: object, schema_name: str) -> None:
    """Raise ``ContractError`` when ``instance`` fails ``schema_name``."""
    schema: dict[str, Any] = load_schema(schema_name)
    validator: Draft202012Validator = Draft202012Validator(schema)
    errors: list[ValidationError] = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.path],
    )
    if not errors:
        return
    first: ValidationError = errors[0]
    location: str = "/".join(str(part) for part in first.path) or "<root>"
    raise ContractError(f"{schema_name} at {location}: {first.message}")


def validate_data_dir(data_dir: Path, *, require_published: bool = False) -> dict[str, Any]:
    """Validate the site data files a client can read.

    ``sample.json`` and ``teams.json`` are always required. ``latest.json``,
    every ``predictions/*.json``, and ``track_record.json`` are checked
    when they are present. ``require_published`` also requires ``latest.json``,
    ``track_record.json``, and at least one week file, which is the tree
    Pages and Results upload.

    Returns:
        ``{"files": int, "paths": list[str]}`` with paths relative to ``data_dir``.

    Raises:
        ContractError: A required file is missing, or a file fails its schema.
    """
    root: Path = Path(data_dir)
    checked: list[str] = []
    _validate_named(root, "sample.json", SLATE_SCHEMA, checked)
    _validate_named(root, "teams.json", TEAMS_SCHEMA, checked)
    latest: Path = root / "latest.json"
    if latest.is_file():
        _validate_named(root, "latest.json", SLATE_SCHEMA, checked)
    elif require_published:
        raise ContractError("latest.json is required before upload.")
    predictions: Path = root / "predictions"
    week_files: list[Path] = []
    if predictions.is_dir():
        week_files = sorted(path for path in predictions.glob("*.json") if path.is_file())
    if require_published and not week_files:
        raise ContractError("predictions/*.json is required before upload.")
    for path in week_files:
        _validate_file(root, path, SLATE_SCHEMA, checked)
    record: Path = root / "track_record.json"
    if record.is_file():
        _validate_named(root, "track_record.json", TRACK_RECORD_SCHEMA, checked)
    elif require_published:
        raise ContractError("track_record.json is required before upload.")
    return {"files": len(checked), "paths": checked}


def main(argv: list[str] | None = None) -> int:
    """Validate ``site/data``. Exit 1 when the contract fails."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate site/data JSON against the v1 contract."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("site/data"),
        help="Directory that holds sample.json, teams.json, and published JSON.",
    )
    parser.add_argument(
        "--require-published",
        action="store_true",
        help="Require latest.json, track_record.json, and one week file.",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        result: dict[str, Any] = validate_data_dir(
            args.data_dir,
            require_published=args.require_published,
        )
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Validated {result['files']} site data file(s) against contract {CONTRACT_VERSION}.")
    return 0


def _validate_named(root: Path, name: str, schema_name: str, checked: list[str]) -> None:
    """Validate one file that must exist directly under ``root``."""
    path: Path = root / name
    if not path.is_file():
        raise ContractError(f"{name} is required.")
    _validate_file(root, path, schema_name, checked)


def _validate_file(root: Path, path: Path, schema_name: str, checked: list[str]) -> None:
    """Decode one UTF-8 JSON document and validate it."""
    label: str = _label(root, path)
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: not valid JSON ({exc}).") from exc
    try:
        validate_instance(payload, schema_name)
    except ContractError as exc:
        raise ContractError(f"{label}: {exc}") from exc
    checked.append(label)


def _label(root: Path, path: Path) -> str:
    """Path relative to the data directory, for error text."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    sys.exit(main())
