#!/usr/bin/env python3
"""Fail when dbt is configured to store one failure relation per test.

Unity Catalog counts tables and views toward the same per-schema quota.
``store_failures`` materializes one relation for every test it covers,
including a test that passes. The project default stays off. A named test
may store failures only when it is listed in ``ALLOWED_STORED_FAILURE_TESTS``
and the total stays within ``MAX_STORED_FAILURE_RELATIONS``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_STORED_FAILURE_RELATIONS: int = 0
ALLOWED_STORED_FAILURE_TESTS: frozenset[str] = frozenset()

_SKIP_DIRECTORIES: frozenset[str] = frozenset({"dbt_packages", "target", "logs"})
_SKIP_FILES: frozenset[str] = frozenset({"profiles.yml", "package-lock.yml"})
_TEST_LIST_KEYS: frozenset[str] = frozenset({"tests", "data_tests"})
_RELATION_TOKENS: frozenset[str] = frozenset({"true", "table", "view"})
_OFF_TOKENS: frozenset[str] = frozenset({"", "false", "none", "null"})
_CONFIG_WORD = re.compile(r"config", re.IGNORECASE)
_STORE_ARGUMENT = re.compile(
    r"""store_failures\s*=\s*(?P<raw>true|false|table|view|ephemeral|"[^"]*"|'[^']*')""",
    re.IGNORECASE,
)
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SQL_LINE_COMMENT = re.compile(r"--.*?$", re.MULTILINE)
_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)


@dataclass(frozen=True)
class StoreFailureScan:
    """Stored-failure relations configured in a dbt project, plus any violations.

    ``relation_count`` counts per-test settings that create a table or a view.
    A project or directory setting is a violation on its own: one flag would
    create a relation for every test, so it is not added into ``relation_count``.
    """

    violations: tuple[str, ...]
    relation_count: int


def _classify(value: object) -> str:
    """Return ``off``, ``relation``, or ``enabled`` for a store_failures value.

    Args:
        value: The YAML or config value attached to ``store_failures``.

    Returns:
        ``off`` when the setting stores nothing, ``relation`` when it creates
        a table or a view, and ``enabled`` for any other non-false value.
    """
    if value is True:
        return "relation"
    if value is False or value is None:
        return "off"
    if isinstance(value, str):
        token: str = value.strip().strip("\"'").lower()
        if token in _OFF_TOKENS:
            return "off"
        if token in _RELATION_TOKENS:
            return "relation"
        return "enabled"
    return "enabled"


def _render(value: object) -> str:
    """Return a short display form of a store_failures value.

    Args:
        value: The raw setting.

    Returns:
        ``true``, ``false``, the stripped string, or ``repr`` for other objects.
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value.strip().strip("\"'")
    return repr(value)


def _bare_key(key: str) -> str:
    """Return a dbt config key without the project-file ``+`` prefix.

    Args:
        key: A YAML mapping key.

    Returns:
        The key with one leading ``+`` removed.
    """
    if key.startswith("+"):
        return key[1:]
    return key


def _iter_project_files(dbt_dir: Path) -> Iterator[Path]:
    """Yield project YAML and SQL files, skipping packages, logs, and profiles.

    Args:
        dbt_dir: The ``dbt/`` directory.

    Yields:
        Files whose suffix is ``.yml``, ``.yaml``, or ``.sql``.
    """
    for path in sorted(dbt_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.name in _SKIP_FILES:
            continue
        if path.suffix.lower() in {".yml", ".yaml", ".sql"}:
            yield path


def _strip_sql_comments(text: str) -> str:
    """Remove SQL and Jinja comments so a comment cannot enable the flag.

    Args:
        text: SQL file contents.

    Returns:
        The same text with comments replaced by spaces.
    """
    without_blocks: str = _SQL_BLOCK_COMMENT.sub(" ", text)
    without_jinja: str = _JINJA_COMMENT.sub(" ", without_blocks)
    return _SQL_LINE_COMMENT.sub(" ", without_jinja)


def _config_bodies(text: str) -> list[str]:
    """Return the argument text of each ``config(...)`` call.

    Args:
        text: SQL with comments already removed.

    Returns:
        The interior of each ``config`` call, in order.
    """
    bodies: list[str] = []
    cursor: int = 0
    for match in _CONFIG_WORD.finditer(text):
        if match.start() < cursor:
            continue
        if match.start() > 0 and (
            text[match.start() - 1].isalnum() or text[match.start() - 1] == "_"
        ):
            continue
        index: int = match.end()
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != "(":
            continue
        depth: int = 0
        body_start: int = index + 1
        closed: bool = False
        for position in range(index, len(text)):
            character: str = text[position]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    bodies.append(text[body_start:position])
                    cursor = position + 1
                    closed = True
                    break
        if not closed:
            break
    return bodies


def _parse_config_token(raw: str) -> object:
    """Turn a ``config()`` token into a bool or a plain string.

    Args:
        raw: The matched token, including optional quotes.

    Returns:
        ``True`` or ``False`` for those words, otherwise the unquoted text.
    """
    token: str = raw.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1]
    lowered: str = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return token


def _next_test_id(
    relative: str, trail: tuple[str, ...], test_name: str, used: dict[str, int]
) -> str:
    """Build a stable id for one test entry and disambiguate repeats.

    Args:
        relative: Project path of the file, with forward slashes.
        trail: Resource and column names above the test list.
        test_name: The test macro or singular name.
        used: Counts of ids already issued in this file.

    Returns:
        An id such as ``models/schema.yml:fct_games.game_id.not_null``.
        A second test with the same path gets a ``#2`` suffix.
    """
    qualified: str = ".".join((*trail, test_name)) if trail else test_name
    base: str = f"{relative}:{qualified}"
    used[base] = used.get(base, 0) + 1
    occurrence: int = used[base]
    if occurrence == 1:
        return base
    return f"{base}#{occurrence}"


def _record(
    *,
    location: str,
    value: object,
    test_id: str | None,
    allowed_tests: frozenset[str],
    relations: list[str],
    violations: list[str],
    cap: int,
) -> None:
    """Record one store_failures setting.

    Args:
        location: File path shown in a violation.
        value: The configured value.
        test_id: The single test this setting belongs to, or None when the
            setting covers every test under a project or directory key.
        allowed_tests: Test ids permitted to store failures.
        relations: Per-test table or view settings found so far.
        violations: Messages collected so far.
        cap: Maximum number of stored-failure relations.
    """
    kind: str = _classify(value)
    if kind == "off":
        return
    rendered: str = _render(value)
    if test_id is None:
        violations.append(
            f"{location} sets store_failures to {rendered} for every test under that config. "
            "A project or directory setting stores one relation per test. "
            "Tables and views both count toward the Unity Catalog schema quota. "
            "Leave store_failures unset or false."
        )
        return
    if kind == "relation":
        relations.append(test_id)
    if test_id not in allowed_tests:
        violations.append(
            f"{location} test {test_id} sets store_failures to {rendered}. "
            "Stored failures stay off unless that test is named in "
            "ALLOWED_STORED_FAILURE_TESTS in scripts/check_dbt_store_failures.py "
            f"and the relation count stays within the cap of {cap}."
        )


def _walk(
    node: object,
    *,
    relative: str,
    trail: tuple[str, ...],
    test_id: str | None,
    used: dict[str, int],
    allowed_tests: frozenset[str],
    relations: list[str],
    violations: list[str],
    cap: int,
) -> None:
    """Walk a YAML node and record store_failures settings.

    Args:
        node: A decoded YAML value.
        relative: File path relative to the dbt directory.
        trail: Named parents used to build a test id.
        test_id: Set when ``node`` is inside one test entry.
        used: Repeat counter for test ids in this file.
        allowed_tests: Test ids permitted to store failures.
        relations: Per-test relation settings found so far.
        violations: Messages collected so far.
        cap: Maximum number of stored-failure relations.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            bare: str = _bare_key(key)
            if bare == "store_failures":
                _record(
                    location=relative,
                    value=value,
                    test_id=test_id,
                    allowed_tests=allowed_tests,
                    relations=relations,
                    violations=violations,
                    cap=cap,
                )
                continue
            if bare in _TEST_LIST_KEYS and isinstance(value, list):
                _walk_tests(
                    value,
                    relative=relative,
                    trail=trail,
                    used=used,
                    allowed_tests=allowed_tests,
                    relations=relations,
                    violations=violations,
                    cap=cap,
                )
                continue
            _walk(
                value,
                relative=relative,
                trail=trail,
                test_id=test_id,
                used=used,
                allowed_tests=allowed_tests,
                relations=relations,
                violations=violations,
                cap=cap,
            )
        return
    if isinstance(node, list):
        for item in node:
            next_trail: tuple[str, ...] = trail
            if isinstance(item, dict):
                name: object = item.get("name")
                if isinstance(name, str) and name:
                    next_trail = trail + (name,)
            _walk(
                item,
                relative=relative,
                trail=next_trail,
                test_id=test_id,
                used=used,
                allowed_tests=allowed_tests,
                relations=relations,
                violations=violations,
                cap=cap,
            )


def _walk_tests(
    items: list[object],
    *,
    relative: str,
    trail: tuple[str, ...],
    used: dict[str, int],
    allowed_tests: frozenset[str],
    relations: list[str],
    violations: list[str],
    cap: int,
) -> None:
    """Walk a schema.yml ``tests`` list. Each entry is one test.

    Args:
        items: The test list.
        relative: File path relative to the dbt directory.
        trail: Resource and column names above the list.
        used: Repeat counter for test ids in this file.
        allowed_tests: Test ids permitted to store failures.
        relations: Per-test relation settings found so far.
        violations: Messages collected so far.
        cap: Maximum number of stored-failure relations.
    """
    for item in items:
        if isinstance(item, str):
            continue
        if isinstance(item, dict) and len(item) == 1:
            test_name, body = next(iter(item.items()))
            if isinstance(test_name, str):
                test_id: str = _next_test_id(relative, trail, test_name, used)
                _walk(
                    body,
                    relative=relative,
                    trail=trail,
                    test_id=test_id,
                    used=used,
                    allowed_tests=allowed_tests,
                    relations=relations,
                    violations=violations,
                    cap=cap,
                )
                continue
        test_id = _next_test_id(relative, trail, "test", used)
        _walk(
            item,
            relative=relative,
            trail=trail,
            test_id=test_id,
            used=used,
            allowed_tests=allowed_tests,
            relations=relations,
            violations=violations,
            cap=cap,
        )


def _scan_yaml(
    path: Path,
    dbt_dir: Path,
    *,
    allowed_tests: frozenset[str],
    relations: list[str],
    violations: list[str],
    cap: int,
) -> None:
    """Scan one YAML file for store_failures settings.

    Args:
        path: The file to read.
        dbt_dir: The dbt directory, used to build a relative location.
        allowed_tests: Test ids permitted to store failures.
        relations: Per-test relation settings found so far.
        violations: Messages collected so far.
        cap: Maximum number of stored-failure relations.
    """
    relative: str = path.relative_to(dbt_dir).as_posix()
    try:
        text: str = path.read_text(encoding="utf-8")
        documents: list[object] = list(yaml.safe_load_all(text))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        violations.append(f"{relative} could not be read as YAML: {exc}")
        return
    used: dict[str, int] = {}
    for document in documents:
        _walk(
            document,
            relative=relative,
            trail=(),
            test_id=None,
            used=used,
            allowed_tests=allowed_tests,
            relations=relations,
            violations=violations,
            cap=cap,
        )


def _scan_sql(
    path: Path,
    dbt_dir: Path,
    *,
    allowed_tests: frozenset[str],
    relations: list[str],
    violations: list[str],
    cap: int,
) -> None:
    """Scan ``config()`` calls in one SQL file.

    A file under ``tests/`` is one singular test. A ``config()`` anywhere else
    is treated as a project-wide setting, because it is not one schema test.

    Args:
        path: The SQL file.
        dbt_dir: The dbt directory, used to build a relative location.
        allowed_tests: Test ids permitted to store failures.
        relations: Per-test relation settings found so far.
        violations: Messages collected so far.
        cap: Maximum number of stored-failure relations.
    """
    relative: str = path.relative_to(dbt_dir).as_posix()
    try:
        text: str = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        violations.append(f"{relative} could not be read: {exc}")
        return
    singular: str | None = relative if relative.startswith("tests/") else None
    for body in _config_bodies(_strip_sql_comments(text)):
        for match in _STORE_ARGUMENT.finditer(body):
            _record(
                location=relative,
                value=_parse_config_token(match.group("raw")),
                test_id=singular,
                allowed_tests=allowed_tests,
                relations=relations,
                violations=violations,
                cap=cap,
            )


def scan_store_failures(
    dbt_dir: Path,
    *,
    allowed_tests: frozenset[str] = ALLOWED_STORED_FAILURE_TESTS,
    max_relations: int = MAX_STORED_FAILURE_RELATIONS,
) -> StoreFailureScan:
    """Scan a dbt project for stored-failure configuration.

    Args:
        dbt_dir: The ``dbt/`` directory of the project.
        allowed_tests: Test ids that may store a failure relation.
        max_relations: Maximum number of table or view failure relations.

    Returns:
        The relation count and any violations. An empty violation tuple means
        the project does not enable store_failures broadly and stays within
        the cap.
    """
    relations: list[str] = []
    violations: list[str] = []
    if not dbt_dir.is_dir():
        return StoreFailureScan(
            violations=(f"missing dbt project directory: {dbt_dir}",),
            relation_count=0,
        )
    for path in _iter_project_files(dbt_dir):
        if path.suffix.lower() == ".sql":
            _scan_sql(
                path,
                dbt_dir,
                allowed_tests=allowed_tests,
                relations=relations,
                violations=violations,
                cap=max_relations,
            )
        else:
            _scan_yaml(
                path,
                dbt_dir,
                allowed_tests=allowed_tests,
                relations=relations,
                violations=violations,
                cap=max_relations,
            )
    if len(relations) > max_relations:
        violations.append(
            f"stored-failure relations configured: {len(relations)}. The cap is {max_relations}."
        )
    return StoreFailureScan(violations=tuple(violations), relation_count=len(relations))


def main(argv: list[str] | None = None) -> int:
    """Check the dbt project and return a process status.

    Args:
        argv: Arguments, excluding the program name. Defaults to ``sys.argv``.

    Returns:
        ``0`` when store_failures stays within the cap, ``1`` otherwise.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Fail if dbt stores a failure relation for every test."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. The dbt project is <root>/dbt.",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    dbt_dir: Path = args.root.resolve() / "dbt"
    scan: StoreFailureScan = scan_store_failures(dbt_dir)
    if scan.violations:
        for violation in scan.violations:
            print(violation, file=sys.stderr)
        print(f"stored-failure relations: {scan.relation_count}", file=sys.stderr)
        return 1
    print(f"stored-failure relations: {scan.relation_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
