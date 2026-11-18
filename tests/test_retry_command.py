"""Bounded retry for transient Databricks errors. Other failures stay fail-closed."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from endzone.data.retry_command import (
    is_transient_databricks_error,
    main,
    run_with_retry,
)

pytestmark = pytest.mark.cross_platform


def test_markers_match_the_deadlock_and_refuse_a_dropped_slate() -> None:
    """A deadlock is transient. A dropped game_id is not, even next to that word."""
    assert is_transient_databricks_error(
        "Deadlock found when trying to get lock; try restarting transaction"
    )
    assert is_transient_databricks_error("TEMPORARILY_UNAVAILABLE")
    assert is_transient_databricks_error("bad column") is False
    assert (
        is_transient_databricks_error(
            "Deadlock found when trying to get lock and this slate drops a game_id"
        )
        is False
    )


def test_transient_failure_is_retried_then_succeeds() -> None:
    """The second attempt is the one that counts. The pause is invoked once."""
    calls: list[int] = []
    pauses: list[float] = []

    def runner(_command: Sequence[str]) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            return {"code": 1, "output": "Deadlock found when trying to get lock"}
        return {"code": 0, "output": ""}

    result: dict[str, Any] = run_with_retry(
        ["dbt", "build"],
        attempts=3,
        pause_seconds=20,
        runner=runner,
        sleeper=pauses.append,
    )
    assert result == {"code": 0, "attempts": 2}
    assert pauses == [20]


def test_ordinary_failure_is_not_retried() -> None:
    """A non-transient exit stops on the first attempt."""
    calls: list[int] = []

    def runner(_command: Sequence[str]) -> dict[str, Any]:
        calls.append(1)
        return {"code": 4, "output": "column missing"}

    result: dict[str, Any] = run_with_retry(
        ["dbt", "build"],
        attempts=3,
        runner=runner,
        sleeper=lambda _seconds: pytest.fail("should not pause"),
    )
    assert result == {"code": 4, "attempts": 1}
    assert calls == [1]


def test_cli_reruns_a_transient_child(tmp_path: Path) -> None:
    """The command after -- runs again when the child prints a deadlock."""
    script: Path = tmp_path / "child.py"
    counter: Path = tmp_path / "count"
    script.write_text(
        "import pathlib, sys\n"
        f"path = pathlib.Path({str(counter)!r})\n"
        "count = int(path.read_text(encoding='utf-8')) if path.exists() else 0\n"
        "count += 1\n"
        "path.write_text(str(count), encoding='utf-8')\n"
        "if count < 2:\n"
        "    sys.stderr.write('Deadlock found when trying to get lock\\n')\n"
        "    sys.exit(1)\n"
        "sys.stdout.write('ok\\n')\n",
        encoding="utf-8",
    )
    code: int = main(
        [
            "--attempts",
            "3",
            "--pause-seconds",
            "0",
            "--",
            sys.executable,
            str(script),
        ]
    )
    assert code == 0
    assert counter.read_text(encoding="utf-8") == "2"
