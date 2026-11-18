"""Rerun a command when Databricks reports a transient failure.

Pages uses this around ingest, ``dbt build``, and scoring. A deadlock or
``TEMPORARILY_UNAVAILABLE`` is retried a bounded number of times. Any other
failure, including a refused published slate, stops on the first exit.
The child output is printed as the child wrote it. This module does not
add credentials to the command or the log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_ATTEMPTS: int = 3
DEFAULT_PAUSE_SECONDS: float = 20.0
_TAIL_CHARS: int = 65536
TRANSIENT_MARKERS: tuple[str, ...] = (
    "deadlock found when trying to get lock",
    "temporarily_unavailable",
)


class RetryError(ValueError):
    """The retry invocation itself is invalid."""


def is_transient_databricks_error(text: str) -> bool:
    """True when the output is a known transient Databricks failure.

    A message that says a slate drops a ``game_id`` is not retried, even
    if a deadlock string is also present.
    """
    folded: str = text.lower()
    if "game_id" in folded and "drop" in folded:
        return False
    return any(marker in folded for marker in TRANSIENT_MARKERS)


def run_with_retry(
    command: Sequence[str],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    runner: Callable[[Sequence[str]], dict[str, Any]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run ``command`` until it exits 0 or the failure is not transient.

    Returns:
        ``code`` and ``attempts`` (how many times the command ran).

    Raises:
        RetryError: ``attempts`` is below 1, or ``command`` is empty.
    """
    if attempts < 1:
        raise RetryError("attempts must be at least 1.")
    if not command:
        raise RetryError("A command is required after --.")
    invoke: Callable[[Sequence[str]], dict[str, Any]] = (
        runner if runner is not None else _subprocess
    )
    last_code: int = 1
    for attempt in range(1, attempts + 1):
        result: dict[str, Any] = invoke(command)
        last_code = int(result["code"])
        if last_code == 0:
            return {"code": 0, "attempts": attempt}
        output: str = str(result.get("output", ""))
        if attempt == attempts or not is_transient_databricks_error(output):
            return {"code": last_code, "attempts": attempt}
        logger.info(
            "Transient Databricks error. Retrying attempt %s of %s.",
            attempt + 1,
            attempts,
        )
        sleeper(pause_seconds)
    return {"code": last_code, "attempts": attempts}


def main(argv: list[str] | None = None) -> int:
    """Run the command after ``--``. Exit with the child's code."""
    raw: list[str] = list(sys.argv[1:] if argv is None else argv)
    option_args: list[str]
    command: list[str]
    if "--" in raw:
        split: int = raw.index("--")
        option_args = raw[:split]
        command = raw[split + 1 :]
    else:
        option_args = raw
        command = []
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Retry a command when Databricks reports a transient error."
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS)
    args: argparse.Namespace = parser.parse_args(option_args)
    if args.pause_seconds < 0:
        print("pause-seconds must be 0 or greater.", file=sys.stderr)
        return 2
    try:
        result: dict[str, Any] = run_with_retry(
            command,
            attempts=args.attempts,
            pause_seconds=args.pause_seconds,
        )
    except RetryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return int(result["code"])


def _subprocess(command: Sequence[str]) -> dict[str, Any]:
    """Run one command, streaming its output and keeping a tail for the check."""
    process: subprocess.Popen[str] = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail: str = ""
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        tail = (tail + line)[-_TAIL_CHARS:]
    code: int = process.wait()
    return {"code": code, "output": tail}


if __name__ == "__main__":
    sys.exit(main())
