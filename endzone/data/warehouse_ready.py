"""Start the SQL warehouse and wait until it is RUNNING.

Pages calls this before ingest, dbt, and scoring. The warehouse id is the
last segment of ``DATABRICKS_HTTP_PATH`` (``/sql/1.0/warehouses/<id>``).
The host, the path, the id, and the token are not written to the log.
A stopped warehouse is started. A warehouse that does not reach RUNNING
inside the timeout fails the job. Published slates are not touched.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from endzone.config import load_databricks_settings
from endzone.data.warehouse import connect_workspace
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_TIMEOUT_SECONDS: float = 600.0
DEFAULT_POLL_SECONDS: float = 10.0
_WAREHOUSE_PATH: re.Pattern[str] = re.compile(
    r"^/sql/1\.0/warehouses/(?P<warehouse_id>[A-Za-z0-9-]+)/?$"
)
_TERMINAL_STATES: frozenset[str] = frozenset({"DELETED", "DELETING"})
_READY_STATE: str = "RUNNING"


class WarehouseNotReadyError(RuntimeError):
    """The SQL warehouse cannot be used. The message has no credentials."""


def warehouse_id_from_http_path(http_path: str) -> str:
    """Return the warehouse id from a SQL HTTP path.

    Raises:
        WarehouseNotReadyError: The path is not ``/sql/1.0/warehouses/<id>``.
            The path itself is not included in the message.
    """
    match: re.Match[str] | None = _WAREHOUSE_PATH.fullmatch(http_path.strip())
    if match is None:
        raise WarehouseNotReadyError(
            "DATABRICKS_HTTP_PATH is not a SQL warehouse path "
            "(/sql/1.0/warehouses/<id>). The warehouse was not started."
        )
    return match.group("warehouse_id")


def ensure_warehouse_running(
    warehouses: Any,
    warehouse_id: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait until the warehouse is RUNNING, starting it when it is stopped.

    ``warehouses`` is the SDK warehouses API (``get`` and ``start``).
    STOPPING is waited out, then started. STARTING is only waited on.
    DELETED and DELETING fail immediately.

    Returns:
        ``state`` (``RUNNING``) and ``started`` (true when this call requested a start).

    Raises:
        WarehouseNotReadyError: The state is terminal, the API call fails, or
            RUNNING does not arrive before ``timeout_seconds``.
    """
    if timeout_seconds <= 0:
        raise WarehouseNotReadyError("The warehouse start timeout must be greater than 0 seconds.")
    deadline: float = clock() + timeout_seconds
    started: bool = False
    while True:
        state: str = _read_state(warehouses, warehouse_id)
        if state == _READY_STATE:
            logger.info("SQL warehouse is RUNNING.")
            return {"state": state, "started": started}
        if state in _TERMINAL_STATES:
            raise WarehouseNotReadyError(
                f"The SQL warehouse is {state}. "
                "Published slates were not changed. "
                "Restore a warehouse in the Databricks SQL warehouses page and rerun."
            )
        if clock() >= deadline:
            shown: str = state if state else "UNKNOWN"
            raise WarehouseNotReadyError(
                "The SQL warehouse did not reach RUNNING within "
                f"{int(timeout_seconds)} seconds (last state {shown}). "
                "Start it in the Databricks SQL warehouses page and rerun. "
                "Published slates were not changed."
            )
        if state == "STOPPED" and not started:
            _request_start(warehouses, warehouse_id)
            started = True
            logger.info("SQL warehouse was stopped. Start requested.")
        elif state == "STOPPING":
            logger.info("SQL warehouse is stopping. Waiting to start it.")
        elif state == "STARTING":
            logger.info("SQL warehouse is starting.")
        else:
            logger.info("SQL warehouse state is %s. Waiting.", state or "UNKNOWN")
        sleeper(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    """Start the configured warehouse and wait. Exit 1 when it is not RUNNING."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Start the Databricks SQL warehouse and wait until it is RUNNING."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for RUNNING. The default is 600.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="Seconds between state checks. The default is 10.",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        _run(timeout_seconds=args.timeout_seconds, poll_seconds=args.poll_seconds)
    except WarehouseNotReadyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _run(*, timeout_seconds: float, poll_seconds: float) -> None:
    """Load credentials from the environment and wait for RUNNING."""
    settings = load_databricks_settings()
    if not settings.host or not settings.http_path or not settings.token:
        raise WarehouseNotReadyError(
            "DATABRICKS_HOST, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN must be set. "
            "The warehouse was not started."
        )
    warehouse_id: str = warehouse_id_from_http_path(settings.http_path)
    client: Any = connect_workspace(settings.host, settings.token)
    ensure_warehouse_running(
        client.warehouses,
        warehouse_id,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def _read_state(warehouses: Any, warehouse_id: str) -> str:
    """Return the upper-case state name. API failures stay credential-free."""
    try:
        info: Any = warehouses.get(warehouse_id)
    except OSError as exc:
        raise WarehouseNotReadyError(
            "Could not read the SQL warehouse state "
            f"({_api_code(exc)}). Published slates were not changed."
        ) from None
    return _state_name(getattr(info, "state", None))


def _request_start(warehouses: Any, warehouse_id: str) -> None:
    """POST a start. A race with STARTING is left for the next poll."""
    try:
        warehouses.start(warehouse_id)
    except OSError as exc:
        logger.info("Warehouse start returned %s. Waiting for RUNNING.", _api_code(exc))


def _state_name(state: object) -> str:
    """Normalize an SDK enum or a plain string to an upper-case name."""
    if state is None:
        return ""
    value: object = getattr(state, "value", state)
    return str(value).strip().upper()


def _api_code(exc: BaseException) -> str:
    """Return an error code. The exception text can contain the warehouse path."""
    code: object = getattr(exc, "error_code", None)
    if isinstance(code, str) and code.strip() != "":
        return code.strip()
    return type(exc).__name__


if __name__ == "__main__":
    sys.exit(main())
