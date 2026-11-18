"""SQL warehouse start. No network, and no credential in the failure text."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from endzone.data.warehouse_ready import (
    WarehouseNotReadyError,
    ensure_warehouse_running,
    main,
    warehouse_id_from_http_path,
)

pytestmark = pytest.mark.cross_platform


class _Clock:
    """A clock that moves only when the waiter sleeps."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Warehouses:
    """Scripted get/start. The id is checked and never copied into errors."""

    def __init__(self, states: list[str], *, fail_start: bool = False) -> None:
        self.states: list[str] = list(states)
        self.starts: int = 0
        self.seen_ids: list[str] = []
        self.fail_start: bool = fail_start

    def get(self, warehouse_id: str) -> SimpleNamespace:
        self.seen_ids.append(warehouse_id)
        if len(self.states) > 1:
            state: str = self.states.pop(0)
        else:
            state = self.states[0]
        return SimpleNamespace(state=state)

    def start(self, warehouse_id: str) -> None:
        self.seen_ids.append(warehouse_id)
        self.starts += 1
        if self.fail_start:
            error: OSError = OSError("token dapi-secret path /sql/1.0/warehouses/wh-secret")
            error.error_code = "INVALID_STATE"  # type: ignore[attr-defined]
            raise error


def test_http_path_yields_the_warehouse_id() -> None:
    """The id is the last path segment. Anything else is refused without an echo."""
    assert warehouse_id_from_http_path("/sql/1.0/warehouses/abc123") == "abc123"
    assert warehouse_id_from_http_path("/sql/1.0/warehouses/abc123/") == "abc123"
    with pytest.raises(WarehouseNotReadyError, match="not a SQL warehouse path") as raised:
        warehouse_id_from_http_path("/sql/1.0/warehouses/abc123/extra")
    assert "abc123" not in str(raised.value)


def test_stopped_warehouse_is_started_once_and_waited_on() -> None:
    """STOPPED requests one start. STARTING only waits. RUNNING returns."""
    clock = _Clock()
    warehouses = _Warehouses(["STOPPED", "STARTING", "RUNNING"])
    result: dict[str, Any] = ensure_warehouse_running(
        warehouses,
        "wh-secret",
        timeout_seconds=60,
        poll_seconds=10,
        sleeper=clock.sleep,
        clock=clock,
    )
    assert result == {"state": "RUNNING", "started": True}
    assert warehouses.starts == 1
    assert clock.now == 20


def test_running_warehouse_is_not_started() -> None:
    """A warehouse that is already RUNNING is left alone."""
    warehouses = _Warehouses(["RUNNING"])
    result: dict[str, Any] = ensure_warehouse_running(
        warehouses,
        "wh-secret",
        sleeper=lambda _seconds: pytest.fail("should not wait"),
        clock=lambda: 0.0,
    )
    assert result["started"] is False
    assert warehouses.starts == 0


def test_deleted_warehouse_fails_before_a_start() -> None:
    """A deleted warehouse is a hard failure. The id stays out of the message."""
    warehouses = _Warehouses(["DELETED"])
    with pytest.raises(WarehouseNotReadyError, match="DELETED") as raised:
        ensure_warehouse_running(warehouses, "wh-secret", clock=lambda: 0.0)
    assert warehouses.starts == 0
    assert "wh-secret" not in str(raised.value)


def test_timeout_names_the_last_state_and_hides_the_id() -> None:
    """The wait is bounded. The message says how to rerun and what was not changed."""
    clock = _Clock()
    warehouses = _Warehouses(["STARTING"])
    with pytest.raises(WarehouseNotReadyError, match="30 seconds") as raised:
        ensure_warehouse_running(
            warehouses,
            "wh-secret",
            timeout_seconds=30,
            poll_seconds=10,
            sleeper=clock.sleep,
            clock=clock,
        )
    message: str = str(raised.value)
    assert "last state STARTING" in message
    assert "Published slates were not changed" in message
    assert "wh-secret" not in message
    assert clock.now >= 30


def test_api_error_reports_the_code_only() -> None:
    """A Databricks error code is logged. The exception text is not."""

    class _Broken:
        def get(self, warehouse_id: str) -> None:
            error: OSError = OSError("dapi-secret-value /sql/1.0/warehouses/wh-secret")
            error.error_code = "BAD_REQUEST"  # type: ignore[attr-defined]
            raise error

    with pytest.raises(WarehouseNotReadyError, match="BAD_REQUEST") as raised:
        ensure_warehouse_running(_Broken(), "wh-secret", clock=lambda: 0.0)
    message: str = str(raised.value)
    assert "dapi-secret-value" not in message
    assert "wh-secret" not in message


def test_main_failure_does_not_print_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI failure names the state and omits the host, path, and token."""
    monkeypatch.setenv("DATABRICKS_HOST", "adb-secret.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/wh-secret")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-secret-value")

    def _client(host: str, token: str) -> SimpleNamespace:
        assert host == "adb-secret.cloud.databricks.com"
        assert token == "dapi-secret-value"
        return SimpleNamespace(warehouses=_Warehouses(["DELETED"]))

    monkeypatch.setattr("endzone.data.warehouse_ready.connect_workspace", _client)
    code: int = main(["--timeout-seconds", "30", "--poll-seconds", "1"])
    assert code == 1
    captured = capsys.readouterr()
    combined: str = captured.out + captured.err
    assert "DELETED" in combined
    assert "dapi-secret-value" not in combined
    assert "wh-secret" not in combined
    assert "adb-secret" not in combined


def test_pages_starts_the_warehouse_before_dbt_and_scoring() -> None:
    """A stopped warehouse is started before the first statement that needs it."""
    text = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    start = text.index("endzone.data.warehouse_ready")
    assert start < text.index("endzone.ingest.load_raw_nflverse")
    assert start < text.index("dbt build")
    assert start < text.index("endzone.modeling.score")
    assert "endzone.data.retry_command" in text
