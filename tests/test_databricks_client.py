"""Tests for DatabricksClient SQL execution.

The warehouse SQL API accepts one statement per cursor.execute. These tests
mock databricks.sql and assert that query and read_table never send a
USE script, and that table names are fully qualified.
"""

import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.databricks_client import (
    DEFAULT_CATALOG,
    DEFAULT_MARTS_SCHEMA,
    DatabricksClient,
)


def _clear_databricks_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop warehouse settings so tests do not inherit the process environment."""
    for name in (
        "DATABRICKS_HOST",
        "DATABRICKS_HTTP_PATH",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CATALOG",
        "DATABRICKS_MARTS_SCHEMA",
    ):
        monkeypatch.delenv(name, raising=False)


def _install_fake_connector(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake databricks.sql module and return (connect, cursor)."""
    cursor = MagicMock()
    cursor.description = [("game_id",), ("season",)]
    cursor.fetchall.return_value = [("G1", 2024)]

    connection = MagicMock()
    connection.cursor.return_value = cursor

    connect = MagicMock(return_value=connection)
    sql_module = types.ModuleType("databricks.sql")
    sql_module.connect = connect

    databricks_module = sys.modules.get("databricks")
    if databricks_module is None:
        databricks_module = types.ModuleType("databricks")
        monkeypatch.setitem(sys.modules, "databricks", databricks_module)
    monkeypatch.setattr(databricks_module, "sql", sql_module, raising=False)
    monkeypatch.setitem(sys.modules, "databricks.sql", sql_module)
    return connect, cursor


def _executed_sql(cursor: MagicMock) -> str:
    """Return the single SQL string passed to cursor.execute."""
    assert cursor.execute.call_count == 1
    statement: str = cursor.execute.call_args.args[0]
    assert isinstance(statement, str)
    return statement


def _assert_one_statement(statement: str) -> None:
    """A warehouse execute payload is one statement and does not start with USE."""
    body: str = statement.strip().rstrip(";").strip()
    assert ";" not in body
    assert "USE CATALOG" not in statement.upper()
    assert "USE SCHEMA" not in statement.upper()


@pytest.fixture
def connected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DatabricksClient, MagicMock, MagicMock]:
    """Client with placeholder credentials and a mocked SQL connector."""
    _clear_databricks_env(monkeypatch)
    connect, cursor = _install_fake_connector(monkeypatch)
    client = DatabricksClient(
        host="example.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/example",
        token="token-example",
        catalog="workspace",
        schema="nfl_dev_marts",
    )
    return client, connect, cursor


class TestDatabricksClientDefaults:
    """Catalog and schema defaults follow the dbt databricks target."""

    def test_default_schema_matches_dbt_marts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset env uses workspace and nfl_dev_marts."""
        _clear_databricks_env(monkeypatch)
        client = DatabricksClient()
        assert client.catalog == DEFAULT_CATALOG == "workspace"
        assert client.schema == DEFAULT_MARTS_SCHEMA == "nfl_dev_marts"

    def test_env_overrides_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DATABRICKS_CATALOG and DATABRICKS_MARTS_SCHEMA override the defaults."""
        _clear_databricks_env(monkeypatch)
        monkeypatch.setenv("DATABRICKS_CATALOG", "analytics")
        monkeypatch.setenv("DATABRICKS_MARTS_SCHEMA", "gold_marts")
        client = DatabricksClient(
            host="example.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/example",
            token="token-example",
        )
        assert client.catalog == "analytics"
        assert client.schema == "gold_marts"

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor arguments win over environment variables."""
        _clear_databricks_env(monkeypatch)
        monkeypatch.setenv("DATABRICKS_CATALOG", "analytics")
        monkeypatch.setenv("DATABRICKS_MARTS_SCHEMA", "gold_marts")
        client = DatabricksClient(catalog="workspace", schema="nfl_dev_marts")
        assert client.catalog == "workspace"
        assert client.schema == "nfl_dev_marts"


class TestDatabricksClientQuery:
    """query() sends exactly one statement and sets catalog on connect."""

    def test_query_executes_one_statement(
        self, connected_client: tuple[DatabricksClient, MagicMock, MagicMock]
    ) -> None:
        """The caller's SQL is executed once, with no USE prefix."""
        client, connect, cursor = connected_client
        statement = "SELECT * FROM fct_games WHERE season = 2024"
        frame: pd.DataFrame = client.query(statement)

        connect.assert_called_once_with(
            server_hostname="example.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/example",
            access_token="token-example",
            catalog="workspace",
            schema="nfl_dev_marts",
        )
        executed: str = _executed_sql(cursor)
        assert executed == statement
        _assert_one_statement(executed)
        assert list(frame.columns) == ["game_id", "season"]
        assert frame.iloc[0]["game_id"] == "G1"

    def test_second_query_reuses_connection(
        self, connected_client: tuple[DatabricksClient, MagicMock, MagicMock]
    ) -> None:
        """Each execute is still one statement when the connection is reused."""
        client, connect, cursor = connected_client
        client.query("SELECT 1")
        client.query("SELECT 2")

        connect.assert_called_once()
        assert cursor.execute.call_count == 2
        for call in cursor.execute.call_args_list:
            _assert_one_statement(call.args[0])


class TestDatabricksClientReadTable:
    """read_table qualifies names and still executes one statement."""

    def test_read_table_qualifies_bare_name(
        self, connected_client: tuple[DatabricksClient, MagicMock, MagicMock]
    ) -> None:
        """A bare marts table becomes catalog.schema.table in backticks."""
        client, _connect, cursor = connected_client
        client.read_table("fct_games")

        executed: str = _executed_sql(cursor)
        assert executed == "SELECT * FROM `workspace`.`nfl_dev_marts`.`fct_games`"
        _assert_one_statement(executed)

    def test_read_table_qualifies_selected_columns(
        self, connected_client: tuple[DatabricksClient, MagicMock, MagicMock]
    ) -> None:
        """Column lists stay unqualified. Only the table name is qualified."""
        client, _connect, cursor = connected_client
        client.read_table("fct_games", columns=["game_id", "season", "home_won"])

        executed: str = _executed_sql(cursor)
        assert (
            executed
            == "SELECT game_id, season, home_won FROM `workspace`.`nfl_dev_marts`.`fct_games`"
        )
        _assert_one_statement(executed)

    def test_read_table_uses_env_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env overrides are quoted into the single SELECT."""
        _clear_databricks_env(monkeypatch)
        _connect, cursor = _install_fake_connector(monkeypatch)
        monkeypatch.setenv("DATABRICKS_CATALOG", "analytics")
        monkeypatch.setenv("DATABRICKS_MARTS_SCHEMA", "gold_marts")
        client = DatabricksClient(
            host="example.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/example",
            token="token-example",
        )
        client.read_table("fct_team_game_epa")

        executed: str = _executed_sql(cursor)
        assert executed == "SELECT * FROM `analytics`.`gold_marts`.`fct_team_game_epa`"
        _assert_one_statement(executed)
        assert _connect.call_args.kwargs["catalog"] == "analytics"
        assert _connect.call_args.kwargs["schema"] == "gold_marts"

    def test_already_qualified_name_is_not_prefixed(
        self, connected_client: tuple[DatabricksClient, MagicMock, MagicMock]
    ) -> None:
        """A three-part name is quoted in place and is not prefixed again."""
        client, _connect, cursor = connected_client
        client.read_table("workspace.nfl_dev_marts.fct_games")

        executed: str = _executed_sql(cursor)
        assert executed == "SELECT * FROM `workspace`.`nfl_dev_marts`.`fct_games`"
        assert executed.count("`workspace`") == 1
        _assert_one_statement(executed)


class TestDatabricksClientFixtureMode:
    """Missing credentials stay in fixture mode and do not open a connection."""

    def test_query_refuses_fixture_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """query() tells the caller to use a fixture instead of connecting."""
        _clear_databricks_env(monkeypatch)
        client = DatabricksClient()
        with pytest.raises(RuntimeError, match="fixture mode"):
            client.query("SELECT 1")
