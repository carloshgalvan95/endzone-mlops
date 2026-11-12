"""Contracts on the scorer marts and ingestion-time freshness for games.

DuckDB executes the mart SQL and the ``cast(null as <data_type>)`` query dbt
uses for a contract. Databricks is not queried. The type tokens are the cast
names both adapters accept. ``string`` describes as VARCHAR on DuckDB and is
the Spark name ``string``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import duckdb
import pytest
import yaml

pytestmark = pytest.mark.cross_platform

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_CONFIG_BLOCK: re.Pattern[str] = re.compile(r"\{\{.*?\}\}", re.DOTALL)

# Measured from the public schedules parquet on 2026-11-10 after the same
# pandas to_parquet(index=False) rewrite as download_nflverse_games.
# Strings stayed text, int32 columns stayed int32, and nullable int32 columns
# (scores, lines, temp, wind) became double. gameday stayed text.
# The warehouse was not described.
FCT_GAMES_CONTRACT: tuple[tuple[str, str], ...] = (
    ("game_id", "string"),
    ("season", "int"),
    ("week", "int"),
    ("game_type", "string"),
    ("away_team", "string"),
    ("home_team", "string"),
    ("away_score", "double"),
    ("home_score", "double"),
    ("away_team_franchise", "string"),
    ("home_team_franchise", "string"),
    ("home_won", "int"),
    ("home_margin", "double"),
    ("total_points", "double"),
    ("home_covered_spread", "int"),
    ("total_over", "int"),
    ("spread_line", "double"),
    ("total_line", "double"),
    ("away_moneyline", "double"),
    ("home_moneyline", "double"),
    ("gameday", "string"),
    ("weekday", "string"),
    ("gametime", "string"),
    ("location", "string"),
    ("roof", "string"),
    ("surface", "string"),
    ("temp", "double"),
    ("wind", "double"),
    ("away_rest", "int"),
    ("home_rest", "int"),
)

FCT_UPCOMING_GAMES_CONTRACT: tuple[tuple[str, str], ...] = (
    ("game_id", "string"),
    ("season", "int"),
    ("week", "int"),
    ("game_type", "string"),
    ("away_team", "string"),
    ("home_team", "string"),
    ("away_team_franchise", "string"),
    ("home_team_franchise", "string"),
    ("gameday", "string"),
    ("weekday", "string"),
    ("gametime", "string"),
    ("spread_line", "double"),
    ("total_line", "double"),
    ("away_moneyline", "double"),
    ("home_moneyline", "double"),
    ("location", "string"),
    ("roof", "string"),
    ("surface", "string"),
    ("temp", "double"),
    ("wind", "double"),
    ("away_rest", "int"),
    ("home_rest", "int"),
)

SCORER_CONTRACTS: dict[str, tuple[tuple[str, str], ...]] = {
    "fct_games": FCT_GAMES_CONTRACT,
    "fct_upcoming_games": FCT_UPCOMING_GAMES_CONTRACT,
}

DUCKDB_CAST_TYPES: dict[str, str] = {
    "string": "VARCHAR",
    "int": "INTEGER",
    "double": "DOUBLE",
    "timestamp": "TIMESTAMP",
}

WARN_AFTER_DAYS: int = 8
ERROR_AFTER_DAYS: int = 15


class DescribedColumn(NamedTuple):
    """One column name and the type DuckDB reports for it."""

    name: str
    data_type: str


def _read_yaml(relative: str) -> dict[str, object]:
    """Load one dbt YAML file as a mapping.

    Args:
        relative: Path relative to the repository root.

    Returns:
        The parsed document.
    """
    text: str = (REPO_ROOT / relative).read_text(encoding="utf-8")
    payload: object = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} did not parse as a mapping")
    return payload


def _models_by_name() -> dict[str, dict[str, object]]:
    """Return mart models keyed by name."""
    payload: dict[str, object] = _read_yaml("dbt/models/marts/schema.yml")
    models: object = payload["models"]
    if not isinstance(models, list):
        raise AssertionError("mart schema models is not a list")
    found: dict[str, dict[str, object]] = {}
    for model in models:
        if not isinstance(model, dict):
            raise AssertionError("a mart entry is not a mapping")
        name: object = model.get("name")
        if not isinstance(name, str):
            raise AssertionError("a mart is missing a name")
        found[name] = model
    return found


def _column_pairs(model: dict[str, object]) -> tuple[tuple[str, str], ...]:
    """Return ``(name, data_type)`` in YAML order.

    Args:
        model: One model entry from the mart schema.

    Returns:
        Column name and contract type, in listed order.
    """
    columns: object = model["columns"]
    if not isinstance(columns, list):
        raise AssertionError("model columns is not a list")
    pairs: list[tuple[str, str]] = []
    for column in columns:
        if not isinstance(column, dict):
            raise AssertionError("a column entry is not a mapping")
        name: object = column.get("name")
        data_type: object = column.get("data_type")
        if not isinstance(name, str) or not isinstance(data_type, str):
            raise AssertionError(f"column {name!r} is missing a data_type string")
        pairs.append((name, data_type))
    return tuple(pairs)


def _render_sql(relative: str, replacements: dict[str, str]) -> str:
    """Remove the config block and substitute dbt calls.

    Args:
        relative: SQL path relative to the repository root.
        replacements: Exact Jinja snippets and the SQL that replaces them.

    Returns:
        SQL DuckDB can execute.
    """
    text: str = (REPO_ROOT / relative).read_text(encoding="utf-8")
    rendered: str = _CONFIG_BLOCK.sub("", text, count=1)
    for needle, replacement in replacements.items():
        rendered = rendered.replace(needle, replacement)
    if "{{" in rendered or "}}" in rendered:
        raise AssertionError(f"Jinja remains in {relative}")
    return rendered


def _describe(connection: duckdb.DuckDBPyConnection, query: str) -> tuple[DescribedColumn, ...]:
    """Describe a query without writing a table.

    Args:
        connection: Open DuckDB connection.
        query: Select statement.

    Returns:
        Column names and DuckDB type names, in select order.
    """
    rows: list[tuple[object, ...]] = connection.execute(f"describe ({query})").fetchall()
    described: list[DescribedColumn] = []
    for row in rows:
        name = row[0]
        data_type = row[1]
        if not isinstance(name, str) or not isinstance(data_type, str):
            raise AssertionError(f"unexpected describe row {row!r}")
        described.append(DescribedColumn(name, data_type))
    return tuple(described)


def _contract_query(columns: tuple[tuple[str, str], ...]) -> str:
    """Build the empty cast query dbt compares to the model.

    Args:
        columns: Contract name and data_type pairs.

    Returns:
        A select of ``cast(null as <data_type>)`` for each column.
    """
    pieces: list[str] = [f"cast(null as {data_type}) as {name}" for name, data_type in columns]
    return "select " + ", ".join(pieces)


def _prepare_games(connection: duckdb.DuckDBPyConnection) -> None:
    """Create a two-row games source and the franchise seed.

    Types follow the rewritten schedules parquet: text, int32, and double.
    ``ingested_at`` is the load timestamp the bronze replace adds. Staging
    does not select it.

    Args:
        connection: Empty DuckDB connection.
    """
    connection.execute(
        """
        create table raw_games (
            game_id varchar,
            season integer,
            week integer,
            game_type varchar,
            away_team varchar,
            home_team varchar,
            away_score double,
            home_score double,
            gameday varchar,
            weekday varchar,
            gametime varchar,
            spread_line double,
            total_line double,
            away_moneyline double,
            home_moneyline double,
            location varchar,
            roof varchar,
            surface varchar,
            temp double,
            wind double,
            away_rest integer,
            home_rest integer,
            gsis double,
            pfr varchar,
            pff double,
            espn varchar,
            old_game_id varchar,
            ingested_at timestamp
        )
        """
    )
    connection.execute(
        """
        insert into raw_games values
        (
            '2024_01_KC_BUF', 2024, 1, 'REG', 'KC', 'BUF',
            17.0, 24.0, '2024-09-08', 'Sunday', '13:00',
            -3.5, 47.5, -150.0, 130.0, 'Home', 'outdoors', 'grass',
            72.0, 8.0, 7, 7, 2024090801.0, 'pfr', 1.0, 'espn', 'old',
            timestamp '2026-11-10 12:00:00'
        ),
        (
            '2026_10_DAL_PHI', 2026, 10, 'REG', 'DAL', 'PHI',
            null, null, '2026-11-15', 'Sunday', '16:25',
            null, null, null, null, 'Home', 'outdoors', 'grass',
            null, null, 7, 7, null, null, null, null, null,
            timestamp '2026-11-10 12:00:00'
        )
        """
    )
    connection.execute(
        "create table team_franchise_map (team_abbr varchar, franchise_abbr varchar)"
    )
    connection.execute(
        """
        insert into team_franchise_map values
        ('KC', 'KC'), ('BUF', 'BUF'), ('DAL', 'DAL'), ('PHI', 'PHI')
        """
    )
    staging_sql: str = _render_sql(
        "dbt/models/staging/stg_nflverse__games.sql",
        {"{{ source('nflverse', 'raw_games') }}": "raw_games"},
    )
    connection.execute(f"create table stg_nflverse__games as {staging_sql}")


def freshness_level(age_days: int, warn_after: int, error_after: int) -> str:
    """Classify an age in days the way the games freshness windows do.

    Args:
        age_days: Whole days since ``max(ingested_at)``.
        warn_after: First age that warns.
        error_after: First age that errors.

    Returns:
        ``pass``, ``warn``, or ``error``.
    """
    if age_days >= error_after:
        return "error"
    if age_days >= warn_after:
        return "warn"
    return "pass"


class TestScorerContracts:
    """fct_games and fct_upcoming_games keep an enforced column contract."""

    @pytest.mark.parametrize("model_name", ["fct_games", "fct_upcoming_games"])
    def test_yaml_contract_matches_the_measured_types(self, model_name: str) -> None:
        """Each scorer mart enforces the measured name and data_type list."""
        model: dict[str, object] = _models_by_name()[model_name]
        config: object = model["config"]
        if not isinstance(config, dict):
            raise AssertionError(f"{model_name} config is missing")
        contract: object = config.get("contract")
        if not isinstance(contract, dict):
            raise AssertionError(f"{model_name} has no contract")
        assert contract.get("enforced") is True
        assert "store_failures" not in config
        assert _column_pairs(model) == SCORER_CONTRACTS[model_name]

    def test_other_marts_are_not_given_this_contract(self) -> None:
        """The contract stays on the two marts the scorer reads."""
        models: dict[str, dict[str, object]] = _models_by_name()
        for name, model in models.items():
            if name in SCORER_CONTRACTS:
                continue
            config: object = model.get("config")
            if isinstance(config, dict):
                assert "contract" not in config

    def test_duckdb_model_types_match_the_contract_casts(self) -> None:
        """DuckDB describes the mart SQL the same way it describes the contract."""
        connection: duckdb.DuckDBPyConnection = duckdb.connect()
        try:
            _prepare_games(connection)
            replacements: dict[str, str] = {
                "{{ ref('stg_nflverse__games') }}": "stg_nflverse__games",
                "{{ ref('team_franchise_map') }}": "team_franchise_map",
            }
            sql_by_model: dict[str, str] = {
                "fct_games": "dbt/models/marts/fct_games.sql",
                "fct_upcoming_games": "dbt/models/marts/fct_upcoming_games.sql",
            }
            for model_name, relative in sql_by_model.items():
                columns: tuple[tuple[str, str], ...] = SCORER_CONTRACTS[model_name]
                model_types: tuple[DescribedColumn, ...] = _describe(
                    connection, _render_sql(relative, replacements)
                )
                contract_types: tuple[DescribedColumn, ...] = _describe(
                    connection, _contract_query(columns)
                )
                assert model_types == contract_types
                assert [column.name for column in model_types] == [name for name, _ in columns]
        finally:
            connection.close()

    @pytest.mark.parametrize(
        ("token", "duckdb_type"),
        [
            ("string", "VARCHAR"),
            ("int", "INTEGER"),
            ("double", "DOUBLE"),
            ("timestamp", "TIMESTAMP"),
        ],
    )
    def test_cast_token_on_duckdb_and_databricks_labels(self, token: str, duckdb_type: str) -> None:
        """Shared cast names: DuckDB describes them, Databricks keeps the label."""
        from dbt.adapters.databricks.column import DatabricksColumn

        assert DatabricksColumn.translate_type(token) == token
        connection: duckdb.DuckDBPyConnection = duckdb.connect()
        try:
            described: str = connection.execute(
                f"describe (select cast(null as {token}) as c)"
            ).fetchone()[1]
        finally:
            connection.close()
        assert described == duckdb_type
        assert DUCKDB_CAST_TYPES[token] == duckdb_type


class TestGamesFreshness:
    """Games freshness is the bronze replace time, not the schedule date."""

    def test_only_games_declares_freshness(self) -> None:
        """raw_games warns at 8 days and errors at 15. Other sources do not."""
        payload: dict[str, object] = _read_yaml("dbt/models/staging/sources.yml")
        sources: object = payload["sources"]
        if not isinstance(sources, list) or len(sources) != 1:
            raise AssertionError("expected one nflverse source")
        source: object = sources[0]
        if not isinstance(source, dict):
            raise AssertionError("source entry is not a mapping")
        assert "freshness" not in source
        assert "loaded_at_field" not in source.get("config", {})
        tables: object = source["tables"]
        if not isinstance(tables, list):
            raise AssertionError("source tables is not a list")
        seen: set[str] = set()
        for table in tables:
            if not isinstance(table, dict):
                raise AssertionError("a source table is not a mapping")
            name: object = table.get("name")
            if not isinstance(name, str):
                raise AssertionError("a source table is missing a name")
            seen.add(name)
            config: object = table.get("config") or {}
            if not isinstance(config, dict):
                raise AssertionError(f"{name} config is not a mapping")
            if name != "raw_games":
                assert "freshness" not in config
                assert "loaded_at_field" not in config
                assert "loaded_at_query" not in config
                continue
            assert config.get("loaded_at_field") == "ingested_at"
            assert "loaded_at_query" not in config
            freshness: object = config.get("freshness")
            if not isinstance(freshness, dict):
                raise AssertionError("raw_games freshness is missing")
            assert freshness.get("warn_after") == {"count": WARN_AFTER_DAYS, "period": "day"}
            assert freshness.get("error_after") == {"count": ERROR_AFTER_DAYS, "period": "day"}
            assert "gameday" not in str(freshness)
        assert seen == {"raw_games", "raw_play_by_play", "raw_depth_charts", "raw_injuries"}

    def test_staging_does_not_select_the_load_timestamp(self) -> None:
        """ingested_at is not in the nflverse parquet fixture and not in staging."""
        staging: str = (REPO_ROOT / "dbt/models/staging/stg_nflverse__games.sql").read_text(
            encoding="utf-8"
        )
        assert "ingested_at" not in staging
        fixture: str = (REPO_ROOT / "tests/fixtures/nflverse_games_columns.txt").read_text(
            encoding="utf-8"
        )
        names: set[str] = {line.strip() for line in fixture.splitlines() if line.strip()}
        assert "ingested_at" not in names
        assert "gameday" in names

    def test_max_ingested_at_ignores_a_future_gameday(self) -> None:
        """A future schedule date does not make a stale load look fresh."""
        connection: duckdb.DuckDBPyConnection = duckdb.connect()
        try:
            connection.execute(
                """
                create table raw_games (
                    gameday varchar,
                    ingested_at timestamp
                )
                """
            )
            connection.execute(
                """
                insert into raw_games values
                ('1999-09-12', timestamp '2026-11-01 12:00:00'),
                ('2026-12-25', timestamp '2026-11-01 12:00:00')
                """
            )
            loaded: object = connection.execute(
                "select max(ingested_at) from raw_games"
            ).fetchone()[0]
            latest_gameday: object = connection.execute(
                "select max(gameday) from raw_games"
            ).fetchone()[0]
        finally:
            connection.close()
        assert str(loaded).startswith("2026-11-01")
        assert latest_gameday == "2026-12-25"

    @pytest.mark.parametrize(
        ("age_days", "level"),
        [
            (7, "pass"),
            (8, "warn"),
            (14, "warn"),
            (15, "error"),
        ],
    )
    def test_weekly_reload_window(self, age_days: int, level: str) -> None:
        """Seven days still passes. Eight warns. Fifteen errors."""
        assert freshness_level(age_days, WARN_AFTER_DAYS, ERROR_AFTER_DAYS) == level
