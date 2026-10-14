"""
Databricks data access layer for reading marts.

Provides abstraction for querying Databricks SQL warehouse with fallback to local fixtures.
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# dbt target databricks (profiles.yml.example) uses schema nfl_dev.
# dbt_project.yml sets models.marts +schema: marts. This repo has no
# generate_schema_name macro, so dbt's default macro appends the custom
# schema: nfl_dev_marts. Override with DATABRICKS_CATALOG and
# DATABRICKS_MARTS_SCHEMA.
DEFAULT_CATALOG: str = "workspace"
DEFAULT_MARTS_SCHEMA: str = "nfl_dev_marts"


def quote_identifier(identifier: str) -> str:
    """Wrap one SQL identifier in Databricks backticks.

    Args:
        identifier: Unquoted catalog, schema, or table name.

    Returns:
        Backtick-quoted identifier. Embedded backticks are doubled.
    """
    escaped: str = identifier.replace("`", "``")
    return f"`{escaped}`"


def qualify_table_name(catalog: str, schema: str, table_name: str) -> str:
    """Build one backticked table reference for a single SELECT.

    A bare name becomes ``catalog.schema.table``. A name that already
    contains dots is quoted part by part and is not prefixed again.

    Args:
        catalog: Unity Catalog name.
        schema: Schema that holds the marts tables.
        table_name: Bare table name or an already qualified name.

    Returns:
        One identifier safe to place after FROM.
    """
    stripped: str = table_name.strip()
    if "." in stripped:
        parts: list[str] = [part.strip().strip("`") for part in stripped.split(".") if part.strip()]
    else:
        parts = [catalog, schema, stripped.strip("`")]
    return ".".join(quote_identifier(part) for part in parts)


def _resolve_setting(explicit: str | None, env_name: str, default: str) -> str:
    """Prefer an explicit argument, then a non-empty env var, then a default.

    Args:
        explicit: Value passed by the caller. None or blank means unset.
        env_name: Environment variable that overrides the default.
        default: Value used when neither argument nor env var is set.

    Returns:
        The resolved setting.
    """
    if explicit is not None and explicit.strip() != "":
        return explicit
    from_env: str | None = os.getenv(env_name)
    if from_env is not None and from_env.strip() != "":
        return from_env.strip()
    return default


class DatabricksClient:
    """Client for reading data from Databricks SQL warehouse or local fixtures."""

    def __init__(
        self,
        host: str | None = None,
        http_path: str | None = None,
        token: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        """
        Initialize Databricks client.

        Args:
            host: Databricks host (without https://)
            http_path: SQL warehouse HTTP path
            token: Databricks access token
            catalog: Unity Catalog name. Defaults to DATABRICKS_CATALOG or workspace.
            schema: Marts schema. Defaults to DATABRICKS_MARTS_SCHEMA or nfl_dev_marts.

        Example:
            >>> client = DatabricksClient()  # Uses env vars or fixture mode
            >>> df = client.query("SELECT * FROM fct_games LIMIT 10")
        """
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        self.catalog = _resolve_setting(catalog, "DATABRICKS_CATALOG", DEFAULT_CATALOG)
        self.schema = _resolve_setting(schema, "DATABRICKS_MARTS_SCHEMA", DEFAULT_MARTS_SCHEMA)

        self._connection = None
        self._use_fixture = False

        if not all([self.host, self.http_path, self.token]):
            logger.warning(
                "Databricks credentials not provided. Using fixture mode for local testing."
            )
            self._use_fixture = True

    def _get_connection(self) -> Any:
        """Get or create Databricks SQL connection."""
        if self._use_fixture:
            raise RuntimeError("Cannot connect in fixture mode. Use read_fixture() instead.")

        if self._connection is None:
            try:
                from databricks import sql
            except ImportError as e:
                raise ImportError(
                    "databricks-sql-connector not installed. "
                    "Install with: pip install databricks-sql-connector"
                ) from e

            # Session catalog and schema are connection parameters.
            # The SQL warehouse rejects a script that starts with USE.
            self._connection = sql.connect(
                server_hostname=self.host,
                http_path=self.http_path,
                access_token=self.token,
                catalog=self.catalog,
                schema=self.schema,
            )
            logger.info(f"Connected to Databricks: {self.host}")

        return self._connection

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame

        Raises:
            RuntimeError: If in fixture mode (use read_fixture instead)

        Example:
            >>> df = client.query("SELECT * FROM fct_games WHERE season = 2024")
        """
        if self._use_fixture:
            raise RuntimeError(
                "Cannot query Databricks in fixture mode. "
                "Use read_fixture() for local testing or provide credentials."
            )

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # One statement per execute. Catalog and schema are set on the connection.
            statement: str = sql.strip()
            logger.debug(f"Executing SQL: {statement[:100]}...")
            cursor.execute(statement)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            df = pd.DataFrame(result, columns=columns)
            logger.info(f"Query returned {len(df)} rows")
            return df

        finally:
            cursor.close()

    def read_table(self, table_name: str, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Read entire table or specific columns.

        Args:
            table_name: Table name (e.g., fct_games)
            columns: Optional list of columns to select

        Returns:
            Table data as DataFrame

        Example:
            >>> df = client.read_table("fct_games", columns=["game_id", "season", "home_won"])
        """
        qualified_name: str = qualify_table_name(self.catalog, self.schema, table_name)
        if columns:
            cols = ", ".join(columns)
            statement = f"SELECT {cols} FROM {qualified_name}"
        else:
            statement = f"SELECT * FROM {qualified_name}"

        return self.query(statement)

    def read_fixture(self, fixture_path: str | Path) -> pd.DataFrame:
        """
        Read local fixture file (parquet or CSV).

        Used for testing without Databricks connection.

        Args:
            fixture_path: Path to fixture file

        Returns:
            Fixture data as DataFrame

        Example:
            >>> df = client.read_fixture("tests/fixtures/fct_games_sample.parquet")
        """
        fixture_path = Path(fixture_path)

        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found: {fixture_path}")

        logger.info(f"Reading fixture: {fixture_path}")

        if fixture_path.suffix == ".parquet":
            return pd.read_parquet(fixture_path)
        elif fixture_path.suffix == ".csv":
            return pd.read_csv(fixture_path)
        else:
            raise ValueError(f"Unsupported fixture format: {fixture_path.suffix}")

    def close(self) -> None:
        """Close Databricks connection if open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Databricks connection closed")

    def __enter__(self) -> "DatabricksClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
