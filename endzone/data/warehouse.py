"""Databricks SQL reads and Unity Catalog Volume uploads.

Missing credentials stay in fixture mode and refuse to open a connection.
Ingest opens a SQL connection without a default catalog, because its
statements are already fully qualified. Mart reads pass catalog and schema
as connection parameters. The warehouse rejects a script that starts with USE.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from endzone.config import (
    DEFAULT_CATALOG,
    DEFAULT_MARTS_SCHEMA,
    load_databricks_settings,
    resolve_setting,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

# Re-exported so callers can keep importing the defaults with the client.
__all__ = [
    "DEFAULT_CATALOG",
    "DEFAULT_MARTS_SCHEMA",
    "DatabricksClient",
    "connect_sql",
    "connect_workspace",
    "qualify_table_name",
    "quote_identifier",
    "upload_volume_file",
]


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


def connect_sql(
    host: str,
    http_path: str,
    token: str,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> Any:
    """Open one Databricks SQL connection.

    Args:
        host: Workspace hostname, without ``https://``.
        http_path: SQL warehouse HTTP path.
        token: Access token.
        catalog: Session catalog. Omit it when every statement is qualified.
        schema: Session schema. Omit it when every statement is qualified.

    Returns:
        A connection from ``databricks.sql.connect``.

    Raises:
        ImportError: The SQL connector is not installed.
    """
    try:
        from databricks import sql
    except ImportError as exc:
        raise ImportError(
            "databricks-sql-connector not installed. "
            "Install with: pip install databricks-sql-connector"
        ) from exc

    parameters: dict[str, str] = {
        "server_hostname": host,
        "http_path": http_path,
        "access_token": token,
    }
    if catalog is not None:
        parameters["catalog"] = catalog
    if schema is not None:
        parameters["schema"] = schema
    return sql.connect(**parameters)


def connect_workspace(host: str, token: str) -> Any:
    """Open a Workspace client for the Files API.

    Args:
        host: Workspace hostname, without ``https://``.
        token: Access token.

    Returns:
        A ``WorkspaceClient`` used for Volume upload, list, and delete.
    """
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(host=host, token=token)


def upload_volume_file(workspace_client: Any, local_path: Path, volume_path: str) -> None:
    """Upload one local file to a Volume path, replacing an existing object.

    Args:
        workspace_client: Databricks Workspace client.
        local_path: File to read in binary mode.
        volume_path: Destination path, including the ``/Volumes`` prefix.
    """
    with open(local_path, "rb") as handle:
        workspace_client.files.upload(volume_path, handle, overwrite=True)


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
        settings = load_databricks_settings()
        self.host = host or settings.host
        self.http_path = http_path or settings.http_path
        self.token = token or settings.token
        self.catalog = resolve_setting(catalog, "DATABRICKS_CATALOG", DEFAULT_CATALOG)
        self.schema = resolve_setting(schema, "DATABRICKS_MARTS_SCHEMA", DEFAULT_MARTS_SCHEMA)

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
            raise RuntimeError("Cannot connect in fixture mode. Provide Databricks credentials.")

        if self._connection is None:
            if self.host is None or self.http_path is None or self.token is None:
                raise RuntimeError(
                    "Cannot connect in fixture mode. Provide Databricks credentials."
                )
            # Session catalog and schema are connection parameters.
            # The SQL warehouse rejects a script that starts with USE.
            self._connection = connect_sql(
                self.host,
                self.http_path,
                self.token,
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
            RuntimeError: If in fixture mode because credentials are missing

        Example:
            >>> df = client.query("SELECT * FROM fct_games WHERE season = 2024")
        """
        if self._use_fixture:
            raise RuntimeError(
                "Cannot query Databricks in fixture mode. Provide Databricks credentials."
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

    def close(self) -> None:
        """Close Databricks connection if open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Databricks connection closed")

    def __enter__(self) -> "DatabricksClient":
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()
