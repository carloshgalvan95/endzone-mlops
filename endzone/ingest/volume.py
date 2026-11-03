"""Unity Catalog Volume paths and the bronze DDL for nflverse landing files."""

from pathlib import Path
from typing import Any

from endzone.data.warehouse import upload_volume_file
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

CATALOG: str = "workspace"
SCHEMA_RAW: str = "nfl_raw"
VOLUME_NAME: str = "landing"

# Hint read_files so a leftover int/double week mix still parses as one int column.
# The upload path already writes nullable int32. This hint is the belt.
DEPTH_CHART_READ_SCHEMA_HINT: str = "season int, week int, depth_team string, club_code string"

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "games": {
        "filename": "games.parquet",
        "table_name": "games",
        "description": "NFL game schedules and results (1999-present)",
        "supports_season_filter": True,
    },
    "pbp": {
        "filename_pattern": "play_by_play_{season}.parquet",
        "table_name": "play_by_play",
        "description": "NFL play-by-play data",
        "supports_season_filter": True,
    },
    "depth_charts": {
        "filename_pattern": "depth_charts_{season}.parquet",
        "table_name": "depth_charts",
        "description": "NFL depth charts (2001-present, 2025+ snapshots mapped to weeks)",
        "supports_season_filter": True,
    },
}


def _volume_root() -> str:
    """Return the landing volume root used by every bronze upload."""
    return f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}"


def upload_to_volume(
    workspace_client: Any,
    local_path: Path,
    dataset: str,
) -> str:
    """Upload a parquet file to a Unity Catalog Volume.

    Args:
        workspace_client: Databricks Workspace client.
        local_path: Path to the local parquet file.
        dataset: Dataset name (games only; pbp uses ``upload_to_volume_pbp_season``).

    Returns:
        Volume path where the file was uploaded.
    """
    volume_path = f"{_volume_root()}/{dataset}/{local_path.name}"
    logger.info(f"Uploading {local_path.name} to {volume_path}")
    upload_volume_file(workspace_client, local_path, volume_path)
    logger.info(f"Successfully uploaded to {volume_path}")
    return volume_path


def upload_to_volume_pbp_season(workspace_client: Any, local_path: Path, season: int) -> str:
    """Upload one season's pbp parquet to a partitioned volume path.

    Args:
        workspace_client: Databricks Workspace client.
        local_path: Path to the local parquet file.
        season: Season number.

    Returns:
        Volume path where the file was uploaded.
    """
    volume_path = f"{_volume_root()}/pbp/{season}/{local_path.name}"
    logger.info(f"Uploading {local_path.name} to {volume_path}")
    upload_volume_file(workspace_client, local_path, volume_path)
    logger.info(f"Successfully uploaded to {volume_path}")
    return volume_path


def upload_to_volume_depth_chart_season(
    workspace_client: Any,
    local_path: Path,
    season: int,
) -> str:
    """Upload one season of depth charts to the landing volume.

    Args:
        workspace_client: Databricks Workspace client.
        local_path: Local parquet path.
        season: Season year used in the volume directory.

    Returns:
        Volume path of the uploaded file.
    """
    volume_path = f"{_volume_root()}/depth_charts/{season}/{local_path.name}"
    logger.info("Uploading %s to %s", local_path.name, volume_path)
    upload_volume_file(workspace_client, local_path, volume_path)
    logger.info("Uploaded depth charts to %s", volume_path)
    return volume_path


def cleanup_stale_pbp_files(workspace_client: Any) -> list[str]:
    """Delete stale pbp files directly under landing/pbp/ (not in season subfolders).

    Old ingest created files like seasons_2022_2023_2024.parquet. These would be
    double-counted when read_files reads the entire pbp folder. Remove them.

    Args:
        workspace_client: Databricks Workspace client.

    Returns:
        List of deleted file paths.
    """
    pbp_root = f"{_volume_root()}/pbp"
    try:
        files = workspace_client.files.list_directory_contents(pbp_root)
    except Exception as exc:
        logger.info(f"No existing pbp directory or cannot list: {exc}")
        return []

    deleted: list[str] = []
    for file_info in files:
        if file_info.is_directory:
            continue
        file_path = file_info.path
        logger.info(f"Deleting stale pbp file: {file_path}")
        try:
            workspace_client.files.delete(file_path)
            deleted.append(file_path)
        except Exception as exc:
            logger.warning(f"Failed to delete {file_path}: {exc}")

    if deleted:
        logger.info(f"Cleaned up {len(deleted)} stale pbp file(s)")
    else:
        logger.info("No stale pbp files found")
    return deleted


def create_schema_and_volume(sql_connection: Any) -> None:
    """Create the bronze schema and landing volume if they do not exist.

    Args:
        sql_connection: Databricks SQL connection.
    """
    cursor = sql_connection.cursor()
    try:
        schema_sql = f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_RAW}"
        logger.info(f"Executing: {schema_sql}")
        cursor.execute(schema_sql)

        volume_sql = f"""
        CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_RAW}.{VOLUME_NAME}
        """
        logger.info(f"Executing: CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_RAW}.{VOLUME_NAME}")
        cursor.execute(volume_sql)
        logger.info(f"Schema {CATALOG}.{SCHEMA_RAW} and volume {VOLUME_NAME} ready")
    finally:
        cursor.close()


def create_bronze_table(
    sql_connection: Any,
    dataset: str,
    volume_path: str | None = None,
) -> None:
    """Create a bronze Delta table from parquet file(s) in a Volume.

    For pbp, reads the entire pbp folder (all season partitions).
    For games, reads the specific file path.

    Args:
        sql_connection: Databricks SQL connection.
        dataset: Dataset name (games, pbp, depth_charts).
        volume_path: Path to a parquet file or folder. ``None`` for pbp and
            depth charts, which read their folders.
    """
    config = DATASET_CONFIG[dataset]
    table_name = f"{CATALOG}.{SCHEMA_RAW}.{config['table_name']}"

    if dataset == "pbp":
        read_path = f"{_volume_root()}/pbp"
    elif dataset == "depth_charts":
        read_path = f"{_volume_root()}/depth_charts"
    else:
        if volume_path is None:
            raise ValueError("volume_path is required for games dataset")
        read_path = volume_path

    read_options = "format => 'parquet'"
    if dataset == "depth_charts":
        read_options = (
            "format => 'parquet',\n        schemaHints => '" + DEPTH_CHART_READ_SCHEMA_HINT + "'"
        )
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {table_name}
    USING DELTA
    COMMENT '{config["description"]} - loaded via nflverse feed'
    AS
    SELECT * FROM read_files(
        '{read_path}',
        {read_options}
    )
    """

    logger.info(f"Creating bronze table {table_name} from {read_path}")
    logger.debug(f"SQL: {create_table_sql}")

    cursor = sql_connection.cursor()
    try:
        cursor.execute(create_table_sql)
        logger.info(f"Successfully created table {table_name}")
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = cursor.fetchone()[0]
        logger.info(f"Table {table_name} contains {row_count} rows")
    finally:
        cursor.close()
