"""
Load raw nflverse data into Databricks Free Edition.

Downloads nflverse datasets (games, play-by-play) from official GitHub releases,
uploads to Unity Catalog Volumes, and creates bronze Delta tables.

Usage:
    python -m src.ingest.load_raw_nflverse --seasons 2022 2023 2024
    python -m src.ingest.load_raw_nflverse --seasons 2022 2023 --dataset pbp

License: nflverse data is CC BY 4.0
Source: https://github.com/nflverse/nflverse-data
"""

import argparse
import datetime
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from databricks.sdk import WorkspaceClient
from databricks.sql import connect

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# nflverse data release URLs
# Games/schedules are in the 'schedules' release tag
NFLVERSE_GAMES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
)
# Play-by-play data is in the 'pbp' release tag, per-season files
NFLVERSE_PBP_URL_PATTERN = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)

# Default season range (1999 is first nflverse season)
# Upper bound is derived dynamically from games.parquet
DEFAULT_SEASON_START = 1999

# Databricks Unity Catalog configuration
CATALOG = "workspace"
SCHEMA_RAW = "nfl_raw"
VOLUME_NAME = "landing"

# Dataset configurations
DATASET_CONFIG = {
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
}


def load_env_from_file() -> None:
    """
    Load environment variables from .env file if it exists.

    This allows running the script without manually exporting all variables.
    Note: In production, use proper secret management (Databricks secrets).
    """
    env_file = Path(".env")
    if env_file.exists():
        logger.info(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()


def parse_seasons(seasons_arg: str) -> list[int]:
    """
    Parse season specification into list of season years.

    Supports ranges (e.g., "1999-2026") and comma-separated lists (e.g., "2022,2023,2024").

    Args:
        seasons_arg: Season specification string

    Returns:
        List of season years (e.g., [1999, 2000, ..., 2026])

    Examples:
        >>> parse_seasons("1999-2003")
        [1999, 2000, 2001, 2002, 2003]
        >>> parse_seasons("2022,2023,2024")
        [2022, 2023, 2024]
        >>> parse_seasons("2022")
        [2022]
    """
    seasons = []

    for part in seasons_arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(part))

    return sorted(set(seasons))


def load_target_schema() -> pa.Schema:
    """
    Load the fixed pbp target schema from JSON.

    Returns:
        Target pyarrow Schema for pbp data
    """
    schema_path = Path(__file__).parent / "schemas" / "pbp_schema.json"

    with open(schema_path) as f:
        schema_dict = json.load(f)

    fields = []
    for col_name, type_str in schema_dict.items():
        pa_type = _parse_arrow_type(type_str)
        fields.append(pa.field(col_name, pa_type))

    return pa.schema(fields)


def _parse_arrow_type(type_str: str) -> pa.DataType:
    """Parse arrow type from string representation."""
    type_str = type_str.strip()

    type_map = {
        "int64": pa.int64(),
        "int32": pa.int32(),
        "float64": pa.float64(),
        "float32": pa.float32(),
        "string": pa.string(),
        "bool": pa.bool_(),
        "double": pa.float64(),
    }

    return type_map.get(type_str, pa.string())


def conform_table_to_schema(table: pa.Table, target_schema: pa.Schema, season: int) -> pa.Table:
    """
    Conform a pyarrow Table to a target schema.

    Adds missing columns as null, removes extra columns, and casts types to match target.
    Raises error on cast failures instead of silently nulling.

    Args:
        table: Input pyarrow Table
        target_schema: Target schema to conform to
        season: Season number (for error messages)

    Returns:
        Conformed pyarrow Table

    Raises:
        ValueError: If column type cannot be cast to target type
    """
    arrays = []
    names = []

    extra_columns = set(table.column_names) - {f.name for f in target_schema}
    if extra_columns:
        logger.warning(
            f"Season {season}: dropping {len(extra_columns)} columns not in target schema: "
            f"{sorted(list(extra_columns)[:10])}" + (" ..." if len(extra_columns) > 10 else "")
        )

    for field in target_schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            if col.type != field.type:
                try:
                    col = col.cast(field.type, safe=False)
                except pa.ArrowInvalid as e:
                    raise ValueError(
                        f"Season {season}: cannot cast column '{field.name}' "
                        f"from {col.type} to {field.type}: {e}"
                    ) from e
            arrays.append(col)
        else:
            arrays.append(pa.nulls(len(table), type=field.type))

        names.append(field.name)

    return pa.table(arrays, names=names)


def get_databricks_config() -> dict[str, str]:
    """
    Get Databricks configuration from environment variables.

    Returns:
        Dict with host, http_path, token

    Raises:
        ValueError: If required environment variables are not set
    """
    host = os.environ.get("DATABRICKS_HOST")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not http_path or not token:
        raise ValueError(
            "Missing required Databricks environment variables. "
            "Set DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN. "
            "See .env.example and docs/runbook.md for setup instructions."
        )

    return {"host": host, "http_path": http_path, "token": token}


def get_current_season() -> int:
    """
    Determine the current NFL season.

    Downloads nflverse games.parquet and uses the max season present.
    Fallback: calendar year if month >= 3, else year - 1 (Jan/Feb still previous season).

    Returns:
        Current NFL season year
    """
    try:
        df = download_parquet_with_validation(NFLVERSE_GAMES_URL)
        current_season = int(df["season"].max())
        logger.info(f"Current NFL season from games.parquet: {current_season}")
        return current_season
    except Exception as e:
        logger.warning(f"Could not determine season from games.parquet: {e}, using fallback")

        now = datetime.datetime.now()
        if now.month >= 3:
            fallback_season = now.year
        else:
            fallback_season = now.year - 1

        logger.info(f"Using fallback current season: {fallback_season}")
        return fallback_season


def get_default_season_range() -> str:
    """
    Get default season range string.

    Returns:
        Season range string like "1999-2026"
    """
    current_season = get_current_season()
    return f"{DEFAULT_SEASON_START}-{current_season}"


def download_parquet_with_validation(url: str) -> pd.DataFrame:
    """
    Download parquet file with explicit HTTP validation.

    Args:
        url: URL to parquet file

    Returns:
        DataFrame loaded from parquet

    Raises:
        ValueError: If HTTP response is not 200 OK
    """
    logger.info(f"Downloading from {url}")
    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        raise ValueError(
            f"Failed to download from {url}: "
            f"HTTP {response.status_code} {response.reason}. "
            f"Verify the URL exists and is accessible."
        )

    logger.debug(f"Successfully fetched {len(response.content)} bytes")
    return pd.read_parquet(BytesIO(response.content))


def download_nflverse_games(seasons: list[int], local_dir: Path) -> Path:
    """
    Download nflverse games.parquet and filter to specified seasons.

    Args:
        seasons: List of seasons to include (e.g., [2022, 2023, 2024])
        local_dir: Local directory to save filtered parquet

    Returns:
        Path to the filtered parquet file
    """
    df = download_parquet_with_validation(NFLVERSE_GAMES_URL)
    logger.info(f"Downloaded {len(df)} total games")

    # Filter to requested seasons
    df = df[df["season"].isin(seasons)]
    logger.info(f"Filtered to {len(df)} games across seasons {seasons}")

    if df.empty:
        raise ValueError(f"No games found for seasons {seasons}")

    # Save filtered data locally
    output_path = local_dir / "games_filtered.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved filtered games to {output_path}")

    return output_path


def download_nflverse_pbp(
    seasons: list[int], local_dir: Path, workspace_client: WorkspaceClient
) -> list[int]:
    """
    Download nflverse play-by-play data for specified seasons.

    Processes ONE SEASON AT A TIME to avoid memory exhaustion. Downloads, conforms to
    fixed target schema, writes locally, uploads to volume, then deletes local file
    before moving to next season.

    Seasons with missing files are skipped with a warning.

    Args:
        seasons: List of seasons to download (e.g., [1999, 2000, 2022, 2023, 2024])
        local_dir: Local directory for temporary files
        workspace_client: Databricks WorkspaceClient for uploads

    Returns:
        List of successfully processed season years
    """
    target_schema = load_target_schema()
    logger.info(f"Loaded target schema with {len(target_schema)} columns")

    processed_seasons = []

    for season in seasons:
        url = NFLVERSE_PBP_URL_PATTERN.format(season=season)
        logger.info(f"Processing play-by-play for season {season}")

        try:
            df = download_parquet_with_validation(url)
            logger.info(f"Downloaded {len(df)} plays for season {season}")

            if "season" not in df.columns:
                df["season"] = season

            table = pa.Table.from_pandas(df)
            del df

            conformed = conform_table_to_schema(table, target_schema, season)
            del table

            local_path = local_dir / f"play_by_play_{season}.parquet"
            pq.write_table(conformed, local_path)
            logger.info(f"Saved conformed season {season} to {local_path} ({len(conformed)} rows)")

            upload_to_volume_pbp_season(workspace_client, local_path, season)

            local_path.unlink()
            logger.debug(f"Deleted local file {local_path}")

            processed_seasons.append(season)

        except ValueError as e:
            if "HTTP 404" in str(e):
                logger.warning(
                    f"Season {season} pbp file not found (HTTP 404), skipping. "
                    f"This is expected for future seasons."
                )
            else:
                logger.warning(f"Failed to process season {season}: {e}, skipping")
        except Exception as e:
            logger.warning(f"Failed to process season {season}: {e}, skipping")

    if not processed_seasons:
        raise ValueError(f"No play-by-play data found for any of seasons {seasons}")

    logger.info(f"Successfully processed {len(processed_seasons)} seasons")
    return processed_seasons


def upload_to_volume_pbp_season(
    workspace_client: WorkspaceClient, local_path: Path, season: int
) -> str:
    """
    Upload one season's pbp parquet to partitioned volume path.

    Args:
        workspace_client: Databricks WorkspaceClient
        local_path: Path to local parquet file
        season: Season number

    Returns:
        Volume path where file was uploaded
    """
    volume_root = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}"
    volume_path = f"{volume_root}/pbp/{season}/{local_path.name}"

    logger.info(f"Uploading {local_path.name} to {volume_path}")

    with open(local_path, "rb") as f:
        workspace_client.files.upload(volume_path, f, overwrite=True)
    logger.info(f"Successfully uploaded to {volume_path}")

    return volume_path


def upload_to_volume(
    workspace_client: WorkspaceClient,
    local_path: Path,
    dataset: str,
) -> str:
    """
    Upload parquet file to Unity Catalog Volume.

    Args:
        workspace_client: Databricks WorkspaceClient
        local_path: Path to local parquet file
        dataset: Dataset name (games only, pbp uses upload_to_volume_pbp_season)

    Returns:
        Volume path where file was uploaded
    """
    volume_root = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}"
    volume_path = f"{volume_root}/{dataset}/{local_path.name}"

    logger.info(f"Uploading {local_path.name} to {volume_path}")

    with open(local_path, "rb") as f:
        workspace_client.files.upload(volume_path, f, overwrite=True)
    logger.info(f"Successfully uploaded to {volume_path}")

    return volume_path


def cleanup_stale_pbp_files(workspace_client: WorkspaceClient) -> list[str]:
    """
    Delete stale pbp files directly under landing/pbp/ (not in season subfolders).

    Old ingest created files like seasons_2022_2023_2024.parquet. These would be
    double-counted when read_files reads the entire pbp folder. Remove them.

    Args:
        workspace_client: Databricks WorkspaceClient

    Returns:
        List of deleted file paths
    """
    volume_root = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}"
    pbp_root = f"{volume_root}/pbp"

    try:
        files = workspace_client.files.list_directory_contents(pbp_root)
    except Exception as e:
        logger.info(f"No existing pbp directory or cannot list: {e}")
        return []

    deleted = []
    for file_info in files:
        if file_info.is_directory:
            continue

        file_path = file_info.path
        logger.info(f"Deleting stale pbp file: {file_path}")
        try:
            workspace_client.files.delete(file_path)
            deleted.append(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")

    if deleted:
        logger.info(f"Cleaned up {len(deleted)} stale pbp file(s)")
    else:
        logger.info("No stale pbp files found")

    return deleted


def create_schema_and_volume(sql_connection: Any) -> None:
    """
    Create schema and volume if they don't exist.

    Args:
        sql_connection: Databricks SQL connection
    """
    cursor = sql_connection.cursor()

    try:
        # Create schema (catalog.schema)
        schema_sql = f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_RAW}"
        logger.info(f"Executing: {schema_sql}")
        cursor.execute(schema_sql)

        # Create volume for landing zone
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
    """
    Create bronze Delta table from parquet file(s) in Volume.

    For pbp dataset, reads from entire pbp folder (all season partitions).
    For games dataset, reads from specific file path.

    Args:
        sql_connection: Databricks SQL connection
        dataset: Dataset name (games, pbp)
        volume_path: Path to parquet file or folder in Volume (None for pbp, uses folder)
    """
    config = DATASET_CONFIG[dataset]
    table_name = f"{CATALOG}.{SCHEMA_RAW}.{config['table_name']}"

    if dataset == "pbp":
        # For pbp, read from entire pbp folder (all season partitions)
        read_path = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}/pbp"
    else:
        # For games, use the specific file path
        if volume_path is None:
            raise ValueError("volume_path is required for games dataset")
        read_path = volume_path

    create_table_sql = f"""
    CREATE OR REPLACE TABLE {table_name}
    USING DELTA
    COMMENT '{config["description"]} - loaded via nflverse feed'
    AS
    SELECT * FROM read_files(
        '{read_path}',
        format => 'parquet'
    )
    """

    logger.info(f"Creating bronze table {table_name} from {read_path}")
    logger.debug(f"SQL: {create_table_sql}")

    cursor = sql_connection.cursor()
    try:
        cursor.execute(create_table_sql)
        logger.info(f"Successfully created table {table_name}")

        # Get row count for verification
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = cursor.fetchone()[0]
        logger.info(f"Table {table_name} contains {row_count} rows")
    finally:
        cursor.close()


def main() -> int:
    """
    Main entry point for data ingestion.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Load nflverse data into Databricks Unity Catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load games and pbp for all seasons (1999-2026)
  python -m src.ingest.load_raw_nflverse --dataset all

  # Load games for specific range
  python -m src.ingest.load_raw_nflverse --seasons 2020-2024 --dataset games

  # Load pbp for specific seasons
  python -m src.ingest.load_raw_nflverse --seasons 2022,2023,2024 --dataset pbp

  # Refresh current season only (weekly update)
  python -m src.ingest.load_raw_nflverse --current-season --dataset all

Environment Variables:
  DATABRICKS_HOST          Databricks workspace URL
  DATABRICKS_HTTP_PATH     SQL Warehouse HTTP path
  DATABRICKS_TOKEN         Personal access token

See .env.example and docs/runbook.md for configuration details.
        """,
    )

    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to load as range (e.g., 1999-2026) or list (e.g., 2022,2023,2024). Default: 1999 through current season",
    )

    parser.add_argument(
        "--current-season",
        action="store_true",
        help="Load only the current season pbp (games still loads full range). Use for weekly refresh.",
    )

    parser.add_argument(
        "--dataset",
        choices=["games", "pbp", "all"],
        default="all",
        help="Dataset to load (default: all)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(args.log_level)

    # Determine seasons to load
    if args.seasons:
        season_range = args.seasons
    else:
        season_range = get_default_season_range()

    all_seasons = parse_seasons(season_range)
    logger.info(
        f"Full season range: {all_seasons[0]}-{all_seasons[-1]} ({len(all_seasons)} seasons)"
    )

    if args.current_season:
        current = get_current_season()
        pbp_seasons = [current]
        games_seasons = all_seasons
        logger.info(f"Current season mode: pbp will load {current} only, games loads full range")
    else:
        pbp_seasons = all_seasons
        games_seasons = all_seasons

    try:
        logger.info(f"Starting nflverse data load for {args.dataset} dataset")

        # Load environment
        load_env_from_file()

        # Get Databricks config
        db_config = get_databricks_config()
        logger.info(f"Connected to Databricks host: {db_config['host']}")

        # Create workspace client (for Files API)
        workspace_client = WorkspaceClient(
            host=db_config["host"],
            token=db_config["token"],
        )

        # Create SQL connection (for schema/table operations)
        sql_connection = connect(
            server_hostname=db_config["host"],
            http_path=db_config["http_path"],
            access_token=db_config["token"],
        )

        try:
            # Create schema and volume (idempotent)
            create_schema_and_volume(sql_connection)

            # Download data to temporary directory
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Load games dataset
                if args.dataset in ("games", "all"):
                    logger.info(
                        f"Loading games dataset for seasons {games_seasons[0]}-{games_seasons[-1]}"
                    )
                    local_file = download_nflverse_games(games_seasons, temp_path)

                    volume_path = upload_to_volume(
                        workspace_client,
                        local_file,
                        "games",
                    )

                    create_bronze_table(
                        sql_connection,
                        "games",
                        volume_path,
                    )

                # Load pbp dataset
                if args.dataset in ("pbp", "all"):
                    logger.info(f"Loading play-by-play dataset for seasons: {pbp_seasons}")

                    cleanup_stale_pbp_files(workspace_client)

                    with TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        processed_seasons = download_nflverse_pbp(
                            pbp_seasons, temp_path, workspace_client
                        )

                    if not processed_seasons:
                        logger.warning("No pbp data processed, skipping bronze table creation")
                    else:
                        create_bronze_table(
                            sql_connection,
                            "pbp",
                            volume_path=None,
                        )

            logger.info("Data load completed successfully")
            return 0

        finally:
            sql_connection.close()

    except Exception as e:
        logger.error(f"Data load failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
