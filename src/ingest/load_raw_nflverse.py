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
import logging
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sql import connect

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# nflverse data release URLs
NFLVERSE_BASE_URL = "https://github.com/nflverse/nflverse-data/releases/latest/download/"

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


def download_nflverse_games(seasons: list[int], local_dir: Path) -> Path:
    """
    Download nflverse games.parquet and filter to specified seasons.

    Args:
        seasons: List of seasons to include (e.g., [2022, 2023, 2024])
        local_dir: Local directory to save filtered parquet

    Returns:
        Path to the filtered parquet file
    """
    url = f"{NFLVERSE_BASE_URL}games.parquet"
    logger.info(f"Downloading nflverse games from {url}")

    df = pd.read_parquet(url)
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


def download_nflverse_pbp(seasons: list[int], local_dir: Path) -> Path:
    """
    Download nflverse play-by-play data for specified seasons.

    Args:
        seasons: List of seasons to download (e.g., [2022, 2023, 2024])
        local_dir: Local directory to save combined parquet

    Returns:
        Path to the combined parquet file
    """
    dfs = []

    for season in seasons:
        url = f"{NFLVERSE_BASE_URL}play_by_play_{season}.parquet"
        logger.info(f"Downloading play-by-play for season {season} from {url}")

        try:
            df = pd.read_parquet(url)
            logger.info(f"Downloaded {len(df)} plays for season {season}")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to download season {season}: {e}")
            raise

    if not dfs:
        raise ValueError(f"No play-by-play data found for seasons {seasons}")

    # Combine all seasons
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total plays across {len(seasons)} seasons")

    # Save combined data locally
    output_path = local_dir / "pbp_filtered.parquet"
    combined_df.to_parquet(output_path, index=False)
    logger.info(f"Saved filtered play-by-play to {output_path}")

    return output_path


def upload_to_volume(
    workspace_client: WorkspaceClient,
    local_path: Path,
    dataset: str,
    seasons: list[int],
) -> str:
    """
    Upload parquet file to Unity Catalog Volume.

    Args:
        workspace_client: Databricks WorkspaceClient
        local_path: Path to local parquet file
        dataset: Dataset name (games, pbp)
        seasons: List of seasons (for path organization)

    Returns:
        Volume path where file was uploaded
    """
    # Create volume path
    volume_root = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}"
    seasons_str = "_".join(map(str, sorted(seasons)))
    volume_path = f"{volume_root}/{dataset}/seasons_{seasons_str}.parquet"

    logger.info(f"Uploading {local_path.name} to {volume_path}")

    # Read file content
    with open(local_path, "rb") as f:
        file_content = f.read()

    # Upload using Files API
    workspace_client.files.upload(volume_path, file_content, overwrite=True)
    logger.info(f"Successfully uploaded to {volume_path}")

    return volume_path


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
    volume_path: str,
) -> None:
    """
    Create bronze Delta table from parquet file in Volume.

    Args:
        sql_connection: Databricks SQL connection
        dataset: Dataset name (games, pbp)
        volume_path: Path to parquet file in Volume
    """
    config = DATASET_CONFIG[dataset]
    table_name = f"{CATALOG}.{SCHEMA_RAW}.{config['table_name']}"

    # Use CREATE OR REPLACE for idempotency
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {table_name}
    USING DELTA
    COMMENT '{config["description"]} - loaded via nflverse feed'
    AS
    SELECT * FROM read_files(
        '{volume_path}',
        format => 'parquet'
    )
    """

    logger.info(f"Creating bronze table {table_name}")
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
  # Load games for 3 seasons
  python -m src.ingest.load_raw_nflverse --seasons 2022 2023 2024

  # Load play-by-play for 2 seasons
  python -m src.ingest.load_raw_nflverse --seasons 2022 2023 --dataset pbp

Environment Variables:
  DATABRICKS_HOST          Databricks workspace URL
  DATABRICKS_HTTP_PATH     SQL Warehouse HTTP path
  DATABRICKS_TOKEN         Personal access token

See .env.example and docs/runbook.md for configuration details.
        """,
    )

    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Seasons to load (e.g., 2022 2023 2024)",
    )

    parser.add_argument(
        "--dataset",
        choices=["games", "pbp"],
        default="games",
        help="Dataset to load (default: games)",
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

    try:
        logger.info(f"Starting nflverse data load for {args.dataset} dataset")
        logger.info(f"Seasons: {args.seasons}")

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

                if args.dataset == "games":
                    local_file = download_nflverse_games(args.seasons, temp_path)
                elif args.dataset == "pbp":
                    local_file = download_nflverse_pbp(args.seasons, temp_path)
                else:
                    raise ValueError(f"Unknown dataset: {args.dataset}")

                # Upload to volume
                volume_path = upload_to_volume(
                    workspace_client,
                    local_file,
                    args.dataset,
                    args.seasons,
                )

                # Create bronze table
                create_bronze_table(
                    sql_connection,
                    args.dataset,
                    volume_path,
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
