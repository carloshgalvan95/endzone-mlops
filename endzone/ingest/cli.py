"""Command line for the bronze nflverse load."""

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from endzone.config import get_databricks_config, load_env_file
from endzone.data.warehouse import connect_sql, connect_workspace
from endzone.ingest.depth_charts import download_nflverse_depth_charts
from endzone.ingest.injuries import download_nflverse_injuries
from endzone.ingest.nflverse import (
    download_nflverse_games,
    download_nflverse_pbp,
    get_current_season,
    get_default_season_range,
    parse_seasons,
)
from endzone.ingest.volume import (
    cleanup_stale_pbp_files,
    create_bronze_table,
    create_schema_and_volume,
    upload_to_volume,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Return the ingest argument parser. Flags stay stable for ``make ingest``."""
    parser = argparse.ArgumentParser(
        description="Load nflverse data into Databricks Unity Catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load games and pbp for all seasons (1999-2026)
  python -m endzone.ingest.load_raw_nflverse --dataset all

  # Load games for specific range
  python -m endzone.ingest.load_raw_nflverse --seasons 2020-2024 --dataset games

  # Load pbp for specific seasons
  python -m endzone.ingest.load_raw_nflverse --seasons 2022,2023,2024 --dataset pbp

  # Refresh current season only (weekly update)
  python -m endzone.ingest.load_raw_nflverse --current-season --dataset all

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
        help=(
            "Seasons to load as range (e.g., 1999-2026) or list (e.g., 2022,2023,2024). "
            "Default: 1999 through current season"
        ),
    )
    parser.add_argument(
        "--current-season",
        action="store_true",
        help="Load only the current season pbp (games still loads full range). Use for weekly refresh.",
    )
    parser.add_argument(
        "--dataset",
        choices=["games", "pbp", "depth_charts", "injuries", "all"],
        default="all",
        help=(
            "Dataset to load (default: all). depth_charts is the weekly starter chart. "
            "injuries is the weekly injury report from 2009."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def _season_plan(args: argparse.Namespace) -> dict[str, list[int]]:
    """Resolve games, pbp, depth-chart, and injury season lists for this invocation."""
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
        logger.info(
            "Current season mode: pbp, depth charts, and injuries load %s only, "
            "games loads full range",
            current,
        )
        return {
            "games": all_seasons,
            "pbp": [current],
            "depth_charts": [current],
            "injuries": [current],
        }
    return {
        "games": all_seasons,
        "pbp": all_seasons,
        "depth_charts": all_seasons,
        "injuries": all_seasons,
    }


def _load_games(
    sql_connection: Any,
    workspace_client: Any,
    seasons: list[int],
    temp_path: Path,
) -> None:
    """Filter games, upload the file, and replace the bronze table."""
    logger.info(f"Loading games dataset for seasons {seasons[0]}-{seasons[-1]}")
    local_file = download_nflverse_games(seasons, temp_path)
    volume_path = upload_to_volume(workspace_client, local_file, "games")
    create_bronze_table(sql_connection, "games", volume_path)


def _load_pbp(
    sql_connection: Any,
    workspace_client: Any,
    seasons: list[int],
) -> None:
    """Upload one pbp file per season, then replace the bronze table."""
    logger.info(f"Loading play-by-play dataset for seasons: {seasons}")
    cleanup_stale_pbp_files(workspace_client)
    with TemporaryDirectory() as temp_dir:
        processed_seasons = download_nflverse_pbp(seasons, Path(temp_dir), workspace_client)
    if not processed_seasons:
        logger.warning("No pbp data processed, skipping bronze table creation")
        return
    create_bronze_table(sql_connection, "pbp", volume_path=None)


def _load_depth_charts(
    sql_connection: Any,
    workspace_client: Any,
    seasons: list[int],
) -> None:
    """Upload one depth-chart file per season, then replace the bronze table."""
    logger.info("Loading depth charts for seasons: %s", seasons)
    with TemporaryDirectory() as depth_dir:
        processed_charts = download_nflverse_depth_charts(
            seasons,
            Path(depth_dir),
            workspace_client,
        )
    if not processed_charts:
        logger.warning("No depth charts processed, skipping bronze table creation")
        return
    create_bronze_table(sql_connection, "depth_charts", volume_path=None)


def _load_injuries(
    sql_connection: Any,
    workspace_client: Any,
    seasons: list[int],
) -> None:
    """Upload one injury file per season, then replace the bronze table."""
    logger.info("Loading injury reports for seasons: %s", seasons)
    with TemporaryDirectory() as injury_dir:
        processed_injuries = download_nflverse_injuries(
            seasons,
            Path(injury_dir),
            workspace_client,
        )
    if not processed_injuries:
        logger.warning("No injury reports processed, skipping bronze table creation")
        return
    create_bronze_table(sql_connection, "injuries", volume_path=None)


def main(argv: list[str] | None = None) -> int:
    """Load nflverse parquet into ``workspace.nfl_raw``.

    Args:
        argv: Arguments, excluding the program name. ``None`` reads ``sys.argv``.

    Returns:
        Exit code. ``0`` for success, ``1`` when the load fails.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.getLogger().setLevel(args.log_level)
    plan = _season_plan(args)

    try:
        logger.info(f"Starting nflverse data load for {args.dataset} dataset")
        load_env_file()
        db_config = get_databricks_config()
        logger.info(f"Connected to Databricks host: {db_config['host']}")
        workspace_client = connect_workspace(db_config["host"], db_config["token"])
        sql_connection = connect_sql(
            db_config["host"],
            db_config["http_path"],
            db_config["token"],
        )
        try:
            create_schema_and_volume(sql_connection)
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                if args.dataset in ("games", "all"):
                    _load_games(sql_connection, workspace_client, plan["games"], temp_path)
                if args.dataset in ("pbp", "all"):
                    _load_pbp(sql_connection, workspace_client, plan["pbp"])
                if args.dataset in ("depth_charts", "all"):
                    _load_depth_charts(sql_connection, workspace_client, plan["depth_charts"])
                if args.dataset in ("injuries", "all"):
                    _load_injuries(sql_connection, workspace_client, plan["injuries"])
            logger.info("Data load completed successfully")
            return 0
        finally:
            sql_connection.close()
    except Exception as exc:
        logger.error(f"Data load failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
