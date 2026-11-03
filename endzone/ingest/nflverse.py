"""Download nflverse games and play-by-play releases."""

import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from endzone.ingest.schema import conform_table_to_schema, load_target_schema
from endzone.ingest.volume import upload_to_volume_pbp_season
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

NFLVERSE_GAMES_URL: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
)
NFLVERSE_PBP_URL_PATTERN: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)
DEFAULT_SEASON_START: int = 1999


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    """Download one URL and raise when the response is not HTTP 200.

    Args:
        url: File URL.
        timeout: Request timeout in seconds.

    Returns:
        Response body.

    Raises:
        ValueError: The response status is not 200.
    """
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download from {url}: "
            f"HTTP {response.status_code} {response.reason}. "
            f"Verify the URL exists and is accessible."
        )
    return response.content


def download_parquet_with_validation(url: str) -> pd.DataFrame:
    """Download a parquet file with explicit HTTP validation.

    Args:
        url: URL of the parquet file.

    Returns:
        DataFrame loaded from the parquet body.

    Raises:
        ValueError: The HTTP response is not 200 OK.
    """
    logger.info(f"Downloading from {url}")
    content = fetch_bytes(url)
    logger.debug(f"Successfully fetched {len(content)} bytes")
    return pd.read_parquet(BytesIO(content))


def parse_seasons(seasons_arg: str) -> list[int]:
    """Parse a season specification into a list of season years.

    Supports ranges (for example ``1999-2026``) and comma-separated lists
    (for example ``2022,2023,2024``).

    Args:
        seasons_arg: Season specification string.

    Returns:
        Sorted unique season years.
    """
    seasons: list[int] = []
    for part in seasons_arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(part))
    return sorted(set(seasons))


def get_current_season() -> int:
    """Determine the current NFL season.

    Downloads nflverse ``games.parquet`` and uses the max season present.
    Fallback: calendar year if month >= 3, else year - 1 (Jan/Feb still
    the previous season).

    Returns:
        Current NFL season year.
    """
    try:
        frame = download_parquet_with_validation(NFLVERSE_GAMES_URL)
        current_season = int(frame["season"].max())
        logger.info(f"Current NFL season from games.parquet: {current_season}")
        return current_season
    except Exception as exc:
        logger.warning(f"Could not determine season from games.parquet: {exc}, using fallback")
        now = datetime.datetime.now()
        if now.month >= 3:
            fallback_season = now.year
        else:
            fallback_season = now.year - 1
        logger.info(f"Using fallback current season: {fallback_season}")
        return fallback_season


def get_default_season_range() -> str:
    """Return the default season range string, such as ``1999-2026``."""
    current_season = get_current_season()
    return f"{DEFAULT_SEASON_START}-{current_season}"


def download_nflverse_games(seasons: list[int], local_dir: Path) -> Path:
    """Download nflverse ``games.parquet`` and filter to the requested seasons.

    Args:
        seasons: Seasons to keep.
        local_dir: Directory for the filtered parquet.

    Returns:
        Path to the filtered parquet file.

    Raises:
        ValueError: No games match the requested seasons.
    """
    frame = download_parquet_with_validation(NFLVERSE_GAMES_URL)
    logger.info(f"Downloaded {len(frame)} total games")
    frame = frame[frame["season"].isin(seasons)]
    logger.info(f"Filtered to {len(frame)} games across seasons {seasons}")
    if frame.empty:
        raise ValueError(f"No games found for seasons {seasons}")
    output_path = local_dir / "games_filtered.parquet"
    frame.to_parquet(output_path, index=False)
    logger.info(f"Saved filtered games to {output_path}")
    return output_path


def download_nflverse_pbp(
    seasons: list[int], local_dir: Path, workspace_client: object
) -> list[int]:
    """Download play-by-play one season at a time and upload each file.

    Seasons with missing files are skipped with a warning.

    Args:
        seasons: Seasons to download.
        local_dir: Directory for the temporary parquet files.
        workspace_client: Databricks client used for the volume upload.

    Returns:
        Season years that were uploaded.

    Raises:
        ValueError: No season produced a file.
    """
    target_schema = load_target_schema()
    logger.info(f"Loaded target schema with {len(target_schema)} columns")
    processed_seasons: list[int] = []

    for season in seasons:
        url = NFLVERSE_PBP_URL_PATTERN.format(season=season)
        logger.info(f"Processing play-by-play for season {season}")
        try:
            frame = download_parquet_with_validation(url)
            logger.info(f"Downloaded {len(frame)} plays for season {season}")
            if "season" not in frame.columns:
                frame["season"] = season
            table = pa.Table.from_pandas(frame)
            del frame
            conformed = conform_table_to_schema(table, target_schema, season)
            del table
            local_path = local_dir / f"play_by_play_{season}.parquet"
            pq.write_table(conformed, local_path)
            logger.info(f"Saved conformed season {season} to {local_path} ({len(conformed)} rows)")
            upload_to_volume_pbp_season(workspace_client, local_path, season)
            local_path.unlink()
            logger.debug(f"Deleted local file {local_path}")
            processed_seasons.append(season)
        except ValueError as exc:
            if "HTTP 404" in str(exc):
                logger.warning(
                    f"Season {season} pbp file not found (HTTP 404), skipping. "
                    f"This is expected for future seasons."
                )
            else:
                logger.warning(f"Failed to process season {season}: {exc}, skipping")
        except Exception as exc:
            logger.warning(f"Failed to process season {season}: {exc}, skipping")

    if not processed_seasons:
        raise ValueError(f"No play-by-play data found for any of seasons {seasons}")
    logger.info(f"Successfully processed {len(processed_seasons)} seasons")
    return processed_seasons
