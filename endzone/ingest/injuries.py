"""Weekly nflverse injury reports, conformed onto one bronze schema.

Releases from 2009 through 2024 include ``date_modified``. The 2025 and 2026
files do not. Those seasons still upload: the timestamp column is null, and
the feature code drops a row it cannot place before kickoff. Seasons before
2009 were not published.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from endzone.ingest.nflverse import download_parquet_with_validation
from endzone.ingest.volume import upload_to_volume_injury_season
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

NFLVERSE_INJURIES_URL_PATTERN: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
)
INJURY_FIRST_SEASON: int = 2009
# Stable bronze columns. date_modified is added as null when the release
# dropped it. season_type, present from 2025, is not part of this list.
INJURY_COLUMNS: tuple[str, ...] = (
    "season",
    "game_type",
    "team",
    "week",
    "gsis_id",
    "position",
    "full_name",
    "first_name",
    "last_name",
    "report_primary_injury",
    "report_secondary_injury",
    "report_status",
    "practice_primary_injury",
    "practice_secondary_injury",
    "practice_status",
    "date_modified",
)
_TEXT_COLUMNS: tuple[str, ...] = tuple(
    column for column in INJURY_COLUMNS if column not in {"season", "week", "date_modified"}
)
_REQUIRED_WITHOUT_TIMESTAMP: tuple[str, ...] = tuple(
    column for column in INJURY_COLUMNS if column != "date_modified"
)


def _stable_week(values: pd.Series) -> pd.Series:
    """Return week as nullable int32 so every season file shares one type."""
    numeric: pd.Series = pd.to_numeric(values, errors="coerce")
    return numeric.round().astype("Int32")


def _utc_timestamp(values: pd.Series) -> pd.Series:
    """Parse a report timestamp as UTC. Unparseable values become null."""
    parsed: pd.Series = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.astype("datetime64[us, UTC]")


def normalize_injury_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Force one season onto ``INJURY_COLUMNS``.

    Args:
        frame: One nflverse injuries release.

    Returns:
        A copy in ``INJURY_COLUMNS`` order. Week is nullable int32. Text is
        stripped. A missing ``date_modified`` column is null on every row.

    Raises:
        ValueError: A column other than ``date_modified`` is missing.
    """
    missing: list[str] = [
        column for column in _REQUIRED_WITHOUT_TIMESTAMP if column not in frame.columns
    ]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Injury frame is missing columns: {missing_list}")
    output: pd.DataFrame = pd.DataFrame(index=frame.index)
    output["season"] = pd.to_numeric(frame["season"], errors="coerce").round().astype("Int32")
    output["week"] = _stable_week(frame["week"])
    for column in _TEXT_COLUMNS:
        output[column] = frame[column].astype("string").str.strip()
    if "date_modified" in frame.columns:
        output["date_modified"] = _utc_timestamp(frame["date_modified"])
    else:
        output["date_modified"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[us, UTC]")
    return output.loc[:, list(INJURY_COLUMNS)]


def download_nflverse_injuries(
    seasons: list[int],
    local_dir: Path,
    workspace_client: object,
) -> list[int]:
    """Download injury reports and upload one parquet per season.

    Seasons before 2009 are skipped. HTTP 404 is skipped. A file that lacks
    a stable column other than ``date_modified`` is an error. Files land at
    ``injuries/{season}/injuries_{season}.parquet``.

    Args:
        seasons: Season years to attempt.
        local_dir: Temporary directory for the parquet files.
        workspace_client: Databricks client used for the volume upload.

    Returns:
        Season years that were uploaded.

    Raises:
        ValueError: No season produced a file, or a file failed the column check.
    """
    processed: list[int] = []
    for season in seasons:
        if season < INJURY_FIRST_SEASON:
            logger.info(
                "Season %s is before injury reports (%s), skipping",
                season,
                INJURY_FIRST_SEASON,
            )
            continue
        url: str = NFLVERSE_INJURIES_URL_PATTERN.format(season=season)
        logger.info("Processing injury reports for season %s", season)
        try:
            frame: pd.DataFrame = download_parquet_with_validation(url)
        except ValueError as exc:
            if "HTTP 404" in str(exc):
                logger.warning("Season %s injury report was not found (HTTP 404), skipping", season)
                continue
            raise
        if "date_modified" not in frame.columns:
            logger.warning(
                "Season %s injury report has no date_modified. "
                "The column is stored null and those rows are not used as pre-game features.",
                season,
            )
        selected: pd.DataFrame = normalize_injury_frame(frame)
        local_path: Path = local_dir / f"injuries_{season}.parquet"
        selected.to_parquet(local_path, index=False)
        upload_to_volume_injury_season(workspace_client, local_path, season)
        local_path.unlink()
        processed.append(season)
    if not processed:
        raise ValueError(f"No injury report data found for seasons {seasons}")
    logger.info("Uploaded injury reports for %s seasons", len(processed))
    return processed
