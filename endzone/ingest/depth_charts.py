"""Weekly depth charts, legacy club codes, and the season upload."""

from pathlib import Path

import pandas as pd

from endzone.ingest.nflverse import NFLVERSE_GAMES_URL, download_parquet_with_validation
from endzone.ingest.snapshots import is_snapshot_depth_chart, map_snapshot_depth_chart
from endzone.ingest.volume import upload_to_volume_depth_chart_season
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

NFLVERSE_DEPTH_CHARTS_URL_PATTERN: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.parquet"
)
DEPTH_CHART_FIRST_SEASON: int = 2001
# Columns present on depth_charts_2001.parquet through depth_charts_2024.parquet.
# 2025 and later are mapped onto this list before the file is uploaded.
DEPTH_CHART_COLUMNS: tuple[str, ...] = (
    "season",
    "club_code",
    "week",
    "game_type",
    "depth_team",
    "last_name",
    "first_name",
    "football_name",
    "formation",
    "gsis_id",
    "jersey_number",
    "position",
    "elias_id",
    "depth_position",
    "full_name",
)
# Column order on depth_charts_2025.parquet. The release is a timestamped
# snapshot, not a weekly chart. There is no week or game_type.
DEPTH_CHART_2025_COLUMNS: tuple[str, ...] = (
    "dt",
    "team",
    "player_name",
    "espn_id",
    "gsis_id",
    "pos_grp_id",
    "pos_grp",
    "pos_id",
    "pos_name",
    "pos_abb",
    "pos_slot",
    "pos_rank",
)


def _has_weekly_depth_chart_columns(frame: pd.DataFrame) -> bool:
    """Return True when the frame already has the 2001-2024 weekly columns."""
    return all(column in frame.columns for column in DEPTH_CHART_COLUMNS)


def _depth_chart_club_map_path() -> Path:
    """Path of the seed that translates legacy depth-chart club codes."""
    return Path(__file__).resolve().parents[2] / "dbt" / "seeds" / "depth_chart_club_map.csv"


def load_depth_chart_club_map() -> pd.DataFrame:
    """Load legacy club-code ranges onto the schedule abbreviation.

    The current nflverse release already uses the schedule codes. Older
    files and NFL.com scraps use ARZ, BLT, CLV, HST, JAC, LAR, and the
    relocated codes in the wrong years. Ranges for one code do not overlap.
    A code that is already the schedule abbreviation is absent, so applying
    the map twice does not change it.

    Returns:
        Columns ``club_code``, ``season_start``, ``season_end``, and
        ``schedule_team``.

    Raises:
        ValueError: The seed is missing a column, or one code has overlapping
            season ranges.
    """
    required: tuple[str, ...] = ("club_code", "season_start", "season_end", "schedule_team")
    frame: pd.DataFrame = pd.read_csv(_depth_chart_club_map_path())
    missing: list[str] = [column for column in required if column not in frame.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Depth chart club map is missing columns: {missing_list}")
    cleaned: pd.DataFrame = pd.DataFrame(
        {
            "club_code": frame["club_code"].astype("string").str.strip(),
            "season_start": pd.to_numeric(frame["season_start"], errors="raise").astype(int),
            "season_end": pd.to_numeric(frame["season_end"], errors="raise").astype(int),
            "schedule_team": frame["schedule_team"].astype("string").str.strip(),
        }
    )
    for club_code, group in cleaned.groupby("club_code", sort=False):
        ordered: pd.DataFrame = group.sort_values(["season_start", "season_end"], kind="mergesort")
        starts: list[int] = ordered["season_start"].astype(int).tolist()
        ends: list[int] = ordered["season_end"].astype(int).tolist()
        for index in range(1, len(starts)):
            if starts[index] <= ends[index - 1]:
                raise ValueError(f"Depth chart club map overlaps for {club_code}")
    return cleaned.reset_index(drop=True)


def schedule_club_codes(club_code: pd.Series, season: pd.Series) -> pd.Series:
    """Translate depth-chart club codes onto the schedule abbreviation.

    Args:
        club_code: Raw ``club_code`` or snapshot ``team``.
        season: Season year for each row. Used when a code changed cities.

    Returns:
        Schedule abbreviations. An unknown code is returned unchanged.
    """
    codes: pd.Series = club_code.astype("string").str.strip()
    seasons: pd.Series = pd.to_numeric(season, errors="coerce")
    translated: pd.Series = codes.copy()
    mapping: pd.DataFrame = load_depth_chart_club_map()
    for row in mapping.itertuples(index=False):
        matched: pd.Series = (
            codes.eq(str(row.club_code))
            & seasons.ge(int(row.season_start))
            & seasons.le(int(row.season_end))
        )
        translated = translated.mask(matched, str(row.schedule_team))
    return translated


def _stable_week(values: pd.Series) -> pd.Series:
    """Return week as nullable int32.

    A file with a null week (the Super Bowl bye) is what pandas reads as
    float64. A file with no null week stays int32. Those two parquet types
    must not both land in one ``read_files`` folder.
    """
    numeric: pd.Series = pd.to_numeric(values, errors="coerce")
    return numeric.round().astype("Int32")


def _stable_depth_team(values: pd.Series) -> pd.Series:
    """Return depth rank as text, so 1, 1.0, and '1' all become '1'."""
    numeric: pd.Series = pd.to_numeric(values, errors="coerce")
    as_text: pd.Series = numeric.round().astype("Int64").astype("string")
    original: pd.Series = values.astype("string").str.strip()
    return as_text.where(numeric.notna(), original)


def normalize_depth_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the weekly bronze schema onto one season.

    Week is nullable int32 in every file. ``depth_team`` is text.
    ``club_code`` is the schedule abbreviation. Column order stays
    ``DEPTH_CHART_COLUMNS``.

    Args:
        frame: A weekly chart or a snapshot that was already mapped onto
            those columns.

    Returns:
        A copy safe to upload next to the other seasons.

    Raises:
        ValueError: A stable column is missing.
    """
    missing: list[str] = [column for column in DEPTH_CHART_COLUMNS if column not in frame.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Depth chart frame is missing columns: {missing_list}")
    output: pd.DataFrame = frame.loc[:, list(DEPTH_CHART_COLUMNS)].copy()
    output["season"] = pd.to_numeric(output["season"], errors="coerce").astype("Int32")
    output["week"] = _stable_week(output["week"])
    output["depth_team"] = _stable_depth_team(output["depth_team"])
    output["club_code"] = schedule_club_codes(output["club_code"], output["season"])
    for column in DEPTH_CHART_COLUMNS:
        if column in {"season", "week", "depth_team", "club_code"}:
            continue
        output[column] = output[column].astype("string")
    return output.loc[:, list(DEPTH_CHART_COLUMNS)]


def _select_weekly_depth_chart_columns(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Keep the stable weekly columns and refuse a file that lacks one.

    Args:
        frame: One season that already uses ``DEPTH_CHART_COLUMNS``.
        season: Season year, used in the error.

    Returns:
        A copy with ``DEPTH_CHART_COLUMNS`` only, in that order.

    Raises:
        ValueError: A required weekly column is absent.
    """
    missing: list[str] = [column for column in DEPTH_CHART_COLUMNS if column not in frame.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Season {season} depth chart missing columns: {missing_list}")
    extra: list[str] = sorted(set(frame.columns) - set(DEPTH_CHART_COLUMNS))
    if extra:
        logger.warning(
            "Season %s depth chart dropping columns not in the stable list: %s",
            season,
            extra[:10],
        )
    return normalize_depth_chart_frame(frame)


def select_depth_chart_columns(
    frame: pd.DataFrame,
    season: int,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the stable depth-chart columns for one season file.

    A 2001-2024 file already has those columns. A 2025-and-later snapshot is
    mapped with ``games``. Anything else is an error.

    Args:
        frame: One season of nflverse depth charts.
        season: Season year, used in errors and written onto mapped rows.
        games: Schedule used only for the timestamped schema.

    Returns:
        A copy with ``DEPTH_CHART_COLUMNS`` only, in that order.

    Raises:
        ValueError: A weekly column is absent, or a snapshot cannot be mapped.
    """
    if _has_weekly_depth_chart_columns(frame):
        return _select_weekly_depth_chart_columns(frame, season)
    if is_snapshot_depth_chart(frame):
        if games is None:
            raise ValueError(
                f"Season {season} depth chart is a timestamped snapshot "
                "and needs the games schedule"
            )
        return map_snapshot_depth_chart(frame, season, games)
    missing: list[str] = [column for column in DEPTH_CHART_COLUMNS if column not in frame.columns]
    missing_list: str = ", ".join(missing)
    raise ValueError(f"Season {season} depth chart missing columns: {missing_list}")


def download_nflverse_depth_charts(
    seasons: list[int],
    local_dir: Path,
    workspace_client: object,
    games: pd.DataFrame | None = None,
) -> list[int]:
    """Download depth charts and upload one parquet per season.

    Seasons before 2001 are skipped. A missing release (HTTP 404) is skipped.
    A weekly file that lacks a stable column is an error. A timestamped file
    (2025 and later) is mapped onto those columns with the games schedule.
    When that mapping is not possible, the season is skipped with a warning
    and ``fct_expected_starter`` keeps the previous-game primary passer.
    Files land at ``depth_charts/{season}/depth_charts_{season}.parquet`` so
    a later refresh replaces one season and leaves the others.

    Args:
        seasons: Season years to attempt.
        local_dir: Temporary directory for the parquet files.
        workspace_client: Databricks client used for the volume upload.
        games: Optional schedule. When omitted, a snapshot season downloads
            ``games.parquet`` once.

    Returns:
        Season years that were uploaded.

    Raises:
        ValueError: No season produced a file, or a weekly file failed the
            column check.
    """
    processed: list[int] = []
    games_frame: pd.DataFrame | None = games
    for season in seasons:
        if season < DEPTH_CHART_FIRST_SEASON:
            logger.info(
                "Season %s is before depth charts (%s), skipping",
                season,
                DEPTH_CHART_FIRST_SEASON,
            )
            continue
        url = NFLVERSE_DEPTH_CHARTS_URL_PATTERN.format(season=season)
        logger.info("Processing depth charts for season %s", season)
        try:
            frame = download_parquet_with_validation(url)
        except ValueError as exc:
            if "HTTP 404" in str(exc):
                logger.warning("Season %s depth chart was not found (HTTP 404), skipping", season)
                continue
            raise
        if is_snapshot_depth_chart(frame):
            try:
                if games_frame is None:
                    games_frame = download_parquet_with_validation(NFLVERSE_GAMES_URL)
                selected = select_depth_chart_columns(frame, season, games_frame)
            except ValueError as exc:
                logger.warning(
                    "Season %s depth chart uses the timestamped schema and was not "
                    "loaded (%s). Skipping this season. fct_expected_starter uses "
                    "the previous-game primary passer.",
                    season,
                    exc,
                )
                continue
        else:
            selected = select_depth_chart_columns(frame, season)
        local_path = local_dir / f"depth_charts_{season}.parquet"
        selected.to_parquet(local_path, index=False)
        upload_to_volume_depth_chart_season(workspace_client, local_path, season)
        local_path.unlink()
        processed.append(season)
    if not processed:
        raise ValueError(f"No depth chart data found for seasons {seasons}")
    logger.info("Uploaded depth charts for %s seasons", len(processed))
    return processed
