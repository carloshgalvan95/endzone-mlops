"""
Load raw nflverse data into Databricks Free Edition.

Downloads nflverse datasets (games, play-by-play, weekly depth charts) from
official GitHub releases, uploads to Unity Catalog Volumes, and creates
bronze Delta tables.

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
# Weekly NFL.com depth charts. The release starts in 2001. There is no 1999 file.
NFLVERSE_DEPTH_CHARTS_URL_PATTERN = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.parquet"
)
DEPTH_CHART_FIRST_SEASON = 2001
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
# Columns the snapshot mapper reads. Extra release columns are ignored.
_SNAPSHOT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "dt",
    "team",
    "gsis_id",
    "pos_grp",
    "pos_abb",
    "pos_rank",
)
# nflverse pos_grp values on the 2025 release, and the weekly formation they are.
# An unknown group is dropped rather than guessed.
POS_GRP_TO_FORMATION: dict[str, str] = {
    "3WR 1TE": "Offense",
    "Base 4-3 D": "Defense",
    "Base 3-4 D": "Defense",
    "Special Teams": "Special Teams",
}
# Hint read_files so a leftover int/double week mix still parses as one int column.
# The upload path already writes nullable int32. This hint is the belt.
DEPTH_CHART_READ_SCHEMA_HINT: str = "season int, week int, depth_team string, club_code string"
# nflverse publishes gametime in Eastern time. Snapshot dt is UTC.
KICKOFF_TIMEZONE = "America/New_York"
_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
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
    "depth_charts": {
        "filename_pattern": "depth_charts_{season}.parquet",
        "table_name": "depth_charts",
        "description": "NFL depth charts (2001-present, 2025+ snapshots mapped to weeks)",
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


def _has_weekly_depth_chart_columns(frame: pd.DataFrame) -> bool:
    """Return True when the frame already has the 2001-2024 weekly columns."""
    return all(column in frame.columns for column in DEPTH_CHART_COLUMNS)


def _is_snapshot_depth_chart(frame: pd.DataFrame) -> bool:
    """Return True for the timestamped schema that starts in 2025.

    A file that still has ``week`` stays on the weekly path. The snapshot
    release has no week.
    """
    if "week" in frame.columns:
        return False
    return all(column in frame.columns for column in _SNAPSHOT_REQUIRED_COLUMNS)


def _blank_string_column(index: pd.Index) -> pd.Series:
    """Return an all-null string column aligned to ``index``.

    An all-null object column is written as a null parquet type. The weekly
    files use string, so the mapped file has to use string too.
    """
    return pd.Series(pd.NA, index=index, dtype="string")


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


def _kickoff_utc(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Build a UTC kickoff from an Eastern ``gameday`` and ``gametime``.

    nflverse ``gametime`` is Eastern (``America/New_York``), including the
    daylight-saving offset. A clock time that cannot be placed on that date
    becomes null, and the caller drops that game.

    Args:
        gameday: Game date as ``YYYY-MM-DD``.
        gametime: Kickoff clock time, usually ``HH:MM``.

    Returns:
        Kickoff timestamps in UTC, null where the pair does not parse.
    """
    combined: pd.Series = (
        gameday.astype("string").str.strip() + " " + gametime.astype("string").str.strip()
    )
    naive: pd.Series = pd.to_datetime(combined, errors="coerce", format="%Y-%m-%d %H:%M")
    failed: pd.Index = naive.index[naive.isna()]
    if len(failed):
        naive.loc[failed] = pd.to_datetime(combined.loc[failed], errors="coerce")
    localized: pd.Series = naive.dt.tz_localize(
        KICKOFF_TIMEZONE,
        ambiguous="NaT",
        nonexistent="NaT",
    )
    return localized.dt.tz_convert("UTC")


def _team_games_for_season(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Expand one season of the schedule to one row per team-game.

    Args:
        games: nflverse games, one row per game. Other seasons are ignored.
        season: Season year to keep.

    Returns:
        Columns ``club_code``, ``week``, ``game_type``, and ``kickoff_at``.

    Raises:
        ValueError: The schedule is missing a column this join needs, or the
            season has no games.
    """
    missing: list[str] = [column for column in _SCHEDULE_COLUMNS if column not in games.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Games schedule is missing columns: {missing_list}")
    season_number: pd.Series = pd.to_numeric(games["season"], errors="coerce")
    schedule: pd.DataFrame = games.loc[season_number == season].copy()
    if schedule.empty:
        raise ValueError(f"Games schedule has no rows for season {season}")
    schedule["kickoff_at"] = _kickoff_utc(schedule["gameday"], schedule["gametime"])
    home: pd.DataFrame = schedule.assign(club_code=schedule["home_team"].astype("string"))
    away: pd.DataFrame = schedule.assign(club_code=schedule["away_team"].astype("string"))
    team_games: pd.DataFrame = pd.concat([home, away], ignore_index=True)
    return team_games.loc[:, ["club_code", "week", "game_type", "kickoff_at"]].copy()


def map_snapshot_depth_chart(
    frame: pd.DataFrame,
    season: int,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Map a timestamped depth chart onto the weekly bronze columns.

    The 2025 release is one league-wide scrape per ``dt`` (UTC, ``Z`` suffix).
    It has ``team``, ``pos_grp``, ``pos_abb``, ``pos_rank``, and ``gsis_id``.
    It does not have ``week`` or ``game_type``.

    For each team-game in ``season``, keep the latest scrape whose timestamp
    is strictly before that game's kickoff, then label those rows with that
    game's ``week`` and ``game_type``. Kickoff is ``gameday`` plus ``gametime``
    in Eastern time. A Thursday game therefore cannot see Friday's scrape, and
    a Sunday game can. Scrapes after the last kickoff (the following offseason)
    are not assigned to this season.

    ``team`` becomes ``club_code``. ``pos_abb`` becomes ``position`` (``QB``
    stays ``QB``). ``pos_rank`` becomes ``depth_team`` as text. ``pos_grp``
    becomes ``formation`` through ``POS_GRP_TO_FORMATION``. ``player_name``
    becomes ``full_name`` when it is present. Given name, jersey number, and
    ``elias_id`` are not on the release, so those columns stay null.

    Args:
        frame: One season in the timestamped schema.
        season: Season year written onto every output row.
        games: nflverse schedule used to place each scrape on a game.

    Returns:
        A frame with ``DEPTH_CHART_COLUMNS`` only, in that order.

    Raises:
        ValueError: The schedule cannot place any scrape before a kickoff,
            or no position group can be mapped.
    """
    chart: pd.DataFrame = frame.copy()
    chart["formation"] = chart["pos_grp"].map(POS_GRP_TO_FORMATION)
    unmapped: pd.Series = chart.loc[chart["formation"].isna(), "pos_grp"]
    if not unmapped.empty:
        unknown_groups: list[str] = sorted(set(unmapped.astype("string").fillna("<null>")))
        logger.warning(
            "Season %s depth chart dropping unmapped position groups: %s",
            season,
            ", ".join(unknown_groups),
        )
    chart = chart.loc[chart["formation"].notna()].copy()
    if chart.empty:
        raise ValueError(f"Season {season} depth chart has no mappable position group")

    chart["snapshot_at"] = pd.to_datetime(chart["dt"], utc=True, errors="coerce")
    bad_timestamps: int = int(chart["snapshot_at"].isna().sum())
    if bad_timestamps:
        logger.warning(
            "Season %s depth chart dropping %s rows with no timestamp",
            season,
            bad_timestamps,
        )
        chart = chart.loc[chart["snapshot_at"].notna()].copy()
    if chart.empty:
        raise ValueError(f"Season {season} depth chart has no parseable snapshot time")

    team_games: pd.DataFrame = _team_games_for_season(games, season)
    missing_kickoff: pd.Series = team_games["kickoff_at"].isna()
    if bool(missing_kickoff.any()):
        logger.warning(
            "Season %s dropping %s team-games with no Eastern kickoff",
            season,
            int(missing_kickoff.sum()),
        )
        team_games = team_games.loc[~missing_kickoff].copy()
    if team_games.empty:
        raise ValueError(f"Season {season} schedule has no kickoff to attach a depth chart to")

    snapshot_times: pd.DataFrame = (
        chart.loc[:, ["dt", "snapshot_at"]]
        .drop_duplicates(subset=["snapshot_at"])
        .sort_values("snapshot_at", kind="mergesort")
    )
    assigned: pd.DataFrame = pd.merge_asof(
        team_games.sort_values("kickoff_at", kind="mergesort"),
        snapshot_times,
        left_on="kickoff_at",
        right_on="snapshot_at",
        direction="backward",
        allow_exact_matches=False,
    )
    unmatched: int = int(assigned["dt"].isna().sum())
    if unmatched:
        logger.warning(
            "Season %s has %s team-games with no depth-chart snapshot strictly before kickoff",
            season,
            unmatched,
        )
    assigned = assigned.dropna(subset=["dt"]).copy()
    if assigned.empty:
        raise ValueError(f"Season {season} has no depth-chart snapshot strictly before a kickoff")

    if "player_name" not in chart.columns:
        chart["player_name"] = pd.NA
    chart_rows: pd.DataFrame = chart.loc[
        :,
        ["dt", "team", "gsis_id", "pos_abb", "pos_rank", "formation", "player_name"],
    ].copy()
    chart_rows["team"] = chart_rows["team"].astype("string")
    assigned["club_code"] = assigned["club_code"].astype("string")
    merged: pd.DataFrame = assigned.merge(
        chart_rows,
        left_on=["club_code", "dt"],
        right_on=["team", "dt"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"Season {season} depth chart snapshots did not match any scheduled team")
    placed_keys: int = int(assigned.loc[:, ["club_code", "dt"]].drop_duplicates().shape[0])
    found_keys: int = int(merged.loc[:, ["club_code", "dt"]].drop_duplicates().shape[0])
    if found_keys < placed_keys:
        logger.warning(
            "Season %s depth chart is missing %s team snapshots that the schedule asked for",
            season,
            placed_keys - found_keys,
        )

    depth_team: pd.Series = (
        pd.to_numeric(merged["pos_rank"], errors="coerce").round().astype("Int64").astype("string")
    )
    output: pd.DataFrame = pd.DataFrame(
        {
            "season": pd.Series(season, index=merged.index, dtype="int32"),
            "club_code": merged["club_code"].astype("string"),
            "week": merged["week"].astype("float64"),
            "game_type": merged["game_type"].astype("string"),
            "depth_team": depth_team,
            "last_name": _blank_string_column(merged.index),
            "first_name": _blank_string_column(merged.index),
            "football_name": _blank_string_column(merged.index),
            "formation": merged["formation"].astype("string"),
            "gsis_id": merged["gsis_id"].astype("string"),
            "jersey_number": _blank_string_column(merged.index),
            "position": merged["pos_abb"].astype("string"),
            "elias_id": _blank_string_column(merged.index),
            "depth_position": _blank_string_column(merged.index),
            "full_name": merged["player_name"].astype("string"),
        }
    )
    logger.info(
        "Season %s depth chart mapped %s snapshot times onto %s team-games (%s rows)",
        season,
        int(assigned["dt"].nunique()),
        int(assigned.loc[:, ["club_code", "week", "game_type"]].drop_duplicates().shape[0]),
        len(output),
    )
    return normalize_depth_chart_frame(output).reset_index(drop=True)


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
    if _is_snapshot_depth_chart(frame):
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
    workspace_client: WorkspaceClient,
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
        if _is_snapshot_depth_chart(frame):
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


def upload_to_volume_depth_chart_season(
    workspace_client: WorkspaceClient,
    local_path: Path,
    season: int,
) -> str:
    """Upload one season of depth charts to the landing volume.

    Args:
        workspace_client: Databricks WorkspaceClient.
        local_path: Local parquet path.
        season: Season year used in the volume directory.

    Returns:
        Volume path of the uploaded file.
    """
    volume_root = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}"
    volume_path = f"{volume_root}/depth_charts/{season}/{local_path.name}"
    logger.info("Uploading %s to %s", local_path.name, volume_path)
    with open(local_path, "rb") as handle:
        workspace_client.files.upload(volume_path, handle, overwrite=True)
    logger.info("Uploaded depth charts to %s", volume_path)
    return volume_path


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
    elif dataset == "depth_charts":
        read_path = f"/Volumes/{CATALOG}/{SCHEMA_RAW}/{VOLUME_NAME}/depth_charts"
    else:
        # For games, use the specific file path
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
        choices=["games", "pbp", "depth_charts", "all"],
        default="all",
        help="Dataset to load (default: all). depth_charts is the weekly starter chart.",
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
        depth_chart_seasons = [current]
        games_seasons = all_seasons
        logger.info(
            "Current season mode: pbp and depth charts load %s only, games loads full range",
            current,
        )
    else:
        pbp_seasons = all_seasons
        depth_chart_seasons = all_seasons
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

                if args.dataset in ("depth_charts", "all"):
                    logger.info("Loading depth charts for seasons: %s", depth_chart_seasons)
                    with TemporaryDirectory() as depth_dir:
                        processed_charts = download_nflverse_depth_charts(
                            depth_chart_seasons,
                            Path(depth_dir),
                            workspace_client,
                        )
                    if not processed_charts:
                        logger.warning("No depth charts processed, skipping bronze table creation")
                    else:
                        create_bronze_table(
                            sql_connection,
                            "depth_charts",
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
