"""Map a timestamped depth-chart release onto the weekly bronze columns."""

import pandas as pd

from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

# nflverse pos_grp values on the 2025 release, and the weekly formation they are.
# An unknown group is dropped rather than guessed.
POS_GRP_TO_FORMATION: dict[str, str] = {
    "3WR 1TE": "Offense",
    "Base 4-3 D": "Defense",
    "Base 3-4 D": "Defense",
    "Special Teams": "Special Teams",
}
# Columns the snapshot mapper reads. Extra release columns are ignored.
_SNAPSHOT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "dt",
    "team",
    "gsis_id",
    "pos_grp",
    "pos_abb",
    "pos_rank",
)
# nflverse publishes gametime in Eastern time. Snapshot dt is UTC.
KICKOFF_TIMEZONE: str = "America/New_York"
_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
)


def is_snapshot_depth_chart(frame: pd.DataFrame) -> bool:
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


def _drop_unmapped_position_groups(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Map ``pos_grp`` to ``formation`` and drop groups that are not in the table.

    Raises:
        ValueError: No row has a known position group.
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
    return chart


def _parse_snapshot_timestamps(chart: pd.DataFrame, season: int) -> pd.DataFrame:
    """Parse ``dt`` as UTC and drop rows that do not parse.

    Raises:
        ValueError: No snapshot time parses.
    """
    stamped: pd.DataFrame = chart.copy()
    stamped["snapshot_at"] = pd.to_datetime(stamped["dt"], utc=True, errors="coerce")
    bad_timestamps: int = int(stamped["snapshot_at"].isna().sum())
    if bad_timestamps:
        logger.warning(
            "Season %s depth chart dropping %s rows with no timestamp",
            season,
            bad_timestamps,
        )
        stamped = stamped.loc[stamped["snapshot_at"].notna()].copy()
    if stamped.empty:
        raise ValueError(f"Season {season} depth chart has no parseable snapshot time")
    return stamped


def _team_games_with_kickoff(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Keep team-games that have an Eastern kickoff.

    Raises:
        ValueError: The season has no kickoff to attach a chart to.
    """
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
    return team_games


def _assign_snapshots_before_kickoff(
    chart: pd.DataFrame,
    team_games: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Label each team-game with the latest scrape strictly before kickoff.

    Raises:
        ValueError: No scrape is strictly before a kickoff.
    """
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
    return assigned


def _join_snapshot_rows(chart: pd.DataFrame, assigned: pd.DataFrame, season: int) -> pd.DataFrame:
    """Attach chart rows for each team and the scrape chosen for that game.

    Raises:
        ValueError: The chosen scrapes do not match any scheduled team.
    """
    chart_rows: pd.DataFrame = chart.copy()
    if "player_name" not in chart_rows.columns:
        chart_rows["player_name"] = pd.NA
    chart_rows = chart_rows.loc[
        :,
        ["dt", "team", "gsis_id", "pos_abb", "pos_rank", "formation", "player_name"],
    ].copy()
    chart_rows["team"] = chart_rows["team"].astype("string")
    placed: pd.DataFrame = assigned.copy()
    placed["club_code"] = placed["club_code"].astype("string")
    merged: pd.DataFrame = placed.merge(
        chart_rows,
        left_on=["club_code", "dt"],
        right_on=["team", "dt"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"Season {season} depth chart snapshots did not match any scheduled team")
    placed_keys: int = int(placed.loc[:, ["club_code", "dt"]].drop_duplicates().shape[0])
    found_keys: int = int(merged.loc[:, ["club_code", "dt"]].drop_duplicates().shape[0])
    if found_keys < placed_keys:
        logger.warning(
            "Season %s depth chart is missing %s team snapshots that the schedule asked for",
            season,
            placed_keys - found_keys,
        )
    return merged


def _snapshot_weekly_columns(merged: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build the weekly column frame before the shared normalizer runs."""
    depth_team: pd.Series = (
        pd.to_numeric(merged["pos_rank"], errors="coerce").round().astype("Int64").astype("string")
    )
    return pd.DataFrame(
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
        A frame with the stable weekly columns only, in that order.

    Raises:
        ValueError: The schedule cannot place any scrape before a kickoff,
            or no position group can be mapped.
    """
    chart = _drop_unmapped_position_groups(frame, season)
    chart = _parse_snapshot_timestamps(chart, season)
    team_games = _team_games_with_kickoff(games, season)
    assigned = _assign_snapshots_before_kickoff(chart, team_games, season)
    merged = _join_snapshot_rows(chart, assigned, season)
    output = _snapshot_weekly_columns(merged, season)
    logger.info(
        "Season %s depth chart mapped %s snapshot times onto %s team-games (%s rows)",
        season,
        int(assigned["dt"].nunique()),
        int(assigned.loc[:, ["club_code", "week", "game_type"]].drop_duplicates().shape[0]),
        len(output),
    )
    # Imported here so depth_charts can import this mapper without a cycle.
    from endzone.ingest.depth_charts import normalize_depth_chart_frame

    return normalize_depth_chart_frame(output).reset_index(drop=True)
