"""Pre-game injury absence counts from depth-chart starters.

The count for a team-game uses the last injury row for each starter whose
``date_modified`` is strictly before kickoff. Kickoff is ``gameday`` plus
``gametime`` in Eastern time (``America/New_York``). A row with no timestamp,
or a timestamp on or after kickoff, is not used. Week number is not a clock.
A game with no injury rows does not need a kickoff. A game that has rows
but no buildable kickoff contributes 0.

QB is excluded. The four model columns are home minus away.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

EASTERN: ZoneInfo = ZoneInfo("America/New_York")

OUT_OR_DOUBTFUL: frozenset[str] = frozenset({"Out", "Doubtful"})

# Depth-chart ``position`` values. Injury-report position is not the group.
POSITION_GROUP_BY_CHART_POSITION: dict[str, str] = {
    "C": "ol",
    "G": "ol",
    "T": "ol",
    "RB": "skill",
    "WR": "skill",
    "TE": "skill",
    "DE": "front_seven",
    "DT": "front_seven",
    "NT": "front_seven",
    "LB": "front_seven",
    "ILB": "front_seven",
    "OLB": "front_seven",
    "MLB": "front_seven",
    "CB": "db",
    "FS": "db",
    "SS": "db",
    "DB": "db",
    "S": "db",
}

INJURY_DIFF_COLUMNS: tuple[str, ...] = (
    "diff_ol_starters_out_or_doubtful",
    "diff_skill_starters_out_or_doubtful",
    "diff_front_seven_starters_out_or_doubtful",
    "diff_db_starters_out_or_doubtful",
)

_GROUP_TO_COLUMN: dict[str, str] = {
    "ol": "diff_ol_starters_out_or_doubtful",
    "skill": "diff_skill_starters_out_or_doubtful",
    "front_seven": "diff_front_seven_starters_out_or_doubtful",
    "db": "diff_db_starters_out_or_doubtful",
}

_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
)
_STARTER_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_type",
    "team",
    "gsis_id",
    "position",
)
_INJURY_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_type",
    "team",
    "gsis_id",
    "report_status",
    "date_modified",
)
_JOIN_KEYS: tuple[str, ...] = ("season", "week", "game_type", "team")
_PLAYER_KEYS: tuple[str, ...] = ("season", "week", "game_type", "team", "gsis_id")
_STARTER_DEPTH: int = 1


def attach_injury_features(
    games: pd.DataFrame,
    depth_chart_starters: pd.DataFrame | None,
    injury_reports: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return the four home-minus-away absence columns, one row per game.

    Both frames omitted leaves every column missing. Passing one without the
    other is an error, matching the starting-QB inputs.

    Args:
        games: One row per game, including schedule team, week, and kickoff.
        depth_chart_starters: Depth 1 chart rows. None only when reports are
            also None.
        injury_reports: Player report rows, including any timestamp. None only
            when starters are also None.

    Returns:
        ``game_id`` plus ``INJURY_DIFF_COLUMNS``. Counts are floats. A team
        with no qualifying absence contributes 0 when both inputs are present.

    Raises:
        ValueError: Exactly one input is missing, ``game_id`` is duplicated,
            or a required column is absent.
    """
    if "game_id" not in games.columns:
        raise ValueError("games is missing game_id")
    game_ids: pd.Series = games["game_id"].astype("string")
    if game_ids.duplicated().any():
        raise ValueError("games game_id must be unique")
    if depth_chart_starters is None and injury_reports is None:
        missing: pd.DataFrame = pd.DataFrame({"game_id": game_ids.to_numpy()})
        for column in INJURY_DIFF_COLUMNS:
            missing[column] = float("nan")
        return missing
    if depth_chart_starters is None or injury_reports is None:
        raise ValueError(
            "depth_chart_starters and injury_reports must both be provided or both be omitted"
        )
    return starter_absence_diffs(games, depth_chart_starters, injury_reports)


def starter_absence_diffs(
    games: pd.DataFrame,
    depth_chart_starters: pd.DataFrame,
    injury_reports: pd.DataFrame,
) -> pd.DataFrame:
    """Count starters listed Out or Doubtful on the last pre-kickoff report.

    Args:
        games: Schedule rows. Kickoff is ``gameday`` plus ``gametime`` in
            Eastern time. A missing clock is ignored when that game has no
            injury rows.
        depth_chart_starters: Chart rows. When ``depth_team`` is present, only
            rank 1 is a starter. QB, fullback, and specialists are not in a
            group. The group comes from chart ``position``.
        injury_reports: Report rows. The latest ``date_modified`` strictly
            before that team-game's kickoff is the report. A null timestamp
            is dropped. Out and Doubtful count. Questionable does not.

    Returns:
        One row per game. Each difference is the home count minus the away
        count. A side with no qualifying starter is 0.

    Raises:
        ValueError: A required column is missing.
    """
    _require_columns(games, _GAME_COLUMNS, "games")
    _require_columns(depth_chart_starters, _STARTER_COLUMNS, "depth_chart_starters")
    _require_columns(injury_reports, _INJURY_COLUMNS, "injury_reports")
    team_games: pd.DataFrame = _team_games(games)
    _warn_unusable_kickoff(injury_reports, team_games)
    reports: pd.DataFrame = _last_pregame_status(injury_reports, team_games)
    starters: pd.DataFrame = _grouped_starters(depth_chart_starters)
    counted: pd.DataFrame = _absence_counts(reports, starters)
    return _home_minus_away(team_games, counted)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Raise when ``frame`` does not have every name in ``columns``."""
    missing: list[str] = [column for column in columns if column not in frame.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"{label} is missing columns: {missing_list}")


def _as_int(values: pd.Series) -> pd.Series:
    """Round a week or season to nullable int."""
    return pd.to_numeric(values, errors="coerce").round().astype("Int64")


def _as_text(values: pd.Series) -> pd.Series:
    """Strip text. Empty strings stay empty so a later filter can drop them."""
    return values.astype("string").str.strip()


def _kickoff_utc(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Return kickoff as UTC. A blank clock stays missing.

    The format is ``%Y-%m-%d %H:%M:%S``. A date-only value must not decide
    the format for the rest of the series. That inference turns every later
    ``HH:MM`` into a missing kickoff when the earliest seasons have no clock.
    """
    day: pd.Series = pd.to_datetime(gameday, errors="coerce")
    clock: pd.Series = _normalized_clock(gametime)
    text: pd.Series = day.dt.strftime("%Y-%m-%d") + " " + clock
    stamp: pd.Series = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    localized: pd.Series = stamp.dt.tz_localize(
        EASTERN,
        ambiguous="NaT",
        nonexistent="shift_forward",
    )
    return localized.dt.tz_convert("UTC")


def _normalized_clock(gametime: pd.Series) -> pd.Series:
    """Return ``HH:MM:SS``, or missing when the clock cannot be read."""
    clock: pd.Series = _as_text(gametime)
    blank: pd.Series = clock.isna() | clock.str.lower().isin(["", "na", "nat", "none", "nan"])
    clock = clock.mask(blank)
    parts: pd.DataFrame = clock.str.split(":", expand=True, n=2)
    width: int = int(parts.shape[1])
    while width < 3:
        parts[width] = pd.NA
        width += 1
    hour: pd.Series = parts[0].astype("string").str.zfill(2)
    minute: pd.Series = parts[1].astype("string").str.zfill(2)
    second: pd.Series = parts[2].astype("string").fillna("00").str.zfill(2)
    normalized: pd.Series = hour + ":" + minute + ":" + second
    return normalized.mask(hour.isna() | minute.isna() | second.isna())


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per side. Kickoff may be missing when the clock is blank."""
    kickoff: pd.Series = _kickoff_utc(games["gameday"], games["gametime"])
    aligned_kickoff: pd.Series = pd.Series(kickoff.to_numpy(), dtype="datetime64[ns, UTC]")
    base: pd.DataFrame = pd.DataFrame(
        {
            "game_id": games["game_id"].astype("string").to_numpy(),
            "season": _as_int(games["season"]).to_numpy(),
            "week": _as_int(games["week"]).to_numpy(),
            "game_type": _as_text(games["game_type"]).to_numpy(),
            "kickoff_utc": aligned_kickoff.to_numpy(),
        }
    )
    base["kickoff_utc"] = aligned_kickoff
    home: pd.DataFrame = base.copy()
    home["team"] = _as_text(games["home_team"]).to_numpy()
    home["side"] = "home"
    away: pd.DataFrame = base.copy()
    away["team"] = _as_text(games["away_team"]).to_numpy()
    away["side"] = "away"
    return pd.concat([home, away], ignore_index=True)


def _warn_unusable_kickoff(injuries: pd.DataFrame, team_games: pd.DataFrame) -> None:
    """Warn when injury rows exist and the kickoff clock cannot be built.

    Games with no report rows are not counted. Their absence columns stay 0
    without a timestamp. Rows on a game with no kickoff are excluded later.
    """
    reports: pd.DataFrame = pd.DataFrame(
        {
            "season": _as_int(injuries["season"]),
            "week": _as_int(injuries["week"]),
            "game_type": _as_text(injuries["game_type"]),
            "team": _as_text(injuries["team"]),
            "gsis_id": _as_text(injuries["gsis_id"]),
        }
    )
    reports = reports.loc[reports["gsis_id"].notna() & reports["gsis_id"].ne("")]
    if reports.empty or team_games.empty:
        return
    keys: pd.DataFrame = reports.loc[:, list(_JOIN_KEYS)].drop_duplicates()
    linked: pd.DataFrame = team_games.merge(keys, on=list(_JOIN_KEYS), how="left", indicator=True)
    unusable: pd.Series = linked["kickoff_utc"].isna() & linked["_merge"].eq("both")
    n_games: int = int(linked.loc[unusable, "game_id"].nunique())
    if n_games == 0:
        return
    logger.warning(
        "%s games have injury rows and no buildable kickoff. "
        "Those rows are excluded and the absence counts are 0.",
        n_games,
    )


def _last_pregame_status(injuries: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest Out or Doubtful row strictly before each kickoff."""
    reports: pd.DataFrame = pd.DataFrame(
        {
            "season": _as_int(injuries["season"]),
            "week": _as_int(injuries["week"]),
            "game_type": _as_text(injuries["game_type"]),
            "team": _as_text(injuries["team"]),
            "gsis_id": _as_text(injuries["gsis_id"]),
            "report_status": _as_text(injuries["report_status"]),
            "date_modified": pd.to_datetime(injuries["date_modified"], utc=True, errors="coerce"),
        }
    )
    reports = reports.loc[reports["gsis_id"].notna() & reports["gsis_id"].ne("")]
    linked: pd.DataFrame = team_games.merge(reports, on=list(_JOIN_KEYS), how="left")
    known: pd.Series = linked["date_modified"].notna() & linked["kickoff_utc"].notna()
    before: pd.DataFrame = linked.loc[known & (linked["date_modified"] < linked["kickoff_utc"])]
    if before.empty:
        return before
    status_rank: pd.Series = before["report_status"].map({"Out": 2, "Doubtful": 1}).fillna(0)
    ordered: pd.DataFrame = before.assign(_status_rank=status_rank).sort_values(
        ["date_modified", "_status_rank"],
        kind="mergesort",
    )
    latest: pd.DataFrame = ordered.drop_duplicates(
        ["game_id", *list(_PLAYER_KEYS)],
        keep="last",
    )
    kept: pd.DataFrame = latest.loc[latest["report_status"].isin(OUT_OR_DOUBTFUL)]
    return kept.drop(columns=["_status_rank"])


def _grouped_starters(starters: pd.DataFrame) -> pd.DataFrame:
    """Depth-1 chart rows in the four groups. QB and specialists are dropped."""
    frame: pd.DataFrame = pd.DataFrame(
        {
            "season": _as_int(starters["season"]),
            "week": _as_int(starters["week"]),
            "game_type": _as_text(starters["game_type"]),
            "team": _as_text(starters["team"]),
            "gsis_id": _as_text(starters["gsis_id"]),
            "position": _as_text(starters["position"]).str.upper(),
        }
    )
    if "depth_team" in starters.columns:
        depth: pd.Series = _as_int(starters["depth_team"])
        frame = frame.loc[depth.eq(_STARTER_DEPTH)]
    frame["position_group"] = frame["position"].map(POSITION_GROUP_BY_CHART_POSITION)
    frame = frame.loc[frame["position_group"].notna() & frame["gsis_id"].notna()]
    frame = frame.loc[frame["gsis_id"].ne("")]
    return frame.drop_duplicates(
        list(_PLAYER_KEYS) + ["position_group"],
        keep="first",
    )


def _absence_counts(reports: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Unique starters listed Out or Doubtful, by game, side, and group."""
    if reports.empty or starters.empty:
        return pd.DataFrame(columns=["game_id", "side", "position_group", "n_starters"])
    matched: pd.DataFrame = reports.merge(starters, on=list(_PLAYER_KEYS), how="inner")
    if matched.empty:
        return pd.DataFrame(columns=["game_id", "side", "position_group", "n_starters"])
    counted: pd.DataFrame = (
        matched.groupby(["game_id", "side", "position_group"], as_index=False)["gsis_id"]
        .nunique()
        .rename(columns={"gsis_id": "n_starters"})
    )
    return counted


def _home_minus_away(team_games: pd.DataFrame, counted: pd.DataFrame) -> pd.DataFrame:
    """Home count minus away count. A missing side is 0."""
    game_ids: pd.Series = team_games["game_id"].drop_duplicates()
    output: pd.DataFrame = pd.DataFrame({"game_id": game_ids.to_numpy()})
    for group, column in _GROUP_TO_COLUMN.items():
        group_rows: pd.DataFrame = counted.loc[counted["position_group"].eq(group)]
        home: pd.DataFrame = group_rows.loc[
            group_rows["side"].eq("home"), ["game_id", "n_starters"]
        ].rename(columns={"n_starters": "home_n"})
        away: pd.DataFrame = group_rows.loc[
            group_rows["side"].eq("away"), ["game_id", "n_starters"]
        ].rename(columns={"n_starters": "away_n"})
        output = output.merge(home, on="game_id", how="left")
        output = output.merge(away, on="game_id", how="left")
        output[column] = output["home_n"].fillna(0).astype(float) - output["away_n"].fillna(
            0
        ).astype(float)
        output = output.drop(columns=["home_n", "away_n"])
    return output.loc[:, ["game_id", *INJURY_DIFF_COLUMNS]].reset_index(drop=True)
