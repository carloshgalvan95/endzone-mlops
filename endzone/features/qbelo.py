"""FiveThirtyEight starting-QB adjustment on the margin-of-victory Elo gap.

The counting formula, the 0.9 and 0.1 quarterback weights, the 3.3 Elo
factor, and the 400-point probability scale are the values on the
FiveThirtyEight methodology page (archived 15 Oct 2020). The team weight
is the same page's "every 20 games" line, written as 0.95 and 0.05.
A starter with no earlier game has adjustment 0. Ratings do not revert
between seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from endzone.features.opponent import ELO_MOV_DIFF_COLUMN

QBELO_DIFF_COLUMN: str = "qbelo538_diff"
MODEL_ELO_MOV: str = "elo_mov"
MODEL_QBELO538: str = "qbelo538"
# Home advantage on the context probabilities. The 2020 page's base is 55
# plus travel. This round keeps the preregistered 65.
QBELO_HOME_ADVANTAGE: float = 65.0
QBELO_SCALE: float = 400.0
QB_ELO_PER_VALUE: float = 3.3
QB_RATING_KEEP: float = 0.9
QB_RATING_NEW: float = 0.1
TEAM_RATING_KEEP: float = 0.95
TEAM_RATING_NEW: float = 0.05
# Game VALUE weights from the methodology page. They are not tuned.
VALUE_PASS_ATTEMPT: float = -2.2
VALUE_COMPLETION: float = 3.7
VALUE_PASSING_YARDS: float = 1.0 / 5.0
VALUE_PASSING_TD: float = 11.3
VALUE_INTERCEPTION: float = -14.1
VALUE_SACK: float = -8.0
VALUE_RUSH_ATTEMPT: float = -1.1
VALUE_RUSHING_YARDS: float = 0.6
VALUE_RUSHING_TD: float = 15.9

QB_BOX_SCORE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "team_franchise",
    "player_id",
    "pass_attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
)
_COUNT_COLUMNS: tuple[str, ...] = QB_BOX_SCORE_COLUMNS[3:]
QBELO_DETAIL_COLUMNS: tuple[str, ...] = (
    "game_id",
    "home_qb_adj",
    "away_qb_adj",
    "home_starter_resolved",
    "away_starter_resolved",
    "home_qb_has_history",
    "away_qb_has_history",
)
QB_COVERAGE_COLUMNS: tuple[str, ...] = (
    "season",
    "n",
    "starter_both_share",
    "history_both_share",
)


def qb_game_value(
    pass_attempts: float,
    completions: float,
    passing_yards: float,
    passing_tds: float,
    interceptions: float,
    sacks: float,
    rush_attempts: float,
    rushing_yards: float,
    rushing_tds: float,
) -> float:
    """Return one game's QB VALUE from the published counting formula.

    Args:
        pass_attempts: Pass attempts.
        completions: Completions.
        passing_yards: Passing yards.
        passing_tds: Passing touchdowns.
        interceptions: Interceptions thrown.
        sacks: Times sacked.
        rush_attempts: Rush attempts.
        rushing_yards: Rushing yards.
        rushing_tds: Rushing touchdowns.

    Returns:
        The game VALUE. A missing count is not accepted here. Callers
        fill a missing count with 0 before calling.

    Raises:
        ValueError: A count is not finite.
    """
    counts: tuple[float, ...] = (
        pass_attempts,
        completions,
        passing_yards,
        passing_tds,
        interceptions,
        sacks,
        rush_attempts,
        rushing_yards,
        rushing_tds,
    )
    if not all(np.isfinite(count) for count in counts):
        raise ValueError("QB VALUE counts must be finite")
    return float(
        VALUE_PASS_ATTEMPT * pass_attempts
        + VALUE_COMPLETION * completions
        + VALUE_PASSING_YARDS * passing_yards
        + VALUE_PASSING_TD * passing_tds
        + VALUE_INTERCEPTION * interceptions
        + VALUE_SACK * sacks
        + VALUE_RUSH_ATTEMPT * rush_attempts
        + VALUE_RUSHING_YARDS * rushing_yards
        + VALUE_RUSHING_TD * rushing_tds
    )


def rating_win_probability(diff: np.ndarray) -> np.ndarray:
    """Home win probability from a rating gap plus home advantage 65.

    The scale is 400. ``diff`` does not already include home advantage.
    The formula is ``1 / (10 ** (-(diff + 65) / 400) + 1)``.

    Args:
        diff: Home rating minus away rating, in Elo points.

    Returns:
        One probability per input.

    Raises:
        ValueError: A gap is not finite.
    """
    gaps: np.ndarray = np.asarray(diff, dtype=float)
    if len(gaps) > 0 and not np.isfinite(gaps).all():
        raise ValueError("rating gaps must be finite")
    exponent: np.ndarray = -((gaps + QBELO_HOME_ADVANTAGE) / QBELO_SCALE)
    return 1.0 / (np.power(10.0, exponent) + 1.0)


def _text_id(values: pd.Series) -> pd.Series:
    """Strip ids. A blank or a text nan is missing."""
    text: pd.Series = values.astype("string").str.strip()
    missing: pd.Series = text.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "<na>"])
    return text.mask(missing)


def _prepare_box_scores(box_scores: pd.DataFrame | None) -> pd.DataFrame:
    """Sum duplicate player-games and fill a missing count with 0."""
    if box_scores is None or len(box_scores) == 0:
        return pd.DataFrame(columns=list(QB_BOX_SCORE_COLUMNS))
    missing: list[str] = [
        column for column in QB_BOX_SCORE_COLUMNS if column not in box_scores.columns
    ]
    if missing:
        joined: str = ", ".join(missing)
        raise ValueError(f"qb box scores are missing columns: {joined}")
    frame: pd.DataFrame = box_scores.loc[:, list(QB_BOX_SCORE_COLUMNS)].copy()
    frame["game_id"] = frame["game_id"].astype("string")
    frame["player_id"] = _text_id(frame["player_id"])
    frame["team_franchise"] = _text_id(frame["team_franchise"])
    for column in _COUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    kept: pd.Series = frame["player_id"].notna() & frame["team_franchise"].notna()
    frame = frame.loc[kept]
    if frame.empty:
        return pd.DataFrame(columns=list(QB_BOX_SCORE_COLUMNS))
    grouped: pd.DataFrame = frame.groupby(
        ["game_id", "player_id", "team_franchise"],
        as_index=False,
        sort=False,
    )[list(_COUNT_COLUMNS)].sum()
    values: list[float] = [
        qb_game_value(*(float(getattr(row, column)) for column in _COUNT_COLUMNS))
        for row in grouped.itertuples(index=False)
    ]
    grouped["value"] = values
    return grouped


def _side_adjustment(
    starter: str | None,
    franchise: str | None,
    resolved: bool,
    player_rating: dict[str, float],
    player_seen: set[str],
    team_rating: dict[str, float],
    team_seen: set[str],
) -> tuple[float, bool, bool]:
    """Pregame adjustment for one side, read before this date's updates."""
    if not resolved or starter is None or starter not in player_seen:
        return 0.0, resolved, False
    team_value: float = (
        team_rating[franchise] if franchise in team_seen and franchise is not None else 0.0
    )
    return float(player_rating[starter] - team_value), True, True


def _team_game_value(
    rows: pd.DataFrame,
    starter: str | None,
) -> float | None:
    """One team-game VALUE. The starter's line wins when it exists."""
    if rows.empty:
        return None
    if starter is not None:
        starter_rows: pd.DataFrame = rows.loc[rows["player_id"] == starter]
        if not starter_rows.empty:
            return float(starter_rows["value"].sum())
    ranked: pd.DataFrame = rows.sort_values(
        ["pass_attempts", "rush_attempts", "player_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["value"])


def qbelo_pregame_table(
    games: pd.DataFrame,
    starters: pd.DataFrame,
    box_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    """Pregame QB adjustments from games strictly before each date.

    The starter ids and sources come from the H5 starter table. This
    function does not choose a starter. A same-day game is not history.
    A player rating starts at 0 and then keeps 0.9 of the old rating plus
    0.1 of the new game VALUE. A team rating keeps 0.95 and adds 0.05.
    A starter with no earlier game has adjustment 0, including when the
    team already has a rating.

    Args:
        games: One row per game, with franchise columns and ``gameday``.
        starters: H5 output. Needs starter ids and sources.
        box_scores: Player-game counting stats. None leaves every
            adjustment at 0.

    Returns:
        One row per game with the adjustment and the coverage flags.

    Raises:
        ValueError: A required column is missing or ``game_id`` repeats.
    """
    required: tuple[str, ...] = (
        "game_id",
        "gameday",
        "home_team_franchise",
        "away_team_franchise",
    )
    missing_games: list[str] = [column for column in required if column not in games.columns]
    if missing_games:
        raise ValueError(f"games are missing columns: {', '.join(missing_games)}")
    starter_columns: tuple[str, ...] = (
        "game_id",
        "home_starter_player_id",
        "away_starter_player_id",
        "home_starter_source",
        "away_starter_source",
    )
    missing_starters: list[str] = [
        column for column in starter_columns if column not in starters.columns
    ]
    if missing_starters:
        raise ValueError(f"starters are missing columns: {', '.join(missing_starters)}")
    if games["game_id"].duplicated().any():
        raise ValueError("games game_id must be unique")

    schedule: pd.DataFrame = games.loc[:, list(required)].copy()
    schedule["game_id"] = schedule["game_id"].astype("string")
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    schedule["home_team_franchise"] = _text_id(schedule["home_team_franchise"])
    schedule["away_team_franchise"] = _text_id(schedule["away_team_franchise"])
    starter_frame: pd.DataFrame = starters.loc[:, list(starter_columns)].copy()
    starter_frame["game_id"] = starter_frame["game_id"].astype("string")
    for column in (
        "home_starter_player_id",
        "away_starter_player_id",
        "home_starter_source",
        "away_starter_source",
    ):
        starter_frame[column] = _text_id(starter_frame[column])
    merged: pd.DataFrame = schedule.merge(starter_frame, on="game_id", how="left")
    boxes: pd.DataFrame = _prepare_box_scores(box_scores)
    known_ids: set[str] = set(merged["game_id"])
    if not boxes.empty:
        boxes = boxes.loc[boxes["game_id"].isin(known_ids)].copy()
    boxes_by_game: dict[str, pd.DataFrame] = {}
    if not boxes.empty:
        for game_id, part in boxes.groupby("game_id", sort=False):
            boxes_by_game[str(game_id)] = part

    player_rating: dict[str, float] = {}
    player_seen: set[str] = set()
    team_rating: dict[str, float] = {}
    team_seen: set[str] = set()
    rows: list[dict[str, object]] = []
    ordered: pd.DataFrame = merged.sort_values(["gameday", "game_id"], kind="mergesort")
    for _gameday, day in ordered.groupby("gameday", sort=False):
        day_rows: list[dict[str, object]] = []
        for game in day.itertuples(index=False):
            record: dict[str, object] = {"game_id": game.game_id}
            for side in ("home", "away"):
                starter_raw: object = getattr(game, f"{side}_starter_player_id")
                source_raw: object = getattr(game, f"{side}_starter_source")
                franchise_raw: object = getattr(game, f"{side}_team_franchise")
                starter: str | None = None if pd.isna(starter_raw) else str(starter_raw)
                franchise: str | None = None if pd.isna(franchise_raw) else str(franchise_raw)
                resolved: bool = starter is not None and not pd.isna(source_raw)
                adjustment, side_resolved, has_history = _side_adjustment(
                    starter,
                    franchise,
                    resolved,
                    player_rating,
                    player_seen,
                    team_rating,
                    team_seen,
                )
                record[f"{side}_qb_adj"] = adjustment
                record[f"{side}_starter_resolved"] = side_resolved
                record[f"{side}_qb_has_history"] = has_history
                record[f"_{side}_starter"] = starter
                record[f"_{side}_franchise"] = franchise
            day_rows.append(record)
        for record in day_rows:
            game_boxes: pd.DataFrame = boxes_by_game.get(str(record["game_id"]), boxes.iloc[0:0])
            for side in ("home", "away"):
                franchise = record[f"_{side}_franchise"]
                if franchise is None or game_boxes.empty:
                    team_value = None
                else:
                    team_rows: pd.DataFrame = game_boxes.loc[
                        game_boxes["team_franchise"] == franchise
                    ]
                    team_value = _team_game_value(team_rows, record[f"_{side}_starter"])
                if team_value is None or franchise is None:
                    continue
                previous: float = team_rating.get(str(franchise), 0.0)
                team_rating[str(franchise)] = TEAM_RATING_KEEP * previous + TEAM_RATING_NEW * float(
                    team_value
                )
                team_seen.add(str(franchise))
            if game_boxes.empty:
                continue
            player_values: pd.Series = game_boxes.groupby("player_id", sort=False)["value"].sum()
            for player_id, value in player_values.items():
                player_key: str = str(player_id)
                previous_player: float = player_rating.get(player_key, 0.0)
                player_rating[player_key] = (
                    QB_RATING_KEEP * previous_player + QB_RATING_NEW * float(value)
                )
                player_seen.add(player_key)
        for record in day_rows:
            rows.append({column: record[column] for column in QBELO_DETAIL_COLUMNS})
    detail: pd.DataFrame = pd.DataFrame(rows, columns=list(QBELO_DETAIL_COLUMNS))
    return detail


def qbelo538_diff_frame(
    elo_diff: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``3.3 * (home_qb_adj - away_qb_adj)`` to ``elo_mov_diff``.

    Args:
        elo_diff: ``game_id`` and ``elo_mov_diff``.
        detail: Output of ``qbelo_pregame_table``.

    Returns:
        ``game_id`` and ``qbelo538_diff``.

    Raises:
        ValueError: A game is missing from the adjustment table.
    """
    if ELO_MOV_DIFF_COLUMN not in elo_diff.columns or "game_id" not in elo_diff.columns:
        raise ValueError("elo diff frame needs game_id and elo_mov_diff")
    merged: pd.DataFrame = elo_diff.loc[:, ["game_id", ELO_MOV_DIFF_COLUMN]].merge(
        detail.loc[:, ["game_id", "home_qb_adj", "away_qb_adj"]],
        on="game_id",
        how="left",
    )
    if merged["home_qb_adj"].isna().any() or merged["away_qb_adj"].isna().any():
        raise ValueError("qbelo adjustment is missing for a game")
    out: pd.DataFrame = pd.DataFrame({"game_id": merged["game_id"].to_numpy()})
    out[QBELO_DIFF_COLUMN] = merged[ELO_MOV_DIFF_COLUMN].to_numpy(
        dtype=float
    ) + QB_ELO_PER_VALUE * (
        merged["home_qb_adj"].to_numpy(dtype=float) - merged["away_qb_adj"].to_numpy(dtype=float)
    )
    return out


def qb_coverage_by_season(
    games: pd.DataFrame,
    detail: pd.DataFrame,
    seasons: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Share of games with both starters resolved, and with prior VALUE.

    Args:
        games: Schedule with ``game_id`` and ``season``.
        detail: Output of ``qbelo_pregame_table``.
        seasons: Seasons to keep. None keeps every season in ``games``.

    Returns:
        One row per season with ``n``, ``starter_both_share``, and
        ``history_both_share``.
    """
    merged: pd.DataFrame = games.loc[:, ["game_id", "season"]].merge(
        detail, on="game_id", how="inner"
    )
    if seasons is not None:
        allowed: set[int] = {int(season) for season in seasons}
        merged = merged.loc[merged["season"].astype(int).isin(allowed)]
    rows: list[dict[str, object]] = []
    if merged.empty:
        return pd.DataFrame(columns=list(QB_COVERAGE_COLUMNS))
    for season, part in merged.groupby(merged["season"].astype(int), sort=True):
        both_starters: pd.Series = part["home_starter_resolved"] & part["away_starter_resolved"]
        both_history: pd.Series = part["home_qb_has_history"] & part["away_qb_has_history"]
        rows.append(
            {
                "season": int(season),
                "n": int(len(part)),
                "starter_both_share": float(both_starters.mean()),
                "history_both_share": float(both_history.mean()),
            }
        )
    return pd.DataFrame(rows, columns=list(QB_COVERAGE_COLUMNS))
