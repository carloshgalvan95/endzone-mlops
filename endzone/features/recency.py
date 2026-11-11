"""Preregistered recency and streak columns.

These columns are challengers to logistic ``h2_full``. They are not a new
reference. The choices below are fixed before any model fit:

- One recent-EPA variant: the cross-season mean of the last 5 team-games.
  A 3-game mean and an exponential mean are not registered.
- Recent point differential uses that same 5-game window.
- Signed streak is the current run of wins or losses, not a 5-game count.
- Elo difference is the pre-game home rating minus the away rating.
  Home advantage is not added.
- None of the four blocks are crossed with the others.
- History is every team-game with ``gameday`` strictly before the game,
  including earlier seasons. Week 1 does not reset the window or the streak.
- Fewer than 5 prior games use the games that exist. Zero prior games stay
  missing. A row is not dropped.
- A tie is margin 0, occupies a slot in the point window, and breaks a streak.
- ``lookback_games`` on the h2 builder does not change these columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

RECENCY_WINDOW_GAMES: int = 5

FEATURE_SET_RECENCY_EPA5: str = "recency_epa5"
FEATURE_SET_RECENCY_MARGIN: str = "recency_margin"
FEATURE_SET_RECENCY_STREAK: str = "recency_streak"
FEATURE_SET_RECENCY_ELO: str = "recency_elo"

RECENCY_CHALLENGER_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_RECENCY_EPA5,
    FEATURE_SET_RECENCY_MARGIN,
    FEATURE_SET_RECENCY_STREAK,
    FEATURE_SET_RECENCY_ELO,
)

# Cross-season recent EPA columns in h2_full that recency_epa5 replaces.
# Season EPA, regressed EPA, win rate, and rest stay on the 10-game recipe.
REPLACED_RECENT_EPA_COLUMNS: tuple[str, ...] = (
    "home_recent_offensive_epa_per_play",
    "home_recent_defensive_epa_per_play",
    "away_recent_offensive_epa_per_play",
    "away_recent_defensive_epa_per_play",
)

RECENCY_EPA5_COLUMNS: tuple[str, ...] = (
    "home_recent5_offensive_epa_per_play",
    "home_recent5_defensive_epa_per_play",
    "away_recent5_offensive_epa_per_play",
    "away_recent5_defensive_epa_per_play",
)
RECENCY_POINT_DIFF_COLUMNS: tuple[str, ...] = (
    "home_recent_point_diff",
    "away_recent_point_diff",
)
RECENCY_STREAK_COLUMNS: tuple[str, ...] = (
    "home_signed_streak",
    "away_signed_streak",
)
RECENCY_ELO_COLUMNS: tuple[str, ...] = ("elo_rating_diff",)
RECENCY_OUTPUT_COLUMNS: tuple[str, ...] = (
    RECENCY_EPA5_COLUMNS + RECENCY_POINT_DIFF_COLUMNS + RECENCY_STREAK_COLUMNS + RECENCY_ELO_COLUMNS
)

_GAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "gameday",
    "home_team_franchise",
    "away_team_franchise",
    "home_won",
)
_EPA_COLUMNS: tuple[str, ...] = (
    "game_id",
    "gameday",
    "team_franchise",
    "avg_offensive_epa",
    "avg_defensive_epa_allowed",
)
_SIDE_TEAM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("home", "home_team_franchise"),
    ("away", "away_team_franchise"),
)


class _TeamArrays(NamedTuple):
    """One team's prior rows, sorted by gameday then game_id."""

    dates: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray


def signed_result_streak(results: Sequence[int]) -> float:
    """Signed length of the current win or loss run.

    ``results`` is chronological. ``1`` is a win, ``-1`` is a loss, and ``0``
    is a tie. An empty sequence is missing. A trailing tie is 0, which is a
    known broken streak, not missing history.

    Args:
        results: Prior results, oldest first.

    Returns:
        The signed streak, or NaN when ``results`` is empty.

    Raises:
        ValueError: A value is not -1, 0, or 1.
    """
    if len(results) == 0:
        return float("nan")
    latest: int = int(results[-1])
    if latest not in (-1, 0, 1):
        raise ValueError("streak results must be -1, 0, or 1")
    if latest == 0:
        return 0.0
    length: int = 0
    for value in reversed(results):
        if int(value) != latest:
            break
        length += 1
    return float(latest * length)


def _result_sign(margin: float) -> int:
    """Map a point margin to a win, a loss, or a tie."""
    if margin > 0.0:
        return 1
    if margin < 0.0:
        return -1
    return 0


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Raise when ``frame`` is missing any of ``columns``."""
    missing: list[str] = [column for column in columns if column not in frame.columns]
    if missing:
        joined: str = ", ".join(missing)
        raise ValueError(f"{label} missing columns: {joined}")


def _nanmean(values: np.ndarray) -> float:
    """Mean that skips NaN. An empty or all-missing slice stays missing."""
    if values.size == 0 or bool(np.isnan(values).all()):
        return float("nan")
    return float(np.nanmean(values))


def _history_end(dates: np.ndarray, game_date: object) -> int:
    """Index of the first row that is not strictly before ``game_date``.

    ``side="left"`` keeps a same-gameday row out of the prefix. A missing
    date has an empty prefix.
    """
    if dates.size == 0 or pd.isna(game_date):
        return 0
    cutoff: np.datetime64 = np.datetime64(pd.Timestamp(game_date), "ns")
    return int(np.searchsorted(dates, cutoff, side="left"))


def _window_mean(values: np.ndarray, end: int, window: int) -> float:
    """Mean of the last ``window`` values in ``values[:end]``."""
    if end <= 0:
        return float("nan")
    start: int = max(0, end - window)
    return _nanmean(values[start:end])


def _group_teams(
    frame: pd.DataFrame,
    primary: str,
    secondary: str,
    secondary_dtype: type[np.generic],
) -> dict[str, _TeamArrays]:
    """Sort team rows and store dates plus two value columns."""
    histories: dict[str, _TeamArrays] = {}
    if frame.empty:
        return histories
    ordered: pd.DataFrame = frame.sort_values(
        ["team_franchise", "gameday", "game_id"],
        kind="mergesort",
    )
    for team, part in ordered.groupby("team_franchise", sort=False):
        histories[str(team)] = _TeamArrays(
            dates=pd.to_datetime(part["gameday"]).to_numpy(dtype="datetime64[ns]"),
            primary=part[primary].to_numpy(dtype=float),
            secondary=part[secondary].to_numpy(dtype=secondary_dtype),
        )
    return histories


def _empty_history() -> _TeamArrays:
    """A team with no prior rows."""
    return _TeamArrays(
        dates=np.array([], dtype="datetime64[ns]"),
        primary=np.array([], dtype=float),
        secondary=np.array([], dtype=float),
    )


def _result_frame(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team for each game that has both final scores.

    Ties stay in the frame with margin 0. A null score is not a result.
    ``home_won`` is not the source: the games mart stores a tie as 0.
    """
    columns: list[str] = ["game_id", "gameday", "team_franchise", "margin", "result"]
    if "home_score" not in games.columns or "away_score" not in games.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        home_score: object = game.home_score
        away_score: object = game.away_score
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        margin: float = float(home_score) - float(away_score)
        sign: int = _result_sign(margin)
        away_sign: int = 0 if sign == 0 else -sign
        gameday: object = game.gameday
        game_id: object = game.game_id
        rows.append(
            {
                "game_id": game_id,
                "gameday": gameday,
                "team_franchise": game.home_team_franchise,
                "margin": margin,
                "result": sign,
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "gameday": gameday,
                "team_franchise": game.away_team_franchise,
                "margin": -margin,
                "result": away_sign,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _elo_rating_diff(games: pd.DataFrame) -> np.ndarray:
    """Home pre-game Elo minus away pre-game Elo, without home advantage.

    The import stays inside the function so this module can be loaded while
    ``pregame`` is still initializing.
    """
    from endzone.features.pregame import elo_home_probabilities

    aligned: pd.DataFrame = games.reset_index(drop=True)
    fitted: dict[str, object] = elo_home_probabilities(aligned)
    home: np.ndarray = np.asarray(fitted["home_ratings"], dtype=float)
    away: np.ndarray = np.asarray(fitted["away_ratings"], dtype=float)
    if len(home) != len(aligned) or len(away) != len(aligned):
        raise ValueError("Elo ratings do not align with the games frame")
    return home - away


def attach_recency_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Build the preregistered recency columns for every game.

    EPA uses ``team_games``. Point differential and streak use final scores
    on ``games``. Elo uses the same update as the baseline probability.
    A same-gameday game is not history. A later gameday in the same week is
    not history. An earlier gameday is history even when the season changes.

    Args:
        games: One row per game. Needs identifiers, franchises, and ``home_won``.
            ``home_score`` and ``away_score`` are optional. Without them the
            point and streak columns stay missing.
        team_games: Team-game EPA. Needs the offensive and defensive means.

    Returns:
        One row per input game, with ``game_id`` and ``RECENCY_OUTPUT_COLUMNS``.

    Raises:
        ValueError: A required column is missing or ``game_id`` is duplicated.
    """
    _require_columns(games, _GAME_COLUMNS, "games")
    _require_columns(team_games, _EPA_COLUMNS, "fct_team_game_epa")
    if games["game_id"].duplicated().any():
        raise ValueError("game_id must be unique to attach recency features")

    game_frame: pd.DataFrame = games.copy()
    game_frame["gameday"] = pd.to_datetime(game_frame["gameday"])
    epa_frame: pd.DataFrame = team_games.loc[:, list(_EPA_COLUMNS)].copy()
    epa_frame["gameday"] = pd.to_datetime(epa_frame["gameday"])
    epa_frame = epa_frame.loc[epa_frame["gameday"].notna()]
    offensive: dict[str, _TeamArrays] = _group_teams(
        epa_frame,
        "avg_offensive_epa",
        "avg_defensive_epa_allowed",
        np.float64,
    )
    results: pd.DataFrame = _result_frame(game_frame)
    results = results.loc[pd.to_datetime(results["gameday"]).notna()] if len(results) else results
    margins: dict[str, _TeamArrays] = _group_teams(results, "margin", "result", np.float64)
    elo_diff: np.ndarray = _elo_rating_diff(game_frame)

    rows: list[dict[str, object]] = []
    for position, game in enumerate(game_frame.itertuples(index=False)):
        record: dict[str, object] = {
            "game_id": game.game_id,
            "elo_rating_diff": float(elo_diff[position]),
        }
        for side, team_column in _SIDE_TEAM_COLUMNS:
            team: str = str(getattr(game, team_column))
            epa_history: _TeamArrays = offensive.get(team, _empty_history())
            result_history: _TeamArrays = margins.get(team, _empty_history())
            epa_end: int = _history_end(epa_history.dates, game.gameday)
            result_end: int = _history_end(result_history.dates, game.gameday)
            record[f"{side}_recent5_offensive_epa_per_play"] = _window_mean(
                epa_history.primary,
                epa_end,
                RECENCY_WINDOW_GAMES,
            )
            record[f"{side}_recent5_defensive_epa_per_play"] = _window_mean(
                epa_history.secondary,
                epa_end,
                RECENCY_WINDOW_GAMES,
            )
            record[f"{side}_recent_point_diff"] = _window_mean(
                result_history.primary,
                result_end,
                RECENCY_WINDOW_GAMES,
            )
            if result_end <= 0:
                record[f"{side}_signed_streak"] = float("nan")
            else:
                record[f"{side}_signed_streak"] = signed_result_streak(
                    result_history.secondary[:result_end].astype(int)
                )
        rows.append(record)

    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame.loc[:, ["game_id", *RECENCY_OUTPUT_COLUMNS]]
