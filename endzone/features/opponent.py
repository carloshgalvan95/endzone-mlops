"""Opponent-adjusted EPA and the H6 Elo difference.

The choices below are locked before any model fit:

- Game EPA is fit with ridge regression on offense, defense, and home terms.
  The penalty is fixed. It is not chosen on the 2016-2023 window.
- A rating for a game uses team-games with ``gameday`` strictly before that
  game. Games on the same gameday share the fit and do not enter it.
- Those adjusted game values replace raw EPA inside the same recent,
  season, and cross-season regression windows as ``h2_full``.
- ``h6_adj_elo`` adds pre-game home Elo minus away Elo. The update uses the
  FiveThirtyEight margin-of-victory multiplier. Home advantage is not stored
  in the column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

RIDGE_ALPHA: float = 1.0
MIN_RIDGE_ROWS: int = 4
MIN_RIDGE_TEAMS: int = 2
ADJ_OFFENSIVE_COLUMN: str = "adj_offensive_epa"
ADJ_DEFENSIVE_COLUMN: str = "adj_defensive_epa_allowed"
ELO_MOV_DIFF_COLUMN: str = "elo_mov_diff"
FEATURE_SET_H6_ADJ: str = "h6_adj"
FEATURE_SET_H6_ADJ_ELO: str = "h6_adj_elo"
H6_CHALLENGER_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_H6_ADJ,
    FEATURE_SET_H6_ADJ_ELO,
)

# Names emitted by the h2 EPA block when the source columns are adjusted EPA.
# Recent means cross seasons. Season and regressed means stay inside a season.
H6_ADJ_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_recent_adj_offensive_epa_per_play",
    "home_recent_adj_defensive_epa_per_play",
    "away_recent_adj_offensive_epa_per_play",
    "away_recent_adj_defensive_epa_per_play",
    "home_season_adj_offensive_epa_per_play",
    "away_season_adj_offensive_epa_per_play",
    "home_season_adj_defensive_epa_per_play",
    "away_season_adj_defensive_epa_per_play",
    "home_regressed_recent_adj_offensive_epa_per_play",
    "home_regressed_recent_adj_defensive_epa_per_play",
    "away_regressed_recent_adj_offensive_epa_per_play",
    "away_regressed_recent_adj_defensive_epa_per_play",
    "home_regressed_season_adj_offensive_epa_per_play",
    "home_regressed_season_adj_defensive_epa_per_play",
    "away_regressed_season_adj_offensive_epa_per_play",
    "away_regressed_season_adj_defensive_epa_per_play",
)
H6_OUTPUT_COLUMNS: tuple[str, ...] = H6_ADJ_FEATURE_COLUMNS + (ELO_MOV_DIFF_COLUMN,)

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
    "season",
    "gameday",
    "team_franchise",
    "avg_offensive_epa",
    "avg_defensive_epa_allowed",
)
_EMPTY_HISTORY_COLUMNS: tuple[str, ...] = (
    "season",
    "gameday",
    ADJ_OFFENSIVE_COLUMN,
    ADJ_DEFENSIVE_COLUMN,
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Raise when ``frame`` is missing any of ``columns``."""
    missing: list[str] = [column for column in columns if column not in frame.columns]
    if missing:
        joined: str = ", ".join(missing)
        raise ValueError(f"{label} missing columns: {joined}")


def fit_opponent_ridge(
    offense: np.ndarray,
    defense: np.ndarray,
    home_sign: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, Any] | None:
    """Fit offense, defense, and home coefficients on one prior sample.

    The design is an offense dummy, a defense dummy, and ``home_sign``
    (+1 at home, -1 away). Ridge penalizes the coefficients and not the
    intercept. Teams are sorted so the column order is stable.

    Args:
        offense: Offense franchise for each team-game.
        defense: Opponent franchise for each team-game.
        home_sign: +1 or -1 for each team-game.
        target: Offensive EPA per play for each team-game.
        alpha: L2 penalty. Must be positive. The default is locked.

    Returns:
        Dict with ``offense`` and ``defense`` coefficient maps, ``home``,
        ``intercept``, ``alpha``, ``n``, and ``teams``. ``None`` when the
        sample has fewer than two teams or fewer than four team-games.

    Raises:
        ValueError: The arrays differ in length, a home sign is not +1 or -1,
            a target is not finite, or ``alpha`` is not positive.
    """
    if alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")
    row_count: int = int(len(target))
    if not (len(offense) == len(defense) == len(home_sign) == row_count):
        raise ValueError("ridge inputs must have the same length")
    if row_count == 0:
        return None
    signs: np.ndarray = np.asarray(home_sign, dtype=float)
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ValueError("home_sign must be +1 or -1")
    observed: np.ndarray = np.asarray(target, dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("ridge target must be finite")
    teams: tuple[str, ...] = tuple(
        sorted({str(team) for team in offense} | {str(team) for team in defense})
    )
    if row_count < MIN_RIDGE_ROWS or len(teams) < MIN_RIDGE_TEAMS:
        return None
    index: dict[str, int] = {team: position for position, team in enumerate(teams)}
    team_count: int = len(teams)
    design: np.ndarray = np.zeros((row_count, 2 * team_count + 1), dtype=float)
    row_index: np.ndarray = np.arange(row_count)
    offense_index: np.ndarray = np.asarray([index[str(team)] for team in offense], dtype=int)
    defense_index: np.ndarray = np.asarray([index[str(team)] for team in defense], dtype=int)
    design[row_index, offense_index] = 1.0
    design[row_index, team_count + defense_index] = 1.0
    design[:, -1] = signs
    model: Ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="cholesky")
    model.fit(design, observed)
    coefficients: np.ndarray = np.asarray(model.coef_, dtype=float)
    return {
        "offense": {team: float(coefficients[position]) for position, team in enumerate(teams)},
        "defense": {
            team: float(coefficients[team_count + position]) for position, team in enumerate(teams)
        },
        "home": float(coefficients[-1]),
        "intercept": float(model.intercept_),
        "alpha": float(alpha),
        "n": row_count,
        "teams": teams,
    }


def adjust_game_epa(
    offensive_epa: float,
    defensive_epa_allowed: float,
    opponent: str,
    home_sign: float,
    fit: dict[str, Any],
) -> dict[str, float]:
    """Remove opponent and venue from one team-game.

    Adjusted offense subtracts the opponent defense coefficient and the home
    term. Adjusted EPA allowed subtracts the opponent offense coefficient and
    the home term from the opponent's side. An opponent missing from the fit
    uses coefficient 0, the ridge target. A missing input stays missing.

    Args:
        offensive_epa: Observed offensive EPA per play.
        defensive_epa_allowed: Observed EPA allowed per play.
        opponent: The other franchise in the game.
        home_sign: +1 when this team is home, -1 when it is away.
        fit: Result of ``fit_opponent_ridge``.

    Returns:
        Dict with ``adj_offensive_epa`` and ``adj_defensive_epa_allowed``.

    Raises:
        ValueError: ``home_sign`` is not +1 or -1.
    """
    sign: float = float(home_sign)
    if sign not in (-1.0, 1.0):
        raise ValueError("home_sign must be +1 or -1")
    home_coef: float = float(fit["home"])
    offense_map: dict[str, float] = fit["offense"]
    defense_map: dict[str, float] = fit["defense"]
    opponent_name: str = str(opponent)
    opponent_defense: float = float(defense_map.get(opponent_name, 0.0))
    opponent_offense: float = float(offense_map.get(opponent_name, 0.0))
    offensive_value: float = float(offensive_epa)
    defensive_value: float = float(defensive_epa_allowed)
    adjusted_offense: float = offensive_value - opponent_defense - home_coef * sign
    adjusted_defense: float = defensive_value - opponent_offense - home_coef * (-sign)
    if not np.isfinite(offensive_value):
        adjusted_offense = float("nan")
    if not np.isfinite(defensive_value):
        adjusted_defense = float("nan")
    return {
        ADJ_OFFENSIVE_COLUMN: adjusted_offense,
        ADJ_DEFENSIVE_COLUMN: adjusted_defense,
    }


def opponent_adjusted_team_games(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    alpha: float = RIDGE_ALPHA,
) -> pd.DataFrame:
    """Adjusted offensive EPA and EPA allowed for every team-game.

    Each gameday is adjusted with a ridge fit on team-games whose gameday is
    strictly earlier. The current gameday does not enter that fit.

    Args:
        games: One row per game, with home and away franchises.
        team_games: Team-game EPA. One row per franchise per game.
        alpha: Locked ridge penalty.

    Returns:
        One row per team-game, with ``game_id``, ``season``, ``gameday``,
        ``team_franchise``, and the two adjusted columns. A team-game that
        cannot be joined, or a date with too little history, stays missing.

    Raises:
        ValueError: A required column is missing, ``game_id`` is duplicated
            on ``games``, or a team-game key is duplicated.
    """
    _require_columns(games, _GAME_COLUMNS, "games")
    _require_columns(team_games, _EPA_COLUMNS, "fct_team_game_epa")
    if games["game_id"].duplicated().any():
        raise ValueError("game_id must be unique to adjust opponent EPA")
    team_key: pd.Series = (
        team_games["game_id"].astype(str) + "\t" + team_games["team_franchise"].astype(str)
    )
    if team_key.duplicated().any():
        raise ValueError("game_id and team_franchise must be unique on team-game EPA")

    schedule: pd.DataFrame = games.loc[
        :, ["game_id", "home_team_franchise", "away_team_franchise"]
    ].copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    merged: pd.DataFrame = team_games.loc[:, list(_EPA_COLUMNS)].copy()
    merged["game_id"] = merged["game_id"].astype(str)
    merged["team_franchise"] = merged["team_franchise"].astype(str)
    merged = merged.merge(schedule, on="game_id", how="left")
    merged["gameday"] = pd.to_datetime(merged["gameday"])
    home_team: pd.Series = merged["home_team_franchise"].astype(str)
    away_team: pd.Series = merged["away_team_franchise"].astype(str)
    known_side: pd.Series = (
        merged["home_team_franchise"].notna() & merged["away_team_franchise"].notna()
    )
    is_home: pd.Series = known_side & (merged["team_franchise"] == home_team)
    is_away: pd.Series = known_side & (merged["team_franchise"] == away_team)
    merged["opponent"] = np.where(is_home, away_team, np.where(is_away, home_team, pd.NA))
    merged["home_sign"] = np.where(is_home, 1.0, np.where(is_away, -1.0, np.nan))
    merged["_order"] = np.arange(len(merged))
    offensive_values: np.ndarray = pd.to_numeric(
        merged["avg_offensive_epa"], errors="coerce"
    ).to_numpy(dtype=float)

    usable: pd.Series = (
        merged["gameday"].notna()
        & merged["opponent"].notna()
        & merged["home_sign"].notna()
        & np.isfinite(offensive_values)
    )
    ordered: pd.DataFrame = merged.loc[usable].sort_values(
        ["gameday", "game_id", "team_franchise"],
        kind="mergesort",
    )
    adjusted_offense: np.ndarray = np.full(len(merged), np.nan, dtype=float)
    adjusted_defense: np.ndarray = np.full(len(merged), np.nan, dtype=float)
    if not ordered.empty:
        dates: np.ndarray = ordered["gameday"].to_numpy(dtype="datetime64[ns]")
        unique_dates: np.ndarray = pd.unique(dates)
        offense_names: np.ndarray = ordered["team_franchise"].to_numpy()
        defense_names: np.ndarray = ordered["opponent"].to_numpy()
        signs: np.ndarray = ordered["home_sign"].to_numpy(dtype=float)
        targets: np.ndarray = ordered["avg_offensive_epa"].to_numpy(dtype=float)
        allowed: np.ndarray = ordered["avg_defensive_epa_allowed"].to_numpy(dtype=float)
        positions: np.ndarray = ordered["_order"].to_numpy(dtype=int)
        for game_date in unique_dates:
            cutoff: np.datetime64 = np.datetime64(game_date, "ns")
            prior_end: int = int(np.searchsorted(dates, cutoff, side="left"))
            current_end: int = int(np.searchsorted(dates, cutoff, side="right"))
            fit: dict[str, Any] | None = fit_opponent_ridge(
                offense_names[:prior_end],
                defense_names[:prior_end],
                signs[:prior_end],
                targets[:prior_end],
                alpha=alpha,
            )
            if fit is None:
                continue
            for position in range(prior_end, current_end):
                adjusted: dict[str, float] = adjust_game_epa(
                    float(targets[position]),
                    float(allowed[position]),
                    str(defense_names[position]),
                    float(signs[position]),
                    fit,
                )
                row_position: int = int(positions[position])
                adjusted_offense[row_position] = adjusted[ADJ_OFFENSIVE_COLUMN]
                adjusted_defense[row_position] = adjusted[ADJ_DEFENSIVE_COLUMN]

    return pd.DataFrame(
        {
            "game_id": merged["game_id"].to_numpy(),
            "season": merged["season"].to_numpy(),
            "gameday": merged["gameday"].to_numpy(),
            "team_franchise": merged["team_franchise"].to_numpy(),
            ADJ_OFFENSIVE_COLUMN: adjusted_offense,
            ADJ_DEFENSIVE_COLUMN: adjusted_defense,
        }
    )


def _empty_history() -> pd.DataFrame:
    """A team with no adjusted games yet."""
    return pd.DataFrame(columns=list(_EMPTY_HISTORY_COLUMNS))


def attach_h6_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    lookback_games: int,
    shrink_factor: float,
    blend_k: float,
) -> pd.DataFrame:
    """Build the H6 columns for every game.

    Adjusted EPA uses ``team_games``. The Elo difference uses the same
    schedule as the baseline Elo probability, with the margin-of-victory
    multiplier turned on. A same-gameday game is not history.

    Args:
        games: One row per game.
        team_games: Team-game EPA.
        lookback_games: Cross-season window for recent adjusted EPA.
        shrink_factor: Fraction of the prior-season gap removed.
        blend_k: Current-season blend strength, ``n / (n + blend_k)``.

    Returns:
        One row per input game, with ``game_id`` and ``H6_OUTPUT_COLUMNS``.

    Raises:
        ValueError: A required column is missing, ``game_id`` is duplicated,
            or the EPA block does not emit the locked column names.
    """
    from endzone.features.pregame import (
        _epa_feature_block,
        _prior_season_league_means,
        elo_home_probabilities,
    )

    _require_columns(games, _GAME_COLUMNS, "games")
    if games["game_id"].duplicated().any():
        raise ValueError("game_id must be unique to attach H6 features")
    if lookback_games < 1:
        raise ValueError("lookback_games must be positive")

    game_frame: pd.DataFrame = games.reset_index(drop=True).copy()
    game_frame["gameday"] = pd.to_datetime(game_frame["gameday"])
    adjusted: pd.DataFrame = opponent_adjusted_team_games(game_frame, team_games)
    adjusted = (
        adjusted.loc[adjusted["gameday"].notna()]
        .sort_values(
            ["gameday", "game_id", "team_franchise"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    history_dates: np.ndarray = pd.to_datetime(adjusted["gameday"]).to_numpy(dtype="datetime64[ns]")
    by_team: dict[str, pd.DataFrame] = {}
    team_dates: dict[str, np.ndarray] = {}
    if not adjusted.empty:
        for team, part in adjusted.groupby("team_franchise", sort=False):
            team_name: str = str(team)
            team_frame: pd.DataFrame = part.reset_index(drop=True)
            by_team[team_name] = team_frame
            team_dates[team_name] = pd.to_datetime(team_frame["gameday"]).to_numpy(
                dtype="datetime64[ns]"
            )

    fitted: dict[str, Any] = elo_home_probabilities(game_frame, margin_of_victory=True)
    home_rating: np.ndarray = np.asarray(fitted["home_ratings"], dtype=float)
    away_rating: np.ndarray = np.asarray(fitted["away_ratings"], dtype=float)
    if len(home_rating) != len(game_frame) or len(away_rating) != len(game_frame):
        raise ValueError("Elo ratings do not align with the games frame")

    rows: list[dict[str, object]] = []
    for position, game in enumerate(game_frame.itertuples(index=False)):
        record: dict[str, object] = {
            "game_id": game.game_id,
            ELO_MOV_DIFF_COLUMN: float(home_rating[position] - away_rating[position]),
        }
        game_date: object = game.gameday
        if pd.isna(game_date):
            prior_end = 0
        else:
            cutoff: np.datetime64 = np.datetime64(pd.Timestamp(game_date), "ns")
            prior_end = int(np.searchsorted(history_dates, cutoff, side="left"))
        prior_all: pd.DataFrame = adjusted.iloc[:prior_end]
        season: int = int(game.season)
        league_means: dict[str, float] = _prior_season_league_means(
            prior_all,
            season - 1,
            offensive_column=ADJ_OFFENSIVE_COLUMN,
            defensive_column=ADJ_DEFENSIVE_COLUMN,
        )
        for side, team_column in (
            ("home", "home_team_franchise"),
            ("away", "away_team_franchise"),
        ):
            team: str = str(getattr(game, team_column))
            team_history: pd.DataFrame = by_team.get(team, _empty_history())
            if team in team_dates and not pd.isna(game_date):
                team_cutoff: np.datetime64 = np.datetime64(pd.Timestamp(game_date), "ns")
                team_end: int = int(np.searchsorted(team_dates[team], team_cutoff, side="left"))
                team_prior: pd.DataFrame = team_history.iloc[:team_end]
            else:
                team_prior = _empty_history()
            block: dict[str, float] = _epa_feature_block(
                side,
                team_prior,
                season,
                lookback_games,
                league_means,
                shrink_factor,
                blend_k,
                ADJ_OFFENSIVE_COLUMN,
                ADJ_DEFENSIVE_COLUMN,
                infix="adj_",
            )
            record.update(block)
        missing_names: list[str] = [
            column for column in H6_ADJ_FEATURE_COLUMNS if column not in record
        ]
        if missing_names:
            joined: str = ", ".join(missing_names)
            raise ValueError(f"adjusted EPA block is missing columns: {joined}")
        rows.append(record)

    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame.loc[:, ["game_id", *H6_OUTPUT_COLUMNS]]
