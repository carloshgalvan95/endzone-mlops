"""
Pre-game feature engineering for NFL predictions.

CRITICAL: All features must be leakage-free. Features for game G use only data
from games strictly before G.

Features computed:
- Rolling/expanding team performance metrics (EPA per play, win rate)
- Rest days
- Prior-game counts (legacy feature set only)
- Home-minus-away differences of the level features (diff sets only)
- Season-to-date defensive EPA allowed (h2 sets only)
- Cross-season regression of season and recent EPA (h2_full only)
- Recent and regressed-season pass and rush EPA per play (h3 sets only)
- Clean EPA, same windows as h2_full, plus a recent clean success rate (h4 sets)
- A shrunk pregame starting-QB rating carried across teams and seasons (h5)
- A simple Elo rating and a no-vig market probability
"""

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

# Columns that identify a game. They are not model inputs.
MODEL_ID_COLUMNS: tuple[str, ...] = ("game_id", "gameday", "season")

# Prior-game counts from the start of the frame. They track calendar time.
GAMES_PLAYED_COLUMNS: tuple[str, ...] = ("home_games_played", "away_games_played")

FEATURE_SET_CURRENT: str = "current"
FEATURE_SET_WITH_GAMES_PLAYED: str = "with_games_played"
FEATURE_SET_DIFF: str = "diff"
FEATURE_SET_DIFF_ONLY: str = "diff_only"
FEATURE_SET_H2_DEF: str = "h2_def"
FEATURE_SET_H2_FULL: str = "h2_full"
FEATURE_SET_H3_SPLIT: str = "h3_split"
FEATURE_SET_H3_REPLACE: str = "h3_replace"
FEATURE_SET_H4_CLEAN: str = "h4_clean"
FEATURE_SET_H4_CLEAN_SR: str = "h4_clean_sr"
FEATURE_SET_H5_QB: str = "h5_qb"

# Level columns, in the pre-phase-1 order. Differentials are appended after
# these so ``current`` and ``with_games_played`` stay the same matrices.
# This is the pre-phase-1 set and stays available for a reproducible rerun.
PREGAME_COMPUTED_COLUMNS: tuple[str, ...] = (
    "home_recent_offensive_epa_per_play",
    "home_recent_defensive_epa_per_play",
    "away_recent_offensive_epa_per_play",
    "away_recent_defensive_epa_per_play",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_rest_days",
    "away_rest_days",
    "home_season_offensive_epa_per_play",
    "away_season_offensive_epa_per_play",
    "home_games_played",
    "away_games_played",
)

# Default model inputs. games_played is computed, then left out of this set.
PREGAME_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in PREGAME_COMPUTED_COLUMNS if column not in GAMES_PLAYED_COLUMNS
)

# Old training matrix, including the games_played clock.
PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED: tuple[str, ...] = PREGAME_COMPUTED_COLUMNS


class DifferentialFeature(NamedTuple):
    """One home-minus-away column and the two level columns it subtracts."""

    home_column: str
    away_column: str
    diff_column: str


# Home minus away. Defensive EPA is EPA allowed and is not sign-flipped.
DIFFERENTIAL_FEATURES: tuple[DifferentialFeature, ...] = (
    DifferentialFeature(
        "home_recent_offensive_epa_per_play",
        "away_recent_offensive_epa_per_play",
        "diff_recent_offensive_epa_per_play",
    ),
    DifferentialFeature(
        "home_recent_defensive_epa_per_play",
        "away_recent_defensive_epa_per_play",
        "diff_recent_defensive_epa_per_play",
    ),
    DifferentialFeature(
        "home_recent_win_rate",
        "away_recent_win_rate",
        "diff_recent_win_rate",
    ),
    DifferentialFeature(
        "home_season_offensive_epa_per_play",
        "away_season_offensive_epa_per_play",
        "diff_season_offensive_epa_per_play",
    ),
    DifferentialFeature(
        "home_rest_days",
        "away_rest_days",
        "diff_rest_days",
    ),
)

DIFF_FEATURE_COLUMNS: tuple[str, ...] = tuple(item.diff_column for item in DIFFERENTIAL_FEATURES)

# ``current`` plus the five differences. Levels stay, so this is not a replacement.
PREGAME_FEATURE_COLUMNS_DIFF: tuple[str, ...] = PREGAME_FEATURE_COLUMNS + DIFF_FEATURE_COLUMNS

# Differences only. Parsimony check against ``diff``.
PREGAME_FEATURE_COLUMNS_DIFF_ONLY: tuple[str, ...] = DIFF_FEATURE_COLUMNS

# Same-season prior games only. Week 1 stays missing. Not part of ``current``.
SEASON_DEFENSIVE_EPA_COLUMNS: tuple[str, ...] = (
    "home_season_defensive_epa_per_play",
    "away_season_defensive_epa_per_play",
)

# Shrunk prior-season finals, then blended toward the current season.
# Recent windows for these columns stay inside one season. The level
# recent columns above still cross seasons.
REGRESSED_EPA_COLUMNS: tuple[str, ...] = (
    "home_regressed_recent_offensive_epa_per_play",
    "home_regressed_recent_defensive_epa_per_play",
    "away_regressed_recent_offensive_epa_per_play",
    "away_regressed_recent_defensive_epa_per_play",
    "home_regressed_season_offensive_epa_per_play",
    "away_regressed_season_offensive_epa_per_play",
    "home_regressed_season_defensive_epa_per_play",
    "away_regressed_season_defensive_epa_per_play",
)

# ``current`` plus the two season defensive EPA columns.
PREGAME_FEATURE_COLUMNS_H2_DEF: tuple[str, ...] = (
    PREGAME_FEATURE_COLUMNS + SEASON_DEFENSIVE_EPA_COLUMNS
)

# ``h2_def`` plus the regressed season and recent EPA columns.
PREGAME_FEATURE_COLUMNS_H2_FULL: tuple[str, ...] = (
    PREGAME_FEATURE_COLUMNS_H2_DEF + REGRESSED_EPA_COLUMNS
)

# Pass and rush EPA per play. Recent uses the cross-season lookback.
# Regressed-season uses the same shrink and blend as the H2 season columns.
# Defensive rates are the opponent's offensive pass or rush rate.
PASS_RUSH_RECENT_COLUMNS: tuple[str, ...] = (
    "home_recent_offensive_pass_epa_per_play",
    "home_recent_offensive_rush_epa_per_play",
    "home_recent_defensive_pass_epa_per_play",
    "home_recent_defensive_rush_epa_per_play",
    "away_recent_offensive_pass_epa_per_play",
    "away_recent_offensive_rush_epa_per_play",
    "away_recent_defensive_pass_epa_per_play",
    "away_recent_defensive_rush_epa_per_play",
)
PASS_RUSH_REGRESSED_SEASON_COLUMNS: tuple[str, ...] = (
    "home_regressed_season_offensive_pass_epa_per_play",
    "home_regressed_season_offensive_rush_epa_per_play",
    "home_regressed_season_defensive_pass_epa_per_play",
    "home_regressed_season_defensive_rush_epa_per_play",
    "away_regressed_season_offensive_pass_epa_per_play",
    "away_regressed_season_offensive_rush_epa_per_play",
    "away_regressed_season_defensive_pass_epa_per_play",
    "away_regressed_season_defensive_rush_epa_per_play",
)
PASS_RUSH_EPA_COLUMNS: tuple[str, ...] = (
    PASS_RUSH_RECENT_COLUMNS + PASS_RUSH_REGRESSED_SEASON_COLUMNS
)

# Combined columns the pass/rush splits stand in for. ``h3_replace`` drops
# these and keeps the rest of ``h2_full`` (win rate, rest, season EPA, and
# the regressed recent block).
H3_REPLACED_COMBINED_COLUMNS: tuple[str, ...] = (
    "home_recent_offensive_epa_per_play",
    "home_recent_defensive_epa_per_play",
    "away_recent_offensive_epa_per_play",
    "away_recent_defensive_epa_per_play",
    "home_regressed_season_offensive_epa_per_play",
    "away_regressed_season_offensive_epa_per_play",
    "home_regressed_season_defensive_epa_per_play",
    "away_regressed_season_defensive_epa_per_play",
)

# ``h2_full`` plus the sixteen pass/rush columns.
PREGAME_FEATURE_COLUMNS_H3_SPLIT: tuple[str, ...] = (
    PREGAME_FEATURE_COLUMNS_H2_FULL + PASS_RUSH_EPA_COLUMNS
)

# ``h2_full`` with the combined recent and regressed-season EPA replaced.
PREGAME_FEATURE_COLUMNS_H3_REPLACE: tuple[str, ...] = (
    tuple(
        column
        for column in PREGAME_FEATURE_COLUMNS_H2_FULL
        if column not in H3_REPLACED_COMBINED_COLUMNS
    )
    + PASS_RUSH_EPA_COLUMNS
)

# Win rate and rest stay on h4. Every other h2_full column is an EPA input.
H2_CONTEXT_COLUMNS: tuple[str, ...] = (
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_rest_days",
    "away_rest_days",
)

# Mart columns for the clean filter. Missing as a pair leaves the h4
# features missing. One column without the other is refused.
CLEAN_OFFENSIVE_EPA_COLUMN: str = "avg_clean_offensive_epa"
CLEAN_DEFENSIVE_EPA_COLUMN: str = "avg_clean_defensive_epa_allowed"
CLEAN_OFFENSIVE_SUCCESS_COLUMN: str = "clean_success_rate"
CLEAN_DEFENSIVE_SUCCESS_COLUMN: str = "clean_defensive_success_rate"
CLEAN_EPA_SOURCE_COLUMNS: tuple[str, ...] = (
    CLEAN_OFFENSIVE_EPA_COLUMN,
    CLEAN_DEFENSIVE_EPA_COLUMN,
)
CLEAN_SUCCESS_SOURCE_COLUMNS: tuple[str, ...] = (
    CLEAN_OFFENSIVE_SUCCESS_COLUMN,
    CLEAN_DEFENSIVE_SUCCESS_COLUMN,
)

# Recent clean success rate. Same cross-season lookback as recent EPA.
CLEAN_SUCCESS_RATE_COLUMNS: tuple[str, ...] = (
    "home_recent_clean_offensive_success_rate",
    "home_recent_clean_defensive_success_rate",
    "away_recent_clean_offensive_success_rate",
    "away_recent_clean_defensive_success_rate",
)


def _clean_epa_feature_name(column: str) -> str:
    """Insert ``clean_`` before offensive or defensive in an EPA feature name."""
    for role in ("offensive", "defensive"):
        token: str = f"_{role}_"
        if token in column:
            return column.replace(token, f"_clean_{role}_", 1)
    raise ValueError(f"{column} is not an offensive or defensive EPA column")


def _h4_columns_from_h2(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Keep context columns and point every EPA column at the clean source."""
    renamed: list[str] = []
    for column in columns:
        if column in H2_CONTEXT_COLUMNS:
            renamed.append(column)
        else:
            renamed.append(_clean_epa_feature_name(column))
    return tuple(renamed)


# ``h2_full`` with every EPA input computed from the clean mart columns.
# Windows, shrinkage, and the blend weight are the same as ``h2_full``.
PREGAME_FEATURE_COLUMNS_H4_CLEAN: tuple[str, ...] = _h4_columns_from_h2(
    PREGAME_FEATURE_COLUMNS_H2_FULL
)

# ``h4_clean`` plus the recent clean success rate, offense and defense.
PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR: tuple[str, ...] = (
    PREGAME_FEATURE_COLUMNS_H4_CLEAN + CLEAN_SUCCESS_RATE_COLUMNS
)

# Clean EPA columns the builder emits, in output order. The h4 sets select
# a subset and put win rate and rest back in the ``h2_full`` positions.
CLEAN_EPA_FEATURE_COLUMNS: tuple[str, ...] = _h4_columns_from_h2(
    tuple(column for column in PREGAME_FEATURE_COLUMNS_H2_FULL if column not in H2_CONTEXT_COLUMNS)
)

# Pregame starting QB. Prior dropbacks shrink EPA per dropback toward
# replacement. 100 prior dropbacks is half weight. Replacement is 0.
# CPOE is not mixed in. It is a different unit, and a blend would be a
# second hypothesis.
QB_PRIOR_DROPBACKS: float = 100.0
REPLACEMENT_EPA_PER_DROPBACK: float = 0.0
QB_RATING_COLUMNS: tuple[str, ...] = (
    "home_qb_rating",
    "away_qb_rating",
    "home_qb_rating_minus_team_pass_epa",
    "away_qb_rating_minus_team_pass_epa",
)
QB_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "home_starter_player_id",
    "away_starter_player_id",
    "home_starter_source",
    "away_starter_source",
    "home_qb_changed",
    "away_qb_changed",
)
QB_OUTPUT_COLUMNS: tuple[str, ...] = QB_RATING_COLUMNS + QB_DIAGNOSTIC_COLUMNS
PREGAME_FEATURE_COLUMNS_H5_QB: tuple[str, ...] = PREGAME_FEATURE_COLUMNS_H2_FULL + QB_RATING_COLUMNS
EXPECTED_STARTER_COLUMNS: tuple[str, ...] = (
    "game_id",
    "home_starter_player_id",
    "away_starter_player_id",
)
QB_DROPBACK_COLUMNS: tuple[str, ...] = (
    "game_id",
    "gameday",
    "team_franchise",
    "passer_player_id",
    "dropbacks",
    "dropback_epa",
)

FEATURE_SET_NAMES: tuple[str, ...] = (
    FEATURE_SET_CURRENT,
    FEATURE_SET_WITH_GAMES_PLAYED,
    FEATURE_SET_DIFF,
    FEATURE_SET_DIFF_ONLY,
    FEATURE_SET_H2_DEF,
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H3_SPLIT,
    FEATURE_SET_H3_REPLACE,
    FEATURE_SET_H4_CLEAN,
    FEATURE_SET_H4_CLEAN_SR,
    FEATURE_SET_H5_QB,
)

FEATURE_SET_COLUMNS: dict[str, tuple[str, ...]] = {
    FEATURE_SET_CURRENT: PREGAME_FEATURE_COLUMNS,
    FEATURE_SET_WITH_GAMES_PLAYED: PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED,
    FEATURE_SET_DIFF: PREGAME_FEATURE_COLUMNS_DIFF,
    FEATURE_SET_DIFF_ONLY: PREGAME_FEATURE_COLUMNS_DIFF_ONLY,
    FEATURE_SET_H2_DEF: PREGAME_FEATURE_COLUMNS_H2_DEF,
    FEATURE_SET_H2_FULL: PREGAME_FEATURE_COLUMNS_H2_FULL,
    FEATURE_SET_H3_SPLIT: PREGAME_FEATURE_COLUMNS_H3_SPLIT,
    FEATURE_SET_H3_REPLACE: PREGAME_FEATURE_COLUMNS_H3_REPLACE,
    FEATURE_SET_H4_CLEAN: PREGAME_FEATURE_COLUMNS_H4_CLEAN,
    FEATURE_SET_H4_CLEAN_SR: PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR,
    FEATURE_SET_H5_QB: PREGAME_FEATURE_COLUMNS_H5_QB,
}

# FiveThirtyEight-style Elo. Home advantage is added only when a game is
# scored, not stored on the team rating. One third of the gap to the mean
# is removed the first time a later season is seen. No margin multiplier.
ELO_K: float = 20.0
ELO_HOME_ADVANTAGE: float = 65.0
ELO_MEAN_RATING: float = 1505.0
ELO_SEASON_REVERSION: float = 1.0 / 3.0
ELO_SCALE: float = 400.0

# One third of the gap to the prior-season league mean is removed.
# The same fraction Elo uses between seasons.
SEASON_SHRINK_FACTOR: float = ELO_SEASON_REVERSION
# Current-season weight is n / (n + k). Half weight at k games.
SEASON_BLEND_K: float = 4.0

MONEYLINE_HOME_COLUMN: str = "home_moneyline"
MONEYLINE_AWAY_COLUMN: str = "away_moneyline"


class TeamEpaSnapshot(NamedTuple):
    """Prior EPA for one team, split into the current season and the season before.

    Recent windows are the last ``lookback`` rows inside that season only.
    Counts are games in the season, not the length of the recent window.
    """

    current_season_n: int
    current_season_offensive: float
    current_season_defensive: float
    current_recent_offensive: float
    current_recent_defensive: float
    prior_season_offensive: float
    prior_season_defensive: float
    prior_recent_offensive: float
    prior_recent_defensive: float


def _mean_or_nan(values: pd.Series) -> float:
    """Mean of a series, or NaN when the series is empty."""
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def _column_mean(rows: pd.DataFrame, column: str) -> float:
    """Mean of one column, or NaN when the column or the rows are missing."""
    if len(rows) == 0 or column not in rows.columns:
        return float("nan")
    return _mean_or_nan(rows[column])


def _refuse_partial_source(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Allow a source pair to be all present or all absent.

    Args:
        frame: Team-game table.
        columns: Columns that have to arrive together.
        label: Short name used in the error.

    Raises:
        ValueError: Some of ``columns`` are present and some are not.
    """
    present: list[str] = [column for column in columns if column in frame.columns]
    if len(present) in (0, len(columns)):
        return
    missing: list[str] = [column for column in columns if column not in frame.columns]
    missing_list: str = ", ".join(missing)
    raise ValueError(f"fct_team_game_epa missing {label} columns: {missing_list}")


def _validate_regression_settings(shrink_factor: float, blend_k: float) -> None:
    """Reject a shrink factor or blend constant that the formula cannot use."""
    if not 0.0 <= float(shrink_factor) <= 1.0:
        raise ValueError("shrink_factor must be in [0, 1]")
    if float(blend_k) <= 0.0:
        raise ValueError("blend_k must be positive")


def shrink_toward_mean(value: float, league_mean: float, shrink_factor: float) -> float:
    """Pull ``value`` part of the way to ``league_mean``.

    ``shrink_factor`` is the fraction of the gap that is removed. ``1/3``
    matches Elo season reversion: two thirds of the gap stays.

    Args:
        value: Prior-season final EPA.
        league_mean: League mean the value is pulled toward.
        shrink_factor: Fraction of the gap removed. Must be in ``[0, 1]``.

    Returns:
        The shrunk value.

    Raises:
        ValueError: ``shrink_factor`` is outside ``[0, 1]``.
    """
    _validate_regression_settings(shrink_factor, blend_k=1.0)
    return float(league_mean) + (1.0 - float(shrink_factor)) * (float(value) - float(league_mean))


def current_season_blend_weight(n_games: int, blend_k: float) -> float:
    """Weight on the current-season mean, ``n / (n + k)``.

    Zero games returns 0, so a caller can use the shrunk prior alone.

    Args:
        n_games: Completed current-season games before this kickoff.
        blend_k: Positive prior strength, in games.

    Returns:
        Weight in ``[0, 1)``.

    Raises:
        ValueError: ``n_games`` is negative or ``blend_k`` is not positive.
    """
    if n_games < 0:
        raise ValueError("n_games must be >= 0")
    _validate_regression_settings(shrink_factor=0.0, blend_k=blend_k)
    return float(n_games) / (float(n_games) + float(blend_k))


def regress_epa_feature(
    current_value: float,
    n_games: int,
    prior_value: float,
    league_mean: float,
    shrink_factor: float = SEASON_SHRINK_FACTOR,
    blend_k: float = SEASON_BLEND_K,
) -> dict[str, float]:
    """Blend a current-season EPA mean with a shrunk prior-season final value.

    At zero current-season games the result is the prior value moved
    ``shrink_factor`` of the way to ``league_mean``. As games accumulate,
    the weight on the current mean is ``n / (n + blend_k)``. A missing
    prior, or a missing league mean, leaves the current mean in place
    after the first game and leaves the result missing before that.

    Args:
        current_value: Mean EPA from current-season games before this kickoff.
        n_games: How many of those games exist. This is the blend count,
            including games that fall outside a recent window.
        prior_value: Final value from the previous season only.
        league_mean: League mean from that same previous season.
        shrink_factor: Fraction of the prior gap removed. Default matches Elo.
        blend_k: Prior strength in games. Default is ``SEASON_BLEND_K``.

    Returns:
        Dict with ``value``, ``shrunk_prior``, and ``weight``. Missing
        inputs stay NaN.

    Raises:
        ValueError: The shrink factor, blend constant, or game count is invalid.
    """
    _validate_regression_settings(shrink_factor, blend_k)
    weight: float = current_season_blend_weight(n_games, blend_k)
    prior_ready: bool = bool(np.isfinite(prior_value)) and bool(np.isfinite(league_mean))
    shrunk_prior: float = (
        shrink_toward_mean(float(prior_value), float(league_mean), shrink_factor)
        if prior_ready
        else float("nan")
    )
    current_ready: bool = bool(np.isfinite(current_value))
    if n_games == 0:
        value: float = shrunk_prior
    elif not prior_ready:
        value = float(current_value) if current_ready else float("nan")
    elif not current_ready:
        value = float("nan")
    else:
        value = weight * float(current_value) + (1.0 - weight) * shrunk_prior
    return {"value": value, "shrunk_prior": shrunk_prior, "weight": weight}


def _team_epa_snapshot(
    team_prior: pd.DataFrame,
    season: int,
    lookback_games: int,
    offensive_column: str = "avg_offensive_epa",
    defensive_column: str = "avg_defensive_epa_allowed",
) -> TeamEpaSnapshot:
    """Season and within-season recent EPA from rows already limited to prior games.

    ``team_prior`` must contain only games with ``gameday`` strictly before
    the game being featurized, and it must be in gameday order. The prior
    season is ``season - 1`` only. An earlier season is not carried forward.
    A missing source column yields NaN means. The game count is still the
    number of prior team-games in the season.
    """
    current_rows: pd.DataFrame = team_prior.loc[team_prior["season"] == season]
    prior_rows: pd.DataFrame = team_prior.loc[team_prior["season"] == season - 1]
    current_recent: pd.DataFrame = current_rows.tail(lookback_games)
    prior_recent: pd.DataFrame = prior_rows.tail(lookback_games)
    return TeamEpaSnapshot(
        current_season_n=int(len(current_rows)),
        current_season_offensive=_column_mean(current_rows, offensive_column),
        current_season_defensive=_column_mean(current_rows, defensive_column),
        current_recent_offensive=_column_mean(current_recent, offensive_column),
        current_recent_defensive=_column_mean(current_recent, defensive_column),
        prior_season_offensive=_column_mean(prior_rows, offensive_column),
        prior_season_defensive=_column_mean(prior_rows, defensive_column),
        prior_recent_offensive=_column_mean(prior_recent, offensive_column),
        prior_recent_defensive=_column_mean(prior_recent, defensive_column),
    )


def _prior_season_league_means(
    prior_games: pd.DataFrame,
    prior_season: int,
    offensive_column: str = "avg_offensive_epa",
    defensive_column: str = "avg_defensive_epa_allowed",
) -> dict[str, float]:
    """League means for one season, from team-games already limited to earlier dates.

    Args:
        prior_games: Every franchise, ``gameday`` strictly before the game.
        prior_season: Season whose completed games form the shrink target.
        offensive_column: Team-game offensive EPA column to average.
        defensive_column: Team-game EPA-allowed column to average.

    Returns:
        Dict with ``offensive`` and ``defensive`` means. Either is NaN when
        that season has no earlier team-games or the column is absent.
    """
    season_rows: pd.DataFrame = prior_games.loc[prior_games["season"] == prior_season]
    return {
        "offensive": _column_mean(season_rows, offensive_column),
        "defensive": _column_mean(season_rows, defensive_column),
    }


def _regressed_epa_for_side(
    side: str,
    snapshot: TeamEpaSnapshot,
    league_means: dict[str, float],
    shrink_factor: float,
    blend_k: float,
    infix: str = "",
) -> dict[str, float]:
    """Regressed recent and season EPA columns for ``home`` or ``away``.

    ``infix`` is ``""`` for the original columns and ``"clean_"`` for the
    clean-EPA columns. Shrinkage and the blend weight do not change.
    """
    specs: tuple[tuple[str, float, float, float], ...] = (
        (
            f"regressed_recent_{infix}offensive_epa_per_play",
            snapshot.current_recent_offensive,
            snapshot.prior_recent_offensive,
            league_means["offensive"],
        ),
        (
            f"regressed_recent_{infix}defensive_epa_per_play",
            snapshot.current_recent_defensive,
            snapshot.prior_recent_defensive,
            league_means["defensive"],
        ),
        (
            f"regressed_season_{infix}offensive_epa_per_play",
            snapshot.current_season_offensive,
            snapshot.prior_season_offensive,
            league_means["offensive"],
        ),
        (
            f"regressed_season_{infix}defensive_epa_per_play",
            snapshot.current_season_defensive,
            snapshot.prior_season_defensive,
            league_means["defensive"],
        ),
    )
    columns: dict[str, float] = {}
    for suffix, current_value, prior_value, league_mean in specs:
        regressed: dict[str, float] = regress_epa_feature(
            current_value,
            snapshot.current_season_n,
            prior_value,
            league_mean,
            shrink_factor=shrink_factor,
            blend_k=blend_k,
        )
        columns[f"{side}_{suffix}"] = regressed["value"]
    return columns


def _epa_feature_block(
    side: str,
    prior: pd.DataFrame,
    season: int,
    lookback_games: int,
    league_means: dict[str, float],
    shrink_factor: float,
    blend_k: float,
    offensive_column: str,
    defensive_column: str,
    infix: str = "",
) -> dict[str, float]:
    """Recent, season, and regressed EPA for one side from one column pair.

    Recent means use the cross-season lookback. Season and regressed-recent
    means stay inside one season. ``infix`` is ``""`` or ``"clean_"``.
    """
    rolling: pd.DataFrame = prior.tail(lookback_games)
    snapshot: TeamEpaSnapshot = _team_epa_snapshot(
        prior,
        season,
        lookback_games,
        offensive_column=offensive_column,
        defensive_column=defensive_column,
    )
    regressed: dict[str, float] = _regressed_epa_for_side(
        side,
        snapshot,
        league_means,
        shrink_factor,
        blend_k,
        infix=infix,
    )
    return {
        f"{side}_recent_{infix}offensive_epa_per_play": _column_mean(rolling, offensive_column),
        f"{side}_recent_{infix}defensive_epa_per_play": _column_mean(rolling, defensive_column),
        f"{side}_season_{infix}offensive_epa_per_play": snapshot.current_season_offensive,
        f"{side}_season_{infix}defensive_epa_per_play": snapshot.current_season_defensive,
        **regressed,
    }


def _recent_clean_success(
    side: str,
    prior: pd.DataFrame,
    lookback_games: int,
) -> dict[str, float]:
    """Cross-season lookback means of clean success rate for one side.

    A missing source column stays missing. The current game is not in
    ``prior``.
    """
    window: pd.DataFrame = prior.tail(lookback_games)
    return {
        f"{side}_recent_clean_offensive_success_rate": _column_mean(
            window, CLEAN_OFFENSIVE_SUCCESS_COLUMN
        ),
        f"{side}_recent_clean_defensive_success_rate": _column_mean(
            window, CLEAN_DEFENSIVE_SUCCESS_COLUMN
        ),
    }


# Mart columns. pass_epa and rush_epa are offensive sums. Play counts are
# the denominators. Defense is the opponent row in the same game. The mart
# does not store a defensive pass or rush column.
PASS_RUSH_SOURCE_COLUMNS: tuple[str, ...] = (
    "pass_epa",
    "rush_epa",
    "pass_plays",
    "rush_plays",
    "opponent_franchise",
)

# Rate column on the team-game frame, in the same order as the feature kinds.
_PASS_RUSH_RATE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("offensive", "pass", "avg_offensive_pass_epa"),
    ("offensive", "rush", "avg_offensive_rush_epa"),
    ("defensive", "pass", "avg_defensive_pass_epa_allowed"),
    ("defensive", "rush", "avg_defensive_rush_epa_allowed"),
)


def _per_play_epa(total: pd.Series, plays: pd.Series) -> pd.Series:
    """EPA per play. Zero or missing play counts stay missing, not zero."""
    totals: pd.Series = pd.to_numeric(total, errors="coerce")
    counts: pd.Series = pd.to_numeric(plays, errors="coerce")
    rates: pd.Series = totals / counts
    return rates.where(counts > 0)


def _attach_pass_rush_rates(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add offensive and defensive pass and rush EPA per play.

    A frame with none of the source columns gets missing rates, so older
    fixtures still build. A frame with some of the columns and not others
    is refused. Defensive rates are the opponent's offensive rates on the
    same ``game_id``.

    Args:
        team_games: Team-game EPA, one row per team per game.

    Returns:
        A copy with the four rate columns.

    Raises:
        ValueError: Only part of the pass/rush source is present, or the
            opponent join changes the row count.
    """
    frame: pd.DataFrame = team_games.copy()
    present: list[str] = [column for column in PASS_RUSH_SOURCE_COLUMNS if column in frame.columns]
    if len(present) == 0:
        for _role, _kind, column in _PASS_RUSH_RATE_COLUMNS:
            frame[column] = np.nan
        return frame
    if len(present) != len(PASS_RUSH_SOURCE_COLUMNS):
        missing: list[str] = [
            column for column in PASS_RUSH_SOURCE_COLUMNS if column not in frame.columns
        ]
        missing_list: str = ", ".join(missing)
        raise ValueError(f"fct_team_game_epa missing pass/rush columns: {missing_list}")

    frame["avg_offensive_pass_epa"] = _per_play_epa(frame["pass_epa"], frame["pass_plays"])
    frame["avg_offensive_rush_epa"] = _per_play_epa(frame["rush_epa"], frame["rush_plays"])
    opponent_rates: pd.DataFrame = frame.loc[
        :,
        ["game_id", "team_franchise", "avg_offensive_pass_epa", "avg_offensive_rush_epa"],
    ].rename(
        columns={
            "team_franchise": "opponent_franchise",
            "avg_offensive_pass_epa": "avg_defensive_pass_epa_allowed",
            "avg_offensive_rush_epa": "avg_defensive_rush_epa_allowed",
        }
    )
    merged: pd.DataFrame = frame.merge(
        opponent_rates,
        on=["game_id", "opponent_franchise"],
        how="left",
    )
    if len(merged) != len(frame):
        raise ValueError("pass/rush opponent join changed the team-game row count")
    return merged


def _pass_rush_means(rows: pd.DataFrame) -> dict[str, float]:
    """Mean of each pass/rush rate. An empty frame or a missing rate is NaN."""
    means: dict[str, float] = {}
    for role, kind, column in _PASS_RUSH_RATE_COLUMNS:
        values: pd.Series = rows[column] if column in rows.columns else pd.Series(dtype=float)
        means[f"{role}_{kind}"] = _mean_or_nan(values)
    return means


def _recent_pass_rush(prior: pd.DataFrame, side: str, lookback_games: int) -> dict[str, float]:
    """Cross-season lookback means of pass and rush EPA per play."""
    window: pd.DataFrame = prior.tail(lookback_games)
    means: dict[str, float] = _pass_rush_means(window)
    return {
        f"{side}_recent_{role}_{kind}_epa_per_play": means[f"{role}_{kind}"]
        for role, kind, _column in _PASS_RUSH_RATE_COLUMNS
    }


def _prior_season_pass_rush_means(prior_games: pd.DataFrame, prior_season: int) -> dict[str, float]:
    """League pass/rush means for one season, from team-games already before this date."""
    season_rows: pd.DataFrame = prior_games.loc[prior_games["season"] == prior_season]
    return _pass_rush_means(season_rows)


def _regressed_season_pass_rush(
    prior: pd.DataFrame,
    side: str,
    season: int,
    league_means: dict[str, float],
    shrink_factor: float,
    blend_k: float,
) -> dict[str, float]:
    """Regressed-season pass and rush EPA for ``home`` or ``away``.

    The current value is the mean of prior games in ``season``. The prior
    value is the mean of season ``season - 1`` only. Shrinkage and the
    blend weight match the combined regressed-season columns.
    """
    current_rows: pd.DataFrame = prior.loc[prior["season"] == season]
    prior_rows: pd.DataFrame = prior.loc[prior["season"] == season - 1]
    current_means: dict[str, float] = _pass_rush_means(current_rows)
    prior_means: dict[str, float] = _pass_rush_means(prior_rows)
    columns: dict[str, float] = {}
    for role, kind, _rate_column in _PASS_RUSH_RATE_COLUMNS:
        key: str = f"{role}_{kind}"
        regressed: dict[str, float] = regress_epa_feature(
            current_means[key],
            int(len(current_rows)),
            prior_means[key],
            league_means[key],
            shrink_factor=shrink_factor,
            blend_k=blend_k,
        )
        columns[f"{side}_regressed_season_{role}_{kind}_epa_per_play"] = regressed["value"]
    return columns


def _require_frame_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Raise when ``frame`` is missing any of ``columns``."""
    missing: list[str] = [column for column in columns if column not in frame.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"{label} missing columns: {missing_list}")


def _player_id_text(values: pd.Series) -> pd.Series:
    """Strip ids. Blank values and the text nan stay missing."""
    text: pd.Series = values.astype("string").str.strip()
    missing: pd.Series = text.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "<na>"])
    return text.mask(missing)


def shrink_qb_rating(
    prior_dropbacks: float,
    prior_epa: float,
    prior_strength: float = QB_PRIOR_DROPBACKS,
    replacement_epa: float = REPLACEMENT_EPA_PER_DROPBACK,
) -> dict[str, float]:
    """Shrink EPA per dropback toward replacement by the number of prior dropbacks.

    ``weight`` is ``n / (n + prior_strength)``. The value is
    ``weight * (prior_epa / n) + (1 - weight) * replacement_epa``.
    ``n`` of 0, or a missing ``n``, returns the replacement and weight 0.
    The strength and the replacement are fixed. They are not tuned per run.

    Args:
        prior_dropbacks: Dropbacks strictly before the game.
        prior_epa: Sum of EPA on those dropbacks.
        prior_strength: Pseudo-dropbacks of the replacement prior. Default 100.
        replacement_epa: EPA per dropback of a replacement passer. Default 0.

    Returns:
        Dict with ``value``, ``weight``, and ``raw_epa_per_dropback``.
        The raw rate is NaN when ``n`` is 0.

    Raises:
        ValueError: ``prior_strength`` is not a positive finite number, or
            ``replacement_epa`` is not finite.
    """
    strength: float = float(prior_strength)
    replacement: float = float(replacement_epa)
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("prior_strength must be a positive finite number")
    if not np.isfinite(replacement):
        raise ValueError("replacement_epa must be finite")
    dropbacks: float = 0.0
    if pd.notna(prior_dropbacks) and float(prior_dropbacks) > 0.0:
        dropbacks = float(prior_dropbacks)
    if dropbacks == 0.0:
        return {
            "value": replacement,
            "weight": 0.0,
            "raw_epa_per_dropback": float("nan"),
        }
    raw: float = float(prior_epa) / dropbacks
    weight: float = dropbacks / (dropbacks + strength)
    value: float = weight * raw + (1.0 - weight) * replacement
    return {"value": value, "weight": weight, "raw_epa_per_dropback": raw}


def _missing_qb_features(game_ids: pd.Series) -> pd.DataFrame:
    """QB columns for a call that did not pass starter or dropback tables."""
    frame = pd.DataFrame({"game_id": game_ids.to_numpy()})
    for column in QB_OUTPUT_COLUMNS:
        frame[column] = np.nan
    return frame


def _prepare_dropbacks(qb_dropbacks: pd.DataFrame) -> pd.DataFrame:
    """Keep dated dropbacks with a passer, a franchise, and a positive count."""
    frame: pd.DataFrame = qb_dropbacks.loc[:, list(QB_DROPBACK_COLUMNS)].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    frame["passer_player_id"] = _player_id_text(frame["passer_player_id"])
    frame["team_franchise"] = _player_id_text(frame["team_franchise"])
    frame["dropbacks"] = pd.to_numeric(frame["dropbacks"], errors="coerce")
    frame["dropback_epa"] = pd.to_numeric(frame["dropback_epa"], errors="coerce").fillna(0.0)
    kept: pd.Series = (
        frame["gameday"].notna()
        & frame["passer_player_id"].notna()
        & frame["team_franchise"].notna()
        & frame["dropbacks"].notna()
        & (frame["dropbacks"] > 0)
    )
    return frame.loc[kept].reset_index(drop=True)


def _prior_player_totals(
    passer_ids: pd.Series,
    query_dates: np.ndarray,
    dropbacks: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Cumulative dropbacks and EPA for each passer strictly before ``query_dates``.

    Same-day rows are excluded. ``searchsorted(..., side='left') - 1`` is the
    last history row before the query date.
    """
    row_count: int = len(passer_ids)
    prior_dropbacks: np.ndarray = np.zeros(row_count, dtype=float)
    prior_epa: np.ndarray = np.zeros(row_count, dtype=float)
    if dropbacks.empty:
        return {"dropbacks": prior_dropbacks, "epa": prior_epa}
    history: pd.DataFrame = (
        dropbacks.groupby(["passer_player_id", "game_id", "gameday"], as_index=False)
        .agg(dropbacks=("dropbacks", "sum"), dropback_epa=("dropback_epa", "sum"))
        .sort_values(["passer_player_id", "gameday", "game_id"], kind="mergesort")
    )
    by_passer: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for passer, group in history.groupby("passer_player_id", sort=False):
        by_passer[str(passer)] = (
            group["gameday"].to_numpy(dtype="datetime64[ns]"),
            np.cumsum(group["dropbacks"].to_numpy(dtype=float)),
            np.cumsum(group["dropback_epa"].to_numpy(dtype=float)),
        )
    queries = pd.DataFrame(
        {
            "passer": passer_ids.astype("string").to_numpy(),
            "date": query_dates,
            "pos": np.arange(row_count),
        }
    )
    known = queries.loc[queries["passer"].notna()]
    for passer, group in known.groupby("passer", sort=False):
        stored: tuple[np.ndarray, np.ndarray, np.ndarray] | None = by_passer.get(str(passer))
        if stored is None:
            continue
        dates, cumulative_dropbacks, cumulative_epa = stored
        prior_index: np.ndarray = (
            np.searchsorted(dates, group["date"].to_numpy(dtype="datetime64[ns]"), side="left") - 1
        )
        valid: np.ndarray = prior_index >= 0
        if not bool(valid.any()):
            continue
        slots: np.ndarray = group["pos"].to_numpy()[valid]
        prior_dropbacks[slots] = cumulative_dropbacks[prior_index[valid]]
        prior_epa[slots] = cumulative_epa[prior_index[valid]]
    return {"dropbacks": prior_dropbacks, "epa": prior_epa}


def _previous_primary_passers(
    franchises: pd.Series,
    query_dates: np.ndarray,
    dropbacks: pd.DataFrame,
) -> pd.Series:
    """Primary passer on each franchise's last team-game strictly before the query.

    Primary means the most dropbacks. Ties keep the lower passer id.
    """
    row_count: int = len(franchises)
    previous: pd.Series = pd.Series(pd.NA, index=range(row_count), dtype="string")
    if dropbacks.empty:
        return previous
    ranked: pd.DataFrame = dropbacks.sort_values(
        ["team_franchise", "game_id", "dropbacks", "passer_player_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    primary: pd.DataFrame = (
        ranked.groupby(["game_id", "team_franchise"], as_index=False)
        .head(1)
        .sort_values(["team_franchise", "gameday", "game_id"], kind="mergesort")
    )
    by_team: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for franchise, group in primary.groupby("team_franchise", sort=False):
        by_team[str(franchise)] = (
            group["gameday"].to_numpy(dtype="datetime64[ns]"),
            group["passer_player_id"].astype("string").to_numpy(),
        )
    queries = pd.DataFrame(
        {
            "franchise": franchises.astype("string").to_numpy(),
            "date": query_dates,
            "pos": np.arange(row_count),
        }
    )
    known = queries.loc[queries["franchise"].notna()]
    for franchise, group in known.groupby("franchise", sort=False):
        stored_team: tuple[np.ndarray, np.ndarray] | None = by_team.get(str(franchise))
        if stored_team is None:
            continue
        dates, passers = stored_team
        prior_index = (
            np.searchsorted(dates, group["date"].to_numpy(dtype="datetime64[ns]"), side="left") - 1
        )
        valid = prior_index >= 0
        if not bool(valid.any()):
            continue
        slots = group["pos"].to_numpy()[valid]
        previous.iloc[slots] = passers[prior_index[valid]]
    return previous


def _recent_team_pass_epa(
    franchises: pd.Series,
    query_dates: np.ndarray,
    dropbacks: pd.DataFrame,
    lookback_games: int,
) -> np.ndarray:
    """EPA per dropback over the last ``lookback_games`` team-games before the query.

    Every passer on those team-games counts. Same-day team-games are excluded.
    """
    rates: np.ndarray = np.full(len(franchises), np.nan)
    if dropbacks.empty:
        return rates
    totals: pd.DataFrame = (
        dropbacks.groupby(["team_franchise", "game_id", "gameday"], as_index=False)
        .agg(dropbacks=("dropbacks", "sum"), dropback_epa=("dropback_epa", "sum"))
        .sort_values(["team_franchise", "gameday", "game_id"], kind="mergesort")
    )
    by_team: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for franchise, group in totals.groupby("team_franchise", sort=False):
        by_team[str(franchise)] = (
            group["gameday"].to_numpy(dtype="datetime64[ns]"),
            np.cumsum(group["dropbacks"].to_numpy(dtype=float)),
            np.cumsum(group["dropback_epa"].to_numpy(dtype=float)),
        )
    queries = pd.DataFrame(
        {
            "franchise": franchises.astype("string").to_numpy(),
            "date": query_dates,
            "pos": np.arange(len(franchises)),
        }
    )
    known = queries.loc[queries["franchise"].notna()]
    for franchise, group in known.groupby("franchise", sort=False):
        stored_totals: tuple[np.ndarray, np.ndarray, np.ndarray] | None = by_team.get(
            str(franchise)
        )
        if stored_totals is None:
            continue
        dates, cumulative_dropbacks, cumulative_epa = stored_totals
        end_index: np.ndarray = (
            np.searchsorted(dates, group["date"].to_numpy(dtype="datetime64[ns]"), side="left") - 1
        )
        valid = end_index >= 0
        if not bool(valid.any()):
            continue
        end_valid: np.ndarray = end_index[valid]
        start: np.ndarray = np.maximum(end_valid - lookback_games + 1, 0)
        window_dropbacks: np.ndarray = cumulative_dropbacks[end_valid] - np.where(
            start > 0, cumulative_dropbacks[start - 1], 0.0
        )
        window_epa: np.ndarray = cumulative_epa[end_valid] - np.where(
            start > 0, cumulative_epa[start - 1], 0.0
        )
        usable: np.ndarray = window_dropbacks > 0
        slots = group["pos"].to_numpy()[valid][usable]
        rates[slots] = window_epa[usable] / window_dropbacks[usable]
    return rates


def attach_qb_features(
    games: pd.DataFrame,
    expected_starters: pd.DataFrame | None,
    qb_dropbacks: pd.DataFrame | None,
    lookback_games: int = 10,
    prior_strength: float = QB_PRIOR_DROPBACKS,
    replacement_epa: float = REPLACEMENT_EPA_PER_DROPBACK,
) -> pd.DataFrame:
    """Pregame QB rating, the gap to recent team passing EPA, and the change flag.

    The depth-chart starter is used when it is present. A null chart falls
    back to the previous game's primary passer for that franchise. The change
    flag is 1 only when the chart and that previous passer both exist and
    differ, and 0 when they match. A missing chart cannot confirm a change,
    so the flag stays missing.

    History is every dropback by that passer with ``gameday`` strictly before
    the game, across teams and seasons. A known starter with no prior
    dropbacks is rated at replacement. An unknown starter stays missing.
    Team passing EPA is EPA per dropback over the last ``lookback_games``
    team-games, counting every passer. CPOE is not an input.

    Args:
        games: One row per game, with franchise columns and ``gameday``.
        expected_starters: Depth-chart QB1 ids. None only when ``qb_dropbacks``
            is also None.
        qb_dropbacks: Who played. Not the starter. None only when
            ``expected_starters`` is also None.
        lookback_games: Team-game window for recent passing EPA.
        prior_strength: Shrinkage strength passed to ``shrink_qb_rating``.
        replacement_epa: Replacement EPA per dropback.

    Returns:
        One row per ``game_id`` with ``QB_OUTPUT_COLUMNS``. Model columns are
        the four rating columns. The rest are diagnostics.

    Raises:
        ValueError: Exactly one of the QB tables is provided, a required
            column is missing, ``game_id`` is duplicated, or ``lookback_games``
            is below 1.
    """
    game_columns: tuple[str, ...] = (
        "game_id",
        "gameday",
        "home_team_franchise",
        "away_team_franchise",
    )
    _require_frame_columns(games, game_columns, "games")
    if games["game_id"].duplicated().any():
        raise ValueError("games game_id must be unique")
    if expected_starters is None and qb_dropbacks is None:
        return _missing_qb_features(games["game_id"])
    if expected_starters is None or qb_dropbacks is None:
        raise ValueError(
            "expected_starters and qb_dropbacks must both be provided or both be omitted"
        )
    if lookback_games < 1:
        raise ValueError("lookback_games must be at least 1")
    _require_frame_columns(expected_starters, EXPECTED_STARTER_COLUMNS, "expected_starters")
    _require_frame_columns(qb_dropbacks, QB_DROPBACK_COLUMNS, "qb_dropbacks")
    if expected_starters["game_id"].duplicated().any():
        raise ValueError("expected_starters game_id must be unique")

    schedule: pd.DataFrame = games.loc[:, list(game_columns)].copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    schedule["home_team_franchise"] = _player_id_text(schedule["home_team_franchise"])
    schedule["away_team_franchise"] = _player_id_text(schedule["away_team_franchise"])
    starters: pd.DataFrame = expected_starters.loc[:, list(EXPECTED_STARTER_COLUMNS)].copy()
    starters["home_starter_player_id"] = _player_id_text(starters["home_starter_player_id"])
    starters["away_starter_player_id"] = _player_id_text(starters["away_starter_player_id"])
    merged: pd.DataFrame = schedule.merge(starters, on="game_id", how="left")
    dropbacks: pd.DataFrame = _prepare_dropbacks(qb_dropbacks)
    home_side = pd.DataFrame(
        {
            "game_id": merged["game_id"].to_numpy(),
            "gameday": merged["gameday"].to_numpy(dtype="datetime64[ns]"),
            "side": "home",
            "franchise": merged["home_team_franchise"].to_numpy(),
            "chart_starter": merged["home_starter_player_id"].to_numpy(),
        }
    )
    away_side = pd.DataFrame(
        {
            "game_id": merged["game_id"].to_numpy(),
            "gameday": merged["gameday"].to_numpy(dtype="datetime64[ns]"),
            "side": "away",
            "franchise": merged["away_team_franchise"].to_numpy(),
            "chart_starter": merged["away_starter_player_id"].to_numpy(),
        }
    )
    sides: pd.DataFrame = pd.concat([home_side, away_side], ignore_index=True)
    query_dates: np.ndarray = sides["gameday"].to_numpy(dtype="datetime64[ns]")
    chart: pd.Series = _player_id_text(sides["chart_starter"])
    previous: pd.Series = _previous_primary_passers(sides["franchise"], query_dates, dropbacks)
    missing_chart: pd.Series = chart.isna()
    missing_previous: pd.Series = previous.isna()
    starter_id: pd.Series = chart.mask(missing_chart, previous)
    source: pd.Series = pd.Series(pd.NA, index=sides.index, dtype="string")
    source.loc[~missing_chart] = "depth_chart"
    source.loc[missing_chart & ~missing_previous] = "previous_passer"
    changed: np.ndarray = np.full(len(sides), np.nan)
    both_known: pd.Series = ~missing_chart & ~missing_previous
    differs: pd.Series = (chart.astype("string") != previous.astype("string")).fillna(False)
    both_known_flags: np.ndarray = both_known.fillna(False).to_numpy(dtype=bool)
    differs_flags: np.ndarray = differs.to_numpy(dtype=bool)
    changed[both_known_flags & differs_flags] = 1.0
    changed[both_known_flags & ~differs_flags] = 0.0
    prior: dict[str, np.ndarray] = _prior_player_totals(starter_id, query_dates, dropbacks)
    ratings: np.ndarray = np.empty(len(sides), dtype=float)
    for index, starter in enumerate(starter_id.tolist()):
        if pd.isna(starter):
            ratings[index] = np.nan
            continue
        ratings[index] = shrink_qb_rating(
            float(prior["dropbacks"][index]),
            float(prior["epa"][index]),
            prior_strength=prior_strength,
            replacement_epa=replacement_epa,
        )["value"]
    team_rates: np.ndarray = _recent_team_pass_epa(
        sides["franchise"],
        query_dates,
        dropbacks,
        lookback_games,
    )
    gap: np.ndarray = np.where(
        np.isfinite(ratings) & np.isfinite(team_rates),
        ratings - team_rates,
        np.nan,
    )
    sides = sides.assign(
        starter_player_id=starter_id.to_numpy(),
        starter_source=source.to_numpy(),
        qb_changed=changed,
        qb_rating=ratings,
        rating_minus_team_pass_epa=gap,
    )
    wide = pd.DataFrame({"game_id": schedule["game_id"].to_numpy()})
    for side_name in ("home", "away"):
        part: pd.DataFrame = sides.loc[sides["side"] == side_name].set_index("game_id")
        aligned: pd.DataFrame = part.reindex(wide["game_id"].to_numpy())
        wide[f"{side_name}_qb_rating"] = aligned["qb_rating"].to_numpy()
        wide[f"{side_name}_qb_rating_minus_team_pass_epa"] = aligned[
            "rating_minus_team_pass_epa"
        ].to_numpy()
        wide[f"{side_name}_starter_player_id"] = aligned["starter_player_id"].to_numpy()
        wide[f"{side_name}_starter_source"] = aligned["starter_source"].to_numpy()
        wide[f"{side_name}_qb_changed"] = aligned["qb_changed"].to_numpy()
    return wide.loc[:, ["game_id", *QB_OUTPUT_COLUMNS]]


def build_pregame_features(
    fct_games: pd.DataFrame,
    fct_team_game_epa: pd.DataFrame,
    lookback_games: int = 10,
    shrink_factor: float = SEASON_SHRINK_FACTOR,
    blend_k: float = SEASON_BLEND_K,
    expected_starters: pd.DataFrame | None = None,
    qb_dropbacks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build pre-game features for each game using only prior game data.

    LEAKAGE PROTECTION: For each game G, only uses data from games with
    gameday strictly before G's gameday.

    Args:
        fct_games: Games fact table with game_id, season, gameday, home/away teams, scores
        fct_team_game_epa: Team-game EPA metrics
        lookback_games: Number of recent games to include in rolling features
        shrink_factor: Fraction of the prior-season gap removed. Default matches Elo.
        blend_k: Current-season blend strength. Weight is ``n / (n + blend_k)``.
        expected_starters: Depth-chart QB1 ids, one row per game. Omit together
            with ``qb_dropbacks`` to leave the QB columns missing.
        qb_dropbacks: Passer dropbacks. This is who played, not the starter.

    Returns:
        DataFrame with game_id and feature columns

    Feature columns:
        - home_recent_offensive_epa_per_play: Home team's avg offensive EPA (last N games)
        - home_recent_defensive_epa_per_play: Home team's avg defensive EPA allowed (last N games)
        - away_recent_offensive_epa_per_play: Away team's avg offensive EPA (last N games)
        - away_recent_defensive_epa_per_play: Away team's avg defensive EPA allowed (last N games)
        - home_recent_win_rate: Home team's win rate (last N games)
        - away_recent_win_rate: Away team's win rate (last N games)
        - home_rest_days: Days since home team's last game
        - away_rest_days: Days since away team's last game
        - home_season_offensive_epa_per_play: Home team's season-to-date avg offensive EPA
        - away_season_offensive_epa_per_play: Away team's season-to-date avg offensive EPA
        - home_season_defensive_epa_per_play: Home season-to-date EPA allowed (h2 sets)
        - away_season_defensive_epa_per_play: Away season-to-date EPA allowed (h2 sets)
        - home_regressed_recent_offensive_epa_per_play: Shrunk prior recent offense, blended
        - home_regressed_recent_defensive_epa_per_play: Shrunk prior recent defense, blended
        - away_regressed_recent_offensive_epa_per_play: Shrunk prior recent offense, blended
        - away_regressed_recent_defensive_epa_per_play: Shrunk prior recent defense, blended
        - home_regressed_season_offensive_epa_per_play: Shrunk prior season offense, blended
        - away_regressed_season_offensive_epa_per_play: Shrunk prior season offense, blended
        - home_regressed_season_defensive_epa_per_play: Shrunk prior season defense, blended
        - away_regressed_season_defensive_epa_per_play: Shrunk prior season defense, blended
        - home and away recent pass and rush EPA per play, offense and defense
        - home and away regressed-season pass and rush EPA per play, offense and defense
        - home_games_played: Count of prior games for home team (legacy set only)
        - away_games_played: Count of prior games for away team (legacy set only)
        - diff_recent_offensive_epa_per_play: Home recent offensive EPA minus away
        - diff_recent_defensive_epa_per_play: Home recent defensive EPA allowed minus away
        - diff_recent_win_rate: Home recent win rate minus away
        - diff_season_offensive_epa_per_play: Home season offensive EPA minus away
        - diff_rest_days: Home rest days minus away rest days

    Missing history handling: Features use NaN (not 0) when insufficient prior games exist.
    A difference is NaN when either side is NaN. Use sklearn
    SimpleImputer(strategy='constant', fill_value=0) for EPA features and for
    difference columns, and SimpleImputer(strategy='constant', fill_value=0.5)
    for level win_rate columns in the training pipeline.
    games_played is still computed so the old matrix can be reproduced. The default
    training set omits it, the differences, the season defensive columns, the
    regressed columns, and the pass/rush columns. See ``feature_columns_for``.

    Season defensive EPA uses only prior games of the same season. Regressed
    columns carry the previous season's final value, shrunk toward that
    season's league mean, and blend in current-season games with weight
    ``n / (n + blend_k)``. The prior season is ``season - 1`` only, and only
    rows with ``gameday`` strictly before this game. Week 1 of the first
    season in the frame stays missing because there is no prior season.

    Pass and rush columns use ``pass_epa / pass_plays`` and
    ``rush_epa / rush_plays`` from the mart. A zero play count stays missing.
    Recent pass and rush use the same cross-season ``lookback_games`` window
    as the level recent EPA columns. Regressed-season pass and rush use the
    same shrinkage and blend as the combined regressed-season columns.
    Defensive pass and rush EPA allowed is the opponent's offensive rate in
    that game, because the mart stores pass and rush EPA on the offense row.
    A team-game frame with none of those source columns leaves the splits
    missing. A frame with only some of them raises ``ValueError``.

    Clean EPA uses ``avg_clean_offensive_epa`` and
    ``avg_clean_defensive_epa_allowed``. Recent, season, and regressed
    columns use the same windows, ``shrink_factor``, and ``blend_k`` as the
    original EPA columns. The league mean is the clean column, not the
    original one. Recent clean success rate uses the same cross-season
    lookback. A frame with none of a clean source pair leaves those features
    missing. A frame with only one column of a pair raises ``ValueError``.

    Starting-QB columns use ``expected_starters`` and ``qb_dropbacks``.
    Both omitted leaves those columns missing. Exactly one raises
    ``ValueError``. The starter id is the depth-chart QB1 when present,
    otherwise the previous game's primary passer. Ratings use only dropbacks
    with ``gameday`` strictly before the game. See ``attach_qb_features``.
    """
    _validate_regression_settings(shrink_factor, blend_k)
    # Validate required columns
    required_games_cols = [
        "game_id",
        "season",
        "gameday",
        "home_team_franchise",
        "away_team_franchise",
        "home_won",
        "home_rest",
        "away_rest",
    ]
    missing = set(required_games_cols) - set(fct_games.columns)
    if missing:
        raise ValueError(f"fct_games missing columns: {missing}")

    required_epa_cols = [
        "game_id",
        "season",
        "gameday",
        "team_franchise",
        "avg_offensive_epa",
        "avg_defensive_epa_allowed",
    ]
    missing = set(required_epa_cols) - set(fct_team_game_epa.columns)
    if missing:
        raise ValueError(f"fct_team_game_epa missing columns: {missing}")

    # Ensure gameday is datetime
    fct_games = fct_games.copy()
    fct_team_game_epa = fct_team_game_epa.copy()
    fct_games["gameday"] = pd.to_datetime(fct_games["gameday"])
    fct_team_game_epa["gameday"] = pd.to_datetime(fct_team_game_epa["gameday"])

    # Sort by gameday (critical for leakage prevention)
    fct_games = fct_games.sort_values("gameday").reset_index(drop=True)
    fct_team_game_epa = fct_team_game_epa.sort_values("gameday").reset_index(drop=True)
    fct_team_game_epa = _attach_pass_rush_rates(fct_team_game_epa)
    _refuse_partial_source(fct_team_game_epa, CLEAN_EPA_SOURCE_COLUMNS, "clean EPA")
    _refuse_partial_source(fct_team_game_epa, CLEAN_SUCCESS_SOURCE_COLUMNS, "clean success")

    features_list = []

    for idx, game in fct_games.iterrows():
        game_id = game["game_id"]
        game_date = game["gameday"]
        home_team = game["home_team_franchise"]
        away_team = game["away_team_franchise"]

        # Get prior games for both teams (strictly before this game's date)
        prior_games = fct_team_game_epa[fct_team_game_epa["gameday"] < game_date].copy()

        # Home team prior games
        home_prior = prior_games[prior_games["team_franchise"] == home_team].copy()
        away_prior = prior_games[prior_games["team_franchise"] == away_team].copy()

        # Rolling win rate (from fct_games via team_franchise)
        home_recent_games = fct_games[
            (fct_games["gameday"] < game_date)
            & (
                (fct_games["home_team_franchise"] == home_team)
                | (fct_games["away_team_franchise"] == home_team)
            )
        ].tail(lookback_games)

        away_recent_games = fct_games[
            (fct_games["gameday"] < game_date)
            & (
                (fct_games["home_team_franchise"] == away_team)
                | (fct_games["away_team_franchise"] == away_team)
            )
        ].tail(lookback_games)

        home_wins = (
            (home_recent_games["home_team_franchise"] == home_team)
            & (home_recent_games["home_won"] == 1)
        ).sum() + (
            (home_recent_games["away_team_franchise"] == home_team)
            & (home_recent_games["home_won"] == 0)
        ).sum()

        away_wins = (
            (away_recent_games["home_team_franchise"] == away_team)
            & (away_recent_games["home_won"] == 1)
        ).sum() + (
            (away_recent_games["away_team_franchise"] == away_team)
            & (away_recent_games["home_won"] == 0)
        ).sum()

        home_recent_win_rate = (
            home_wins / len(home_recent_games) if len(home_recent_games) > 0 else float("nan")
        )
        away_recent_win_rate = (
            away_wins / len(away_recent_games) if len(away_recent_games) > 0 else float("nan")
        )

        # Season-to-date EPA (expanding window)
        # Use season column, not gameday.year (January playoffs belong to prior season)
        game_season = int(game["season"])
        league_means: dict[str, float] = _prior_season_league_means(prior_games, game_season - 1)
        clean_league_means: dict[str, float] = _prior_season_league_means(
            prior_games,
            game_season - 1,
            offensive_column=CLEAN_OFFENSIVE_EPA_COLUMN,
            defensive_column=CLEAN_DEFENSIVE_EPA_COLUMN,
        )
        home_epa: dict[str, float] = _epa_feature_block(
            "home",
            home_prior,
            game_season,
            lookback_games,
            league_means,
            shrink_factor,
            blend_k,
            "avg_offensive_epa",
            "avg_defensive_epa_allowed",
        )
        away_epa: dict[str, float] = _epa_feature_block(
            "away",
            away_prior,
            game_season,
            lookback_games,
            league_means,
            shrink_factor,
            blend_k,
            "avg_offensive_epa",
            "avg_defensive_epa_allowed",
        )
        home_clean: dict[str, float] = _epa_feature_block(
            "home",
            home_prior,
            game_season,
            lookback_games,
            clean_league_means,
            shrink_factor,
            blend_k,
            CLEAN_OFFENSIVE_EPA_COLUMN,
            CLEAN_DEFENSIVE_EPA_COLUMN,
            infix="clean_",
        )
        away_clean: dict[str, float] = _epa_feature_block(
            "away",
            away_prior,
            game_season,
            lookback_games,
            clean_league_means,
            shrink_factor,
            blend_k,
            CLEAN_OFFENSIVE_EPA_COLUMN,
            CLEAN_DEFENSIVE_EPA_COLUMN,
            infix="clean_",
        )
        home_clean_success: dict[str, float] = _recent_clean_success(
            "home", home_prior, lookback_games
        )
        away_clean_success: dict[str, float] = _recent_clean_success(
            "away", away_prior, lookback_games
        )
        pass_rush_league: dict[str, float] = _prior_season_pass_rush_means(
            prior_games, game_season - 1
        )
        home_pass_rush_recent: dict[str, float] = _recent_pass_rush(
            home_prior, "home", lookback_games
        )
        away_pass_rush_recent: dict[str, float] = _recent_pass_rush(
            away_prior, "away", lookback_games
        )
        home_pass_rush_season: dict[str, float] = _regressed_season_pass_rush(
            home_prior,
            "home",
            game_season,
            pass_rush_league,
            shrink_factor,
            blend_k,
        )
        away_pass_rush_season: dict[str, float] = _regressed_season_pass_rush(
            away_prior,
            "away",
            game_season,
            pass_rush_league,
            shrink_factor,
            blend_k,
        )

        # Rest days (from fct_games.home_rest/away_rest)
        home_rest_days = game["home_rest"] if pd.notna(game["home_rest"]) else 7
        away_rest_days = game["away_rest"] if pd.notna(game["away_rest"]) else 7

        features_list.append(
            {
                "game_id": game_id,
                "home_recent_win_rate": home_recent_win_rate,
                "away_recent_win_rate": away_recent_win_rate,
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                "home_games_played": len(home_prior),
                "away_games_played": len(away_prior),
                **home_epa,
                **away_epa,
                **home_clean,
                **away_clean,
                **home_clean_success,
                **away_clean_success,
                **home_pass_rush_recent,
                **away_pass_rush_recent,
                **home_pass_rush_season,
                **away_pass_rush_season,
            }
        )

    frame: pd.DataFrame = add_home_minus_away_features(pd.DataFrame(features_list))
    ordered_columns: list[str] = [
        "game_id",
        *PREGAME_COMPUTED_COLUMNS,
        *DIFF_FEATURE_COLUMNS,
        *SEASON_DEFENSIVE_EPA_COLUMNS,
        *REGRESSED_EPA_COLUMNS,
        *PASS_RUSH_EPA_COLUMNS,
        *CLEAN_EPA_FEATURE_COLUMNS,
        *CLEAN_SUCCESS_RATE_COLUMNS,
        *QB_OUTPUT_COLUMNS,
    ]
    qb_features: pd.DataFrame = attach_qb_features(
        fct_games,
        expected_starters,
        qb_dropbacks,
        lookback_games=lookback_games,
    )
    frame = frame.merge(qb_features, on="game_id", how="left")
    return frame.loc[:, ordered_columns]


def add_home_minus_away_features(features: pd.DataFrame) -> pd.DataFrame:
    """Append home-minus-away columns. A missing side stays missing.

    Each difference is the home level minus the away level. Defensive EPA is
    EPA allowed, and that gap is not flipped. The function does not look at
    another row. Leakage protection is the level columns it subtracts.

    Args:
        features: One row per game, including both sides of each pair in
            ``DIFFERENTIAL_FEATURES``.

    Returns:
        A copy with ``DIFF_FEATURE_COLUMNS`` appended, in that order.

    Raises:
        ValueError: A level column required by a difference is missing.
    """
    missing: list[str] = [
        column
        for item in DIFFERENTIAL_FEATURES
        for column in (item.home_column, item.away_column)
        if column not in features.columns
    ]
    if missing:
        missing_list: str = ", ".join(sorted(set(missing)))
        raise ValueError(f"features missing columns for differentials: {missing_list}")

    frame: pd.DataFrame = features.copy()
    for item in DIFFERENTIAL_FEATURES:
        frame[item.diff_column] = frame[item.home_column] - frame[item.away_column]
    return frame


def feature_columns_for(feature_set: str) -> tuple[str, ...]:
    """Return the model columns for a named feature set.

    ``current`` is the default and omits ``home_games_played``,
    ``away_games_played``, the differential columns, season defensive EPA,
    and the regressed columns.
    ``with_games_played`` is the previous matrix.
    ``diff`` is ``current`` plus home-minus-away columns.
    ``diff_only`` is the differences and nothing else.
    ``h2_def`` is ``current`` plus season-to-date defensive EPA allowed.
    ``h2_full`` is ``h2_def`` plus the regressed season and recent EPA columns.
    ``h3_split`` is ``h2_full`` plus recent and regressed-season pass/rush EPA.
    ``h3_replace`` drops the combined recent and regressed-season EPA that
    those splits replace and keeps the rest of ``h2_full``.
    ``h4_clean`` is ``h2_full`` with every EPA input taken from the clean
    mart columns. Win rate and rest stay. ``h4_clean_sr`` adds the recent
    clean success rate for offense and defense.
    ``h5_qb`` is ``h2_full`` plus the home and away starting-QB ratings and
    each rating minus that team's recent passing EPA.

    Args:
        feature_set: One of ``FEATURE_SET_NAMES``.

    Returns:
        Column names in training-matrix order.

    Raises:
        ValueError: The name is not a known feature set.
    """
    columns: tuple[str, ...] | None = FEATURE_SET_COLUMNS.get(feature_set)
    if columns is None:
        known: str = ", ".join(FEATURE_SET_NAMES)
        raise ValueError(f"Unknown feature set {feature_set!r}. Known sets: {known}.")
    return columns


def prepare_training_data(
    fct_games: pd.DataFrame,
    features: pd.DataFrame,
    target_column: str = "home_won",
    feature_set: str = FEATURE_SET_CURRENT,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.

    Args:
        fct_games: Games fact table with target column
        features: Feature DataFrame from build_pregame_features
        target_column: Name of target column in fct_games
        feature_set: One of ``FEATURE_SET_NAMES``. Default is ``current``.

    Returns:
        Tuple of (X: feature DataFrame, y: target Series)

    Example:
        >>> X, y = prepare_training_data(games, features)
        >>> print(X.shape, y.shape)
    """
    merged = features.merge(
        fct_games[["game_id", target_column, "gameday", "season"]], on="game_id", how="inner"
    )

    feature_cols = list(feature_columns_for(feature_set))

    X = merged[list(MODEL_ID_COLUMNS) + feature_cols].copy()
    y = merged[target_column].copy()

    return X, y


def split_by_season(
    X: pd.DataFrame,
    y: pd.Series,
    train_end_season: int = 2023,
    val_season: int = 2024,
    test_season: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test by season (time-based split).

    Args:
        X: Feature DataFrame with 'season' column
        y: Target Series
        train_end_season: Last season to include in training (inclusive)
        val_season: Validation season
        test_season: Test season

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = split_by_season(X, y)
    """
    if "season" not in X.columns:
        raise ValueError("X must have 'season' column for time-based split")

    train_mask = X["season"] <= train_end_season
    val_mask = X["season"] == val_season
    test_mask = X["season"] == test_season

    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    X_test = X[test_mask].copy()

    y_train = y[train_mask].copy()
    y_val = y[val_mask].copy()
    y_test = y[test_mask].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def _expected_home_win(
    home_rating: float,
    away_rating: float,
    home_advantage: float,
    scale: float,
) -> float:
    """Map a rating gap plus home advantage to a home win probability."""
    exponent: float = -((home_rating + home_advantage) - away_rating) / scale
    return 1.0 / (1.0 + 10.0**exponent)


def _regress_rating(rating: float, mean_rating: float, season_reversion: float) -> float:
    """Move a rating part of the way back to the league mean."""
    return mean_rating + (1.0 - season_reversion) * (rating - mean_rating)


def _elo_outcome(row: pd.Series, score_columns: bool) -> float | None:
    """Return the home result used to update Elo, or None when it is unknown.

    A tied score is 0.5 so the rating does not treat the tie as a home loss.
    Without scores, ``home_won`` is the result. A null label does not update.
    """
    if score_columns:
        home_score = row["home_score"]
        away_score = row["away_score"]
        if pd.notna(home_score) and pd.notna(away_score) and float(home_score) == float(away_score):
            return 0.5
    if pd.isna(row["home_won"]):
        return None
    return float(row["home_won"])


def elo_home_probabilities(
    games: pd.DataFrame,
    k: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    mean_rating: float = ELO_MEAN_RATING,
    season_reversion: float = ELO_SEASON_REVERSION,
    scale: float = ELO_SCALE,
) -> dict[str, Any]:
    """Home win probabilities from games strictly before each kickoff.

    Ratings start at ``mean_rating``. Games that share a ``gameday`` are
    scored from the same ratings, then updated together, so one game cannot
    see a same-day result. The first time a later season appears, every
    stored rating moves ``season_reversion`` of the way back to the mean.
    Home advantage is added in the probability only.

    Args:
        games: One row per game. Needs identifiers, franchises, and ``home_won``.
            ``home_score`` and ``away_score`` are optional and mark ties.
        k: Elo step size.
        home_advantage: Elo points added to the home team before the game.
        mean_rating: Starting rating and the season-regression target.
        season_reversion: Fraction of the gap to the mean removed between seasons.
        scale: Denominator in the logistic rating curve.

    Returns:
        Dict with ``probabilities`` (a Series aligned to ``games.index``) and
        the settings that were used.

    Raises:
        ValueError: A required column is missing or a setting is out of range.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not 0.0 <= season_reversion <= 1.0:
        raise ValueError("season_reversion must be in [0, 1]")

    required: list[str] = [
        "game_id",
        "season",
        "gameday",
        "home_team_franchise",
        "away_team_franchise",
        "home_won",
    ]
    missing: set[str] = set(required) - set(games.columns)
    if missing:
        raise ValueError(f"games missing columns for Elo: {sorted(missing)}")

    frame: pd.DataFrame = games.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    score_columns: bool = "home_score" in frame.columns and "away_score" in frame.columns
    ordered: pd.DataFrame = frame.sort_values(
        ["gameday", "season", "game_id"],
        kind="mergesort",
    )
    probabilities: pd.Series = pd.Series(np.nan, index=frame.index, dtype=float)
    ratings: dict[str, float] = {}
    pending: list[tuple[str, str, float, float]] = []
    current_season: int | None = None

    def apply_pending() -> None:
        """Apply results from earlier gamedays to the stored ratings."""
        for home_team, away_team, actual, expected in pending:
            delta: float = k * (actual - expected)
            ratings[home_team] = ratings[home_team] + delta
            ratings[away_team] = ratings[away_team] - delta
        pending.clear()

    def regress_all() -> None:
        """Regress every stored rating toward the mean once."""
        for team, rating in list(ratings.items()):
            ratings[team] = _regress_rating(rating, mean_rating, season_reversion)

    # One pass per gameday. Updates wait until the next date, so a game cannot
    # see a result from the same gameday. A new season regresses after those
    # prior-day updates and before any game in the new season is scored.
    for _gameday, day in ordered.groupby("gameday", sort=False):
        apply_pending()
        for season_value, part in day.groupby("season", sort=True):
            season: int = int(season_value)
            if current_season is None:
                current_season = season
            elif season > current_season:
                regress_all()
                current_season = season
            for row_index, row in part.iterrows():
                home_team: str = str(row["home_team_franchise"])
                away_team: str = str(row["away_team_franchise"])
                home_rating: float = ratings.get(home_team, mean_rating)
                away_rating: float = ratings.get(away_team, mean_rating)
                expected: float = _expected_home_win(
                    home_rating, away_rating, home_advantage, scale
                )
                probabilities.loc[row_index] = expected
                ratings.setdefault(home_team, mean_rating)
                ratings.setdefault(away_team, mean_rating)
                outcome: float | None = _elo_outcome(row, score_columns)
                if outcome is None:
                    continue
                pending.append((home_team, away_team, outcome, expected))

    return {
        "probabilities": probabilities,
        "k": k,
        "home_advantage": home_advantage,
        "mean_rating": mean_rating,
        "season_reversion": season_reversion,
        "scale": scale,
    }


def american_odds_to_implied_probability(american_odds: float) -> float:
    """Convert one American moneyline to a raw implied probability.

    Positive odds ``o`` are ``100 / (o + 100)``. Negative odds ``o`` are
    ``-o / (-o + 100)``. The result still includes the vig.

    Args:
        american_odds: Closing American moneyline. Zero is invalid.

    Returns:
        Implied probability in (0, 1).

    Raises:
        ValueError: The odds are zero or not finite.
    """
    odds: float = float(american_odds)
    if odds == 0.0 or not np.isfinite(odds):
        raise ValueError("American odds must be a non-zero finite number")
    if odds > 0.0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def no_vig_home_probability(home_moneyline: float, away_moneyline: float) -> float:
    """Remove the vig by normalizing the two raw implied probabilities.

    Args:
        home_moneyline: Closing American moneyline for the home team.
        away_moneyline: Closing American moneyline for the away team.

    Returns:
        Home win probability after the two raw prices are scaled to sum to 1.

    Raises:
        ValueError: Either price is not a usable American moneyline.
    """
    home_raw: float = american_odds_to_implied_probability(home_moneyline)
    away_raw: float = american_odds_to_implied_probability(away_moneyline)
    total: float = home_raw + away_raw
    if total <= 0.0:
        raise ValueError("Implied probabilities must sum to a positive number")
    return home_raw / total


def market_home_probabilities(games: pd.DataFrame) -> dict[str, Any]:
    """No-vig home probabilities from closing moneylines, when the columns exist.

    A frame without ``home_moneyline`` and ``away_moneyline`` is skipped.
    Rows with a null or invalid price are left as NaN and counted as missing.
    This does not read another game. The closing line is a column on the row.

    Args:
        games: Game table. Moneyline columns are optional.

    Returns:
        Dict with ``status`` (``ok`` or ``skipped``), ``reason``,
        ``probabilities`` (None when the columns are absent), ``n_scored``,
        and ``n_missing_lines``.
    """
    missing_columns: list[str] = [
        column
        for column in (MONEYLINE_HOME_COLUMN, MONEYLINE_AWAY_COLUMN)
        if column not in games.columns
    ]
    if missing_columns:
        return {
            "status": "skipped",
            "reason": "moneyline_columns_absent",
            "probabilities": None,
            "n_scored": 0,
            "n_missing_lines": 0,
        }

    values: list[float] = []
    n_missing: int = 0
    home_lines = games[MONEYLINE_HOME_COLUMN].tolist()
    away_lines = games[MONEYLINE_AWAY_COLUMN].tolist()
    for home_line, away_line in zip(home_lines, away_lines, strict=True):
        if pd.isna(home_line) or pd.isna(away_line):
            values.append(float("nan"))
            n_missing += 1
            continue
        try:
            values.append(no_vig_home_probability(float(home_line), float(away_line)))
        except ValueError:
            values.append(float("nan"))
            n_missing += 1
    probabilities: pd.Series = pd.Series(values, index=games.index, dtype=float)
    n_scored: int = int(probabilities.notna().sum())
    if n_scored == 0:
        return {
            "status": "skipped",
            "reason": "no_valid_moneylines",
            "probabilities": probabilities,
            "n_scored": 0,
            "n_missing_lines": n_missing,
        }
    return {
        "status": "ok",
        "reason": None,
        "probabilities": probabilities,
        "n_scored": n_scored,
        "n_missing_lines": n_missing,
    }
