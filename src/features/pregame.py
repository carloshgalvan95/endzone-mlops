"""
Pre-game feature engineering for NFL predictions.

CRITICAL: All features must be leakage-free. Features for game G use only data
from games strictly before G.

Features computed:
- Rolling/expanding team performance metrics (EPA per play, win rate)
- Rest days
- Prior-game counts (legacy feature set only)
- Home-minus-away differences of the level features (diff sets only)
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

FEATURE_SET_NAMES: tuple[str, ...] = (
    FEATURE_SET_CURRENT,
    FEATURE_SET_WITH_GAMES_PLAYED,
    FEATURE_SET_DIFF,
    FEATURE_SET_DIFF_ONLY,
)

FEATURE_SET_COLUMNS: dict[str, tuple[str, ...]] = {
    FEATURE_SET_CURRENT: PREGAME_FEATURE_COLUMNS,
    FEATURE_SET_WITH_GAMES_PLAYED: PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED,
    FEATURE_SET_DIFF: PREGAME_FEATURE_COLUMNS_DIFF,
    FEATURE_SET_DIFF_ONLY: PREGAME_FEATURE_COLUMNS_DIFF_ONLY,
}

# FiveThirtyEight-style Elo. Home advantage is added only when a game is
# scored, not stored on the team rating. One third of the gap to the mean
# is removed the first time a later season is seen. No margin multiplier.
ELO_K: float = 20.0
ELO_HOME_ADVANTAGE: float = 65.0
ELO_MEAN_RATING: float = 1505.0
ELO_SEASON_REVERSION: float = 1.0 / 3.0
ELO_SCALE: float = 400.0

MONEYLINE_HOME_COLUMN: str = "home_moneyline"
MONEYLINE_AWAY_COLUMN: str = "away_moneyline"


def build_pregame_features(
    fct_games: pd.DataFrame,
    fct_team_game_epa: pd.DataFrame,
    lookback_games: int = 10,
) -> pd.DataFrame:
    """
    Build pre-game features for each game using only prior game data.

    LEAKAGE PROTECTION: For each game G, only uses data from games with
    gameday strictly before G's gameday.

    Args:
        fct_games: Games fact table with game_id, season, gameday, home/away teams, scores
        fct_team_game_epa: Team-game EPA metrics
        lookback_games: Number of recent games to include in rolling features

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
    training set omits it and omits the differences. See ``feature_columns_for``.
    """
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

        # Compute rolling features (last N games)
        home_rolling = home_prior.tail(lookback_games)
        away_rolling = away_prior.tail(lookback_games)

        # Rolling EPA features (use NaN if no prior games, not 0)
        home_recent_off_epa = (
            home_rolling["avg_offensive_epa"].mean() if len(home_rolling) > 0 else float("nan")
        )
        home_recent_def_epa = (
            home_rolling["avg_defensive_epa_allowed"].mean()
            if len(home_rolling) > 0
            else float("nan")
        )
        away_recent_off_epa = (
            away_rolling["avg_offensive_epa"].mean() if len(away_rolling) > 0 else float("nan")
        )
        away_recent_def_epa = (
            away_rolling["avg_defensive_epa_allowed"].mean()
            if len(away_rolling) > 0
            else float("nan")
        )

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
        game_season = game["season"]
        home_season = home_prior[home_prior["season"] == game_season]
        away_season = away_prior[away_prior["season"] == game_season]

        home_season_off_epa = (
            home_season["avg_offensive_epa"].mean() if len(home_season) > 0 else float("nan")
        )
        away_season_off_epa = (
            away_season["avg_offensive_epa"].mean() if len(away_season) > 0 else float("nan")
        )

        # Rest days (from fct_games.home_rest/away_rest)
        home_rest_days = game["home_rest"] if pd.notna(game["home_rest"]) else 7
        away_rest_days = game["away_rest"] if pd.notna(game["away_rest"]) else 7

        features_list.append(
            {
                "game_id": game_id,
                "home_recent_offensive_epa_per_play": home_recent_off_epa,
                "home_recent_defensive_epa_per_play": home_recent_def_epa,
                "away_recent_offensive_epa_per_play": away_recent_off_epa,
                "away_recent_defensive_epa_per_play": away_recent_def_epa,
                "home_recent_win_rate": home_recent_win_rate,
                "away_recent_win_rate": away_recent_win_rate,
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                "home_season_offensive_epa_per_play": home_season_off_epa,
                "away_season_offensive_epa_per_play": away_season_off_epa,
                "home_games_played": len(home_prior),
                "away_games_played": len(away_prior),
            }
        )

    return add_home_minus_away_features(pd.DataFrame(features_list))


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
    ``away_games_played``, and the differential columns.
    ``with_games_played`` is the previous matrix.
    ``diff`` is ``current`` plus home-minus-away columns.
    ``diff_only`` is the differences and nothing else.

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
