"""
Pre-game feature engineering for NFL predictions.

CRITICAL: All features must be leakage-free. Features for game G use only data
from games strictly before G.

Features computed:
- Rolling/expanding team performance metrics (EPA per play, win rate, point differential)
- Rest days
- Home field advantage indicator
"""

import pandas as pd

# Columns that identify a game. They are not model inputs.
MODEL_ID_COLUMNS: tuple[str, ...] = ("game_id", "gameday", "season")

# Model inputs produced by build_pregame_features, in training-matrix order.
PREGAME_FEATURE_COLUMNS: tuple[str, ...] = (
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
        - home_games_played: Count of prior games for home team
        - away_games_played: Count of prior games for away team

    Missing history handling: Features use NaN (not 0) when insufficient prior games exist.
    Use sklearn SimpleImputer(strategy='constant', fill_value=0) for EPA features and
    SimpleImputer(strategy='constant', fill_value=0.5) for win_rate in training pipeline.
    Games_played features allow models to learn uncertainty from sample size.
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

    return pd.DataFrame(features_list)


def prepare_training_data(
    fct_games: pd.DataFrame,
    features: pd.DataFrame,
    target_column: str = "home_won",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.

    Args:
        fct_games: Games fact table with target column
        features: Feature DataFrame from build_pregame_features
        target_column: Name of target column in fct_games

    Returns:
        Tuple of (X: feature DataFrame, y: target Series)

    Example:
        >>> X, y = prepare_training_data(games, features)
        >>> print(X.shape, y.shape)
    """
    merged = features.merge(
        fct_games[["game_id", target_column, "gameday", "season"]], on="game_id", how="inner"
    )

    feature_cols = list(PREGAME_FEATURE_COLUMNS)

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
