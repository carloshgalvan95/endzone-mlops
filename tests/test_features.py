"""
Tests for feature engineering.

Critical tests:
- No data leakage (features for game G do not use data from G or later)
- Split logic is correct
- Features compute correctly
"""

import pandas as pd
import pytest

from src.features.pregame import (
    DIFF_FEATURE_COLUMNS,
    FEATURE_SET_DIFF,
    FEATURE_SET_DIFF_ONLY,
    FEATURE_SET_NAMES,
    FEATURE_SET_WITH_GAMES_PLAYED,
    GAMES_PLAYED_COLUMNS,
    PREGAME_FEATURE_COLUMNS,
    PREGAME_FEATURE_COLUMNS_DIFF,
    PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED,
    build_pregame_features,
    feature_columns_for,
    prepare_training_data,
    split_by_season,
)


@pytest.fixture
def synthetic_games() -> pd.DataFrame:
    """Create synthetic game data for testing leakage."""
    data = {
        "game_id": ["G1", "G2", "G3", "G4", "G5"],
        "season": [2023, 2023, 2023, 2024, 2024],
        "gameday": pd.to_datetime(
            ["2023-09-10", "2023-09-17", "2023-09-24", "2024-09-08", "2024-09-15"]
        ),
        "home_team_franchise": ["KC", "BUF", "KC", "KC", "BUF"],
        "away_team_franchise": ["BUF", "MIA", "LAC", "BUF", "MIA"],
        "home_score": [30, 21, 28, 35, 24],
        "away_score": [20, 14, 17, 28, 21],
        "home_won": [1, 1, 1, 1, 1],
        "home_rest": [7, 7, 7, 365, 7],
        "away_rest": [7, 7, 7, 365, 7],
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_team_game_epa(synthetic_games: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic EPA data matching synthetic games."""
    rows = []

    for _, game in synthetic_games.iterrows():
        rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "gameday": game["gameday"],
                "team_franchise": game["home_team_franchise"],
                "avg_offensive_epa": 0.15,
                "avg_defensive_epa_allowed": -0.10,
            }
        )
        rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "gameday": game["gameday"],
                "team_franchise": game["away_team_franchise"],
                "avg_offensive_epa": 0.10,
                "avg_defensive_epa_allowed": -0.05,
            }
        )

    return pd.DataFrame(rows)


class TestBuildPregameFeatures:
    """Test feature building logic."""

    def test_no_leakage(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """
        Verify no data leakage: features for game G do not use data from G or later.

        Test strategy: For first game in dataset, features should all be defaults
        (0 or 0.5) since no prior games exist.
        """
        features = build_pregame_features(
            synthetic_games, synthetic_team_game_epa, lookback_games=5
        )

        # First game (G1) should have no prior data
        first_game_features = features[features["game_id"] == "G1"].iloc[0]

        # Expected defaults when no prior games exist (NaN, not 0)
        assert pd.isna(first_game_features["home_recent_offensive_epa_per_play"])
        assert pd.isna(first_game_features["away_recent_offensive_epa_per_play"])
        assert pd.isna(first_game_features["home_recent_win_rate"])
        assert pd.isna(first_game_features["away_recent_win_rate"])
        assert first_game_features["home_games_played"] == 0
        assert first_game_features["away_games_played"] == 0

        # Second game (G2) for BUF home should only use G1 (where BUF was away)
        second_game_features = features[features["game_id"] == "G2"].iloc[0]

        # BUF played G1 as away team, so should have 1 prior game
        # Win rate should be 0 (lost G1), EPA should be from G1
        assert second_game_features["home_recent_win_rate"] == 0.0

    def test_required_columns_validation(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """Missing required columns should raise ValueError."""
        # Drop required column from games
        games_missing = synthetic_games.drop(columns=["home_team_franchise"])

        with pytest.raises(ValueError, match="fct_games missing columns"):
            build_pregame_features(games_missing, synthetic_team_game_epa)

        # Drop required column from EPA
        epa_missing = synthetic_team_game_epa.drop(columns=["avg_offensive_epa"])

        with pytest.raises(ValueError, match="fct_team_game_epa missing columns"):
            build_pregame_features(synthetic_games, epa_missing)

    def test_feature_output_shape(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """Output should have one row per game with expected feature columns."""
        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)

        assert len(features) == len(synthetic_games)
        assert "game_id" in features.columns

        expected_features = [
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
            *DIFF_FEATURE_COLUMNS,
        ]

        for feature_name in expected_features:
            assert feature_name in features.columns

    def test_january_playoff_season_filter(self) -> None:
        """
        January playoff games use season column, not gameday.year.

        A January 2024 playoff game (season 2023) should include only
        September-December 2023 games in season-to-date features, not
        January 2024 games from season 2024.
        """
        # Regular season games in 2023 (Sep-Dec)
        games_data = {
            "game_id": ["G1_SEP", "G2_OCT", "G3_DEC", "G4_JAN_PLAYOFF"],
            "season": [2023, 2023, 2023, 2023],
            "gameday": pd.to_datetime(["2023-09-10", "2023-10-15", "2023-12-24", "2024-01-14"]),
            "home_team_franchise": ["KC", "KC", "KC", "KC"],
            "away_team_franchise": ["BUF", "MIA", "LAC", "BUF"],
            "home_score": [28, 30, 24, 35],
            "away_score": [21, 24, 21, 28],
            "home_won": [1, 1, 1, 1],
            "home_rest": [7, 7, 7, 21],
            "away_rest": [7, 7, 7, 21],
        }
        games_df = pd.DataFrame(games_data)

        # EPA for KC in those games
        epa_rows = []
        for _, game in games_df.iterrows():
            epa_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "gameday": game["gameday"],
                    "team_franchise": "KC",
                    "avg_offensive_epa": 0.20,
                    "avg_defensive_epa_allowed": -0.10,
                }
            )
        epa_df = pd.DataFrame(epa_rows)

        features = build_pregame_features(games_df, epa_df, lookback_games=10)

        # For January playoff game (G4_JAN_PLAYOFF), season-to-date should include G1-G3
        jan_playoff_features = features[features["game_id"] == "G4_JAN_PLAYOFF"].iloc[0]

        # KC has 3 prior games in season 2023, all with EPA 0.20
        assert (
            pytest.approx(jan_playoff_features["home_season_offensive_epa_per_play"], abs=0.001)
            == 0.20
        ), "Season-to-date should include all 3 prior games from season 2023"
        assert jan_playoff_features["home_games_played"] == 3, (
            "KC should have 3 prior games before playoff"
        )


class TestPrepareTrainingData:
    """Test training data preparation."""

    def test_prepare_training_data(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """Prepare training data merges features and target correctly."""
        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        X, y = prepare_training_data(synthetic_games, features, target_column="home_won")

        assert len(X) == len(synthetic_games)
        assert len(y) == len(synthetic_games)
        assert "game_id" in X.columns
        assert "season" in X.columns
        assert all(y == 1)
        assert set(GAMES_PLAYED_COLUMNS).isdisjoint(X.columns)
        assert set(DIFF_FEATURE_COLUMNS).isdisjoint(X.columns)
        assert set(DIFF_FEATURE_COLUMNS) <= set(features.columns)
        assert list(feature_columns_for("current")) == list(PREGAME_FEATURE_COLUMNS)

        legacy_x, _legacy_y = prepare_training_data(
            synthetic_games,
            features,
            feature_set=FEATURE_SET_WITH_GAMES_PLAYED,
        )
        assert set(GAMES_PLAYED_COLUMNS) <= set(legacy_x.columns)
        assert set(DIFF_FEATURE_COLUMNS).isdisjoint(legacy_x.columns)

    def test_feature_set_registry_adds_diff_without_changing_current(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """``diff`` appends gaps. ``diff_only`` is only the gaps. The old sets stay put."""
        assert FEATURE_SET_NAMES == ("current", "with_games_played", "diff", "diff_only")
        assert feature_columns_for("current") == PREGAME_FEATURE_COLUMNS
        assert feature_columns_for("with_games_played") == PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED
        assert feature_columns_for(FEATURE_SET_DIFF) == PREGAME_FEATURE_COLUMNS_DIFF
        assert feature_columns_for(FEATURE_SET_DIFF_ONLY) == DIFF_FEATURE_COLUMNS
        assert (
            PREGAME_FEATURE_COLUMNS_DIFF[: len(PREGAME_FEATURE_COLUMNS)] == PREGAME_FEATURE_COLUMNS
        )
        assert "home_games_played" not in feature_columns_for(FEATURE_SET_DIFF)
        assert "home_recent_offensive_epa_per_play" not in feature_columns_for(
            FEATURE_SET_DIFF_ONLY
        )

        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        diff_only, _target = prepare_training_data(
            synthetic_games,
            features,
            feature_set=FEATURE_SET_DIFF_ONLY,
        )
        model_columns = [
            column for column in diff_only.columns if column not in ("game_id", "gameday", "season")
        ]
        assert model_columns == list(DIFF_FEATURE_COLUMNS)

    def test_unknown_feature_set_is_rejected(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """A typo in the feature set name fails before a model is fit."""
        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        with pytest.raises(ValueError, match="Unknown feature set"):
            prepare_training_data(synthetic_games, features, feature_set="both")


class TestSplitBySeason:
    """Test time-based season splitting."""

    def test_split_by_season(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """Split by season produces correct splits."""
        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        X, y = prepare_training_data(synthetic_games, features)

        X_train, X_val, X_test, y_train, y_val, y_test = split_by_season(
            X, y, train_end_season=2023, val_season=2024, test_season=2025
        )

        # Train should have 2023 games (G1, G2, G3)
        assert len(X_train) == 3
        assert all(X_train["season"] == 2023)

        # Val should have 2024 games (G4, G5)
        assert len(X_val) == 2
        assert all(X_val["season"] == 2024)

        # Test should be empty (no 2025 games in fixture)
        assert len(X_test) == 0

    def test_split_requires_season_column(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """Split should fail if season column missing."""
        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        X, y = prepare_training_data(synthetic_games, features)

        X_no_season = X.drop(columns=["season"])

        with pytest.raises(ValueError, match="must have 'season' column"):
            split_by_season(X_no_season, y)
