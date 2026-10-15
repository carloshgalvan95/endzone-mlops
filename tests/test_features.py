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
    CLEAN_SUCCESS_RATE_COLUMNS,
    DIFF_FEATURE_COLUMNS,
    FEATURE_SET_DIFF,
    FEATURE_SET_DIFF_ONLY,
    FEATURE_SET_H2_DEF,
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H3_REPLACE,
    FEATURE_SET_H3_SPLIT,
    FEATURE_SET_H4_CLEAN,
    FEATURE_SET_H4_CLEAN_SR,
    FEATURE_SET_H5_QB,
    FEATURE_SET_NAMES,
    FEATURE_SET_WITH_GAMES_PLAYED,
    GAMES_PLAYED_COLUMNS,
    H2_CONTEXT_COLUMNS,
    H3_REPLACED_COMBINED_COLUMNS,
    PASS_RUSH_EPA_COLUMNS,
    PREGAME_FEATURE_COLUMNS,
    PREGAME_FEATURE_COLUMNS_DIFF,
    PREGAME_FEATURE_COLUMNS_H2_DEF,
    PREGAME_FEATURE_COLUMNS_H2_FULL,
    PREGAME_FEATURE_COLUMNS_H3_REPLACE,
    PREGAME_FEATURE_COLUMNS_H3_SPLIT,
    PREGAME_FEATURE_COLUMNS_H4_CLEAN,
    PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR,
    PREGAME_FEATURE_COLUMNS_H5_QB,
    PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED,
    QB_RATING_COLUMNS,
    REGRESSED_EPA_COLUMNS,
    SEASON_BLEND_K,
    SEASON_DEFENSIVE_EPA_COLUMNS,
    SEASON_SHRINK_FACTOR,
    build_pregame_features,
    current_season_blend_weight,
    feature_columns_for,
    prepare_training_data,
    regress_epa_feature,
    shrink_toward_mean,
    split_by_season,
)


def _same_value(left: object, right: object) -> bool:
    """True when both values are missing or they match within float tolerance."""
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == pytest.approx(right))


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
        assert FEATURE_SET_NAMES == (
            "current",
            "with_games_played",
            "diff",
            "diff_only",
            "h2_def",
            "h2_full",
            "h3_split",
            "h3_replace",
            "h4_clean",
            "h4_clean_sr",
            "h5_qb",
        )
        assert feature_columns_for("current") == PREGAME_FEATURE_COLUMNS
        assert feature_columns_for("with_games_played") == PREGAME_FEATURE_COLUMNS_WITH_GAMES_PLAYED
        assert feature_columns_for(FEATURE_SET_DIFF) == PREGAME_FEATURE_COLUMNS_DIFF
        assert feature_columns_for(FEATURE_SET_DIFF_ONLY) == DIFF_FEATURE_COLUMNS
        assert feature_columns_for(FEATURE_SET_H2_DEF) == PREGAME_FEATURE_COLUMNS_H2_DEF
        assert feature_columns_for(FEATURE_SET_H2_FULL) == PREGAME_FEATURE_COLUMNS_H2_FULL
        assert (
            PREGAME_FEATURE_COLUMNS_DIFF[: len(PREGAME_FEATURE_COLUMNS)] == PREGAME_FEATURE_COLUMNS
        )
        assert PREGAME_FEATURE_COLUMNS_H2_DEF[: len(PREGAME_FEATURE_COLUMNS)] == (
            PREGAME_FEATURE_COLUMNS
        )
        assert PREGAME_FEATURE_COLUMNS_H2_DEF[len(PREGAME_FEATURE_COLUMNS) :] == (
            SEASON_DEFENSIVE_EPA_COLUMNS
        )
        assert PREGAME_FEATURE_COLUMNS_H2_FULL[: len(PREGAME_FEATURE_COLUMNS_H2_DEF)] == (
            PREGAME_FEATURE_COLUMNS_H2_DEF
        )
        assert PREGAME_FEATURE_COLUMNS_H2_FULL[len(PREGAME_FEATURE_COLUMNS_H2_DEF) :] == (
            REGRESSED_EPA_COLUMNS
        )
        assert set(SEASON_DEFENSIVE_EPA_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS)
        assert set(REGRESSED_EPA_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS)
        assert set(REGRESSED_EPA_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS_H2_DEF)
        assert "home_games_played" not in feature_columns_for(FEATURE_SET_DIFF)
        assert "home_games_played" not in feature_columns_for(FEATURE_SET_H2_FULL)
        assert "diff_recent_win_rate" not in feature_columns_for(FEATURE_SET_H2_FULL)
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

        h2_full, _h2_target = prepare_training_data(
            synthetic_games,
            features,
            feature_set=FEATURE_SET_H2_FULL,
        )
        h2_columns = [
            column for column in h2_full.columns if column not in ("game_id", "gameday", "season")
        ]
        assert h2_columns == list(PREGAME_FEATURE_COLUMNS_H2_FULL)
        assert feature_columns_for(FEATURE_SET_H3_SPLIT) == PREGAME_FEATURE_COLUMNS_H3_SPLIT
        assert feature_columns_for(FEATURE_SET_H3_REPLACE) == PREGAME_FEATURE_COLUMNS_H3_REPLACE
        assert PREGAME_FEATURE_COLUMNS_H3_SPLIT[: len(PREGAME_FEATURE_COLUMNS_H2_FULL)] == (
            PREGAME_FEATURE_COLUMNS_H2_FULL
        )
        assert PREGAME_FEATURE_COLUMNS_H3_SPLIT[len(PREGAME_FEATURE_COLUMNS_H2_FULL) :] == (
            PASS_RUSH_EPA_COLUMNS
        )
        kept_h2_columns: tuple[str, ...] = tuple(
            column
            for column in PREGAME_FEATURE_COLUMNS_H2_FULL
            if column not in H3_REPLACED_COMBINED_COLUMNS
        )
        assert PREGAME_FEATURE_COLUMNS_H3_REPLACE[: len(kept_h2_columns)] == kept_h2_columns
        assert set(H3_REPLACED_COMBINED_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS_H3_REPLACE)
        assert set(PASS_RUSH_EPA_COLUMNS) <= set(PREGAME_FEATURE_COLUMNS_H3_SPLIT)
        assert set(PASS_RUSH_EPA_COLUMNS) <= set(PREGAME_FEATURE_COLUMNS_H3_REPLACE)
        assert set(PASS_RUSH_EPA_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS)
        assert set(PASS_RUSH_EPA_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS_H2_FULL)
        assert "home_recent_offensive_epa_per_play" not in feature_columns_for(
            FEATURE_SET_H3_REPLACE
        )
        assert "home_regressed_season_offensive_epa_per_play" not in feature_columns_for(
            FEATURE_SET_H3_REPLACE
        )
        assert "home_regressed_recent_offensive_epa_per_play" in feature_columns_for(
            FEATURE_SET_H3_REPLACE
        )
        assert "home_recent_win_rate" in feature_columns_for(FEATURE_SET_H3_REPLACE)
        assert "home_season_defensive_epa_per_play" in feature_columns_for(FEATURE_SET_H3_REPLACE)
        assert "home_games_played" not in feature_columns_for(FEATURE_SET_H3_SPLIT)

        h3_split, _h3_target = prepare_training_data(
            synthetic_games,
            features,
            feature_set=FEATURE_SET_H3_SPLIT,
        )
        h3_columns = [
            column for column in h3_split.columns if column not in ("game_id", "gameday", "season")
        ]
        assert h3_columns == list(PREGAME_FEATURE_COLUMNS_H3_SPLIT)
        assert feature_columns_for(FEATURE_SET_H4_CLEAN) == PREGAME_FEATURE_COLUMNS_H4_CLEAN
        assert feature_columns_for(FEATURE_SET_H4_CLEAN_SR) == PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR
        assert len(PREGAME_FEATURE_COLUMNS_H4_CLEAN) == len(PREGAME_FEATURE_COLUMNS_H2_FULL)
        assert PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR[: len(PREGAME_FEATURE_COLUMNS_H4_CLEAN)] == (
            PREGAME_FEATURE_COLUMNS_H4_CLEAN
        )
        assert PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR[len(PREGAME_FEATURE_COLUMNS_H4_CLEAN) :] == (
            CLEAN_SUCCESS_RATE_COLUMNS
        )
        for dirty, clean in zip(
            PREGAME_FEATURE_COLUMNS_H2_FULL,
            PREGAME_FEATURE_COLUMNS_H4_CLEAN,
            strict=True,
        ):
            if dirty in H2_CONTEXT_COLUMNS:
                assert clean == dirty
            else:
                expected = dirty.replace("_offensive_", "_clean_offensive_").replace(
                    "_defensive_", "_clean_defensive_"
                )
                assert clean == expected
        assert set(H2_CONTEXT_COLUMNS) <= set(PREGAME_FEATURE_COLUMNS_H4_CLEAN)
        assert "home_recent_offensive_epa_per_play" not in PREGAME_FEATURE_COLUMNS_H4_CLEAN
        assert "home_recent_clean_offensive_epa_per_play" in PREGAME_FEATURE_COLUMNS_H4_CLEAN
        assert "home_regressed_season_clean_defensive_epa_per_play" in (
            PREGAME_FEATURE_COLUMNS_H4_CLEAN
        )
        assert "home_games_played" not in PREGAME_FEATURE_COLUMNS_H4_CLEAN_SR
        assert feature_columns_for(FEATURE_SET_H5_QB) == PREGAME_FEATURE_COLUMNS_H5_QB
        assert PREGAME_FEATURE_COLUMNS_H5_QB[: len(PREGAME_FEATURE_COLUMNS_H2_FULL)] == (
            PREGAME_FEATURE_COLUMNS_H2_FULL
        )
        assert PREGAME_FEATURE_COLUMNS_H5_QB[len(PREGAME_FEATURE_COLUMNS_H2_FULL) :] == (
            QB_RATING_COLUMNS
        )
        assert "home_qb_rating" in PREGAME_FEATURE_COLUMNS_H5_QB
        assert "home_starter_player_id" not in PREGAME_FEATURE_COLUMNS_H5_QB

    def test_unknown_feature_set_is_rejected(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """A typo in the feature set name fails before a model is fit."""
        features = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        with pytest.raises(ValueError, match="Unknown feature set"):
            prepare_training_data(synthetic_games, features, feature_set="both")


class TestSeasonRegressionMath:
    """Shrinkage and the current-season blend are pure functions."""

    def test_one_third_shrink_matches_the_elo_fraction(self) -> None:
        """Default shrinkage removes one third of the gap, the same fraction as Elo."""
        assert SEASON_SHRINK_FACTOR == pytest.approx(1.0 / 3.0)
        assert shrink_toward_mean(0.30, 0.0, 1.0 / 3.0) == pytest.approx(0.20)
        assert shrink_toward_mean(0.30, 0.0, 0.0) == pytest.approx(0.30)
        assert shrink_toward_mean(0.30, 0.12, 1.0) == pytest.approx(0.12)
        with pytest.raises(ValueError, match="shrink_factor"):
            shrink_toward_mean(0.30, 0.0, 1.5)
        with pytest.raises(ValueError, match="shrink_factor"):
            shrink_toward_mean(0.30, 0.0, -0.1)

    def test_blend_weight_is_n_over_n_plus_k(self) -> None:
        """Zero games stays on the prior. k games is half weight. k must be positive."""
        assert SEASON_BLEND_K == pytest.approx(4.0)
        assert current_season_blend_weight(0, 4.0) == pytest.approx(0.0)
        assert current_season_blend_weight(4, 4.0) == pytest.approx(0.5)
        assert current_season_blend_weight(8, 4.0) == pytest.approx(8.0 / 12.0)
        with pytest.raises(ValueError, match="n_games"):
            current_season_blend_weight(-1, 4.0)
        with pytest.raises(ValueError, match="blend_k"):
            current_season_blend_weight(1, 0.0)
        with pytest.raises(ValueError, match="blend_k"):
            regress_epa_feature(0.1, 1, 0.2, 0.0, blend_k=-1.0)

    def test_week_one_uses_the_shrunk_prior_and_later_weeks_blend(self) -> None:
        """n = 0 returns the shrunk prior. Later weeks move toward the current mean."""
        league_mean: float = 0.175
        prior_value: float = 0.45
        shrunk: float = shrink_toward_mean(prior_value, league_mean, 1.0 / 3.0)
        week_one = regress_epa_feature(
            float("nan"),
            0,
            prior_value,
            league_mean,
            shrink_factor=1.0 / 3.0,
            blend_k=4.0,
        )
        assert week_one["weight"] == pytest.approx(0.0)
        assert week_one["shrunk_prior"] == pytest.approx(shrunk)
        assert week_one["value"] == pytest.approx(shrunk)

        blended = regress_epa_feature(
            0.90,
            1,
            prior_value,
            league_mean,
            shrink_factor=1.0 / 3.0,
            blend_k=4.0,
        )
        assert blended["weight"] == pytest.approx(1.0 / 5.0)
        assert blended["value"] == pytest.approx((1.0 / 5.0) * 0.90 + (4.0 / 5.0) * shrunk)

        no_prior = regress_epa_feature(0.90, 2, float("nan"), league_mean, blend_k=4.0)
        assert no_prior["weight"] == pytest.approx(2.0 / 6.0)
        assert pd.isna(no_prior["shrunk_prior"])
        assert no_prior["value"] == pytest.approx(0.90)

        cold_start = regress_epa_feature(float("nan"), 0, float("nan"), float("nan"))
        assert pd.isna(cold_start["value"])
        assert cold_start["weight"] == pytest.approx(0.0)


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


class TestCleanEpaFeatureSets:
    """h4 reads the clean columns and leaves the h2 matrix unchanged."""

    def test_identical_clean_inputs_match_h2_and_a_partial_pair_is_refused(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_team_game_epa: pd.DataFrame,
    ) -> None:
        """Same clean and original rates produce the same EPA features."""
        baseline = build_pregame_features(synthetic_games, synthetic_team_game_epa)
        assert pd.isna(
            baseline.loc[baseline["game_id"] == "G5", "home_recent_clean_offensive_epa_per_play"]
        ).all()
        copied = synthetic_team_game_epa.copy()
        copied["avg_clean_offensive_epa"] = copied["avg_offensive_epa"]
        copied["avg_clean_defensive_epa_allowed"] = copied["avg_defensive_epa_allowed"]
        copied["clean_success_rate"] = 0.55
        copied["clean_defensive_success_rate"] = 0.45
        matched = build_pregame_features(synthetic_games, copied)
        row = matched.loc[matched["game_id"] == "G5"].iloc[0]
        baseline_row = baseline.loc[baseline["game_id"] == "G5"].iloc[0]
        for dirty, clean in zip(
            PREGAME_FEATURE_COLUMNS_H2_FULL,
            PREGAME_FEATURE_COLUMNS_H4_CLEAN,
            strict=True,
        ):
            assert _same_value(row[clean], row[dirty])
            assert _same_value(row[dirty], baseline_row[dirty])
        assert row["home_recent_clean_offensive_success_rate"] == pytest.approx(0.55)
        assert row["away_recent_clean_defensive_success_rate"] == pytest.approx(0.45)

        offensive_only = copied.drop(columns=["avg_clean_defensive_epa_allowed"])
        with pytest.raises(ValueError, match="clean EPA"):
            build_pregame_features(synthetic_games, offensive_only)
        success_only = copied.drop(columns=["clean_defensive_success_rate"])
        with pytest.raises(ValueError, match="clean success"):
            build_pregame_features(synthetic_games, success_only)
