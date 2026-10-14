"""Leakage tests for pregame features.

Every feature for game G must match the feature built from games whose
gameday is strictly before G. Synthetic frames only. No warehouse.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pregame import PREGAME_FEATURE_COLUMNS, build_pregame_features

LOOKBACK_GAMES: int = 2


def _games_and_epa() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two seasons, a shared gameday, and a week-1 game with prior history.

    Dates are unique except G5 and G8, which share 2024-09-08 so a same-day
    game cannot leak into the other. EPA values are distinct so a window
    that is too wide, too short, or open on the right fails the expected mean.
    """
    games = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G7"],
            "season": [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
            "gameday": pd.to_datetime(
                [
                    "2023-09-10",
                    "2023-09-17",
                    "2023-10-01",
                    "2023-10-08",
                    "2024-09-08",
                    "2024-09-08",
                    "2024-09-08",
                    "2024-09-15",
                ]
            ),
            "home_team_franchise": ["A", "A", "A", "B", "A", "E", "A", "A"],
            "away_team_franchise": ["B", "C", "D", "A", "B", "F", "D", "C"],
            "home_score": [20, 21, 10, 30, 40, 3, 14, 17],
            "away_score": [10, 7, 13, 7, 10, 6, 7, 14],
            "home_won": [1, 1, 0, 1, 1, 0, 1, 1],
            "home_rest": [7, 7, 7, 7, 100, 7, 100, 7],
            "away_rest": [7, 7, 7, 7, np.nan, 7, 7, 7],
        }
    )
    offense = {
        ("G1", "A"): 0.10,
        ("G1", "B"): 0.50,
        ("G2", "A"): 0.30,
        ("G2", "C"): 0.70,
        ("G3", "A"): 0.40,
        ("G3", "D"): 0.20,
        ("G4", "B"): 0.60,
        ("G4", "A"): 0.80,
        ("G5", "A"): 9.00,
        ("G5", "B"): 9.00,
        ("G6", "E"): 8.00,
        ("G6", "F"): 8.00,
        ("G8", "A"): 77.00,
        ("G8", "D"): 77.00,
        ("G7", "A"): 1.00,
        ("G7", "C"): 1.00,
    }
    defense = {key: -value for key, value in offense.items()}
    defense[("G5", "A")] = 9.00
    defense[("G5", "B")] = 9.00
    defense[("G8", "A")] = 77.00
    defense[("G8", "D")] = 77.00
    defense[("G7", "A")] = 1.00
    defense[("G7", "C")] = 1.00

    epa_rows: list[dict[str, object]] = []
    for _, game in games.iterrows():
        for side, team_column in (
            ("home", "home_team_franchise"),
            ("away", "away_team_franchise"),
        ):
            team = game[team_column]
            key = (game["game_id"], team)
            epa_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "gameday": game["gameday"],
                    "team_franchise": team,
                    "avg_offensive_epa": offense[key],
                    "avg_defensive_epa_allowed": defense[key],
                    "side": side,
                }
            )
    epa = pd.DataFrame(epa_rows).drop(columns=["side"])
    return games, epa


def _feature_row(features: pd.DataFrame, game_id: str) -> pd.Series:
    """Return the single feature row for a game id."""
    matched = features.loc[features["game_id"] == game_id]
    assert len(matched) == 1
    return matched.iloc[0]


def _assert_equal(actual: object, expected: float, column: str) -> None:
    """Compare one feature value, treating NaN as a value."""
    if pd.isna(expected):
        assert pd.isna(actual), column
        return
    assert actual == pytest.approx(expected), column


def _assert_row(features: pd.DataFrame, game_id: str, expected: dict[str, float]) -> None:
    """Assert every expected column, and that the expectation covers all features."""
    row = _feature_row(features, game_id)
    assert set(expected) == set(PREGAME_FEATURE_COLUMNS)
    for column, expected_value in expected.items():
        _assert_equal(row[column], expected_value, f"{game_id}.{column}")


def _history_before(games: pd.DataFrame, epa: pd.DataFrame, game: pd.Series) -> pd.DataFrame:
    """Rebuild features using only this game plus rows strictly before its gameday."""
    earlier_games = games.loc[games["gameday"] < game["gameday"]]
    current_game = games.loc[games["game_id"] == game["game_id"]]
    truncated_games = pd.concat([earlier_games, current_game], ignore_index=True)
    truncated_epa = epa.loc[epa["gameday"] < game["gameday"]]
    return build_pregame_features(
        truncated_games,
        truncated_epa,
        lookback_games=LOOKBACK_GAMES,
    )


class TestPregameLeakage:
    """Prior-game rule for every pregame column."""

    def test_output_columns_match_the_training_matrix(self) -> None:
        """The builder's columns are exactly the training features plus game_id."""
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)

        assert list(features.columns) == ["game_id", *PREGAME_FEATURE_COLUMNS]
        assert len(features) == len(games)

    def test_each_feature_matches_history_truncated_before_the_game(self) -> None:
        """For every game, every feature equals the truncated-history rebuild."""
        games, epa = _games_and_epa()
        full = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)

        for _, game in games.iterrows():
            truncated = _history_before(games, epa, game)
            full_row = _feature_row(full, game["game_id"])
            truncated_row = _feature_row(truncated, game["game_id"])
            for column in PREGAME_FEATURE_COLUMNS:
                _assert_equal(
                    full_row[column],
                    truncated_row[column],
                    f"{game['game_id']}.{column}",
                )

    def test_current_and_later_outcomes_do_not_change_features(self) -> None:
        """Poisoning this game and every game on or after its date leaves features put."""
        games, epa = _games_and_epa()
        baseline = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)

        for _, game in games.iterrows():
            poisoned_games = games.copy()
            poisoned_epa = epa.copy()
            on_or_after = poisoned_games["gameday"] >= game["gameday"]
            poisoned_games.loc[on_or_after, "home_score"] = 99
            poisoned_games.loc[on_or_after, "away_score"] = 0
            poisoned_games.loc[on_or_after, "home_won"] = 0
            epa_on_or_after = poisoned_epa["gameday"] >= game["gameday"]
            poisoned_epa.loc[epa_on_or_after, "avg_offensive_epa"] = 99.0
            poisoned_epa.loc[epa_on_or_after, "avg_defensive_epa_allowed"] = 99.0

            poisoned = build_pregame_features(
                poisoned_games,
                poisoned_epa,
                lookback_games=LOOKBACK_GAMES,
            )
            baseline_row = _feature_row(baseline, game["game_id"])
            poisoned_row = _feature_row(poisoned, game["game_id"])
            for column in PREGAME_FEATURE_COLUMNS:
                _assert_equal(
                    poisoned_row[column],
                    baseline_row[column],
                    f"{game['game_id']}.{column}",
                )

    def test_first_game_in_the_frame_has_no_history(self) -> None:
        """The earliest game has empty rolling and season features and zero games played."""
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)
        missing = float("nan")
        _assert_row(
            features,
            "G1",
            {
                "home_recent_offensive_epa_per_play": missing,
                "home_recent_defensive_epa_per_play": missing,
                "away_recent_offensive_epa_per_play": missing,
                "away_recent_defensive_epa_per_play": missing,
                "home_recent_win_rate": missing,
                "away_recent_win_rate": missing,
                "home_rest_days": 7,
                "away_rest_days": 7,
                "home_season_offensive_epa_per_play": missing,
                "away_season_offensive_epa_per_play": missing,
                "home_games_played": 0,
                "away_games_played": 0,
            },
        )

    def test_first_game_of_a_season_keeps_prior_seasons_out_of_season_epa(self) -> None:
        """Week 1 uses prior-season games for the rolling window and not for season EPA.

        Team A has four 2023 games. Lookback is 2, so the 2024 opener uses the
        last two of those games (0.40 and 0.80), not the older two and not the
        opener itself. Season EPA is missing. games_played still counts all four.
        """
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)
        missing = float("nan")
        _assert_row(
            features,
            "G5",
            {
                "home_recent_offensive_epa_per_play": 0.60,
                "home_recent_defensive_epa_per_play": -0.60,
                "away_recent_offensive_epa_per_play": 0.55,
                "away_recent_defensive_epa_per_play": -0.55,
                "home_recent_win_rate": 0.0,
                "away_recent_win_rate": 0.5,
                "home_rest_days": 100,
                "away_rest_days": 7,
                "home_season_offensive_epa_per_play": missing,
                "away_season_offensive_epa_per_play": missing,
                "home_games_played": 4,
                "away_games_played": 2,
            },
        )

    def test_same_gameday_games_do_not_see_each_other(self) -> None:
        """Two games on 2024-09-08 for team A share the same prior window."""
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)
        opener = _feature_row(features, "G5")
        same_day = _feature_row(features, "G8")

        for column in PREGAME_FEATURE_COLUMNS:
            if column == "away_recent_offensive_epa_per_play":
                continue
            if column == "away_recent_defensive_epa_per_play":
                continue
            if column == "away_recent_win_rate":
                continue
            if column == "away_rest_days":
                continue
            if column == "away_season_offensive_epa_per_play":
                continue
            if column == "away_games_played":
                continue
            _assert_equal(same_day[column], opener[column], column)

        assert same_day["home_games_played"] == 4
        assert same_day["home_recent_offensive_epa_per_play"] == pytest.approx(0.60)

    def test_first_game_for_a_new_team_is_empty_on_a_shared_gameday(self) -> None:
        """A team with no prior games stays empty even when other games share the date."""
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)
        missing = float("nan")
        _assert_row(
            features,
            "G6",
            {
                "home_recent_offensive_epa_per_play": missing,
                "home_recent_defensive_epa_per_play": missing,
                "away_recent_offensive_epa_per_play": missing,
                "away_recent_defensive_epa_per_play": missing,
                "home_recent_win_rate": missing,
                "away_recent_win_rate": missing,
                "home_rest_days": 7,
                "away_rest_days": 7,
                "home_season_offensive_epa_per_play": missing,
                "away_season_offensive_epa_per_play": missing,
                "home_games_played": 0,
                "away_games_played": 0,
            },
        )

    def test_rolling_window_at_the_next_week_includes_only_the_last_n_prior_games(
        self,
    ) -> None:
        """The game after the opener uses the lookback window, which now crosses the season.

        Season EPA for that game is the mean of the two same-day openers only.
        It does not average in 2023, and it does not include this game's own EPA.
        """
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)
        _assert_row(
            features,
            "G7",
            {
                "home_recent_offensive_epa_per_play": 43.0,
                "home_recent_defensive_epa_per_play": 43.0,
                "away_recent_offensive_epa_per_play": 0.70,
                "away_recent_defensive_epa_per_play": -0.70,
                "home_recent_win_rate": 1.0,
                "away_recent_win_rate": 0.0,
                "home_rest_days": 7,
                "away_rest_days": 7,
                "home_season_offensive_epa_per_play": 43.0,
                "away_season_offensive_epa_per_play": float("nan"),
                "home_games_played": 6,
                "away_games_played": 1,
            },
        )
