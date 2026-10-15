"""Leakage tests for pregame features.

Every feature for game G must match the feature built from games whose
gameday is strictly before G. Synthetic frames only. No warehouse.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pregame import (
    DIFF_FEATURE_COLUMNS,
    DIFFERENTIAL_FEATURES,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_MEAN_RATING,
    ELO_SCALE,
    ELO_SEASON_REVERSION,
    GAMES_PLAYED_COLUMNS,
    PASS_RUSH_EPA_COLUMNS,
    PREGAME_COMPUTED_COLUMNS,
    PREGAME_FEATURE_COLUMNS,
    REGRESSED_EPA_COLUMNS,
    SEASON_BLEND_K,
    SEASON_DEFENSIVE_EPA_COLUMNS,
    SEASON_SHRINK_FACTOR,
    build_pregame_features,
    elo_home_probabilities,
    regress_epa_feature,
)

CHECKED_FEATURE_COLUMNS: tuple[str, ...] = (
    PREGAME_COMPUTED_COLUMNS
    + DIFF_FEATURE_COLUMNS
    + SEASON_DEFENSIVE_EPA_COLUMNS
    + REGRESSED_EPA_COLUMNS
    + PASS_RUSH_EPA_COLUMNS
)

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
    assert set(expected) == set(PREGAME_COMPUTED_COLUMNS)
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

        assert list(features.columns) == ["game_id", *CHECKED_FEATURE_COLUMNS]
        assert set(PREGAME_FEATURE_COLUMNS).isdisjoint(GAMES_PLAYED_COLUMNS)
        assert set(PREGAME_FEATURE_COLUMNS).isdisjoint(DIFF_FEATURE_COLUMNS)
        assert len(features) == len(games)

    def test_each_feature_matches_history_truncated_before_the_game(self) -> None:
        """For every game, every feature equals the truncated-history rebuild."""
        games, epa = _games_and_epa()
        full = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)

        for _, game in games.iterrows():
            truncated = _history_before(games, epa, game)
            full_row = _feature_row(full, game["game_id"])
            truncated_row = _feature_row(truncated, game["game_id"])
            for column in CHECKED_FEATURE_COLUMNS:
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
            for column in CHECKED_FEATURE_COLUMNS:
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

        for column in PREGAME_COMPUTED_COLUMNS:
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

    def test_differentials_are_home_minus_away_from_prior_games_only(self) -> None:
        """Each gap is home minus away, and a missing side stays missing.

        G5 is the 2024 opener. Home recent offense (0.60) beats away (0.55),
        so that difference is positive. Home recent win rate (0.0) is below
        away (0.5), so that difference is negative. Defensive EPA is allowed
        EPA and is not flipped: home -0.60 minus away -0.55 is -0.05.
        Season EPA is missing on both sides. Rest is 100 minus the away
        default of 7. The opener's own EPA is not in the gap.
        """
        games, epa = _games_and_epa()
        features = build_pregame_features(games, epa, lookback_games=LOOKBACK_GAMES)

        for _, row in features.iterrows():
            for item in DIFFERENTIAL_FEATURES:
                actual = row[item.diff_column]
                home_value = row[item.home_column]
                away_value = row[item.away_column]
                if pd.isna(home_value) or pd.isna(away_value):
                    assert pd.isna(actual), item.diff_column
                    continue
                assert actual == pytest.approx(home_value - away_value), item.diff_column

        opener = _feature_row(features, "G5")
        assert opener["diff_recent_offensive_epa_per_play"] == pytest.approx(0.05)
        assert opener["diff_recent_offensive_epa_per_play"] > 0
        assert opener["diff_recent_defensive_epa_per_play"] == pytest.approx(-0.05)
        assert opener["diff_recent_defensive_epa_per_play"] < 0
        assert opener["diff_recent_win_rate"] == pytest.approx(-0.5)
        assert opener["diff_recent_win_rate"] < 0
        assert pd.isna(opener["diff_season_offensive_epa_per_play"])
        assert opener["diff_rest_days"] == pytest.approx(93)
        assert opener["diff_rest_days"] > 0

        first = _feature_row(features, "G1")
        assert pd.isna(first["diff_recent_offensive_epa_per_play"])
        assert pd.isna(first["diff_recent_win_rate"])
        assert first["diff_rest_days"] == pytest.approx(0)

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


def _regression_frame(include_old_season: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prior season, week 1, the next week, a same-day game, and a future decoy.

    The 2023-09-20 row is labeled season 2023 but dated after week 1, so it is
    not a completed prior game for that opener. The optional 2022 game is two
    seasons back and must not enter the 2024 carry-over.
    """
    rows: list[dict[str, object]] = []
    if include_old_season:
        rows.append(
            {
                "game_id": "OLD",
                "season": 2022,
                "gameday": "2022-09-11",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 10,
                "away_score": 7,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 5.0,
                "home_def": 5.0,
                "away_off": 5.0,
                "away_def": 5.0,
            }
        )
    rows.extend(
        [
            {
                "game_id": "P1",
                "season": 2023,
                "gameday": "2023-09-10",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 20,
                "away_score": 10,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 0.30,
                "home_def": 0.10,
                "away_off": -0.20,
                "away_def": 0.40,
            },
            {
                "game_id": "P2",
                "season": 2023,
                "gameday": "2023-09-17",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 24,
                "away_score": 17,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 0.60,
                "home_def": -0.20,
                "away_off": 0.00,
                "away_def": 0.20,
            },
            {
                "game_id": "W1",
                "season": 2024,
                "gameday": "2024-09-08",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 30,
                "away_score": 14,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 9.00,
                "home_def": 9.00,
                "away_off": 9.00,
                "away_def": 9.00,
            },
            {
                "game_id": "SAME",
                "season": 2024,
                "gameday": "2024-09-08",
                "home_team_franchise": "A",
                "away_team_franchise": "C",
                "home_score": 21,
                "away_score": 3,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 8.00,
                "home_def": 8.00,
                "away_off": 8.00,
                "away_def": 8.00,
            },
            {
                "game_id": "W2",
                "season": 2024,
                "gameday": "2024-09-15",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 17,
                "away_score": 14,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 1.00,
                "home_def": 1.00,
                "away_off": 1.00,
                "away_def": 1.00,
            },
            {
                "game_id": "LATE",
                "season": 2023,
                "gameday": "2024-09-20",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 3,
                "away_score": 0,
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
                "home_off": 99.0,
                "home_def": 99.0,
                "away_off": 99.0,
                "away_def": 99.0,
            },
        ]
    )
    games = pd.DataFrame(rows)
    games["gameday"] = pd.to_datetime(games["gameday"])
    epa_rows: list[dict[str, object]] = []
    for row in rows:
        for side in ("home", "away"):
            epa_rows.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "gameday": pd.Timestamp(str(row["gameday"])),
                    "team_franchise": row[f"{side}_team_franchise"],
                    "avg_offensive_epa": row[f"{side}_off"],
                    "avg_defensive_epa_allowed": row[f"{side}_def"],
                }
            )
    epa = pd.DataFrame(epa_rows)
    games = games.drop(columns=["home_off", "home_def", "away_off", "away_def"])
    return games, epa


class TestSeasonRegressionLeakage:
    """Prior-season carry-over uses only completed games from the previous season."""

    def test_week_one_is_the_shrunk_prior_and_ignores_the_current_game(self) -> None:
        """Raw season defense is missing. The regressed value does not use week 1 EPA."""
        games, epa = _regression_frame()
        features = build_pregame_features(games, epa, lookback_games=10)
        opener = _feature_row(features, "W1")
        same_day = _feature_row(features, "SAME")
        league_offense: float = 0.175
        prior_offense: float = 0.45
        shrunk_offense: float = league_offense + (2.0 / 3.0) * (prior_offense - league_offense)
        shrunk_defense: float = 0.125 + (2.0 / 3.0) * (-0.05 - 0.125)
        assert SEASON_SHRINK_FACTOR == pytest.approx(1.0 / 3.0)
        assert SEASON_BLEND_K == pytest.approx(4.0)
        assert pd.isna(opener["home_season_offensive_epa_per_play"])
        assert pd.isna(opener["home_season_defensive_epa_per_play"])
        assert pd.isna(opener["away_season_defensive_epa_per_play"])
        assert opener["home_regressed_season_offensive_epa_per_play"] == pytest.approx(
            shrunk_offense
        )
        assert opener["home_regressed_season_defensive_epa_per_play"] == pytest.approx(
            shrunk_defense
        )
        assert opener["home_regressed_recent_offensive_epa_per_play"] == pytest.approx(
            shrunk_offense
        )
        formula = regress_epa_feature(
            float("nan"),
            0,
            prior_offense,
            league_offense,
            shrink_factor=SEASON_SHRINK_FACTOR,
            blend_k=SEASON_BLEND_K,
        )
        assert opener["home_regressed_season_offensive_epa_per_play"] == pytest.approx(
            formula["value"]
        )
        for column in (
            "home_season_defensive_epa_per_play",
            "home_regressed_season_offensive_epa_per_play",
            "home_regressed_season_defensive_epa_per_play",
            "home_regressed_recent_offensive_epa_per_play",
            "home_regressed_recent_defensive_epa_per_play",
        ):
            _assert_equal(same_day[column], opener[column], column)

    def test_recent_carry_over_uses_the_prior_season_window_only(self) -> None:
        """Lookback 1 keeps the last prior-season game and not the one before it."""
        games, epa = _regression_frame()
        features = build_pregame_features(games, epa, lookback_games=1)
        opener = _feature_row(features, "W1")
        league_offense: float = 0.175
        last_game_offense: float = 0.60
        season_offense: float = 0.45
        shrunk_recent: float = league_offense + (2.0 / 3.0) * (last_game_offense - league_offense)
        shrunk_season: float = league_offense + (2.0 / 3.0) * (season_offense - league_offense)
        assert opener["home_regressed_recent_offensive_epa_per_play"] == pytest.approx(
            shrunk_recent
        )
        assert opener["home_regressed_season_offensive_epa_per_play"] == pytest.approx(
            shrunk_season
        )
        assert shrunk_recent != pytest.approx(shrunk_season)

    def test_later_weeks_blend_current_season_games_with_the_shrunk_prior(self) -> None:
        """Two week-1 games are in the current mean. Week 2's own EPA is not."""
        games, epa = _regression_frame()
        features = build_pregame_features(games, epa, lookback_games=10)
        second = _feature_row(features, "W2")
        # W1 and SAME are both before W2, so n = 2 for team A, not 1.
        # SAME is the same franchise on week 1. Both count.
        current_offense: float = (9.0 + 8.0) / 2.0
        shrunk: float = 0.175 + (2.0 / 3.0) * (0.45 - 0.175)
        weight: float = 2.0 / (2.0 + SEASON_BLEND_K)
        expected: float = weight * current_offense + (1.0 - weight) * shrunk
        assert second["home_season_offensive_epa_per_play"] == pytest.approx(current_offense)
        assert second["home_season_defensive_epa_per_play"] == pytest.approx((9.0 + 8.0) / 2.0)
        assert second["home_regressed_season_offensive_epa_per_play"] == pytest.approx(expected)
        assert second["home_regressed_season_offensive_epa_per_play"] != pytest.approx(1.0)

    def test_only_the_previous_season_and_only_games_already_played_are_carried(
        self,
    ) -> None:
        """A two-seasons-back game and a not-yet-played prior-season row do not move week 1."""
        games, epa = _regression_frame(include_old_season=False)
        with_old_games, with_old_epa = _regression_frame(include_old_season=True)
        baseline = build_pregame_features(games, epa, lookback_games=10)
        with_old = build_pregame_features(with_old_games, with_old_epa, lookback_games=10)
        baseline_row = _feature_row(baseline, "W1")
        old_row = _feature_row(with_old, "W1")
        for column in SEASON_DEFENSIVE_EPA_COLUMNS + REGRESSED_EPA_COLUMNS:
            _assert_equal(old_row[column], baseline_row[column], column)

        poisoned_epa = epa.copy()
        late = poisoned_epa["game_id"] == "LATE"
        poisoned_epa.loc[late, "avg_offensive_epa"] = -99.0
        poisoned_epa.loc[late, "avg_defensive_epa_allowed"] = -99.0
        poisoned = build_pregame_features(games, poisoned_epa, lookback_games=10)
        poisoned_row = _feature_row(poisoned, "W1")
        second = _feature_row(poisoned, "W2")
        for column in REGRESSED_EPA_COLUMNS:
            _assert_equal(poisoned_row[column], baseline_row[column], column)
            _assert_equal(
                second[column],
                _feature_row(baseline, "W2")[column],
                f"W2.{column}",
            )

        changed_epa = epa.copy()
        prior = changed_epa["game_id"] == "P2"
        home_prior = prior & (changed_epa["team_franchise"] == "A")
        changed_epa.loc[home_prior, "avg_offensive_epa"] = 0.0
        changed = build_pregame_features(games, changed_epa, lookback_games=10)
        changed_row = _feature_row(changed, "W1")
        assert changed_row["home_regressed_season_offensive_epa_per_play"] != pytest.approx(
            baseline_row["home_regressed_season_offensive_epa_per_play"]
        )
        assert pd.isna(changed_row["home_season_offensive_epa_per_play"])


def _pass_rush_row(
    game_id: str,
    season: int,
    gameday: str,
    team: str,
    opponent: str,
    off_pass: float,
    off_rush: float,
    pass_plays: int = 10,
    rush_plays: int = 10,
) -> dict[str, object]:
    """One team-game with pass and rush totals that divide back to the rates."""
    return {
        "game_id": game_id,
        "season": season,
        "gameday": pd.Timestamp(gameday),
        "team_franchise": team,
        "opponent_franchise": opponent,
        "avg_offensive_epa": 0.0,
        "avg_defensive_epa_allowed": 0.0,
        "pass_plays": pass_plays,
        "rush_plays": rush_plays,
        "pass_epa": float(off_pass) * pass_plays if pass_plays else 0.0,
        "rush_epa": float(off_rush) * rush_plays if rush_plays else 0.0,
    }


def _pass_rush_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two prior-season games, a same-day week 1, and a week 2 that must not leak."""
    games = pd.DataFrame(
        {
            "game_id": ["P1", "P2", "W1", "SAME", "W2"],
            "season": [2023, 2023, 2024, 2024, 2024],
            "gameday": pd.to_datetime(
                ["2023-09-10", "2023-09-17", "2024-09-08", "2024-09-08", "2024-09-15"]
            ),
            "home_team_franchise": ["A", "A", "A", "A", "A"],
            "away_team_franchise": ["B", "C", "B", "D", "B"],
            "home_score": [21, 17, 24, 10, 28],
            "away_score": [14, 20, 7, 14, 21],
            "home_won": [1, 0, 1, 0, 1],
            "home_rest": [7, 7, 7, 7, 7],
            "away_rest": [7, 7, 7, 7, 7],
        }
    )
    rates: dict[tuple[str, str], tuple[float, float]] = {
        ("P1", "A"): (0.20, 0.10),
        ("P1", "B"): (0.50, -0.30),
        ("P2", "A"): (0.40, 0.00),
        ("P2", "C"): (0.10, 0.20),
        ("W1", "A"): (9.00, 9.00),
        ("W1", "B"): (7.00, 7.00),
        ("SAME", "A"): (8.00, 8.00),
        ("SAME", "D"): (6.00, 6.00),
        ("W2", "A"): (1.00, 1.00),
        ("W2", "B"): (1.00, 1.00),
    }
    opponents: dict[tuple[str, str], str] = {
        ("P1", "A"): "B",
        ("P1", "B"): "A",
        ("P2", "A"): "C",
        ("P2", "C"): "A",
        ("W1", "A"): "B",
        ("W1", "B"): "A",
        ("SAME", "A"): "D",
        ("SAME", "D"): "A",
        ("W2", "A"): "B",
        ("W2", "B"): "A",
    }
    epa_rows: list[dict[str, object]] = []
    for (game_id, team), (off_pass, off_rush) in rates.items():
        season = 2023 if game_id.startswith("P") else 2024
        gameday = {
            "P1": "2023-09-10",
            "P2": "2023-09-17",
            "W1": "2024-09-08",
            "SAME": "2024-09-08",
            "W2": "2024-09-15",
        }[game_id]
        epa_rows.append(
            _pass_rush_row(
                game_id,
                season,
                gameday,
                team,
                opponents[(game_id, team)],
                off_pass,
                off_rush,
            )
        )
    return games, pd.DataFrame(epa_rows)


class TestPassRushLeakage:
    """Pass and rush splits use only prior games, and defense is the opponent rate."""

    def test_week_one_uses_the_prior_window_and_the_shrunk_season(self) -> None:
        """Lookback 10 averages both prior games. The current game's 9.0 stays out.

        Defensive pass EPA is the opponent's offensive pass rate, not the
        team's own. The regressed-season value is the prior-season mean
        shrunk one third toward the 2023 league mean. Week 1 has no blend.
        """
        games, epa = _pass_rush_games()
        features = build_pregame_features(games, epa, lookback_games=10)
        opener = _feature_row(features, "W1")
        same_day = _feature_row(features, "SAME")
        league_pass: float = 0.30
        league_rush: float = 0.0
        prior_off_pass: float = 0.30
        prior_off_rush: float = 0.05
        prior_def_pass: float = 0.30
        prior_def_rush: float = -0.05
        shrunk_off_rush: float = league_rush + (2.0 / 3.0) * (prior_off_rush - league_rush)
        shrunk_def_rush: float = league_rush + (2.0 / 3.0) * (prior_def_rush - league_rush)
        assert opener["home_recent_offensive_pass_epa_per_play"] == pytest.approx(prior_off_pass)
        assert opener["home_recent_offensive_rush_epa_per_play"] == pytest.approx(prior_off_rush)
        assert opener["home_recent_defensive_pass_epa_per_play"] == pytest.approx(prior_def_pass)
        assert opener["home_recent_defensive_rush_epa_per_play"] == pytest.approx(prior_def_rush)
        assert opener["home_recent_offensive_rush_epa_per_play"] != pytest.approx(prior_def_rush)
        assert opener["home_regressed_season_offensive_pass_epa_per_play"] == pytest.approx(
            league_pass
        )
        assert opener["home_regressed_season_offensive_rush_epa_per_play"] == pytest.approx(
            shrunk_off_rush
        )
        assert opener["home_regressed_season_defensive_pass_epa_per_play"] == pytest.approx(
            league_pass
        )
        assert opener["home_regressed_season_defensive_rush_epa_per_play"] == pytest.approx(
            shrunk_def_rush
        )
        assert opener["home_regressed_season_offensive_pass_epa_per_play"] != pytest.approx(9.0)
        for column in PASS_RUSH_EPA_COLUMNS:
            if column.startswith("away_"):
                continue
            _assert_equal(same_day[column], opener[column], column)

    def test_recent_window_is_the_last_prior_game_only(self) -> None:
        """Lookback 1 keeps P2 and drops P1. The season mean still uses both."""
        games, epa = _pass_rush_games()
        features = build_pregame_features(games, epa, lookback_games=1)
        opener = _feature_row(features, "W1")
        assert opener["home_recent_offensive_pass_epa_per_play"] == pytest.approx(0.40)
        assert opener["home_recent_defensive_rush_epa_per_play"] == pytest.approx(0.20)
        assert opener["home_regressed_season_offensive_pass_epa_per_play"] == pytest.approx(0.30)
        assert opener["home_regressed_season_offensive_rush_epa_per_play"] == pytest.approx(
            (2.0 / 3.0) * 0.05
        )

    def test_later_week_blends_current_season_rates_and_ignores_itself(self) -> None:
        """W1 and SAME are the current season. W2's own rate of 1.0 is not."""
        games, epa = _pass_rush_games()
        features = build_pregame_features(games, epa, lookback_games=10)
        second = _feature_row(features, "W2")
        # Recent crosses seasons: P1, P2, W1, and SAME. W2 itself is excluded.
        recent_off_pass: float = (0.20 + 0.40 + 9.0 + 8.0) / 4.0
        recent_def_pass: float = (0.50 + 0.10 + 7.0 + 6.0) / 4.0
        current_off_pass: float = (9.0 + 8.0) / 2.0
        shrunk_off_pass: float = 0.30
        weight: float = 2.0 / (2.0 + SEASON_BLEND_K)
        expected: float = weight * current_off_pass + (1.0 - weight) * shrunk_off_pass
        assert second["home_recent_offensive_pass_epa_per_play"] == pytest.approx(recent_off_pass)
        assert second["home_recent_defensive_pass_epa_per_play"] == pytest.approx(recent_def_pass)
        assert second["home_regressed_season_offensive_pass_epa_per_play"] == pytest.approx(
            expected
        )
        assert second["home_regressed_season_offensive_pass_epa_per_play"] != pytest.approx(1.0)
        earlier_games = games.loc[games["gameday"] < pd.Timestamp("2024-09-15")]
        current_game = games.loc[games["game_id"] == "W2"]
        truncated = build_pregame_features(
            pd.concat([earlier_games, current_game], ignore_index=True),
            epa.loc[epa["gameday"] < pd.Timestamp("2024-09-15")],
            lookback_games=10,
        )
        truncated_row = _feature_row(truncated, "W2")
        for column in PASS_RUSH_EPA_COLUMNS:
            _assert_equal(truncated_row[column], second[column], column)

    def test_same_day_and_future_pass_rates_do_not_move_week_one(self) -> None:
        """Poisoning week 1, the same day, and week 2 leaves the opener put."""
        games, epa = _pass_rush_games()
        baseline = build_pregame_features(games, epa, lookback_games=10)
        poisoned = epa.copy()
        later = poisoned["gameday"] >= pd.Timestamp("2024-09-08")
        poisoned.loc[later, "pass_epa"] = 500.0
        poisoned.loc[later, "rush_epa"] = -500.0
        updated = build_pregame_features(games, poisoned, lookback_games=10)
        baseline_row = _feature_row(baseline, "W1")
        updated_row = _feature_row(updated, "W1")
        for column in PASS_RUSH_EPA_COLUMNS:
            _assert_equal(updated_row[column], baseline_row[column], column)

    def test_zero_pass_plays_stay_missing_and_a_partial_source_is_refused(self) -> None:
        """A zero denominator is not a zero rate. Some source columns and not all fail."""
        games = pd.DataFrame(
            {
                "game_id": ["Z1", "Z2"],
                "season": [2023, 2023],
                "gameday": pd.to_datetime(["2023-09-10", "2023-09-17"]),
                "home_team_franchise": ["A", "A"],
                "away_team_franchise": ["B", "B"],
                "home_score": [10, 14],
                "away_score": [7, 10],
                "home_won": [1, 1],
                "home_rest": [7, 7],
                "away_rest": [7, 7],
            }
        )
        epa = pd.DataFrame(
            [
                _pass_rush_row("Z1", 2023, "2023-09-10", "A", "B", 0.0, 0.20, pass_plays=0),
                _pass_rush_row("Z1", 2023, "2023-09-10", "B", "A", 0.50, 0.10),
                _pass_rush_row("Z2", 2023, "2023-09-17", "A", "B", 1.00, 1.00),
                _pass_rush_row("Z2", 2023, "2023-09-17", "B", "A", 1.00, 1.00),
            ]
        )
        features = build_pregame_features(games, epa, lookback_games=10)
        second = _feature_row(features, "Z2")
        assert pd.isna(second["home_recent_offensive_pass_epa_per_play"])
        assert second["home_recent_offensive_rush_epa_per_play"] == pytest.approx(0.20)
        assert second["home_recent_defensive_pass_epa_per_play"] == pytest.approx(0.50)
        partial = epa.drop(columns=["pass_plays"])
        with pytest.raises(ValueError, match="missing pass/rush columns"):
            build_pregame_features(games, partial, lookback_games=10)


def _elo_games() -> pd.DataFrame:
    """A short schedule with a shared gameday, a tie, and a second season."""
    return pd.DataFrame(
        {
            "game_id": ["E1", "E2", "E3", "E4", "E5"],
            "season": [2023, 2023, 2023, 2023, 2024],
            "gameday": pd.to_datetime(
                ["2023-09-10", "2023-09-17", "2023-09-17", "2023-09-24", "2024-09-08"]
            ),
            "home_team_franchise": ["A", "A", "B", "A", "A"],
            "away_team_franchise": ["B", "C", "D", "B", "B"],
            "home_score": [20, 10, 14, 21, 17],
            "away_score": [10, 10, 7, 24, 20],
            "home_won": [1, 0, 1, 0, 0],
        }
    )


def _elo_map(games: pd.DataFrame) -> dict[str, float]:
    """Home probabilities keyed by game id."""
    fitted = elo_home_probabilities(games)
    probabilities = fitted["probabilities"]
    return dict(zip(games["game_id"], probabilities.tolist(), strict=True))


class TestEloLeakage:
    """Elo for game G uses only games with an earlier gameday."""

    def test_probability_matches_history_truncated_before_the_gameday(self) -> None:
        """Each game matches a rebuild that drops same-day and later games."""
        games = _elo_games()
        full = _elo_map(games)
        for _, game in games.iterrows():
            earlier = games.loc[games["gameday"] < game["gameday"]]
            current = games.loc[games["game_id"] == game["game_id"]]
            truncated = pd.concat([earlier, current], ignore_index=True)
            rebuilt = _elo_map(truncated)
            assert rebuilt[game["game_id"]] == pytest.approx(full[game["game_id"]])

    def test_same_day_results_do_not_change_each_other(self) -> None:
        """Flipping the other game on 2023-09-17 leaves both probabilities put."""
        games = _elo_games()
        baseline = _elo_map(games)
        poisoned = games.copy()
        poisoned.loc[poisoned["game_id"] == "E2", "home_won"] = 1
        poisoned.loc[poisoned["game_id"] == "E2", "home_score"] = 30
        poisoned.loc[poisoned["game_id"] == "E2", "away_score"] = 0
        updated = _elo_map(poisoned)
        assert updated["E2"] == pytest.approx(baseline["E2"])
        assert updated["E3"] == pytest.approx(baseline["E3"])
        assert updated["E4"] != pytest.approx(baseline["E4"])

    def test_current_and_later_results_do_not_change_the_current_probability(self) -> None:
        """Poisoning this game and every later game leaves this probability put."""
        games = _elo_games()
        baseline = _elo_map(games)
        for _, game in games.iterrows():
            poisoned = games.copy()
            on_or_after = poisoned["gameday"] >= game["gameday"]
            poisoned.loc[on_or_after, "home_won"] = 0
            poisoned.loc[on_or_after, "home_score"] = 0
            poisoned.loc[on_or_after, "away_score"] = 40
            updated = _elo_map(poisoned)
            assert updated[game["game_id"]] == pytest.approx(baseline[game["game_id"]])

    def test_opening_game_uses_the_mean_and_home_advantage(self) -> None:
        """Equal teams on the first game are separated only by home advantage."""
        games = _elo_games()
        opening = _elo_map(games)["E1"]
        expected = 1.0 / (1.0 + 10.0 ** (-(ELO_HOME_ADVANTAGE) / ELO_SCALE))
        assert opening == pytest.approx(expected)
        assert opening > 0.5
        assert ELO_K == 20.0
        assert ELO_MEAN_RATING == 1505.0
        assert ELO_SEASON_REVERSION == pytest.approx(1.0 / 3.0)

    def test_a_win_raises_the_next_home_probability_and_the_offseason_pulls_it_back(
        self,
    ) -> None:
        """The next-season opener is closer to the cold start than a same-season rematch."""
        same_season = pd.DataFrame(
            {
                "game_id": ["S1", "S2"],
                "season": [2023, 2023],
                "gameday": pd.to_datetime(["2023-09-10", "2023-09-17"]),
                "home_team_franchise": ["A", "A"],
                "away_team_franchise": ["B", "B"],
                "home_score": [24, 21],
                "away_score": [10, 17],
                "home_won": [1, 1],
            }
        )
        next_season = same_season.copy()
        next_season.loc[next_season["game_id"] == "S2", "season"] = 2024
        next_season.loc[next_season["game_id"] == "S2", "gameday"] = pd.Timestamp("2024-09-08")
        cold = _elo_map(same_season)["S1"]
        same = _elo_map(same_season)["S2"]
        later = _elo_map(next_season)["S2"]
        assert same > cold
        assert cold < later < same

    def test_a_tie_does_not_move_the_rating_as_a_home_loss(self) -> None:
        """Equal scores update as a draw even when home_won is stored as 0."""
        tied = pd.DataFrame(
            {
                "game_id": ["T1", "T2"],
                "season": [2023, 2023],
                "gameday": pd.to_datetime(["2023-09-10", "2023-09-17"]),
                "home_team_franchise": ["A", "A"],
                "away_team_franchise": ["B", "C"],
                "home_score": [17, 10],
                "away_score": [17, 7],
                "home_won": [0, 1],
            }
        )
        decided = tied.copy()
        decided.loc[decided["game_id"] == "T1", "away_score"] = 20
        tied_next = _elo_map(tied)["T2"]
        loss_next = _elo_map(decided)["T2"]
        assert tied_next > loss_next

    def test_a_missing_label_does_not_update_the_rating(self) -> None:
        """An unknown result still gets a probability and does not move the next game."""
        games = pd.DataFrame(
            {
                "game_id": ["U1", "U2"],
                "season": [2023, 2023],
                "gameday": pd.to_datetime(["2023-09-10", "2023-09-17"]),
                "home_team_franchise": ["A", "A"],
                "away_team_franchise": ["B", "B"],
                "home_won": [np.nan, 1],
            }
        )
        probabilities = _elo_map(games)
        cold = 1.0 / (1.0 + 10.0 ** (-(ELO_HOME_ADVANTAGE) / ELO_SCALE))
        assert probabilities["U1"] == pytest.approx(cold)
        assert probabilities["U2"] == pytest.approx(cold)

    def test_rejects_missing_columns_and_bad_settings(self) -> None:
        """Elo refuses a frame it cannot order and settings outside their range."""
        games = _elo_games().drop(columns=["home_won"])
        with pytest.raises(ValueError, match="missing columns"):
            elo_home_probabilities(games)
        with pytest.raises(ValueError, match="k must be positive"):
            elo_home_probabilities(_elo_games(), k=0)
        with pytest.raises(ValueError, match="scale must be positive"):
            elo_home_probabilities(_elo_games(), scale=0)
        with pytest.raises(ValueError, match="season_reversion"):
            elo_home_probabilities(_elo_games(), season_reversion=1.5)
