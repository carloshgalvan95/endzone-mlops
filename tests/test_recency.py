"""Preregistered recency columns and the 2016-2023 evaluation command.

Synthetic frames only. No warehouse, and no fitted log loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.pregame import (
    ELO_HOME_ADVANTAGE,
    ELO_POINTS_PER_SPREAD,
    PREGAME_FEATURE_COLUMNS_H2_FULL,
    PREGAME_FEATURE_COLUMNS_RECENCY_ELO,
    PREGAME_FEATURE_COLUMNS_RECENCY_EPA5,
    PREGAME_FEATURE_COLUMNS_RECENCY_MARGIN,
    PREGAME_FEATURE_COLUMNS_RECENCY_STREAK,
    build_pregame_features,
    elo_home_probabilities,
    feature_columns_for,
    prepare_training_data,
)
from endzone.features.recency import (
    RECENCY_CHALLENGER_FEATURE_SETS,
    RECENCY_ELO_COLUMNS,
    RECENCY_EPA5_COLUMNS,
    RECENCY_OUTPUT_COLUMNS,
    RECENCY_POINT_DIFF_COLUMNS,
    RECENCY_STREAK_COLUMNS,
    RECENCY_WINDOW_GAMES,
    REPLACED_RECENT_EPA_COLUMNS,
    signed_result_streak,
)
from endzone.modeling.evaluation import require_aligned_oof
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.models import (
    _imputation_preprocessor,
    feature_column_groups,
    fit_logistic_pipeline,
)
from endzone.modeling.train import recency_challenger_argv
from scripts.train_recency import main as train_recency_main
from scripts.train_recency import require_written_cohort

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _close(actual: object, expected: float, label: str) -> None:
    """Compare one value and treat NaN as a value."""
    if pd.isna(expected):
        assert pd.isna(actual), label
        return
    assert actual == pytest.approx(expected), label


def _games_from(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Add the rest columns the pregame builder requires."""
    frame: pd.DataFrame = pd.DataFrame(rows)
    frame["home_rest"] = 7
    frame["away_rest"] = 7
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    return frame


def _epa_from(games: pd.DataFrame, offense: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Team-game EPA. Defense is the negative of the listed offense unless overridden."""
    rows: list[dict[str, object]] = []
    for _, game in games.iterrows():
        for team in (game["home_team_franchise"], game["away_team_franchise"]):
            value: float = offense[(str(game["game_id"]), str(team))]
            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "gameday": game["gameday"],
                    "team_franchise": team,
                    "avg_offensive_epa": value,
                    "avg_defensive_epa_allowed": -value,
                }
            )
    return pd.DataFrame(rows)


def _slate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A tie, a week boundary, a same-day game, and a later season.

    ``TIE`` stores ``home_won`` 0, which is what the games mart does for an
    equal score. The streak has to use the scores.
    """
    games: pd.DataFrame = _games_from(
        [
            {
                "game_id": "S0",
                "season": 2023,
                "gameday": "2023-12-25",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 20,
                "away_score": 10,
                "home_won": 1,
            },
            {
                "game_id": "TIE",
                "season": 2023,
                "gameday": "2023-12-31",
                "home_team_franchise": "A",
                "away_team_franchise": "C",
                "home_score": 17,
                "away_score": 17,
                "home_won": 0,
            },
            {
                "game_id": "THU",
                "season": 2024,
                "gameday": "2024-09-05",
                "home_team_franchise": "A",
                "away_team_franchise": "D",
                "home_score": 27,
                "away_score": 7,
                "home_won": 1,
            },
            {
                "game_id": "SUN",
                "season": 2024,
                "gameday": "2024-09-08",
                "home_team_franchise": "A",
                "away_team_franchise": "E",
                "home_score": 30,
                "away_score": 0,
                "home_won": 1,
            },
            {
                "game_id": "SAME",
                "season": 2024,
                "gameday": "2024-09-08",
                "home_team_franchise": "F",
                "away_team_franchise": "G",
                "home_score": 40,
                "away_score": 0,
                "home_won": 1,
            },
            {
                "game_id": "MON",
                "season": 2024,
                "gameday": "2024-09-09",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 3,
                "away_score": 30,
                "home_won": 0,
            },
            {
                "game_id": "NEXT",
                "season": 2024,
                "gameday": "2024-09-15",
                "home_team_franchise": "A",
                "away_team_franchise": "C",
                "home_score": 14,
                "away_score": 10,
                "home_won": 1,
            },
            {
                "game_id": "Y25",
                "season": 2025,
                "gameday": "2025-09-07",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 0,
                "away_score": 50,
                "home_won": 0,
            },
        ]
    )
    offense: dict[tuple[str, str], float] = {}
    listed: dict[str, float] = {
        "S0": 0.10,
        "TIE": 0.30,
        "THU": 0.50,
        "SUN": 9.00,
        "SAME": 7.00,
        "MON": 0.70,
        "NEXT": 0.90,
        "Y25": 5.00,
    }
    for _, game in games.iterrows():
        game_id: str = str(game["game_id"])
        home_value: float = listed[game_id]
        offense[(game_id, str(game["home_team_franchise"]))] = home_value
        offense[(game_id, str(game["away_team_franchise"]))] = -home_value
    offense[("S0", "B")] = -0.20
    return games, _epa_from(games, offense)


def _value(features: pd.DataFrame, game_id: str, column: str) -> object:
    """Return one cell."""
    matched: pd.DataFrame = features.loc[features["game_id"] == game_id]
    assert len(matched) == 1
    return matched.iloc[0][column]


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ((), float("nan")),
        ((0,), 0.0),
        ((1, 0), 0.0),
        ((1, 1, 1), 3.0),
        ((-1, -1), -2.0),
        ((1, 1, -1), -1.0),
        ((-1, 0, 1, 1), 2.0),
        ((1, 0, 1), 1.0),
    ],
)
def test_signed_streak_breaks_on_a_tie_and_stays_missing_without_history(
    results: tuple[int, ...],
    expected: float,
) -> None:
    """A tie is 0. An empty history is missing, not a streak of zero."""
    _close(signed_result_streak(results), expected, str(results))


def test_feature_sets_are_four_single_changes() -> None:
    """Each challenger changes one block. h2_full itself is unchanged."""
    assert RECENCY_WINDOW_GAMES == 5
    assert feature_columns_for("h2_full") == PREGAME_FEATURE_COLUMNS_H2_FULL
    assert set(RECENCY_OUTPUT_COLUMNS).isdisjoint(PREGAME_FEATURE_COLUMNS_H2_FULL)
    replaced: dict[str, str] = dict(
        zip(REPLACED_RECENT_EPA_COLUMNS, RECENCY_EPA5_COLUMNS, strict=True)
    )
    assert feature_columns_for("recency_epa5") == tuple(
        replaced.get(column, column) for column in PREGAME_FEATURE_COLUMNS_H2_FULL
    )
    assert feature_columns_for("recency_epa5") == PREGAME_FEATURE_COLUMNS_RECENCY_EPA5
    assert set(REPLACED_RECENT_EPA_COLUMNS).isdisjoint(feature_columns_for("recency_epa5"))
    assert feature_columns_for("recency_margin") == PREGAME_FEATURE_COLUMNS_RECENCY_MARGIN
    assert feature_columns_for("recency_streak") == PREGAME_FEATURE_COLUMNS_RECENCY_STREAK
    assert feature_columns_for("recency_elo") == PREGAME_FEATURE_COLUMNS_RECENCY_ELO
    assert feature_columns_for("recency_margin") == (
        PREGAME_FEATURE_COLUMNS_H2_FULL + RECENCY_POINT_DIFF_COLUMNS
    )
    assert feature_columns_for("recency_streak") == (
        PREGAME_FEATURE_COLUMNS_H2_FULL + RECENCY_STREAK_COLUMNS
    )
    assert feature_columns_for("recency_elo") == (
        PREGAME_FEATURE_COLUMNS_H2_FULL + RECENCY_ELO_COLUMNS
    )
    for feature_set in RECENCY_CHALLENGER_FEATURE_SETS:
        columns: set[str] = set(feature_columns_for(feature_set))
        families: list[set[str]] = [
            set(RECENCY_EPA5_COLUMNS),
            set(RECENCY_POINT_DIFF_COLUMNS),
            set(RECENCY_STREAK_COLUMNS),
            set(RECENCY_ELO_COLUMNS),
        ]
        present: int = sum(1 for family in families if family <= columns)
        assert present == 1


def test_chronology_ties_season_boundary_and_missing_history() -> None:
    """Hand values for the pre-kickoff rules on one slate."""
    games, epa = _slate()
    features: pd.DataFrame = build_pregame_features(games, epa, lookback_games=10)
    expected: dict[str, dict[str, float]] = {
        "S0": {
            "home_recent5_offensive_epa_per_play": float("nan"),
            "home_recent5_defensive_epa_per_play": float("nan"),
            "home_recent_point_diff": float("nan"),
            "home_signed_streak": float("nan"),
            "elo_rating_diff": 0.0,
        },
        "THU": {
            "home_recent5_offensive_epa_per_play": 0.20,
            "home_recent5_defensive_epa_per_play": -0.20,
            "home_recent_point_diff": 5.0,
            "home_signed_streak": 0.0,
        },
        "SUN": {
            "home_recent5_offensive_epa_per_play": 0.30,
            "home_recent5_defensive_epa_per_play": -0.30,
            "home_recent_point_diff": 10.0,
            "home_signed_streak": 1.0,
            "away_recent5_offensive_epa_per_play": float("nan"),
            "away_signed_streak": float("nan"),
        },
        "MON": {
            "home_recent5_offensive_epa_per_play": (0.10 + 0.30 + 0.50 + 9.00) / 4.0,
            "home_recent_point_diff": 15.0,
            "home_signed_streak": 2.0,
            "away_recent_point_diff": -10.0,
            "away_signed_streak": -1.0,
            "away_recent5_offensive_epa_per_play": -0.20,
        },
        "NEXT": {
            "home_recent5_offensive_epa_per_play": (0.10 + 0.30 + 0.50 + 9.00 + 0.70) / 5.0,
            "home_recent_point_diff": (10.0 + 0.0 + 20.0 + 30.0 - 27.0) / 5.0,
            "home_signed_streak": -1.0,
        },
    }
    for game_id, columns in expected.items():
        for column, value in columns.items():
            _close(_value(features, game_id, column), value, f"{game_id}.{column}")

    poisoned_games: pd.DataFrame = games.copy()
    poisoned_epa: pd.DataFrame = epa.copy()
    later: pd.Series = poisoned_games["gameday"] >= pd.Timestamp("2024-09-08")
    poisoned_games.loc[later, "home_score"] = 99
    poisoned_games.loc[later, "away_score"] = 0
    poisoned_games.loc[later, "home_won"] = 0
    poisoned_epa.loc[
        poisoned_epa["game_id"].isin(["SUN", "SAME", "MON", "NEXT", "Y25"]),
        ["avg_offensive_epa", "avg_defensive_epa_allowed"],
    ] = 99.0
    poisoned: pd.DataFrame = build_pregame_features(poisoned_games, poisoned_epa, lookback_games=10)
    for game_id in ("S0", "THU", "SUN"):
        for column in RECENCY_OUTPUT_COLUMNS:
            _close(
                _value(poisoned, game_id, column),
                _value(features, game_id, column),
                f"poisoned {game_id}.{column}",
            )


def test_five_game_window_does_not_follow_the_h2_lookback() -> None:
    """Six prior games: lookback 2 and lookback 10 disagree, and the 5-game mean does not."""
    rows: list[dict[str, object]] = []
    offense: dict[tuple[str, str], float] = {}
    for index in range(7):
        game_id: str = f"G{index}"
        rows.append(
            {
                "game_id": game_id,
                "season": 2022,
                "gameday": str((pd.Timestamp("2022-09-01") + pd.Timedelta(days=7 * index)).date()),
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 10 + index,
                "away_score": 0,
                "home_won": 1,
            }
        )
        offense[(game_id, "A")] = 0.1 * (index + 1)
        offense[(game_id, "B")] = -0.1 * (index + 1)
    games: pd.DataFrame = _games_from(rows)
    epa: pd.DataFrame = _epa_from(games, offense)
    wide: pd.DataFrame = build_pregame_features(games, epa, lookback_games=10)
    short: pd.DataFrame = build_pregame_features(games, epa, lookback_games=2)
    _close(_value(wide, "G6", "home_recent5_offensive_epa_per_play"), 0.4, "wide recent5")
    _close(_value(short, "G6", "home_recent5_offensive_epa_per_play"), 0.4, "short recent5")
    _close(_value(wide, "G6", "home_recent_offensive_epa_per_play"), 0.35, "lookback 10")
    _close(_value(short, "G6", "home_recent_offensive_epa_per_play"), 0.55, "lookback 2")
    _close(_value(wide, "G6", "home_recent_point_diff"), 13.0, "point window")
    _close(_value(short, "G6", "home_recent_point_diff"), 13.0, "point window short")
    _close(_value(wide, "G6", "home_signed_streak"), 6.0, "streak")
    _close(_value(short, "G6", "home_signed_streak"), 6.0, "streak short")


def test_input_order_does_not_change_recency_columns() -> None:
    """A shuffled schedule matches the chronological build on game_id."""
    games, epa = _slate()
    baseline: pd.DataFrame = build_pregame_features(games, epa, lookback_games=10)
    shuffled_games: pd.DataFrame = games.sample(frac=1, random_state=1)
    shuffled_epa: pd.DataFrame = epa.sample(frac=1, random_state=2)
    shuffled: pd.DataFrame = build_pregame_features(shuffled_games, shuffled_epa, lookback_games=10)
    merged: pd.DataFrame = baseline.merge(shuffled, on="game_id", suffixes=("_base", "_shuf"))
    for column in RECENCY_OUTPUT_COLUMNS:
        for _, row in merged.iterrows():
            _close(row[f"{column}_shuf"], row[f"{column}_base"], f"{row['game_id']}.{column}")


def test_challenger_matrices_keep_games_with_missing_history() -> None:
    """A team with no prior game stays in every matrix."""
    games, epa = _slate()
    features: pd.DataFrame = build_pregame_features(games, epa, lookback_games=10)
    for feature_set in ("h2_full", *RECENCY_CHALLENGER_FEATURE_SETS):
        matrix, target = prepare_training_data(games, features, feature_set=feature_set)
        assert len(matrix) == len(games)
        assert len(target) == len(games)
        assert set(matrix["game_id"]) == set(games["game_id"])
    streak_matrix, _target = prepare_training_data(games, features, feature_set="recency_streak")
    opener: pd.DataFrame = streak_matrix.loc[streak_matrix["game_id"] == "S0"]
    assert len(opener) == 1
    assert pd.isna(opener.iloc[0]["home_signed_streak"])


def test_elo_difference_omits_home_advantage_and_same_day_results() -> None:
    """The feature is the rating gap. The point spread still adds home field."""
    games: pd.DataFrame = _games_from(
        [
            {
                "game_id": "D1",
                "season": 2022,
                "gameday": "2022-09-11",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_score": 21,
                "away_score": 7,
                "home_won": 1,
            },
            {
                "game_id": "D1B",
                "season": 2022,
                "gameday": "2022-09-11",
                "home_team_franchise": "C",
                "away_team_franchise": "D",
                "home_score": 7,
                "away_score": 21,
                "home_won": 0,
            },
            {
                "game_id": "D2",
                "season": 2022,
                "gameday": "2022-09-18",
                "home_team_franchise": "A",
                "away_team_franchise": "C",
                "home_score": 10,
                "away_score": 10,
                "home_won": 0,
            },
        ]
    )
    offense: dict[tuple[str, str], float] = {
        ("D1", "A"): 0.1,
        ("D1", "B"): -0.1,
        ("D1B", "C"): -0.1,
        ("D1B", "D"): 0.1,
        ("D2", "A"): 0.2,
        ("D2", "C"): -0.2,
    }
    features: pd.DataFrame = build_pregame_features(games, _epa_from(games, offense))
    fitted: dict[str, object] = elo_home_probabilities(games)
    home: np.ndarray = np.asarray(fitted["home_ratings"], dtype=float)
    away: np.ndarray = np.asarray(fitted["away_ratings"], dtype=float)
    spreads: np.ndarray = np.asarray(fitted["point_spreads"], dtype=float)
    assert home - away == pytest.approx(spreads * ELO_POINTS_PER_SPREAD - ELO_HOME_ADVANTAGE)
    _close(_value(features, "D1", "elo_rating_diff"), 0.0, "same-day home")
    _close(_value(features, "D1B", "elo_rating_diff"), 0.0, "same-day other")
    assert float(_value(features, "D2", "elo_rating_diff")) > 0.0


def test_h2_full_imputer_does_not_gain_a_form_step() -> None:
    """The frozen recipe's transformer list stays EPA, win rate, and rest."""
    groups: dict[str, list[str]] = feature_column_groups(list(PREGAME_FEATURE_COLUMNS_H2_FULL))
    assert groups["form"] == []
    preprocessor = _imputation_preprocessor(list(PREGAME_FEATURE_COLUMNS_H2_FULL))
    names: list[str] = [name for name, _step, _columns in preprocessor.transformers]
    assert names == ["epa_imputer", "win_rate_imputer", "other_passthrough"]
    form_groups: dict[str, list[str]] = feature_column_groups(
        [
            "home_recent_point_diff",
            "home_signed_streak",
            "elo_rating_diff",
            "home_recent5_offensive_epa_per_play",
            "home_rest_days",
        ]
    )
    assert form_groups["form"] == [
        "home_recent_point_diff",
        "home_signed_streak",
        "elo_rating_diff",
    ]
    assert form_groups["epa"] == ["home_recent5_offensive_epa_per_play"]
    assert form_groups["diff"] == []
    assert form_groups["other"] == ["home_rest_days"]


def test_form_scaler_fits_the_training_rows_only() -> None:
    """A missing streak becomes 0. An evaluation row does not enter the mean."""
    frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "gameday": pd.to_datetime(["2020-09-01", "2020-09-08"]),
            "season": [2020, 2020],
            "home_signed_streak": [2.0, np.nan],
            "home_rest_days": [7.0, 7.0],
        }
    )
    pipeline = fit_logistic_pipeline(frame, pd.Series([0, 1]))
    mean: np.ndarray = pipeline.named_steps["scaler"].mean_
    assert mean[0] == pytest.approx(1.0)
    assert mean[1] == pytest.approx(7.0)


def test_holdout_refuses_recency_sets_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025, so it cannot select these sets."""
    for feature_set in RECENCY_CHALLENGER_FEATURE_SETS:
        args: argparse.Namespace = argparse.Namespace(feature_set=feature_set)
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(args)


def test_recency_command_locks_the_development_window() -> None:
    """The argv, the YAML, and the reference recipe stay on 2016-2023."""
    argv: list[str] = recency_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "recency"
    assert argv[argv.index("--reference-feature-set") + 1] == "h2_full"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "recency")
    assert spec.default is False
    assert spec.reference_feature_set == "h2_full"
    assert spec.output_dir == argv[argv.index("--output-dir") + 1]
    assert spec.logistic_feature_sets == ("h2_full", *RECENCY_CHALLENGER_FEATURE_SETS)
    assert spec.note.endswith("No model is promoted from this run.")


def test_aligned_oof_rejects_a_dropped_game_and_a_later_season() -> None:
    """The challenger has to score the reference games and no later season."""
    reference: pd.DataFrame = pd.DataFrame(
        {"game_id": ["a", "b"], "season": [2016, 2016], "y_true": [1, 0]}
    )
    matched: dict[str, int] = require_aligned_oof(
        reference,
        reference.copy(),
        candidate_name="recency_epa5",
        allowed_seasons=tuple(range(2016, 2024)),
    )
    assert matched == {"n": 2}
    dropped: pd.DataFrame = reference.iloc[[0]].copy()
    with pytest.raises(ValueError, match="game_ids"):
        require_aligned_oof(
            reference,
            dropped,
            candidate_name="recency_epa5",
            allowed_seasons=tuple(range(2016, 2024)),
        )
    later: pd.DataFrame = reference.copy()
    later.loc[later.index[0], "season"] = 2025
    with pytest.raises(ValueError, match="2025"):
        require_aligned_oof(
            reference,
            later,
            candidate_name="recency_epa5",
            allowed_seasons=tuple(range(2016, 2024)),
        )


@pytest.mark.cross_platform
def test_written_cohort_reads_utf8_oof_files(tmp_path: Path) -> None:
    """The command checks the files it asks Carlos to keep."""
    reference: pd.DataFrame = pd.DataFrame(
        {"game_id": ["g1", "g2"], "season": [2018, 2019], "y_true": [1, 0], "y_proba": [0.6, 0.4]}
    )
    reference.to_csv(tmp_path / "oof_logistic_h2_full.csv", index=False, encoding="utf-8")
    for feature_set in RECENCY_CHALLENGER_FEATURE_SETS:
        reference.to_csv(
            tmp_path / f"oof_logistic_{feature_set}.csv",
            index=False,
            encoding="utf-8",
        )
    summary: dict[str, int] = require_written_cohort(tmp_path)
    assert summary == {"n": 2, "variants": len(RECENCY_CHALLENGER_FEATURE_SETS)}
    shorter: pd.DataFrame = reference.iloc[[0]]
    shorter.to_csv(
        tmp_path / "oof_logistic_recency_streak.csv",
        index=False,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recency_streak"):
        require_written_cohort(tmp_path)


@pytest.mark.cross_platform
def test_train_recency_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point the fixed command at 2025."""
    monkeypatch.setattr("sys.argv", ["train_recency.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_recency_main()


@pytest.mark.cross_platform
def test_make_and_dev_script_call_the_fixed_command() -> None:
    """The Make and PowerShell targets do not pass their own season flags."""
    makefile: str = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-recency:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_recency.py" in recipe
    assert "walk-forward" not in recipe
    assert "--cv-end-season" not in recipe
    script: str = (_REPO_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")
    assert "function Invoke-TrainRecency" in script
    assert "python scripts/train_recency.py" in script
    assert '"train-recency" { Invoke-TrainRecency }' in script
