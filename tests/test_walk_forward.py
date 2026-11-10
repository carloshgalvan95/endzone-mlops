"""Walk-forward week backtest on a local fixture. No warehouse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from endzone.features.pregame import build_pregame_features
from endzone.modeling.registry import metrics_from_walk_forward_report, promotion_decision
from endzone.modeling.registry_command import fit_reference_pipeline
from endzone.modeling.registry_command import main as registry_main
from endzone.modeling.walk_forward import (
    WalkForwardSettings,
    _features_for_week,
    default_slate,
    main,
    run_walk_forward,
    week_start_elo_probabilities,
    write_walk_forward_report,
)
from tests.test_registry import _release_mlflow_sqlalchemy_engines, _sqlite_uri

_HOMES: tuple[str, ...] = ("KC", "BUF", "PHI", "DAL")
_AWAYS: tuple[str, ...] = ("LV", "NYJ", "SF", "SEA")


def _slate(season_weeks: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Completed games and team EPA. ``season_weeks`` maps a season to its week count."""
    rows: list[dict[str, object]] = []
    for season, n_weeks in season_weeks.items():
        for week in range(1, n_weeks + 1):
            for index, (home, away) in enumerate(zip(_HOMES, _AWAYS, strict=True)):
                home_won: int = 0 if (week + index) % 2 else 1
                rows.append(
                    {
                        "game_id": f"{season}-{week}-{index}",
                        "season": season,
                        "week": week,
                        "gameday": pd.Timestamp("2023-09-07")
                        + pd.Timedelta(days=(season - 2023) * 400 + week * 7 + index),
                        "home_team": home,
                        "away_team": away,
                        "home_team_franchise": home,
                        "away_team_franchise": away,
                        "home_score": 24 if home_won else 17,
                        "away_score": 17 if home_won else 24,
                        "home_won": home_won,
                        "home_rest": 7,
                        "away_rest": 7,
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                    }
                )
    games = pd.DataFrame(rows)
    epa_rows: list[dict[str, object]] = []
    for _, game in games.iterrows():
        for franchise, won in (
            (game["home_team_franchise"], int(game["home_won"]) == 1),
            (game["away_team_franchise"], int(game["home_won"]) == 0),
        ):
            epa_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": int(game["season"]),
                    "week": int(game["week"]),
                    "gameday": game["gameday"],
                    "team_franchise": franchise,
                    "avg_offensive_epa": 0.12 if won else -0.04,
                    "avg_defensive_epa_allowed": -0.03 if won else 0.08,
                }
            )
    return games, pd.DataFrame(epa_rows)


def _settings(**overrides: object) -> WalkForwardSettings:
    """Score 2024 weeks 1 and 2. The registered cutoff is 2024 week 3."""
    values: dict[str, object] = {
        "v1_through_season": 2023,
        "start_season": 2024,
        "slate_season": 2024,
        "slate_week": 3,
        "lookback_games": 10,
        "logistic_c": 1.0,
        "logistic_max_iter": 1000,
        "random_state": 42,
        "n_bootstrap": 20,
        "bootstrap_seed": 42,
        "max_null_fraction": 0.5,
    }
    values.update(overrides)
    return WalkForwardSettings(**values)  # type: ignore[arg-type]


def _run() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """One backtest on six 2023 weeks and two 2024 weeks."""
    games, team_games = _slate({2023: 6, 2024: 2})
    report = run_walk_forward(games, team_games, _settings())
    return report, games, team_games


def test_walk_forward_uses_only_prior_games_and_keeps_v1_capped() -> None:
    """2024 week 2 trains on week 1. v1 does not, and it stays the same size."""
    report, _, _ = _run()
    weeks = {row["week"]: row for row in report["weeks"]}
    assert list(weeks) == [1, 2]
    assert weeks[1]["n_train_candidate"] == weeks[1]["n_train_v1"]
    assert weeks[2]["n_train_candidate"] == weeks[1]["n_train_candidate"] + weeks[1]["n"]
    assert weeks[2]["n_train_v1"] == weeks[1]["n_train_v1"]
    assert report["aggregate"]["n"] == weeks[1]["n"] + weeks[2]["n"]
    assert report["aggregate"]["market_n"] == report["aggregate"]["n"]
    assert report["slate"] == {"season": 2024, "week": 3}
    metrics_from_walk_forward_report(report)


def test_later_week_results_do_not_change_an_earlier_score() -> None:
    """Flipping 2024 week 2 leaves the week 1 log loss alone."""
    report, games, team_games = _run()
    poisoned_games = games.copy()
    poisoned_team = team_games.copy()
    week_two = poisoned_games["game_id"].astype(str).str.startswith("2024-2-")
    poisoned_games.loc[week_two, "home_won"] = 1 - poisoned_games.loc[week_two, "home_won"].astype(
        int
    )
    poisoned_games.loc[week_two, "home_score"] = 3
    poisoned_games.loc[week_two, "away_score"] = 40
    epa_week_two = poisoned_team["game_id"].astype(str).str.startswith("2024-2-")
    poisoned_team.loc[epa_week_two, "avg_offensive_epa"] = 9.0
    again = run_walk_forward(poisoned_games, poisoned_team, _settings())
    before = report["weeks"][0]
    after = again["weeks"][0]
    assert before["week"] == 1 and after["week"] == 1
    assert before["candidate_log_loss"] == after["candidate_log_loss"]
    assert before["v1_log_loss"] == after["v1_log_loss"]
    assert before["elo_log_loss"] == after["elo_log_loss"]


def test_same_week_results_are_hidden_from_later_games_in_that_week() -> None:
    """Thursday's score is not a feature for Sunday in the scored week."""
    games, team_games = _slate({2023: 2, 2024: 1})
    history = games.loc[games["season"] == 2023].copy()
    week = games.loc[games["season"] == 2024].copy()
    original = _features_for_week(history, week, team_games, lookback_games=10)
    poisoned_games = games.copy()
    poisoned_team = team_games.copy()
    thursday = poisoned_games["game_id"].astype(str) == "2024-1-0"
    poisoned_games.loc[thursday, "home_won"] = 1 - int(
        poisoned_games.loc[thursday, "home_won"].iloc[0]
    )
    poisoned_team.loc[poisoned_team["game_id"].astype(str) == "2024-1-0", "avg_offensive_epa"] = 8.0
    poisoned_week = poisoned_games.loc[poisoned_games["season"] == 2024].copy()
    changed = _features_for_week(history, poisoned_week, poisoned_team, lookback_games=10)
    sunday = "2024-1-3"
    left = original.loc[original["game_id"].astype(str) == sunday].reset_index(drop=True)
    right = changed.loc[changed["game_id"].astype(str) == sunday].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_week_start_elo_ignores_same_week_results() -> None:
    """A result inside the week does not move Elo for the other game."""
    history = pd.DataFrame(
        [
            {
                "game_id": "h1",
                "season": 2024,
                "week": 1,
                "gameday": pd.Timestamp("2024-09-08"),
                "home_team_franchise": "PHI",
                "away_team_franchise": "SF",
                "home_score": 20,
                "away_score": 10,
                "home_won": 1,
            }
        ]
    )
    week = pd.DataFrame(
        [
            {
                "game_id": "w-thu",
                "season": 2024,
                "week": 2,
                "gameday": pd.Timestamp("2024-09-12"),
                "home_team_franchise": "BUF",
                "away_team_franchise": "NYJ",
                "home_score": 30,
                "away_score": 7,
                "home_won": 1,
            },
            {
                "game_id": "w-sun",
                "season": 2024,
                "week": 2,
                "gameday": pd.Timestamp("2024-09-15"),
                "home_team_franchise": "PHI",
                "away_team_franchise": "KC",
                "home_score": 21,
                "away_score": 20,
                "home_won": 1,
            },
        ]
    )
    first = week_start_elo_probabilities(history, week)
    flipped = week.copy()
    flipped.loc[flipped["game_id"] == "w-thu", "home_won"] = 0
    flipped.loc[flipped["game_id"] == "w-thu", "home_score"] = 7
    flipped.loc[flipped["game_id"] == "w-thu", "away_score"] = 30
    second = week_start_elo_probabilities(history, flipped)
    pd.testing.assert_series_equal(first, second)
    moved_history = history.copy()
    moved_history.loc[0, "home_won"] = 0
    moved_history.loc[0, "home_score"] = 7
    moved_history.loc[0, "away_score"] = 30
    third = week_start_elo_probabilities(moved_history, week)
    assert not np.isclose(float(first.loc["w-sun"]), float(third.loc["w-sun"]))


def test_game_ids_returns_the_same_row_as_the_full_build() -> None:
    """Limiting the emitted games does not change the prior-game math."""
    games, team_games = _slate({2023: 2, 2024: 1})
    full = build_pregame_features(games, team_games, lookback_games=10)
    one_id = "2024-1-2"
    subset = build_pregame_features(games, team_games, lookback_games=10, game_ids={one_id})
    expected = full.loc[full["game_id"].astype(str) == one_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset.reset_index(drop=True), expected)
    with pytest.raises(ValueError, match="not found"):
        build_pregame_features(games, team_games, game_ids={"missing"})
    with pytest.raises(ValueError, match="empty"):
        build_pregame_features(games, team_games, game_ids=set())


def test_fit_reference_pipeline_stops_before_the_slate() -> None:
    """A week-2 cutoff keeps week 1. A week-1 cutoff does not."""
    games, team_games = _slate({2023: 6, 2024: 2})
    before_week_one = fit_reference_pipeline(
        games,
        team_games,
        through_season=2023,
        before_season=2024,
        before_week=1,
        logistic_max_iter=200,
    )
    before_week_two = fit_reference_pipeline(
        games,
        team_games,
        through_season=2023,
        before_season=2024,
        before_week=2,
        logistic_max_iter=200,
    )
    assert before_week_one["cutoff_season"] == 2024
    assert before_week_one["cutoff_week"] == 1
    assert before_week_two["n_train"] == before_week_one["n_train"] + 4
    assert before_week_one["through_season"] == 2023


def test_default_slate_is_the_week_after_the_last_completed_week() -> None:
    """A season that ends at week 2 proposes week 3. Pass flags to override it."""
    games, _ = _slate({2024: 2})
    assert default_slate(games) == {"season": 2024, "week": 3}


def test_report_round_trip_keeps_the_gate(tmp_path: Path) -> None:
    """The written JSON is what verify and register-recent read."""
    report, _, _ = _run()
    paths = write_walk_forward_report(report, tmp_path)
    loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    tags = metrics_from_walk_forward_report(loaded)
    decision = promotion_decision(tags)
    assert loaded["gate"] == "non_inferiority"
    assert loaded["margin"] == pytest.approx(0.002)
    assert decision["allowed"] is loaded["passes"]
    if loaded["passes"]:
        verified = main(["--verify", "--report", paths["json"]])
        assert verified["allowed"] is True
        return
    with pytest.raises(SystemExit) as raised:
        main(["--verify", "--report", paths["json"]])
    assert raised.value.code == 1


def test_verify_accepts_the_warehouse_gap_without_gate_fields(tmp_path: Path) -> None:
    """The first warehouse JSON has no gate fields. Non-inferiority still passes.

    Candidate 0.6239 against v1 0.6235 is inside the 0.002 margin. The file
    matches that run's shape: one pooled row, ``beats_v1`` false, and no
    ``gate`` key. Verify does not need a second backtest.
    """
    report = _synthetic_report(beats=False)
    report["aggregate"]["candidate_log_loss"] = 0.6239
    report["aggregate"]["v1_log_loss"] = 0.6235
    report["aggregate"]["log_loss_diff"] = 0.6239 - 0.6235
    report["aggregate"]["beats_v1"] = False
    report["aggregate"]["n"] = 706
    report["aggregate"]["market_n"] = 706
    report["aggregate"]["elo_log_loss"] = 0.6454
    report["aggregate"]["baseline_home_log_loss"] = 0.6934
    report["aggregate"]["market_log_loss"] = 0.6028
    report["aggregate"]["log_loss_diff_ci_low"] = -0.0006
    report["aggregate"]["log_loss_diff_ci_high"] = 0.0013
    week = report["weeks"][0]
    week["n"] = 706
    week["market_n"] = 706
    week["candidate_log_loss"] = 0.6239
    week["v1_log_loss"] = 0.6235
    week["elo_log_loss"] = 0.6454
    week["baseline_home_log_loss"] = 0.6934
    week["market_log_loss"] = 0.6028
    report["weeks"] = [week]
    assert "gate" not in report
    path = tmp_path / "walk_forward.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    verified = main(["--verify", "--report", str(path)])
    assert verified["allowed"] is True
    assert verified["reason"] == "walk_forward_refresh_passes"
    assert verified["gate"] == "non_inferiority"


def test_reported_gate_has_to_match_the_losses(tmp_path: Path) -> None:
    """A hand-edited passes flag fails even when the other numbers are consistent."""
    report = _synthetic_report(beats=False)
    report["gate"] = "non_inferiority"
    report["margin"] = 0.002
    report["passes"] = True
    path = tmp_path / "walk_forward.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main(["--verify", "--report", str(path)])
    assert raised.value.code == 1


def test_register_recent_refuses_a_different_recipe_before_a_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed C is a challenger. This command does not register it."""

    def refuse_load(*args: object, **kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise AssertionError("a different recipe must not load data")

    monkeypatch.setattr("endzone.modeling.registry_command.load_data", refuse_load)
    report = _synthetic_report(beats=True)
    report["logistic_c"] = 0.1
    monkeypatch.setattr(
        "endzone.modeling.registry_command.read_walk_forward_report",
        lambda report_path: report,
    )
    with pytest.raises(SystemExit) as raised:
        registry_main(
            [
                "register-recent",
                "--report",
                "reports/walk_forward/challenger.json",
                "--tracking-uri",
                "sqlite:///reports/unused.db",
                "--registry-uri",
                "sqlite:///reports/unused.db",
            ]
        )
    assert raised.value.code == 1


def test_verify_exits_when_the_candidate_loses(tmp_path: Path) -> None:
    """A candidate outside the refresh margin is a result, and the gate refuses it."""
    report = _synthetic_report(beats=False)
    path = tmp_path / "walk_forward.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main(["--verify", "--report", str(path)])
    assert raised.value.code == 1


def test_register_recent_refuses_before_a_fit_when_the_report_loses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A losing report does not load the warehouse and does not write a version."""

    def refuse_load(*args: object, **kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise AssertionError("a losing report must not load data")

    monkeypatch.setattr("endzone.modeling.registry_command.load_data", refuse_load)
    path = Path("reports/walk_forward/losing.json")
    # The path has to exist. Use a temp file via the monkeypatched reader instead.
    monkeypatch.setattr(
        "endzone.modeling.registry_command.read_walk_forward_report",
        lambda report: _synthetic_report(beats=False),
    )
    with pytest.raises(SystemExit) as raised:
        registry_main(
            [
                "register-recent",
                "--report",
                str(path),
                "--tracking-uri",
                "sqlite:///reports/unused.db",
                "--registry-uri",
                "sqlite:///reports/unused.db",
            ]
        )
    assert raised.value.code == 1


def test_register_recent_writes_a_version_and_leaves_the_alias_unset(tmp_path: Path) -> None:
    """A winning report registers the fresh fit. Promote is still a separate call."""
    import mlflow

    games, team_games = _slate({2023: 6, 2024: 2})
    fixture = tmp_path / "fixtures"
    fixture.mkdir()
    games.to_parquet(fixture / "fct_games.parquet")
    team_games.to_parquet(fixture / "fct_team_game_epa.parquet")
    report = _synthetic_report(beats=True)
    report_path = tmp_path / "walk_forward.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    uri = _sqlite_uri(tmp_path / "mlflow.db")
    try:
        registered = registry_main(
            [
                "register-recent",
                "--report",
                str(report_path),
                "--fixture-path",
                str(fixture),
                "--tracking-uri",
                uri,
                "--registry-uri",
                uri,
            ]
        )
        assert registered["alias"] is None
        assert registered["allowed"] is True
        assert registered["cutoff_season"] == 2026
        assert registered["cutoff_week"] == 10
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        versions = client.search_model_versions("name='endzone-winprob'")
        assert len(versions) == 1
        promoted = registry_main(
            ["promote", "--version", registered["model_version"], "--registry-uri", uri]
        )
    finally:
        _release_mlflow_sqlalchemy_engines()
    assert promoted["alias"] == "prod"
    assert promoted["reason"] == "walk_forward_refresh_passes"


def _synthetic_report(beats: bool) -> dict[str, Any]:
    """A two-week report whose aggregate matches the rows. No warehouse numbers."""
    candidate = 0.50 if beats else 0.70
    v1_loss = 0.60
    weeks = []
    for week in (1, 2):
        weeks.append(
            {
                "season": 2024,
                "week": week,
                "n": 4,
                "n_train_candidate": 20 + week,
                "n_train_v1": 20,
                "candidate_log_loss": candidate,
                "v1_log_loss": v1_loss,
                "elo_log_loss": 0.65,
                "baseline_home_log_loss": 0.69,
                "market_n": 4,
                "market_log_loss": 0.55,
            }
        )
    return {
        "protocol": "walk_forward_week",
        "model": "logistic",
        "feature_set": "h2_full",
        "v1_through_season": 2023,
        "start_season": 2024,
        "slate": {"season": 2026, "week": 10},
        "lookback_games": 10,
        "logistic_c": 1.0,
        "logistic_max_iter": 1000,
        "random_state": 42,
        "n_bootstrap": 20,
        "bootstrap_seed": 42,
        "weeks": weeks,
        "aggregate": {
            "n": 8,
            "candidate_log_loss": candidate,
            "v1_log_loss": v1_loss,
            "elo_log_loss": 0.65,
            "baseline_home_log_loss": 0.69,
            "market_log_loss": 0.55,
            "market_n": 8,
            "log_loss_diff": candidate - v1_loss,
            "log_loss_diff_ci_low": -0.02,
            "log_loss_diff_ci_high": -0.01 if beats else 0.02,
            "beats_v1": candidate < v1_loss,
        },
        "notes": ["Synthetic fixture. Not a warehouse result."],
    }
