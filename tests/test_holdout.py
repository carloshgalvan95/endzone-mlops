"""Single-split holdout: each model logs validation metrics on a local fixture."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from endzone.modeling.holdout import _fit_holdout_model
from endzone.modeling.train import build_parser, main
from tests.test_train import _reset_mlflow_tracking


@pytest.fixture(autouse=True)
def _isolate_mlflow(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep holdout runs off the repo SQLite file."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield
    _reset_mlflow_tracking()


def _write_fixture(directory: Path, seasons: list[int], games_per_season: int = 8) -> Path:
    """Write games and team EPA for the seasons a holdout split needs."""
    rows: list[dict[str, object]] = []
    game_number: int = 1
    for season in seasons:
        for index in range(games_per_season):
            rows.append(
                {
                    "game_id": f"G{game_number}",
                    "season": season,
                    "gameday": pd.Timestamp("2022-09-01") + pd.Timedelta(days=7 * game_number),
                    "home_team_franchise": ["KC", "BUF", "PHI", "DAL"][index % 4],
                    "away_team_franchise": ["BUF", "MIA", "DAL", "NYJ"][index % 4],
                    "home_score": 24 if index % 2 == 0 else 17,
                    "away_score": 17 if index % 2 == 0 else 24,
                    "home_won": 1 if index % 2 == 0 else 0,
                    "home_rest": 7,
                    "away_rest": 7,
                }
            )
            game_number += 1
    games = pd.DataFrame(rows)
    epa_rows: list[dict[str, object]] = []
    for _, game in games.iterrows():
        for franchise in (game["home_team_franchise"], game["away_team_franchise"]):
            epa_rows.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "gameday": game["gameday"],
                    "team_franchise": franchise,
                    "avg_offensive_epa": 0.1,
                    "avg_defensive_epa_allowed": -0.05,
                }
            )
    fixture_dir: Path = directory / "fixtures"
    fixture_dir.mkdir()
    games.to_parquet(fixture_dir / "fct_games.parquet")
    pd.DataFrame(epa_rows).to_parquet(fixture_dir / "fct_team_game_epa.parquet")
    return fixture_dir


def _tracking_uri(tmp_path: Path) -> str:
    """SQLite URI under tmp_path. POSIX so the same string works on Windows."""
    return "sqlite:///" + Path(tmp_path, "mlflow.db").as_posix()


def _run_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: Path,
    model: str,
) -> dict[str, str]:
    """Run one holdout and return the logged params."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", _tracking_uri(tmp_path))
    monkeypatch.setattr(
        "endzone.modeling.tracking.LOCAL_ARTIFACT_DIR",
        str(tmp_path / "mlartifacts"),
    )
    main(
        [
            "--model",
            model,
            "--fixture-path",
            str(fixture),
            "--lookback",
            "3",
            "--max-iter",
            "200",
            "--n-estimators",
            "5",
            "--seed",
            "42",
        ]
    )
    client = mlflow.MlflowClient(tracking_uri=_tracking_uri(tmp_path))
    experiment = client.get_experiment_by_name("nfl_pregame")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
    assert len(runs) == 1
    run = runs[0]
    assert "val_log_loss" in run.data.metrics
    assert run.data.params["model_type"] == model
    return dict(run.data.params)


@pytest.mark.parametrize("model", ["baseline-home", "logistic", "logistic_l2", "gbm"])
def test_holdout_logs_each_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Train, validation, and an empty test season log one run per model."""
    fixture: Path = _write_fixture(tmp_path, [2022, 2023, 2024])
    params: dict[str, str] = _run_holdout(tmp_path, monkeypatch, fixture, model)
    if model == "logistic_l2":
        assert params["c_selection"] == "inner_season"
        assert "inner_seasons" in params


def test_logistic_l2_falls_back_when_one_training_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One training season has no inner split, so C stays at the fallback."""
    fixture: Path = _write_fixture(tmp_path, [2023, 2024])
    params: dict[str, str] = _run_holdout(tmp_path, monkeypatch, fixture, "logistic_l2")
    assert params["c_selection"] == "fallback"
    assert "inner_seasons" not in params


def test_h5_fixture_without_quarterback_tables_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """h5_qb without starter tables names the missing columns."""
    fixture: Path = _write_fixture(tmp_path, [2022, 2023, 2024])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _tracking_uri(tmp_path))
    monkeypatch.setattr(
        "endzone.modeling.tracking.LOCAL_ARTIFACT_DIR",
        str(tmp_path / "mlartifacts"),
    )
    with pytest.raises(ValueError, match="h5_qb"):
        main(
            [
                "--model",
                "logistic",
                "--feature-set",
                "h5_qb",
                "--fixture-path",
                str(fixture),
                "--lookback",
                "3",
                "--max-iter",
                "50",
                "--seed",
                "42",
            ]
        )


def test_unknown_model_is_rejected() -> None:
    """A model name outside the four factories fails before a fit."""
    args = build_parser().parse_args(["--model", "logistic"])
    args.model = "not-a-model"
    empty = pd.DataFrame()
    target = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Unknown model type"):
        _fit_holdout_model(args, empty, target, empty, target)
