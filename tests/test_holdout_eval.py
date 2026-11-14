"""One-shot 2024 and 2025 holdout: isolation, schema, and the overwrite guard."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.pregame import (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_PKG_CORE_INJ_L2,
    PREGAME_FEATURE_COLUMNS_H2_FULL,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
)
from endzone.modeling.holdout import run_holdout
from endzone.modeling.holdout_eval import (
    BOOTSTRAP_SEED,
    CV_LOG_LOSS,
    CV_N,
    EVAL_SEASONS,
    HOLDOUT_EXPERIMENT_NAME,
    HOLDOUT_JSON_NAME,
    HOLDOUT_MARKDOWN_NAME,
    MAX_ITER,
    N_BOOTSTRAP,
    POOLED_SLICE,
    REFERENCE_C,
    REFERENCE_FEATURE_SET,
    SEED,
    SELECTED_C,
    SELECTED_FEATURE_SET,
    SELECTED_N_COLUMNS,
    log_holdout_run,
    render_holdout_markdown,
    run_holdout_evaluation,
    write_holdout_reports,
)
from endzone.modeling.models import (
    model_feature_columns,
    train_baseline_home,
    train_logistic_regression,
)
from scripts.run_holdout import build_parser, main

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "feature_set",
        "role",
        "n",
        "log_loss",
        "brier",
        "auc",
        "accuracy",
        "vs_h2_full",
        "vs_market",
    }
)
_COMPARISON_KEYS: frozenset[str] = frozenset(
    {
        "log_loss_diff",
        "log_loss_diff_ci_low",
        "log_loss_diff_ci_high",
        "n",
        "n_bootstrap",
    }
)
_CALIBRATION_KEYS: frozenset[str] = frozenset(
    {"model", "feature_set", "slice", "n", "n_bins", "ece", "bins"}
)
_INFORMATION_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "feature_set",
        "slice",
        "status",
        "reason",
        "n",
        "n_bootstrap",
        "n_bootstrap_used",
        "intercept",
        "intercept_ci_low",
        "intercept_ci_high",
        "market_coef",
        "market_ci_low",
        "market_ci_high",
        "model_coef",
        "model_ci_low",
        "model_ci_high",
    }
)


def _slate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seasons 2022 through 2025, with both labels in every season."""
    games: list[dict[str, object]] = []
    epa_rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for week in range(6):
            day: pd.Timestamp = pd.Timestamp(f"{season}-09-11") + pd.Timedelta(days=7 * week)
            pairings: tuple[tuple[str, str], ...] = (
                (("A", "B"), ("C", "D")) if week % 2 == 0 else (("A", "C"), ("B", "D"))
            )
            for index, (home, away) in enumerate(pairings):
                game_id: str = f"{season}-{week}-{index}"
                home_won: int = 0 if (week + index) % 2 == 0 else 1
                games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week + 1,
                        "game_type": "REG",
                        "gameday": day,
                        "gametime": "13:00",
                        "home_team": home,
                        "away_team": away,
                        "home_team_franchise": home,
                        "away_team_franchise": away,
                        "home_won": home_won,
                        "home_score": 24 if home_won else 14,
                        "away_score": 17 if home_won else 21,
                        "home_rest": 7,
                        "away_rest": 7,
                        "home_moneyline": -140 if home_won else 120,
                        "away_moneyline": 120 if home_won else -140,
                    }
                )
                for team, offense in ((home, 0.2), (away, -0.05)):
                    epa_rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "gameday": day,
                            "team_franchise": team,
                            "avg_offensive_epa": offense,
                            "avg_defensive_epa_allowed": 0.02,
                        }
                    )
    return pd.DataFrame(games), pd.DataFrame(epa_rows)


def _starter_tables(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One chart starter and one dropback row per team-game."""
    starter_rows: list[dict[str, object]] = []
    dropback_rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        starter_rows.append(
            {
                "game_id": game.game_id,
                "home_starter_player_id": f"QB-{game.home_team_franchise}",
                "away_starter_player_id": f"QB-{game.away_team_franchise}",
            }
        )
        for franchise in (game.home_team_franchise, game.away_team_franchise):
            dropback_rows.append(
                {
                    "game_id": game.game_id,
                    "gameday": game.gameday,
                    "team_franchise": franchise,
                    "passer_player_id": f"QB-{franchise}",
                    "dropbacks": 30,
                    "dropback_epa": 3.0,
                }
            )
    return pd.DataFrame(starter_rows), pd.DataFrame(dropback_rows)


def _empty_injuries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chart and report frames with the columns the absence count reads."""
    starters: pd.DataFrame = pd.DataFrame(
        columns=["season", "week", "game_type", "team", "gsis_id", "position"]
    )
    reports: pd.DataFrame = pd.DataFrame(
        columns=[
            "season",
            "week",
            "game_type",
            "team",
            "gsis_id",
            "report_status",
            "date_modified",
        ]
    )
    return starters, reports


def _evaluate(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Fit the synthetic slate and record every logistic and home-prior call."""
    recorded: list[dict[str, object]] = []
    real_logistic = train_logistic_regression
    real_prior = train_baseline_home

    def spy_logistic(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[object, dict[str, float]]:
        recorded.append(
            {
                "kind": "logistic",
                "C": C,
                "seasons": sorted(int(season) for season in X_train["season"].unique()),
                "eval_seasons": sorted(int(season) for season in X_val["season"].unique()),
                "n_columns": len(model_feature_columns(X_train)),
            }
        )
        return real_logistic(
            X_train,
            y_train,
            X_val,
            y_val,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            sample_weight=sample_weight,
        )

    def spy_prior(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[object, dict[str, float]]:
        recorded.append(
            {
                "kind": "baseline-home",
                "seasons": sorted(int(season) for season in X_train["season"].unique()),
                "eval_seasons": sorted(int(season) for season in X_val["season"].unique()),
            }
        )
        return real_prior(X_train, y_train, X_val, y_val)

    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.train_logistic_regression",
        spy_logistic,
    )
    monkeypatch.setattr("endzone.modeling.holdout_eval.train_baseline_home", spy_prior)
    games, team_games = _slate()
    starters, dropbacks = _starter_tables(games)
    depth_charts, injuries = _empty_injuries()
    report: dict[str, object] = run_holdout_evaluation(
        games,
        team_games,
        depth_chart_starters=depth_charts,
        injury_reports=injuries,
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        max_iter=200,
        n_bootstrap=20,
    )
    return report, recorded


def test_locked_recipe_constants() -> None:
    """The command's defaults are the preregistered recipe, not a search."""
    signature = inspect.signature(run_holdout_evaluation)
    assert signature.parameters["lookback"].default == 10
    assert signature.parameters["max_iter"].default == MAX_ITER == 1000
    assert signature.parameters["seed"].default == SEED == 42
    assert signature.parameters["n_bootstrap"].default == N_BOOTSTRAP == 1000
    assert signature.parameters["bootstrap_seed"].default == BOOTSTRAP_SEED == 42
    assert EVAL_SEASONS == (2024, 2025)
    assert SELECTED_C == 0.1
    assert REFERENCE_C == 1.0
    assert SELECTED_FEATURE_SET == FEATURE_SET_PKG_CORE_INJ_L2
    assert REFERENCE_FEATURE_SET == FEATURE_SET_H2_FULL
    assert len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ) == SELECTED_N_COLUMNS == 47
    assert CV_LOG_LOSS == 0.6330
    assert CV_N == 2191
    assert HOLDOUT_EXPERIMENT_NAME == "nfl_pregame/holdout_2024_2025"
    destinations: set[str] = {
        action.dest for action in build_parser()._actions if action.dest != "help"
    }
    assert destinations == {"allow_overwrite"}


def test_training_rows_exclude_the_scored_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """2024 is out of the 2024 fit. 2025 is out of both fits. 2024 is in the 2025 fit."""
    _report, recorded = _evaluate(monkeypatch)
    logistic: list[dict[str, object]] = [row for row in recorded if row["kind"] == "logistic"]
    assert len(logistic) == 4
    by_width: dict[int, list[dict[str, object]]] = {}
    for row in logistic:
        by_width.setdefault(int(row["n_columns"]), []).append(row)
    selected_calls: list[dict[str, object]] = by_width[len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ)]
    reference_calls: list[dict[str, object]] = by_width[len(PREGAME_FEATURE_COLUMNS_H2_FULL)]
    assert {row["C"] for row in selected_calls} == {0.1}
    assert {row["C"] for row in reference_calls} == {1.0}
    for row in logistic:
        seasons: set[int] = set(row["seasons"])  # type: ignore[arg-type]
        eval_seasons: set[int] = set(row["eval_seasons"])  # type: ignore[arg-type]
        assert 2025 not in seasons
        assert eval_seasons.isdisjoint(seasons)
        assert len(eval_seasons) == 1
    selected_2024: dict[str, object] = next(
        row for row in selected_calls if row["eval_seasons"] == [2024]
    )
    selected_2025: dict[str, object] = next(
        row for row in selected_calls if row["eval_seasons"] == [2025]
    )
    assert 2024 not in set(selected_2024["seasons"])  # type: ignore[arg-type]
    assert 2024 in set(selected_2025["seasons"])  # type: ignore[arg-type]
    assert max(selected_2024["seasons"]) == 2023  # type: ignore[type-var]
    assert max(selected_2025["seasons"]) == 2024  # type: ignore[type-var]
    prior_calls: list[dict[str, object]] = [
        row for row in recorded if row["kind"] == "baseline-home"
    ]
    assert len(prior_calls) == 2
    for row in prior_calls:
        assert 2025 not in set(row["seasons"])  # type: ignore[arg-type]
        assert set(row["eval_seasons"]).isdisjoint(set(row["seasons"]))  # type: ignore[arg-type]


def test_report_schema_matches_the_aligned_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pooled and per-season files share counts, intervals, calibration, and market info."""
    report, _recorded = _evaluate(monkeypatch)
    paths: dict[str, Path] = write_holdout_reports(report, tmp_path)
    loaded: dict[str, object] = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert set(loaded) == {
        "metadata",
        "parameters",
        "pooled",
        "seasons",
        "calibration",
        "market_information",
        "oof",
    }
    metadata: dict[str, object] = loaded["metadata"]  # type: ignore[assignment]
    assert metadata["adr"] == "ADR-010"
    assert metadata["issue"] == 61
    assert metadata["experiment"] == HOLDOUT_EXPERIMENT_NAME
    assert metadata["runs_once"] is True
    assert metadata["publish_regardless"] is True
    assert metadata["no_selection_after_2025"] is True
    assert metadata["registered"] is False
    assert metadata["promoted"] is False
    parameters: dict[str, object] = loaded["parameters"]  # type: ignore[assignment]
    assert parameters["eval_seasons"] == [2024, 2025]
    assert parameters["cv_log_loss"] == 0.6330
    assert parameters["cv_n"] == 2191
    assert parameters["n_bootstrap"] == 20
    assert parameters["bootstrap_seed"] == 42
    selected_spec: dict[str, object] = next(
        row
        for row in parameters["models"]  # type: ignore[union-attr]
        if row["feature_set"] == SELECTED_FEATURE_SET
    )
    reference_spec: dict[str, object] = next(
        row
        for row in parameters["models"]  # type: ignore[union-attr]
        if row["feature_set"] == REFERENCE_FEATURE_SET
    )
    assert selected_spec["C"] == 0.1
    assert selected_spec["n_columns"] == 47
    assert selected_spec["role"] == "selected"
    assert reference_spec["C"] == 1.0
    assert reference_spec["role"] == "prior_reference"
    seasons: list[dict[str, object]] = loaded["seasons"]  # type: ignore[assignment]
    assert [row["eval_season"] for row in seasons] == [2024, 2025]
    season_2024: dict[str, object] = seasons[0]
    season_2025: dict[str, object] = seasons[1]
    assert season_2024["train_season_max"] == 2023
    assert 2024 not in season_2024["train_seasons"]  # type: ignore[operator]
    assert season_2025["train_season_max"] == 2024
    assert 2024 in season_2025["train_seasons"]  # type: ignore[operator]
    assert 2025 not in season_2025["train_seasons"]  # type: ignore[operator]
    pooled: dict[str, object] = loaded["pooled"]  # type: ignore[assignment]
    assert pooled["slice"] == POOLED_SLICE
    _assert_metric_block(pooled["rows"], pooled["n"], n_bootstrap=20)  # type: ignore[arg-type]
    for season in seasons:
        _assert_metric_block(season["rows"], season["n_eval"], n_bootstrap=20)  # type: ignore[arg-type]
        for fit in season["fits"]:  # type: ignore[union-attr]
            assert max(fit["train_seasons"]) < season["eval_season"]
            assert season["eval_season"] not in fit["train_seasons"]
            assert 2025 not in fit["train_seasons"]
    assert pooled["n"] == season_2024["n_eval"] + season_2025["n_eval"]
    _assert_calibration(loaded["calibration"], loaded)  # type: ignore[arg-type]
    _assert_market_information(loaded["market_information"], loaded)  # type: ignore[arg-type]
    _assert_oof(loaded["oof"], pooled["n"], season_2024["n_eval"], season_2025["n_eval"])  # type: ignore[arg-type]
    markdown: str = paths["markdown"].read_text(encoding="utf-8")
    assert markdown == render_holdout_markdown(report)
    selected_row: dict[str, object] = next(
        row
        for row in pooled["rows"]  # type: ignore[union-attr]
        if row["feature_set"] == SELECTED_FEATURE_SET
    )
    assert f"{selected_row['log_loss']:.6f}" in markdown
    assert "pooled" in markdown
    assert "runs once" in markdown
    assert "does not register or promote" in markdown
    assert "\u2014" not in markdown
    assert "\u2013" not in markdown
    assert paths["json"].name == HOLDOUT_JSON_NAME
    assert paths["markdown"].name == HOLDOUT_MARKDOWN_NAME


def _assert_metric_block(rows: list[dict[str, object]], n_games: int, *, n_bootstrap: int) -> None:
    """Every comparison row shares n, a finite score, and a closed interval."""
    assert len(rows) == 6
    assert {row["feature_set"] for row in rows} == {
        SELECTED_FEATURE_SET,
        REFERENCE_FEATURE_SET,
        "elo",
        "elo_mov",
        "none",
        "market",
    }
    for row in rows:
        assert set(row) == _METRIC_KEYS
        assert row["n"] == n_games
        assert n_games > 0
        for name in ("log_loss", "brier", "auc", "accuracy"):
            assert isinstance(row[name], float)
            assert np.isfinite(row[name])
        for key in ("vs_h2_full", "vs_market"):
            comparison: dict[str, object] = row[key]  # type: ignore[assignment]
            assert set(comparison) == _COMPARISON_KEYS
            assert comparison["n"] == n_games
            assert comparison["n_bootstrap"] == n_bootstrap
            difference: float = comparison["log_loss_diff"]  # type: ignore[assignment]
            low: float = comparison["log_loss_diff_ci_low"]  # type: ignore[assignment]
            high: float = comparison["log_loss_diff_ci_high"]  # type: ignore[assignment]
            assert low <= difference <= high


def _assert_calibration(rows: list[dict[str, object]], report: dict[str, object]) -> None:
    """Ten equal-count bins when the slice has at least ten games."""
    assert len(rows) == 18
    slices: set[str] = set()
    for row in rows:
        assert set(row) == _CALIBRATION_KEYS
        slices.add(str(row["slice"]))
        n_games: int = int(row["n"])
        assert row["n_bins"] == min(10, n_games)
        bins: list[dict[str, object]] = row["bins"]  # type: ignore[assignment]
        assert len(bins) == row["n_bins"]
        assert [item["bin"] for item in bins] == list(range(1, int(row["n_bins"]) + 1))
        assert isinstance(row["ece"], float)
        assert sum(int(item["n"]) for item in bins) == n_games
    assert slices == {"2024", "2025", POOLED_SLICE}
    pooled_n: int = int(report["pooled"]["n"])  # type: ignore[index]
    pooled_rows: list[dict[str, object]] = [row for row in rows if row["slice"] == POOLED_SLICE]
    assert {row["n"] for row in pooled_rows} == {pooled_n}


def _assert_market_information(rows: list[dict[str, object]], report: dict[str, object]) -> None:
    """Each non-market model has market_coef and model_coef on every slice."""
    assert len(rows) == 15
    assert all(row["model"] != "market" for row in rows)
    for row in rows:
        assert set(row) == _INFORMATION_KEYS
        assert row["status"] == "scored"
        assert isinstance(row["market_coef"], float)
        assert isinstance(row["model_coef"], float)
        assert np.isfinite(row["market_coef"])
        assert np.isfinite(row["model_coef"])
        assert row["n_bootstrap"] == 20
    pooled_n: int = int(report["pooled"]["n"])  # type: ignore[index]
    pooled_rows: list[dict[str, object]] = [row for row in rows if row["slice"] == POOLED_SLICE]
    assert {row["n"] for row in pooled_rows} == {pooled_n}


def _assert_oof(oof: dict[str, object], pooled_n: int, n_2024: int, n_2025: int) -> None:
    """Game rows are the aligned seasons, with no null probability."""
    assert set(oof) == {"n", "n_2024", "n_2025", "model_keys", "rows"}
    assert oof["n"] == pooled_n == n_2024 + n_2025
    assert oof["n_2024"] == n_2024
    assert oof["n_2025"] == n_2025
    rows: list[dict[str, object]] = oof["rows"]  # type: ignore[assignment]
    assert len(rows) == pooled_n
    game_ids: list[str] = [str(row["game_id"]) for row in rows]
    assert len(game_ids) == len(set(game_ids))
    keys: list[str] = list(oof["model_keys"])  # type: ignore[arg-type]
    assert len(keys) == 6
    for row in rows:
        assert set(row) == {"game_id", "season", "y_true", "probabilities"}
        assert row["season"] in (2024, 2025)
        assert row["y_true"] in (0, 1)
        probabilities: dict[str, float] = row["probabilities"]  # type: ignore[assignment]
        assert set(probabilities) == set(keys)
        assert all(
            isinstance(value, float) and np.isfinite(value) for value in probabilities.values()
        )
    assert sum(row["season"] == 2024 for row in rows) == n_2024
    assert sum(row["season"] == 2025 for row in rows) == n_2025


def test_cv_holdout_still_refuses_the_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    """train.py --val-season still refuses pkg_core_inj_l2 before any warehouse read."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("warehouse")

    monkeypatch.setattr("endzone.modeling.holdout.load_data", boom)
    with pytest.raises(ValueError, match="2016-2023"):
        run_holdout(argparse.Namespace(feature_set=FEATURE_SET_PKG_CORE_INJ_L2))


@pytest.mark.cross_platform
def test_existing_report_stops_before_the_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-shot guard fails before a loader runs."""
    (tmp_path / HOLDOUT_JSON_NAME).write_text("{}\n", encoding="utf-8")

    def boom() -> dict[str, object]:
        raise AssertionError("warehouse")

    monkeypatch.setattr("scripts.run_holdout.run_holdout_from_warehouse", boom)
    with pytest.raises(SystemExit, match="already exists"):
        main([], report_dir=tmp_path)
    with pytest.raises(SystemExit, match="once"):
        main([], report_dir=tmp_path)


@pytest.mark.cross_platform
def test_allow_overwrite_is_the_only_extra_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--allow-overwrite reaches the eval. Any other flag is rejected."""
    (tmp_path / HOLDOUT_JSON_NAME).write_text("{}\n", encoding="utf-8")
    called: dict[str, int] = {"n": 0}

    def fake() -> dict[str, object]:
        called["n"] += 1
        return {"metadata": {}}

    monkeypatch.setattr("scripts.run_holdout.run_holdout_from_warehouse", fake)
    monkeypatch.setattr(
        "scripts.run_holdout.write_holdout_reports",
        lambda _report, directory: {
            "json": directory / HOLDOUT_JSON_NAME,
            "markdown": directory / HOLDOUT_MARKDOWN_NAME,
        },
    )
    monkeypatch.setattr("scripts.run_holdout.log_holdout_run", lambda *_args, **_kwargs: None)
    main(["--allow-overwrite"], report_dir=tmp_path)
    assert called["n"] == 1
    with pytest.raises(SystemExit):
        main(["--season", "2025"], report_dir=tmp_path)


def test_log_names_the_holdout_experiment_and_does_not_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking uses the holdout experiment and never the registry."""
    seen: dict[str, object] = {}

    def activate(name: str, uri: str) -> None:
        seen["experiment"] = name
        seen["uri"] = uri

    class _Run:
        def __enter__(self) -> _Run:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr("endzone.modeling.holdout_eval.activate_tracking_experiment", activate)
    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.configured_tracking_uri", lambda: "sqlite:///tmp.db"
    )
    monkeypatch.setattr("endzone.modeling.holdout_eval.mlflow.start_run", lambda: _Run())
    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.mlflow.log_params", lambda params: seen.update(params=params)
    )
    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.mlflow.log_metrics",
        lambda metrics: seen.update(metrics=metrics),
    )
    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.mlflow.set_tags", lambda tags: seen.update(tags=tags)
    )
    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.mlflow.log_artifact", lambda path: seen.update(artifact=path)
    )

    def refuse_register(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("registered")

    monkeypatch.setattr(
        "endzone.modeling.holdout_eval.mlflow.register_model",
        refuse_register,
        raising=False,
    )
    report: dict[str, object] = {
        "parameters": {
            "lookback": 10,
            "seed": 42,
            "max_iter": 1000,
            "n_bootstrap": 1000,
            "bootstrap_seed": 42,
        },
        "pooled": {
            "rows": [
                {"feature_set": SELECTED_FEATURE_SET, "log_loss": 0.6},
                {"feature_set": REFERENCE_FEATURE_SET, "log_loss": 0.7},
                {"feature_set": "market", "log_loss": 0.5},
            ]
        },
    }
    log_holdout_run(report, artifact_path=Path("holdout.json"))
    assert seen["experiment"] == HOLDOUT_EXPERIMENT_NAME
    assert seen["tags"]["registered"] == "false"  # type: ignore[index]
    assert seen["tags"]["promoted"] == "false"  # type: ignore[index]
    assert seen["metrics"]["log_loss_selected"] == 0.6  # type: ignore[index]


@pytest.mark.cross_platform
def test_frozen_holdout_report_is_tracked() -> None:
    """The one warehouse report is in the tree, so a clone refuses a second run."""
    json_path: Path = _REPO_ROOT / "docs" / "experiments" / "reports" / HOLDOUT_JSON_NAME
    markdown_path: Path = _REPO_ROOT / "docs" / "experiments" / "reports" / HOLDOUT_MARKDOWN_NAME
    assert json_path.is_file()
    assert markdown_path.is_file()
    ignored: subprocess.CompletedProcess[str] = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "-q",
            "docs/experiments/reports/holdout_2024_2025.json",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 1
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "docs/experiments/reports/holdout_2024_2025.json",
            "docs/experiments/reports/holdout_2024_2025.md",
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.cross_platform
def test_make_and_dev_ps1_call_the_holdout_script() -> None:
    """Both entrypoints run the same command and do not hard-code a Windows path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("holdout:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/run_holdout.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert '"holdout" { Invoke-Holdout }' in script
    assert "python scripts/run_holdout.py" in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script
    assert "\u2014" not in makefile
    assert "\u2013" not in makefile


def test_module_entry_accepts_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """python scripts/run_holdout.py is the same main the tests call."""
    monkeypatch.setattr(sys, "argv", ["run_holdout.py", "--allow-overwrite"])
    monkeypatch.setattr(
        "scripts.run_holdout.run_holdout_from_warehouse",
        lambda: {"metadata": {}},
    )
    monkeypatch.setattr(
        "scripts.run_holdout.write_holdout_reports",
        lambda _report, directory: {
            "json": directory / HOLDOUT_JSON_NAME,
            "markdown": directory / HOLDOUT_MARKDOWN_NAME,
        },
    )
    monkeypatch.setattr("scripts.run_holdout.log_holdout_run", lambda *_args, **_kwargs: None)
    (tmp_path / HOLDOUT_JSON_NAME).write_text("{}\n", encoding="utf-8")
    main(report_dir=tmp_path)
