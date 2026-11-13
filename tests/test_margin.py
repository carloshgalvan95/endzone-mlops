"""The margin candidates stay on the locked 2016-2023 recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.opponent import ELO_MOV_DIFF_COLUMN
from endzone.features.qbelo import MODEL_ELO_MOV, rating_win_probability
from endzone.modeling.cv import (
    RollingOriginConfig,
    make_rolling_origin_config,
    rolling_origin_cv,
)
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.margin import (
    MARGIN_RIDGE_ALPHA,
    home_margins,
    logit_correlation,
    logit_mean_probabilities,
    margin_ridge_fold_predictor,
    margin_win_probability,
    training_residual_sigma,
)
from endzone.modeling.models import (
    MODEL_MARGIN_MEAN,
    MODEL_MARGIN_RIDGE,
    fit_ridge_pipeline,
)
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.points_eval import standard_normal_cdf
from endzone.modeling.train import margin_challenger_argv
from scripts.train_margin import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_context_baselines,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_margin import (
    main as train_margin_main,
)
from tests.test_ensembles import _with_schedule
from tests.test_h6 import _multiweek_slate
from tests.test_pkg_core import _starter_tables
from tests.test_qbelo538 import _empty_injuries

FEATURE_SET_H2_FULL = "h2_full"
FEATURE_SET_PKG_CORE_INJ_L2 = "pkg_core_inj_l2"


def _toy_schedule() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Four training games, two evaluation games, and one 2024 game."""
    games: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["t1", "t2", "t3", "t4", "e1", "e2", "future"],
            "season": [2016, 2016, 2016, 2016, 2017, 2017, 2024],
            "gameday": [
                "2016-09-11",
                "2016-09-18",
                "2016-09-25",
                "2016-10-02",
                "2017-09-10",
                "2017-09-17",
                "2024-09-08",
            ],
            "home_score": [27, 10, 31, 20, 24, 13, 70],
            "away_score": [14, 24, 17, 20, 10, 27, 0],
        }
    )
    frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "gameday": games["gameday"],
            "season": games["season"],
            "signal": [1.0, -1.2, 0.4, 0.1, 0.8, -0.7, 9.0],
        }
    )
    target: pd.Series = pd.Series([1, 0, 1, 0, 1, 0, 1])
    return games, frame, target


def test_margin_probability_is_clipped_phi() -> None:
    """p is Phi(mu / sigma), clipped to the logit bounds."""
    assert margin_win_probability(np.array([0.0]), 12.0)[0] == pytest.approx(0.5)
    assert margin_win_probability(np.array([10.0]), 10.0)[0] == pytest.approx(
        standard_normal_cdf(1.0)
    )
    clipped: np.ndarray = margin_win_probability(np.array([100.0, -100.0]), 1.0)
    assert clipped[0] == pytest.approx(1.0 - 1e-6)
    assert clipped[1] == pytest.approx(1e-6)


def test_logit_mean_uses_equal_weights_and_clips_first() -> None:
    """The mean is the sigmoid of the average logit, after the 1e-6 clip."""
    left: np.ndarray = np.array([0.2, 0.0])
    right: np.ndarray = np.array([0.8, 1.0])
    mixed: np.ndarray = logit_mean_probabilities(left, right)
    clipped_left: np.ndarray = np.clip(left, 1e-6, 1.0 - 1e-6)
    clipped_right: np.ndarray = np.clip(right, 1e-6, 1.0 - 1e-6)
    logits: np.ndarray = 0.5 * (
        np.log(clipped_left / (1.0 - clipped_left)) + np.log(clipped_right / (1.0 - clipped_right))
    )
    expected: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(mixed, expected)
    assert mixed[1] == pytest.approx(0.5)
    assert np.isfinite(mixed[1])


def test_sigma_uses_training_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval labels and eval scores do not change sigma or the training fit."""
    games, frame, target = _toy_schedule()
    train: pd.DataFrame = frame.loc[frame["season"] == 2016].reset_index(drop=True)
    evaluation: pd.DataFrame = frame.loc[frame["season"] == 2017].reset_index(drop=True)
    train_target: pd.Series = target.loc[frame["season"] == 2016].reset_index(drop=True)
    eval_target: pd.Series = target.loc[frame["season"] == 2017].reset_index(drop=True)
    calls: list[dict[str, object]] = []
    real = fit_ridge_pipeline

    def spy(x_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0):
        calls.append(
            {
                "ids": [str(game_id) for game_id in x_train["game_id"]],
                "y": np.asarray(y_train, dtype=float).copy(),
                "alpha": alpha,
            }
        )
        return real(x_train, y_train, alpha=alpha)

    monkeypatch.setattr("endzone.modeling.margin.fit_ridge_pipeline", spy)
    predictor = margin_ridge_fold_predictor(games)
    first = predictor(train, train_target, evaluation, eval_target)
    flipped: pd.Series = pd.Series(1 - eval_target.to_numpy())
    second = predictor(train, train_target, evaluation, flipped)
    assert len(calls) == 2
    assert calls[0]["ids"] == ["t1", "t2", "t3", "t4"]
    assert "e1" not in calls[0]["ids"]
    assert "future" not in calls[0]["ids"]
    assert calls[0]["alpha"] == pytest.approx(MARGIN_RIDGE_ALPHA)
    np.testing.assert_allclose(
        calls[0]["y"],  # type: ignore[arg-type]
        home_margins(games).loc[["t1", "t2", "t3", "t4"]].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        calls[0]["y"],  # type: ignore[arg-type]
        calls[1]["y"],  # type: ignore[arg-type]
    )
    assert first.extras["sigma"] == pytest.approx(second.extras["sigma"])
    np.testing.assert_allclose(first.probabilities, second.probabilities)
    hand_pipe = real(
        train,
        pd.Series(calls[0]["y"], index=train.index),  # type: ignore[arg-type]
        alpha=MARGIN_RIDGE_ALPHA,
    )
    columns: list[str] = ["signal"]
    train_hat: np.ndarray = np.asarray(hand_pipe.predict(train[columns]), dtype=float)
    assert first.extras["sigma"] == pytest.approx(
        training_residual_sigma(np.asarray(calls[0]["y"], dtype=float), train_hat)
    )
    changed: pd.DataFrame = games.copy()
    changed.loc[changed["game_id"] == "e1", "home_score"] = 3
    changed_prediction = margin_ridge_fold_predictor(changed)(
        train,
        train_target,
        evaluation,
        eval_target,
    )
    assert changed_prediction.extras["sigma"] == pytest.approx(first.extras["sigma"])
    assert changed_prediction.extras["margin_mae"] != pytest.approx(first.extras["margin_mae"])


def test_scored_folds_exclude_2024_and_later() -> None:
    """A later season can sit in the frame and still stay out of every fold."""
    games, frame, target = _toy_schedule()
    with pytest.raises(ValueError, match="2024"):
        make_rolling_origin_config(cv_start_season=2016, cv_end_season=2024)
    scored: dict[str, object] = rolling_origin_cv(
        frame,
        target,
        margin_ridge_fold_predictor(games),
        RollingOriginConfig(cv_start_season=2017, cv_end_season=2017, n_bootstrap=0),
    )
    oof: pd.DataFrame = scored["oof"]  # type: ignore[assignment]
    folds: list[dict[str, object]] = scored["folds"]  # type: ignore[assignment]
    assert set(int(season) for season in oof["season"]) == {2017}
    assert int(oof["season"].max()) < 2024
    assert folds[0]["train_seasons"] == [2016]
    assert "future" not in set(oof["game_id"].astype(str))
    without_future: pd.DataFrame = frame.loc[frame["game_id"] != "future"].reset_index(drop=True)
    without_target: pd.Series = target.loc[frame["game_id"] != "future"].reset_index(drop=True)
    again: dict[str, object] = rolling_origin_cv(
        without_future,
        without_target,
        margin_ridge_fold_predictor(games.loc[games["game_id"] != "future"]),
        RollingOriginConfig(cv_start_season=2017, cv_end_season=2017, n_bootstrap=0),
    )
    assert folds[0]["sigma"] == pytest.approx(again["folds"][0]["sigma"])  # type: ignore[index]


def test_margin_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023 and the two fixed candidates."""
    argv: list[str] = margin_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "margin"
    assert argv[argv.index("--reference-model") + 1] == "logistic"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_margin"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "margin")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_margin"
    assert spec.report_profile == "margin"
    assert spec.reference_feature_set == FEATURE_SET_PKG_CORE_INJ_L2
    assert spec.logistic_feature_sets == ("h2_full", "pkg_core_inj_l2")
    assert [row.model for row in spec.model_specs] == [
        "logistic",
        "logistic",
        MODEL_MARGIN_RIDGE,
        MODEL_MARGIN_MEAN,
    ]
    assert spec.note.endswith("No model is promoted from this run.")
    fixed: dict[tuple[str, str], float | None] = {
        (row.model, row.feature_set): row.fixed_c for row in spec.model_specs
    }
    assert fixed[("logistic", "h2_full")] is None
    assert fixed[("logistic", "pkg_core_inj_l2")] == pytest.approx(0.1)
    assert fixed[(MODEL_MARGIN_RIDGE, "pkg_core_inj_l2")] is None
    assert fixed[(MODEL_MARGIN_MEAN, "pkg_core_inj_l2")] is None
    assert locked_logistic_h2_full_n() == 2191


def test_margin_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_margin.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_margin_main()


@pytest.mark.cross_platform
def test_margin_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-margin:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_margin.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_margin.py" in script
    assert '"train-margin" { Invoke-TrainMargin }' in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script


def _write_oof(path: Path, frame: pd.DataFrame) -> None:
    """Write one OOF file as UTF-8."""
    frame.to_csv(path, index=False, encoding="utf-8")


@pytest.mark.cross_platform
def test_written_margin_cohort_reads_utf8(tmp_path: Path) -> None:
    """The command checks the OOF files and the context labels."""
    reference: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2018, 2019],
            "y_true": [1, 0],
            "y_proba": [0.6, 0.4],
        }
    )
    _write_oof(tmp_path / "oof_logistic_pkg_core_inj_l2.csv", reference)
    for file_name in (
        "oof_logistic_h2_full.csv",
        "oof_elo_elo.csv",
        "oof_elo_mov_elo_mov.csv",
        "oof_margin_ridge_pkg_core_inj_pkg_core_inj_l2.csv",
        "oof_margin_mean_pkg_core_inj_l2.csv",
    ):
        _write_oof(tmp_path / file_name, reference)
    acceptance: pd.DataFrame = pd.DataFrame(
        [
            {"model": "market", "feature_set": "market", "decision": "benchmark"},
            {"model": "elo_mov", "feature_set": "elo_mov", "decision": "baseline"},
        ]
    )
    acceptance.to_csv(tmp_path / "acceptance.csv", index=False, encoding="utf-8")
    summary: dict[str, int] = require_written_cohort(tmp_path)
    require_market_benchmark(tmp_path)
    require_context_baselines(tmp_path)
    assert summary["n"] == 2
    assert "margin cohort" in cohort_message(summary["n"], 2191)


def test_margin_report_scores_each_candidate_alone() -> None:
    """The mean is the logit average. Sigma and the correlation are descriptive."""
    games, epa = _multiweek_slate(weeks=6)
    games = _with_schedule(games)
    starters, dropbacks = _starter_tables(games)
    empty_starters, empty_reports = _empty_injuries()
    report: dict[str, object] = run_phase1_baselines(
        games,
        epa,
        config=RollingOriginConfig(
            cv_start_season=2017,
            cv_end_season=2017,
            n_bootstrap=8,
            bootstrap_seed=42,
        ),
        lookback_games=4,
        logistic_c=1.0,
        logistic_max_iter=200,
        phase=2,
        hypothesis="margin",
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        depth_chart_starters=empty_starters,
        injury_reports=empty_reports,
        random_state=42,
    )
    by_key: dict[tuple[str, str], dict[str, object]] = {
        (str(row["model"]), str(row["feature_set"])): row
        for row in report["results"]  # type: ignore[index]
    }
    defender: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)]
    ridge: dict[str, object] = by_key[(MODEL_MARGIN_RIDGE, FEATURE_SET_PKG_CORE_INJ_L2)]
    averaged: dict[str, object] = by_key[(MODEL_MARGIN_MEAN, FEATURE_SET_PKG_CORE_INJ_L2)]
    assert defender["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert ridge["params"]["ridge_alpha"] == pytest.approx(MARGIN_RIDGE_ALPHA)  # type: ignore[index]
    assert averaged["params"]["weights"] == "0.5,0.5"  # type: ignore[index]
    assert averaged["params"]["learned_weights"] is False  # type: ignore[index]
    ridge_oof: pd.DataFrame = ridge["oof"]  # type: ignore[assignment]
    defender_oof: pd.DataFrame = defender["oof"]  # type: ignore[assignment]
    mean_oof: pd.DataFrame = averaged["oof"]  # type: ignore[assignment]
    assert int(ridge_oof["season"].max()) < 2024
    assert int(mean_oof["season"].max()) < 2024
    for fold in ridge["folds"]:  # type: ignore[union-attr]
        assert int(fold["eval_season"]) < 2024
        assert max(int(season) for season in fold["train_seasons"]) < int(fold["eval_season"])
        assert float(fold["sigma"]) > 0.0
    order: list[str] = [str(game_id) for game_id in mean_oof["game_id"]]
    ridge_by_id: pd.DataFrame = ridge_oof.assign(
        game_id=ridge_oof["game_id"].astype(str)
    ).set_index("game_id")
    defender_by_id: pd.DataFrame = defender_oof.assign(
        game_id=defender_oof["game_id"].astype(str)
    ).set_index("game_id")
    expected: np.ndarray = logit_mean_probabilities(
        ridge_by_id.loc[order, "y_proba"].to_numpy(dtype=float),
        defender_by_id.loc[order, "y_proba"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(mean_oof["y_proba"].to_numpy(dtype=float), expected)
    diagnostics: pd.DataFrame = report["margin_diagnostics"]  # type: ignore[assignment]
    summary: pd.DataFrame = diagnostics.loc[diagnostics["row_kind"] == "summary"]
    assert len(summary) == 1
    assert summary.iloc[0]["logit_correlation"] == pytest.approx(
        logit_correlation(
            ridge_by_id.loc[order, "y_proba"].to_numpy(dtype=float),
            defender_by_id.loc[order, "y_proba"].to_numpy(dtype=float),
        )
    )
    assert float(summary.iloc[0]["margin_mae"]) > 0.0
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] in {"accepted", "rejected", "inconclusive"}
    for model_name in (MODEL_MARGIN_RIDGE, MODEL_MARGIN_MEAN):
        assert decisions[(model_name, FEATURE_SET_PKG_CORE_INJ_L2)] in {
            "accepted",
            "rejected",
            "inconclusive",
        }
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[(MODEL_ELO_MOV, MODEL_ELO_MOV)] == "baseline"
    elo_oof: pd.DataFrame = by_key[(MODEL_ELO_MOV, MODEL_ELO_MOV)]["oof"]  # type: ignore[assignment]
    features_for_gap: pd.DataFrame = report["dataset_frame"]  # type: ignore[assignment]
    assert int(elo_oof["season"].max()) < 2024
    gap: pd.DataFrame = elo_oof.merge(
        features_for_gap.loc[:, ["game_id", ELO_MOV_DIFF_COLUMN]],
        on="game_id",
        how="left",
    )
    np.testing.assert_allclose(
        gap["y_proba"].to_numpy(dtype=float),
        rating_win_probability(gap[ELO_MOV_DIFF_COLUMN].to_numpy(dtype=float)),
    )
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert {MODEL_MARGIN_RIDGE, MODEL_MARGIN_MEAN} <= set(calibration["model"])
    assert {MODEL_MARGIN_RIDGE, MODEL_MARGIN_MEAN} <= set(information["model"])
    assert "## Margin diagnostics" in str(report["markdown"])
