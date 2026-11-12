"""The booster and the mean stay on the locked 2016-2023 recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from endzone.features.pregame import FEATURE_SET_H2_FULL, FEATURE_SET_PKG_CORE_INJ_L2
from endzone.modeling.cv import RollingOriginConfig, rolling_origin_cv
from endzone.modeling.ensembles import (
    HIST_GRADIENT_BOOSTING_PARAMS,
    MODEL_ENS_MEAN,
    MODEL_GBM_PKG_CORE_INJ,
    ensemble_mean_result,
    fit_hist_gradient_boosting,
    hist_gbm_fold_predictor,
    hist_gradient_boosting_estimator,
    mean_member_oof,
    mean_probabilities,
    positive_class_probability,
)
from endzone.modeling.hypotheses import CvModelSpec, experiment_for
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.train import ensembles_challenger_argv
from scripts.train_ensembles import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_ensembles import (
    main as train_ensembles_main,
)
from tests.test_h6 import _multiweek_slate
from tests.test_pkg_core import _starter_tables


def _with_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Add the kickoff columns the absence count reads."""
    frame: pd.DataFrame = games.copy()
    parts: pd.Series = frame["game_id"].astype(str).str.split("-", expand=True)
    frame["week"] = parts[1].astype(int) + 1
    frame["game_type"] = "REG"
    frame["gametime"] = "13:00"
    frame["home_team"] = frame["home_team_franchise"]
    frame["away_team"] = frame["away_team_franchise"]
    return frame


def _labeled_frame(
    seasons: tuple[int, ...], rows_per_season: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Small numeric frame. One column matches the label on even rows."""
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    for season in seasons:
        for index in range(rows_per_season):
            label: int = index % 2
            rows.append(
                {
                    "game_id": f"{season}-{index}",
                    "season": season,
                    "gameday": f"{season}-09-{index + 1:02d}",
                    "signal": float(label),
                }
            )
            labels.append(label)
    frame: pd.DataFrame = pd.DataFrame(rows)
    target: pd.Series = pd.Series(labels, dtype=int)
    return frame, target


def test_ensembles_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023 and the two fixed candidates."""
    argv: list[str] = ensembles_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "ensembles"
    assert argv[argv.index("--reference-model") + 1] == "logistic"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_ensembles"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "ensembles")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_ensembles"
    assert spec.tag == "ensembles"
    assert spec.report_profile == "ensembles"
    assert spec.reference_model == "logistic"
    assert spec.reference_feature_set == FEATURE_SET_PKG_CORE_INJ_L2
    assert spec.output_dir == "reports/phase2_ensembles"
    assert spec.logistic_feature_sets == ("h2_full", "pkg_core_inj_l2")
    assert spec.note.endswith("No model is promoted from this run.")
    assert [row.model for row in spec.model_specs] == [
        "logistic",
        "logistic",
        MODEL_GBM_PKG_CORE_INJ,
        MODEL_ENS_MEAN,
    ]
    fixed: dict[str, float | None] = {
        row.feature_set: row.fixed_c for row in spec.model_specs if row.model == "logistic"
    }
    assert fixed["h2_full"] is None
    assert fixed["pkg_core_inj_l2"] == pytest.approx(0.1)
    params = HIST_GRADIENT_BOOSTING_PARAMS
    assert params.learning_rate == pytest.approx(0.05)
    assert params.max_iter == 300
    assert params.max_leaf_nodes == 15
    assert params.min_samples_leaf == 50
    assert params.l2_regularization == pytest.approx(1.0)
    assert params.max_features == pytest.approx(1.0)
    assert params.random_state == 42
    assert params.early_stopping is False
    estimator: HistGradientBoostingClassifier = hist_gradient_boosting_estimator()
    assert estimator.early_stopping is False
    assert estimator.random_state == 42
    assert locked_logistic_h2_full_n() == 2191


def test_ensembles_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_ensembles.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_ensembles_main()


@pytest.mark.cross_platform
def test_ensembles_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-ensembles:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_ensembles.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_ensembles.py" in script
    assert '"train-ensembles" { Invoke-TrainEnsembles }' in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script


@pytest.mark.cross_platform
def test_written_cohort_and_market_benchmark_read_utf8(tmp_path: Path) -> None:
    """The command checks the OOF files and refuses an accepted market."""
    reference: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2018, 2019],
            "y_true": [1, 0],
            "y_proba": [0.6, 0.4],
        }
    )
    names: tuple[str, ...] = (
        "oof_logistic_pkg_core_inj_l2.csv",
        "oof_logistic_h2_full.csv",
        "oof_gbm_pkg_core_inj_pkg_core_inj_l2.csv",
        "oof_ens_mean_pkg_core_inj_l2.csv",
    )
    for name in names:
        reference.to_csv(tmp_path / name, index=False, encoding="utf-8")
    summary: dict[str, int] = require_written_cohort(tmp_path)
    assert summary == {"n": 2, "variants": 3}
    assert "not that table" in cohort_message(2, 2191)
    assert "matches the locked" in cohort_message(2191, 2191)
    acceptance: pd.DataFrame = pd.DataFrame(
        [{"model": "market", "feature_set": "market", "decision": "benchmark"}]
    )
    acceptance.to_csv(tmp_path / "acceptance.csv", index=False, encoding="utf-8")
    require_market_benchmark(tmp_path)
    acceptance.loc[0, "decision"] = "accepted"
    acceptance.to_csv(tmp_path / "acceptance.csv", index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark"):
        require_market_benchmark(tmp_path)


def test_booster_is_deterministic_and_keeps_missing_values() -> None:
    """Two fits match, and a missing value is a split the zero fill does not see."""
    rng: np.random.Generator = np.random.default_rng(1)
    n_rows: int = 120
    signal: np.ndarray = np.zeros(n_rows)
    signal[:60] = np.nan
    noise: np.ndarray = rng.normal(size=n_rows)
    frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(n_rows)],
            "season": [2016] * n_rows,
            "gameday": ["2016-09-11"] * n_rows,
            "signal": signal,
            "noise": noise,
        }
    )
    target: pd.Series = pd.Series([1] * 60 + [0] * 60)
    first: HistGradientBoostingClassifier = fit_hist_gradient_boosting(frame, target)
    second: HistGradientBoostingClassifier = fit_hist_gradient_boosting(frame, target)
    probe: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["nan", "zero"],
            "season": [2016, 2016],
            "gameday": ["2016-09-11", "2016-09-11"],
            "signal": [np.nan, 0.0],
            "noise": [0.0, 0.0],
        }
    )
    native: np.ndarray = positive_class_probability(first, probe)
    repeated: np.ndarray = positive_class_probability(second, probe)
    np.testing.assert_array_equal(native, repeated)
    assert native[0] > 0.9
    assert native[1] < 0.1
    filled: pd.DataFrame = frame.copy()
    filled["signal"] = filled["signal"].fillna(0.0)
    imputed: HistGradientBoostingClassifier = fit_hist_gradient_boosting(filled, target)
    collapsed: np.ndarray = positive_class_probability(imputed, probe)
    assert collapsed[0] == pytest.approx(collapsed[1])
    assert abs(float(collapsed[0]) - float(native[0])) > 0.2


def test_fold_fit_ignores_the_scored_season_and_its_labels() -> None:
    """Training seasons stay before the scored season. Eval labels do not move the fit."""
    frame, target = _labeled_frame((2015, 2016, 2017, 2018), rows_per_season=40)
    train_mask: pd.Series = frame["season"] < 2017
    eval_mask: pd.Series = frame["season"] == 2017
    train_frame: pd.DataFrame = frame.loc[train_mask].reset_index(drop=True)
    eval_frame: pd.DataFrame = frame.loc[eval_mask].reset_index(drop=True)
    train_target: pd.Series = target.loc[train_mask].reset_index(drop=True)
    eval_target: pd.Series = target.loc[eval_mask].reset_index(drop=True)
    flipped: pd.Series = pd.Series(1 - eval_target.to_numpy(), dtype=int)
    predictor = hist_gbm_fold_predictor()
    original: np.ndarray = predictor(train_frame, train_target, eval_frame, eval_target)
    relabeled: np.ndarray = predictor(train_frame, train_target, eval_frame, flipped)
    np.testing.assert_array_equal(original, relabeled)
    train_only: np.ndarray = positive_class_probability(
        fit_hist_gradient_boosting(train_frame, train_target),
        eval_frame,
    )
    np.testing.assert_array_equal(original, train_only)
    leaked_frame: pd.DataFrame = pd.concat([train_frame, eval_frame], ignore_index=True)
    leaked_target: pd.Series = pd.concat([train_target, pd.Series([1] * len(eval_frame))])
    leaked: np.ndarray = positive_class_probability(
        fit_hist_gradient_boosting(leaked_frame, leaked_target),
        eval_frame,
    )
    assert not np.array_equal(original, leaked)
    seen: list[list[int]] = []
    real_fit = fit_hist_gradient_boosting

    def _spy(x_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingClassifier:
        seen.append(sorted(int(season) for season in x_train["season"].unique()))
        return real_fit(x_train, y_train)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("endzone.modeling.ensembles.fit_hist_gradient_boosting", _spy)
    try:
        scored: dict[str, object] = rolling_origin_cv(
            frame,
            target,
            hist_gbm_fold_predictor(),
            RollingOriginConfig(cv_start_season=2016, cv_end_season=2017, n_bootstrap=0),
        )
    finally:
        monkeypatch.undo()
    oof: pd.DataFrame = scored["oof"]  # type: ignore[assignment]
    assert set(int(season) for season in oof["season"]) == {2016, 2017}
    assert 2018 not in set(int(season) for season in oof["season"])
    for seasons in seen:
        assert 2018 not in seasons
    for fold in scored["folds"]:  # type: ignore[union-attr]
        assert int(fold["eval_season"]) not in [int(season) for season in fold["train_seasons"]]
        assert all(int(season) < int(fold["eval_season"]) for season in fold["train_seasons"])
    assert seen[0] == [2015]
    assert seen[1] == [2015, 2016]


def test_ensemble_is_the_unweighted_mean_of_the_same_folds() -> None:
    """The mean uses member probabilities and refuses a different game list."""
    np.testing.assert_array_equal(
        mean_probabilities(np.array([0.25, 0.5]), np.array([0.75, 0.5])),
        np.array([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="probabilities"):
        mean_probabilities(np.array([0.2]), np.array([1.2]))
    logistic: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2016, 2017],
            "y_true": [1, 0],
            "y_proba": [0.25, 0.5],
        }
    )
    booster: pd.DataFrame = logistic.copy()
    booster["y_proba"] = [0.75, 0.5]
    averaged: pd.DataFrame = mean_member_oof(logistic, booster)
    np.testing.assert_array_equal(averaged["y_proba"].to_numpy(), np.array([0.5, 0.5]))
    swapped: pd.DataFrame = booster.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="different games"):
        mean_member_oof(logistic, swapped)
    folds: list[dict[str, object]] = [
        {
            "season": 2016,
            "status": "scored",
            "reason": None,
            "n_train": 4,
            "n_eval": 1,
            "train_seasons": [2015],
            "eval_season": 2016,
            "log_loss": 0.5,
            "brier": 0.2,
            "auc": 0.6,
            "accuracy": 0.5,
        }
    ]
    shared: dict[str, object] = {"random_state": 42}
    logistic_result: dict[str, object] = {
        "model": "logistic",
        "feature_set": "pkg_core_inj_l2",
        "status": "ok",
        "folds": folds,
        "oof": logistic.iloc[:1].copy(),
    }
    booster_result: dict[str, object] = {
        "model": MODEL_GBM_PKG_CORE_INJ,
        "feature_set": "pkg_core_inj_l2",
        "status": "ok",
        "folds": folds,
        "oof": booster.iloc[:1].copy(),
    }
    built: dict[str, object] = ensemble_mean_result(
        [logistic_result, booster_result],
        CvModelSpec(MODEL_ENS_MEAN, "pkg_core_inj_l2"),
        shared,
    )
    assert built["status"] == "ok"
    assert built["params"]["learned_weights"] is False  # type: ignore[index]
    assert built["params"]["stacking"] is False  # type: ignore[index]
    oof: pd.DataFrame = built["oof"]  # type: ignore[assignment]
    assert oof["y_proba"].iloc[0] == pytest.approx(0.5)
    assert built["folds"][0]["train_seasons"] == [2015]  # type: ignore[index]
    other_folds: list[dict[str, object]] = [dict(folds[0], train_seasons=[2014])]
    booster_result["folds"] = other_folds
    with pytest.raises(ValueError, match="training seasons"):
        ensemble_mean_result(
            [logistic_result, booster_result],
            CvModelSpec(MODEL_ENS_MEAN, "pkg_core_inj_l2"),
            shared,
        )


def test_ensemble_report_matches_the_member_mean() -> None:
    """The CV report's mean is the two member probabilities, with no future season."""
    games, epa = _multiweek_slate(weeks=8)
    games = _with_schedule(games)
    starters, dropbacks = _starter_tables(games)
    empty_starters: pd.DataFrame = pd.DataFrame(
        columns=["season", "week", "game_type", "team", "gsis_id", "position"]
    )
    empty_reports: pd.DataFrame = pd.DataFrame(
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
        hypothesis="ensembles",
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
    logistic: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)]
    booster: dict[str, object] = by_key[(MODEL_GBM_PKG_CORE_INJ, FEATURE_SET_PKG_CORE_INJ_L2)]
    ensemble: dict[str, object] = by_key[(MODEL_ENS_MEAN, FEATURE_SET_PKG_CORE_INJ_L2)]
    assert logistic["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert booster["params"]["learning_rate"] == pytest.approx(0.05)  # type: ignore[index]
    assert booster["params"]["max_iter"] == 300  # type: ignore[index]
    assert booster["params"]["early_stopping"] is False  # type: ignore[index]
    assert booster["params"]["missing_values"] == "native"  # type: ignore[index]
    assert ensemble["params"]["learned_weights"] is False  # type: ignore[index]
    logistic_oof: pd.DataFrame = logistic["oof"]  # type: ignore[assignment]
    booster_oof: pd.DataFrame = booster["oof"]  # type: ignore[assignment]
    ensemble_oof: pd.DataFrame = ensemble["oof"]  # type: ignore[assignment]
    expected: np.ndarray = (
        logistic_oof["y_proba"].to_numpy(dtype=float) + booster_oof["y_proba"].to_numpy(dtype=float)
    ) / 2.0
    np.testing.assert_allclose(ensemble_oof["y_proba"].to_numpy(dtype=float), expected)
    assert list(ensemble_oof["game_id"]) == list(logistic_oof["game_id"])
    assert set(int(season) for season in ensemble_oof["season"]) == {2017}
    for row in (booster, ensemble):
        for fold in row["folds"]:  # type: ignore[index]
            if fold["status"] != "scored":
                continue
            assert all(int(season) < int(fold["eval_season"]) for season in fold["train_seasons"])
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] in {"accepted", "rejected", "inconclusive"}
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[(MODEL_GBM_PKG_CORE_INJ, FEATURE_SET_PKG_CORE_INJ_L2)] in {
        "accepted",
        "rejected",
        "inconclusive",
    }
    assert decisions[(MODEL_ENS_MEAN, FEATURE_SET_PKG_CORE_INJ_L2)] in {
        "accepted",
        "rejected",
        "inconclusive",
    }
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert MODEL_GBM_PKG_CORE_INJ in set(calibration["model"])
    assert MODEL_ENS_MEAN in set(calibration["model"])
    assert MODEL_GBM_PKG_CORE_INJ in set(information["model"])
    assert MODEL_ENS_MEAN in set(information["model"])
    assert "No blend is published." in str(report["markdown"])
