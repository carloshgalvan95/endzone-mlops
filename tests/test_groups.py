"""The group-logistic candidates stay on the locked 2016-2023 recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.opponent import ELO_MOV_DIFF_COLUMN
from endzone.features.pregame import PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ
from endzone.features.qbelo import MODEL_ELO_MOV, rating_win_probability
from endzone.modeling.cv import (
    RollingOriginConfig,
    make_rolling_origin_config,
    rolling_origin_cv,
)
from endzone.modeling.groups import (
    GROUP_COLUMNS,
    GROUP_LOGISTIC_C,
    GROUP_NAMES,
    HOLDOUT_REFUSED_NAMES,
    group_logistic_fold_predictor,
    logit_mean_probabilities,
    verify_group_partition,
)
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.models import (
    MODEL_GROUP_MEAN,
    MODEL_GROUP_MEAN_DEF,
    fit_logistic_pipeline,
)
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.train import groups_challenger_argv
from scripts.train_groups import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_context_baselines,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_groups import (
    main as train_groups_main,
)
from tests.test_ensembles import _with_schedule
from tests.test_h6 import _multiweek_slate
from tests.test_pkg_core import _starter_tables
from tests.test_qbelo538 import _empty_injuries

FEATURE_SET_H2_FULL = "h2_full"
FEATURE_SET_PKG_CORE_INJ_L2 = "pkg_core_inj_l2"


def test_group_partition_is_exact_and_disjoint() -> None:
    """The four groups are 36, 4, 1, and 6 columns and cover the defender."""
    groups: dict[str, tuple[str, ...]] = verify_group_partition()
    assert tuple(groups) == GROUP_NAMES
    assert [len(groups[name]) for name in GROUP_NAMES] == [36, 4, 1, 6]
    assert sum(len(columns) for columns in groups.values()) == 47
    assert len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ) == 47
    flat: list[str] = [column for columns in groups.values() for column in columns]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ)
    assert groups["qb"] == (
        "home_qb_rating",
        "away_qb_rating",
        "home_qb_rating_minus_team_pass_epa",
        "away_qb_rating_minus_team_pass_epa",
    )
    assert groups["ratings"] == ("elo_mov_diff",)
    assert groups["injuries_rest"][-2:] == ("home_rest_days", "away_rest_days")
    assert "home_recent_point_diff" in groups["epa_form"]
    assert "home_rest_days" not in groups["epa_form"]


def test_logit_means_use_equal_weights_and_clip_first() -> None:
    """group_mean is the 4-way logit average. group_mean_def is the 2-way one."""
    members: list[np.ndarray] = [
        np.array([0.2, 0.0]),
        np.array([0.4, 1.0]),
        np.array([0.6, 0.2]),
        np.array([0.8, 0.9]),
    ]
    mixed: np.ndarray = logit_mean_probabilities(members)
    clipped: list[np.ndarray] = [np.clip(item, 1e-6, 1.0 - 1e-6) for item in members]
    logits: np.ndarray = np.mean(
        [np.log(item / (1.0 - item)) for item in clipped],
        axis=0,
    )
    expected: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(mixed, expected)
    assert np.isfinite(mixed).all()
    pair: np.ndarray = logit_mean_probabilities([mixed, np.array([0.3, 0.7])])
    left: np.ndarray = np.clip(mixed, 1e-6, 1.0 - 1e-6)
    right: np.ndarray = np.clip(np.array([0.3, 0.7]), 1e-6, 1.0 - 1e-6)
    pair_logit: np.ndarray = 0.5 * (np.log(left / (1.0 - left)) + np.log(right / (1.0 - right)))
    np.testing.assert_allclose(pair, 1.0 / (1.0 + np.exp(-pair_logit)))


def test_eval_labels_do_not_change_the_training_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flipped evaluation label does not change the rows passed to the fit."""
    frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["t1", "t2", "t3", "t4", "e1", "e2"],
            "gameday": [
                "2016-09-11",
                "2016-09-18",
                "2016-09-25",
                "2016-10-02",
                "2017-09-10",
                "2017-09-17",
            ],
            "season": [2016, 2016, 2016, 2016, 2017, 2017],
            "elo_mov_diff": [10.0, -20.0, 5.0, -4.0, 12.0, -8.0],
        }
    )
    target: pd.Series = pd.Series([1, 0, 1, 0, 1, 0])
    train: pd.DataFrame = frame.loc[frame["season"] == 2016].reset_index(drop=True)
    evaluation: pd.DataFrame = frame.loc[frame["season"] == 2017].reset_index(drop=True)
    train_target: pd.Series = target.loc[frame["season"] == 2016].reset_index(drop=True)
    eval_target: pd.Series = target.loc[frame["season"] == 2017].reset_index(drop=True)
    calls: list[dict[str, object]] = []
    real = fit_logistic_pipeline

    def spy(x_train: pd.DataFrame, y_train: pd.Series, **kwargs: object):
        calls.append(
            {
                "ids": [str(game_id) for game_id in x_train["game_id"]],
                "columns": [column for column in x_train.columns if column != "game_id"],
                "y": np.asarray(y_train, dtype=float).copy(),
                "C": kwargs.get("C"),
            }
        )
        return real(x_train, y_train, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("endzone.modeling.groups.fit_logistic_pipeline", spy)
    predictor = group_logistic_fold_predictor(
        GROUP_COLUMNS["ratings"], max_iter=50, random_state=42
    )
    first = predictor(train, train_target, evaluation, eval_target)
    second = predictor(train, train_target, evaluation, pd.Series(1 - eval_target.to_numpy()))
    assert len(calls) == 2
    assert calls[0]["ids"] == ["t1", "t2", "t3", "t4"]
    assert "e1" not in calls[0]["ids"]
    assert calls[0]["C"] == pytest.approx(GROUP_LOGISTIC_C)
    assert "elo_mov_diff" in calls[0]["columns"]
    np.testing.assert_allclose(calls[0]["y"], calls[1]["y"])  # type: ignore[arg-type]
    np.testing.assert_allclose(first, second)


def test_scored_folds_exclude_2024_and_later() -> None:
    """A later season can sit in the frame and still stay out of every fold."""
    frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["t1", "t2", "t3", "t4", "e1", "e2", "future"],
            "gameday": [
                "2016-09-11",
                "2016-09-18",
                "2016-09-25",
                "2016-10-02",
                "2017-09-10",
                "2017-09-17",
                "2024-09-08",
            ],
            "season": [2016, 2016, 2016, 2016, 2017, 2017, 2024],
            "elo_mov_diff": [10.0, -20.0, 5.0, -4.0, 12.0, -8.0, 400.0],
        }
    )
    target: pd.Series = pd.Series([1, 0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="2024"):
        make_rolling_origin_config(cv_start_season=2016, cv_end_season=2024)
    predictor = group_logistic_fold_predictor(
        GROUP_COLUMNS["ratings"], max_iter=50, random_state=42
    )
    scored: dict[str, object] = rolling_origin_cv(
        frame,
        target,
        predictor,
        RollingOriginConfig(cv_start_season=2017, cv_end_season=2017, n_bootstrap=0),
    )
    oof: pd.DataFrame = scored["oof"]  # type: ignore[assignment]
    folds: list[dict[str, object]] = scored["folds"]  # type: ignore[assignment]
    assert set(int(season) for season in oof["season"]) == {2017}
    assert int(oof["season"].max()) < 2024
    assert folds[0]["train_seasons"] == [2016]
    assert "future" not in set(oof["game_id"].astype(str))
    without: pd.DataFrame = frame.loc[frame["game_id"] != "future"].reset_index(drop=True)
    without_target: pd.Series = target.loc[frame["game_id"] != "future"].reset_index(drop=True)
    again: dict[str, object] = rolling_origin_cv(
        without,
        without_target,
        predictor,
        RollingOriginConfig(cv_start_season=2017, cv_end_season=2017, n_bootstrap=0),
    )
    np.testing.assert_allclose(
        oof["y_proba"].to_numpy(dtype=float),
        again["oof"]["y_proba"].to_numpy(dtype=float),  # type: ignore[index]
    )


def test_groups_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023 and the two fixed candidates."""
    argv: list[str] = groups_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--hypothesis") + 1] == "groups"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_groups"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "groups")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_groups"
    assert spec.report_profile == "groups"
    assert spec.logistic_feature_sets == ("h2_full", "pkg_core_inj_l2")
    assert [row.model for row in spec.model_specs] == [
        "logistic",
        "logistic",
        MODEL_GROUP_MEAN,
        MODEL_GROUP_MEAN_DEF,
    ]
    assert spec.note.endswith("No model is promoted from this run.")
    fixed: dict[tuple[str, str], float | None] = {
        (row.model, row.feature_set): row.fixed_c for row in spec.model_specs
    }
    assert fixed[("logistic", "h2_full")] is None
    assert fixed[("logistic", "pkg_core_inj_l2")] == pytest.approx(0.1)
    assert fixed[(MODEL_GROUP_MEAN, "pkg_core_inj_l2")] is None
    assert locked_logistic_h2_full_n() == 2191


def test_holdout_refuses_group_names_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025."""
    for name in (
        MODEL_GROUP_MEAN,
        MODEL_GROUP_MEAN_DEF,
        "epa_form",
        "qb",
        "ratings",
        "injuries_rest",
    ):
        assert name in HOLDOUT_REFUSED_NAMES
        as_model: argparse.Namespace = argparse.Namespace(model=name, feature_set="current")
        as_feature: argparse.Namespace = argparse.Namespace(feature_set=name)
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(as_model)
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(as_feature)


def test_groups_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_groups.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_groups_main()


@pytest.mark.cross_platform
def test_groups_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-groups:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_groups.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_groups.py" in script
    assert '"train-groups" { Invoke-TrainGroups }' in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script


def _write_oof(path: Path, frame: pd.DataFrame) -> None:
    """Write one OOF file as UTF-8."""
    frame.to_csv(path, index=False, encoding="utf-8")


@pytest.mark.cross_platform
def test_written_groups_cohort_reads_utf8(tmp_path: Path) -> None:
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
        "oof_group_mean_pkg_core_inj_l2.csv",
        "oof_group_mean_def_pkg_core_inj_l2.csv",
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
    assert "groups cohort" in cohort_message(summary["n"], 2191)


def test_group_report_scores_each_candidate_alone() -> None:
    """The averages match the logit means. The four groups are not candidates."""
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
        hypothesis="groups",
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
    grouped: dict[str, object] = by_key[(MODEL_GROUP_MEAN, FEATURE_SET_PKG_CORE_INJ_L2)]
    blended: dict[str, object] = by_key[(MODEL_GROUP_MEAN_DEF, FEATURE_SET_PKG_CORE_INJ_L2)]
    assert defender["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert grouped["params"]["C"] == pytest.approx(GROUP_LOGISTIC_C)  # type: ignore[index]
    assert grouped["params"]["weights"] == "0.25,0.25,0.25,0.25"  # type: ignore[index]
    assert grouped["params"]["learned_weights"] is False  # type: ignore[index]
    assert blended["params"]["weights"] == "0.5,0.5"  # type: ignore[index]
    assert blended["params"]["learned_weights"] is False  # type: ignore[index]
    members: list[dict[str, object]] = list(grouped["group_members"])  # type: ignore[arg-type]
    assert [str(member["model"]) for member in members] == list(GROUP_NAMES)
    order: list[str] = [str(game_id) for game_id in grouped["oof"]["game_id"]]  # type: ignore[index]
    member_probabilities: list[np.ndarray] = []
    for member in members:
        member_oof: pd.DataFrame = member["oof"]  # type: ignore[assignment]
        assert int(member_oof["season"].max()) < 2024
        indexed: pd.DataFrame = member_oof.assign(
            game_id=member_oof["game_id"].astype(str)
        ).set_index("game_id")
        member_probabilities.append(indexed.loc[order, "y_proba"].to_numpy(dtype=float))
    np.testing.assert_allclose(
        grouped["oof"]["y_proba"].to_numpy(dtype=float),  # type: ignore[index]
        logit_mean_probabilities(member_probabilities),
    )
    defender_oof: pd.DataFrame = defender["oof"]  # type: ignore[assignment]
    defender_indexed: pd.DataFrame = defender_oof.assign(
        game_id=defender_oof["game_id"].astype(str)
    ).set_index("game_id")
    blend_order: list[str] = [str(game_id) for game_id in blended["oof"]["game_id"]]  # type: ignore[index]
    grouped_indexed: pd.DataFrame = (
        grouped["oof"]
        .assign(  # type: ignore[union-attr]
            game_id=grouped["oof"]["game_id"].astype(str)  # type: ignore[index]
        )
        .set_index("game_id")
    )
    np.testing.assert_allclose(
        blended["oof"]["y_proba"].to_numpy(dtype=float),  # type: ignore[index]
        logit_mean_probabilities(
            [
                grouped_indexed.loc[blend_order, "y_proba"].to_numpy(dtype=float),
                defender_indexed.loc[blend_order, "y_proba"].to_numpy(dtype=float),
            ]
        ),
    )
    for fold in grouped["folds"]:  # type: ignore[union-attr]
        assert int(fold["eval_season"]) < 2024
        assert max(int(season) for season in fold["train_seasons"]) < int(fold["eval_season"])
    diagnostics: pd.DataFrame = report["group_diagnostics"]  # type: ignore[assignment]
    group_rows: pd.DataFrame = diagnostics.loc[diagnostics["row_kind"] == "group"]
    assert list(group_rows["group"]) == list(GROUP_NAMES)
    assert group_rows["log_loss"].notna().all()
    summary: pd.DataFrame = diagnostics.loc[diagnostics["row_kind"] == "summary"]
    assert len(summary) == 1
    assert summary.iloc[0]["logit_correlation"] == pytest.approx(
        float(
            np.corrcoef(
                np.log(
                    np.clip(
                        grouped_indexed.loc[order, "y_proba"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
                    )
                    / (
                        1
                        - np.clip(
                            grouped_indexed.loc[order, "y_proba"].to_numpy(dtype=float),
                            1e-6,
                            1 - 1e-6,
                        )
                    )
                ),
                np.log(
                    np.clip(
                        defender_indexed.loc[order, "y_proba"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
                    )
                    / (
                        1
                        - np.clip(
                            defender_indexed.loc[order, "y_proba"].to_numpy(dtype=float),
                            1e-6,
                            1 - 1e-6,
                        )
                    )
                ),
            )[0, 1]
        )
    )
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] in {"accepted", "rejected", "inconclusive"}
    for model_name in (MODEL_GROUP_MEAN, MODEL_GROUP_MEAN_DEF):
        assert decisions[(model_name, FEATURE_SET_PKG_CORE_INJ_L2)] in {
            "accepted",
            "rejected",
            "inconclusive",
        }
    for name in GROUP_NAMES:
        assert name not in {model for model, _feature_set in decisions}
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[(MODEL_ELO_MOV, MODEL_ELO_MOV)] == "baseline"
    elo_oof: pd.DataFrame = by_key[(MODEL_ELO_MOV, MODEL_ELO_MOV)]["oof"]  # type: ignore[assignment]
    features_for_gap: pd.DataFrame = report["dataset_frame"]  # type: ignore[assignment]
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
    assert {MODEL_GROUP_MEAN, MODEL_GROUP_MEAN_DEF} <= set(calibration["model"])
    assert {MODEL_GROUP_MEAN, MODEL_GROUP_MEAN_DEF} <= set(information["model"])
    assert "## Group diagnostics" in str(report["markdown"])
