"""The prior-shrink candidates stay on the locked 2016-2023 recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.opponent import ELO_MOV_DIFF_COLUMN
from endzone.features.pregame import (
    DEFENSIVE_BLEND_K,
    FEATURE_SET_H2_FULL,
    FEATURE_SET_PKG_CORE_INJ_L2,
    FEATURE_SET_PKG_CORE_INJ_L2_K39,
    FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY,
    OFFENSIVE_BLEND_K,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
    SEASON_BLEND_K,
    build_pregame_features,
    feature_columns_for,
)
from endzone.features.qbelo import MODEL_ELO_MOV, rating_win_probability
from endzone.modeling.cv import RollingOriginConfig, _logistic_fold_predictor
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.models import (
    TRAIN_SEASON_HALF_LIFE,
    fit_logistic_pipeline,
    season_decay_weights,
)
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.train import prior_shrink_challenger_argv
from scripts.train_prior_shrink import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_context_baselines,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_prior_shrink import (
    main as train_prior_shrink_main,
)
from tests.test_ensembles import _with_schedule
from tests.test_h6 import _multiweek_slate
from tests.test_pkg_core import _starter_tables
from tests.test_qbelo538 import _empty_injuries


def _frames_match(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Fail unless two feature frames have the same values, including nulls."""
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=0.0, atol=0.0)


def test_shared_blend_matches_the_defender_columns() -> None:
    """Explicit k 4 on both sides leaves every existing column unchanged."""
    games, epa = _multiweek_slate(weeks=6)
    games = _with_schedule(games)
    starters, dropbacks = _starter_tables(games)
    empty_starters, empty_reports = _empty_injuries()
    shared: dict[str, object] = {
        "lookback_games": 4,
        "expected_starters": starters,
        "qb_dropbacks": dropbacks,
        "depth_chart_starters": empty_starters,
        "injury_reports": empty_reports,
    }
    default: pd.DataFrame = build_pregame_features(games, epa, **shared)
    explicit: pd.DataFrame = build_pregame_features(
        games,
        epa,
        blend_k_offense=SEASON_BLEND_K,
        blend_k_defense=SEASON_BLEND_K,
        **shared,
    )
    _frames_match(default, explicit)
    columns: list[str] = list(feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2))
    _frames_match(default.loc[:, columns], explicit.loc[:, columns])


def test_split_blend_changes_only_regressed_epa() -> None:
    """k 3 and k 9 move offense and defense. Raw recent, Elo, QB, and injuries stay."""
    games, epa = _multiweek_slate(weeks=6)
    games = _with_schedule(games)
    perturbed: pd.DataFrame = epa.copy()
    later: pd.Series = (perturbed["season"] == 2017) & (perturbed["team_franchise"] == "A")
    perturbed.loc[later, "avg_offensive_epa"] = 1.5
    perturbed.loc[later, "avg_defensive_epa_allowed"] = -0.4
    starters, dropbacks = _starter_tables(games)
    empty_starters, empty_reports = _empty_injuries()
    shared: dict[str, object] = {
        "lookback_games": 4,
        "expected_starters": starters,
        "qb_dropbacks": dropbacks,
        "depth_chart_starters": empty_starters,
        "injury_reports": empty_reports,
    }
    default: pd.DataFrame = build_pregame_features(games, perturbed, **shared)
    split: pd.DataFrame = build_pregame_features(
        games,
        perturbed,
        blend_k_offense=OFFENSIVE_BLEND_K,
        blend_k_defense=DEFENSIVE_BLEND_K,
        **shared,
    )
    columns: tuple[str, ...] = feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2)
    stable: list[str] = [column for column in columns if "regressed" not in column]
    _frames_match(default.loc[:, stable], split.loc[:, stable])
    changed_offense: bool = False
    changed_defense: bool = False
    for column in columns:
        if "regressed" not in column:
            continue
        left: np.ndarray = default[column].to_numpy(dtype=float)
        right: np.ndarray = split[column].to_numpy(dtype=float)
        if np.allclose(left, right, equal_nan=True):
            continue
        if "offensive" in column:
            changed_offense = True
        if "defensive" in column:
            changed_defense = True
    assert changed_offense
    assert changed_defense


def test_candidate_columns_match_the_defender() -> None:
    """Both new sets keep the defender names and order."""
    defender: tuple[str, ...] = feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2)
    assert feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2_K39) == defender
    assert feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY) == defender
    assert defender == PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ


def test_season_decay_weights_use_the_last_training_season() -> None:
    """A season two years back has weight 0.5 ** (2 / 5). The latest season is 1."""
    weights: np.ndarray = season_decay_weights(pd.Series([2014, 2016, 2016]))
    assert weights[0] == pytest.approx(0.5 ** (2.0 / TRAIN_SEASON_HALF_LIFE))
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        season_decay_weights(pd.Series([2016]), half_life=0.0)


def test_unweighted_logistic_matches_a_missing_sample_weight() -> None:
    """None keeps the old fit. A real weight changes the coefficient."""
    frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(6)],
            "gameday": ["2014-09-01"] * 3 + ["2016-09-01"] * 3,
            "season": [2014, 2014, 2014, 2016, 2016, 2016],
            "signal": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )
    target: pd.Series = pd.Series([0, 1, 0, 1, 0, 1])
    plain = fit_logistic_pipeline(frame, target, max_iter=200, random_state=42)
    omitted = fit_logistic_pipeline(
        frame,
        target,
        max_iter=200,
        random_state=42,
        sample_weight=None,
    )
    np.testing.assert_allclose(
        plain.named_steps["classifier"].coef_,
        omitted.named_steps["classifier"].coef_,
    )
    weighted = fit_logistic_pipeline(
        frame,
        target,
        max_iter=200,
        random_state=42,
        sample_weight=season_decay_weights(frame["season"]),
    )
    assert not np.allclose(
        plain.named_steps["classifier"].coef_,
        weighted.named_steps["classifier"].coef_,
    )


def test_fold_weights_use_only_training_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decay fit sees seasons before the scored season and does not weight it."""
    seen: dict[str, object] = {}

    def spy(seasons: pd.Series, half_life: float) -> np.ndarray:
        seen["seasons"] = [int(season) for season in seasons]
        seen["half_life"] = half_life
        return np.ones(len(seasons), dtype=float)

    monkeypatch.setattr("endzone.modeling.cv.season_decay_weights", spy)
    train: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "gameday": ["2015-09-01", "2015-09-08", "2016-09-01", "2016-09-08"],
            "season": [2015, 2015, 2016, 2016],
            "signal": [0.0, 1.0, 0.0, 1.0],
        }
    )
    train_target: pd.Series = pd.Series([0, 1, 0, 1])
    evaluation: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["e", "f"],
            "gameday": ["2017-09-01", "2017-09-08"],
            "season": [2017, 2017],
            "signal": [0.2, 0.8],
        }
    )
    eval_target: pd.Series = pd.Series([1, 0])
    plain = _logistic_fold_predictor(1.0, 200, 42)
    plain(train, train_target, evaluation, eval_target)
    assert seen == {}
    weighted = _logistic_fold_predictor(1.0, 200, 42, season_half_life=TRAIN_SEASON_HALF_LIFE)
    probabilities: np.ndarray = weighted(train, train_target, evaluation, eval_target)
    assert len(probabilities) == len(evaluation)
    assert set(seen["seasons"]) == {2015, 2016}  # type: ignore[arg-type]
    assert seen["half_life"] == pytest.approx(TRAIN_SEASON_HALF_LIFE)
    assert 2017 not in seen["seasons"]  # type: ignore[operator]


def test_prior_shrink_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023 and the two fixed candidates."""
    argv: list[str] = prior_shrink_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "prior_shrink"
    assert argv[argv.index("--reference-model") + 1] == "logistic"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_prior_shrink"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "prior_shrink")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_prior_shrink"
    assert spec.report_profile == "prior_shrink"
    assert spec.reference_feature_set == FEATURE_SET_PKG_CORE_INJ_L2
    assert spec.logistic_feature_sets == (
        "h2_full",
        "pkg_core_inj_l2",
        "pkg_core_inj_l2_k39",
        "pkg_core_inj_l2_k39_decay",
    )
    assert spec.note.endswith("No model is promoted from this run.")
    fixed: dict[str, float | None] = {row.feature_set: row.fixed_c for row in spec.model_specs}
    assert fixed["h2_full"] is None
    assert fixed["pkg_core_inj_l2"] == pytest.approx(0.1)
    assert fixed["pkg_core_inj_l2_k39"] == pytest.approx(0.1)
    assert fixed["pkg_core_inj_l2_k39_decay"] == pytest.approx(0.1)
    assert locked_logistic_h2_full_n() == 2191


def test_holdout_refuses_the_prior_shrink_sets_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025."""
    for feature_set in (
        FEATURE_SET_PKG_CORE_INJ_L2_K39,
        FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY,
    ):
        args: argparse.Namespace = argparse.Namespace(feature_set=feature_set)
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(args)


def test_prior_shrink_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_prior_shrink.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_prior_shrink_main()


@pytest.mark.cross_platform
def test_prior_shrink_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-prior-shrink:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[
        0
    ]
    assert "python scripts/train_prior_shrink.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_prior_shrink.py" in script
    assert '"train-prior-shrink" { Invoke-TrainPriorShrink }' in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script


def _write_oof(path: Path, frame: pd.DataFrame) -> None:
    """Write one OOF file as UTF-8."""
    frame.to_csv(path, index=False, encoding="utf-8")


@pytest.mark.cross_platform
def test_written_prior_shrink_cohort_reads_utf8(tmp_path: Path) -> None:
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
        "oof_logistic_pkg_core_inj_l2_k39.csv",
        "oof_logistic_pkg_core_inj_l2_k39_decay.csv",
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
    assert "prior shrink cohort" in cohort_message(summary["n"], 2191)


def test_prior_shrink_report_scores_each_candidate_alone() -> None:
    """The decay row records the half-life. elo_mov stays a baseline probability."""
    games, epa = _multiweek_slate(weeks=6)
    games = _with_schedule(games)
    starters, dropbacks = _starter_tables(games)
    empty_starters, empty_reports = _empty_injuries()
    features: pd.DataFrame = build_pregame_features(
        games,
        epa,
        lookback_games=4,
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        depth_chart_starters=empty_starters,
        injury_reports=empty_reports,
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
        hypothesis="prior_shrink",
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
    reference: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)]
    unchanged: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2_K39)]
    decay: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY)]
    assert reference["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert unchanged["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert unchanged["params"]["blend_k_offense"] == pytest.approx(OFFENSIVE_BLEND_K)  # type: ignore[index]
    assert unchanged["params"]["blend_k_defense"] == pytest.approx(DEFENSIVE_BLEND_K)  # type: ignore[index]
    assert "season_half_life" not in unchanged["params"]  # type: ignore[operator]
    assert decay["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert decay["params"]["season_half_life"] == pytest.approx(TRAIN_SEASON_HALF_LIFE)  # type: ignore[index]
    assert decay["params"]["sample_weight"] == "train_season_half_life"  # type: ignore[index]
    oof: pd.DataFrame = by_key[(MODEL_ELO_MOV, MODEL_ELO_MOV)]["oof"]  # type: ignore[assignment]
    assert set(int(season) for season in oof["season"]) == {2017}
    assert int(oof["season"].max()) < 2024
    for feature_set in (FEATURE_SET_PKG_CORE_INJ_L2_K39, FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY):
        candidate_oof: pd.DataFrame = by_key[("logistic", feature_set)]["oof"]  # type: ignore[assignment]
        assert int(candidate_oof["season"].max()) < 2024
    merged: pd.DataFrame = oof.merge(
        features.loc[:, ["game_id", ELO_MOV_DIFF_COLUMN]],
        on="game_id",
        how="left",
    )
    expected: np.ndarray = rating_win_probability(merged[ELO_MOV_DIFF_COLUMN].to_numpy(dtype=float))
    np.testing.assert_allclose(merged["y_proba"].to_numpy(dtype=float), expected)
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] in {"accepted", "rejected", "inconclusive"}
    for feature_set in (FEATURE_SET_PKG_CORE_INJ_L2_K39, FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY):
        assert decisions[("logistic", feature_set)] in {"accepted", "rejected", "inconclusive"}
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[(MODEL_ELO_MOV, MODEL_ELO_MOV)] == "baseline"
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert {
        FEATURE_SET_PKG_CORE_INJ_L2_K39,
        FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY,
    } <= set(calibration["feature_set"])
    assert {
        FEATURE_SET_PKG_CORE_INJ_L2_K39,
        FEATURE_SET_PKG_CORE_INJ_L2_K39_DECAY,
    } <= set(information["feature_set"])
    assert ("qbelo538", "qbelo538") not in by_key
