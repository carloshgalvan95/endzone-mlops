"""Injury coverage, home-field context, and the locked 2016-2023 recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from endzone.features.home_field import HOME_FIELD_COLUMNS, attach_home_field_features
from endzone.features.injuries import (
    INJ_REPORT_COVERED_COLUMN,
    INJURY_DIFF_COLUMNS,
    QUESTIONABLE_WEIGHT,
    WEIGHTED_INJURY_DIFF_COLUMNS,
    attach_injury_features,
    attach_weighted_injury_features,
)
from endzone.features.pregame import (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_PKG_CORE_INJ_L2,
    FEATURE_SET_PKG_CORE_INJ_L2_HFA_CTX,
    FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q,
    PREGAME_FEATURE_COLUMNS_PKG_CORE,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_HFA_CTX,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_INJ_COV_Q,
    feature_columns_for,
)
from endzone.modeling.cv import RollingOriginConfig
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.models import feature_column_groups
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.report import write_phase1_outputs
from endzone.modeling.train import closing_challenger_argv
from scripts.train_closing import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_context_baselines,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_closing import (
    main as train_closing_main,
)
from tests.test_ensembles import _with_schedule
from tests.test_h6 import _multiweek_slate
from tests.test_pkg_core import _starter_tables
from tests.test_qbelo538 import _empty_injuries


def _game(
    game_id: str = "g1",
    season: int = 2016,
    gameday: str = "2016-09-11",
    gametime: str = "13:00",
    home: str = "A",
    away: str = "B",
) -> pd.DataFrame:
    """One scheduled game with the columns the absence count requires."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": 1,
                "game_type": "REG",
                "gameday": gameday,
                "gametime": gametime,
                "home_team": home,
                "away_team": away,
            }
        ]
    )


def _starter(team: str, gsis_id: str, position: str, depth: int = 1) -> dict[str, object]:
    """One depth-chart row on the synthetic week-1 slate."""
    return {
        "season": 2016,
        "week": 1,
        "game_type": "REG",
        "team": team,
        "gsis_id": gsis_id,
        "position": position,
        "depth_team": depth,
    }


def _report(team: str, gsis_id: str, status: str, modified: str) -> dict[str, object]:
    """One injury row. ``modified`` is UTC."""
    return {
        "season": 2016,
        "week": 1,
        "game_type": "REG",
        "team": team,
        "gsis_id": gsis_id,
        "report_status": status,
        "date_modified": modified,
    }


def _weighted_row(
    games: pd.DataFrame,
    starters: list[dict[str, object]],
    reports: list[dict[str, object]],
) -> pd.Series:
    """Return the weighted row for the only game."""
    frame: pd.DataFrame = attach_weighted_injury_features(
        games,
        pd.DataFrame(starters),
        pd.DataFrame(reports),
    )
    return frame.iloc[0]


def test_questionable_weight_is_locked_and_out_or_doubtful_stays_put() -> None:
    """Questionable is 0.27. The four Out or Doubtful columns ignore it."""
    games: pd.DataFrame = _game()
    starters: list[dict[str, object]] = [
        _starter("A", "ol", "T"),
        _starter("B", "ol", "G"),
        _starter("A", "wr", "WR"),
    ]
    reports: list[dict[str, object]] = [
        _report("A", "ol", "Out", "2016-09-10 12:00:00+00:00"),
        _report("B", "ol", "Questionable", "2016-09-10 12:00:00+00:00"),
        _report("A", "wr", "Out", "2016-09-09 12:00:00+00:00"),
        _report("A", "wr", "Questionable", "2016-09-10 12:00:00+00:00"),
    ]
    weighted: pd.Series = _weighted_row(games, starters, reports)
    original: pd.Series = attach_injury_features(
        games, pd.DataFrame(starters), pd.DataFrame(reports)
    ).iloc[0]
    assert QUESTIONABLE_WEIGHT == pytest.approx(0.27)
    assert weighted["diff_ol_starters_absent_w"] == pytest.approx(1.0 - 0.27)
    assert weighted["diff_skill_starters_absent_w"] == pytest.approx(0.27)
    assert original["diff_ol_starters_out_or_doubtful"] == 1.0
    assert original["diff_skill_starters_out_or_doubtful"] == 0.0
    doubtful: pd.Series = _weighted_row(
        games,
        [_starter("A", "cb", "CB"), _starter("B", "cb", "CB")],
        [
            _report("A", "cb", "Doubtful", "2016-09-10 12:00:00+00:00"),
            _report("B", "cb", "Out", "2016-09-10 12:00:00+00:00"),
        ],
    )
    assert doubtful["diff_db_starters_absent_w"] == pytest.approx(0.0)


def test_same_timestamp_keeps_out_over_questionable_for_the_weight() -> None:
    """A tie on the clock keeps Out, so the weight is 1.0 rather than 0.27."""
    weighted: pd.Series = _weighted_row(
        _game(),
        [_starter("A", "wr", "WR")],
        [
            _report("A", "wr", "Questionable", "2016-09-10 12:00:00+00:00"),
            _report("A", "wr", "Out", "2016-09-10 12:00:00+00:00"),
        ],
    )
    assert weighted["diff_skill_starters_absent_w"] == pytest.approx(1.0)


def test_injury_coverage_is_zero_before_2009_and_when_inputs_are_omitted() -> None:
    """Coverage is 1.0 only for 2009 or later when both tables are passed."""
    games: pd.DataFrame = pd.concat(
        [_game(game_id="early", season=2008), _game(game_id="later", season=2009)],
        ignore_index=True,
    )
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
    covered: pd.DataFrame = attach_weighted_injury_features(games, starters, reports)
    assert list(covered[INJ_REPORT_COVERED_COLUMN]) == [0.0, 1.0]
    assert covered[INJ_REPORT_COVERED_COLUMN].isna().sum() == 0
    omitted: pd.DataFrame = attach_weighted_injury_features(games, None, None)
    assert list(omitted[INJ_REPORT_COVERED_COLUMN]) == [0.0, 0.0]
    assert omitted[list(WEIGHTED_INJURY_DIFF_COLUMNS)].isna().all().all()
    with pytest.raises(ValueError, match="both be provided"):
        attach_weighted_injury_features(games, starters, None)


def test_feature_sets_keep_the_defender_and_add_the_locked_columns() -> None:
    """The defender stays 47 columns. The candidates are 48 and 50."""
    assert feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2) == PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ
    assert len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ) == 47
    assert len(PREGAME_FEATURE_COLUMNS_PKG_CORE) == 43
    coverage: tuple[str, ...] = feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q)
    context: tuple[str, ...] = feature_columns_for(FEATURE_SET_PKG_CORE_INJ_L2_HFA_CTX)
    assert coverage == PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_INJ_COV_Q
    assert context == PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_HFA_CTX
    assert len(coverage) == 48
    assert len(context) == 50
    assert coverage[:43] == PREGAME_FEATURE_COLUMNS_PKG_CORE
    assert coverage[43:] == WEIGHTED_INJURY_DIFF_COLUMNS + (INJ_REPORT_COVERED_COLUMN,)
    assert set(INJURY_DIFF_COLUMNS).isdisjoint(coverage)
    assert context[:47] == PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ
    assert context[47:] == HOME_FIELD_COLUMNS
    groups: dict[str, list[str]] = feature_column_groups(list(coverage) + list(HOME_FIELD_COLUMNS))
    assert set(WEIGHTED_INJURY_DIFF_COLUMNS) <= set(groups["diff"])
    assert INJ_REPORT_COVERED_COLUMN in groups["other"]
    assert set(HOME_FIELD_COLUMNS) <= set(groups["other"])


def _schedule_row(
    game_id: str,
    season: int,
    gameday: str,
    home: str,
    away: str,
    *,
    location: str = "Home",
    surface: str | None = "grass",
    gametime: str | None = "13:00",
) -> dict[str, object]:
    """One game with the columns the home-field features read."""
    return {
        "game_id": game_id,
        "season": season,
        "gameday": gameday,
        "home_team_franchise": home,
        "away_team_franchise": away,
        "location": location,
        "surface": surface,
        "gametime": gametime,
    }


def test_divisional_flag_follows_the_realignment_and_relocations() -> None:
    """1999-2001 and 2002+ divisions, including OAK, SD, and STL."""
    rows: list[dict[str, object]] = [
        _schedule_row("e2001", 2001, "2001-09-09", "BUF", "IND"),
        _schedule_row("e2002", 2002, "2002-09-08", "BUF", "IND"),
        _schedule_row("sea2001", 2001, "2001-09-09", "SEA", "DEN"),
        _schedule_row("sea2002", 2002, "2002-09-08", "SEA", "DEN"),
        _schedule_row("nfc2002", 2002, "2002-09-08", "SEA", "SF"),
        _schedule_row("ari2001", 2001, "2001-09-09", "ARI", "DAL"),
        _schedule_row("ari2002", 2002, "2002-09-08", "ARI", "DAL"),
        _schedule_row("stl", 2001, "2001-09-09", "SF", "STL"),
        _schedule_row("oak", 2001, "2001-09-09", "DEN", "OAK"),
        _schedule_row("sd", 2016, "2016-09-11", "KC", "SD"),
        _schedule_row("lv", 2020, "2020-09-13", "DEN", "LV"),
        _schedule_row("hou", 2002, "2002-09-08", "HOU", "IND"),
        _schedule_row("hou_early", 2001, "2001-09-09", "HOU", "TEN"),
        _schedule_row("other", 2016, "2016-09-11", "BUF", "KC"),
        _schedule_row("unknown", 2016, "2016-09-11", "BUF", "ZZZ"),
    ]
    flags: pd.Series = attach_home_field_features(pd.DataFrame(rows)).set_index("game_id")[
        "is_divisional_game"
    ]
    expected: dict[str, float] = {
        "e2001": 1.0,
        "e2002": 0.0,
        "sea2001": 1.0,
        "sea2002": 0.0,
        "nfc2002": 1.0,
        "ari2001": 1.0,
        "ari2002": 0.0,
        "stl": 1.0,
        "oak": 1.0,
        "sd": 1.0,
        "lv": 1.0,
        "hou": 1.0,
        "hou_early": 0.0,
        "other": 0.0,
        "unknown": 0.0,
    }
    for game_id, value in expected.items():
        assert flags.loc[game_id] == value


def test_surface_mismatch_uses_only_an_earlier_home_surface() -> None:
    """The prior home surface is the latest known one strictly before gameday."""
    rows: list[dict[str, object]] = [
        _schedule_row("old", 2016, "2016-09-11", "SEA", "NYG", surface="grass"),
        _schedule_row("newer", 2016, "2016-09-18", "SEA", "NYG", surface="fieldturf"),
        _schedule_row("same_day", 2016, "2016-09-25", "SEA", "NYG", surface="grass"),
        _schedule_row(
            "neutral", 2016, "2016-09-20", "SEA", "NYG", surface="grass", location="Neutral"
        ),
        _schedule_row("blank", 2016, "2016-09-25", "NYG", "SEA", surface="  "),
        _schedule_row("visitor", 2016, "2016-09-25", "NYG", "SEA", surface="grass"),
        _schedule_row("dess", 2016, "2016-10-02", "NYG", "SEA", surface=" dessograss "),
        _schedule_row("unknown", 2016, "2016-10-02", "DAL", "NYG", surface="hybrid"),
        _schedule_row("no_prior", 2016, "2016-09-11", "BUF", "MIA", surface="fieldturf"),
    ]
    flags: pd.Series = attach_home_field_features(pd.DataFrame(rows)).set_index("game_id")[
        "away_surface_mismatch"
    ]
    assert flags.loc["visitor"] == 1.0
    assert flags.loc["dess"] == 0.0
    assert flags.loc["blank"] == 0.0
    assert flags.loc["unknown"] == 0.0
    assert flags.loc["no_prior"] == 0.0
    assert flags.loc["same_day"] == 0.0


def test_west_coast_early_kickoff_excludes_neutral_sites_and_the_boundary() -> None:
    """14:00 ET is out. A neutral site is out. The home team must not be western."""
    rows: list[dict[str, object]] = [
        _schedule_row("early", 2016, "2016-09-11", "NYG", "SEA", gametime="13:00"),
        _schedule_row("boundary", 2016, "2016-09-11", "NYG", "SEA", gametime="14:00"),
        _schedule_row("just_before", 2016, "2016-09-11", "NYG", "SEA", gametime="13:59"),
        _schedule_row("seconds", 2016, "2016-09-11", "NYG", "DEN", gametime="13:00:00"),
        _schedule_row("neutral", 2016, "2016-09-11", "NYG", "SEA", location="Neutral"),
        _schedule_row("blank", 2016, "2016-09-11", "NYG", "SEA", gametime=""),
        _schedule_row("malformed", 2016, "2016-09-11", "NYG", "SEA", gametime="noon"),
        _schedule_row("home_west", 2016, "2016-09-11", "SEA", "NYG"),
        _schedule_row("both_west", 2016, "2016-09-11", "SF", "SEA"),
        _schedule_row("east", 2016, "2016-09-11", "NYG", "BUF"),
    ]
    flags: pd.Series = attach_home_field_features(pd.DataFrame(rows)).set_index("game_id")[
        "west_coast_early_kickoff"
    ]
    assert flags.loc["early"] == 1.0
    assert flags.loc["just_before"] == 1.0
    assert flags.loc["seconds"] == 1.0
    assert flags.loc["boundary"] == 0.0
    assert flags.loc["neutral"] == 0.0
    assert flags.loc["blank"] == 0.0
    assert flags.loc["malformed"] == 0.0
    assert flags.loc["home_west"] == 0.0
    assert flags.loc["both_west"] == 0.0
    assert flags.loc["east"] == 0.0


def test_home_field_and_coverage_columns_have_no_missing_values() -> None:
    """The passthrough columns are finite even when the source fields are blank."""
    games: pd.DataFrame = pd.DataFrame(
        [
            _schedule_row("a", 2008, "2008-09-07", "BUF", "MIA", surface=None, gametime=None),
            _schedule_row("b", 2016, "2016-09-11", "NYG", "SEA", location="Home", surface=""),
        ]
    )
    home_field: pd.DataFrame = attach_home_field_features(games)
    assert set(HOME_FIELD_COLUMNS) <= set(home_field.columns)
    assert home_field[list(HOME_FIELD_COLUMNS)].isna().sum().sum() == 0
    assert set(home_field[list(HOME_FIELD_COLUMNS)].stack().unique()).issubset({0.0, 1.0})
    covered: pd.DataFrame = attach_weighted_injury_features(games, None, None)
    assert covered[INJ_REPORT_COVERED_COLUMN].isna().sum() == 0


def test_closing_command_locks_the_development_window() -> None:
    """The recipe is 2016-2023, C 0.1 on the candidates, and no 2024."""
    argv: list[str] = closing_challenger_argv()
    assert "2024" not in argv
    assert "2025" not in argv
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_closing"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    spec = experiment_for(2, "closing")
    assert spec.experiment_name == "nfl_pregame/phase2_closing"
    assert spec.report_profile == "closing"
    assert spec.default is False
    assert [item.feature_set for item in spec.model_specs] == [
        FEATURE_SET_H2_FULL,
        FEATURE_SET_PKG_CORE_INJ_L2,
        FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q,
        FEATURE_SET_PKG_CORE_INJ_L2_HFA_CTX,
    ]
    assert spec.model_specs[0].fixed_c is None
    assert spec.model_specs[1].fixed_c == pytest.approx(0.1)
    assert spec.model_specs[2].fixed_c == pytest.approx(0.1)
    assert spec.model_specs[3].fixed_c == pytest.approx(0.1)
    assert spec.note.endswith("No model is promoted from this run.")
    assert locked_logistic_h2_full_n() == 2191


def test_holdout_refuses_closing_feature_sets_before_a_warehouse_read() -> None:
    """The 2024-2025 split does not score either candidate."""
    for feature_set in (
        FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q,
        FEATURE_SET_PKG_CORE_INJ_L2_HFA_CTX,
    ):
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(argparse.Namespace(feature_set=feature_set))


def test_closing_command_refuses_extra_arguments() -> None:
    """The script has no flags. The window stays inside the module."""
    original: list[str] = sys.argv
    sys.argv = ["train_closing.py", "--cv-end-season", "2025"]
    try:
        with pytest.raises(SystemExit, match="does not take arguments"):
            train_closing_main()
    finally:
        sys.argv = original


@pytest.mark.cross_platform
def test_closing_recipe_is_in_the_makefile_and_dev_script() -> None:
    """Make and PowerShell call the same script and do not hard-code a drive."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-closing:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_closing.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert '"train-closing" { Invoke-TrainClosing }' in script
    assert "C:\\" not in script


@pytest.mark.cross_platform
def test_written_closing_cohort_reads_utf8(tmp_path: Path) -> None:
    """The command checks the OOF files, comparison.json, and the context labels."""
    reference: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2018, 2019],
            "y_true": [1, 0],
            "y_proba": [0.6, 0.4],
        }
    )
    reference.to_csv(tmp_path / "oof_logistic_pkg_core_inj_l2.csv", index=False, encoding="utf-8")
    for file_name in (
        "oof_logistic_h2_full.csv",
        "oof_elo_elo.csv",
        "oof_elo_mov_elo_mov.csv",
        "oof_logistic_pkg_core_inj_l2_inj_cov_q.csv",
        "oof_logistic_pkg_core_inj_l2_hfa_ctx.csv",
    ):
        reference.to_csv(tmp_path / file_name, index=False, encoding="utf-8")
    (tmp_path / "comparison.json").write_text("[]\n", encoding="utf-8")
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
    assert "closing cohort" in cohort_message(summary["n"], 2191)


def test_closing_report_scores_both_candidates(tmp_path: Path) -> None:
    """One outer fold scores both logistics, writes comparison.json, and stays before 2024."""
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
        hypothesis="closing",
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
    coverage: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q)]
    context: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2_HFA_CTX)]
    assert defender["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert coverage["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert context["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert by_key[("logistic", FEATURE_SET_H2_FULL)]["params"]["C"] == pytest.approx(1.0)  # type: ignore[index]
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[("elo_mov", "elo_mov")] == "baseline"
    for feature_set in (
        FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q,
        FEATURE_SET_PKG_CORE_INJ_L2_HFA_CTX,
    ):
        assert decisions[("logistic", feature_set)] in {"accepted", "rejected", "inconclusive"}
        oof: pd.DataFrame = by_key[("logistic", feature_set)]["oof"]  # type: ignore[assignment]
        assert int(oof["season"].max()) < 2024
        assert oof["y_proba"].notna().all()
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    assert "ece" in calibration.columns
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert len(information) > 0
    written: dict[str, object] = write_phase1_outputs(report, tmp_path)  # type: ignore[arg-type]
    payload: list[dict[str, object]] = json.loads(
        (tmp_path / "comparison.json").read_text(encoding="utf-8")
    )
    assert any(row["feature_set"] == FEATURE_SET_PKG_CORE_INJ_L2_INJ_COV_Q for row in payload)
    extra_paths: dict[str, str] = written["extra_paths"]  # type: ignore[assignment]
    assert "comparison_json" in extra_paths
