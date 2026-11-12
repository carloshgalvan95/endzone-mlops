"""The injury candidates stay on the locked 2016-2023 recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from endzone.features.pregame import FEATURE_SET_H2_FULL, PKG_CORE_INJ_FEATURE_SETS
from endzone.modeling.cv import RollingOriginConfig
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import _fixed_c, experiment_for
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.scoring import logistic_c_for_spec
from endzone.modeling.train import pkg_core_inj_challenger_argv
from scripts.train_pkg_core_inj import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_pkg_core_inj import (
    main as train_pkg_core_inj_main,
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


def test_pkg_core_inj_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023, with C 0.1 only on the L2 row."""
    argv: list[str] = pkg_core_inj_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "pkg_core_inj"
    assert argv[argv.index("--reference-feature-set") + 1] == "h2_full"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_pkg_core_inj"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "pkg_core_inj")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_pkg_core_inj"
    assert spec.tag == "pkg-core-inj"
    assert spec.report_profile == "pkg_core_inj"
    assert spec.log_qb is True
    assert spec.qb_subset is False
    assert spec.output_dir == "reports/phase2_pkg_core_inj"
    assert spec.logistic_feature_sets == (
        "h2_full",
        "pkg_core",
        "pkg_core_inj",
        "pkg_core_inj_l2",
    )
    assert spec.note.endswith("No model is promoted from this run.")
    fixed: dict[str, float | None] = {row.feature_set: row.fixed_c for row in spec.model_specs}
    assert fixed["h2_full"] is None
    assert fixed["pkg_core"] is None
    assert fixed["pkg_core_inj"] is None
    assert fixed["pkg_core_inj_l2"] == pytest.approx(0.1)
    assert locked_logistic_h2_full_n() == 2191


def test_fixed_c_overrides_the_command_and_is_refused_on_logistic_l2() -> None:
    """The L2 candidate keeps C at 0.1. The tuned model cannot also lock C."""
    spec = experiment_for(2, "pkg_core_inj")
    logistic = next(row for row in spec.model_specs if row.feature_set == "pkg_core_inj")
    tuned = next(row for row in spec.model_specs if row.feature_set == "pkg_core_inj_l2")
    assert logistic_c_for_spec(logistic, 1.0) == pytest.approx(1.0)
    assert logistic_c_for_spec(tuned, 1.0) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="logistic"):
        _fixed_c(0.1, Path("logistic_l2.yaml"), "logistic_l2")


def test_holdout_refuses_both_injury_sets_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025."""
    for feature_set in PKG_CORE_INJ_FEATURE_SETS:
        args: argparse.Namespace = argparse.Namespace(feature_set=feature_set)
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(args)


def test_pkg_core_inj_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_pkg_core_inj.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_pkg_core_inj_main()


@pytest.mark.cross_platform
def test_pkg_core_inj_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-pkg-core-inj:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[
        0
    ]
    assert "python scripts/train_pkg_core_inj.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_pkg_core_inj.py" in script
    assert '"train-pkg-core-inj" { Invoke-TrainPkgCoreInj }' in script
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
    names: tuple[str, ...] = ("h2_full", "pkg_core", "pkg_core_inj", "pkg_core_inj_l2")
    for feature_set in names:
        reference.to_csv(
            tmp_path / f"oof_logistic_{feature_set}.csv",
            index=False,
            encoding="utf-8",
        )
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


def test_injury_report_marks_the_market_and_locks_c() -> None:
    """pkg_core is context. The two injury rows are the candidates. C is 0.1 on one."""
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
        hypothesis="pkg_core_inj",
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        depth_chart_starters=empty_starters,
        injury_reports=empty_reports,
        random_state=42,
    )
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] == "reference"
    assert decisions[("logistic", "pkg_core")] in {"accepted", "rejected", "inconclusive"}
    for feature_set in PKG_CORE_INJ_FEATURE_SETS:
        assert decisions[("logistic", feature_set)] in {"accepted", "rejected", "inconclusive"}
    by_set: dict[str, dict[str, object]] = {
        str(row["feature_set"]): row
        for row in report["results"]  # type: ignore[index]
    }
    assert by_set["pkg_core_inj"]["params"]["C"] == pytest.approx(1.0)  # type: ignore[index]
    assert by_set["pkg_core_inj_l2"]["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert "benchmark" in str(report["markdown"])
    assert "No blend is published." in str(report["markdown"])
