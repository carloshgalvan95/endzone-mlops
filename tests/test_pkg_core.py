"""pkg_core is the locked union of three existing feature blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from endzone.features.pregame import (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H5_QB,
    FEATURE_SET_H6_ADJ_ELO,
    FEATURE_SET_PKG_CORE,
    FEATURE_SET_RECENCY_MARGIN,
    PKG_CORE_CHALLENGER_FEATURE_SETS,
    PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO,
    PREGAME_FEATURE_COLUMNS_PKG_CORE,
    PREGAME_FEATURE_COLUMNS_RECENCY_MARGIN,
    QB_RATING_COLUMNS,
    feature_columns_for,
)
from endzone.modeling.cv import RollingOriginConfig
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.models import feature_column_groups
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.train import pkg_core_challenger_argv
from scripts.train_pkg_core import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_pkg_core import (
    main as train_pkg_core_main,
)
from tests.test_h6 import _multiweek_slate


def _ordered_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """First occurrence of each name, in the order the groups are given."""
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for column in group:
            if column in seen:
                continue
            seen.add(column)
            ordered.append(column)
    return tuple(ordered)


def test_pkg_core_keeps_every_source_column_and_adds_none() -> None:
    """The candidate is the three blocks, with a repeated name stored once."""
    expected: tuple[str, ...] = _ordered_union(
        PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO,
        QB_RATING_COLUMNS,
        PREGAME_FEATURE_COLUMNS_RECENCY_MARGIN,
    )
    assert feature_columns_for(FEATURE_SET_PKG_CORE) == expected
    assert PREGAME_FEATURE_COLUMNS_PKG_CORE == expected
    assert len(expected) == len(set(expected))
    assert set(PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO) <= set(expected)
    assert set(QB_RATING_COLUMNS) <= set(expected)
    assert set(PREGAME_FEATURE_COLUMNS_RECENCY_MARGIN) <= set(expected)
    assert set(expected) <= (
        set(PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO)
        | set(QB_RATING_COLUMNS)
        | set(PREGAME_FEATURE_COLUMNS_RECENCY_MARGIN)
    )
    assert expected[: len(PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO)] == (
        PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO
    )
    groups: dict[str, list[str]] = feature_column_groups(list(expected))
    assert "elo_mov_diff" in groups["form"]
    assert "home_recent_point_diff" in groups["form"]
    assert "away_recent_point_diff" in groups["form"]
    assert "home_qb_rating" in groups["epa"]
    assert "home_qb_rating_minus_team_pass_epa" in groups["epa"]
    assert "home_recent_adj_offensive_epa_per_play" in groups["epa"]
    assert "home_recent_offensive_epa_per_play" in groups["epa"]
    assert "home_recent_win_rate" in groups["win_rate"]
    assert "home_rest_days" in groups["other"]


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


def test_pkg_core_without_starters_fails_before_a_fit() -> None:
    """Missing QB ratings are all null, so the candidate is named and not fit."""
    games, epa = _multiweek_slate()
    with pytest.raises(ValueError, match="pkg_core") as exc_info:
        run_phase1_baselines(
            games,
            epa,
            config=RollingOriginConfig(
                cv_start_season=2017,
                cv_end_season=2017,
                n_bootstrap=8,
                bootstrap_seed=42,
            ),
            phase=2,
            hypothesis="pkg_core",
        )
    message: str = str(exc_info.value)
    assert "home_qb_rating" in message
    assert "h5_qb" in message
    assert "run make dbt-build-databricks" in message


def test_pkg_core_report_marks_the_market_and_scores_the_context_rows() -> None:
    """The candidate and the three context sets share one comparison."""
    games, epa = _multiweek_slate(weeks=8)
    starters, dropbacks = _starter_tables(games)
    report: dict[str, object] = run_phase1_baselines(
        games,
        epa,
        config=RollingOriginConfig(
            cv_start_season=2017,
            cv_end_season=2017,
            n_bootstrap=20,
            bootstrap_seed=42,
        ),
        lookback_games=4,
        logistic_max_iter=200,
        phase=2,
        hypothesis="pkg_core",
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        random_state=42,
    )
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[("logistic", "h2_full")] == "reference"
    for feature_set in PKG_CORE_CHALLENGER_FEATURE_SETS:
        assert decisions[("logistic", feature_set)] in {"accepted", "rejected", "inconclusive"}
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    assert set(calibration["model"]) >= {"logistic", "elo", "market"}
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert set(information["feature_set"]) == {
        FEATURE_SET_H2_FULL,
        FEATURE_SET_PKG_CORE,
        FEATURE_SET_H5_QB,
        FEATURE_SET_RECENCY_MARGIN,
        FEATURE_SET_H6_ADJ_ELO,
    }
    assert "blend" not in information.columns
    markdown: str = str(report["markdown"])
    assert "benchmark" in markdown
    assert "## Calibration" in markdown
    assert "## Market information" in markdown
    assert "pkg_core is the preregistered candidate." in markdown
    assert "No blend is published." in markdown


def test_holdout_refuses_pkg_core_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025, so it cannot select this set."""
    args: argparse.Namespace = argparse.Namespace(feature_set=FEATURE_SET_PKG_CORE)
    with pytest.raises(ValueError, match="2016-2023"):
        run_holdout(args)


def test_pkg_core_command_locks_the_development_window() -> None:
    """The argv, the YAML, and the reference recipe stay on 2016-2023."""
    argv: list[str] = pkg_core_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "pkg_core"
    assert argv[argv.index("--reference-feature-set") + 1] == "h2_full"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_pkg_core"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "pkg_core")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_pkg_core"
    assert spec.tag == "pkg-core"
    assert spec.report_profile == "pkg_core"
    assert spec.log_qb is True
    assert spec.qb_subset is False
    assert spec.qb_feature_set == "h5_qb"
    assert spec.reference_feature_set == "h2_full"
    assert spec.output_dir == "reports/phase2_pkg_core"
    assert spec.logistic_feature_sets == ("h2_full", *PKG_CORE_CHALLENGER_FEATURE_SETS)
    assert spec.note.endswith("No model is promoted from this run.")
    assert locked_logistic_h2_full_n() == 2191


def test_pkg_core_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_pkg_core.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_pkg_core_main()


@pytest.mark.cross_platform
def test_pkg_core_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-pkg-core:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_pkg_core.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_pkg_core.py" in script
    assert '"train-pkg-core" { Invoke-TrainPkgCore }' in script
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
    reference.to_csv(tmp_path / "oof_logistic_h2_full.csv", index=False, encoding="utf-8")
    for feature_set in PKG_CORE_CHALLENGER_FEATURE_SETS:
        reference.to_csv(
            tmp_path / f"oof_logistic_{feature_set}.csv",
            index=False,
            encoding="utf-8",
        )
    summary: dict[str, int] = require_written_cohort(tmp_path)
    assert summary == {"n": 2, "variants": len(PKG_CORE_CHALLENGER_FEATURE_SETS)}
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
