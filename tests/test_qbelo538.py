"""The QB Elo candidate stays on the locked 2016-2023 recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.opponent import ELO_MOV_DIFF_COLUMN
from endzone.features.pregame import (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_PKG_CORE_INJ_L2,
    FEATURE_SET_PKG_CORE_INJ_L2_QBELO538,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_QBELO538,
    build_pregame_features,
)
from endzone.features.qbelo import (
    MODEL_ELO_MOV,
    MODEL_QBELO538,
    QB_BOX_SCORE_COLUMNS,
    QB_COVERAGE_COLUMNS,
    QB_ELO_PER_VALUE,
    QBELO_DIFF_COLUMN,
    qb_game_value,
    qbelo538_diff_frame,
    qbelo_pregame_table,
    rating_win_probability,
)
from endzone.modeling.cv import RollingOriginConfig
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.train import qbelo538_challenger_argv
from scripts.train_qbelo538 import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_context_baselines,
    require_market_benchmark,
    require_qb_coverage,
    require_written_cohort,
)
from scripts.train_qbelo538 import (
    main as train_qbelo538_main,
)
from tests.test_ensembles import _with_schedule
from tests.test_h6 import _multiweek_slate
from tests.test_pkg_core import _starter_tables

_HAND_COUNTS: tuple[float, ...] = (10, 6, 100, 1, 1, 2, 3, 20, 1)


def _schedule_row(
    game_id: str,
    gameday: str,
    home: str,
    away: str,
    season: int = 2020,
) -> dict[str, object]:
    """One game row for the QB rating table."""
    return {
        "game_id": game_id,
        "season": season,
        "gameday": pd.Timestamp(gameday),
        "home_team_franchise": home,
        "away_team_franchise": away,
    }


def _starter_row(game_id: str, home: str, away: str) -> dict[str, object]:
    """Depth-chart starters. The ids are already resolved."""
    return {
        "game_id": game_id,
        "home_starter_player_id": home,
        "away_starter_player_id": away,
        "home_starter_source": "depth_chart",
        "away_starter_source": "depth_chart",
    }


def _box_row(
    game_id: str,
    franchise: str,
    player_id: str,
    counts: tuple[float, ...] = _HAND_COUNTS,
) -> dict[str, object]:
    """One player-game line. Missing counts in ``counts`` are not allowed."""
    row: dict[str, object] = {
        "game_id": game_id,
        "team_franchise": franchise,
        "player_id": player_id,
    }
    for column, value in zip(QB_BOX_SCORE_COLUMNS[3:], counts, strict=True):
        row[column] = value
    return row


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


def test_hand_computed_value_matches_the_published_weights() -> None:
    """10 attempts, 6 completions, 100 yards, and the rest of the hand line."""
    value: float = qb_game_value(*_HAND_COUNTS)
    assert value == pytest.approx(26.0)
    assert value == pytest.approx(
        -2.2 * 10 + 3.7 * 6 + 100 / 5 + 11.3 - 14.1 - 8 * 2 - 1.1 * 3 + 0.6 * 20 + 15.9
    )


def test_a_qb_with_no_history_has_adjustment_zero() -> None:
    """The second game's new starter does not inherit a negative team gap."""
    games: pd.DataFrame = pd.DataFrame(
        [
            _schedule_row("g1", "2020-09-13", "HOME", "AWAY"),
            _schedule_row("g2", "2020-09-20", "HOME", "AWAY"),
        ]
    )
    starters: pd.DataFrame = pd.DataFrame(
        [
            _starter_row("g1", "QB-A", "QB-B"),
            _starter_row("g2", "QB-NEW", "QB-B"),
        ]
    )
    boxes: pd.DataFrame = pd.DataFrame([_box_row("g1", "HOME", "QB-A")])
    detail: pd.DataFrame = qbelo_pregame_table(games, starters, boxes).set_index("game_id")
    assert detail.loc["g1", "home_qb_adj"] == pytest.approx(0.0)
    assert detail.loc["g1", "away_qb_adj"] == pytest.approx(0.0)
    assert bool(detail.loc["g1", "home_qb_has_history"]) is False
    assert detail.loc["g2", "home_qb_adj"] == pytest.approx(0.0)
    assert bool(detail.loc["g2", "home_starter_resolved"]) is True
    assert bool(detail.loc["g2", "home_qb_has_history"]) is False


def test_rating_uses_only_games_strictly_before_the_date() -> None:
    """A later box score and a same-day box score do not move the earlier rating."""
    value: float = qb_game_value(*_HAND_COUNTS)
    games: pd.DataFrame = pd.DataFrame(
        [
            _schedule_row("g1", "2020-09-13", "HOME", "AWAY"),
            _schedule_row("g-same", "2020-09-13", "OTHER", "OTHER2"),
            _schedule_row("g2", "2020-09-20", "HOME", "AWAY"),
            _schedule_row("g-future", "2020-09-27", "HOME", "AWAY"),
        ]
    )
    starters: pd.DataFrame = pd.DataFrame(
        [
            _starter_row("g1", "QB-A", "QB-B"),
            _starter_row("g-same", "QB-A", "QB-C"),
            _starter_row("g2", "QB-A", "QB-B"),
            _starter_row("g-future", "QB-A", "QB-B"),
        ]
    )
    boxes: pd.DataFrame = pd.DataFrame(
        [
            _box_row("g1", "HOME", "QB-A"),
            _box_row("g-same", "OTHER", "QB-A", counts=(1, 1, 10000, 0, 0, 0, 0, 0, 0)),
            _box_row("g-future", "HOME", "QB-A", counts=(1, 1, 10000, 0, 0, 0, 0, 0, 0)),
        ]
    )
    detail: pd.DataFrame = qbelo_pregame_table(games, starters, boxes).set_index("game_id")
    without_future: pd.DataFrame = qbelo_pregame_table(
        games.loc[games["game_id"] != "g-future"],
        starters.loc[starters["game_id"] != "g-future"],
        boxes.loc[boxes["game_id"] != "g-future"],
    ).set_index("game_id")
    assert detail.loc["g1", "home_qb_adj"] == pytest.approx(0.0)
    assert detail.loc["g-same", "home_qb_adj"] == pytest.approx(0.0)
    assert detail.loc["g2", "home_qb_adj"] == pytest.approx(without_future.loc["g2", "home_qb_adj"])
    # One prior game for QB-A is not enough once the same-day line is history
    # for the next date. The same-day line is excluded from g1 and g-same.
    assert detail.loc["g2", "home_qb_adj"] != pytest.approx(0.1 * value - 0.05 * value)
    only_first: pd.DataFrame = qbelo_pregame_table(
        games.loc[games["game_id"].isin(["g1", "g2"])],
        starters.loc[starters["game_id"].isin(["g1", "g2"])],
        boxes.loc[boxes["game_id"] == "g1"],
    ).set_index("game_id")
    assert only_first.loc["g2", "home_qb_adj"] == pytest.approx(0.05 * value)
    elo: pd.DataFrame = pd.DataFrame({"game_id": ["g1", "g2"], ELO_MOV_DIFF_COLUMN: [12.0, -4.0]})
    diff: pd.DataFrame = qbelo538_diff_frame(elo, only_first.reset_index()).set_index("game_id")
    assert diff.loc["g2", QBELO_DIFF_COLUMN] == pytest.approx(
        -4.0 + QB_ELO_PER_VALUE * 0.05 * value
    )


def test_candidate_columns_replace_only_the_margin_gap() -> None:
    """The new set keeps the defender width and swaps one column."""
    assert len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_QBELO538) == len(
        PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ
    )
    changed: list[tuple[str, str]] = [
        (old, new)
        for old, new in zip(
            PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
            PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ_L2_QBELO538,
            strict=True,
        )
        if old != new
    ]
    assert changed == [(ELO_MOV_DIFF_COLUMN, QBELO_DIFF_COLUMN)]


def test_qbelo_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023 and the one fixed candidate."""
    argv: list[str] = qbelo538_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "qbelo538"
    assert argv[argv.index("--reference-model") + 1] == "logistic"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_qbelo538"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "qbelo538")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_qbelo538"
    assert spec.report_profile == "qbelo538"
    assert spec.reference_feature_set == FEATURE_SET_PKG_CORE_INJ_L2
    assert spec.logistic_feature_sets == (
        "h2_full",
        "pkg_core_inj_l2",
        "pkg_core_inj_l2_qbelo538",
    )
    assert spec.note.endswith("No model is promoted from this run.")
    fixed: dict[str, float | None] = {row.feature_set: row.fixed_c for row in spec.model_specs}
    assert fixed["h2_full"] is None
    assert fixed["pkg_core_inj_l2"] == pytest.approx(0.1)
    assert fixed["pkg_core_inj_l2_qbelo538"] == pytest.approx(0.1)
    assert locked_logistic_h2_full_n() == 2191


def test_holdout_refuses_the_qb_elo_set_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025."""
    args: argparse.Namespace = argparse.Namespace(feature_set=FEATURE_SET_PKG_CORE_INJ_L2_QBELO538)
    with pytest.raises(ValueError, match="2016-2023"):
        run_holdout(args)


def test_qbelo_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_qbelo538.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_qbelo538_main()


@pytest.mark.cross_platform
def test_qbelo_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-qbelo538:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_qbelo538.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_qbelo538.py" in script
    assert '"train-qbelo538" { Invoke-TrainQbelo538 }' in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script


def _write_oof(path: Path, frame: pd.DataFrame) -> None:
    """Write one OOF file as UTF-8."""
    frame.to_csv(path, index=False, encoding="utf-8")


@pytest.mark.cross_platform
def test_written_qbelo_cohort_reads_utf8(tmp_path: Path) -> None:
    """The command checks the OOF files, the context labels, and coverage."""
    reference: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2018, 2019],
            "y_true": [1, 0],
            "y_proba": [0.6, 0.4],
        }
    )
    _write_oof(tmp_path / "oof_logistic_pkg_core_inj_l2.csv", reference)
    for _name, file_name in (
        ("h2_full", "oof_logistic_h2_full.csv"),
        ("elo", "oof_elo_elo.csv"),
        (MODEL_ELO_MOV, "oof_elo_mov_elo_mov.csv"),
        (MODEL_QBELO538, "oof_qbelo538_qbelo538.csv"),
        ("candidate", "oof_logistic_pkg_core_inj_l2_qbelo538.csv"),
    ):
        _write_oof(tmp_path / file_name, reference)
    acceptance: pd.DataFrame = pd.DataFrame(
        [
            {"model": "market", "feature_set": "market", "decision": "benchmark"},
            {"model": "elo_mov", "feature_set": "elo_mov", "decision": "baseline"},
            {"model": "qbelo538", "feature_set": "qbelo538", "decision": "baseline"},
        ]
    )
    acceptance.to_csv(tmp_path / "acceptance.csv", index=False, encoding="utf-8")
    coverage: pd.DataFrame = pd.DataFrame(
        {
            "season": list(range(2016, 2024)),
            "n": [10] * 8,
            "starter_both_share": [1.0] * 8,
            "history_both_share": [0.5] * 8,
        }
    )
    assert list(coverage.columns) == list(QB_COVERAGE_COLUMNS)
    coverage.to_csv(tmp_path / "qb_coverage.csv", index=False, encoding="utf-8")
    summary: dict[str, int] = require_written_cohort(tmp_path)
    require_market_benchmark(tmp_path)
    require_context_baselines(tmp_path)
    require_qb_coverage(tmp_path)
    assert summary["n"] == 2
    assert "qbelo538 cohort" in cohort_message(summary["n"], 2191)


def test_feature_gap_ignores_a_future_box_score() -> None:
    """The built column for an earlier game does not see a later box score."""
    games, epa = _multiweek_slate(weeks=3)
    games = _with_schedule(games)
    starters, dropbacks = _starter_tables(games)
    empty_starters, empty_reports = _empty_injuries()
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        rows.append(
            _box_row(
                str(game.game_id),
                str(game.home_team_franchise),
                f"QB-{game.home_team_franchise}",
                counts=(10, 6, 100, 1, 0, 0, 1, 10, 0),
            )
        )
        rows.append(
            _box_row(
                str(game.game_id),
                str(game.away_team_franchise),
                f"QB-{game.away_team_franchise}",
                counts=(10, 4, 40, 0, 1, 2, 1, 5, 0),
            )
        )
    boxes: pd.DataFrame = pd.DataFrame(rows)
    last_id: str = str(games.sort_values(["gameday", "game_id"]).iloc[-1]["game_id"])
    poisoned: pd.DataFrame = boxes.copy()
    poisoned.loc[poisoned["game_id"] == last_id, "passing_yards"] = 50000.0
    shared: dict[str, object] = {
        "lookback_games": 2,
        "expected_starters": starters,
        "qb_dropbacks": dropbacks,
        "depth_chart_starters": empty_starters,
        "injury_reports": empty_reports,
    }
    original: pd.DataFrame = build_pregame_features(games, epa, qb_box_scores=boxes, **shared)
    changed: pd.DataFrame = build_pregame_features(games, epa, qb_box_scores=poisoned, **shared)
    early: pd.Series = original.loc[original["game_id"] != last_id, QBELO_DIFF_COLUMN]
    early_changed: pd.Series = changed.loc[changed["game_id"] != last_id, QBELO_DIFF_COLUMN]
    np.testing.assert_allclose(early.to_numpy(dtype=float), early_changed.to_numpy(dtype=float))
    first_day = games["gameday"].min()
    first_ids: set[str] = set(games.loc[games["gameday"] == first_day, "game_id"].astype(str))
    first: pd.DataFrame = original.loc[original["game_id"].isin(first_ids)]
    np.testing.assert_allclose(
        first[QBELO_DIFF_COLUMN].to_numpy(dtype=float),
        first[ELO_MOV_DIFF_COLUMN].to_numpy(dtype=float),
    )
    bare: pd.DataFrame = build_pregame_features(games, epa, **shared)
    np.testing.assert_allclose(
        bare[QBELO_DIFF_COLUMN].to_numpy(dtype=float),
        bare[ELO_MOV_DIFF_COLUMN].to_numpy(dtype=float),
    )


def test_qbelo_report_scores_the_gap_and_keeps_context_rows_baseline() -> None:
    """Context probabilities use the 65 and 400 formula. Coverage stays in the window."""
    games, epa = _multiweek_slate(weeks=6)
    games = _with_schedule(games)
    starters, dropbacks = _starter_tables(games)
    empty_starters, empty_reports = _empty_injuries()
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        rows.append(
            _box_row(
                str(game.game_id),
                str(game.home_team_franchise),
                f"QB-{game.home_team_franchise}",
            )
        )
    boxes: pd.DataFrame = pd.DataFrame(rows)
    features: pd.DataFrame = build_pregame_features(
        games,
        epa,
        lookback_games=4,
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        depth_chart_starters=empty_starters,
        injury_reports=empty_reports,
        qb_box_scores=boxes,
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
        hypothesis="qbelo538",
        expected_starters=starters,
        qb_dropbacks=dropbacks,
        depth_chart_starters=empty_starters,
        injury_reports=empty_reports,
        qb_box_scores=boxes,
        random_state=42,
    )
    by_key: dict[tuple[str, str], dict[str, object]] = {
        (str(row["model"]), str(row["feature_set"])): row
        for row in report["results"]  # type: ignore[index]
    }
    candidate: dict[str, object] = by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2_QBELO538)]
    assert candidate["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert by_key[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)]["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    for model_name, column in (
        (MODEL_ELO_MOV, ELO_MOV_DIFF_COLUMN),
        (MODEL_QBELO538, QBELO_DIFF_COLUMN),
    ):
        oof: pd.DataFrame = by_key[(model_name, model_name)]["oof"]  # type: ignore[assignment]
        assert set(int(season) for season in oof["season"]) == {2017}
        assert int(oof["season"].max()) < 2024
        merged: pd.DataFrame = oof.merge(
            features.loc[:, ["game_id", column]],
            on="game_id",
            how="left",
        )
        expected: np.ndarray = rating_win_probability(merged[column].to_numpy(dtype=float))
        np.testing.assert_allclose(merged["y_proba"].to_numpy(dtype=float), expected)
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] in {"accepted", "rejected", "inconclusive"}
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2_QBELO538)] in {
        "accepted",
        "rejected",
        "inconclusive",
    }
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[(MODEL_ELO_MOV, MODEL_ELO_MOV)] == "baseline"
    assert decisions[(MODEL_QBELO538, MODEL_QBELO538)] == "baseline"
    coverage: pd.DataFrame = report["qb_coverage"]  # type: ignore[assignment]
    assert list(coverage["season"]) == [2017]
    assert float(coverage.iloc[0]["starter_both_share"]) == pytest.approx(1.0)
    assert 0.0 <= float(coverage.iloc[0]["history_both_share"]) <= 1.0
    markdown: str = str(report["markdown"])
    assert "## QB coverage" in markdown
    assert "| elo_mov |" in markdown
    assert "| qbelo538 |" in markdown
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert {MODEL_ELO_MOV, MODEL_QBELO538} <= set(calibration["model"])
    assert {MODEL_ELO_MOV, MODEL_QBELO538} <= set(information["model"])
