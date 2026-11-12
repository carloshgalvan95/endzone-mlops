"""The defender-plus-Elo mean and stack stay on the locked 2016-2023 recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.pregame import (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_PKG_CORE_INJ_L2,
    elo_home_probabilities,
)
from endzone.modeling.cv import RollingOriginConfig, train_logistic_regression
from endzone.modeling.ensembles import (
    LOGIT_CLIP,
    MODEL_ENS_MEAN_ELO,
    MODEL_STACK_ELO,
    STACK_INNER_START_SEASON,
    STACKER_C,
    collect_inner_member_rows,
    elo_mean_result,
    fit_probability_stacker,
    mean_aligned_oof,
    mean_probabilities,
    stacker_probabilities,
    stacker_training_rows,
)
from endzone.modeling.hypotheses import CvModelSpec, experiment_for
from endzone.modeling.models import MODEL_LOGISTIC
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.report import _market_information_lines
from endzone.modeling.train import ensembles_r2_challenger_argv
from scripts.train_ensembles_r2 import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_ensembles_r2 import (
    main as train_ensembles_r2_main,
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


def _copy_season(
    games: pd.DataFrame,
    epa: pd.DataFrame,
    source: int,
    dest: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat one season a year earlier so an inner fold has history."""
    copied_games: pd.DataFrame = games.loc[games["season"] == source].copy()
    copied_games["season"] = dest
    copied_games["game_id"] = (
        copied_games["game_id"].astype(str).str.replace(f"{source}-", f"{dest}-", n=1, regex=False)
    )
    copied_games["gameday"] = pd.to_datetime(copied_games["gameday"]) - pd.DateOffset(
        years=source - dest
    )
    copied_epa: pd.DataFrame = epa.loc[epa["season"] == source].copy()
    copied_epa["season"] = dest
    copied_epa["game_id"] = (
        copied_epa["game_id"].astype(str).str.replace(f"{source}-", f"{dest}-", n=1, regex=False)
    )
    if "gameday" in copied_epa.columns:
        copied_epa["gameday"] = pd.to_datetime(copied_epa["gameday"]) - pd.DateOffset(
            years=source - dest
        )
    return (
        pd.concat([copied_games, games], ignore_index=True),
        pd.concat([copied_epa, epa], ignore_index=True),
    )


def _history_slate() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Seasons 2009, 2010, 2011, and a 2018 sentinel the stacker must not see."""
    matrix_rows: list[dict[str, object]] = []
    game_rows: list[dict[str, object]] = []
    labels: list[int] = []
    homes: tuple[tuple[str, str], ...] = (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"))
    for season in (2009, 2010, 2011, 2018):
        for index, (home, away) in enumerate(homes):
            label: int = index % 2
            game_id: str = f"{season}-{index}"
            day: pd.Timestamp = pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * index)
            matrix_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "gameday": day,
                    "signal": float(label),
                }
            )
            game_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "gameday": day,
                    "home_team_franchise": home,
                    "away_team_franchise": away,
                    "home_won": label,
                    "home_score": 24 if label else 14,
                    "away_score": 13 if label else 21,
                }
            )
            labels.append(label)
    matrix: pd.DataFrame = pd.DataFrame(matrix_rows)
    target: pd.Series = pd.Series(labels, dtype=int)
    games: pd.DataFrame = pd.DataFrame(game_rows)
    return matrix, target, games


def test_ensembles_r2_command_locks_the_development_window() -> None:
    """The argv and the YAML stay on 2016-2023 and the two fixed candidates."""
    argv: list[str] = ensembles_r2_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert "2024" not in argv
    assert "2025" not in argv
    assert argv[argv.index("--hypothesis") + 1] == "ensembles_r2"
    assert argv[argv.index("--reference-feature-set") + 1] == "pkg_core_inj_l2"
    assert argv[argv.index("--output-dir") + 1] == "reports/phase2_ensembles_r2"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    spec = experiment_for(2, "ensembles_r2")
    assert spec.experiment_name == "nfl_pregame/phase2_ensembles_r2"
    assert spec.reference_model == "logistic"
    assert spec.reference_feature_set == FEATURE_SET_PKG_CORE_INJ_L2
    assert spec.output_dir == "reports/phase2_ensembles_r2"
    assert spec.report_profile == "ensembles_r2"
    assert spec.logistic_feature_sets == ("h2_full", "pkg_core_inj_l2")
    assert spec.note.endswith("No model is promoted from this run.")
    assert [row.model for row in spec.model_specs] == [
        "logistic",
        "logistic",
        MODEL_ENS_MEAN_ELO,
        MODEL_STACK_ELO,
    ]
    fixed: dict[str, float | None] = {
        row.feature_set: row.fixed_c for row in spec.model_specs if row.model == "logistic"
    }
    assert fixed["h2_full"] is None
    assert fixed["pkg_core_inj_l2"] == pytest.approx(0.1)
    assert STACKER_C == pytest.approx(1.0)
    assert LOGIT_CLIP == pytest.approx(1e-6)
    assert STACK_INNER_START_SEASON == 2010
    assert locked_logistic_h2_full_n() == 2191


def test_ensembles_r2_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_ensembles_r2.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_ensembles_r2_main()


@pytest.mark.cross_platform
def test_ensembles_r2_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-ensembles-r2:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[
        0
    ]
    assert "python scripts/train_ensembles_r2.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_ensembles_r2.py" in script
    assert '"train-ensembles-r2" { Invoke-TrainEnsemblesR2 }' in script
    assert "C:\\" not in recipe
    assert "C:\\" not in script


@pytest.mark.cross_platform
def test_written_r2_cohort_and_market_benchmark_read_utf8(tmp_path: Path) -> None:
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
        "oof_elo_elo.csv",
        "oof_ens_mean_elo_pkg_core_inj_l2.csv",
        "oof_stack_elo_pkg_core_inj_l2.csv",
    )
    for name in names:
        reference.to_csv(tmp_path / name, index=False, encoding="utf-8")
    summary: dict[str, int] = require_written_cohort(tmp_path)
    assert summary == {"n": 2, "variants": 4}
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


def test_mean_equals_the_average_of_the_members() -> None:
    """The mean follows the defender's game order when Elo is reversed."""
    left: np.ndarray = np.array([0.25, 0.75])
    right: np.ndarray = np.array([0.75, 0.25])
    np.testing.assert_array_equal(mean_probabilities(left, right), np.array([0.5, 0.5]))
    logistic: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2017, 2017],
            "y_true": [0, 1],
            "y_proba": [0.25, 0.75],
        }
    )
    elo: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["g2", "g1"],
            "season": [2017, 2017],
            "y_true": [1, 0],
            "y_proba": [0.25, 0.75],
        }
    )
    aligned: pd.DataFrame = mean_aligned_oof(logistic, elo)
    np.testing.assert_array_equal(aligned["y_proba"].to_numpy(dtype=float), np.array([0.5, 0.5]))
    assert list(aligned["game_id"]) == ["g1", "g2"]
    folds: list[dict[str, object]] = [
        {
            "season": 2017,
            "status": "scored",
            "reason": None,
            "n_train": 4,
            "n_eval": 2,
            "train_seasons": [2016],
            "eval_season": 2017,
        }
    ]
    built: dict[str, object] = elo_mean_result(
        [
            {
                "model": MODEL_LOGISTIC,
                "feature_set": FEATURE_SET_PKG_CORE_INJ_L2,
                "status": "ok",
                "folds": folds,
                "oof": logistic,
            },
            {
                "model": "elo",
                "feature_set": "elo",
                "status": "ok",
                "folds": folds,
                "oof": elo,
            },
        ],
        CvModelSpec(MODEL_ENS_MEAN_ELO, FEATURE_SET_PKG_CORE_INJ_L2),
        {},
    )
    assert built["status"] == "ok"
    assert built["params"]["learned_weights"] is False  # type: ignore[index]
    oof: pd.DataFrame = built["oof"]  # type: ignore[assignment]
    np.testing.assert_array_equal(oof["y_proba"].to_numpy(dtype=float), np.array([0.5, 0.5]))


def test_stacker_is_deterministic_with_clipped_probabilities() -> None:
    """Two stacker fits match, including probabilities that sit on 0 and 1."""
    seasons: np.ndarray = np.array([2010, 2010, 2011, 2011])
    labels: np.ndarray = np.array([0, 1, 0, 1])
    p_logistic: np.ndarray = np.array([0.0, 1.0, 0.25, 0.75])
    p_elo: np.ndarray = np.array([0.2, 0.8, 0.0, 1.0])
    first = fit_probability_stacker(seasons, labels, p_logistic, p_elo)
    second = fit_probability_stacker(seasons, labels, p_logistic, p_elo)
    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.intercept_, second.intercept_)
    assert first.C == pytest.approx(1.0)
    assert first.fit_intercept is True
    predicted: np.ndarray = stacker_probabilities(first, p_logistic, p_elo)
    repeated: np.ndarray = stacker_probabilities(second, p_logistic, p_elo)
    np.testing.assert_array_equal(predicted, repeated)
    assert predicted.shape == (4,)
    assert np.isfinite(predicted).all()


def test_stacker_training_rows_exclude_the_scored_season_and_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inner rows start in 2010, and season S and every later season stay out."""
    matrix, labels, games = _history_slate()
    seen_member: list[tuple[int, int]] = []
    seen_elo: list[set[int]] = []

    def spy_logistic(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
        **kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        train_max: int = int(train_frame["season"].astype(int).max())
        eval_season: int = int(eval_frame["season"].astype(int).min())
        assert train_max < eval_season
        seen_member.append((train_max, eval_season))
        return train_logistic_regression(
            train_frame,
            train_target,
            eval_frame,
            eval_target,
            **kwargs,  # type: ignore[arg-type]
        )

    def spy_elo(frame: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        seasons: set[int] = {int(season) for season in frame["season"]}
        assert 2018 not in seasons
        assert max(seasons) <= 2011
        seen_elo.append(seasons)
        return elo_home_probabilities(frame, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("endzone.modeling.cv.train_logistic_regression", spy_logistic)
    monkeypatch.setattr("endzone.modeling.cv.elo_home_probabilities", spy_elo)
    empty: pd.DataFrame = pd.DataFrame(columns=["game_id", "season", "y_true", "y_proba"])
    inner: pd.DataFrame = collect_inner_member_rows(
        matrix,
        labels,
        games,
        empty,
        empty,
        through_season=2011,
        c_value=0.1,
        max_iter=200,
        random_state=42,
    )
    assert seen_member
    assert seen_elo
    assert set(int(season) for season in inner["season"]) <= {2010, 2011}
    assert 2009 not in set(int(season) for season in inner["season"])
    assert 2018 not in set(int(season) for season in inner["season"])
    scored_season: int = 2016
    training: pd.DataFrame = stacker_training_rows(inner, scored_season)
    assert not training.empty
    assert int(training["season"].min()) >= STACK_INNER_START_SEASON
    assert int(training["season"].max()) < scored_season
    assert scored_season not in set(int(season) for season in training["season"])
    later: pd.DataFrame = pd.concat(
        [
            inner,
            pd.DataFrame(
                {
                    "game_id": ["later"],
                    "season": [scored_season],
                    "y_true": [1],
                    "p_logistic": [0.6],
                    "p_elo": [0.4],
                }
            ),
        ],
        ignore_index=True,
    )
    held: pd.DataFrame = stacker_training_rows(later, scored_season)
    assert int(held["season"].max()) < scored_season
    assert scored_season not in set(int(season) for season in held["season"])


def test_market_information_markdown_names_the_model() -> None:
    """Two rows on the same feature set stay distinct by model name."""
    table: pd.DataFrame = pd.DataFrame(
        {
            "model": ["logistic", MODEL_ENS_MEAN_ELO, MODEL_STACK_ELO],
            "feature_set": [FEATURE_SET_PKG_CORE_INJ_L2] * 3,
            "status": ["ok", "ok", "ok"],
            "n": [8, 8, 8],
            "market_coef": [0.1, 0.2, 0.3],
            "market_ci_low": [0.0, 0.0, 0.0],
            "market_ci_high": [0.2, 0.3, 0.4],
            "model_coef": [0.4, 0.5, 0.6],
            "model_ci_low": [0.1, 0.2, 0.3],
            "model_ci_high": [0.5, 0.6, 0.7],
        }
    )
    rendered: str = "\n".join(_market_information_lines(table))
    header: str = next(line for line in rendered.splitlines() if line.startswith("| model |"))
    assert header.index("model") < header.index("feature_set")
    assert rendered.count(FEATURE_SET_PKG_CORE_INJ_L2) >= 3
    assert "| logistic |" in rendered
    assert f"| {MODEL_ENS_MEAN_ELO} |" in rendered
    assert f"| {MODEL_STACK_ELO} |" in rendered


def test_round2_report_mean_matches_members_and_stacker_excludes_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline mean is the member average, and the stacker fit stops before S."""
    games, epa = _multiweek_slate(weeks=8)
    games, epa = _copy_season(games, epa, 2016, 2015)
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
    fitted_seasons: list[np.ndarray] = []
    real_fit = fit_probability_stacker

    def spy_fit(
        seasons: np.ndarray,
        y_true: np.ndarray,
        p_logistic: np.ndarray,
        p_elo: np.ndarray,
    ) -> object:
        fitted_seasons.append(np.asarray(seasons, dtype=int).copy())
        return real_fit(seasons, y_true, p_logistic, p_elo)

    monkeypatch.setattr("endzone.modeling.ensembles.fit_probability_stacker", spy_fit)
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
        hypothesis="ensembles_r2",
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
    elo: dict[str, object] = by_key[("elo", "elo")]
    mean_row: dict[str, object] = by_key[(MODEL_ENS_MEAN_ELO, FEATURE_SET_PKG_CORE_INJ_L2)]
    stack_row: dict[str, object] = by_key[(MODEL_STACK_ELO, FEATURE_SET_PKG_CORE_INJ_L2)]
    assert logistic["params"]["C"] == pytest.approx(0.1)  # type: ignore[index]
    assert mean_row["status"] == "ok"
    assert stack_row["status"] == "ok"
    assert stack_row["params"]["C"] == pytest.approx(1.0)  # type: ignore[index]
    assert stack_row["params"]["logit_clip"] == pytest.approx(1e-6)  # type: ignore[index]
    logistic_oof: pd.DataFrame = logistic["oof"]  # type: ignore[assignment]
    elo_oof: pd.DataFrame = elo["oof"]  # type: ignore[assignment]
    mean_oof: pd.DataFrame = mean_row["oof"]  # type: ignore[assignment]
    stack_oof: pd.DataFrame = stack_row["oof"]  # type: ignore[assignment]
    expected: pd.DataFrame = mean_aligned_oof(logistic_oof, elo_oof)
    np.testing.assert_allclose(
        mean_oof["y_proba"].to_numpy(dtype=float),
        expected["y_proba"].to_numpy(dtype=float),
    )
    assert list(mean_oof["game_id"]) == list(logistic_oof["game_id"])
    assert set(int(season) for season in stack_oof["season"]) == {2017}
    assert fitted_seasons
    for seasons in fitted_seasons:
        assert int(seasons.min()) >= STACK_INNER_START_SEASON
        assert int(seasons.max()) < 2017
        assert 2017 not in set(int(season) for season in seasons)
    coefficients: pd.DataFrame = report["stacker_coefficients"]  # type: ignore[assignment]
    scored: pd.DataFrame = coefficients.loc[coefficients["status"] == "scored"]
    assert len(scored) == 1
    assert int(scored.iloc[0]["season"]) == 2017
    inner_seasons: list[int] = [
        int(season) for season in str(scored.iloc[0]["inner_seasons"]).split(",") if season
    ]
    assert inner_seasons
    assert max(inner_seasons) < 2017
    assert min(inner_seasons) >= STACK_INNER_START_SEASON
    assert np.isfinite(float(scored.iloc[0]["intercept"]))
    assert np.isfinite(float(scored.iloc[0]["coef_logit_pkg_core_inj_l2"]))
    assert np.isfinite(float(scored.iloc[0]["coef_logit_elo"]))
    markdown: str = str(report["markdown"])
    assert "## Stacker coefficients" in markdown
    assert f"| {MODEL_ENS_MEAN_ELO} |" in markdown
    assert f"| {MODEL_STACK_ELO} |" in markdown
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("logistic", FEATURE_SET_PKG_CORE_INJ_L2)] == "reference"
    assert decisions[("logistic", FEATURE_SET_H2_FULL)] in {"accepted", "rejected", "inconclusive"}
    assert decisions[("market", "market")] == "benchmark"
    assert decisions[(MODEL_ENS_MEAN_ELO, FEATURE_SET_PKG_CORE_INJ_L2)] in {
        "accepted",
        "rejected",
        "inconclusive",
    }
    assert decisions[(MODEL_STACK_ELO, FEATURE_SET_PKG_CORE_INJ_L2)] in {
        "accepted",
        "rejected",
        "inconclusive",
    }
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert MODEL_ENS_MEAN_ELO in set(information["model"])
    assert MODEL_STACK_ELO in set(information["model"])
