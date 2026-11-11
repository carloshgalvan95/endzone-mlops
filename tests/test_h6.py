"""H6 opponent adjustment, Elo margin, calibration, and the fixed command."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from endzone.features.opponent import (
    ELO_MOV_DIFF_COLUMN,
    H6_ADJ_FEATURE_COLUMNS,
    H6_CHALLENGER_FEATURE_SETS,
    RIDGE_ALPHA,
    adjust_game_epa,
    fit_opponent_ridge,
    opponent_adjusted_team_games,
)
from endzone.features.pregame import (
    ELO_HOME_ADVANTAGE,
    H2_CONTEXT_COLUMNS,
    PREGAME_FEATURE_COLUMNS_H6_ADJ,
    PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO,
    build_pregame_features,
    elo_home_probabilities,
    feature_columns_for,
    margin_of_victory_multiplier,
)
from endzone.modeling.cv import RollingOriginConfig
from endzone.modeling.dataset import validate_feature_columns
from endzone.modeling.evaluation import (
    calibration_table_for_results,
    label_market_benchmark,
    market_information_regression,
    reliability_by_decile,
)
from endzone.modeling.holdout import run_holdout
from endzone.modeling.hypotheses import experiment_for
from endzone.modeling.models import feature_column_groups
from endzone.modeling.pipeline import run_phase1_baselines
from endzone.modeling.train import h6_challenger_argv
from scripts.train_h6 import (
    cohort_message,
    locked_logistic_h2_full_n,
    require_market_benchmark,
    require_written_cohort,
)
from scripts.train_h6 import (
    main as train_h6_main,
)


def _ridge_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A offense is strong. B defense faces that offense."""
    offense: np.ndarray = np.array(["A", "B", "A", "B", "C", "A"] * 4)
    defense: np.ndarray = np.array(["B", "A", "C", "C", "B", "B"] * 4)
    home_sign: np.ndarray = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0] * 4)
    target: np.ndarray = np.array([2.0, 0.0, 1.5, 0.2, 0.4, 1.8] * 4)
    return offense, defense, home_sign, target


def test_ridge_matches_an_independent_fit_and_removes_a_weak_defense() -> None:
    """The locked design matches sklearn, and a weak defense is removed."""
    offense, defense, home_sign, target = _ridge_sample()
    fitted: dict[str, object] | None = fit_opponent_ridge(
        offense, defense, home_sign, target, alpha=RIDGE_ALPHA
    )
    assert fitted is not None
    teams: list[str] = list(fitted["teams"])  # type: ignore[arg-type]
    team_count: int = len(teams)
    index: dict[str, int] = {team: position for position, team in enumerate(teams)}
    design: np.ndarray = np.zeros((len(target), 2 * team_count + 1), dtype=float)
    rows: np.ndarray = np.arange(len(target))
    design[rows, [index[str(team)] for team in offense]] = 1.0
    design[rows, [team_count + index[str(team)] for team in defense]] = 1.0
    design[:, -1] = home_sign
    independent: Ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="cholesky")
    independent.fit(design, target)
    offense_map: dict[str, float] = fitted["offense"]  # type: ignore[assignment]
    defense_map: dict[str, float] = fitted["defense"]  # type: ignore[assignment]
    for position, team in enumerate(teams):
        assert offense_map[team] == pytest.approx(float(independent.coef_[position]))
        assert defense_map[team] == pytest.approx(float(independent.coef_[team_count + position]))
    assert fitted["home"] == pytest.approx(float(independent.coef_[-1]))
    assert defense_map["B"] > defense_map["A"]
    against_weak: dict[str, float] = adjust_game_epa(2.0, 0.1, "B", 1.0, fitted)
    against_strong: dict[str, float] = adjust_game_epa(2.0, 0.1, "A", 1.0, fitted)
    assert against_weak["adj_offensive_epa"] < against_strong["adj_offensive_epa"]
    unseen: dict[str, float] = adjust_game_epa(0.5, 0.25, "Z", 1.0, fitted)
    home_coef: float = float(fitted["home"])  # type: ignore[arg-type]
    assert unseen["adj_offensive_epa"] == pytest.approx(0.5 - home_coef)
    assert fit_opponent_ridge(offense[:3], defense[:3], home_sign[:3], target[:3]) is None


def test_margin_multiplier_matches_the_published_formula() -> None:
    """A 14-point win and a tie use the FiveThirtyEight factor."""
    blowout: float = margin_of_victory_multiplier(14.0, 100.0, 1.0)
    assert blowout == pytest.approx(math.log(15.0) * 2.2 / 2.3)
    tie: float = margin_of_victory_multiplier(0.0, 40.0, 0.5)
    assert tie == pytest.approx(math.log(2.0) * 2.2)


def _two_game_slate(home_score: float, away_score: float) -> pd.DataFrame:
    """One decided game, then a second game on a later date."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2020, 2020],
            "gameday": pd.to_datetime(["2020-09-10", "2020-09-17"]),
            "home_team_franchise": ["H", "H"],
            "away_team_franchise": ["A", "A"],
            "home_won": [1, 1],
            "home_score": [home_score, 20.0],
            "away_score": [away_score, 10.0],
        }
    )


def test_margin_of_victory_moves_a_blowout_more_than_a_close_win() -> None:
    """The default Elo path is unchanged. The margin path is not."""
    close: pd.DataFrame = _two_game_slate(21.0, 20.0)
    blowout: pd.DataFrame = _two_game_slate(35.0, 7.0)
    plain: dict[str, object] = elo_home_probabilities(blowout)
    flagged_off: dict[str, object] = elo_home_probabilities(blowout, margin_of_victory=False)
    assert np.allclose(plain["probabilities"], flagged_off["probabilities"])  # type: ignore[arg-type]
    assert plain["margin_of_victory"] is False
    moved: dict[str, object] = elo_home_probabilities(blowout, margin_of_victory=True)
    close_moved: dict[str, object] = elo_home_probabilities(close, margin_of_victory=True)
    blowout_gap: float = float(moved["home_ratings"].iloc[1] - moved["away_ratings"].iloc[1])  # type: ignore[union-attr]
    close_gap: float = float(
        close_moved["home_ratings"].iloc[1] - close_moved["away_ratings"].iloc[1]  # type: ignore[union-attr]
    )
    plain_gap: float = float(plain["home_ratings"].iloc[1] - plain["away_ratings"].iloc[1])  # type: ignore[union-attr]
    assert blowout_gap > close_gap
    assert blowout_gap != pytest.approx(plain_gap)
    assert moved["margin_of_victory"] is True
    assert ELO_HOME_ADVANTAGE == 65


def _team_games_for(games: pd.DataFrame, offensive: float) -> pd.DataFrame:
    """One offensive EPA for every team-game."""
    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        for team in (game.home_team_franchise, game.away_team_franchise):
            rows.append(
                {
                    "game_id": game.game_id,
                    "season": game.season,
                    "gameday": game.gameday,
                    "team_franchise": team,
                    "avg_offensive_epa": offensive,
                    "avg_defensive_epa_allowed": offensive / 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_adjusted_epa_ignores_same_day_and_later_games() -> None:
    """A game's adjusted EPA uses only earlier gamedays."""
    games: pd.DataFrame = pd.DataFrame(
        {
            "game_id": ["early", "same", "later"],
            "season": [2020, 2020, 2020],
            "gameday": pd.to_datetime(["2020-09-10", "2020-09-10", "2020-09-17"]),
            "home_team_franchise": ["A", "C", "A"],
            "away_team_franchise": ["B", "D", "B"],
            "home_won": [1, 0, 1],
            "home_score": [24, 10, 21],
            "away_score": [10, 17, 14],
        }
    )
    baseline: pd.DataFrame = opponent_adjusted_team_games(games, _team_games_for(games, 0.2))
    poisoned_games: pd.DataFrame = games.copy()
    poisoned_games.loc[poisoned_games["game_id"] != "early", "home_score"] = 70
    poisoned_epa: pd.DataFrame = _team_games_for(games, 0.2)
    poisoned_epa.loc[poisoned_epa["game_id"] != "early", "avg_offensive_epa"] = 9.0
    poisoned: pd.DataFrame = opponent_adjusted_team_games(poisoned_games, poisoned_epa)
    early_base: pd.DataFrame = baseline.loc[baseline["game_id"] == "early"].reset_index(drop=True)
    early_poison: pd.DataFrame = poisoned.loc[poisoned["game_id"] == "early"].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_base, early_poison)
    later_base: pd.Series = baseline.loc[baseline["game_id"] == "later", "adj_offensive_epa"]
    later_poison: pd.Series = poisoned.loc[poisoned["game_id"] == "later", "adj_offensive_epa"]
    assert not np.allclose(
        later_base.to_numpy(dtype=float),
        later_poison.to_numpy(dtype=float),
        equal_nan=True,
    )


def _multiweek_slate(weeks: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enough weeks that week-1 nulls are under half the rows."""
    games: list[dict[str, object]] = []
    epa_rows: list[dict[str, object]] = []
    for season in (2016, 2017):
        for week in range(weeks):
            day: pd.Timestamp = pd.Timestamp(f"{season}-09-04") + pd.Timedelta(days=7 * week)
            pairings: tuple[tuple[str, str], ...] = (
                (("A", "B"), ("C", "D")) if week % 2 == 0 else (("A", "C"), ("B", "D"))
            )
            for index, (home, away) in enumerate(pairings):
                game_id: str = f"{season}-{week}-{index}"
                home_won: int = 0 if (week + index) % 3 == 0 else 1
                games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "gameday": day,
                        "home_team_franchise": home,
                        "away_team_franchise": away,
                        "home_won": home_won,
                        "home_score": 27 if home_won else 14,
                        "away_score": 13 if home_won else 24,
                        "home_rest": 7,
                        "away_rest": 7,
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                    }
                )
                for team, offense in ((home, 0.4), (away, -0.1)):
                    epa_rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "gameday": day,
                            "team_franchise": team,
                            "avg_offensive_epa": offense if team == "A" else offense / 2.0,
                            "avg_defensive_epa_allowed": 0.05,
                        }
                    )
    return pd.DataFrame(games), pd.DataFrame(epa_rows)


def test_h6_columns_match_h2_and_stay_under_half_null() -> None:
    """Adjusted names replace EPA, and a short slate is usable."""
    renamed: set[str] = {
        column for column in PREGAME_FEATURE_COLUMNS_H6_ADJ if column not in H2_CONTEXT_COLUMNS
    }
    assert renamed == set(H6_ADJ_FEATURE_COLUMNS)
    assert feature_columns_for("h6_adj") == PREGAME_FEATURE_COLUMNS_H6_ADJ
    assert feature_columns_for("h6_adj_elo") == PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO
    assert ELO_MOV_DIFF_COLUMN in PREGAME_FEATURE_COLUMNS_H6_ADJ_ELO
    assert ELO_MOV_DIFF_COLUMN not in PREGAME_FEATURE_COLUMNS_H6_ADJ
    games, epa = _multiweek_slate()
    features: pd.DataFrame = build_pregame_features(games, epa, lookback_games=10)
    validate_feature_columns(features, ("h6_adj", "h6_adj_elo"))
    groups: dict[str, list[str]] = feature_column_groups(
        ["elo_mov_diff", "home_recent_adj_offensive_epa_per_play", "home_rest_days"]
    )
    assert groups["form"] == ["elo_mov_diff"]
    assert groups["epa"] == ["home_recent_adj_offensive_epa_per_play"]
    assert groups["other"] == ["home_rest_days"]


def test_reliability_deciles_and_brier_components() -> None:
    """The fixture's bin components are fixed, and they are not the raw Brier."""
    labels: np.ndarray = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    probabilities: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    summary: dict[str, object] = reliability_by_decile(labels, probabilities, n_bins=2)
    bins: list[dict[str, float]] = summary["bins"]  # type: ignore[assignment]
    assert bins[0]["mean_predicted"] == pytest.approx(0.25)
    assert bins[0]["observed_rate"] == pytest.approx(0.0)
    assert bins[1]["mean_predicted"] == pytest.approx(0.75)
    assert bins[1]["observed_rate"] == pytest.approx(1.0)
    assert summary["reliability"] == pytest.approx(0.0625)
    assert summary["resolution"] == pytest.approx(0.25)
    assert summary["uncertainty"] == pytest.approx(0.25)
    assert summary["ece"] == pytest.approx(0.25)
    binned: float = (
        float(summary["reliability"]) - float(summary["resolution"]) + float(summary["uncertainty"])
    )
    assert binned == pytest.approx(0.0625)
    assert summary["brier"] == pytest.approx(0.075)
    assert summary["brier"] != pytest.approx(binned)


def test_market_information_returns_coefficients_and_no_blend() -> None:
    """The regression is descriptive. It does not return a blended probability."""
    labels: np.ndarray = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=float)
    market: np.ndarray = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.35, 0.65])
    model: np.ndarray = np.array([0.25, 0.35, 0.45, 0.55, 0.7, 0.75, 0.4, 0.6])
    fitted: dict[str, object] = market_information_regression(
        labels,
        market,
        model,
        n_bootstrap=20,
        seed=42,
    )
    assert fitted["status"] == "scored"
    assert fitted["n"] == 8
    assert fitted["n_bootstrap_used"] == 20
    assert isinstance(fitted["market_coef"], float)
    assert isinstance(fitted["model_coef"], float)
    assert float(fitted["market_ci_low"]) <= float(fitted["market_ci_high"])
    assert not any("blend" in str(key) for key in fitted)
    assert not any(isinstance(value, np.ndarray) for value in fitted.values())


def test_market_row_is_relabeled_benchmark() -> None:
    """The numeric accept label does not stay on the market row."""
    original: list[dict[str, object]] = [
        {"model": "market", "feature_set": "market", "decision": "accepted"},
        {"model": "logistic", "feature_set": "h6_adj", "decision": "rejected"},
    ]
    labeled: list[dict[str, object]] = label_market_benchmark(original)  # type: ignore[arg-type]
    assert labeled[0]["decision"] == "benchmark"
    assert original[0]["decision"] == "accepted"
    assert labeled[1]["decision"] == "rejected"


def test_h6_report_marks_the_market_and_keeps_calibration() -> None:
    """One small rolling-origin run writes the descriptive sections."""
    games, epa = _multiweek_slate(weeks=6)
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
        phase=2,
        hypothesis="h6",
        random_state=42,
    )
    decisions: dict[tuple[str, str], str] = {
        (str(row["model"]), str(row["feature_set"])): str(row["decision"])
        for row in report["acceptance"]  # type: ignore[index]
    }
    assert decisions[("market", "market")] == "benchmark"
    assert "accepted" not in decisions[("market", "market")]
    calibration: pd.DataFrame = report["calibration"]  # type: ignore[assignment]
    assert set(calibration["model"]) >= {"logistic", "elo", "market"}
    information: pd.DataFrame = report["market_information"]  # type: ignore[assignment]
    assert set(information["feature_set"]) == {"h2_full", "h6_adj", "h6_adj_elo"}
    assert "blend" not in information.columns
    markdown: str = str(report["markdown"])
    assert "benchmark" in markdown
    assert "## Calibration" in markdown
    assert "## Market information" in markdown
    assert "No blend is published." in markdown
    table: pd.DataFrame = calibration_table_for_results(report["results"])  # type: ignore[arg-type]
    assert not table.empty


def test_holdout_refuses_h6_sets_before_a_warehouse_read() -> None:
    """The single split scores 2024 and 2025, so it cannot select these sets."""
    for feature_set in H6_CHALLENGER_FEATURE_SETS:
        args: argparse.Namespace = argparse.Namespace(feature_set=feature_set)
        with pytest.raises(ValueError, match="2016-2023"):
            run_holdout(args)


def test_h6_command_locks_the_development_window() -> None:
    """The argv, the YAML, and the reference recipe stay on 2016-2023."""
    argv: list[str] = h6_challenger_argv()
    assert argv[argv.index("--cv-start-season") + 1] == "2016"
    assert argv[argv.index("--cv-end-season") + 1] == "2023"
    assert argv[argv.index("--lookback") + 1] == "10"
    assert argv[argv.index("--hypothesis") + 1] == "h6"
    assert argv[argv.index("--reference-feature-set") + 1] == "h2_full"
    assert argv[argv.index("--C") + 1] == "1.0"
    assert argv[argv.index("--max-iter") + 1] == "1000"
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--n-bootstrap") + 1] == "1000"
    assert argv[argv.index("--bootstrap-seed") + 1] == "42"
    assert "2024" not in argv
    assert "2025" not in argv
    spec = experiment_for(2, "h6")
    assert spec.default is False
    assert spec.experiment_name == "nfl_pregame/phase2_h6"
    assert spec.tag == "h6-opponent-adjusted"
    assert spec.report_profile == "h6"
    assert spec.reference_feature_set == "h2_full"
    assert spec.output_dir == argv[argv.index("--output-dir") + 1]
    assert spec.logistic_feature_sets == ("h2_full", *H6_CHALLENGER_FEATURE_SETS)
    assert spec.note.endswith("No model is promoted from this run.")
    assert locked_logistic_h2_full_n() == 2191


def test_h6_command_refuses_extra_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller cannot point this command at 2024 or 2025."""
    monkeypatch.setattr("sys.argv", ["train_h6.py", "--cv-end-season", "2025"])
    with pytest.raises(SystemExit, match="does not take arguments"):
        train_h6_main()


@pytest.mark.cross_platform
def test_h6_recipe_is_windows_safe() -> None:
    """Make and PowerShell call the script and do not bake in a machine path."""
    makefile: str = Path("Makefile").read_text(encoding="utf-8")
    recipe: str = makefile.split("train-h6:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    assert "python scripts/train_h6.py" in recipe
    script: str = Path("scripts/dev.ps1").read_text(encoding="utf-8")
    assert "python scripts/train_h6.py" in script
    assert '"train-h6" { Invoke-TrainH6 }' in script
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
    for feature_set in H6_CHALLENGER_FEATURE_SETS:
        reference.to_csv(
            tmp_path / f"oof_logistic_{feature_set}.csv",
            index=False,
            encoding="utf-8",
        )
    summary: dict[str, int] = require_written_cohort(tmp_path)
    assert summary == {"n": 2, "variants": 2}
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
