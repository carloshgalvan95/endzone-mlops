"""Sprint 2 margin and total rule. Synthetic games only. No warehouse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endzone.features.pregame import (
    ELO_HOME_ADVANTAGE,
    ELO_POINTS_PER_SPREAD,
    elo_home_probabilities,
    feature_columns_for,
)
from endzone.modeling.points import attach_score_estimates, run_score_experiment, write_score_report
from endzone.modeling.points_eval import (
    HIDDEN_FAILED,
    HIDDEN_NOT_RUN,
    decide_mae_rule,
    paired_mae_difference,
    split_estimated_score,
)
from endzone.modeling.points_spec import load_score_spec
from tests.test_weekly_predictions import _reset_mlflow_tracking


def test_elo_spread_divides_home_field_by_25() -> None:
    """A first game has no history, so the spread is home field over 25."""
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "season": 2023,
                "gameday": "2023-09-10",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_won": 1,
                "home_score": 24,
                "away_score": 17,
            }
        ]
    )
    fitted = elo_home_probabilities(games)
    assert fitted["point_spreads"].iloc[0] == pytest.approx(
        ELO_HOME_ADVANTAGE / ELO_POINTS_PER_SPREAD
    )
    poisoned = games.copy()
    poisoned["home_score"] = 3
    poisoned["away_score"] = 40
    poisoned["home_won"] = 0
    again = elo_home_probabilities(poisoned)
    assert again["point_spreads"].iloc[0] == pytest.approx(fitted["point_spreads"].iloc[0])


def test_split_score_keeps_margin_and_total() -> None:
    """Home and away points rebuild the margin and the total."""
    parts = split_estimated_score(3.0, 45.0)
    assert parts["home"] == pytest.approx(24.0)
    assert parts["away"] == pytest.approx(21.0)
    assert parts["home"] - parts["away"] == pytest.approx(3.0)
    assert parts["home"] + parts["away"] == pytest.approx(45.0)


def test_acceptance_rule_matches_adr_007() -> None:
    """The interval has to sit below 0, and at least 5 of 8 seasons must improve."""
    assert (
        decide_mae_rule(
            -0.4,
            -0.6,
            -0.1,
            5,
            8,
            min_seasons_improved=5,
            season_count=8,
        )
        == "accepted"
    )
    assert (
        decide_mae_rule(
            -0.4,
            -0.6,
            0.01,
            8,
            8,
            min_seasons_improved=5,
            season_count=8,
        )
        == "rejected"
    )
    assert (
        decide_mae_rule(
            -0.4,
            -0.6,
            -0.1,
            4,
            8,
            min_seasons_improved=5,
            season_count=8,
        )
        == "rejected"
    )
    assert (
        decide_mae_rule(
            -0.4,
            -0.6,
            -0.1,
            5,
            7,
            min_seasons_improved=5,
            season_count=8,
        )
        == "inconclusive"
    )
    assert (
        decide_mae_rule(
            None,
            None,
            None,
            None,
            None,
            min_seasons_improved=5,
            season_count=8,
        )
        == "inconclusive"
    )


def test_paired_mae_interval_is_negative_when_the_model_is_closer() -> None:
    """Model error 1 versus baseline error 5. The difference interval stays below 0."""
    actual = np.array([0.0, 10.0, 20.0, 30.0])
    model = actual + 1.0
    baseline = actual + 5.0
    paired = paired_mae_difference(actual, model, baseline, n_bootstrap=200, seed=42)
    assert paired["mae"] == pytest.approx(1.0)
    assert paired["baseline_mae"] == pytest.approx(5.0)
    assert paired["mae_diff"] == pytest.approx(-4.0)
    assert paired["mae_diff_ci_high"] is not None
    assert float(paired["mae_diff_ci_high"]) < 0.0


def test_market_margin_uses_the_negated_home_line() -> None:
    """A home favorite of 3 is spread_line -3, so the market margin is +3."""
    spec = load_score_spec()
    assert spec.market_margin_multiplier == pytest.approx(-1.0)
    spread_line = np.array([-3.0, 7.0, np.nan])
    market = spread_line * spec.market_margin_multiplier
    assert market[0] == pytest.approx(3.0)
    assert market[1] == pytest.approx(-7.0)
    assert not np.isfinite(market[2])


def _signal_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Eight eval seasons where two h2_full columns equal the margin and the total."""
    columns: tuple[str, ...] = feature_columns_for("h2_full")
    generator = np.random.default_rng(7)
    games: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    teams: tuple[str, ...] = ("A", "B", "C", "D")
    for season in range(2015, 2024):
        for index in range(12):
            margin: float = float(generator.normal(0.0, 10.0))
            total: float = float(40.0 + generator.normal(0.0, 6.0))
            game_id: str = f"{season}-{index}"
            home: str = teams[index % 4]
            away: str = teams[(index + 1) % 4]
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": index + 1,
                    "gameday": pd.Timestamp("2015-09-01")
                    + pd.Timedelta(days=(season - 2015) * 40 + index),
                    "home_team": home,
                    "away_team": away,
                    "home_team_franchise": home,
                    "away_team_franchise": away,
                    "home_won": int(margin > 0.0),
                    "home_score": (total + margin) / 2.0,
                    "away_score": (total - margin) / 2.0,
                    "spread_line": -3.0,
                    "total_line": 43.0,
                    "home_rest": 7,
                    "away_rest": 7,
                }
            )
            row: dict[str, object] = {column: 0.0 for column in columns}
            row["game_id"] = game_id
            row["home_rest_days"] = margin
            row["away_rest_days"] = total
            feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    slate_features = features.iloc[[0]].copy()
    slate_features["game_id"] = "SLATE1"
    slate_features["home_rest_days"] = 6.0
    slate_features["away_rest_days"] = 44.0
    features = pd.concat([features, slate_features], ignore_index=True)
    slate = pd.DataFrame(
        [
            {
                "game_id": "SLATE1",
                "season": 2024,
                "week": 1,
                "gameday": pd.Timestamp("2024-09-08"),
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )
    return pd.DataFrame(games), features, slate


def test_signal_beats_the_baselines_and_registers(tmp_path: Path) -> None:
    """A column that copies the target passes the pre-registered rule."""
    games, features, slate = _signal_frame()
    spec = load_score_spec()
    tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    try:
        result = run_score_experiment(
            games,
            features,
            spec,
            slate=slate,
            tracking_uri=tracking_uri,
            register=True,
            output_dir=tmp_path / "reports",
        )
    finally:
        _reset_mlflow_tracking()
    margin = result["score_model"]["margin"]
    total = result["score_model"]["total"]
    assert margin["decision"] == "accepted"
    assert total["decision"] == "accepted"
    assert result["score_model"]["accepted"] is True
    assert result["score_model"]["hidden_reason"] is None
    assert margin["mae"] < margin["baseline_mae"]
    assert total["mae"] < total["baseline_mae"]
    assert margin["mae_diff_ci_high"] < 0.0
    assert total["mae_diff_ci_high"] < 0.0
    assert margin["seasons_improved"] >= 5
    assert total["seasons_improved"] >= 5
    assert margin["market_mae"] is not None
    assert total["market_mae"] is not None
    assert margin["rmse"] is not None
    assert margin["coverage_80"] is not None
    assert margin["implied_win_log_loss"] is not None
    assert margin["winprob_log_loss_benchmark"] == pytest.approx(0.638)
    assert total["implied_win_log_loss"] is None
    line = result["lines"]["SLATE1"]
    assert line["estimated_home_points"] == pytest.approx(
        (line["estimated_total"] + line["estimated_margin"]) / 2.0
    )
    assert line["estimated_away_points"] == pytest.approx(
        (line["estimated_total"] - line["estimated_margin"]) / 2.0
    )
    assert result["versions"]["margin"] is not None
    assert result["versions"]["total"] is not None
    report = json.loads((tmp_path / "reports" / "cv_report.json").read_text(encoding="utf-8"))
    assert report["margin"]["decision"] == "accepted"
    assert report["rule"]["market_is_a_gate"] is False
    assert len(report["margin"]["seasons"]) == 8
    written = write_score_report(report, tmp_path / "again")
    assert Path(written["markdown"]).read_text(encoding="utf-8").startswith("# h2_full")


def test_eval_targets_do_not_change_earlier_predictions(tmp_path: Path) -> None:
    """Changing 2023 scores leaves 2016-2022 model MAE where it was."""
    games, features, _slate = _signal_frame()
    spec = load_score_spec()
    tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    try:
        run_score_experiment(
            games,
            features,
            spec,
            tracking_uri=tracking_uri,
            register=False,
            output_dir=tmp_path / "first",
        )
        poisoned = games.copy()
        poisoned.loc[poisoned["season"] == 2023, "home_score"] = 70.0
        poisoned.loc[poisoned["season"] == 2023, "away_score"] = 0.0
        run_score_experiment(
            poisoned,
            features,
            spec,
            tracking_uri=tracking_uri,
            register=False,
            output_dir=tmp_path / "second",
        )
    finally:
        _reset_mlflow_tracking()
    before = json.loads((tmp_path / "first" / "cv_report.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "second" / "cv_report.json").read_text(encoding="utf-8"))
    early_before = [row for row in before["margin"]["seasons"] if row["season"] < 2023]
    early_after = [row for row in after["margin"]["seasons"] if row["season"] < 2023]
    assert early_before == early_after
    late_before = next(row for row in before["margin"]["seasons"] if row["season"] == 2023)
    late_after = next(row for row in after["margin"]["seasons"] if row["season"] == 2023)
    assert late_before["mae"] != late_after["mae"]


def test_missing_scores_hide_the_estimate() -> None:
    """No final score means the rule was not run. The page gets null points."""
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "season": 2024,
                "gameday": "2024-09-08",
                "home_team_franchise": "A",
                "away_team_franchise": "B",
                "home_won": 1,
            }
        ]
    )
    features = pd.DataFrame([{"game_id": "G1"}])
    slate = pd.DataFrame([{"game_id": "NEXT"}])
    attached = attach_score_estimates(games, features, slate, register=False)
    assert attached["score_model"]["accepted"] is False
    assert attached["score_model"]["hidden_reason"] == HIDDEN_NOT_RUN
    assert attached["lines"]["NEXT"]["estimated_home_points"] is None
    assert attached["lines"]["NEXT"]["estimated_away_points"] is None


def test_failed_rule_uses_the_hidden_sentence(tmp_path: Path) -> None:
    """A model with no signal loses to Elo and ties the league mean.

    Margin equals the Elo spread, so the baseline MAE is 0. Totals are
    constant, so ridge and the league mean match. Neither target passes.
    """
    games, features, slate = _signal_frame()
    spec = load_score_spec()
    elo_input = games.loc[
        :,
        [
            "game_id",
            "season",
            "gameday",
            "home_team_franchise",
            "away_team_franchise",
            "home_won",
            "home_score",
            "away_score",
        ],
    ]
    spreads = elo_home_probabilities(
        elo_input,
        points_per_spread=spec.elo_points_per_spread,
    )["point_spreads"]
    games = games.copy()
    games["home_score"] = (45.0 + spreads.to_numpy()) / 2.0
    games["away_score"] = (45.0 - spreads.to_numpy()) / 2.0
    games["home_won"] = (games["home_score"] > games["away_score"]).astype(int)
    noise = features.copy()
    for column in feature_columns_for("h2_full"):
        noise[column] = 0.0
    tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    try:
        result = run_score_experiment(
            games,
            noise,
            spec,
            slate=slate,
            tracking_uri=tracking_uri,
            register=True,
            output_dir=None,
        )
    finally:
        _reset_mlflow_tracking()
    assert result["score_model"]["accepted"] is False
    assert result["score_model"]["hidden_reason"] == HIDDEN_FAILED
    assert result["lines"]["SLATE1"]["estimated_home_points"] is None
    assert result["versions"]["margin"] is None
    assert result["versions"]["total"] is None
