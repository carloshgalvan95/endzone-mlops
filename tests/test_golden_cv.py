"""Golden snapshot of the rolling-origin CV report.

The games and team-game tables are built here with numpy Generator seed 42.
Regenerate the committed snapshot on purpose:

    UPDATE_GOLDEN=1 pytest tests/test_golden_cv.py
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.train.train import (
    cv_model_specs_for,
    logistic_feature_sets_for,
    make_rolling_origin_config,
    run_phase1_baselines,
)

GOLDEN_PATH: Path = Path(__file__).parent / "golden" / "cv_report.json"
GOLDEN_SEED: int = 42
GOLDEN_ABSOLUTE_TOLERANCE: float = 1e-9
_METRIC_DECIMALS: int = 12
_SELECTION_HYPOTHESES: tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h5b", "ridge")
_TEAMS: tuple[str, ...] = ("A", "B", "C", "D")
_PAIRINGS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
    ((1, 0), (3, 2)),
)


def synthetic_cv_frames(seed: int = GOLDEN_SEED) -> dict[str, pd.DataFrame]:
    """Build a deterministic games table and team-game EPA table.

    Seasons 2014 through 2025 are present. Each season has eight games and
    both classes. One 2020 game has no moneyline.

    Args:
        seed: Generator seed for the EPA noise.

    Returns:
        Dict with ``games`` and ``team_games``.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    game_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    for season in range(2014, 2026):
        for week, pairs in enumerate(_PAIRINGS):
            gameday: pd.Timestamp = pd.Timestamp(year=season, month=9, day=1) + pd.Timedelta(
                days=7 * week
            )
            for pair_index, (home_index, away_index) in enumerate(pairs):
                home_won: int = int((season + week + pair_index) % 2 == 0)
                game_id: str = f"{season}W{week}G{pair_index}"
                home_team: str = _TEAMS[home_index]
                away_team: str = _TEAMS[away_index]
                if home_won == 1:
                    home_line: float = -5000.0
                    away_line: float = 4000.0
                    home_score: int = 27
                    away_score: int = 17
                else:
                    home_line = 4000.0
                    away_line = -5000.0
                    home_score = 14
                    away_score = 24
                if season == 2020 and week == 0 and pair_index == 0:
                    home_line = np.nan
                    away_line = np.nan
                game_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "gameday": gameday,
                        "home_team_franchise": home_team,
                        "away_team_franchise": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_won": home_won,
                        "home_rest": 7 + (week % 3),
                        "away_rest": 6 + pair_index,
                        "home_moneyline": home_line,
                        "away_moneyline": away_line,
                    }
                )
                for team, opponent, sign in (
                    (home_team, away_team, 1.0 if home_won == 1 else -1.0),
                    (away_team, home_team, -1.0 if home_won == 1 else 1.0),
                ):
                    offense: float = 0.12 * sign + float(rng.normal(0.0, 0.05))
                    team_rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "gameday": gameday,
                            "team_franchise": team,
                            "opponent_franchise": opponent,
                            "avg_offensive_epa": offense,
                            "avg_defensive_epa_allowed": -offense + float(rng.normal(0.0, 0.02)),
                            "pass_plays": 20,
                            "rush_plays": 12,
                            "pass_epa": offense * 20.0,
                            "rush_epa": offense * 4.0,
                        }
                    )
    return {
        "games": pd.DataFrame(game_rows),
        "team_games": pd.DataFrame(team_rows),
    }


def _spec_rows(phase: int, hypothesis: str | None = None) -> list[dict[str, str]]:
    """Return the public model specs for one phase as plain dicts."""
    return [
        {"model": spec.model, "feature_set": spec.feature_set}
        for spec in cv_model_specs_for(phase, hypothesis)
    ]


def _feature_set_selection() -> dict[str, Any]:
    """Snapshot feature-set selection for every current phase and hypothesis."""
    selection: dict[str, Any] = {
        "phase1": {
            "feature_sets": list(logistic_feature_sets_for(1)),
            "model_specs": _spec_rows(1),
        }
    }
    for hypothesis in _SELECTION_HYPOTHESES:
        selection[hypothesis] = {
            "feature_sets": list(logistic_feature_sets_for(2, hypothesis)),
            "model_specs": _spec_rows(2, hypothesis),
        }
    return selection


def _snapshot_folds(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the split fields a refactor of the fold loop would move."""
    return [
        {
            "season": int(fold["season"]),
            "status": fold["status"],
            "n_train": int(fold["n_train"]),
            "n_eval": int(fold["n_eval"]),
            "train_seasons": [int(season) for season in fold["train_seasons"]],
        }
        for fold in folds
    ]


def snapshot_cv_report(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a CV report to metrics, fold sizes, and decision labels.

    Args:
        report: Result of ``run_phase1_baselines``.

    Returns:
        A JSON-ready dict. Floats are still raw and are rounded on compare.
    """
    decisions: dict[tuple[str, str], str] = {
        (row["model"], row["feature_set"]): str(row["decision"]) for row in report["acceptance"]
    }
    rows: list[dict[str, Any]] = []
    for result in report["results"]:
        identity: tuple[str, str] = (result["model"], result["feature_set"])
        pooled: dict[str, Any] = result["pooled"] or {}
        comparison: dict[str, Any] = result["comparison"] or {}
        n_value: object = pooled.get("n")
        rows.append(
            {
                "model": result["model"],
                "feature_set": result["feature_set"],
                "decision": decisions[identity],
                "n": None if n_value is None else int(n_value),
                "log_loss": pooled.get("log_loss"),
                "brier": pooled.get("brier"),
                "auc": pooled.get("auc"),
                "accuracy": pooled.get("accuracy"),
                "log_loss_diff": comparison.get("log_loss_diff"),
                "log_loss_diff_ci_low": comparison.get("log_loss_diff_ci_low"),
                "log_loss_diff_ci_high": comparison.get("log_loss_diff_ci_high"),
                "folds": _snapshot_folds(result["folds"]),
            }
        )
    return {
        "dataset_seed": GOLDEN_SEED,
        "phase": report["phase"],
        "hypothesis": report["hypothesis"],
        "reference_model": report["reference_model"],
        "reference_feature_set": report["reference_feature_set"],
        "config": {
            "cv_start_season": report["config"]["cv_start_season"],
            "cv_end_season": report["config"]["cv_end_season"],
            "n_bootstrap": report["config"]["n_bootstrap"],
            "bootstrap_seed": report["config"]["bootstrap_seed"],
        },
        "feature_set_selection": _feature_set_selection(),
        "rows": rows,
    }


def _canonicalize(value: object) -> object:
    """Round floats so the snapshot file is stable past 1e-9."""
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number: float = float(value)
        if not math.isfinite(number):
            return None
        return round(number, _METRIC_DECIMALS)
    return value


def assert_golden_match(actual: object, expected: object, path: str = "report") -> None:
    """Fail when a snapshot drifts. Floats must agree within 1e-9.

    Args:
        actual: Fresh snapshot.
        expected: Committed snapshot.
        path: Location used in the assertion message.

    Raises:
        AssertionError: A label, a fold size, or a metric moved.
    """
    _assert_match(_canonicalize(actual), _canonicalize(expected), path)


def _assert_match(actual: object, expected: object, path: str) -> None:
    """Walk two canonical snapshots and raise on the first difference."""
    if isinstance(expected, float):
        if not isinstance(actual, float):
            raise AssertionError(f"{path}: expected a float, got {actual!r}")
        difference: float = abs(actual - expected)
        if difference > GOLDEN_ABSOLUTE_TOLERANCE:
            raise AssertionError(
                f"{path}: {actual:.12f} differs from {expected:.12f} by {difference:.3e}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(
                f"{path}: length {len(actual) if isinstance(actual, list) else actual!r}"
            )
        for index, expected_item in enumerate(expected):
            _assert_match(actual[index], expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected an object, got {actual!r}")
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys {sorted(actual)} != {sorted(expected)}")
        for key in expected:
            _assert_match(actual[key], expected[key], f"{path}.{key}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _row(snapshot: dict[str, Any], model: str, feature_set: str) -> dict[str, Any]:
    """Return one snapshot row."""
    for row in snapshot["rows"]:
        if row["model"] == model and row["feature_set"] == feature_set:
            return row
    raise AssertionError(f"missing row {model}/{feature_set}")


def _assert_split_contract(snapshot: dict[str, Any], games: pd.DataFrame) -> None:
    """Lock the window, the bootstrap, and the fold boundaries of this fixture."""
    config: dict[str, int] = snapshot["config"]
    assert config == {
        "cv_start_season": 2016,
        "cv_end_season": 2023,
        "n_bootstrap": 1000,
        "bootstrap_seed": 42,
    }
    assert set(games["season"].unique()) >= {2024, 2025}
    baseline: dict[str, Any] = _row(snapshot, "baseline-home", "none")
    folds: list[dict[str, Any]] = baseline["folds"]
    assert [fold["season"] for fold in folds] == list(range(2016, 2024))
    assert folds[0] == {
        "season": 2016,
        "status": "scored",
        "n_train": 16,
        "n_eval": 8,
        "train_seasons": [2014, 2015],
    }
    assert folds[-1]["season"] == 2023
    assert folds[-1]["n_train"] == 72
    assert folds[-1]["n_eval"] == 8
    assert folds[-1]["train_seasons"] == list(range(2014, 2023))
    market: dict[str, Any] = _row(snapshot, "market", "market")
    market_2020: dict[str, Any] = next(fold for fold in market["folds"] if fold["season"] == 2020)
    assert market_2020["n_eval"] == 7
    logistic_sets: list[str] = [
        row["feature_set"] for row in snapshot["rows"] if row["model"] == "logistic"
    ]
    assert logistic_sets == snapshot["feature_set_selection"]["h3"]["feature_sets"]
    assert snapshot["reference_model"] == "logistic"
    assert snapshot["reference_feature_set"] == "h2_full"
    decisions: set[str] = {row["decision"] for row in snapshot["rows"]}
    assert "reference" in decisions
    assert "accepted" in decisions
    assert "rejected" in decisions


def test_golden_cv_report() -> None:
    """The public CV path matches the committed snapshot."""
    frames: dict[str, pd.DataFrame] = synthetic_cv_frames()
    config = make_rolling_origin_config()
    assert config.n_bootstrap == 1000
    assert config.bootstrap_seed == 42
    with threadpool_limits(limits=1):
        report: dict[str, Any] = run_phase1_baselines(
            frames["games"],
            frames["team_games"],
            config=config,
            phase=2,
            hypothesis="h3",
        )
    snapshot: dict[str, Any] = snapshot_cv_report(report)
    _assert_split_contract(snapshot, frames["games"])
    payload: object = _canonicalize(snapshot)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not GOLDEN_PATH.is_file():
        raise AssertionError(
            f"Missing {GOLDEN_PATH.name}. Regenerate with "
            "UPDATE_GOLDEN=1 pytest tests/test_golden_cv.py"
        )
    expected: object = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert_golden_match(payload, expected)


def test_golden_comparison_fails_when_a_metric_moves() -> None:
    """A 1e-6 change in log loss is outside the 1e-9 band."""
    base: dict[str, object] = {"log_loss": 0.5, "n": 8, "decision": "rejected"}
    moved: dict[str, object] = {"log_loss": 0.5 + 1e-6, "n": 8, "decision": "rejected"}
    with pytest.raises(AssertionError, match="log_loss"):
        assert_golden_match(moved, base)


def test_golden_comparison_fails_on_fold_size_or_decision() -> None:
    """Fold sizes and decision labels are exact."""
    base: dict[str, object] = {
        "decision": "accepted",
        "folds": [{"season": 2016, "n_train": 16, "n_eval": 8}],
    }
    resized: dict[str, object] = {
        "decision": "accepted",
        "folds": [{"season": 2016, "n_train": 16, "n_eval": 9}],
    }
    relabeled: dict[str, object] = {
        "decision": "rejected",
        "folds": [{"season": 2016, "n_train": 16, "n_eval": 8}],
    }
    with pytest.raises(AssertionError, match="n_eval"):
        assert_golden_match(resized, base)
    with pytest.raises(AssertionError, match="decision"):
        assert_golden_match(relabeled, base)
