"""Warehouse reference check. The fit itself stays behind the CV entry point."""

from typing import Any

import pandas as pd
import pytest

from endzone.features.pregame import FEATURE_SET_H2_FULL
from endzone.modeling.pipeline import run_reference_check
from endzone.modeling.reference import (
    compare_reference_metrics,
    format_reference_check,
    load_reference_metrics,
)
from endzone.modeling.train import main


def _report_from_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """Build the pooled rows ``compare_reference_metrics`` reads."""
    results: list[dict[str, Any]] = []
    for row in reference["models"]:
        pooled: dict[str, float] = {
            field: float(value)
            for field, value in row.items()
            if field not in {"model", "feature_set"}
        }
        results.append(
            {
                "model": row["model"],
                "feature_set": row["feature_set"],
                "pooled": pooled,
            }
        )
    return {"results": results}


def test_committed_reference_matches_the_h5_table() -> None:
    """The file stores the published logistic, Elo, market, and home-prior rows."""
    reference: dict[str, Any] = load_reference_metrics()
    indexed: dict[tuple[str, str], dict[str, Any]] = {
        (row["model"], row["feature_set"]): row for row in reference["models"]
    }
    logistic: dict[str, Any] = indexed[("logistic", "h2_full")]
    assert logistic["n"] == 2191
    assert logistic["log_loss"] == 0.6377
    assert logistic["brier"] == 0.2235
    assert logistic["auc"] == 0.6843
    assert logistic["accuracy"] == 0.6426
    assert indexed[("elo", "elo")]["n"] == 2191
    assert indexed[("elo", "elo")]["log_loss"] == 0.6529
    assert indexed[("market", "market")]["n"] == 2190
    assert indexed[("market", "market")]["log_loss"] == 0.6120
    assert indexed[("baseline-home", "none")]["n"] == 2191
    assert indexed[("baseline-home", "none")]["log_loss"] == 0.6888
    assert reference["cv_start_season"] == 2016
    assert reference["cv_end_season"] == 2023
    assert reference["n_bootstrap"] == 1000
    assert reference["bootstrap_seed"] == 42
    assert reference["tolerance"] == 1e-4


def test_compare_reference_metrics_passes_inside_tolerance() -> None:
    """A drift of 5e-5 stays inside the 1e-4 band."""
    reference: dict[str, Any] = load_reference_metrics()
    report: dict[str, Any] = _report_from_reference(reference)
    report["results"][0]["pooled"]["log_loss"] += 5e-5
    result: dict[str, Any] = compare_reference_metrics(report, reference)
    assert result["ok"]
    assert result["failures"] == []


def test_compare_reference_metrics_fails_on_log_loss_n_elo_or_market() -> None:
    """Log loss, n, Elo, and the market are each enough to fail the check."""
    reference: dict[str, Any] = load_reference_metrics()
    cases: tuple[tuple[str, str, str, float], ...] = (
        ("logistic", "h2_full", "log_loss", 0.0002),
        ("logistic", "h2_full", "n", 1.0),
        ("elo", "elo", "log_loss", 0.0002),
        ("market", "market", "log_loss", -0.0002),
        ("market", "market", "n", -1.0),
    )
    for model_name, feature_set, field, delta in cases:
        report: dict[str, Any] = _report_from_reference(reference)
        for row in report["results"]:
            if row["model"] == model_name and row["feature_set"] == feature_set:
                row["pooled"][field] += delta
        result: dict[str, Any] = compare_reference_metrics(report, reference)
        assert result["ok"] is False
        assert any(
            failure["model"] == model_name
            and failure["feature_set"] == feature_set
            and failure["field"] == field
            for failure in result["failures"]
        )
        message: str = format_reference_check(result)
        assert model_name in message
        assert field in message


def test_run_reference_check_reuses_the_cv_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check calls ``run_phase1_baselines`` and skips the warehouse when given frames."""
    reference: dict[str, Any] = load_reference_metrics()
    captured: dict[str, Any] = {}

    def fake_run(*args: object, **kwargs: object) -> dict[str, Any]:
        del args
        captured["kwargs"] = kwargs
        return _report_from_reference(reference)

    def fail_load(*args: object, **kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        del args, kwargs
        raise AssertionError("load_data should wait for omitted frames")

    monkeypatch.setattr("endzone.modeling.pipeline.run_phase1_baselines", fake_run)
    monkeypatch.setattr("endzone.modeling.pipeline.load_data", fail_load)
    result: dict[str, Any] = run_reference_check(
        games=pd.DataFrame({"game_id": [1]}),
        team_games=pd.DataFrame({"game_id": [1]}),
    )
    assert result["ok"]
    kwargs: dict[str, Any] = captured["kwargs"]
    assert kwargs["logistic_feature_sets"] == (FEATURE_SET_H2_FULL,)
    assert kwargs["reference_model"] == "logistic"
    assert kwargs["reference_feature_set"] == FEATURE_SET_H2_FULL
    assert kwargs["phase"] == 2
    assert kwargs["hypothesis"] == "h3"
    config: Any = kwargs["config"]
    assert config.cv_start_season == 2016
    assert config.cv_end_season == 2023
    assert config.n_bootstrap == 1000
    assert config.bootstrap_seed == 42


def test_run_reference_check_loads_the_warehouse_when_frames_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the frames reads the marts through ``load_data``."""
    reference: dict[str, Any] = load_reference_metrics()
    seen: dict[str, object] = {}

    def fake_load(fixture_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        seen["fixture_path"] = fixture_path
        return pd.DataFrame({"game_id": [1]}), pd.DataFrame({"game_id": [1]})

    monkeypatch.setattr("endzone.modeling.pipeline.load_data", fake_load)
    monkeypatch.setattr(
        "endzone.modeling.pipeline.run_phase1_baselines",
        lambda *args, **kwargs: _report_from_reference(reference),
    )
    result: dict[str, Any] = run_reference_check(fixture_path="fixtures/tiny")
    assert result["ok"]
    assert seen["fixture_path"] == "fixtures/tiny"


def test_verify_reference_cli_exits_when_the_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--verify-reference`` returns a non-zero status when a locked metric moves."""
    monkeypatch.setattr(
        "endzone.modeling.train.run_reference_check",
        lambda **kwargs: {
            "ok": False,
            "tolerance": 1e-4,
            "failures": [
                {
                    "model": "logistic",
                    "feature_set": "h2_full",
                    "field": "log_loss",
                    "observed": 0.64,
                    "expected": 0.6377,
                    "abs_diff": 0.0023,
                }
            ],
            "observed": [],
        },
    )
    with pytest.raises(SystemExit) as caught:
        main(["--verify-reference"])
    assert caught.value.code == 1


def test_verify_reference_cli_returns_when_the_check_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching refit does not raise."""
    monkeypatch.setattr(
        "endzone.modeling.train.run_reference_check",
        lambda **kwargs: {
            "ok": True,
            "tolerance": 1e-4,
            "failures": [],
            "observed": [
                {
                    "model": "logistic",
                    "feature_set": "h2_full",
                    "log_loss": 0.6377,
                    "n": 2191,
                }
            ],
        },
    )
    main(["--verify-reference"])
