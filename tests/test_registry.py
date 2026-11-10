"""Registry gate, sqlite alias round trip, and the register command. No warehouse."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from endzone.modeling.models import fit_logistic_pipeline
from endzone.modeling.registry import (
    TAG_CI_HIGH,
    TAG_CI_LOW,
    TAG_CUTOFF_SEASON,
    TAG_CUTOFF_WEEK,
    TAG_CV_LOG_LOSS,
    TAG_CV_N,
    TAG_EVALUATION,
    TAG_FEATURE_SET,
    TAG_LOG_LOSS_DIFF,
    TAG_MODEL,
    TAG_THROUGH_SEASON,
    TAG_V1_LOG_LOSS,
    WALK_FORWARD_EVALUATION,
    ProdAliasError,
    assert_can_register,
    format_scored_version,
    load_prod_model,
    metrics_from_comparison_csv,
    metrics_from_reference_check,
    promote_registered_version,
    promotion_decision,
    register_model_version,
)
from endzone.modeling.registry_command import main

_GOLDEN_LOSS: str = "0.6377"
_GOLDEN_N: str = "2191"


def _release_mlflow_sqlalchemy_engines() -> None:
    """Dispose cached SQLAlchemy engines that keep a SQLite file open."""
    import mlflow
    from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
    from mlflow.tracking._model_registry.utils import _get_store_registry
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    for store_cls in (TrackingStore, RegistryStore, SqlAlchemyJobStore):
        engine_map = getattr(store_cls, "_engine_map", None)
        if not isinstance(engine_map, dict):
            continue
        for engine in list(engine_map.values()):
            engine.dispose()
        engine_map.clear()
    tracking_cache = _tracking_store_registry._get_store_with_resolved_uri
    tracking_cache.cache_clear()
    registry_cache = _get_store_registry()._get_store_with_resolved_uri
    registry_cache.cache_clear()
    if mlflow.active_run() is not None:
        mlflow.end_run()
    from mlflow.tracking._model_registry import utils as registry_utils

    registry_utils._registry_uri = None
    os.environ.pop("MLFLOW_REGISTRY_URI", None)
    mlflow.set_tracking_uri("sqlite:///:memory:")


def _tiny_pipeline() -> tuple[object, pd.DataFrame]:
    """A one-column logistic fit. The registry stores it, the gate does not use it."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
    target = pd.Series([0, 1, 0, 1, 0, 1])
    pipeline = fit_logistic_pipeline(frame, target, max_iter=200)
    return pipeline, frame.head(1)


def _golden_tags(**overrides: str) -> dict[str, str]:
    """Tags that match the locked logistic h2_full row."""
    tags: dict[str, str] = {
        TAG_MODEL: "logistic",
        TAG_FEATURE_SET: "h2_full",
        TAG_CV_LOG_LOSS: _GOLDEN_LOSS,
        TAG_CV_N: _GOLDEN_N,
    }
    tags.update(overrides)
    return tags


def _sqlite_uri(path: Path) -> str:
    """A sqlite URI whose path uses forward slashes."""
    return "sqlite:///" + path.as_posix()


def test_promotion_gate_matches_the_golden_row_and_refuses_drift() -> None:
    """h2_full passes inside the golden tolerance and fails outside it."""
    matched = promotion_decision(_golden_tags())
    assert matched["allowed"] is True
    assert matched["reason"] == "matches_golden_reference"
    drifted = promotion_decision(_golden_tags(**{TAG_CV_LOG_LOSS: "0.6400"}))
    assert drifted["allowed"] is False
    assert drifted["reason"] == "outside_golden_tolerance"
    wrong_n = promotion_decision(_golden_tags(**{TAG_CV_N: "2000"}))
    assert wrong_n["allowed"] is False
    assert wrong_n["reason"] == "outside_golden_tolerance"
    missing = promotion_decision({TAG_MODEL: "logistic", TAG_FEATURE_SET: "h2_full"})
    assert missing["allowed"] is False
    assert missing["reason"] == "missing_cv_metrics"


def test_challenger_must_beat_the_golden_row_by_the_pre_registered_margin() -> None:
    """A non-reference row needs the ADR-005 accept and a 0.003 gap under golden."""
    accepted = promotion_decision(
        {
            TAG_MODEL: "logistic",
            TAG_FEATURE_SET: "h5_candidate",
            TAG_CV_LOG_LOSS: "0.6330",
            TAG_CV_N: "2191",
            TAG_LOG_LOSS_DIFF: "-0.0040",
            TAG_CI_LOW: "-0.0080",
            TAG_CI_HIGH: "-0.0010",
        }
    )
    assert accepted["allowed"] is True
    assert accepted["reason"] == "accepted_by_pre_registered_rule"
    short = promotion_decision(
        {
            TAG_MODEL: "logistic",
            TAG_FEATURE_SET: "h5_candidate",
            TAG_CV_LOG_LOSS: "0.6360",
            TAG_CV_N: "2191",
            TAG_LOG_LOSS_DIFF: "-0.0040",
            TAG_CI_LOW: "-0.0080",
            TAG_CI_HIGH: "-0.0010",
        }
    )
    assert short["allowed"] is False
    assert short["reason"] == "short_of_golden_margin"
    rejected = promotion_decision(
        {
            TAG_MODEL: "logistic",
            TAG_FEATURE_SET: "h5_candidate",
            TAG_CV_LOG_LOSS: "0.6200",
            TAG_CV_N: "2191",
            TAG_LOG_LOSS_DIFF: "-0.0100",
            TAG_CI_LOW: "-0.0200",
            TAG_CI_HIGH: "0.0010",
        }
    )
    assert rejected["allowed"] is False
    assert rejected["reason"] == "rejected"


def _walk_forward_tags(**overrides: str) -> dict[str, str]:
    """Tags for a fresh-window fit that beats the through-2023 log loss."""
    tags: dict[str, str] = {
        TAG_MODEL: "logistic",
        TAG_FEATURE_SET: "h2_full",
        TAG_EVALUATION: WALK_FORWARD_EVALUATION,
        TAG_CV_LOG_LOSS: "0.60",
        TAG_V1_LOG_LOSS: "0.66",
        TAG_CV_N: "100",
        TAG_LOG_LOSS_DIFF: "-0.06",
        TAG_CUTOFF_SEASON: "2026",
        TAG_CUTOFF_WEEK: "10",
        TAG_THROUGH_SEASON: "2026",
    }
    tags.update(overrides)
    return tags


def test_fresh_window_without_a_walk_forward_tag_is_refused() -> None:
    """A through-2026 fit cannot ride the golden 2016-2023 tags onto prod."""
    fresh = promotion_decision(_golden_tags(**{TAG_THROUGH_SEASON: "2026"}))
    assert fresh["allowed"] is False
    assert fresh["reason"] == "fresh_window_needs_walk_forward"
    still_golden = promotion_decision(_golden_tags(**{TAG_THROUGH_SEASON: "2023"}))
    assert still_golden["allowed"] is True
    assert still_golden["reason"] == "matches_golden_reference"


def test_walk_forward_promotion_requires_a_lower_log_loss_than_v1() -> None:
    """The fresh window is promoted only when its own backtest beats v1."""
    won = promotion_decision(_walk_forward_tags())
    assert won["allowed"] is True
    assert won["reason"] == "walk_forward_beats_v1"
    lost = promotion_decision(
        _walk_forward_tags(
            **{
                TAG_CV_LOG_LOSS: "0.70",
                TAG_V1_LOG_LOSS: "0.66",
                TAG_LOG_LOSS_DIFF: "0.04",
            }
        )
    )
    assert lost["allowed"] is False
    assert lost["reason"] == "walk_forward_does_not_beat_v1"
    inconsistent = promotion_decision(_walk_forward_tags(**{TAG_LOG_LOSS_DIFF: "0"}))
    assert inconsistent["allowed"] is False
    assert inconsistent["reason"] == "walk_forward_tags_inconsistent"
    other = promotion_decision(_walk_forward_tags(**{TAG_FEATURE_SET: "h5_qb"}))
    assert other["allowed"] is False
    assert other["reason"] == "walk_forward_not_h2_full"


def test_unity_catalog_register_refuses_sqlite_tracking() -> None:
    """UC cannot read a sqlite artifact, so that pair is refused before a write."""
    assert_can_register("sqlite:///mlflow.db", "sqlite:///mlflow.db")
    assert_can_register("databricks", "databricks-uc")
    with pytest.raises(ValueError, match="Unity Catalog"):
        assert_can_register("sqlite:///mlflow.db", "databricks-uc")


def test_scored_version_keeps_a_bare_number_when_there_is_no_alias() -> None:
    """The week file records version@alias only when an alias was loaded."""
    assert format_scored_version("3", "prod") == "3@prod"
    assert format_scored_version("3", None) == "3"
    assert format_scored_version("local", "") == "local"


def test_comparison_csv_requires_one_reference_row(tmp_path: Path) -> None:
    """The register command reads one logistic h2_full row and nothing else."""
    path = tmp_path / "comparison.csv"
    frame = pd.DataFrame(
        [
            {"model": "logistic", "feature_set": "h2_full", "log_loss": 0.6377, "n": 2191},
            {"model": "market", "feature_set": "h2_full", "log_loss": 0.6120, "n": 2191},
        ]
    )
    frame.to_csv(path, index=False)
    tags = metrics_from_comparison_csv(str(path))
    assert tags[TAG_CV_LOG_LOSS] == "0.6377"
    assert tags[TAG_CV_N] == "2191"
    frame = frame.iloc[1:]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one logistic h2_full"):
        metrics_from_comparison_csv(str(path))
    with pytest.raises(ValueError, match="did not match"):
        metrics_from_reference_check({"ok": False, "observed": []})


def test_sqlite_register_promote_and_load_round_trip(tmp_path: Path) -> None:
    """Register leaves the alias unset. Promote sets it and a second version can take it."""
    import mlflow

    uri = _sqlite_uri(tmp_path / "mlflow.db")
    pipeline, example = _tiny_pipeline()
    try:
        first = register_model_version(
            pipeline,
            example=example,
            model_name="endzone-winprob",
            tags=_golden_tags(),
            tracking_uri=uri,
            registry_uri=uri,
            run_name="register-one",
            through_season=2023,
        )
        with pytest.raises(ProdAliasError):
            load_prod_model("endzone-winprob", uri)
        promoted = promote_registered_version("endzone-winprob", first.model_version, uri)
        assert promoted["allowed"] is True
        assert promoted["alias"] == "prod"
        loaded = load_prod_model("endzone-winprob", uri)
        assert loaded.version == first.model_version
        assert loaded.alias == "prod"
        second = register_model_version(
            pipeline,
            example=example,
            model_name="endzone-winprob",
            tags=_golden_tags(),
            tracking_uri=uri,
            registry_uri=uri,
            run_name="register-two",
            through_season=2023,
        )
        moved = promote_registered_version("endzone-winprob", second.model_version, uri)
        assert moved["alias"] == "prod"
        loaded_again = load_prod_model("endzone-winprob", uri)
        assert loaded_again.version == second.model_version
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        versions = client.search_model_versions("name='endzone-winprob'")
        found: set[str] = {str(item.version) for item in versions}
        assert found == {first.model_version, second.model_version}
    finally:
        _release_mlflow_sqlalchemy_engines()


def test_promote_refusal_does_not_set_the_alias(tmp_path: Path) -> None:
    """A version that misses the gate keeps every alias unchanged."""
    uri = _sqlite_uri(tmp_path / "mlflow.db")
    pipeline, example = _tiny_pipeline()
    try:
        logged = register_model_version(
            pipeline,
            example=example,
            model_name="endzone-winprob",
            tags=_golden_tags(**{TAG_CV_LOG_LOSS: "0.7000"}),
            tracking_uri=uri,
            registry_uri=uri,
            run_name="register-bad",
        )
        decision = promote_registered_version("endzone-winprob", logged.model_version, uri)
        assert decision["allowed"] is False
        assert decision["alias"] is None
        with pytest.raises(ProdAliasError):
            load_prod_model("endzone-winprob", uri)
    finally:
        _release_mlflow_sqlalchemy_engines()


def test_register_command_stops_before_a_write_when_the_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed reference check does not fit and does not create a version."""
    uri = _sqlite_uri(tmp_path / "mlflow.db")

    def failed_check(**kwargs: object) -> dict[str, Any]:
        return {
            "ok": False,
            "tolerance": 0.0001,
            "failures": [
                {
                    "model": "logistic",
                    "feature_set": "h2_full",
                    "field": "log_loss",
                    "observed": 0.7,
                    "expected": 0.6377,
                    "abs_diff": 0.0623,
                }
            ],
            "observed": [],
        }

    def refuse_fit(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("a failed check must not fit")

    def refuse_load(*args: object, **kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise AssertionError("a failed check must not load data")

    monkeypatch.setattr("endzone.modeling.registry_command.run_reference_check", failed_check)
    monkeypatch.setattr("endzone.modeling.registry_command.fit_reference_pipeline", refuse_fit)
    monkeypatch.setattr("endzone.modeling.registry_command.load_data", refuse_load)
    with pytest.raises(SystemExit) as raised:
        main(["register", "--tracking-uri", uri, "--registry-uri", uri])
    assert raised.value.code == 1


def test_register_command_writes_a_version_and_does_not_set_the_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passing check registers one version and leaves prod unset."""
    import mlflow

    uri = _sqlite_uri(tmp_path / "mlflow.db")
    pipeline, example = _tiny_pipeline()

    def passing_check(**kwargs: object) -> dict[str, Any]:
        return {
            "ok": True,
            "tolerance": 0.0001,
            "failures": [],
            "observed": [
                {
                    "model": "logistic",
                    "feature_set": "h2_full",
                    "log_loss": 0.6377,
                    "n": 2191,
                }
            ],
        }

    def fake_fit(*args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "pipeline": pipeline,
            "example": example,
            "n_train": 4,
            "through_season": 2023,
        }

    monkeypatch.setattr("endzone.modeling.registry_command.run_reference_check", passing_check)
    monkeypatch.setattr("endzone.modeling.registry_command.fit_reference_pipeline", fake_fit)
    monkeypatch.setattr(
        "endzone.modeling.registry_command.load_data",
        lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()),
    )
    try:
        result = main(["register", "--tracking-uri", uri, "--registry-uri", uri])
        assert result["alias"] is None
        assert result["model_version"].isdigit()
        assert result["allowed"] is True
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        versions = client.search_model_versions("name='endzone-winprob'")
        assert len(versions) == 1
        with pytest.raises(ProdAliasError):
            load_prod_model("endzone-winprob", uri)
        promoted = main(["promote", "--version", result["model_version"], "--registry-uri", uri])
    finally:
        _release_mlflow_sqlalchemy_engines()
    assert promoted["alias"] == "prod"


def test_promote_command_rejects_the_alias_name() -> None:
    """prod is an alias. The version argument has to be a positive integer."""
    with pytest.raises(SystemExit):
        main(["promote", "--version", "prod", "--registry-uri", "sqlite:///unused.db"])
