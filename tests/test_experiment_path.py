"""Databricks experiment paths stay absolute. SQLite names stay as written."""

from __future__ import annotations

import mlflow
import pytest

from endzone.modeling.tracking import (
    DEFAULT_DATABRICKS_EXPERIMENT_ROOT,
    activate_tracking_experiment,
    databricks_experiment_parent,
    resolve_experiment_name,
)

_RELATIVE: str = "nfl_pregame/registry"


def test_sqlite_and_other_stores_keep_the_relative_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Databricks root must not rename local or non-workspace experiments."""
    monkeypatch.setenv("ENDZONE_MLFLOW_EXPERIMENT_ROOT", "/Users/me/endzone")
    for tracking_uri in (
        "sqlite:///mlflow.db",
        "http://127.0.0.1:5000",
        "databricks-uc",
        "./mlruns",
    ):
        assert resolve_experiment_name(_RELATIVE, tracking_uri) == _RELATIVE


def test_databricks_prefixes_a_relative_name_and_keeps_an_absolute_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default root is /Shared/endzone-mlops. A leading slash is already absolute."""
    monkeypatch.delenv("ENDZONE_MLFLOW_EXPERIMENT_ROOT", raising=False)
    expected: str = f"{DEFAULT_DATABRICKS_EXPERIMENT_ROOT}/{_RELATIVE}"
    assert resolve_experiment_name(_RELATIVE, "databricks") == expected
    assert resolve_experiment_name(_RELATIVE, "databricks://lab") == expected
    absolute: str = "/Users/me/nfl_pregame"
    assert resolve_experiment_name(absolute, "databricks") == absolute
    assert resolve_experiment_name(f"  {_RELATIVE}  ", "databricks") == expected


def test_experiment_root_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENDZONE_MLFLOW_EXPERIMENT_ROOT replaces the default folder, including a trailing slash."""
    monkeypatch.setenv("ENDZONE_MLFLOW_EXPERIMENT_ROOT", "/Users/me/endzone/")
    assert resolve_experiment_name(_RELATIVE, "databricks") == f"/Users/me/endzone/{_RELATIVE}"
    assert resolve_experiment_name("/Shared/kept", "databricks") == "/Shared/kept"
    monkeypatch.setenv("ENDZONE_MLFLOW_EXPERIMENT_ROOT", "Shared/relative")
    with pytest.raises(ValueError, match="ENDZONE_MLFLOW_EXPERIMENT_ROOT"):
        resolve_experiment_name(_RELATIVE, "databricks")


def test_parent_folder_is_the_directory_above_the_experiment() -> None:
    """mkdirs must not occupy the experiment path itself."""
    assert (
        databricks_experiment_parent("/Shared/endzone-mlops/nfl_pregame/registry")
        == "/Shared/endzone-mlops/nfl_pregame"
    )
    assert databricks_experiment_parent("/registry") is None
    with pytest.raises(ValueError, match="absolute"):
        databricks_experiment_parent("nfl_pregame/registry")


def test_databricks_activation_uses_the_absolute_path_and_creates_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """set_experiment sees the workspace path. mkdirs sees only its parent."""
    seen: dict[str, object] = {}

    class FakeWorkspace:
        def mkdirs(self, path: str) -> None:
            seen["mkdirs"] = path

    class FakeClient:
        def __init__(self, profile: str | None = None) -> None:
            seen["profile"] = profile
            self.workspace = FakeWorkspace()

    import databricks.sdk

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", FakeClient)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: seen.__setitem__("uri", uri))
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: seen.__setitem__("experiment", name))
    monkeypatch.delenv("ENDZONE_MLFLOW_EXPERIMENT_ROOT", raising=False)

    activate_tracking_experiment(_RELATIVE, "databricks")
    assert seen["uri"] == "databricks"
    assert seen["profile"] is None
    assert seen["experiment"] == f"{DEFAULT_DATABRICKS_EXPERIMENT_ROOT}/{_RELATIVE}"
    assert seen["mkdirs"] == f"{DEFAULT_DATABRICKS_EXPERIMENT_ROOT}/nfl_pregame"

    activate_tracking_experiment("/Users/me/nfl_pregame", "databricks://lab")
    assert seen["profile"] == "lab"
    assert seen["experiment"] == "/Users/me/nfl_pregame"
    assert seen["mkdirs"] == "/Users/me"


def test_sqlite_activation_keeps_the_relative_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A local store does not call the workspace client."""
    import databricks.sdk

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("sqlite tracking must not create a workspace folder")

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", refuse)
    monkeypatch.setenv("ENDZONE_MLFLOW_EXPERIMENT_ROOT", "/Users/me/endzone")
    uri: str = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous: str = mlflow.get_tracking_uri()
    try:
        activate_tracking_experiment(_RELATIVE, uri)
        experiment = mlflow.MlflowClient(tracking_uri=uri).get_experiment_by_name(_RELATIVE)
    finally:
        mlflow.set_tracking_uri(previous)
    assert experiment is not None
    assert experiment.name == _RELATIVE
