"""MLflow experiments, tags, dataset lineage, and the reviewed skops artifact."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.utils.uri import is_databricks_uri

from endzone.config import resolve_setting, resolve_tracking_uri
from endzone.modeling.evaluation import (
    MIN_LOG_LOSS_IMPROVEMENT,
    acceptance_decisions,
    decide_log_loss_improvement,
)
from endzone.modeling.hypotheses import load_catalog
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

# Fluent SQLite tracking stores artifacts under ./mlruns unless an experiment
# sets another location. Keep them in a gitignored directory.
LOCAL_ARTIFACT_DIR: str = "mlartifacts"
# Types skops refuses by default that these pipelines are known to contain.
# Anything else fails closed. Do not replace this with skops' full untrusted list.
REVIEWED_SKOPS_TRUSTED_TYPES: frozenset[str] = frozenset(
    {
        "numpy.dtype",
        "sklearn.tree._tree.Tree",
    }
)
CV_SCHEME: str = "rolling_season_2016_2023"
CV_DATASET_NAME: str = "nfl_pregame_cv"
# Databricks rejects a relative experiment name. Relative names are stored
# under this workspace folder unless ENDZONE_MLFLOW_EXPERIMENT_ROOT is set.
DEFAULT_DATABRICKS_EXPERIMENT_ROOT: str = "/Shared/endzone-mlops"
_ENV_EXPERIMENT_ROOT: str = "ENDZONE_MLFLOW_EXPERIMENT_ROOT"
BASELINE_MODEL_NAMES: frozenset[str] = frozenset({"baseline-home", "elo", "market"})
SCORED_FEATURE_MODELS: frozenset[str] = frozenset({"logistic", "logistic_l2"})


def get_git_sha() -> str:
    """Return the current git commit SHA, or ``unknown`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def resolve_skops_trusted_types(model: object) -> dict[str, list[str]]:
    """Return the reviewed skops types required to reload this model.

    An untrusted type outside the allowlist raises ``ValueError``.

    Returns:
        Dict with key ``trusted_types``, sorted and safe to pass to
        ``mlflow.sklearn.log_model``.
    """
    import skops.io

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path: Path = Path(tmp_dir) / "model.skops"
        skops.io.dump(model, model_path)
        untrusted: set[str] = set(skops.io.get_untrusted_types(file=model_path))

    unknown: set[str] = untrusted - REVIEWED_SKOPS_TRUSTED_TYPES
    if unknown:
        unknown_list: str = ", ".join(sorted(unknown))
        allowlist: str = ", ".join(sorted(REVIEWED_SKOPS_TRUSTED_TYPES))
        raise ValueError(
            "Refusing to trust skops types that are not on the reviewed "
            f"allowlist: {unknown_list}. Allowlist: {allowlist}."
        )
    return {"trusted_types": sorted(untrusted)}


def log_sklearn_model(model: object) -> dict[str, list[str]]:
    """Log a fitted sklearn model on the active MLflow run.

    The artifact name is ``model``. Trusted skops types are limited to the
    reviewed types this specific model needs.
    """
    trusted: dict[str, list[str]] = resolve_skops_trusted_types(model)
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=trusted["trusted_types"],
    )
    return trusted


class RegisteredModelLog(NamedTuple):
    """A logged sklearn model and the registry version created for it."""

    trusted_types: list[str]
    model_version: str
    model_uri: str
    run_id: str


def log_registered_sklearn_model(
    model: object,
    *,
    signature: Any,
    input_example: pd.DataFrame,
    registered_model_name: str,
) -> RegisteredModelLog:
    """Log a sklearn model with a signature and register that version.

    The artifact name stays ``model``. Registration uses the active tracking
    store. A missing registry version raises ``RuntimeError``.

    Args:
        model: Fitted estimator.
        signature: MLflow model signature for the feature columns.
        input_example: One or more rows of those columns.
        registered_model_name: Registry name, such as ``endzone-winprob``.

    Returns:
        Trusted skops types, the registry version, the model URI, and the run id.
    """
    trusted: dict[str, list[str]] = resolve_skops_trusted_types(model)
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        skops_trusted_types=trusted["trusted_types"],
    )
    version: object = info.registered_model_version
    if version is None or str(version) == "":
        raise RuntimeError(f"MLflow logged the model but did not register {registered_model_name}.")
    return RegisteredModelLog(
        trusted_types=list(trusted["trusted_types"]),
        model_version=str(version),
        model_uri=str(info.model_uri),
        run_id=str(info.run_id),
    )


def resolve_experiment_name(
    experiment_name: str,
    tracking_uri: str,
    *,
    root: str | None = None,
) -> str:
    """Return the experiment name for this tracking store.

    SQLite, file, and HTTP stores keep the name, including when
    ``ENDZONE_MLFLOW_EXPERIMENT_ROOT`` is set. A Databricks tracking URI
    (``databricks`` or ``databricks://...``) prefixes a relative name with
    the workspace root. A name that already starts with ``/`` is unchanged.
    ``databricks-uc`` is a registry URI and is not rewritten.

    Args:
        experiment_name: Logical name such as ``nfl_pregame/registry``.
        tracking_uri: Tracking URI passed to ``mlflow.set_tracking_uri``.
        root: Workspace folder. ``None`` reads ``ENDZONE_MLFLOW_EXPERIMENT_ROOT``,
            then ``/Shared/endzone-mlops``.

    Returns:
        The name ``set_experiment`` should receive.

    Raises:
        ValueError: The name is blank, or a Databricks root is not an
            absolute workspace path.
    """
    name: str = experiment_name.strip()
    if name == "":
        raise ValueError("Experiment name must not be blank.")
    if not is_databricks_uri(tracking_uri):
        return name
    if name.startswith("/"):
        return name
    return f"{_databricks_experiment_root(root)}/{name}"


def databricks_experiment_parent(experiment_path: str) -> str | None:
    """Return the workspace folder that must exist before ``set_experiment``.

    Databricks stores an experiment as a workspace object. Creating it does
    not create the parent folder. The experiment path itself is not a folder.

    Args:
        experiment_path: Absolute workspace path of the experiment.

    Returns:
        The parent folder, or ``None`` when the experiment sits at the
        workspace root.

    Raises:
        ValueError: The path is not absolute.
    """
    if not experiment_path.startswith("/"):
        raise ValueError("Databricks experiment path must be absolute.")
    parent: str = experiment_path.rsplit("/", 1)[0]
    if parent == "":
        return None
    return parent


def ensure_databricks_experiment_parent(experiment_path: str, tracking_uri: str) -> None:
    """Create the parent workspace folder if it is missing.

    ``WorkspaceClient.workspace.mkdirs`` creates missing parents and does
    nothing when the folder already exists. A ``databricks://profile`` URI
    uses that profile. ``databricks`` and a secret-scope URI use the default
    client, which reads the same ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN``
    MLflow uses.

    Raises:
        ValueError: The experiment path is not absolute.
    """
    parent: str | None = databricks_experiment_parent(experiment_path)
    if parent is None:
        return
    from databricks.sdk import WorkspaceClient

    profile: str | None = _databricks_profile(tracking_uri)
    client = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    logger.info("Ensuring Databricks workspace folder %s", parent)
    client.workspace.mkdirs(parent)


def _databricks_experiment_root(explicit: str | None) -> str:
    """Return a clean absolute workspace folder.

    Raises:
        ValueError: The root is blank, relative, or contains an empty or
            ``..`` segment.
    """
    raw: str = resolve_setting(explicit, _ENV_EXPERIMENT_ROOT, DEFAULT_DATABRICKS_EXPERIMENT_ROOT)
    root: str = raw.rstrip("/")
    parts: list[str] = root.split("/")
    if root == "" or not root.startswith("/") or any(part in ("", "..") for part in parts[1:]):
        raise ValueError(
            "ENDZONE_MLFLOW_EXPERIMENT_ROOT must be an absolute Databricks "
            "workspace path such as /Shared/endzone-mlops."
        )
    return root


def _databricks_profile(tracking_uri: str) -> str | None:
    """Return a profile name from ``databricks://profile``, or ``None``."""
    if not tracking_uri.startswith("databricks://"):
        return None
    profile: str = tracking_uri.removeprefix("databricks://").split("/", 1)[0]
    if profile == "" or ":" in profile:
        return None
    return profile


def activate_tracking_experiment(experiment_name: str, tracking_uri: str) -> None:
    """Point MLflow at a tracking URI and activate an experiment.

    Scheme URIs such as ``sqlite:///mlflow.db`` keep the given experiment
    name. A Databricks tracking URI turns a relative name into an absolute
    workspace path and creates the parent folder when it is missing.
    ``databricks`` targets keep the workspace artifact root. A new SQLite
    experiment stores artifacts under ``mlartifacts/``.
    """
    mlflow.set_tracking_uri(tracking_uri)
    resolved_name: str = resolve_experiment_name(experiment_name, tracking_uri)
    if is_databricks_uri(tracking_uri):
        logger.info("Using Databricks MLflow experiment %s", resolved_name)
        ensure_databricks_experiment_parent(resolved_name, tracking_uri)
        mlflow.set_experiment(resolved_name)
        return
    is_databricks_target: bool = tracking_uri == "databricks" or tracking_uri.startswith(
        "databricks"
    )
    if is_databricks_target:
        logger.info("Using Databricks MLflow tracking")
        mlflow.set_experiment(resolved_name)
        return

    logger.info(f"Using local MLflow tracking: {tracking_uri}")
    if not tracking_uri.startswith("sqlite:"):
        mlflow.set_experiment(resolved_name)
        return

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(resolved_name)
    if experiment is not None:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        return

    artifact_location: str = Path(LOCAL_ARTIFACT_DIR).resolve().as_uri()
    experiment_id: str = client.create_experiment(
        resolved_name,
        artifact_location=artifact_location,
    )
    mlflow.set_experiment(experiment_id=experiment_id)


def parent_run_name(
    phase: int,
    hypothesis: str | None,
    timestamp: datetime | None = None,
) -> str:
    """Name the parent run for one CV invocation.

    Returns:
        A name such as ``phase1-<timestamp>`` or ``phase2-h3-<timestamp>``.
    """
    clock: datetime = timestamp if timestamp is not None else datetime.now(UTC)
    stamp: str = clock.strftime("%Y%m%dT%H%M%SZ")
    if phase != 2 or hypothesis is None:
        return f"phase1-{stamp}"
    spec = load_catalog().by_tag(hypothesis)
    if spec is None:
        return f"phase2-{stamp}"
    return f"phase2-{spec.short_name}-{stamp}"


def mlflow_decision_tag(
    result: dict[str, Any],
    reference_model: str,
    reference_feature_set: str,
) -> str:
    """Tag one scored model with the acceptance rule.

    The reference row is ``reference``. Home prior, Elo, and the market are
    ``baseline`` unless that model is the reference. A candidate is
    ``accepted`` only when pooled log loss improves by at least 0.003 and
    the 95% interval lies entirely below 0.
    """
    if result["model"] == reference_model and result["feature_set"] == reference_feature_set:
        return "reference"
    if result["model"] in BASELINE_MODEL_NAMES:
        return "baseline"
    comparison: dict[str, Any] = result.get("comparison") or {}
    labeled: dict[str, Any] = decide_log_loss_improvement(
        comparison.get("log_loss_diff"),
        comparison.get("log_loss_diff_ci_low"),
        comparison.get("log_loss_diff_ci_high"),
        min_improvement=MIN_LOG_LOSS_IMPROVEMENT,
    )
    return str(labeled["decision"])


def _child_decision_tag(
    result: dict[str, Any],
    reference_model: str,
    reference_feature_set: str,
    acceptance_rows: list[dict[str, Any]] | None,
) -> str:
    """Return the child decision tag.

    A row already labeled ``benchmark`` keeps that label. Every other row
    uses ``mlflow_decision_tag``.
    """
    for row in acceptance_rows or []:
        if row.get("model") == result["model"] and row.get("feature_set") == result["feature_set"]:
            if row.get("decision") == "benchmark":
                return "benchmark"
    return mlflow_decision_tag(result, reference_model, reference_feature_set)


def select_best_feature_set(
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
) -> str:
    """Choose the feature set recorded on the parent run.

    An accepted logistic set wins. If more than one is accepted, the lowest
    pooled log loss wins. If none is accepted, the reference feature set is kept.
    """
    decisions: list[dict[str, Any]] = acceptance_decisions(
        results,
        reference_model,
        reference_feature_set,
    )
    scored_losses: list[tuple[str, str, float]] = []
    for result in results:
        if result["model"] not in SCORED_FEATURE_MODELS or result["status"] != "ok":
            continue
        pooled: dict[str, Any] = result.get("pooled") or {}
        value: object = pooled.get("log_loss")
        if value is None or pd.isna(value):
            continue
        scored_losses.append((str(result["model"]), str(result["feature_set"]), float(value)))
    accepted_keys: set[tuple[str, str]] = {
        (str(row["model"]), str(row["feature_set"]))
        for row in decisions
        if row["decision"] == "accepted"
    }
    accepted: list[tuple[str, str, float]] = [
        item for item in scored_losses if (item[0], item[1]) in accepted_keys
    ]
    if not accepted:
        return reference_feature_set
    return min(accepted, key=lambda item: item[2])[1]


def _description_dataset_source(description: str) -> Any:
    """Dataset source that stores a short description of the CV matrix.

    A plain sentence has no URI scheme, so MLflow would otherwise warn and
    treat it as a local file path. Passing the local source instance skips
    that resolver.
    """
    from mlflow.data.dataset_source_registry import get_registered_sources

    local_sources: list[Any] = [
        source_cls
        for source_cls in get_registered_sources()
        if source_cls._get_source_type() == "local"
    ]
    if not local_sources:
        raise ValueError("MLflow did not register a local dataset source")
    return local_sources[-1](description)


def _cv_pandas_dataset(frame: pd.DataFrame, source: str) -> Any:
    """Build one pandas dataset for the CV matrix, named ``nfl_pregame_cv``."""
    return mlflow.data.from_pandas(
        frame,
        source=_description_dataset_source(source),
        name=CV_DATASET_NAME,
        targets="home_won",
    )


def _log_dataset_lineage(dataset: Any, n_rows: int, n_columns: int) -> str:
    """Log one CV dataset input and return its digest."""
    digest: str = str(dataset.digest)
    mlflow.set_tag("data_digest", digest)
    mlflow.log_input(dataset, context="training")
    mlflow.log_params(
        {
            "data_n_rows": str(n_rows),
            "data_n_columns": str(n_columns),
        }
    )
    return digest


def _log_paired_difference_metrics(comparison: dict[str, Any] | None) -> None:
    """Log the paired log loss gap under the decision metric names."""
    if comparison is None:
        return
    for source_name in ("log_loss_diff", "log_loss_diff_ci_low", "log_loss_diff_ci_high"):
        metric_value: object = comparison.get(source_name)
        if metric_value is None or pd.isna(metric_value):
            continue
        mlflow.log_metric(source_name, float(metric_value))
        mlflow.log_metric(f"cv_{source_name}", float(metric_value))


def _log_child_run(
    result: dict[str, Any],
    paths: dict[str, Any],
    phase_tag: str,
    hypothesis: str | None,
    run_note: str,
    git_commit: str,
    reference_model: str,
    reference_feature_set: str,
    dataset: Any | None,
    dataset_frame: pd.DataFrame | None,
    acceptance_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Log one nested child and return its model, feature set, and run id."""
    run_name: str = f"{result['model']}-{result['feature_set']}"
    with mlflow.start_run(run_name=run_name, nested=True) as active_run:
        mlflow.set_tag("phase", phase_tag)
        if hypothesis is not None:
            mlflow.set_tag("hypothesis", hypothesis)
        mlflow.set_tag("feature_set", result["feature_set"])
        mlflow.set_tag("cv_scheme", CV_SCHEME)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag(
            "decision",
            _child_decision_tag(
                result,
                reference_model,
                reference_feature_set,
                acceptance_rows,
            ),
        )
        mlflow.set_tag("mlflow.note.content", run_note)
        if dataset is not None and dataset_frame is not None:
            _log_dataset_lineage(
                dataset,
                n_rows=len(dataset_frame),
                n_columns=int(dataset_frame.shape[1]),
            )
        params: dict[str, Any] = {
            key: value
            for key, value in result["params"].items()
            if value is not None and not isinstance(value, dict | list)
        }
        params["model_type"] = result["model"]
        params["feature_set"] = result["feature_set"]
        mlflow.log_params(params)
        pooled: dict[str, Any] = result["pooled"] or {}
        for metric_name in ("log_loss", "brier", "auc", "accuracy"):
            metric_value = pooled.get(metric_name)
            if metric_value is None or pd.isna(metric_value):
                continue
            mlflow.log_metric(f"cv_{metric_name}", float(metric_value))
        if pooled.get("n") is not None:
            mlflow.log_metric("cv_n", float(pooled["n"]))
        _log_paired_difference_metrics(result.get("comparison"))
        scored_folds: int = sum(1 for fold in result["folds"] if fold["status"] == "scored")
        skipped_folds: int = sum(1 for fold in result["folds"] if fold["status"] != "scored")
        mlflow.log_metric("cv_folds_scored", float(scored_folds))
        mlflow.log_metric("cv_folds_skipped", float(skipped_folds))
        oof_key: str = f"{result['model']}|{result['feature_set']}"
        oof_path: str | None = paths["oof_paths"].get(oof_key)
        if oof_path is not None:
            mlflow.log_artifact(oof_path)
        return {
            "model": result["model"],
            "feature_set": result["feature_set"],
            "run_id": active_run.info.run_id,
        }


def log_phase1_to_mlflow(
    report: dict[str, Any],
    paths: dict[str, Any],
    tracking_uri: str,
    experiment_name: str | None = None,
    phase_tag: str | None = None,
    hypothesis: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Log one parent run and one nested child per scored model.

    The parent carries the comparison report, the dataset digest, and
    ``best_feature_set``. Each child is the same invocation with
    ``nested=True``. Omitted tag arguments use the phase 1 experiment file.

    Returns:
        Dict with ``run_ids``, ``parent_run_id``, ``parent_run_name``,
        ``data_digest``, ``best_feature_set``, ``experiment_name``, and
        ``tracking_uri``.
    """
    phase1 = load_catalog().phase1()
    resolved_experiment: str = (
        phase1.experiment_name if experiment_name is None else experiment_name
    )
    resolved_phase_tag: str = phase1.phase_tag if phase_tag is None else phase_tag
    resolved_uri: str = resolve_tracking_uri(tracking_uri)
    activate_tracking_experiment(resolved_experiment, resolved_uri)
    reference_model: str = str(report["reference_model"])
    reference_feature_set: str = str(report["reference_feature_set"])
    best_feature_set: str = select_best_feature_set(
        report["results"],
        reference_model,
        reference_feature_set,
    )
    run_note: str = phase1.note if note is None else note
    git_commit: str = get_git_sha()
    phase_value: int = int(report.get("phase") or resolved_phase_tag)
    parent_name: str = parent_run_name(phase_value, hypothesis)
    dataset_frame: pd.DataFrame | None = report.get("dataset_frame")
    dataset_source: str = str(
        report.get("dataset_source") or "pregame features joined to fct_games"
    )
    dataset: Any | None = None
    if dataset_frame is not None:
        dataset = _cv_pandas_dataset(dataset_frame, dataset_source)
    run_ids: list[dict[str, str]] = []
    with mlflow.start_run(run_name=parent_name) as parent_run:
        mlflow.set_tag("phase", resolved_phase_tag)
        if hypothesis is not None:
            mlflow.set_tag("hypothesis", hypothesis)
        mlflow.set_tag("cv_scheme", CV_SCHEME)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("reference_model", reference_model)
        mlflow.set_tag("reference_feature_set", reference_feature_set)
        mlflow.set_tag("best_feature_set", best_feature_set)
        mlflow.set_tag("mlflow.note.content", run_note)
        digest: str | None = None
        if dataset is not None and dataset_frame is not None:
            digest = _log_dataset_lineage(
                dataset,
                n_rows=len(dataset_frame),
                n_columns=int(dataset_frame.shape[1]),
            )
        mlflow.log_artifact(paths["markdown_path"])
        mlflow.log_artifact(paths["csv_path"])
        for extra_path in paths.get("extra_paths", {}).values():
            mlflow.log_artifact(extra_path)
        acceptance_rows: list[dict[str, Any]] = list(report.get("acceptance") or [])
        for result in report["results"]:
            if result["status"] != "ok":
                continue
            run_ids.append(
                _log_child_run(
                    result,
                    paths,
                    resolved_phase_tag,
                    hypothesis,
                    run_note,
                    git_commit,
                    reference_model,
                    reference_feature_set,
                    dataset,
                    dataset_frame,
                    acceptance_rows,
                )
            )
        parent_run_id: str = parent_run.info.run_id
    return {
        "run_ids": run_ids,
        "parent_run_id": parent_run_id,
        "parent_run_name": parent_name,
        "data_digest": digest,
        "best_feature_set": best_feature_set,
        "experiment_name": resolved_experiment,
        "tracking_uri": resolved_uri,
    }
