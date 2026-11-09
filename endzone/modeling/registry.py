"""Model registry names, the prod alias, and the promotion gate.

Weekly scoring loads alias ``prod``. It does not set that alias. Registration
writes a version with CV tags and leaves the alias alone. Promotion is a
separate call that sets the alias only when the tags pass the gate.

Unity Catalog name: ``workspace.nfl_dev.endzone_winprob`` (underscores, three
levels). A sqlite registry, including tests, uses ``endzone-winprob``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, NamedTuple

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from endzone.config import (
    configured_registered_model_name,
    unity_catalog_registry,
)
from endzone.modeling.evaluation import MIN_LOG_LOSS_IMPROVEMENT, decide_log_loss_improvement
from endzone.modeling.reference import load_reference_metrics
from endzone.modeling.tracking import (
    RegisteredModelLog,
    activate_tracking_experiment,
    get_git_sha,
    log_registered_sklearn_model,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

PROD_ALIAS: str = "prod"
REFERENCE_MODEL_NAME: str = "logistic"
REFERENCE_FEATURE_SET: str = "h2_full"
REGISTRY_EXPERIMENT_NAME: str = "nfl_pregame/registry"
TAG_MODEL: str = "model"
TAG_FEATURE_SET: str = "feature_set"
TAG_CV_LOG_LOSS: str = "cv_log_loss"
TAG_CV_N: str = "cv_n"
TAG_LOG_LOSS_DIFF: str = "log_loss_diff"
TAG_CI_LOW: str = "log_loss_diff_ci_low"
TAG_CI_HIGH: str = "log_loss_diff_ci_high"
TAG_THROUGH_SEASON: str = "through_season"
TAG_GIT_COMMIT: str = "git_commit"


class ProdModel(NamedTuple):
    """A model loaded from the ``prod`` alias."""

    model: object
    version: str
    alias: str
    model_name: str
    model_uri: str


class ProdAliasError(Exception):
    """The ``prod`` alias is not on this registered model."""


def format_scored_version(version: str, alias: str | None) -> str:
    """Record the registry version and, when one was used, its alias.

    A loaded prod model is ``3@prod``. A fallback fit with no alias stays
    ``3`` or ``local``.
    """
    text: str = version.strip()
    if alias is None or alias.strip() == "":
        return text
    return f"{text}@{alias.strip()}"


def format_tag_number(value: object) -> str:
    """Render a metric as a stable registry tag.

    Numpy integers from a CSV count as numbers. Booleans and strings do not.

    Raises:
        ValueError: The value is missing, boolean, or not finite.
    """
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"Metric {value!r} is not a number.")
    try:
        number: float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {value!r} is not a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Metric {value!r} is not finite.")
    if number.is_integer():
        return str(int(number))
    return format(number, ".10g")


@contextmanager
def pinned_registry(registry_uri: str) -> Iterator[None]:
    """Point the MLflow registry at ``registry_uri`` for one block, then restore it.

    An unset registry is restored as unset. Writing back the resolved URI
    would pin later clients to a store they did not choose.
    """
    from mlflow.tracking._model_registry import utils as registry_utils

    previous_global: str | None = registry_utils._registry_uri
    previous_env: str | None = os.environ.get("MLFLOW_REGISTRY_URI")
    mlflow.set_registry_uri(registry_uri)
    try:
        yield
    finally:
        registry_utils._registry_uri = previous_global
        if previous_env is None:
            os.environ.pop("MLFLOW_REGISTRY_URI", None)
        else:
            os.environ["MLFLOW_REGISTRY_URI"] = previous_env


def assert_can_register(tracking_uri: str, registry_uri: str) -> None:
    """Refuse a Unity Catalog register whose artifacts would stay on sqlite.

    A sqlite tracking URI cannot be copied into Unity Catalog. Tests and the
    weekly fallback use sqlite for both. The explicit register command uses
    ``databricks`` tracking with ``databricks-uc``.

    Raises:
        ValueError: The registry is Unity Catalog and tracking is not Databricks.
    """
    if not unity_catalog_registry(registry_uri):
        return
    tracking: str = tracking_uri.strip()
    remote: bool = tracking == "databricks" or tracking.startswith("databricks")
    if remote:
        return
    raise ValueError(
        "Unity Catalog registration needs MLFLOW_TRACKING_URI=databricks "
        "so the artifact is stored where the registry can read it. "
        "A sqlite tracking URI is for tests and for the weekly fallback."
    )


def promotion_decision(
    tags: Mapping[str, str],
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether one registered version may receive the ``prod`` alias.

    Logistic ``h2_full`` passes when its logged rolling-origin log loss and
    ``n`` match ``docs/experiments/reference_metrics.json`` within that file's
    tolerance. That is the golden bar. It does not have to beat itself, and
    it does not have to beat the market.

    Any other model passes only when the pre-registered rule accepts it
    (log loss at least 0.003 lower, 95% interval entirely below 0) and its
    log loss is at least 0.003 below the golden ``h2_full`` log loss.

    Returns:
        Dict with ``allowed``, ``reason``, and the numbers that were compared.
    """
    locked: Mapping[str, Any] = reference if reference is not None else load_reference_metrics()
    golden: dict[str, Any] = _golden_row(locked, REFERENCE_MODEL_NAME, REFERENCE_FEATURE_SET)
    tolerance: float = float(locked["tolerance"])
    golden_loss: float = float(golden["log_loss"])
    golden_n: float = float(golden["n"])
    model_name: str = str(tags.get(TAG_MODEL, "")).strip()
    feature_set: str = str(tags.get(TAG_FEATURE_SET, "")).strip()
    cv_log_loss: float | None = _parse_metric(tags.get(TAG_CV_LOG_LOSS))
    cv_n: float | None = _parse_metric(tags.get(TAG_CV_N))
    diff: float | None = _parse_metric(tags.get(TAG_LOG_LOSS_DIFF))
    ci_low: float | None = _parse_metric(tags.get(TAG_CI_LOW))
    ci_high: float | None = _parse_metric(tags.get(TAG_CI_HIGH))
    decision: dict[str, Any] = {
        "allowed": False,
        "reason": "missing_cv_metrics",
        "model": model_name,
        "feature_set": feature_set,
        "cv_log_loss": cv_log_loss,
        "cv_n": cv_n,
        "golden_log_loss": golden_loss,
        "golden_n": golden_n,
        "tolerance": tolerance,
        "log_loss_diff": diff,
        "log_loss_diff_ci_low": ci_low,
        "log_loss_diff_ci_high": ci_high,
    }
    if cv_log_loss is None or cv_n is None:
        return decision
    if model_name == REFERENCE_MODEL_NAME and feature_set == REFERENCE_FEATURE_SET:
        loss_ok: bool = abs(cv_log_loss - golden_loss) <= tolerance
        n_ok: bool = abs(cv_n - golden_n) <= tolerance
        decision["allowed"] = loss_ok and n_ok
        decision["reason"] = (
            "matches_golden_reference" if loss_ok and n_ok else "outside_golden_tolerance"
        )
        return decision
    labeled: dict[str, Any] = decide_log_loss_improvement(diff, ci_low, ci_high)
    rule: str = str(labeled["decision"])
    beats_golden: bool = cv_log_loss <= golden_loss - MIN_LOG_LOSS_IMPROVEMENT
    if rule == "accepted" and beats_golden:
        decision["allowed"] = True
        decision["reason"] = "accepted_by_pre_registered_rule"
        return decision
    if rule == "accepted" and not beats_golden:
        decision["reason"] = "short_of_golden_margin"
        return decision
    decision["reason"] = rule
    return decision


def metrics_from_reference_check(result: Mapping[str, Any]) -> dict[str, str]:
    """Pull logistic ``h2_full`` tags from a passing reference check.

    Raises:
        ValueError: The check failed or did not observe the reference row.
    """
    if result.get("ok") is not True:
        raise ValueError("Reference check did not match the golden metrics.")
    observed: object = result.get("observed")
    if not isinstance(observed, list):
        raise ValueError("Reference check has no observed rows.")
    for row in observed:
        if not isinstance(row, dict):
            continue
        if (
            row.get("model") != REFERENCE_MODEL_NAME
            or row.get("feature_set") != REFERENCE_FEATURE_SET
        ):
            continue
        return {
            TAG_MODEL: REFERENCE_MODEL_NAME,
            TAG_FEATURE_SET: REFERENCE_FEATURE_SET,
            TAG_CV_LOG_LOSS: format_tag_number(row.get("log_loss")),
            TAG_CV_N: format_tag_number(row.get("n")),
        }
    raise ValueError("Reference check did not observe logistic h2_full.")


def metrics_from_comparison_csv(path: str) -> dict[str, str]:
    """Read the logistic ``h2_full`` row from a comparison CSV.

    Raises:
        ValueError: The file has no single reference row, or a metric is blank.
    """
    frame: pd.DataFrame = pd.read_csv(path)
    required: set[str] = {"model", "feature_set", "log_loss", "n"}
    missing: list[str] = sorted(required - set(frame.columns))
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Comparison CSV is missing {missing_list}.")
    matched: pd.DataFrame = frame.loc[
        (frame["model"].astype(str) == REFERENCE_MODEL_NAME)
        & (frame["feature_set"].astype(str) == REFERENCE_FEATURE_SET)
    ]
    if len(matched) != 1:
        raise ValueError(f"Comparison CSV needs one logistic h2_full row. Found {len(matched)}.")
    row: pd.Series = matched.iloc[0]
    return {
        TAG_MODEL: REFERENCE_MODEL_NAME,
        TAG_FEATURE_SET: REFERENCE_FEATURE_SET,
        TAG_CV_LOG_LOSS: format_tag_number(row["log_loss"]),
        TAG_CV_N: format_tag_number(row["n"]),
    }


def register_model_version(
    model: object,
    *,
    example: pd.DataFrame,
    model_name: str,
    tags: Mapping[str, str],
    tracking_uri: str,
    registry_uri: str,
    run_name: str,
    through_season: int | None = None,
) -> RegisteredModelLog:
    """Log one sklearn model and register it. Does not set an alias.

    Args:
        model: Fitted estimator.
        example: One row of the training columns, used as the signature.
        model_name: Registry name for this URI.
        tags: CV tags stored on the version. Values are strings.
        tracking_uri: Where the run and artifacts are written.
        registry_uri: Where the version is created.
        run_name: MLflow run name.
        through_season: Optional cutoff stored as a tag.

    Returns:
        The registered version. The ``prod`` alias is unchanged.
    """
    assert_can_register(tracking_uri, registry_uri)
    with pinned_registry(registry_uri):
        activate_tracking_experiment(REGISTRY_EXPERIMENT_NAME, tracking_uri)
        signature = infer_signature(example, model.predict_proba(example)[:, 1])  # type: ignore[attr-defined]
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("stage", "registry")
            mlflow.set_tag("model_alias", "none")
            mlflow.set_tag("git_commit", get_git_sha())
            mlflow.set_tag(
                "mlflow.note.content",
                "Registered without an alias. Promote with make promote-model.",
            )
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            if through_season is not None:
                mlflow.set_tag(TAG_THROUGH_SEASON, str(int(through_season)))
            logged: RegisteredModelLog = log_registered_sklearn_model(
                model,
                signature=signature,
                input_example=example,
                registered_model_name=model_name,
            )
        _write_version_tags(
            model_name,
            logged.model_version,
            tags,
            registry_uri,
            through_season=through_season,
        )
    logger.info(
        "Registered %s version %s. No alias was set.",
        model_name,
        logged.model_version,
    )
    return logged


def read_version_tags(model_name: str, version: str, registry_uri: str) -> dict[str, str]:
    """Return the string tags stored on one model version.

    Raises:
        mlflow.MlflowException: The version does not exist or the registry
            cannot be read.
    """
    with pinned_registry(registry_uri):
        client = mlflow.MlflowClient(registry_uri=registry_uri)
        info = client.get_model_version(model_name, str(version))
    raw: object = getattr(info, "tags", None) or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def promote_registered_version(
    model_name: str,
    version: str,
    registry_uri: str,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Set alias ``prod`` when the version's tags pass the gate.

    A refusal leaves every alias as it was. This does not delete a version
    and it does not create one.

    Returns:
        The promotion decision, plus ``model_name``, ``model_version``, and
        ``alias`` (``prod`` only when ``allowed`` is true).
    """
    tags: dict[str, str] = read_version_tags(model_name, version, registry_uri)
    decision: dict[str, Any] = promotion_decision(tags, reference)
    decision["model_name"] = model_name
    decision["model_version"] = str(version)
    decision["alias"] = None
    if not decision["allowed"]:
        logger.error(
            "Refusing to set alias %s on %s version %s: %s.",
            PROD_ALIAS,
            model_name,
            version,
            decision["reason"],
        )
        return decision
    with pinned_registry(registry_uri):
        client = mlflow.MlflowClient(registry_uri=registry_uri)
        client.set_registered_model_alias(model_name, PROD_ALIAS, str(version))
    decision["alias"] = PROD_ALIAS
    logger.info("Set alias %s on %s version %s.", PROD_ALIAS, model_name, version)
    return decision


def load_prod_model(model_name: str, registry_uri: str) -> ProdModel:
    """Load ``models:/<name>@prod``.

    Restores the tracking and registry URIs that were set on entry.

    Raises:
        ProdAliasError: The alias is missing.
        mlflow.MlflowException: The registry or the artifact cannot be read.
    """
    previous_tracking: str = mlflow.get_tracking_uri()
    try:
        with pinned_registry(registry_uri):
            if registry_uri.startswith("sqlite:") or registry_uri.startswith("file:"):
                mlflow.set_tracking_uri(registry_uri)
            elif registry_uri == "databricks" or registry_uri.startswith("databricks"):
                mlflow.set_tracking_uri("databricks")
            client = mlflow.MlflowClient(registry_uri=registry_uri)
            try:
                version_info = client.get_model_version_by_alias(model_name, PROD_ALIAS)
            except mlflow.MlflowException as exc:
                if not _alias_is_missing(exc):
                    raise
                raise ProdAliasError(f"{model_name} has no alias {PROD_ALIAS}: {exc}") from exc
            version: str = str(version_info.version)
            model_uri: str = f"models:/{model_name}@{PROD_ALIAS}"
            model: object = mlflow.sklearn.load_model(model_uri)
            return ProdModel(
                model=model,
                version=version,
                alias=PROD_ALIAS,
                model_name=model_name,
                model_uri=model_uri,
            )
    finally:
        mlflow.set_tracking_uri(previous_tracking)


def default_model_name(registry_uri: str) -> str:
    """Registry name for ``registry_uri``, including an env override."""
    return configured_registered_model_name(registry_uri)


def _write_version_tags(
    model_name: str,
    version: str,
    tags: Mapping[str, str],
    registry_uri: str,
    *,
    through_season: int | None,
) -> None:
    """Copy CV tags onto the model version so promote can read them later."""
    client = mlflow.MlflowClient(registry_uri=registry_uri)
    stored: dict[str, str] = {str(key): str(value) for key, value in tags.items()}
    stored[TAG_GIT_COMMIT] = get_git_sha()
    if through_season is not None:
        stored[TAG_THROUGH_SEASON] = str(int(through_season))
    for key, value in stored.items():
        client.set_model_version_tag(model_name, str(version), key, value)


def _golden_row(reference: Mapping[str, Any], model_name: str, feature_set: str) -> dict[str, Any]:
    """Return one model row from a reference-metrics object.

    Raises:
        ValueError: The row is missing.
    """
    models: object = reference.get("models")
    if not isinstance(models, list):
        raise ValueError("Reference metrics have no models list.")
    for row in models:
        if (
            isinstance(row, dict)
            and row.get("model") == model_name
            and row.get("feature_set") == feature_set
        ):
            return row
    raise ValueError(f"Reference metrics have no {model_name} {feature_set} row.")


def _alias_is_missing(exc: mlflow.MlflowException) -> bool:
    """True when the registry answered that the model or alias does not exist.

    A connection or permission error is not a missing alias. The caller
    re-raises those so the weekly log says the registry could not be read.
    """
    code: str = str(getattr(exc, "error_code", "") or "")
    if code == "RESOURCE_DOES_NOT_EXIST":
        return True
    text: str = str(exc).lower()
    return "does not exist" in text or "not found" in text


def _parse_metric(value: object) -> float | None:
    """Return a finite float, or None when the tag is blank or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number: float = float(value)
        if math.isfinite(number):
            return number
        return None
    text: str = str(value).strip()
    if text == "":
        return None
    try:
        parsed: float = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
