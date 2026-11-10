"""Model registry names, the prod alias, and the promotion gate.

Weekly scoring loads alias ``prod``. It does not set that alias. Registration
writes a version with CV tags and leaves the alias alone. Promotion is a
separate call that sets the alias only when the tags pass the gate.

Unity Catalog name: ``workspace.nfl_dev.endzone_winprob`` (underscores, three
levels). A sqlite registry, including tests, uses ``endzone-winprob``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
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
TAG_EVALUATION: str = "evaluation"
TAG_V1_LOG_LOSS: str = "v1_log_loss"
TAG_CUTOFF_SEASON: str = "cutoff_season"
TAG_CUTOFF_WEEK: str = "cutoff_week"
TAG_GATE: str = "gate"
TAG_GATE_MARGIN: str = "gate_margin"
TAG_PASSES: str = "passes"
TAG_LOGISTIC_C: str = "logistic_c"
TAG_LOGISTIC_MAX_ITER: str = "logistic_max_iter"
TAG_LOOKBACK_GAMES: str = "lookback_games"
TAG_RANDOM_STATE: str = "random_state"
WALK_FORWARD_EVALUATION: str = "walk_forward"
WALK_FORWARD_PROTOCOL: str = "walk_forward_week"
GATE_NON_INFERIORITY: str = "non_inferiority"
GATE_STRICT_SUPERIORITY: str = "strict_superiority"
# A same-recipe refresh may be this much worse than current prod and still
# promote. 0.002 is below the 0.003 feature-adoption rule, so a refresh is
# not a back door for a new feature. The warehouse gap on the first run was
# about 0.0004, inside this margin. A different recipe does not use it.
REFRESH_LOG_LOSS_MARGIN: float = 0.002
_WALK_FORWARD_DIFF_TOLERANCE: float = 1e-5
_RECIPE_C_TOLERANCE: float = 1e-12


class ProdModel(NamedTuple):
    """A model loaded from the ``prod`` alias."""

    model: object
    version: str
    alias: str
    model_name: str
    model_uri: str


class ProductionRecipe(NamedTuple):
    """Model type, feature set, and logistic hyperparameters of alias ``prod``.

    Version 1 and the weekly fallback both use these values. A walk-forward
    refresh is the same recipe on a later cutoff. Anything else is a challenger.
    """

    model: str
    feature_set: str
    logistic_c: float
    logistic_max_iter: int
    lookback_games: int
    random_state: int


PROD_RECIPE: ProductionRecipe = ProductionRecipe(
    model=REFERENCE_MODEL_NAME,
    feature_set=REFERENCE_FEATURE_SET,
    logistic_c=1.0,
    logistic_max_iter=1000,
    lookback_games=10,
    random_state=42,
)


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
    it does not have to beat the market. A ``through_season`` past
    ``cv_end_season`` does not use that bar. It needs ``evaluation``
    ``walk_forward``.

    A walk-forward version of the same recipe passes when its pooled log loss
    is at most ``REFRESH_LOG_LOSS_MARGIN`` above the current prod loss on the
    same weeks. That prod loss is the report's v1 column while prod is the
    through-2023 fit. A different model, feature set, or hyperparameter has
    to be strictly lower. The 0.003 feature rule is a different gate and is
    not applied here. The bootstrap interval is descriptive.

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
    if str(tags.get(TAG_EVALUATION, "")).strip() == WALK_FORWARD_EVALUATION:
        return _decide_walk_forward(tags, decision)
    through_season_value: float | None = _parse_metric(tags.get(TAG_THROUGH_SEASON))
    cv_end_season: float = float(locked["cv_end_season"])
    if through_season_value is not None and through_season_value > cv_end_season:
        decision["reason"] = "fresh_window_needs_walk_forward"
        decision["through_season"] = through_season_value
        return decision
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


def read_walk_forward_report(path: str) -> dict[str, Any]:
    """Load a walk-forward JSON report.

    Raises:
        ValueError: The path is missing, or the file is not a JSON object.
    """
    report_path: Path = Path(path)
    if not report_path.is_file():
        raise ValueError(f"Walk-forward report not found: {report_path}")
    payload: object = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Walk-forward report must be a JSON object.")
    return payload


def recipe_matches(candidate: ProductionRecipe, defender: ProductionRecipe) -> bool:
    """Return True when two recipes are the same model, features, and hyperparameters."""
    if candidate.model != defender.model or candidate.feature_set != defender.feature_set:
        return False
    if candidate.logistic_max_iter != defender.logistic_max_iter:
        return False
    if candidate.lookback_games != defender.lookback_games:
        return False
    if candidate.random_state != defender.random_state:
        return False
    return abs(candidate.logistic_c - defender.logistic_c) <= _RECIPE_C_TOLERANCE


def walk_forward_promotion_gate(
    candidate_loss: float, defender_loss: float, same_recipe: bool
) -> dict[str, Any]:
    """Choose non-inferiority or strict superiority from the recipe comparison.

    The same recipe may sit up to ``REFRESH_LOG_LOSS_MARGIN`` above the
    defender. A different recipe has to be strictly lower. The margin is not
    applied to a challenger. Equality with the defender passes only for a
    refresh.

    Returns:
        Dict with ``gate``, ``margin``, ``passes``, ``limit``, and ``same_recipe``.
    """
    if same_recipe:
        margin: float = REFRESH_LOG_LOSS_MARGIN
        gate: str = GATE_NON_INFERIORITY
        passes: bool = candidate_loss <= defender_loss + margin
    else:
        margin = 0.0
        gate = GATE_STRICT_SUPERIORITY
        passes = candidate_loss < defender_loss
    return {
        "gate": gate,
        "margin": margin,
        "passes": passes,
        "limit": defender_loss + margin,
        "same_recipe": same_recipe,
    }


def recipe_from_walk_forward(payload: Mapping[str, Any]) -> ProductionRecipe:
    """Read the fitted recipe from a walk-forward report.

    Raises:
        ValueError: A recipe field is missing or not the right type.
    """
    model_name: object = payload.get("model")
    feature_set: object = payload.get("feature_set")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Walk-forward report is missing model.")
    if not isinstance(feature_set, str) or not feature_set.strip():
        raise ValueError("Walk-forward report is missing feature_set.")
    logistic_c: float = _require_float(payload.get("logistic_c"), "logistic_c")
    if logistic_c <= 0:
        raise ValueError("logistic_c must be positive.")
    return ProductionRecipe(
        model=model_name.strip(),
        feature_set=feature_set.strip(),
        logistic_c=logistic_c,
        logistic_max_iter=_require_int(
            payload.get("logistic_max_iter"), "logistic_max_iter", minimum=1
        ),
        lookback_games=_require_int(payload.get("lookback_games"), "lookback_games", minimum=1),
        random_state=_require_int(payload.get("random_state"), "random_state"),
    )


def metrics_from_walk_forward_report(payload: Mapping[str, Any]) -> dict[str, str]:
    """Turn a walk-forward report into registry tags.

    The aggregate log loss has to match the week rows. ``beats_v1`` has to
    match a strict improvement. ``passes`` has to match the refresh or
    challenger gate. A hand edit of one field fails here. A report written
    before the gate fields existed still validates. This function fills the
    tags from the losses and the recipe.

    Raises:
        ValueError: The report is not the walk-forward protocol, or its
            numbers disagree.
    """
    _validate_walk_forward_report(payload)
    aggregate: Mapping[str, Any] = payload["aggregate"]
    slate: Mapping[str, Any] = payload["slate"]
    recipe: ProductionRecipe = recipe_from_walk_forward(payload)
    gate: dict[str, Any] = walk_forward_promotion_gate(
        float(aggregate["candidate_log_loss"]),
        float(aggregate["v1_log_loss"]),
        recipe_matches(recipe, PROD_RECIPE),
    )
    included_season: int = _max_included_season(payload)
    tags: dict[str, str] = {
        TAG_MODEL: recipe.model,
        TAG_FEATURE_SET: recipe.feature_set,
        TAG_EVALUATION: WALK_FORWARD_EVALUATION,
        TAG_CV_LOG_LOSS: format_tag_number(aggregate["candidate_log_loss"]),
        TAG_V1_LOG_LOSS: format_tag_number(aggregate["v1_log_loss"]),
        TAG_CV_N: format_tag_number(aggregate["n"]),
        TAG_LOG_LOSS_DIFF: format_tag_number(aggregate["log_loss_diff"]),
        TAG_CUTOFF_SEASON: str(int(slate["season"])),
        TAG_CUTOFF_WEEK: str(int(slate["week"])),
        TAG_THROUGH_SEASON: str(included_season),
        TAG_LOGISTIC_C: format_tag_number(recipe.logistic_c),
        TAG_LOGISTIC_MAX_ITER: str(recipe.logistic_max_iter),
        TAG_LOOKBACK_GAMES: str(recipe.lookback_games),
        TAG_RANDOM_STATE: str(recipe.random_state),
        TAG_GATE: str(gate["gate"]),
        TAG_GATE_MARGIN: format_tag_number(gate["margin"]),
        TAG_PASSES: "true" if gate["passes"] else "false",
    }
    ci_low: object = aggregate.get("log_loss_diff_ci_low")
    ci_high: object = aggregate.get("log_loss_diff_ci_high")
    if ci_low is not None and ci_high is not None:
        tags[TAG_CI_LOW] = format_tag_number(ci_low)
        tags[TAG_CI_HIGH] = format_tag_number(ci_high)
    return tags


def _recipe_from_tags(tags: Mapping[str, str]) -> ProductionRecipe | None:
    """Return the recipe stored on a version, or None when a field is missing."""
    model_name: str = str(tags.get(TAG_MODEL, "")).strip()
    feature_set: str = str(tags.get(TAG_FEATURE_SET, "")).strip()
    logistic_c: float | None = _parse_metric(tags.get(TAG_LOGISTIC_C))
    max_iter: float | None = _parse_metric(tags.get(TAG_LOGISTIC_MAX_ITER))
    lookback: float | None = _parse_metric(tags.get(TAG_LOOKBACK_GAMES))
    seed: float | None = _parse_metric(tags.get(TAG_RANDOM_STATE))
    if (
        not model_name
        or not feature_set
        or logistic_c is None
        or max_iter is None
        or lookback is None
        or seed is None
    ):
        return None
    if not max_iter.is_integer() or not lookback.is_integer() or not seed.is_integer():
        return None
    if logistic_c <= 0:
        return None
    return ProductionRecipe(
        model=model_name,
        feature_set=feature_set,
        logistic_c=logistic_c,
        logistic_max_iter=int(max_iter),
        lookback_games=int(lookback),
        random_state=int(seed),
    )


def _decide_walk_forward(tags: Mapping[str, str], decision: dict[str, Any]) -> dict[str, Any]:
    """Allow prod when the recipe's own gate passes.

    The same recipe uses non-inferiority. A different recipe uses strict
    superiority. The comparison is recomputed from the loss tags and the
    recipe tags. A stored ``passes`` flag that disagrees is refused.
    """
    candidate: float | None = _parse_metric(tags.get(TAG_CV_LOG_LOSS))
    v1_loss: float | None = _parse_metric(tags.get(TAG_V1_LOG_LOSS))
    sample_n: float | None = _parse_metric(tags.get(TAG_CV_N))
    diff: float | None = _parse_metric(tags.get(TAG_LOG_LOSS_DIFF))
    cutoff_season: float | None = _parse_metric(tags.get(TAG_CUTOFF_SEASON))
    cutoff_week: float | None = _parse_metric(tags.get(TAG_CUTOFF_WEEK))
    recipe: ProductionRecipe | None = _recipe_from_tags(tags)
    decision["v1_log_loss"] = v1_loss
    decision["cutoff_season"] = cutoff_season
    decision["cutoff_week"] = cutoff_week
    if (
        candidate is None
        or v1_loss is None
        or sample_n is None
        or diff is None
        or cutoff_season is None
        or cutoff_week is None
        or cutoff_week < 1
        or recipe is None
    ):
        decision["reason"] = "missing_walk_forward_metrics"
        return decision
    if abs((candidate - v1_loss) - diff) > _WALK_FORWARD_DIFF_TOLERANCE:
        decision["reason"] = "walk_forward_tags_inconsistent"
        return decision
    if sample_n <= 0:
        decision["reason"] = "missing_walk_forward_metrics"
        return decision
    same_recipe: bool = recipe_matches(recipe, PROD_RECIPE)
    gate: dict[str, Any] = walk_forward_promotion_gate(candidate, v1_loss, same_recipe)
    decision["gate"] = gate["gate"]
    decision["margin"] = gate["margin"]
    decision["passes"] = gate["passes"]
    decision["same_recipe"] = same_recipe
    if not _stored_gate_matches(tags, gate):
        decision["reason"] = "walk_forward_gate_inconsistent"
        return decision
    if gate["passes"]:
        decision["allowed"] = True
        decision["reason"] = (
            "walk_forward_refresh_passes" if same_recipe else "walk_forward_challenger_passes"
        )
        return decision
    decision["reason"] = (
        "walk_forward_refresh_fails" if same_recipe else "walk_forward_challenger_fails"
    )
    return decision


def _stored_gate_matches(tags: Mapping[str, str], gate: Mapping[str, Any]) -> bool:
    """Return False when a stored gate field disagrees with the recomputed one.

    Missing gate tags are filled by the recomputation. A present tag has to
    match. That is what stops a hand-edited ``passes`` flag.
    """
    stored_gate: str = str(tags.get(TAG_GATE, "")).strip()
    if stored_gate and stored_gate != gate["gate"]:
        return False
    stored_margin: float | None = _parse_metric(tags.get(TAG_GATE_MARGIN))
    if (
        stored_margin is not None
        and abs(stored_margin - float(gate["margin"])) > _WALK_FORWARD_DIFF_TOLERANCE
    ):
        return False
    stored_passes: str = str(tags.get(TAG_PASSES, "")).strip().lower()
    if stored_passes in {"true", "false"}:
        expected: str = "true" if gate["passes"] else "false"
        if stored_passes != expected:
            return False
    elif stored_passes:
        return False
    return True


def _validate_walk_forward_report(payload: Mapping[str, Any]) -> None:
    """Raise when a walk-forward report disagrees with itself."""
    if payload.get("protocol") != WALK_FORWARD_PROTOCOL:
        raise ValueError("Walk-forward report protocol is not walk_forward_week.")
    recipe_from_walk_forward(payload)
    v1_through: int = _require_int(payload.get("v1_through_season"), "v1_through_season")
    _require_int(payload.get("lookback_games"), "lookback_games", minimum=1)
    _require_int(payload.get("logistic_max_iter"), "logistic_max_iter", minimum=1)
    _require_int(payload.get("random_state"), "random_state")
    n_bootstrap: int = _require_int(payload.get("n_bootstrap"), "n_bootstrap", minimum=0)
    _require_int(payload.get("bootstrap_seed"), "bootstrap_seed")
    logistic_c: float = _require_float(payload.get("logistic_c"), "logistic_c")
    if logistic_c <= 0:
        raise ValueError("logistic_c must be positive.")
    slate: object = payload.get("slate")
    if not isinstance(slate, Mapping):
        raise ValueError("Walk-forward report is missing slate.")
    slate_season: int = _require_int(slate.get("season"), "slate.season")
    slate_week: int = _require_int(slate.get("week"), "slate.week", minimum=1)
    weeks: object = payload.get("weeks")
    if not isinstance(weeks, list) or len(weeks) == 0:
        raise ValueError("Walk-forward report has no scored weeks.")
    aggregate: object = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("Walk-forward report is missing aggregate.")
    ordered: list[Mapping[str, Any]] = []
    for index, week in enumerate(weeks):
        if not isinstance(week, Mapping):
            raise ValueError(f"Week row {index} is not an object.")
        ordered.append(week)
    ordered.sort(
        key=lambda row: (
            _require_int(row.get("season"), "season"),
            _require_int(row.get("week"), "week"),
        )
    )
    previous_train: int = -1
    v1_train_counts: set[int] = set()
    total_n: int = 0
    weighted: dict[str, float] = {
        "candidate_log_loss": 0.0,
        "v1_log_loss": 0.0,
        "elo_log_loss": 0.0,
        "baseline_home_log_loss": 0.0,
    }
    market_n: int = 0
    market_weighted: float = 0.0
    slate_key: tuple[int, int] = (slate_season, slate_week)
    for week in ordered:
        season: int = _require_int(week.get("season"), "season")
        week_number: int = _require_int(week.get("week"), "week", minimum=1)
        if season <= v1_through:
            raise ValueError(f"{season} week {week_number} is inside the v1 window.")
        if (season, week_number) >= slate_key:
            raise ValueError(f"{season} week {week_number} is not before the slate.")
        row_n: int = _require_int(week.get("n"), "n", minimum=1)
        train_candidate: int = _require_int(
            week.get("n_train_candidate"), "n_train_candidate", minimum=1
        )
        train_v1: int = _require_int(week.get("n_train_v1"), "n_train_v1", minimum=1)
        if train_candidate < train_v1:
            raise ValueError(
                f"{season} week {week_number} trains the candidate on fewer games than v1."
            )
        if train_candidate < previous_train:
            raise ValueError(
                "Candidate training count goes backwards. Later weeks must not drop games."
            )
        previous_train = train_candidate
        v1_train_counts.add(train_v1)
        total_n += row_n
        for key in weighted:
            weighted[key] += _require_float(week.get(key), key) * row_n
        row_market_n: int = _require_int(week.get("market_n"), "market_n", minimum=0)
        if row_market_n > row_n:
            raise ValueError(f"{season} week {week_number} has more market rows than games.")
        if row_market_n > 0:
            market_n += row_market_n
            market_weighted += (
                _require_float(week.get("market_log_loss"), "market_log_loss") * row_market_n
            )
    if len(v1_train_counts) != 1:
        raise ValueError("v1 training count changes across weeks.")
    aggregate_n: int = _require_int(aggregate.get("n"), "aggregate.n", minimum=1)
    if aggregate_n != total_n:
        raise ValueError("Aggregate n does not match the week rows.")
    for key, total in weighted.items():
        pooled: float = total / float(total_n)
        reported: float = _require_float(aggregate.get(key), f"aggregate.{key}")
        if abs(pooled - reported) > _WALK_FORWARD_DIFF_TOLERANCE:
            raise ValueError(f"Aggregate {key} does not match the week rows.")
    candidate_loss: float = _require_float(
        aggregate.get("candidate_log_loss"), "aggregate.candidate_log_loss"
    )
    v1_loss: float = _require_float(aggregate.get("v1_log_loss"), "aggregate.v1_log_loss")
    diff: float = _require_float(aggregate.get("log_loss_diff"), "aggregate.log_loss_diff")
    if abs((candidate_loss - v1_loss) - diff) > _WALK_FORWARD_DIFF_TOLERANCE:
        raise ValueError("log_loss_diff is not candidate minus v1.")
    beats: object = aggregate.get("beats_v1")
    if not isinstance(beats, bool) or beats is not (candidate_loss < v1_loss):
        raise ValueError("beats_v1 does not match the aggregate log loss.")
    _require_reported_gate(payload, candidate_loss, v1_loss)
    reported_market_n: int = _require_int(
        aggregate.get("market_n"), "aggregate.market_n", minimum=0
    )
    if reported_market_n != market_n:
        raise ValueError("Aggregate market_n does not match the week rows.")
    if market_n == 0:
        if aggregate.get("market_log_loss") is not None:
            raise ValueError("market_log_loss must be null when market_n is 0.")
    else:
        pooled_market: float = market_weighted / float(market_n)
        reported_market: float = _require_float(
            aggregate.get("market_log_loss"), "aggregate.market_log_loss"
        )
        if abs(pooled_market - reported_market) > _WALK_FORWARD_DIFF_TOLERANCE:
            raise ValueError("Aggregate market_log_loss does not match the week rows.")
    if n_bootstrap > 0 and aggregate_n >= 2:
        _require_float(aggregate.get("log_loss_diff_ci_low"), "aggregate.log_loss_diff_ci_low")
        _require_float(aggregate.get("log_loss_diff_ci_high"), "aggregate.log_loss_diff_ci_high")


def _require_reported_gate(
    payload: Mapping[str, Any], candidate_loss: float, defender_loss: float
) -> None:
    """Check gate, margin, and passes when the report includes them.

    A report from before those fields existed is still valid. The gate is
    then computed from the losses and the recipe. If any one of the three
    fields is present, all three have to be present and match that computation.
    """
    present: set[str] = {key for key in ("gate", "margin", "passes") if key in payload}
    if not present:
        return
    missing: list[str] = sorted({"gate", "margin", "passes"} - present)
    if missing:
        raise ValueError(f"Walk-forward report is missing {', '.join(missing)}.")
    expected: dict[str, Any] = walk_forward_promotion_gate(
        candidate_loss,
        defender_loss,
        recipe_matches(recipe_from_walk_forward(payload), PROD_RECIPE),
    )
    if payload.get("gate") != expected["gate"]:
        raise ValueError("gate does not match the recipe.")
    margin: float = _require_float(payload.get("margin"), "margin")
    if abs(margin - float(expected["margin"])) > _WALK_FORWARD_DIFF_TOLERANCE:
        raise ValueError("margin does not match the gate.")
    passes: object = payload.get("passes")
    if not isinstance(passes, bool) or passes is not expected["passes"]:
        raise ValueError("passes does not match the gate.")


def _max_included_season(payload: Mapping[str, Any]) -> int:
    """Last season that still has a completed game before the slate."""
    slate: Mapping[str, Any] = payload["slate"]
    slate_season: int = int(slate["season"])
    slate_week: int = int(slate["week"])
    weeks: list[Mapping[str, Any]] = list(payload["weeks"])
    seasons: list[int] = [int(payload["v1_through_season"])]
    for week in weeks:
        season: int = int(week["season"])
        week_number: int = int(week["week"])
        if (season, week_number) < (slate_season, slate_week):
            seasons.append(season)
    return max(seasons)


def _require_int(value: object, label: str, minimum: int | None = None) -> int:
    """Return an integer, accepting an integral float from JSON."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    number: int
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{label} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    return number


def _require_float(value: object, label: str) -> float:
    """Return a finite float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number.")
    number: float = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


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
