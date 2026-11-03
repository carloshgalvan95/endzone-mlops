"""Pooled metrics, paired bootstrap, and the 0.003 acceptance rule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Accept a set only when pooled log loss falls by at least this much.
MIN_LOG_LOSS_IMPROVEMENT: float = 0.003


def binary_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | int | None]:
    """Pooled log loss, Brier, AUC, and accuracy at probability 0.5.

    AUC is missing when the labels have one class. An empty sample leaves
    every estimate missing.
    """
    n_rows: int = int(len(y_true))
    empty: dict[str, float | int | None] = {
        "log_loss": None,
        "brier": None,
        "auc": None,
        "accuracy": None,
        "n": n_rows,
    }
    if n_rows == 0:
        return empty
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_proba, dtype=float)
    hard: np.ndarray = (probabilities >= 0.5).astype(int)
    auc: float | None = None
    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, probabilities))
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auc": auc,
        "accuracy": float(accuracy_score(labels, hard)),
        "n": n_rows,
    }


def paired_log_loss_difference(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_reference: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Paired bootstrap of model log loss minus reference log loss.

    The same row indexes are drawn for both probability vectors. A negative
    difference means the model has lower log loss than the reference.

    Raises:
        ValueError: The arrays differ in length, a value is missing, or
            ``n_bootstrap`` is negative.
    """
    labels = np.asarray(y_true)
    model_probability = np.asarray(y_proba, dtype=float)
    reference_probability = np.asarray(y_reference, dtype=float)
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    if not (len(labels) == len(model_probability) == len(reference_probability)):
        raise ValueError("y_true, y_proba, and y_reference must have the same length")
    if len(labels) > 0 and (
        not np.isfinite(model_probability).all() or not np.isfinite(reference_probability).all()
    ):
        raise ValueError("probabilities must be finite")

    empty: dict[str, float | int | None] = {
        "log_loss": None,
        "reference_log_loss": None,
        "log_loss_diff": None,
        "log_loss_diff_ci_low": None,
        "log_loss_diff_ci_high": None,
        "n": int(len(labels)),
        "n_bootstrap": int(n_bootstrap),
    }
    if len(labels) == 0:
        return empty

    model_loss: float = float(log_loss(labels, model_probability, labels=[0, 1]))
    reference_loss: float = float(log_loss(labels, reference_probability, labels=[0, 1]))
    difference: float = model_loss - reference_loss
    if n_bootstrap == 0 or len(labels) < 2:
        return {
            "log_loss": model_loss,
            "reference_log_loss": reference_loss,
            "log_loss_diff": difference,
            "log_loss_diff_ci_low": None,
            "log_loss_diff_ci_high": None,
            "n": int(len(labels)),
            "n_bootstrap": int(n_bootstrap),
        }

    generator = np.random.default_rng(seed)
    row_count: int = len(labels)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        drawn = generator.integers(0, row_count, size=row_count)
        drawn_difference: float = float(
            log_loss(labels[drawn], model_probability[drawn], labels=[0, 1])
        ) - float(log_loss(labels[drawn], reference_probability[drawn], labels=[0, 1]))
        samples.append(drawn_difference)
    return {
        "log_loss": model_loss,
        "reference_log_loss": reference_loss,
        "log_loss_diff": difference,
        "log_loss_diff_ci_low": float(np.percentile(samples, 2.5)),
        "log_loss_diff_ci_high": float(np.percentile(samples, 97.5)),
        "n": row_count,
        "n_bootstrap": int(n_bootstrap),
    }


def _empty_comparison() -> dict[str, float | int | None]:
    """Comparison fields when two out-of-fold tables cannot be paired."""
    return {
        "log_loss": None,
        "reference_log_loss": None,
        "log_loss_diff": None,
        "log_loss_diff_ci_low": None,
        "log_loss_diff_ci_high": None,
        "n": 0,
        "n_bootstrap": 0,
    }


def compare_oof_to_reference(
    model_oof: pd.DataFrame | None,
    reference_oof: pd.DataFrame | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Pair two out-of-fold tables on ``game_id`` and bootstrap the log loss gap."""
    if model_oof is None or reference_oof is None or model_oof.empty or reference_oof.empty:
        return _empty_comparison()
    left: pd.DataFrame = model_oof[["game_id", "y_true", "y_proba"]].rename(
        columns={"y_proba": "model_proba"}
    )
    right: pd.DataFrame = reference_oof[["game_id", "y_proba"]].rename(
        columns={"y_proba": "reference_proba"}
    )
    paired: pd.DataFrame = left.merge(right, on="game_id", how="inner")
    if paired.empty:
        return _empty_comparison()
    return paired_log_loss_difference(
        paired["y_true"].to_numpy(),
        paired["model_proba"].to_numpy(dtype=float),
        paired["reference_proba"].to_numpy(dtype=float),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def _is_missing_metric(value: object) -> bool:
    """Return True when a metric was not computed."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def decide_log_loss_improvement(
    log_loss_diff: float | None,
    ci_low: float | None,
    ci_high: float | None,
    min_improvement: float = MIN_LOG_LOSS_IMPROVEMENT,
) -> dict[str, Any]:
    """Apply the acceptance rule to one paired log loss difference.

    Accept only when pooled log loss is at least ``min_improvement`` lower
    than the reference and the 95% interval lies entirely below zero. A
    missing interval is inconclusive. Every other scored comparison is rejected.
    """
    missing: bool = (
        _is_missing_metric(log_loss_diff)
        or _is_missing_metric(ci_low)
        or _is_missing_metric(ci_high)
    )
    decision: str = "inconclusive"
    if not missing:
        assert log_loss_diff is not None
        assert ci_low is not None
        assert ci_high is not None
        difference: float = float(log_loss_diff)
        lower: float = float(ci_low)
        upper: float = float(ci_high)
        improves: bool = difference <= -float(min_improvement)
        interval_below_zero: bool = lower <= upper and upper < 0.0
        decision = "accepted" if improves and interval_below_zero else "rejected"
    return {
        "decision": decision,
        "min_improvement": float(min_improvement),
        "log_loss_diff": log_loss_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def acceptance_decisions(
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
    min_improvement: float = MIN_LOG_LOSS_IMPROVEMENT,
) -> list[dict[str, Any]]:
    """Label each CV row. The reference row is not a candidate.

    The reference decision is ``reference``.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        comparison: dict[str, Any] = result.get("comparison") or {}
        is_reference: bool = (
            result["model"] == reference_model and result["feature_set"] == reference_feature_set
        )
        if is_reference:
            labeled: dict[str, Any] = {
                "decision": "reference",
                "min_improvement": float(min_improvement),
                "log_loss_diff": comparison.get("log_loss_diff"),
                "ci_low": comparison.get("log_loss_diff_ci_low"),
                "ci_high": comparison.get("log_loss_diff_ci_high"),
            }
        else:
            labeled = decide_log_loss_improvement(
                comparison.get("log_loss_diff"),
                comparison.get("log_loss_diff_ci_low"),
                comparison.get("log_loss_diff_ci_high"),
                min_improvement=min_improvement,
            )
        rows.append(
            {
                "model": result["model"],
                "feature_set": result["feature_set"],
                **labeled,
            }
        )
    return rows


def resolve_cv_reference(
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
) -> dict[str, Any]:
    """Return the one result row that is the paired reference.

    Raises:
        ValueError: There is not exactly one match.
    """
    matches: list[dict[str, Any]] = [
        result
        for result in results
        if result["model"] == reference_model and result["feature_set"] == reference_feature_set
    ]
    if len(matches) != 1:
        available: str = ", ".join(f"{item['model']}/{item['feature_set']}" for item in results)
        raise ValueError(
            f"Reference {reference_model}/{reference_feature_set} is not in the CV results. "
            f"Available: {available}."
        )
    return matches[0]


def attach_reference_comparison(
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Pair every scored model to one reference and store the log loss gap.

    The reference row is included. Its difference is zero when it scored.
    A skipped model keeps ``comparison`` as None. ``results`` is updated in place.
    """
    reference: dict[str, Any] = resolve_cv_reference(
        results,
        reference_model,
        reference_feature_set,
    )
    reference_oof: pd.DataFrame | None = reference["oof"]
    for result in results:
        if result["status"] != "ok" or result["oof"] is None:
            result["comparison"] = None
            continue
        result["comparison"] = compare_oof_to_reference(
            result["oof"],
            reference_oof,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return reference


def _oof_for(
    results: list[dict[str, Any]],
    model_name: str,
    feature_set: str,
) -> pd.DataFrame | None:
    """Return one scored model's out-of-fold table, or None when it is absent."""
    for result in results:
        if result["model"] == model_name and result["feature_set"] == feature_set:
            oof: pd.DataFrame | None = result.get("oof")
            return oof
    return None


def qb_change_subset_comparison(
    features: pd.DataFrame,
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
    candidate_feature_set: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Paired log loss on games where either depth-chart starter changed.

    The section is descriptive. It is not an acceptance decision. A game
    counts when ``home_qb_changed`` or ``away_qb_changed`` is 1. Fewer than
    two out-of-fold games, or a single class, skips the comparison.
    """
    home_source: pd.Series = features["home_starter_source"]
    away_source: pd.Series = features["away_starter_source"]
    changed: pd.Series = features["home_qb_changed"].eq(1) | features["away_qb_changed"].eq(1)
    changed_ids: set[object] = set(features.loc[changed, "game_id"])
    summary: dict[str, Any] = {
        "n_depth_chart_starters": int((home_source == "depth_chart").sum())
        + int((away_source == "depth_chart").sum()),
        "n_previous_passer_starters": int((home_source == "previous_passer").sum())
        + int((away_source == "previous_passer").sum()),
        "n_qb_change_games": len(changed_ids),
        "status": "skipped",
        "reason": "too_few_games",
        "comparison": None,
    }
    candidate: pd.DataFrame | None = _oof_for(results, "logistic", candidate_feature_set)
    reference: pd.DataFrame | None = _oof_for(results, reference_model, reference_feature_set)
    if candidate is None or reference is None or candidate.empty or reference.empty:
        return summary
    candidate_ids: set[object] = set(candidate["game_id"]).intersection(changed_ids)
    shared_ids: set[object] = candidate_ids.intersection(set(reference["game_id"]))
    if len(shared_ids) < 2:
        return summary
    labels: pd.Series = candidate.loc[candidate["game_id"].isin(shared_ids), "y_true"]
    if int(labels.nunique()) < 2:
        summary["reason"] = "single_class"
        return summary
    summary["status"] = "scored"
    summary["reason"] = None
    summary["comparison"] = compare_oof_to_reference(
        candidate.loc[candidate["game_id"].isin(shared_ids)],
        reference.loc[reference["game_id"].isin(shared_ids)],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return summary
