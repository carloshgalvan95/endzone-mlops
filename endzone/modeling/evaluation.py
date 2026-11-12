"""Pooled metrics, paired bootstrap, and the 0.003 acceptance rule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Accept a set only when pooled log loss falls by at least this much.
MIN_LOG_LOSS_IMPROVEMENT: float = 0.003
N_CALIBRATION_BINS: int = 10
LOGIT_CLIP: float = 1e-6
CALIBRATION_MODELS: frozenset[str] = frozenset(
    {
        "logistic",
        "logistic_l2",
        "elo",
        "market",
        "gbm_pkg_core_inj",
        "ens_mean",
    }
)
# Descriptive market-information rows. Baseline models are not in this set.
INFORMATION_MODELS: frozenset[str] = frozenset(
    {"logistic", "logistic_l2", "gbm_pkg_core_inj", "ens_mean"}
)
CALIBRATION_COLUMNS: tuple[str, ...] = (
    "model",
    "feature_set",
    "row_kind",
    "bin",
    "n",
    "mean_predicted",
    "observed_rate",
    "ece",
    "reliability",
    "resolution",
    "uncertainty",
    "brier",
)
MARKET_INFORMATION_COLUMNS: tuple[str, ...] = (
    "model",
    "feature_set",
    "status",
    "reason",
    "n",
    "intercept",
    "intercept_ci_low",
    "intercept_ci_high",
    "market_coef",
    "market_ci_low",
    "market_ci_high",
    "model_coef",
    "model_ci_low",
    "model_ci_high",
    "n_bootstrap",
    "n_bootstrap_used",
)


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


def require_aligned_oof(
    reference_oof: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    *,
    candidate_name: str,
    allowed_seasons: tuple[int, ...],
) -> dict[str, int]:
    """Require the candidate OOF table to be the reference games, in order.

    The paired log-loss comparison is only the adoption comparison when both
    models score the same games. A shorter candidate table is a dropped game,
    not a smaller sample to report quietly.

    Args:
        reference_oof: Out-of-fold rows for the reference model.
        candidate_oof: Out-of-fold rows for one challenger.
        candidate_name: Name used in the error.
        allowed_seasons: Seasons that may appear. Any other season fails.

    Returns:
        Dict with ``n``, the shared row count.

    Raises:
        ValueError: A required column is missing, a table is empty, a game
            or label differs, or a season is outside ``allowed_seasons``.
    """
    required: tuple[str, ...] = ("game_id", "season", "y_true")
    for label, frame in (("reference", reference_oof), (candidate_name, candidate_oof)):
        missing: list[str] = [column for column in required if column not in frame.columns]
        if missing:
            joined: str = ", ".join(missing)
            raise ValueError(f"{label} OOF is missing columns: {joined}")
        if len(frame) == 0:
            raise ValueError(f"{label} OOF has no rows")
    allowed: set[int] = {int(season) for season in allowed_seasons}
    for label, frame in (("reference", reference_oof), (candidate_name, candidate_oof)):
        seasons: set[int] = {int(season) for season in frame["season"].tolist()}
        outside: set[int] = seasons - allowed
        if outside:
            outside_list: str = ", ".join(str(season) for season in sorted(outside))
            raise ValueError(f"{label} OOF scores seasons outside the protocol: {outside_list}")
    reference_ids: list[str] = [str(game_id) for game_id in reference_oof["game_id"].tolist()]
    candidate_ids: list[str] = [str(game_id) for game_id in candidate_oof["game_id"].tolist()]
    if candidate_ids != reference_ids:
        raise ValueError(
            f"{candidate_name} OOF game_ids do not match the reference. "
            f"reference n={len(reference_ids)} candidate n={len(candidate_ids)}."
        )
    reference_labels: list[int] = [int(label) for label in reference_oof["y_true"].tolist()]
    candidate_labels: list[int] = [int(label) for label in candidate_oof["y_true"].tolist()]
    if candidate_labels != reference_labels:
        raise ValueError(f"{candidate_name} OOF labels do not match the reference")
    return {"n": len(reference_ids)}


def label_market_benchmark(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy acceptance rows and label the market row ``benchmark``.

    The numeric rule can mark the market accepted. That label is not used
    for this row. Logistic rows are unchanged.

    Args:
        decisions: Rows from ``acceptance_decisions``.

    Returns:
        A new list. The market row's decision is ``benchmark``.
    """
    labeled: list[dict[str, Any]] = []
    for row in decisions:
        copied: dict[str, Any] = dict(row)
        if copied.get("model") == "market" and copied.get("feature_set") == "market":
            copied["decision"] = "benchmark"
        labeled.append(copied)
    return labeled


def _as_binary_sample(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate one probability sample.

    Raises:
        ValueError: The arrays differ in length, a label is not 0 or 1, or a
            probability is not finite and inside ``[0, 1]``.
    """
    labels: np.ndarray = np.asarray(y_true, dtype=float)
    probabilities: np.ndarray = np.asarray(y_proba, dtype=float)
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_proba must have the same length")
    if len(labels) > 0 and not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    if len(probabilities) > 0 and (
        not np.isfinite(probabilities).all()
        or bool((probabilities < 0.0).any())
        or bool((probabilities > 1.0).any())
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return labels, probabilities


def reliability_by_decile(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = N_CALIBRATION_BINS,
    game_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Equal-count reliability table, ECE, and the decile Brier components.

    Rows are ordered by probability, then by ``game_ids`` when those are
    passed, and split into up to ``n_bins`` groups of nearly equal size.
    Reliability, resolution, and uncertainty describe those bins. ``brier``
    is the score of the game probabilities. The binned components are not
    a substitute for that score when probabilities vary inside a bin.

    Args:
        y_true: Home-win labels, 0 or 1.
        y_proba: Predicted home-win probabilities.
        n_bins: Maximum number of equal-count bins. Fewer are used when the
            sample is smaller.
        game_ids: Optional ids used only to break probability ties.

    Returns:
        Dict with ``bins``, ``n``, ``n_bins``, ``ece``, ``reliability``,
        ``resolution``, ``uncertainty``, ``brier``, and ``base_rate``.
        An empty sample leaves the estimates missing.

    Raises:
        ValueError: ``n_bins`` is below 1, or the sample is not a valid
            binary probability sample.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    labels, probabilities = _as_binary_sample(y_true, y_proba)
    empty_bins: list[dict[str, float | int]] = []
    if len(labels) == 0:
        return {
            "bins": empty_bins,
            "n": 0,
            "n_bins": 0,
            "ece": None,
            "reliability": None,
            "resolution": None,
            "uncertainty": None,
            "brier": None,
            "base_rate": None,
        }
    if game_ids is None:
        tie_break: np.ndarray = np.arange(len(labels))
    else:
        if len(game_ids) != len(labels):
            raise ValueError("game_ids must have the same length as y_true")
        tie_break = np.asarray([str(game_id) for game_id in game_ids])
    ordering: pd.DataFrame = pd.DataFrame({"probability": probabilities, "tie_break": tie_break})
    order: np.ndarray = ordering.sort_values(
        ["probability", "tie_break"],
        kind="mergesort",
    ).index.to_numpy()
    bin_count: int = min(int(n_bins), int(len(labels)))
    slices: list[np.ndarray] = [
        chunk for chunk in np.array_split(order, bin_count) if len(chunk) > 0
    ]
    bins: list[dict[str, float | int]] = []
    for bin_number, indexes in enumerate(slices, start=1):
        bins.append(
            {
                "bin": bin_number,
                "n": int(len(indexes)),
                "mean_predicted": float(probabilities[indexes].mean()),
                "observed_rate": float(labels[indexes].mean()),
            }
        )
    sample_size: int = int(len(labels))
    base_rate: float = float(labels.mean())
    reliability: float = 0.0
    resolution: float = 0.0
    calibration_error: float = 0.0
    for item in bins:
        weight: float = float(item["n"]) / float(sample_size)
        gap: float = float(item["mean_predicted"]) - float(item["observed_rate"])
        reliability += weight * gap * gap
        outcome_gap: float = float(item["observed_rate"]) - base_rate
        resolution += weight * outcome_gap * outcome_gap
        calibration_error += weight * abs(gap)
    return {
        "bins": bins,
        "n": sample_size,
        "n_bins": len(bins),
        "ece": calibration_error,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": base_rate * (1.0 - base_rate),
        "brier": float(brier_score_loss(labels, probabilities)),
        "base_rate": base_rate,
    }


def calibration_table_for_results(
    results: list[dict[str, Any]],
    *,
    n_bins: int = N_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Reliability rows for scored probability models, Elo, and the market.

    A skipped model is omitted. The table is informative and is not a gate.

    Args:
        results: Scored CV rows, each with ``oof`` when status is ``ok``.
        n_bins: Maximum equal-count bins per model.

    Returns:
        A dataframe with ``CALIBRATION_COLUMNS``. Bin rows and one summary
        row per scored model.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("model") not in CALIBRATION_MODELS or result.get("status") != "ok":
            continue
        oof: pd.DataFrame | None = result.get("oof")
        if oof is None or len(oof) == 0:
            continue
        summary: dict[str, Any] = reliability_by_decile(
            oof["y_true"].to_numpy(),
            oof["y_proba"].to_numpy(dtype=float),
            n_bins=n_bins,
            game_ids=oof["game_id"].to_numpy(),
        )
        shared: dict[str, Any] = {
            "model": result["model"],
            "feature_set": result["feature_set"],
            "ece": summary["ece"],
            "reliability": summary["reliability"],
            "resolution": summary["resolution"],
            "uncertainty": summary["uncertainty"],
            "brier": summary["brier"],
        }
        for item in summary["bins"]:
            rows.append(
                {
                    **shared,
                    "row_kind": "bin",
                    "bin": item["bin"],
                    "n": item["n"],
                    "mean_predicted": item["mean_predicted"],
                    "observed_rate": item["observed_rate"],
                }
            )
        rows.append(
            {
                **shared,
                "row_kind": "summary",
                "bin": None,
                "n": summary["n"],
                "mean_predicted": None,
                "observed_rate": summary["base_rate"],
            }
        )
    return pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))


def logit_probability(probability: np.ndarray, clip: float = LOGIT_CLIP) -> np.ndarray:
    """Logit of a probability, with 0 and 1 moved inside ``(clip, 1 - clip)``.

    Args:
        probability: Values in ``[0, 1]``.
        clip: Distance kept between a clipped probability and 0 or 1.

    Returns:
        The logit, one value per input.

    Raises:
        ValueError: ``clip`` is outside ``(0, 0.5)``, or a probability is
            not finite and inside ``[0, 1]``.
    """
    if not 0.0 < clip < 0.5:
        raise ValueError("logit clip must be in (0, 0.5)")
    values: np.ndarray = np.asarray(probability, dtype=float)
    if len(values) > 0 and (
        not np.isfinite(values).all() or bool((values < 0.0).any()) or bool((values > 1.0).any())
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    clipped: np.ndarray = np.clip(values, clip, 1.0 - clip)
    return np.log(clipped / (1.0 - clipped))


def _information_coefficients(labels: np.ndarray, design: np.ndarray, seed: int) -> np.ndarray:
    """Intercept, market coefficient, and model coefficient for one sample."""
    estimator: LogisticRegression = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    estimator.fit(design, labels.astype(int))
    coefficients: np.ndarray = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    if len(coefficients) != 2:
        raise ValueError("market information regression must have two coefficients")
    return np.asarray(
        [float(estimator.intercept_[0]), float(coefficients[0]), float(coefficients[1])],
        dtype=float,
    )


def _blank_information(n_rows: int, n_bootstrap: int, reason: str) -> dict[str, Any]:
    """A skipped information row. Coefficients stay missing."""
    return {
        "status": "skipped",
        "reason": reason,
        "n": n_rows,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_used": 0,
        "intercept": None,
        "intercept_ci_low": None,
        "intercept_ci_high": None,
        "market_coef": None,
        "market_ci_low": None,
        "market_ci_high": None,
        "model_coef": None,
        "model_ci_low": None,
        "model_ci_high": None,
    }


def market_information_regression(
    y_true: np.ndarray,
    market_probability: np.ndarray,
    model_probability: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Logistic regression of the result on logit(market) and logit(model).

    The fit is descriptive. It does not return a blended probability, and
    nothing in the dict is a publishing weight. Intervals are percentiles of
    a row bootstrap. A draw with one class is skipped.

    Args:
        y_true: Home-win labels on the shared out-of-fold games.
        market_probability: No-vig market probability on those games.
        model_probability: Model probability on those games.
        n_bootstrap: Bootstrap draws requested. Zero leaves the interval missing.
        seed: Seed for the bootstrap generator and the logistic solver.

    Returns:
        Dict of coefficients, intervals, and sample size. Keys do not include
        a blended probability.

    Raises:
        ValueError: The arrays disagree, a probability is invalid, or
            ``n_bootstrap`` is negative.
    """
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    labels, market = _as_binary_sample(y_true, market_probability)
    _labels, model = _as_binary_sample(y_true, model_probability)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        reason: str = "single_class" if len(labels) >= 2 else "too_few_games"
        return _blank_information(len(labels), n_bootstrap, reason)
    design: np.ndarray = np.column_stack([logit_probability(market), logit_probability(model)])
    point: np.ndarray = _information_coefficients(labels, design, seed)
    draws: list[np.ndarray] = []
    if n_bootstrap > 0 and len(labels) >= 2:
        generator: np.random.Generator = np.random.default_rng(seed)
        attempts: int = 0
        attempt_limit: int = max(n_bootstrap * 5, n_bootstrap)
        while len(draws) < n_bootstrap and attempts < attempt_limit:
            attempts += 1
            drawn: np.ndarray = generator.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[drawn])) < 2:
                continue
            draws.append(_information_coefficients(labels[drawn], design[drawn], seed))
    intervals: list[tuple[float, float] | tuple[None, None]] = []
    if len(draws) >= 2:
        stacked: np.ndarray = np.vstack(draws)
        for column in range(3):
            intervals.append(
                (
                    float(np.percentile(stacked[:, column], 2.5)),
                    float(np.percentile(stacked[:, column], 97.5)),
                )
            )
    else:
        intervals = [(None, None), (None, None), (None, None)]
    payload: dict[str, Any] = {
        "status": "scored",
        "reason": None,
        "n": int(len(labels)),
        "n_bootstrap": int(n_bootstrap),
        "n_bootstrap_used": len(draws),
        "intercept": float(point[0]),
        "intercept_ci_low": intervals[0][0],
        "intercept_ci_high": intervals[0][1],
        "market_coef": float(point[1]),
        "market_ci_low": intervals[1][0],
        "market_ci_high": intervals[1][1],
        "model_coef": float(point[2]),
        "model_ci_low": intervals[2][0],
        "model_ci_high": intervals[2][1],
    }
    return payload


def _market_oof(results: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Return the market out-of-fold table when it scored."""
    for result in results:
        if result.get("model") == "market" and result.get("feature_set") == "market":
            oof: pd.DataFrame | None = result.get("oof")
            if result.get("status") == "ok" and oof is not None and len(oof) > 0:
                return oof
    return None


def market_information_table_for_results(
    results: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """One descriptive regression per scored model, on games shared with the market.

    Logistic rows, the locked booster, and the equal mean are included.
    No blended probability is stored. The mean ensemble is already a
    probability. This regression does not turn it into a weight.

    Args:
        results: Scored CV rows.
        n_bootstrap: Bootstrap draws for each logistic row.
        seed: Bootstrap seed.

    Returns:
        A dataframe with ``MARKET_INFORMATION_COLUMNS``.
    """
    market: pd.DataFrame | None = _market_oof(results)
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("model") not in INFORMATION_MODELS or result.get("status") != "ok":
            continue
        model_oof: pd.DataFrame | None = result.get("oof")
        identity: dict[str, Any] = {
            "model": result["model"],
            "feature_set": result["feature_set"],
        }
        if model_oof is None or len(model_oof) == 0 or market is None:
            blank: dict[str, Any] = _blank_information(0, n_bootstrap, "market_not_scored")
            rows.append({**identity, **blank})
            continue
        left: pd.DataFrame = model_oof.loc[:, ["game_id", "y_true", "y_proba"]].rename(
            columns={"y_proba": "model_proba"}
        )
        right: pd.DataFrame = market.loc[:, ["game_id", "y_true", "y_proba"]].rename(
            columns={"y_true": "market_label", "y_proba": "market_proba"}
        )
        paired: pd.DataFrame = left.merge(right, on="game_id", how="inner")
        if paired.empty:
            blank = _blank_information(0, n_bootstrap, "no_shared_games")
            rows.append({**identity, **blank})
            continue
        if not paired["y_true"].astype(int).equals(paired["market_label"].astype(int)):
            raise ValueError(
                f"{result['feature_set']} labels do not match the market on shared games"
            )
        fitted: dict[str, Any] = market_information_regression(
            paired["y_true"].to_numpy(),
            paired["market_proba"].to_numpy(dtype=float),
            paired["model_proba"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({**identity, **fitted})
    return pd.DataFrame(rows, columns=list(MARKET_INFORMATION_COLUMNS))
