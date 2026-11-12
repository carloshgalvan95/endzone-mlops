"""Fixed histogram gradient boosting and the unweighted mean of two folds.

``gbm_pkg_core_inj`` is ``HistGradientBoostingClassifier`` on the
``pkg_core_inj_l2`` columns. Missing values stay missing. The booster does
not impute them and does not scale them. ``ens_mean`` is the arithmetic mean
of that booster and logistic ``pkg_core_inj_l2`` on the same fold. No weight
is learned, and nothing is fit on out-of-fold predictions.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from endzone.modeling.cv import FoldPredictor, cv_result_record
from endzone.modeling.evaluation import binary_classification_metrics
from endzone.modeling.hypotheses import CvModelSpec
from endzone.modeling.models import MODEL_LOGISTIC, model_feature_columns

MODEL_GBM_PKG_CORE_INJ: str = "gbm_pkg_core_inj"
MODEL_ENS_MEAN: str = "ens_mean"


class HistGradientBoostingParams(NamedTuple):
    """Locked booster settings. These are not chosen on the scored window."""

    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    max_features: float = 1.0
    random_state: int = 42
    early_stopping: bool = False


HIST_GRADIENT_BOOSTING_PARAMS: HistGradientBoostingParams = HistGradientBoostingParams()


def hist_gradient_boosting_param_fields() -> dict[str, Any]:
    """MLflow params for the locked booster, including native missing values."""
    params: HistGradientBoostingParams = HIST_GRADIENT_BOOSTING_PARAMS
    return {
        "learning_rate": params.learning_rate,
        "max_iter": params.max_iter,
        "max_leaf_nodes": params.max_leaf_nodes,
        "min_samples_leaf": params.min_samples_leaf,
        "l2_regularization": params.l2_regularization,
        "max_features": params.max_features,
        "random_state": params.random_state,
        "early_stopping": params.early_stopping,
        "missing_values": "native",
    }


def _numeric_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Model columns as float64. ``NA`` becomes ``NaN`` and is not filled.

    Raises:
        ValueError: The frame has no model columns, or a column is not numeric.
    """
    columns: list[str] = model_feature_columns(frame)
    if not columns:
        raise ValueError("feature matrix has no model columns")
    matrix: pd.DataFrame = frame.loc[:, columns]
    try:
        numeric: pd.DataFrame = matrix.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("booster features must be numeric") from exc
    return numeric


def hist_gradient_boosting_estimator() -> HistGradientBoostingClassifier:
    """Return an unfitted booster with the locked parameters."""
    params: HistGradientBoostingParams = HIST_GRADIENT_BOOSTING_PARAMS
    return HistGradientBoostingClassifier(
        learning_rate=params.learning_rate,
        max_iter=params.max_iter,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        l2_regularization=params.l2_regularization,
        max_features=params.max_features,
        random_state=params.random_state,
        early_stopping=params.early_stopping,
    )


def fit_hist_gradient_boosting(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> HistGradientBoostingClassifier:
    """Fit the locked booster on ``x_train`` only.

    One thread keeps the histogram build deterministic. The scored-season
    frame is not an argument.

    Args:
        x_train: Training rows, including the identifier columns.
        y_train: Labels aligned with ``x_train``.

    Returns:
        The fitted classifier.
    """
    model: HistGradientBoostingClassifier = hist_gradient_boosting_estimator()
    matrix: pd.DataFrame = _numeric_feature_matrix(x_train)
    labels: pd.Series = pd.to_numeric(pd.Series(y_train), errors="coerce")
    if labels.isna().any():
        raise ValueError("booster target must be numeric 0/1 with no nulls")
    with threadpool_limits(limits=1):
        model.fit(matrix, labels.to_numpy())
    return model


def positive_class_probability(
    model: HistGradientBoostingClassifier,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Probability of class 1. Missing features stay missing.

    Raises:
        ValueError: The fitted classes do not include 1.
    """
    classes: list[int] = [int(label) for label in model.classes_]
    if 1 not in classes:
        raise ValueError("booster has no positive class")
    matrix: pd.DataFrame = _numeric_feature_matrix(frame)
    with threadpool_limits(limits=1):
        probabilities: np.ndarray = np.asarray(
            model.predict_proba(matrix)[:, classes.index(1)],
            dtype=float,
        )
    return probabilities


def hist_gbm_fold_predictor() -> FoldPredictor:
    """Booster refit on each fold's training seasons.

    The evaluation labels are ignored. They are not a fit input.
    """

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> np.ndarray:
        del eval_target
        model: HistGradientBoostingClassifier = fit_hist_gradient_boosting(
            train_frame,
            train_target,
        )
        return positive_class_probability(model, eval_frame)

    return predict


def mean_probabilities(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Unweighted mean of two probability vectors.

    Raises:
        ValueError: The shapes differ, or a value is not a finite probability.
    """
    first: np.ndarray = np.asarray(left, dtype=float)
    second: np.ndarray = np.asarray(right, dtype=float)
    if first.shape != second.shape:
        raise ValueError("ensemble members must have the same number of probabilities")
    if len(first) > 0 and (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or bool((first < 0.0).any())
        or bool((first > 1.0).any())
        or bool((second < 0.0).any())
        or bool((second > 1.0).any())
    ):
        raise ValueError("ensemble members must be finite probabilities in [0, 1]")
    return (first + second) / 2.0


def mean_member_oof(logistic_oof: pd.DataFrame, booster_oof: pd.DataFrame) -> pd.DataFrame:
    """Average two out-of-fold tables that scored the same games in the same order.

    Args:
        logistic_oof: Rows from logistic ``pkg_core_inj_l2``.
        booster_oof: Rows from ``gbm_pkg_core_inj`` on the same folds.

    Returns:
        A copy of ``logistic_oof`` whose ``y_proba`` is the mean.

    Raises:
        ValueError: Game ids, seasons, or labels differ.
    """
    required: tuple[str, ...] = ("game_id", "season", "y_true", "y_proba")
    for label, frame in (("logistic", logistic_oof), ("booster", booster_oof)):
        missing: list[str] = [column for column in required if column not in frame.columns]
        if missing:
            joined: str = ", ".join(missing)
            raise ValueError(f"{label} OOF is missing columns: {joined}")
    if [str(game_id) for game_id in logistic_oof["game_id"]] != [
        str(game_id) for game_id in booster_oof["game_id"]
    ]:
        raise ValueError("ensemble members scored different games")
    if [int(season) for season in logistic_oof["season"]] != [
        int(season) for season in booster_oof["season"]
    ]:
        raise ValueError("ensemble members scored different seasons")
    if [int(label) for label in logistic_oof["y_true"]] != [
        int(label) for label in booster_oof["y_true"]
    ]:
        raise ValueError("ensemble member labels do not match")
    averaged: pd.DataFrame = logistic_oof.loc[:, list(required)].copy()
    averaged["y_proba"] = mean_probabilities(
        logistic_oof["y_proba"].to_numpy(dtype=float),
        booster_oof["y_proba"].to_numpy(dtype=float),
    )
    return averaged


def _fold_schedule(fold: dict[str, Any]) -> tuple[Any, ...]:
    """Fields that must match before two folds can be averaged."""
    return (
        int(fold["season"]),
        str(fold["status"]),
        tuple(int(season) for season in fold["train_seasons"]),
        int(fold["n_train"]),
        int(fold["n_eval"]),
        int(fold["eval_season"]),
    )


def _require_same_folds(
    logistic_folds: list[dict[str, Any]],
    booster_folds: list[dict[str, Any]],
) -> None:
    """Refuse a mean when the members did not use the same training seasons."""
    if len(logistic_folds) != len(booster_folds):
        raise ValueError("ensemble members do not have the same folds")
    for logistic_fold, booster_fold in zip(logistic_folds, booster_folds, strict=True):
        if _fold_schedule(logistic_fold) != _fold_schedule(booster_fold):
            raise ValueError(
                "ensemble members used different training seasons on "
                f"season {logistic_fold['season']}"
            )


def _folds_from_mean(
    member_folds: list[dict[str, Any]],
    averaged: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Per-season metrics of the averaged probabilities.

    Training seasons are copied from the member fold. They are not refit.
    """
    folds: list[dict[str, Any]] = []
    for fold in member_folds:
        copied: dict[str, Any] = {
            "season": int(fold["season"]),
            "status": fold["status"],
            "reason": fold["reason"],
            "n_train": int(fold["n_train"]),
            "n_eval": int(fold["n_eval"]),
            "train_seasons": [int(season) for season in fold["train_seasons"]],
            "eval_season": int(fold["eval_season"]),
            "log_loss": None,
            "brier": None,
            "auc": None,
            "accuracy": None,
        }
        if fold["status"] != "scored":
            folds.append(copied)
            continue
        season_rows: pd.DataFrame = averaged.loc[
            averaged["season"].astype(int) == int(fold["season"])
        ]
        if len(season_rows) != int(fold["n_eval"]):
            raise ValueError(
                f"ensemble season {fold['season']} has {len(season_rows)} rows, "
                f"expected {fold['n_eval']}"
            )
        metrics: dict[str, float | int | None] = binary_classification_metrics(
            season_rows["y_true"].to_numpy(),
            season_rows["y_proba"].to_numpy(dtype=float),
        )
        copied["log_loss"] = metrics["log_loss"]
        copied["brier"] = metrics["brier"]
        copied["auc"] = metrics["auc"]
        copied["accuracy"] = metrics["accuracy"]
        folds.append(copied)
    return folds


def _member_result(
    results: list[dict[str, Any]],
    model_name: str,
    feature_set: str,
) -> dict[str, Any] | None:
    """Return the one scored row for this model and feature set."""
    matches: list[dict[str, Any]] = [
        result
        for result in results
        if result.get("model") == model_name and result.get("feature_set") == feature_set
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _ensemble_params(shared: dict[str, Any], feature_set: str) -> dict[str, Any]:
    """Params that record an equal mean and no stacker."""
    return {
        **shared,
        "model_type": MODEL_ENS_MEAN,
        "feature_set": feature_set,
        "members": (f"{MODEL_LOGISTIC}:{feature_set},{MODEL_GBM_PKG_CORE_INJ}:{feature_set}"),
        "weights": "equal",
        "learned_weights": False,
        "stacking": False,
    }


def ensemble_mean_result(
    results: list[dict[str, Any]],
    spec: CvModelSpec,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Build ``ens_mean`` from the two member out-of-fold tables.

    Both members must already be scored on ``spec.feature_set`` with the
    same fold schedule. This function does not fit a model.

    Args:
        results: Rows scored before the ensemble, including both members.
        spec: The ``ens_mean`` row. Its feature set selects the members.
        shared: Params stored on every row of this comparison.

    Returns:
        A CV result record. Status is ``skipped`` when a member did not score.
    """
    params: dict[str, Any] = _ensemble_params(shared, spec.feature_set)
    logistic: dict[str, Any] | None = _member_result(results, MODEL_LOGISTIC, spec.feature_set)
    booster: dict[str, Any] | None = _member_result(
        results,
        MODEL_GBM_PKG_CORE_INJ,
        spec.feature_set,
    )
    if (
        logistic is None
        or booster is None
        or logistic.get("status") != "ok"
        or booster.get("status") != "ok"
        or logistic.get("oof") is None
        or booster.get("oof") is None
    ):
        return cv_result_record(
            MODEL_ENS_MEAN,
            spec.feature_set,
            "skipped",
            "missing_member",
            params,
            None,
        )
    _require_same_folds(list(logistic["folds"]), list(booster["folds"]))
    averaged: pd.DataFrame = mean_member_oof(logistic["oof"], booster["oof"])
    folds: list[dict[str, Any]] = _folds_from_mean(list(logistic["folds"]), averaged)
    pooled: dict[str, float | int | None] = binary_classification_metrics(
        averaged["y_true"].to_numpy(),
        averaged["y_proba"].to_numpy(dtype=float),
    )
    return cv_result_record(
        MODEL_ENS_MEAN,
        spec.feature_set,
        "ok",
        None,
        params,
        {"pooled": pooled, "folds": folds, "oof": averaged},
    )
