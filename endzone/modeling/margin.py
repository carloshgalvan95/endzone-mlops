"""Point-margin ridge converted to a home win probability, and its logit mean.

``margin_ridge_pkg_core_inj`` fits ``Ridge(alpha=1.0)`` on the home margin
with the defender's columns, the same imputation, and the same scaler.
The home win probability is ``Phi(mu / sigma)``. Sigma is the training
residual standard deviation of that fold. ``margin_mean`` is the equal
logit average of that probability and logistic ``pkg_core_inj_l2``. No
weight is learned. This module does not fit the ADR-007 score models.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from endzone.modeling.cv import FoldPrediction, FoldPredictor, cv_result_record
from endzone.modeling.evaluation import (
    LOGIT_CLIP,
    binary_classification_metrics,
    logit_probability,
)
from endzone.modeling.hypotheses import CvModelSpec
from endzone.modeling.models import (
    MODEL_LOGISTIC,
    MODEL_MARGIN_MEAN,
    MODEL_MARGIN_RIDGE,
    fit_ridge_pipeline,
    model_feature_columns,
)
from endzone.modeling.points_eval import standard_normal_cdf

MARGIN_RIDGE_ALPHA: float = 1.0
MARGIN_MEAN_WEIGHT: float = 0.5
MARGIN_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "row_kind",
    "season",
    "n",
    "sigma",
    "margin_mae",
    "logit_correlation",
)


def home_margins(games: pd.DataFrame) -> pd.Series:
    """Home score minus away score, indexed by ``game_id``.

    Args:
        games: One row per game, with scores.

    Returns:
        Float margins. Ties are 0, the same score difference the mart stores.

    Raises:
        ValueError: A required column is missing, a game id repeats, or a
            score is not numeric.
    """
    required: tuple[str, ...] = ("game_id", "home_score", "away_score")
    missing: list[str] = [column for column in required if column not in games.columns]
    if missing:
        joined: str = ", ".join(missing)
        raise ValueError(f"games are missing margin columns: {joined}")
    frame: pd.DataFrame = games.loc[:, list(required)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    if frame["game_id"].duplicated().any():
        raise ValueError("game_id must be unique")
    home: pd.Series = pd.to_numeric(frame["home_score"], errors="coerce")
    away: pd.Series = pd.to_numeric(frame["away_score"], errors="coerce")
    margin: pd.Series = home - away
    if margin.isna().any() or not np.isfinite(margin.to_numpy(dtype=float)).all():
        raise ValueError("home_score and away_score must be finite numbers")
    return pd.Series(margin.to_numpy(dtype=float), index=frame["game_id"].to_numpy())


def _lookup_margin(frame: pd.DataFrame, margins: pd.Series) -> np.ndarray:
    """Return margins for ``frame`` in row order."""
    game_ids: list[str] = [str(game_id) for game_id in frame["game_id"].tolist()]
    missing: list[str] = [game_id for game_id in game_ids if game_id not in margins.index]
    if missing:
        raise ValueError(f"margin is missing for game_id {missing[0]}")
    return margins.loc[game_ids].to_numpy(dtype=float)


def training_residual_sigma(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Standard deviation of training residuals, with ``ddof=1``.

    Args:
        actual: Training margins.
        predicted: Fitted margins on those same rows.

    Returns:
        A positive finite sigma.

    Raises:
        ValueError: The arrays differ in length, or sigma is not positive.
    """
    observed: np.ndarray = np.asarray(actual, dtype=float)
    fitted: np.ndarray = np.asarray(predicted, dtype=float)
    if len(observed) != len(fitted):
        raise ValueError("actual and predicted margins must have the same length")
    if len(observed) < 2:
        raise ValueError("sigma needs at least two training rows")
    if not np.isfinite(observed).all() or not np.isfinite(fitted).all():
        raise ValueError("training margins must be finite")
    sigma: float = float(np.std(observed - fitted, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("training residual sigma must be positive")
    return sigma


def margin_win_probability(mu: np.ndarray, sigma: float) -> np.ndarray:
    """Map predicted margins to home win probabilities.

    ``p = Phi(mu / sigma)``, then clip to ``[1e-6, 1 - 1e-6]``.

    Args:
        mu: Predicted home margins.
        sigma: Training residual standard deviation. Must be positive.

    Returns:
        One probability per margin.

    Raises:
        ValueError: ``sigma`` is not positive, or a margin is not finite.
    """
    if not math.isfinite(sigma) or float(sigma) <= 0.0:
        raise ValueError("sigma must be positive")
    center: np.ndarray = np.asarray(mu, dtype=float)
    if len(center) > 0 and not np.isfinite(center).all():
        raise ValueError("predicted margins must be finite")
    width: float = float(sigma)
    raw: np.ndarray = np.array(
        [standard_normal_cdf(float(value) / width) for value in center],
        dtype=float,
    )
    return np.clip(raw, LOGIT_CLIP, 1.0 - LOGIT_CLIP)


def logit_mean_probabilities(
    left: np.ndarray,
    right: np.ndarray,
    weight: float = MARGIN_MEAN_WEIGHT,
) -> np.ndarray:
    """Equal logit average of two probability vectors.

    Both inputs are clipped to ``[1e-6, 1 - 1e-6]`` before the logit.
    ``weight`` is the share on ``left``. The preregistered value is 0.5.

    Args:
        left: First probability vector.
        right: Second probability vector, same length.
        weight: Weight on ``left``. The remainder is the weight on ``right``.

    Returns:
        The sigmoid of the weighted logits.

    Raises:
        ValueError: The shapes differ, or ``weight`` is outside ``[0, 1]``.
    """
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("logit weight must be between 0 and 1")
    first: np.ndarray = logit_probability(np.asarray(left, dtype=float))
    second: np.ndarray = logit_probability(np.asarray(right, dtype=float))
    if first.shape != second.shape:
        raise ValueError("logit members must have the same number of probabilities")
    mixed: np.ndarray = (float(weight) * first) + ((1.0 - float(weight)) * second)
    return 1.0 / (1.0 + np.exp(-mixed))


def margin_ridge_fold_predictor(games: pd.DataFrame) -> FoldPredictor:
    """Ridge refit on each fold's training margins.

    The evaluation labels are ignored. Sigma uses the training residuals
    only. The scored season does not enter the fit.

    Args:
        games: Schedule with ``home_score`` and ``away_score``.

    Returns:
        A fold predictor. Extras are ``sigma`` and ``margin_mae``.
    """
    margins: pd.Series = home_margins(games)

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> FoldPrediction:
        del train_target, eval_target
        train_margin: np.ndarray = _lookup_margin(train_frame, margins)
        eval_margin: np.ndarray = _lookup_margin(eval_frame, margins)
        pipeline: Pipeline = fit_ridge_pipeline(
            train_frame,
            pd.Series(train_margin, index=train_frame.index),
            alpha=MARGIN_RIDGE_ALPHA,
        )
        columns: list[str] = model_feature_columns(train_frame)
        train_hat: np.ndarray = np.asarray(pipeline.predict(train_frame[columns]), dtype=float)
        eval_hat: np.ndarray = np.asarray(pipeline.predict(eval_frame[columns]), dtype=float)
        sigma: float = training_residual_sigma(train_margin, train_hat)
        probabilities: np.ndarray = margin_win_probability(eval_hat, sigma)
        margin_mae: float = float(np.mean(np.abs(eval_margin - eval_hat)))
        return FoldPrediction(
            probabilities,
            {"sigma": sigma, "margin_mae": margin_mae},
        )

    return predict


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


def _require_same_games(reference_oof: pd.DataFrame, other_oof: pd.DataFrame) -> pd.DataFrame:
    """Return ``other_oof`` in ``reference_oof`` order.

    Raises:
        ValueError: A game, a season, or a label does not match.
    """
    required: tuple[str, ...] = ("game_id", "season", "y_true", "y_proba")
    for label, frame in (("margin", reference_oof), ("defender", other_oof)):
        missing: list[str] = [column for column in required if column not in frame.columns]
        if missing:
            joined: str = ", ".join(missing)
            raise ValueError(f"{label} OOF is missing columns: {joined}")
    reference_ids: list[str] = [str(game_id) for game_id in reference_oof["game_id"]]
    other: pd.DataFrame = other_oof.copy()
    other["game_id"] = other["game_id"].map(str)
    other = other.drop_duplicates("game_id").set_index("game_id")
    if set(reference_ids) != set(other.index.astype(str)):
        raise ValueError("margin members scored different games")
    ordered: pd.DataFrame = other.loc[reference_ids]
    if [int(season) for season in reference_oof["season"]] != [
        int(season) for season in ordered["season"]
    ]:
        raise ValueError("margin members scored different seasons")
    if [int(label) for label in reference_oof["y_true"]] != [
        int(label) for label in ordered["y_true"]
    ]:
        raise ValueError("margin member labels do not match")
    return ordered


def logit_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Pearson correlation of two probability vectors in logit space.

    Returns:
        The correlation, or None when a vector has no variation.
    """
    first: np.ndarray = logit_probability(np.asarray(left, dtype=float))
    second: np.ndarray = logit_probability(np.asarray(right, dtype=float))
    if len(first) < 2 or float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _mean_params(shared: dict[str, Any], feature_set: str) -> dict[str, Any]:
    """Params that record an equal logit mean and no fitted weight."""
    return {
        **shared,
        "model_type": MODEL_MARGIN_MEAN,
        "feature_set": feature_set,
        "members": (f"{MODEL_MARGIN_RIDGE}:{feature_set},{MODEL_LOGISTIC}:{feature_set}"),
        "weights": f"{MARGIN_MEAN_WEIGHT:g},{1.0 - MARGIN_MEAN_WEIGHT:g}",
        "learned_weights": False,
        "logit_clip": LOGIT_CLIP,
    }


def _folds_from_probabilities(
    member_folds: list[dict[str, Any]],
    averaged: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Per-season metrics of the logit-averaged probabilities."""
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
                f"margin_mean season {fold['season']} has {len(season_rows)} rows, "
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


def margin_mean_result(
    results: list[dict[str, Any]],
    spec: CvModelSpec,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Build ``margin_mean`` from the ridge and the defender.

    Both members must already be scored on ``spec.feature_set``. This
    function does not fit a model and does not learn a weight.

    Args:
        results: Rows scored before the mean, including both members.
        spec: The ``margin_mean`` row. Its feature set selects the members.
        shared: Params stored on every row of this comparison.

    Returns:
        A CV result record. Status is ``skipped`` when a member did not score.
    """
    params: dict[str, Any] = _mean_params(shared, spec.feature_set)
    ridge: dict[str, Any] | None = _member_result(results, MODEL_MARGIN_RIDGE, spec.feature_set)
    defender: dict[str, Any] | None = _member_result(results, MODEL_LOGISTIC, spec.feature_set)
    if (
        ridge is None
        or defender is None
        or ridge.get("status") != "ok"
        or defender.get("status") != "ok"
        or ridge.get("oof") is None
        or defender.get("oof") is None
    ):
        return cv_result_record(
            MODEL_MARGIN_MEAN,
            spec.feature_set,
            "skipped",
            "missing_member",
            params,
            None,
        )
    ordered: pd.DataFrame = _require_same_games(ridge["oof"], defender["oof"])
    averaged: pd.DataFrame = ridge["oof"].loc[:, ["game_id", "season", "y_true", "y_proba"]].copy()
    averaged["y_proba"] = logit_mean_probabilities(
        ridge["oof"]["y_proba"].to_numpy(dtype=float),
        ordered["y_proba"].to_numpy(dtype=float),
        weight=MARGIN_MEAN_WEIGHT,
    )
    folds: list[dict[str, Any]] = _folds_from_probabilities(list(ridge["folds"]), averaged)
    pooled: dict[str, float | int | None] = binary_classification_metrics(
        averaged["y_true"].to_numpy(),
        averaged["y_proba"].to_numpy(dtype=float),
    )
    return cv_result_record(
        MODEL_MARGIN_MEAN,
        spec.feature_set,
        "ok",
        None,
        params,
        {"pooled": pooled, "folds": folds, "oof": averaged},
    )


def margin_diagnostics_table(results: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Per-fold sigma, margin MAE, and the logit correlation with the defender.

    The correlation is one pooled number. It is not an acceptance decision.

    Args:
        results: Scored rows, including the ridge and logistic defender.

    Returns:
        The diagnostics table, or None when the ridge row is absent.
    """
    ridge: dict[str, Any] | None = next(
        (result for result in results if result.get("model") == MODEL_MARGIN_RIDGE),
        None,
    )
    if ridge is None or ridge.get("status") != "ok":
        return None
    rows: list[dict[str, Any]] = []
    weighted_error: float = 0.0
    weighted_n: int = 0
    for fold in ridge.get("folds") or []:
        if fold.get("status") != "scored":
            continue
        n_eval: int = int(fold["n_eval"])
        margin_mae: float = float(fold["margin_mae"])
        rows.append(
            {
                "row_kind": "fold",
                "season": int(fold["season"]),
                "n": n_eval,
                "sigma": float(fold["sigma"]),
                "margin_mae": margin_mae,
                "logit_correlation": None,
            }
        )
        weighted_error += margin_mae * n_eval
        weighted_n += n_eval
    defender: dict[str, Any] | None = _member_result(
        results,
        MODEL_LOGISTIC,
        str(ridge["feature_set"]),
    )
    correlation: float | None = None
    if (
        defender is not None
        and defender.get("status") == "ok"
        and defender.get("oof") is not None
        and ridge.get("oof") is not None
    ):
        ordered = _require_same_games(ridge["oof"], defender["oof"])
        correlation = logit_correlation(
            ridge["oof"]["y_proba"].to_numpy(dtype=float),
            ordered["y_proba"].to_numpy(dtype=float),
        )
    pooled_mae: float | None = None if weighted_n == 0 else weighted_error / weighted_n
    rows.append(
        {
            "row_kind": "summary",
            "season": None,
            "n": weighted_n,
            "sigma": None,
            "margin_mae": pooled_mae,
            "logit_correlation": correlation,
        }
    )
    return pd.DataFrame(rows, columns=list(MARGIN_DIAGNOSTIC_COLUMNS))
