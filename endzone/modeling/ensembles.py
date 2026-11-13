"""Fixed histogram gradient boosting, means, and the defender-plus-Elo stacker.

``gbm_pkg_core_inj`` is ``HistGradientBoostingClassifier`` on the
``pkg_core_inj_l2`` columns. Missing values stay missing. The booster does
not impute them and does not scale them. ``ens_mean`` is the arithmetic mean
of that booster and logistic ``pkg_core_inj_l2`` on the same fold. No weight
is learned, and nothing is fit on out-of-fold predictions.

``ens_mean_elo`` is the arithmetic mean of logistic ``pkg_core_inj_l2`` and
Elo. ``stack_elo`` fits a logistic regression on the logits of those two
probabilities. For outer season S the stacker sees only inner out-of-fold
rows from 2010 through S-1. It does not see season S.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from endzone.features.pregame import prepare_training_data
from endzone.modeling.cv import (
    FoldPredictor,
    _cv_fold_skip_reason,
    _elo_fold_predictor,
    _logistic_fold_predictor,
    cv_result_record,
)
from endzone.modeling.evaluation import (
    LOGIT_CLIP,
    binary_classification_metrics,
    logit_probability,
)
from endzone.modeling.hypotheses import CvModelSpec
from endzone.modeling.models import (
    MODEL_GROUP_MEAN,
    MODEL_GROUP_MEAN_DEF,
    MODEL_LOGISTIC,
    MODEL_MARGIN_MEAN,
    model_feature_columns,
)

MODEL_GBM_PKG_CORE_INJ: str = "gbm_pkg_core_inj"
MODEL_ENS_MEAN: str = "ens_mean"
MODEL_ENS_MEAN_ELO: str = "ens_mean_elo"
MODEL_STACK_ELO: str = "stack_elo"
MODEL_ELO: str = "elo"
# Derived rows are means or a stacker. They are not fit inside ``_score_model_spec``.
DERIVED_CV_MODELS: frozenset[str] = frozenset(
    {
        MODEL_ENS_MEAN,
        MODEL_ENS_MEAN_ELO,
        MODEL_STACK_ELO,
        MODEL_MARGIN_MEAN,
        MODEL_GROUP_MEAN,
        MODEL_GROUP_MEAN_DEF,
    }
)
# Inner out-of-fold seasons for the stacker start here. This is not a search.
STACK_INNER_START_SEASON: int = 2010
STACKER_C: float = 1.0
STACKER_MAX_ITER: int = 1000
STACKER_SEED: int = 42
STACKER_COLUMNS: tuple[str, ...] = (
    "season",
    "status",
    "reason",
    "n_train",
    "inner_seasons",
    "intercept",
    "coef_logit_pkg_core_inj_l2",
    "coef_logit_elo",
)


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


def _oof_frame(result: dict[str, Any] | None) -> pd.DataFrame | None:
    """Return a scored out-of-fold table, or None."""
    if result is None or result.get("status") != "ok":
        return None
    oof: pd.DataFrame | None = result.get("oof")
    if oof is None or len(oof) == 0:
        return None
    return oof


def mean_aligned_oof(reference_oof: pd.DataFrame, other_oof: pd.DataFrame) -> pd.DataFrame:
    """Average two probability tables on ``game_id``, in the reference order.

    The members do not have to be in the same row order. They do have to
    score the same games and the same labels.

    Raises:
        ValueError: A game, a season, or a label does not match.
    """
    required: tuple[str, ...] = ("game_id", "season", "y_true", "y_proba")
    for label, frame in (("reference", reference_oof), ("other", other_oof)):
        missing: list[str] = [column for column in required if column not in frame.columns]
        if missing:
            joined: str = ", ".join(missing)
            raise ValueError(f"{label} OOF is missing columns: {joined}")
    reference_ids: list[str] = [str(game_id) for game_id in reference_oof["game_id"]]
    other_ids: set[str] = {str(game_id) for game_id in other_oof["game_id"]}
    if set(reference_ids) != other_ids:
        raise ValueError("ensemble members scored different games")
    other_by_id: pd.DataFrame = other_oof.copy()
    other_by_id["game_id"] = other_by_id["game_id"].map(str)
    other_by_id = other_by_id.drop_duplicates("game_id").set_index("game_id")
    ordered: pd.DataFrame = other_by_id.loc[reference_ids]
    if [int(season) for season in reference_oof["season"]] != [
        int(season) for season in ordered["season"]
    ]:
        raise ValueError("ensemble members scored different seasons")
    if [int(label) for label in reference_oof["y_true"]] != [
        int(label) for label in ordered["y_true"]
    ]:
        raise ValueError("ensemble member labels do not match")
    averaged: pd.DataFrame = reference_oof.loc[:, list(required)].copy()
    averaged["y_proba"] = mean_probabilities(
        reference_oof["y_proba"].to_numpy(dtype=float),
        ordered["y_proba"].to_numpy(dtype=float),
    )
    return averaged


def elo_mean_result(
    results: list[dict[str, Any]],
    spec: CvModelSpec,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Unweighted mean of logistic ``spec.feature_set`` and Elo on each fold.

    Both probabilities are the ones already scored for that fold. This
    function does not fit a model and does not learn a weight.

    Args:
        results: Scored rows, including the logistic member and Elo.
        spec: The ``ens_mean_elo`` row.
        shared: Params stored on every row of this comparison.

    Returns:
        A CV result record. Status is ``skipped`` when a member did not score.
    """
    params: dict[str, Any] = {
        **shared,
        "model_type": MODEL_ENS_MEAN_ELO,
        "feature_set": spec.feature_set,
        "members": f"{MODEL_LOGISTIC}:{spec.feature_set},{MODEL_ELO}:elo",
        "weights": "equal",
        "learned_weights": False,
        "stacking": False,
    }
    logistic: dict[str, Any] | None = _member_result(results, MODEL_LOGISTIC, spec.feature_set)
    elo: dict[str, Any] | None = _member_result(results, MODEL_ELO, MODEL_ELO)
    logistic_oof: pd.DataFrame | None = _oof_frame(logistic)
    elo_oof: pd.DataFrame | None = _oof_frame(elo)
    if logistic is None or logistic_oof is None or elo_oof is None:
        return cv_result_record(
            MODEL_ENS_MEAN_ELO,
            spec.feature_set,
            "skipped",
            "missing_member",
            params,
            None,
        )
    averaged: pd.DataFrame = mean_aligned_oof(logistic_oof, elo_oof)
    folds: list[dict[str, Any]] = _folds_from_mean(list(logistic["folds"]), averaged)
    pooled: dict[str, float | int | None] = binary_classification_metrics(
        averaged["y_true"].to_numpy(),
        averaged["y_proba"].to_numpy(dtype=float),
    )
    return cv_result_record(
        MODEL_ENS_MEAN_ELO,
        spec.feature_set,
        "ok",
        None,
        params,
        {"pooled": pooled, "folds": folds, "oof": averaged},
    )


def _member_rows_from_outer(
    logistic_oof: pd.DataFrame,
    elo_oof: pd.DataFrame,
    season: int,
) -> pd.DataFrame | None:
    """Published outer probabilities for one season, or None when it was not scored."""
    logistic_rows: pd.DataFrame = logistic_oof.loc[logistic_oof["season"].astype(int) == season]
    if logistic_rows.empty:
        return None
    elo_rows: pd.DataFrame = elo_oof.loc[elo_oof["season"].astype(int) == season].copy()
    elo_rows["game_id"] = elo_rows["game_id"].map(str)
    elo_by_id: pd.DataFrame = elo_rows.drop_duplicates("game_id").set_index("game_id")
    game_ids: list[str] = [str(game_id) for game_id in logistic_rows["game_id"]]
    if any(game_id not in elo_by_id.index for game_id in game_ids):
        raise ValueError(f"Elo did not score every logistic game in season {season}")
    ordered: pd.DataFrame = elo_by_id.loc[game_ids]
    if [int(label) for label in logistic_rows["y_true"]] != [
        int(label) for label in ordered["y_true"]
    ]:
        raise ValueError(f"Elo labels do not match logistic labels in season {season}")
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": season,
            "y_true": [int(label) for label in logistic_rows["y_true"]],
            "p_logistic": logistic_rows["y_proba"].to_numpy(dtype=float),
            "p_elo": ordered["y_proba"].to_numpy(dtype=float),
        }
    )


def _fit_member_season(
    matrix: pd.DataFrame,
    labels: pd.Series,
    games: pd.DataFrame,
    season: int,
    c_value: float,
    max_iter: int,
    random_state: int,
) -> pd.DataFrame | None:
    """Train both members on seasons before ``season`` and predict that season.

    Returns:
        One row per scored game, or None when the fold cannot be scored.
        The returned seasons are only ``season``.
    """
    before: pd.Series = matrix["season"].astype(int) < season
    during: pd.Series = matrix["season"].astype(int) == season
    train_frame: pd.DataFrame = matrix.loc[before].reset_index(drop=True)
    eval_frame: pd.DataFrame = matrix.loc[during].reset_index(drop=True)
    train_target: pd.Series = labels.loc[before].reset_index(drop=True)
    eval_target: pd.Series = labels.loc[during].reset_index(drop=True)
    if _cv_fold_skip_reason(train_target, eval_target) is not None:
        return None
    logistic_probability: np.ndarray = np.asarray(
        _logistic_fold_predictor(c_value, max_iter, random_state)(
            train_frame,
            train_target,
            eval_frame,
            eval_target,
        ),
        dtype=float,
    )
    elo_probability: np.ndarray = np.asarray(
        _elo_fold_predictor(games)(train_frame, train_target, eval_frame, eval_target),
        dtype=float,
    )
    return pd.DataFrame(
        {
            "game_id": [str(game_id) for game_id in eval_frame["game_id"]],
            "season": season,
            "y_true": [int(label) for label in eval_target],
            "p_logistic": logistic_probability,
            "p_elo": elo_probability,
        }
    )


def collect_inner_member_rows(
    matrix: pd.DataFrame,
    labels: pd.Series,
    games: pd.DataFrame,
    logistic_oof: pd.DataFrame,
    elo_oof: pd.DataFrame,
    *,
    through_season: int,
    c_value: float,
    max_iter: int,
    random_state: int,
) -> pd.DataFrame:
    """Inner member predictions for seasons ``2010`` through ``through_season``.

    A season already present in both outer tables is reused. That outer row
    was itself trained on seasons strictly before it. Earlier seasons are fit
    here on seasons strictly before the inner season.

    Args:
        matrix: Defender matrix, including seasons before the outer window.
        labels: Labels aligned with ``matrix``.
        games: Schedule rows used by Elo. Later seasons may be present.
            The Elo fold keeps only the inner training games and that season.
        logistic_oof: Outer defender probabilities.
        elo_oof: Outer Elo probabilities.
        through_season: Last inner season to keep. The outer scored season
            is not included.
        c_value: C for the defender refits.
        max_iter: Solver cap for those refits.
        random_state: Seed for those refits.

    Returns:
        Columns ``game_id``, ``season``, ``y_true``, ``p_logistic``, ``p_elo``.
        Empty when no inner season can be scored.
    """
    present: list[int] = sorted(int(season) for season in pd.unique(matrix["season"]))
    wanted: list[int] = [
        season for season in present if STACK_INNER_START_SEASON <= season <= int(through_season)
    ]
    frames: list[pd.DataFrame] = []
    for season in wanted:
        published: pd.DataFrame | None = _member_rows_from_outer(logistic_oof, elo_oof, season)
        if published is not None:
            frames.append(published)
            continue
        fitted: pd.DataFrame | None = _fit_member_season(
            matrix,
            labels,
            games,
            season,
            c_value,
            max_iter,
            random_state,
        )
        if fitted is not None:
            frames.append(fitted)
    if not frames:
        return pd.DataFrame(columns=["game_id", "season", "y_true", "p_logistic", "p_elo"])
    return pd.concat(frames, ignore_index=True)


def stacker_training_rows(inner_rows: pd.DataFrame, season_s: int) -> pd.DataFrame:
    """Rows the stacker may see for outer season ``season_s``.

    The season column is the inner season T. T runs from 2010 through
    ``season_s - 1``. Season ``season_s`` and every later season are removed.

    Raises:
        ValueError: The filter still contains the scored season or a later one.
    """
    if inner_rows.empty:
        return inner_rows.copy()
    seasons: pd.Series = inner_rows["season"].astype(int)
    kept: pd.DataFrame = inner_rows.loc[
        (seasons >= STACK_INNER_START_SEASON) & (seasons < int(season_s))
    ].copy()
    if not kept.empty and int(kept["season"].max()) >= int(season_s):
        raise ValueError(f"stacker training includes season {season_s} or later")
    return kept.reset_index(drop=True)


def _stacker_design(p_logistic: np.ndarray, p_elo: np.ndarray) -> np.ndarray:
    """Two logit columns. Probabilities are clipped before the logit."""
    return np.column_stack(
        [
            logit_probability(np.asarray(p_logistic, dtype=float), clip=LOGIT_CLIP),
            logit_probability(np.asarray(p_elo, dtype=float), clip=LOGIT_CLIP),
        ]
    )


def fit_probability_stacker(
    seasons: np.ndarray,
    y_true: np.ndarray,
    p_logistic: np.ndarray,
    p_elo: np.ndarray,
) -> LogisticRegression:
    """Fit the locked stacker. ``seasons`` is checked by the caller.

    The design is an intercept plus ``logit(p_pkg_core_inj_l2)`` and
    ``logit(p_elo)``. C is 1.0. Nothing in this function selects C.

    Args:
        seasons: Inner season of each training row. Not a model column.
        y_true: Labels for those inner rows.
        p_logistic: Defender probabilities for those rows.
        p_elo: Elo probabilities for those rows.

    Returns:
        The fitted logistic regression.

    Raises:
        ValueError: The season, label, and probability columns differ in length.
    """
    if len(seasons) != len(y_true) or len(seasons) != len(p_logistic) or len(seasons) != len(p_elo):
        raise ValueError("stacker training columns have different lengths")
    model: LogisticRegression = LogisticRegression(
        C=STACKER_C,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=STACKER_MAX_ITER,
        random_state=STACKER_SEED,
    )
    model.fit(_stacker_design(p_logistic, p_elo), np.asarray(y_true, dtype=int))
    return model


def stacker_probabilities(
    model: LogisticRegression,
    p_logistic: np.ndarray,
    p_elo: np.ndarray,
) -> np.ndarray:
    """Probability of class 1 from a fitted stacker."""
    classes: list[int] = [int(label) for label in model.classes_]
    if 1 not in classes:
        raise ValueError("stacker has no positive class")
    design: np.ndarray = _stacker_design(p_logistic, p_elo)
    return np.asarray(model.predict_proba(design)[:, classes.index(1)], dtype=float)


def _blank_stacker_row(season: int, reason: str) -> dict[str, Any]:
    """One descriptive row when the stacker was not fit for ``season``."""
    return {
        "season": season,
        "status": "skipped",
        "reason": reason,
        "n_train": 0,
        "inner_seasons": "",
        "intercept": None,
        "coef_logit_pkg_core_inj_l2": None,
        "coef_logit_elo": None,
    }


def _scored_stacker_row(
    season: int,
    train_rows: pd.DataFrame,
    model: LogisticRegression,
) -> dict[str, Any]:
    """Intercept and the two logit coefficients for one outer season."""
    coefficients: np.ndarray = np.asarray(model.coef_, dtype=float).reshape(-1)
    inner: list[int] = sorted(int(value) for value in pd.unique(train_rows["season"]))
    return {
        "season": season,
        "status": "scored",
        "reason": None,
        "n_train": int(len(train_rows)),
        "inner_seasons": ",".join(str(value) for value in inner),
        "intercept": float(model.intercept_[0]),
        "coef_logit_pkg_core_inj_l2": float(coefficients[0]),
        "coef_logit_elo": float(coefficients[1]),
    }


def stack_elo_result(
    games: pd.DataFrame,
    features: pd.DataFrame,
    results: list[dict[str, Any]],
    spec: CvModelSpec,
    shared: dict[str, Any],
    *,
    cv_start_season: int,
    cv_end_season: int,
    logistic_max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    """Score ``stack_elo`` without letting the stacker see the outer season.

    For outer season S, members for each inner season T in 2010..S-1 are
    trained on seasons before T. The stacker is fit on those inner rows
    only, then applied to the outer probabilities for S.

    Args:
        games: Schedule, including seasons before the outer window.
        features: Pregame features for the defender matrix.
        results: Scored rows, including the defender and Elo.
        spec: The ``stack_elo`` row. Its feature set is the defender set.
        shared: Params stored on every row of this comparison.
        cv_start_season: First outer season.
        cv_end_season: Last outer season.
        logistic_max_iter: Solver cap for defender refits before the window.
        random_state: Seed for those refits. The stacker seed stays 42.

    Returns:
        A CV result record. ``stacker_coefficients`` is a descriptive table.
    """
    params: dict[str, Any] = {
        **shared,
        "model_type": MODEL_STACK_ELO,
        "feature_set": spec.feature_set,
        "members": f"{MODEL_LOGISTIC}:{spec.feature_set},{MODEL_ELO}:elo",
        "C": STACKER_C,
        "fit_intercept": True,
        "logit_clip": LOGIT_CLIP,
        "inner_start_season": STACK_INNER_START_SEASON,
        "max_iter": STACKER_MAX_ITER,
        "stacker_seed": STACKER_SEED,
        "learned_weights": True,
        "weights_fit_on": "inner_oof_before_scored_season",
    }
    logistic: dict[str, Any] | None = _member_result(results, MODEL_LOGISTIC, spec.feature_set)
    elo: dict[str, Any] | None = _member_result(results, MODEL_ELO, MODEL_ELO)
    logistic_oof: pd.DataFrame | None = _oof_frame(logistic)
    elo_oof: pd.DataFrame | None = _oof_frame(elo)
    empty_table: pd.DataFrame = pd.DataFrame(columns=list(STACKER_COLUMNS))
    if logistic is None or logistic_oof is None or elo_oof is None:
        skipped: dict[str, Any] = cv_result_record(
            MODEL_STACK_ELO,
            spec.feature_set,
            "skipped",
            "missing_member",
            params,
            None,
        )
        skipped["stacker_coefficients"] = empty_table
        return skipped
    defender_c: object = (logistic.get("params") or {}).get("C")
    if defender_c is None:
        raise ValueError("pkg_core_inj_l2 did not record C")
    matrix: pd.DataFrame
    labels: pd.Series
    matrix, labels = prepare_training_data(games, features, feature_set=spec.feature_set)
    matrix = matrix.reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    inner_rows: pd.DataFrame = collect_inner_member_rows(
        matrix,
        labels,
        games,
        logistic_oof,
        elo_oof,
        through_season=int(cv_end_season) - 1,
        c_value=float(defender_c),
        max_iter=logistic_max_iter,
        random_state=random_state,
    )
    oof_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for season in range(int(cv_start_season), int(cv_end_season) + 1):
        outer: pd.DataFrame | None = _member_rows_from_outer(logistic_oof, elo_oof, season)
        train_rows: pd.DataFrame = stacker_training_rows(inner_rows, season)
        if outer is None:
            fold_rows.append(_blank_outer_fold(season, "no_eval_rows"))
            coefficient_rows.append(_blank_stacker_row(season, "no_eval_rows"))
            continue
        if train_rows.empty or int(pd.Series(train_rows["y_true"]).nunique()) < 2:
            reason: str = "no_inner_oof" if train_rows.empty else "single_class_inner"
            fold_rows.append(_blank_outer_fold(season, reason))
            coefficient_rows.append(_blank_stacker_row(season, reason))
            continue
        model: LogisticRegression = fit_probability_stacker(
            train_rows["season"].to_numpy(),
            train_rows["y_true"].to_numpy(),
            train_rows["p_logistic"].to_numpy(dtype=float),
            train_rows["p_elo"].to_numpy(dtype=float),
        )
        probabilities: np.ndarray = stacker_probabilities(
            model,
            outer["p_logistic"].to_numpy(dtype=float),
            outer["p_elo"].to_numpy(dtype=float),
        )
        predicted: pd.DataFrame = outer.loc[:, ["game_id", "season", "y_true"]].copy()
        predicted["y_proba"] = probabilities
        oof_frames.append(predicted)
        metrics: dict[str, float | int | None] = binary_classification_metrics(
            predicted["y_true"].to_numpy(),
            probabilities,
        )
        inner_seasons: list[int] = sorted(int(value) for value in pd.unique(train_rows["season"]))
        fold_rows.append(
            {
                "season": season,
                "status": "scored",
                "reason": None,
                "n_train": int(len(train_rows)),
                "n_eval": int(len(predicted)),
                "train_seasons": inner_seasons,
                "eval_season": season,
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
            }
        )
        coefficient_rows.append(_scored_stacker_row(season, train_rows, model))
    coefficients: pd.DataFrame = pd.DataFrame(coefficient_rows, columns=list(STACKER_COLUMNS))
    if not oof_frames:
        skipped = cv_result_record(
            MODEL_STACK_ELO,
            spec.feature_set,
            "skipped",
            "no_scored_folds",
            params,
            {"pooled": None, "folds": fold_rows, "oof": pd.DataFrame()},
        )
        skipped["stacker_coefficients"] = coefficients
        return skipped
    oof: pd.DataFrame = pd.concat(oof_frames, ignore_index=True)
    pooled: dict[str, float | int | None] = binary_classification_metrics(
        oof["y_true"].to_numpy(),
        oof["y_proba"].to_numpy(dtype=float),
    )
    record: dict[str, Any] = cv_result_record(
        MODEL_STACK_ELO,
        spec.feature_set,
        "ok",
        None,
        params,
        {"pooled": pooled, "folds": fold_rows, "oof": oof},
    )
    record["stacker_coefficients"] = coefficients
    return record


def _blank_outer_fold(season: int, reason: str) -> dict[str, Any]:
    """A fold row for an outer season the stacker did not score."""
    return {
        "season": season,
        "status": "skipped",
        "reason": reason,
        "n_train": 0,
        "n_eval": 0,
        "train_seasons": [],
        "eval_season": season,
        "log_loss": None,
        "brier": None,
        "auc": None,
        "accuracy": None,
    }
