"""Rolling-origin folds and the predictors that score one season."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from endzone.features.pregame import elo_home_probabilities
from endzone.modeling.evaluation import binary_classification_metrics
from endzone.modeling.hypotheses import FINAL_VALIDATION_SEASON, HELD_OUT_TEST_SEASON
from endzone.modeling.models import (
    model_feature_columns,
    select_logistic_l2_c,
    train_baseline_home,
    train_logistic_regression,
)

_FOLD_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "season",
        "status",
        "reason",
        "n_train",
        "n_eval",
        "train_seasons",
        "eval_season",
        "log_loss",
        "brier",
        "auc",
        "accuracy",
    }
)


class FoldPrediction(NamedTuple):
    """Probabilities for one scored season, plus fields stored on that fold."""

    probabilities: np.ndarray
    extras: dict[str, Any]


# predict(...) -> probabilities, or probabilities plus fold fields such as chosen_c.
FoldPredictor = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    np.ndarray | FoldPrediction,
]


class RollingOriginConfig(NamedTuple):
    """Season window and bootstrap settings for rolling-origin CV."""

    cv_start_season: int = 2016
    cv_end_season: int = 2023
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42


def make_rolling_origin_config(**overrides: Any) -> RollingOriginConfig:
    """Build a CV config and reject a window that would score 2024 or 2025.

    Raises:
        ValueError: The season window or bootstrap count cannot be run.
        TypeError: An override is not a config field.
    """
    config = RollingOriginConfig(**overrides)
    if config.cv_start_season > config.cv_end_season:
        raise ValueError("cv_start_season must be <= cv_end_season")
    if config.cv_end_season >= FINAL_VALIDATION_SEASON:
        raise ValueError(
            "Rolling-origin CV scores seasons before "
            f"{FINAL_VALIDATION_SEASON}. {FINAL_VALIDATION_SEASON} is the final "
            f"validation season and {HELD_OUT_TEST_SEASON} is not scored."
        )
    if config.n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    return config


def _cv_fold_skip_reason(train_target: pd.Series, eval_target: pd.Series) -> str | None:
    """Return why a CV fold cannot be scored, or None when it can."""
    if len(train_target) == 0:
        return "no_train_rows"
    if len(eval_target) == 0:
        return "no_eval_rows"
    if train_target.nunique() < 2:
        return "single_class_train"
    if eval_target.nunique() < 2:
        return "single_class_eval"
    return None


def _align_cv_frame(
    frame: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reset indexes and require unique game ids and binary labels."""
    if "season" not in frame.columns or "game_id" not in frame.columns:
        raise ValueError("frame must include game_id and season")
    if len(frame) != len(target):
        raise ValueError("frame and target must have the same number of rows")
    if len(frame) == 0:
        raise ValueError("frame must contain at least one row")
    aligned: pd.DataFrame = frame.reset_index(drop=True)
    labels: pd.Series = pd.to_numeric(pd.Series(target).reset_index(drop=True), errors="coerce")
    if labels.isna().any():
        raise ValueError("target must be numeric 0/1 with no nulls")
    if not np.isin(labels.to_numpy(), [0, 1]).all():
        raise ValueError("target must contain only classes 0 and 1")
    if aligned["game_id"].duplicated().any():
        raise ValueError("game_id must be unique")
    return aligned, labels


def _prediction_parts(predicted: np.ndarray | FoldPrediction) -> tuple[np.ndarray, dict[str, Any]]:
    """Split a fold predictor result into probabilities and extra fold fields."""
    if isinstance(predicted, FoldPrediction):
        return predicted.probabilities, dict(predicted.extras)
    return predicted, {}


def _score_season(
    season: int,
    aligned: pd.DataFrame,
    labels: pd.Series,
    predict_fold: FoldPredictor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one season, or record why it was skipped."""
    if season >= FINAL_VALIDATION_SEASON:
        raise ValueError(
            f"Refusing to score season {season}. "
            f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are held out."
        )
    train_mask = aligned["season"] < season
    eval_mask = aligned["season"] == season
    train_frame: pd.DataFrame = aligned.loc[train_mask]
    eval_frame: pd.DataFrame = aligned.loc[eval_mask]
    train_target: pd.Series = labels.loc[train_mask]
    eval_target: pd.Series = labels.loc[eval_mask]
    reason: str | None = _cv_fold_skip_reason(train_target, eval_target)
    fold: dict[str, Any] = {
        "season": season,
        "status": "skipped" if reason is not None else "scored",
        "reason": reason,
        "n_train": int(len(train_frame)),
        "n_eval": int(len(eval_frame)),
        "train_seasons": sorted(int(value) for value in train_frame["season"].unique()),
        "eval_season": season,
        "log_loss": None,
        "brier": None,
        "auc": None,
        "accuracy": None,
    }
    if reason is not None:
        return fold, []
    probability_values, extras = _prediction_parts(
        predict_fold(train_frame, train_target, eval_frame, eval_target)
    )
    overlap: set[str] = set(extras).intersection(_FOLD_RESERVED_KEYS)
    if overlap:
        overlap_list: str = ", ".join(sorted(overlap))
        raise ValueError(f"fold extras overwrite reserved fields: {overlap_list}")
    fold.update(extras)
    probabilities = np.asarray(probability_values, dtype=float)
    if len(probabilities) != len(eval_frame):
        raise ValueError(
            f"predict_fold returned {len(probabilities)} probabilities for {len(eval_frame)} rows"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite probabilities for season {season}")
    fold_metrics: dict[str, float | int | None] = binary_classification_metrics(
        eval_target.to_numpy(),
        probabilities,
    )
    fold["log_loss"] = fold_metrics["log_loss"]
    fold["brier"] = fold_metrics["brier"]
    fold["auc"] = fold_metrics["auc"]
    fold["accuracy"] = fold_metrics["accuracy"]
    oof_rows: list[dict[str, Any]] = []
    for game_id, label, probability in zip(
        eval_frame["game_id"].tolist(),
        eval_target.tolist(),
        probabilities.tolist(),
        strict=True,
    ):
        oof_rows.append(
            {
                "game_id": game_id,
                "season": season,
                "y_true": int(label),
                "y_proba": float(probability),
            }
        )
    return fold, oof_rows


def rolling_origin_cv(
    frame: pd.DataFrame,
    target: pd.Series,
    predict_fold: FoldPredictor,
    config: RollingOriginConfig | None = None,
) -> dict[str, Any]:
    """Train on seasons before S and score S, for each S in the CV window.

    2024 and 2025 are never scored, even when they are rows in ``frame``.
    Folds with no rows or a single class are recorded and skipped. Pooled
    metrics use only the scored out-of-fold rows. The bootstrap fields are
    stored for the caller. This function does not resample.

    Returns:
        Dict with ``pooled``, ``folds``, ``oof``, and ``config``.
    """
    resolved: RollingOriginConfig = make_rolling_origin_config(
        **(config._asdict() if config is not None else {})
    )
    aligned, labels = _align_cv_frame(frame, target)
    folds: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for season in range(resolved.cv_start_season, resolved.cv_end_season + 1):
        fold, season_rows = _score_season(season, aligned, labels, predict_fold)
        folds.append(fold)
        oof_rows.extend(season_rows)
    oof: pd.DataFrame = pd.DataFrame(
        oof_rows,
        columns=["game_id", "season", "y_true", "y_proba"],
    )
    if not oof.empty and bool((oof["season"] >= FINAL_VALIDATION_SEASON).any()):
        raise ValueError("CV scored a final-validation or held-out test season")
    if oof.empty:
        pooled: dict[str, float | int | None] = binary_classification_metrics(
            np.array([]),
            np.array([]),
        )
    else:
        pooled = binary_classification_metrics(
            oof["y_true"].to_numpy(),
            oof["y_proba"].to_numpy(dtype=float),
        )
    return {
        "pooled": pooled,
        "folds": folds,
        "oof": oof,
        "config": resolved._asdict(),
    }


def cv_result_record(
    model_name: str,
    feature_set: str,
    status: str,
    reason: str | None,
    params: dict[str, Any],
    cv_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bundle one model's CV output before the reference comparison is attached."""
    scored: bool = cv_result is not None and not cv_result["oof"].empty
    if not scored:
        skip_reason: str | None = reason or "no_scored_folds"
        return {
            "model": model_name,
            "feature_set": feature_set,
            "status": "skipped",
            "reason": skip_reason,
            "params": params,
            "pooled": None,
            "folds": [] if cv_result is None else cv_result["folds"],
            "oof": None,
            "comparison": None,
        }
    return {
        "model": model_name,
        "feature_set": feature_set,
        "status": status,
        "reason": reason,
        "params": params,
        "pooled": cv_result["pooled"],
        "folds": cv_result["folds"],
        "oof": cv_result["oof"],
        "comparison": None,
    }


def _model_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Probability of class 1 from a fitted baseline or training pipeline."""
    feature_columns: list[str] = model_feature_columns(frame)
    if hasattr(model, "named_steps"):
        classes = list(model.named_steps["classifier"].classes_)
    else:
        classes = list(model.classes_)
    positive_index: int = classes.index(1)
    return np.asarray(model.predict_proba(frame[feature_columns])[:, positive_index], dtype=float)


def _baseline_fold_predictor() -> FoldPredictor:
    """Home-prior probability refit on each fold's training seasons."""

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> np.ndarray:
        model, _metrics = train_baseline_home(train_frame, train_target, eval_frame, eval_target)
        return _model_probability(model, eval_frame)

    return predict


def _logistic_fold_predictor(c_value: float, max_iter: int, random_state: int) -> FoldPredictor:
    """Logistic pipeline refit on each fold's training seasons."""

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> np.ndarray:
        model, _metrics = train_logistic_regression(
            train_frame,
            train_target,
            eval_frame,
            eval_target,
            C=c_value,
            max_iter=max_iter,
            random_state=random_state,
        )
        return _model_probability(model, eval_frame)

    return predict


def _logistic_l2_fold_predictor(
    c_grid: tuple[float, ...],
    fallback_c: float,
    max_iter: int,
    random_state: int,
) -> FoldPredictor:
    """L2 logistic whose C is chosen on the training seasons of this fold.

    The scored season is used only after C is fixed. ``chosen_c`` and
    ``c_selection`` are stored on the fold.
    """

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> FoldPrediction:
        selection: dict[str, Any] = select_logistic_l2_c(
            train_frame,
            train_target,
            c_grid=c_grid,
            fallback_c=fallback_c,
            max_iter=max_iter,
            random_state=random_state,
        )
        model, _metrics = train_logistic_regression(
            train_frame,
            train_target,
            eval_frame,
            eval_target,
            C=float(selection["chosen_c"]),
            max_iter=max_iter,
            random_state=random_state,
        )
        return FoldPrediction(
            probabilities=_model_probability(model, eval_frame),
            extras={
                "chosen_c": float(selection["chosen_c"]),
                "c_selection": str(selection["c_selection"]),
            },
        )

    return predict


def _elo_fold_predictor(games: pd.DataFrame) -> FoldPredictor:
    """Elo probabilities from the fold's prior games plus the scored season.

    Games outside the fold are not passed in, so a later season cannot move
    the rating used for an earlier season.
    """

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> np.ndarray:
        del train_target, eval_target
        fold_ids: set[object] = set(train_frame["game_id"]).union(set(eval_frame["game_id"]))
        subset: pd.DataFrame = games.loc[games["game_id"].isin(fold_ids)]
        fitted: dict[str, Any] = elo_home_probabilities(subset)
        mapping: pd.Series = pd.Series(
            fitted["probabilities"].to_numpy(),
            index=subset["game_id"].to_numpy(),
        )
        return mapping.loc[eval_frame["game_id"].to_numpy()].to_numpy(dtype=float)

    return predict


def _lookup_fold_predictor(probabilities_by_game: pd.Series) -> FoldPredictor:
    """Return stored probabilities for the evaluation game ids."""

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> np.ndarray:
        del train_frame, train_target, eval_target
        return probabilities_by_game.loc[eval_frame["game_id"].to_numpy()].to_numpy(dtype=float)

    return predict
