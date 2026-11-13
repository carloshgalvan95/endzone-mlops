"""Four disjoint logistics on the defender columns, and two logit averages.

``epa_form``, ``qb``, ``ratings``, and ``injuries_rest`` partition
``pkg_core_inj_l2``. Each group is logistic regression with C fixed at
0.1, the defender imputation groups, and a ``StandardScaler``. The fit
uses only that fold's training seasons. ``group_mean`` is the equal
logit average of the four. ``group_mean_def`` is the equal logit average
of ``group_mean`` and logistic ``pkg_core_inj_l2``. No weight is learned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from endzone.features.injuries import INJURY_DIFF_COLUMNS
from endzone.features.opponent import ELO_MOV_DIFF_COLUMN, H6_ADJ_FEATURE_COLUMNS
from endzone.features.pregame import (
    FEATURE_SET_PKG_CORE_INJ_L2,
    PREGAME_FEATURE_COLUMNS_H2_FULL,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
    QB_RATING_COLUMNS,
    prepare_training_data,
)
from endzone.features.recency import RECENCY_POINT_DIFF_COLUMNS
from endzone.modeling.cv import (
    FoldPredictor,
    RollingOriginConfig,
    cv_result_record,
    rolling_origin_cv,
)
from endzone.modeling.evaluation import (
    LOGIT_CLIP,
    binary_classification_metrics,
    logit_probability,
)
from endzone.modeling.hypotheses import CvModelSpec
from endzone.modeling.models import (
    IDENTIFIER_COLUMNS,
    MODEL_GROUP_MEAN,
    MODEL_GROUP_MEAN_DEF,
    MODEL_LOGISTIC,
    fit_logistic_pipeline,
    model_feature_columns,
)

GROUP_LOGISTIC_C: float = 0.1
GROUP_NAMES: tuple[str, ...] = ("epa_form", "qb", "ratings", "injuries_rest")
WIN_RATE_COLUMNS: tuple[str, ...] = ("home_recent_win_rate", "away_recent_win_rate")
REST_COLUMNS: tuple[str, ...] = ("home_rest_days", "away_rest_days")
GROUP_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "row_kind",
    "group",
    "n",
    "log_loss",
    "logit_correlation",
)


def _unadjusted_h2_epa_columns() -> tuple[str, ...]:
    """The sixteen raw EPA columns inside ``h2_full``."""
    return tuple(column for column in PREGAME_FEATURE_COLUMNS_H2_FULL if "epa" in column)


def _declared_groups() -> dict[str, tuple[str, ...]]:
    """The preregistered column lists, before the partition check."""
    return {
        "epa_form": (
            H6_ADJ_FEATURE_COLUMNS
            + _unadjusted_h2_epa_columns()
            + WIN_RATE_COLUMNS
            + RECENCY_POINT_DIFF_COLUMNS
        ),
        "qb": QB_RATING_COLUMNS,
        "ratings": (ELO_MOV_DIFF_COLUMN,),
        "injuries_rest": INJURY_DIFF_COLUMNS + REST_COLUMNS,
    }


def verify_group_partition() -> dict[str, tuple[str, ...]]:
    """Require the four groups to be a disjoint partition of the defender.

    Returns:
        Group name to columns, in preregistered order.

    Raises:
        ValueError: A group has the wrong size, a column repeats, or the
            union is not ``PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ``.
    """
    groups: dict[str, tuple[str, ...]] = _declared_groups()
    if tuple(groups) != GROUP_NAMES:
        raise ValueError("group names must be epa_form, qb, ratings, injuries_rest")
    expected_sizes: dict[str, int] = {
        "epa_form": 36,
        "qb": 4,
        "ratings": 1,
        "injuries_rest": 6,
    }
    if len(H6_ADJ_FEATURE_COLUMNS) != 16 or len(_unadjusted_h2_epa_columns()) != 16:
        raise ValueError("epa_form requires 16 adjusted and 16 unadjusted EPA columns")
    seen: list[str] = []
    for name, columns in groups.items():
        if len(columns) != expected_sizes[name]:
            raise ValueError(f"{name} has {len(columns)} columns, expected {expected_sizes[name]}")
        for column in columns:
            if column in seen:
                raise ValueError(f"{column} is in more than one group")
            seen.append(column)
    defender: tuple[str, ...] = PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ
    if len(seen) != len(defender) or set(seen) != set(defender):
        raise ValueError("the four groups are not a partition of pkg_core_inj_l2")
    return groups


GROUP_COLUMNS: dict[str, tuple[str, ...]] = verify_group_partition()
HOLDOUT_REFUSED_NAMES: frozenset[str] = frozenset(GROUP_NAMES) | frozenset(
    {MODEL_GROUP_MEAN, MODEL_GROUP_MEAN_DEF}
)


def logit_mean_probabilities(members: list[np.ndarray]) -> np.ndarray:
    """Equal logit average of two or more probability vectors.

    Each vector is clipped to ``[1e-6, 1 - 1e-6]`` before the logit.
    Every member has the same weight, ``1 / n``.

    Args:
        members: Probability vectors of the same length. At least two.

    Returns:
        The sigmoid of the average logit.

    Raises:
        ValueError: Fewer than two members are given, or the shapes differ.
    """
    if len(members) < 2:
        raise ValueError("a logit average needs at least two members")
    logits: list[np.ndarray] = [
        logit_probability(np.asarray(member, dtype=float)) for member in members
    ]
    if any(logits[0].shape != item.shape for item in logits):
        raise ValueError("logit members must have the same number of probabilities")
    weight: float = 1.0 / float(len(logits))
    mixed: np.ndarray = np.zeros(logits[0].shape, dtype=float)
    for item in logits:
        mixed = mixed + (weight * item)
    return 1.0 / (1.0 + np.exp(-mixed))


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


def group_logistic_fold_predictor(
    columns: tuple[str, ...],
    max_iter: int,
    random_state: int,
) -> FoldPredictor:
    """Logistic refit on one group's columns. C stays 0.1.

    Evaluation labels are ignored. The scored season is not a training row.

    Args:
        columns: The group's columns. They must already be on the frame.
        max_iter: Iterations for ``lbfgs``. This is not a search.
        random_state: Seed for the classifier.

    Returns:
        A fold predictor.
    """
    selected: tuple[str, ...] = tuple(columns)

    def predict(
        train_frame: pd.DataFrame,
        train_target: pd.Series,
        eval_frame: pd.DataFrame,
        eval_target: pd.Series,
    ) -> np.ndarray:
        del eval_target
        train_subset: pd.DataFrame = _subset_frame(train_frame, selected)
        eval_subset: pd.DataFrame = _subset_frame(eval_frame, selected)
        pipeline: Pipeline = fit_logistic_pipeline(
            train_subset,
            train_target,
            C=GROUP_LOGISTIC_C,
            max_iter=max_iter,
            random_state=random_state,
        )
        return _positive_probability(pipeline, eval_subset)

    return predict


def _subset_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Keep identifiers and the group's columns, in that order."""
    missing: list[str] = [column for column in columns if column not in frame.columns]
    if missing:
        joined: str = ", ".join(missing)
        raise ValueError(f"group columns are missing from the frame: {joined}")
    identifiers: list[str] = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    return frame.loc[:, identifiers + list(columns)]


def _positive_probability(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Home-win probability from a fitted logistic pipeline."""
    columns: list[str] = model_feature_columns(frame)
    classes: list[int] = [int(label) for label in pipeline.named_steps["classifier"].classes_]
    if 1 not in classes:
        raise ValueError("logistic fit has no positive class")
    positive_index: int = classes.index(1)
    return np.asarray(
        pipeline.predict_proba(frame[columns])[:, positive_index],
        dtype=float,
    )


def _align_oof(reference_oof: pd.DataFrame, other_oof: pd.DataFrame) -> pd.DataFrame:
    """Return ``other_oof`` in ``reference_oof`` game order.

    Raises:
        ValueError: A game, a season, or a label does not match.
    """
    required: tuple[str, ...] = ("game_id", "season", "y_true", "y_proba")
    for label, frame in (("reference", reference_oof), ("member", other_oof)):
        missing: list[str] = [column for column in required if column not in frame.columns]
        if missing:
            joined: str = ", ".join(missing)
            raise ValueError(f"{label} OOF is missing columns: {joined}")
    reference_ids: list[str] = [str(game_id) for game_id in reference_oof["game_id"]]
    other: pd.DataFrame = other_oof.copy()
    other["game_id"] = other["game_id"].map(str)
    other = other.drop_duplicates("game_id").set_index("game_id")
    if set(reference_ids) != set(other.index.astype(str)):
        raise ValueError("group members scored different games")
    ordered: pd.DataFrame = other.loc[reference_ids].reset_index()
    if [int(season) for season in reference_oof["season"]] != [
        int(season) for season in ordered["season"]
    ]:
        raise ValueError("group members scored different seasons")
    if [int(label) for label in reference_oof["y_true"]] != [
        int(label) for label in ordered["y_true"]
    ]:
        raise ValueError("group member labels do not match")
    return ordered


def _folds_from_probabilities(
    member_folds: list[dict[str, Any]],
    averaged: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Per-season metrics of an averaged probability table."""
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
                f"group average season {fold['season']} has {len(season_rows)} rows, "
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


def _average_oof(
    member_oofs: list[pd.DataFrame],
    member_folds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build pooled metrics, folds, and an OOF table from aligned members."""
    aligned: list[pd.DataFrame] = [member_oofs[0]]
    for other in member_oofs[1:]:
        aligned.append(_align_oof(member_oofs[0], other))
    averaged: pd.DataFrame = member_oofs[0].loc[:, ["game_id", "season", "y_true"]].copy()
    averaged["y_proba"] = logit_mean_probabilities(
        [frame["y_proba"].to_numpy(dtype=float) for frame in aligned]
    )
    pooled: dict[str, float | int | None] = binary_classification_metrics(
        averaged["y_true"].to_numpy(),
        averaged["y_proba"].to_numpy(dtype=float),
    )
    return {
        "pooled": pooled,
        "folds": _folds_from_probabilities(member_folds, averaged),
        "oof": averaged,
    }


def _weight_text(count: int) -> str:
    """Equal weights as a comma-separated string."""
    weight: float = 1.0 / float(count)
    return ",".join(f"{weight:g}" for _ in range(count))


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


def group_mean_result(
    games: pd.DataFrame,
    features: pd.DataFrame,
    spec: CvModelSpec,
    shared: dict[str, Any],
    cv_config: RollingOriginConfig,
    max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    """Fit the four group logistics and average them in logit space.

    Each group uses C 0.1. The weights are 1/4. Nothing is fit on the
    averaged probabilities.

    Args:
        games: Schedule used to build the training matrix.
        features: Pregame feature frame that contains the defender columns.
        spec: The ``group_mean`` row. Its feature set must be the defender.
        shared: Params stored on every row of this comparison.
        cv_config: Outer rolling-origin window.
        max_iter: Iterations for each group logistic.
        random_state: Seed for each group logistic.

    Returns:
        A CV result record. ``group_members`` holds the four group fits.
        Status is ``skipped`` when a group does not score.
    """
    if spec.feature_set != FEATURE_SET_PKG_CORE_INJ_L2:
        raise ValueError("group_mean scores the pkg_core_inj_l2 columns")
    verify_group_partition()
    params: dict[str, Any] = {
        **shared,
        "model_type": MODEL_GROUP_MEAN,
        "feature_set": spec.feature_set,
        "members": ",".join(GROUP_NAMES),
        "weights": _weight_text(len(GROUP_NAMES)),
        "learned_weights": False,
        "C": GROUP_LOGISTIC_C,
        "regularization": "l2",
        "max_iter": max_iter,
        "probability_clip": LOGIT_CLIP,
        "imputation": "feature_column_groups",
        "scaler": "standard",
    }
    matrix: pd.DataFrame
    target: pd.Series
    matrix, target = prepare_training_data(games, features, feature_set=spec.feature_set)
    members: list[dict[str, Any]] = []
    for name in GROUP_NAMES:
        scored: dict[str, Any] = rolling_origin_cv(
            matrix,
            target,
            group_logistic_fold_predictor(GROUP_COLUMNS[name], max_iter, random_state),
            cv_config,
        )
        record: dict[str, Any] = cv_result_record(
            name,
            spec.feature_set,
            "ok",
            None,
            {"C": GROUP_LOGISTIC_C, "columns": len(GROUP_COLUMNS[name])},
            scored,
        )
        members.append(record)
        if record["status"] != "ok" or record.get("oof") is None:
            skipped: dict[str, Any] = cv_result_record(
                MODEL_GROUP_MEAN,
                spec.feature_set,
                "skipped",
                "missing_member",
                params,
                None,
            )
            skipped["group_members"] = members
            return skipped
    averaged: dict[str, Any] = _average_oof(
        [member["oof"] for member in members],
        list(members[0]["folds"]),
    )
    result: dict[str, Any] = cv_result_record(
        MODEL_GROUP_MEAN,
        spec.feature_set,
        "ok",
        None,
        params,
        averaged,
    )
    result["group_members"] = members
    return result


def group_mean_def_result(
    results: list[dict[str, Any]],
    spec: CvModelSpec,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Average ``group_mean`` and the defender with equal logit weights.

    No weight is fit. Both probabilities are clipped before the logit.

    Args:
        results: Rows scored before this one, including both members.
        spec: The ``group_mean_def`` row.
        shared: Params stored on every row of this comparison.

    Returns:
        A CV result record. Status is ``skipped`` when a member did not score.
    """
    params: dict[str, Any] = {
        **shared,
        "model_type": MODEL_GROUP_MEAN_DEF,
        "feature_set": spec.feature_set,
        "members": f"{MODEL_GROUP_MEAN},{MODEL_LOGISTIC}:{spec.feature_set}",
        "weights": _weight_text(2),
        "learned_weights": False,
        "probability_clip": LOGIT_CLIP,
    }
    grouped: dict[str, Any] | None = _member_result(results, MODEL_GROUP_MEAN, spec.feature_set)
    defender: dict[str, Any] | None = _member_result(results, MODEL_LOGISTIC, spec.feature_set)
    if (
        grouped is None
        or defender is None
        or grouped.get("status") != "ok"
        or defender.get("status") != "ok"
        or grouped.get("oof") is None
        or defender.get("oof") is None
    ):
        return cv_result_record(
            MODEL_GROUP_MEAN_DEF,
            spec.feature_set,
            "skipped",
            "missing_member",
            params,
            None,
        )
    ordered: pd.DataFrame = _align_oof(grouped["oof"], defender["oof"])
    averaged: dict[str, Any] = _average_oof(
        [grouped["oof"], ordered],
        list(grouped["folds"]),
    )
    return cv_result_record(
        MODEL_GROUP_MEAN_DEF,
        spec.feature_set,
        "ok",
        None,
        params,
        averaged,
    )


def group_diagnostics_table(results: list[dict[str, Any]]) -> pd.DataFrame | None:
    """OOF log loss of each group, and the logit correlation with the defender.

    The four groups are not candidates. The correlation is not a gate.

    Args:
        results: Scored rows, including ``group_mean`` and the defender.

    Returns:
        The diagnostics table, or None when ``group_mean`` did not score.
    """
    grouped: dict[str, Any] | None = next(
        (result for result in results if result.get("model") == MODEL_GROUP_MEAN),
        None,
    )
    if grouped is None or grouped.get("status") != "ok":
        return None
    members: list[dict[str, Any]] = list(grouped.get("group_members") or [])
    rows: list[dict[str, Any]] = []
    for member in members:
        pooled: dict[str, Any] = member.get("pooled") or {}
        rows.append(
            {
                "row_kind": "group",
                "group": str(member.get("model")),
                "n": pooled.get("n"),
                "log_loss": pooled.get("log_loss"),
                "logit_correlation": None,
            }
        )
    defender: dict[str, Any] | None = _member_result(
        results,
        MODEL_LOGISTIC,
        str(grouped["feature_set"]),
    )
    correlation: float | None = None
    if (
        defender is not None
        and defender.get("status") == "ok"
        and defender.get("oof") is not None
        and grouped.get("oof") is not None
    ):
        ordered = _align_oof(grouped["oof"], defender["oof"])
        correlation = logit_correlation(
            grouped["oof"]["y_proba"].to_numpy(dtype=float),
            ordered["y_proba"].to_numpy(dtype=float),
        )
    summary_pooled: dict[str, Any] = grouped.get("pooled") or {}
    rows.append(
        {
            "row_kind": "summary",
            "group": MODEL_GROUP_MEAN,
            "n": summary_pooled.get("n"),
            "log_loss": summary_pooled.get("log_loss"),
            "logit_correlation": correlation,
        }
    )
    return pd.DataFrame(rows, columns=list(GROUP_DIAGNOSTIC_COLUMNS))
