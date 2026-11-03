"""Model factories for the home prior, logistic regression, fold-tuned L2, and GBM."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_LOGISTIC: str = "logistic"
MODEL_LOGISTIC_L2: str = "logistic_l2"
MODEL_BASELINE_HOME: str = "baseline-home"
IDENTIFIER_COLUMNS: tuple[str, ...] = ("game_id", "gameday", "season")
# Fixed-C logistic is already L2. logistic_l2 chooses C inside each training fold.
# The grid matches the sweep endpoints (0.1 and 10) plus one step on each side.
DEFAULT_L2_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


class LogisticL2Settings(NamedTuple):
    """C grid and solver settings for one L2 fit. The scored season is not included."""

    c_grid: tuple[float, ...] = DEFAULT_L2_C_GRID
    fallback_c: float = 1.0
    max_iter: int = 1000
    random_state: int = 42


class LogisticL2Fit(NamedTuple):
    """A fitted L2 logistic pipeline and the C chosen on the training seasons."""

    pipeline: Pipeline
    metrics: dict[str, float]
    chosen_c: float
    c_selection: str
    inner_log_loss: float | None
    inner_seasons: tuple[int, ...]


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model columns, dropping game identifiers."""
    return [column for column in frame.columns if column not in IDENTIFIER_COLUMNS]


def feature_column_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Split model columns into imputation groups.

    Level EPA is filled with 0 and level win rate with 0.5. A starting-QB
    rating is filled with 0, the replacement level. A clean success rate is
    filled with 0.5. Difference columns are filled with 0. Other columns,
    including rest days, are left as they are.

    Returns:
        Dict with keys ``epa``, ``win_rate``, ``success_rate``, ``other``,
        and ``diff``. A group can be empty.
    """
    diff_features: list[str] = [column for column in feature_columns if column.startswith("diff_")]
    level_features: list[str] = [
        column for column in feature_columns if column not in diff_features
    ]
    epa_features: list[str] = [
        column for column in level_features if "epa" in column or "qb_rating" in column
    ]
    win_rate_features: list[str] = [column for column in level_features if "win_rate" in column]
    success_rate_features: list[str] = [
        column for column in level_features if "success_rate" in column
    ]
    grouped: set[str] = set(epa_features + win_rate_features + success_rate_features)
    other_features: list[str] = [column for column in level_features if column not in grouped]
    return {
        "epa": epa_features,
        "win_rate": win_rate_features,
        "success_rate": success_rate_features,
        "other": other_features,
        "diff": diff_features,
    }


def _imputation_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Build the column-wise imputer used before a classifier.

    Empty groups are omitted. When the matrix has no difference columns the
    transformer list matches the pre-phase-2 pipeline: EPA, win rate, other.

    Raises:
        ValueError: There are no model columns to impute.
    """
    groups: dict[str, list[str]] = feature_column_groups(feature_columns)
    transformers: list[tuple[str, Any, list[str]]] = []
    if groups["epa"]:
        transformers.append(
            ("epa_imputer", SimpleImputer(strategy="constant", fill_value=0), groups["epa"])
        )
    if groups["win_rate"]:
        transformers.append(
            (
                "win_rate_imputer",
                SimpleImputer(strategy="constant", fill_value=0.5),
                groups["win_rate"],
            )
        )
    if groups["success_rate"]:
        transformers.append(
            (
                "success_rate_imputer",
                SimpleImputer(strategy="constant", fill_value=0.5),
                groups["success_rate"],
            )
        )
    if groups["other"]:
        transformers.append(("other_passthrough", "passthrough", groups["other"]))
    if groups["diff"]:
        transformers.append(
            ("diff_imputer", SimpleImputer(strategy="constant", fill_value=0), groups["diff"])
        )
    if not transformers:
        raise ValueError("feature matrix has no model columns")
    return ColumnTransformer(transformers=transformers)


def _validation_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Accuracy, Brier, log loss, and ROC AUC on one validation split."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_baseline_home(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[DummyClassifier, dict[str, float]]:
    """Train a home baseline whose probability is the training home win rate.

    ``predict_proba`` is that rate on every row. Hard labels are the majority
    class. When the training home win rate is above one half, every hard
    label is home.
    """
    train_features: pd.DataFrame = X_train.drop(columns=list(IDENTIFIER_COLUMNS))
    val_features: pd.DataFrame = X_val.drop(columns=list(IDENTIFIER_COLUMNS))
    model = DummyClassifier(strategy="prior", random_state=42)
    model.fit(train_features, y_train)
    y_pred_proba: np.ndarray = model.predict_proba(val_features)[:, 1]
    y_pred: np.ndarray = model.predict(val_features)
    return model, _validation_metrics(y_val, y_pred, y_pred_proba)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Train logistic regression with imputation and standard scaling.

    Scaling is applied only in this pipeline, after imputation, so lbfgs can
    converge when ``C`` is large and feature scales differ.
    """
    feature_cols: list[str] = model_feature_columns(X_train)
    pipeline = Pipeline(
        [
            ("preprocessor", _imputation_preprocessor(feature_cols)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train[feature_cols], y_train)
    y_pred_proba: np.ndarray = pipeline.predict_proba(X_val[feature_cols])[:, 1]
    y_pred: np.ndarray = pipeline.predict(X_val[feature_cols])
    return pipeline, _validation_metrics(y_val, y_pred, y_pred_proba)


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Train gradient boosting with the same imputation pipeline, unscaled."""
    feature_cols: list[str] = model_feature_columns(X_train)
    pipeline = Pipeline(
        [
            ("preprocessor", _imputation_preprocessor(feature_cols)),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train[feature_cols], y_train)
    y_pred_proba: np.ndarray = pipeline.predict_proba(X_val[feature_cols])[:, 1]
    y_pred: np.ndarray = pipeline.predict(X_val[feature_cols])
    return pipeline, _validation_metrics(y_val, y_pred, y_pred_proba)


def _validate_c_grid(c_grid: tuple[float, ...]) -> tuple[float, ...]:
    """Return a grid of finite positive C values.

    Raises:
        ValueError: The grid is empty or a value is not finite and positive.
    """
    if len(c_grid) == 0:
        raise ValueError("c_grid must contain at least one positive C")
    values: list[float] = []
    for value in c_grid:
        c_value: float = float(value)
        if not np.isfinite(c_value) or c_value <= 0.0:
            raise ValueError(f"every C in c_grid must be finite and positive, got {value!r}")
        values.append(c_value)
    return tuple(values)


def _format_regularization_c(value: float) -> str:
    """Render one C without trailing zeros, stable enough for an MLflow param."""
    return f"{float(value):.6g}"


def _usable_inner_seasons(seasons: pd.Series, labels: pd.Series) -> list[int]:
    """Seasons that can score an expanding inner split.

    The earliest season is never an inner validation season. A season is
    usable when both the prior rows and that season have two classes.
    """
    ordered: list[int] = sorted(int(season) for season in pd.unique(seasons))
    usable: list[int] = []
    for season in ordered[1:]:
        prior_labels: pd.Series = labels.loc[seasons < season]
        season_labels: pd.Series = labels.loc[seasons == season]
        if prior_labels.nunique() < 2 or season_labels.nunique() < 2:
            continue
        usable.append(season)
    return usable


def _aligned_training_frame(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reset indexes and require a numeric target plus a season column."""
    if "season" not in train_frame.columns:
        raise ValueError("training frame must include season")
    frame: pd.DataFrame = train_frame.reset_index(drop=True)
    labels: pd.Series = pd.to_numeric(
        pd.Series(train_target).reset_index(drop=True), errors="coerce"
    )
    if len(frame) != len(labels):
        raise ValueError("train_frame and train_target must have the same number of rows")
    if labels.isna().any():
        raise ValueError("train_target must be numeric 0/1 with no nulls")
    return frame, labels


def select_logistic_l2_c(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    c_grid: tuple[float, ...] = DEFAULT_L2_C_GRID,
    fallback_c: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Choose C from training seasons only.

    For each season after the first training season, fit on the seasons
    strictly before it and score log loss on that season. The C with the
    lowest mean inner log loss wins. Ties keep the smaller C. The caller
    must not pass the season that the outer fold will score.

    Returns:
        Dict with ``chosen_c``, ``c_selection`` (``inner_season`` or
        ``fallback``), ``fallback_reason``, ``inner_log_loss``, and
        ``inner_seasons``.
    """
    settings = LogisticL2Settings(
        c_grid=c_grid,
        fallback_c=fallback_c,
        max_iter=max_iter,
        random_state=random_state,
    )
    return _select_logistic_l2_c(train_frame, train_target, settings)


def _select_logistic_l2_c(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    settings: LogisticL2Settings,
) -> dict[str, Any]:
    """Choose C using ``settings``. See ``select_logistic_l2_c``."""
    grid: tuple[float, ...] = _validate_c_grid(settings.c_grid)
    fallback: float = float(settings.fallback_c)
    if not np.isfinite(fallback) or fallback <= 0.0:
        raise ValueError(f"fallback_c must be finite and positive, got {settings.fallback_c!r}")
    frame, labels = _aligned_training_frame(train_frame, train_target)
    seasons: pd.Series = frame["season"]
    inner_seasons: list[int] = _usable_inner_seasons(seasons, labels)
    if not inner_seasons:
        return {
            "chosen_c": fallback,
            "c_selection": "fallback",
            "fallback_reason": "no_inner_season",
            "inner_log_loss": None,
            "inner_seasons": (),
        }
    scored: list[dict[str, float]] = []
    for c_value in grid:
        losses: list[float] = []
        for season in inner_seasons:
            prior_mask = seasons < season
            season_mask = seasons == season
            _model, metrics = train_logistic_regression(
                frame.loc[prior_mask],
                labels.loc[prior_mask],
                frame.loc[season_mask],
                labels.loc[season_mask],
                C=c_value,
                max_iter=settings.max_iter,
                random_state=settings.random_state,
            )
            losses.append(float(metrics["log_loss"]))
        mean_loss: float = float(np.mean(losses))
        if np.isfinite(mean_loss):
            scored.append({"c": c_value, "mean_log_loss": mean_loss})
    if not scored:
        return {
            "chosen_c": fallback,
            "c_selection": "fallback",
            "fallback_reason": "non_finite_inner_log_loss",
            "inner_log_loss": None,
            "inner_seasons": tuple(inner_seasons),
        }
    best: dict[str, float] = min(scored, key=lambda item: (item["mean_log_loss"], item["c"]))
    return {
        "chosen_c": float(best["c"]),
        "c_selection": "inner_season",
        "fallback_reason": None,
        "inner_log_loss": float(best["mean_log_loss"]),
        "inner_seasons": tuple(inner_seasons),
    }


def fit_logistic_l2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    c_grid: tuple[float, ...] = DEFAULT_L2_C_GRID,
    fallback_c: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticL2Fit:
    """Tune L2 C on the training seasons, then refit and score ``X_val``.

    ``X_val`` is not used to choose C. The returned metrics are the
    validation metrics of the refit.
    """
    selection: dict[str, Any] = select_logistic_l2_c(
        X_train,
        y_train,
        c_grid=c_grid,
        fallback_c=fallback_c,
        max_iter=max_iter,
        random_state=random_state,
    )
    pipeline, metrics = train_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        C=float(selection["chosen_c"]),
        max_iter=max_iter,
        random_state=random_state,
    )
    return LogisticL2Fit(
        pipeline=pipeline,
        metrics=metrics,
        chosen_c=float(selection["chosen_c"]),
        c_selection=str(selection["c_selection"]),
        inner_log_loss=selection["inner_log_loss"],
        inner_seasons=tuple(selection["inner_seasons"]),
    )


def chosen_c_param_fields(folds: list[dict[str, Any]]) -> dict[str, str]:
    """MLflow params for the C chosen on each scored fold.

    Returns:
        ``chosen_c_by_fold`` and one ``chosen_c_<season>`` key per scored fold
        that recorded a C. Empty when no fold recorded one.
    """
    pieces: list[str] = []
    fields: dict[str, str] = {}
    for fold in folds:
        if fold.get("status") != "scored" or fold.get("chosen_c") is None:
            continue
        season: int = int(fold["season"])
        text: str = _format_regularization_c(float(fold["chosen_c"]))
        pieces.append(f"{season}:{text}")
        fields[f"chosen_c_{season}"] = text
    if pieces:
        fields["chosen_c_by_fold"] = ",".join(pieces)
    return fields
