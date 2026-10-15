"""
Train NFL pregame prediction models with MLflow tracking.

Usage:
    python -m src.train.train --model logistic --lookback 10 --seed 42
    python -m src.train.train --model gbm --max-depth 5 --n-estimators 100
    python -m src.train.train --model baseline-home
    python -m src.train.train --cv
    python -m src.train.train --cv --phase 2
    python -m src.train.train --cv --phase 2 --hypothesis h3
    python -m src.train.train --cv --phase 2 --hypothesis h4
    python -m src.train.train --cv --phase 2 --hypothesis h5
    python -m src.train.train --cv --phase 2 --hypothesis ridge
"""

import argparse
import os
import subprocess
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import mlflow
import mlflow.sklearn
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

from src.data.databricks_client import DatabricksClient
from src.features.pregame import (
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_MEAN_RATING,
    ELO_SCALE,
    ELO_SEASON_REVERSION,
    FEATURE_SET_COLUMNS,
    FEATURE_SET_CURRENT,
    FEATURE_SET_DIFF,
    FEATURE_SET_DIFF_ONLY,
    FEATURE_SET_H2_DEF,
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H3_REPLACE,
    FEATURE_SET_H3_SPLIT,
    FEATURE_SET_H4_CLEAN,
    FEATURE_SET_H4_CLEAN_SR,
    FEATURE_SET_H5_QB,
    FEATURE_SET_NAMES,
    FEATURE_SET_WITH_GAMES_PLAYED,
    QB_PRIOR_DROPBACKS,
    REPLACEMENT_EPA_PER_DROPBACK,
    SEASON_BLEND_K,
    SEASON_SHRINK_FACTOR,
    build_pregame_features,
    elo_home_probabilities,
    feature_columns_for,
    market_home_probabilities,
    prepare_training_data,
    split_by_season,
)
from src.utils.config import DEFAULT_MLFLOW_TRACKING_URI, resolve_tracking_uri
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Fluent SQLite tracking stores artifacts under ./mlruns unless an experiment
# sets another location. Keep them in a gitignored directory.
LOCAL_ARTIFACT_DIR: str = "mlartifacts"

# Types skops refuses by default that these pipelines are known to contain.
# Anything else fails closed. Do not replace this with skops' full untrusted list.
REVIEWED_SKOPS_TRUSTED_TYPES: frozenset[str] = frozenset(
    {
        "numpy.dtype",
        "sklearn.tree._tree.Tree",
    }
)

# Phase 1 scores 2016-2023 only. 2024 stays the final validation. 2025 is held out.
FINAL_VALIDATION_SEASON: int = 2024
HELD_OUT_TEST_SEASON: int = 2025
PHASE1_EXPERIMENT_NAME: str = "nfl_pregame/phase1_baselines"
PHASE1_TAG: str = "1"
PHASE2_EXPERIMENT_NAME: str = "nfl_pregame/phase2_h1"
PHASE2_H2_EXPERIMENT_NAME: str = "nfl_pregame/phase2_h2"
PHASE2_H3_EXPERIMENT_NAME: str = "nfl_pregame/phase2_h3"
PHASE2_H4_EXPERIMENT_NAME: str = "nfl_pregame/phase2_h4"
PHASE2_H5_EXPERIMENT_NAME: str = "nfl_pregame/phase2_h5"
PHASE2_RIDGE_EXPERIMENT_NAME: str = "nfl_pregame/phase2_ridge"
PHASE2_TAG: str = "2"
HYPOTHESIS_H1: str = "h1-diff"
HYPOTHESIS_H2: str = "h2-season-def-regression"
HYPOTHESIS_H3: str = "h3-pass-rush-epa"
HYPOTHESIS_H4: str = "h4-clean-epa"
HYPOTHESIS_H5: str = "h5-starting-qb"
HYPOTHESIS_RIDGE: str = "ridge-reference"
HYPOTHESIS_ALIASES: dict[str, str] = {
    "h1": HYPOTHESIS_H1,
    "h1-diff": HYPOTHESIS_H1,
    "h2": HYPOTHESIS_H2,
    "h2-season-def-regression": HYPOTHESIS_H2,
    "h3": HYPOTHESIS_H3,
    "h3-pass-rush-epa": HYPOTHESIS_H3,
    "h4": HYPOTHESIS_H4,
    "h4-clean-epa": HYPOTHESIS_H4,
    "h5": HYPOTHESIS_H5,
    "h5-starting-qb": HYPOTHESIS_H5,
    "ridge": HYPOTHESIS_RIDGE,
    "ridge-reference": HYPOTHESIS_RIDGE,
}
MODEL_LOGISTIC: str = "logistic"
MODEL_LOGISTIC_L2: str = "logistic_l2"
# Fixed-C logistic is already L2. logistic_l2 chooses C inside each training fold.
# The grid matches the sweep endpoints (0.1 and 10) plus one step on each side.
DEFAULT_L2_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_SCHEME: str = "rolling_season_2016_2023"
PHASE1_MARKDOWN_NAME: str = "comparison.md"
PHASE1_CSV_NAME: str = "comparison.csv"
REFERENCE_MODEL_NAME: str = "baseline-home"
BASELINE_MODEL_NAMES: frozenset[str] = frozenset({REFERENCE_MODEL_NAME, "elo", "market"})
CV_DATASET_NAME: str = "nfl_pregame_cv"
# Accept a set only when pooled log loss falls by at least this much.
MIN_LOG_LOSS_IMPROVEMENT: float = 0.003
# A requested feature column at or above this null share is not a real signal.
# The H4 pre-rebuild run filled absent clean columns and scored a constant.
DEFAULT_MAX_NULL_FRACTION: float = 0.5
FEATURE_COLUMN_REBUILD_HINT: str = "run make dbt-build-databricks"
PHASE1_LOGISTIC_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_CURRENT,
    FEATURE_SET_WITH_GAMES_PLAYED,
)
PHASE2_LOGISTIC_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_CURRENT,
    FEATURE_SET_DIFF,
    FEATURE_SET_DIFF_ONLY,
)
PHASE2_H2_LOGISTIC_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_CURRENT,
    FEATURE_SET_H2_DEF,
    FEATURE_SET_H2_FULL,
)
PHASE2_H3_LOGISTIC_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H3_SPLIT,
    FEATURE_SET_H3_REPLACE,
)
PHASE2_H4_LOGISTIC_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H4_CLEAN,
    FEATURE_SET_H4_CLEAN_SR,
)
PHASE2_H5_LOGISTIC_FEATURE_SETS: tuple[str, ...] = (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_H5_QB,
)
# Phase 1 stays on the home prior. Phase 2 H1 and H2 use logistic ``current``.
# H3, H4, and ridge use logistic ``h2_full``. The hypothesis map is the phase 2 default.
DEFAULT_REFERENCE_BY_PHASE: dict[int, tuple[str, str]] = {
    1: (REFERENCE_MODEL_NAME, "none"),
    2: ("logistic", FEATURE_SET_CURRENT),
}
DEFAULT_REFERENCE_BY_HYPOTHESIS: dict[str, tuple[str, str]] = {
    HYPOTHESIS_H1: ("logistic", FEATURE_SET_CURRENT),
    HYPOTHESIS_H2: ("logistic", FEATURE_SET_CURRENT),
    HYPOTHESIS_H3: ("logistic", FEATURE_SET_H2_FULL),
    HYPOTHESIS_H4: ("logistic", FEATURE_SET_H2_FULL),
    HYPOTHESIS_H5: ("logistic", FEATURE_SET_H2_FULL),
    HYPOTHESIS_RIDGE: ("logistic", FEATURE_SET_H2_FULL),
}
MODEL_DEFAULT_FEATURE_SET: dict[str, str] = {
    REFERENCE_MODEL_NAME: "none",
    MODEL_LOGISTIC: FEATURE_SET_CURRENT,
    MODEL_LOGISTIC_L2: FEATURE_SET_H2_FULL,
    "elo": "elo",
    "market": "market",
}
# Models whose feature set can be accepted. Baselines are not in this set.
SCORED_FEATURE_MODELS: frozenset[str] = frozenset({MODEL_LOGISTIC, MODEL_LOGISTIC_L2})
PHASE1_NOTE: str = (
    "Phase 1 rolling-origin baselines. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)
PHASE2_NOTE: str = (
    "Phase 2 hypothesis h1-diff. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)
PHASE2_H2_NOTE: str = (
    "Phase 2 hypothesis h2-season-def-regression. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)
PHASE2_H3_NOTE: str = (
    "Phase 2 hypothesis h3-pass-rush-epa. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)
PHASE2_H4_NOTE: str = (
    "Phase 2 hypothesis h4-clean-epa. "
    "Shrinkage and the blend weight are fixed. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)
PHASE2_H5_NOTE: str = (
    "Phase 2 hypothesis h5-starting-qb. "
    "QB shrinkage and the replacement level are fixed. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)
PHASE2_RIDGE_NOTE: str = (
    "Phase 2 ridge reference. L2 logistic chooses C inside each training fold. "
    f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
    "No model is promoted from this run."
)


class CvModelSpec(NamedTuple):
    """One model and feature set scored in a rolling-origin comparison."""

    model: str
    feature_set: str


PHASE2_RIDGE_MODEL_SPECS: tuple[CvModelSpec, ...] = (
    CvModelSpec(MODEL_LOGISTIC, FEATURE_SET_H2_FULL),
    CvModelSpec(MODEL_LOGISTIC_L2, FEATURE_SET_H2_FULL),
    CvModelSpec(MODEL_LOGISTIC_L2, FEATURE_SET_H3_SPLIT),
    CvModelSpec(MODEL_LOGISTIC_L2, FEATURE_SET_H3_REPLACE),
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

COMPARISON_COLUMNS: tuple[str, ...] = (
    "model",
    "feature_set",
    "status",
    "reason",
    "n",
    "log_loss",
    "brier",
    "auc",
    "accuracy",
    "log_loss_diff",
    "log_loss_diff_ci_low",
    "log_loss_diff_ci_high",
)


class RollingOriginConfig(NamedTuple):
    """Season window and bootstrap settings for rolling-origin CV."""

    cv_start_season: int = 2016
    cv_end_season: int = 2023
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def load_data(
    fixture_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load fct_games and fct_team_game_epa from Databricks or fixtures.

    Args:
        fixture_path: Optional directory path to fixtures for testing

    Returns:
        Tuple of (fct_games, fct_team_game_epa) DataFrames
    """
    if fixture_path:
        fixture_dir = Path(fixture_path)
        games_file = fixture_dir / "fct_games.parquet"
        epa_file = fixture_dir / "fct_team_game_epa.parquet"

        if not games_file.exists() or not epa_file.exists():
            raise FileNotFoundError(
                f"Fixtures not found in {fixture_dir}. Expected: {games_file.name}, {epa_file.name}"
            )

        logger.info(f"Loading data from fixtures: {fixture_dir}")
        fct_games = pd.read_parquet(games_file)
        fct_team_game_epa = pd.read_parquet(epa_file)

    else:
        logger.info("Loading data from Databricks")
        with DatabricksClient() as client:
            fct_games = client.read_table("fct_games")
            fct_team_game_epa = client.read_table("fct_team_game_epa")

    logger.info(f"Loaded {len(fct_games)} games, {len(fct_team_game_epa)} team-game records")
    return fct_games, fct_team_game_epa


def load_qb_inputs(
    fixture_path: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the expected-starter and dropback marts for H5.

    A fixture directory without both parquet files returns None so the
    column check can name the missing ratings. A warehouse read uses
    ``fct_expected_starter`` and ``fct_qb_dropbacks``. A missing warehouse
    table surfaces the SQL error.

    Args:
        fixture_path: Directory used by tests. None reads the warehouse.

    Returns:
        ``(expected_starters, qb_dropbacks)``. Both are None when the
        fixture files are absent.
    """
    if fixture_path:
        fixture_dir = Path(fixture_path)
        starter_file = fixture_dir / "fct_expected_starter.parquet"
        dropback_file = fixture_dir / "fct_qb_dropbacks.parquet"
        if not starter_file.exists() or not dropback_file.exists():
            logger.info("H5 fixture tables are absent in %s", fixture_dir)
            return None, None
        logger.info("Loading H5 tables from fixtures: %s", fixture_dir)
        return pd.read_parquet(starter_file), pd.read_parquet(dropback_file)
    logger.info("Loading H5 tables from Databricks")
    with DatabricksClient() as client:
        return client.read_table("fct_expected_starter"), client.read_table("fct_qb_dropbacks")


def train_baseline_home(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[DummyClassifier, dict]:
    """Train a home baseline whose probability is the training home win rate.

    ``predict_proba`` is that rate on every row, so log loss and Brier stay
    finite. Hard labels are the majority class. When the training home win
    rate is above one half, every hard label is home and accuracy matches
    an always-home rule.

    Returns:
        Tuple of (model, metrics_dict)
    """
    id_columns: list[str] = ["game_id", "gameday", "season"]
    train_features: pd.DataFrame = X_train.drop(columns=id_columns)
    val_features: pd.DataFrame = X_val.drop(columns=id_columns)
    model = DummyClassifier(strategy="prior", random_state=42)
    model.fit(train_features, y_train)

    y_pred_proba = model.predict_proba(val_features)[:, 1]
    y_pred = model.predict(val_features)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "brier_score": brier_score_loss(y_val, y_pred_proba),
        "log_loss": log_loss(y_val, y_pred_proba),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    return model, metrics


def feature_column_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Split model columns into imputation groups.

    Level EPA is filled with 0 and level win rate with 0.5. A starting-QB
    rating is filled with 0, the replacement level. A clean success
    rate is filled with 0.5, the same uninformative rate. Difference
    columns are filled with 0. A missing home-minus-away win rate is not a
    0.5 home edge. Other columns, including rest days, are left as they are.

    Args:
        feature_columns: Model columns in training-matrix order, without ids.

    Returns:
        Dict with keys ``epa``, ``win_rate``, ``success_rate``, ``other``,
        and ``diff``. Each value keeps the input order. A group can be empty.
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

    Empty groups are omitted so a difference-only matrix does not pass an
    empty column list to sklearn. When the matrix has no difference columns
    the transformer list matches the pre-phase-2 pipeline: EPA, win rate, other.

    Args:
        feature_columns: Model columns, without game identifiers.

    Returns:
        An unfitted ``ColumnTransformer``.

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


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """Train logistic regression with imputation and standard scaling.

    Scaling is applied only in this pipeline, after imputation, so lbfgs
    can converge when ``C`` is large and feature scales differ. Gradient
    boosting keeps its own unscaled pipeline.

    Returns:
        Tuple of (pipeline, metrics_dict)
    """
    feature_cols = [col for col in X_train.columns if col not in ["game_id", "gameday", "season"]]

    preprocessor = _imputation_preprocessor(feature_cols)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
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

    y_pred_proba = pipeline.predict_proba(X_val[feature_cols])[:, 1]
    y_pred = pipeline.predict(X_val[feature_cols])

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "brier_score": brier_score_loss(y_val, y_pred_proba),
        "log_loss": log_loss(y_val, y_pred_proba),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    return pipeline, metrics


class LogisticL2Fit(NamedTuple):
    """A fitted L2 logistic pipeline and the C chosen on the training seasons."""

    pipeline: Pipeline
    metrics: dict[str, float]
    chosen_c: float
    c_selection: str
    inner_log_loss: float | None
    inner_seasons: tuple[int, ...]


def _validate_c_grid(c_grid: tuple[float, ...]) -> tuple[float, ...]:
    """Return a grid of finite positive C values.

    Args:
        c_grid: Candidate inverse regularization strengths.

    Returns:
        The same values as floats.

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

    The earliest season has no prior rows, so it is never an inner validation
    season. A season is usable when both the prior rows and that season have
    two classes.

    Args:
        seasons: Season of each training row, aligned with ``labels``.
        labels: Binary labels for those rows.

    Returns:
        Inner validation seasons, ascending.
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

    Args:
        train_frame: Training games, including ``season``.
        train_target: Binary labels aligned with ``train_frame``.
        c_grid: Positive C values to try.
        fallback_c: C used when no inner season can be scored.
        max_iter: Logistic solver iteration cap.
        random_state: Seed for each inner fit.

    Returns:
        Dict with ``chosen_c``, ``c_selection`` (``inner_season`` or
        ``fallback``), ``fallback_reason``, ``inner_log_loss``, and
        ``inner_seasons``.

    Raises:
        ValueError: The frame has no season, the labels do not align, or
            the grid or fallback C is invalid.
    """
    grid: tuple[float, ...] = _validate_c_grid(c_grid)
    fallback: float = float(fallback_c)
    if not np.isfinite(fallback) or fallback <= 0.0:
        raise ValueError(f"fallback_c must be finite and positive, got {fallback_c!r}")
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
                max_iter=max_iter,
                random_state=random_state,
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

    Args:
        X_train: Training matrix, including ``season``.
        y_train: Training labels.
        X_val: Rows scored after C is chosen.
        y_val: Labels for ``X_val``.
        c_grid: Candidate C values.
        fallback_c: C used when the training seasons cannot form an inner split.
        max_iter: Logistic solver iteration cap.
        random_state: Seed for the inner fits and the refit.

    Returns:
        Fitted pipeline, validation metrics, and the chosen C.
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


def _chosen_c_param_fields(folds: list[dict[str, Any]]) -> dict[str, str]:
    """MLflow params for the C chosen on each scored fold.

    Args:
        folds: Fold dicts from rolling-origin CV.

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


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Train gradient boosting classifier with imputation pipeline.

    Returns:
        Tuple of (pipeline, metrics_dict)
    """
    feature_cols = [col for col in X_train.columns if col not in ["game_id", "gameday", "season"]]

    preprocessor = _imputation_preprocessor(feature_cols)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
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

    y_pred_proba = pipeline.predict_proba(X_val[feature_cols])[:, 1]
    y_pred = pipeline.predict(X_val[feature_cols])

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "brier_score": brier_score_loss(y_val, y_pred_proba),
        "log_loss": log_loss(y_val, y_pred_proba),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    return pipeline, metrics


def resolve_skops_trusted_types(model: object) -> dict[str, list[str]]:
    """Return the reviewed skops types required to reload this model.

    The model is written to a temporary skops file and inspected with
    ``skops.io.get_untrusted_types``. Only types on
    ``REVIEWED_SKOPS_TRUSTED_TYPES`` are returned. An untrusted type outside
    that allowlist raises ``ValueError`` instead of being trusted.

    Args:
        model: Fitted sklearn estimator or pipeline.

    Returns:
        Dict with key ``trusted_types``, sorted and safe to pass to
        ``mlflow.sklearn.log_model``.
    """
    import skops.io

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path: Path = Path(tmp_dir) / "model.skops"
        skops.io.dump(model, model_path)
        untrusted: set[str] = set(skops.io.get_untrusted_types(file=model_path))

    unknown: set[str] = untrusted - REVIEWED_SKOPS_TRUSTED_TYPES
    if unknown:
        unknown_list: str = ", ".join(sorted(unknown))
        allowlist: str = ", ".join(sorted(REVIEWED_SKOPS_TRUSTED_TYPES))
        raise ValueError(
            "Refusing to trust skops types that are not on the reviewed "
            f"allowlist: {unknown_list}. Allowlist: {allowlist}."
        )
    return {"trusted_types": sorted(untrusted)}


def log_sklearn_model(model: object) -> dict[str, list[str]]:
    """Log a fitted sklearn model on the active MLflow run.

    The artifact name is ``model``. Trusted skops types are limited to the
    reviewed types this specific model needs.

    Args:
        model: Fitted estimator produced by a training function.

    Returns:
        Dict with key ``trusted_types`` listing the types that were trusted.
    """
    trusted: dict[str, list[str]] = resolve_skops_trusted_types(model)
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=trusted["trusted_types"],
    )
    return trusted


def activate_tracking_experiment(experiment_name: str, tracking_uri: str) -> None:
    """Point MLflow at a tracking URI and activate an experiment.

    Scheme URIs such as ``sqlite:///mlflow.db`` are used as given.
    ``databricks`` targets keep the workspace artifact root. A new SQLite
    experiment stores artifacts under ``mlartifacts/`` so they do not land
    in the filesystem tracking directory.

    Args:
        experiment_name: Experiment to create or activate.
        tracking_uri: URI already normalized by ``resolve_tracking_uri``.
    """
    mlflow.set_tracking_uri(tracking_uri)
    is_databricks_target: bool = tracking_uri == "databricks" or tracking_uri.startswith(
        "databricks"
    )
    if is_databricks_target:
        logger.info("Using Databricks MLflow tracking")
        mlflow.set_experiment(experiment_name)
        return

    logger.info(f"Using local MLflow tracking: {tracking_uri}")
    if not tracking_uri.startswith("sqlite:"):
        mlflow.set_experiment(experiment_name)
        return

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        return

    artifact_location: str = Path(LOCAL_ARTIFACT_DIR).resolve().as_uri()
    experiment_id: str = client.create_experiment(
        experiment_name,
        artifact_location=artifact_location,
    )
    mlflow.set_experiment(experiment_id=experiment_id)


def make_rolling_origin_config(**overrides: Any) -> RollingOriginConfig:
    """Build a CV config and reject a window that would score 2024 or 2025.

    Args:
        **overrides: Fields of ``RollingOriginConfig`` to replace.

    Returns:
        Validated config.

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


def binary_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float | int | None]:
    """Pooled log loss, Brier, AUC, and accuracy at probability 0.5.

    Args:
        y_true: Binary labels.
        y_proba: Predicted probabilities of class 1.

    Returns:
        Dict with ``log_loss``, ``brier``, ``auc``, ``accuracy``, and ``n``.
        AUC is missing when the labels have one class. An empty sample
        leaves every estimate missing.
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

    Args:
        y_true: Binary labels shared by both models.
        y_proba: Model probabilities.
        y_reference: Reference probabilities, one per row of ``y_true``.
        n_bootstrap: Resamples. Zero keeps the point difference and skips
            the interval.
        seed: Seed for the resampler.

    Returns:
        Dict with both log losses, the difference, the 2.5 and 97.5
        percentiles, and the row count.

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


def rolling_origin_cv(
    frame: pd.DataFrame,
    target: pd.Series,
    predict_fold: FoldPredictor,
    config: RollingOriginConfig | None = None,
) -> dict[str, Any]:
    """Train on seasons before S and score S, for each S in the CV window.

    2024 and 2025 are never scored, even when they are rows in ``frame``.
    Folds with no rows or a single class are recorded and skipped. Pooled
    metrics use only the scored out-of-fold rows.

    Args:
        frame: One row per game, including ``game_id`` and ``season``.
            Aligned by position with ``target``.
        target: Binary home-win labels.
        predict_fold: Called with the training frame, training labels,
            evaluation frame, and evaluation labels. Returns one probability
            per evaluation row.
        config: Season window and bootstrap settings. The bootstrap fields
            are stored for the caller. This function does not resample.

    Returns:
        Dict with ``pooled``, ``folds``, ``oof``, and ``config``.

    Raises:
        ValueError: The frame cannot be split, a fold returns a bad
            probability vector, or a held-out season was scored.
    """
    resolved: RollingOriginConfig = make_rolling_origin_config(
        **(config._asdict() if config is not None else {})
    )
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

    folds: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for season in range(resolved.cv_start_season, resolved.cv_end_season + 1):
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
            folds.append(fold)
            continue
        predicted = predict_fold(train_frame, train_target, eval_frame, eval_target)
        extras: dict[str, Any] = {}
        if isinstance(predicted, FoldPrediction):
            extras = dict(predicted.extras)
            probability_values = predicted.probabilities
        else:
            probability_values = predicted
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
        folds.append(fold)
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


def _model_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Probability of class 1 from a fitted baseline or training pipeline."""
    feature_columns: list[str] = [
        column for column in frame.columns if column not in ("game_id", "gameday", "season")
    ]
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
    """Pair two out-of-fold tables on ``game_id`` and bootstrap the log loss gap.

    Args:
        model_oof: Scored games for the candidate.
        reference_oof: Scored games for the reference model.
        n_bootstrap: Resamples passed to ``paired_log_loss_difference``.
        seed: Bootstrap seed.

    Returns:
        The paired difference on the games present in both tables.
    """
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


def _cv_result_record(
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


def resolve_reference_spec(
    phase: int,
    reference_model: str | None = None,
    reference_feature_set: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, str]:
    """Choose the paired-comparison reference for a CV phase.

    Phase 1 defaults to the home prior. Phase 2 H1 and H2 default to
    logistic ``current``. Phase 2 H3, H4, and the ridge comparison default to
    logistic ``h2_full``. Passing a model or a feature set replaces that
    default. When only the model is set, its usual feature set is used,
    except logistic on H3, H4, or ridge, which stays on ``h2_full`` unless a
    feature set is passed.

    Args:
        phase: ``1`` or ``2``.
        reference_model: Model name, or None for the phase default.
        reference_feature_set: Feature set name, or None to derive it.
        hypothesis: Phase 2 hypothesis name. None keeps H1.

    Returns:
        Dict with ``model`` and ``feature_set``.

    Raises:
        ValueError: The phase is unknown, or the model has no default set.
    """
    if phase not in DEFAULT_REFERENCE_BY_PHASE:
        known_phases: str = ", ".join(str(item) for item in sorted(DEFAULT_REFERENCE_BY_PHASE))
        raise ValueError(f"Unknown phase {phase}. Known phases: {known_phases}.")
    if phase == 2:
        resolved_hypothesis: str | None = resolve_hypothesis(phase, hypothesis)
        if resolved_hypothesis is None:
            raise ValueError("Phase 2 requires a hypothesis.")
        default_model, default_feature_set = DEFAULT_REFERENCE_BY_HYPOTHESIS[resolved_hypothesis]
    else:
        resolve_hypothesis(phase, hypothesis)
        default_model, default_feature_set = DEFAULT_REFERENCE_BY_PHASE[phase]
    model_name: str = default_model if reference_model is None else reference_model
    if reference_feature_set is not None:
        feature_set = reference_feature_set
    elif reference_model is None or model_name == default_model:
        feature_set = default_feature_set
    elif model_name in MODEL_DEFAULT_FEATURE_SET:
        feature_set = MODEL_DEFAULT_FEATURE_SET[model_name]
    else:
        known_models: str = ", ".join(sorted(MODEL_DEFAULT_FEATURE_SET))
        raise ValueError(
            f"Unknown reference model {model_name!r}. "
            f"Pass a feature set, or use one of: {known_models}."
        )
    return {"model": model_name, "feature_set": feature_set}


def resolve_hypothesis(phase: int, hypothesis: str | None = None) -> str | None:
    """Map a CLI hypothesis name to the tag stored on the run.

    Phase 1 has no hypothesis. Phase 2 defaults to H1 so the differential
    report stays reproducible. ``h2`` selects the season-defense hypothesis.
    ``h3`` selects the pass/rush split. ``h4`` selects clean EPA.
    ``ridge`` selects the L2 reference comparison.

    Args:
        phase: ``1`` or ``2``.
        hypothesis: Short or full hypothesis name, or None for the default.

    Returns:
        ``None`` for phase 1, otherwise ``h1-diff``,
        ``h2-season-def-regression``, ``h3-pass-rush-epa``,
        ``h4-clean-epa``, ``h5-starting-qb``, or ``ridge-reference``.

    Raises:
        ValueError: Phase 1 was given a hypothesis, or the name is unknown.
    """
    if phase == 1:
        if hypothesis is not None:
            raise ValueError("Phase 1 does not take a hypothesis. Omit --hypothesis.")
        return None
    if phase != 2:
        raise ValueError(f"Unknown phase {phase}. Known phases: 1, 2.")
    if hypothesis is None:
        return HYPOTHESIS_H1
    resolved: str | None = HYPOTHESIS_ALIASES.get(hypothesis.strip().lower())
    if resolved is None:
        known: str = ", ".join(sorted(HYPOTHESIS_ALIASES))
        raise ValueError(f"Unknown hypothesis {hypothesis!r}. Known hypotheses: {known}.")
    return resolved


def logistic_feature_sets_for(phase: int, hypothesis: str | None = None) -> tuple[str, ...]:
    """Return the logistic feature sets scored by a phase.

    Phase 1 scores ``current`` and ``with_games_played``. Phase 2 H1 scores
    ``current``, ``diff``, and ``diff_only``. Phase 2 H2 scores ``current``,
    ``h2_def``, and ``h2_full``. Phase 2 H3 scores ``h2_full``, ``h3_split``,
    and ``h3_replace``. Phase 2 H4 scores ``h2_full``, ``h4_clean``, and
    ``h4_clean_sr``. Phase 2 H5 scores ``h2_full`` and ``h5_qb``.
    The ridge comparison's logistic row is ``h2_full``
    only. Its L2 rows are returned by ``cv_model_specs_for``.

    Args:
        phase: ``1`` or ``2``.
        hypothesis: Phase 2 hypothesis name. None keeps H1.

    Returns:
        Feature set names in report order.

    Raises:
        ValueError: The phase or hypothesis is unknown.
    """
    if phase == 1:
        resolve_hypothesis(phase, hypothesis)
        return PHASE1_LOGISTIC_FEATURE_SETS
    if phase == 2:
        resolved: str | None = resolve_hypothesis(phase, hypothesis)
        if resolved == HYPOTHESIS_H2:
            return PHASE2_H2_LOGISTIC_FEATURE_SETS
        if resolved == HYPOTHESIS_H3:
            return PHASE2_H3_LOGISTIC_FEATURE_SETS
        if resolved == HYPOTHESIS_H4:
            return PHASE2_H4_LOGISTIC_FEATURE_SETS
        if resolved == HYPOTHESIS_H5:
            return PHASE2_H5_LOGISTIC_FEATURE_SETS
        if resolved == HYPOTHESIS_RIDGE:
            return (FEATURE_SET_H2_FULL,)
        return PHASE2_LOGISTIC_FEATURE_SETS
    raise ValueError(f"Unknown phase {phase}. Known phases: 1, 2.")


def cv_model_specs_for(phase: int, hypothesis: str | None = None) -> tuple[CvModelSpec, ...]:
    """Return the model and feature-set pairs a phase scores, before baselines.

    Phase 1 and hypotheses H1, H2, H3, H4, and H5 score fixed-C logistic regression
    on ``logistic_feature_sets_for``. The ridge comparison scores fixed-C
    logistic on ``h2_full``, then L2 logistic on ``h2_full``, ``h3_split``,
    and ``h3_replace``.

    Args:
        phase: ``1`` or ``2``.
        hypothesis: Phase 2 hypothesis name. None keeps H1.

    Returns:
        Specs in report order.

    Raises:
        ValueError: The phase or hypothesis is unknown.
    """
    resolved: str | None = resolve_hypothesis(phase, hypothesis)
    if resolved == HYPOTHESIS_RIDGE:
        return PHASE2_RIDGE_MODEL_SPECS
    return tuple(
        CvModelSpec(MODEL_LOGISTIC, feature_set)
        for feature_set in logistic_feature_sets_for(phase, hypothesis=hypothesis)
    )


def phase_tracking_settings(phase: int, hypothesis: str | None = None) -> dict[str, Any]:
    """MLflow experiment, tags, note, and default output directory for a phase.

    Args:
        phase: ``1`` or ``2``.
        hypothesis: Phase 2 hypothesis name. None keeps H1.

    Returns:
        Dict with ``experiment_name``, ``phase_tag``, ``hypothesis``,
        ``note``, ``output_dir``, and ``title``. ``hypothesis`` is None
        for phase 1.

    Raises:
        ValueError: The phase or hypothesis is unknown.
    """
    if phase == 1:
        resolve_hypothesis(phase, hypothesis)
        return {
            "experiment_name": PHASE1_EXPERIMENT_NAME,
            "phase_tag": PHASE1_TAG,
            "hypothesis": None,
            "note": PHASE1_NOTE,
            "output_dir": "reports/phase1",
            "title": "Phase 1 baselines",
        }
    if phase == 2:
        resolved: str | None = resolve_hypothesis(phase, hypothesis)
        if resolved == HYPOTHESIS_H2:
            return {
                "experiment_name": PHASE2_H2_EXPERIMENT_NAME,
                "phase_tag": PHASE2_TAG,
                "hypothesis": HYPOTHESIS_H2,
                "note": PHASE2_H2_NOTE,
                "output_dir": "reports/phase2_h2",
                "title": "Phase 2 H2 season defensive regression",
            }
        if resolved == HYPOTHESIS_H3:
            return {
                "experiment_name": PHASE2_H3_EXPERIMENT_NAME,
                "phase_tag": PHASE2_TAG,
                "hypothesis": HYPOTHESIS_H3,
                "note": PHASE2_H3_NOTE,
                "output_dir": "reports/phase2_h3",
                "title": "Phase 2 H3 pass and rush EPA",
            }
        if resolved == HYPOTHESIS_H4:
            return {
                "experiment_name": PHASE2_H4_EXPERIMENT_NAME,
                "phase_tag": PHASE2_TAG,
                "hypothesis": HYPOTHESIS_H4,
                "note": PHASE2_H4_NOTE,
                "output_dir": "reports/phase2_h4",
                "title": "Phase 2 H4 clean EPA",
            }
        if resolved == HYPOTHESIS_H5:
            return {
                "experiment_name": PHASE2_H5_EXPERIMENT_NAME,
                "phase_tag": PHASE2_TAG,
                "hypothesis": HYPOTHESIS_H5,
                "note": PHASE2_H5_NOTE,
                "output_dir": "reports/phase2_h5",
                "title": "Phase 2 H5 starting QB",
            }
        if resolved == HYPOTHESIS_RIDGE:
            return {
                "experiment_name": PHASE2_RIDGE_EXPERIMENT_NAME,
                "phase_tag": PHASE2_TAG,
                "hypothesis": HYPOTHESIS_RIDGE,
                "note": PHASE2_RIDGE_NOTE,
                "output_dir": "reports/phase2_ridge",
                "title": "Phase 2 ridge reference",
            }
        return {
            "experiment_name": PHASE2_EXPERIMENT_NAME,
            "phase_tag": PHASE2_TAG,
            "hypothesis": HYPOTHESIS_H1,
            "note": PHASE2_NOTE,
            "output_dir": "reports/phase2",
            "title": "Phase 2 H1 differential features",
        }
    raise ValueError(f"Unknown phase {phase}. Known phases: 1, 2.")


def resolve_cv_reference(
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
) -> dict[str, Any]:
    """Return the one result row that is the paired reference.

    Args:
        results: Model records from a rolling-origin run.
        reference_model: ``model`` value to match.
        reference_feature_set: ``feature_set`` value to match.

    Returns:
        The matching result dict.

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
    A skipped model keeps ``comparison`` as None.

    Args:
        results: Model records. Updated in place.
        reference_model: Model name of the reference.
        reference_feature_set: Feature set of the reference.
        n_bootstrap: Resamples passed to ``compare_oof_to_reference``.
        seed: Bootstrap seed.

    Returns:
        The reference result dict.

    Raises:
        ValueError: The reference is not one of ``results``.
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
    """Apply the H1 acceptance rule to one paired log loss difference.

    Accept only when pooled log loss is at least ``min_improvement`` lower
    than the reference and the 95% interval lies entirely below zero. That
    is how "the interval excludes 0" is applied to an improvement: the upper
    bound is negative. A missing interval is inconclusive. Every other
    scored comparison is rejected.

    Args:
        log_loss_diff: Candidate log loss minus reference log loss.
        ci_low: 2.5 percentile of the paired bootstrap.
        ci_high: 97.5 percentile of the paired bootstrap.
        min_improvement: Minimum log loss decrease required to accept.

    Returns:
        Dict with ``decision`` (``accepted``, ``rejected``, or
        ``inconclusive``), ``min_improvement``, ``log_loss_diff``,
        ``ci_low``, and ``ci_high``.
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
    """Label each CV row with the H1 rule. The reference row is not a candidate.

    Args:
        results: Model records that already have ``comparison`` attached.
        reference_model: Model name of the reference.
        reference_feature_set: Feature set of the reference.
        min_improvement: Passed to ``decide_log_loss_improvement``.

    Returns:
        One dict per result, with ``model``, ``feature_set``, and the
        decision fields. The reference decision is ``reference``.
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


def validate_feature_columns(
    frame: pd.DataFrame,
    feature_sets: tuple[str, ...],
    max_null_fraction: float = DEFAULT_MAX_NULL_FRACTION,
) -> dict[str, dict[str, float]]:
    """Refuse a requested feature set whose columns are missing or mostly null.

    The check runs on the feature frame before any fold is fit. A column
    that is absent, or whose null share is at least ``max_null_fraction``,
    fails the set. The default share is one half. Filling those nulls in
    the logistic imputer would score a constant, which is what happened
    when the clean EPA columns had not been rebuilt.

    Args:
        frame: One row per game, including every column the sets will use.
        feature_sets: Names understood by ``feature_columns_for``.
        max_null_fraction: Inclusive null-share limit in ``(0, 1]``.

    Returns:
        Dict of feature set to ``{column: null_fraction}`` for sets that
        passed. Fractions are in ``[0, 1]``.

    Raises:
        ValueError: The threshold is invalid, the frame has no rows, or a
            set has a missing or too-null column. The message names the
            feature set, the columns, and tells the reader to run
            ``make dbt-build-databricks``.
    """
    threshold: float = float(max_null_fraction)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("max_null_fraction must be in (0, 1]")
    if len(frame) == 0:
        raise ValueError("feature frame has no rows to validate")

    problems: list[str] = []
    passed: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    row_count: int = len(frame)
    for feature_set in feature_sets:
        if feature_set in seen:
            continue
        seen.add(feature_set)
        columns: tuple[str, ...] = feature_columns_for(feature_set)
        missing: list[str] = [column for column in columns if column not in frame.columns]
        null_columns: list[str] = []
        fractions: dict[str, float] = {}
        for column in columns:
            if column not in frame.columns:
                continue
            null_fraction: float = float(frame[column].isna().sum()) / float(row_count)
            fractions[column] = null_fraction
            if null_fraction >= threshold:
                null_columns.append(f"{column} ({null_fraction:.0%} null)")
        if missing or null_columns:
            missing_text: str = ", ".join(missing) if missing else "none"
            null_text: str = ", ".join(null_columns) if null_columns else "none"
            problems.append(
                f"{feature_set}: missing columns: {missing_text}. "
                f"Columns at or above {threshold:.0%} null: {null_text}."
            )
            continue
        passed[feature_set] = fractions
    if problems:
        detail: str = " ".join(problems)
        raise ValueError(
            "Feature columns are missing or too null to score. "
            f"{detail} Hint: {FEATURE_COLUMN_REBUILD_HINT}."
        )
    return passed


def _oof_for(
    results: list[dict[str, Any]], model_name: str, feature_set: str
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

    Args:
        features: Pregame frame with starter diagnostics.
        results: CV rows, including out-of-fold tables.
        reference_model: Reference model name.
        reference_feature_set: Reference feature set.
        candidate_feature_set: Feature set whose gap is described.
        n_bootstrap: Resamples for the paired interval.
        seed: Bootstrap seed.

    Returns:
        Dict with starter counts, ``n_qb_change_games``, ``status``
        (``scored`` or ``skipped``), ``reason``, and ``comparison``.
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


def run_phase1_baselines(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    config: RollingOriginConfig | None = None,
    lookback_games: int = 10,
    logistic_c: float = 1.0,
    logistic_max_iter: int = 1000,
    random_state: int = 42,
    logistic_feature_sets: tuple[str, ...] | None = None,
    reference_model: str | None = None,
    reference_feature_set: str | None = None,
    phase: int = 1,
    hypothesis: str | None = None,
    max_null_fraction: float = DEFAULT_MAX_NULL_FRACTION,
    expected_starters: pd.DataFrame | None = None,
    qb_dropbacks: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score logistic feature sets, the home prior, Elo, and the market.

    The market is skipped when the moneyline columns are absent or every
    line is missing. 2024 and 2025 are not scored. Phase 1 logs logistic
    ``current`` and ``with_games_played`` and pairs every model to the home
    prior. Phase 2 H1 logs logistic ``current``, ``diff``, and ``diff_only``.
    Phase 2 H2 logs logistic ``current``, ``h2_def``, and ``h2_full``.
    Phase 2 H3 logs logistic ``h2_full``, ``h3_split``, and ``h3_replace``.
    Phase 2 H4 logs logistic ``h2_full``, ``h4_clean``, and ``h4_clean_sr``.
    Phase 2 H5 logs logistic ``h2_full`` and ``h5_qb``.
    The ridge comparison logs logistic ``h2_full`` and ``logistic_l2`` on
    ``h2_full``, ``h3_split``, and ``h3_replace``. ``logistic_l2`` chooses
    C inside each training fold. H1 and H2 pair every model to logistic
    ``current``. H3, H4, H5, and ridge pair every model to logistic ``h2_full``.
    ``reference_model`` and ``reference_feature_set`` override that default.
    The phase 1 call with no extra arguments is unchanged.

    Args:
        games: Completed games, one row per game.
        team_games: Team-game EPA used to build the logistic matrices.
        config: Rolling-origin window. Defaults to 2016-2023.
        lookback_games: Rolling window passed to ``build_pregame_features``.
        logistic_c: Inverse regularization strength.
        logistic_max_iter: Logistic solver iteration cap.
        random_state: Seed for the logistic fit and the stored config.
        logistic_feature_sets: Logistic sets to score. None uses the phase default.
        reference_model: Reference model name. None uses the phase default.
        reference_feature_set: Reference feature set. None uses the phase default.
        phase: ``1`` keeps the phase 1 report. ``2`` is a phase 2 hypothesis.
        hypothesis: ``h1``, ``h2``, ``h3``, ``h4``, ``h5``, or ``ridge``. None on phase 2 keeps H1.
        max_null_fraction: Fail a requested column at or above this null share.
        expected_starters: Depth-chart QB1 table for H5. None leaves QB columns missing.
        qb_dropbacks: Passer dropbacks for H5. None leaves QB columns missing.

    Returns:
        Dict with ``results``, ``comparison_table``, ``markdown``, ``config``,
        ``reference_model``, and ``reference_feature_set``.

    Raises:
        ValueError: The phase, a feature set, or the reference cannot be used.
    """
    resolved: RollingOriginConfig = make_rolling_origin_config(
        **(config._asdict() if config is not None else {})
    )
    reference_spec: dict[str, str] = resolve_reference_spec(
        phase,
        reference_model=reference_model,
        reference_feature_set=reference_feature_set,
        hypothesis=hypothesis,
    )
    if logistic_feature_sets is None:
        model_specs: tuple[CvModelSpec, ...] = cv_model_specs_for(phase, hypothesis=hypothesis)
    else:
        model_specs = tuple(
            CvModelSpec(MODEL_LOGISTIC, feature_set_name)
            for feature_set_name in logistic_feature_sets
        )
    for spec in model_specs:
        feature_columns_for(spec.feature_set)
    tracking: dict[str, Any] = phase_tracking_settings(phase, hypothesis=hypothesis)
    features: pd.DataFrame = build_pregame_features(
        games,
        team_games,
        lookback_games=lookback_games,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
    )
    requested_sets: list[str] = []
    for spec in model_specs:
        if spec.feature_set in FEATURE_SET_COLUMNS and spec.feature_set not in requested_sets:
            requested_sets.append(spec.feature_set)
    if (
        reference_spec["feature_set"] in FEATURE_SET_COLUMNS
        and reference_spec["feature_set"] not in requested_sets
    ):
        requested_sets.append(reference_spec["feature_set"])
    validate_feature_columns(
        features,
        tuple(requested_sets),
        max_null_fraction=max_null_fraction,
    )
    shared_params: dict[str, Any] = {
        "lookback_games": lookback_games,
        "cv_start_season": resolved.cv_start_season,
        "cv_end_season": resolved.cv_end_season,
        "n_bootstrap": resolved.n_bootstrap,
        "bootstrap_seed": resolved.bootstrap_seed,
        "random_state": random_state,
        "reference_model": reference_spec["model"],
        "reference_feature_set": reference_spec["feature_set"],
    }
    if tracking["hypothesis"] in {
        HYPOTHESIS_H2,
        HYPOTHESIS_H3,
        HYPOTHESIS_H4,
        HYPOTHESIS_H5,
        HYPOTHESIS_RIDGE,
    }:
        shared_params["shrink_factor"] = SEASON_SHRINK_FACTOR
        shared_params["blend_k"] = SEASON_BLEND_K
    if tracking["hypothesis"] == HYPOTHESIS_H5:
        shared_params["qb_prior_dropbacks"] = QB_PRIOR_DROPBACKS
        shared_params["replacement_epa_per_dropback"] = REPLACEMENT_EPA_PER_DROPBACK
    results: list[dict[str, Any]] = []

    current_matrix, current_target = prepare_training_data(
        games,
        features,
        feature_set=FEATURE_SET_CURRENT,
    )
    baseline_cv: dict[str, Any] = rolling_origin_cv(
        current_matrix,
        current_target,
        _baseline_fold_predictor(),
        resolved,
    )
    baseline_reason: str | None = None if not baseline_cv["oof"].empty else "no_scored_folds"
    results.append(
        _cv_result_record(
            REFERENCE_MODEL_NAME,
            "none",
            "ok" if baseline_reason is None else "skipped",
            baseline_reason,
            {**shared_params, "model_type": REFERENCE_MODEL_NAME, "features": "none"},
            baseline_cv,
        )
    )

    for spec in model_specs:
        matrix, target = prepare_training_data(games, features, feature_set=spec.feature_set)
        if spec.model == MODEL_LOGISTIC_L2:
            fold_predictor: FoldPredictor = _logistic_l2_fold_predictor(
                DEFAULT_L2_C_GRID,
                logistic_c,
                logistic_max_iter,
                random_state,
            )
        elif spec.model == MODEL_LOGISTIC:
            fold_predictor = _logistic_fold_predictor(logistic_c, logistic_max_iter, random_state)
        else:
            raise ValueError(f"Unknown CV model {spec.model!r}.")
        logistic_cv: dict[str, Any] = rolling_origin_cv(
            matrix,
            target,
            fold_predictor,
            resolved,
        )
        logistic_reason: str | None = None if not logistic_cv["oof"].empty else "no_scored_folds"
        columns: str = ",".join(feature_columns_for(spec.feature_set))
        model_params: dict[str, Any] = {
            **shared_params,
            "model_type": spec.model,
            "feature_set": spec.feature_set,
            "features": columns,
            "max_iter": logistic_max_iter,
            "regularization": "l2",
        }
        if spec.model == MODEL_LOGISTIC_L2:
            model_params["c_grid"] = ",".join(
                _format_regularization_c(value) for value in DEFAULT_L2_C_GRID
            )
            model_params["c_fallback"] = logistic_c
            model_params.update(_chosen_c_param_fields(logistic_cv["folds"]))
        else:
            model_params["C"] = logistic_c
        results.append(
            _cv_result_record(
                spec.model,
                spec.feature_set,
                "ok" if logistic_reason is None else "skipped",
                logistic_reason,
                model_params,
                logistic_cv,
            )
        )

    elo_cv: dict[str, Any] = rolling_origin_cv(
        current_matrix,
        current_target,
        _elo_fold_predictor(games),
        resolved,
    )
    elo_reason: str | None = None if not elo_cv["oof"].empty else "no_scored_folds"
    results.append(
        _cv_result_record(
            "elo",
            "elo",
            "ok" if elo_reason is None else "skipped",
            elo_reason,
            {
                **shared_params,
                "model_type": "elo",
                "feature_set": "elo",
                "features": "elo_rating",
                "elo_k": ELO_K,
                "elo_home_advantage": ELO_HOME_ADVANTAGE,
                "elo_mean_rating": ELO_MEAN_RATING,
                "elo_season_reversion": ELO_SEASON_REVERSION,
                "elo_scale": ELO_SCALE,
            },
            elo_cv,
        )
    )

    market: dict[str, Any] = market_home_probabilities(games)
    if market["status"] != "ok":
        results.append(
            _cv_result_record(
                "market",
                "market",
                "skipped",
                str(market["reason"]),
                {
                    **shared_params,
                    "model_type": "market",
                    "feature_set": "market",
                    "features": "no_vig_closing_moneyline",
                },
                None,
            )
        )
    else:
        market_probability: pd.Series = market["probabilities"]
        kept: pd.Series = market_probability.notna()
        market_frame: pd.DataFrame = pd.DataFrame(
            {
                "game_id": games.loc[kept, "game_id"].to_numpy(),
                "gameday": games.loc[kept, "gameday"].to_numpy(),
                "season": games.loc[kept, "season"].to_numpy(),
            }
        )
        market_target: pd.Series = pd.Series(games.loc[kept, "home_won"].to_numpy())
        by_game: pd.Series = pd.Series(
            market_probability.loc[kept].to_numpy(dtype=float),
            index=market_frame["game_id"].to_numpy(),
        )
        market_cv: dict[str, Any] = rolling_origin_cv(
            market_frame,
            market_target,
            _lookup_fold_predictor(by_game),
            resolved,
        )
        market_reason: str | None = None if not market_cv["oof"].empty else "no_scored_folds"
        results.append(
            _cv_result_record(
                "market",
                "market",
                "ok" if market_reason is None else "skipped",
                market_reason,
                {
                    **shared_params,
                    "model_type": "market",
                    "feature_set": "market",
                    "features": "no_vig_closing_moneyline",
                    "n_missing_lines": market["n_missing_lines"],
                },
                market_cv,
            )
        )

    attach_reference_comparison(
        results,
        reference_spec["model"],
        reference_spec["feature_set"],
        n_bootstrap=resolved.n_bootstrap,
        seed=resolved.bootstrap_seed,
    )

    comparison_table: pd.DataFrame = _comparison_table(results)
    apply_acceptance: bool = phase == 2
    qb_subset: dict[str, Any] | None = None
    if tracking["hypothesis"] == HYPOTHESIS_H5:
        qb_subset = qb_change_subset_comparison(
            features,
            results,
            reference_spec["model"],
            reference_spec["feature_set"],
            FEATURE_SET_H5_QB,
            n_bootstrap=resolved.n_bootstrap,
            seed=resolved.bootstrap_seed,
        )
    report: dict[str, Any] = {
        "results": results,
        "comparison_table": comparison_table,
        "config": resolved._asdict(),
        "lookback_games": lookback_games,
        "reference_model": reference_spec["model"],
        "reference_feature_set": reference_spec["feature_set"],
        "phase": phase,
        "hypothesis": tracking["hypothesis"],
        "title": tracking["title"],
        "apply_acceptance": apply_acceptance,
        "min_log_loss_improvement": MIN_LOG_LOSS_IMPROVEMENT,
        "markdown": "",
        "dataset_frame": _cv_dataset_frame(games, features, resolved.cv_end_season),
        "dataset_source": (
            "fct_games joined to pregame features, "
            f"seasons through {resolved.cv_end_season}, lookback {lookback_games}"
        ),
        "qb_change_subset": qb_subset,
    }
    if apply_acceptance:
        report["acceptance"] = acceptance_decisions(
            results,
            reference_spec["model"],
            reference_spec["feature_set"],
        )
    report["markdown"] = render_phase1_report(report)
    return report


def _comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per model with pooled metrics and the paired log loss gap."""
    rows: list[dict[str, Any]] = []
    for result in results:
        pooled: dict[str, Any] = result["pooled"] or {}
        comparison: dict[str, Any] = result["comparison"] or {}
        rows.append(
            {
                "model": result["model"],
                "feature_set": result["feature_set"],
                "status": result["status"],
                "reason": result["reason"],
                "n": pooled.get("n") if result["pooled"] else comparison.get("n"),
                "log_loss": pooled.get("log_loss"),
                "brier": pooled.get("brier"),
                "auc": pooled.get("auc"),
                "accuracy": pooled.get("accuracy"),
                "log_loss_diff": comparison.get("log_loss_diff"),
                "log_loss_diff_ci_low": comparison.get("log_loss_diff_ci_low"),
                "log_loss_diff_ci_high": comparison.get("log_loss_diff_ci_high"),
            }
        )
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def _format_metric(value: object) -> str:
    """Render one report cell. Missing values become ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        return str(value)
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a GitHub-flavored table."""
    head: str = "| " + " | ".join(headers) + " |"
    separator: str = "| " + " | ".join("---" for _ in headers) + " |"
    body: list[str] = [
        "| " + " | ".join(_format_metric(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([head, separator, *body])


def _render_acceptance_section(report: dict[str, Any]) -> list[str]:
    """Render the H1 rule and the per-row decision. Numbers come from ``report``."""
    minimum: float = float(report.get("min_log_loss_improvement", MIN_LOG_LOSS_IMPROVEMENT))
    rows: list[dict[str, Any]] = report.get("acceptance") or []
    lines: list[str] = [
        "## Acceptance",
        "",
        "Accept a feature set only when both are true:",
        f"pooled log loss improves by at least {minimum:.3f}",
        f"(`log_loss_diff` <= -{minimum:.3f}) and the 95% paired interval",
        "excludes 0 by lying entirely below 0 (`log_loss_diff_ci_high` < 0).",
        "Otherwise a scored row is rejected. A row without an interval is",
        "inconclusive. The reference row is not a candidate.",
        "",
    ]
    if not rows:
        lines.append("No acceptance rows were computed.")
        lines.append("")
        return lines
    table_rows: list[list[object]] = [
        [
            row["model"],
            row["feature_set"],
            row["log_loss_diff"],
            row["ci_low"],
            row["ci_high"],
            row["decision"],
        ]
        for row in rows
    ]
    lines.append(
        _markdown_table(
            ["model", "feature_set", "log_loss_diff", "ci_low", "ci_high", "decision"],
            table_rows,
        )
    )
    lines.append("")
    return lines


def _format_optional_float(value: float | None) -> str:
    """Format a metric, or ``missing`` when the interval was not computed."""
    if value is None or pd.isna(value):
        return "missing"
    return f"{float(value):.4f}"


def _qb_change_subset_lines(subset: dict[str, Any]) -> list[str]:
    """Markdown for the descriptive QB-change slice. Not an acceptance rule."""
    lines: list[str] = [
        "## QB change subset",
        "",
        "Games where the depth-chart starter differs from the previous primary passer.",
        "This section is descriptive. It is not an acceptance decision.",
        f"Depth-chart starter sides: {subset['n_depth_chart_starters']}.",
        f"Previous-passer fallback sides: {subset['n_previous_passer_starters']}.",
        f"Games with a QB change: {subset['n_qb_change_games']}.",
    ]
    if subset["status"] != "scored" or subset["comparison"] is None:
        reason: str = str(subset["reason"] or "not_scored")
        lines.append(f"Subset comparison skipped ({reason}).")
    else:
        comparison: dict[str, Any] = subset["comparison"]
        lines.append(
            "On "
            f"{comparison['n']} out-of-fold games with a QB change, "
            f"h5_qb log loss is {_format_optional_float(comparison['log_loss'])} "
            f"and the reference is {_format_optional_float(comparison['reference_log_loss'])}. "
            "The difference is "
            f"{_format_optional_float(comparison['log_loss_diff'])} "
            "(interval "
            f"{_format_optional_float(comparison['log_loss_diff_ci_low'])} to "
            f"{_format_optional_float(comparison['log_loss_diff_ci_high'])})."
        )
    lines.append("")
    return lines


def render_phase1_report(report: dict[str, Any]) -> str:
    """Render the rolling-origin comparison. Numbers come only from ``report``."""
    config: dict[str, Any] = report["config"]
    table: pd.DataFrame = report["comparison_table"]
    title: str = str(report.get("title") or "Phase 1 baselines")
    reference_feature_set: str = str(report.get("reference_feature_set") or "none")
    if reference_feature_set == "none":
        reference_line: str = f"Reference model: `{report['reference_model']}`."
    else:
        reference_line = (
            f"Reference model: `{report['reference_model']}` "
            f"on feature set `{reference_feature_set}`."
        )
    lines: list[str] = [
        f"# {title}",
        "",
        "Rolling-origin CV. For each season S from",
        f"{config['cv_start_season']} through {config['cv_end_season']}, a model",
        "trains on seasons strictly before S and scores S.",
        f"{FINAL_VALIDATION_SEASON} is the final validation and is not scored.",
        f"{HELD_OUT_TEST_SEASON} is not scored.",
        "",
        reference_line,
        "`log_loss_diff` is candidate log loss minus reference log loss on the",
        "same out-of-fold games. A negative difference favors the candidate.",
        "The interval is the 2.5 and 97.5 percentiles of a paired bootstrap",
        f"({config['n_bootstrap']} resamples, seed {config['bootstrap_seed']}).",
        "Accuracy treats a probability of 0.5 or higher as a home win.",
        "",
        "Elo uses K "
        f"{ELO_K:g}, home advantage {ELO_HOME_ADVANTAGE:g}, mean rating "
        f"{ELO_MEAN_RATING:g}, and season reversion {ELO_SEASON_REVERSION:.3f} "
        f"on a scale of {ELO_SCALE:g}. There is no margin-of-victory multiplier.",
        "The market probability is the no-vig closing moneyline. It is skipped",
        "when those columns are absent.",
        "",
        f"Lookback games: {report['lookback_games']}.",
        "",
        "## Comparison",
        "",
        _markdown_table(
            list(COMPARISON_COLUMNS),
            [list(row) for row in table.itertuples(index=False, name=None)],
        ),
        "",
        "## Folds",
        "",
    ]
    subset: dict[str, Any] | None = report.get("qb_change_subset")
    if subset is not None:
        fold_at: int = lines.index("## Folds")
        lines[fold_at:fold_at] = _qb_change_subset_lines(subset)
    if report.get("hypothesis") == HYPOTHESIS_H4:
        regression_note = [
            "Clean EPA keeps pass and run plays only.",
            "no_play, kneels, spikes, and garbage time are excluded.",
            "Garbage time is pre-snap wp below 0.05 or above 0.95.",
            "A missing wp is kept. Those cutoffs are fixed.",
            "Recent, season, and regressed clean EPA use the same windows",
            f"as h2_full, shrunk by {SEASON_SHRINK_FACTOR:.3f} toward the",
            "clean league mean, then blended with weight",
            f"n/(n+{SEASON_BLEND_K:g}).",
            "h4_clean_sr adds the recent clean success rate.",
            "A missing success rate is filled with 0.5.",
        ]
        lookback_index: int = lines.index(f"Lookback games: {report['lookback_games']}.")
        lines[lookback_index + 1 : lookback_index + 1] = regression_note
    elif report.get("hypothesis") == HYPOTHESIS_H5:
        regression_note = [
            "The starter is the offense QB1 on that week's depth chart.",
            "A missing chart uses the previous game's primary passer.",
            "The rating is EPA per dropback, shrunk toward replacement 0",
            f"with weight n/(n+{QB_PRIOR_DROPBACKS:g}).",
            "Only dropbacks from earlier gamedays count, across teams and seasons.",
            "h5_qb is h2_full plus the home and away ratings and each rating",
            "minus that team's recent passing EPA.",
            "CPOE is stored on the dropback mart and is not in the rating.",
        ]
        lookback_index = lines.index(f"Lookback games: {report['lookback_games']}.")
        lines[lookback_index + 1 : lookback_index + 1] = regression_note
    elif report.get("hypothesis") in {HYPOTHESIS_H2, HYPOTHESIS_H3, HYPOTHESIS_RIDGE}:
        regression_note = [
            "Season defensive EPA is the same-season prior mean of EPA allowed.",
            "Regressed EPA starts from the previous season's final value,",
            f"shrunk by {SEASON_SHRINK_FACTOR:.3f} toward that season's league mean,",
            "then blends in current-season games with weight",
            f"n/(n+{SEASON_BLEND_K:g}). Week 1 of a season with no prior stays missing.",
        ]
        if report.get("hypothesis") in {HYPOTHESIS_H3, HYPOTHESIS_RIDGE}:
            regression_note.extend(
                [
                    "Pass and rush EPA per play split offense and defense.",
                    "Recent splits use the lookback window on prior games,",
                    "including prior seasons. Regressed-season splits use the",
                    "same shrinkage and blend. Defensive splits are the",
                    "opponent's offensive pass or rush rate from that game.",
                ]
            )
        if report.get("hypothesis") == HYPOTHESIS_RIDGE:
            grid_text: str = ", ".join(
                _format_regularization_c(value) for value in DEFAULT_L2_C_GRID
            )
            regression_note.extend(
                [
                    "logistic_l2 is L2 logistic regression. C is chosen inside",
                    "each training fold by an expanding season split on those",
                    "training seasons only. The scored season is not used to",
                    f"pick C. The grid is {grid_text}.",
                    "A fold with no usable inner season keeps the fallback C.",
                    "The fold table records chosen_c.",
                ]
            )
        lookback_index: int = lines.index(f"Lookback games: {report['lookback_games']}.")
        lines[lookback_index + 1 : lookback_index + 1] = regression_note
    if report.get("hypothesis"):
        lines.insert(2, f"Hypothesis: `{report['hypothesis']}`.")
        lines.insert(3, "")
    for result in report["results"]:
        lines.append(f"### {result['model']} ({result['feature_set']})")
        lines.append("")
        if result["status"] != "ok":
            lines.append(f"Skipped ({result['reason']}).")
            lines.append("")
            if not result["folds"]:
                continue
        records_chosen_c: bool = any("chosen_c" in fold for fold in result["folds"])
        fold_headers: list[str] = [
            "season",
            "status",
            "reason",
            "n_train",
            "n_eval",
            "train_seasons",
            "log_loss",
        ]
        if records_chosen_c:
            fold_headers.extend(["chosen_c", "c_selection"])
        fold_rows: list[list[object]] = []
        for fold in result["folds"]:
            fold_row: list[object] = [
                fold["season"],
                fold["status"],
                fold["reason"],
                fold["n_train"],
                fold["n_eval"],
                ",".join(str(season) for season in fold["train_seasons"]),
                fold["log_loss"],
            ]
            if records_chosen_c:
                fold_row.extend([fold.get("chosen_c"), fold.get("c_selection")])
            fold_rows.append(fold_row)
        lines.append(_markdown_table(fold_headers, fold_rows))
        lines.append("")
    if report.get("apply_acceptance"):
        lines.extend(_render_acceptance_section(report))
    lines.append("No model is promoted from this report.")
    lines.append("")
    return "\n".join(lines)


def write_phase1_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write the comparison markdown, the comparison CSV, and per-model OOF files.

    Args:
        report: Result of ``run_phase1_baselines``.
        output_dir: Directory to create if needed.

    Returns:
        Dict with ``markdown_path``, ``csv_path``, and ``oof_paths``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path: Path = output_dir / PHASE1_MARKDOWN_NAME
    csv_path: Path = output_dir / PHASE1_CSV_NAME
    markdown_path.write_text(report["markdown"], encoding="utf-8")
    table: pd.DataFrame = report["comparison_table"]
    table.to_csv(csv_path, index=False)
    oof_paths: dict[str, str] = {}
    for result in report["results"]:
        oof: pd.DataFrame | None = result["oof"]
        if oof is None:
            continue
        file_name: str = f"oof_{result['model']}_{result['feature_set']}.csv"
        oof_path: Path = output_dir / file_name
        oof.to_csv(oof_path, index=False)
        oof_paths[f"{result['model']}|{result['feature_set']}"] = str(oof_path)
    return {
        "markdown_path": str(markdown_path),
        "csv_path": str(csv_path),
        "oof_paths": oof_paths,
    }


def _cv_dataset_frame(
    games: pd.DataFrame,
    features: pd.DataFrame,
    cv_end_season: int,
) -> pd.DataFrame:
    """Rows that can enter rolling-origin CV, in a stable column order.

    Seasons after ``cv_end_season`` stay out, so 2024 and 2025 are not part
    of the digest when the window ends in 2023. The frame is the full
    feature matrix plus the label, not one feature set.

    Args:
        games: Game table with ``game_id``, ``gameday``, ``season``, and ``home_won``.
        features: Output of ``build_pregame_features``.
        cv_end_season: Last season included in the CV window.

    Returns:
        One row per game with season at or before ``cv_end_season``, sorted
        by gameday and game id.
    """
    labels: pd.DataFrame = games.loc[:, ["game_id", "gameday", "season", "home_won"]].copy()
    merged: pd.DataFrame = features.merge(labels, on="game_id", how="inner")
    kept: pd.DataFrame = merged.loc[merged["season"] <= cv_end_season].copy()
    ordered: pd.DataFrame = kept.sort_values(["gameday", "game_id"], kind="mergesort")
    id_columns: list[str] = ["game_id", "gameday", "season", "home_won"]
    feature_columns: list[str] = [column for column in features.columns if column != "game_id"]
    return ordered.loc[:, id_columns + feature_columns].reset_index(drop=True)


def parent_run_name(
    phase: int,
    hypothesis: str | None,
    timestamp: datetime | None = None,
) -> str:
    """Name the parent run for one CV invocation.

    Args:
        phase: ``1`` or ``2``.
        hypothesis: Resolved hypothesis tag, or None for phase 1.
        timestamp: Clock used in the name. Defaults to the current UTC time.

    Returns:
        A name such as ``phase1-<timestamp>`` or ``phase2-h3-<timestamp>``.
    """
    clock: datetime = timestamp if timestamp is not None else datetime.now(timezone.utc)
    stamp: str = clock.strftime("%Y%m%dT%H%M%SZ")
    if phase != 2 or hypothesis is None:
        return f"phase1-{stamp}"
    short_names: dict[str, str] = {
        HYPOTHESIS_H1: "h1",
        HYPOTHESIS_H2: "h2",
        HYPOTHESIS_H3: "h3",
        HYPOTHESIS_H4: "h4",
        HYPOTHESIS_H5: "h5",
        HYPOTHESIS_RIDGE: "ridge",
    }
    short: str | None = short_names.get(hypothesis)
    if short is None:
        return f"phase2-{stamp}"
    return f"phase2-{short}-{stamp}"


def mlflow_decision_tag(
    result: dict[str, Any],
    reference_model: str,
    reference_feature_set: str,
) -> str:
    """Tag one scored model with the acceptance rule.

    The reference row is ``reference``. Home prior, Elo, and the market are
    ``baseline`` unless that model is the reference. A logistic candidate is
    ``accepted`` only when pooled log loss improves by at least 0.003 and
    the 95% interval lies entirely below 0. A missing interval is
    ``inconclusive``. Every other candidate is ``rejected``.

    Args:
        result: One CV result dict with ``comparison`` attached.
        reference_model: Model name of the reference.
        reference_feature_set: Feature set of the reference.

    Returns:
        ``accepted``, ``rejected``, ``inconclusive``, ``reference``, or ``baseline``.
    """
    if result["model"] == reference_model and result["feature_set"] == reference_feature_set:
        return "reference"
    if result["model"] in BASELINE_MODEL_NAMES:
        return "baseline"
    comparison: dict[str, Any] = result.get("comparison") or {}
    labeled: dict[str, Any] = decide_log_loss_improvement(
        comparison.get("log_loss_diff"),
        comparison.get("log_loss_diff_ci_low"),
        comparison.get("log_loss_diff_ci_high"),
    )
    return str(labeled["decision"])


def select_best_feature_set(
    results: list[dict[str, Any]],
    reference_model: str,
    reference_feature_set: str,
) -> str:
    """Choose the feature set recorded on the parent run.

    An accepted logistic set wins. If more than one is accepted, the lowest
    pooled log loss wins. If none is accepted, the reference feature set is
    kept.

    Args:
        results: CV result dicts with ``comparison`` and ``pooled`` attached.
        reference_model: Model name of the reference.
        reference_feature_set: Feature set of the reference.

    Returns:
        Feature set name, or ``none`` when the reference has no feature set
        and no candidate was accepted.
    """
    decisions: list[dict[str, Any]] = acceptance_decisions(
        results,
        reference_model,
        reference_feature_set,
    )
    scored_losses: list[tuple[str, str, float]] = []
    for result in results:
        if result["model"] not in SCORED_FEATURE_MODELS or result["status"] != "ok":
            continue
        pooled: dict[str, Any] = result.get("pooled") or {}
        value: object = pooled.get("log_loss")
        if value is None or pd.isna(value):
            continue
        scored_losses.append((str(result["model"]), str(result["feature_set"]), float(value)))
    accepted_keys: set[tuple[str, str]] = {
        (str(row["model"]), str(row["feature_set"]))
        for row in decisions
        if row["decision"] == "accepted"
    }
    accepted: list[tuple[str, str, float]] = [
        item for item in scored_losses if (item[0], item[1]) in accepted_keys
    ]
    if not accepted:
        return reference_feature_set
    return min(accepted, key=lambda item: item[2])[1]


def _description_dataset_source(description: str) -> Any:
    """Dataset source that stores a short description of the CV matrix.

    A plain sentence has no URI scheme, so MLflow would otherwise warn and
    treat it as a local file path. The local source class still holds that
    sentence, and passing the instance skips the ambiguous resolver.

    Args:
        description: Where the rows came from, in one sentence.

    Returns:
        A local MLflow dataset source.

    Raises:
        ValueError: MLflow did not register a local dataset source.
    """
    from mlflow.data.dataset_source_registry import get_registered_sources

    local_sources: list[Any] = [
        source_cls
        for source_cls in get_registered_sources()
        if source_cls._get_source_type() == "local"
    ]
    if not local_sources:
        raise ValueError("MLflow did not register a local dataset source")
    return local_sources[-1](description)


def _cv_pandas_dataset(frame: pd.DataFrame, source: str) -> Any:
    """Build one pandas dataset for the CV matrix.

    Args:
        frame: Dataset frame from ``_cv_dataset_frame``.
        source: Short description of where the rows came from.

    Returns:
        An ``mlflow.data`` pandas dataset named ``nfl_pregame_cv``.
    """
    return mlflow.data.from_pandas(
        frame,
        source=_description_dataset_source(source),
        name=CV_DATASET_NAME,
        targets="home_won",
    )


def _log_dataset_lineage(dataset: Any, n_rows: int, n_columns: int) -> str:
    """Log one CV dataset input and return its digest.

    Parent and children receive the same dataset object, so the digest matches.

    Args:
        dataset: Dataset from ``_cv_pandas_dataset``.
        n_rows: Row count of the matrix.
        n_columns: Column count of the matrix.

    Returns:
        The dataset digest stored on the ``data_digest`` tag.
    """
    digest: str = str(dataset.digest)
    mlflow.set_tag("data_digest", digest)
    mlflow.log_input(dataset, context="training")
    mlflow.log_params(
        {
            "data_n_rows": str(n_rows),
            "data_n_columns": str(n_columns),
        }
    )
    return digest


def _log_paired_difference_metrics(comparison: dict[str, Any] | None) -> None:
    """Log the paired log loss gap under the decision metric names."""
    if comparison is None:
        return
    for source_name in ("log_loss_diff", "log_loss_diff_ci_low", "log_loss_diff_ci_high"):
        metric_value: object = comparison.get(source_name)
        if metric_value is None or pd.isna(metric_value):
            continue
        mlflow.log_metric(source_name, float(metric_value))
        mlflow.log_metric(f"cv_{source_name}", float(metric_value))


def log_phase1_to_mlflow(
    report: dict[str, Any],
    paths: dict[str, Any],
    tracking_uri: str,
    experiment_name: str = PHASE1_EXPERIMENT_NAME,
    phase_tag: str = PHASE1_TAG,
    hypothesis: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Log one parent run and one nested child per scored model.

    The parent carries the comparison report, the dataset digest, and
    ``best_feature_set``. Each child is the same invocation with
    ``nested=True``. Phase 1 callers can omit the tag arguments. Phase 2
    H1 passes ``hypothesis`` ``h1-diff``. Phase 2 H2 passes
    ``h2-season-def-regression``. Phase 2 H3 passes ``h3-pass-rush-epa``.
    Phase 2 H4 passes ``h4-clean-epa``. The ridge comparison passes
    ``ridge-reference``. All of them use
    ``phase_tag`` ``2``.

    Args:
        report: Result of ``run_phase1_baselines``.
        paths: Paths from ``write_phase1_outputs``.
        tracking_uri: MLflow tracking URI, passed through ``resolve_tracking_uri``.
        experiment_name: Experiment to create or activate.
        phase_tag: Value of the ``phase`` tag.
        hypothesis: Value of the ``hypothesis`` tag. Omitted when None.
        note: Run note. The phase 1 note is used when this is None.

    Returns:
        Dict with ``run_ids`` (the children), ``parent_run_id``,
        ``parent_run_name``, ``data_digest``, ``best_feature_set``,
        ``experiment_name``, and ``tracking_uri``.
    """
    resolved_uri: str = resolve_tracking_uri(tracking_uri)
    activate_tracking_experiment(experiment_name, resolved_uri)
    reference_model: str = str(report["reference_model"])
    reference_feature_set: str = str(report["reference_feature_set"])
    best_feature_set: str = select_best_feature_set(
        report["results"],
        reference_model,
        reference_feature_set,
    )
    run_note: str = PHASE1_NOTE if note is None else note
    git_commit: str = get_git_sha()
    phase_value: int = int(report.get("phase") or phase_tag)
    parent_name: str = parent_run_name(phase_value, hypothesis)
    dataset_frame: pd.DataFrame | None = report.get("dataset_frame")
    dataset_source: str = str(
        report.get("dataset_source") or "pregame features joined to fct_games"
    )
    dataset: Any | None = None
    if dataset_frame is not None:
        dataset = _cv_pandas_dataset(dataset_frame, dataset_source)
    run_ids: list[dict[str, str]] = []
    with mlflow.start_run(run_name=parent_name) as parent_run:
        mlflow.set_tag("phase", phase_tag)
        if hypothesis is not None:
            mlflow.set_tag("hypothesis", hypothesis)
        mlflow.set_tag("cv_scheme", CV_SCHEME)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("reference_model", reference_model)
        mlflow.set_tag("reference_feature_set", reference_feature_set)
        mlflow.set_tag("best_feature_set", best_feature_set)
        mlflow.set_tag("mlflow.note.content", run_note)
        digest: str | None = None
        if dataset is not None and dataset_frame is not None:
            digest = _log_dataset_lineage(
                dataset,
                n_rows=len(dataset_frame),
                n_columns=int(dataset_frame.shape[1]),
            )
        mlflow.log_artifact(paths["markdown_path"])
        mlflow.log_artifact(paths["csv_path"])
        for result in report["results"]:
            if result["status"] != "ok":
                continue
            run_name: str = f"{result['model']}-{result['feature_set']}"
            with mlflow.start_run(run_name=run_name, nested=True) as active_run:
                mlflow.set_tag("phase", phase_tag)
                if hypothesis is not None:
                    mlflow.set_tag("hypothesis", hypothesis)
                mlflow.set_tag("feature_set", result["feature_set"])
                mlflow.set_tag("cv_scheme", CV_SCHEME)
                mlflow.set_tag("git_commit", git_commit)
                mlflow.set_tag(
                    "decision",
                    mlflow_decision_tag(result, reference_model, reference_feature_set),
                )
                mlflow.set_tag("mlflow.note.content", run_note)
                if dataset is not None and dataset_frame is not None:
                    _log_dataset_lineage(
                        dataset,
                        n_rows=len(dataset_frame),
                        n_columns=int(dataset_frame.shape[1]),
                    )
                params: dict[str, Any] = {
                    key: value
                    for key, value in result["params"].items()
                    if value is not None and not isinstance(value, dict | list)
                }
                params["model_type"] = result["model"]
                params["feature_set"] = result["feature_set"]
                mlflow.log_params(params)
                pooled: dict[str, Any] = result["pooled"] or {}
                for metric_name in ("log_loss", "brier", "auc", "accuracy"):
                    metric_value = pooled.get(metric_name)
                    if metric_value is None or pd.isna(metric_value):
                        continue
                    mlflow.log_metric(f"cv_{metric_name}", float(metric_value))
                if pooled.get("n") is not None:
                    mlflow.log_metric("cv_n", float(pooled["n"]))
                _log_paired_difference_metrics(result.get("comparison"))
                scored_folds: int = sum(1 for fold in result["folds"] if fold["status"] == "scored")
                skipped_folds: int = sum(
                    1 for fold in result["folds"] if fold["status"] != "scored"
                )
                mlflow.log_metric("cv_folds_scored", float(scored_folds))
                mlflow.log_metric("cv_folds_skipped", float(skipped_folds))
                oof_key: str = f"{result['model']}|{result['feature_set']}"
                oof_path: str | None = paths["oof_paths"].get(oof_key)
                if oof_path is not None:
                    mlflow.log_artifact(oof_path)
                run_ids.append(
                    {
                        "model": result["model"],
                        "feature_set": result["feature_set"],
                        "run_id": active_run.info.run_id,
                    }
                )
        parent_run_id: str = parent_run.info.run_id
    return {
        "run_ids": run_ids,
        "parent_run_id": parent_run_id,
        "parent_run_name": parent_name,
        "data_digest": digest,
        "best_feature_set": best_feature_set,
        "experiment_name": experiment_name,
        "tracking_uri": resolved_uri,
    }


def _run_phase1_cli(args: argparse.Namespace) -> None:
    """Fit a rolling-origin CV phase, write the report, and log MLflow."""
    config: RollingOriginConfig = make_rolling_origin_config(
        cv_start_season=args.cv_start_season,
        cv_end_season=args.cv_end_season,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    settings: dict[str, Any] = phase_tracking_settings(args.phase, hypothesis=args.hypothesis)
    output_dir: str = args.output_dir
    if args.phase != 1 and args.output_dir == "reports/phase1":
        output_dir = str(settings["output_dir"])
    logger.info(
        "Running phase %s rolling-origin CV for seasons %s through %s",
        args.phase,
        config.cv_start_season,
        config.cv_end_season,
    )
    fixture_path: str | None = args.fixture_path if args.fixture_path else None
    games, team_games = load_data(fixture_path=fixture_path)
    expected_starters: pd.DataFrame | None = None
    qb_dropbacks: pd.DataFrame | None = None
    if settings["hypothesis"] == HYPOTHESIS_H5:
        expected_starters, qb_dropbacks = load_qb_inputs(fixture_path)
    report: dict[str, Any] = run_phase1_baselines(
        games,
        team_games,
        config=config,
        lookback_games=args.lookback,
        logistic_c=args.C,
        logistic_max_iter=args.max_iter,
        random_state=args.seed,
        reference_model=args.reference_model,
        reference_feature_set=args.reference_feature_set,
        phase=args.phase,
        hypothesis=args.hypothesis,
        max_null_fraction=args.max_null_fraction,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
    )
    paths: dict[str, Any] = write_phase1_outputs(report, Path(output_dir))
    tracking_uri: str = resolve_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    )
    logged: dict[str, Any] = log_phase1_to_mlflow(
        report,
        paths,
        tracking_uri,
        experiment_name=str(settings["experiment_name"]),
        phase_tag=str(settings["phase_tag"]),
        hypothesis=settings["hypothesis"],
        note=str(settings["note"]),
    )
    logger.info("Wrote %s and %s", paths["markdown_path"], paths["csv_path"])
    run_list: str = ", ".join(item["run_id"] for item in logged["run_ids"])
    logger.info("MLflow runs tagged phase=%s: %s", settings["phase_tag"], run_list)


def main(argv: list[str] | None = None) -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train NFL pregame prediction models")

    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline-home", "logistic", "logistic_l2", "gbm"],
        default="logistic",
        help=(
            "Model type to train. logistic_l2 is L2 logistic regression "
            "with C tuned on the training seasons only."
        ),
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Number of recent games for rolling features",
    )

    parser.add_argument(
        "--train-end-season",
        type=int,
        default=2023,
        help="Last season to include in training",
    )

    parser.add_argument(
        "--val-season",
        type=int,
        default=2024,
        help="Validation season",
    )

    parser.add_argument(
        "--test-season",
        type=int,
        default=2025,
        help="Test season",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        choices=list(FEATURE_SET_NAMES),
        default=FEATURE_SET_CURRENT,
        help=("Columns for a single split. --cv selects the logistic sets for --phase."),
    )

    parser.add_argument(
        "--cv",
        action="store_true",
        help=(
            "Run rolling-origin CV on 2016-2023. "
            "--phase 1 is the baseline report. "
            "--phase 2 --hypothesis h1 scores diff feature sets. "
            "--phase 2 --hypothesis h2 scores season defense and regressed EPA. "
            "--phase 2 --hypothesis h3 scores pass and rush splits against h2_full. "
            "--phase 2 --hypothesis h4 scores clean EPA against h2_full. "
            "--phase 2 --hypothesis h5 scores starting QB ratings against h2_full. "
            "--phase 2 --hypothesis ridge scores logistic_l2 against logistic h2_full. "
            "Does not score 2024 or 2025."
        ),
    )

    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "1 keeps the phase 1 report (home prior, current, with_games_played). "
            "2 scores the hypothesis from --hypothesis. H1 and H2 use logistic "
            "current. H3, H4, H5, and ridge use logistic h2_full."
        ),
    )

    parser.add_argument(
        "--hypothesis",
        type=str,
        choices=sorted(HYPOTHESIS_ALIASES),
        default=None,
        help=(
            "Phase 2 hypothesis. Default h1 scores diff and diff_only "
            "and writes reports/phase2. h2 scores h2_def and h2_full "
            "and writes reports/phase2_h2. h3 scores h3_split and h3_replace "
            "against h2_full and writes reports/phase2_h3. h4 scores h4_clean "
            "and h4_clean_sr against h2_full and writes reports/phase2_h4. "
            "h5 scores h5_qb against h2_full and writes reports/phase2_h5. "
            "ridge scores "
            "logistic_l2 against logistic h2_full and writes reports/phase2_ridge. "
            "Phase 1 rejects this flag."
        ),
    )

    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help=(
            "Model for the paired log loss reference. "
            "Default is baseline-home for phase 1 and logistic for phase 2."
        ),
    )

    parser.add_argument(
        "--reference-feature-set",
        type=str,
        default=None,
        help=(
            "Feature set of the reference model. "
            "Default is none for the home prior, current for logistic on H1 and H2, "
            "and h2_full for logistic on H3, H4, H5, and ridge."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/phase1",
        help=(
            "Directory for the comparison report. "
            "Phase 2 h1 with this default writes reports/phase2. "
            "Phase 2 h2 with this default writes reports/phase2_h2. "
            "Phase 2 h3 with this default writes reports/phase2_h3. "
            "Phase 2 h4 with this default writes reports/phase2_h4. "
            "Phase 2 h5 with this default writes reports/phase2_h5. "
            "Phase 2 ridge with this default writes reports/phase2_ridge."
        ),
    )

    parser.add_argument(
        "--max-null-fraction",
        type=float,
        default=DEFAULT_MAX_NULL_FRACTION,
        help=(
            "Refuse a requested CV feature column whose null share is at "
            "least this value. Default is 0.5. The error names the feature "
            "set and says to run make dbt-build-databricks."
        ),
    )

    parser.add_argument("--cv-start-season", type=int, default=2016)
    parser.add_argument("--cv-end-season", type=int, default=2023)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument(
        "--fixture-path",
        type=str,
        help="Path to fixture directory (for testing without Databricks)",
    )

    # Logistic regression hyperparameters
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (logistic)",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations (logistic)",
    )

    # Gradient boosting hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting stages (gbm)",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum tree depth (gbm)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (gbm)",
    )

    args = parser.parse_args(argv)

    if args.cv:
        _run_phase1_cli(args)
        return

    # Plain local paths such as ./mlruns fall back to sqlite inside
    # resolve_tracking_uri (MLflow 3.16 refuses the file store).
    # sqlite:// and databricks pass through unchanged.
    tracking_uri = resolve_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    )
    activate_tracking_experiment("nfl_pregame", tracking_uri)

    # Load data
    logger.info("Loading data...")
    fct_games, fct_team_game_epa = load_data(fixture_path=args.fixture_path)

    # Build features
    logger.info(f"Building features with lookback={args.lookback}...")
    expected_starters: pd.DataFrame | None = None
    qb_dropbacks: pd.DataFrame | None = None
    if args.feature_set == FEATURE_SET_H5_QB:
        expected_starters, qb_dropbacks = load_qb_inputs(args.fixture_path)
    features = build_pregame_features(
        fct_games=fct_games,
        fct_team_game_epa=fct_team_game_epa,
        lookback_games=args.lookback,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
    )
    if args.feature_set == FEATURE_SET_H5_QB:
        validate_feature_columns(features, (FEATURE_SET_H5_QB,))

    # Prepare training data
    X, y = prepare_training_data(
        fct_games,
        features,
        target_column="home_won",
        feature_set=args.feature_set,
    )

    # Split by season
    logger.info(
        f"Splitting data: train <= {args.train_end_season}, "
        f"val = {args.val_season}, test = {args.test_season}"
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_by_season(
        X,
        y,
        train_end_season=args.train_end_season,
        val_season=args.val_season,
        test_season=args.test_season,
    )

    logger.info(f"Train: {len(X_train)} games, Val: {len(X_val)} games, Test: {len(X_test)} games")

    # Train model
    with mlflow.start_run():
        # Log common parameters
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("feature_set", args.feature_set)
        mlflow.log_param("lookback_games", args.lookback)
        mlflow.log_param("train_end_season", args.train_end_season)
        mlflow.log_param("val_season", args.val_season)
        mlflow.log_param("test_season", args.test_season)
        mlflow.log_param("random_seed", args.seed)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        # Log feature list
        feature_cols = [
            col for col in X_train.columns if col not in ["game_id", "gameday", "season"]
        ]
        mlflow.log_param("features", ",".join(feature_cols))

        # Log git commit
        git_sha = get_git_sha()
        mlflow.set_tag("git_commit", git_sha)

        # Train model based on type
        if args.model == "baseline-home":
            logger.info("Training baseline model (training home win rate probability)")
            model, metrics = train_baseline_home(X_train, y_train, X_val, y_val)

        elif args.model == "logistic":
            logger.info(f"Training logistic regression (C={args.C}, max_iter={args.max_iter})")
            mlflow.log_param("C", args.C)
            mlflow.log_param("max_iter", args.max_iter)
            mlflow.log_param("regularization", "l2")

            model, metrics = train_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                C=args.C,
                max_iter=args.max_iter,
                random_state=args.seed,
            )

        elif args.model == "logistic_l2":
            logger.info(
                "Training L2 logistic regression with C tuned on the training seasons "
                f"(fallback C={args.C}, max_iter={args.max_iter})"
            )
            mlflow.log_param("max_iter", args.max_iter)
            mlflow.log_param("regularization", "l2")
            mlflow.log_param(
                "c_grid",
                ",".join(_format_regularization_c(value) for value in DEFAULT_L2_C_GRID),
            )
            mlflow.log_param("c_fallback", args.C)
            fitted: LogisticL2Fit = fit_logistic_l2(
                X_train,
                y_train,
                X_val,
                y_val,
                fallback_c=args.C,
                max_iter=args.max_iter,
                random_state=args.seed,
            )
            model = fitted.pipeline
            metrics = fitted.metrics
            mlflow.log_param("chosen_c", _format_regularization_c(fitted.chosen_c))
            mlflow.log_param("c_selection", fitted.c_selection)
            if fitted.inner_seasons:
                mlflow.log_param(
                    "inner_seasons",
                    ",".join(str(season) for season in fitted.inner_seasons),
                )

        elif args.model == "gbm":
            logger.info(
                f"Training gradient boosting (n_estimators={args.n_estimators}, "
                f"max_depth={args.max_depth}, lr={args.learning_rate})"
            )
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("learning_rate", args.learning_rate)

            model, metrics = train_gradient_boosting(
                X_train,
                y_train,
                X_val,
                y_val,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                random_state=args.seed,
            )

        else:
            raise ValueError(f"Unknown model type: {args.model}")

        # Log metrics
        logger.info(f"Validation metrics: {metrics}")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)

        trusted_types: dict[str, list[str]] = log_sklearn_model(model)
        logger.info(f"Trusted skops types: {trusted_types['trusted_types']}")

        logger.info("Training complete. MLflow run logged.")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
