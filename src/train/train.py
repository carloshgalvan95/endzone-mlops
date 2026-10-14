"""
Train NFL pregame prediction models with MLflow tracking.

Usage:
    python -m src.train.train --model logistic --lookback 10 --seed 42
    python -m src.train.train --model gbm --max-depth 5 --n-estimators 100
    python -m src.train.train --model baseline-home
    python -m src.train.train --cv
"""

import argparse
import os
import subprocess
import tempfile
from collections.abc import Callable
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
    FEATURE_SET_CURRENT,
    FEATURE_SET_WITH_GAMES_PLAYED,
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
CV_SCHEME: str = "rolling_season_2016_2023"
PHASE1_MARKDOWN_NAME: str = "comparison.md"
PHASE1_CSV_NAME: str = "comparison.csv"
REFERENCE_MODEL_NAME: str = "baseline-home"

# predict(train_x, train_y, eval_x, eval_y) -> P(home win) for eval rows.
FoldPredictor = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], np.ndarray]

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

    epa_features = [col for col in feature_cols if "epa" in col]
    win_rate_features = [col for col in feature_cols if "win_rate" in col]
    other_features = [
        col for col in feature_cols if col not in epa_features and col not in win_rate_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("epa_imputer", SimpleImputer(strategy="constant", fill_value=0), epa_features),
            (
                "win_rate_imputer",
                SimpleImputer(strategy="constant", fill_value=0.5),
                win_rate_features,
            ),
            ("other_passthrough", "passthrough", other_features),
        ]
    )

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

    epa_features = [col for col in feature_cols if "epa" in col]
    win_rate_features = [col for col in feature_cols if "win_rate" in col]
    other_features = [
        col for col in feature_cols if col not in epa_features and col not in win_rate_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("epa_imputer", SimpleImputer(strategy="constant", fill_value=0), epa_features),
            (
                "win_rate_imputer",
                SimpleImputer(strategy="constant", fill_value=0.5),
                win_rate_features,
            ),
            ("other_passthrough", "passthrough", other_features),
        ]
    )

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
        probabilities = np.asarray(
            predict_fold(train_frame, train_target, eval_frame, eval_target),
            dtype=float,
        )
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


def run_phase1_baselines(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    config: RollingOriginConfig | None = None,
    lookback_games: int = 10,
    logistic_c: float = 1.0,
    logistic_max_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Score the home prior, both logistic feature sets, Elo, and the market.

    The market is skipped when the moneyline columns are absent or every
    line is missing. 2024 and 2025 are not scored. The reference for the
    paired log loss interval is the home prior.

    Args:
        games: Completed games, one row per game.
        team_games: Team-game EPA used to build the logistic matrices.
        config: Rolling-origin window. Defaults to 2016-2023.
        lookback_games: Rolling window passed to ``build_pregame_features``.
        logistic_c: Inverse regularization strength.
        logistic_max_iter: Logistic solver iteration cap.
        random_state: Seed for the logistic fit and the stored config.

    Returns:
        Dict with ``results``, ``comparison_table``, ``markdown``, and ``config``.
    """
    resolved: RollingOriginConfig = make_rolling_origin_config(
        **(config._asdict() if config is not None else {})
    )
    features: pd.DataFrame = build_pregame_features(
        games,
        team_games,
        lookback_games=lookback_games,
    )
    shared_params: dict[str, Any] = {
        "lookback_games": lookback_games,
        "cv_start_season": resolved.cv_start_season,
        "cv_end_season": resolved.cv_end_season,
        "n_bootstrap": resolved.n_bootstrap,
        "bootstrap_seed": resolved.bootstrap_seed,
        "random_state": random_state,
    }
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

    for feature_set in (FEATURE_SET_CURRENT, FEATURE_SET_WITH_GAMES_PLAYED):
        matrix, target = prepare_training_data(games, features, feature_set=feature_set)
        logistic_cv: dict[str, Any] = rolling_origin_cv(
            matrix,
            target,
            _logistic_fold_predictor(logistic_c, logistic_max_iter, random_state),
            resolved,
        )
        logistic_reason: str | None = None if not logistic_cv["oof"].empty else "no_scored_folds"
        columns: str = ",".join(feature_columns_for(feature_set))
        results.append(
            _cv_result_record(
                "logistic",
                feature_set,
                "ok" if logistic_reason is None else "skipped",
                logistic_reason,
                {
                    **shared_params,
                    "model_type": "logistic",
                    "feature_set": feature_set,
                    "features": columns,
                    "C": logistic_c,
                    "max_iter": logistic_max_iter,
                },
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

    reference_oof: pd.DataFrame | None = results[0]["oof"]
    for result in results:
        if result["status"] != "ok" or result["oof"] is None:
            result["comparison"] = None
            continue
        result["comparison"] = compare_oof_to_reference(
            result["oof"],
            reference_oof,
            n_bootstrap=resolved.n_bootstrap,
            seed=resolved.bootstrap_seed,
        )

    comparison_table: pd.DataFrame = _comparison_table(results)
    report: dict[str, Any] = {
        "results": results,
        "comparison_table": comparison_table,
        "config": resolved._asdict(),
        "lookback_games": lookback_games,
        "reference_model": REFERENCE_MODEL_NAME,
        "markdown": "",
    }
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


def render_phase1_report(report: dict[str, Any]) -> str:
    """Render the phase 1 comparison. Numbers come only from ``report``."""
    config: dict[str, Any] = report["config"]
    table: pd.DataFrame = report["comparison_table"]
    lines: list[str] = [
        "# Phase 1 baselines",
        "",
        "Rolling-origin CV. For each season S from",
        f"{config['cv_start_season']} through {config['cv_end_season']}, a model",
        "trains on seasons strictly before S and scores S.",
        f"{FINAL_VALIDATION_SEASON} is the final validation and is not scored.",
        f"{HELD_OUT_TEST_SEASON} is not scored.",
        "",
        f"Reference model: `{report['reference_model']}`.",
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
    for result in report["results"]:
        lines.append(f"### {result['model']} ({result['feature_set']})")
        lines.append("")
        if result["status"] != "ok":
            lines.append(f"Skipped ({result['reason']}).")
            lines.append("")
            if not result["folds"]:
                continue
        fold_rows: list[list[object]] = []
        for fold in result["folds"]:
            fold_rows.append(
                [
                    fold["season"],
                    fold["status"],
                    fold["reason"],
                    fold["n_train"],
                    fold["n_eval"],
                    ",".join(str(season) for season in fold["train_seasons"]),
                    fold["log_loss"],
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "season",
                    "status",
                    "reason",
                    "n_train",
                    "n_eval",
                    "train_seasons",
                    "log_loss",
                ],
                fold_rows,
            )
        )
        lines.append("")
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


def log_phase1_to_mlflow(
    report: dict[str, Any],
    paths: dict[str, Any],
    tracking_uri: str,
    experiment_name: str = PHASE1_EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Log one phase=1 run per scored model.

    Args:
        report: Result of ``run_phase1_baselines``.
        paths: Paths from ``write_phase1_outputs``.
        tracking_uri: MLflow tracking URI, passed through ``resolve_tracking_uri``.
        experiment_name: Experiment to create or activate.

    Returns:
        Dict with ``run_ids``, ``experiment_name``, and ``tracking_uri``.
    """
    resolved_uri: str = resolve_tracking_uri(tracking_uri)
    activate_tracking_experiment(experiment_name, resolved_uri)
    run_ids: list[dict[str, str]] = []
    for result in report["results"]:
        if result["status"] != "ok":
            continue
        run_name: str = f"{result['model']}-{result['feature_set']}"
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tag("phase", PHASE1_TAG)
            mlflow.set_tag("feature_set", result["feature_set"])
            mlflow.set_tag("cv_scheme", CV_SCHEME)
            mlflow.set_tag("git_commit", get_git_sha())
            mlflow.set_tag(
                "mlflow.note.content",
                "Phase 1 rolling-origin baselines. "
                f"{FINAL_VALIDATION_SEASON} and {HELD_OUT_TEST_SEASON} are not scored. "
                "No model is promoted from this run.",
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
            comparison: dict[str, Any] | None = result["comparison"]
            if comparison is not None:
                for source_name, metric_name in (
                    ("log_loss_diff", "cv_log_loss_diff"),
                    ("log_loss_diff_ci_low", "cv_log_loss_diff_ci_low"),
                    ("log_loss_diff_ci_high", "cv_log_loss_diff_ci_high"),
                ):
                    metric_value = comparison.get(source_name)
                    if metric_value is None or pd.isna(metric_value):
                        continue
                    mlflow.log_metric(metric_name, float(metric_value))
            scored_folds: int = sum(1 for fold in result["folds"] if fold["status"] == "scored")
            skipped_folds: int = sum(1 for fold in result["folds"] if fold["status"] != "scored")
            mlflow.log_metric("cv_folds_scored", float(scored_folds))
            mlflow.log_metric("cv_folds_skipped", float(skipped_folds))
            mlflow.log_artifact(paths["markdown_path"])
            mlflow.log_artifact(paths["csv_path"])
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
    return {
        "run_ids": run_ids,
        "experiment_name": experiment_name,
        "tracking_uri": resolved_uri,
    }


def _run_phase1_cli(args: argparse.Namespace) -> None:
    """Fit the phase 1 baselines, write the report, and log MLflow."""
    config: RollingOriginConfig = make_rolling_origin_config(
        cv_start_season=args.cv_start_season,
        cv_end_season=args.cv_end_season,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    logger.info(
        "Running phase 1 rolling-origin CV for seasons %s through %s",
        config.cv_start_season,
        config.cv_end_season,
    )
    fixture_path: str | None = args.fixture_path if args.fixture_path else None
    games, team_games = load_data(fixture_path=fixture_path)
    report: dict[str, Any] = run_phase1_baselines(
        games,
        team_games,
        config=config,
        lookback_games=args.lookback,
        logistic_c=args.C,
        logistic_max_iter=args.max_iter,
        random_state=args.seed,
    )
    paths: dict[str, Any] = write_phase1_outputs(report, Path(args.output_dir))
    tracking_uri: str = resolve_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    )
    logged: dict[str, Any] = log_phase1_to_mlflow(report, paths, tracking_uri)
    logger.info("Wrote %s and %s", paths["markdown_path"], paths["csv_path"])
    run_list: str = ", ".join(item["run_id"] for item in logged["run_ids"])
    logger.info("MLflow runs tagged phase=%s: %s", PHASE1_TAG, run_list)


def main(argv: list[str] | None = None) -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train NFL pregame prediction models")

    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline-home", "logistic", "gbm"],
        default="logistic",
        help="Model type to train",
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
        choices=[FEATURE_SET_CURRENT, FEATURE_SET_WITH_GAMES_PLAYED],
        default=FEATURE_SET_CURRENT,
        help="Columns for a single split. --cv always logs both feature sets.",
    )

    parser.add_argument(
        "--cv",
        action="store_true",
        help=("Run phase 1 rolling-origin CV on 2016-2023. Does not score 2024 or 2025."),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/phase1",
        help="Directory for the phase 1 comparison report",
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
    features = build_pregame_features(
        fct_games=fct_games,
        fct_team_game_epa=fct_team_game_epa,
        lookback_games=args.lookback,
    )

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

            model, metrics = train_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                C=args.C,
                max_iter=args.max_iter,
                random_state=args.seed,
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
