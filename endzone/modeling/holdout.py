"""Single train, validation, and test split. This path does not run rolling-origin CV."""

from __future__ import annotations

import argparse
from typing import Any

import mlflow
import pandas as pd

from endzone.config import configured_tracking_uri
from endzone.features.opponent import H6_CHALLENGER_FEATURE_SETS
from endzone.features.pregame import (
    FEATURE_SET_H5_QB,
    FEATURE_SET_PKG_CORE,
    PKG_CORE_INJ_FEATURE_SETS,
    PRIOR_SHRINK_FEATURE_SETS,
    QBELO538_FEATURE_SETS,
    build_pregame_features,
    prepare_training_data,
    split_by_season,
)
from endzone.features.recency import RECENCY_CHALLENGER_FEATURE_SETS
from endzone.modeling.dataset import load_data, load_qb_inputs, validate_feature_columns
from endzone.modeling.models import (
    DEFAULT_L2_C_GRID,
    LogisticL2Fit,
    _format_regularization_c,
    fit_logistic_l2,
    train_baseline_home,
    train_gradient_boosting,
    train_logistic_regression,
)
from endzone.modeling.tracking import (
    activate_tracking_experiment,
    get_git_sha,
    log_sklearn_model,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

HOLDOUT_EXPERIMENT_NAME: str = "nfl_pregame"


def _log_split_params(args: argparse.Namespace, sizes: dict[str, int], columns: list[str]) -> None:
    """Log the split, the seed, and the feature list for one holdout run."""
    mlflow.log_param("model_type", args.model)
    mlflow.log_param("feature_set", args.feature_set)
    mlflow.log_param("lookback_games", args.lookback)
    mlflow.log_param("train_end_season", args.train_end_season)
    mlflow.log_param("val_season", args.val_season)
    mlflow.log_param("test_season", args.test_season)
    mlflow.log_param("random_seed", args.seed)
    mlflow.log_param("train_size", sizes["train"])
    mlflow.log_param("val_size", sizes["val"])
    mlflow.log_param("test_size", sizes["test"])
    mlflow.log_param("features", ",".join(columns))
    mlflow.set_tag("git_commit", get_git_sha())


def _fit_holdout_model(
    args: argparse.Namespace,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Any, dict[str, float]]:
    """Fit the model named by ``--model`` and log its hyperparameters."""
    if args.model == "baseline-home":
        logger.info("Training baseline model (training home win rate probability)")
        return train_baseline_home(x_train, y_train, x_val, y_val)
    if args.model == "logistic":
        logger.info(f"Training logistic regression (C={args.C}, max_iter={args.max_iter})")
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("regularization", "l2")
        return train_logistic_regression(
            x_train,
            y_train,
            x_val,
            y_val,
            C=args.C,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
    if args.model == "logistic_l2":
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
            x_train,
            y_train,
            x_val,
            y_val,
            fallback_c=args.C,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        mlflow.log_param("chosen_c", _format_regularization_c(fitted.chosen_c))
        mlflow.log_param("c_selection", fitted.c_selection)
        if fitted.inner_seasons:
            mlflow.log_param(
                "inner_seasons",
                ",".join(str(season) for season in fitted.inner_seasons),
            )
        return fitted.pipeline, fitted.metrics
    if args.model == "gbm":
        logger.info(
            f"Training gradient boosting (n_estimators={args.n_estimators}, "
            f"max_depth={args.max_depth}, lr={args.learning_rate})"
        )
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        return train_gradient_boosting(
            x_train,
            y_train,
            x_val,
            y_val,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown model type: {args.model}")


def run_holdout(args: argparse.Namespace) -> None:
    """Train one model on seasons through the train cut and log the validation metrics.

    Recency, H6, pkg_core, the injury candidates, the ensemble
    candidates, the QB Elo candidate, the prior-shrink candidates, and
    the margin candidates are refused before any warehouse read. That
    split scores 2024 and 2025. The preregistered commands are
    ``python scripts/train_recency.py``, ``python scripts/train_h6.py``,
    ``python scripts/train_pkg_core.py``,
    ``python scripts/train_pkg_core_inj.py``,
    ``python scripts/train_ensembles.py``,
    ``python scripts/train_ensembles_r2.py``,
    ``python scripts/train_qbelo538.py``,
    ``python scripts/train_prior_shrink.py``, and
    ``python scripts/train_margin.py``.
    """
    if args.feature_set in RECENCY_CHALLENGER_FEATURE_SETS:
        raise ValueError(
            f"{args.feature_set} is a preregistered recency challenger. "
            "Score it with python scripts/train_recency.py. "
            "That command is rolling-origin CV for seasons 2016-2023. "
            "This split scores 2024 and 2025."
        )
    if args.feature_set in H6_CHALLENGER_FEATURE_SETS:
        raise ValueError(
            f"{args.feature_set} is a preregistered H6 challenger. "
            "Score it with python scripts/train_h6.py. "
            "That command is rolling-origin CV for seasons 2016-2023. "
            "This split scores 2024 and 2025."
        )
    if args.feature_set == FEATURE_SET_PKG_CORE:
        raise ValueError(
            "pkg_core is a preregistered feature package. "
            "Score it with python scripts/train_pkg_core.py. "
            "That command is rolling-origin CV for seasons 2016-2023. "
            "This split scores 2024 and 2025."
        )
    if args.feature_set in PKG_CORE_INJ_FEATURE_SETS:
        raise ValueError(
            f"{args.feature_set} is a preregistered injury candidate. "
            "Score it with python scripts/train_pkg_core_inj.py. "
            "That command is rolling-origin CV for seasons 2016-2023. "
            "This split scores 2024 and 2025."
        )
    if args.feature_set in QBELO538_FEATURE_SETS:
        raise ValueError(
            f"{args.feature_set} is a preregistered QB Elo candidate. "
            "Score it with python scripts/train_qbelo538.py. "
            "That command is rolling-origin CV for seasons 2016-2023. "
            "This split scores 2024 and 2025."
        )
    if args.feature_set in PRIOR_SHRINK_FEATURE_SETS:
        raise ValueError(
            f"{args.feature_set} is a preregistered prior-shrink candidate. "
            "Score it with python scripts/train_prior_shrink.py. "
            "That command is rolling-origin CV for seasons 2016-2023. "
            "This split scores 2024 and 2025."
        )
    tracking_uri: str = configured_tracking_uri()
    activate_tracking_experiment(HOLDOUT_EXPERIMENT_NAME, tracking_uri)
    logger.info("Loading data...")
    games: pd.DataFrame
    team_games: pd.DataFrame
    games, team_games = load_data(fixture_path=args.fixture_path)
    logger.info(f"Building features with lookback={args.lookback}...")
    expected_starters: pd.DataFrame | None = None
    qb_dropbacks: pd.DataFrame | None = None
    if args.feature_set == FEATURE_SET_H5_QB:
        expected_starters, qb_dropbacks = load_qb_inputs(args.fixture_path)
    features: pd.DataFrame = build_pregame_features(
        fct_games=games,
        fct_team_game_epa=team_games,
        lookback_games=args.lookback,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
    )
    if args.feature_set == FEATURE_SET_H5_QB:
        validate_feature_columns(features, (FEATURE_SET_H5_QB,))
    frame: pd.DataFrame
    target: pd.Series
    frame, target = prepare_training_data(
        games,
        features,
        target_column="home_won",
        feature_set=args.feature_set,
    )
    logger.info(
        f"Splitting data: train <= {args.train_end_season}, "
        f"val = {args.val_season}, test = {args.test_season}"
    )
    x_train, x_val, x_test, y_train, y_val, _y_test = split_by_season(
        frame,
        target,
        train_end_season=args.train_end_season,
        val_season=args.val_season,
        test_season=args.test_season,
    )
    logger.info(f"Train: {len(x_train)} games, Val: {len(x_val)} games, Test: {len(x_test)} games")
    with mlflow.start_run():
        feature_columns: list[str] = [
            column for column in x_train.columns if column not in ["game_id", "gameday", "season"]
        ]
        _log_split_params(
            args,
            {"train": len(x_train), "val": len(x_val), "test": len(x_test)},
            feature_columns,
        )
        model: Any
        metrics: dict[str, float]
        model, metrics = _fit_holdout_model(args, x_train, y_train, x_val, y_val)
        logger.info(f"Validation metrics: {metrics}")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        trusted_types: dict[str, list[str]] = log_sklearn_model(model)
        logger.info(f"Trusted skops types: {trusted_types['trusted_types']}")
        logger.info("Training complete. MLflow run logged.")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
