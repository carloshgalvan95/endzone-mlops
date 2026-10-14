"""
Train NFL pregame prediction models with MLflow tracking.

Usage:
    python -m src.train.train --model logistic --lookback 10 --seed 42
    python -m src.train.train --model gbm --max-depth 5 --n-estimators 100
    python -m src.train.train --model baseline-home
"""

import argparse
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data.databricks_client import DatabricksClient
from src.features.pregame import build_pregame_features, prepare_training_data, split_by_season
from src.utils.config import resolve_tracking_uri
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


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
    """
    Train baseline model (always predict home team wins).

    Returns:
        Tuple of (model, metrics_dict)
    """
    model = DummyClassifier(strategy="constant", constant=1, random_state=42)
    model.fit(X_train.drop(columns=["game_id", "gameday", "season"]), y_train)

    y_pred_proba = model.predict_proba(X_val.drop(columns=["game_id", "gameday", "season"]))[:, 1]
    y_pred = model.predict(X_val.drop(columns=["game_id", "gameday", "season"]))

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
    """
    Train logistic regression model with imputation pipeline.

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
            ("classifier", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)),
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


def main() -> None:
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

    args = parser.parse_args()

    # Set up MLflow. Local paths become absolute file URIs so Windows
    # drive letters are not parsed as an invalid remote store.
    tracking_uri = resolve_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    if tracking_uri == "databricks" or tracking_uri.startswith("databricks"):
        logger.info("Using Databricks MLflow tracking")
    else:
        logger.info(f"Using local MLflow tracking: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("nfl_pregame")

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
    X, y = prepare_training_data(fct_games, features, target_column="home_won")

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
            logger.info("Training baseline model (always predict home wins)")
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

        # Log model (trust numpy.dtype for sklearn Pipeline compatibility)
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["numpy.dtype"])

        logger.info("Training complete. MLflow run logged.")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
