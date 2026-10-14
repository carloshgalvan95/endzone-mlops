"""
Tests for training pipeline.

Tests:
- Train with synthetic data using local MLflow
- Model types all work
- Git SHA extraction
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.train.train import get_git_sha, load_data, train_baseline_home, train_logistic_regression
from src.utils.config import resolve_tracking_uri


@pytest.fixture
def train_fixture_path(tmp_path: Path) -> Path:
    """Create temporary fixture files for testing."""
    # Create synthetic games with both home wins and away wins
    games_data = {
        "game_id": [f"G{i}" for i in range(1, 51)],
        "season": [2023] * 30 + [2024] * 20,
        "gameday": pd.date_range("2023-09-01", periods=50, freq="7D"),
        "home_team_franchise": ["KC", "BUF", "PHI"] * 16 + ["KC", "BUF"],
        "away_team_franchise": ["BUF", "MIA", "DAL"] * 16 + ["MIA", "DAL"],
        "home_score": [28, 21, 24, 30, 17] * 10,
        "away_score": [21, 28, 21, 24, 28] * 10,
        "home_won": [1, 0, 1, 1, 0] * 10,
        "home_rest": [7] * 50,
        "away_rest": [7] * 50,
    }
    games_df = pd.DataFrame(games_data)

    # Create synthetic EPA data
    epa_rows = []
    for _, game in games_df.iterrows():
        epa_rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "gameday": game["gameday"],
                "team_franchise": game["home_team_franchise"],
                "avg_offensive_epa": 0.15,
                "avg_defensive_epa_allowed": -0.10,
            }
        )
        epa_rows.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "gameday": game["gameday"],
                "team_franchise": game["away_team_franchise"],
                "avg_offensive_epa": 0.10,
                "avg_defensive_epa_allowed": -0.05,
            }
        )

    epa_df = pd.DataFrame(epa_rows)

    # Save to parquet
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    games_df.to_parquet(fixture_dir / "fct_games.parquet")
    epa_df.to_parquet(fixture_dir / "fct_team_game_epa.parquet")

    return fixture_dir


class TestGetGitSha:
    """Test git SHA extraction."""

    def test_get_git_sha(self) -> None:
        """Git SHA should return a string."""
        sha = get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0


class TestLoadData:
    """Test data loading."""

    def test_load_data_from_fixture(self, train_fixture_path: Path) -> None:
        """Load data from fixtures."""
        games, epa = load_data(fixture_path=str(train_fixture_path))

        assert len(games) == 50
        assert len(epa) == 100
        assert "game_id" in games.columns
        assert "team_franchise" in epa.columns

    def test_load_data_missing_fixture(self, tmp_path: Path) -> None:
        """Missing fixture should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data(fixture_path=str(tmp_path / "nonexistent"))


class TestTrainModels:
    """Test model training functions."""

    def test_train_baseline_home(self, train_fixture_path: Path) -> None:
        """Train baseline model with synthetic data."""
        from src.features.pregame import (
            build_pregame_features,
            prepare_training_data,
            split_by_season,
        )
        from src.train.train import load_data

        games, epa = load_data(fixture_path=str(train_fixture_path))
        features = build_pregame_features(games, epa, lookback_games=5)
        X, y = prepare_training_data(games, features)
        X_train, X_val, X_test, y_train, y_val, y_test = split_by_season(
            X, y, train_end_season=2023, val_season=2024, test_season=2025
        )

        model, metrics = train_baseline_home(X_train, y_train, X_val, y_val)

        assert model is not None
        assert "accuracy" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "roc_auc" in metrics

        # Baseline always predicts 1 (home wins)
        # With 60% home wins in fixture, baseline should have ~60% accuracy
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_logistic_regression(self, train_fixture_path: Path) -> None:
        """Train logistic regression with synthetic data."""
        from src.features.pregame import (
            build_pregame_features,
            prepare_training_data,
            split_by_season,
        )
        from src.train.train import load_data

        games, epa = load_data(fixture_path=str(train_fixture_path))
        features = build_pregame_features(games, epa, lookback_games=5)
        X, y = prepare_training_data(games, features)
        X_train, X_val, X_test, y_train, y_val, y_test = split_by_season(
            X, y, train_end_season=2023, val_season=2024, test_season=2025
        )

        model, metrics = train_logistic_regression(
            X_train, y_train, X_val, y_val, C=1.0, max_iter=100, random_state=42
        )

        assert model is not None
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["brier_score"] <= 1
        assert metrics["log_loss"] >= 0


class TestEndToEndTraining:
    """End-to-end training test with local MLflow."""

    def test_end_to_end_train(
        self, train_fixture_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        End-to-end training pipeline test.

        Uses temporary MLflow directory to avoid polluting workspace.
        """
        import mlflow

        from src.features.pregame import (
            build_pregame_features,
            prepare_training_data,
            split_by_season,
        )
        from src.train.train import load_data, train_logistic_regression

        # Allow file store for testing
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

        # Use temporary MLflow directory
        with tempfile.TemporaryDirectory() as tmp_mlflow:
            tracking_uri = resolve_tracking_uri(tmp_mlflow)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("test_experiment")

            # Load data
            games, epa = load_data(fixture_path=str(train_fixture_path))

            # Build features
            features = build_pregame_features(games, epa, lookback_games=5)

            # Prepare data
            X, y = prepare_training_data(games, features)

            # Split
            X_train, X_val, X_test, y_train, y_val, y_test = split_by_season(
                X, y, train_end_season=2023, val_season=2024, test_season=2025
            )

            # Train with MLflow tracking
            with mlflow.start_run():
                mlflow.log_param("test_run", True)
                model, metrics = train_logistic_regression(
                    X_train, y_train, X_val, y_val, C=1.0, max_iter=100, random_state=42
                )
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                mlflow.sklearn.log_model(model, "model", skops_trusted_types=["numpy.dtype"])

                run_id = mlflow.active_run().info.run_id

            # Verify MLflow logged correctly
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)

            assert run.data.params["test_run"] == "True"
            assert "val_accuracy" in run.data.metrics
