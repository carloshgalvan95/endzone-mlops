"""
Tests for training pipeline.

Tests:
- Train with synthetic data using local MLflow
- Model types all work
- Git SHA extraction
- Each model type logs to a temporary SQLite store
"""

import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.train.train import (
    REVIEWED_SKOPS_TRUSTED_TYPES,
    get_git_sha,
    load_data,
    log_sklearn_model,
    resolve_skops_trusted_types,
    train_baseline_home,
    train_gradient_boosting,
    train_logistic_regression,
)

# In-memory store: no SQLite file, so it cannot stay locked on Windows.
_NEUTRAL_TRACKING_URI: str = "sqlite:///:memory:"


def _release_mlflow_sqlalchemy_engines() -> None:
    """Dispose cached SQLAlchemy engines that keep a SQLite file open."""
    from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
    from mlflow.tracking._model_registry.utils import _get_store_registry
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    for store_cls in (TrackingStore, RegistryStore, SqlAlchemyJobStore):
        engine_map = getattr(store_cls, "_engine_map", None)
        if not isinstance(engine_map, dict):
            continue
        for engine in list(engine_map.values()):
            engine.dispose()
        engine_map.clear()

    tracking_cache = _tracking_store_registry._get_store_with_resolved_uri
    tracking_cache.cache_clear()
    registry_cache = _get_store_registry()._get_store_with_resolved_uri
    registry_cache.cache_clear()


def _reset_mlflow_tracking() -> None:
    """End any open run, drop SQLite engines, and leave a non-file tracking URI."""
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.end_run()
    _release_mlflow_sqlalchemy_engines()
    mlflow.set_tracking_uri(_NEUTRAL_TRACKING_URI)


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore MLflow's process-global tracking URI after each test."""
    import mlflow.tracking._tracking_service.utils as tracking_utils

    monkeypatch.setattr(tracking_utils, "_tracking_uri", tracking_utils._tracking_uri)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    yield
    _reset_mlflow_tracking()


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

        id_columns: list[str] = ["game_id", "gameday", "season"]
        val_features: pd.DataFrame = X_val.drop(columns=id_columns)
        home_win_rate: float = float(y_train.mean())
        probabilities: np.ndarray = model.predict_proba(val_features)[:, 1]
        hard_labels: np.ndarray = model.predict(val_features)

        # Constant probability is the training home win rate, not a hard 1.0.
        assert home_win_rate > 0.5
        assert probabilities == pytest.approx(home_win_rate)
        assert float(probabilities.min()) == pytest.approx(float(probabilities.max()))

        # Above one half, hard labels stay home, so accuracy matches always-home.
        assert set(hard_labels.tolist()) == {1}
        assert metrics["accuracy"] == pytest.approx(float((y_val == 1).mean()))
        hard_ones: np.ndarray = np.ones(len(y_val))
        assert metrics["log_loss"] < log_loss(y_val, hard_ones)
        assert metrics["brier_score"] < brier_score_loss(y_val, hard_ones)

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
        step_names: list[str] = list(model.named_steps)
        assert step_names == ["preprocessor", "scaler", "classifier"]
        assert isinstance(model.named_steps["scaler"], StandardScaler)


class TestEndToEndTraining:
    """End-to-end training test with local MLflow."""

    def test_end_to_end_train(self, train_fixture_path: Path, tmp_path: Path) -> None:
        """
        End-to-end training pipeline test.

        ``tmp_path`` is left in place when the test returns, so Windows can
        still have the SQLite file open without a cleanup PermissionError.
        """
        import mlflow

        from src.features.pregame import (
            build_pregame_features,
            prepare_training_data,
            split_by_season,
        )
        from src.train.train import load_data, train_logistic_regression

        # SQLite URI with a POSIX path so the same string works on Windows.
        tracking_uri = "sqlite:///" + Path(tmp_path, "mlflow.db").as_posix()
        artifact_location = Path(tmp_path, "mlartifacts").resolve().as_uri()
        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_id = mlflow.create_experiment(
                "test_experiment",
                artifact_location=artifact_location,
            )
            mlflow.set_experiment(experiment_id=experiment_id)

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
                log_sklearn_model(model)

                run_id = mlflow.active_run().info.run_id

            # Verify MLflow logged correctly
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)

            assert run.data.params["test_run"] == "True"
            assert "val_accuracy" in run.data.metrics
        finally:
            _reset_mlflow_tracking()


def _train_val_from_fixture(
    fixture_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build the season split used by the training tests."""
    from src.features.pregame import (
        build_pregame_features,
        prepare_training_data,
        split_by_season,
    )

    games, epa = load_data(fixture_path=str(fixture_path))
    features = build_pregame_features(games, epa, lookback_games=5)
    features_frame, target = prepare_training_data(games, features)
    X_train, X_val, _X_test, y_train, y_val, _y_test = split_by_season(
        features_frame,
        target,
        train_end_season=2023,
        val_season=2024,
        test_season=2025,
    )
    return X_train, X_val, y_train, y_val


class _UnreviewedEstimator(BaseEstimator, ClassifierMixin):
    """Estimator whose type is intentionally absent from the skops allowlist."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_UnreviewedEstimator":
        """Store class labels. The values are unused."""
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return a constant away label so the estimator is a classifier."""
        row_count: int = int(np.asarray(X).shape[0])
        return np.zeros(row_count, dtype=int)


class TestSkopsTrustedTypes:
    """Reviewed skops allowlist, applied per model."""

    def test_unreviewed_type_is_refused(self) -> None:
        """A type outside the allowlist is not trusted."""
        estimator = _UnreviewedEstimator()
        features = pd.DataFrame({"home_recent_offensive_epa_per_play": [0.1, -0.2]})
        target = pd.Series([1, 0])
        estimator.fit(features, target)

        with pytest.raises(ValueError, match="reviewed"):
            resolve_skops_trusted_types(estimator)


class TestLogEachModelType:
    """Log baseline, logistic, and gradient boosting without a skops error."""

    def test_logs_each_model_to_sqlite_store(
        self, train_fixture_path: Path, tmp_path: Path
    ) -> None:
        """Each model type logs to a temporary SQLite store.

        ``tmp_path`` stays until the test returns. Engines are disposed in
        ``finally`` so Windows can delete the SQLite file afterwards.
        """
        import mlflow

        X_train, X_val, y_train, y_val = _train_val_from_fixture(train_fixture_path)
        trainers = {
            "baseline": lambda: train_baseline_home(X_train, y_train, X_val, y_val),
            "logistic": lambda: train_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                C=10.0,
                max_iter=200,
                random_state=42,
            ),
            "gbm": lambda: train_gradient_boosting(
                X_train,
                y_train,
                X_val,
                y_val,
                n_estimators=10,
                max_depth=2,
                learning_rate=0.1,
                random_state=42,
            ),
        }
        expected_types: dict[str, list[str]] = {
            "baseline": [],
            "logistic": ["numpy.dtype"],
            "gbm": ["numpy.dtype", "sklearn.tree._tree.Tree"],
        }

        tracking_uri: str = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
        artifact_location: str = (tmp_path / "mlartifacts").resolve().as_uri()
        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_id: str = mlflow.create_experiment(
                "test_model_types",
                artifact_location=artifact_location,
            )
            mlflow.set_experiment(experiment_id=experiment_id)
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

            for model_name, trainer in trainers.items():
                with mlflow.start_run(run_name=model_name):
                    model, _metrics = trainer()
                    if model_name == "gbm":
                        assert "scaler" not in model.named_steps
                    trusted: dict[str, list[str]] = log_sklearn_model(model)
                    run_id: str = mlflow.active_run().info.run_id

                assert trusted["trusted_types"] == expected_types[model_name]
                assert set(trusted["trusted_types"]) <= REVIEWED_SKOPS_TRUSTED_TYPES
                logged_models = client.search_logged_models(experiment_ids=[experiment_id])
                matched = [item for item in logged_models if item.source_run_id == run_id]
                assert len(matched) == 1
                assert matched[0].name == "model"
                assert matched[0].status.name == "READY"
                reloaded = mlflow.sklearn.load_model(matched[0].model_uri)
                score_frame: pd.DataFrame = X_val.drop(columns=["game_id", "gameday", "season"])
                reloaded_proba: np.ndarray = reloaded.predict_proba(score_frame)
                assert reloaded_proba.shape[0] == len(score_frame)
        finally:
            _reset_mlflow_tracking()


class TestLogisticScaling:
    """StandardScaler keeps lbfgs from warning at C=10."""

    def test_fixture_c10_has_no_convergence_warning(self, train_fixture_path: Path) -> None:
        """Sweep settings (C=10, 1000 iterations) stay quiet on fixture data."""
        X_train, X_val, y_train, y_val = _train_val_from_fixture(train_fixture_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            train_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                C=10.0,
                max_iter=1000,
                random_state=42,
            )

        convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
        assert convergence == []

    def test_scaling_removes_lbfgs_convergence_warning(self) -> None:
        """Ill-conditioned columns warn without scaling and stay quiet with it."""
        rng = np.random.default_rng(42)
        row_count: int = 2000
        features = pd.DataFrame(
            {
                "home_recent_offensive_epa_per_play": rng.normal(0.05, 0.15, row_count),
                "away_recent_offensive_epa_per_play": rng.normal(-0.02, 0.12, row_count),
                "home_recent_defensive_epa_per_play": rng.normal(0.05, 0.15, row_count),
                "away_recent_defensive_epa_per_play": rng.normal(-0.02, 0.12, row_count),
                "home_recent_win_rate": rng.uniform(0.0, 1.0, row_count),
                "away_recent_win_rate": rng.uniform(0.0, 1.0, row_count),
                "home_rest_days": rng.integers(0, 21, row_count).astype(float),
                "away_rest_days": rng.integers(0, 21, row_count).astype(float),
                "home_games_played": rng.integers(0, 400, row_count).astype(float),
                "away_games_played": rng.integers(0, 400, row_count).astype(float),
                "home_spread": rng.normal(0.0, 1.0, row_count) * 1e6,
            }
        )
        target = pd.Series((rng.random(row_count) < 0.55).astype(int))
        assert set(target.tolist()) == {0, 1}

        with warnings.catch_warnings(record=True) as unscaled_caught:
            warnings.simplefilter("always", ConvergenceWarning)
            LogisticRegression(C=10.0, max_iter=100, solver="lbfgs", random_state=42).fit(
                features, target
            )
        assert any(issubclass(item.category, ConvergenceWarning) for item in unscaled_caught)

        with warnings.catch_warnings(record=True) as scaled_caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model, _metrics = train_logistic_regression(
                features,
                target,
                features,
                target,
                C=10.0,
                max_iter=100,
                random_state=42,
            )

        assert not any(issubclass(item.category, ConvergenceWarning) for item in scaled_caught)
        assert list(model.named_steps) == ["preprocessor", "scaler", "classifier"]
        assert isinstance(model.named_steps["scaler"], StandardScaler)
