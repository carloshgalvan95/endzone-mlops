"""
Tests for utility modules.
Follows DoD: smoke tests for pure functions.
"""

from pathlib import Path
from urllib.parse import urlparse

import pytest

from src.utils.config import Config, load_config, resolve_tracking_uri
from src.utils.logging import setup_logger


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config loads with defaults when no env vars set."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        config = Config()
        assert config.data_dir == Path("./data")
        assert config.mlflow_tracking_uri == "sqlite:///mlflow.db"
        assert config.enable_live_feed is False

        loaded = load_config()
        assert loaded.mlflow_tracking_uri == "sqlite:///mlflow.db"

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config respects environment variables."""
        monkeypatch.setenv("DATA_DIR", "/tmp/test_data")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
        monkeypatch.setenv("ENABLE_LIVE_FEED", "true")

        config = load_config()
        assert config.data_dir == Path("/tmp/test_data")
        assert config.mlflow_tracking_uri == "databricks"
        assert config.enable_live_feed is True

    def test_config_immutable(self) -> None:
        """Config is frozen after creation."""
        config = Config()
        with pytest.raises(Exception):
            config.data_dir = Path("/new/path")  # type: ignore


class TestResolveTrackingUri:
    """MLflow tracking URI normalization."""

    def test_databricks_passthrough(self) -> None:
        """Databricks targets stay unchanged, including prefixed names."""
        assert resolve_tracking_uri("databricks") == "databricks"
        prefixed: str = "databricks-uc"
        assert resolve_tracking_uri(prefixed) == prefixed

    def test_existing_scheme_passthrough(self) -> None:
        """Values that already contain a scheme are returned unchanged."""
        remote_uri: str = "http://127.0.0.1:5000"
        sqlite_uri: str = "sqlite:///mlflow.db"
        assert resolve_tracking_uri(remote_uri) == remote_uri
        assert resolve_tracking_uri(sqlite_uri) == sqlite_uri

    def test_relative_path_becomes_file_uri(self) -> None:
        """A relative path becomes the absolute URI from Path.as_uri."""
        relative_path: str = "./mlruns"
        expected: str = Path(relative_path).resolve().as_uri()
        resolved: str = resolve_tracking_uri(relative_path)
        assert resolved == expected
        parsed = urlparse(resolved)
        assert parsed.scheme == "file"
        assert parsed.path.endswith("/mlruns")


class TestLogging:
    """Test logging setup."""

    def test_logger_creation(self) -> None:
        """Logger is created with correct name and level."""
        logger = setup_logger("test_logger", level="DEBUG")
        assert logger.name == "test_logger"
        assert logger.level == 10  # DEBUG = 10

    def test_logger_singleton_behavior(self) -> None:
        """Calling setup_logger twice returns same logger."""
        logger1 = setup_logger("test_singleton")
        logger2 = setup_logger("test_singleton")
        assert logger1 is logger2

    def test_logger_with_file(self, tmp_path: Path) -> None:
        """Logger can write to file."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file_logger", log_file=str(log_file))
        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
