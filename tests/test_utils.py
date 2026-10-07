"""
Tests for utility modules.
Follows DoD: smoke tests for pure functions.
"""

from pathlib import Path

import pytest

from src.utils.config import Config, load_config
from src.utils.logging import setup_logger


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self) -> None:
        """Config loads with defaults when no env vars set."""
        config = Config()
        assert config.data_dir == Path("./data")
        assert config.mlflow_tracking_uri == "./mlruns"
        assert config.enable_live_feed is False

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
