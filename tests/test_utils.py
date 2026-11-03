"""Tests for utility modules."""

from pathlib import Path

from endzone.utils.logging import setup_logger


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
