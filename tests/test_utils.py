"""Tests for utility modules."""

import logging
from pathlib import Path

import pytest

from endzone.utils.logging import setup_logger


class TestLogging:
    """Logger name, level, singleton, and file output."""

    @pytest.mark.cross_platform
    def test_logger_name_level_singleton_and_file(self, tmp_path: Path) -> None:
        """One name is one logger, and a file handler writes the message."""
        logger: logging.Logger = setup_logger("test_logger", level="DEBUG")
        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG

        singleton: logging.Logger = setup_logger("test_singleton")
        assert setup_logger("test_singleton") is singleton

        log_file: Path = tmp_path / "test.log"
        file_logger: logging.Logger = setup_logger("test_file_logger", log_file=str(log_file))
        file_logger.info("Test message")
        assert "Test message" in log_file.read_text(encoding="utf-8")
