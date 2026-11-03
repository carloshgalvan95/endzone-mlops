"""
Logging utilities for endzone-mlops.
Follows Wilson 14: clear logging, structured output.
"""

import logging
import sys


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """
    Set up logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of calling module).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Pipeline started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Format: timestamp - name - level - message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
