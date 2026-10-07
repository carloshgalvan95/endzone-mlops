"""
Configuration utilities for endzone-mlops.
Follows Wilson pattern: config injected, never hardcoded.
"""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    """Application configuration."""

    model_config = ConfigDict(frozen=True)

    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    mlflow_tracking_uri: str = Field(default="./mlruns")
    enable_live_feed: bool = Field(default=False)


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config object with validated settings.

    Example:
        >>> config = load_config()
        >>> print(config.data_dir)
        ./data
    """
    return Config(
        data_dir=Path(os.getenv("DATA_DIR", "./data")),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        enable_live_feed=os.getenv("ENABLE_LIVE_FEED", "false").lower() == "true",
    )


def get_config_dict() -> dict[str, Any]:
    """
    Get configuration as dictionary.

    Returns:
        Dict of configuration values.
    """
    config = load_config()
    return config.model_dump()
