"""
Configuration utilities for endzone-mlops.
Follows Wilson pattern: config injected, never hardcoded.
"""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Relative SQLite file in the repo root. The filesystem tracking backend
# (./mlruns) is in maintenance mode on current MLflow.
DEFAULT_MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"


class Config(BaseModel):
    """Application configuration."""

    model_config = ConfigDict(frozen=True)

    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    mlflow_tracking_uri: str = Field(default=DEFAULT_MLFLOW_TRACKING_URI)
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
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI),
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


def resolve_tracking_uri(value: str) -> str:
    """Normalize an MLflow tracking URI for POSIX and Windows.

    Scheme URIs and Databricks targets are returned unchanged. A local
    filesystem path is resolved to an absolute URI that MLflow accepts
    on both platforms.

    Args:
        value: Tracking URI or local filesystem path.

    Returns:
        Tracking URI safe for ``mlflow.set_tracking_uri``.
    """
    has_scheme: bool = "://" in value
    is_databricks_target: bool = value == "databricks" or value.startswith("databricks")
    if has_scheme or is_databricks_target:
        return value
    return Path(value).resolve().as_uri()
