"""MLflow tracking URI helpers."""

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Relative SQLite file in the repo root. The filesystem tracking backend
# (./mlruns) is in maintenance mode on current MLflow.
DEFAULT_MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"


def resolve_tracking_uri(value: str) -> str:
    """Normalize an MLflow tracking URI for POSIX and Windows.

    Scheme URIs and Databricks targets are returned unchanged. A plain
    local path such as ``./mlruns`` or ``mlruns`` is refused by the
    MLflow 3.16 file store, so this logs a warning and falls back to
    ``sqlite:///mlflow.db``.
    """
    has_scheme: bool = "://" in value
    is_databricks_target: bool = value == "databricks" or value.startswith("databricks")
    if has_scheme or is_databricks_target:
        return value
    logger.warning(
        "MLflow 3.16 refuses the file store for tracking URI %r. Falling back to %s.",
        value,
        DEFAULT_MLFLOW_TRACKING_URI,
    )
    return DEFAULT_MLFLOW_TRACKING_URI
