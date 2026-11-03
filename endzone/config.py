"""Process settings from the environment and an optional ``.env`` file.

This is the only module that reads ``.env``, ``DATABRICKS_*``, and
``MLFLOW_TRACKING_URI``. A variable already set in the process wins, so
``set -a; source .env; set +a`` still overrides the file. A missing variable
is filled from ``.env`` in the working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_CATALOG: str = "workspace"
DEFAULT_MARTS_SCHEMA: str = "nfl_dev_marts"
DEFAULT_STAGING_SCHEMA: str = "nfl_dev_staging"
DEFAULT_RAW_SCHEMA: str = "nfl_raw"
DEFAULT_MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

_MISSING_CREDENTIALS: str = (
    "Missing required Databricks environment variables. "
    "Set DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN. "
    "See .env.example and docs/runbook.md for setup instructions."
)

# The working-directory file is read once. Later calls still see variables a
# test or a shell export changed after that read, and they do not put a
# deleted variable back.
_cwd_env_loaded: bool = False


@dataclass(frozen=True)
class DatabricksSettings:
    """Warehouse credentials and the Unity Catalog names the client qualifies."""

    host: str | None
    http_path: str | None
    token: str | None
    catalog: str
    marts_schema: str
    staging_schema: str
    raw_schema: str


def load_env_file(path: Path | None = None, *, override: bool = False) -> bool:
    """Load a dotenv file without interpolating values.

    Args:
        path: File to read. ``None`` reads ``.env`` in the working directory
            once per process.
        override: When ``False``, a variable already in the environment stays.

    Returns:
        ``True`` when a file was loaded.
    """
    global _cwd_env_loaded
    if path is None:
        if _cwd_env_loaded:
            return False
        _cwd_env_loaded = True
        dotenv_path = Path.cwd() / ".env"
    else:
        dotenv_path = path
    if not dotenv_path.is_file():
        return False
    loaded: bool = load_dotenv(dotenv_path, override=override, interpolate=False)
    if loaded and path is None:
        logger.info("Loading environment from %s", dotenv_path)
    return loaded


def resolve_setting(explicit: str | None, env_name: str, default: str) -> str:
    """Prefer an explicit argument, then a non-empty env var, then a default.

    Args:
        explicit: Value passed by the caller. ``None`` or blank means unset.
        env_name: Environment variable that overrides the default.
        default: Value used when neither argument nor env var is set.

    Returns:
        The resolved setting, with surrounding whitespace removed.
    """
    load_env_file()
    if explicit is not None and explicit.strip() != "":
        return explicit.strip()
    from_env: str | None = os.environ.get(env_name)
    if from_env is not None and from_env.strip() != "":
        return from_env.strip()
    return default


def _optional_env(name: str) -> str | None:
    """Return a non-empty environment value, or ``None``."""
    value: str | None = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def load_databricks_settings() -> DatabricksSettings:
    """Read warehouse settings from the environment and ``.env``.

    Returns:
        Credentials (``None`` when unset) and catalog or schema names.
    """
    load_env_file()
    return DatabricksSettings(
        host=_optional_env("DATABRICKS_HOST"),
        http_path=_optional_env("DATABRICKS_HTTP_PATH"),
        token=_optional_env("DATABRICKS_TOKEN"),
        catalog=resolve_setting(None, "DATABRICKS_CATALOG", DEFAULT_CATALOG),
        marts_schema=resolve_setting(None, "DATABRICKS_MARTS_SCHEMA", DEFAULT_MARTS_SCHEMA),
        staging_schema=resolve_setting(None, "DATABRICKS_STAGING_SCHEMA", DEFAULT_STAGING_SCHEMA),
        raw_schema=resolve_setting(None, "DATABRICKS_RAW_SCHEMA", DEFAULT_RAW_SCHEMA),
    )


def get_databricks_config() -> dict[str, str]:
    """Return host, HTTP path, and token, or raise when one is missing.

    Returns:
        Dict with ``host``, ``http_path``, and ``token``.

    Raises:
        ValueError: A required credential is unset or blank.
    """
    settings: DatabricksSettings = load_databricks_settings()
    if not settings.host or not settings.http_path or not settings.token:
        raise ValueError(_MISSING_CREDENTIALS)
    return {
        "host": settings.host,
        "http_path": settings.http_path,
        "token": settings.token,
    }


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


def configured_tracking_uri() -> str:
    """Return the MLflow tracking URI from the environment, then normalize it.

    Returns:
        A scheme URI. A plain path falls back to ``sqlite:///mlflow.db``.
    """
    load_env_file()
    raw: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    if raw is None:
        raw = DEFAULT_MLFLOW_TRACKING_URI
    return resolve_tracking_uri(raw)


load_env_file()
