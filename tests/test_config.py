"""Settings loaded from the environment and from a dotenv file."""

import os
from pathlib import Path

import pytest

from endzone.config import (
    DEFAULT_MLFLOW_TRACKING_URI,
    configured_registered_model_name,
    configured_registry_uri,
    configured_tracking_uri,
    get_databricks_config,
    load_env_file,
    resolve_setting,
    resolve_tracking_uri,
)


class TestGetDatabricksConfig:
    """Test Databricks configuration loading."""

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Get config when all env vars are set."""
        monkeypatch.setenv("DATABRICKS_HOST", "test-host.databricks.com")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/test")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-token")

        config = get_databricks_config()

        assert config["host"] == "test-host.databricks.com"
        assert config["http_path"] == "/sql/1.0/warehouses/test"
        assert config["token"] == "dapi-test-token"

    def test_config_missing_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise error when DATABRICKS_HOST is missing."""
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/test")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-token")

        with pytest.raises(ValueError, match="Missing required Databricks"):
            get_databricks_config()

    def test_config_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise error when DATABRICKS_TOKEN is missing."""
        monkeypatch.setenv("DATABRICKS_HOST", "test-host.databricks.com")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/test")
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

        with pytest.raises(ValueError, match="Missing required Databricks"):
            get_databricks_config()


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

    def test_plain_local_path_falls_back_to_sqlite(self, caplog: pytest.LogCaptureFixture) -> None:
        """MLflow 3.16 refuses ./mlruns and mlruns, so both become sqlite."""
        with caplog.at_level("WARNING", logger="endzone.config"):
            relative_path: str = resolve_tracking_uri("./mlruns")
            bare_path: str = resolve_tracking_uri("mlruns")

        assert relative_path == DEFAULT_MLFLOW_TRACKING_URI
        assert bare_path == DEFAULT_MLFLOW_TRACKING_URI
        assert relative_path == "sqlite:///mlflow.db"
        assert "file store" in caplog.text
        assert "./mlruns" in caplog.text
        assert "mlruns" in caplog.text

    def test_absolute_local_path_falls_back_to_sqlite(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Any scheme-less path is a file store and falls back to sqlite."""
        with caplog.at_level("WARNING", logger="endzone.config"):
            resolved: str = resolve_tracking_uri(str(tmp_path))

        assert resolved == DEFAULT_MLFLOW_TRACKING_URI
        assert "Falling back" in caplog.text

    def test_file_uri_passthrough(self) -> None:
        """An explicit file URI is already a scheme and is left unchanged."""
        file_uri: str = "file:///tmp/mlruns"
        assert resolve_tracking_uri(file_uri) == file_uri


class TestEnvFile:
    """``.env`` fills gaps and does not replace a shell export."""

    def test_file_fills_an_unset_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dotenv assignment is visible when the process has no value."""
        env_file: Path = tmp_path / ".env"
        env_file.write_text(
            "\n# comment\nENDZONE_CONFIG_PROBE=from-file\nexport ENDZONE_CONFIG_OTHER=exported\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("ENDZONE_CONFIG_PROBE", raising=False)
        monkeypatch.delenv("ENDZONE_CONFIG_OTHER", raising=False)

        loaded: bool = load_env_file(env_file)

        assert loaded is True
        assert os.environ["ENDZONE_CONFIG_PROBE"] == "from-file"
        assert os.environ["ENDZONE_CONFIG_OTHER"] == "exported"

    def test_shell_value_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An existing environment variable is left as the shell set it."""
        env_file: Path = tmp_path / ".env"
        env_file.write_text("ENDZONE_CONFIG_PROBE=from-file\n", encoding="utf-8")
        monkeypatch.setenv("ENDZONE_CONFIG_PROBE", "from-shell")

        load_env_file(env_file)

        assert os.environ["ENDZONE_CONFIG_PROBE"] == "from-shell"

    def test_missing_file_is_skipped(self, tmp_path: Path) -> None:
        """A path that is not a file does not raise."""
        assert load_env_file(tmp_path / "missing.env") is False

    def test_quotes_are_stripped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A quoted dotenv value is stored without the quotes."""
        env_file: Path = tmp_path / ".env"
        env_file.write_text('ENDZONE_CONFIG_PROBE="quoted"\n', encoding="utf-8")
        monkeypatch.delenv("ENDZONE_CONFIG_PROBE", raising=False)

        load_env_file(env_file)

        assert os.environ["ENDZONE_CONFIG_PROBE"] == "quoted"

    def test_blank_schema_env_uses_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank override falls through to the built-in schema name."""
        monkeypatch.setenv("DATABRICKS_RAW_SCHEMA", "  ")
        assert resolve_setting(None, "DATABRICKS_RAW_SCHEMA", "nfl_raw") == "nfl_raw"
        assert resolve_setting("  ", "DATABRICKS_RAW_SCHEMA", "nfl_raw") == "nfl_raw"

    def test_tracking_uri_comes_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MLFLOW_TRACKING_URI is read here and a plain path falls back to sqlite."""
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert configured_tracking_uri() == DEFAULT_MLFLOW_TRACKING_URI
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///other.db")
        assert configured_tracking_uri() == "sqlite:///other.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")
        assert configured_tracking_uri() == DEFAULT_MLFLOW_TRACKING_URI

    def test_registry_uri_and_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unity Catalog gets a three-level name. sqlite keeps endzone-winprob."""
        monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("ENDZONE_REGISTERED_MODEL", raising=False)
        monkeypatch.delenv("DATABRICKS_CATALOG", raising=False)
        monkeypatch.delenv("ENDZONE_UC_SCHEMA", raising=False)
        monkeypatch.delenv("ENDZONE_UC_MODEL", raising=False)
        assert configured_registry_uri() == DEFAULT_MLFLOW_TRACKING_URI
        monkeypatch.setenv("MLFLOW_REGISTRY_URI", "databricks-uc")
        assert configured_registry_uri() == "databricks-uc"
        assert configured_registered_model_name("sqlite:///mlflow.db") == "endzone-winprob"
        assert (
            configured_registered_model_name("databricks-uc") == "workspace.nfl_dev.endzone_winprob"
        )
        monkeypatch.setenv("ENDZONE_REGISTERED_MODEL", "workspace.other.custom")
        assert configured_registered_model_name("databricks-uc") == "workspace.other.custom"
