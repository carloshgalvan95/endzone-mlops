"""
Unit tests for nflverse data ingestion module.

Tests use mocks to avoid network calls and Databricks dependencies in CI.
Follows Wilson patterns: test pure functions, mock I/O boundaries.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingest.load_raw_nflverse import (
    DATASET_CONFIG,
    create_bronze_table,
    create_schema_and_volume,
    download_nflverse_games,
    download_nflverse_pbp,
    get_databricks_config,
    upload_to_volume,
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


class TestDownloadNflverseGames:
    """Test nflverse games data download."""

    @patch("src.ingest.load_raw_nflverse.pd.read_parquet")
    def test_download_and_filter_games(
        self,
        mock_read_parquet: Mock,
        tmp_path: Path,
    ) -> None:
        """Download games and filter to specified seasons."""
        # Mock data with multiple seasons
        mock_df = pd.DataFrame(
            {
                "season": [2022, 2022, 2023, 2023, 2024],
                "week": [1, 2, 1, 2, 1],
                "game_id": [
                    "2022_01_A_B",
                    "2022_02_C_D",
                    "2023_01_E_F",
                    "2023_02_G_H",
                    "2024_01_I_J",
                ],
            }
        )
        mock_read_parquet.return_value = mock_df

        # Download and filter to 2022 and 2023
        result_path = download_nflverse_games([2022, 2023], tmp_path)

        # Verify parquet was read
        mock_read_parquet.assert_called_once()
        assert "games.parquet" in mock_read_parquet.call_args[0][0]

        # Verify filtered data was saved - read with pyarrow directly to avoid mock
        assert result_path.exists()
        table = pq.read_table(result_path)
        result_df = table.to_pandas()
        assert len(result_df) == 4  # Only 2022 and 2023 games
        assert set(result_df["season"]) == {2022, 2023}

    @patch("src.ingest.load_raw_nflverse.pd.read_parquet")
    def test_download_games_no_matches(
        self,
        mock_read_parquet: Mock,
        tmp_path: Path,
    ) -> None:
        """Raise error when no games match filter."""
        mock_df = pd.DataFrame(
            {
                "season": [2020, 2021],
                "week": [1, 1],
                "game_id": ["2020_01_A_B", "2021_01_C_D"],
            }
        )
        mock_read_parquet.return_value = mock_df

        with pytest.raises(ValueError, match="No games found for seasons"):
            download_nflverse_games([2024, 2025], tmp_path)


class TestDownloadNflversePbp:
    """Test nflverse play-by-play data download."""

    @patch("src.ingest.load_raw_nflverse.pd.read_parquet")
    def test_download_and_combine_pbp(
        self,
        mock_read_parquet: Mock,
        tmp_path: Path,
    ) -> None:
        """Download play-by-play for multiple seasons and combine."""

        # Mock data for each season
        def mock_read(url: str | Path) -> pd.DataFrame:
            url_str = str(url)
            if "2022" in url_str:
                return pd.DataFrame({"play_id": [1, 2], "season": [2022, 2022]})
            elif "2023" in url_str:
                return pd.DataFrame({"play_id": [3, 4], "season": [2023, 2023]})
            else:
                raise ValueError(f"Unexpected URL: {url_str}")

        mock_read_parquet.side_effect = mock_read

        # Download seasons 2022 and 2023
        result_path = download_nflverse_pbp([2022, 2023], tmp_path)

        # Verify both seasons were downloaded
        assert mock_read_parquet.call_count == 2

        # Verify combined data - read with pyarrow directly to avoid mock
        assert result_path.exists()
        table = pq.read_table(result_path)
        result_df = table.to_pandas()
        assert len(result_df) == 4  # 2 plays per season
        assert set(result_df["season"]) == {2022, 2023}


class TestUploadToVolume:
    """Test Unity Catalog Volume upload."""

    def test_upload_constructs_correct_path(self, tmp_path: Path) -> None:
        """Upload to correct Volume path."""
        # Create mock workspace client
        mock_client = MagicMock()
        mock_client.files.upload = Mock()

        # Create test file
        test_file = tmp_path / "test_data.parquet"
        test_file.write_bytes(b"test content")

        # Upload
        volume_path = upload_to_volume(
            mock_client,
            test_file,
            dataset="games",
            seasons=[2022, 2023],
        )

        # Verify path format
        assert volume_path == "/Volumes/workspace/nfl_raw/landing/games/seasons_2022_2023.parquet"

        # Verify upload was called
        mock_client.files.upload.assert_called_once()
        call_args = mock_client.files.upload.call_args
        assert call_args[0][0] == volume_path
        assert call_args[0][1] == b"test content"
        assert call_args[1]["overwrite"] is True


class TestCreateSchemaAndVolume:
    """Test schema and volume creation."""

    def test_creates_schema_and_volume(self) -> None:
        """Execute SQL to create schema and volume."""
        # Mock SQL connection and cursor
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Create schema and volume
        create_schema_and_volume(mock_connection)

        # Verify cursor was used
        mock_connection.cursor.assert_called_once()

        # Verify SQL was executed (2 calls: schema and volume)
        assert mock_cursor.execute.call_count == 2

        # Verify schema creation SQL
        schema_call = mock_cursor.execute.call_args_list[0][0][0]
        assert "CREATE SCHEMA IF NOT EXISTS" in schema_call
        assert "workspace.nfl_raw" in schema_call

        # Verify volume creation SQL
        volume_call = mock_cursor.execute.call_args_list[1][0][0]
        assert "CREATE VOLUME IF NOT EXISTS" in volume_call


class TestCreateBronzeTable:
    """Test bronze Delta table creation."""

    def test_creates_table_from_volume(self) -> None:
        """Create Delta table using read_files from Volume."""
        # Mock SQL connection and cursor
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100,)  # Row count
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        # Create bronze table
        create_bronze_table(
            mock_connection,
            dataset="games",
            volume_path="/Volumes/workspace/nfl_raw/landing/games/test.parquet",
        )

        # Verify SQL was executed
        assert mock_cursor.execute.call_count == 2  # CREATE TABLE + COUNT

        # Verify CREATE TABLE SQL
        create_call = mock_cursor.execute.call_args_list[0][0][0]
        assert "CREATE OR REPLACE TABLE" in create_call
        assert "workspace.nfl_raw.games" in create_call
        assert "read_files" in create_call
        assert "/Volumes/workspace/nfl_raw/landing/games/test.parquet" in create_call

        # Verify COUNT SQL
        count_call = mock_cursor.execute.call_args_list[1][0][0]
        assert "SELECT COUNT(*)" in count_call


class TestDatasetConfig:
    """Test dataset configuration constants."""

    def test_games_config_exists(self) -> None:
        """Games dataset is configured."""
        assert "games" in DATASET_CONFIG
        assert DATASET_CONFIG["games"]["table_name"] == "games"
        assert DATASET_CONFIG["games"]["supports_season_filter"] is True

    def test_pbp_config_exists(self) -> None:
        """Play-by-play dataset is configured."""
        assert "pbp" in DATASET_CONFIG
        assert DATASET_CONFIG["pbp"]["table_name"] == "play_by_play"
        assert DATASET_CONFIG["pbp"]["supports_season_filter"] is True
