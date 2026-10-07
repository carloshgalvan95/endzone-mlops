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
    NFLVERSE_GAMES_URL,
    NFLVERSE_PBP_URL_PATTERN,
    create_bronze_table,
    create_schema_and_volume,
    download_nflverse_games,
    download_nflverse_pbp,
    download_parquet_with_validation,
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


class TestDownloadParquetWithValidation:
    """Test parquet download with HTTP validation."""

    @patch("src.ingest.load_raw_nflverse.requests.get")
    def test_successful_download(self, mock_get: Mock) -> None:
        """Download succeeds with HTTP 200."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_df.to_parquet()
        mock_get.return_value = mock_response

        result = download_parquet_with_validation("https://example.com/data.parquet")

        assert len(result) == 3
        mock_get.assert_called_once_with("https://example.com/data.parquet", timeout=60)

    @patch("src.ingest.load_raw_nflverse.requests.get")
    def test_http_404_raises_error(self, mock_get: Mock) -> None:
        """HTTP 404 raises ValueError with clear message."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="HTTP 404 Not Found"):
            download_parquet_with_validation("https://example.com/missing.parquet")


class TestDownloadNflverseGames:
    """Test nflverse games data download."""

    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_download_and_filter_games(
        self,
        mock_download: Mock,
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
        mock_download.return_value = mock_df

        # Download and filter to 2022 and 2023
        result_path = download_nflverse_games([2022, 2023], tmp_path)

        # Verify correct URL was used
        mock_download.assert_called_once_with(NFLVERSE_GAMES_URL)

        # Verify filtered data was saved
        assert result_path.exists()
        table = pq.read_table(result_path)
        result_df = table.to_pandas()
        assert len(result_df) == 4  # Only 2022 and 2023 games
        assert set(result_df["season"]) == {2022, 2023}

    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_download_games_no_matches(
        self,
        mock_download: Mock,
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
        mock_download.return_value = mock_df

        with pytest.raises(ValueError, match="No games found for seasons"):
            download_nflverse_games([2024, 2025], tmp_path)


class TestDownloadNflversePbp:
    """Test nflverse play-by-play data download."""

    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_download_and_combine_pbp(
        self,
        mock_download: Mock,
        tmp_path: Path,
    ) -> None:
        """Download play-by-play for multiple seasons and combine."""

        # Mock data for each season
        def mock_download_func(url: str) -> pd.DataFrame:
            if "2022" in url:
                return pd.DataFrame({"play_id": [1, 2], "season": [2022, 2022]})
            elif "2023" in url:
                return pd.DataFrame({"play_id": [3, 4], "season": [2023, 2023]})
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_download.side_effect = mock_download_func

        # Download seasons 2022 and 2023
        result_path = download_nflverse_pbp([2022, 2023], tmp_path)

        # Verify both seasons were downloaded with correct URLs
        assert mock_download.call_count == 2
        expected_urls = [
            NFLVERSE_PBP_URL_PATTERN.format(season=2022),
            NFLVERSE_PBP_URL_PATTERN.format(season=2023),
        ]
        actual_urls = [call[0][0] for call in mock_download.call_args_list]
        assert actual_urls == expected_urls

        # Verify combined data
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
        uploaded = call_args[0][1]
        assert hasattr(uploaded, "read"), "upload must receive a binary file object"
        assert Path(uploaded.name) == test_file
        assert uploaded.mode == "rb"
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
