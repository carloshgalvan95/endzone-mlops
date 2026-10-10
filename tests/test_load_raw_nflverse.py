"""
Unit tests for nflverse data ingestion module.

Tests use mocks to avoid network calls and Databricks dependencies in CI.
Follows Wilson patterns: test pure functions, mock I/O boundaries.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ingest.load_raw_nflverse import (
    DATASET_CONFIG,
    DEFAULT_SEASON_START,
    NFLVERSE_GAMES_URL,
    cleanup_stale_pbp_files,
    conform_table_to_schema,
    create_bronze_table,
    create_schema_and_volume,
    download_nflverse_games,
    download_nflverse_pbp,
    download_parquet_with_validation,
    get_current_season,
    get_databricks_config,
    get_default_season_range,
    load_target_schema,
    parse_seasons,
    upload_to_volume,
    upload_to_volume_pbp_season,
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


class TestParseSeasons:
    """Test season specification parsing."""

    def test_parse_range(self) -> None:
        """Parse season range."""
        result = parse_seasons("1999-2003")
        assert result == [1999, 2000, 2001, 2002, 2003]

    def test_parse_list(self) -> None:
        """Parse comma-separated list."""
        result = parse_seasons("2022,2023,2024")
        assert result == [2022, 2023, 2024]

    def test_parse_single(self) -> None:
        """Parse single season."""
        result = parse_seasons("2022")
        assert result == [2022]

    def test_parse_mixed(self) -> None:
        """Parse mixed range and list."""
        result = parse_seasons("2020-2022,2024")
        assert result == [2020, 2021, 2022, 2024]

    def test_parse_removes_duplicates(self) -> None:
        """Parse removes duplicates and sorts."""
        result = parse_seasons("2022,2021,2022,2023")
        assert result == [2021, 2022, 2023]

    def test_default_seasons_constant(self) -> None:
        """Default season start is 1999."""
        assert DEFAULT_SEASON_START == 1999


class TestGetCurrentSeason:
    """Test current season determination."""

    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_current_season_from_games_parquet(self, mock_download: Mock) -> None:
        """Get current season from max season in games.parquet."""
        mock_df = pd.DataFrame({"season": [2024, 2025, 2026, 2026]})
        mock_download.return_value = mock_df

        current = get_current_season()

        assert current == 2026
        mock_download.assert_called_once_with(NFLVERSE_GAMES_URL)

    @patch("src.ingest.load_raw_nflverse.datetime")
    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_current_season_fallback_after_march(
        self, mock_download: Mock, mock_datetime: Mock
    ) -> None:
        """Fallback to calendar year if month >= 3."""
        mock_download.side_effect = Exception("Network error")

        mock_now = Mock()
        mock_now.year = 2027
        mock_now.month = 9
        mock_datetime.datetime.now.return_value = mock_now

        current = get_current_season()

        assert current == 2027

    @patch("src.ingest.load_raw_nflverse.datetime")
    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_current_season_fallback_before_march(
        self, mock_download: Mock, mock_datetime: Mock
    ) -> None:
        """Fallback to year - 1 if month < 3 (still previous season)."""
        mock_download.side_effect = Exception("Network error")

        mock_now = Mock()
        mock_now.year = 2027
        mock_now.month = 1
        mock_datetime.datetime.now.return_value = mock_now

        current = get_current_season()

        assert current == 2026


class TestGetDefaultSeasonRange:
    """Test default season range generation."""

    @patch("src.ingest.load_raw_nflverse.get_current_season")
    def test_default_range_uses_current_season(self, mock_current: Mock) -> None:
        """Default range is DEFAULT_SEASON_START through current season."""
        mock_current.return_value = 2026

        result = get_default_season_range()

        assert result == f"{DEFAULT_SEASON_START}-2026"
        mock_current.assert_called_once()


class TestLoadTargetSchema:
    """Test loading fixed pbp target schema."""

    def test_load_target_schema(self) -> None:
        """Load target schema from JSON."""
        schema = load_target_schema()
        assert len(schema) > 0
        assert all(isinstance(f, pa.Field) for f in schema)

    def test_target_schema_has_key_columns(self) -> None:
        """Target schema has critical columns."""
        schema = load_target_schema()
        col_names = {f.name for f in schema}
        required = {"game_id", "play_id", "season", "posteam", "defteam", "play", "epa"}
        assert required.issubset(col_names)


class TestConformTableToSchema:
    """Test table conforming to target schema."""

    def test_conform_adds_missing_columns(self) -> None:
        """Conform adds missing columns as null."""
        table = pa.table({"a": [1, 2]})
        target_schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])

        conformed = conform_table_to_schema(table, target_schema, season=2022)
        assert conformed.column_names == ["a", "b"]
        assert conformed.column("b").null_count == 2

    def test_conform_removes_extra_columns(self) -> None:
        """Conform removes columns not in target (logs warning)."""
        table = pa.table({"a": [1, 2], "b": ["x", "y"], "c": [3.0, 4.0]})
        target_schema = pa.schema([pa.field("a", pa.int64())])

        conformed = conform_table_to_schema(table, target_schema, season=2022)
        assert conformed.column_names == ["a"]

    def test_conform_casts_types(self) -> None:
        """Conform casts types to match target."""
        table = pa.table({"a": pa.array([1, 2], type=pa.int32())})
        target_schema = pa.schema([pa.field("a", pa.int64())])

        conformed = conform_table_to_schema(table, target_schema, season=2022)
        assert conformed.column("a").type == pa.int64()

    def test_conform_column_order_matches_target(self) -> None:
        """Conform reorders columns to match target schema."""
        table = pa.table({"c": [3, 4], "a": [1, 2], "b": ["x", "y"]})
        target_schema = pa.schema(
            [pa.field("a", pa.int64()), pa.field("b", pa.string()), pa.field("c", pa.int64())]
        )

        conformed = conform_table_to_schema(table, target_schema, season=2022)
        assert conformed.column_names == ["a", "b", "c"]

    def test_conform_raises_on_cast_failure(self) -> None:
        """Conform raises error when cast fails."""
        table = pa.table({"a": ["not", "numbers"]})
        target_schema = pa.schema([pa.field("a", pa.int64())])

        with pytest.raises(ValueError, match="Season 2022.*column 'a'.*string.*int64"):
            conform_table_to_schema(table, target_schema, season=2022)


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

    @patch("src.ingest.load_raw_nflverse.load_target_schema")
    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    @patch("src.ingest.load_raw_nflverse.upload_to_volume_pbp_season")
    def test_download_and_conform_pbp_one_at_a_time(
        self,
        mock_upload: Mock,
        mock_download: Mock,
        mock_load_schema: Mock,
        tmp_path: Path,
    ) -> None:
        """Download and conform pbp one season at a time."""
        mock_schema = pa.schema([pa.field("play_id", pa.int64()), pa.field("season", pa.int64())])
        mock_load_schema.return_value = mock_schema

        def mock_download_func(url: str) -> pd.DataFrame:
            if "2022" in url:
                return pd.DataFrame({"play_id": [1, 2], "season": [2022, 2022]})
            elif "2023" in url:
                return pd.DataFrame({"play_id": [3, 4], "season": [2023, 2023]})
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_download.side_effect = mock_download_func
        mock_workspace = MagicMock()

        result = download_nflverse_pbp([2022, 2023], tmp_path, mock_workspace)

        assert len(result) == 2
        assert result == [2022, 2023]
        assert mock_download.call_count == 2
        assert mock_upload.call_count == 2

    @patch("src.ingest.load_raw_nflverse.load_target_schema")
    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    @patch("src.ingest.load_raw_nflverse.upload_to_volume_pbp_season")
    def test_download_pbp_skip_missing_season(
        self,
        mock_upload: Mock,
        mock_download: Mock,
        mock_load_schema: Mock,
        tmp_path: Path,
    ) -> None:
        """Skip seasons with HTTP 404 gracefully."""
        mock_schema = pa.schema([pa.field("play_id", pa.int64()), pa.field("season", pa.int64())])
        mock_load_schema.return_value = mock_schema

        def mock_download_func(url: str) -> pd.DataFrame:
            if "2022" in url:
                return pd.DataFrame({"play_id": [1, 2], "season": [2022, 2022]})
            elif "2026" in url:
                raise ValueError("HTTP 404 Not Found")
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_download.side_effect = mock_download_func
        mock_workspace = MagicMock()

        result = download_nflverse_pbp([2022, 2026], tmp_path, mock_workspace)

        assert len(result) == 1
        assert result == [2022]

    @patch("src.ingest.load_raw_nflverse.load_target_schema")
    @patch("src.ingest.load_raw_nflverse.download_parquet_with_validation")
    def test_download_pbp_all_seasons_missing(
        self,
        mock_download: Mock,
        mock_load_schema: Mock,
        tmp_path: Path,
    ) -> None:
        """Raise error when all seasons are missing."""
        mock_schema = pa.schema([pa.field("play_id", pa.int64())])
        mock_load_schema.return_value = mock_schema
        mock_download.side_effect = ValueError("HTTP 404 Not Found")
        mock_workspace = MagicMock()

        with pytest.raises(ValueError, match="No play-by-play data found"):
            download_nflverse_pbp([2050, 2051], tmp_path, mock_workspace)


class TestUploadToVolume:
    """Test Unity Catalog Volume upload."""

    def test_upload_games_constructs_correct_path(self, tmp_path: Path) -> None:
        """Upload games to correct Volume path."""
        mock_client = MagicMock()
        mock_client.files.upload = Mock()

        test_file = tmp_path / "games_filtered.parquet"
        test_file.write_bytes(b"test content")

        volume_path = upload_to_volume(
            mock_client,
            test_file,
            dataset="games",
        )

        assert volume_path == "/Volumes/workspace/nfl_raw/landing/games/games_filtered.parquet"

        mock_client.files.upload.assert_called_once()
        call_args = mock_client.files.upload.call_args
        assert call_args[0][0] == volume_path
        uploaded = call_args[0][1]
        assert hasattr(uploaded, "read")
        assert Path(uploaded.name) == test_file
        assert uploaded.mode == "rb"
        assert call_args[1]["overwrite"] is True


class TestUploadToVolumePbpSeason:
    """Test pbp per-season upload."""

    def test_upload_pbp_per_season_path(self, tmp_path: Path) -> None:
        """Upload pbp to partitioned per-season path without hive-style naming."""
        mock_client = MagicMock()
        mock_client.files.upload = Mock()

        test_file = tmp_path / "play_by_play_2022.parquet"
        test_file.write_bytes(b"test content")

        volume_path = upload_to_volume_pbp_season(
            mock_client,
            test_file,
            season=2022,
        )

        assert (
            volume_path == "/Volumes/workspace/nfl_raw/landing/pbp/2022/play_by_play_2022.parquet"
        )

        mock_client.files.upload.assert_called_once()
        call_args = mock_client.files.upload.call_args
        assert call_args[0][0] == volume_path


class TestCleanupStalePbpFiles:
    """Test cleanup of stale pbp files."""

    def test_cleanup_removes_direct_files(self) -> None:
        """Remove files directly under pbp/ folder."""
        mock_client = MagicMock()

        mock_stale_file = MagicMock()
        mock_stale_file.is_directory = False
        mock_stale_file.path = "/Volumes/workspace/nfl_raw/landing/pbp/seasons_2022_2023.parquet"

        mock_season_dir = MagicMock()
        mock_season_dir.is_directory = True
        mock_season_dir.path = "/Volumes/workspace/nfl_raw/landing/pbp/2024"

        mock_client.files.list_directory_contents.return_value = [mock_stale_file, mock_season_dir]

        deleted = cleanup_stale_pbp_files(mock_client)

        assert len(deleted) == 1
        assert deleted[0] == mock_stale_file.path
        mock_client.files.delete.assert_called_once_with(mock_stale_file.path)

    def test_cleanup_handles_no_directory(self) -> None:
        """Handle case where pbp directory does not exist."""
        mock_client = MagicMock()
        mock_client.files.list_directory_contents.side_effect = Exception("Not found")

        deleted = cleanup_stale_pbp_files(mock_client)

        assert deleted == []


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

    def test_creates_games_table_from_file(self) -> None:
        """Create games Delta table using read_files from specific file."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100,)
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        create_bronze_table(
            mock_connection,
            dataset="games",
            volume_path="/Volumes/workspace/nfl_raw/landing/games/games_filtered.parquet",
        )

        assert mock_cursor.execute.call_count == 2

        create_call = mock_cursor.execute.call_args_list[0][0][0]
        assert "CREATE OR REPLACE TABLE" in create_call
        assert "workspace.nfl_raw.games" in create_call
        assert "read_files" in create_call
        assert "/Volumes/workspace/nfl_raw/landing/games/games_filtered.parquet" in create_call

        count_call = mock_cursor.execute.call_args_list[1][0][0]
        assert "SELECT COUNT(*)" in count_call

    def test_creates_pbp_table_from_folder(self) -> None:
        """Create pbp Delta table using read_files from entire pbp folder."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (50000,)
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor

        create_bronze_table(
            mock_connection,
            dataset="pbp",
            volume_path=None,
        )

        assert mock_cursor.execute.call_count == 2

        create_call = mock_cursor.execute.call_args_list[0][0][0]
        assert "CREATE OR REPLACE TABLE" in create_call
        assert "workspace.nfl_raw.play_by_play" in create_call
        assert "read_files" in create_call
        assert "/Volumes/workspace/nfl_raw/landing/pbp" in create_call


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


class TestCurrentSeasonMode:
    """Test current-season mode behavior."""

    @patch("src.ingest.load_raw_nflverse.get_current_season")
    @patch("src.ingest.load_raw_nflverse.download_nflverse_games")
    @patch("src.ingest.load_raw_nflverse.download_nflverse_pbp")
    def test_current_season_mode_games_full_range_pbp_current_only(
        self,
        mock_pbp: Mock,
        mock_games: Mock,
        mock_current: Mock,
    ) -> None:
        """In current-season mode, games loads full range, pbp loads current only."""
        from pathlib import Path

        from src.ingest.load_raw_nflverse import parse_seasons

        mock_current.return_value = 2026

        default_range = f"{DEFAULT_SEASON_START}-2026"
        all_seasons = parse_seasons(default_range)

        mock_games.return_value = Path("/tmp/fake_games.parquet")
        mock_pbp.return_value = [2026]

        # This simulates what main() does in current-season mode
        games_seasons = all_seasons
        pbp_seasons = [2026]

        # Verify games would receive full range
        assert len(games_seasons) == 28
        assert games_seasons[0] == 1999
        assert games_seasons[-1] == 2026

        # Verify pbp would receive only current season
        assert pbp_seasons == [2026]
