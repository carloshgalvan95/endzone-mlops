"""Injury ingest writes one bronze schema, including a null report clock."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from endzone.ingest.cli import _season_plan
from endzone.ingest.injuries import (
    INJURY_COLUMNS,
    INJURY_FIRST_SEASON,
    download_nflverse_injuries,
    normalize_injury_frame,
)
from endzone.ingest.volume import DATASET_CONFIG, create_bronze_table


def _injury_row() -> dict[str, object]:
    """One report row with every stable column except the clock."""
    return {column: "x" for column in INJURY_COLUMNS if column != "date_modified"} | {
        "season": 2024,
        "week": 1,
    }


def test_normalize_adds_a_null_clock_and_keeps_week_as_int() -> None:
    """A 2025-shaped frame still has date_modified, stored null."""
    frame: pd.DataFrame = pd.DataFrame([_injury_row()])
    selected: pd.DataFrame = normalize_injury_frame(frame)
    assert list(selected.columns) == list(INJURY_COLUMNS)
    assert selected["date_modified"].isna().all()
    assert str(selected["week"].dtype) == "Int32"
    broken: pd.DataFrame = frame.drop(columns=["gsis_id"])
    with pytest.raises(ValueError, match="gsis_id"):
        normalize_injury_frame(broken)


@pytest.mark.cross_platform
def test_null_clock_parquet_is_a_timestamp(tmp_path: Path) -> None:
    """read_files can merge a season that published no date_modified."""
    selected: pd.DataFrame = normalize_injury_frame(pd.DataFrame([_injury_row()]))
    path: Path = tmp_path / "injuries_2025.parquet"
    selected.to_parquet(path, index=False)
    clock = pq.read_schema(path).field("date_modified")
    week = pq.read_schema(path).field("week")
    assert week.type == pa.int32()
    assert clock.type.tz == "UTC"


def test_release_fixtures_record_the_2025_column_drop() -> None:
    """The stable bronze list has date_modified. The 2025 release does not."""
    stable: set[str] = set(
        Path("tests/fixtures/nflverse_injuries_columns.txt").read_text(encoding="utf-8").split()
    )
    later: set[str] = set(
        Path("tests/fixtures/nflverse_injuries_2025_columns.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    assert stable == set(INJURY_COLUMNS)
    assert "date_modified" not in later
    assert "season_type" in later


def test_download_skips_early_seasons_and_a_missing_file(tmp_path: Path) -> None:
    """2008 has no release. HTTP 404 is skipped. A good file is uploaded."""
    uploaded: list[str] = []

    def fake_download(url: str) -> pd.DataFrame:
        if "2010" in url:
            raise ValueError("Failed to download: HTTP 404")
        return pd.DataFrame([_injury_row()])

    def fake_upload(client: object, local_path: Path, season: int) -> str:
        del client
        uploaded.append(local_path.name)
        return f"/Volumes/workspace/nfl_raw/landing/injuries/{season}/{local_path.name}"

    with (
        patch(
            "endzone.ingest.injuries.download_parquet_with_validation",
            side_effect=fake_download,
        ),
        patch("endzone.ingest.injuries.upload_to_volume_injury_season", side_effect=fake_upload),
    ):
        processed: list[int] = download_nflverse_injuries(
            [2008, 2010, 2016],
            tmp_path,
            MagicMock(),
        )
    assert processed == [2016]
    assert uploaded == ["injuries_2016.parquet"]
    assert INJURY_FIRST_SEASON == 2009


def test_download_refuses_a_broken_file_before_upload(tmp_path: Path) -> None:
    """A missing stable column does not upload."""
    broken: pd.DataFrame = pd.DataFrame([_injury_row()]).drop(columns=["gsis_id"])
    with (
        patch(
            "endzone.ingest.injuries.download_parquet_with_validation",
            return_value=broken,
        ),
        patch("endzone.ingest.injuries.upload_to_volume_injury_season") as upload,
        pytest.raises(ValueError, match="gsis_id"),
    ):
        download_nflverse_injuries([2016], tmp_path, MagicMock())
    upload.assert_not_called()


def test_bronze_replace_reads_the_injury_folder() -> None:
    """The table is workspace.nfl_raw.injuries and week is hinted as int."""
    cursor = MagicMock()
    cursor.fetchone.return_value = (10,)
    connection = MagicMock()
    connection.cursor.return_value = cursor
    create_bronze_table(connection, dataset="injuries", volume_path=None)
    statement: str = cursor.execute.call_args_list[0][0][0]
    assert "workspace.nfl_raw.injuries" in statement
    assert "/Volumes/workspace/nfl_raw/landing/injuries" in statement
    assert "schemaHints" in statement
    assert "date_modified timestamp" in statement
    assert DATASET_CONFIG["injuries"]["table_name"] == "injuries"


def test_current_season_plan_loads_injuries_for_that_year_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weekly refresh does not redownload every injury season."""
    monkeypatch.setattr("endzone.ingest.cli.get_current_season", lambda: 2026)
    monkeypatch.setattr(
        "endzone.ingest.cli.get_default_season_range",
        lambda: "1999-2026",
    )
    plan: dict[str, list[int]] = _season_plan(argparse.Namespace(seasons=None, current_season=True))
    assert plan["injuries"] == [2026]
    assert plan["games"][0] == 1999
    assert plan["games"][-1] == 2026


def test_injury_marts_keep_every_report_and_rank_one_chart() -> None:
    """SQL does not drop a post-kickoff row and does not promote depth 2."""
    reports: str = Path("dbt/models/marts/fct_injury_reports.sql").read_text(encoding="utf-8")
    starters: str = Path("dbt/models/marts/fct_depth_chart_starters.sql").read_text(
        encoding="utf-8"
    )
    lowered_reports: str = reports.lower()
    lowered_starters: str = starters.lower()
    assert "date_modified" in lowered_reports
    assert "gameday" not in lowered_reports
    assert "row_number" not in lowered_reports
    assert "depth_rank = 1" in lowered_starters
    assert "depth_chart_club_map" in lowered_starters
    assert "position = 'qb'" not in lowered_starters
    schema: str = Path("dbt/models/marts/schema.yml").read_text(encoding="utf-8")
    assert "store_failures" not in schema
    assert "fct_injury_reports" in schema
    assert "fct_depth_chart_starters" in schema
