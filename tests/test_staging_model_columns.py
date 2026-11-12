"""
Test that dbt staging models only reference columns that exist in source data.

This test parses staging model SQL files and validates that all column references
exist in the real nflverse parquet schemas (captured in fixtures).

Prevents UNRESOLVED_COLUMN errors in Databricks by catching nonexistent columns in CI.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cross_platform


def load_column_fixture(fixture_name: str) -> set[str]:
    """
    Load column names from fixture file.

    Args:
        fixture_name: Name of fixture file (e.g., 'nflverse_pbp_columns.txt')

    Returns:
        Set of column names
    """
    fixture_path = Path(__file__).parent / "fixtures" / fixture_name
    text = fixture_path.read_text(encoding="utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def extract_columns_from_sql(sql_path: Path) -> set[str]:
    """
    Extract column names from dbt staging model SQL file.

    Parses the SELECT statement to find all column references in the source CTE.
    Handles column aliases (e.g., 'raw_col as clean_col').

    Args:
        sql_path: Path to SQL file

    Returns:
        Set of raw column names (before 'as' alias)
    """
    content = sql_path.read_text(encoding="utf-8")

    # Find the 'with source as (...' CTE
    source_cte_match = re.search(
        r"with\s+source\s+as\s*\((.*?)\)\s*select\s+\*\s+from\s+source",
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if not source_cte_match:
        raise ValueError(f"Could not find 'with source as' CTE in {sql_path}")

    source_cte = source_cte_match.group(1)

    # Extract SELECT clause (between 'select' and 'from')
    select_match = re.search(
        r"select\s+(.*?)\s+from\s+{{",
        source_cte,
        re.DOTALL | re.IGNORECASE,
    )

    if not select_match:
        raise ValueError(f"Could not find SELECT clause in source CTE of {sql_path}")

    select_clause = select_match.group(1)

    # Extract column names (handle aliases with 'as')
    columns = set()
    for line in select_clause.split("\n"):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("--"):
            continue

        # Remove trailing comma
        line = line.rstrip(",")

        # Handle 'column as alias' pattern
        if " as " in line.lower():
            # Extract the part before 'as'
            raw_col = re.split(r"\s+as\s+", line, flags=re.IGNORECASE)[0].strip()
            columns.add(raw_col)
        else:
            # Plain column name
            columns.add(line)

    return columns


class TestStagingModelColumns:
    """Test that staging models only reference valid source columns."""

    @pytest.mark.parametrize(
        ("relative_sql", "fixture_name"),
        [
            ("dbt/models/staging/stg_nflverse__games.sql", "nflverse_games_columns.txt"),
            ("dbt/models/staging/stg_nflverse__pbp.sql", "nflverse_pbp_columns.txt"),
            (
                "dbt/models/staging/stg_nflverse__depth_charts.sql",
                "nflverse_depth_chart_columns.txt",
            ),
            (
                "dbt/models/staging/stg_nflverse__injuries.sql",
                "nflverse_injuries_columns.txt",
            ),
        ],
    )
    def test_staging_columns_exist_on_the_source(
        self, relative_sql: str, fixture_name: str
    ) -> None:
        """Every selected source column exists on the captured nflverse schema."""
        sql_path = Path(__file__).resolve().parents[1] / relative_sql
        fixture_columns = load_column_fixture(fixture_name)
        referenced_columns = extract_columns_from_sql(sql_path)
        missing = referenced_columns - fixture_columns
        assert not missing, (
            f"{relative_sql} references columns missing from tests/fixtures/{fixture_name}: "
            f"{sorted(missing)}"
        )

    def test_2025_depth_chart_release_has_no_week(self) -> None:
        """The 2025 release is a snapshot. Staging still reads the weekly columns."""
        weekly = load_column_fixture("nflverse_depth_chart_columns.txt")
        snapshot = load_column_fixture("nflverse_depth_chart_2025_columns.txt")
        assert "week" in weekly
        assert "game_type" in weekly
        assert "week" not in snapshot
        assert "dt" in snapshot
        assert "team" in snapshot
        assert "pos_grp" in snapshot
        assert "pos_abb" in snapshot
        assert "pos_rank" in snapshot
        assert "gsis_id" in snapshot


class TestColumnFixtures:
    """Test that column fixtures are valid."""

    def test_known_pbp_columns_present(self) -> None:
        """Fixture contains known critical pbp columns."""
        columns = load_column_fixture("nflverse_pbp_columns.txt")

        required = {
            "game_id",
            "play_id",
            "season",
            "posteam",
            "defteam",
            "play_type",
            "epa",
            "wpa",
            "play",
        }
        missing = required - columns
        assert not missing, f"Critical pbp columns missing from fixture: {missing}"

    def test_known_games_columns_present(self) -> None:
        """Fixture contains known critical games columns."""
        columns = load_column_fixture("nflverse_games_columns.txt")

        required = {"game_id", "season", "week", "away_team", "home_team"}
        missing = required - columns
        assert not missing, f"Critical games columns missing from fixture: {missing}"
