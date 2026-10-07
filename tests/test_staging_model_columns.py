"""
Test that dbt staging models only reference columns that exist in source data.

This test parses staging model SQL files and validates that all column references
exist in the real nflverse parquet schemas (captured in fixtures).

Prevents UNRESOLVED_COLUMN errors in Databricks by catching nonexistent columns in CI.
"""

import re
from pathlib import Path

import pytest


def load_column_fixture(fixture_name: str) -> set[str]:
    """
    Load column names from fixture file.

    Args:
        fixture_name: Name of fixture file (e.g., 'nflverse_pbp_columns.txt')

    Returns:
        Set of column names
    """
    fixture_path = Path(__file__).parent / "fixtures" / fixture_name
    with open(fixture_path) as f:
        return {line.strip() for line in f if line.strip()}


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
    with open(sql_path) as f:
        content = f.read()

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

    def test_stg_nflverse_games_columns_exist(self) -> None:
        """All columns in stg_nflverse__games.sql exist in real games parquet."""
        sql_path = (
            Path(__file__).parent.parent / "dbt" / "models" / "staging" / "stg_nflverse__games.sql"
        )
        fixture_columns = load_column_fixture("nflverse_games_columns.txt")

        if not sql_path.exists():
            pytest.skip(f"Staging model not found: {sql_path}")

        referenced_columns = extract_columns_from_sql(sql_path)

        # Check each referenced column exists in fixture
        missing = referenced_columns - fixture_columns
        if missing:
            pytest.fail(
                f"stg_nflverse__games.sql references columns that don't exist in real games parquet: {sorted(missing)}\n"
                f"Available columns are in tests/fixtures/nflverse_games_columns.txt"
            )

    def test_stg_nflverse_pbp_columns_exist(self) -> None:
        """All columns in stg_nflverse__pbp.sql exist in real pbp parquet."""
        sql_path = (
            Path(__file__).parent.parent / "dbt" / "models" / "staging" / "stg_nflverse__pbp.sql"
        )
        fixture_columns = load_column_fixture("nflverse_pbp_columns.txt")

        if not sql_path.exists():
            pytest.skip(f"Staging model not found: {sql_path}")

        referenced_columns = extract_columns_from_sql(sql_path)

        # Check each referenced column exists in fixture
        missing = referenced_columns - fixture_columns
        if missing:
            pytest.fail(
                f"stg_nflverse__pbp.sql references columns that don't exist in real pbp parquet: {sorted(missing)}\n"
                f"Available columns are in tests/fixtures/nflverse_pbp_columns.txt"
            )


class TestColumnFixtures:
    """Test that column fixtures are valid."""

    def test_games_fixture_not_empty(self) -> None:
        """Games column fixture has content."""
        columns = load_column_fixture("nflverse_games_columns.txt")
        assert len(columns) > 0, "Games column fixture is empty"
        assert len(columns) >= 40, f"Games fixture should have ~46 columns, got {len(columns)}"

    def test_pbp_fixture_not_empty(self) -> None:
        """PBP column fixture has content."""
        columns = load_column_fixture("nflverse_pbp_columns.txt")
        assert len(columns) > 0, "PBP column fixture is empty"
        assert len(columns) >= 350, f"PBP fixture should have ~372 columns, got {len(columns)}"

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
