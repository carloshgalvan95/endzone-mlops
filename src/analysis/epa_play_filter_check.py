"""Count play types inside the EPA play filter, and kneel or spike rows.

``fct_team_game_epa`` keeps rows with ``play = 1``. This check asks which
``play_type`` values survive that filter, and how many ``qb_kneel = 1`` and
``qb_spike = 1`` rows exist. Staging does not have to carry the flag columns.
When they are absent, the counts come from the raw play-by-play table.

The check does not change a dbt model. It reads through the same Databricks
SQL client training uses. One statement per ``cursor.execute``.
"""

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.databricks_client import (
    DEFAULT_CATALOG,
    DatabricksClient,
    qualify_table_name,
    quote_identifier,
)
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

STAGING_TABLE: str = "stg_nflverse__pbp"
RAW_TABLE: str = "play_by_play"
DEFAULT_STAGING_SCHEMA: str = "nfl_dev"
DEFAULT_RAW_SCHEMA: str = "nfl_raw"
FLAG_COLUMNS: tuple[str, ...] = ("qb_kneel", "qb_spike")
DEFAULT_OUTPUT: str = "reports/epa_filter_check.md"


def _resolve_setting(explicit: str | None, env_name: str, default: str) -> str:
    """Prefer an explicit argument, then a non-empty env var, then a default.

    Args:
        explicit: Value passed by the caller. None or blank means unset.
        env_name: Environment variable that overrides the default.
        default: Value used when neither argument nor env var is set.

    Returns:
        The resolved setting.
    """
    if explicit is not None and explicit.strip() != "":
        return explicit.strip()
    from_env: str | None = os.getenv(env_name)
    if from_env is not None and from_env.strip() != "":
        return from_env.strip()
    return default


def resolve_epa_filter_locations(
    catalog: str | None = None,
    staging_schema: str | None = None,
    raw_schema: str | None = None,
) -> dict[str, str]:
    """Resolve the staging view and the raw play-by-play table.

    Staging lives in ``nfl_dev`` (the dbt target schema). Marts live in
    ``nfl_dev_marts``. The raw table is ``workspace.nfl_raw.play_by_play``.
    ``DATABRICKS_CATALOG``, ``DATABRICKS_STAGING_SCHEMA``, and
    ``DATABRICKS_RAW_SCHEMA`` override those defaults.

    Args:
        catalog: Unity Catalog name, or None to use the env var or default.
        staging_schema: Schema of ``stg_nflverse__pbp``.
        raw_schema: Schema of the bronze play-by-play table.

    Returns:
        Dict with ``catalog``, ``staging_schema``, ``raw_schema``,
        ``staging_table``, and ``raw_table``. Table values are qualified
        and backtick-quoted.
    """
    resolved_catalog: str = _resolve_setting(catalog, "DATABRICKS_CATALOG", DEFAULT_CATALOG)
    resolved_staging: str = _resolve_setting(
        staging_schema,
        "DATABRICKS_STAGING_SCHEMA",
        DEFAULT_STAGING_SCHEMA,
    )
    resolved_raw: str = _resolve_setting(raw_schema, "DATABRICKS_RAW_SCHEMA", DEFAULT_RAW_SCHEMA)
    return {
        "catalog": resolved_catalog,
        "staging_schema": resolved_staging,
        "raw_schema": resolved_raw,
        "staging_table": qualify_table_name(resolved_catalog, resolved_staging, STAGING_TABLE),
        "raw_table": qualify_table_name(resolved_catalog, resolved_raw, RAW_TABLE),
    }


def describe_table_sql(qualified_table: str) -> str:
    """One ``DESCRIBE TABLE`` statement.

    Args:
        qualified_table: Backtick-quoted table reference.

    Returns:
        A single SQL statement with no trailing semicolon.
    """
    return f"DESCRIBE TABLE {qualified_table}"


def play_type_counts_sql(qualified_table: str) -> str:
    """Count ``play = 1`` rows by ``play_type``.

    Args:
        qualified_table: Backtick-quoted staging view.

    Returns:
        A single SQL statement with no trailing semicolon.
    """
    return (
        "SELECT play_type, COUNT(*) AS n_plays "
        f"FROM {qualified_table} "
        "WHERE play = 1 "
        "GROUP BY play_type "
        "ORDER BY n_plays DESC"
    )


def flag_counts_sql(
    qualified_table: str,
    columns: tuple[str, ...] = FLAG_COLUMNS,
) -> str:
    """Count flag = 1 rows, and the subset that also has ``play = 1``.

    Args:
        qualified_table: Backtick-quoted table that has the flag columns.
        columns: Flag column names. Defaults to kneel and spike.

    Returns:
        A single SQL statement with no trailing semicolon.
    """
    selects: list[str] = []
    play_column: str = quote_identifier("play")
    for column in columns:
        quoted: str = quote_identifier(column)
        selects.append(
            f"SUM(CASE WHEN {quoted} = 1 THEN 1 ELSE 0 END) AS {quote_identifier(column + '_n')}"
        )
        selects.append(
            f"SUM(CASE WHEN {quoted} = 1 AND {play_column} = 1 THEN 1 ELSE 0 END) "
            f"AS {quote_identifier(column + '_play_n')}"
        )
    return f"SELECT {', '.join(selects)} FROM {qualified_table}"


def _cell_is_missing(value: object) -> bool:
    """True for None and floating NaN. Other objects are present."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def column_names_from_describe(described: pd.DataFrame) -> list[str]:
    """Read column names from a ``DESCRIBE TABLE`` result.

    A blank name or a name that starts with ``#`` ends the column list.
    Databricks prints partition metadata after that break.

    Args:
        described: Result of ``DESCRIBE TABLE``.

    Returns:
        Column names in table order, before the metadata break.

    Raises:
        ValueError: The result has neither ``col_name`` nor ``column_name``.
    """
    lowered: dict[str, str] = {str(column).lower(): str(column) for column in described.columns}
    if "col_name" in lowered:
        key: str = lowered["col_name"]
    elif "column_name" in lowered:
        key = lowered["column_name"]
    else:
        raise ValueError("DESCRIBE result has no col_name column")
    names: list[str] = []
    for raw in described[key].tolist():
        if _cell_is_missing(raw):
            break
        text: str = str(raw).strip()
        if text == "" or text.startswith("#"):
            break
        names.append(text)
    return names


def _frame_column(frame: pd.DataFrame, name: str) -> str:
    """Return the actual column whose name matches ``name``, ignoring case.

    Args:
        frame: Query result.
        name: Expected column name, without backticks.

    Returns:
        The column label on ``frame``.

    Raises:
        ValueError: ``name`` is not in the result.
    """
    for column in frame.columns:
        cleaned: str = str(column).strip("`")
        if cleaned.lower() == name.lower():
            return str(column)
    raise ValueError(f"query result has no {name} column")


def _count_cell(value: object) -> int:
    """Convert one aggregate cell to an int.

    Args:
        value: A count from the warehouse or a test double.

    Returns:
        The count as an int.

    Raises:
        ValueError: The cell is missing.
    """
    if _cell_is_missing(value):
        raise ValueError("flag count is missing")
    return int(value)


def _play_type_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn the play-type aggregate into rows with a share of ``play = 1``.

    Args:
        frame: Columns ``play_type`` and ``n_plays``.

    Returns:
        One dict per play type, in the query's order. An empty frame is an
        empty list.
    """
    if frame.empty:
        return []
    type_column: str = _frame_column(frame, "play_type")
    count_column: str = _frame_column(frame, "n_plays")
    counts: list[int] = [_count_cell(value) for value in frame[count_column].tolist()]
    total: int = sum(counts)
    rows: list[dict[str, Any]] = []
    for raw_type, n_plays in zip(frame[type_column].tolist(), counts, strict=True):
        if _cell_is_missing(raw_type):
            label = "(null)"
        else:
            label = str(raw_type)
        share: float = 0.0 if total == 0 else n_plays / total
        rows.append({"play_type": label, "n_plays": n_plays, "share": share})
    return rows


def _flag_rows(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read kneel and spike counts from a one-row aggregate.

    Args:
        frame: One row of flag aggregates.
        columns: Flag names, without the ``_n`` suffix.

    Returns:
        One dict per flag, with ``n_flag_1`` and ``n_flag_1_and_play_1``.

    Raises:
        ValueError: The aggregate returned no row, or a count column is absent.
    """
    if frame.empty:
        raise ValueError("flag count query returned no rows")
    row: pd.Series = frame.iloc[0]
    flags: list[dict[str, Any]] = []
    for column in columns:
        n_name: str = _frame_column(frame, f"{column}_n")
        play_name: str = _frame_column(frame, f"{column}_play_n")
        flags.append(
            {
                "column": column,
                "n_flag_1": _count_cell(row[n_name]),
                "n_flag_1_and_play_1": _count_cell(row[play_name]),
            }
        )
    return flags


def run_epa_play_filter_check(
    client: Any,
    locations: dict[str, str],
    flag_columns: tuple[str, ...] = FLAG_COLUMNS,
) -> dict[str, Any]:
    """Query play types on staging, then kneel and spike counts.

    ``client.query`` runs one statement at a time. Flag columns that are
    missing from staging are read from the raw table instead.

    Args:
        client: Object with ``query(sql) -> DataFrame``. The Databricks
            client is the production object. Tests pass a double.
        locations: Result of ``resolve_epa_filter_locations``.
        flag_columns: Columns counted as kneel or spike flags.

    Returns:
        Dict with ``staging_table``, ``flag_source`` (``staging`` or
        ``raw``), ``flag_table``, ``play_types``, and ``flags``.

    Raises:
        ValueError: The flag columns are on neither table, or a query
            result is missing a required column.
    """
    staging_table: str = locations["staging_table"]
    raw_table: str = locations["raw_table"]
    staging_columns: list[str] = column_names_from_describe(
        client.query(describe_table_sql(staging_table))
    )
    play_types: list[dict[str, Any]] = _play_type_rows(
        client.query(play_type_counts_sql(staging_table))
    )
    missing: list[str] = [column for column in flag_columns if column not in staging_columns]
    if not missing:
        flag_source = "staging"
        flag_table = staging_table
        flag_frame: pd.DataFrame = client.query(flag_counts_sql(staging_table, flag_columns))
    else:
        raw_columns: list[str] = column_names_from_describe(
            client.query(describe_table_sql(raw_table))
        )
        absent: list[str] = [column for column in flag_columns if column not in raw_columns]
        if absent:
            joined: str = ", ".join(absent)
            raise ValueError(f"{joined} not on staging {staging_table} or raw {raw_table}")
        flag_source = "raw"
        flag_table = raw_table
        flag_frame = client.query(flag_counts_sql(raw_table, flag_columns))
    return {
        "staging_table": staging_table,
        "flag_source": flag_source,
        "flag_table": flag_table,
        "play_types": play_types,
        "flags": _flag_rows(flag_frame, flag_columns),
    }


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a GitHub-flavored table. Floats use four decimal places."""
    rendered: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            if isinstance(cell, float):
                cells.append(f"{cell:.4f}")
            else:
                cells.append(str(cell))
        rendered.append(cells)
    head: str = "| " + " | ".join(headers) + " |"
    separator: str = "| " + " | ".join("---" for _ in headers) + " |"
    body: list[str] = ["| " + " | ".join(cells) + " |" for cells in rendered]
    return "\n".join([head, separator, *body])


def render_epa_filter_report(report: dict[str, Any]) -> str:
    """Render the play-type and flag-count report.

    Args:
        report: Result of ``run_epa_play_filter_check``.

    Returns:
        Markdown text. It does not contain an em dash.
    """
    lines: list[str] = [
        "# EPA play filter check",
        "",
        f"Staging table: `{report['staging_table']}`.",
        "Rows with `play = 1`, grouped by `play_type`.",
        "This is the filter `fct_team_game_epa` uses before it averages EPA.",
        "No dbt model was changed.",
        "",
        "## Play types",
        "",
    ]
    play_types: list[dict[str, Any]] = list(report["play_types"])
    if not play_types:
        lines.append("No rows with play = 1.")
    else:
        lines.append(
            _markdown_table(
                ["play_type", "n_plays", "share"],
                [[row["play_type"], row["n_plays"], row["share"]] for row in play_types],
            )
        )
    lines.append("")
    if report["flag_source"] == "staging":
        lines.append(f"Kneel and spike counts come from staging `{report['flag_table']}`.")
    else:
        lines.append(
            "`qb_kneel` and `qb_spike` are not both on "
            f"`{report['staging_table']}`. Counts come from `{report['flag_table']}`."
        )
    lines.append("")
    lines.append("`n_flag_1` counts rows where the flag equals 1.")
    lines.append("`n_flag_1_and_play_1` counts those rows that also have `play = 1`.")
    lines.append("")
    lines.append(
        _markdown_table(
            ["column", "n_flag_1", "n_flag_1_and_play_1"],
            [
                [row["column"], row["n_flag_1"], row["n_flag_1_and_play_1"]]
                for row in report["flags"]
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def write_epa_filter_report(report: dict[str, Any], output_path: Path) -> dict[str, str]:
    """Write the markdown report, creating parent directories when needed.

    Args:
        report: Result of ``run_epa_play_filter_check``.
        output_path: Destination file.

    Returns:
        Dict with ``markdown_path`` and ``markdown``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown: str = render_epa_filter_report(report)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return {"markdown_path": str(output_path), "markdown": markdown}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the check CLI.

    Args:
        argv: Arguments, or None to read the process arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Count play types inside the EPA play filter, plus kneel and spike rows."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Markdown path. Default: reports/epa_filter_check.md",
    )
    parser.add_argument("--catalog", default=None, help="Unity Catalog. Default: workspace.")
    parser.add_argument(
        "--staging-schema",
        default=None,
        help="Schema of stg_nflverse__pbp. Default: nfl_dev.",
    )
    parser.add_argument(
        "--raw-schema",
        default=None,
        help="Schema of the raw play-by-play table. Default: nfl_raw.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Query the warehouse and write ``reports/epa_filter_check.md``.

    Uses ``DatabricksClient`` the same way training does. Catalog and schema
    overrides change the qualified table names. The connection still comes
    from ``DATABRICKS_HOST``, ``DATABRICKS_HTTP_PATH``, and ``DATABRICKS_TOKEN``.

    Args:
        argv: CLI arguments. None reads the process arguments.

    Returns:
        Dict with ``report`` and ``markdown_path``.

    Raises:
        RuntimeError: The client is in fixture mode because credentials
            are missing.
        ValueError: The flag columns cannot be found.
    """
    args = _parse_args(argv)
    locations: dict[str, str] = resolve_epa_filter_locations(
        catalog=args.catalog,
        staging_schema=args.staging_schema,
        raw_schema=args.raw_schema,
    )
    client = DatabricksClient()
    try:
        report: dict[str, Any] = run_epa_play_filter_check(client, locations)
    finally:
        client.close()
    written: dict[str, str] = write_epa_filter_report(report, Path(args.output))
    return {"report": report, "markdown_path": written["markdown_path"]}


if __name__ == "__main__":  # pragma: no cover
    main()
