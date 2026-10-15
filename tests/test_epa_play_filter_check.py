"""Offline tests for the EPA play-filter diagnostic."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.analysis.epa_play_filter_check import (
    FLAG_COLUMNS,
    column_names_from_describe,
    describe_table_sql,
    flag_counts_sql,
    main,
    play_type_counts_sql,
    render_epa_filter_report,
    resolve_epa_filter_locations,
    run_epa_play_filter_check,
    write_epa_filter_report,
)


class _QueryClient:
    """Records SQL and returns frames chosen by the statement text."""

    def __init__(self, routes: dict[str, pd.DataFrame]) -> None:
        self.routes = routes
        self.queries: list[str] = []

    def query(self, sql: str) -> pd.DataFrame:
        """Return the first route whose key is contained in ``sql``."""
        self.queries.append(sql)
        for key, frame in self.routes.items():
            if key in sql:
                return frame.copy()
        raise AssertionError(f"unexpected SQL: {sql}")


def _describe(names: list[str | None], column: str = "col_name") -> pd.DataFrame:
    """A DESCRIBE-shaped frame."""
    return pd.DataFrame({column: names, "data_type": ["int"] * len(names)})


def _play_types(*pairs: tuple[object, int]) -> pd.DataFrame:
    """Play-type aggregate rows."""
    return pd.DataFrame(
        {"play_type": [item[0] for item in pairs], "n_plays": [item[1] for item in pairs]}
    )


def _flags(kneel: int, kneel_play: int, spike: int, spike_play: int) -> pd.DataFrame:
    """One aggregate row for the two flags."""
    return pd.DataFrame(
        {
            "qb_kneel_n": [kneel],
            "qb_kneel_play_n": [kneel_play],
            "qb_spike_n": [spike],
            "qb_spike_play_n": [spike_play],
        }
    )


def _locations() -> dict[str, str]:
    """Default qualified names."""
    return resolve_epa_filter_locations(
        catalog="workspace", staging_schema="nfl_dev_staging", raw_schema="nfl_raw"
    )


class TestLocationsAndSql:
    """Table names follow the dbt schemas, and each query is one statement."""

    def test_defaults_and_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit arguments win, then env vars, then nfl_dev_staging and nfl_raw."""
        monkeypatch.delenv("DATABRICKS_CATALOG", raising=False)
        monkeypatch.delenv("DATABRICKS_STAGING_SCHEMA", raising=False)
        monkeypatch.delenv("DATABRICKS_RAW_SCHEMA", raising=False)
        defaults = resolve_epa_filter_locations()
        assert defaults["catalog"] == "workspace"
        assert defaults["staging_schema"] == "nfl_dev_staging"
        assert defaults["raw_schema"] == "nfl_raw"
        assert defaults["staging_table"] == "`workspace`.`nfl_dev_staging`.`stg_nflverse__pbp`"
        assert defaults["raw_table"] == "`workspace`.`nfl_raw`.`play_by_play`"

        monkeypatch.setenv("DATABRICKS_CATALOG", " other ")
        monkeypatch.setenv("DATABRICKS_STAGING_SCHEMA", " env_staging ")
        monkeypatch.setenv("DATABRICKS_RAW_SCHEMA", "")
        from_env = resolve_epa_filter_locations(catalog="  ", staging_schema=None, raw_schema=None)
        assert from_env["catalog"] == "other"
        assert from_env["staging_schema"] == "env_staging"
        assert from_env["raw_schema"] == "nfl_raw"

        explicit = resolve_epa_filter_locations(
            catalog="workspace",
            staging_schema="custom_staging",
            raw_schema="custom_raw",
        )
        assert explicit["staging_table"].endswith("`custom_staging`.`stg_nflverse__pbp`")
        assert "`custom_raw`.`play_by_play`" in explicit["raw_table"]

    def test_statements_are_single_selects(self) -> None:
        """DESCRIBE and the two aggregates do not chain statements."""
        staging = "`workspace`.`nfl_dev_staging`.`stg_nflverse__pbp`"
        describe_sql = describe_table_sql(staging)
        play_sql = play_type_counts_sql(staging)
        flag_sql = flag_counts_sql(staging)
        for statement in (describe_sql, play_sql, flag_sql):
            assert ";" not in statement
        assert describe_sql.startswith("DESCRIBE TABLE ")
        assert "WHERE play = 1" in play_sql
        assert "GROUP BY play_type" in play_sql
        assert "ORDER BY n_plays DESC" in play_sql
        assert "qb_kneel" in flag_sql
        assert "qb_spike" in flag_sql
        assert "play" in flag_sql


class TestDescribeParsing:
    """Column names stop before Databricks partition metadata."""

    def test_col_name_stops_at_a_blank_or_a_hash(self) -> None:
        """A blank cell and a hash comment both end the list."""
        blank = column_names_from_describe(
            _describe(["play", "play_type", "", "# Partition Information"])
        )
        assert blank == ["play", "play_type"]
        hashed = column_names_from_describe(_describe(["play", "# col_name"]))
        assert hashed == ["play"]
        missing_cell = column_names_from_describe(_describe(["play", None, "qb_kneel"]))
        assert missing_cell == ["play"]
        nan_cell = column_names_from_describe(
            pd.DataFrame({"col_name": ["play", float("nan")], "data_type": ["int", None]})
        )
        assert nan_cell == ["play"]

    def test_column_name_is_accepted_and_a_missing_key_is_not(self) -> None:
        """Some drivers name the field column_name. Anything else fails."""
        names = column_names_from_describe(_describe(["play", "qb_kneel"], column="column_name"))
        assert names == ["play", "qb_kneel"]
        with pytest.raises(ValueError, match="col_name"):
            column_names_from_describe(pd.DataFrame({"data_type": ["int"]}))


class TestRun:
    """Staging is preferred for the flags. Raw is the fallback."""

    def test_flags_on_staging_do_not_touch_the_raw_table(self) -> None:
        """A staging view that already has the flags is the only flag source."""
        locations = _locations()
        client = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(
                    ["play", "play_type", "qb_kneel", "qb_spike"]
                ),
                "SELECT play_type": _play_types(("pass", 80), (None, 20), ("qb_kneel", 0)),
                "qb_kneel": _flags(4, 3, 2, 1),
            }
        )
        report = run_epa_play_filter_check(client, locations)
        assert report["flag_source"] == "staging"
        assert report["flag_table"] == locations["staging_table"]
        assert all("play_by_play" not in sql for sql in client.queries)
        assert [row["play_type"] for row in report["play_types"]] == ["pass", "(null)", "qb_kneel"]
        assert report["play_types"][0]["share"] == pytest.approx(0.8)
        assert report["play_types"][2]["share"] == pytest.approx(0.0)
        assert report["flags"][0] == {
            "column": "qb_kneel",
            "n_flag_1": 4,
            "n_flag_1_and_play_1": 3,
        }
        assert report["flags"][1]["column"] == "qb_spike"
        markdown = render_epa_filter_report(report)
        assert "Kneel and spike counts come from staging" in markdown
        assert "(null)" in markdown
        assert "0.8000" in markdown
        assert "\u2014" not in markdown
        assert ";" not in " ".join(client.queries)

    def test_missing_staging_flags_are_read_from_raw(self) -> None:
        """qb_kneel and qb_spike are not on the staging view in this repo."""
        locations = _locations()
        client = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(["play", "play_type"]),
                "DESCRIBE TABLE `workspace`.`nfl_raw`": _describe(["play", "qb_kneel", "qb_spike"]),
                "SELECT play_type": _play_types(),
                "FROM `workspace`.`nfl_raw`.`play_by_play`": _flags(9, 8, 1, 0),
            }
        )
        report = run_epa_play_filter_check(client, locations)
        assert report["flag_source"] == "raw"
        assert report["flag_table"] == locations["raw_table"]
        assert report["play_types"] == []
        assert report["flags"][0]["n_flag_1"] == 9
        assert report["flags"][1]["n_flag_1_and_play_1"] == 0
        markdown = render_epa_filter_report(report)
        assert "No rows with play = 1." in markdown
        assert "are not both on" in markdown
        assert "play_by_play" in markdown

    def test_flags_missing_from_both_tables_raise(self) -> None:
        """A clear error is better than a warehouse error on an unknown column."""
        locations = _locations()
        client = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(["play"]),
                "DESCRIBE TABLE `workspace`.`nfl_raw`": _describe(["play"]),
                "SELECT play_type": _play_types(("pass", 1)),
            }
        )
        with pytest.raises(ValueError, match="qb_kneel, qb_spike"):
            run_epa_play_filter_check(client, locations, flag_columns=FLAG_COLUMNS)

    def test_empty_flag_aggregate_and_unknown_columns_raise(self) -> None:
        """A zero-row aggregate and a renamed count column are both errors."""
        locations = _locations()
        empty_flags = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(
                    ["play", "qb_kneel", "qb_spike"]
                ),
                "SELECT play_type": _play_types(("pass", 1)),
                "qb_kneel": pd.DataFrame(
                    columns=["qb_kneel_n", "qb_kneel_play_n", "qb_spike_n", "qb_spike_play_n"]
                ),
            }
        )
        with pytest.raises(ValueError, match="no rows"):
            run_epa_play_filter_check(empty_flags, locations)

        missing_count = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(
                    ["play", "qb_kneel", "qb_spike"]
                ),
                "SELECT play_type": pd.DataFrame({"other": [1]}),
                "qb_kneel": _flags(1, 1, 1, 1),
            }
        )
        with pytest.raises(ValueError, match="play_type"):
            run_epa_play_filter_check(missing_count, locations)

        missing_flag_column = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(
                    ["play", "qb_kneel", "qb_spike"]
                ),
                "SELECT play_type": _play_types(("pass", 2)),
                "qb_kneel": pd.DataFrame({"qb_kneel_n": [1]}),
            }
        )
        with pytest.raises(ValueError, match="qb_kneel_play_n"):
            run_epa_play_filter_check(missing_flag_column, locations)

    def test_backtick_aliases_and_a_null_count_are_handled(self) -> None:
        """Quoted aliases still match, and a null aggregate is an error."""
        locations = _locations()
        quoted = pd.DataFrame(
            {
                "`qb_kneel_n`": [5],
                "`qb_kneel_play_n`": [4],
                "`qb_spike_n`": [None],
                "`qb_spike_play_n`": [0],
            }
        )
        client = _QueryClient(
            {
                "DESCRIBE TABLE `workspace`.`nfl_dev_staging`": _describe(["qb_kneel", "qb_spike"]),
                "SELECT play_type": _play_types(("run", 3)),
                "qb_kneel": quoted,
            }
        )
        with pytest.raises(ValueError, match="flag count is missing"):
            run_epa_play_filter_check(client, locations)


class TestReportFileAndCli:
    """The markdown file and the CLI use the same client as training."""

    def test_write_creates_the_parent_directory(self, tmp_path: Path) -> None:
        """The report path can sit under a directory that does not exist yet."""
        report: dict[str, Any] = {
            "staging_table": "`workspace`.`nfl_dev_staging`.`stg_nflverse__pbp`",
            "flag_source": "raw",
            "flag_table": "`workspace`.`nfl_raw`.`play_by_play`",
            "play_types": [{"play_type": "pass", "n_plays": 1, "share": 1.0}],
            "flags": [
                {"column": "qb_kneel", "n_flag_1": 1, "n_flag_1_and_play_1": 1},
                {"column": "qb_spike", "n_flag_1": 0, "n_flag_1_and_play_1": 0},
            ],
        }
        destination = tmp_path / "nested" / "epa_filter_check.md"
        written = write_epa_filter_report(report, destination)
        assert Path(written["markdown_path"]) == destination
        assert destination.read_text(encoding="utf-8") == written["markdown"]
        assert "No dbt model was changed." in written["markdown"]

    def test_main_writes_the_report_through_a_mocked_connection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI arguments select the schemas. The double stands in for the warehouse."""
        queries: list[str] = []
        closed: list[bool] = []

        class FakeClient:
            """Minimal Databricks client double."""

            def query(self, sql: str) -> pd.DataFrame:
                queries.append(sql)
                if sql.startswith("DESCRIBE") and "custom_staging" in sql:
                    return _describe(["play", "play_type"])
                if sql.startswith("DESCRIBE") and "custom_raw" in sql:
                    return _describe(["play", "qb_kneel", "qb_spike"])
                if sql.startswith("SELECT play_type"):
                    return _play_types(("pass", 10), ("run", 5))
                if "custom_raw" in sql:
                    return _flags(2, 2, 1, 1)
                raise AssertionError(sql)

            def close(self) -> None:
                closed.append(True)

        monkeypatch.setattr(
            "src.analysis.epa_play_filter_check.DatabricksClient",
            FakeClient,
        )
        output = tmp_path / "epa_filter_check.md"
        result = main(
            [
                "--output",
                str(output),
                "--catalog",
                "workspace",
                "--staging-schema",
                "custom_staging",
                "--raw-schema",
                "custom_raw",
            ]
        )
        assert closed == [True]
        assert Path(result["markdown_path"]) == output
        text = output.read_text(encoding="utf-8")
        assert "custom_staging" in text
        assert "pass" in text
        assert len(queries) == 4
        assert all(";" not in sql for sql in queries)

    def test_main_closes_the_client_when_the_warehouse_is_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing credentials stay in fixture mode and still close the client."""
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_HTTP_PATH", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
        output = tmp_path / "missing.md"
        with pytest.raises(RuntimeError, match="fixture mode"):
            main(["--output", str(output)])
        assert not output.exists()
