"""Job summary rows for a Pages or Results run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from endzone.site.run_summary import (
    collect_summary,
    format_run_summary,
    load_run_facts,
    main,
)

pytestmark = pytest.mark.cross_platform

_HOST: str = "example.cloud.databricks.com"
_HTTP_PATH: str = "/sql/1.0/warehouses/example"
_TOKEN: str = "dapi-test-token-value"


def _slate(**overrides: object) -> dict[str, object]:
    """One published week. Callers replace the fields a case cares about."""
    document: dict[str, object] = {
        "sample": False,
        "season": 2026,
        "week": 10,
        "model_version": "2@prod",
        "model_alias": "prod",
        "data_digest": "abc123def456",
        "games": [{"game_id": "G1"}, {"game_id": "G2"}, {"game_id": "G3"}],
        "score_model": {
            "margin": {"model_version": "4"},
            "total": {"model_version": "5"},
        },
    }
    document.update(overrides)
    return document


def _write_json(path: Path, payload: object) -> None:
    """Write one UTF-8 JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_status(directory: Path, name: str, lines: list[str]) -> None:
    """Write one key=value status file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell(summary: str, label: str) -> str:
    """Return the table cell for ``label``."""
    prefix: str = f"| {label} | "
    for line in summary.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].removesuffix(" |").strip()
    raise AssertionError(f"{label} is missing from the summary")


def test_normal_week_shows_the_loaded_alias_and_point_versions() -> None:
    """A scored week names 2@prod, the digest, finals, the alert, and the archive."""
    rows = collect_summary(
        _slate(),
        {"available": True, "data_digest": "from-the-record"},
        {
            "score_gate": {"skip": "false", "reason": "missing", "season": "2026", "week": "10"},
            "score": {
                "model_version": "2@prod",
                "model_alias": "prod",
                "model_source": "registry",
                "data_digest": "abc123def456",
            },
            "results": {"updated": "2", "unchanged": "0", "outcome": "finals applied"},
            "model_alert": {"status": "within_gap", "action": "none"},
            "archive": {"committed": "true", "status": "committed"},
        },
    )
    summary = format_run_summary(rows)
    assert _cell(summary, "Season") == "2026"
    assert _cell(summary, "Week") == "10"
    assert _cell(summary, "Games") == "3"
    assert _cell(summary, "Model version") == "2@prod"
    assert _cell(summary, "Alias") == "2@prod"
    assert _cell(summary, "Data digest") == "abc123def456"
    assert _cell(summary, "Score gate") == "not skipped"
    assert _cell(summary, "Results") == "finals applied"
    assert _cell(summary, "Model alert") == "within_gap"
    assert _cell(summary, "Archive") == "committed"
    assert _cell(summary, "Margin model version") == "4"
    assert _cell(summary, "Total model version") == "5"


def test_skipped_gate_names_the_published_week() -> None:
    """A week already on disk is skipped. That is not a weekly fit."""
    summary = format_run_summary(
        collect_summary(
            _slate(),
            None,
            {
                "score_gate": {
                    "skip": "true",
                    "reason": "published",
                    "season": "2026",
                    "week": "10",
                },
                "results": {"updated": "0", "unchanged": "4", "outcome": "unchanged"},
            },
        )
    )
    assert _cell(summary, "Score gate") == "skipped, week already published"
    assert _cell(summary, "Results") == "unchanged"
    assert _cell(summary, "Model version") == "2@prod"
    assert "weekly fit" not in summary
    assert "alias prod is missing" not in summary
    assert "Margin model version" in summary


def test_fallback_week_names_the_reason_and_not_an_alias() -> None:
    """A missing alias is a weekly fit. The summary says why, not ``@prod``."""
    summary = format_run_summary(
        collect_summary(
            _slate(model_version="8", model_alias=None),
            None,
            {
                "score_gate": {"skip": "false", "reason": "missing"},
                "score": {
                    "model_version": "8",
                    "model_source": "refit",
                    "fallback_reason": "alias_missing",
                    "data_digest": "feedface",
                },
            },
        )
    )
    assert _cell(summary, "Model version") == "8"
    assert _cell(summary, "Alias") == "fallback: alias prod is missing"
    assert _cell(summary, "Data digest") == "feedface"
    assert "@prod" not in summary


def test_missing_files_stay_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No slate and no status files still render, and the point rows stay out."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    summary = format_run_summary(load_run_facts(tmp_path / "site", [tmp_path / "missing"]))
    assert _cell(summary, "Season") == "unavailable"
    assert _cell(summary, "Week") == "unavailable"
    assert _cell(summary, "Games") == "unavailable"
    assert _cell(summary, "Data digest") == "unavailable"
    assert _cell(summary, "Score gate") == "not run"
    assert _cell(summary, "Results") == "not run"
    assert _cell(summary, "Model alert") == "not run"
    assert _cell(summary, "Archive") == "not run"
    assert "Margin model version" not in summary
    assert "Total model version" not in summary
    assert main(["--site-data", str(tmp_path / "site")]) == 0


def test_track_record_fills_a_missing_slate(tmp_path: Path) -> None:
    """The last tracked week is used when latest.json is absent."""
    site = tmp_path / "site"
    _write_json(
        site / "track_record.json",
        {
            "available": True,
            "data_digest": "from-track",
            "seasons": [{"season": 2026, "weeks": [{"week": 9}, {"week": 10}]}],
        },
    )
    summary = format_run_summary(load_run_facts(site, []))
    assert _cell(summary, "Season") == "2026"
    assert _cell(summary, "Week") == "10"
    assert _cell(summary, "Data digest") == "from-track"


def test_summary_drops_a_host_a_warehouse_path_and_a_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warehouse connection values never reach the summary, from files or the environment."""
    monkeypatch.setenv("DATABRICKS_HOST", _HOST)
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", _HTTP_PATH)
    monkeypatch.setenv("DATABRICKS_TOKEN", _TOKEN)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"https://{_HOST}")
    site = tmp_path / "site"
    _write_json(
        site / "latest.json",
        _slate(
            model_version=f"2@prod {_HOST}",
            model_alias=_TOKEN,
            data_digest=f"{_HTTP_PATH} {_TOKEN}",
            games=[{"game_id": _HOST}],
        ),
    )
    status = tmp_path / "status"
    _write_status(
        status,
        "score",
        [
            "model_source=refit",
            f"fallback_reason=registry {_HOST} token {_TOKEN}",
            f"data_digest={_HTTP_PATH}",
        ],
    )
    destination = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    assert (
        main(
            [
                "--site-data",
                str(site),
                "--status-dir",
                str(status),
                "--summary-file",
                str(destination),
            ]
        )
        == 0
    )
    text = destination.read_text(encoding="utf-8")
    forbidden: tuple[str, ...] = (
        _HOST,
        _HTTP_PATH,
        _TOKEN,
        f"https://{_HOST}",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_HTTP_PATH",
    )
    for secret in forbidden:
        assert secret not in text
    assert _cell(text, "Season") == "2026"
    assert "[redacted]" in text


def test_cli_appends_the_summary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command writes UTF-8 markdown and returns 0."""
    site = tmp_path / "site"
    _write_json(site / "latest.json", _slate())
    destination = tmp_path / "summary.md"
    destination.write_text("existing\n", encoding="utf-8")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main(["--site-data", str(site), "--summary-file", str(destination)]) == 0
    text = destination.read_text(encoding="utf-8")
    assert text.startswith("existing\n")
    assert "| Season | 2026 |" in text


