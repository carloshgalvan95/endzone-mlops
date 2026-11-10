"""The Pages score gate skips a week that is already on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from endzone.site.score_gate import main, score_gate_decision

pytestmark = pytest.mark.cross_platform


def _schedule(*rows: dict[str, object]) -> pd.DataFrame:
    """One schedule table. Missing scores are null."""
    return pd.DataFrame(list(rows))


def _write_week(root: Path, season: int, week: int) -> None:
    """Write a minimal real slate for the gate's presence check."""
    directory: Path = root / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "sample": False,
        "season": season,
        "week": week,
        "games": [{"game_id": "G1"}],
    }
    (directory / f"{season}-w{week}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_gate_skips_when_the_target_week_file_exists(tmp_path: Path) -> None:
    """The next unplayed week is not scored when its file is already present."""
    _write_week(tmp_path, 2026, 10)
    schedule = _schedule(
        {
            "game_id": "G1",
            "season": 2026,
            "week": 10,
            "gameday": "2026-11-12",
            "home_score": None,
            "away_score": None,
        }
    )
    decision = score_gate_decision(tmp_path, schedule, None, None, replace_published=False)
    assert decision["skip"] is True
    assert decision["reason"] == "published"
    assert decision["season"] == 2026
    assert decision["week"] == 10


def test_gate_scores_when_the_file_is_missing(tmp_path: Path) -> None:
    """A missing week file is the first publish, so the fit still runs."""
    schedule = _schedule(
        {
            "game_id": "G1",
            "season": 2026,
            "week": 10,
            "gameday": "2026-11-12",
            "home_score": None,
            "away_score": None,
        }
    )
    decision = score_gate_decision(tmp_path, schedule, None, None, replace_published=False)
    assert decision == {
        "skip": False,
        "reason": "missing",
        "season": 2026,
        "week": 10,
    }


def test_replace_published_is_never_skipped(tmp_path: Path) -> None:
    """The manual rebuild flag scores even when the week file exists."""
    _write_week(tmp_path, 2026, 10)
    decision = score_gate_decision(
        tmp_path,
        pd.DataFrame(),
        2026,
        10,
        replace_published=True,
    )
    assert decision["skip"] is False
    assert decision["reason"] == "replace_published"


def test_gate_cli_writes_skip_and_does_not_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """replace_published does not download a schedule. The output file is the contract."""

    def refuse_download(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("downloaded")

    monkeypatch.setattr(
        "endzone.site.score_gate.download_parquet_with_validation",
        refuse_download,
    )
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    _write_week(tmp_path / "site", 2026, 10)
    destination = tmp_path / "github_output.txt"
    status = main(
        [
            "--output-dir",
            str(tmp_path / "site"),
            "--season",
            "2026",
            "--week",
            "10",
            "--replace-published",
            "--github-output",
            str(destination),
        ]
    )
    text = destination.read_text(encoding="utf-8")
    assert status == 0
    assert "skip=false\n" in text
    assert "reason=replace_published\n" in text
    assert "season=2026\n" in text
    assert "week=10\n" in text
