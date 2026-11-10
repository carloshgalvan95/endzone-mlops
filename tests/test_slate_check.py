"""The Tuesday slate check: expected week, missing file, finals, and one issue."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from endzone.site.slate_check import (
    expected_publish_week,
    find_open_issue,
    main,
    previous_finals_are_due,
    previous_scheduled_week,
    publish_window,
    slate_alert_title,
    slate_check_findings,
    slate_finals_complete,
    sync_slate_issue,
)

_MEXICO = ZoneInfo("America/Mexico_City")


def _schedule(*rows: dict[str, object]) -> pd.DataFrame:
    """Schedule rows. Scores may be null."""
    return pd.DataFrame(list(rows))


def _game(
    game_id: str, *, status: str, home: int | None = None, away: int | None = None
) -> dict[str, Any]:
    """One slate game. Scores are omitted when the game is still scheduled."""
    record: dict[str, Any] = {"game_id": game_id, "status": status}
    if home is not None and away is not None:
        record["home_score"] = home
        record["away_score"] = away
    return record


def _slate(season: int, week: int, games: list[dict[str, Any]]) -> dict[str, Any]:
    """A minimal published document."""
    return {"sample": False, "season": season, "week": week, "games": games}


class _Issues:
    """In-memory issue client. Records creates and returns the open list."""

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items: list[dict[str, Any]] = list(items or [])
        self.created: list[str] = []
        self.labels: list[str] = []

    def ensure_label(self, name: str, color: str, description: str) -> None:
        self.labels.append(name)

    def list_all_open(self) -> list[dict[str, Any]]:
        return list(self.items)

    def create(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        item: dict[str, Any] = {"number": 40 + len(self.items), "title": title, "body": body}
        self.items.append(item)
        self.created.append(title)
        return item


def test_publish_window_starts_on_tuesday() -> None:
    """Tuesday is the start. Monday still belongs to the window that began last Tuesday."""
    tuesday = publish_window(date(2026, 11, 10))
    assert tuesday == {"start": date(2026, 11, 10), "end": date(2026, 11, 17)}
    monday = publish_window(date(2026, 11, 9))
    assert monday == {"start": date(2026, 11, 3), "end": date(2026, 11, 10)}


def test_expected_week_uses_select_slate_inside_the_window() -> None:
    """Tuesday 10 Nov 2026 expects 2026 week 10. July and a playoff Tuesday differ."""
    regular = _schedule(
        {
            "game_id": "W9",
            "season": 2026,
            "week": 9,
            "gameday": "2026-11-09",
            "home_score": 20,
            "away_score": 17,
        },
        {
            "game_id": "W10",
            "season": 2026,
            "week": 10,
            "gameday": "2026-11-12",
            "home_score": None,
            "away_score": None,
        },
    )
    assert expected_publish_week(regular, date(2026, 11, 10)) == {"season": 2026, "week": 10}
    offseason = _schedule(
        {
            "game_id": "OPENER",
            "season": 2026,
            "week": 1,
            "gameday": "2026-09-10",
            "home_score": None,
            "away_score": None,
        }
    )
    assert expected_publish_week(offseason, date(2026, 7, 14)) is None
    playoff = _schedule(
        {
            "game_id": "SB",
            "season": 2026,
            "week": 22,
            "gameday": "2027-02-14",
            "home_score": None,
            "away_score": None,
        }
    )
    # 2027-02-09 is a Tuesday. The game that Sunday is inside the window.
    assert expected_publish_week(playoff, date(2027, 2, 9)) == {"season": 2026, "week": 22}


def test_previous_week_and_when_finals_are_due() -> None:
    """Week 10 follows week 9. Finals are due from 08:00 CT Tuesday, not before."""
    schedule = _schedule(
        {"game_id": "A", "season": 2026, "week": 9, "gameday": "2026-11-08"},
        {"game_id": "B", "season": 2026, "week": 10, "gameday": "2026-11-12"},
    )
    assert previous_scheduled_week(schedule, 2026, 10) == {"season": 2026, "week": 9}
    assert previous_scheduled_week(schedule, 2026, 9) is None
    before = datetime(2026, 11, 10, 13, 59, tzinfo=UTC)
    at_eight = datetime(2026, 11, 10, 14, 0, tzinfo=UTC)
    alert_hour = datetime(2026, 11, 10, 14, 7, tzinfo=UTC)
    assert previous_finals_are_due(before) is False
    assert previous_finals_are_due(at_eight) is True
    assert previous_finals_are_due(alert_hour) is True
    monday = datetime(2026, 11, 9, 23, 0, tzinfo=_MEXICO)
    assert previous_finals_are_due(monday) is True
    with pytest.raises(ValueError, match="timezone"):
        previous_finals_are_due(datetime(2026, 11, 10, 8, 0))


def test_findings_cover_a_missing_file_incomplete_finals_and_the_offseason() -> None:
    """One missing piece alerts. A complete previous week, or no games, stays quiet."""
    current = _slate(2026, 10, [_game("G1", status="scheduled")])
    complete = _slate(2026, 9, [_game("G0", status="final", home=24, away=17)])
    partial = _slate(
        2026,
        9,
        [
            _game("G0", status="final", home=24, away=17),
            _game("G2", status="scheduled"),
        ],
    )
    expected = {"season": 2026, "week": 10}
    previous = {"season": 2026, "week": 9}
    missing = slate_check_findings(expected, None, previous, complete, previous_finals_due=True)
    assert missing["alert"] is True
    assert missing["problems"][0]["kind"] == "missing_slate"
    assert slate_alert_title(2026, 10) == "Slate check failed: 2026 week 10"
    finals = slate_check_findings(expected, current, previous, partial, previous_finals_due=True)
    assert finals["alert"] is True
    assert finals["problems"] == [
        {
            "kind": "missing_finals",
            "season": 2026,
            "week": 9,
            "games": 2,
            "unfinished": 1,
        }
    ]
    early = slate_check_findings(expected, current, previous, partial, previous_finals_due=False)
    assert early["alert"] is False
    quiet = slate_check_findings(expected, current, previous, complete, previous_finals_due=True)
    assert quiet["alert"] is False
    assert quiet["reason"] == "ok"
    offseason = slate_check_findings(None, None, None, None, previous_finals_due=True)
    assert offseason["alert"] is False
    assert offseason["reason"] == "outside_season"
    assert slate_finals_complete(None)["complete"] is False
    assert slate_finals_complete(complete)["reason"] == "complete"


def test_sync_opens_one_issue_and_a_second_call_does_not() -> None:
    """An open issue with the same title blocks a duplicate. A pull request does not."""
    findings = slate_check_findings(
        {"season": 2026, "week": 10},
        None,
        {"season": 2026, "week": 9},
        _slate(2026, 9, [_game("G0", status="final", home=1, away=0)]),
        previous_finals_due=True,
    )
    client = _Issues(
        [{"number": 3, "title": "Slate check failed: 2026 week 10", "pull_request": {"url": "x"}}]
    )
    created = sync_slate_issue(findings, client)
    assert created["action"] == "created"
    assert client.created == ["Slate check failed: 2026 week 10"]
    again = sync_slate_issue(findings, client)
    assert again["action"] == "unchanged"
    assert again["number"] == created["number"]
    assert client.created == ["Slate check failed: 2026 week 10"]
    assert find_open_issue(client.items, "Slate check failed: 2026 week 9")["found"] is False
    quiet = slate_check_findings(
        {"season": 2026, "week": 10},
        _slate(2026, 10, [_game("G1", status="scheduled")]),
        {"season": 2026, "week": 9},
        _slate(2026, 9, [_game("G0", status="final", home=1, away=0)]),
        previous_finals_due=True,
    )
    skipped = sync_slate_issue(quiet, client)
    assert skipped["action"] == "skipped"
    assert len(client.created) == 1


@pytest.mark.cross_platform
def test_main_is_quiet_outside_the_season_and_fails_without_a_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """July does nothing. A missing November slate without a token fails closed."""
    games = tmp_path / "games.csv"
    games.write_text(
        "game_id,season,week,gameday,home_score,away_score\n"
        "OPENER,2026,1,2026-09-10,,\n"
        "W9,2026,9,2026-11-08,21,17\n"
        "W10,2026,10,2026-11-12,,\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    quiet = main(
        [
            "--archive-dir",
            str(archive),
            "--games-path",
            str(games),
            "--now",
            "2026-07-14T18:00:00+00:00",
        ]
    )
    assert quiet == 0
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    missing = main(
        [
            "--archive-dir",
            str(archive),
            "--games-path",
            str(games),
            "--now",
            "2026-11-10T14:07:00+00:00",
            "--repo",
            "owner/endzone-mlops",
        ]
    )
    assert missing == 1
    dry = main(
        [
            "--archive-dir",
            str(archive),
            "--games-path",
            str(games),
            "--now",
            "2026-11-10T14:07:00+00:00",
            "--dry-run",
        ]
    )
    assert dry == 0


@pytest.mark.cross_platform
def test_corrupt_slate_fails_closed(tmp_path: Path) -> None:
    """A week file that is not JSON is not treated as a quiet week."""
    games = tmp_path / "games.csv"
    games.write_text(
        "game_id,season,week,gameday,home_score,away_score\nW10,2026,10,2026-11-12,,\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive" / "predictions"
    archive.mkdir(parents=True)
    (archive / "2026-w10.json").write_text("{", encoding="utf-8")
    status = main(
        [
            "--archive-dir",
            str(tmp_path / "archive"),
            "--games-path",
            str(games),
            "--now",
            "2026-11-10T16:13:00+00:00",
            "--dry-run",
        ]
    )
    assert status == 1


def test_slate_check_workflow_does_not_score_or_deploy() -> None:
    """The alert job can read the repo and open an issue. It cannot publish."""
    text = Path(".github/workflows/slate-check.yml").read_text(encoding="utf-8")
    folded = text.lower()
    assert "workflow_dispatch:" in text
    assert 'cron: "7 14 * * 2"' in text
    assert 'cron: "13 16 * * 2"' in text
    assert "endzone.site.slate_check" in text
    assert "issues: write" in text
    assert "contents: read" in text
    assert "contents: write" not in text
    assert "pages: write" not in text
    assert "id-token: write" not in text
    assert "databricks" not in folded
    assert "endzone.modeling.score" not in text
    assert "endzone.site.archive publish" not in text
    assert "deploy-pages" not in text
    assert "group: pages" not in text
