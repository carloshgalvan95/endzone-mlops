"""Archive and restore of published weekly slates. No network."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from endzone.modeling.weekly import _WEEK_FILE as SCORE_WEEK_FILE
from endzone.modeling.weekly import validate_prediction_document
from endzone.site.archive import (
    _WEEK_FILE as ARCHIVE_WEEK_FILE,
)
from endzone.site.archive import (
    ARCHIVE_BRANCH,
    ArchiveError,
    archive_commit_message,
    archived_game_ids,
    backfill_archive,
    main,
    publish_archive,
    restore_from_repo,
    restore_slates,
    stage_archive,
)

_STAMP: str = "20261109T042837Z"
_STAMP_LATER: str = "20261109T120000Z"
_AUTHOR: str = "Nia Pérez"
_EMAIL: str = "nia@example.com"


def _target(decision: str, baseline_name: str) -> dict[str, Any]:
    """One score-model target with no MAE."""
    return {
        "decision": decision,
        "baseline_name": baseline_name,
        "mae": None,
        "baseline_mae": None,
        "market_mae": None,
    }


def _game(game_id: str) -> dict[str, Any]:
    """One scheduled game. The reason carries a non-ASCII name on purpose."""
    return {
        "game_id": game_id,
        "kickoff": "2026-11-08",
        "home": "BUF",
        "away": "MIA",
        "model_home_wp": 0.5,
        "market_home_wp": None,
        "diff": None,
        "model_version": "local",
        "generated_at": "2026-11-09T04:28:37Z",
        "status": "scheduled",
        "estimated_margin": None,
        "estimated_total": None,
        "estimated_home_points": None,
        "estimated_away_points": None,
    }


def _document(season: int, week: int, game_ids: list[str]) -> dict[str, Any]:
    """A valid week document. ``José`` is stored as UTF-8, not an escape."""
    document: dict[str, Any] = {
        "sample": False,
        "season": season,
        "week": week,
        "weeks_available": [week],
        "generated_at": "2026-11-09T04:28:37Z",
        "model_name": "endzone-winprob",
        "model_version": "local",
        "feature_set": "h2_full",
        "threshold_pp": 5,
        "score_model": {
            "feature_set": "h2_full",
            "accepted": False,
            "hidden_reason": "José",
            "margin": _target("not_run", "Elo/25"),
            "total": _target("not_run", "League mean"),
        },
        "games": [_game(game_id) for game_id in game_ids],
    }
    validate_prediction_document(document)
    return document


def _dump(document: dict[str, Any]) -> str:
    """Serialize the way the tests persist JSON: UTF-8, no ASCII escapes."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _write_json(path: Path, document: dict[str, Any]) -> None:
    """Write one JSON object with an explicit UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(document), encoding="utf-8")


def _site(root: Path, weeks: dict[int, list[str]], *, season: int = 2026) -> None:
    """Write week files and point latest.json at the highest week."""
    latest_week: int = max(weeks)
    for week, game_ids in weeks.items():
        _write_json(
            root / "predictions" / f"{season}-w{week}.json",
            _document(season, week, game_ids),
        )
    _write_json(root / "latest.json", _document(season, latest_week, weeks[latest_week]))


def _ids(path: Path) -> set[str]:
    """Game ids in a UTF-8 JSON file."""
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {str(game["game_id"]) for game in document["games"]}


def test_week_file_pattern_matches_the_score_writer() -> None:
    """Archive names and score names stay the same pattern."""
    assert ARCHIVE_WEEK_FILE.pattern == SCORE_WEEK_FILE.pattern


def test_commit_message_names_the_weeks() -> None:
    """The subject matches the workflow's Conventional Commit shape."""
    one: str = archive_commit_message([{"season": 2026, "week": 9}], backfill=False)
    assert one == "chore(data): archive 2026 week 9"
    several: str = archive_commit_message(
        [{"season": 2026, "week": 10}, {"season": 2026, "week": 9}, {"season": 2026, "week": 9}],
        backfill=False,
    )
    assert several == "chore(data): archive 2026 weeks 9, 10"
    seeded: str = archive_commit_message([{"season": 2026, "week": 9}], backfill=True)
    assert seeded == "chore(data): backfill 2026 week 9"
    assert archive_commit_message([], backfill=False) == "chore(data): archive published slates"


def test_stage_keeps_the_earlier_file_when_the_week_is_replaced(tmp_path: Path) -> None:
    """A new tip does not delete the snapshot that was archived first."""
    archive = tmp_path / "archive"
    site = tmp_path / "site"
    _site(site, {9: ["G1", "G2"]})
    first: dict[str, Any] = stage_archive(archive, site, archived_at=_STAMP)
    assert first["changed_weeks"] == [{"season": 2026, "week": 9}]
    assert first["message"] == "chore(data): archive 2026 week 9"
    history: list[Path] = sorted((archive / "history" / "2026-w9").glob("*.json"))
    assert len(history) == 1
    assert "José" in history[0].read_text(encoding="utf-8")

    _site(site, {9: ["G1", "G2", "G3"]})
    second: dict[str, Any] = stage_archive(archive, site, archived_at=_STAMP_LATER)
    assert second["changed_weeks"] == [{"season": 2026, "week": 9}]
    stored: list[Path] = sorted((archive / "history" / "2026-w9").glob("*.json"))
    assert len(stored) == 2
    assert _ids(stored[0]) == {"G1", "G2"}
    assert _ids(archive / "predictions" / "2026-w9.json") == {"G1", "G2", "G3"}
    assert "José" in (archive / "predictions" / "2026-w9.json").read_text(encoding="utf-8")


def test_stage_refuses_a_game_id_that_exists_only_in_history(tmp_path: Path) -> None:
    """No-shrink checks every archived snapshot, not only the tip."""
    archive = tmp_path / "archive"
    site = tmp_path / "site"
    _site(site, {9: ["G1", "G2"]})
    stage_archive(archive, site, archived_at=_STAMP)
    tip: Path = archive / "predictions" / "2026-w9.json"
    before: bytes = tip.read_bytes()
    _write_json(
        archive / "history" / "2026-w9" / "20261108T000000Z.json",
        _document(2026, 9, ["G1", "G2", "G_OLD"]),
    )
    assert "G_OLD" in archived_game_ids(archive, 2026, 9)

    _site(site, {9: ["G1", "G2", "G3"]})
    with pytest.raises(ArchiveError, match="G_OLD"):
        stage_archive(archive, site, archived_at=_STAMP_LATER)
    assert tip.read_bytes() == before
    assert (archive / "history" / "2026-w9" / "20261108T000000Z.json").is_file()


def test_stage_allows_added_games(tmp_path: Path) -> None:
    """Adding a game id is not a shrink."""
    archive = tmp_path / "archive"
    site = tmp_path / "site"
    _site(site, {9: ["G1"]})
    stage_archive(archive, site, archived_at=_STAMP)
    _site(site, {9: ["G1", "G2"]})
    staged: dict[str, Any] = stage_archive(archive, site, archived_at=_STAMP_LATER)
    assert staged["changed_weeks"] == [{"season": 2026, "week": 9}]
    assert _ids(archive / "predictions" / "2026-w9.json") == {"G1", "G2"}


def test_stage_refuses_a_missing_archived_week(tmp_path: Path) -> None:
    """A deploy that omits a stored week fails before it rewrites the tip."""
    archive = tmp_path / "archive"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _site(first, {8: ["A1"], 9: ["G1"]})
    stage_archive(archive, first, archived_at=_STAMP)
    week_eight: bytes = (archive / "predictions" / "2026-w8.json").read_bytes()
    _site(second, {9: ["G1", "G2"]})
    with pytest.raises(ArchiveError, match="2026 week 8"):
        stage_archive(archive, second, archived_at=_STAMP_LATER)
    assert (archive / "predictions" / "2026-w8.json").read_bytes() == week_eight


def test_stage_refuses_a_colon_in_the_snapshot_name(tmp_path: Path) -> None:
    """Snapshot names stay free of characters Windows rejects."""
    site = tmp_path / "site"
    _site(site, {9: ["G1"]})
    with pytest.raises(ArchiveError, match="no colon"):
        stage_archive(tmp_path / "archive", site, archived_at="2026-11-09T04:28:37Z")


def test_identical_bytes_do_not_add_a_snapshot(tmp_path: Path) -> None:
    """A second stage of the same file does not grow history."""
    archive = tmp_path / "archive"
    site = tmp_path / "site"
    _site(site, {9: ["G1", "G2"]})
    stage_archive(archive, site, archived_at=_STAMP)
    again: dict[str, Any] = stage_archive(archive, site, archived_at=_STAMP_LATER)
    assert again["changed_weeks"] == []
    assert list((archive / "history" / "2026-w9").glob("*.json")) == [
        archive / "history" / "2026-w9" / f"{_STAMP}.json"
    ]


def test_restore_prefers_the_archive_and_fills_gaps_from_the_site(tmp_path: Path) -> None:
    """The archive wins for a week it has. A week it lacks comes from the site."""
    archive = tmp_path / "archive"
    _site(archive, {9: ["G1", "G2", "G3"]})
    short: dict[str, Any] = _document(2026, 9, ["G1"])
    week_eight: dict[str, Any] = _document(2026, 8, ["A1"])
    latest: dict[str, Any] = _document(2026, 9, ["G1"])
    latest["weeks_available"] = [8, 9]
    pages: dict[str, str] = {
        "https://example.test/data/latest.json": _dump(latest),
        "https://example.test/data/predictions/2026-w9.json": _dump(short),
        "https://example.test/data/predictions/2026-w8.json": _dump(week_eight),
    }
    requested: list[str] = []

    def fetch_text(url: str) -> str:
        requested.append(url)
        if url not in pages:
            raise ValueError(f"HTTP 404 for {url}")
        return pages[url]

    output = tmp_path / "out"
    _write_json(output / "predictions" / "2026-w9.json", short)
    restored: dict[str, Any] = restore_slates(
        output,
        archive_dir=archive,
        pages_base="https://example.test",
        fetch_text=fetch_text,
    )
    assert restored["copied_from_archive"] == 1
    assert restored["copied_from_site"] == 1
    assert restored["archive_latest"] is True
    assert _ids(output / "predictions" / "2026-w9.json") == {"G1", "G2", "G3"}
    assert _ids(output / "predictions" / "2026-w8.json") == {"A1"}
    assert _ids(output / "latest.json") == {"G1", "G2", "G3"}
    assert "https://example.test/data/predictions/2026-w9.json" not in requested
    assert "https://example.test/data/predictions/2026-w8.json" in requested


def test_restore_uses_the_live_site_when_no_archive_is_given(tmp_path: Path) -> None:
    """No archive directory means the live site is the whole restore."""
    latest: dict[str, Any] = _document(2026, 9, ["G1"])
    pages: dict[str, str] = {
        "https://example.test/data/latest.json": _dump(latest),
        "https://example.test/data/predictions/2026-w9.json": _dump(latest),
    }
    restored: dict[str, Any] = restore_slates(
        tmp_path / "out",
        archive_dir=None,
        pages_base="https://example.test/",
        fetch_text=lambda url: pages[url],
    )
    assert restored["copied_from_archive"] == 0
    assert restored["copied_from_site"] == 1
    assert _ids(tmp_path / "out" / "predictions" / "2026-w9.json") == {"G1"}


def test_restore_fails_closed_on_an_unreadable_archive_week(tmp_path: Path) -> None:
    """A corrupt archive file is not replaced by the shorter live slate."""
    archive = tmp_path / "archive" / "predictions"
    archive.mkdir(parents=True)
    (archive / "2026-w9.json").write_text("{", encoding="utf-8")
    called: list[str] = []

    def fetch_text(url: str) -> str:
        called.append(url)
        return _dump(_document(2026, 9, ["G1"]))

    output = tmp_path / "out"
    with pytest.raises(ArchiveError, match="2026-w9.json"):
        restore_slates(
            output,
            archive_dir=tmp_path / "archive",
            pages_base="https://example.test",
            fetch_text=fetch_text,
        )
    assert called == []
    assert not (output / "predictions" / "2026-w9.json").exists()


def test_backfill_seeds_the_archive_from_a_fixture_site(tmp_path: Path) -> None:
    """Backfill copies the live weeks and records a backfill commit."""
    work, _bare = _repo(tmp_path)
    latest: dict[str, Any] = _document(2026, 9, ["G1", "G2"])
    latest["weeks_available"] = [9]
    pages: dict[str, str] = {
        "https://example.test/data/latest.json": _dump(latest),
        "https://example.test/data/predictions/2026-w9.json": _dump(latest),
    }
    result: dict[str, Any] = backfill_archive(
        work,
        "https://example.test",
        archived_at=_STAMP,
        author_name=_AUTHOR,
        author_email=_EMAIL,
        fetch_text=lambda url: pages[url],
    )
    assert result["committed"] is True
    assert result["message"] == "chore(data): backfill 2026 week 9"
    assert _log_subject(work) == "chore(data): backfill 2026 week 9"
    assert "Co-authored-by" not in _log_body(work)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git with an explicit UTF-8 encoding and no signing."""
    env: dict[str, str] = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Ada López"
    env["GIT_AUTHOR_EMAIL"] = "ada@example.com"
    env["GIT_COMMITTER_NAME"] = "Ada López"
    env["GIT_COMMITTER_EMAIL"] = "ada@example.com"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.quotepath=false",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=env,
        timeout=60,
    )


def _git_ok(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git and fail the test when it exits non-zero."""
    completed: subprocess.CompletedProcess[str] = _git(repo, *args)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """A local clone and a bare remote. Nothing leaves the machine."""
    work: Path = tmp_path / "work"
    work.mkdir()
    _git_ok(work, "init", "-b", "main")
    _git_ok(work, "commit", "--allow-empty", "-m", "init")
    bare: Path = tmp_path / "remote.git"
    _git_ok(tmp_path, "init", "--bare", "-b", "main", str(bare))
    _git_ok(work, "remote", "add", "origin", str(bare))
    _git_ok(work, "push", "-u", "origin", "main")
    return work, bare


def _publish(work: Path, site: Path, stamp: str) -> dict[str, Any]:
    """Push one site directory to the local archive branch."""
    return publish_archive(
        work,
        site,
        archived_at=stamp,
        author_name=_AUTHOR,
        author_email=_EMAIL,
    )


def _log_body(repo: Path, revision: str = ARCHIVE_BRANCH) -> str:
    """Full commit message, decoded as UTF-8."""
    completed: subprocess.CompletedProcess[str] = _git_ok(
        repo,
        "log",
        "-1",
        "--format=%B",
        revision,
    )
    return completed.stdout


def _log_subject(repo: Path, revision: str = ARCHIVE_BRANCH) -> str:
    """Commit subject line."""
    completed: subprocess.CompletedProcess[str] = _git_ok(
        repo,
        "log",
        "-1",
        "--format=%s",
        revision,
    )
    return completed.stdout.strip()


def _identity(repo: Path, revision: str = ARCHIVE_BRANCH) -> list[str]:
    """Author and committer name and email."""
    completed: subprocess.CompletedProcess[str] = _git_ok(
        repo,
        "log",
        "-1",
        "--format=%an%n%ae%n%cn%n%ce",
        revision,
    )
    return completed.stdout.splitlines()


def test_publish_keeps_history_off_main(tmp_path: Path) -> None:
    """The archive branch is an orphan, and an update keeps the first blob."""
    work, _bare = _repo(tmp_path)
    site = tmp_path / "site"
    _site(site, {9: ["G1", "G2"]})
    first: dict[str, Any] = _publish(work, site, _STAMP)
    assert first["committed"] is True
    assert first["message"] == "chore(data): archive 2026 week 9"
    assert _identity(work) == [_AUTHOR, _EMAIL, _AUTHOR, _EMAIL]
    assert "Co-authored-by" not in _log_body(work)
    parents: list[str] = _git_ok(
        work,
        "rev-list",
        "--parents",
        "-n",
        "1",
        ARCHIVE_BRANCH,
    ).stdout.split()
    assert len(parents) == 1
    names: set[str] = set(
        _git_ok(work, "ls-tree", "-r", "--name-only", ARCHIVE_BRANCH).stdout.splitlines()
    )
    assert "predictions/2026-w9.json" in names
    assert "history/2026-w9/20261109T042837Z.json" in names
    assert "endzone/site/archive.py" not in names
    assert "pyproject.toml" not in names
    merge_base: subprocess.CompletedProcess[str] = _git(work, "merge-base", "main", ARCHIVE_BRANCH)
    assert merge_base.returncode != 0

    _site(site, {9: ["G1", "G2", "G3"]})
    second: dict[str, Any] = _publish(work, site, _STAMP_LATER)
    assert second["committed"] is True
    previous: str = _git_ok(
        work,
        "show",
        f"{first['sha']}:predictions/2026-w9.json",
    ).stdout
    assert "G1" in previous
    assert "G3" not in previous
    assert "José" in previous
    unchanged: dict[str, Any] = _publish(work, site, "20261109T130000Z")
    assert unchanged["committed"] is False
    assert _git_ok(work, "rev-parse", ARCHIVE_BRANCH).stdout.strip() == second["sha"]
    status: str = _git_ok(work, "status", "--porcelain").stdout
    assert status == ""
    assert _git_ok(work, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


def test_publish_does_not_push_a_shorter_slate(tmp_path: Path) -> None:
    """A refused shrink leaves the remote branch on the previous commit."""
    work, _bare = _repo(tmp_path)
    site = tmp_path / "site"
    _site(site, {9: ["G1", "G2"]})
    first: dict[str, Any] = _publish(work, site, _STAMP)
    _site(site, {9: ["G1"]})
    with pytest.raises(ArchiveError, match="drops 1"):
        _publish(work, site, _STAMP_LATER)
    assert _git_ok(work, "rev-parse", ARCHIVE_BRANCH).stdout.strip() == first["sha"]
    remote: str = _git_ok(work, "ls-remote", "--heads", "origin", ARCHIVE_BRANCH).stdout
    assert first["sha"] in remote


def test_publish_refuses_main(tmp_path: Path) -> None:
    """The archive push cannot target the branch that stays free of week JSON."""
    work, _bare = _repo(tmp_path)
    site = tmp_path / "site"
    _site(site, {9: ["G1"]})
    with pytest.raises(ArchiveError, match="main"):
        publish_archive(
            work,
            site,
            branch="main",
            archived_at=_STAMP,
            author_name=_AUTHOR,
            author_email=_EMAIL,
        )


def test_restore_from_the_branch_beats_the_live_site(tmp_path: Path) -> None:
    """Once the branch exists, a shorter live file does not replace it."""
    work, _bare = _repo(tmp_path)
    site = tmp_path / "site"
    _site(site, {9: ["G1", "G2", "G3"]})
    _publish(work, site, _STAMP)
    short: dict[str, Any] = _document(2026, 9, ["G1"])
    pages: dict[str, str] = {
        "https://example.test/data/latest.json": _dump(short),
        "https://example.test/data/predictions/2026-w9.json": _dump(short),
    }
    output = tmp_path / "out"
    restored: dict[str, Any] = restore_from_repo(
        work,
        output,
        pages_base="https://example.test",
        fetch_text=lambda url: pages[url],
    )
    assert restored["copied_from_archive"] == 1
    assert restored["copied_from_site"] == 0
    assert _ids(output / "predictions" / "2026-w9.json") == {"G1", "G2", "G3"}


def test_restore_falls_back_when_the_branch_is_absent(tmp_path: Path) -> None:
    """Before the first backfill, the live site is still the restore source."""
    work, _bare = _repo(tmp_path)
    latest: dict[str, Any] = _document(2026, 9, ["G1", "G2"])
    pages: dict[str, str] = {
        "https://example.test/data/latest.json": _dump(latest),
        "https://example.test/data/predictions/2026-w9.json": _dump(latest),
    }
    output = tmp_path / "out"
    restored: dict[str, Any] = restore_from_repo(
        work,
        output,
        pages_base="https://example.test",
        fetch_text=lambda url: pages[url],
    )
    assert restored["copied_from_archive"] == 0
    assert restored["copied_from_site"] == 1
    assert _ids(output / "predictions" / "2026-w9.json") == {"G1", "G2"}


def test_restore_does_not_fall_back_when_the_remote_cannot_be_read(tmp_path: Path) -> None:
    """A git failure is not treated as an empty archive."""
    work, _bare = _repo(tmp_path)
    _git_ok(work, "remote", "set-url", "origin", str(tmp_path / "missing"))
    called: list[str] = []

    def fetch_text(url: str) -> str:
        called.append(url)
        return "{}"

    with pytest.raises(ArchiveError, match="Reading origin"):
        restore_from_repo(
            work,
            tmp_path / "out",
            pages_base="https://example.test",
            fetch_text=fetch_text,
        )
    assert called == []


def test_publish_cli_uses_the_author_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command records the author from the environment, not from git config."""
    work, _bare = _repo(tmp_path)
    site = tmp_path / "site"
    _site(site, {9: ["G1"]})
    monkeypatch.setenv("ARCHIVE_GIT_NAME", _AUTHOR)
    monkeypatch.setenv("ARCHIVE_GIT_EMAIL", _EMAIL)
    result: dict[str, Any] = main(
        [
            "publish",
            "--repo",
            str(work),
            "--site-data",
            str(site),
            "--archived-at",
            _STAMP,
        ]
    )
    assert result["message"] == "chore(data): archive 2026 week 9"
    assert _identity(work) == [_AUTHOR, _EMAIL, _AUTHOR, _EMAIL]
