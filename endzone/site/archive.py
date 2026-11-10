"""Durable copy of published weekly slates.

The live Pages site is the page readers open. It is not the only copy.
``published-slates`` is an orphan branch, not a parent of ``main``, and it
is never merged. ``predictions/<season>-w<week>.json`` and ``latest.json``
at the tip are the current files. ``history/`` keeps every earlier file, so
replacing a week does not throw the previous bytes away. ``main`` stays
free of generated prediction JSON.

A restore copies the branch first and fills weeks the branch does not have
from the live site. A write that drops a ``game_id`` already stored for
that season and week fails. Adding games is allowed. The Pages deploy runs
only after this push succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from endzone.modeling.weekly import (
    DEFAULT_OUTPUT_DIR,
    restore_published_predictions,
    validate_prediction_document,
)
from endzone.utils.logging import setup_logger
from endzone.utils.step_status import (
    append_github_output,
    status_directory,
    write_step_status,
)

logger = setup_logger(__name__)

ARCHIVE_BRANCH: str = "published-slates"
BOT_NAME: str = "github-actions[bot]"
BOT_EMAIL: str = "41898282+github-actions[bot]@users.noreply.github.com"
_STAMP: re.Pattern[str] = re.compile(r"^\d{8}T\d{6}Z$")
_WEEK_FILE: re.Pattern[str] = re.compile(r"^(?P<season>\d+)-w(?P<week>\d+)\.json$")
_WEEK_DIR: re.Pattern[str] = re.compile(r"^(?P<season>\d+)-w(?P<week>\d+)$")
_HISTORY_FILE: re.Pattern[str] = re.compile(
    r"^history/(?P<season>\d+)-w(?P<week>\d+)/(?P<stamp>\d{8}T\d{6}Z)(?:-\d+)?\.json$"
)
_LATEST_HISTORY: re.Pattern[str] = re.compile(
    r"^history/latest/(?P<stamp>\d{8}T\d{6}Z)(?:-\d+)?\.json$"
)
_BRANCH_NAME: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_PROTECTED_BRANCHES: frozenset[str] = frozenset({"main", "master"})
_TOKEN_IN_URL: re.Pattern[str] = re.compile(r"x-access-token:[^@\s]+@")
_ALLOWED_ROOT_FILES: frozenset[str] = frozenset({"README.md", ".gitattributes", "latest.json"})
_README: str = """# Published slates

This branch is the durable copy of the weekly prediction JSON. The Pages
and Results workflows write it. Do not edit it by hand, and do not merge
it into main.

`predictions/<season>-w<week>.json` is the current file for that week.
`latest.json` is the file the page loads. `history/` keeps every earlier
copy. The file name is the UTC time of that archive write, with no colon,
so it is safe on Windows. A later week file does not delete the earlier one.

main does not store these files. See ADR-006 on main.
"""
_ATTRIBUTES: str = "* -text\n"


class ArchiveError(Exception):
    """The published-slate archive could not be read or written."""


def archived_at_text(clock: datetime | None = None) -> str:
    """Return a UTC stamp safe for a Windows file name.

    The shape is ``YYYYMMDDTHHMMSSZ``. It has no colon.
    """
    moment: datetime = clock if clock is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def archive_commit_message(weeks: list[dict[str, int]], *, backfill: bool) -> str:
    """Conventional Commit subject for one archive push.

    One week is ``chore(data): archive 2026 week 9``. A backfill uses
    ``backfill`` in place of ``archive``. An empty list is the generic
    subject used when only the branch notes changed.
    """
    verb: str = "backfill" if backfill else "archive"
    if not weeks:
        return f"chore(data): {verb} published slates"
    by_season: dict[int, list[int]] = {}
    seen: set[tuple[int, int]] = set()
    ordered: list[dict[str, int]] = sorted(weeks, key=lambda item: (item["season"], item["week"]))
    for item in ordered:
        key: tuple[int, int] = (int(item["season"]), int(item["week"]))
        if key in seen:
            continue
        seen.add(key)
        by_season.setdefault(key[0], []).append(key[1])
    parts: list[str] = []
    for season, week_numbers in by_season.items():
        if len(week_numbers) == 1:
            parts.append(f"{season} week {week_numbers[0]}")
            continue
        listed: str = ", ".join(str(week) for week in week_numbers)
        parts.append(f"{season} weeks {listed}")
    return f"chore(data): {verb} {', '.join(parts)}"


def archived_game_ids(archive_root: Path, season: int, week: int) -> set[str]:
    """Game ids already stored for this season and week.

    The tip file, every history snapshot, ``latest.json`` when it names
    this week, and history copies of ``latest.json`` all count. A file
    that cannot be read fails closed.

    Raises:
        ArchiveError: A stored file is unreadable or is not that week.
    """
    root: Path = Path(archive_root)
    target_season: int = int(season)
    target_week: int = int(week)
    ids: set[str] = set()
    tip: Path = root / "predictions" / f"{target_season}-w{target_week}.json"
    if tip.is_file():
        ids.update(_ids_in_archive_file(tip, target_season, target_week, require_name=True))
    history_dir: Path = root / "history" / f"{target_season}-w{target_week}"
    if history_dir.is_dir():
        for path in sorted(history_dir.glob("*.json")):
            if path.is_file():
                ids.update(
                    _ids_in_archive_file(path, target_season, target_week, require_name=False)
                )
    latest: Path = root / "latest.json"
    if latest.is_file():
        ids.update(_ids_in_latest(latest, target_season, target_week))
    latest_history: Path = root / "history" / "latest"
    if latest_history.is_dir():
        for path in sorted(latest_history.glob("*.json")):
            if path.is_file():
                ids.update(_ids_in_latest(path, target_season, target_week))
    return ids


def stage_archive(
    archive_root: Path,
    site_data: Path,
    *,
    archived_at: str,
    backfill: bool = False,
) -> dict[str, Any]:
    """Copy week files into the archive directory.

    History files are written before the tip. An identical payload is not
    stored twice. Nothing is written when a check fails.

    Returns:
        ``changed_weeks``, ``latest_changed``, and ``message``.

    Raises:
        ArchiveError: The stamp is unsafe, a slate is invalid, a week
            already archived is missing, or a ``game_id`` would be dropped.
    """
    stamp: str = _require_stamp(archived_at)
    incoming: dict[str, Any] = _load_incoming(Path(site_data))
    root: Path = Path(archive_root)
    _refuse_missing_weeks(root, incoming["weeks"])
    _refuse_dropped_ids(root, incoming["weeks"])
    _refuse_dropped_ids(root, [incoming["latest"]])
    changed: list[dict[str, int]] = []
    for item in incoming["weeks"]:
        if _store_slate(root, item, stamp, latest=False):
            changed.append({"season": int(item["season"]), "week": int(item["week"])})
    latest_changed: bool = _store_slate(root, incoming["latest"], stamp, latest=True)
    _ensure_notes(root)
    return {
        "changed_weeks": changed,
        "latest_changed": latest_changed,
        "message": archive_commit_message(changed, backfill=backfill),
    }


def restore_slates(
    output_dir: Path,
    *,
    archive_dir: Path | None = None,
    pages_base: str | None = None,
    fetch_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Copy week files, preferring the archive over the live site.

    A week file already taken from the archive is not replaced by the live
    site. Weeks the archive does not have are copied from the site. An
    unreadable archived week fails before any live file is written.

    Returns:
        ``copied_from_archive``, ``copied_from_site``, ``copied``, and
        ``archive_latest``.

    Raises:
        ArchiveError: The archive directory is missing or a stored week is
            unreadable, or neither source was given.
    """
    destination: Path = Path(output_dir)
    copied_from_archive: int = 0
    archive_latest: bool = False
    if archive_dir is not None:
        archived: dict[str, Any] = _copy_archive(destination, Path(archive_dir))
        copied_from_archive = int(archived["copied"])
        archive_latest = bool(archived["latest"])
    pages: str = "" if pages_base is None else pages_base.strip()
    if pages == "" and archive_dir is None:
        raise ArchiveError("Restore needs an archive directory or the live site URL.")
    copied_from_site: int = 0
    if pages != "":
        already_latest: bool = archive_latest or (destination / "latest.json").is_file()
        site: dict[str, int] = restore_published_predictions(
            destination,
            pages,
            fetch_text,
            skip_existing=True,
            write_latest=not already_latest,
        )
        copied_from_site = int(site["copied"])
    logger.info(
        "Restored %s week file(s) from the archive and %s from the live site.",
        copied_from_archive,
        copied_from_site,
    )
    return {
        "copied_from_archive": copied_from_archive,
        "copied_from_site": copied_from_site,
        "copied": copied_from_archive + copied_from_site,
        "archive_latest": archive_latest,
    }


def restore_from_repo(
    repo: Path,
    output_dir: Path,
    *,
    remote: str = "origin",
    branch: str = ARCHIVE_BRANCH,
    pages_base: str | None = None,
    fetch_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Restore from ``branch`` on ``remote``, then fill gaps from the live site.

    A missing branch falls back to the live site. Any other git failure
    does not fall back.

    Raises:
        ArchiveError: The remote cannot be read, or a stored week is bad.
    """
    _refuse_bad_ref(remote, branch)
    work: Path = Path(repo)
    if not _remote_has_branch(work, remote, branch):
        logger.info("Archive branch %s is not on %s. Using the live site.", branch, remote)
        return restore_slates(
            output_dir, archive_dir=None, pages_base=pages_base, fetch_text=fetch_text
        )
    with _checked_out_archive(work, remote, branch) as archive_dir:
        return restore_slates(
            output_dir,
            archive_dir=archive_dir,
            pages_base=pages_base,
            fetch_text=fetch_text,
        )


@contextmanager
def open_archive_readonly(
    repo: Path,
    *,
    remote: str = "origin",
    branch: str = ARCHIVE_BRANCH,
) -> Iterator[Path]:
    """Yield a detached checkout of the archive branch.

    The branch is not created when it is missing, and nothing is pushed.
    Callers read the files and leave them unchanged.

    Raises:
        ArchiveError: The branch is missing, or git cannot read it.
    """
    _refuse_bad_ref(remote, branch)
    if not _remote_has_branch(Path(repo), remote, branch):
        raise ArchiveError(f"Archive branch {branch} is not on {remote}.")
    with _checked_out_archive(Path(repo), remote, branch) as archive_dir:
        yield archive_dir


def publish_archive(
    repo: Path,
    site_data: Path,
    *,
    remote: str = "origin",
    branch: str = ARCHIVE_BRANCH,
    archived_at: str,
    author_name: str,
    author_email: str,
    backfill: bool = False,
) -> dict[str, Any]:
    """Commit ``site_data`` onto the archive branch and push it.

    The push is not forced. Identical bytes do not add a commit. A failed
    check or a rejected push leaves the remote branch as it was.

    Returns:
        The ``stage_archive`` fields plus ``committed``, ``pushed``, and
        ``sha``.

    Raises:
        ArchiveError: The slate is refused, git fails, or the push is rejected.
    """
    _refuse_bad_ref(remote, branch)
    _refuse_blank_author(author_name, author_email)
    parent: Path = Path(tempfile.mkdtemp(prefix="endzone-archive-"))
    dest: Path = parent / "wt"
    state: dict[str, bool] = {"created": False, "pushed": False}
    try:
        _prepare_worktree(Path(repo), dest, remote, branch, state)
        staged: dict[str, Any] = stage_archive(
            dest,
            Path(site_data),
            archived_at=archived_at,
            backfill=backfill,
        )
        if _porcelain(dest) == "":
            sha: str = "" if state["created"] else _head(dest)
            logger.info("Published slates are unchanged. No archive commit.")
            return {**staged, "committed": False, "pushed": False, "sha": sha}
        _stage_paths(dest)
        _assert_archive_index(dest)
        sha = _commit_archive(parent, dest, str(staged["message"]), author_name, author_email)
        _push_archive(dest, remote, branch)
        _point_local_branch(dest, branch)
        state["pushed"] = True
        logger.info("Pushed %s (%s).", staged["message"], sha)
        return {**staged, "committed": True, "pushed": True, "sha": sha}
    finally:
        _remove_worktree(Path(repo), dest, branch, state)
        shutil.rmtree(parent, ignore_errors=True)


def backfill_archive(
    repo: Path,
    pages_base: str,
    *,
    remote: str = "origin",
    branch: str = ARCHIVE_BRANCH,
    archived_at: str,
    author_name: str,
    author_email: str,
    fetch_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Seed the archive from the live site and push that commit.

    This does not score and it does not deploy. The no-shrink rule still
    applies when the branch already has weeks.

    Raises:
        ArchiveError: The live site has no real week, or the push is refused.
    """
    pages: str = pages_base.strip()
    if pages == "":
        raise ArchiveError("Backfill needs the live site URL.")
    scratch_parent: Path = Path(tempfile.mkdtemp(prefix="endzone-backfill-"))
    scratch: Path = scratch_parent / "site"
    scratch.mkdir()
    try:
        copied: dict[str, int] = restore_published_predictions(
            scratch,
            pages,
            fetch_text,
            write_latest=True,
        )
        if copied["copied"] == 0:
            raise ArchiveError("The live site has no published week to archive.")
        return publish_archive(
            Path(repo),
            scratch,
            remote=remote,
            branch=branch,
            archived_at=archived_at,
            author_name=author_name,
            author_email=author_email,
            backfill=True,
        )
    finally:
        shutil.rmtree(scratch_parent, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the archive command parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Restore and archive published weekly slates. Does not fit a model."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    restore = commands.add_parser(
        "restore",
        help="Copy the archive first, then weeks it does not have from the live site.",
    )
    restore.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    restore.add_argument("--archive-dir", default=None)
    restore.add_argument("--pages-base", default=None)
    restore.add_argument("--repo", default=".")
    restore.add_argument("--remote", default="origin")
    restore.add_argument("--branch", default=ARCHIVE_BRANCH)

    publish = commands.add_parser(
        "publish",
        help="Commit site week files to the archive branch and push.",
    )
    publish.add_argument("--site-data", required=True)
    publish.add_argument("--repo", default=".")
    publish.add_argument("--remote", default="origin")
    publish.add_argument("--branch", default=ARCHIVE_BRANCH)
    publish.add_argument("--archived-at", default=None)

    backfill = commands.add_parser(
        "backfill",
        help="Seed the archive branch from the live site. Does not score or deploy.",
    )
    backfill.add_argument("--pages-base", required=True)
    backfill.add_argument("--repo", default=".")
    backfill.add_argument("--remote", default="origin")
    backfill.add_argument("--branch", default=ARCHIVE_BRANCH)
    backfill.add_argument("--archived-at", default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run restore, publish, or backfill.

    Raises:
        SystemExit: The archive command failed.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    try:
        result: dict[str, Any] = _run(args)
    except ArchiveError as exc:
        if args.command != "restore":
            _emit_archive_status({"committed": False, "status": "failed"})
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    _print_result(args.command, result)
    if args.command != "restore":
        committed: bool = bool(result.get("committed"))
        _emit_archive_status(
            {
                "committed": committed,
                "status": "committed" if committed else "nothing to commit",
            }
        )
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch one parsed archive command."""
    if args.command == "restore":
        if args.archive_dir:
            return restore_slates(
                Path(args.output_dir),
                archive_dir=Path(args.archive_dir),
                pages_base=args.pages_base,
            )
        return restore_from_repo(
            Path(args.repo),
            Path(args.output_dir),
            remote=args.remote,
            branch=args.branch,
            pages_base=args.pages_base,
        )
    author: dict[str, str] = _author_from_env()
    stamp: str = args.archived_at or archived_at_text()
    if args.command == "publish":
        return publish_archive(
            Path(args.repo),
            Path(args.site_data),
            remote=args.remote,
            branch=args.branch,
            archived_at=stamp,
            author_name=author["author_name"],
            author_email=author["author_email"],
        )
    return backfill_archive(
        Path(args.repo),
        args.pages_base,
        remote=args.remote,
        branch=args.branch,
        archived_at=stamp,
        author_name=author["author_name"],
        author_email=author["author_email"],
    )


def _emit_archive_status(fields: Mapping[str, object]) -> None:
    """Record committed or nothing to commit. The remote URL stays out.

    A status-file error is logged and does not change the archive result.
    """
    try:
        append_github_output(fields)
        directory: Path | None = status_directory()
        if directory is None:
            return
        write_step_status(directory, "archive", fields)
    except OSError as exc:
        logger.warning("Archive status was not written: %s", type(exc).__name__)


def _print_result(command: str, result: Mapping[str, Any]) -> None:
    """Write one status line for the workflow log."""
    if command == "restore":
        print(
            "Restored "
            f"{result['copied_from_archive']} archive week(s) and "
            f"{result['copied_from_site']} live week(s)."
        )
        return
    if result.get("committed"):
        print(f"Pushed {result['message']} ({result['sha']}).")
        return
    print("Published slates are unchanged. No archive commit.")


def _author_from_env() -> dict[str, str]:
    """Author for a workflow commit. The default is the Actions bot."""
    name: str = os.environ.get("ARCHIVE_GIT_NAME", BOT_NAME).strip()
    email: str = os.environ.get("ARCHIVE_GIT_EMAIL", BOT_EMAIL).strip()
    return {"author_name": name, "author_email": email}


def _require_stamp(archived_at: str) -> str:
    """Reject a stamp that is not safe as a file name."""
    if _STAMP.fullmatch(archived_at) is None:
        raise ArchiveError("Archive timestamp must look like 20261109T042837Z, with no colon.")
    return archived_at


def _refuse_bad_ref(remote: str, branch: str) -> None:
    """Reject a ref that could point git at the wrong branch."""
    if remote.startswith("-") or branch.startswith("-"):
        raise ArchiveError("Remote and branch must not start with a dash.")
    if _BRANCH_NAME.fullmatch(branch) is None:
        raise ArchiveError(f"Archive branch name is not allowed: {branch}.")
    if branch in _PROTECTED_BRANCHES:
        raise ArchiveError(f"Refusing to archive onto {branch}.")


def _refuse_blank_author(author_name: str, author_email: str) -> None:
    """Require both halves of a git identity."""
    if author_name.strip() == "" or author_email.strip() == "" or "@" not in author_email:
        raise ArchiveError("Archive commits need an author name and email.")


def _load_incoming(site_data: Path) -> dict[str, Any]:
    """Read the week files and ``latest.json`` a deploy would publish."""
    if not site_data.is_dir():
        raise ArchiveError("Site data directory does not exist.")
    weeks: list[dict[str, Any]] = []
    predictions: Path = site_data / "predictions"
    if predictions.is_dir():
        for path in sorted(predictions.glob("*.json")):
            if not path.is_file() or _WEEK_FILE.match(path.name) is None:
                continue
            record: dict[str, Any] | None = _load_slate_file(path)
            if record is not None:
                weeks.append(record)
    latest_path: Path = site_data / "latest.json"
    if not latest_path.is_file():
        raise ArchiveError("Refusing to archive without latest.json.")
    latest: dict[str, Any] | None = _load_slate_file(latest_path)
    if latest is None:
        raise ArchiveError("Refusing to archive the example slate.")
    if not weeks:
        raise ArchiveError("Refusing to archive a slate with no week files.")
    _require_latest_match(latest, weeks)
    return {"weeks": weeks, "latest": latest}


def _load_slate_file(path: Path) -> dict[str, Any] | None:
    """Return one slate record, or None when the file is an example slate."""
    raw: bytes = _read_bytes(path)
    document: dict[str, Any] = _parse_object(path.name, raw)
    if document.get("sample") is True:
        return None
    return _slate_record(path.name, document, raw)


def _slate_record(name: str, document: dict[str, Any], raw: bytes) -> dict[str, Any]:
    """Validate one document and collect its game ids."""
    try:
        validate_prediction_document(document)
    except (TypeError, ValueError) as exc:
        raise ArchiveError(f"{name} is not a publishable slate: {exc}") from exc
    season: object = document.get("season")
    week: object = document.get("week")
    if isinstance(season, bool) or not isinstance(season, int):
        raise ArchiveError(f"{name} has no integer season.")
    if isinstance(week, bool) or not isinstance(week, int):
        raise ArchiveError(f"{name} has no integer week.")
    if name != "latest.json" and name != f"{season}-w{week}.json":
        raise ArchiveError(f"{name} does not match season {season} week {week}.")
    game_ids: set[str] = set()
    games: object = document.get("games")
    if isinstance(games, list):
        for game in games:
            if isinstance(game, dict) and game.get("game_id") is not None:
                game_ids.add(str(game["game_id"]))
    if not game_ids:
        raise ArchiveError(f"{name} has no game_id values.")
    return {
        "name": name,
        "season": season,
        "week": week,
        "game_ids": game_ids,
        "raw": raw,
    }


def _require_latest_match(latest: dict[str, Any], weeks: list[dict[str, Any]]) -> None:
    """Require ``latest.json`` to carry the same ids as its week file."""
    matches: list[dict[str, Any]] = [
        item
        for item in weeks
        if item["season"] == latest["season"] and item["week"] == latest["week"]
    ]
    if not matches:
        raise ArchiveError(
            f"latest.json is {latest['season']} week {latest['week']}, "
            "and that week file is missing."
        )
    if matches[0]["game_ids"] != latest["game_ids"]:
        raise ArchiveError(
            f"latest.json game ids do not match {latest['season']} week {latest['week']}."
        )


def _refuse_missing_weeks(archive_root: Path, weeks: list[dict[str, Any]]) -> None:
    """Fail when this deploy omits a week the archive already stores."""
    incoming: set[tuple[int, int]] = {(int(item["season"]), int(item["week"])) for item in weeks}
    for season, week in sorted(_archived_week_keys(archive_root)):
        if (season, week) not in incoming:
            raise ArchiveError(
                f"Refusing to archive: {season} week {week} is already archived "
                "and this deploy does not include it."
            )


def _archived_week_keys(archive_root: Path) -> set[tuple[int, int]]:
    """Season and week pairs present as a tip file or a history directory."""
    keys: set[tuple[int, int]] = set()
    predictions: Path = archive_root / "predictions"
    if predictions.is_dir():
        for path in predictions.glob("*.json"):
            match: re.Match[str] | None = _WEEK_FILE.match(path.name)
            if match is not None and path.is_file():
                keys.add((int(match.group("season")), int(match.group("week"))))
    history: Path = archive_root / "history"
    if history.is_dir():
        for path in history.iterdir():
            if not path.is_dir() or path.name == "latest":
                continue
            match = _WEEK_DIR.match(path.name)
            if match is not None:
                keys.add((int(match.group("season")), int(match.group("week"))))
    return keys


def _refuse_dropped_ids(archive_root: Path, slates: list[dict[str, Any]]) -> None:
    """Fail when an incoming slate drops a game id the archive already has."""
    for item in slates:
        season: int = int(item["season"])
        week: int = int(item["week"])
        existing: set[str] = archived_game_ids(archive_root, season, week)
        incoming: set[str] = set(item["game_ids"])
        dropped: set[str] = existing - incoming
        if not dropped:
            continue
        shown: list[str] = sorted(dropped)[:20]
        suffix: str = ""
        if len(dropped) > len(shown):
            suffix = f" and {len(dropped) - len(shown)} more"
        dropped_list: str = ", ".join(shown)
        raise ArchiveError(
            f"Refusing to archive {season} week {week}: the archive has "
            f"{len(existing)} game_id(s) and this slate drops {len(dropped)} "
            f"({dropped_list}{suffix})."
        )


def _store_slate(archive_root: Path, item: dict[str, Any], stamp: str, *, latest: bool) -> bool:
    """Write history, then the tip. Return True when either file is new.

    A tip that already has these bytes is left alone, including its history.
    The earlier copy was stored when those bytes were first archived.
    """
    raw: bytes = bytes(item["raw"])
    if latest:
        history_dir: Path = archive_root / "history" / "latest"
        tip: Path = archive_root / "latest.json"
    else:
        history_dir = archive_root / "history" / f"{int(item['season'])}-w{int(item['week'])}"
        tip = archive_root / "predictions" / str(item["name"])
    if tip.is_file() and tip.read_bytes() == raw:
        return False
    wrote_history: bool = _snapshot(history_dir, stamp, raw)
    wrote_tip: bool = _write_bytes(tip, raw)
    return wrote_history or wrote_tip


def _snapshot(directory: Path, stamp: str, raw: bytes) -> bool:
    """Store ``raw`` under ``directory`` without replacing an older file."""
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.glob("*.json"):
        if existing.is_file() and existing.read_bytes() == raw:
            return False
    index: int = 1
    while index < 1000:
        name: str = f"{stamp}.json" if index == 1 else f"{stamp}-{index}.json"
        candidate: Path = directory / name
        if not candidate.exists():
            candidate.write_bytes(raw)
            return True
        index += 1
    raise ArchiveError(f"Too many snapshots in {directory.name}.")


def _write_bytes(path: Path, raw: bytes) -> bool:
    """Write ``raw`` when the current file differs. Return True when it did."""
    if path.is_file() and path.read_bytes() == raw:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return True


def _ensure_notes(archive_root: Path) -> None:
    """Keep the branch readable and stop git from rewriting JSON newlines."""
    _write_bytes(archive_root / "README.md", _README.encode("utf-8"))
    _write_bytes(archive_root / ".gitattributes", _ATTRIBUTES.encode("utf-8"))


def _ids_in_archive_file(
    path: Path,
    season: int,
    week: int,
    *,
    require_name: bool,
) -> set[str]:
    """Game ids in one stored week file."""
    document: dict[str, Any] = _parse_object(path.name, _read_bytes(path))
    if document.get("sample") is True:
        raise ArchiveError(f"{path.name} is an example slate and cannot be used as an archive.")
    record_name: str = path.name if require_name else "latest.json"
    record: dict[str, Any] = _slate_record(record_name, document, b"")
    if require_name and path.name != f"{season}-w{week}.json":
        raise ArchiveError(f"{path.name} does not match season {season} week {week}.")
    if int(record["season"]) != season or int(record["week"]) != week:
        raise ArchiveError(
            f"{path.name} says season {record['season']} week {record['week']}, "
            f"expected {season} week {week}."
        )
    return set(record["game_ids"])


def _ids_in_latest(path: Path, season: int, week: int) -> set[str]:
    """Game ids when this latest snapshot names the requested week."""
    document: dict[str, Any] = _parse_object(path.name, _read_bytes(path))
    if document.get("sample") is True:
        raise ArchiveError(f"{path.name} is an example slate and cannot be used as an archive.")
    file_season: object = document.get("season")
    file_week: object = document.get("week")
    if file_season != season or file_week != week:
        return set()
    record: dict[str, Any] = _slate_record("latest.json", document, b"")
    return set(record["game_ids"])


def _copy_archive(output_dir: Path, archive_dir: Path) -> dict[str, Any]:
    """Copy tip files after every archived week has been read."""
    if not archive_dir.is_dir():
        raise ArchiveError("Archive directory does not exist.")
    planned: list[tuple[Path, bytes]] = []
    predictions: Path = archive_dir / "predictions"
    if predictions.is_dir():
        for path in sorted(predictions.glob("*.json")):
            if not path.is_file() or _WEEK_FILE.match(path.name) is None:
                continue
            record: dict[str, Any] | None = _load_slate_file(path)
            if record is None:
                continue
            planned.append((output_dir / "predictions" / path.name, bytes(record["raw"])))
    latest_raw: bytes | None = None
    latest_path: Path = archive_dir / "latest.json"
    if latest_path.is_file():
        latest: dict[str, Any] | None = _load_slate_file(latest_path)
        if latest is None:
            raise ArchiveError("Archived latest.json is the example slate.")
        latest_raw = bytes(latest["raw"])
    for destination, raw in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    latest_copied: bool = False
    if latest_raw is not None:
        target: Path = output_dir / "latest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(latest_raw)
        latest_copied = True
    return {"copied": len(planned), "latest": latest_copied}


def _read_bytes(path: Path) -> bytes:
    """Read one archive or site file."""
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArchiveError(f"Could not read {path.name}: {exc}.") from exc


def _parse_object(name: str, raw: bytes) -> dict[str, Any]:
    """Decode UTF-8 JSON. Any other encoding fails closed."""
    try:
        payload: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"Could not read {name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ArchiveError(f"{name} is not a JSON object.")
    return payload


def _prepare_worktree(
    repo: Path,
    dest: Path,
    remote: str,
    branch: str,
    state: dict[str, bool],
) -> None:
    """Check out the archive branch, or start an orphan branch."""
    head: subprocess.CompletedProcess[str] = _run_git(repo, ["rev-parse", "--verify", "HEAD"])
    if head.returncode != 0:
        raise ArchiveError("The repository has no commit to base an archive worktree on.")
    if _remote_has_branch(repo, remote, branch):
        ref: str = f"refs/remotes/{remote}/{branch}"
        _git_ok(
            repo,
            ["fetch", remote, f"refs/heads/{branch}:{ref}"],
            action="Fetching the archive",
        )
        _git_ok(
            repo,
            ["worktree", "add", "--detach", str(dest), ref],
            action="Checking out the archive",
        )
        return
    _git_ok(
        repo,
        ["worktree", "add", "--detach", str(dest), "HEAD"],
        action="Preparing a new archive branch",
    )
    _git_ok(dest, ["checkout", "--orphan", branch], action="Creating the archive branch")
    state["created"] = True
    listed: subprocess.CompletedProcess[str] = _run_git(dest, ["ls-files"])
    if listed.returncode != 0:
        raise ArchiveError(f"Clearing the new archive branch failed: {_redact(listed.stderr)}")
    if listed.stdout.strip() != "":
        _git_ok(dest, ["rm", "-rf", "."], action="Clearing files from the new archive branch")
    _git_ok(dest, ["clean", "-fdx"], action="Clearing untracked files from the new archive branch")


@contextmanager
def _checked_out_archive(repo: Path, remote: str, branch: str) -> Iterator[Path]:
    """Yield a detached worktree of the archive branch, then remove it."""
    parent: Path = Path(tempfile.mkdtemp(prefix="endzone-archive-"))
    dest: Path = parent / "wt"
    state: dict[str, bool] = {"created": False, "pushed": False}
    try:
        _prepare_worktree(repo, dest, remote, branch, state)
        yield dest
    finally:
        _remove_worktree(repo, dest, branch, state)
        shutil.rmtree(parent, ignore_errors=True)


def _remove_worktree(repo: Path, dest: Path, branch: str, state: Mapping[str, bool]) -> None:
    """Drop the temporary worktree. Delete a branch this attempt did not push."""
    if dest.exists():
        _run_git(repo, ["worktree", "remove", "--force", str(dest)])
    if state.get("created") and not state.get("pushed"):
        _run_git(repo, ["branch", "-D", branch])


def _stage_paths(dest: Path) -> None:
    """Stage archive files, including JSON that a root gitignore would skip."""
    paths: list[str] = [
        name
        for name in ("README.md", ".gitattributes", "latest.json", "predictions", "history")
        if (dest / name).exists()
    ]
    if not paths:
        raise ArchiveError("The archive worktree has no files to commit.")
    _git_ok(dest, ["add", "-f", "--", *paths], action="Staging archive files")


def _assert_archive_index(dest: Path) -> None:
    """Refuse a commit that picked up files from main."""
    listed: subprocess.CompletedProcess[str] = _git_ok(
        dest,
        ["ls-files"],
        action="Listing staged archive files",
    )
    for name in listed.stdout.splitlines():
        if name in _ALLOWED_ROOT_FILES:
            continue
        if name.startswith("predictions/") and _WEEK_FILE.match(Path(name).name):
            continue
        if _HISTORY_FILE.match(name) or _LATEST_HISTORY.match(name):
            continue
        raise ArchiveError(f"Refusing to archive unexpected path {name}.")


def _commit_archive(
    parent: Path,
    dest: Path,
    message: str,
    author_name: str,
    author_email: str,
) -> str:
    """Commit with an empty hooks directory so no trailer is appended."""
    if message.strip() == "":
        raise ArchiveError("Archive changes have no commit message.")
    hooks: Path = parent / "hooks"
    hooks.mkdir(exist_ok=True)
    commit_env: dict[str, str] = os.environ.copy()
    commit_env["GIT_AUTHOR_NAME"] = author_name
    commit_env["GIT_AUTHOR_EMAIL"] = author_email
    commit_env["GIT_COMMITTER_NAME"] = author_name
    commit_env["GIT_COMMITTER_EMAIL"] = author_email
    _git_ok(
        dest,
        ["-c", f"core.hooksPath={hooks}", "commit", "--no-verify", "-m", message],
        env=commit_env,
        action="Committing the archive",
    )
    return _head(dest)


def _push_archive(dest: Path, remote: str, branch: str) -> None:
    """Push the archive commit. Never force it."""
    _git_ok(
        dest,
        ["push", remote, f"HEAD:refs/heads/{branch}"],
        action="Pushing the archive",
    )


def _point_local_branch(dest: Path, branch: str) -> None:
    """Move the local branch ref to the commit that was just pushed.

    A detached worktree does not move the branch on commit. The next fetch
    in this clone should see the same SHA the remote has.
    """
    ref: str = f"refs/heads/{branch}"
    updated: subprocess.CompletedProcess[str] = _run_git(dest, ["update-ref", ref, "HEAD"])
    if updated.returncode == 0:
        return
    current: subprocess.CompletedProcess[str] = _run_git(dest, ["symbolic-ref", "-q", "HEAD"])
    if current.returncode == 0 and current.stdout.strip() == ref:
        return
    detail: str = _redact(updated.stderr or updated.stdout)
    raise ArchiveError(f"Updating the local archive branch failed: {detail}")


def _porcelain(repo: Path) -> str:
    """Return ``git status --porcelain``, stripped."""
    completed: subprocess.CompletedProcess[str] = _git_ok(
        repo,
        ["status", "--porcelain"],
        action="Reading archive status",
    )
    return completed.stdout.strip()


def _head(repo: Path) -> str:
    """Return the current commit SHA."""
    completed: subprocess.CompletedProcess[str] = _git_ok(
        repo,
        ["rev-parse", "HEAD"],
        action="Reading the archive revision",
    )
    return completed.stdout.strip()


def _remote_has_branch(repo: Path, remote: str, branch: str) -> bool:
    """True when ``remote`` already has ``branch``. Other git errors raise."""
    completed: subprocess.CompletedProcess[str] = _git_ok(
        repo,
        ["ls-remote", "--heads", remote, f"refs/heads/{branch}"],
        action=f"Reading {remote} {branch}",
    )
    return completed.stdout.strip() != ""


def _git_ok(
    repo: Path,
    args: list[str],
    *,
    action: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run git and raise ``ArchiveError`` when it fails."""
    completed: subprocess.CompletedProcess[str] = _run_git(repo, args, env=env)
    if completed.returncode != 0:
        detail: str = _redact(completed.stderr or completed.stdout)
        raise ArchiveError(f"{action} failed: {detail}")
    return completed


def _run_git(
    repo: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run git with UTF-8 output and no credential prompt."""
    command: list[str] = [
        "git",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.quotepath=false",
        "-c",
        "commit.gpgsign=false",
        *args,
    ]
    run_env: dict[str, str] = os.environ.copy()
    run_env["GIT_TERMINAL_PROMPT"] = "0"
    if env is not None:
        run_env.update(env)
    try:
        return subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=run_env,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise ArchiveError("Git timed out.") from exc
    except OSError as exc:
        raise ArchiveError(f"Could not run git: {exc}.") from exc


def _redact(text: str) -> str:
    """Drop credential material that git may echo in a remote URL."""
    return _TOKEN_IN_URL.sub("x-access-token:***@", text).strip()


if __name__ == "__main__":
    main()
