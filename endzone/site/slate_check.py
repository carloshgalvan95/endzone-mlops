"""Open one issue when the Tuesday slate or the previous week's finals are missing.

The expected week is ``select_slate`` on the games whose ``gameday`` falls
in the current Tuesday-to-Tuesday window (America/Mexico_City). That is the
scorer's week picker, limited to the week a Tuesday publish is responsible
for. A window with no games is the offseason: the command exits without
calling GitHub.

The current week's file is ``predictions/<season>-w<week>.json`` on
``published-slates``. After 08:00 on Tuesday, the previous scheduled week
must have a final score on every game. That is the track record state
``complete``. The command does not score, deploy, or write to a branch.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from endzone.ingest.nflverse import NFLVERSE_GAMES_URL, download_parquet_with_validation
from endzone.modeling.weekly import select_slate
from endzone.site.archive import ARCHIVE_BRANCH, ArchiveError, open_archive_readonly
from endzone.site.model_alert import GithubIssueClient, ModelAlertError
from endzone.site.results import read_games_file
from endzone.site.track_record import result_side
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

MEXICO_CITY: ZoneInfo = ZoneInfo("America/Mexico_City")
# Tuesday 08:00 local. Monday night finals should be on the slate by then.
PREVIOUS_FINALS_HOUR: int = 8
SLATE_ALERT_LABEL: str = "area:observability"
SLATE_ALERT_LABEL_COLOR: str = "1d76db"
SLATE_ALERT_LABEL_DESCRIPTION: str = (
    "A published week file is missing, or the previous week is missing finals."
)


class SlateCheckError(Exception):
    """The slate check could not decide, or the issue could not be opened."""


def publish_window(today: date) -> dict[str, date]:
    """Return the Tuesday-to-Tuesday window that contains ``today``.

    ``start`` is the Tuesday on or before ``today``. ``end`` is the next
    Tuesday and is exclusive. Game dates in that half-open span are the
    week a Tuesday publish owns.

    Returns:
        Dict with ``start`` and ``end``.
    """
    days_since_tuesday: int = (today.weekday() - 1) % 7
    start: date = today - timedelta(days=days_since_tuesday)
    return {"start": start, "end": start + timedelta(days=7)}


def expected_publish_week(schedule: pd.DataFrame, today: date) -> dict[str, int] | None:
    """Return the week ``select_slate`` would publish for this Tuesday window.

    Games outside the window are ignored, so the offseason and next season's
    schedule do not invent a week. Inside the window, ``select_slate`` picks
    the earliest season and week. That is the same function the scorer uses.
    An empty window means there is no slate to expect.

    Returns:
        Dict with ``season`` and ``week``, or None outside the season.

    Raises:
        ValueError: A row in the window has no season, week, or game id.
    """
    window: dict[str, date] = publish_window(today)
    in_window: pd.DataFrame = _games_in_window(schedule, window["start"], window["end"])
    if in_window.empty:
        return None
    chosen: dict[str, Any] = select_slate(in_window, None, None)
    return {"season": int(chosen["season"]), "week": int(chosen["week"])}


def previous_scheduled_week(
    schedule: pd.DataFrame,
    season: int,
    week: int,
) -> dict[str, int] | None:
    """Return the latest scheduled week strictly before ``season`` and ``week``.

    Week 1 uses the previous season's last week when that season is on the
    schedule. A schedule that does not contain an earlier week returns None.

    Returns:
        Dict with ``season`` and ``week``, or None.
    """
    if schedule.empty or "season" not in schedule.columns or "week" not in schedule.columns:
        return None
    seasons: pd.Series = pd.to_numeric(schedule["season"], errors="coerce")
    weeks: pd.Series = pd.to_numeric(schedule["week"], errors="coerce")
    pairs: set[tuple[int, int]] = set()
    for season_value, week_value in zip(seasons.tolist(), weeks.tolist(), strict=True):
        if pd.isna(season_value) or pd.isna(week_value):
            continue
        pairs.add((int(season_value), int(week_value)))
    current: tuple[int, int] = (int(season), int(week))
    prior: list[tuple[int, int]] = sorted(pair for pair in pairs if pair < current)
    if not prior:
        return None
    chosen: tuple[int, int] = prior[-1]
    return {"season": chosen[0], "week": chosen[1]}


def previous_finals_are_due(moment: datetime) -> bool:
    """True when the previous week should already have every final.

    Before 08:00 America/Mexico_City on Tuesday, Monday night's score may
    still be absent. Every later hour, including the rest of the week, asks
    for those finals.

    Raises:
        ValueError: ``moment`` has no timezone.
    """
    if moment.tzinfo is None:
        raise ValueError("moment must include a timezone.")
    local: datetime = moment.astimezone(MEXICO_CITY)
    if local.weekday() == 1 and local.hour < PREVIOUS_FINALS_HOUR:
        return False
    return True


def slate_finals_complete(document: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return whether every game in one slate is final.

    A game is final when ``result_side`` accepts it, the same rule the
    track record uses for state ``complete``. A missing document, an example
    slate, or an empty games list is not complete.

    Returns:
        Dict with ``complete``, ``reason``, ``games``, and ``unfinished``.
    """
    if document is None or document.get("sample") is True:
        return {"complete": False, "reason": "missing", "games": 0, "unfinished": 0}
    games: object = document.get("games")
    if not isinstance(games, list) or len(games) == 0:
        return {"complete": False, "reason": "no_games", "games": 0, "unfinished": 0}
    unfinished: int = 0
    for game in games:
        if not isinstance(game, dict):
            unfinished += 1
            continue
        if result_side(game.get("status"), game.get("home_score"), game.get("away_score")) is None:
            unfinished += 1
    complete: bool = unfinished == 0
    return {
        "complete": complete,
        "reason": "complete" if complete else "unfinished",
        "games": len(games),
        "unfinished": unfinished,
    }


def slate_check_findings(
    expected: Mapping[str, int] | None,
    current_document: Mapping[str, Any] | None,
    previous: Mapping[str, int] | None,
    previous_document: Mapping[str, Any] | None,
    *,
    previous_finals_due: bool,
) -> dict[str, Any]:
    """Return whether this week should open an issue.

    Outside the season (``expected`` is None) nothing is missing. Otherwise
    a missing current file is a problem. Previous-week finals are a problem
    only when ``previous_finals_due`` is true and an earlier week exists.

    Returns:
        Dict with ``alert``, ``reason``, ``season``, ``week``, and
        ``problems``.
    """
    if expected is None:
        return {
            "alert": False,
            "reason": "outside_season",
            "season": None,
            "week": None,
            "problems": [],
        }
    season: int = int(expected["season"])
    week: int = int(expected["week"])
    problems: list[dict[str, Any]] = []
    if current_document is None:
        problems.append({"kind": "missing_slate", "season": season, "week": week})
    if previous_finals_due and previous is not None:
        finals: dict[str, Any] = slate_finals_complete(previous_document)
        if not finals["complete"]:
            problems.append(
                {
                    "kind": "missing_finals",
                    "season": int(previous["season"]),
                    "week": int(previous["week"]),
                    "games": int(finals["games"]),
                    "unfinished": int(finals["unfinished"]),
                }
            )
    return {
        "alert": len(problems) > 0,
        "reason": "missing" if problems else "ok",
        "season": season,
        "week": week,
        "problems": problems,
    }


def slate_alert_title(season: int, week: int) -> str:
    """Return the one issue title for this season and week.

    The title is stable so a second run finds the open issue and does not
    open another. The body lists whether the file, the finals, or both
    are missing.
    """
    return f"Slate check failed: {int(season)} week {int(week)}"


def slate_alert_body(findings: Mapping[str, Any]) -> str:
    """Return the issue body. It tells the operator which dispatch to run."""
    season: int = int(findings["season"])
    week: int = int(findings["week"])
    lines: list[str] = [
        f"The Tuesday slate check found a problem for {season} week {week}.",
        "",
    ]
    problems: object = findings.get("problems")
    items: list[object] = list(problems) if isinstance(problems, list) else []
    for problem in items:
        if not isinstance(problem, dict):
            continue
        lines.append(_problem_line(problem))
    lines.extend(
        [
            "",
            "GitHub scheduled workflows are best effort. A backup cron may have been skipped.",
            "",
            "If predictions/"
            f"{season}-w{week}.json is missing, open Actions, Pages, Run workflow. "
            f"Set season to {season} and week to {week}. Leave replace_published off. "
            "That scores the week only when the file is not already on published-slates. "
            "It does not rebuild a slate that is already published.",
            "",
            "If the previous week is missing finals, open Actions, Results, Run workflow. "
            "That overlays nflverse scores onto the published slate. It does not rescore.",
            "",
            "This check does not score, deploy, or write to a branch. "
            "Close this issue after the file is present and the previous week is complete. "
            "A later run with the same title does not open a second issue.",
        ]
    )
    return "\n".join(lines) + "\n"


def find_open_issue(issues: list[Mapping[str, Any]], title: str) -> dict[str, Any]:
    """Return the first open issue whose title matches exactly.

    Pull requests are ignored. The match is the whole title, so a different
    week does not count.

    Returns:
        Dict with ``found`` and ``number``.
    """
    for item in issues:
        if "pull_request" in item:
            continue
        if item.get("title") != title:
            continue
        number: object = item.get("number")
        if isinstance(number, bool) or not isinstance(number, int):
            continue
        return {"found": True, "number": number}
    return {"found": False, "number": None}


def sync_slate_issue(findings: Mapping[str, Any], client: Any) -> dict[str, Any]:
    """Open one issue when the check failed, unless that title is already open.

    A quiet week does not call GitHub. An existing open issue with the same
    title is left as it is.

    Returns:
        Dict with ``action`` (``skipped``, ``unchanged``, or ``created``)
        and ``number``.
    """
    if not findings.get("alert"):
        return {"action": "skipped", "reason": findings.get("reason"), "number": None}
    season: int = int(findings["season"])
    week: int = int(findings["week"])
    title: str = slate_alert_title(season, week)
    body: str = slate_alert_body(findings)
    client.ensure_label(
        SLATE_ALERT_LABEL,
        SLATE_ALERT_LABEL_COLOR,
        SLATE_ALERT_LABEL_DESCRIPTION,
    )
    existing: dict[str, Any] = find_open_issue(client.list_all_open(), title)
    if existing["found"]:
        logger.info(
            "Open issue #%s already has this title. Not opening another.",
            existing["number"],
        )
        return {"action": "unchanged", "number": existing["number"], "reason": "already_open"}
    created: dict[str, Any] = client.create(title, body, [SLATE_ALERT_LABEL])
    return {"action": "created", "number": created.get("number"), "reason": "opened"}


def read_published_slate(archive: Path, season: int, week: int) -> dict[str, Any] | None:
    """Read one tip file from a ``published-slates`` checkout.

    A missing file returns None. An example slate returns None. A file that
    exists and is not JSON fails closed.

    Raises:
        SlateCheckError: The file exists but cannot be read as a JSON object.
    """
    path: Path = Path(archive) / "predictions" / f"{int(season)}-w{int(week)}.json"
    if not path.is_file():
        return None
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SlateCheckError(f"Could not read {path.name}: {exc}.") from exc
    if not isinstance(payload, dict) or payload.get("sample") is True:
        return None
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the slate-check parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Open one issue when the current week's slate is missing or the "
            "previous week has no finals. Does not score or deploy."
        )
    )
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--archive-dir", type=str, default=None)
    parser.add_argument("--remote", type=str, default="origin")
    parser.add_argument("--branch", type=str, default=ARCHIVE_BRANCH)
    parser.add_argument(
        "--games-path",
        type=str,
        default=None,
        help="Local games parquet or CSV. Omit it to download the nflverse schedules file.",
    )
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Timezone-aware timestamp. The default is the current UTC time.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the decision and do not call GitHub.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the check. Return 0 when there is nothing to open or the issue exists.

    Return 1 when the schedule, the archive, or GitHub cannot be used. A
    missing slate with a successful issue create returns 0.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    try:
        _run(args)
    except (SlateCheckError, ArchiveError, ModelAlertError, OSError, ValueError) as exc:
        logger.error("Slate check failed: %s", exc)
        return 1
    return 0


def _run(args: argparse.Namespace) -> None:
    """Load the schedule and the archive, then open at most one issue."""
    moment: datetime = _parse_moment(args.now)
    schedule: pd.DataFrame = _load_schedule(args.games_path)
    today: date = moment.astimezone(MEXICO_CITY).date()
    expected: dict[str, int] | None = expected_publish_week(schedule, today)
    if expected is None:
        logger.info("No scheduled games in the week of %s. No issue.", today.isoformat())
        return
    with _archive_dir(args) as archive:
        current: dict[str, Any] | None = read_published_slate(
            archive, expected["season"], expected["week"]
        )
        previous: dict[str, int] | None = previous_scheduled_week(
            schedule, expected["season"], expected["week"]
        )
        previous_document: dict[str, Any] | None = None
        if previous is not None:
            previous_document = read_published_slate(archive, previous["season"], previous["week"])
        findings: dict[str, Any] = slate_check_findings(
            expected,
            current,
            previous,
            previous_document,
            previous_finals_due=previous_finals_are_due(moment),
        )
    logger.info(
        "Slate check %s for %s week %s (%s problem(s)).",
        findings["reason"],
        findings["season"],
        findings["week"],
        len(findings["problems"]),
    )
    if not findings["alert"] or args.dry_run:
        return
    repo: str = args.repo or os.environ.get("GITHUB_REPOSITORY") or ""
    token: str = os.environ.get("GITHUB_TOKEN") or ""
    if repo.strip() == "" or token.strip() == "":
        raise SlateCheckError("GITHUB_REPOSITORY or GITHUB_TOKEN is unset. No issue was opened.")
    result: dict[str, Any] = sync_slate_issue(findings, GithubIssueClient(repo, token))
    logger.info("Slate check issue action=%s number=%s.", result["action"], result["number"])


def _parse_moment(text: str | None) -> datetime:
    """Parse a timezone-aware timestamp. Empty means now."""
    if text is None or text.strip() == "":
        return datetime.now(UTC)
    try:
        moment: datetime = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SlateCheckError(f"--now is not a timestamp: {text}.") from exc
    if moment.tzinfo is None:
        raise SlateCheckError("--now must include a timezone.")
    return moment


def _load_schedule(games_path: str | None) -> pd.DataFrame:
    """Read a local games file, or download the nflverse schedules release."""
    if games_path:
        return read_games_file(Path(games_path))
    logger.info("Downloading the nflverse schedule from %s", NFLVERSE_GAMES_URL)
    return download_parquet_with_validation(NFLVERSE_GAMES_URL)


def _archive_dir(args: argparse.Namespace) -> Any:
    """Yield a directory of published week files. Does not push."""
    if args.archive_dir:
        return _local_archive(Path(args.archive_dir))
    if not args.repo:
        raise SlateCheckError("The check needs --repo or --archive-dir.")
    return open_archive_readonly(Path(args.repo), remote=args.remote, branch=args.branch)


@contextmanager
def _local_archive(path: Path) -> Iterator[Path]:
    """Yield a directory that is already a checkout. Does not push."""
    if not path.is_dir():
        raise SlateCheckError(f"Archive directory does not exist: {path}.")
    yield path


def _games_in_window(schedule: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows whose ``gameday`` is on or after ``start`` and before ``end``."""
    if schedule.empty:
        return schedule.copy()
    if "gameday" not in schedule.columns:
        raise ValueError("Schedule is missing gameday.")
    frame: pd.DataFrame = schedule.copy()
    days: pd.Series = pd.to_datetime(frame["gameday"], errors="coerce").dt.normalize()
    # A row with no date is not inside a Tuesday window. Drop it.
    known: pd.Series = days.notna()
    frame = frame.loc[known].copy()
    days = days.loc[known]
    start_stamp: pd.Timestamp = pd.Timestamp(start)
    end_stamp: pd.Timestamp = pd.Timestamp(end)
    chosen: pd.DataFrame = frame.loc[(days >= start_stamp) & (days < end_stamp)].copy()
    return chosen.reset_index(drop=True)


def _problem_line(problem: Mapping[str, Any]) -> str:
    """One bullet for the issue body."""
    kind: object = problem.get("kind")
    season: object = problem.get("season")
    week: object = problem.get("week")
    if kind == "missing_slate":
        return f"- predictions/{season}-w{week}.json is not on published-slates."
    if kind == "missing_finals":
        unfinished: object = problem.get("unfinished")
        games: object = problem.get("games")
        if games == 0:
            return (
                f"- {season} week {week} has no slate, so its finals are missing "
                f"(predictions/{season}-w{week}.json)."
            )
        return f"- {season} week {week} is missing finals for {unfinished} of {games} games."
    return "- The slate check found a problem it could not describe."


if __name__ == "__main__":
    raise SystemExit(main())
