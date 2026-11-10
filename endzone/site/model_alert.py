"""Alert when the published model is worse than its validated baseline.

The metric is rolling log loss on the same decided games the track record
uses. It runs after Results writes ``track_record.json``. A trip opens or
updates one GitHub issue. It does not change a week file, and a failure
here must not fail the publish.

The gap is model log loss minus market log loss. The reference file supplies
the validated gap (logistic ``h2_full`` minus the no-vig market). The alert
trips when the window gap is at least that reference gap plus a margin.
Defaults: last 64 decided games, at least 48 of them, and a margin of 0.02.
A window that matches the reference table stays quiet. Trailing the market
by the reference gap is the known state. Override with the CLI flags or
``ENDZONE_ALERT_WINDOW_GAMES``, ``ENDZONE_ALERT_MIN_GAMES``, and
``ENDZONE_ALERT_GAP_MARGIN``.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from endzone.modeling.reference import reference_market_gap
from endzone.site.archive import ArchiveError, open_archive_readonly
from endzone.site.track_record import (
    TrackRecordError,
    binary_log_loss,
    clip_probability,
    finished_market_games,
)
from endzone.utils.logging import setup_logger
from endzone.utils.step_status import status_directory, write_step_status

logger = setup_logger(__name__)

DEFAULT_WINDOW_GAMES: int = 64
DEFAULT_MIN_GAMES: int = 48
DEFAULT_GAP_MARGIN: float = 0.02
ALERT_LABEL: str = "model-behind-market"
ALERT_LABEL_COLOR: str = "b60205"
ALERT_LABEL_DESCRIPTION: str = (
    "Published log loss gap exceeds the reference CV gap plus the margin."
)
ALERT_TITLE: str = "Published model is worse than its validated baseline"
_ENV_WINDOW: str = "ENDZONE_ALERT_WINDOW_GAMES"
_ENV_MIN: str = "ENDZONE_ALERT_MIN_GAMES"
_ENV_MARGIN: str = "ENDZONE_ALERT_GAP_MARGIN"
_API_ROOT: str = "https://api.github.com"
_API_VERSION: str = "2022-11-28"


class ModelAlertError(Exception):
    """The alert could not be delivered. The publish should continue."""


def evaluate_market_gap(
    games: list[Mapping[str, Any]],
    *,
    window_games: int = DEFAULT_WINDOW_GAMES,
    min_games: int = DEFAULT_MIN_GAMES,
    gap_margin: float = DEFAULT_GAP_MARGIN,
    reference_gap: float | None = None,
) -> dict[str, Any]:
    """Compare model and market log loss on the most recent decided games.

    ``games`` is oldest first. Rows that are not a 0/1 label with two
    probabilities in ``[0, 1]`` are dropped before the window is cut.
    ``reference_gap`` defaults to logistic ``h2_full`` minus the market in
    the locked reference file. The alert trips when the window has at least
    ``min_games`` rows and model log loss minus market log loss is greater
    than or equal to ``reference_gap + gap_margin``. A window that matches
    the reference gap stays quiet.

    Returns:
        Dict with ``tripped``, ``reason``, ``n_games``, both log losses,
        ``gap``, ``reference_gap``, ``gap_margin``, ``gap_threshold``,
        and the last season and week.

    Raises:
        ValueError: A setting is out of range, or ``min_games`` exceeds
            ``window_games``.
    """
    if window_games < 1:
        raise ValueError("window_games must be >= 1.")
    if min_games < 1:
        raise ValueError("min_games must be >= 1.")
    if min_games > window_games:
        raise ValueError("min_games cannot exceed window_games.")
    if gap_margin < 0 or gap_margin != gap_margin:
        raise ValueError("gap_margin must be a finite number >= 0.")
    locked_gap: float = reference_market_gap() if reference_gap is None else float(reference_gap)
    if locked_gap != locked_gap:
        raise ValueError("reference_gap must be finite.")
    gap_threshold: float = locked_gap + float(gap_margin)
    kept: list[dict[str, Any]] = []
    for game in games:
        usable: dict[str, Any] | None = _usable_row(game)
        if usable is not None:
            kept.append(usable)
    window: list[dict[str, Any]] = kept[-window_games:]
    last_season: int | None = None
    last_week: int | None = None
    if window:
        last_season = int(window[-1]["season"]) if window[-1].get("season") is not None else None
        last_week = int(window[-1]["week"]) if window[-1].get("week") is not None else None
    base: dict[str, Any] = {
        "tripped": False,
        "reason": "below_minimum_sample",
        "n_games": len(window),
        "model_log_loss": None,
        "market_log_loss": None,
        "gap": None,
        "window_games": int(window_games),
        "min_games": int(min_games),
        "reference_gap": locked_gap,
        "gap_margin": float(gap_margin),
        "gap_threshold": gap_threshold,
        "last_season": last_season,
        "last_week": last_week,
    }
    if len(window) < min_games:
        return base
    labels: list[int] = [int(row["label"]) for row in window]
    model_loss: float | None = binary_log_loss(
        labels,
        [float(row["model_probability"]) for row in window],
    )
    market_loss: float | None = binary_log_loss(
        labels,
        [float(row["market_probability"]) for row in window],
    )
    if model_loss is None or market_loss is None:
        return base
    gap: float = model_loss - market_loss
    tripped: bool = gap >= float(gap_threshold)
    base["model_log_loss"] = model_loss
    base["market_log_loss"] = market_loss
    base["gap"] = gap
    base["tripped"] = tripped
    base["reason"] = "log_loss_gap" if tripped else "within_gap"
    return base


def alert_issue_body(decision: Mapping[str, Any]) -> str:
    """Stable issue body. A timestamp is omitted so an unchanged window is not edited."""
    model_loss: float = float(decision["model_log_loss"])
    market_loss: float = float(decision["market_log_loss"])
    gap: float = float(decision["gap"])
    lines: list[str] = [
        "The published model is worse than its validated baseline.",
        "The gap is model log loss minus no-vig market log loss.",
        "",
        f"Window: last {int(decision['n_games'])} decided games "
        f"(cap {int(decision['window_games'])}).",
        f"Minimum sample: {int(decision['min_games'])}.",
        f"Model log loss: {model_loss:.6f}.",
        f"Market log loss: {market_loss:.6f}.",
        f"Gap (model minus market): {gap:.6f}.",
        f"Reference CV gap: {float(decision['reference_gap']):.6f}.",
        f"Margin: {float(decision['gap_margin']):.6f}.",
        f"Bar (reference gap plus margin): {float(decision['gap_threshold']):.6f}.",
        f"Last game in the window: season {decision['last_season']} week {decision['last_week']}.",
        "",
        "Lower log loss is closer to the results.",
        "A window that matches the reference table stays quiet.",
        "Trailing the market by the reference gap is the known state.",
        "This issue is not a betting signal.",
        "",
        "Close this issue after reading it. The next Results run updates this same issue "
        "when the window still trips. It does not open a second one.",
        "",
        "Definition: docs/adr/ADR-009-model-registry.md.",
    ]
    return "\n".join(lines) + "\n"


def sync_alert_issue(
    decision: Mapping[str, Any],
    client: Any,
) -> dict[str, Any]:
    """Open or update the single labeled issue when the alert trips.

    A window that does not trip does not touch GitHub. An existing open issue
    with the same title and body is left as it is, so an hourly Results run
    does not edit it again.

    Returns:
        Dict with ``action`` (``skipped``, ``created``, ``updated``, or
        ``unchanged``) and ``number``.
    """
    if not decision.get("tripped"):
        return {"action": "skipped", "reason": decision.get("reason"), "number": None}
    title: str = ALERT_TITLE
    body: str = alert_issue_body(decision)
    client.ensure_label(ALERT_LABEL, ALERT_LABEL_COLOR, ALERT_LABEL_DESCRIPTION)
    open_issues: list[dict[str, Any]] = [
        item for item in client.list_open(ALERT_LABEL) if "pull_request" not in item
    ]
    if not open_issues:
        created: dict[str, Any] = client.create(title, body, [ALERT_LABEL])
        return {"action": "created", "number": created.get("number"), "reason": "log_loss_gap"}
    chosen: dict[str, Any] = min(open_issues, key=lambda item: int(item["number"]))
    if len(open_issues) > 1:
        logger.warning(
            "Found %s open %s issues. Updating #%s and leaving the others.",
            len(open_issues),
            ALERT_LABEL,
            chosen["number"],
        )
    if chosen.get("title") == title and chosen.get("body") == body:
        return {"action": "unchanged", "number": chosen.get("number"), "reason": "log_loss_gap"}
    client.update(int(chosen["number"]), title, body)
    return {"action": "updated", "number": chosen.get("number"), "reason": "log_loss_gap"}


class GithubIssueClient:
    """Minimal Issues API client. The token is not written to the log."""

    def __init__(
        self,
        repo: str,
        token: str,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if "/" not in repo or repo.startswith("/") or repo.endswith("/"):
            raise ModelAlertError(f"GITHUB_REPOSITORY must look like owner/name, got {repo!r}.")
        if token.strip() == "":
            raise ModelAlertError("GITHUB_TOKEN is empty.")
        self._repo: str = repo
        self._token: str = token
        self._opener: Callable[..., Any] = opener if opener is not None else urllib.request.urlopen

    def list_open(self, label: str) -> list[dict[str, Any]]:
        """Return the first page of open issues with ``label``."""
        path: str = f"/repos/{self._repo}/issues?state=open&labels={label}&per_page=100"
        payload: object = self._read("GET", path, None)
        if not isinstance(payload, list):
            raise ModelAlertError("GitHub issues list was not a list.")
        return [item for item in payload if isinstance(item, dict)]

    def list_all_open(self) -> list[dict[str, Any]]:
        """Return open issues, following pages of 100.

        The API includes pull requests. Callers that match titles should
        skip items that contain ``pull_request``.
        """
        found: list[dict[str, Any]] = []
        page: int = 1
        while page <= 20:
            path: str = f"/repos/{self._repo}/issues?state=open&per_page=100&page={page}"
            payload: object = self._read("GET", path, None)
            if not isinstance(payload, list):
                raise ModelAlertError("GitHub issues list was not a list.")
            batch: list[dict[str, Any]] = [item for item in payload if isinstance(item, dict)]
            found.extend(batch)
            if len(batch) < 100:
                return found
            page += 1
        return found

    def create(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        """Open one issue and return the API object."""
        payload: object = self._read(
            "POST",
            f"/repos/{self._repo}/issues",
            {"title": title, "body": body, "labels": labels},
        )
        if not isinstance(payload, dict):
            raise ModelAlertError("GitHub create issue did not return an object.")
        return payload

    def update(self, number: int, title: str, body: str) -> dict[str, Any]:
        """Replace the title and body of one issue."""
        payload: object = self._read(
            "PATCH",
            f"/repos/{self._repo}/issues/{int(number)}",
            {"title": title, "body": body},
        )
        if not isinstance(payload, dict):
            raise ModelAlertError("GitHub update issue did not return an object.")
        return payload

    def ensure_label(self, name: str, color: str, description: str) -> None:
        """Create the label when it is missing. An existing label is left as it is."""
        try:
            self._read(
                "POST",
                f"/repos/{self._repo}/labels",
                {"name": name, "color": color, "description": description},
            )
        except ModelAlertError as exc:
            if "422" in str(exc):
                return
            raise

    def _read(self, method: str, path: str, payload: dict[str, Any] | None) -> object:
        """Send one API call and decode the JSON body.

        Raises:
            ModelAlertError: The HTTP call failed or the body was not JSON.
        """
        url: str = f"{_API_ROOT}{path}"
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("X-GitHub-Api-Version", _API_VERSION)
        request.add_header("User-Agent", "endzone-mlops")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        status: int
        raw: str
        try:
            with self._opener(request, timeout=30) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read().decode("utf-8", errors="replace")
            if method == "POST" and path.endswith("/labels") and status == 422:
                raise ModelAlertError("GitHub label request returned 422.") from exc
            raise ModelAlertError(f"GitHub {method} {path} returned {status}.") from exc
        except urllib.error.URLError as exc:
            raise ModelAlertError(f"GitHub {method} {path} failed: {exc.reason}.") from exc
        except TimeoutError as exc:
            raise ModelAlertError(f"GitHub {method} {path} timed out.") from exc
        if status >= 400:
            raise ModelAlertError(f"GitHub {method} {path} returned {status}.")
        if raw.strip() == "":
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelAlertError(f"GitHub {method} {path} did not return JSON.") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the alert parser. Defaults match the documented thresholds."""
    parser = argparse.ArgumentParser(
        description="Open one issue when published log loss exceeds the reference gap plus a margin."
    )
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--site-data", type=str, default=None)
    parser.add_argument("--archive-dir", type=str, default=None)
    parser.add_argument("--remote", type=str, default="origin")
    parser.add_argument("--branch", type=str, default="published-slates")
    parser.add_argument(
        "--games-json",
        type=str,
        default=None,
        help="JSON list of decided games. Skips the archive. For tests.",
    )
    parser.add_argument("--window-games", type=int, default=None)
    parser.add_argument("--min-games", type=int, default=None)
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Extra log loss above the reference CV gap. The default is 0.02.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the decision and do not call GitHub.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate the window and, when it trips, sync one issue.

    Returns 0 even when the registry of issues cannot be reached. The
    Results publish must not fail because of this step.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    try:
        status: dict[str, Any] = _run(args)
    except (TrackRecordError, ArchiveError, ModelAlertError, OSError, ValueError) as exc:
        logger.warning("Model alert did not update an issue: %s", exc)
        status = {"status": "failed", "action": "none"}
    try:
        _emit_alert_status(status)
    except OSError as exc:
        logger.warning("Model alert status was not written: %s", type(exc).__name__)
    return 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Load games, evaluate, and sync. Raises on a bad archive or a bad API call.

    Returns:
        ``status`` and ``action``. The action is ``none`` when no issue
        was touched. A dry run that would open an issue says ``dry_run``.
    """
    window_games: int = _setting_int(args.window_games, _ENV_WINDOW, DEFAULT_WINDOW_GAMES)
    min_games: int = _setting_int(args.min_games, _ENV_MIN, DEFAULT_MIN_GAMES)
    gap_margin: float = _setting_float(args.margin, _ENV_MARGIN, DEFAULT_GAP_MARGIN)
    games: list[Mapping[str, Any]] = _load_games(args)
    decision: dict[str, Any] = evaluate_market_gap(
        games,
        window_games=window_games,
        min_games=min_games,
        gap_margin=gap_margin,
    )
    logger.info(
        "Model alert %s (n=%s, gap=%s, reference_gap=%s, margin=%s, bar=%s).",
        decision["reason"],
        decision["n_games"],
        _format_optional(decision["gap"]),
        f"{float(decision['reference_gap']):.6f}",
        f"{float(decision['gap_margin']):.6f}",
        f"{float(decision['gap_threshold']):.6f}",
    )
    if not decision["tripped"]:
        return {"status": str(decision["reason"]), "action": "none"}
    if args.dry_run:
        return {"status": str(decision["reason"]), "action": "dry_run"}
    repo: str = args.repo or os.environ.get("GITHUB_REPOSITORY") or ""
    token: str = os.environ.get("GITHUB_TOKEN") or ""
    if repo.strip() == "" or token.strip() == "":
        logger.warning(
            "Model alert tripped and GITHUB_REPOSITORY or GITHUB_TOKEN is unset. "
            "No issue was opened."
        )
        return {"status": str(decision["reason"]), "action": "not_opened"}
    result: dict[str, Any] = sync_alert_issue(decision, GithubIssueClient(repo, token))
    logger.info("Model alert issue action=%s number=%s.", result["action"], result["number"])
    return {"status": str(decision["reason"]), "action": str(result["action"])}


def _emit_alert_status(status: Mapping[str, Any]) -> None:
    """Record the alert status. The issue body and the token stay out of the file."""
    directory: Path | None = status_directory()
    if directory is None:
        return
    write_step_status(
        directory,
        "model_alert",
        {"status": status.get("status"), "action": status.get("action")},
    )


def _load_games(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    """Read decided games from JSON, an archive directory, or the git branch."""
    if args.games_json:
        payload: object = json.loads(Path(args.games_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--games-json must be a list.")
        return [item for item in payload if isinstance(item, dict)]
    site_data: Path | None = Path(args.site_data) if args.site_data else None
    if args.archive_dir:
        return finished_market_games(Path(args.archive_dir), site_data)
    if not args.repo:
        raise ValueError("The alert needs --repo, --archive-dir, or --games-json.")
    with open_archive_readonly(Path(args.repo), remote=args.remote, branch=args.branch) as archive:
        return finished_market_games(archive, site_data)


def _usable_row(game: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a clipped row, or None when the game cannot enter log loss."""
    label: object = game.get("label")
    if isinstance(label, bool) or label not in (0, 1):
        return None
    model_probability: float | None = clip_probability(_as_float(game.get("model_probability")))
    market_probability: float | None = clip_probability(_as_float(game.get("market_probability")))
    if model_probability is None or market_probability is None:
        return None
    season: object = game.get("season")
    week: object = game.get("week")
    return {
        "season": season,
        "week": week,
        "label": int(label),
        "model_probability": model_probability,
        "market_probability": market_probability,
    }


def _as_float(value: object) -> float | None:
    """Return a float, or None for a blank or non-numeric cell."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _setting_int(explicit: int | None, env_name: str, default: int) -> int:
    """Prefer the CLI flag, then the environment, then the default."""
    if explicit is not None:
        return int(explicit)
    raw: str | None = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _setting_float(explicit: float | None, env_name: str, default: float) -> float:
    """Prefer the CLI flag, then the environment, then the default."""
    if explicit is not None:
        return float(explicit)
    raw: str | None = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw.strip())


def _format_optional(value: object) -> str:
    """Render a gap for the log. Missing stays ``n/a``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
