"""Job summary for one Pages or Results run.

The command reads the slate, the track record, and the step status files
those jobs already wrote. It appends markdown to ``GITHUB_STEP_SUMMARY``.
It does not fit a model, does not call Databricks, and does not read
``.env``. A missing file stays a visible gap. The command returns 0.

``python -m endzone.site.run_summary``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from endzone.utils.step_status import read_status_directory

_ROW_LABELS: tuple[tuple[str, str], ...] = (
    ("season", "Season"),
    ("week", "Week"),
    ("games", "Games"),
    ("model_version", "Model version"),
    ("alias", "Alias"),
    ("data_digest", "Data digest"),
    ("score_gate", "Score gate"),
    ("results", "Results"),
    ("model_alert", "Model alert"),
    ("archive", "Archive"),
    ("margin_model_version", "Margin model version"),
    ("total_model_version", "Total model version"),
)
_OPTIONAL_ROWS: frozenset[str] = frozenset({"margin_model_version", "total_model_version"})
_FALLBACK_TEXT: dict[str, str] = {
    "alias_missing": "alias prod is missing",
    "registry_unreadable": "registry could not be read",
    "loaded_model_could_not_score": "loaded model could not score this slate",
    "weekly_fit": "weekly fit",
    "score_failed": "score failed",
}
_HOST: re.Pattern[str] = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:cloud\.databricks\.com|azuredatabricks\.net|gcp\.databricks\.com)\b",
    re.IGNORECASE,
)
_URL: re.Pattern[str] = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)
_HTTP_PATH: re.Pattern[str] = re.compile(r"/sql/(?:1\.0|protocolv1)/[^\s)>\]\"']*", re.IGNORECASE)
_TOKEN: re.Pattern[str] = re.compile(r"\bdapi[a-z0-9-]{8,}\b", re.IGNORECASE)
_ENV_ASSIGN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|HOST|HTTP_PATH|CREDENTIAL|API_KEY|URI)\b\s*=\s*\S+",
    re.IGNORECASE,
)
_SENSITIVE_KEY: re.Pattern[str] = re.compile(
    r"(TOKEN|SECRET|PASSWORD|HOST|HTTP_PATH|CREDENTIAL|API_KEY|URI)",
    re.IGNORECASE,
)
_REDACTED: str = "[redacted]"


def redact_text(text: str) -> str:
    """Remove hosts, HTTP paths, tokens, and sensitive environment values.

    The check is on the rendered text. A slate field that happens to carry
    one of those strings is replaced. The function does not print the
    environment.
    """
    cleaned: str = text.replace("\r", " ").replace("\n", " ")
    cleaned = _URL.sub(_REDACTED, cleaned)
    cleaned = _HOST.sub(_REDACTED, cleaned)
    cleaned = _HTTP_PATH.sub(_REDACTED, cleaned)
    cleaned = _TOKEN.sub(_REDACTED, cleaned)
    cleaned = _ENV_ASSIGN.sub(_REDACTED, cleaned)
    for secret in _sensitive_env_values():
        if secret in cleaned:
            cleaned = cleaned.replace(secret, _REDACTED)
    return " ".join(cleaned.split())


def collect_summary(
    slate: Mapping[str, Any] | None,
    track_record: Mapping[str, Any] | None,
    statuses: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    """Build the display rows for one run.

    ``statuses`` is keyed by step file stem (``score_gate``, ``score``,
    ``results``, ``model_alert``, ``archive``). Missing inputs stay
    explicit. Margin and total versions are included only when the slate
    carries those keys.

    Returns:
        Row id to display text. Optional rows are absent when the slate
        has no such key.
    """
    document: Mapping[str, Any] = slate if isinstance(slate, Mapping) else {}
    record: Mapping[str, Any] = track_record if isinstance(track_record, Mapping) else {}
    score: Mapping[str, str] = statuses.get("score", {})
    gate: Mapping[str, str] = statuses.get("score_gate", {})
    when: dict[str, str] = _season_week(document, record, score, gate)
    rows: dict[str, str] = {
        "season": when["season"],
        "week": when["week"],
        "games": _game_count(document, score),
        "model_version": _model_version(document, score),
        "alias": _alias_text(document, score, gate),
        "data_digest": _data_digest(document, record, score),
        "score_gate": _score_gate_text(gate),
        "results": _results_text(statuses.get("results", {})),
        "model_alert": _alert_text(statuses.get("model_alert", {})),
        "archive": _archive_text(statuses.get("archive", {})),
    }
    rows.update(_point_versions(document))
    return rows


def format_run_summary(rows: Mapping[str, str]) -> str:
    """Render ``rows`` as a GitHub step summary table.

    Every cell is redacted. Optional point-model rows are skipped when
    they are absent.
    """
    lines: list[str] = ["## Run summary", "", "| Field | Value |", "| --- | --- |"]
    for key, label in _ROW_LABELS:
        if key not in rows:
            if key in _OPTIONAL_ROWS:
                continue
            value: str = "unavailable"
        else:
            value = rows[key]
        cell: str = _table_cell(redact_text(value))
        if cell == "":
            cell = "unavailable"
        lines.append(f"| {label} | {cell} |")
    lines.append("")
    return "\n".join(lines)


def load_run_facts(site_data: Path, status_dirs: list[Path]) -> dict[str, str]:
    """Read the slate, the track record, and any status directories.

    A missing or unreadable file is skipped. Later status directories
    replace the same step stem.

    Returns:
        The display rows from ``collect_summary``.
    """
    root: Path = Path(site_data)
    statuses: dict[str, dict[str, str]] = {}
    for directory in status_dirs:
        found: dict[str, dict[str, str]] = read_status_directory(Path(directory))
        statuses.update(found)
    return collect_summary(
        _read_json(root / "latest.json"),
        _read_json(root / "track_record.json"),
        statuses,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the run-summary parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Write the Pages or Results job summary from the slate, the track "
            "record, and step status files. Does not fit a model."
        )
    )
    parser.add_argument(
        "--site-data",
        type=str,
        default="site/data",
        help="Directory that holds latest.json and track_record.json.",
    )
    parser.add_argument(
        "--status-dir",
        action="append",
        default=None,
        help="Directory of key=value step files. Repeat for more than one.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Append markdown here. The default is GITHUB_STEP_SUMMARY, else stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the summary and return 0.

    A missing input or a write error still returns 0. The deploy must not
    fail because this step failed. The workflow also sets continue-on-error.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    directories: list[Path] = [Path(item) for item in (args.status_dir or [])]
    try:
        rows: dict[str, str] = load_run_facts(Path(args.site_data), directories)
        markdown: str = format_run_summary(rows)
    except OSError:
        markdown = "## Run summary\n\nThe summary could not be read.\n"
    _write_summary(markdown, args.summary_file)
    return 0


def _write_summary(markdown: str, explicit: str | None) -> None:
    """Append the summary. Stdout is the local fallback."""
    destination: str | None = explicit or os.environ.get("GITHUB_STEP_SUMMARY")
    payload: str = markdown if markdown.endswith("\n") else markdown + "\n"
    if destination is None or destination.strip() == "":
        sys.stdout.write(payload)
        return
    try:
        with Path(destination).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except OSError:
        sys.stdout.write(payload)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Load one JSON object, or None when the file is missing or unusable."""
    if not path.is_file():
        return None
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _season_week(
    slate: Mapping[str, Any],
    track_record: Mapping[str, Any],
    score: Mapping[str, str],
    gate: Mapping[str, str],
) -> dict[str, str]:
    """Pick season and week from the slate, then the score, the gate, the record."""
    season: str = _int_text(slate.get("season"))
    week: str = _int_text(slate.get("week"))
    if season == "unavailable":
        season = _int_text(score.get("season"))
    if week == "unavailable":
        week = _int_text(score.get("week"))
    if season == "unavailable":
        season = _int_text(gate.get("season"))
    if week == "unavailable":
        week = _int_text(gate.get("week"))
    tracked: dict[str, str] | None = _last_tracked_week(track_record)
    if tracked is not None:
        if season == "unavailable":
            season = tracked["season"]
        if week == "unavailable":
            week = tracked["week"]
    return {"season": season, "week": week}


def _last_tracked_week(track_record: Mapping[str, Any]) -> dict[str, str] | None:
    """Season and week of the last row in the track record, if both are ints."""
    seasons: object = track_record.get("seasons")
    if not isinstance(seasons, list) or len(seasons) == 0:
        return None
    last_season: object = seasons[-1]
    if not isinstance(last_season, Mapping):
        return None
    season: str = _int_text(last_season.get("season"))
    weeks: object = last_season.get("weeks")
    if not isinstance(weeks, list) or len(weeks) == 0:
        return None
    last_week: object = weeks[-1]
    if not isinstance(last_week, Mapping):
        return None
    week: str = _int_text(last_week.get("week"))
    if season == "unavailable" or week == "unavailable":
        return None
    return {"season": season, "week": week}


def _game_count(slate: Mapping[str, Any], score: Mapping[str, str]) -> str:
    """Count games on the slate, then the score status file."""
    games: object = slate.get("games")
    if isinstance(games, list):
        return str(len(games))
    counted: str = _int_text(score.get("games"))
    if counted != "unavailable":
        return counted
    return "unavailable"


def _model_version(slate: Mapping[str, Any], score: Mapping[str, str]) -> str:
    """Prefer the version the score step recorded, then the slate."""
    recorded: str = score.get("model_version", "").strip()
    if recorded != "":
        return recorded
    value: object = slate.get("model_version")
    if isinstance(value, str) and value.strip() != "":
        return value.strip()
    return "unavailable"


def _alias_text(
    slate: Mapping[str, Any],
    score: Mapping[str, str],
    gate: Mapping[str, str],
) -> str:
    """The alias that was loaded, or the fallback reason.

    A skipped gate is not a fallback. The slate's ``version@alias`` still
    describes the file that stayed published.
    """
    reason: str = score.get("fallback_reason", "").strip()
    source: str = score.get("model_source", "").strip()
    if reason != "":
        return "fallback: " + _FALLBACK_TEXT.get(reason, reason)
    version: str = _model_version(slate, score)
    if "@" in version and version != "unavailable":
        return version
    alias: str = score.get("model_alias", "").strip()
    if alias == "":
        slate_alias: object = slate.get("model_alias")
        if isinstance(slate_alias, str):
            alias = slate_alias.strip()
    if source == "registry" and alias != "" and version not in {"", "unavailable"}:
        return f"{version}@{alias}"
    if source == "refit":
        return "fallback: weekly fit"
    if gate.get("skip") == "true" and gate.get("reason") == "published":
        return "not scored"
    return "unavailable"


def _data_digest(
    slate: Mapping[str, Any],
    track_record: Mapping[str, Any],
    score: Mapping[str, str],
) -> str:
    """Digest from the score step, then the slate, then the track record."""
    recorded: str = score.get("data_digest", "").strip()
    if recorded != "":
        return recorded
    for source in (slate, track_record):
        value: object = source.get("data_digest")
        if isinstance(value, str) and value.strip() != "":
            return value.strip()
    return "unavailable"


def _score_gate_text(gate: Mapping[str, str]) -> str:
    """Say whether the score gate skipped a week that was already published."""
    if not gate:
        return "not run"
    skip: str = gate.get("skip", "")
    reason: str = gate.get("reason", "")
    if skip == "true" and reason == "published":
        return "skipped, week already published"
    if skip == "true":
        return "skipped"
    if skip == "false":
        return "not skipped"
    return "not run"


def _results_text(results: Mapping[str, str]) -> str:
    """Finals applied, unchanged, or not run."""
    if not results:
        return "not run"
    outcome: str = results.get("outcome", "").strip()
    if outcome in {"finals applied", "unchanged", "failed"}:
        return outcome
    updated: str = results.get("updated", "")
    if updated.isdigit() and int(updated) > 0:
        return "finals applied"
    if updated.isdigit():
        return "unchanged"
    return "not run"


def _alert_text(alert: Mapping[str, str]) -> str:
    """Model alert status. An action is included when one was recorded."""
    if not alert:
        return "not run"
    status: str = alert.get("status", "").strip() or "unavailable"
    action: str = alert.get("action", "").strip()
    if action != "" and action != "none":
        return f"{status} ({action})"
    return status


def _archive_text(archive: Mapping[str, str]) -> str:
    """Committed, nothing to commit, failed, or not run."""
    if not archive:
        return "not run"
    if archive.get("status") == "failed" or archive.get("failed") == "true":
        return "failed"
    if archive.get("committed") == "true" or archive.get("status") == "committed":
        return "committed"
    if archive.get("committed") == "false" or archive.get("status") == "nothing to commit":
        return "nothing to commit"
    return "not run"


def _point_versions(slate: Mapping[str, Any]) -> dict[str, str]:
    """Margin and total versions when the slate JSON has those keys."""
    score_model: object = slate.get("score_model")
    if not isinstance(score_model, Mapping):
        return {}
    found: dict[str, str] = {}
    for name, row_id in (
        ("margin", "margin_model_version"),
        ("total", "total_model_version"),
    ):
        target: object = score_model.get(name)
        if not isinstance(target, Mapping) or "model_version" not in target:
            continue
        version: object = target.get("model_version")
        if version is None:
            found[row_id] = "none"
        elif isinstance(version, str) and version.strip() != "":
            found[row_id] = version.strip()
        elif isinstance(version, int) and not isinstance(version, bool):
            found[row_id] = str(version)
        else:
            found[row_id] = "none"
    return found


def _int_text(value: object) -> str:
    """Render an int, or ``unavailable`` for a blank or non-numeric cell."""
    if isinstance(value, bool) or value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        text: str = value.strip()
        if text.isdigit():
            return str(int(text))
    return "unavailable"


def _table_cell(value: str) -> str:
    """Keep one cell from breaking the markdown table."""
    return value.replace("|", "/")


def _sensitive_env_values() -> list[str]:
    """Values of sensitive variables, longest first, so a short one cannot cut a long one.

    Names are matched. The values are used only as search needles. They are
    not written into the summary.
    """
    found: list[str] = []
    for key, value in os.environ.items():
        if _SENSITIVE_KEY.search(key) is None:
            continue
        if len(value) < 12:
            continue
        found.append(value)
    found.sort(key=len, reverse=True)
    return found


if __name__ == "__main__":
    raise SystemExit(main())
