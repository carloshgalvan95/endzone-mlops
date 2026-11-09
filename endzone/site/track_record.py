"""Weekly track record from the archived public slates.

The numbers come from the files on ``published-slates``. Prediction fields
come from the earliest archived copy of each game. Final scores come from
the newest copy, which is the site file after the results overlay when that
file exists. A later rescore is not used. This module writes
``site/data/track_record.json`` and does not modify a week file.

If the archive cannot be read, the written document says so and contains
no week rows. A corrupt archived week fails instead of publishing a partial
record. See ADR-008.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from endzone.site.archive import ARCHIVE_BRANCH, ArchiveError, open_archive_readonly
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

PROBABILITY_CLIP: float = 1e-6
SMALL_SAMPLE_DECIDED: int = 48
METHOD_NAME: str = "ADR-008"
METHOD_URL: str = (
    "https://github.com/carloshgalvan95/endzone-mlops/blob/main/docs/adr/"
    "ADR-008-weekly-track-record.md"
)
TRACK_RECORD_NAME: str = "track_record.json"
_WEEK_FILE: re.Pattern[str] = re.compile(r"^(?P<season>\d+)-w(?P<week>\d+)\.json$")
_WEEK_DIR: re.Pattern[str] = re.compile(r"^(?P<season>\d+)-w(?P<week>\d+)$")
_GENERATED_AT: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ASSET_URL: re.Pattern[str] = re.compile(
    r'(?P<prefix>(?:href|src)=")(?P<url>assets/[A-Za-z0-9._/-]+)(?P<suffix>")'
)
_COUNT_FIELDS: tuple[str, ...] = (
    "games",
    "finished",
    "unfinished",
    "ties",
    "decided",
    "pick_hits",
    "pick_misses",
    "pick_unavailable",
    "tie_hits",
    "market_games",
    "market_missing",
    "score_games",
    "score_missing",
    "prediction_games_from_archive",
    "prediction_games_from_this_build",
)


class TrackRecordError(Exception):
    """The track record could not be computed without misleading the page."""


def rounded_percent(probability: float) -> int:
    """Round a home win probability to a percent, half away from zero.

    This matches the page, which uses JavaScript ``Math.round``. Python's
    ``round`` would turn 50.5 into 50 and call a home pick a tie.
    """
    return int(math.floor(float(probability) * 100.0 + 0.5))


def pick_side(model_home_wp: float | None) -> str | None:
    """Return ``home``, ``away``, or ``tie`` for the frozen model pick.

    The percent is rounded. 50 is a tie pick, above 50 is home, and below
    50 is away. A missing or out-of-range probability has no pick.
    """
    if isinstance(model_home_wp, bool) or model_home_wp is None:
        return None
    try:
        probability: float = float(model_home_wp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    percent: int = rounded_percent(probability)
    if percent == 50:
        return "tie"
    if percent > 50:
        return "home"
    return "away"


def result_side(status: object, home_score: object, away_score: object) -> str | None:
    """Return ``home``, ``away``, or ``tie`` when the game is final.

    A game counts only when ``status`` is ``final`` and both scores are
    non-negative integers. Equal scores are a tie. Anything else is
    unfinished, including a final label with a missing score.
    """
    if status != "final":
        return None
    home: int | None = _score_int(home_score)
    away: int | None = _score_int(away_score)
    if home is None or away is None:
        return None
    if home == away:
        return "tie"
    if home > away:
        return "home"
    return "away"


def clip_probability(probability: float | None) -> float | None:
    """Clip a probability into ``[clip, 1 - clip]``.

    ``None`` when the value is missing, non-finite, or outside ``[0, 1]``.
    A stored 0 or 1 is clipped so the log loss stays finite.
    """
    if isinstance(probability, bool) or probability is None:
        return None
    try:
        value: float = float(probability)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return min(max(value, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)


def binary_log_loss(labels: list[int], probabilities: list[float]) -> float | None:
    """Mean binary log loss. Probabilities must already be clipped.

    Empty input is ``None``. The log is natural, the same base as the
    training metric. A label must be 0 or 1.

    Raises:
        ValueError: The two sequences differ in length, or a label is not 0 or 1.
    """
    if len(labels) != len(probabilities):
        raise ValueError("Log loss labels and probabilities must have the same length.")
    if not labels:
        return None
    total: float = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in (0, 1):
            raise ValueError("Log loss labels must be 0 or 1.")
        if not 0.0 < probability < 1.0:
            raise ValueError("Log loss probabilities must be clipped away from 0 and 1.")
        total += -(label * math.log(probability) + (1 - label) * math.log(1.0 - probability))
    return total / len(labels)


def sample_label(weeks_with_finals: int, decided: int) -> str | None:
    """Label a season total that is too small to read as a result.

    One week with finals is ``one_week``. More weeks, but fewer than
    ``SMALL_SAMPLE_DECIDED`` decided games, is ``small``. No finals is
    ``no_finals``. A larger sample has no label.
    """
    if weeks_with_finals <= 0:
        return "no_finals"
    if weeks_with_finals == 1:
        return "one_week"
    if decided < SMALL_SAMPLE_DECIDED:
        return "small"
    return None


def metrics_from_games(games: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Count hits, log loss, and score error for one list of games.

    Each game uses ``model_home_wp``, ``market_home_wp``, ``status``,
    ``home_score``, ``away_score``, ``estimated_margin``, and
    ``estimated_total``. ``prediction_from_archive`` defaults to true.

    A pick hit matches the game card: the rounded pick equals the result,
    including a tie pick on a tied score. Log loss uses finished games that
    have a winner and both probabilities. Ties are left out of log loss.
    Margin error and total error use finished games that published both
    estimates, including ties. Unfinished games stay out of every rate.
    """
    return dict(_measure(games)["metrics"])


def _measure(games: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return public counts plus the terms a season total has to re-mean."""
    counts: dict[str, int] = {name: 0 for name in _COUNT_FIELDS}
    labels: list[int] = []
    model_probs: list[float] = []
    market_probs: list[float] = []
    margin_errors: list[float] = []
    total_errors: list[float] = []
    for game in games:
        counts["games"] += 1
        if game.get("prediction_from_archive", True) is True:
            counts["prediction_games_from_archive"] += 1
        else:
            counts["prediction_games_from_this_build"] += 1
        result: str | None = result_side(
            game.get("status"), game.get("home_score"), game.get("away_score")
        )
        if result is None:
            counts["unfinished"] += 1
            continue
        counts["finished"] += 1
        _count_pick(counts, pick_side(_finite_float(game.get("model_home_wp"))), result)
        if result == "tie":
            counts["ties"] += 1
        else:
            counts["decided"] += 1
            _collect_market(game, result, counts, labels, model_probs, market_probs)
        _collect_score(game, counts, margin_errors, total_errors)
    metrics: dict[str, Any] = dict(counts)
    metrics["model_log_loss"] = _round6(binary_log_loss(labels, model_probs))
    metrics["market_log_loss"] = _round6(binary_log_loss(labels, market_probs))
    metrics["margin_mae"] = _round6(_mean(margin_errors))
    metrics["total_mae"] = _round6(_mean(total_errors))
    return {
        "metrics": metrics,
        "labels": labels,
        "model_probs": model_probs,
        "market_probs": market_probs,
        "margin_errors": margin_errors,
        "total_errors": total_errors,
    }


def unavailable_document(generated_at: str) -> dict[str, Any]:
    """A track record that refuses to invent weeks.

    The page shows this when the archive branch cannot be read. It has no
    season rows and no rates.
    """
    stamp: str = _require_generated_at(generated_at)
    return _document(available=False, reason="archive_unreachable", generated_at=stamp, seasons=[])


def build_track_record(
    archive_dir: Path | None,
    site_data: Path | None,
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Build the track record from an archive directory and optional site files.

    ``archive_dir`` is read only. When it is missing, the result is an
    unavailable document and ``site_data`` is not used as a substitute.
    A week that exists only under ``site_data`` is a first publish. A game
    that already has an archived copy keeps that copy's probabilities.

    Raises:
        TrackRecordError: An archived or site week file cannot be read, or
            it disagrees with its season and week.
    """
    stamp: str = _require_generated_at(generated_at)
    if archive_dir is None or not Path(archive_dir).is_dir():
        return unavailable_document(stamp)
    archive: Path = Path(archive_dir)
    site_docs: dict[tuple[int, int], dict[str, Any]] = _site_weeks(site_data)
    keys: set[tuple[int, int]] = _archived_keys(archive)
    keys.update(site_docs)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for season, week in sorted(keys):
        snapshots: list[dict[str, Any]] = _week_snapshots(archive, season, week)
        site_doc: dict[str, Any] | None = site_docs.get((season, week))
        if not snapshots and site_doc is None:
            continue
        games: list[dict[str, Any]] = _assemble_week(snapshots, site_doc)
        if not games:
            continue
        grouped.setdefault(season, []).append(_week_row(season, week, games))
    seasons: list[dict[str, Any]] = []
    for season, weeks in grouped.items():
        seasons.append(
            {
                "season": season,
                "weeks": [_public_week(week) for week in weeks],
                "summary": _summary(weeks),
            }
        )
    return _document(available=True, reason=None, generated_at=stamp, seasons=seasons)


def finished_market_games(
    archive_dir: Path | None,
    site_data: Path | None,
) -> list[dict[str, Any]]:
    """Decided games that enter the track-record log loss, oldest week first.

    Ties and games with a missing or unusable price are omitted. Probabilities
    are clipped the same way as the track record. A missing archive returns
    an empty list and does not fall back to site files. A corrupt week file
    raises ``TrackRecordError``.

    Each row has ``season``, ``week``, ``game_id``, ``label`` (1 for a home
    win), ``model_probability``, and ``market_probability``.
    """
    if archive_dir is None or not Path(archive_dir).is_dir():
        return []
    archive: Path = Path(archive_dir)
    site_docs: dict[tuple[int, int], dict[str, Any]] = _site_weeks(site_data)
    keys: set[tuple[int, int]] = _archived_keys(archive)
    keys.update(site_docs)
    rows: list[dict[str, Any]] = []
    for season, week in sorted(keys):
        snapshots: list[dict[str, Any]] = _week_snapshots(archive, season, week)
        site_doc: dict[str, Any] | None = site_docs.get((season, week))
        if not snapshots and site_doc is None:
            continue
        games: list[dict[str, Any]] = _assemble_week(snapshots, site_doc)
        for game in games:
            result: str | None = result_side(
                game.get("status"),
                game.get("home_score"),
                game.get("away_score"),
            )
            if result not in {"home", "away"}:
                continue
            model_probability: float | None = clip_probability(
                _finite_float(game.get("model_home_wp"))
            )
            market_probability: float | None = clip_probability(
                _finite_float(game.get("market_home_wp"))
            )
            if model_probability is None or market_probability is None:
                continue
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": str(game.get("game_id", "")),
                    "label": 1 if result == "home" else 0,
                    "model_probability": model_probability,
                    "market_probability": market_probability,
                }
            )
    return rows


def assert_derived_output(
    output: Path,
    archive_dir: Path | None,
    site_data: Path | None,
) -> None:
    """Refuse a path that would replace a published slate.

    ``site/data/track_record.json`` is allowed. A week file name, ``latest.json``,
    anything inside the archive, and anything inside ``predictions/`` are not.

    Raises:
        TrackRecordError: The path is a published slate or sits inside one.
    """
    path: Path = Path(output)
    if path.name == "latest.json" or _WEEK_FILE.match(path.name):
        raise TrackRecordError(f"Refusing to write {path.name}. That name is a published slate.")
    if archive_dir is not None and _is_inside(path, Path(archive_dir)):
        raise TrackRecordError("Refusing to write inside the archive.")
    if site_data is not None and _is_inside(path, Path(site_data) / "predictions"):
        raise TrackRecordError("Refusing to write inside published week files.")


def write_track_record(document: Mapping[str, Any], output: Path) -> None:
    """Write one derived JSON file as UTF-8 with ``\\n`` newlines.

    The write is the output path only. A crash leaves the previous output
    in place until the temp file replaces it.

    Raises:
        TrackRecordError: The output directory cannot be written.
    """
    path: Path = Path(output)
    payload: str = json.dumps(dict(document), indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError as exc:
        raise TrackRecordError(f"Could not write {path.name}: {exc}.") from exc
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def stamp_asset_urls(index_path: Path) -> dict[str, str]:
    """Append a content hash to local ``assets/`` URLs in one HTML file.

    GitHub Pages caches files for 10 minutes and the URLs have no version.
    The hash changes when the file bytes change, so a new HTML file loads
    the new CSS and JavaScript. URLs that already have a query are left
    alone. Missing files are left alone.

    Returns:
        Map of asset URL to the 12 character hash that was written.

    Raises:
        TrackRecordError: The HTML file cannot be read or written.
    """
    path: Path = Path(index_path)
    try:
        html: str = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrackRecordError(f"Could not read {path.name}: {exc}.") from exc
    found: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        url: str = match.group("url")
        asset: Path = path.parent / url
        if not asset.is_file():
            return match.group(0)
        digest: str = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
        found[url] = digest
        return f"{match.group('prefix')}{url}?v={digest}{match.group('suffix')}"

    updated: str = _ASSET_URL.sub(replace, html)
    if updated == html:
        return found
    try:
        path.write_text(updated, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise TrackRecordError(f"Could not write {path.name}: {exc}.") from exc
    return found


def build_parser() -> argparse.ArgumentParser:
    """Build the track record command parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Write the weekly track record from archived slates. "
            "Does not fit a model and does not edit a week file."
        )
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=ARCHIVE_BRANCH)
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--site-data", default="site/data")
    parser.add_argument("--output", default=f"site/data/{TRACK_RECORD_NAME}")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--stamp-html", default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Write the track record, then optionally stamp asset URLs.

    A missing archive writes an unavailable document and exits 0. A corrupt
    week exits 1 and deletes a stale ``track_record.json`` so the deploy
    cannot keep an old record.

    Raises:
        SystemExit: The archive was readable but a week file was not.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    output: Path = Path(args.output)
    try:
        document: dict[str, Any] = _produce(args, output)
    except TrackRecordError as exc:
        _remove_stale_record(output)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if args.stamp_html:
        stamp_asset_urls(Path(args.stamp_html))
    _print_result(document, output)
    return document


def _produce(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    """Read the archive, or record that it was unreachable, then write."""
    stamp: str = args.generated_at or _utc_stamp(None)
    site_data: Path | None = Path(args.site_data) if args.site_data else None
    if args.archive_dir:
        archive: Path = Path(args.archive_dir)
        assert_derived_output(output, archive if archive.is_dir() else None, site_data)
        if archive.is_dir():
            document: dict[str, Any] = build_track_record(archive, site_data, generated_at=stamp)
        else:
            logger.info("Archive directory %s is missing. No week record will be written.", archive)
            document = unavailable_document(stamp)
        write_track_record(document, output)
        return document
    assert_derived_output(output, None, site_data)
    try:
        with open_archive_readonly(
            Path(args.repo), remote=args.remote, branch=args.branch
        ) as archive:
            assert_derived_output(output, archive, site_data)
            document = build_track_record(archive, site_data, generated_at=stamp)
    except ArchiveError as exc:
        logger.info("Published-slate archive is unreachable: %s", exc)
        document = unavailable_document(stamp)
    write_track_record(document, output)
    return document


def _print_result(document: Mapping[str, Any], output: Path) -> None:
    """Write one status line for the workflow log."""
    if not document.get("available"):
        print(f"Archive is unreachable. Wrote {output.name} with no week results.")
        return
    seasons: object = document.get("seasons")
    week_count: int = 0
    if isinstance(seasons, list):
        for season in seasons:
            if isinstance(season, dict) and isinstance(season.get("weeks"), list):
                week_count += len(season["weeks"])
    print(f"Wrote track record for {week_count} week(s) to {output.name}.")


def _remove_stale_record(output: Path) -> None:
    """Delete only a derived track record. Never a week file."""
    if output.name != TRACK_RECORD_NAME or not output.is_file():
        return
    try:
        output.unlink()
    except OSError as exc:
        logger.warning("Could not remove stale %s: %s", output.name, exc)


def _document(
    *,
    available: bool,
    reason: str | None,
    generated_at: str,
    seasons: list[dict[str, Any]],
) -> dict[str, Any]:
    """One public track record document."""
    return {
        "available": available,
        "reason": reason,
        "generated_at": generated_at,
        "method": METHOD_NAME,
        "method_url": METHOD_URL,
        "probability_clip": PROBABILITY_CLIP,
        "small_sample_decided": SMALL_SAMPLE_DECIDED,
        "seasons": seasons,
    }


def _count_pick(counts: dict[str, int], pick: str | None, result: str) -> None:
    """Count one finished game as a hit, a miss, or a missing pick."""
    if pick is None:
        counts["pick_unavailable"] += 1
        return
    if pick == result:
        counts["pick_hits"] += 1
        if result == "tie":
            counts["tie_hits"] += 1
        return
    counts["pick_misses"] += 1


def _collect_market(
    game: Mapping[str, Any],
    result: str,
    counts: dict[str, int],
    labels: list[int],
    model_probs: list[float],
    market_probs: list[float],
) -> None:
    """Keep one decided game when both probabilities can be clipped."""
    model_probability: float | None = clip_probability(_finite_float(game.get("model_home_wp")))
    market_probability: float | None = clip_probability(_finite_float(game.get("market_home_wp")))
    if model_probability is None or market_probability is None:
        counts["market_missing"] += 1
        return
    counts["market_games"] += 1
    labels.append(1 if result == "home" else 0)
    model_probs.append(model_probability)
    market_probs.append(market_probability)


def _collect_score(
    game: Mapping[str, Any],
    counts: dict[str, int],
    margin_errors: list[float],
    total_errors: list[float],
) -> None:
    """Keep margin and total error when both estimates were published."""
    margin: float | None = _finite_float(game.get("estimated_margin"))
    total: float | None = _finite_float(game.get("estimated_total"))
    home: int | None = _score_int(game.get("home_score"))
    away: int | None = _score_int(game.get("away_score"))
    if margin is None or total is None or home is None or away is None:
        counts["score_missing"] += 1
        return
    counts["score_games"] += 1
    margin_errors.append(abs(margin - (home - away)))
    total_errors.append(abs(total - (home + away)))


def _week_row(season: int, week: int, games: list[Mapping[str, Any]]) -> dict[str, Any]:
    """One week row plus the raw terms used to pool the season."""
    measured: dict[str, Any] = _measure(games)
    metrics: dict[str, Any] = dict(measured["metrics"])
    finished: int = int(metrics["finished"])
    unfinished: int = int(metrics["unfinished"])
    if finished == 0:
        state: str = "no_finals"
    elif unfinished > 0:
        state = "partial"
    else:
        state = "complete"
    return {
        "season": season,
        "week": week,
        "state": state,
        **metrics,
        "_labels": measured["labels"],
        "_model_probs": measured["model_probs"],
        "_market_probs": measured["market_probs"],
        "_margin_errors": measured["margin_errors"],
        "_total_errors": measured["total_errors"],
    }


def _public_week(week: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the raw terms before the document is written."""
    return {key: value for key, value in week.items() if not str(key).startswith("_")}


def _summary(weeks: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool week rows. Means are over games, not over weekly means."""
    totals: dict[str, int] = {name: 0 for name in _COUNT_FIELDS}
    for week in weeks:
        for name in _COUNT_FIELDS:
            totals[name] += int(week[name])
    labels: list[int] = [label for week in weeks for label in week["_labels"]]
    model_probs: list[float] = [value for week in weeks for value in week["_model_probs"]]
    market_probs: list[float] = [value for week in weeks for value in week["_market_probs"]]
    margin_errors: list[float] = [value for week in weeks for value in week["_margin_errors"]]
    total_errors: list[float] = [value for week in weeks for value in week["_total_errors"]]
    if totals["market_games"] != len(labels) or totals["score_games"] != len(margin_errors):
        raise TrackRecordError("Pooled counts do not match the stored game terms.")
    weeks_with_finals: int = sum(1 for week in weeks if int(week["finished"]) > 0)
    summary: dict[str, Any] = {
        "weeks": len(weeks),
        "weeks_with_finals": weeks_with_finals,
        "sample_label": sample_label(weeks_with_finals, totals["decided"]),
    }
    summary.update(totals)
    summary["model_log_loss"] = _round6(binary_log_loss(labels, model_probs))
    summary["market_log_loss"] = _round6(binary_log_loss(labels, market_probs))
    summary["margin_mae"] = _round6(_mean(margin_errors))
    summary["total_mae"] = _round6(_mean(total_errors))
    return summary


def _assemble_week(
    snapshots: list[dict[str, Any]],
    site_doc: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Join earliest probabilities to the newest final scores.

    Snapshots are oldest first. The site file, when present, is the newest
    score source and is not a probability source for a game the archive
    already has.
    """
    archived: list[dict[str, dict[str, Any]]] = [_index_games(document) for document in snapshots]
    site_index: dict[str, dict[str, Any]] = _index_games(site_doc) if site_doc is not None else {}
    order: list[str] = []
    seen: set[str] = set()
    for indexed in [*archived, site_index]:
        for game_id in indexed:
            if game_id not in seen:
                seen.add(game_id)
                order.append(game_id)
    score_sources: list[dict[str, dict[str, Any]]] = []
    if site_doc is not None:
        score_sources.append(site_index)
    score_sources.extend(reversed(archived))
    games: list[dict[str, Any]] = []
    for game_id in order:
        prediction: dict[str, Any] | None = None
        from_archive: bool = False
        for indexed in archived:
            if game_id in indexed:
                prediction = indexed[game_id]
                from_archive = True
                break
        if prediction is None and game_id in site_index:
            prediction = site_index[game_id]
        if prediction is None:
            continue
        score_row: dict[str, Any] = prediction
        for indexed in score_sources:
            if game_id in indexed:
                score_row = indexed[game_id]
                break
        games.append(
            {
                "game_id": game_id,
                "model_home_wp": prediction.get("model_home_wp"),
                "market_home_wp": prediction.get("market_home_wp"),
                "estimated_margin": prediction.get("estimated_margin"),
                "estimated_total": prediction.get("estimated_total"),
                "status": score_row.get("status"),
                "home_score": score_row.get("home_score"),
                "away_score": score_row.get("away_score"),
                "prediction_from_archive": from_archive,
            }
        )
    return games


def _week_snapshots(archive: Path, season: int, week: int) -> list[dict[str, Any]]:
    """History files oldest first, then the tip file."""
    snapshots: list[dict[str, Any]] = []
    history: Path = archive / "history" / f"{season}-w{week}"
    if history.is_dir():
        for path in sorted(item for item in history.glob("*.json") if item.is_file()):
            document: dict[str, Any] | None = _read_slate(path, season, week)
            if document is not None:
                snapshots.append(document)
    tip: Path = archive / "predictions" / f"{season}-w{week}.json"
    if tip.is_file():
        document = _read_slate(tip, season, week)
        if document is not None:
            snapshots.append(document)
    return snapshots


def _archived_keys(archive: Path) -> set[tuple[int, int]]:
    """Season and week pairs present as a tip file or a history directory."""
    keys: set[tuple[int, int]] = set()
    predictions: Path = archive / "predictions"
    if predictions.is_dir():
        for path in predictions.glob("*.json"):
            match: re.Match[str] | None = _WEEK_FILE.match(path.name)
            if match is not None and path.is_file():
                keys.add((int(match.group("season")), int(match.group("week"))))
    history: Path = archive / "history"
    if history.is_dir():
        for path in history.iterdir():
            if not path.is_dir() or path.name == "latest":
                continue
            match = _WEEK_DIR.match(path.name)
            if match is not None:
                keys.add((int(match.group("season")), int(match.group("week"))))
    return keys


def _site_weeks(site_data: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    """Week files under ``site/data/predictions``. Example slates are skipped."""
    found: dict[tuple[int, int], dict[str, Any]] = {}
    if site_data is None:
        return found
    predictions: Path = Path(site_data) / "predictions"
    if not predictions.is_dir():
        return found
    for path in sorted(item for item in predictions.glob("*.json") if item.is_file()):
        match: re.Match[str] | None = _WEEK_FILE.match(path.name)
        if match is None:
            continue
        season: int = int(match.group("season"))
        week: int = int(match.group("week"))
        document: dict[str, Any] | None = _read_slate(path, season, week)
        if document is not None:
            found[(season, week)] = document
    return found


def _read_slate(path: Path, season: int, week: int) -> dict[str, Any] | None:
    """Read one slate. An example file is skipped. A bad file fails closed."""
    document: dict[str, Any] = _read_json(path)
    if document.get("sample") is True:
        return None
    file_season: int | None = _plain_int(document.get("season"))
    file_week: int | None = _plain_int(document.get("week"))
    if file_season != season or file_week != week:
        raise TrackRecordError(
            f"{path.name} says season {file_season} week {file_week}, "
            f"expected {season} week {week}."
        )
    if not isinstance(document.get("games"), list):
        raise TrackRecordError(f"{path.name} has no games list.")
    return document


def _read_json(path: Path) -> dict[str, Any]:
    """Decode one UTF-8 JSON object."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackRecordError(f"Could not read {path.name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise TrackRecordError(f"{path.name} is not a JSON object.")
    return payload


def _index_games(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index games by id. Two different rows for one id fail closed."""
    games: object = document.get("games")
    if not isinstance(games, list):
        raise TrackRecordError("A slate has no games list.")
    indexed: dict[str, dict[str, Any]] = {}
    for game in games:
        if not isinstance(game, dict):
            raise TrackRecordError("A slate game is not an object.")
        game_id: object = game.get("game_id")
        if not isinstance(game_id, str) or game_id == "":
            raise TrackRecordError("A slate game has no game_id.")
        previous: dict[str, Any] | None = indexed.get(game_id)
        if previous is not None and previous != game:
            raise TrackRecordError(f"Conflicting rows for {game_id}.")
        indexed[game_id] = game
    return indexed


def _finite_float(value: object) -> float | None:
    """Return a finite float, or None for a blank or non-numeric cell."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number: float = float(value)
        if math.isfinite(number):
            return number
    return None


def _score_int(value: object) -> int | None:
    """Return a non-negative integer score. Booleans are not scores."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _plain_int(value: object) -> int | None:
    """Return an int. Booleans are not ints here."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None when there is nothing to average."""
    if not values:
        return None
    return sum(values) / len(values)


def _round6(value: float | None) -> float | None:
    """Round a metric to 6 decimal places for a stable JSON file."""
    if value is None:
        return None
    return float(f"{value:.6f}")


def _require_generated_at(generated_at: str) -> str:
    """Require an ISO UTC stamp with a Z suffix and no fractional seconds."""
    if _GENERATED_AT.fullmatch(generated_at) is None:
        raise TrackRecordError("generated_at must look like 2026-11-09T15:04:00Z.")
    return generated_at


def _utc_stamp(clock: datetime | None) -> str:
    """Return the current UTC minute stamp used on week files."""
    moment: datetime = clock if clock is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_inside(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or a file inside it."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
