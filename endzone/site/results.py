"""Overlay final nflverse scores onto published weekly prediction JSON.

Predictions stay frozen. This command does not fit a model, does not run
dbt, and does not read the warehouse. It starts from week files already on
disk (optionally restored from the live site) and attaches scores only when
nflverse has both of them.

``select_slate`` and ``load_upcoming_games`` drop finished games, so this
job never uses them.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from endzone.ingest.nflverse import NFLVERSE_GAMES_URL, download_parquet_with_validation
from endzone.modeling.weekly import (
    _WEEK_FILE,
    DEFAULT_OUTPUT_DIR,
    FINAL_GAME_STATUS,
    SCHEDULED_GAME_STATUS,
    _dumps,
    restore_published_predictions,
    scoreline_result,
    validate_prediction_document,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

_SCORE_COLUMNS: tuple[str, ...] = ("game_id", "home_score", "away_score")
_FROZEN_GAME_FIELDS: tuple[str, ...] = (
    "game_id",
    "kickoff",
    "home",
    "away",
    "model_home_wp",
    "market_home_wp",
    "diff",
    "model_version",
    "generated_at",
)
_FROZEN_DOCUMENT_FIELDS: tuple[str, ...] = (
    "sample",
    "season",
    "week",
    "generated_at",
    "model_name",
    "model_version",
    "feature_set",
)


def final_scores_by_game_id(games: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Index finished games by ``game_id``.

    A row counts only when both scores are present. ``status`` is ``final``
    for those rows. ``result`` is ``home``, ``away``, or ``tie``. Equal
    scores, including 0-0, are a tie. A row with a missing score is omitted.
    This does not treat a scoreless row as final.

    Returns:
        Mapping of ``game_id`` to ``home_score``, ``away_score``, ``result``,
        and ``status``.

    Raises:
        ValueError: A score column is missing, or two finished rows disagree.
    """
    missing: list[str] = [column for column in _SCORE_COLUMNS if column not in games.columns]
    if missing:
        raise ValueError(f"Games file missing columns: {', '.join(missing)}.")
    finals: dict[str, dict[str, Any]] = {}
    for _, row in games.iterrows():
        game_id: str = _game_id(row["game_id"])
        if game_id == "":
            continue
        home_score: int | None = _finished_score(row["home_score"])
        away_score: int | None = _finished_score(row["away_score"])
        if home_score is None or away_score is None:
            continue
        result: dict[str, Any] = {
            "home_score": home_score,
            "away_score": away_score,
            "result": scoreline_result(home_score, away_score),
            "status": FINAL_GAME_STATUS,
        }
        previous: dict[str, Any] | None = finals.get(game_id)
        if previous is not None and previous != result:
            raise ValueError(f"Conflicting final scores for {game_id}.")
        finals[game_id] = result
    return finals


def apply_final_scores(
    document: dict[str, Any],
    finals: dict[str, dict[str, Any]],
    results_updated_at: str,
) -> dict[str, Any]:
    """Return a copy with final scores attached.

    Prediction fields and ``generated_at`` stay as published. Every game
    gets ``status``. A finished nflverse row sets ``final`` and integer
    scores. A game that already has ``status`` ``final`` and both scores
    keeps those scores when nflverse has no finished row. Every other game
    is ``scheduled`` and has no score fields. An example document
    (``sample: true``) is returned unchanged.

    Returns:
        The same document when it is an example, otherwise a copy that
        includes ``results_updated_at``.
    """
    if document.get("sample") is True:
        return document
    games_value: object = document.get("games")
    games_in: list[Any] = list(games_value) if isinstance(games_value, list) else []
    games_out: list[Any] = []
    for game in games_in:
        if not isinstance(game, dict):
            games_out.append(game)
            continue
        merged: dict[str, Any] = _overlay_game(game, finals)
        games_out.append(merged)
        _assert_frozen_game(game, merged)
    updated: dict[str, Any] = _with_results_stamp(document, games_out, results_updated_at)
    _assert_frozen_document(document, updated)
    return updated


def refresh_results(
    output_dir: Path,
    *,
    games_path: Path | None = None,
    results_updated_at: str,
) -> dict[str, Any]:
    """Overlay finals onto every non-example week file and ``latest.json``.

    When ``latest.json`` is missing, it is copied from the newest real week
    file so a Pages deploy still has a current slate. Example files are not
    rewritten. ``games_path`` reads a local parquet or CSV. ``None`` downloads
    the nflverse schedules file and keeps the seasons those week files name.

    Returns:
        ``updated``, ``skipped_examples``, ``seasons``, ``final_games``, and
        ``matched_games``.

    Raises:
        ValueError: There is no real week to update, the games file is
            unusable, or a document fails validation.
    """
    root: Path = Path(output_dir)
    _ensure_latest_file(root)
    if _newest_real_week_file(root) is None and not _real_latest(root):
        raise ValueError(
            "No published week files to update. Restore the live site or "
            "choose an output directory that already has week JSON."
        )
    seasons: list[int] = _seasons_to_fetch(root)
    if not seasons:
        raise ValueError("Published week files are missing a season.")
    games: pd.DataFrame = _load_games(games_path, seasons)
    finals: dict[str, dict[str, Any]] = final_scores_by_game_id(games)
    planned: list[tuple[Path, dict[str, Any]]] = []
    skipped_examples: int = 0
    matched_games: int = 0
    for path in _prediction_files(root):
        document: dict[str, Any] = _read_document(path)
        if document.get("sample") is True:
            skipped_examples += 1
            continue
        updated: dict[str, Any] = apply_final_scores(document, finals, results_updated_at)
        validate_prediction_document(updated)
        matched_games += _matched_game_count(updated, finals)
        planned.append((path, updated))
    for path, updated in planned:
        path.write_text(_dumps(updated), encoding="utf-8")
    logger.info(
        "Overlaid final scores on %s file(s). %s finished nflverse game(s), %s matched.",
        len(planned),
        len(finals),
        matched_games,
    )
    return {
        "updated": len(planned),
        "skipped_examples": skipped_examples,
        "seasons": seasons,
        "final_games": len(finals),
        "matched_games": matched_games,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the results-refresh parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Overlay final nflverse scores onto published week JSON. Does not refit the model."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory that holds latest.json and predictions/<season>-w<week>.json.",
    )
    parser.add_argument(
        "--restore-published",
        type=str,
        default=None,
        help=(
            "Base URL of the live site. Copy non-example week files into the "
            "output directory before overlaying scores."
        ),
    )
    parser.add_argument(
        "--games-path",
        type=str,
        default=None,
        help="Local games parquet or CSV. Omit it to download the nflverse schedules file.",
    )
    parser.add_argument(
        "--results-updated-at",
        type=str,
        default=None,
        help="ISO UTC timestamp ending in Z. The default is the current time.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Restore published weeks when asked, then overlay final scores."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    output_dir: Path = Path(args.output_dir)
    if args.restore_published:
        restore_published_predictions(output_dir, args.restore_published)
    stamp: str = args.results_updated_at or _utc_now_text()
    games_path: Path | None = Path(args.games_path) if args.games_path else None
    return refresh_results(
        output_dir,
        games_path=games_path,
        results_updated_at=stamp,
    )


def _utc_now_text(clock: datetime | None = None) -> str:
    """Return an ISO UTC timestamp with a Z suffix and no microseconds."""
    moment: datetime = clock if clock is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _game_id(value: object) -> str:
    """Normalize a game id. Empty means the cell is missing."""
    if value is None or isinstance(value, bool):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()


def _finished_score(value: object) -> int | None:
    """Return a non-negative integer score, or None when the cell is empty."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    try:
        number: float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None
    return int(number)


def _overlay_game(game: dict[str, Any], finals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attach one game's status and scores without touching the prediction."""
    merged: dict[str, Any] = dict(game)
    merged.pop("home_won", None)
    final: dict[str, Any] | None = finals.get(_game_id(game.get("game_id")))
    if final is not None:
        merged["status"] = FINAL_GAME_STATUS
        merged["home_score"] = final["home_score"]
        merged["away_score"] = final["away_score"]
        merged["result"] = final["result"]
        return merged
    if game.get("status") == FINAL_GAME_STATUS:
        home_score: int | None = _finished_score(merged.get("home_score"))
        away_score: int | None = _finished_score(merged.get("away_score"))
        if home_score is not None and away_score is not None:
            merged["status"] = FINAL_GAME_STATUS
            merged["home_score"] = home_score
            merged["away_score"] = away_score
            merged["result"] = scoreline_result(home_score, away_score)
            return merged
    merged["status"] = SCHEDULED_GAME_STATUS
    merged.pop("home_score", None)
    merged.pop("away_score", None)
    merged.pop("result", None)
    return merged


def _assert_frozen_game(original: dict[str, Any], merged: dict[str, Any]) -> None:
    """Raise when an overlay changes a published prediction field."""
    for key in _FROZEN_GAME_FIELDS:
        if merged.get(key) != original.get(key):
            raise ValueError(f"Refusing to change {key} on {original.get('game_id')}.")


def _assert_frozen_document(original: dict[str, Any], updated: dict[str, Any]) -> None:
    """Raise when an overlay changes a published document field."""
    for key in _FROZEN_DOCUMENT_FIELDS:
        if updated.get(key) != original.get(key):
            raise ValueError(f"Refusing to change document field {key}.")


def _with_results_stamp(
    document: dict[str, Any],
    games: list[Any],
    results_updated_at: str,
) -> dict[str, Any]:
    """Copy ``document``, replace games, and set the results stamp."""
    updated: dict[str, Any] = {}
    stamped: bool = False
    for key, value in document.items():
        if key == "results_updated_at":
            continue
        if key == "games":
            updated["games"] = games
            continue
        updated[key] = value
        if key == "generated_at":
            updated["results_updated_at"] = results_updated_at
            stamped = True
    if "games" not in updated:
        updated["games"] = games
    if not stamped:
        updated["results_updated_at"] = results_updated_at
    return updated


def _matched_game_count(document: dict[str, Any], finals: dict[str, dict[str, Any]]) -> int:
    """Count games in one document that have a finished nflverse row."""
    games: object = document.get("games")
    if not isinstance(games, list):
        return 0
    matched: int = 0
    for game in games:
        if isinstance(game, dict) and _game_id(game.get("game_id")) in finals:
            matched += 1
    return matched


def _load_games(games_path: Path | None, seasons: list[int]) -> pd.DataFrame:
    """Load nflverse games and keep the requested seasons."""
    if games_path is None:
        frame: pd.DataFrame = download_parquet_with_validation(NFLVERSE_GAMES_URL)
    else:
        frame = _read_games_file(Path(games_path))
    if "season" not in frame.columns:
        raise ValueError("Games file is missing season.")
    numeric_season: pd.Series = pd.to_numeric(frame["season"], errors="coerce")
    return frame.loc[numeric_season.isin(seasons)].reset_index(drop=True)


def _read_games_file(path: Path) -> pd.DataFrame:
    """Read a local games parquet or CSV."""
    suffix: str = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Games file must be parquet or CSV: {path.name}.")


def _ensure_latest_file(output_dir: Path) -> None:
    """Copy the newest real week onto ``latest.json`` when that file is absent.

    ``restore_published_predictions`` writes week files and does not write
    ``latest.json``. The page loads ``latest.json``, so a results deploy
    needs it. An example file that is already present is left alone.
    """
    latest_path: Path = output_dir / "latest.json"
    if latest_path.is_file():
        return
    newest: Path | None = _newest_real_week_file(output_dir)
    if newest is None:
        return
    latest_path.write_text(newest.read_text(encoding="utf-8"), encoding="utf-8")


def _newest_real_week_file(output_dir: Path) -> Path | None:
    """Return the highest season and week file that is not an example."""
    for path in reversed(_week_files(output_dir)):
        try:
            document: dict[str, Any] = _read_document(path)
        except ValueError:
            continue
        if document.get("sample") is True:
            continue
        return path
    return None


def _real_latest(output_dir: Path) -> bool:
    """True when ``latest.json`` exists and is not an example slate."""
    latest_path: Path = output_dir / "latest.json"
    if not latest_path.is_file():
        return False
    try:
        document: dict[str, Any] = _read_document(latest_path)
    except ValueError:
        return False
    return document.get("sample") is not True


def _seasons_to_fetch(output_dir: Path) -> list[int]:
    """Seasons named by real week files, plus the season on ``latest.json``."""
    seasons: set[int] = set()
    for path in _week_files(output_dir):
        try:
            document: dict[str, Any] = _read_document(path)
        except ValueError:
            continue
        if document.get("sample") is True:
            continue
        match = _WEEK_FILE.match(path.name)
        if match is not None:
            seasons.add(int(match.group("season")))
    latest_path: Path = output_dir / "latest.json"
    if latest_path.is_file():
        try:
            latest: dict[str, Any] = _read_document(latest_path)
        except ValueError:
            latest = {}
        if latest.get("sample") is not True:
            season: object = latest.get("season")
            if isinstance(season, int) and not isinstance(season, bool):
                seasons.add(season)
    return sorted(seasons)


def _prediction_files(output_dir: Path) -> list[Path]:
    """Week files in season-week order, then ``latest.json`` when present."""
    files: list[Path] = _week_files(output_dir)
    latest_path: Path = output_dir / "latest.json"
    if latest_path.is_file():
        files.append(latest_path)
    return files


def _week_files(output_dir: Path) -> list[Path]:
    """Non-directory week JSON files, sorted by season and week."""
    predictions: Path = output_dir / "predictions"
    if not predictions.is_dir():
        return []
    found: list[tuple[int, int, Path]] = []
    for path in predictions.glob("*.json"):
        match = _WEEK_FILE.match(path.name)
        if match is None:
            continue
        found.append((int(match.group("season")), int(match.group("week")), path))
    found.sort()
    return [path for _, _, path in found]


def _read_document(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Raises:
        ValueError: The file is missing or is not a JSON object.
    """
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object.")
    return payload


if __name__ == "__main__":
    main()
