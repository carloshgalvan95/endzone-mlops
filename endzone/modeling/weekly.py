"""Weekly prediction documents: slate selection, team marks, and JSON files.

The page reads ``site/data``. Real score output stays untracked. A shipped
sample is labeled ``sample: true`` and has no season or week. Each game
has ``status`` ``scheduled`` or ``final``. Scores are attached after publish
and only when the game is final. They are not part of the fit.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from endzone.modeling.points_eval import HIDDEN_NOT_RUN
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

REGISTERED_MODEL_NAME: str = "endzone-winprob"
WEEKLY_FEATURE_SET: str = "h2_full"
WEEKLY_EXPERIMENT_NAME: str = "nfl_pregame/weekly"
DEFAULT_OUTPUT_DIR: Path = Path("site/data")
DEFAULT_THRESHOLD_PP: int = 5
NFLVERSE_TEAMS_URL: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv"
)
EASTERN: ZoneInfo = ZoneInfo("America/New_York")
HEX_COLOR: re.Pattern[str] = re.compile(r"^#[0-9A-Fa-f]{6}$")
_WEEK_FILE: re.Pattern[str] = re.compile(r"^(?P<season>\d+)-w(?P<week>\d+)\.json$")
PREDICTION_GAME_FIELDS: tuple[str, ...] = (
    "game_id",
    "kickoff",
    "home",
    "away",
    "model_home_wp",
    "market_home_wp",
    "diff",
    "model_version",
    "generated_at",
    "status",
    "estimated_margin",
    "estimated_total",
    "estimated_home_points",
    "estimated_away_points",
)
SCORE_MODEL_TARGET_FIELDS: tuple[str, ...] = (
    "decision",
    "baseline_name",
    "mae",
    "baseline_mae",
    "market_mae",
)
SCHEDULED_GAME_STATUS: str = "scheduled"
FINAL_GAME_STATUS: str = "final"
_RESULTS_UPDATED_AT: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SLATE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_team_franchise",
    "away_team_franchise",
    "home_rest",
    "away_rest",
)


def parse_optional_int(value: str | None) -> int | None:
    """Return an int, or None for an empty, ``auto``, or ``none`` flag."""
    if value is None:
        return None
    text: str = value.strip().lower()
    if text in {"", "auto", "none"}:
        return None
    return int(text)


def completed_games(games: pd.DataFrame) -> pd.DataFrame:
    """Keep games that already have a home-win label.

    Raises:
        ValueError: ``home_won`` is missing, or every row is unplayed.
    """
    if "home_won" not in games.columns:
        raise ValueError("fct_games is missing home_won.")
    played: pd.DataFrame = games.loc[games["home_won"].notna()].copy()
    if played.empty:
        raise ValueError("No completed games to fit h2_full.")
    return played


def games_before_week(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Completed games strictly before ``season`` and ``week``.

    A row counts when ``season`` is earlier, or the season matches and
    ``week`` is earlier. Games from the week being scored stay out, including
    ones that already have a final. A Sunday fit must not train on Thursday.

    Raises:
        ValueError: ``season`` or ``week`` is missing, or nothing is left.
    """
    played: pd.DataFrame = completed_games(games)
    for column in ("season", "week"):
        if column not in played.columns:
            raise ValueError(f"Completed games are missing {column}.")
    season_values: pd.Series = pd.to_numeric(played["season"], errors="coerce")
    week_values: pd.Series = pd.to_numeric(played["week"], errors="coerce")
    if season_values.isna().any() or week_values.isna().any():
        raise ValueError("Completed games need season and week before the fit cutoff.")
    target_season: int = int(season)
    target_week: int = int(week)
    before: pd.Series = (season_values < target_season) | (
        (season_values == target_season) & (week_values < target_week)
    )
    kept: pd.DataFrame = played.loc[before].copy()
    if kept.empty:
        raise ValueError(f"No completed games before season {target_season} week {target_week}.")
    return kept


def team_games_before_week(
    team_games: pd.DataFrame,
    season: int,
    week: int,
    history_ids: set[str],
) -> pd.DataFrame:
    """Team-game rows that may be used as pregame features for this week.

    Rows whose ``game_id`` is not in ``history_ids`` are dropped. When
    ``season`` and ``week`` are present, a row dated on or after the scored
    week is dropped too, even if its id was passed in. A row with a missing
    week stays when its id is in the history set.

    Raises:
        ValueError: ``game_id`` is missing.
    """
    if "game_id" not in team_games.columns:
        raise ValueError("Team-game features are missing game_id.")
    frame: pd.DataFrame = team_games.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    allowed: pd.Series = frame["game_id"].isin(history_ids)
    if "season" in frame.columns and "week" in frame.columns:
        season_values: pd.Series = pd.to_numeric(frame["season"], errors="coerce")
        week_values: pd.Series = pd.to_numeric(frame["week"], errors="coerce")
        known: pd.Series = season_values.notna() & week_values.notna()
        target_season: int = int(season)
        target_week: int = int(week)
        earlier: pd.Series = (season_values < target_season) | (
            (season_values == target_season) & (week_values < target_week)
        )
        allowed = allowed & ((~known) | earlier)
    return frame.loc[allowed].reset_index(drop=True)


def select_slate(
    upcoming: pd.DataFrame,
    season: int | None,
    week: int | None,
) -> dict[str, Any]:
    """Pick one unplayed week.

    The default season is the earliest season that still has an unplayed
    game. The default week is the earliest unplayed week in that season.

    Returns:
        Dict with ``season``, ``week``, and ``games``.

    Raises:
        ValueError: The upcoming table is empty or the requested week is not in it.
    """
    if upcoming.empty:
        raise ValueError("No upcoming games to score. fct_upcoming_games is empty.")
    frame: pd.DataFrame = upcoming.copy()
    if frame["season"].isna().any() or frame["week"].isna().any():
        raise ValueError("Upcoming games need season and week.")
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    chosen_season: int = int(frame["season"].min()) if season is None else int(season)
    season_rows: pd.DataFrame = frame.loc[frame["season"] == chosen_season]
    if season_rows.empty:
        raise ValueError(f"No unplayed games in season {chosen_season}.")
    chosen_week: int = int(season_rows["week"].min()) if week is None else int(week)
    slate: pd.DataFrame = season_rows.loc[season_rows["week"] == chosen_week].copy()
    if slate.empty:
        raise ValueError(f"No unplayed games for season {chosen_season} week {chosen_week}.")
    if slate["game_id"].duplicated().any():
        raise ValueError("Upcoming slate has duplicate game_id values.")
    ordered: pd.DataFrame = slate.sort_values(
        ["gameday", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return {"season": chosen_season, "week": chosen_week, "games": ordered}


def select_scheduled_week(
    games: pd.DataFrame,
    upcoming: pd.DataFrame,
    season: int,
    week: int,
) -> dict[str, Any]:
    """Every scheduled game in one season and week, played and unplayed.

    Completed rows come from ``fct_games``. Rows that still have no final
    come from ``fct_upcoming_games``. Scores and ``home_won`` are not copied
    onto the slate. A ``game_id`` that appears in both tables is kept once.

    Returns:
        Dict with ``season``, ``week``, and ``games``.

    Raises:
        ValueError: That week has no games, or a table repeats ``game_id``.
    """
    target_season: int = int(season)
    target_week: int = int(week)
    played: pd.DataFrame = _schedule_rows(games, target_season, target_week)
    remaining: pd.DataFrame = _schedule_rows(upcoming, target_season, target_week)
    # Prefer the upcoming row when a game is listed twice. It has no result.
    combined: pd.DataFrame = pd.concat([remaining, played], ignore_index=True)
    if combined.empty:
        raise ValueError(f"No scheduled games for season {target_season} week {target_week}.")
    combined["game_id"] = combined["game_id"].astype(str)
    combined = combined.drop_duplicates(subset=["game_id"], keep="first")
    ordered: pd.DataFrame = combined.sort_values(
        ["gameday", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return {"season": target_season, "week": target_week, "games": ordered}


def _schedule_rows(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Schedule columns for one week, without scores or labels.

    An empty table contributes no rows. A non-empty table must name the week.

    Raises:
        ValueError: A required schedule column is missing, or ``game_id`` repeats.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(_SLATE_COLUMNS))
    if "season" not in frame.columns or "week" not in frame.columns:
        raise ValueError("Schedule rows need season and week.")
    if frame["season"].isna().any() or frame["week"].isna().any():
        raise ValueError("Schedule rows need season and week.")
    working: pd.DataFrame = frame.copy()
    working["season"] = working["season"].astype(int)
    working["week"] = working["week"].astype(int)
    chosen: pd.DataFrame = working.loc[
        (working["season"] == season) & (working["week"] == week)
    ].copy()
    if chosen.empty:
        return pd.DataFrame(columns=list(_SLATE_COLUMNS))
    missing: list[str] = [column for column in _SLATE_COLUMNS if column not in chosen.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Scheduled games missing columns: {missing_list}.")
    if chosen["game_id"].duplicated().any():
        raise ValueError("Scheduled slate has duplicate game_id values.")
    keep: list[str] = [
        column
        for column in (*_SLATE_COLUMNS, "gametime", "home_moneyline", "away_moneyline")
        if column in chosen.columns
    ]
    return chosen.loc[:, keep].reset_index(drop=True)


def slate_for_features(slate: pd.DataFrame) -> pd.DataFrame:
    """Add the columns ``build_pregame_features`` needs on an unplayed slate.

    ``home_won`` is missing on purpose. The feature builder does not treat
    that row as a result.

    Raises:
        ValueError: A required schedule column is missing, or a franchise is null.
    """
    missing: list[str] = [column for column in _SLATE_COLUMNS if column not in slate.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Upcoming games missing columns: {missing_list}.")
    frame: pd.DataFrame = slate.copy()
    if frame["home_team_franchise"].isna().any() or frame["away_team_franchise"].isna().any():
        raise ValueError("Upcoming games need home_team_franchise and away_team_franchise.")
    frame["home_won"] = pd.NA
    if "home_moneyline" not in frame.columns:
        frame["home_moneyline"] = pd.NA
    if "away_moneyline" not in frame.columns:
        frame["away_moneyline"] = pd.NA
    if "gametime" not in frame.columns:
        frame["gametime"] = pd.NA
    return frame.reset_index(drop=True)


def kickoff_text(gameday: object, gametime: object) -> str:
    """Format kickoff as an Eastern timestamp, or the date when time is absent.

    nflverse ``gametime`` is Eastern (``America/New_York``). The page shows
    that instant in Mexico City.
    """
    day: pd.Timestamp = pd.to_datetime(gameday)
    if pd.isna(day):
        raise ValueError("gameday is missing.")
    date_text: str = f"{day.year:04d}-{day.month:02d}-{day.day:02d}"
    clock: tuple[int, int, int] | None = _clock_parts(gametime)
    if clock is None:
        return date_text
    hour, minute, second = clock
    stamp: datetime = datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=EASTERN)
    return stamp.isoformat()


def _clock_parts(gametime: object) -> tuple[int, int, int] | None:
    """Parse ``HH:MM`` or ``HH:MM:SS``. None means the time is absent."""
    if gametime is None:
        return None
    if isinstance(gametime, float) and math.isnan(gametime):
        return None
    try:
        if pd.isna(gametime):
            return None
    except (TypeError, ValueError):
        return None
    text: str = str(gametime).strip()
    if text == "" or text.lower() in {"nat", "none", "nan"}:
        return None
    clock: str = text.split(" ")[-1]
    parts: list[str] = clock.split(":")
    if len(parts) < 2:
        return None
    hour: int = int(float(parts[0]))
    minute: int = int(float(parts[1]))
    second: int = int(float(parts[2])) if len(parts) > 2 else 0
    return hour, minute, second


def _round_probability(value: float) -> float:
    """Round a probability to 6 decimal places for a stable JSON file."""
    return round(float(value), 6)


def default_score_model() -> dict[str, Any]:
    """Accuracy block used when the pre-registered test has not been run."""
    return {
        "feature_set": WEEKLY_FEATURE_SET,
        "accepted": False,
        "hidden_reason": HIDDEN_NOT_RUN,
        "margin": _blank_score_target("not_run", "Elo/25"),
        "total": _blank_score_target("not_run", "League mean"),
    }


def _blank_score_target(decision: str, baseline_name: str) -> dict[str, Any]:
    """One target with no MAE yet."""
    return {
        "decision": decision,
        "baseline_name": baseline_name,
        "mae": None,
        "baseline_mae": None,
        "market_mae": None,
    }


def _round_points(value: object) -> float | None:
    """Store a point estimate to 6 decimals. Display rounding is separate."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    number: float = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        return None
    return round(number, 6)


def build_prediction_document(
    *,
    season: int,
    week: int,
    games: pd.DataFrame,
    model_probabilities: pd.Series,
    market_probabilities: dict[str, float | None],
    model_name: str,
    model_version: str,
    feature_set: str,
    generated_at: str,
    sample: bool = False,
    score_lines: dict[str, dict[str, float | None]] | None = None,
    score_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the weekly JSON document.

    ``diff`` is model home win probability minus market home win probability.
    A missing market line leaves both ``market_home_wp`` and ``diff`` null.
    Point estimates stay null unless both Sprint 2 targets passed ADR-007.
    Home points are ``(total + margin) / 2``. Away points are
    ``(total - margin) / 2``.

    Returns:
        Document with one record per game. Each record has
        ``PREDICTION_GAME_FIELDS``.
    """
    model: dict[str, Any] = score_model if score_model is not None else default_score_model()
    publish_points: bool = model.get("accepted") is True
    lines: dict[str, dict[str, float | None]] = score_lines or {}
    records: list[dict[str, Any]] = []
    for _, row in games.iterrows():
        game_id: str = str(row["game_id"])
        model_probability: float = _round_probability(float(model_probabilities.loc[game_id]))
        market_raw: float | None = market_probabilities.get(game_id)
        market_probability: float | None = (
            None if market_raw is None else _round_probability(market_raw)
        )
        difference: float | None = (
            None
            if market_probability is None
            else _round_probability(model_probability - market_probability)
        )
        estimate: dict[str, float | None] = lines.get(game_id, {}) if publish_points else {}
        records.append(
            {
                "game_id": game_id,
                "kickoff": kickoff_text(row["gameday"], row.get("gametime")),
                "home": str(row["home_team"]),
                "away": str(row["away_team"]),
                "model_home_wp": model_probability,
                "market_home_wp": market_probability,
                "diff": difference,
                "model_version": model_version,
                "generated_at": generated_at,
                "status": SCHEDULED_GAME_STATUS,
                "estimated_margin": _round_points(estimate.get("estimated_margin")),
                "estimated_total": _round_points(estimate.get("estimated_total")),
                "estimated_home_points": _round_points(estimate.get("estimated_home_points")),
                "estimated_away_points": _round_points(estimate.get("estimated_away_points")),
            }
        )
    return {
        "sample": sample,
        "season": int(season),
        "week": int(week),
        "weeks_available": [int(week)],
        "generated_at": generated_at,
        "model_name": model_name,
        "model_version": model_version,
        "feature_set": feature_set,
        "threshold_pp": DEFAULT_THRESHOLD_PP,
        "score_model": model,
        "games": records,
    }


def validate_prediction_document(document: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a weekly document breaks the public schema.

    Prediction fields stay required. Every game has ``status`` of
    ``scheduled`` or ``final``. Scores are integers and are present only
    when ``status`` is ``final``. A final score of 0-0 is allowed. The
    winner is the higher score. Equal scores are a tie. ``result`` is
    optional and must match that scoreline when it is set.
    """
    required: tuple[str, ...] = (
        "sample",
        "season",
        "week",
        "generated_at",
        "model_name",
        "model_version",
        "feature_set",
        "score_model",
        "games",
    )
    missing: list[str] = [key for key in required if key not in document]
    if missing:
        raise ValueError(f"Prediction document missing keys: {', '.join(missing)}.")
    _validate_results_updated_at(document.get("results_updated_at"))
    if not isinstance(document["games"], list) or len(document["games"]) == 0:
        raise ValueError("Prediction document needs at least one game.")
    for game in document["games"]:
        absent: list[str] = [key for key in PREDICTION_GAME_FIELDS if key not in game]
        if absent:
            raise ValueError(f"Game {game.get('game_id')!r} missing keys: {', '.join(absent)}.")
        _validate_optional_result(game)
        model_probability: float = float(game["model_home_wp"])
        if not 0.0 <= model_probability <= 1.0:
            raise ValueError(f"model_home_wp for {game['game_id']} is outside [0, 1].")
        market_probability: object = game["market_home_wp"]
        if market_probability is None:
            if game["diff"] is not None:
                raise ValueError(
                    f"diff must be null when market_home_wp is null ({game['game_id']})."
                )
            continue
        market_value: float = float(market_probability)
        if not 0.0 <= market_value <= 1.0:
            raise ValueError(f"market_home_wp for {game['game_id']} is outside [0, 1].")
        expected: float = _round_probability(model_probability - market_value)
        if game["diff"] != expected:
            raise ValueError(f"diff for {game['game_id']} is {game['diff']}, expected {expected}.")
    _validate_score_model(document["score_model"], document["games"])


def _validate_score_model(score_model: object, games: list[dict[str, Any]]) -> None:
    """Require the accuracy block, and null points unless both targets passed."""
    if not isinstance(score_model, dict):
        raise ValueError("score_model must be an object.")
    for key in ("feature_set", "accepted", "hidden_reason", "margin", "total"):
        if key not in score_model:
            raise ValueError(f"score_model is missing {key}.")
    accepted: object = score_model["accepted"]
    if not isinstance(accepted, bool):
        raise ValueError("score_model.accepted must be true or false.")
    if accepted:
        if score_model["hidden_reason"] is not None:
            raise ValueError("An accepted score model has no hidden reason.")
    elif not isinstance(score_model["hidden_reason"], str) or not score_model["hidden_reason"]:
        raise ValueError("A hidden score model needs a reason.")
    for name in ("margin", "total"):
        target: object = score_model[name]
        if not isinstance(target, dict):
            raise ValueError(f"score_model.{name} must be an object.")
        absent: list[str] = [key for key in SCORE_MODEL_TARGET_FIELDS if key not in target]
        if absent:
            raise ValueError(f"score_model.{name} missing keys: {', '.join(absent)}.")
    for game in games:
        points: list[object] = [
            game["estimated_margin"],
            game["estimated_total"],
            game["estimated_home_points"],
            game["estimated_away_points"],
        ]
        if accepted:
            if any(point is None or not math.isfinite(float(point)) for point in points):
                raise ValueError(f"Accepted score model needs points for {game['game_id']}.")
            continue
        if any(point is not None for point in points):
            raise ValueError(f"Hidden score model must leave points null for {game['game_id']}.")


def _validate_results_updated_at(value: object) -> None:
    """Allow a missing stamp. A present stamp is ISO UTC with a Z suffix."""
    if value is None:
        return
    if isinstance(value, str) and _RESULTS_UPDATED_AT.match(value):
        return
    raise ValueError("results_updated_at must be an ISO UTC timestamp ending in Z.")


def _optional_score(game: dict[str, Any], key: str) -> int | None:
    """Return one score, or None when that cell is absent.

    Raises:
        ValueError: The cell is present and is not a non-negative integer.
    """
    if key not in game or game[key] is None:
        return None
    value: object = game[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} for {game.get('game_id')} must be an integer.")
    if value < 0:
        raise ValueError(f"{key} for {game.get('game_id')} must be zero or greater.")
    return value


def scoreline_result(home_score: int, away_score: int) -> str:
    """Return ``home``, ``away``, or ``tie`` from a final scoreline.

    Equal scores are a tie. Otherwise the higher score wins. This does not
    look at ``status``. Callers decide whether the game is final.
    """
    if home_score == away_score:
        return "tie"
    if home_score > away_score:
        return "home"
    return "away"


def _validate_optional_result(game: dict[str, Any]) -> None:
    """Require scheduled or final, and scores only on a final game."""
    status: object = game.get("status")
    if status not in (SCHEDULED_GAME_STATUS, FINAL_GAME_STATUS):
        raise ValueError(f"status for {game.get('game_id')} must be scheduled or final.")
    home_score: int | None = _optional_score(game, "home_score")
    away_score: int | None = _optional_score(game, "away_score")
    if status == SCHEDULED_GAME_STATUS:
        if home_score is not None or away_score is not None:
            raise ValueError(f"scheduled game {game.get('game_id')} must not include scores.")
        if game.get("result") is not None:
            raise ValueError(f"scheduled game {game.get('game_id')} must not include result.")
        return
    if home_score is None or away_score is None:
        raise ValueError(
            f"final game {game.get('game_id')} needs integer home_score and away_score."
        )
    expected: str = scoreline_result(home_score, away_score)
    result: object = game.get("result")
    if result is not None and result != expected:
        raise ValueError(f"result for {game.get('game_id')} does not match the scores.")


def weeks_present(output_dir: Path, season: int) -> list[int]:
    """Return sorted weeks that already have a non-example file for ``season``."""
    predictions: Path = Path(output_dir) / "predictions"
    if not predictions.is_dir():
        return []
    weeks: list[int] = []
    for path in sorted(predictions.glob(f"{int(season)}-w*.json")):
        match: re.Match[str] | None = _WEEK_FILE.match(path.name)
        if match is None or int(match.group("season")) != int(season):
            continue
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable prediction file %s: %s", path.name, exc)
            continue
        if payload.get("sample") is True:
            continue
        weeks.append(int(match.group("week")))
    return sorted(set(weeks))


def week_prediction_path(output_dir: Path, season: int, week: int) -> Path:
    """Path of ``predictions/<season>-w<week>.json`` under ``output_dir``."""
    return Path(output_dir) / "predictions" / f"{int(season)}-w{int(week)}.json"


def published_game_ids(output_dir: Path, season: int, week: int) -> set[str]:
    """Game ids already stored for this season and week.

    The week file contributes its ids when it is not an example. ``latest.json``
    contributes ids only when it is that same season and week. A file that
    cannot be read fails closed so a score cannot replace it blindly.

    Raises:
        ValueError: A file exists but is not a JSON object, or the week file
            names a different season or week.
    """
    root: Path = Path(output_dir)
    target_season: int = int(season)
    target_week: int = int(week)
    ids: set[str] = _game_ids_in_file(
        week_prediction_path(root, target_season, target_week),
        target_season,
        target_week,
        require_week_match=True,
    )
    ids.update(
        _game_ids_in_file(
            root / "latest.json",
            target_season,
            target_week,
            require_week_match=False,
        )
    )
    return ids


def refuse_dropped_game_ids(
    output_dir: Path,
    season: int,
    week: int,
    incoming_ids: set[str],
) -> None:
    """Raise when a new slate would drop a game id that is already published.

    Adding games is allowed. Replacing probabilities for the same ids is the
    caller's decision. Dropping an id is not.

    Raises:
        ValueError: At least one published ``game_id`` is absent from
            ``incoming_ids``.
    """
    existing: set[str] = published_game_ids(output_dir, season, week)
    dropped: set[str] = existing - {str(game_id) for game_id in incoming_ids}
    if not dropped:
        return
    shown: list[str] = sorted(dropped)[:20]
    suffix: str = ""
    if len(dropped) > len(shown):
        suffix = f" and {len(dropped) - len(shown)} more"
    dropped_list: str = ", ".join(shown)
    raise ValueError(
        f"Refusing to replace {int(season)} week {int(week)}: the published file "
        f"has {len(existing)} game_id(s) and the new slate drops {len(dropped)} "
        f"({dropped_list}{suffix})."
    )


def kept_published_week(output_dir: Path, season: int, week: int) -> dict[str, Any] | None:
    """Return the on-disk week when a score should keep it.

    A missing file, or an example slate, returns None so the caller can score.
    The week file itself is not rewritten. ``latest.json`` is pointed at that
    file when the bytes differ.

    Returns:
        Dict with ``document``, ``paths``, ``model_version``, ``run_id``,
        and ``kept_published`` set to True. None when there is nothing to keep.

    Raises:
        ValueError: The week file exists, is not an example, and has no games
            or names a different season or week.
    """
    path: Path = week_prediction_path(output_dir, season, week)
    if not path.is_file():
        return None
    document: dict[str, Any] = _read_json_object(path)
    if document.get("sample") is True:
        return None
    file_season: object = document.get("season")
    file_week: object = document.get("week")
    if file_season != int(season) or file_week != int(week):
        raise ValueError(
            f"{path.name} says season {file_season} week {file_week}, "
            f"expected {int(season)} week {int(week)}."
        )
    games: object = document.get("games")
    if not isinstance(games, list) or len(games) == 0:
        raise ValueError(f"Refusing to score over {path.name}: the published file has no games.")
    latest: Path = Path(output_dir) / "latest.json"
    _point_latest_at(path, latest)
    teams: Path = Path(output_dir) / "teams.json"
    logger.info(
        "Keeping published %s week %s (%s games). "
        "Pass --replace-published to rebuild the full week.",
        int(season),
        int(week),
        len(games),
    )
    return {
        "document": document,
        "paths": {
            "week": str(path),
            "latest": str(latest),
            "teams": str(teams) if teams.is_file() else "",
        },
        "model_version": str(document.get("model_version", "")),
        "run_id": "",
        "kept_published": True,
    }


def _point_latest_at(week_path: Path, latest_path: Path) -> None:
    """Copy the week file onto ``latest.json`` when the bytes differ."""
    payload: str = week_path.read_text(encoding="utf-8")
    if latest_path.is_file() and latest_path.read_text(encoding="utf-8") == payload:
        return
    latest_path.write_text(payload, encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Raises:
        ValueError: The file is missing, unreadable, or not an object.
    """
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Refusing to replace unreadable predictions file {path.name}: {exc}."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Refusing to replace {path.name}: it is not a JSON object.")
    return payload


def _game_ids_in_file(
    path: Path,
    season: int,
    week: int,
    *,
    require_week_match: bool,
) -> set[str]:
    """Game ids in one predictions file, or an empty set when it does not apply."""
    if not path.is_file():
        return set()
    document: dict[str, Any] = _read_json_object(path)
    if document.get("sample") is True:
        return set()
    file_season: object = document.get("season")
    file_week: object = document.get("week")
    matches: bool = file_season == season and file_week == week
    if not matches:
        if require_week_match:
            raise ValueError(
                f"{path.name} says season {file_season} week {file_week}, "
                f"expected {season} week {week}."
            )
        return set()
    games: object = document.get("games")
    if not isinstance(games, list):
        raise ValueError(f"{path.name} is missing a games list.")
    ids: set[str] = set()
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id: object = game.get("game_id")
        if game_id is None:
            continue
        ids.add(str(game_id))
    return ids


def write_prediction_bundle(
    document: dict[str, Any],
    teams: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Write the week file, ``latest.json``, and ``teams.json``.

    ``weeks_available`` lists every non-example week file for this season
    already in ``output_dir``, including the file just written. Sibling week
    files get the same list so the page selector stays complete. A week file
    that already exists cannot lose a ``game_id``. The write fails first.

    Returns:
        Paths for ``week``, ``latest``, and ``teams``.

    Raises:
        ValueError: The document is invalid, or it drops a published game id.
    """
    validate_prediction_document(document)
    incoming_ids: set[str] = {str(game["game_id"]) for game in document["games"]}
    refuse_dropped_game_ids(
        output_dir,
        int(document["season"]),
        int(document["week"]),
        incoming_ids,
    )
    root: Path = Path(output_dir)
    predictions: Path = root / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    season: int = int(document["season"])
    week: int = int(document["week"])
    week_path: Path = predictions / f"{season}-w{week}.json"
    latest_path: Path = root / "latest.json"
    teams_path: Path = root / "teams.json"
    week_path.write_text(_dumps(document), encoding="utf-8")
    weeks: list[int] = weeks_present(root, season)
    document["weeks_available"] = weeks
    payload: str = _dumps(document)
    week_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    for path in sorted(predictions.glob(f"{season}-w*.json")):
        if path == week_path:
            continue
        _record_weeks_available(path, weeks)
    teams_path.write_text(_dumps(teams), encoding="utf-8")
    logger.info("Wrote %s and %s", week_path, latest_path)
    return {
        "week": str(week_path),
        "latest": str(latest_path),
        "teams": str(teams_path),
    }


def _record_weeks_available(path: Path, weeks: list[int]) -> None:
    """Set ``weeks_available`` on one week file. Example files are left alone."""
    try:
        document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not update weeks on %s: %s", path.name, exc)
        return
    if document.get("sample") is True or document.get("weeks_available") == weeks:
        return
    document["weeks_available"] = list(weeks)
    path.write_text(_dumps(document), encoding="utf-8")


def restore_published_predictions(
    output_dir: Path,
    base_url: str,
    fetch_text: Callable[[str], str] | None = None,
    *,
    skip_existing: bool = False,
    write_latest: bool = False,
) -> dict[str, int]:
    """Copy non-example week files from a published site into ``output_dir``.

    An example ``latest.json``, a blank URL, or a failed read copies nothing.
    One missing week file does not drop the weeks that did download.
    ``skip_existing`` leaves a week file that is already on disk, so an
    archive restore can fill only the weeks it does not have. ``write_latest``
    also writes a real ``latest.json`` unless that file is already present
    and ``skip_existing`` is set.

    Returns:
        Dict with ``copied``, the number of week files written.
    """
    root_url: str = base_url.strip().rstrip("/")
    if root_url == "":
        return {"copied": 0}
    fetcher: Callable[[str], str] = fetch_text or _fetch_published_text
    try:
        latest_payload: dict[str, Any] = json.loads(fetcher(f"{root_url}/data/latest.json"))
    except (OSError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        logger.warning("Could not read published latest.json: %s", exc)
        return {"copied": 0}
    if not isinstance(latest_payload, dict) or latest_payload.get("sample") is True:
        logger.info("Published latest.json is an example slate. No week files were copied.")
        return {"copied": 0}
    season: object = latest_payload.get("season")
    if isinstance(season, bool) or not isinstance(season, int):
        return {"copied": 0}
    weeks: list[int] = _published_weeks(latest_payload)
    predictions: Path = Path(output_dir) / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    copied: int = 0
    for week in weeks:
        destination: Path = predictions / f"{season}-w{week}.json"
        if skip_existing and destination.is_file():
            continue
        url: str = f"{root_url}/data/predictions/{season}-w{week}.json"
        try:
            payload: dict[str, Any] = json.loads(fetcher(url))
        except (OSError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
            logger.warning("Skipping published week %s: %s", week, exc)
            continue
        if not _published_week_document(payload, season, week):
            continue
        try:
            destination.write_text(_dumps(payload), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Skipping published week %s: %s", week, exc)
            continue
        copied += 1
    if write_latest:
        latest_path: Path = Path(output_dir) / "latest.json"
        latest_week: object = latest_payload.get("week")
        if not (skip_existing and latest_path.is_file()) and _published_week_document(
            latest_payload, season, latest_week
        ):
            latest_path.write_text(_dumps(latest_payload), encoding="utf-8")
    logger.info("Copied %s published week file(s) into %s", copied, predictions)
    return {"copied": copied}


def _published_weeks(latest_payload: dict[str, Any]) -> list[int]:
    """Week numbers listed on a published latest document, plus its own week."""
    raw_weeks: object = latest_payload.get("weeks_available")
    values: list[object] = list(raw_weeks) if isinstance(raw_weeks, list) else []
    if not values:
        values = [latest_payload.get("week")]
    elif latest_payload.get("week") not in values:
        values.append(latest_payload.get("week"))
    weeks: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        weeks.append(value)
    return sorted(set(weeks))


def _published_week_document(payload: object, season: int, week: int) -> bool:
    """True when a downloaded file is a real week for this season."""
    if not isinstance(payload, dict) or payload.get("sample") is True:
        return False
    return payload.get("season") == season and payload.get("week") == week


def _fetch_published_text(url: str) -> str:
    """GET one published JSON file. A non-200 response raises ``ValueError``."""
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Published predictions returned HTTP {response.status_code}.")
    return response.text


def _dumps(payload: dict[str, Any]) -> str:
    """Serialize JSON with stable spacing. NaN is refused."""
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def load_team_catalog(source: Path | None = None) -> pd.DataFrame:
    """Load nflverse ``teams_colors_logos``.

    ``None`` downloads the release CSV. A ``.json`` file is a catalog this
    module already wrote. Any other path is read as CSV.

    Raises:
        ValueError: The download is not HTTP 200.
    """
    if source is None:
        logger.info("Downloading team colors and logos from %s", NFLVERSE_TEAMS_URL)
        response = requests.get(NFLVERSE_TEAMS_URL, timeout=60)
        if response.status_code != 200:
            raise ValueError(
                "Failed to download team colors and logos: "
                f"HTTP {response.status_code} from {NFLVERSE_TEAMS_URL}."
            )
        return pd.read_csv(BytesIO(response.content))
    path: Path = Path(source)
    if path.suffix.lower() == ".json":
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        teams: object = payload["teams"]
        rows: list[Any] = list(teams.values()) if isinstance(teams, dict) else list(teams)
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def team_catalog_document(frame: pd.DataFrame) -> dict[str, Any]:
    """Map team abbreviations to name, validated colors, and a logo URL.

    The logo is ``team_logo_espn`` when that URL is present. ESPN's
    ``500-dark`` asset is a light mark, so that one team uses
    ``team_logo_wikipedia`` instead. Colors that are not ``#RRGGBB`` are
    dropped. Logo image files are not stored.
    """
    teams: dict[str, dict[str, Any]] = {}
    if "team_abbr" not in frame.columns and "abbr" not in frame.columns:
        raise ValueError("Team catalog needs team_abbr or abbr.")
    sort_column: str = "team_abbr" if "team_abbr" in frame.columns else "abbr"
    ordered: pd.DataFrame = frame.sort_values(sort_column, kind="mergesort")
    for _, row in ordered.iterrows():
        abbr: str = _abbr(row)
        name: str = _team_name(row, abbr)
        teams[abbr] = {
            "abbr": abbr,
            "name": name,
            "team_color": _hex_color(row.get("team_color")),
            "team_color2": _hex_color(row.get("team_color2")),
            "logo": _logo_url(row),
        }
    return {
        "source": "nflverse teams_colors_logos",
        "source_url": NFLVERSE_TEAMS_URL,
        "teams": teams,
    }


def _abbr(row: pd.Series) -> str:
    """Read ``team_abbr`` or the ``abbr`` key written into teams.json."""
    for column in ("team_abbr", "abbr"):
        if column not in row.index:
            continue
        value: object = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text: str = str(value).strip()
        if text:
            return text
    raise ValueError("Team catalog row is missing team_abbr.")


def _team_name(row: pd.Series, abbr: str) -> str:
    """Prefer the short nickname, then the full name, then the abbreviation."""
    for column in ("team_nick", "name", "team_name"):
        if column not in row.index:
            continue
        value: object = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text: str = str(value).strip()
        if text:
            return text
    return abbr


def _hex_color(value: object) -> str | None:
    """Return a ``#RRGGBB`` color, or None when the cell is not one."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text: str = str(value).strip()
    if HEX_COLOR.match(text):
        return text
    return None


def _logo_url(row: pd.Series) -> str | None:
    """Pick an ESPN logo URL, falling back to Wikipedia for a dark variant."""
    espn: str = _url_text(
        row.get("team_logo_espn") if "team_logo_espn" in row.index else row.get("logo")
    )
    wiki: str = _url_text(row.get("team_logo_wikipedia"))
    if espn and "500-dark" not in espn:
        return espn
    if wiki:
        return wiki
    return espn or None


def _url_text(value: object) -> str:
    """Return a stripped URL string, or empty when the cell is missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()
