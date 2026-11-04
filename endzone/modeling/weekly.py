"""Weekly prediction documents: slate selection, team marks, and JSON files.

The page reads ``site/data``. Real score output stays untracked. A shipped
sample is labeled ``sample: true`` and has no season or week.
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
)
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
) -> dict[str, Any]:
    """Build the weekly JSON document.

    ``diff`` is model home win probability minus market home win probability.
    A missing market line leaves both ``market_home_wp`` and ``diff`` null.

    Returns:
        Document with one record per game. Each record has
        ``PREDICTION_GAME_FIELDS``.
    """
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
        "games": records,
    }


def validate_prediction_document(document: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a weekly document breaks the public schema."""
    required: tuple[str, ...] = (
        "sample",
        "season",
        "week",
        "generated_at",
        "model_name",
        "model_version",
        "feature_set",
        "games",
    )
    missing: list[str] = [key for key in required if key not in document]
    if missing:
        raise ValueError(f"Prediction document missing keys: {', '.join(missing)}.")
    if not isinstance(document["games"], list) or len(document["games"]) == 0:
        raise ValueError("Prediction document needs at least one game.")
    for game in document["games"]:
        absent: list[str] = [key for key in PREDICTION_GAME_FIELDS if key not in game]
        if absent:
            raise ValueError(f"Game {game.get('game_id')!r} missing keys: {', '.join(absent)}.")
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


def write_prediction_bundle(
    document: dict[str, Any],
    teams: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Write the week file, ``latest.json``, and ``teams.json``.

    ``weeks_available`` lists every non-example week file for this season
    already in ``output_dir``, including the file just written. Sibling week
    files get the same list so the page selector stays complete.

    Returns:
        Paths for ``week``, ``latest``, and ``teams``.
    """
    validate_prediction_document(document)
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
) -> dict[str, int]:
    """Copy non-example week files from a published site into ``output_dir``.

    An example ``latest.json``, a blank URL, or a failed read copies nothing.
    One missing week file does not drop the weeks that did download.

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
        url: str = f"{root_url}/data/predictions/{season}-w{week}.json"
        try:
            payload: dict[str, Any] = json.loads(fetcher(url))
        except (OSError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
            logger.warning("Skipping published week %s: %s", week, exc)
            continue
        if not _published_week_document(payload, season, week):
            continue
        destination: Path = predictions / f"{season}-w{week}.json"
        try:
            destination.write_text(_dumps(payload), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Skipping published week %s: %s", week, exc)
            continue
        copied += 1
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
