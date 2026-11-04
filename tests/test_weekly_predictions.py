"""Weekly scoring and the prediction JSON schema. No warehouse."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from endzone.features.pregame import build_pregame_features
from endzone.modeling.score import WeeklyScoreSettings, build_parser, score_upcoming_week
from endzone.modeling.weekly import (
    PREDICTION_GAME_FIELDS,
    build_prediction_document,
    load_team_catalog,
    parse_optional_int,
    select_slate,
    team_catalog_document,
    validate_prediction_document,
    write_prediction_bundle,
)

_FIXED_CLOCK: datetime = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def _release_mlflow_sqlalchemy_engines() -> None:
    """Dispose cached SQLAlchemy engines that keep a SQLite file open."""
    from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
    from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
    from mlflow.tracking._model_registry.utils import _get_store_registry
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry

    for store_cls in (TrackingStore, RegistryStore, SqlAlchemyJobStore):
        engine_map = getattr(store_cls, "_engine_map", None)
        if not isinstance(engine_map, dict):
            continue
        for engine in list(engine_map.values()):
            engine.dispose()
        engine_map.clear()
    tracking_cache = _tracking_store_registry._get_store_with_resolved_uri
    tracking_cache.cache_clear()
    registry_cache = _get_store_registry()._get_store_with_resolved_uri
    registry_cache.cache_clear()


def _reset_mlflow_tracking() -> None:
    """End any open run and drop SQLite engines."""
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.end_run()
    _release_mlflow_sqlalchemy_engines()
    mlflow.set_tracking_uri("sqlite:///:memory:")


def _catalog() -> pd.DataFrame:
    """Two teams: one normal ESPN logo, one dark ESPN logo that must fall back."""
    return pd.DataFrame(
        [
            {
                "team_abbr": "BUF",
                "team_nick": "Bills",
                "team_color": "#00338D",
                "team_color2": "navy",
                "team_logo_espn": "https://example.test/buf.png",
                "team_logo_wikipedia": "https://example.test/buf-wiki.png",
            },
            {
                "team_abbr": "CAR",
                "team_nick": "Panthers",
                "team_color": "blue",
                "team_color2": "#000000",
                "team_logo_espn": "https://a.espncdn.com/i/teamlogos/nfl/500-dark/car.png",
                "team_logo_wikipedia": "https://example.test/car.png",
            },
        ]
    )


def _completed_and_slate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two seasons of completed games plus the next unplayed week."""
    games: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    pairs: tuple[tuple[str, str], ...] = (("BUF", "MIA"), ("KC", "DEN"))
    for season, n_weeks, start in ((2023, 4, "2023-09-10"), (2024, 8, "2024-09-08")):
        start_day: pd.Timestamp = pd.Timestamp(start)
        for week in range(1, n_weeks + 1):
            gameday: pd.Timestamp = start_day + pd.Timedelta(days=7 * (week - 1))
            for index, (home, away) in enumerate(pairs):
                if week % 2 == 0:
                    home, away = away, home
                home_won: int = 0 if (week + index) % 2 else 1
                game_id: str = f"{season}W{week}{home}{away}"
                games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "gameday": gameday,
                        "gametime": "13:00",
                        "home_team": home,
                        "away_team": away,
                        "home_team_franchise": home,
                        "away_team_franchise": away,
                        "home_won": home_won,
                        "home_rest": 7,
                        "away_rest": 7,
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                    }
                )
                for team, offense in ((home, 0.1 * week), (away, -0.05 * week)):
                    team_rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "gameday": gameday,
                            "team_franchise": team,
                            "avg_offensive_epa": offense,
                            "avg_defensive_epa_allowed": -offense,
                        }
                    )
    upcoming_day: pd.Timestamp = pd.Timestamp("2024-09-08") + pd.Timedelta(days=7 * 8)
    upcoming = pd.DataFrame(
        [
            {
                "game_id": "2024W9BUFMIA",
                "season": 2024,
                "week": 9,
                "gameday": upcoming_day,
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_team_franchise": "BUF",
                "away_team_franchise": "MIA",
                "home_rest": 7,
                "away_rest": 7,
                "home_moneyline": -150,
                "away_moneyline": 130,
            },
            {
                "game_id": "2024W9KCDEN",
                "season": 2024,
                "week": 9,
                "gameday": upcoming_day,
                "gametime": "16:25",
                "home_team": "KC",
                "away_team": "DEN",
                "home_team_franchise": "KC",
                "away_team_franchise": "DEN",
                "home_rest": 7,
                "away_rest": 6,
                "home_moneyline": pd.NA,
                "away_moneyline": pd.NA,
            },
        ]
    )
    return pd.DataFrame(games), pd.DataFrame(team_rows), upcoming


def test_select_slate_defaults_to_the_next_unplayed_week() -> None:
    """The earliest remaining season and week is the default slate."""
    _games, _team_games, upcoming = _completed_and_slate()
    later = upcoming.copy()
    later["season"] = 2025
    later["week"] = 1
    combined = pd.concat([upcoming, later], ignore_index=True)
    chosen = select_slate(combined, None, None)
    assert chosen["season"] == 2024
    assert chosen["week"] == 9
    assert len(chosen["games"]) == 2


def test_select_slate_rejects_a_played_week() -> None:
    """A week that is not in the upcoming table is an error."""
    _games, _team_games, upcoming = _completed_and_slate()
    with pytest.raises(ValueError, match="week 3"):
        select_slate(upcoming, 2024, 3)


def test_unplayed_row_is_not_win_rate_history() -> None:
    """A missing home_won is not a loss in the rolling win rate."""
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "season": 2024,
                "gameday": pd.Timestamp("2024-09-08"),
                "home_team_franchise": "BUF",
                "away_team_franchise": "MIA",
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
            },
            {
                "game_id": "G2",
                "season": 2024,
                "gameday": pd.Timestamp("2024-09-15"),
                "home_team_franchise": "BUF",
                "away_team_franchise": "KC",
                "home_won": 1,
                "home_rest": 7,
                "away_rest": 7,
            },
        ]
    )
    epa_rows: list[dict[str, object]] = []
    for game_id, gameday, home, away, offense in (
        ("G1", "2024-09-08", "BUF", "MIA", 0.2),
        ("G2", "2024-09-15", "BUF", "KC", 0.3),
    ):
        for team, value in ((home, offense), (away, -offense)):
            epa_rows.append(
                {
                    "game_id": game_id,
                    "season": 2024,
                    "gameday": pd.Timestamp(gameday),
                    "team_franchise": team,
                    "avg_offensive_epa": value,
                    "avg_defensive_epa_allowed": -value,
                }
            )
    epa = pd.DataFrame(epa_rows)
    baseline = build_pregame_features(games, epa, lookback_games=10)
    unplayed = games.iloc[[0]].copy()
    unplayed["game_id"] = "GAP"
    unplayed["gameday"] = pd.Timestamp("2024-09-12")
    unplayed["home_won"] = pd.NA
    mixed = pd.concat([games, unplayed], ignore_index=True)
    scored = build_pregame_features(mixed, epa, lookback_games=10)
    base_rate = baseline.loc[baseline["game_id"] == "G2", "home_recent_win_rate"].iloc[0]
    mixed_rate = scored.loc[scored["game_id"] == "G2", "home_recent_win_rate"].iloc[0]
    assert base_rate == pytest.approx(1.0)
    assert mixed_rate == pytest.approx(base_rate)


def test_prediction_document_schema_and_market_gap() -> None:
    """diff is model minus market, and a missing line leaves diff null."""
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "gameday": "2026-11-08",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "MIA",
            },
            {
                "game_id": "G2",
                "gameday": "2026-11-08",
                "gametime": pd.NA,
                "home_team": "KC",
                "away_team": "DEN",
            },
        ]
    )
    document = build_prediction_document(
        season=2026,
        week=9,
        games=games,
        model_probabilities=pd.Series({"G1": 0.58123456, "G2": 0.42}),
        market_probabilities={"G1": 0.55, "G2": None},
        model_name="endzone-winprob",
        model_version="3",
        feature_set="h2_full",
        generated_at="2026-11-04T12:00:00Z",
    )
    validate_prediction_document(document)
    first, second = document["games"]
    assert set(PREDICTION_GAME_FIELDS) <= set(first)
    assert first["model_home_wp"] == pytest.approx(0.581235)
    assert first["diff"] == pytest.approx(
        round(first["model_home_wp"] - first["market_home_wp"], 6)
    )
    assert first["kickoff"].endswith("-05:00") or first["kickoff"].endswith("-04:00")
    assert second["market_home_wp"] is None
    assert second["diff"] is None
    assert second["kickoff"] == "2026-11-08"
    assert document["sample"] is False
    assert document["model_version"] == "3"


def test_team_catalog_logo_and_hex() -> None:
    """ESPN is preferred, a dark ESPN asset falls back, and bad hex is dropped."""
    document = team_catalog_document(_catalog())
    bills = document["teams"]["BUF"]
    panthers = document["teams"]["CAR"]
    assert bills["logo"] == "https://example.test/buf.png"
    assert bills["team_color"] == "#00338D"
    assert bills["team_color2"] is None
    assert panthers["logo"] == "https://example.test/car.png"
    assert panthers["team_color"] is None
    assert panthers["team_color2"] == "#000000"


def test_team_catalog_json_round_trip(tmp_path: Path) -> None:
    """A written teams.json loads back to the same abbreviations."""
    document = team_catalog_document(_catalog())
    path = tmp_path / "teams.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = team_catalog_document(load_team_catalog(path))
    assert set(loaded["teams"]) == {"BUF", "CAR"}
    assert loaded["teams"]["BUF"]["name"] == "Bills"


def test_write_bundle_names(tmp_path: Path) -> None:
    """The week file, latest.json, and teams.json land under the output directory."""
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "gameday": "2026-11-08",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "MIA",
            }
        ]
    )
    document = build_prediction_document(
        season=2026,
        week=9,
        games=games,
        model_probabilities=pd.Series({"G1": 0.5}),
        market_probabilities={"G1": 0.4},
        model_name="endzone-winprob",
        model_version="1",
        feature_set="h2_full",
        generated_at="2026-11-04T12:00:00Z",
    )
    paths = write_prediction_bundle(document, team_catalog_document(_catalog()), tmp_path)
    week_path = Path(paths["week"])
    latest = json.loads(Path(paths["latest"]).read_text(encoding="utf-8"))
    assert week_path.name == "2026-w9.json"
    assert latest["games"][0]["game_id"] == "G1"
    assert "BUF" in json.loads(Path(paths["teams"]).read_text(encoding="utf-8"))["teams"]


def test_parse_optional_int_and_parser() -> None:
    """auto means the scorer picks the next unplayed week."""
    assert parse_optional_int("auto") is None
    assert parse_optional_int("9") == 9
    args = build_parser().parse_args(["--season", "2026", "--week", "auto"])
    assert parse_optional_int(args.season) == 2026
    assert parse_optional_int(args.week) is None


def test_score_week_registers_model_and_writes_json(tmp_path: Path) -> None:
    """A fake slate fits, registers, and writes probabilities in (0, 1)."""
    import mlflow

    games, team_games, upcoming = _completed_and_slate()
    tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    settings = WeeklyScoreSettings(
        output_dir=tmp_path / "out",
        tracking_uri=tracking_uri,
        logistic_max_iter=400,
        generated_at=_FIXED_CLOCK,
    )
    try:
        result = score_upcoming_week(games, team_games, upcoming, _catalog(), settings)
    finally:
        _reset_mlflow_tracking()

    document: dict[str, Any] = result["document"]
    validate_prediction_document(document)
    assert document["sample"] is False
    assert document["season"] == 2024
    assert document["week"] == 9
    assert document["feature_set"] == "h2_full"
    assert document["model_name"] == "endzone-winprob"
    assert document["generated_at"] == "2026-11-04T12:00:00Z"
    assert result["model_version"].isdigit()
    by_id = {game["game_id"]: game for game in document["games"]}
    lined = by_id["2024W9BUFMIA"]
    missing_line = by_id["2024W9KCDEN"]
    assert 0.0 < lined["model_home_wp"] < 1.0
    assert lined["market_home_wp"] is not None
    assert lined["diff"] == pytest.approx(
        round(lined["model_home_wp"] - lined["market_home_wp"], 6)
    )
    assert missing_line["market_home_wp"] is None
    assert missing_line["diff"] is None
    assert lined["kickoff"] == "2024-11-03T13:00:00-05:00"
    week_file = json.loads(Path(result["paths"]["week"]).read_text(encoding="utf-8"))
    assert week_file["model_version"] == result["model_version"]
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions("name='endzone-winprob'")
    assert len(versions) == 1
    _reset_mlflow_tracking()


def test_shipped_sample_is_labeled_example() -> None:
    """The page sample is example data, not a model fit."""
    sample_path = Path("site/data/sample.json")
    document = json.loads(sample_path.read_text(encoding="utf-8"))
    assert document["sample"] is True
    assert document["is_example"] is True
    validate_prediction_document(document)
    teams = json.loads(Path("site/data/teams.json").read_text(encoding="utf-8"))
    assert teams["source"] == "nflverse teams_colors_logos"
    assert teams["teams"]["KC"]["logo"].startswith("https://")
    assert not teams["teams"]["KC"]["logo"].endswith(".svg")
