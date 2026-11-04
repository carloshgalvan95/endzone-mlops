"""Weekly scoring and the prediction JSON schema. No warehouse."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from endzone.features.pregame import build_pregame_features, no_vig_home_probability
from endzone.modeling.score import (
    LOCAL_MODEL_VERSION,
    WeeklyScoreSettings,
    _market_by_game,
    _register_weekly_model,
    build_parser,
    score_upcoming_week,
)
from endzone.modeling.weekly import (
    PREDICTION_GAME_FIELDS,
    build_prediction_document,
    load_team_catalog,
    parse_optional_int,
    restore_published_predictions,
    select_slate,
    slate_for_features,
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
    assert document["score_model"]["accepted"] is False
    assert first["estimated_home_points"] is None
    assert first["estimated_away_points"] is None


def test_accepted_points_follow_margin_and_total() -> None:
    """Home is (total + margin) / 2. A failed model cannot carry a score."""
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
    score_model: dict[str, Any] = {
        "feature_set": "h2_full",
        "accepted": True,
        "hidden_reason": None,
        "margin": {
            "decision": "accepted",
            "baseline_name": "Elo/25",
            "mae": 10.0,
            "baseline_mae": 11.0,
            "market_mae": 10.2,
        },
        "total": {
            "decision": "accepted",
            "baseline_name": "League mean",
            "mae": 9.0,
            "baseline_mae": 10.0,
            "market_mae": 8.5,
        },
    }
    document = build_prediction_document(
        season=2026,
        week=9,
        games=games,
        model_probabilities=pd.Series({"G1": 0.6}),
        market_probabilities={"G1": 0.55},
        model_name="endzone-winprob",
        model_version="1",
        feature_set="h2_full",
        generated_at="2026-11-04T12:00:00Z",
        score_lines={
            "G1": {
                "estimated_margin": 3.0,
                "estimated_total": 45.0,
                "estimated_home_points": 24.0,
                "estimated_away_points": 21.0,
            }
        },
        score_model=score_model,
    )
    validate_prediction_document(document)
    game = document["games"][0]
    assert game["estimated_home_points"] == pytest.approx(24.0)
    assert game["estimated_away_points"] == pytest.approx(21.0)
    rejected = dict(score_model)
    rejected["accepted"] = False
    rejected["hidden_reason"] = "Estimate hidden because it did not pass its pre-registered test."
    hidden = build_prediction_document(
        season=2026,
        week=9,
        games=games,
        model_probabilities=pd.Series({"G1": 0.6}),
        market_probabilities={"G1": 0.55},
        model_name="endzone-winprob",
        model_version="1",
        feature_set="h2_full",
        generated_at="2026-11-04T12:00:00Z",
        score_lines={
            "G1": {
                "estimated_margin": 3.0,
                "estimated_total": 45.0,
                "estimated_home_points": 24.0,
                "estimated_away_points": 21.0,
            }
        },
        score_model=rejected,
    )
    validate_prediction_document(hidden)
    assert hidden["games"][0]["estimated_home_points"] is None
    leaked = dict(hidden)
    leaked["games"] = [dict(hidden["games"][0])]
    leaked["games"][0]["estimated_margin"] = 3.0
    leaked["games"][0]["estimated_total"] = 45.0
    leaked["games"][0]["estimated_home_points"] = 24.0
    leaked["games"][0]["estimated_away_points"] = 21.0
    with pytest.raises(ValueError, match="Hidden score model"):
        validate_prediction_document(leaked)


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
    restore_args = build_parser().parse_args(["--restore-published", "https://example.test"])
    assert restore_args.restore_published == "https://example.test"


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
    assert document["score_model"]["accepted"] is False
    assert "pre-registered test has not been run" in document["score_model"]["hidden_reason"]
    assert lined["estimated_home_points"] is None
    assert lined["estimated_away_points"] is None
    assert lined["kickoff"] == "2024-11-03T13:00:00-05:00"
    week_file = json.loads(Path(result["paths"]["week"]).read_text(encoding="utf-8"))
    assert week_file["model_version"] == result["model_version"]
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions("name='endzone-winprob'")
    assert len(versions) == 1
    _reset_mlflow_tracking()


def test_shipped_sample_is_labeled_example() -> None:
    """The page sample is an example slate, not a season and week."""
    sample_path = Path("site/data/sample.json")
    document = json.loads(sample_path.read_text(encoding="utf-8"))
    assert document["sample"] is True
    assert document["is_example"] is True
    assert document["season"] is None
    assert document["week"] is None
    assert document["weeks_available"] == []
    assert document["label"].startswith("Example slate")
    validate_prediction_document(document)
    for game in document["games"]:
        assert "2026" not in str(game["game_id"])
        assert "W09" not in str(game["game_id"])
    page = Path("site/index.html").read_text(encoding="utf-8")
    script = Path("site/assets/app.js").read_text(encoding="utf-8")
    assert "Week 9" not in page
    assert ">2026<" not in page
    assert "Model accuracy" in page
    assert "WINS" in script
    assert "Estimate hidden because it did not pass its pre-registered test." in script
    assert 'loadJson("data/latest.json")' in script
    assert 'loadJson("data/sample.json")' in script
    teams = json.loads(Path("site/data/teams.json").read_text(encoding="utf-8"))
    assert teams["source"] == "nflverse teams_colors_logos"
    assert teams["teams"]["KC"]["logo"].startswith("https://")
    assert not teams["teams"]["KC"]["logo"].endswith(".svg")


def _upcoming_row(game_id: str, home_line: object, away_line: object) -> dict[str, object]:
    """One unplayed slate row with the columns the feature builder requires."""
    return {
        "game_id": game_id,
        "season": 2026,
        "week": 9,
        "gameday": "2026-11-08",
        "gametime": "13:00",
        "home_team": "BAL",
        "away_team": "JAX",
        "home_team_franchise": "BAL",
        "away_team_franchise": "JAX",
        "home_rest": 7,
        "away_rest": 7,
        "home_moneyline": home_line,
        "away_moneyline": away_line,
    }


def test_missing_moneyline_does_not_blank_a_priced_game() -> None:
    """A null or NaN line on one game leaves the priced game's market probability."""
    expected: float = no_vig_home_probability(390, -520)
    nullable = pd.DataFrame(
        [_upcoming_row("priced", 390, -520), _upcoming_row("open", pd.NA, pd.NA)]
    )
    nullable["home_moneyline"] = pd.Series([390, pd.NA], dtype="Int64")
    nullable["away_moneyline"] = pd.Series([-520, pd.NA], dtype="Int64")
    nullable_market = _market_by_game(slate_for_features(nullable))
    assert nullable_market["priced"] == pytest.approx(expected)
    assert nullable_market["open"] is None

    as_float = pd.DataFrame(
        [
            _upcoming_row("priced", 390.0, -520.0),
            _upcoming_row("open", float("nan"), float("nan")),
        ]
    )
    float_market = _market_by_game(slate_for_features(as_float))
    assert float_market["priced"] == pytest.approx(expected)
    assert float_market["open"] is None


def test_write_bundle_keeps_weeks_already_on_disk(tmp_path: Path) -> None:
    """A new week lists every non-example file already in the predictions directory."""
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    previous = {
        "sample": False,
        "season": 2026,
        "week": 8,
        "weeks_available": [8],
        "games": [],
    }
    example = {"sample": True, "season": 2026, "week": 7, "weeks_available": [7]}
    (predictions / "2026-w8.json").write_text(json.dumps(previous), encoding="utf-8")
    (predictions / "2026-w7.json").write_text(json.dumps(example), encoding="utf-8")
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
    latest = json.loads(Path(paths["latest"]).read_text(encoding="utf-8"))
    kept = json.loads((predictions / "2026-w8.json").read_text(encoding="utf-8"))
    untouched = json.loads((predictions / "2026-w7.json").read_text(encoding="utf-8"))
    assert latest["weeks_available"] == [8, 9]
    assert document["weeks_available"] == [8, 9]
    assert kept["weeks_available"] == [8, 9]
    assert untouched["sample"] is True
    assert untouched["weeks_available"] == [7]


def test_restore_skips_an_example_and_copies_real_weeks(tmp_path: Path) -> None:
    """The live example slate is not copied. Real week files are."""
    example = {"sample": True, "season": 2026, "week": 9, "weeks_available": [9]}
    skipped = restore_published_predictions(
        tmp_path / "example",
        "https://example.test/",
        fetch_text=lambda _url: json.dumps(example),
    )
    assert skipped["copied"] == 0
    assert not (tmp_path / "example" / "predictions").exists()

    real_week = {"sample": False, "season": 2026, "week": 8, "games": []}
    latest = {"sample": False, "season": 2026, "week": 9, "weeks_available": [8, 9]}
    pages = {
        "https://example.test/data/latest.json": json.dumps(latest),
        "https://example.test/data/predictions/2026-w8.json": json.dumps(real_week),
    }

    def fetch_text(url: str) -> str:
        if url not in pages:
            raise ValueError(f"HTTP 404 for {url}")
        return pages[url]

    copied = restore_published_predictions(tmp_path / "real", "https://example.test", fetch_text)
    assert copied["copied"] == 1
    stored = json.loads(
        (tmp_path / "real" / "predictions" / "2026-w8.json").read_text(encoding="utf-8")
    )
    assert stored["week"] == 8
    assert not (tmp_path / "real" / "predictions" / "2026-w9.json").exists()

    def fail(_url: str) -> str:
        raise ValueError("HTTP 404")

    failed = restore_published_predictions(tmp_path / "down", "https://example.test", fail)
    assert failed["copied"] == 0


def test_local_registry_failure_still_writes_the_week(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sqlite registry that cannot register still publishes the week JSON."""
    import mlflow

    def refuse_registry(*_args: object, **_kwargs: object) -> object:
        raise mlflow.MlflowException("registry unsupported")

    monkeypatch.setattr(
        "endzone.modeling.score.log_registered_sklearn_model",
        refuse_registry,
    )
    games, team_games, upcoming = _completed_and_slate()
    tracking_uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    settings = WeeklyScoreSettings(
        output_dir=tmp_path / "out",
        tracking_uri=tracking_uri,
        logistic_max_iter=200,
        generated_at=_FIXED_CLOCK,
    )
    try:
        result = score_upcoming_week(games, team_games, upcoming, _catalog(), settings)
    finally:
        _reset_mlflow_tracking()
    assert result["model_version"] == LOCAL_MODEL_VERSION
    assert result["document"]["games"][0]["model_version"] == LOCAL_MODEL_VERSION
    _reset_mlflow_tracking()


def test_databricks_registry_failure_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Databricks tracking URI still fails when registration fails."""
    import mlflow

    def refuse_registry(*_args: object, **_kwargs: object) -> object:
        raise mlflow.MlflowException("registry unsupported")

    monkeypatch.setattr(
        "endzone.modeling.score.log_registered_sklearn_model",
        refuse_registry,
    )
    with pytest.raises(mlflow.MlflowException):
        _register_weekly_model(
            object(),
            signature=None,
            input_example=pd.DataFrame(),
            model_name="endzone-winprob",
            tracking_uri="databricks",
        )
