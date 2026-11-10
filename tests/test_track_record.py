"""Track record math and the guarantee that published slates are not rewritten."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from endzone.site.track_record import (
    PROBABILITY_CLIP,
    SMALL_SAMPLE_DECIDED,
    TrackRecordError,
    assert_derived_output,
    binary_log_loss,
    build_track_record,
    clip_probability,
    main,
    metrics_from_games,
    pick_side,
    rounded_percent,
    sample_label,
    stamp_asset_urls,
    write_track_record,
)

_STAMP: str = "2026-11-09T15:04:00Z"


def _game(game_id: str, **overrides: Any) -> dict[str, Any]:
    """One finished game. Callers override the cells a test cares about."""
    game: dict[str, Any] = {
        "game_id": game_id,
        "kickoff": "2026-11-08",
        "home": "BBB",
        "away": "AAA",
        "model_home_wp": 0.7,
        "market_home_wp": 0.6,
        "diff": 0.1,
        "model_version": "local",
        "generated_at": "2026-11-09T04:28:37Z",
        "status": "final",
        "home_score": 24,
        "away_score": 17,
        "estimated_margin": 3.0,
        "estimated_total": 44.0,
        "estimated_home_points": 23.5,
        "estimated_away_points": 20.5,
    }
    game.update(overrides)
    return game


def _document(season: int, week: int, games: list[dict[str, Any]]) -> dict[str, Any]:
    """A minimal slate. ``José`` in a caller-supplied name stays UTF-8."""
    return {"sample": False, "season": season, "week": week, "games": games}


def _write(path: Path, document: dict[str, Any]) -> None:
    """Write one fixture as UTF-8 with LF newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _bytes(root: Path) -> dict[str, bytes]:
    """File bytes under ``root``, keyed by the relative path."""
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def _finished(game_id: str, **overrides: Any) -> dict[str, Any]:
    """The view dict ``metrics_from_games`` reads."""
    game: dict[str, Any] = {
        "model_home_wp": 0.7,
        "market_home_wp": 0.6,
        "status": "final",
        "home_score": 24,
        "away_score": 17,
        "estimated_margin": 7.0,
        "estimated_total": 41.0,
        "prediction_from_archive": True,
    }
    game.update(overrides)
    game["game_id"] = game_id
    return game


def test_percent_rounding_matches_the_page() -> None:
    """50.5 is a home pick. Python's banker round would call it a tie."""
    assert rounded_percent(0.505) == 51
    assert rounded_percent(0.495) == 50
    assert rounded_percent(0.485) == 49
    assert rounded_percent(0.5) == 50
    assert pick_side(0.505) == "home"
    assert pick_side(0.495) == "tie"
    assert pick_side(0.485) == "away"
    assert pick_side(0.5) == "tie"
    assert pick_side(None) is None
    assert pick_side(1.2) is None


def test_log_loss_clips_zero_and_one() -> None:
    """A stored 0 or 1 stays finite, and a value outside [0, 1] is dropped."""
    assert clip_probability(0.0) == PROBABILITY_CLIP
    assert clip_probability(1.0) == pytest.approx(1.0 - PROBABILITY_CLIP)
    assert clip_probability(1.2) is None
    assert clip_probability(None) is None
    assert clip_probability(float("nan")) is None
    certain_miss: float | None = binary_log_loss([1], [clip_probability(0.0) or 0.0])
    certain_hit: float | None = binary_log_loss([1], [clip_probability(1.0) or 0.0])
    assert certain_miss is not None and math.isfinite(certain_miss)
    assert certain_hit is not None and math.isfinite(certain_hit)
    assert certain_miss == pytest.approx(-math.log(PROBABILITY_CLIP))
    assert certain_hit == pytest.approx(-math.log(1.0 - PROBABILITY_CLIP))
    assert binary_log_loss([], []) is None
    metrics: dict[str, Any] = metrics_from_games(
        [_finished("g1", model_home_wp=0.0, market_home_wp=1.0, home_score=21, away_score=7)]
    )
    assert metrics["model_log_loss"] == pytest.approx(-math.log(PROBABILITY_CLIP), abs=1e-6)
    assert metrics["market_log_loss"] == pytest.approx(-math.log(1.0 - PROBABILITY_CLIP), abs=1e-6)
    assert math.isfinite(metrics["model_log_loss"])


def test_tie_push_and_unfinished_are_explicit() -> None:
    """A final tie is not a decided game. A scheduled game is unfinished."""
    tied: dict[str, Any] = metrics_from_games(
        [
            _finished(
                "tie",
                model_home_wp=0.8,
                market_home_wp=0.7,
                home_score=20,
                away_score=20,
                estimated_margin=1.0,
                estimated_total=40.0,
            )
        ]
    )
    assert tied["finished"] == 1
    assert tied["ties"] == 1
    assert tied["decided"] == 0
    assert tied["pick_hits"] == 0
    assert tied["pick_misses"] == 1
    assert tied["tie_hits"] == 0
    assert tied["model_log_loss"] is None
    assert tied["market_games"] == 0
    assert tied["margin_mae"] == pytest.approx(1.0)
    assert tied["total_mae"] == pytest.approx(0.0)
    called: dict[str, Any] = metrics_from_games(
        [_finished("called", model_home_wp=0.5, home_score=17, away_score=17)]
    )
    assert called["pick_hits"] == 1
    assert called["tie_hits"] == 1
    assert called["pick_misses"] == 0
    waiting: dict[str, Any] = metrics_from_games(
        [_finished("later", status="scheduled", home_score=None, away_score=None)]
    )
    assert waiting["finished"] == 0
    assert waiting["unfinished"] == 1
    assert waiting["pick_hits"] == 0
    assert waiting["model_log_loss"] is None
    assert waiting["margin_mae"] is None
    assert waiting["score_missing"] == 0
    missing_pick: dict[str, Any] = metrics_from_games([_finished("blank", model_home_wp=None)])
    assert missing_pick["pick_unavailable"] == 1
    assert missing_pick["pick_hits"] == 0
    assert missing_pick["pick_misses"] == 0
    assert (
        missing_pick["pick_hits"] + missing_pick["pick_misses"] + missing_pick["pick_unavailable"]
        == missing_pick["finished"]
    )


def test_market_log_loss_uses_only_games_that_have_a_price() -> None:
    """The model is scored on the same games as the market, not on a larger set."""
    priced: dict[str, Any] = _finished(
        "priced",
        model_home_wp=0.8,
        market_home_wp=0.25,
        home_score=10,
        away_score=20,
    )
    missing: dict[str, Any] = _finished(
        "missing",
        model_home_wp=0.1,
        market_home_wp=None,
        home_score=27,
        away_score=7,
    )
    metrics: dict[str, Any] = metrics_from_games([priced, missing])
    assert metrics["decided"] == 2
    assert metrics["market_games"] == 1
    assert metrics["market_missing"] == 1
    alone: dict[str, Any] = metrics_from_games([priced])
    assert metrics["model_log_loss"] == alone["model_log_loss"]
    assert metrics["market_log_loss"] == alone["market_log_loss"]
    assert metrics["model_log_loss"] != metrics["market_log_loss"]


def test_sample_label_marks_one_week_and_a_short_season() -> None:
    """One finished week is called out. 48 decided games is the next cut."""
    assert sample_label(0, 0) == "no_finals"
    assert sample_label(1, 14) == "one_week"
    assert sample_label(2, SMALL_SAMPLE_DECIDED - 1) == "small"
    assert sample_label(2, SMALL_SAMPLE_DECIDED) is None


def test_season_mean_is_pooled_across_games(tmp_path: Path) -> None:
    """Two weeks are averaged by game, not by averaging the weekly means."""
    archive: Path = tmp_path / "archive"
    _write(
        archive / "predictions" / "2026-w1.json",
        _document(
            2026,
            1,
            [
                _game(
                    "w1",
                    model_home_wp=0.9,
                    market_home_wp=0.4,
                    home_score=20,
                    away_score=10,
                    estimated_margin=10.0,
                    estimated_total=30.0,
                )
            ],
        ),
    )
    _write(
        archive / "predictions" / "2026-w2.json",
        _document(
            2026,
            2,
            [
                _game(
                    "w2a",
                    model_home_wp=0.6,
                    market_home_wp=0.55,
                    home_score=30,
                    away_score=10,
                    estimated_margin=11.0,
                    estimated_total=40.0,
                ),
                _game(
                    "w2b",
                    model_home_wp=0.6,
                    market_home_wp=0.55,
                    home_score=21,
                    away_score=10,
                    estimated_margin=20.0,
                    estimated_total=31.0,
                ),
            ],
        ),
    )
    document: dict[str, Any] = build_track_record(archive, None, generated_at=_STAMP)
    summary: dict[str, Any] = document["seasons"][0]["summary"]
    assert summary["weeks"] == 2
    assert summary["weeks_with_finals"] == 2
    assert summary["sample_label"] == "small"
    assert summary["margin_mae"] == pytest.approx(6.0)
    assert summary["total_mae"] == pytest.approx(0.0)
    weekly_mean: float = (-math.log(0.9) + -math.log(0.6)) / 2
    pooled: float = (-math.log(0.9) + 2 * -math.log(0.6)) / 3
    assert pooled != pytest.approx(weekly_mean)
    assert summary["model_log_loss"] == pytest.approx(pooled, abs=1e-6)


def test_partial_week_and_week_with_no_finals(tmp_path: Path) -> None:
    """A week in progress and a week with no finals both stay in the table."""
    archive: Path = tmp_path / "archive"
    _write(
        archive / "predictions" / "2026-w3.json",
        _document(
            2026,
            3,
            [
                _game("done", home_score=24, away_score=10),
                _game("later", status="scheduled", home_score=None, away_score=None),
            ],
        ),
    )
    _write(
        archive / "predictions" / "2026-w4.json",
        _document(
            2026,
            4,
            [_game("waiting", status="scheduled", home_score=None, away_score=None)],
        ),
    )
    document: dict[str, Any] = build_track_record(archive, None, generated_at=_STAMP)
    weeks: dict[int, dict[str, Any]] = {
        week["week"]: week for week in document["seasons"][0]["weeks"]
    }
    assert weeks[3]["state"] == "partial"
    assert weeks[3]["finished"] == 1
    assert weeks[3]["unfinished"] == 1
    assert weeks[3]["games"] == 2
    assert weeks[4]["state"] == "no_finals"
    assert weeks[4]["finished"] == 0
    assert weeks[4]["model_log_loss"] is None
    assert weeks[4]["margin_mae"] is None
    assert document["seasons"][0]["summary"]["sample_label"] == "one_week"
    assert document["seasons"][0]["summary"]["finished"] == 1


def test_earliest_archive_beats_a_later_rescore(tmp_path: Path) -> None:
    """Probabilities come from the first archived copy. Scores come from the newest."""
    archive: Path = tmp_path / "archive"
    site: Path = tmp_path / "site"
    original: dict[str, Any] = _game(
        "g1",
        model_home_wp=0.8,
        market_home_wp=0.6,
        status="scheduled",
        home_score=None,
        away_score=None,
        estimated_margin=3.0,
        estimated_total=40.0,
    )
    rescored: dict[str, Any] = _game(
        "g1",
        model_home_wp=0.2,
        market_home_wp=0.1,
        status="final",
        home_score=3,
        away_score=1,
        estimated_margin=50.0,
        estimated_total=10.0,
    )
    newest: dict[str, Any] = _game(
        "g1",
        model_home_wp=0.05,
        market_home_wp=0.05,
        status="final",
        home_score=30,
        away_score=10,
        estimated_margin=0.0,
        estimated_total=0.0,
    )
    added: dict[str, Any] = _game(
        "g2",
        model_home_wp=0.9,
        market_home_wp=0.4,
        home_score=17,
        away_score=14,
        estimated_margin=1.0,
        estimated_total=31.0,
    )
    _write(
        archive / "history" / "2026-w9" / "20261101T000000Z.json", _document(2026, 9, [original])
    )
    _write(archive / "predictions" / "2026-w9.json", _document(2026, 9, [rescored]))
    _write(site / "predictions" / "2026-w9.json", _document(2026, 9, [newest, added]))
    before_archive: dict[str, bytes] = _bytes(archive)
    before_site: dict[str, bytes] = _bytes(site)
    document: dict[str, Any] = build_track_record(archive, site, generated_at=_STAMP)
    assert _bytes(archive) == before_archive
    assert _bytes(site) == before_site
    week: dict[str, Any] = document["seasons"][0]["weeks"][0]
    assert week["finished"] == 2
    assert week["pick_hits"] == 2
    assert week["prediction_games_from_archive"] == 1
    assert week["prediction_games_from_this_build"] == 1
    # Actual margin is 20 from the site file. The estimate 3 is the earliest one.
    assert week["margin_mae"] == pytest.approx((abs(3.0 - 20.0) + abs(1.0 - 3.0)) / 2)


def test_missing_archive_does_not_score_the_site_files(tmp_path: Path) -> None:
    """An unreachable archive does not turn a site file into a track record."""
    site: Path = tmp_path / "site"
    _write(
        site / "predictions" / "2026-w9.json",
        _document(2026, 9, [_game("g1", home="José", model_home_wp=0.99)]),
    )
    before: dict[str, bytes] = _bytes(site)
    missing: dict[str, Any] = build_track_record(tmp_path / "missing", site, generated_at=_STAMP)
    assert missing["available"] is False
    assert missing["reason"] == "archive_unreachable"
    assert missing["seasons"] == []
    assert _bytes(site) == before
    empty: Path = tmp_path / "empty"
    empty.mkdir()
    first: dict[str, Any] = build_track_record(empty, site, generated_at=_STAMP)
    assert first["available"] is True
    assert first["seasons"][0]["weeks"][0]["prediction_games_from_this_build"] == 1
    assert _bytes(site) == before


@pytest.mark.cross_platform
def test_read_only_trees_stay_read_only(tmp_path: Path) -> None:
    """Building the record does not need write permission on the slates."""
    archive: Path = tmp_path / "archive"
    site: Path = tmp_path / "site"
    _write(archive / "predictions" / "2026-w9.json", _document(2026, 9, [_game("g1", home="José")]))
    _write(site / "predictions" / "2026-w9.json", _document(2026, 9, [_game("g1", home="José")]))
    before_archive: dict[str, bytes] = _bytes(archive)
    before_site: dict[str, bytes] = _bytes(site)
    locked: list[Path] = [archive, site]
    locked.extend(path for path in archive.rglob("*") if path.is_dir() or path.is_file())
    locked.extend(path for path in site.rglob("*") if path.is_dir() or path.is_file())
    for path in locked:
        path.chmod(0o555 if path.is_dir() else 0o444)
    output: Path = tmp_path / "out" / "track_record.json"
    try:
        document: dict[str, Any] = build_track_record(archive, site, generated_at=_STAMP)
        write_track_record(document, output)
    finally:
        for path in locked:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert _bytes(archive) == before_archive
    assert _bytes(site) == before_site
    assert "José" in before_archive["predictions/2026-w9.json"].decode("utf-8")
    assert b"\r\n" not in output.read_bytes()
    assert document["seasons"][0]["weeks"][0]["games"] == 1


@pytest.mark.cross_platform
def test_writer_keeps_utf8_and_refuses_a_published_name(tmp_path: Path) -> None:
    """The derived file is UTF-8. A week file name is never the output."""
    output: Path = tmp_path / "track_record.json"
    write_track_record({"note": "José", "available": False, "seasons": []}, output)
    raw: bytes = output.read_bytes()
    assert b"\r\n" not in raw
    assert "José" in raw.decode("utf-8")
    assert "\\u00e9" not in raw.decode("utf-8")
    week: Path = tmp_path / "predictions" / "2026-w9.json"
    _write(week, _document(2026, 9, [_game("g1")]))
    original: bytes = week.read_bytes()
    with pytest.raises(TrackRecordError, match="published slate"):
        assert_derived_output(week, None, tmp_path)
    assert week.read_bytes() == original
    archive: Path = tmp_path / "archive"
    archive.mkdir()
    with pytest.raises(TrackRecordError, match="inside the archive"):
        assert_derived_output(archive / "track_record.json", archive, None)


@pytest.mark.cross_platform
def test_corrupt_week_removes_a_stale_record_and_writes_nothing(tmp_path: Path) -> None:
    """A bad archived file fails closed. An old track record is not left behind."""
    archive: Path = tmp_path / "archive"
    week: Path = archive / "predictions" / "2026-w9.json"
    week.parent.mkdir(parents=True)
    week.write_text("{", encoding="utf-8", newline="\n")
    output: Path = tmp_path / "site" / "track_record.json"
    output.parent.mkdir()
    output.write_text('{"available": true}\n', encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--archive-dir",
                str(archive),
                "--site-data",
                str(tmp_path / "site"),
                "--output",
                str(output),
                "--generated-at",
                _STAMP,
            ]
        )
    assert raised.value.code == 1
    assert not output.exists()
    assert week.read_text(encoding="utf-8") == "{"
    history: Path = archive / "history" / "2026-w9" / "20261101T000000Z.json"
    _write(week, _document(2026, 9, [_game("g1")]))
    history.parent.mkdir(parents=True)
    history.write_text("not-json", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit):
        main(
            [
                "--archive-dir",
                str(archive),
                "--site-data",
                str(tmp_path / "site"),
                "--output",
                str(output),
                "--generated-at",
                _STAMP,
            ]
        )
    assert not output.exists()
    assert week.read_bytes() == _bytes(archive)["predictions/2026-w9.json"]


@pytest.mark.cross_platform
def test_command_writes_only_the_derived_file(tmp_path: Path) -> None:
    """The command's only new file is the track record."""
    archive: Path = tmp_path / "archive"
    site: Path = tmp_path / "site"
    _write(archive / "predictions" / "2026-w9.json", _document(2026, 9, [_game("g1", home="José")]))
    _write(
        site / "predictions" / "2026-w9.json",
        _document(2026, 9, [_game("g1", model_home_wp=0.01, home_score=10, away_score=30)]),
    )
    output: Path = site / "track_record.json"
    before_archive: dict[str, bytes] = _bytes(archive)
    before_predictions: bytes = (site / "predictions" / "2026-w9.json").read_bytes()
    result: dict[str, Any] = main(
        [
            "--archive-dir",
            str(archive),
            "--site-data",
            str(site),
            "--output",
            str(output),
            "--generated-at",
            _STAMP,
        ]
    )
    assert result["available"] is True
    assert _bytes(archive) == before_archive
    assert (site / "predictions" / "2026-w9.json").read_bytes() == before_predictions
    saved: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    # Site scores are an away win. The archived 0.7 pick is home, so this is a miss.
    assert saved["seasons"][0]["weeks"][0]["pick_hits"] == 0
    assert saved["method"] == "ADR-008"


def test_sample_slate_and_mismatched_week_are_not_silent(tmp_path: Path) -> None:
    """An example file is skipped. A file that names the wrong week fails."""
    archive: Path = tmp_path / "archive"
    example: dict[str, Any] = _document(2026, 1, [_game("ex")])
    example["sample"] = True
    _write(archive / "predictions" / "2026-w1.json", example)
    document: dict[str, Any] = build_track_record(archive, None, generated_at=_STAMP)
    assert document["available"] is True
    assert document["seasons"] == []
    _write(
        archive / "predictions" / "2026-w2.json",
        _document(2026, 9, [_game("nope")]),
    )
    with pytest.raises(TrackRecordError, match="expected 2026 week 2"):
        build_track_record(archive, None, generated_at=_STAMP)


@pytest.mark.cross_platform
def test_stamp_rewrites_asset_urls_and_not_json(tmp_path: Path) -> None:
    """A content hash is added once. A week file beside the HTML is untouched."""
    asset: Path = tmp_path / "assets" / "app.js"
    asset.parent.mkdir(parents=True)
    asset.write_text("console.log('José');\n", encoding="utf-8", newline="\n")
    index: Path = tmp_path / "index.html"
    index.write_text(
        '<link rel="stylesheet" href="assets/missing.css">\n'
        '<script src="assets/app.js" defer></script>\n',
        encoding="utf-8",
        newline="\n",
    )
    week: Path = tmp_path / "data" / "predictions" / "2026-w9.json"
    _write(week, _document(2026, 9, [_game("g1")]))
    original: bytes = week.read_bytes()
    stamped: dict[str, str] = stamp_asset_urls(index)
    text: str = index.read_text(encoding="utf-8")
    assert stamped["assets/app.js"]
    assert f"assets/app.js?v={stamped['assets/app.js']}" in text
    assert "assets/missing.css" in text
    assert "assets/missing.css?" not in text
    again: dict[str, str] = stamp_asset_urls(index)
    assert again == {}
    assert index.read_text(encoding="utf-8").count("?v=") == 1
    assert week.read_bytes() == original


_OG_IMAGE_URL: str = "https://carloshgalvan95.github.io/endzone-mlops/assets/og-image.png"


def _png_size(payload: bytes) -> tuple[int, int]:
    """Return width and height from a PNG IHDR chunk."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise AssertionError("og-image.png is not a PNG")
    width: int = int.from_bytes(payload[16:20], "big")
    height: int = int.from_bytes(payload[20:24], "big")
    return width, height


def test_open_graph_image_stays_unhashed_in_the_built_site(tmp_path: Path) -> None:
    """Pages hashes CSS and JS. The Open Graph file keeps a stable absolute URL.

    The Pages artifact is the site directory after ``stamp_asset_urls``.
    """
    source_root: Path = Path("site")
    source_image: Path = source_root / "assets" / "og-image.png"
    source_bytes: bytes = source_image.read_bytes()
    assert _png_size(source_bytes) == (1200, 627)

    built: Path = tmp_path / "site"
    assets: Path = built / "assets"
    assets.mkdir(parents=True)
    index: Path = built / "index.html"
    index.write_text(
        (source_root / "index.html").read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    for asset in (source_root / "assets").iterdir():
        if asset.is_file():
            (assets / asset.name).write_bytes(asset.read_bytes())

    stamped: dict[str, str] = stamp_asset_urls(index)
    html: str = index.read_text(encoding="utf-8")
    names: list[str] = sorted(path.name for path in assets.glob("og-image*.png"))

    assert names == ["og-image.png"]
    assert (assets / "og-image.png").read_bytes() == source_bytes
    assert f'property="og:image" content="{_OG_IMAGE_URL}"' in html
    assert f'name="twitter:image" content="{_OG_IMAGE_URL}"' in html
    assert f"{_OG_IMAGE_URL}?" not in html
    assert "og-image.png?v=" not in html
    assert "assets/og-image.png" not in stamped
    assert "assets/styles.css?v=" in html
    assert "assets/app.js?v=" in html
