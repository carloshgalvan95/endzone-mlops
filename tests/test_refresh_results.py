"""Final scores overlay onto published week JSON, and the page's side mapping."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from endzone.modeling.weekly import build_prediction_document, validate_prediction_document
from endzone.site.results import (
    apply_final_scores,
    build_parser,
    final_scores_by_game_id,
    main,
    refresh_results,
)

_STAMP = "2026-11-09T08:30:00Z"
_GENERATED = "2026-11-04T12:00:00Z"
_FROZEN_GAME = (
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
_FROZEN_DOCUMENT = (
    "sample",
    "season",
    "week",
    "generated_at",
    "model_name",
    "model_version",
    "feature_set",
)


def _document(
    games: list[dict[str, object]],
    *,
    week: int = 9,
    weeks: list[int] | None = None,
) -> dict[str, Any]:
    """Build a valid scheduled week, then replace the game rows."""
    frame = pd.DataFrame(
        [
            {
                "game_id": game["game_id"],
                "gameday": "2026-11-08",
                "gametime": "13:00",
                "home_team": game["home"],
                "away_team": game["away"],
            }
            for game in games
        ]
    )
    market: dict[str, float | None] = {
        str(game["game_id"]): game.get("market", 0.5) for game in games
    }
    document = build_prediction_document(
        season=2026,
        week=week,
        games=frame,
        model_probabilities=pd.Series(
            {str(game["game_id"]): float(game.get("model", 0.6)) for game in games}
        ),
        market_probabilities=market,
        model_name="endzone-winprob",
        model_version="3",
        feature_set="h2_full",
        generated_at=_GENERATED,
    )
    document["weeks_available"] = weeks if weeks is not None else [week]
    by_id = {str(game["game_id"]): game for game in games}
    for record in document["games"]:
        extra = by_id[record["game_id"]]
        for key in ("status", "home_score", "away_score", "result", "home_won"):
            if key in extra:
                record[key] = extra[key]
    return document


def _write_games(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _week_games() -> list[dict[str, object]]:
    return [
        {"game_id": "G_HOME", "home": "BUF", "away": "MIA", "model": 0.58, "market": 0.55},
        {"game_id": "G_AWAY", "home": "LAR", "away": "SEA", "model": 0.47, "market": None},
        {"game_id": "G_TIE", "home": "DET", "away": "GB", "model": 0.55, "market": 0.53},
        {"game_id": "G_ZERO", "home": "NYJ", "away": "NE", "model": 0.5, "market": 0.5},
        {"game_id": "G_OPEN", "home": "KC", "away": "DEN", "model": 0.71, "market": 0.72},
        {"game_id": "G_MISSING", "home": "PHI", "away": "DAL", "model": 0.61, "market": 0.6},
    ]


def _games_rows() -> list[dict[str, object]]:
    return [
        {"game_id": "G_HOME", "season": 2026, "home_score": 24.0, "away_score": 17.0},
        {"game_id": "G_AWAY", "season": 2026, "home_score": 20.0, "away_score": 28.0},
        {"game_id": "G_TIE", "season": 2026, "home_score": 20.0, "away_score": 20.0},
        {"game_id": "G_ZERO", "season": 2026, "home_score": 0.0, "away_score": 0.0},
        {"game_id": "G_OPEN", "season": 2026, "home_score": None, "away_score": None},
        {"game_id": "G_EXTRA", "season": 2026, "home_score": 3, "away_score": 2},
        {"game_id": "G_OLD", "season": 2024, "home_score": 40, "away_score": 0},
    ]


def test_overlay_sets_status_and_leaves_predictions_frozen() -> None:
    """Finished games get integer scores. Unplayed games stay scheduled."""
    original = _document(_week_games())
    before = json.loads(json.dumps(original))
    finals = final_scores_by_game_id(pd.DataFrame(_games_rows()))
    updated = apply_final_scores(original, finals, _STAMP)
    validate_prediction_document(updated)
    assert original == before
    for key in _FROZEN_DOCUMENT:
        assert updated[key] == before[key]
    assert updated["results_updated_at"] == _STAMP
    assert updated["generated_at"] == _GENERATED
    by_id = {game["game_id"]: game for game in updated["games"]}
    assert list(by_id) == [game["game_id"] for game in before["games"]]
    assert "G_EXTRA" not in by_id

    home = by_id["G_HOME"]
    assert home["status"] == "final"
    assert home["home_score"] == 24
    assert home["away_score"] == 17
    assert type(home["home_score"]) is int
    assert type(home["away_score"]) is int
    assert home["result"] == "home"
    assert "home_won" not in home
    assert "est_home_score" not in home
    assert "est_away_score" not in home

    away = by_id["G_AWAY"]
    assert away["status"] == "final"
    assert away["result"] == "away"
    assert away["home_score"] == 20
    assert away["away_score"] == 28

    tie = by_id["G_TIE"]
    assert tie["status"] == "final"
    assert tie["result"] == "tie"
    assert tie["home_score"] == tie["away_score"] == 20

    zero = by_id["G_ZERO"]
    assert zero["status"] == "final"
    assert zero["home_score"] == 0
    assert zero["away_score"] == 0
    assert zero["result"] == "tie"

    for game_id in ("G_OPEN", "G_MISSING"):
        game = by_id[game_id]
        assert game["status"] == "scheduled"
        assert "home_score" not in game
        assert "away_score" not in game
        assert "result" not in game
        assert "home_won" not in game

    for game in updated["games"]:
        previous = next(item for item in before["games"] if item["game_id"] == game["game_id"])
        for key in _FROZEN_GAME:
            assert game[key] == previous[key]


def test_scores_without_final_status_do_not_become_a_result() -> None:
    """A score pair is not a final unless status is already final or nflverse says so."""
    document = _document(
        [
            {
                "game_id": "G_BARE",
                "home": "BUF",
                "away": "MIA",
                "model": 0.6,
                "market": 0.55,
                "status": "scheduled",
                "home_score": 10,
                "away_score": 7,
            }
        ]
    )
    updated = apply_final_scores(document, {}, _STAMP)
    game = updated["games"][0]
    assert game["status"] == "scheduled"
    assert "home_score" not in game
    assert "away_score" not in game
    validate_prediction_document(updated)


def test_existing_final_is_kept_when_nflverse_has_no_row() -> None:
    """A published final stays when the schedules file has no finished row."""
    document = _document(
        [
            {
                "game_id": "G_KEPT",
                "home": "BUF",
                "away": "MIA",
                "model": 0.6,
                "market": 0.55,
                "status": "final",
                "home_score": 21,
                "away_score": 14,
                "home_won": True,
            }
        ]
    )
    updated = apply_final_scores(document, {}, _STAMP)
    game = updated["games"][0]
    assert game["status"] == "final"
    assert game["home_score"] == 21
    assert game["away_score"] == 14
    assert game["result"] == "home"
    assert "home_won" not in game
    assert game["model_home_wp"] == document["games"][0]["model_home_wp"]
    validate_prediction_document(updated)


def test_a_later_correction_replaces_scores_only() -> None:
    """An nflverse correction updates the scoreline and leaves the prediction time."""
    document = _document(
        [{"game_id": "G_HOME", "home": "BUF", "away": "MIA", "model": 0.58, "market": 0.55}]
    )
    first = apply_final_scores(
        document,
        final_scores_by_game_id(
            pd.DataFrame(
                [{"game_id": "G_HOME", "season": 2026, "home_score": 24, "away_score": 17}]
            )
        ),
        _STAMP,
    )
    second = apply_final_scores(
        first,
        final_scores_by_game_id(
            pd.DataFrame(
                [{"game_id": "G_HOME", "season": 2026, "home_score": 27, "away_score": 17}]
            )
        ),
        "2026-11-10T08:30:00Z",
    )
    game = second["games"][0]
    assert game["home_score"] == 27
    assert game["result"] == "home"
    assert game["generated_at"] == _GENERATED
    assert second["generated_at"] == _GENERATED
    assert second["results_updated_at"] == "2026-11-10T08:30:00Z"
    assert game["model_home_wp"] == first["games"][0]["model_home_wp"]


def test_validation_accepts_final_scores_and_rejects_a_bad_status() -> None:
    """Optional result must match the scoreline. Scheduled games cannot carry scores."""
    final_doc = _document(
        [
            {
                "game_id": "G_HOME",
                "home": "BUF",
                "away": "MIA",
                "model": 0.58,
                "market": 0.55,
                "status": "final",
                "home_score": 0,
                "away_score": 0,
                "result": "tie",
            }
        ]
    )
    validate_prediction_document(final_doc)
    assert "home_won" not in final_doc["games"][0]

    missing_result = _document(
        [
            {
                "game_id": "G_HOME",
                "home": "BUF",
                "away": "MIA",
                "model": 0.58,
                "market": 0.55,
                "status": "final",
                "home_score": 3,
                "away_score": 1,
            }
        ]
    )
    validate_prediction_document(missing_result)

    scheduled_with_scores = _document(
        [
            {
                "game_id": "G_HOME",
                "home": "BUF",
                "away": "MIA",
                "model": 0.58,
                "market": 0.55,
                "status": "scheduled",
                "home_score": 3,
                "away_score": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="must not include scores"):
        validate_prediction_document(scheduled_with_scores)

    wrong_result = _document(
        [
            {
                "game_id": "G_HOME",
                "home": "BUF",
                "away": "MIA",
                "model": 0.58,
                "market": 0.55,
                "status": "final",
                "home_score": 3,
                "away_score": 1,
                "result": "away",
            }
        ]
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_prediction_document(wrong_result)

    no_status = _document(
        [{"game_id": "G_HOME", "home": "BUF", "away": "MIA", "model": 0.58, "market": 0.55}]
    )
    del no_status["games"][0]["status"]
    with pytest.raises(ValueError, match="status"):
        validate_prediction_document(no_status)


def test_refresh_writes_week_files_and_skips_examples(tmp_path: Path) -> None:
    """Example files stay byte-for-byte. latest.json is copied from the newest week."""
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    week9 = _document(
        _week_games()
        + [{"game_id": "G_OLD", "home": "PIT", "away": "CLE", "model": 0.52, "market": 0.51}],
        week=9,
        weeks=[8, 9],
    )
    week8 = _document(
        [
            {"game_id": "G_TIE", "home": "DET", "away": "GB", "model": 0.55, "market": 0.53},
            {"game_id": "G_AWAY", "home": "LAR", "away": "SEA", "model": 0.47, "market": 0.4},
        ],
        week=8,
        weeks=[8, 9],
    )
    (predictions / "2026-w9.json").write_text(json.dumps(week9), encoding="utf-8")
    (predictions / "2026-w8.json").write_text(json.dumps(week8), encoding="utf-8")
    example = {"sample": True, "season": 2026, "week": 7, "games": [{"game_id": "EXAMPLE-01"}]}
    example_path = predictions / "2026-w7.json"
    example_path.write_text(json.dumps(example), encoding="utf-8")
    notes_path = predictions / "notes.json"
    notes_path.write_text("{}\n", encoding="utf-8")
    sample_path = tmp_path / "sample.json"
    sample_path.write_text('{"sample": true}\n', encoding="utf-8")
    games_path = tmp_path / "games.parquet"
    _write_games(games_path, _games_rows())

    example_bytes = example_path.read_bytes()
    notes_bytes = notes_path.read_bytes()
    sample_bytes = sample_path.read_bytes()
    result = refresh_results(tmp_path, games_path=games_path, results_updated_at=_STAMP)

    assert example_path.read_bytes() == example_bytes
    assert notes_path.read_bytes() == notes_bytes
    assert sample_path.read_bytes() == sample_bytes
    assert result["skipped_examples"] == 1
    assert result["seasons"] == [2026]
    assert result["updated"] == 3
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    stored9 = json.loads((predictions / "2026-w9.json").read_text(encoding="utf-8"))
    stored8 = json.loads((predictions / "2026-w8.json").read_text(encoding="utf-8"))
    assert latest == stored9
    assert latest["week"] == 9
    assert stored9["games"][0]["status"] == "final"
    old = next(game for game in stored9["games"] if game["game_id"] == "G_OLD")
    assert old["status"] == "scheduled"
    assert "home_score" not in old
    assert stored8["games"][0]["result"] == "tie"
    text = (predictions / "2026-w9.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  " in text
    assert "NaN" not in text
    validate_prediction_document(stored9)
    validate_prediction_document(stored8)


def test_conflicting_finals_and_an_empty_directory(tmp_path: Path) -> None:
    """Two different finals for one id fail. An empty directory does not download."""
    with pytest.raises(ValueError, match="Conflicting"):
        final_scores_by_game_id(
            pd.DataFrame(
                [
                    {"game_id": "G", "home_score": 1, "away_score": 0},
                    {"game_id": "G", "home_score": 2, "away_score": 0},
                ]
            )
        )
    with pytest.raises(ValueError, match="home_score"):
        final_scores_by_game_id(pd.DataFrame([{"game_id": "G", "season": 2026}]))

    def boom(_url: str) -> pd.DataFrame:
        raise AssertionError("download")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("endzone.site.results.download_parquet_with_validation", boom)
    try:
        with pytest.raises(ValueError, match="No published week"):
            refresh_results(tmp_path, results_updated_at=_STAMP)
    finally:
        monkeypatch.undo()


@pytest.mark.cross_platform
def test_cli_restores_then_overlays_a_local_games_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command copies published weeks before it reads a local games file."""
    games_path = tmp_path / "games.parquet"
    _write_games(
        games_path,
        [{"game_id": "G_HOME", "season": 2026, "home_score": 24, "away_score": 17}],
    )
    output = tmp_path / "out"

    def fake_restore(output_dir: Path, base_url: str, fetch_text: object = None) -> dict[str, int]:
        assert base_url == "https://example.test"
        assert fetch_text is None
        predictions = Path(output_dir) / "predictions"
        predictions.mkdir(parents=True)
        document = _document(
            [{"game_id": "G_HOME", "home": "BUF", "away": "MIA", "model": 0.58, "market": 0.55}]
        )
        (predictions / "2026-w9.json").write_text(json.dumps(document), encoding="utf-8")
        return {"copied": 1}

    monkeypatch.setattr("endzone.site.results.restore_published_predictions", fake_restore)
    monkeypatch.setattr(
        "endzone.site.results.download_parquet_with_validation",
        lambda _url: (_ for _ in ()).throw(AssertionError("download")),
    )
    result = main(
        [
            "--output-dir",
            str(output),
            "--restore-published",
            "https://example.test",
            "--games-path",
            str(games_path),
            "--results-updated-at",
            _STAMP,
        ]
    )
    assert result["updated"] == 2
    stored = json.loads((output / "predictions" / "2026-w9.json").read_text(encoding="utf-8"))
    assert stored["games"][0]["status"] == "final"
    assert stored["games"][0]["result"] == "home"
    assert (
        build_parser().parse_args(["--games-path", "games.parquet"]).games_path == "games.parquet"
    )


def test_results_module_does_not_score_or_drop_finished_games() -> None:
    """The refresh path never selects an unplayed slate or fits the model."""
    source = Path("endzone/site/results.py").read_text(encoding="utf-8")
    assert "select_slate(" not in source
    assert "load_upcoming_games(" not in source
    assert "endzone.modeling.score" not in source
    workflow = Path(".github/workflows/refresh-results.yml").read_text(encoding="utf-8")
    folded = workflow.lower()
    assert "databricks" not in folded
    assert "dbt" not in folded
    assert "endzone.modeling.score" not in workflow
    assert "endzone.site.results" in workflow
    assert "workflow_dispatch" in workflow
    assert "contents: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "group: pages" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "uv sync --frozen" in workflow
    assert "path: site" in workflow
    pages = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert pages.index("endzone.modeling.score") < pages.index("endzone.site.results")


@pytest.mark.cross_platform
def test_frontend_pick_and_result_sides() -> None:
    """The page maps the frozen pick and the final winner from the same rules."""
    node = shutil.which("node")
    assert node is not None
    script = r"""
const view = require("./site/assets/app.js");
const cases = [
  [view.pickSide(0.71), "home"],
  [view.pickSide(0.42), "away"],
  [view.pickSide(0.5), "tie"],
  [view.pickSide(0.495), "tie"],
  [view.resultSide({status: "scheduled", home_score: 24, away_score: 17}), null],
  [view.resultSide({status: "final", home_score: 0, away_score: 0}), "tie"],
  [view.resultSide({status: "final", home_score: 24, away_score: 17}), "home"],
  [view.resultSide({status: "final", home_score: 14, away_score: 17}), "away"],
  [view.resultSide({home_score: 24, away_score: 17}), null],
  [view.marketCaption(0.7, "home", "BUF", "MIA"), "favored BUF \u00b7 matched"],
  [view.marketCaption(0.46, "home", "LAR", "SEA"), "favored SEA \u00b7 did not match"],
  [view.marketCaption(0.5, "tie", "DET", "GB"), "market even"],
  [view.marketCaption(null, "home", "BUF", "MIA"), ""],
  [view.pickMatched("home", "home"), true],
  [view.pickMatched("away", "away"), true],
  [view.pickMatched("tie", "tie"), true],
  [view.pickMatched("home", "tie"), false],
  [view.pickMatched("tie", "away"), false],
  [view.pickMatched("away", "home"), false],
  [view.pickMatched("home", null), null],
  [view.estimateLabel(17, 24), "est. 17-24"],
  [view.estimateLabel(17.4, 23.6), "est. 17-24"],
  [view.estimateLabel(null, 24), ""],
  [view.estimateLabel(17, null), ""]
];
for (const [got, expected] of cases) {
  if (got !== expected) {
    console.error(JSON.stringify({got, expected}));
    process.exit(1);
  }
}
"""
    completed = subprocess.run(
        [node, "-e", script],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.cross_platform
def test_frontend_estimate_sits_beside_the_final_score() -> None:
    """A final estimate sits on the score row. The date line stays free of est.

    The report leaves Node as ASCII JSON. A Windows code page must not turn
    the page's middle dot into a different character before Python reads it.
    """
    node = shutil.which("node")
    assert node is not None
    script = r"""
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.className = "";
    this.children = [];
    this.attributes = {};
    this.hidden = false;
    this.value = "";
    this.selected = false;
    this._text = "";
    this.style = { setProperty() {} };
    const self = this;
    this.classList = {
      add(name) {
        const parts = self.className.split(/\s+/).filter(Boolean);
        if (!parts.includes(name)) parts.push(name);
        self.className = parts.join(" ");
      },
      remove(name) {
        self.className = self.className.split(/\s+/).filter((part) => part && part !== name).join(" ");
      }
    };
  }
  get textContent() {
    if (this.children.length) return this.children.map((child) => child.textContent || "").join("");
    return this._text;
  }
  set textContent(value) {
    this.children = [];
    this._text = value == null ? "" : String(value);
  }
  appendChild(child) {
    this.children.push(child);
    this._text = "";
    return child;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener() {}
  remove() {}
  get options() { return this.children.filter((child) => child.tagName === "OPTION"); }
}
const byId = {};
function mount(id, tag) {
  const node = new El(tag);
  node.id = id;
  byId[id] = node;
  return node;
}
["games-body", "games-cards", "thr-text", "accuracy-block", "accuracy-note", "season-kicker", "week-title", "example-flag"].forEach((id) => mount(id, "div"));
mount("week-select", "select");
mount("thr-select", "select");
global.document = {
  createElement: (tag) => new El(tag),
  createElementNS: (_ns, tag) => new El(tag),
  createTextNode: (text) => {
    const node = new El("#text");
    node._text = String(text);
    return node;
  },
  getElementById: (id) => byId[id] || null
};
const slate = {
  sample: false,
  season: 2026,
  week: 9,
  weeks_available: [9],
  threshold_pp: 5,
  score_model: { hidden_reason: null, margin: {}, total: {} },
  games: [
    {
      kickoff: "2026-11-08T16:25:00-05:00",
      status: "final",
      home: "IND",
      away: "DAL",
      home_score: 23,
      away_score: 27,
      model_home_wp: 0.38,
      market_home_wp: 0.34,
      estimated_home_points: 23.2,
      estimated_away_points: 26.6
    },
    {
      kickoff: "2026-11-08T13:00:00-05:00",
      status: "final",
      home: "CAR",
      away: "DEN",
      home_score: 21,
      away_score: 26,
      model_home_wp: 0.46,
      market_home_wp: 0.42,
      estimated_home_points: null,
      estimated_away_points: null
    },
    {
      kickoff: "2026-11-08T20:20:00-05:00",
      status: "scheduled",
      home: "CHI",
      away: "TB",
      home_score: null,
      away_score: null,
      model_home_wp: 0.42,
      market_home_wp: 0.48,
      estimated_home_points: 24.6,
      estimated_away_points: 18.4
    }
  ]
};
global.fetch = (url) => {
  const key = String(url);
  const payload = key.indexOf("teams") !== -1 ? { teams: {} } : slate;
  return Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
};
require("./site/assets/app.js");
function tokens(className) { return String(className || "").split(/\s+/).filter(Boolean); }
function hasClass(node, name) { return tokens(node.className).includes(name); }
function findAll(node, name, out) {
  out = out || [];
  if (node && hasClass(node, name)) out.push(node);
  (node && node.children || []).forEach((child) => findAll(child, name, out));
  return out;
}
function summarize(root) {
  const lines = findAll(root, "score-line");
  return {
    lines: lines.map((line) => line.children.map((child) => child.className)),
    estimates: lines.map((line) => {
      const est = (line.children || []).find((child) => hasClass(child, "est-line"));
      return est ? { className: est.className, text: est.textContent, hidden: est.attributes["aria-hidden"] || "" } : null;
    }),
    meta: findAll(root, "result-meta").map((meta) => ({
      text: meta.textContent,
      estCount: findAll(meta, "est-line").length
    })),
    pending: findAll(root, "est-score").map((node) => node.textContent)
  };
}
setTimeout(() => {
  const report = {
    table: summarize(byId["games-body"]),
    cards: summarize(byId["games-cards"])
  };
  process.stdout.write(JSON.stringify(report).replace(/[^\u0000-\u007F]/g, function (ch) {
    return "\\u" + ("0000" + ch.charCodeAt(0).toString(16)).slice(-4);
  }));
  process.exit(0);
}, 50);
"""
    completed = subprocess.run(
        [node, "-e", script],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.isascii()
    report = json.loads(completed.stdout)
    expected_order = ["score-stack", "result-mark", "est-line"]
    for surface in ("table", "cards"):
        lines = report[surface]["lines"]
        assert lines[0] == expected_order
        assert lines[1] == ["score-stack", "result-mark", "est-line is-placeholder"]
        assert len(lines) == 2
        shown = report[surface]["estimates"][0]
        assert shown["text"] == "est. 27-23"
        assert "\u2013" not in shown["text"]
        assert "\u2014" not in shown["text"]
        hidden = report[surface]["estimates"][1]
        assert hidden["text"] == ""
        assert hidden["hidden"] == "true"
        assert "is-placeholder" in hidden["className"]
        for meta in report[surface]["meta"]:
            assert meta["text"].startswith("Final")
            assert "Sun 8 Nov" in meta["text"]
            assert "est." not in meta["text"]
            assert meta["estCount"] == 0
        assert report[surface]["pending"] == ["TB 18 \u00b7 25 CHI"]
