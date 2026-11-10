"""Track-record alert: the gap, the trip rule, and one deduplicated issue."""

from __future__ import annotations

import io
import json
import math
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from endzone.modeling.reference import reference_market_gap
from endzone.site.model_alert import (
    ALERT_LABEL,
    ALERT_TITLE,
    DEFAULT_GAP_MARGIN,
    DEFAULT_MIN_GAMES,
    DEFAULT_WINDOW_GAMES,
    GithubIssueClient,
    ModelAlertError,
    alert_issue_body,
    evaluate_market_gap,
    main,
    sync_alert_issue,
)
from endzone.site.track_record import TrackRecordError, finished_market_games


def _row(
    label: int,
    model: float,
    market: float,
    *,
    season: int = 2026,
    week: int = 9,
    game_id: str = "g1",
) -> dict[str, Any]:
    """One decided game in the shape ``evaluate_market_gap`` reads."""
    return {
        "label": label,
        "model_probability": model,
        "market_probability": market,
        "season": season,
        "week": week,
        "game_id": game_id,
    }


class _Issues:
    """In-memory stand-in for the GitHub client."""

    def __init__(self, open_items: list[dict[str, Any]] | None = None) -> None:
        self.items: list[dict[str, Any]] = list(open_items or [])
        self.created: list[dict[str, Any]] = []
        self.updated: list[int] = []
        self.labels: list[str] = []

    def ensure_label(self, name: str, color: str, description: str) -> None:
        self.labels.append(name)

    def list_open(self, label: str) -> list[dict[str, Any]]:
        return [item for item in self.items if item.get("state", "open") == "open"]

    def create(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        number: int = len(self.items) + 1
        item: dict[str, Any] = {
            "number": number,
            "title": title,
            "body": body,
            "state": "open",
            "labels": labels,
        }
        self.items.append(item)
        self.created.append(item)
        return item

    def update(self, number: int, title: str, body: str) -> dict[str, Any]:
        for item in self.items:
            if item["number"] == number:
                item["title"] = title
                item["body"] = body
                self.updated.append(number)
                return item
        raise AssertionError(f"missing issue {number}")


def _tripped() -> dict[str, Any]:
    """A two-game window whose model is far from the results and the market is not."""
    games: list[dict[str, Any]] = [
        _row(1, 0.2, 0.9, game_id="a"),
        _row(0, 0.8, 0.1, game_id="b", week=10),
    ]
    decision = evaluate_market_gap(
        games,
        window_games=4,
        min_games=2,
        gap_margin=0.02,
        reference_gap=0.0,
    )
    assert decision["tripped"] is True
    return decision


def test_gap_trips_only_on_the_last_window_and_only_above_the_sample() -> None:
    """The last N games count. A short window and a negative gap do not trip."""
    tripped = _tripped()
    assert tripped["reason"] == "log_loss_gap"
    assert tripped["n_games"] == 2
    assert tripped["gap"] > 0.02
    quiet = evaluate_market_gap(
        [_row(1, 0.7, 0.7), _row(0, 0.3, 0.3, week=10)],
        window_games=4,
        min_games=2,
        gap_margin=0.02,
        reference_gap=0.0,
    )
    assert quiet["tripped"] is False
    assert quiet["reason"] == "within_gap"
    short = evaluate_market_gap(
        [_row(1, 0.2, 0.9)],
        window_games=4,
        min_games=2,
        gap_margin=0.02,
        reference_gap=0.0,
    )
    assert short["tripped"] is False
    assert short["reason"] == "below_minimum_sample"
    early_bad = [
        _row(1, 0.1, 0.9, week=1, game_id="old-a"),
        _row(0, 0.9, 0.1, week=1, game_id="old-b"),
        _row(1, 0.7, 0.7, week=2, game_id="new-a"),
        _row(0, 0.3, 0.3, week=2, game_id="new-b"),
    ]
    recent = evaluate_market_gap(
        early_bad, window_games=2, min_games=2, gap_margin=0.02, reference_gap=0.0
    )
    assert recent["tripped"] is False
    assert recent["last_week"] == 2
    late_bad = early_bad[2:] + early_bad[:2]
    recent_bad = evaluate_market_gap(
        late_bad, window_games=2, min_games=2, gap_margin=0.02, reference_gap=0.0
    )
    assert recent_bad["tripped"] is True
    assert recent_bad["last_week"] == 1
    ahead = evaluate_market_gap(
        [_row(1, 0.9, 0.2), _row(0, 0.1, 0.8, week=10)],
        window_games=4,
        min_games=2,
        gap_margin=0.02,
        reference_gap=0.0,
    )
    assert ahead["gap"] < 0
    assert ahead["tripped"] is False
    defaults = evaluate_market_gap([_row(1, 0.2, 0.9) for _ in range(10)])
    assert defaults["tripped"] is False
    assert defaults["reason"] == "below_minimum_sample"
    assert defaults["window_games"] == DEFAULT_WINDOW_GAMES
    assert defaults["min_games"] == DEFAULT_MIN_GAMES
    assert defaults["gap_margin"] == DEFAULT_GAP_MARGIN
    assert defaults["reference_gap"] == pytest.approx(reference_market_gap())
    assert defaults["gap_threshold"] == pytest.approx(
        defaults["reference_gap"] + DEFAULT_GAP_MARGIN
    )


def test_an_equal_gap_trips_and_a_slightly_smaller_gap_does_not() -> None:
    """The comparison is greater than or equal to reference gap plus margin."""
    games = [_row(1, 0.6, 0.8)]
    measured = evaluate_market_gap(
        games, window_games=1, min_games=1, gap_margin=0.0, reference_gap=0.0
    )
    gap = float(measured["gap"])
    assert gap > 0
    at_threshold = evaluate_market_gap(
        games, window_games=1, min_games=1, gap_margin=0.0, reference_gap=gap
    )
    assert at_threshold["tripped"] is True
    under = evaluate_market_gap(
        games, window_games=1, min_games=1, gap_margin=0.0, reference_gap=gap + 1e-9
    )
    assert under["tripped"] is False
    with pytest.raises(ValueError, match="min_games"):
        evaluate_market_gap(games, window_games=2, min_games=4, gap_margin=0.02, reference_gap=0.0)


def _games_at_gap(gap: float) -> list[dict[str, Any]]:
    """Two home wins whose model-minus-market log loss equals ``gap``."""
    market: float = 0.7
    model: float = market * math.exp(-gap)
    return [
        _row(1, model, market, game_id="a"),
        _row(1, model, market, game_id="b", week=10),
    ]


def test_reference_table_stays_quiet_and_the_margin_trips() -> None:
    """The known CV gap is quiet. Reference gap plus the margin trips."""
    reference_gap: float = reference_market_gap()
    assert reference_gap == pytest.approx(0.6377 - 0.6120)
    quiet = evaluate_market_gap(_games_at_gap(reference_gap), window_games=4, min_games=2)
    assert quiet["tripped"] is False
    assert quiet["reason"] == "within_gap"
    assert quiet["gap"] == pytest.approx(reference_gap, abs=1e-9)
    assert quiet["reference_gap"] == pytest.approx(reference_gap)
    assert quiet["gap_margin"] == DEFAULT_GAP_MARGIN
    bar: float = reference_gap + DEFAULT_GAP_MARGIN
    at_bar = evaluate_market_gap(_games_at_gap(bar), window_games=4, min_games=2)
    assert at_bar["gap"] == pytest.approx(bar, abs=1e-9)
    assert at_bar["tripped"] is True
    assert at_bar["reason"] == "log_loss_gap"
    worse = evaluate_market_gap(_games_at_gap(bar + 0.01), window_games=4, min_games=2)
    assert worse["tripped"] is True
    body = alert_issue_body(at_bar)
    assert "This issue is not a betting signal." in body
    assert "stays quiet" in body


def test_issue_sync_creates_once_and_skips_an_unchanged_body() -> None:
    """One open issue is updated. A pull request with the label is ignored."""
    decision = _tripped()
    body = alert_issue_body(decision)
    assert "2026" in body or "season" in body
    assert body.endswith("\n")
    assert "ADR-009" in body
    client = _Issues(
        [
            {"number": 4, "title": "not this", "body": "pr", "pull_request": {"url": "x"}},
            {"number": 9, "title": "old", "body": "old"},
            {"number": 3, "title": "older", "body": "older"},
        ]
    )
    updated = sync_alert_issue(decision, client)
    assert updated["action"] == "updated"
    assert updated["number"] == 3
    assert client.updated == [3]
    assert client.created == []
    again = sync_alert_issue(decision, client)
    assert again["action"] == "unchanged"
    assert again["number"] == 3
    assert client.updated == [3]
    quiet = dict(decision)
    quiet["tripped"] = False
    quiet["reason"] = "within_gap"
    skipped = sync_alert_issue(quiet, client)
    assert skipped["action"] == "skipped"
    assert client.labels == [ALERT_LABEL, ALERT_LABEL]
    fresh = _Issues()
    created = sync_alert_issue(decision, fresh)
    assert created["action"] == "created"
    assert fresh.items[0]["title"] == ALERT_TITLE
    assert fresh.items[0]["body"] == body


def test_github_client_sends_the_token_and_treats_422_as_an_existing_label() -> None:
    """The Authorization header is set. A 422 on label create is not a failure."""
    seen: dict[str, str] = {}

    class _Response:
        status: int = 201

        def read(self) -> bytes:
            return b'{"number": 7}'

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def opener(request: Any, timeout: int = 30) -> _Response:
        seen["auth"] = request.get_header("Authorization")
        seen["method"] = request.method
        return _Response()

    client = GithubIssueClient("owner/repo", "secret-token", opener=opener)
    created = client.create("title", "body", [ALERT_LABEL])
    assert created["number"] == 7
    assert seen["auth"] == "Bearer secret-token"
    assert seen["method"] == "POST"

    def label_exists(request: Any, timeout: int = 30) -> _Response:
        raise urllib.error.HTTPError(
            request.full_url,
            422,
            "exists",
            hdrs=None,
            fp=io.BytesIO(b"{}"),
        )

    existing = GithubIssueClient("owner/repo", "secret-token", opener=label_exists)
    existing.ensure_label(ALERT_LABEL, "b60205", "already there")
    with pytest.raises(ModelAlertError, match="owner/name"):
        GithubIssueClient("noslash", "secret-token")


def test_finished_games_use_the_earliest_archive_probability(tmp_path: Path) -> None:
    """The alert reads the same earliest probability as the track record."""
    archive: Path = tmp_path / "archive"
    early = {
        "sample": False,
        "season": 2026,
        "week": 9,
        "games": [
            {
                "game_id": "g1",
                "model_home_wp": 0.2,
                "market_home_wp": 0.6,
                "status": "scheduled",
            }
        ],
    }
    tip = {
        "sample": False,
        "season": 2026,
        "week": 9,
        "games": [
            {
                "game_id": "g1",
                "model_home_wp": 0.9,
                "market_home_wp": 0.6,
                "status": "final",
                "home_score": 24,
                "away_score": 17,
            }
        ],
    }
    history = archive / "history" / "2026-w9" / "2026-11-01T00-00-00Z.json"
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps(early) + "\n", encoding="utf-8", newline="\n")
    tip_path = archive / "predictions" / "2026-w9.json"
    tip_path.parent.mkdir(parents=True)
    tip_path.write_text(json.dumps(tip) + "\n", encoding="utf-8", newline="\n")
    before = tip_path.read_bytes()
    rows = finished_market_games(archive, None)
    assert tip_path.read_bytes() == before
    assert len(rows) == 1
    assert rows[0]["model_probability"] == pytest.approx(0.2)
    assert rows[0]["label"] == 1
    assert rows[0]["game_id"] == "g1"
    assert finished_market_games(tmp_path / "missing", None) == []
    corrupt = archive / "predictions" / "2026-w10.json"
    corrupt.write_text("{", encoding="utf-8", newline="\n")
    with pytest.raises(TrackRecordError):
        finished_market_games(archive, None)


def test_cli_does_not_fail_when_the_window_is_short_or_github_is_unconfigured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run, a missing token, and a missing archive all return 0."""
    path = tmp_path / "games.json"
    path.write_text(json.dumps([_row(1, 0.2, 0.9)]), encoding="utf-8", newline="\n")
    status_dir = tmp_path / "run-status"
    monkeypatch.setenv("ENDZONE_RUN_STATUS_DIR", str(status_dir))
    assert (
        main(
            [
                "--games-json",
                str(path),
                "--window-games",
                "1",
                "--min-games",
                "1",
                "--margin",
                "0.01",
                "--dry-run",
            ]
        )
        == 0
    )
    saved = (status_dir / "model_alert.txt").read_text(encoding="utf-8")
    assert "status=log_loss_gap\n" in saved
    assert "action=dry_run\n" in saved
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert (
        main(
            [
                "--games-json",
                str(path),
                "--window-games",
                "1",
                "--min-games",
                "1",
                "--margin",
                "0.01",
            ]
        )
        == 0
    )
    assert main(["--archive-dir", str(tmp_path / "missing"), "--min-games", "1"]) == 0


def test_results_workflow_runs_the_alert_without_failing_the_job() -> None:
    """The alert is after the track record, and a GitHub error does not fail publish."""
    text = Path(".github/workflows/refresh-results.yml").read_text(encoding="utf-8")
    assert "endzone.site.model_alert" in text
    assert "continue-on-error: true" in text
    assert "issues: write" in text
    assert text.index("endzone.site.track_record") < text.index("endzone.site.model_alert")
    assert text.index("endzone.site.model_alert") < text.index("latest.json")
    registry = Path(".github/workflows/model-registry.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in registry
    assert "schedule:" not in registry
    assert "group: model-registry" in registry
    assert "group: pages" not in registry
    assert "endzone.modeling.registry_command" in registry
