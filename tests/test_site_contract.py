"""The v1 site data contract matches the writers and the shipped examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from endzone.modeling.weekly import build_prediction_document, team_catalog_document
from endzone.site.contract import (
    CONTRACT_VERSION,
    SLATE_SCHEMA,
    TRACK_RECORD_SCHEMA,
    ContractError,
    main,
    validate_data_dir,
    validate_instance,
)
from endzone.site.results import apply_final_scores
from endzone.site.track_record import build_track_record, unavailable_document

pytestmark = pytest.mark.cross_platform

_STAMP: str = "2026-11-17T18:00:00Z"
_ROOT: Path = Path(__file__).resolve().parents[1]


def test_shipped_examples_match_v1() -> None:
    """sample.json and teams.json are the files the page already ships."""
    result: dict[str, Any] = validate_data_dir(_ROOT / "site" / "data")
    assert result["files"] == 2
    assert result["paths"] == ["sample.json", "teams.json"]
    sample: dict[str, Any] = json.loads(
        (_ROOT / "site" / "data" / "sample.json").read_text(encoding="utf-8")
    )
    assert sample["contract_version"] == CONTRACT_VERSION
    assert sample["sample"] is True
    assert sample["season"] is None


def test_writer_slate_covers_points_market_gap_and_a_tie(tmp_path: Path) -> None:
    """A scored week, a missing price, and a 0-0 tie all validate."""
    document: dict[str, Any] = build_prediction_document(
        season=2026,
        week=10,
        games=_games(),
        model_probabilities=pd.Series({"G1": 0.61, "G2": 0.5}),
        market_probabilities={"G1": 0.58, "G2": None},
        model_name="endzone-winprob",
        model_version="3@prod",
        feature_set="h2_full",
        generated_at=_STAMP,
        model_alias="prod",
        score_model=_accepted_score_model(),
        score_lines={
            "G1": {
                "estimated_margin": 3.0,
                "estimated_total": 41.0,
                "estimated_home_points": 22.0,
                "estimated_away_points": 19.0,
            },
            "G2": {
                "estimated_margin": 0.0,
                "estimated_total": 44.0,
                "estimated_home_points": 22.0,
                "estimated_away_points": 22.0,
            },
        },
    )
    assert document["contract_version"] == CONTRACT_VERSION
    validate_instance(document, SLATE_SCHEMA)
    final: dict[str, Any] = apply_final_scores(
        document,
        {
            "G2": {
                "home_score": 0,
                "away_score": 0,
                "result": "tie",
                "status": "final",
            }
        },
        _STAMP,
    )
    tied: dict[str, Any] = next(game for game in final["games"] if game["game_id"] == "G2")
    assert tied["result"] == "tie"
    assert tied["home_score"] == 0
    validate_instance(final, SLATE_SCHEMA)
    archive: Path = tmp_path / "archive"
    (archive / "predictions").mkdir(parents=True)
    (archive / "predictions" / "2026-w10.json").write_text(
        json.dumps(final),
        encoding="utf-8",
    )
    record: dict[str, Any] = build_track_record(archive, None, generated_at=_STAMP)
    assert record["contract_version"] == CONTRACT_VERSION
    assert record["seasons"][0]["weeks"][0]["ties"] == 1
    validate_instance(record, TRACK_RECORD_SCHEMA)


def test_hidden_score_model_and_unavailable_record() -> None:
    """The default hidden score block and an unreachable archive still match v1."""
    document: dict[str, Any] = build_prediction_document(
        season=2026,
        week=1,
        games=_games().iloc[:1],
        model_probabilities=pd.Series({"G1": 0.5}),
        market_probabilities={"G1": None},
        model_name="endzone-winprob",
        model_version="local",
        feature_set="h2_full",
        generated_at=_STAMP,
    )
    assert document["model_alias"] is None
    assert document["games"][0]["estimated_margin"] is None
    validate_instance(document, SLATE_SCHEMA)
    missing: dict[str, Any] = unavailable_document(_STAMP)
    assert missing["available"] is False
    assert missing["reason"] == "archive_unreachable"
    validate_instance(missing, TRACK_RECORD_SCHEMA)


def test_team_catalog_writer_matches_v1() -> None:
    """A catalog row with a dropped color still matches teams.schema.json."""
    frame: pd.DataFrame = pd.DataFrame(
        [
            {
                "team_abbr": "BUF",
                "team_nick": "Bills",
                "team_color": "#00338D",
                "team_color2": "navy",
                "team_logo_espn": "https://example.test/buf.png",
            }
        ]
    )
    document: dict[str, Any] = team_catalog_document(frame)
    assert document["contract_version"] == CONTRACT_VERSION
    assert document["teams"]["BUF"]["team_color2"] is None
    validate_instance(document, "teams.schema.json")


def test_schema_rejects_a_broken_slate() -> None:
    """A dropped game list, a bad probability, and a major version bump fail."""
    document: dict[str, Any] = build_prediction_document(
        season=2026,
        week=1,
        games=_games().iloc[:1],
        model_probabilities=pd.Series({"G1": 0.5}),
        market_probabilities={"G1": 0.4},
        model_name="endzone-winprob",
        model_version="local",
        feature_set="h2_full",
        generated_at=_STAMP,
    )
    broken: dict[str, Any] = dict(document)
    del broken["games"]
    with pytest.raises(ContractError, match="games"):
        validate_instance(broken, SLATE_SCHEMA)
    wrong_type: dict[str, Any] = json.loads(json.dumps(document))
    wrong_type["games"][0]["model_home_wp"] = "high"
    with pytest.raises(ContractError, match="model_home_wp"):
        validate_instance(wrong_type, SLATE_SCHEMA)
    next_major: dict[str, Any] = dict(document)
    next_major["contract_version"] = "2.0.0"
    with pytest.raises(ContractError, match="contract_version"):
        validate_instance(next_major, SLATE_SCHEMA)
    scheduled: dict[str, Any] = json.loads(json.dumps(document))
    scheduled["games"][0]["home_score"] = 0
    with pytest.raises(ContractError, match="does not allow"):
        validate_instance(scheduled, SLATE_SCHEMA)


def test_cli_require_published(tmp_path: Path) -> None:
    """Upload validation fails closed when the published files are absent."""
    sample: Path = _ROOT / "site" / "data" / "sample.json"
    teams: Path = _ROOT / "site" / "data" / "teams.json"
    (tmp_path / "sample.json").write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "teams.json").write_text(teams.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--data-dir", str(tmp_path)]) == 0
    assert main(["--data-dir", str(tmp_path), "--require-published"]) == 1


def _games() -> pd.DataFrame:
    """Two upcoming rows. The second has no kickoff clock."""
    return pd.DataFrame(
        [
            {
                "game_id": "G1",
                "gameday": "2026-11-15",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "MIA",
            },
            {
                "game_id": "G2",
                "gameday": "2026-11-16",
                "gametime": None,
                "home_team": "KC",
                "away_team": "DEN",
            },
        ]
    )


def _accepted_score_model() -> dict[str, Any]:
    """The page block the score command stores after ADR-007 accepts both targets."""
    target: dict[str, Any] = {
        "decision": "accepted",
        "baseline_name": "Elo/25",
        "mae": 10.155811,
        "rmse": 13.148404,
        "baseline_mae": 10.412221,
        "market_mae": 9.800548,
        "mae_diff": -0.256409,
        "mae_diff_ci_low": -0.368272,
        "mae_diff_ci_high": -0.133364,
        "coverage_80": 0.812871,
        "implied_win_log_loss": 0.63746,
        "winprob_log_loss_benchmark": 0.638,
        "seasons_improved": 8,
        "seasons_scored": 8,
        "n": 2191,
        "model_version": "1",
    }
    total: dict[str, Any] = dict(target)
    total["baseline_name"] = "League mean"
    total["implied_win_log_loss"] = None
    total["winprob_log_loss_benchmark"] = None
    return {
        "feature_set": "h2_full",
        "accepted": True,
        "hidden_reason": None,
        "margin": target,
        "total": total,
    }
