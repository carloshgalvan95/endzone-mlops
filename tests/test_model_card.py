"""The model card stays aligned with the locked reference metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CARD: Path = Path("docs/model_card.md")
_REFERENCE: Path = Path("docs/experiments/reference_metrics.json")

_LOCKED_FIELDS: tuple[str, ...] = ("n", "log_loss", "brier", "auc", "accuracy")


def _read(path: Path) -> str:
    """Read a text file as UTF-8."""
    return path.read_text(encoding="utf-8")


def _json_literal(value: object) -> str:
    """Spell a JSON number the way the reference file spells it."""
    return json.dumps(value)


def test_card_quotes_reference_metrics() -> None:
    """Every locked field in the reference file appears in the card."""
    reference: dict[str, Any] = json.loads(_read(_REFERENCE))
    card: str = _read(_CARD)
    assert "docs/experiments/reference_metrics.json" in card
    assert str(reference["cv_start_season"]) in card
    assert str(reference["cv_end_season"]) in card
    assert str(reference["n_bootstrap"]) in card
    assert str(reference["bootstrap_seed"]) in card
    assert _json_literal(reference["tolerance"]) in card
    for row in reference["models"]:
        assert row["model"] in card
        assert row["feature_set"] in card
        for field in _LOCKED_FIELDS:
            if field not in row:
                continue
            assert _json_literal(row[field]) in card


def test_market_has_the_lower_log_loss_and_nothing_is_a_pick() -> None:
    """The card states the market result and refuses a betting reading."""
    reference: dict[str, Any] = json.loads(_read(_REFERENCE))
    indexed: dict[tuple[str, str], dict[str, Any]] = {
        (row["model"], row["feature_set"]): row for row in reference["models"]
    }
    market_loss: float = float(indexed[("market", "market")]["log_loss"])
    model_loss: float = float(indexed[("logistic", "h2_full")]["log_loss"])
    assert market_loss < model_loss
    card: str = _read(_CARD)
    assert "The market has the lower log loss." in card
    assert "Nothing in this card is a betting pick." in card
    assert "not betting advice" in card
    lowered: str = card.lower()
    assert "bet the" not in lowered
    assert "pick:" not in lowered
