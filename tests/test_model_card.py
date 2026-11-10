"""The model card stays aligned with the locked metrics and the site link."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CARD: Path = Path("docs/model_card.md")
_REFERENCE: Path = Path("docs/experiments/reference_metrics.json")
_H5_REPORT: Path = Path("docs/experiments/reports/phase2_h5_comparison.md")
_H2_REPORT: Path = Path("docs/experiments/reports/phase2_h2_comparison.md")
_README: Path = Path("README.md")
_PAGE: Path = Path("site/index.html")
_CARD_URL: str = "https://github.com/carloshgalvan95/endzone-mlops/blob/main/docs/model_card.md"
# A hyphen inside an identifier (ADR-009) is not a minus sign.
_NUMBER: re.Pattern[str] = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_URL: re.Pattern[str] = re.compile(r"https?://\S+")
_ADR: re.Pattern[str] = re.compile(r"ADR-\d+")

_LOCKED_FIELDS: tuple[str, ...] = ("n", "log_loss", "brier", "auc", "accuracy")
_PAIRED_MODELS: tuple[tuple[str, str], ...] = (
    ("baseline-home", "none"),
    ("logistic", "h2_full"),
    ("logistic", "h5_qb"),
    ("elo", "elo"),
    ("market", "market"),
)


def _read(path: Path) -> str:
    """Read a text file as UTF-8."""
    return path.read_text(encoding="utf-8")


def _token_present(haystack: str, token: str) -> bool:
    """Return True when ``token`` is its own number, allowing a sentence period."""
    return re.search(rf"(?<![\w.]){re.escape(token)}(?!\d)(?!\.\d)", haystack) is not None


def _json_literal(value: object) -> str:
    """Spell a JSON number the way the reference file spells it."""
    return json.dumps(value)


def _comparison_rows(report: str) -> dict[tuple[str, str], dict[str, str]]:
    """Parse the first comparison table in a frozen CV report."""
    rows: dict[tuple[str, str], dict[str, str]] = {}
    in_table: bool = False
    for line in report.splitlines():
        if line.startswith("| model | feature_set |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells: list[str] = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "---":
            continue
        key: tuple[str, str] = (cells[0], cells[1])
        rows[key] = {
            "n": cells[4],
            "log_loss": cells[5],
            "brier": cells[6],
            "auc": cells[7],
            "accuracy": cells[8],
            "log_loss_diff": cells[9],
            "ci_low": cells[10],
            "ci_high": cells[11],
        }
    return rows


def _h2_full_folds(report: str) -> list[dict[str, str]]:
    """Parse the logistic ``h2_full`` fold table from the frozen H5 report."""
    marker: str = "### logistic (h2_full)"
    start: int = report.index(marker)
    section: str = report[start:]
    next_heading: int = section.index("\n### ", len(marker))
    folds: list[dict[str, str]] = []
    for line in section[:next_heading].splitlines():
        if not line.startswith("|") or line.startswith("| season") or line.startswith("| ---"):
            continue
        cells: list[str] = [cell.strip() for cell in line.strip("|").split("|")]
        folds.append(
            {
                "season": cells[0],
                "n_train": cells[3],
                "n_eval": cells[4],
                "log_loss": cells[6],
            }
        )
    return folds


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


def test_card_quotes_the_frozen_h5_comparison() -> None:
    """Paired metrics and the h2_full folds match the frozen H5 report."""
    card: str = _read(_CARD)
    report: str = _read(_H5_REPORT)
    assert "docs/experiments/reports/phase2_h5_comparison.md" in card
    rows: dict[tuple[str, str], dict[str, str]] = _comparison_rows(report)
    for key in _PAIRED_MODELS:
        quoted: dict[str, str] = rows[key]
        for field in (
            "n",
            "log_loss",
            "brier",
            "auc",
            "accuracy",
            "log_loss_diff",
            "ci_low",
            "ci_high",
        ):
            assert quoted[field] in card
    for fold in _h2_full_folds(report):
        assert fold["season"] in card
        assert fold["n_train"] in card
        assert fold["n_eval"] in card
        assert fold["log_loss"] in card


def test_card_numbers_come_from_the_cited_files() -> None:
    """A number in the card is a token in the reference file or a frozen report."""
    card: str = _ADR.sub(" ", _URL.sub(" ", _read(_CARD)))
    sources: str = "\n".join(_read(path) for path in (_REFERENCE, _H5_REPORT, _H2_REPORT))
    missing: list[str] = []
    for token in _NUMBER.findall(card):
        if not _token_present(sources, token):
            missing.append(token)
    assert missing == []


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


def test_readme_and_site_link_the_model_card() -> None:
    """The README and the weekly page both point at the card."""
    readme: str = _read(_README)
    page: str = _read(_PAGE)
    assert "docs/model_card.md" in readme
    assert readme.count("docs/model_card.md") >= 2
    assert page.count(_CARD_URL) == 2
    assert 'rel="noopener"' in page


def test_site_link_leaves_the_game_cards_in_place() -> None:
    """The page link is an anchor. The slate markup is unchanged."""
    page: str = _read(_PAGE)
    assert 'id="games-cards"' in page
    assert 'class="cards"' in page
    assert 'id="games-body"' in page
    assert page.index('id="how-to-read"') < page.index("<footer")
    assert "Illustrative card, not a live prediction." in page
    script: str = _read(Path("site/assets/app.js"))
    css: str = _read(Path("site/assets/styles.css"))
    assert "model_card" not in script
    assert "model_card" not in css


def test_card_has_no_em_dash_or_en_dash() -> None:
    """The card uses commas, periods, colons, and hyphens."""
    card: str = _read(_CARD)
    assert "\u2014" not in card
    assert "\u2013" not in card
