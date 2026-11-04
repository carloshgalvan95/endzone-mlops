"""The Sprint 2 file is pre-registered and stays out of the win-prob catalog."""

from pathlib import Path

import pytest

from endzone.modeling.hypotheses import load_catalog
from endzone.modeling.points_spec import ScoreExperimentSpec, load_score_spec


def test_score_file_is_not_a_classification_hypothesis() -> None:
    """Phase 2 aliases do not include the margin and total experiment."""
    catalog = load_catalog()
    assert "s2" not in catalog.by_id_map
    assert "s2" not in catalog.alias_names()
    assert catalog.by_tag("s2-margin-total") is None


def test_preregistered_thresholds_match_adr_007() -> None:
    """Numbers locked before any fit. ADR-007. Do not retune them after a result."""
    spec: ScoreExperimentSpec = load_score_spec()
    assert spec.experiment_id == "s2"
    assert spec.task == "score"
    assert spec.feature_set == "h2_full"
    assert spec.model == "ridge"
    assert spec.ridge_alpha == pytest.approx(1.0)
    assert spec.random_state == 42
    assert spec.cv_start_season == 2016
    assert spec.cv_end_season == 2023
    assert spec.season_count == 8
    assert spec.min_seasons_improved == 5
    assert spec.n_bootstrap == 1000
    assert spec.bootstrap_seed == 42
    assert spec.elo_points_per_spread == pytest.approx(25.0)
    assert spec.interval_coverage == pytest.approx(0.8)
    assert spec.winprob_log_loss_benchmark == pytest.approx(0.638)
    assert spec.market_margin_multiplier == pytest.approx(1.0)
    assert spec.experiment_name == "nfl_pregame/sprint2_score"
    assert spec.hypothesis_tag == "s2-margin-total"
    assert spec.output_dir == "reports/sprint2"
    margin = spec.target("margin")
    total = spec.target("total")
    assert margin.baseline == "elo_spread"
    assert margin.registered_model_name == "endzone-margin"
    assert total.baseline == "league_mean"
    assert total.registered_model_name == "endzone-total"
    assert [item.name for item in spec.targets] == ["margin", "total"]


def test_score_file_rejects_a_broken_shape(tmp_path: Path) -> None:
    """An unknown key, a bad window, or a second score file fails closed."""
    source: Path = Path("configs/experiments/s2.yaml")
    text: str = source.read_text(encoding="utf-8")
    (tmp_path / "s2.yaml").write_text(text + "extra: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_score_spec(tmp_path)

    window: str = text.replace("season_count: 8", "season_count: 7", 1)
    (tmp_path / "s2.yaml").write_text(window, encoding="utf-8")
    with pytest.raises(ValueError, match="season_count"):
        load_score_spec(tmp_path)

    (tmp_path / "s2.yaml").write_text(text, encoding="utf-8")
    (tmp_path / "s2b.yaml").write_text(text.replace("id: s2", "id: s2b", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_score_spec(tmp_path)
