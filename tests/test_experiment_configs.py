"""Experiment files in configs/experiments are the hypothesis registry."""

from pathlib import Path

import pytest
import yaml

from endzone.modeling.hypotheses import (
    ExperimentSpec,
    experiment_for,
    experiments_directory,
    load_catalog,
    resolve_hypothesis,
)
from endzone.modeling.train import build_parser


def _classification_paths() -> list[Path]:
    """Win-probability experiment files. The score file is a separate task."""
    paths: list[Path] = []
    for path in sorted(experiments_directory().glob("*.yaml")):
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("task") == "score":
            continue
        paths.append(path)
    return paths


@pytest.mark.parametrize("path", _classification_paths(), ids=lambda path: path.stem)
def test_experiment_file_matches_the_catalog(path: Path) -> None:
    """The file id, aliases, model rows, and tracking fields round-trip."""
    spec: ExperimentSpec = load_catalog().by_id(path.stem)
    assert spec.experiment_id == path.stem
    assert spec.phase in (1, 2)
    if spec.phase == 1:
        assert spec.aliases == ()
        assert spec.tag is None
        with pytest.raises(ValueError, match="Phase 1 does not take a hypothesis"):
            resolve_hypothesis(1, "h1")
        return
    assert spec.tag in spec.aliases
    for alias in spec.aliases:
        assert resolve_hypothesis(2, alias) == spec.tag
        assert experiment_for(2, alias).experiment_id == spec.experiment_id
    resolved: ExperimentSpec = experiment_for(2, spec.tag)
    assert resolved.logistic_feature_sets == spec.logistic_feature_sets
    assert resolved.model_specs == spec.model_specs
    assert resolved.experiment_name == spec.experiment_name
    assert resolved.output_dir == spec.output_dir
    assert resolved.phase_tag == spec.phase_tag
    assert resolved.note.endswith("No model is promoted from this run.")


def test_catalog_has_one_default_per_phase() -> None:
    """Phase 2 with no flag uses the single default file. Phase 1 is the other."""
    catalog = load_catalog()
    phase2_defaults: list[ExperimentSpec] = [
        item for item in catalog.experiments if item.phase == 2 and item.default
    ]
    assert len(phase2_defaults) == 1
    assert resolve_hypothesis(2, None) == phase2_defaults[0].tag
    assert resolve_hypothesis(1, None) is None
    assert catalog.phase1().experiment_id == "phase1"


def test_unknown_hypothesis_names_the_known_aliases() -> None:
    """A typo lists the aliases from the files, not a hardcoded table."""
    with pytest.raises(ValueError, match="Unknown hypothesis 'not-a-hypothesis'") as caught:
        resolve_hypothesis(2, "not-a-hypothesis")
    message: str = str(caught.value)
    for alias in load_catalog().alias_names():
        assert alias in message


def test_cli_choices_are_the_yaml_aliases() -> None:
    """``--hypothesis`` accepts every alias and nothing else."""
    action = next(item for item in build_parser()._actions if item.dest == "hypothesis")
    assert tuple(action.choices) == load_catalog().alias_names()
