"""MLproject entry points use the same flags as the training and ingest CLIs."""

import argparse
from pathlib import Path

from mlflow.projects._project_spec import load_project

from endzone.ingest.cli import _build_parser as build_ingest_parser
from endzone.modeling.score import build_parser as build_score_parser
from endzone.modeling.train import build_parser as build_train_parser

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]

_TRAIN_CV_FLAGS: dict[str, str] = {
    "hypothesis": "--hypothesis",
    "reference_model": "--reference-model",
    "reference_feature_set": "--reference-feature-set",
    "output_dir": "--output-dir",
}


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Return every flag the parser accepts."""
    options: set[str] = set()
    for action in parser._actions:
        options.update(action.option_strings)
    return options


def test_mlflow_loads_entry_points() -> None:
    """MLflow reads ingest, train_cv, verify_reference, and score from MLproject."""
    project = load_project(str(_REPO_ROOT))
    names: set[str] = set(project._entry_points)
    assert names == {"ingest", "train_cv", "verify_reference", "score"}


def test_score_parameters_match_the_cli() -> None:
    """The score entry point passes season, week, and the output directory."""
    project = load_project(str(_REPO_ROOT))
    entry = project.get_entry_point("score")
    rendered: str = entry.compute_command({}, None)
    score_flags: set[str] = _option_strings(build_score_parser())
    assert "endzone.modeling.score" in rendered
    for flag in ("--season", "--week", "--output-dir"):
        assert flag in score_flags
        assert flag in rendered
    assert entry.parameters["season"].default == "auto"
    assert entry.parameters["week"].default == "auto"
    assert entry.parameters["output_dir"].default == "site/data"


def test_train_cv_parameters_match_cli_flags() -> None:
    """Typed train_cv parameters are the phase 2 CV flags, with H5b defaults."""
    project = load_project(str(_REPO_ROOT))
    entry = project.get_entry_point("train_cv")
    assert set(entry.parameters) == set(_TRAIN_CV_FLAGS)
    rendered: str = entry.compute_command({}, None)
    train_flags: set[str] = _option_strings(build_train_parser())
    for parameter_name, flag in _TRAIN_CV_FLAGS.items():
        assert flag in train_flags
        assert flag in rendered
        assert parameter_name in entry.command
    assert "--cv" in rendered
    assert "--phase" in rendered
    assert entry.parameters["hypothesis"].default == "h5b"
    assert entry.parameters["reference_model"].default == "logistic"
    assert entry.parameters["reference_feature_set"].default == "h2_full"
    assert entry.parameters["output_dir"].default == "reports/phase2_h5b"


def test_ingest_and_verify_reference_match_the_clis() -> None:
    """ingest and verify_reference call the modules those Make targets call."""
    project = load_project(str(_REPO_ROOT))
    ingest_command: str = project.get_entry_point("ingest").compute_command({}, None)
    verify_command: str = project.get_entry_point("verify_reference").compute_command({}, None)
    assert "endzone.ingest.load_raw_nflverse" in ingest_command
    assert "--dataset" in ingest_command
    assert "--dataset" in _option_strings(build_ingest_parser())
    assert "--verify-reference" in verify_command
    assert "--verify-reference" in _option_strings(build_train_parser())
