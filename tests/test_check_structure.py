"""Tests for the CCDS layout checker."""

from pathlib import Path

import pytest

from scripts.check_structure import (
    REQUIRED_DIRECTORIES,
    REQUIRED_FILES,
    find_structure_violations,
    list_tracked_paths,
    main,
)

pytestmark = pytest.mark.cross_platform

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _write_valid_tree(root: Path) -> None:
    """Create the required directories and files under ``root``."""
    for relative in REQUIRED_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_FILES:
        path: Path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def test_real_repository_passes() -> None:
    """The checkout matches the required layout and has no forbidden paths."""
    violations: list[str] = find_structure_violations(REPO_ROOT, list_tracked_paths(REPO_ROOT))
    assert violations == []


def test_real_repository_cli_passes() -> None:
    """The command-line checker accepts this repository."""
    status: int = main(["--root", str(REPO_ROOT)])
    assert status == 0


def test_missing_directory_fails(tmp_path: Path) -> None:
    """A tree without the CCDS directories is rejected."""
    violations: list[str] = find_structure_violations(tmp_path, [])
    assert "missing directory: data/raw" in violations
    assert "missing directory: reports/figures" in violations
    assert "missing file: notebooks/README.md" in violations


def test_top_level_src_directory_fails(tmp_path: Path) -> None:
    """A top-level src directory is forbidden after the package rename."""
    _write_valid_tree(tmp_path)
    (tmp_path / "src").mkdir()
    violations: list[str] = find_structure_violations(tmp_path, [])
    assert any("top-level src/" in item for item in violations)


def test_tracked_src_path_fails(tmp_path: Path) -> None:
    """A tracked path under src/ fails even when the directory is already gone."""
    _write_valid_tree(tmp_path)
    violations: list[str] = find_structure_violations(tmp_path, ["src/train/train.py"])
    assert any(item.endswith("src/train/train.py") for item in violations)


def test_tracked_data_file_fails(tmp_path: Path) -> None:
    """Parquet, csv, and any other non-gitkeep file under data/ are forbidden."""
    _write_valid_tree(tmp_path)
    violations: list[str] = find_structure_violations(
        tmp_path,
        ["data/raw/.gitkeep", "data/raw/games.parquet", "data/interim/notes.txt"],
    )
    assert "forbidden tracked data file: data/raw/games.parquet" in violations
    assert "forbidden tracked data file: data/interim/notes.txt" in violations
    assert not any(item.endswith("data/raw/.gitkeep") for item in violations)


def test_untracked_data_file_is_allowed(tmp_path: Path) -> None:
    """A cache file on disk is fine when git does not track it."""
    _write_valid_tree(tmp_path)
    (tmp_path / "data" / "raw" / "games.parquet").write_bytes(b"cache")
    violations: list[str] = find_structure_violations(tmp_path, ["data/raw/.gitkeep"])
    assert violations == []


def test_tracked_model_and_report_artifacts_fail(tmp_path: Path) -> None:
    """Serialized models and generated reports stay out of git."""
    _write_valid_tree(tmp_path)
    violations: list[str] = find_structure_violations(
        tmp_path,
        ["models/model.pkl", "reports/phase1/comparison.md", "reports/figures/plot.png"],
    )
    assert "forbidden tracked model artifact: models/model.pkl" in violations
    assert "forbidden tracked report output: reports/phase1/comparison.md" in violations
    assert "forbidden tracked report output: reports/figures/plot.png" in violations


def test_python_outside_allowed_roots_fails(tmp_path: Path) -> None:
    """Product code belongs in endzone/, tests/, or scripts/."""
    _write_valid_tree(tmp_path)
    violations: list[str] = find_structure_violations(
        tmp_path,
        [
            "endzone/train/train.py",
            "tests/test_check_structure.py",
            "scripts/check_structure.py",
            "notebooks/explore.py",
        ],
    )
    assert any("notebooks/explore.py" in item for item in violations)
    assert not any("endzone/train/train.py" in item for item in violations)


def test_cli_reports_git_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A directory that is not a git checkout fails closed."""
    status: int = main(["--root", str(tmp_path)])
    captured: str = capsys.readouterr().err
    assert status == 1
    assert "git ls-files failed" in captured
