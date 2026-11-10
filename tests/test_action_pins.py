"""Tests for the GitHub Actions pin check.

The check only reads local files. It is marked ``cross_platform`` so the
pull-request subset on Windows and macOS runs it.
"""

from pathlib import Path

import pytest

from scripts.check_action_pins import find_pin_violations, main

pytestmark = pytest.mark.cross_platform

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_SHA: str = "a" * 40


def _write(root: Path, relative: str, text: str) -> None:
    """Write a UTF-8 workflow file under ``root``."""
    path: Path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workflow(steps: str) -> str:
    """Return a one-job workflow whose steps are ``steps``."""
    return (
        f"name: demo\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n{steps}"
    )


def _step(uses_line: str) -> str:
    """Return one step whose ``uses`` line is ``uses_line``."""
    return f"      - name: Step\n        {uses_line}\n"


def test_real_repository_pins_actions() -> None:
    """This checkout pins every third-party workflow action."""
    assert find_pin_violations(REPO_ROOT) == []


def test_real_repository_cli_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """The command-line checker accepts this repository."""
    status: int = main(["--root", str(REPO_ROOT)])
    captured: str = capsys.readouterr().out
    assert status == 0
    assert "action pins ok" in captured


@pytest.mark.parametrize(
    ("uses_line", "fragment"),
    [
        ("uses: actions/checkout@v4", "40-character"),
        ("uses: actions/checkout@" + ("b" * 39), "40-character"),
        ("uses: actions/checkout@" + ("b" * 41), "40-character"),
        ("uses: actions/checkout@" + ("A" * 40) + " # v4.4.0", "40-character"),
        ("uses: actions/checkout@" + _SHA, "vX.Y.Z"),
        ("uses: actions/checkout@" + _SHA + " # v4.4", "vX.Y.Z"),
        ("uses: actions/checkout@" + _SHA + " # v4", "vX.Y.Z"),
        (
            "uses: carloshgalvan95/endzone-mlops/.github/workflows/ci.yml@main",
            "40-character",
        ),
        ("uses: docker://alpine:3.19", "40-character"),
    ],
)
def test_unpinned_uses_fails(tmp_path: Path, uses_line: str, fragment: str) -> None:
    """A tag, a short SHA, a branch, or a missing version comment fails."""
    _write(tmp_path, ".github/workflows/ci.yml", _workflow(_step(uses_line)))
    violations: list[str] = find_pin_violations(tmp_path)
    assert len(violations) == 1
    assert fragment in violations[0]
    assert violations[0].startswith(".github/workflows/ci.yml:")


def test_pinned_sha_passes(tmp_path: Path) -> None:
    """A 40-hex SHA with a three-part version comment is accepted."""
    line: str = f"uses: actions/checkout@{_SHA} # v4.4.0"
    _write(tmp_path, ".github/workflows/ci.yml", _workflow(_step(line)))
    assert find_pin_violations(tmp_path) == []


def test_local_action_is_exempt(tmp_path: Path) -> None:
    """A ``./`` action does not need a commit SHA."""
    steps: str = (
        _step("uses: ./.github/actions/build")
        + _step("uses: ./.github/actions/build@v1")
        + _step(f"uses: actions/checkout@{_SHA} # v4.4.0")
    )
    _write(tmp_path, ".github/workflows/ci.yml", _workflow(steps))
    assert find_pin_violations(tmp_path) == []


def test_run_only_workflow_passes(tmp_path: Path) -> None:
    """A job that only runs shell commands has nothing to pin."""
    text: str = _workflow("      - name: Echo\n        run: echo ok\n")
    _write(tmp_path, ".github/workflows/ci.yml", text)
    assert find_pin_violations(tmp_path) == []


def test_uses_text_inside_run_block_is_ignored(tmp_path: Path) -> None:
    """Text inside a ``run:`` block is not an action reference."""
    steps: str = (
        "      - name: Script\n"
        "        run: |\n"
        "          echo uses: actions/checkout@v4\n"
        + _step(f"uses: actions/checkout@{_SHA} # v4.4.0")
    )
    _write(tmp_path, ".github/workflows/ci.yml", _workflow(steps))
    assert find_pin_violations(tmp_path) == []


def test_folded_condition_does_not_hide_a_later_uses(tmp_path: Path) -> None:
    """A folded ``if:`` block does not swallow the next step."""
    text: str = (
        "name: demo\n"
        "on: [push]\n"
        "jobs:\n"
        "  archive:\n"
        "    if: >-\n"
        "      always() &&\n"
        "      needs.build.result == 'success'\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n" + _step("uses: actions/checkout@v4")
    )
    _write(tmp_path, ".github/workflows/pages.yml", text)
    violations: list[str] = find_pin_violations(tmp_path)
    assert len(violations) == 1
    assert violations[0].startswith(".github/workflows/pages.yml:")
    assert "40-character" in violations[0]


def test_composite_action_tag_fails(tmp_path: Path) -> None:
    """A composite action is checked even when workflows are already pinned."""
    _write(
        tmp_path,
        ".github/workflows/ci.yml",
        _workflow(_step(f"uses: actions/checkout@{_SHA} # v4.4.0")),
    )
    _write(
        tmp_path,
        ".github/actions/build/action.yml",
        "name: build\nruns:\n  using: composite\n  steps:\n    - uses: actions/checkout@v4\n",
    )
    violations: list[str] = find_pin_violations(tmp_path)
    assert violations == [
        ".github/actions/build/action.yml:5: uses is not a 40-character commit SHA: "
        "actions/checkout@v4"
    ]


def test_invalid_yaml_fails(tmp_path: Path) -> None:
    """A workflow that is not YAML fails closed."""
    _write(tmp_path, ".github/workflows/ci.yml", "jobs: [\n")
    violations: list[str] = find_pin_violations(tmp_path)
    assert len(violations) == 1
    assert violations[0].startswith(".github/workflows/ci.yml: invalid YAML:")


def test_missing_workflows_fail(tmp_path: Path) -> None:
    """A tree with no workflows and no composite actions fails."""
    violations: list[str] = find_pin_violations(tmp_path)
    assert violations == ["no GitHub Actions workflows or composite actions found"]


def test_cli_reports_a_tag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command prints the violation and returns status 1."""
    _write(tmp_path, ".github/workflows/ci.yml", _workflow(_step("uses: actions/checkout@v4")))
    status: int = main(["--root", str(tmp_path)])
    captured: str = capsys.readouterr().err
    assert status == 1
    assert "40-character" in captured
