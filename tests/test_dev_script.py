"""Windows PowerShell helpers stay valid UTF-8 and parse."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPTS: tuple[str, ...] = ("scripts/dev.ps1", "scripts/train_sweep.ps1")


@pytest.mark.cross_platform
@pytest.mark.parametrize("relative", _SCRIPTS)
def test_powershell_script_is_utf8_and_parses(relative: str) -> None:
    """The Windows helper decodes as UTF-8, and PowerShell accepts the syntax.

    Linux and macOS skip the parser when PowerShell is not installed. The
    UTF-8 read still runs. Windows CI has ``pwsh`` and parses the file.
    """
    path: Path = Path(relative)
    text: str = path.read_text(encoding="utf-8")
    assert "\x00" not in text
    pwsh: str | None = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("PowerShell is not installed")
    command: str = (
        "$errors = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        "$env:ENDZONE_PS1, [ref]$null, [ref]$errors); "
        "if ($errors) { $errors | ForEach-Object { Write-Output $_.ToString() }; exit 1 }"
    )
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [pwsh, "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "ENDZONE_PS1": str(path.resolve())},
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
