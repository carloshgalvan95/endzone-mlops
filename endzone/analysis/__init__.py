"""Diagnostic audits of model inputs.

Importing this package does not import the audit module. ``python -m
endzone.analysis.feature_audit`` would otherwise warn that the module was already
in ``sys.modules`` before runpy executed it.
"""

from typing import Any

__all__ = ["run_feature_audit"]


def __getattr__(name: str) -> Any:
    if name == "run_feature_audit":
        from endzone.analysis.feature_audit import run_feature_audit

        return run_feature_audit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
