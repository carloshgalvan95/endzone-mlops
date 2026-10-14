"""Diagnostic audits of model inputs.

Importing this package does not import the audit module. ``python -m
src.analysis.feature_audit`` would otherwise warn that the module was already
in ``sys.modules`` before runpy executed it.
"""

from typing import Any

__all__ = ["run_feature_audit"]


def __getattr__(name: str) -> Any:
    """Resolve public names without importing them at package import time.

    Args:
        name: Attribute requested from the package.

    Returns:
        The requested callable.

    Raises:
        AttributeError: ``name`` is not a public export.
    """
    if name == "run_feature_audit":
        from src.analysis.feature_audit import run_feature_audit

        return run_feature_audit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
