"""Compare a refit to the committed warehouse reference file."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REFERENCE_METRICS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "docs" / "experiments" / "reference_metrics.json"
)
_REFERENCE_POOLED_FIELDS: frozenset[str] = frozenset({"n", "log_loss", "brier", "auc", "accuracy"})
_REFERENCE_FILE_KEYS: tuple[str, ...] = (
    "cv_start_season",
    "cv_end_season",
    "n_bootstrap",
    "bootstrap_seed",
    "tolerance",
    "models",
)


def _reference_metric_number(value: object) -> float | None:
    """Return a finite float, or None when the value is missing or not numeric."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number: float = float(value)
    if not math.isfinite(number):
        return None
    return number


def load_reference_metrics(path: Path | None = None) -> dict[str, Any]:
    """Read the committed warehouse reference.

    Args:
        path: JSON file. The default is ``docs/experiments/reference_metrics.json``.

    Raises:
        ValueError: A required field is missing, or a model row is not usable.
    """
    metrics_path: Path = REFERENCE_METRICS_PATH if path is None else path
    payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{metrics_path} must contain an object")
    missing: list[str] = [key for key in _REFERENCE_FILE_KEYS if key not in payload]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"{metrics_path} is missing {missing_list}")
    tolerance: float | None = _reference_metric_number(payload["tolerance"])
    if tolerance is None or tolerance < 0.0:
        raise ValueError("tolerance must be a finite number >= 0")
    models: object = payload["models"]
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("models must be a non-empty list")
    seen: set[tuple[str, str]] = set()
    for row in models:
        if not isinstance(row, dict):
            raise ValueError("each reference model must be an object")
        model_name: object = row.get("model")
        feature_set: object = row.get("feature_set")
        if not isinstance(model_name, str) or not isinstance(feature_set, str):
            raise ValueError("each reference model needs model and feature_set strings")
        identity: tuple[str, str] = (model_name, feature_set)
        if identity in seen:
            raise ValueError(f"duplicate reference row {model_name}/{feature_set}")
        seen.add(identity)
        metric_keys: list[str] = [key for key in row if key not in {"model", "feature_set"}]
        if not metric_keys:
            raise ValueError(f"{model_name}/{feature_set} has no metrics")
        for key in metric_keys:
            if key not in _REFERENCE_POOLED_FIELDS:
                known: str = ", ".join(sorted(_REFERENCE_POOLED_FIELDS))
                raise ValueError(
                    f"{model_name}/{feature_set} has unknown metric {key}. Known: {known}."
                )
            if _reference_metric_number(row[key]) is None:
                raise ValueError(f"{model_name}/{feature_set} {key} must be a finite number")
    for key in ("cv_start_season", "cv_end_season", "n_bootstrap", "bootstrap_seed"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            raise ValueError(f"{key} must be an integer")
    return payload


def compare_reference_metrics(
    report: dict[str, Any],
    reference: dict[str, Any],
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Compare pooled reference rows to a rolling-origin report.

    A field fails when its absolute difference is greater than ``tolerance``.
    Sample size uses the same rule, so a different ``n`` fails.
    """
    band: float = float(reference["tolerance"]) if tolerance is None else float(tolerance)
    indexed: dict[tuple[str, str], dict[str, Any]] = {
        (result["model"], result["feature_set"]): result for result in report["results"]
    }
    failures: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for expected_row in reference["models"]:
        model_name: str = str(expected_row["model"])
        feature_set: str = str(expected_row["feature_set"])
        result: dict[str, Any] | None = indexed.get((model_name, feature_set))
        pooled: dict[str, Any] = {}
        if result is not None and result.get("pooled") is not None:
            pooled = result["pooled"]
        observed_row: dict[str, Any] = {"model": model_name, "feature_set": feature_set}
        for field, expected_value in expected_row.items():
            if field in {"model", "feature_set"}:
                continue
            expected_number: float | None = _reference_metric_number(expected_value)
            actual_number: float | None = _reference_metric_number(pooled.get(field))
            observed_row[field] = actual_number
            if expected_number is None or actual_number is None:
                failures.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "field": field,
                        "observed": actual_number,
                        "expected": expected_number,
                        "abs_diff": None,
                    }
                )
                continue
            difference: float = abs(actual_number - expected_number)
            if difference > band:
                failures.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "field": field,
                        "observed": actual_number,
                        "expected": expected_number,
                        "abs_diff": difference,
                    }
                )
        observed.append(observed_row)
    return {
        "ok": len(failures) == 0,
        "tolerance": band,
        "failures": failures,
        "observed": observed,
    }


def format_reference_check(result: dict[str, Any]) -> str:
    """Render a reference check for the process log."""
    tolerance: float = float(result["tolerance"])
    if result["ok"]:
        lines: list[str] = [f"Reference metrics match within {tolerance:.1e}."]
        for row in result["observed"]:
            log_loss_value: object = row.get("log_loss")
            n_value: object = row.get("n")
            rendered_loss: str = (
                f"{float(log_loss_value):.6f}" if isinstance(log_loss_value, int | float) else "n/a"
            )
            rendered_n: str = (
                str(int(n_value))
                if isinstance(n_value, int | float) and not isinstance(n_value, bool)
                else "n/a"
            )
            lines.append(
                f"{row['model']} {row['feature_set']} log_loss={rendered_loss} n={rendered_n}"
            )
        return "\n".join(lines)
    lines = [
        "Reference metrics differ from docs/experiments/reference_metrics.json "
        f"(tolerance {tolerance:.1e})."
    ]
    for failure in result["failures"]:
        lines.append(
            f"{failure['model']} {failure['feature_set']} {failure['field']}: "
            f"observed {failure['observed']} expected {failure['expected']} "
            f"abs_diff {failure['abs_diff']}"
        )
    return "\n".join(lines)
