"""MAE, paired bootstrap, interval coverage, and the Sprint 2 acceptance rule.

ADR-007. The thresholds come from the hypothesis file. This module does not
move them. Market MAE, RMSE, coverage, and implied win log loss are reported.
They are not gates.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Shown when the week file has no score-model result yet.
HIDDEN_NOT_RUN: str = "Estimate hidden because the pre-registered test has not been run."
# Shown when a target was scored and did not pass ADR-007.
HIDDEN_FAILED: str = "Estimate hidden because it did not pass its pre-registered test."
_PROBABILITY_CLIP: float = 1e-6


def normal_interval_z(coverage: float) -> float:
    """Standard normal quantile for a central interval.

    An 80% interval uses the 0.90 quantile, about 1.2816.

    Raises:
        ValueError: ``coverage`` is not strictly between 0 and 1.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    tail: float = (1.0 - float(coverage)) / 2.0
    return float(NormalDist().inv_cdf(1.0 - tail))


def standard_normal_cdf(value: float) -> float:
    """Phi(value) for one standardized number."""
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def split_estimated_score(margin: float, total: float) -> dict[str, float]:
    """Turn a margin and a total into home and away points.

    Home is ``(total + margin) / 2``. Away is ``(total - margin) / 2``.
    Display rounding happens later. These values stay exact.

    Raises:
        ValueError: Either input is not finite.
    """
    if not math.isfinite(margin) or not math.isfinite(total):
        raise ValueError("margin and total must be finite")
    return {
        "home": (float(total) + float(margin)) / 2.0,
        "away": (float(total) - float(margin)) / 2.0,
    }


def absolute_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | None]:
    """Pooled MAE and RMSE. An empty sample leaves both estimates missing."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    usable: np.ndarray = np.isfinite(actual) & np.isfinite(predicted)
    n_rows: int = int(usable.sum())
    if n_rows == 0:
        return {"mae": None, "rmse": None, "n": 0}
    error: np.ndarray = actual[usable] - predicted[usable]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "n": n_rows,
    }


def paired_mae_difference(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_baseline: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Paired bootstrap of model MAE minus baseline MAE.

    The same row indexes are drawn for both prediction vectors. A negative
    difference means the model has lower MAE than the baseline.

    Raises:
        ValueError: The arrays differ in length, or ``n_bootstrap`` is negative.
    """
    actual = np.asarray(y_true, dtype=float)
    model = np.asarray(y_model, dtype=float)
    baseline = np.asarray(y_baseline, dtype=float)
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    if not (len(actual) == len(model) == len(baseline)):
        raise ValueError("y_true, y_model, and y_baseline must have the same length")
    usable: np.ndarray = np.isfinite(actual) & np.isfinite(model) & np.isfinite(baseline)
    actual = actual[usable]
    model = model[usable]
    baseline = baseline[usable]
    empty: dict[str, float | int | None] = {
        "mae": None,
        "baseline_mae": None,
        "mae_diff": None,
        "mae_diff_ci_low": None,
        "mae_diff_ci_high": None,
        "n": int(len(actual)),
        "n_bootstrap": int(n_bootstrap),
    }
    if len(actual) == 0:
        return empty
    model_mae: float = float(np.mean(np.abs(actual - model)))
    baseline_mae: float = float(np.mean(np.abs(actual - baseline)))
    difference: float = model_mae - baseline_mae
    if n_bootstrap == 0 or len(actual) < 2:
        return {
            "mae": model_mae,
            "baseline_mae": baseline_mae,
            "mae_diff": difference,
            "mae_diff_ci_low": None,
            "mae_diff_ci_high": None,
            "n": int(len(actual)),
            "n_bootstrap": int(n_bootstrap),
        }
    generator = np.random.default_rng(seed)
    row_count: int = len(actual)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        drawn: np.ndarray = generator.integers(0, row_count, size=row_count)
        drawn_difference: float = float(np.mean(np.abs(actual[drawn] - model[drawn]))) - float(
            np.mean(np.abs(actual[drawn] - baseline[drawn]))
        )
        samples.append(drawn_difference)
    return {
        "mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_diff": difference,
        "mae_diff_ci_low": float(np.percentile(samples, 2.5)),
        "mae_diff_ci_high": float(np.percentile(samples, 97.5)),
        "n": row_count,
        "n_bootstrap": int(n_bootstrap),
    }


def season_improvement_table(
    seasons: np.ndarray,
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_baseline: np.ndarray,
) -> dict[str, Any]:
    """Per-season MAE and how many seasons the model is strictly better.

    A season counts as improved only when its model MAE is lower than its
    baseline MAE. Ties do not count.
    """
    frame = pd.DataFrame(
        {
            "season": np.asarray(seasons),
            "y_true": np.asarray(y_true, dtype=float),
            "y_model": np.asarray(y_model, dtype=float),
            "y_baseline": np.asarray(y_baseline, dtype=float),
        }
    )
    rows: list[dict[str, float | int | bool]] = []
    improved: int = 0
    for season, part in frame.groupby("season", sort=True):
        actual: np.ndarray = part["y_true"].to_numpy(dtype=float)
        model: np.ndarray = part["y_model"].to_numpy(dtype=float)
        baseline: np.ndarray = part["y_baseline"].to_numpy(dtype=float)
        usable: np.ndarray = np.isfinite(actual) & np.isfinite(model) & np.isfinite(baseline)
        if int(usable.sum()) == 0:
            continue
        model_mae: float = float(np.mean(np.abs(actual[usable] - model[usable])))
        baseline_mae: float = float(np.mean(np.abs(actual[usable] - baseline[usable])))
        better: bool = model_mae < baseline_mae
        if better:
            improved += 1
        rows.append(
            {
                "season": int(season),
                "n": int(usable.sum()),
                "mae": model_mae,
                "baseline_mae": baseline_mae,
                "improved": better,
            }
        )
    return {
        "seasons_improved": improved,
        "seasons_scored": len(rows),
        "seasons": rows,
    }


def normal_interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    coverage: float,
) -> dict[str, float | int | None]:
    """Share of outcomes inside a normal interval of the requested mass.

    ``sigma`` is one residual standard deviation per row. Non-positive or
    missing sigma values are left out of the share.
    """
    z_value: float = normal_interval_z(coverage)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    width = np.asarray(sigma, dtype=float)
    if not (len(actual) == len(predicted) == len(width)):
        raise ValueError("y_true, y_pred, and sigma must have the same length")
    usable: np.ndarray = (
        np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(width) & (width > 0.0)
    )
    n_rows: int = int(usable.sum())
    if n_rows == 0:
        return {"coverage": None, "n": 0, "z": z_value}
    inside: np.ndarray = np.abs(actual[usable] - predicted[usable]) <= (z_value * width[usable])
    return {"coverage": float(np.mean(inside)), "n": n_rows, "z": z_value}


def implied_win_log_loss(
    y_home_won: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float | int | None]:
    """Log loss of ``Phi(mu / sigma)`` against the home-win label.

    Rows with a missing label, a non-binary label, or a non-positive sigma
    are left out.
    """
    labels = np.asarray(y_home_won, dtype=float)
    center = np.asarray(mu, dtype=float)
    width = np.asarray(sigma, dtype=float)
    if not (len(labels) == len(center) == len(width)):
        raise ValueError("y_home_won, mu, and sigma must have the same length")
    usable: np.ndarray = (
        np.isfinite(labels)
        & np.isin(labels, [0.0, 1.0])
        & np.isfinite(center)
        & np.isfinite(width)
        & (width > 0.0)
    )
    n_rows: int = int(usable.sum())
    if n_rows == 0:
        return {"log_loss": None, "n": 0}
    raw: np.ndarray = np.array(
        [
            standard_normal_cdf(float(center_i) / float(width_i))
            for center_i, width_i in zip(center[usable], width[usable], strict=True)
        ],
        dtype=float,
    )
    clipped: np.ndarray = np.clip(raw, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
    return {
        "log_loss": float(log_loss(labels[usable], clipped, labels=[0, 1])),
        "n": n_rows,
    }


def decide_mae_rule(
    mae_diff: float | None,
    ci_low: float | None,
    ci_high: float | None,
    seasons_improved: int | None,
    seasons_scored: int | None,
    *,
    min_seasons_improved: int,
    season_count: int,
) -> str:
    """Accept only when the MAE interval is below 0 and enough seasons improve.

    A missing interval, or a window that did not score every required season,
    is inconclusive. Every other scored comparison is rejected.
    """
    missing: bool = any(
        value is None or (isinstance(value, float) and not math.isfinite(value))
        for value in (mae_diff, ci_low, ci_high, seasons_improved, seasons_scored)
    )
    if missing:
        return "inconclusive"
    assert mae_diff is not None
    assert ci_low is not None
    assert ci_high is not None
    assert seasons_improved is not None
    assert seasons_scored is not None
    if int(seasons_scored) != int(season_count):
        return "inconclusive"
    interval_below_zero: bool = float(ci_low) <= float(ci_high) and float(ci_high) < 0.0
    beats_baseline: bool = float(mae_diff) < 0.0 and interval_below_zero
    enough_seasons: bool = int(seasons_improved) >= int(min_seasons_improved)
    if beats_baseline and enough_seasons:
        return "accepted"
    return "rejected"


def summarize_regression(
    *,
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_baseline: np.ndarray,
    y_market: np.ndarray,
    sigma: np.ndarray,
    seasons: np.ndarray,
    y_home_won: np.ndarray | None,
    n_bootstrap: int,
    seed: int,
    min_seasons_improved: int,
    season_count: int,
    interval_coverage: float,
    include_implied_win: bool,
) -> dict[str, Any]:
    """Pooled metrics and the acceptance decision for one target."""
    paired: dict[str, float | int | None] = paired_mae_difference(
        y_true,
        y_model,
        y_baseline,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    errors: dict[str, float | int | None] = absolute_error_metrics(y_true, y_model)
    by_season: dict[str, Any] = season_improvement_table(seasons, y_true, y_model, y_baseline)
    covered: dict[str, float | int | None] = normal_interval_coverage(
        y_true,
        y_model,
        sigma,
        interval_coverage,
    )
    market_actual = np.asarray(y_true, dtype=float)
    market_pred = np.asarray(y_market, dtype=float)
    market_mask: np.ndarray = np.isfinite(market_actual) & np.isfinite(market_pred)
    market_mae: float | None = None
    market_n: int = int(market_mask.sum())
    if market_n > 0:
        market_mae = float(np.mean(np.abs(market_actual[market_mask] - market_pred[market_mask])))
    implied: dict[str, float | int | None] = {"log_loss": None, "n": 0}
    if include_implied_win:
        if y_home_won is None:
            raise ValueError("implied win log loss needs home-win labels")
        implied = implied_win_log_loss(y_home_won, y_model, sigma)
    decision: str = decide_mae_rule(
        _as_float(paired["mae_diff"]),
        _as_float(paired["mae_diff_ci_low"]),
        _as_float(paired["mae_diff_ci_high"]),
        _as_int(by_season["seasons_improved"]),
        _as_int(by_season["seasons_scored"]),
        min_seasons_improved=min_seasons_improved,
        season_count=season_count,
    )
    return {
        "decision": decision,
        "mae": errors["mae"],
        "rmse": errors["rmse"],
        "baseline_mae": paired["baseline_mae"],
        "mae_diff": paired["mae_diff"],
        "mae_diff_ci_low": paired["mae_diff_ci_low"],
        "mae_diff_ci_high": paired["mae_diff_ci_high"],
        "n": errors["n"],
        "n_bootstrap": paired["n_bootstrap"],
        "market_mae": market_mae,
        "market_n": market_n,
        "coverage_80": covered["coverage"],
        "coverage_n": covered["n"],
        "coverage_z": covered["z"],
        "seasons_improved": by_season["seasons_improved"],
        "seasons_scored": by_season["seasons_scored"],
        "seasons": by_season["seasons"],
        "implied_win_log_loss": implied["log_loss"],
        "implied_win_n": implied["n"],
    }


def _as_float(value: object) -> float | None:
    """Return a finite float, or None."""
    if value is None:
        return None
    try:
        number: float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value: object) -> int | None:
    """Return an int, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
