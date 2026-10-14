"""Phase 0 audit of the current pregame inputs.

The audit does not add, drop, or redefine features. It profiles the matrix
from ``prepare_training_data`` and the logistic pipeline from ``train.py``.

2025 is included only in missingness and in the train-versus-2024-versus-2025
distribution. It is not scored and it is not used to choose a model.
Season-wise univariate CV uses seasons inside the configured window
(default 2016 through 2023). Permutation importance is fit on the training
seasons and scored on 2024.
"""

import argparse
import os
from pathlib import Path
from typing import Any, NamedTuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score

from src.features.pregame import (
    MODEL_ID_COLUMNS,
    build_pregame_features,
    prepare_training_data,
    split_by_season,
)
from src.train.train import (
    activate_tracking_experiment,
    get_git_sha,
    load_data,
    train_logistic_regression,
)
from src.utils.config import DEFAULT_MLFLOW_TRACKING_URI, resolve_tracking_uri
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

PHASE0_EXPERIMENT_NAME: str = "nfl_pregame/phase0_audit"
PHASE0_TAG: str = "0-audit"
MARKDOWN_NAME: str = "feature_audit.md"
CSV_NAME: str = "feature_audit.csv"
SLICE_NAMES: tuple[str, ...] = ("train", "val", "test")

FEATURE_TABLE_COLUMNS: tuple[str, ...] = (
    "feature",
    "missing_count",
    "missing_rate",
    "train_missing_rate",
    "val_missing_rate",
    "test_missing_rate",
    "train_min",
    "train_max",
    "train_mean",
    "train_median",
    "val_min",
    "val_max",
    "val_mean",
    "val_median",
    "val_out_of_range_count",
    "test_min",
    "test_max",
    "test_mean",
    "test_median",
    "test_out_of_range_count",
    "out_of_range",
    "time_counter",
    "cv_log_loss",
    "cv_log_loss_ci_low",
    "cv_log_loss_ci_high",
    "cv_auc",
    "cv_auc_ci_low",
    "cv_auc_ci_high",
    "cv_n_oof",
    "cv_folds_used",
    "cv_folds_skipped",
    "max_abs_spearman",
    "most_correlated_with",
    "redundant",
    "permutation_importance_mean",
    "permutation_importance_std",
)


class FeatureAuditConfig(NamedTuple):
    """Settings for one input audit. Defaults match the training split."""

    train_end_season: int = 2023
    val_season: int = 2024
    test_season: int = 2025
    cv_start_season: int = 2016
    cv_end_season: int = 2023
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    correlation_threshold: float = 0.90
    permutation_repeats: int = 10
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    random_state: int = 42


def make_feature_audit_config(**overrides: Any) -> FeatureAuditConfig:
    """Build a config and reject ranges that cannot be run.

    Args:
        **overrides: Fields of ``FeatureAuditConfig`` to replace.

    Returns:
        Validated config.

    Raises:
        ValueError: A season window, bootstrap count, repeat count, or
            correlation threshold is outside its allowed range.
        TypeError: An override is not a config field.
    """
    config = FeatureAuditConfig(**overrides)
    if config.cv_start_season > config.cv_end_season:
        raise ValueError("cv_start_season must be <= cv_end_season")
    if config.n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    if config.permutation_repeats < 1:
        raise ValueError("permutation_repeats must be >= 1")
    if not 0 < config.correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1]")
    return config


def _feature_names(features: pd.DataFrame) -> list[str]:
    """Return model columns, excluding game identifiers."""
    return [column for column in features.columns if column not in MODEL_ID_COLUMNS]


def _validate_inputs(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Align rows and reject a frame that cannot be audited.

    Returns:
        Dict with keys ``features``, ``target``, and ``feature_names``.
    """
    if "season" not in features.columns:
        raise ValueError("features must include a season column")
    if len(features) == 0:
        raise ValueError("features must contain at least one row")
    if len(features) != len(target):
        raise ValueError("features and target must have the same number of rows")

    aligned_features = features.reset_index(drop=True)
    numeric_target = pd.to_numeric(pd.Series(target).reset_index(drop=True), errors="coerce")
    if numeric_target.isna().any():
        raise ValueError("target must be numeric 0/1 with no nulls")
    if not np.isin(numeric_target.to_numpy(), [0, 1]).all():
        raise ValueError("target must contain only classes 0 and 1")
    if set(numeric_target.astype(int).unique().tolist()) != {0, 1}:
        raise ValueError("target must include both classes 0 and 1")

    feature_names = _feature_names(aligned_features)
    if not feature_names:
        raise ValueError("features must include at least one model column")
    for name in feature_names:
        if not pd.api.types.is_numeric_dtype(aligned_features[name]):
            raise ValueError(f"feature {name} must be numeric")

    return {
        "features": aligned_features,
        "target": numeric_target.astype(int),
        "feature_names": feature_names,
    }


def _slice_frames(
    features: pd.DataFrame,
    target: pd.Series,
    config: FeatureAuditConfig,
) -> dict[str, Any]:
    """Split with the same season rule as training.

    ``test_target`` is returned so the split stays complete. Callers must
    not score it.
    """
    train_x, val_x, test_x, train_y, val_y, test_y = split_by_season(
        features,
        target,
        train_end_season=config.train_end_season,
        val_season=config.val_season,
        test_season=config.test_season,
    )
    return {
        "train_features": train_x,
        "val_features": val_x,
        "test_features": test_x,
        "train_target": train_y,
        "val_target": val_y,
        "test_target": test_y,
    }


def _numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    """Summarize one numeric series. Sample std is omitted below two points."""
    observed = values.dropna()
    missing_count = int(values.isna().sum())
    missing_rate = float(missing_count / len(values)) if len(values) else None
    if observed.empty:
        return {
            "n": int(len(values)),
            "n_missing": missing_count,
            "missing_rate": missing_rate,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "n": int(len(values)),
        "n_missing": missing_count,
        "missing_rate": missing_rate,
        "min": float(observed.min()),
        "max": float(observed.max()),
        "mean": float(observed.mean()),
        "median": float(observed.median()),
    }


def _season_medians(features: pd.DataFrame, column: str) -> list[float]:
    """Median of one feature by season, skipping seasons that are all null."""
    medians: list[tuple[int, float]] = []
    for season, part in features.groupby("season", sort=True):
        observed = part[column].dropna()
        if observed.empty:
            continue
        medians.append((int(season), float(observed.median())))
    medians.sort(key=lambda item: item[0])
    return [value for _, value in medians]


def _strictly_monotone(values: list[float]) -> bool:
    """True when at least three points strictly rise or strictly fall."""
    if len(values) < 3:
        return False
    increasing = all(later > earlier for earlier, later in zip(values, values[1:]))
    decreasing = all(later < earlier for earlier, later in zip(values, values[1:]))
    return increasing or decreasing


# A later slice is a clock when more than this share of its rows sits
# outside the closed training range. 5 percent is the phase 1 bar.
TIME_COUNTER_OUT_OF_RANGE_SHARE: float = 0.05


def _out_of_range_share_exceeds(count: int | None, n_rows: int, threshold: float) -> bool:
    """True when ``count / n_rows`` is strictly above ``threshold``.

    A null count or an empty slice is not a material share.
    """
    if count is None or n_rows <= 0:
        return False
    return (count / n_rows) > threshold


def _count_outside(values: pd.Series, low: float | None, high: float | None) -> int | None:
    """Count non-null values outside the closed training range."""
    if low is None or high is None:
        return None
    observed = values.dropna()
    if observed.empty:
        return 0
    outside = (observed < low) | (observed > high)
    return int(outside.sum())


def _distribution_flags(
    features: pd.DataFrame,
    slices: dict[str, Any],
    column: str,
) -> dict[str, Any]:
    """Compare 2024 and 2025 to the training range, and flag a time counter.

    A time counter is a strictly monotone season median (at least three
    seasons), or a later slice where the share of rows outside the closed
    training range is above ``TIME_COUNTER_OUT_OF_RANGE_SHARE``. An
    entirely-outside rule missed columns that only partly leave the
    training range. The rule is not limited to the ``games_played`` name.
    """
    train_summary = _numeric_summary(slices["train_features"][column])
    val_summary = _numeric_summary(slices["val_features"][column])
    test_summary = _numeric_summary(slices["test_features"][column])
    train_min = train_summary["min"]
    train_max = train_summary["max"]
    val_outside = _count_outside(slices["val_features"][column], train_min, train_max)
    test_outside = _count_outside(slices["test_features"][column], train_min, train_max)
    out_of_range = bool((val_outside or 0) > 0 or (test_outside or 0) > 0)

    val_rows: int = len(slices["val_features"])
    test_rows: int = len(slices["test_features"])
    material_shift: bool = _out_of_range_share_exceeds(
        val_outside, val_rows, TIME_COUNTER_OUT_OF_RANGE_SHARE
    ) or _out_of_range_share_exceeds(test_outside, test_rows, TIME_COUNTER_OUT_OF_RANGE_SHARE)
    monotone: bool = _strictly_monotone(_season_medians(features, column))
    time_counter: bool = material_shift or monotone

    return {
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
        "val_out_of_range_count": val_outside,
        "test_out_of_range_count": test_outside,
        "out_of_range": out_of_range,
        "time_counter": time_counter,
    }


def _positive_probability(pipeline: Any, frame: pd.DataFrame) -> np.ndarray:
    """Probability of class 1 from a fitted logistic pipeline."""
    classifier = pipeline.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)
    return pipeline.predict_proba(frame)[:, positive_index]


def _fold_skip_reason(train_target: pd.Series, eval_target: pd.Series) -> str | None:
    """Return why a CV fold cannot be fit, or None when it can."""
    if len(train_target) == 0:
        return "no_train_rows"
    if len(eval_target) == 0:
        return "no_eval_rows"
    if train_target.nunique() < 2:
        return "single_class_train"
    if eval_target.nunique() < 2:
        return "single_class_eval"
    return None


def bootstrap_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Point log loss and AUC, plus a percentile bootstrap interval.

    Resamples are drawn with replacement from the pooled out-of-fold rows.
    A resample with one class is kept for log loss and skipped for AUC.

    Args:
        y_true: Binary labels.
        y_proba: Predicted probabilities of class 1.
        n_bootstrap: Number of resamples. Zero skips the interval.
        seed: Seed for the resampler.

    Returns:
        Dict of point estimates, interval endpoints, and sample counts.
    """
    empty: dict[str, float | int | None] = {
        "log_loss": None,
        "log_loss_ci_low": None,
        "log_loss_ci_high": None,
        "auc": None,
        "auc_ci_low": None,
        "auc_ci_high": None,
        "n_oof": int(len(y_true)),
        "n_auc_resamples": 0,
    }
    if len(y_true) == 0:
        return empty

    point_log_loss = float(log_loss(y_true, y_proba, labels=[0, 1]))
    both_classes = len(np.unique(y_true)) > 1
    point_auc = float(roc_auc_score(y_true, y_proba)) if both_classes else None
    if n_bootstrap == 0 or len(y_true) < 2:
        return {
            "log_loss": point_log_loss,
            "log_loss_ci_low": None,
            "log_loss_ci_high": None,
            "auc": point_auc,
            "auc_ci_low": None,
            "auc_ci_high": None,
            "n_oof": int(len(y_true)),
            "n_auc_resamples": 0,
        }

    generator = np.random.default_rng(seed)
    log_loss_samples: list[float] = []
    auc_samples: list[float] = []
    row_count = len(y_true)
    for _ in range(n_bootstrap):
        drawn = generator.integers(0, row_count, size=row_count)
        y_drawn = y_true[drawn]
        proba_drawn = y_proba[drawn]
        log_loss_samples.append(float(log_loss(y_drawn, proba_drawn, labels=[0, 1])))
        if len(np.unique(y_drawn)) > 1:
            auc_samples.append(float(roc_auc_score(y_drawn, proba_drawn)))

    auc_low: float | None = None
    auc_high: float | None = None
    if auc_samples:
        auc_low = float(np.percentile(auc_samples, 2.5))
        auc_high = float(np.percentile(auc_samples, 97.5))
    return {
        "log_loss": point_log_loss,
        "log_loss_ci_low": float(np.percentile(log_loss_samples, 2.5)),
        "log_loss_ci_high": float(np.percentile(log_loss_samples, 97.5)),
        "auc": point_auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "n_oof": int(row_count),
        "n_auc_resamples": len(auc_samples),
    }


def _univariate_one_feature(
    features: pd.DataFrame,
    target: pd.Series,
    feature_name: str,
    config: FeatureAuditConfig,
) -> dict[str, Any]:
    """Fit a one-column logistic on each prior-season fold and pool the scores."""
    labels: list[int] = []
    probabilities: list[float] = []
    skipped: list[dict[str, Any]] = []
    folds_used = 0

    for season in range(config.cv_start_season, config.cv_end_season + 1):
        train_mask = features["season"] < season
        eval_mask = features["season"] == season
        train_target = target.loc[train_mask]
        eval_target = target.loc[eval_mask]
        reason = _fold_skip_reason(train_target, eval_target)
        if reason is not None:
            skipped.append({"feature": feature_name, "season": season, "reason": reason})
            continue

        model_columns = ["season", feature_name]
        try:
            pipeline, _fold_metrics = train_logistic_regression(
                features.loc[train_mask, model_columns],
                train_target,
                features.loc[eval_mask, model_columns],
                eval_target,
                C=config.logistic_c,
                max_iter=config.logistic_max_iter,
                random_state=config.random_state,
            )
        except ValueError as exc:
            logger.warning(
                "Univariate fit failed for %s season %s: %s",
                feature_name,
                season,
                exc,
            )
            skipped.append({"feature": feature_name, "season": season, "reason": "fit_failed"})
            continue
        eval_frame = features.loc[eval_mask, [feature_name]]
        proba = _positive_probability(pipeline, eval_frame)
        labels.extend(int(value) for value in eval_target.to_numpy())
        probabilities.extend(float(value) for value in proba)
        folds_used += 1

    metrics = bootstrap_binary_metrics(
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        n_bootstrap=config.n_bootstrap,
        seed=config.bootstrap_seed,
    )
    return {
        "metrics": metrics,
        "folds_used": folds_used,
        "folds_skipped": len(skipped),
        "skipped": skipped,
    }


def _spearman_report(
    train_features: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Spearman correlation on training rows, and pairs at or above the threshold."""
    matrix = train_features[feature_names].corr(method="spearman")
    pairs: list[dict[str, Any]] = []
    per_feature: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(feature_names):
        others = matrix[left].drop(labels=[left])
        absolute = others.abs()
        if absolute.dropna().empty:
            per_feature[left] = {
                "max_abs_spearman": None,
                "most_correlated_with": None,
                "redundant": False,
            }
        else:
            partner = str(absolute.idxmax())
            value = float(absolute.loc[partner])
            per_feature[left] = {
                "max_abs_spearman": value,
                "most_correlated_with": partner,
                "redundant": bool(value >= threshold),
            }
        for right in feature_names[left_index + 1 :]:
            raw = matrix.loc[left, right]
            if pd.isna(raw):
                continue
            if abs(float(raw)) >= threshold:
                pairs.append({"left": left, "right": right, "spearman": float(raw)})
    return {"matrix": matrix, "pairs": pairs, "per_feature": per_feature}


def _permutation_report(
    slices: dict[str, Any],
    feature_names: list[str],
    config: FeatureAuditConfig,
) -> dict[str, Any]:
    """Permute the current logistic model on 2024. Skip when 2024 cannot score.

    Importance is the drop in negative log loss. A larger value means the
    2024 log loss gets worse when that column is shuffled. Correlated inputs
    share credit, so a small value does not prove the column is useless.
    """
    train_features = slices["train_features"]
    val_features = slices["val_features"]
    train_target = slices["train_target"]
    val_target = slices["val_target"]
    skipped: dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        "importances": {},
    }
    if len(train_features) == 0:
        skipped["reason"] = "no_training_rows"
        return skipped
    if train_target.nunique() < 2:
        skipped["reason"] = "single_class_train"
        return skipped
    if len(val_features) == 0:
        skipped["reason"] = "no_validation_rows"
        return skipped
    if val_target.nunique() < 2:
        skipped["reason"] = "single_class_validation"
        return skipped

    try:
        pipeline, _metrics = train_logistic_regression(
            train_features,
            train_target,
            val_features,
            val_target,
            C=config.logistic_c,
            max_iter=config.logistic_max_iter,
            random_state=config.random_state,
        )
    except ValueError as exc:
        logger.warning("Permutation fit failed: %s", exc)
        skipped["reason"] = "fit_failed"
        return skipped
    scored = permutation_importance(
        pipeline,
        val_features[feature_names],
        val_target,
        scoring="neg_log_loss",
        n_repeats=config.permutation_repeats,
        random_state=config.random_state,
    )
    importances: dict[str, dict[str, float]] = {}
    for index, name in enumerate(feature_names):
        importances[name] = {
            "mean": float(scored.importances_mean[index]),
            "std": float(scored.importances_std[index]),
        }
    return {"status": "ok", "reason": None, "importances": importances}


def _assemble_table(
    feature_names: list[str],
    features: pd.DataFrame,
    distributions: dict[str, dict[str, Any]],
    univariate: dict[str, dict[str, Any]],
    correlation: dict[str, Any],
    permutation: dict[str, Any],
) -> pd.DataFrame:
    """One row per feature with every audit field."""
    rows: list[dict[str, Any]] = []
    overall_missing = features[feature_names].isna()
    for name in feature_names:
        distribution = distributions[name]
        metrics = univariate[name]["metrics"]
        corr = correlation["per_feature"][name]
        importance = permutation["importances"].get(name, {})
        missing_count = int(overall_missing[name].sum())
        row: dict[str, Any] = {
            "feature": name,
            "missing_count": missing_count,
            "missing_rate": float(missing_count / len(features)),
            "train_missing_rate": distribution["train"]["missing_rate"],
            "val_missing_rate": distribution["val"]["missing_rate"],
            "test_missing_rate": distribution["test"]["missing_rate"],
            "train_min": distribution["train"]["min"],
            "train_max": distribution["train"]["max"],
            "train_mean": distribution["train"]["mean"],
            "train_median": distribution["train"]["median"],
            "val_min": distribution["val"]["min"],
            "val_max": distribution["val"]["max"],
            "val_mean": distribution["val"]["mean"],
            "val_median": distribution["val"]["median"],
            "val_out_of_range_count": distribution["val_out_of_range_count"],
            "test_min": distribution["test"]["min"],
            "test_max": distribution["test"]["max"],
            "test_mean": distribution["test"]["mean"],
            "test_median": distribution["test"]["median"],
            "test_out_of_range_count": distribution["test_out_of_range_count"],
            "out_of_range": distribution["out_of_range"],
            "time_counter": distribution["time_counter"],
            "cv_log_loss": metrics["log_loss"],
            "cv_log_loss_ci_low": metrics["log_loss_ci_low"],
            "cv_log_loss_ci_high": metrics["log_loss_ci_high"],
            "cv_auc": metrics["auc"],
            "cv_auc_ci_low": metrics["auc_ci_low"],
            "cv_auc_ci_high": metrics["auc_ci_high"],
            "cv_n_oof": metrics["n_oof"],
            "cv_folds_used": univariate[name]["folds_used"],
            "cv_folds_skipped": univariate[name]["folds_skipped"],
            "max_abs_spearman": corr["max_abs_spearman"],
            "most_correlated_with": corr["most_correlated_with"],
            "redundant": corr["redundant"],
            "permutation_importance_mean": importance.get("mean"),
            "permutation_importance_std": importance.get("std"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=list(FEATURE_TABLE_COLUMNS))


def _is_missing_cell(value: object) -> bool:
    """True for None and NaN. False for text."""
    if value is None or isinstance(value, str):
        return value is None
    return bool(pd.isna(value))


def _format_cell(value: object) -> str:
    """Render one markdown cell. Missing values become ``n/a``."""
    if isinstance(value, bool | np.bool_):
        return "yes" if bool(value) else "no"
    if _is_missing_cell(value):
        return "n/a"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a GitHub-flavored table."""
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_format_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the audit as markdown. Numbers come only from ``report``."""
    table: pd.DataFrame = report["feature_table"]
    config: dict[str, Any] = report["config"]
    lines: list[str] = [
        "# Phase 0 feature input audit",
        "",
        "Diagnostic only. Features are unchanged. 2025 appears in the",
        "distribution tables and is not scored. Univariate CV trains on seasons",
        "strictly before the scored season. Permutation importance uses the",
        "current logistic pipeline, fit on the training seasons and scored on",
        f"{config['val_season']}.",
        "",
        "## Settings",
        "",
        f"- Train seasons: through {config['train_end_season']}",
        f"- Validation season: {config['val_season']}",
        f"- Test season (distribution only): {config['test_season']}",
        f"- Univariate CV seasons: {config['cv_start_season']} through {config['cv_end_season']}",
        f"- Bootstrap resamples: {config['n_bootstrap']} (seed {config['bootstrap_seed']})",
        f"- Spearman redundancy threshold: {config['correlation_threshold']}",
        f"- Permutation repeats: {config['permutation_repeats']}",
        "",
        "The logistic step is the training pipeline: EPA columns filled with 0,",
        "win-rate columns filled with 0.5, other columns left as-is, then scaled.",
        "",
        "## Missingness",
        "",
        _markdown_table(
            ["feature", "missing_rate", "train", "val", "test"],
            [
                [
                    row.feature,
                    row.missing_rate,
                    row.train_missing_rate,
                    row.val_missing_rate,
                    row.test_missing_rate,
                ]
                for row in table.itertuples(index=False)
            ],
        ),
        "",
        "## Distribution",
        "",
        "Out of range counts values in that slice outside the closed training",
        "min and max. `time_counter` is yes when the share of 2024 or 2025",
        "rows outside that range is above 0.05, or when season medians are",
        "strictly monotone across at least three seasons.",
        "",
        _markdown_table(
            [
                "feature",
                "train_min",
                "train_max",
                "val_min",
                "val_max",
                "val_out_of_range",
                "test_min",
                "test_max",
                "test_out_of_range",
                "time_counter",
            ],
            [
                [
                    row.feature,
                    row.train_min,
                    row.train_max,
                    row.val_min,
                    row.val_max,
                    row.val_out_of_range_count,
                    row.test_min,
                    row.test_max,
                    row.test_out_of_range_count,
                    row.time_counter,
                ]
                for row in table.itertuples(index=False)
            ],
        ),
        "",
        "## Univariate signal",
        "",
        "Each number is a one-feature logistic on rolling-origin season folds.",
        "Intervals are the 2.5 and 97.5 percentiles of a bootstrap of the",
        "pooled out-of-fold rows.",
        "",
        _markdown_table(
            [
                "feature",
                "log_loss",
                "log_loss_low",
                "log_loss_high",
                "auc",
                "auc_low",
                "auc_high",
                "folds_used",
            ],
            [
                [
                    row.feature,
                    row.cv_log_loss,
                    row.cv_log_loss_ci_low,
                    row.cv_log_loss_ci_high,
                    row.cv_auc,
                    row.cv_auc_ci_low,
                    row.cv_auc_ci_high,
                    row.cv_folds_used,
                ]
                for row in table.itertuples(index=False)
            ],
        ),
        "",
        "## Correlation and redundancy",
        "",
        "Spearman correlation on training rows. A feature is redundant when its",
        f"strongest other-feature absolute correlation is at least {config['correlation_threshold']}.",
        "",
    ]

    matrix: pd.DataFrame = report["correlation_matrix"]
    matrix_headers = ["feature", *list(matrix.columns)]
    matrix_rows: list[list[object]] = []
    for name in matrix.index:
        matrix_rows.append([name, *[matrix.loc[name, column] for column in matrix.columns]])
    lines.append(_markdown_table(matrix_headers, matrix_rows))
    lines.append("")
    pairs: list[dict[str, Any]] = report["redundant_pairs"]
    if pairs:
        lines.append("Redundant pairs:")
        lines.append("")
        lines.append(
            _markdown_table(
                ["left", "right", "spearman"],
                [[pair["left"], pair["right"], pair["spearman"]] for pair in pairs],
            )
        )
    else:
        lines.append("No pair reached the redundancy threshold.")
    lines.extend(
        [
            "",
            "## Permutation importance",
            "",
        ]
    )
    if report["permutation_status"] != "ok":
        lines.append(
            f"Skipped ({report['permutation_reason']}). The logistic model was not scored."
        )
    else:
        lines.append(
            "Mean drop in negative log loss on the validation season when the column is shuffled."
        )
        lines.append("")
        lines.append(
            _markdown_table(
                ["feature", "importance_mean", "importance_std"],
                [
                    [
                        row.feature,
                        row.permutation_importance_mean,
                        row.permutation_importance_std,
                    ]
                    for row in table.itertuples(index=False)
                ],
            )
        )
    skipped_folds: list[dict[str, Any]] = report["skipped_folds"]
    lines.extend(["", "## Skipped univariate folds", ""])
    if skipped_folds:
        lines.append(
            _markdown_table(
                ["feature", "season", "reason"],
                [[item["feature"], item["season"], item["reason"]] for item in skipped_folds],
            )
        )
    else:
        lines.append("Every configured fold was fit.")
    lines.append("")
    return "\n".join(lines)


def run_feature_audit(
    features: pd.DataFrame,
    target: pd.Series,
    config: FeatureAuditConfig | None = None,
) -> dict[str, Any]:
    """Profile each current model input and return tables plus markdown.

    Args:
        features: Matrix from ``prepare_training_data`` (identifiers allowed).
        target: Binary home-win labels, aligned by row position.
        config: Audit settings. Defaults match the training split.

    Returns:
        Dict with ``feature_table``, ``correlation_matrix``, ``redundant_pairs``,
        ``skipped_folds``, ``permutation_status``, ``permutation_reason``,
        ``config``, and ``markdown``.

    Raises:
        ValueError: The frame or config cannot be audited.
    """
    resolved = make_feature_audit_config(**(config or FeatureAuditConfig())._asdict())
    checked = _validate_inputs(features, target)
    aligned: pd.DataFrame = checked["features"]
    labels: pd.Series = checked["target"]
    feature_names: list[str] = checked["feature_names"]
    slices = _slice_frames(aligned, labels, resolved)

    distributions = {name: _distribution_flags(aligned, slices, name) for name in feature_names}
    univariate = {
        name: _univariate_one_feature(aligned, labels, name, resolved) for name in feature_names
    }
    correlation = _spearman_report(
        slices["train_features"], feature_names, resolved.correlation_threshold
    )
    permutation = _permutation_report(slices, feature_names, resolved)
    feature_table = _assemble_table(
        feature_names,
        aligned,
        distributions,
        univariate,
        correlation,
        permutation,
    )
    skipped_folds: list[dict[str, Any]] = []
    for name in feature_names:
        skipped_folds.extend(univariate[name]["skipped"])

    report: dict[str, Any] = {
        "feature_table": feature_table,
        "correlation_matrix": correlation["matrix"],
        "redundant_pairs": correlation["pairs"],
        "skipped_folds": skipped_folds,
        "permutation_status": permutation["status"],
        "permutation_reason": permutation["reason"],
        "config": resolved._asdict(),
        "markdown": "",
    }
    report["markdown"] = render_markdown_report(report)
    return report


def write_audit_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write the markdown report and the feature CSV.

    Args:
        report: Result of ``run_feature_audit``.
        output_dir: Directory to create if needed.

    Returns:
        Dict with ``markdown_path`` and ``csv_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / MARKDOWN_NAME
    csv_path = output_dir / CSV_NAME
    markdown_path.write_text(report["markdown"], encoding="utf-8")
    table: pd.DataFrame = report["feature_table"]
    table.to_csv(csv_path, index=False)
    return {"markdown_path": str(markdown_path), "csv_path": str(csv_path)}


def log_audit_to_mlflow(
    report: dict[str, Any],
    paths: dict[str, str],
    tracking_uri: str,
    extra_params: dict[str, Any] | None = None,
    experiment_name: str = PHASE0_EXPERIMENT_NAME,
) -> dict[str, str]:
    """Log the report files on a run tagged ``phase=0-audit``.

    Args:
        report: Result of ``run_feature_audit``.
        paths: Paths returned by ``write_audit_outputs``.
        tracking_uri: MLflow tracking URI, passed through ``resolve_tracking_uri``.
        extra_params: More params, such as the lookback used to build features.
        experiment_name: Experiment to create or activate.

    Returns:
        Dict with ``run_id``, ``experiment_name``, and ``tracking_uri``.
    """
    resolved_uri = resolve_tracking_uri(tracking_uri)
    activate_tracking_experiment(experiment_name, resolved_uri)
    table: pd.DataFrame = report["feature_table"]
    with mlflow.start_run(run_name="phase0-input-audit") as active_run:
        mlflow.set_tag("phase", PHASE0_TAG)
        mlflow.set_tag("git_commit", get_git_sha())
        mlflow.set_tag(
            "mlflow.note.content",
            "Phase 0 input audit. Diagnostic only. Features are unchanged. "
            "2025 is profiled for range shift and is not used to score or select a model.",
        )
        params: dict[str, Any] = {
            "train_end_season": report["config"]["train_end_season"],
            "val_season": report["config"]["val_season"],
            "test_season": report["config"]["test_season"],
            "cv_start_season": report["config"]["cv_start_season"],
            "cv_end_season": report["config"]["cv_end_season"],
            "n_bootstrap": report["config"]["n_bootstrap"],
            "bootstrap_seed": report["config"]["bootstrap_seed"],
            "correlation_threshold": report["config"]["correlation_threshold"],
            "permutation_repeats": report["config"]["permutation_repeats"],
            "n_features": int(len(table)),
            "permutation_status": report["permutation_status"],
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        mlflow.log_metric("n_features_out_of_range", float(table["out_of_range"].sum()))
        mlflow.log_metric("n_features_time_counter", float(table["time_counter"].sum()))
        mlflow.log_metric("n_redundant_features", float(table["redundant"].sum()))
        mlflow.log_artifact(paths["markdown_path"])
        mlflow.log_artifact(paths["csv_path"])
        run_id = active_run.info.run_id
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "tracking_uri": resolved_uri,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the audit CLI."""
    parser = argparse.ArgumentParser(
        description="Audit pregame model inputs. Does not change features."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/phase0",
        help="Directory for feature_audit.md and feature_audit.csv",
    )
    parser.add_argument(
        "--fixture-path",
        type=str,
        default=None,
        help="Fixture directory with fct_games.parquet and fct_team_game_epa.parquet",
    )
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--train-end-season", type=int, default=2023)
    parser.add_argument("--val-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--cv-start-season", type=int, default=2016)
    parser.add_argument("--cv-end-season", type=int, default=2023)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Build current features, audit them, write the report, and log MLflow.

    Args:
        argv: CLI arguments. ``None`` reads the process arguments.

    Returns:
        Dict with ``report``, ``paths``, and ``run``.
    """
    args = _parse_args(argv)
    config = make_feature_audit_config(
        train_end_season=args.train_end_season,
        val_season=args.val_season,
        test_season=args.test_season,
        cv_start_season=args.cv_start_season,
        cv_end_season=args.cv_end_season,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        correlation_threshold=args.correlation_threshold,
        permutation_repeats=args.permutation_repeats,
        logistic_c=args.C,
        logistic_max_iter=args.max_iter,
        random_state=args.seed,
    )
    fixture_path: str | None = args.fixture_path if args.fixture_path else None
    games, team_games = load_data(fixture_path=fixture_path)
    built = build_pregame_features(
        games,
        team_games,
        lookback_games=args.lookback,
    )
    model_frame, target = prepare_training_data(games, built)
    report = run_feature_audit(model_frame, target, config)
    paths = write_audit_outputs(report, Path(args.output_dir))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    run = log_audit_to_mlflow(
        report,
        paths,
        tracking_uri=tracking_uri,
        extra_params={"lookback_games": args.lookback},
    )
    logger.info("Wrote %s and %s", paths["markdown_path"], paths["csv_path"])
    logger.info("MLflow run %s tagged phase=%s", run["run_id"], PHASE0_TAG)
    return {"report": report, "paths": paths, "run": run}


if __name__ == "__main__":  # pragma: no cover
    main()
