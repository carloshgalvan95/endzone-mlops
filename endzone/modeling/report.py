"""Markdown and CSV comparison reports. Numbers come only from the report dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from endzone.features.pregame import (
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_MEAN_RATING,
    ELO_SCALE,
    ELO_SEASON_REVERSION,
    QB_PRIOR_DROPBACKS,
    SEASON_BLEND_K,
    SEASON_SHRINK_FACTOR,
)
from endzone.modeling.evaluation import MIN_LOG_LOSS_IMPROVEMENT
from endzone.modeling.hypotheses import (
    FINAL_VALIDATION_SEASON,
    HELD_OUT_TEST_SEASON,
    report_profile_for,
)
from endzone.modeling.models import DEFAULT_L2_C_GRID, _format_regularization_c

PHASE1_MARKDOWN_NAME: str = "comparison.md"
PHASE1_CSV_NAME: str = "comparison.csv"
COMPARISON_COLUMNS: tuple[str, ...] = (
    "model",
    "feature_set",
    "status",
    "reason",
    "n",
    "log_loss",
    "brier",
    "auc",
    "accuracy",
    "log_loss_diff",
    "log_loss_diff_ci_low",
    "log_loss_diff_ci_high",
)


def comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per model with pooled metrics and the paired log loss gap."""
    rows: list[dict[str, Any]] = []
    for result in results:
        pooled: dict[str, Any] = result["pooled"] or {}
        comparison: dict[str, Any] = result["comparison"] or {}
        rows.append(
            {
                "model": result["model"],
                "feature_set": result["feature_set"],
                "status": result["status"],
                "reason": result["reason"],
                "n": pooled.get("n") if result["pooled"] else comparison.get("n"),
                "log_loss": pooled.get("log_loss"),
                "brier": pooled.get("brier"),
                "auc": pooled.get("auc"),
                "accuracy": pooled.get("accuracy"),
                "log_loss_diff": comparison.get("log_loss_diff"),
                "log_loss_diff_ci_low": comparison.get("log_loss_diff_ci_low"),
                "log_loss_diff_ci_high": comparison.get("log_loss_diff_ci_high"),
            }
        )
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def _format_metric(value: object) -> str:
    """Render one report cell. Missing values become ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        return str(value)
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a GitHub-flavored table."""
    head: str = "| " + " | ".join(headers) + " |"
    separator: str = "| " + " | ".join("---" for _ in headers) + " |"
    body: list[str] = [
        "| " + " | ".join(_format_metric(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([head, separator, *body])


def _render_acceptance_section(report: dict[str, Any]) -> list[str]:
    """Render the acceptance rule and the per-row decision."""
    minimum: float = float(report.get("min_log_loss_improvement", MIN_LOG_LOSS_IMPROVEMENT))
    rows: list[dict[str, Any]] = report.get("acceptance") or []
    lines: list[str] = [
        "## Acceptance",
        "",
        "Accept a feature set only when both are true:",
        f"pooled log loss improves by at least {minimum:.3f}",
        f"(`log_loss_diff` <= -{minimum:.3f}) and the 95% paired interval",
        "excludes 0 by lying entirely below 0 (`log_loss_diff_ci_high` < 0).",
        "Otherwise a scored row is rejected. A row without an interval is",
        "inconclusive. The reference row is not a candidate.",
        "",
    ]
    if not rows:
        lines.append("No acceptance rows were computed.")
        lines.append("")
        return lines
    table_rows: list[list[object]] = [
        [
            row["model"],
            row["feature_set"],
            row["log_loss_diff"],
            row["ci_low"],
            row["ci_high"],
            row["decision"],
        ]
        for row in rows
    ]
    lines.append(
        _markdown_table(
            ["model", "feature_set", "log_loss_diff", "ci_low", "ci_high", "decision"],
            table_rows,
        )
    )
    lines.append("")
    return lines


def _format_optional_float(value: float | None) -> str:
    """Format a metric, or ``missing`` when the interval was not computed."""
    if value is None or pd.isna(value):
        return "missing"
    return f"{float(value):.4f}"


def _qb_change_subset_lines(subset: dict[str, Any]) -> list[str]:
    """Markdown for the descriptive QB-change slice. Not an acceptance rule."""
    lines: list[str] = [
        "## QB change subset",
        "",
        "Games where the depth-chart starter differs from the previous primary passer.",
        "This section is descriptive. It is not an acceptance decision.",
        f"Depth-chart starter sides: {subset['n_depth_chart_starters']}.",
        f"Previous-passer fallback sides: {subset['n_previous_passer_starters']}.",
        f"Games with a QB change: {subset['n_qb_change_games']}.",
    ]
    if subset["status"] != "scored" or subset["comparison"] is None:
        reason: str = str(subset["reason"] or "not_scored")
        lines.append(f"Subset comparison skipped ({reason}).")
    else:
        comparison: dict[str, Any] = subset["comparison"]
        lines.append(
            "On "
            f"{comparison['n']} out-of-fold games with a QB change, "
            f"h5_qb log loss is {_format_optional_float(comparison['log_loss'])} "
            f"and the reference is {_format_optional_float(comparison['reference_log_loss'])}. "
            "The difference is "
            f"{_format_optional_float(comparison['log_loss_diff'])} "
            "(interval "
            f"{_format_optional_float(comparison['log_loss_diff_ci_low'])} to "
            f"{_format_optional_float(comparison['log_loss_diff_ci_high'])})."
        )
    lines.append("")
    return lines


def _season_regression_lines() -> list[str]:
    """Shared season-defense paragraph for H2, H3, and ridge."""
    return [
        "Season defensive EPA is the same-season prior mean of EPA allowed.",
        "Regressed EPA starts from the previous season's final value,",
        f"shrunk by {SEASON_SHRINK_FACTOR:.3f} toward that season's league mean,",
        "then blends in current-season games with weight",
        f"n/(n+{SEASON_BLEND_K:g}). Week 1 of a season with no prior stays missing.",
    ]


def _pass_rush_lines() -> list[str]:
    """Pass and rush sentences appended after the season-defense paragraph."""
    return [
        "Pass and rush EPA per play split offense and defense.",
        "Recent splits use the lookback window on prior games,",
        "including prior seasons. Regressed-season splits use the",
        "same shrinkage and blend. Defensive splits are the",
        "opponent's offensive pass or rush rate from that game.",
    ]


def _profile_lines(profile: str) -> list[str]:
    """Extra sentences inserted after the lookback line. Empty for phase 1 and H1."""
    if profile == "clean_epa":
        return [
            "Clean EPA keeps pass and run plays only.",
            "no_play, kneels, spikes, and garbage time are excluded.",
            "Garbage time is pre-snap wp below 0.05 or above 0.95.",
            "A missing wp is kept. Those cutoffs are fixed.",
            "Recent, season, and regressed clean EPA use the same windows",
            f"as h2_full, shrunk by {SEASON_SHRINK_FACTOR:.3f} toward the",
            "clean league mean, then blended with weight",
            f"n/(n+{SEASON_BLEND_K:g}).",
            "h4_clean_sr adds the recent clean success rate.",
            "A missing success rate is filled with 0.5.",
        ]
    if profile == "starting_qb":
        return [
            "The starter is the offense QB1 on that week's depth chart.",
            "A missing chart uses the previous game's primary passer.",
            "The rating is EPA per dropback, shrunk toward replacement 0",
            f"with weight n/(n+{QB_PRIOR_DROPBACKS:g}).",
            "Only dropbacks from earlier gamedays count, across teams and seasons.",
            "h5_qb is h2_full plus the home and away ratings and each rating",
            "minus that team's recent passing EPA.",
            "CPOE is stored on the dropback mart and is not in the rating.",
        ]
    if profile == "season_regression":
        return _season_regression_lines()
    if profile == "pass_rush":
        return _season_regression_lines() + _pass_rush_lines()
    if profile == "ridge":
        grid_text: str = ", ".join(_format_regularization_c(value) for value in DEFAULT_L2_C_GRID)
        return (
            _season_regression_lines()
            + _pass_rush_lines()
            + [
                "logistic_l2 is L2 logistic regression. C is chosen inside",
                "each training fold by an expanding season split on those",
                "training seasons only. The scored season is not used to",
                f"pick C. The grid is {grid_text}.",
                "A fold with no usable inner season keeps the fallback C.",
                "The fold table records chosen_c.",
            ]
        )
    return []


def _fold_section(result: dict[str, Any]) -> list[str]:
    """Markdown for one model's fold table, or a skip line."""
    lines: list[str] = [f"### {result['model']} ({result['feature_set']})", ""]
    if result["status"] != "ok":
        lines.append(f"Skipped ({result['reason']}).")
        lines.append("")
        if not result["folds"]:
            return lines
    records_chosen_c: bool = any("chosen_c" in fold for fold in result["folds"])
    fold_headers: list[str] = [
        "season",
        "status",
        "reason",
        "n_train",
        "n_eval",
        "train_seasons",
        "log_loss",
    ]
    if records_chosen_c:
        fold_headers.extend(["chosen_c", "c_selection"])
    fold_rows: list[list[object]] = []
    for fold in result["folds"]:
        fold_row: list[object] = [
            fold["season"],
            fold["status"],
            fold["reason"],
            fold["n_train"],
            fold["n_eval"],
            ",".join(str(season) for season in fold["train_seasons"]),
            fold["log_loss"],
        ]
        if records_chosen_c:
            fold_row.extend([fold.get("chosen_c"), fold.get("c_selection")])
        fold_rows.append(fold_row)
    lines.append(_markdown_table(fold_headers, fold_rows))
    lines.append("")
    return lines


def render_phase1_report(report: dict[str, Any]) -> str:
    """Render the rolling-origin comparison. Numbers come only from ``report``."""
    config: dict[str, Any] = report["config"]
    table: pd.DataFrame = report["comparison_table"]
    title: str = str(report.get("title") or "Phase 1 baselines")
    reference_feature_set: str = str(report.get("reference_feature_set") or "none")
    if reference_feature_set == "none":
        reference_line: str = f"Reference model: `{report['reference_model']}`."
    else:
        reference_line = (
            f"Reference model: `{report['reference_model']}` "
            f"on feature set `{reference_feature_set}`."
        )
    lines: list[str] = [
        f"# {title}",
        "",
        "Rolling-origin CV. For each season S from",
        f"{config['cv_start_season']} through {config['cv_end_season']}, a model",
        "trains on seasons strictly before S and scores S.",
        f"{FINAL_VALIDATION_SEASON} is the final validation and is not scored.",
        f"{HELD_OUT_TEST_SEASON} is not scored.",
        "",
        reference_line,
        "`log_loss_diff` is candidate log loss minus reference log loss on the",
        "same out-of-fold games. A negative difference favors the candidate.",
        "The interval is the 2.5 and 97.5 percentiles of a paired bootstrap",
        f"({config['n_bootstrap']} resamples, seed {config['bootstrap_seed']}).",
        "Accuracy treats a probability of 0.5 or higher as a home win.",
        "",
        "Elo uses K "
        f"{ELO_K:g}, home advantage {ELO_HOME_ADVANTAGE:g}, mean rating "
        f"{ELO_MEAN_RATING:g}, and season reversion {ELO_SEASON_REVERSION:.3f} "
        f"on a scale of {ELO_SCALE:g}. There is no margin-of-victory multiplier.",
        "The market probability is the no-vig closing moneyline. It is skipped",
        "when those columns are absent.",
        "",
        f"Lookback games: {report['lookback_games']}.",
        "",
        "## Comparison",
        "",
        _markdown_table(
            list(COMPARISON_COLUMNS),
            [list(row) for row in table.itertuples(index=False, name=None)],
        ),
        "",
        "## Folds",
        "",
    ]
    subset: dict[str, Any] | None = report.get("qb_change_subset")
    if subset is not None:
        fold_at: int = lines.index("## Folds")
        lines[fold_at:fold_at] = _qb_change_subset_lines(subset)
    profile_lines: list[str] = _profile_lines(report_profile_for(report.get("hypothesis")))
    if profile_lines:
        lookback_index: int = lines.index(f"Lookback games: {report['lookback_games']}.")
        lines[lookback_index + 1 : lookback_index + 1] = profile_lines
    if report.get("hypothesis"):
        lines.insert(2, f"Hypothesis: `{report['hypothesis']}`.")
        lines.insert(3, "")
    for result in report["results"]:
        lines.extend(_fold_section(result))
    if report.get("apply_acceptance"):
        lines.extend(_render_acceptance_section(report))
    lines.append("No model is promoted from this report.")
    lines.append("")
    return "\n".join(lines)


def write_phase1_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write the comparison markdown, the comparison CSV, and per-model OOF files.

    Returns:
        Dict with ``markdown_path``, ``csv_path``, and ``oof_paths``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path: Path = output_dir / PHASE1_MARKDOWN_NAME
    csv_path: Path = output_dir / PHASE1_CSV_NAME
    markdown_path.write_text(report["markdown"], encoding="utf-8")
    table: pd.DataFrame = report["comparison_table"]
    table.to_csv(csv_path, index=False)
    oof_paths: dict[str, str] = {}
    for result in report["results"]:
        oof: pd.DataFrame | None = result["oof"]
        if oof is None:
            continue
        file_name: str = f"oof_{result['model']}_{result['feature_set']}.csv"
        oof_path: Path = output_dir / file_name
        oof.to_csv(oof_path, index=False)
        oof_paths[f"{result['model']}|{result['feature_set']}"] = str(oof_path)
    return {
        "markdown_path": str(markdown_path),
        "csv_path": str(csv_path),
        "oof_paths": oof_paths,
    }
