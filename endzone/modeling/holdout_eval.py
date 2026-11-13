"""One-shot 2024 validation and 2025 test for the locked Phase 2 winner.

The recipe is fixed before the warehouse read. Logistic ``pkg_core_inj_l2``
and logistic ``h2_full`` are refit on ``season < S`` for each scored season.
Elo, the margin-of-victory Elo gap, the home prior, and the no-vig market
are the comparison rows. This module does not register or promote a model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from endzone.config import configured_tracking_uri
from endzone.features.opponent import ELO_MOV_DIFF_COLUMN
from endzone.features.pregame import (
    FEATURE_SET_H2_FULL,
    FEATURE_SET_PKG_CORE_INJ_L2,
    PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ,
    build_pregame_features,
    elo_home_probabilities,
    market_home_probabilities,
    prepare_training_data,
)
from endzone.features.qbelo import rating_win_probability
from endzone.modeling.dataset import load_data, load_injury_inputs, load_qb_inputs
from endzone.modeling.evaluation import (
    N_CALIBRATION_BINS,
    binary_classification_metrics,
    market_information_regression,
    paired_log_loss_difference,
    reliability_by_decile,
)
from endzone.modeling.models import (
    IDENTIFIER_COLUMNS,
    model_feature_columns,
    train_baseline_home,
    train_logistic_regression,
)
from endzone.modeling.tracking import activate_tracking_experiment, get_git_sha
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

HOLDOUT_EXPERIMENT_NAME: str = "nfl_pregame/holdout_2024_2025"
HOLDOUT_JSON_NAME: str = "holdout_2024_2025.json"
HOLDOUT_MARKDOWN_NAME: str = "holdout_2024_2025.md"
SELECTED_MODEL: str = "logistic"
SELECTED_FEATURE_SET: str = FEATURE_SET_PKG_CORE_INJ_L2
REFERENCE_FEATURE_SET: str = FEATURE_SET_H2_FULL
SELECTED_C: float = 0.1
REFERENCE_C: float = 1.0
LOOKBACK: int = 10
MAX_ITER: int = 1000
SEED: int = 42
N_BOOTSTRAP: int = 1000
BOOTSTRAP_SEED: int = 42
EVAL_SEASONS: tuple[int, ...] = (2024, 2025)
POOLED_SLICE: str = "pooled_2024_2025"
CV_LOG_LOSS: float = 0.6330
CV_N: int = 2191
SELECTED_N_COLUMNS: int = 47
SELECTED_KEY: str = f"{SELECTED_MODEL}|{SELECTED_FEATURE_SET}"
REFERENCE_KEY: str = f"{SELECTED_MODEL}|{REFERENCE_FEATURE_SET}"
MARKET_KEY: str = "market|market"

_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model": SELECTED_MODEL,
        "feature_set": SELECTED_FEATURE_SET,
        "role": "selected",
        "C": SELECTED_C,
    },
    {
        "model": SELECTED_MODEL,
        "feature_set": REFERENCE_FEATURE_SET,
        "role": "prior_reference",
        "C": REFERENCE_C,
    },
    {"model": "elo", "feature_set": "elo", "role": "context", "C": None},
    {"model": "elo_mov", "feature_set": "elo_mov", "role": "context", "C": None},
    {"model": "baseline-home", "feature_set": "none", "role": "context", "C": None},
    {"model": "market", "feature_set": "market", "role": "benchmark", "C": None},
)


def default_report_dir() -> Path:
    """Return the frozen-report directory at the repository root."""
    return Path(__file__).resolve().parents[2] / "docs" / "experiments" / "reports"


def refuse_existing_holdout(report_dir: Path, *, allow_overwrite: bool) -> None:
    """Stop when the frozen JSON is already on disk.

    The check is the path only. It does not read the warehouse.

    Raises:
        SystemExit: The JSON exists and ``allow_overwrite`` is false.
    """
    path: Path = report_dir / HOLDOUT_JSON_NAME
    if path.exists() and not allow_overwrite:
        raise SystemExit(
            f"{path} already exists. The frozen holdout report runs only once. "
            "Pass --allow-overwrite only to replace that file on purpose."
        )


def run_holdout_from_warehouse() -> dict[str, Any]:
    """Load the marts and score the locked 2024 and 2025 holdout.

    Returns:
        The report dict. Nothing is written and nothing is registered.
    """
    games, team_games = load_data()
    expected_starters, qb_dropbacks = load_qb_inputs(None)
    depth_chart_starters, injury_reports = load_injury_inputs(None)
    return run_holdout_evaluation(
        games,
        team_games,
        depth_chart_starters=depth_chart_starters,
        injury_reports=injury_reports,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
    )


def run_holdout_evaluation(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    depth_chart_starters: pd.DataFrame | None = None,
    injury_reports: pd.DataFrame | None = None,
    expected_starters: pd.DataFrame | None = None,
    qb_dropbacks: pd.DataFrame | None = None,
    lookback: int = LOOKBACK,
    max_iter: int = MAX_ITER,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score 2024 and 2025 with training seasons strictly before each one.

    Pooled metrics concatenate the two out-of-sample prediction sets.
    Published rows share the games where every model has a finite probability.

    Returns:
        Metadata, parameters, per-season and pooled metrics, calibration,
        market-information coefficients, and the aligned game-level rows.

    Raises:
        ValueError: The locked column count moved, a scored season has no
            aligned games, or a training season is not strictly earlier.
    """
    if len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ) != SELECTED_N_COLUMNS:
        raise ValueError(
            f"{SELECTED_FEATURE_SET} must stay at {SELECTED_N_COLUMNS} columns, "
            f"found {len(PREGAME_FEATURE_COLUMNS_PKG_CORE_INJ)}"
        )
    features: pd.DataFrame = build_pregame_features(
        games,
        team_games,
        lookback_games=lookback,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
        depth_chart_starters=depth_chart_starters,
        injury_reports=injury_reports,
    )
    matrices: dict[str, tuple[pd.DataFrame, pd.Series]] = {
        SELECTED_FEATURE_SET: _labeled_matrix(games, features, SELECTED_FEATURE_SET),
        REFERENCE_FEATURE_SET: _labeled_matrix(games, features, REFERENCE_FEATURE_SET),
    }
    predictions: dict[str, pd.DataFrame] = {}
    fits: list[dict[str, Any]] = []
    for spec in _MODEL_SPECS:
        if spec["model"] != "logistic":
            continue
        feature_set: str = str(spec["feature_set"])
        strength: float = float(spec["C"])
        matrix, target = matrices[feature_set]
        for eval_season in EVAL_SEASONS:
            frame, fit = _fit_logistic_season(
                matrix,
                target,
                eval_season=eval_season,
                feature_set=feature_set,
                strength=strength,
                max_iter=max_iter,
                seed=seed,
            )
            _store_predictions(predictions, spec, frame)
            fits.append(fit)
    selected_matrix, selected_target = matrices[SELECTED_FEATURE_SET]
    for eval_season in EVAL_SEASONS:
        frame, fit = _fit_home_prior_season(
            selected_matrix,
            selected_target,
            eval_season=eval_season,
        )
        baseline = next(spec for spec in _MODEL_SPECS if spec["model"] == "baseline-home")
        _store_predictions(predictions, baseline, frame)
        fits.append(fit)
    _store_lookup_predictions(predictions, games, features)
    season_frames: dict[int, pd.DataFrame] = {
        season: _align_season(predictions, season) for season in EVAL_SEASONS
    }
    for season, aligned in season_frames.items():
        if len(aligned) == 0:
            raise ValueError(f"season {season} has no games with every probability finite")
    pooled: pd.DataFrame = pd.concat(
        [season_frames[season] for season in EVAL_SEASONS],
        ignore_index=True,
    )
    season_reports: list[dict[str, Any]] = []
    for season in EVAL_SEASONS:
        season_fits: list[dict[str, Any]] = [
            fit for fit in fits if int(fit["eval_season"]) == season
        ]
        selected_fit: dict[str, Any] = next(
            fit for fit in season_fits if fit["feature_set"] == SELECTED_FEATURE_SET
        )
        labeled_count: int = _labeled_count(selected_matrix, season)
        season_reports.append(
            {
                "eval_season": season,
                "train_season_max": int(selected_fit["train_season_max"]),
                "train_seasons": list(selected_fit["train_seasons"]),
                "n_train": int(selected_fit["n_train"]),
                "n_labeled": labeled_count,
                "n_eval": int(len(season_frames[season])),
                "n_excluded_unaligned": labeled_count - int(len(season_frames[season])),
                "fits": season_fits,
                "rows": _metric_rows(
                    season_frames[season],
                    n_bootstrap=n_bootstrap,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
        )
    report: dict[str, Any] = {
        "metadata": _metadata(),
        "parameters": _parameters(
            lookback=lookback,
            max_iter=max_iter,
            seed=seed,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        ),
        "pooled": {
            "slice": POOLED_SLICE,
            "n": int(len(pooled)),
            "rows": _metric_rows(pooled, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed),
        },
        "seasons": season_reports,
        "calibration": _calibration_rows(season_frames, pooled),
        "market_information": _market_rows(
            season_frames,
            pooled,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        ),
        "oof": _oof_summary(season_frames),
    }
    return _plain(report)


def write_holdout_reports(report: dict[str, Any], directory: Path) -> dict[str, Path]:
    """Write the JSON and markdown for one holdout report.

    Returns:
        Dict with ``json`` and ``markdown`` paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    json_path: Path = directory / HOLDOUT_JSON_NAME
    markdown_path: Path = directory / HOLDOUT_MARKDOWN_NAME
    payload: str = json.dumps(report, indent=2, allow_nan=False) + "\n"
    json_path.write_text(payload, encoding="utf-8")
    markdown_path.write_text(render_holdout_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_holdout_markdown(report: dict[str, Any]) -> str:
    """Render the holdout report. Numbers come only from ``report``."""
    lines: list[str] = [
        "# 2024 validation and 2025 test",
        "",
        "ADR-010. This holdout runs once. The result is published whether or not",
        "holdout log loss is worse than the 2016-2023 CV log loss. No model is",
        "selected after seeing 2025. This command does not register or promote a model.",
        "The market row is a benchmark. On the reference table the market has the",
        "lower log loss. This is not betting advice.",
        "",
        f"Selected model: logistic `{SELECTED_FEATURE_SET}`, C = {SELECTED_C}, "
        f"{SELECTED_N_COLUMNS} columns. CV log loss {CV_LOG_LOSS}, n = {CV_N}.",
        f"Prior reference: logistic `{REFERENCE_FEATURE_SET}`, C = {REFERENCE_C}.",
        "",
    ]
    lines.extend(_markdown_table("Pooled 2024-2025", report["pooled"]["rows"]))
    for season in report["seasons"]:
        title: str = (
            f"Season {season['eval_season']} "
            f"(train seasons through {season['train_season_max']}, "
            f"n_train {season['n_train']}, n_eval {season['n_eval']})"
        )
        lines.extend(_markdown_table(title, season["rows"]))
    lines.append("## Calibration")
    lines.append("")
    for item in report["calibration"]:
        lines.append(
            f"- `{item['model']}` `{item['feature_set']}` {item['slice']}: "
            f"ECE {_format_number(item['ece'])}, n_bins {item['n_bins']}, n {item['n']}"
        )
    lines.append("")
    lines.append("## Market information")
    lines.append("")
    for item in report["market_information"]:
        lines.append(
            f"- `{item['model']}` `{item['feature_set']}` {item['slice']}: "
            f"market_coef {_format_number(item['market_coef'])}, "
            f"model_coef {_format_number(item['model_coef'])}, n {item['n']}"
        )
    lines.append("")
    return "\n".join(lines)


def log_holdout_run(report: dict[str, Any], *, artifact_path: Path | None = None) -> None:
    """Log params and metrics for the holdout. Does not register a model.

    The experiment name is ``nfl_pregame/holdout_2024_2025``.
    ``activate_tracking_experiment`` is the only tracking-path rewrite.
    """
    activate_tracking_experiment(HOLDOUT_EXPERIMENT_NAME, configured_tracking_uri())
    selected: dict[str, Any] = _row(report["pooled"]["rows"], SELECTED_FEATURE_SET)
    reference: dict[str, Any] = _row(report["pooled"]["rows"], REFERENCE_FEATURE_SET)
    market: dict[str, Any] = _row(report["pooled"]["rows"], "market")
    metrics: dict[str, float] = {}
    for name, row in (
        ("log_loss_selected", selected),
        ("log_loss_h2_full", reference),
        ("log_loss_market", market),
    ):
        if isinstance(row["log_loss"], float):
            metrics[name] = row["log_loss"]
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": SELECTED_MODEL,
                "feature_set": SELECTED_FEATURE_SET,
                "C": SELECTED_C,
                "reference_C": REFERENCE_C,
                "lookback": report["parameters"]["lookback"],
                "seed": report["parameters"]["seed"],
                "max_iter": report["parameters"]["max_iter"],
                "n_bootstrap": report["parameters"]["n_bootstrap"],
                "bootstrap_seed": report["parameters"]["bootstrap_seed"],
                "eval_seasons": "2024,2025",
            }
        )
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "git_commit": get_git_sha(),
                "registered": "false",
                "promoted": "false",
                "stage": "holdout",
            }
        )
        if artifact_path is not None:
            mlflow.log_artifact(str(artifact_path))


def _labeled_matrix(
    games: pd.DataFrame,
    features: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature rows whose label is 0 or 1."""
    matrix, target = prepare_training_data(games, features, feature_set=feature_set)
    matrix = matrix.reset_index(drop=True)
    target = target.reset_index(drop=True)
    labels: pd.Series = pd.to_numeric(target, errors="coerce")
    keep: pd.Series = labels.isin([0, 1])
    return matrix.loc[keep].reset_index(drop=True), labels.loc[keep].astype(int).reset_index(
        drop=True
    )


def _fit_logistic_season(
    matrix: pd.DataFrame,
    target: pd.Series,
    *,
    eval_season: int,
    feature_set: str,
    strength: float,
    max_iter: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one logistic on ``season < eval_season`` and score that season."""
    train_index, eval_index = _season_indexes(matrix, eval_season)
    train_frame: pd.DataFrame = matrix.loc[train_index]
    eval_frame: pd.DataFrame = matrix.loc[eval_index]
    pipeline, _metrics = train_logistic_regression(
        train_frame,
        target.loc[train_index],
        eval_frame,
        target.loc[eval_index],
        C=strength,
        max_iter=max_iter,
        random_state=seed,
    )
    columns: list[str] = model_feature_columns(train_frame)
    probabilities: np.ndarray = pipeline.predict_proba(eval_frame[columns])[:, 1]
    frame: pd.DataFrame = _prediction_frame(eval_frame, target.loc[eval_index], probabilities)
    fit: dict[str, Any] = _fit_record(
        model="logistic",
        feature_set=feature_set,
        eval_season=eval_season,
        train_frame=train_frame,
        strength=strength,
        n_columns=len(columns),
    )
    return frame, fit


def _fit_home_prior_season(
    matrix: pd.DataFrame,
    target: pd.Series,
    *,
    eval_season: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit the training home-win rate on ``season < eval_season``."""
    train_index, eval_index = _season_indexes(matrix, eval_season)
    train_frame: pd.DataFrame = matrix.loc[train_index]
    eval_frame: pd.DataFrame = matrix.loc[eval_index]
    model, _metrics = train_baseline_home(
        train_frame,
        target.loc[train_index],
        eval_frame,
        target.loc[eval_index],
    )
    positive: int = list(model.classes_).index(1)
    probabilities: np.ndarray = model.predict_proba(
        eval_frame.drop(columns=list(IDENTIFIER_COLUMNS))
    )[:, positive]
    frame: pd.DataFrame = _prediction_frame(eval_frame, target.loc[eval_index], probabilities)
    fit: dict[str, Any] = _fit_record(
        model="baseline-home",
        feature_set="none",
        eval_season=eval_season,
        train_frame=train_frame,
        strength=None,
        n_columns=None,
    )
    return frame, fit


def _season_indexes(matrix: pd.DataFrame, eval_season: int) -> tuple[pd.Index, pd.Index]:
    """Return train and eval indexes. Training seasons are strictly earlier."""
    seasons: pd.Series = pd.to_numeric(matrix["season"], errors="raise")
    train_mask: pd.Series = seasons < eval_season
    eval_mask: pd.Series = seasons == eval_season
    if int(train_mask.sum()) == 0 or int(eval_mask.sum()) == 0:
        raise ValueError(f"season {eval_season} needs earlier training rows and eval rows")
    train_seasons: set[int] = {int(season) for season in seasons.loc[train_mask].unique()}
    if max(train_seasons) >= eval_season or eval_season in train_seasons:
        raise ValueError(f"season {eval_season} leaked into its training rows")
    return matrix.index[train_mask.to_numpy()], matrix.index[eval_mask.to_numpy()]


def _fit_record(
    *,
    model: str,
    feature_set: str,
    eval_season: int,
    train_frame: pd.DataFrame,
    strength: float | None,
    n_columns: int | None,
) -> dict[str, Any]:
    """Describe one refit. The eval season is absent from ``train_seasons``."""
    train_seasons: list[int] = sorted(
        int(season) for season in pd.to_numeric(train_frame["season"]).unique()
    )
    return {
        "model": model,
        "feature_set": feature_set,
        "eval_season": eval_season,
        "C": strength,
        "n_columns": n_columns,
        "train_seasons": train_seasons,
        "train_season_max": max(train_seasons),
        "n_train": int(len(train_frame)),
    }


def _prediction_frame(
    eval_frame: pd.DataFrame,
    target: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """One scored season of finite probabilities."""
    return pd.DataFrame(
        {
            "game_id": eval_frame["game_id"].astype(str).to_numpy(),
            "season": pd.to_numeric(eval_frame["season"]).astype(int).to_numpy(),
            "y_true": target.astype(int).to_numpy(),
            "y_proba": np.asarray(probabilities, dtype=float),
        }
    )


def _store_predictions(
    predictions: dict[str, pd.DataFrame],
    spec: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    """Append one season's predictions under the model key."""
    key: str = f"{spec['model']}|{spec['feature_set']}"
    finite: pd.Series = np.isfinite(frame["y_proba"].to_numpy(dtype=float))
    kept: pd.DataFrame = frame.loc[finite]
    if key in predictions:
        predictions[key] = pd.concat([predictions[key], kept], ignore_index=True)
        return
    predictions[key] = kept.reset_index(drop=True)


def _store_lookup_predictions(
    predictions: dict[str, pd.DataFrame],
    games: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Attach Elo, the margin gap, and the market. None of these is refit."""
    elo = elo_home_probabilities(games)
    elo_probability: np.ndarray = (
        pd.Series(elo["probabilities"]).reindex(games.index).to_numpy(dtype=float)
    )
    elo_frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": games["game_id"].astype(str).to_numpy(),
            "season": pd.to_numeric(games["season"]).astype(int).to_numpy(),
            "y_true": pd.to_numeric(games["home_won"], errors="coerce").to_numpy(),
            "y_proba": elo_probability,
        }
    )
    elo_frame = elo_frame.loc[elo_frame["y_true"].isin([0, 1])].copy()
    elo_frame["y_true"] = elo_frame["y_true"].astype(int)
    _store_predictions(predictions, {"model": "elo", "feature_set": "elo"}, elo_frame)

    gaps: pd.DataFrame = features.loc[:, ["game_id", ELO_MOV_DIFF_COLUMN]].copy()
    gaps["game_id"] = gaps["game_id"].astype(str)
    finite_gaps: pd.Series = np.isfinite(pd.to_numeric(gaps[ELO_MOV_DIFF_COLUMN], errors="coerce"))
    rated: pd.DataFrame = gaps.loc[finite_gaps]
    rated_probability: np.ndarray = rating_win_probability(
        pd.to_numeric(rated[ELO_MOV_DIFF_COLUMN]).to_numpy(dtype=float)
    )
    labels: pd.DataFrame = elo_frame.loc[:, ["game_id", "season", "y_true"]]
    mov: pd.DataFrame = labels.merge(
        pd.DataFrame({"game_id": rated["game_id"].to_numpy(), "y_proba": rated_probability}),
        on="game_id",
        how="inner",
    )
    _store_predictions(predictions, {"model": "elo_mov", "feature_set": "elo_mov"}, mov)

    market = market_home_probabilities(games)
    if market["status"] != "ok":
        raise ValueError("market probabilities are required for the holdout")
    market_probability: np.ndarray = (
        pd.Series(market["probabilities"]).reindex(games.index).to_numpy(dtype=float)
    )
    market_frame: pd.DataFrame = labels.merge(
        pd.DataFrame(
            {
                "game_id": games["game_id"].astype(str).to_numpy(),
                "y_proba": market_probability,
            }
        ),
        on="game_id",
        how="inner",
    )
    _store_predictions(predictions, {"model": "market", "feature_set": "market"}, market_frame)


def _align_season(predictions: dict[str, pd.DataFrame], season: int) -> pd.DataFrame:
    """Keep games in this season where every model has a finite probability."""
    keys: list[str] = [f"{spec['model']}|{spec['feature_set']}" for spec in _MODEL_SPECS]
    missing: list[str] = [key for key in keys if key not in predictions]
    if missing:
        raise ValueError(f"missing predictions for {missing}")
    aligned: pd.DataFrame | None = None
    for key in keys:
        frame: pd.DataFrame = predictions[key]
        part: pd.DataFrame = frame.loc[frame["season"] == season, ["game_id", "y_proba"]].rename(
            columns={"y_proba": key}
        )
        if aligned is None:
            labels: pd.DataFrame = frame.loc[
                frame["season"] == season, ["game_id", "season", "y_true"]
            ]
            aligned = labels.merge(part, on="game_id", how="inner")
            continue
        aligned = aligned.merge(part, on="game_id", how="inner")
    if aligned is None:
        raise ValueError(f"season {season} produced no prediction rows")
    finite: np.ndarray = np.isfinite(aligned[keys].to_numpy(dtype=float)).all(axis=1)
    kept: pd.DataFrame = aligned.loc[finite].sort_values("game_id").reset_index(drop=True)
    if kept["game_id"].duplicated().any():
        raise ValueError(f"season {season} has duplicate game ids")
    return kept


def _metric_rows(
    aligned: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Log loss, Brier, AUC, accuracy, and both paired comparisons."""
    labels: np.ndarray = aligned["y_true"].to_numpy(dtype=int)
    reference: np.ndarray = aligned[REFERENCE_KEY].to_numpy(dtype=float)
    market: np.ndarray = aligned[MARKET_KEY].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for spec in _MODEL_SPECS:
        key: str = f"{spec['model']}|{spec['feature_set']}"
        probabilities: np.ndarray = aligned[key].to_numpy(dtype=float)
        metrics: dict[str, Any] = binary_classification_metrics(labels, probabilities)
        versus_reference: dict[str, Any] = _comparison(
            paired_log_loss_difference(
                labels, probabilities, reference, n_bootstrap, bootstrap_seed
            )
        )
        versus_market: dict[str, Any] = _comparison(
            paired_log_loss_difference(labels, probabilities, market, n_bootstrap, bootstrap_seed)
        )
        rows.append(
            {
                "model": spec["model"],
                "feature_set": spec["feature_set"],
                "role": spec["role"],
                "n": int(metrics["n"]),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "vs_h2_full": versus_reference,
                "vs_market": versus_market,
            }
        )
    return rows


def _comparison(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the published bootstrap fields."""
    return {
        "log_loss_diff": result["log_loss_diff"],
        "log_loss_diff_ci_low": result["log_loss_diff_ci_low"],
        "log_loss_diff_ci_high": result["log_loss_diff_ci_high"],
        "n": result["n"],
        "n_bootstrap": result["n_bootstrap"],
    }


def _calibration_rows(
    season_frames: dict[int, pd.DataFrame],
    pooled: pd.DataFrame,
) -> list[dict[str, Any]]:
    """ECE and equal-count bins for every model and every slice."""
    slices: list[tuple[str, pd.DataFrame]] = [(POOLED_SLICE, pooled)]
    slices.extend((str(season), season_frames[season]) for season in EVAL_SEASONS)
    rows: list[dict[str, Any]] = []
    for slice_name, frame in slices:
        for spec in _MODEL_SPECS:
            key: str = f"{spec['model']}|{spec['feature_set']}"
            summary: dict[str, Any] = reliability_by_decile(
                frame["y_true"].to_numpy(dtype=int),
                frame[key].to_numpy(dtype=float),
                n_bins=N_CALIBRATION_BINS,
                game_ids=frame["game_id"].to_numpy(),
            )
            rows.append(
                {
                    "model": spec["model"],
                    "feature_set": spec["feature_set"],
                    "slice": slice_name,
                    "n": summary["n"],
                    "n_bins": summary["n_bins"],
                    "ece": summary["ece"],
                    "bins": [
                        {
                            "bin": item["bin"],
                            "n": item["n"],
                            "mean_predicted": item["mean_predicted"],
                            "observed_rate": item["observed_rate"],
                        }
                        for item in summary["bins"]
                    ],
                }
            )
    return rows


def _market_rows(
    season_frames: dict[int, pd.DataFrame],
    pooled: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Descriptive market and model coefficients. The market row is the covariate."""
    slices: list[tuple[str, pd.DataFrame]] = [(POOLED_SLICE, pooled)]
    slices.extend((str(season), season_frames[season]) for season in EVAL_SEASONS)
    rows: list[dict[str, Any]] = []
    for slice_name, frame in slices:
        market: np.ndarray = frame[MARKET_KEY].to_numpy(dtype=float)
        labels: np.ndarray = frame["y_true"].to_numpy(dtype=int)
        for spec in _MODEL_SPECS:
            if spec["model"] == "market":
                continue
            key: str = f"{spec['model']}|{spec['feature_set']}"
            summary: dict[str, Any] = market_information_regression(
                labels,
                market,
                frame[key].to_numpy(dtype=float),
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "model": spec["model"],
                    "feature_set": spec["feature_set"],
                    "slice": slice_name,
                    "status": summary["status"],
                    "reason": summary["reason"],
                    "n": summary["n"],
                    "n_bootstrap": summary["n_bootstrap"],
                    "n_bootstrap_used": summary["n_bootstrap_used"],
                    "intercept": summary["intercept"],
                    "intercept_ci_low": summary["intercept_ci_low"],
                    "intercept_ci_high": summary["intercept_ci_high"],
                    "market_coef": summary["market_coef"],
                    "market_ci_low": summary["market_ci_low"],
                    "market_ci_high": summary["market_ci_high"],
                    "model_coef": summary["model_coef"],
                    "model_ci_low": summary["model_ci_low"],
                    "model_ci_high": summary["model_ci_high"],
                }
            )
    return rows


def _oof_summary(season_frames: dict[int, pd.DataFrame]) -> dict[str, Any]:
    """Aligned game rows. Every probability is finite."""
    keys: list[str] = [f"{spec['model']}|{spec['feature_set']}" for spec in _MODEL_SPECS]
    rows: list[dict[str, Any]] = []
    for season in EVAL_SEASONS:
        frame: pd.DataFrame = season_frames[season]
        for record in frame.to_dict(orient="records"):
            rows.append(
                {
                    "game_id": record["game_id"],
                    "season": int(record["season"]),
                    "y_true": int(record["y_true"]),
                    "probabilities": {key: float(record[key]) for key in keys},
                }
            )
    return {
        "n": len(rows),
        "n_2024": int(len(season_frames[2024])),
        "n_2025": int(len(season_frames[2025])),
        "model_keys": keys,
        "rows": rows,
    }


def _metadata() -> dict[str, Any]:
    """Commitments that stay true after the numbers are known."""
    return {
        "adr": "ADR-010",
        "issue": 61,
        "experiment": HOLDOUT_EXPERIMENT_NAME,
        "selected_model": SELECTED_MODEL,
        "selected_feature_set": SELECTED_FEATURE_SET,
        "runs_once": True,
        "publish_regardless": True,
        "no_selection_after_2025": True,
        "registered": False,
        "promoted": False,
        "git_sha": get_git_sha(),
    }


def _parameters(
    *,
    lookback: int,
    max_iter: int,
    seed: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Locked recipe plus the bootstrap settings used for this report."""
    models: list[dict[str, Any]] = []
    for spec in _MODEL_SPECS:
        row: dict[str, Any] = {
            "model": spec["model"],
            "feature_set": spec["feature_set"],
            "role": spec["role"],
            "C": spec["C"],
        }
        if spec["feature_set"] == SELECTED_FEATURE_SET:
            row["n_columns"] = SELECTED_N_COLUMNS
        models.append(row)
    return {
        "lookback": lookback,
        "seed": seed,
        "max_iter": max_iter,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "n_calibration_bins": N_CALIBRATION_BINS,
        "eval_seasons": list(EVAL_SEASONS),
        "cv_log_loss": CV_LOG_LOSS,
        "cv_n": CV_N,
        "models": models,
    }


def _labeled_count(matrix: pd.DataFrame, season: int) -> int:
    """How many labeled rows the feature matrix has in one season."""
    seasons: pd.Series = pd.to_numeric(matrix["season"])
    return int((seasons == season).sum())


def _row(rows: list[dict[str, Any]], feature_set: str) -> dict[str, Any]:
    """Return the metric row for one feature set."""
    return next(row for row in rows if row["feature_set"] == feature_set)


def _markdown_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    """One metric table, including both paired differences."""
    lines: list[str] = [
        f"## {title}",
        "",
        "| model | feature_set | n | log_loss | brier | auc | accuracy | vs_h2_full | vs_market |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {feature_set} | {n} | {log_loss} | {brier} | {auc} | {accuracy} | "
            "{versus_reference} | {versus_market} |".format(
                model=row["model"],
                feature_set=row["feature_set"],
                n=row["n"],
                log_loss=_format_number(row["log_loss"]),
                brier=_format_number(row["brier"]),
                auc=_format_number(row["auc"]),
                accuracy=_format_number(row["accuracy"]),
                versus_reference=_format_interval(row["vs_h2_full"]),
                versus_market=_format_interval(row["vs_market"]),
            )
        )
    lines.append("")
    return lines


def _format_number(value: object) -> str:
    """Six decimals, or a blank when the metric is missing."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def _format_interval(comparison: dict[str, Any]) -> str:
    """Difference and the 95 percent interval."""
    return (
        f"{_format_number(comparison['log_loss_diff'])} "
        f"[{_format_number(comparison['log_loss_diff_ci_low'])}, "
        f"{_format_number(comparison['log_loss_diff_ci_high'])}]"
    )


def _plain(value: Any) -> Any:
    """JSON-safe copies of numpy values. Non-finite floats become null."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.floating):
        return _plain(float(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"cannot store {type(value).__name__} in the holdout report")
