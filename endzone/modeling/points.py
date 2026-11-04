"""Sprint 2 margin and total ridge regressions.

Canonical command: ``python -m endzone.modeling.points``.
Acceptance is ADR-007. A target is registered only when that rule passes.
2024 and 2025 are not CV folds. The win-probability reference is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from endzone.config import configured_tracking_uri
from endzone.features.pregame import (
    build_pregame_features,
    elo_home_probabilities,
    feature_columns_for,
)
from endzone.modeling.dataset import load_data, validate_feature_columns
from endzone.modeling.models import fit_ridge_pipeline
from endzone.modeling.points_eval import (
    HIDDEN_FAILED,
    HIDDEN_NOT_RUN,
    split_estimated_score,
    summarize_regression,
)
from endzone.modeling.points_spec import ScoreExperimentSpec, load_score_spec
from endzone.modeling.tracking import (
    activate_tracking_experiment,
    get_git_sha,
    log_registered_sklearn_model,
    log_sklearn_model,
)
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

_LOCAL_MODEL_VERSION: str = "local"
_BASELINE_LABELS: dict[str, str] = {
    "elo_spread": "Elo/25",
    "league_mean": "League mean",
}
_TARGET_COLUMN: dict[str, str] = {"margin": "margin", "total": "total_points"}
_ID_COLUMNS: tuple[str, ...] = ("game_id", "gameday", "season")


def attach_score_estimates(
    games: pd.DataFrame,
    features: pd.DataFrame,
    slate: pd.DataFrame,
    *,
    tracking_uri: str | None = None,
    register: bool = True,
) -> dict[str, Any]:
    """Score one slate, or hide the estimate when ADR-007 does not pass.

    Games without ``home_score`` and ``away_score`` are not a result. The
    estimate stays null and no MLflow run is opened.

    Returns:
        Dict with ``score_model`` and ``lines`` keyed by ``game_id``.
    """
    game_ids: list[str] = [str(game_id) for game_id in slate["game_id"].tolist()]
    if "home_score" not in games.columns or "away_score" not in games.columns:
        return _not_run(game_ids)
    scored: pd.Series = (
        pd.to_numeric(games["home_score"], errors="coerce").notna()
        & pd.to_numeric(games["away_score"], errors="coerce").notna()
    )
    if int(scored.sum()) == 0:
        return _not_run(game_ids)
    spec: ScoreExperimentSpec = load_score_spec()
    return run_score_experiment(
        games,
        features,
        spec,
        slate=slate,
        tracking_uri=tracking_uri,
        register=register,
        output_dir=None,
    )


def run_score_experiment(
    games: pd.DataFrame,
    features: pd.DataFrame,
    spec: ScoreExperimentSpec,
    *,
    slate: pd.DataFrame | None = None,
    tracking_uri: str | None = None,
    register: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run rolling-origin CV, log it, and predict a slate when both targets pass.

    The serving fit, when a target is accepted, uses every completed game in
    ``games``. The acceptance folds still stop at ``cv_end_season``.

    Returns:
        Dict with ``score_model``, ``lines``, ``versions``, and ``paths``.
    """
    if mlflow.active_run() is not None:
        raise RuntimeError("Sprint 2 tracking must start outside an open MLflow run.")
    columns: tuple[str, ...] = feature_columns_for(spec.feature_set)
    frame: pd.DataFrame = _modeling_frame(games, features, columns)
    finite: pd.DataFrame = _finite_targets(frame)
    if finite.empty:
        game_ids: list[str] = [] if slate is None else [str(row) for row in slate["game_id"]]
        return _not_run(game_ids)
    validate_feature_columns(finite, (spec.feature_set,))
    cv_frame: pd.DataFrame = finite.loc[finite["season"] <= spec.cv_end_season].copy()
    summaries: dict[str, dict[str, Any]] = _cross_validate(cv_frame, columns, spec)
    accepted: bool = all(item["decision"] == "accepted" for item in summaries.values())
    hidden_reason: str | None = None if accepted else HIDDEN_FAILED
    serving: dict[str, Pipeline] = {}
    for name, summary in summaries.items():
        if summary["decision"] != "accepted":
            continue
        serving[name] = fit_ridge_pipeline(
            _design(finite, columns),
            finite[_TARGET_COLUMN[name]],
            alpha=spec.ridge_alpha,
        )
    versions: dict[str, str | None] = {"margin": None, "total": None}
    uri: str = tracking_uri or configured_tracking_uri()
    activate_tracking_experiment(spec.experiment_name, uri)
    with mlflow.start_run(run_name="sprint2-score"):
        _log_parent(spec, summaries)
        for name, summary in summaries.items():
            with mlflow.start_run(run_name=f"s2-{name}", nested=True):
                _log_target(name, summary, spec)
                pipeline: Pipeline | None = serving.get(name)
                if pipeline is not None and register:
                    versions[name] = _register_ridge(
                        pipeline,
                        _design(finite, columns),
                        spec.target(name).registered_model_name,
                        uri,
                    )
        if output_dir is not None:
            paths: dict[str, str] = write_score_report(
                _report_document(spec, summaries, accepted),
                Path(output_dir),
            )
            mlflow.log_artifact(paths["json"])
            mlflow.log_artifact(paths["markdown"])
        else:
            paths = {}
    lines: dict[str, dict[str, float | None]] = _lines_for_slate(
        slate,
        features,
        columns,
        serving if accepted else {},
    )
    score_model: dict[str, Any] = _score_model_document(
        spec,
        summaries,
        accepted=accepted,
        hidden_reason=hidden_reason,
        versions=versions,
    )
    for name in ("margin", "total"):
        row: dict[str, Any] = score_model[name]
        logger.info(
            "%s decision=%s mae=%s baseline_mae=%s market_mae=%s seasons=%s/%s",
            name,
            row["decision"],
            row["mae"],
            row["baseline_mae"],
            row["market_mae"],
            row["seasons_improved"],
            row["seasons_scored"],
        )
    return {
        "score_model": score_model,
        "lines": lines,
        "versions": versions,
        "paths": paths,
    }


def write_score_report(document: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write the CV JSON and a short markdown table.

    Returns:
        Paths for ``json`` and ``markdown``.
    """
    folder: Path = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    json_path: Path = folder / "cv_report.json"
    markdown_path: Path = folder / "cv_report.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(document), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def build_parser() -> argparse.ArgumentParser:
    """Build the Sprint 2 parser."""
    parser = argparse.ArgumentParser(
        description="Cross-validate ridge margin and total models. ADR-007."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Report directory. The default is the directory in the hypothesis file.",
    )
    parser.add_argument("--fixture-path", type=str, default=None)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Log the run and skip the model registry. Acceptance is unchanged.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Fit the pre-registered comparison and write the CV report."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    spec: ScoreExperimentSpec = load_score_spec()
    games, team_games = load_data(args.fixture_path if args.fixture_path else None)
    features: pd.DataFrame = build_pregame_features(
        games,
        team_games,
        lookback_games=args.lookback,
    )
    validate_feature_columns(features, (spec.feature_set,))
    output_dir: Path = Path(args.output_dir) if args.output_dir else Path(spec.output_dir)
    result: dict[str, Any] = run_score_experiment(
        games,
        features,
        spec,
        slate=None,
        tracking_uri=args.tracking_uri,
        register=not args.no_register,
        output_dir=output_dir,
    )
    logger.info("Wrote %s", result["paths"].get("json", output_dir))
    return result


def _not_run(game_ids: list[str]) -> dict[str, Any]:
    """Page payload used when scores are not available to test the rule."""
    spec: ScoreExperimentSpec = load_score_spec()
    blank: dict[str, dict[str, Any]] = {}
    for target in spec.targets:
        blank[target.name] = _blank_target(_BASELINE_LABELS[target.baseline], "not_run")
    return {
        "score_model": {
            "feature_set": spec.feature_set,
            "accepted": False,
            "hidden_reason": HIDDEN_NOT_RUN,
            "margin": blank["margin"],
            "total": blank["total"],
        },
        "lines": {game_id: _null_line() for game_id in game_ids},
        "versions": {"margin": None, "total": None},
        "paths": {},
    }


def _blank_target(baseline_name: str, decision: str) -> dict[str, Any]:
    """One target with no metrics."""
    return {
        "decision": decision,
        "baseline_name": baseline_name,
        "mae": None,
        "rmse": None,
        "baseline_mae": None,
        "market_mae": None,
        "mae_diff": None,
        "mae_diff_ci_low": None,
        "mae_diff_ci_high": None,
        "coverage_80": None,
        "implied_win_log_loss": None,
        "winprob_log_loss_benchmark": None,
        "seasons_improved": None,
        "seasons_scored": None,
        "n": None,
        "model_version": None,
    }


def _null_line() -> dict[str, None]:
    """Point fields for a game whose estimate is hidden."""
    return {
        "estimated_margin": None,
        "estimated_total": None,
        "estimated_home_points": None,
        "estimated_away_points": None,
    }


def _modeling_frame(
    games: pd.DataFrame,
    features: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Join scores onto the pregame matrix. One row per game."""
    required: tuple[str, ...] = (
        "game_id",
        "season",
        "gameday",
        "home_team_franchise",
        "away_team_franchise",
        "home_won",
        "home_score",
        "away_score",
    )
    missing: list[str] = [column for column in required if column not in games.columns]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Games are missing columns for margin and total: {missing_list}.")
    absent_features: list[str] = [column for column in columns if column not in features.columns]
    if absent_features:
        missing_list = ", ".join(absent_features)
        raise ValueError(f"Feature matrix is missing columns: {missing_list}.")
    keep: list[str] = list(required)
    for column in ("spread_line", "total_line"):
        if column in games.columns:
            keep.append(column)
    base: pd.DataFrame = games.loc[:, keep].copy()
    base["game_id"] = base["game_id"].astype(str)
    matrix: pd.DataFrame = features.loc[:, ["game_id", *columns]].copy()
    matrix["game_id"] = matrix["game_id"].astype(str)
    merged: pd.DataFrame = base.merge(matrix, on="game_id", how="inner")
    if merged.empty:
        raise ValueError("No games matched the feature matrix.")
    if merged["game_id"].duplicated().any():
        raise ValueError("game_id must be unique after joining features.")
    home_score: pd.Series = pd.to_numeric(merged["home_score"], errors="coerce")
    away_score: pd.Series = pd.to_numeric(merged["away_score"], errors="coerce")
    merged["margin"] = home_score - away_score
    merged["total_points"] = home_score + away_score
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce")
    return merged


def _finite_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a finite margin, total, and season."""
    margin: np.ndarray = frame["margin"].to_numpy(dtype=float)
    total: np.ndarray = frame["total_points"].to_numpy(dtype=float)
    season: np.ndarray = frame["season"].to_numpy(dtype=float)
    usable: np.ndarray = np.isfinite(margin) & np.isfinite(total) & np.isfinite(season)
    return frame.loc[usable].copy()


def _design(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Feature matrix plus the identifier columns the ridge pipeline drops."""
    design: pd.DataFrame = frame.loc[:, [*_ID_COLUMNS, *columns]].copy()
    design.loc[:, list(columns)] = design.loc[:, list(columns)].astype(float)
    return design


def _residual_sigma(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Training residual standard deviation. Non-positive values are missing."""
    if len(actual) < 2:
        return None
    sigma: float = float(np.std(actual - predicted, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        return None
    return sigma


def _cross_validate(
    cv_frame: pd.DataFrame,
    columns: tuple[str, ...],
    spec: ScoreExperimentSpec,
) -> dict[str, dict[str, Any]]:
    """Score each season from strictly earlier seasons."""
    elo_input: pd.DataFrame = cv_frame.loc[
        :,
        [
            "game_id",
            "season",
            "gameday",
            "home_team_franchise",
            "away_team_franchise",
            "home_won",
            "home_score",
            "away_score",
        ],
    ].copy()
    fitted: dict[str, Any] = elo_home_probabilities(
        elo_input,
        points_per_spread=spec.elo_points_per_spread,
    )
    elo_by_id: pd.Series = pd.Series(
        np.asarray(fitted["point_spreads"], dtype=float),
        index=elo_input["game_id"].astype(str),
    )
    collected: dict[str, list[dict[str, Any]]] = {"margin": [], "total": []}
    for season in range(spec.cv_start_season, spec.cv_end_season + 1):
        train: pd.DataFrame = cv_frame.loc[cv_frame["season"] < season]
        eval_rows: pd.DataFrame = cv_frame.loc[cv_frame["season"] == season]
        if train.empty or eval_rows.empty:
            continue
        design_train: pd.DataFrame = _design(train, columns)
        design_eval: pd.DataFrame = _design(eval_rows, columns)
        margin_pipe: Pipeline = fit_ridge_pipeline(
            design_train,
            train["margin"],
            alpha=spec.ridge_alpha,
        )
        total_pipe: Pipeline = fit_ridge_pipeline(
            design_train,
            train["total_points"],
            alpha=spec.ridge_alpha,
        )
        margin_hat: np.ndarray = np.asarray(margin_pipe.predict(design_eval), dtype=float)
        total_hat: np.ndarray = np.asarray(total_pipe.predict(design_eval), dtype=float)
        margin_sigma: float | None = _residual_sigma(
            train["margin"].to_numpy(dtype=float),
            np.asarray(margin_pipe.predict(design_train), dtype=float),
        )
        total_sigma: float | None = _residual_sigma(
            train["total_points"].to_numpy(dtype=float),
            np.asarray(total_pipe.predict(design_train), dtype=float),
        )
        total_baseline: float = float(train["total_points"].mean())
        eval_ids: list[str] = [str(game_id) for game_id in eval_rows["game_id"].tolist()]
        margin_market: np.ndarray = _market_numbers(
            eval_rows,
            "spread_line",
            spec.market_margin_multiplier,
        )
        total_market: np.ndarray = _market_numbers(eval_rows, "total_line", 1.0)
        home_won: np.ndarray = pd.to_numeric(eval_rows["home_won"], errors="coerce").to_numpy(
            dtype=float
        )
        actual_margin: np.ndarray = eval_rows["margin"].to_numpy(dtype=float)
        actual_total: np.ndarray = eval_rows["total_points"].to_numpy(dtype=float)
        elo_hat: np.ndarray = elo_by_id.loc[eval_ids].to_numpy(dtype=float)
        for index, game_id in enumerate(eval_ids):
            collected["margin"].append(
                {
                    "game_id": game_id,
                    "season": int(season),
                    "y_true": float(actual_margin[index]),
                    "y_model": float(margin_hat[index]),
                    "y_baseline": float(elo_hat[index]),
                    "y_market": float(margin_market[index]),
                    "sigma": margin_sigma,
                    "home_won": float(home_won[index]),
                }
            )
            collected["total"].append(
                {
                    "game_id": game_id,
                    "season": int(season),
                    "y_true": float(actual_total[index]),
                    "y_model": float(total_hat[index]),
                    "y_baseline": total_baseline,
                    "y_market": float(total_market[index]),
                    "sigma": total_sigma,
                    "home_won": float(home_won[index]),
                }
            )
    summaries: dict[str, dict[str, Any]] = {}
    for name in ("margin", "total"):
        table: pd.DataFrame = pd.DataFrame(collected[name])
        summaries[name] = _summarize_collected(table, spec, include_implied_win=name == "margin")
    return summaries


def _market_numbers(frame: pd.DataFrame, column: str, multiplier: float) -> np.ndarray:
    """Closing line times the pre-registered sign. Missing lines stay NaN."""
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    values: np.ndarray = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values * float(multiplier)


def _summarize_collected(
    table: pd.DataFrame,
    spec: ScoreExperimentSpec,
    *,
    include_implied_win: bool,
) -> dict[str, Any]:
    """Turn one target's out-of-fold rows into the acceptance summary."""
    if table.empty:
        empty_true = np.array([], dtype=float)
        return summarize_regression(
            y_true=empty_true,
            y_model=empty_true,
            y_baseline=empty_true,
            y_market=empty_true,
            sigma=empty_true,
            seasons=np.array([], dtype=int),
            y_home_won=empty_true if include_implied_win else None,
            n_bootstrap=spec.n_bootstrap,
            seed=spec.bootstrap_seed,
            min_seasons_improved=spec.min_seasons_improved,
            season_count=spec.season_count,
            interval_coverage=spec.interval_coverage,
            include_implied_win=include_implied_win,
        )
    sigma: np.ndarray = table["sigma"].to_numpy(dtype=float)
    return summarize_regression(
        y_true=table["y_true"].to_numpy(dtype=float),
        y_model=table["y_model"].to_numpy(dtype=float),
        y_baseline=table["y_baseline"].to_numpy(dtype=float),
        y_market=table["y_market"].to_numpy(dtype=float),
        sigma=sigma,
        seasons=table["season"].to_numpy(),
        y_home_won=table["home_won"].to_numpy(dtype=float) if include_implied_win else None,
        n_bootstrap=spec.n_bootstrap,
        seed=spec.bootstrap_seed,
        min_seasons_improved=spec.min_seasons_improved,
        season_count=spec.season_count,
        interval_coverage=spec.interval_coverage,
        include_implied_win=include_implied_win,
    )


def _lines_for_slate(
    slate: pd.DataFrame | None,
    features: pd.DataFrame,
    columns: tuple[str, ...],
    serving: dict[str, Pipeline],
) -> dict[str, dict[str, float | None]]:
    """Predicted points per slate game. An unaccepted pair stays null."""
    if slate is None:
        return {}
    game_ids: list[str] = [str(game_id) for game_id in slate["game_id"].tolist()]
    if "margin" not in serving or "total" not in serving:
        return {game_id: _null_line() for game_id in game_ids}
    matrix: pd.DataFrame = features.copy()
    matrix["game_id"] = matrix["game_id"].astype(str)
    ordered: pd.DataFrame = pd.DataFrame({"game_id": game_ids}).merge(
        matrix,
        on="game_id",
        how="left",
    )
    if ordered["game_id"].duplicated().any() or len(ordered) != len(game_ids):
        raise ValueError("Slate features must contain one row per game.")
    missing_feature_rows: pd.Series = ordered.loc[:, list(columns)].isna().all(axis=1)
    if bool(missing_feature_rows.any()):
        raise ValueError("Slate features are missing for at least one game.")
    # The design helper needs gameday and season. They are not model inputs.
    if "gameday" not in ordered.columns:
        ordered["gameday"] = pd.NaT
    if "season" not in ordered.columns:
        ordered["season"] = 0
    design: pd.DataFrame = _design(ordered, columns)
    margin_hat: np.ndarray = np.asarray(serving["margin"].predict(design), dtype=float)
    total_hat: np.ndarray = np.asarray(serving["total"].predict(design), dtype=float)
    lines: dict[str, dict[str, float | None]] = {}
    for index, game_id in enumerate(game_ids):
        margin_value: float = float(margin_hat[index])
        total_value: float = float(total_hat[index])
        if not math.isfinite(margin_value) or not math.isfinite(total_value):
            lines[game_id] = _null_line()
            continue
        parts: dict[str, float] = split_estimated_score(margin_value, total_value)
        lines[game_id] = {
            "estimated_margin": margin_value,
            "estimated_total": total_value,
            "estimated_home_points": parts["home"],
            "estimated_away_points": parts["away"],
        }
    return lines


def _score_model_document(
    spec: ScoreExperimentSpec,
    summaries: dict[str, dict[str, Any]],
    *,
    accepted: bool,
    hidden_reason: str | None,
    versions: dict[str, str | None],
) -> dict[str, Any]:
    """Accuracy block stored on the weekly document and in the CV report."""
    document: dict[str, Any] = {
        "feature_set": spec.feature_set,
        "accepted": accepted,
        "hidden_reason": hidden_reason,
    }
    for target in spec.targets:
        summary: dict[str, Any] = summaries[target.name]
        benchmark: float | None = (
            spec.winprob_log_loss_benchmark if target.name == "margin" else None
        )
        document[target.name] = {
            "decision": summary["decision"],
            "baseline_name": _BASELINE_LABELS[target.baseline],
            "mae": _round_metric(summary["mae"]),
            "rmse": _round_metric(summary["rmse"]),
            "baseline_mae": _round_metric(summary["baseline_mae"]),
            "market_mae": _round_metric(summary["market_mae"]),
            "mae_diff": _round_metric(summary["mae_diff"]),
            "mae_diff_ci_low": _round_metric(summary["mae_diff_ci_low"]),
            "mae_diff_ci_high": _round_metric(summary["mae_diff_ci_high"]),
            "coverage_80": _round_metric(summary["coverage_80"]),
            "implied_win_log_loss": _round_metric(summary["implied_win_log_loss"]),
            "winprob_log_loss_benchmark": benchmark,
            "seasons_improved": summary["seasons_improved"],
            "seasons_scored": summary["seasons_scored"],
            "n": summary["n"],
            "model_version": versions.get(target.name),
        }
    return document


def _report_document(
    spec: ScoreExperimentSpec,
    summaries: dict[str, dict[str, Any]],
    accepted: bool,
) -> dict[str, Any]:
    """CV report. Season rows keep the per-year MAE that the rule counts."""
    body: dict[str, Any] = _score_model_document(
        spec,
        summaries,
        accepted=accepted,
        hidden_reason=None if accepted else HIDDEN_FAILED,
        versions={"margin": None, "total": None},
    )
    body["rule"] = {
        "min_seasons_improved": spec.min_seasons_improved,
        "season_count": spec.season_count,
        "cv_start_season": spec.cv_start_season,
        "cv_end_season": spec.cv_end_season,
        "ridge_alpha": spec.ridge_alpha,
        "n_bootstrap": spec.n_bootstrap,
        "bootstrap_seed": spec.bootstrap_seed,
        "elo_points_per_spread": spec.elo_points_per_spread,
        "market_margin_multiplier": spec.market_margin_multiplier,
        "interval_coverage": spec.interval_coverage,
        "market_is_a_gate": False,
    }
    for name, summary in summaries.items():
        seasons: list[dict[str, Any]] = []
        for row in summary["seasons"]:
            seasons.append(
                {
                    "season": row["season"],
                    "n": row["n"],
                    "mae": _round_metric(row["mae"]),
                    "baseline_mae": _round_metric(row["baseline_mae"]),
                    "improved": bool(row["improved"]),
                }
            )
        body[name]["seasons"] = seasons
    return body


def _round_metric(value: object) -> float | None:
    """Six decimal places for a stored metric. Missing stays null."""
    if value is None:
        return None
    try:
        number: float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _log_parent(spec: ScoreExperimentSpec, summaries: dict[str, dict[str, Any]]) -> None:
    """Params and tags on the Sprint 2 parent run."""
    mlflow.log_param("feature_set", spec.feature_set)
    mlflow.log_param("model", spec.model)
    mlflow.log_param("ridge_alpha", spec.ridge_alpha)
    mlflow.log_param("cv_start_season", spec.cv_start_season)
    mlflow.log_param("cv_end_season", spec.cv_end_season)
    mlflow.log_param("n_bootstrap", spec.n_bootstrap)
    mlflow.log_param("bootstrap_seed", spec.bootstrap_seed)
    mlflow.log_param("elo_points_per_spread", spec.elo_points_per_spread)
    mlflow.set_tag("phase", spec.phase_tag)
    mlflow.set_tag("hypothesis", spec.hypothesis_tag)
    mlflow.set_tag("git_commit", get_git_sha())
    for name, summary in summaries.items():
        _log_metrics(name, summary)


def _log_target(name: str, summary: dict[str, Any], spec: ScoreExperimentSpec) -> None:
    """One nested run per target. ``decision`` is the acceptance label."""
    mlflow.set_tag("decision", str(summary["decision"]))
    mlflow.set_tag("target", name)
    mlflow.set_tag("hypothesis", spec.hypothesis_tag)
    mlflow.set_tag("git_commit", get_git_sha())
    mlflow.log_param("baseline", spec.target(name).baseline)
    mlflow.log_param("ridge_alpha", spec.ridge_alpha)
    _log_metrics(name, summary)


def _log_metrics(name: str, summary: dict[str, Any]) -> None:
    """Log finite metrics. Missing market or coverage values are skipped."""
    fields: dict[str, object] = {
        "mae": summary["mae"],
        "rmse": summary["rmse"],
        "baseline_mae": summary["baseline_mae"],
        "mae_diff": summary["mae_diff"],
        "mae_diff_ci_low": summary["mae_diff_ci_low"],
        "mae_diff_ci_high": summary["mae_diff_ci_high"],
        "market_mae": summary["market_mae"],
        "coverage_80": summary["coverage_80"],
        "implied_win_log_loss": summary["implied_win_log_loss"],
        "seasons_improved": summary["seasons_improved"],
        "n": summary["n"],
    }
    for key, value in fields.items():
        number: float | None = _round_metric(value)
        if number is None and not isinstance(value, int | float):
            continue
        if number is None:
            continue
        mlflow.log_metric(f"{name}_{key}", number)


def _local_registry(tracking_uri: str) -> bool:
    """True for a sqlite or file store. A Databricks URI still has to register."""
    text: str = tracking_uri.strip()
    return text.startswith("sqlite:") or text.startswith("file:")


def _register_ridge(
    pipeline: Pipeline,
    design: pd.DataFrame,
    model_name: str,
    tracking_uri: str,
) -> str:
    """Register one accepted pipeline. A local registry failure uses ``local``."""
    example: pd.DataFrame = design.head(1)
    signature = infer_signature(example, pipeline.predict(example))
    try:
        logged = log_registered_sklearn_model(
            pipeline,
            signature=signature,
            input_example=example,
            registered_model_name=model_name,
        )
    except (RuntimeError, mlflow.MlflowException) as exc:
        if not _local_registry(tracking_uri):
            raise
        logger.warning(
            "Local MLflow did not register %s (%s). The version is %s.",
            model_name,
            exc,
            _LOCAL_MODEL_VERSION,
        )
        try:
            log_sklearn_model(pipeline)
        except mlflow.MlflowException as log_exc:
            logger.warning("Local MLflow did not store the model artifact: %s", log_exc)
        return _LOCAL_MODEL_VERSION
    return logged.model_version


def _markdown_report(document: dict[str, Any]) -> str:
    """A short table. The JSON file is the record of the numbers."""
    rule: dict[str, Any] = document["rule"]
    lines: list[str] = [
        f"# {document['feature_set']} margin and total",
        "",
        "Pre-registered rule: ADR-007. "
        f"Ridge alpha {rule['ridge_alpha']:g} on {document['feature_set']}. "
        f"Rolling-origin CV {rule['cv_start_season']}-{rule['cv_end_season']}. "
        "A target is accepted only when the 95% paired bootstrap interval of "
        "model MAE minus baseline MAE lies entirely below 0 and the model is "
        f"strictly better in at least {rule['min_seasons_improved']} of "
        f"{rule['season_count']} seasons. Market MAE, RMSE, 80% coverage, and "
        "implied win log loss are reported and are not gates.",
        "",
        "| Target | Decision | MAE | Baseline | Diff | CI low | CI high | Seasons | "
        "Market MAE | RMSE | 80% coverage | Implied log loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("margin", "total"):
        row: dict[str, Any] = document[name]
        lines.append(
            "| {name} | {decision} | {mae} | {baseline} | {diff} | {low} | {high} | "
            "{seasons} | {market} | {rmse} | {coverage} | {implied} |".format(
                name=name,
                decision=row["decision"],
                mae=_cell(row["mae"]),
                baseline=_cell(row["baseline_mae"]),
                diff=_cell(row["mae_diff"]),
                low=_cell(row["mae_diff_ci_low"]),
                high=_cell(row["mae_diff_ci_high"]),
                seasons=f"{row['seasons_improved']}/{row['seasons_scored']}",
                market=_cell(row["market_mae"]),
                rmse=_cell(row["rmse"]),
                coverage=_cell(row["coverage_80"]),
                implied=_cell(row["implied_win_log_loss"]),
            )
        )
    lines.append("")
    for name in ("margin", "total"):
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| Season | n | MAE | Baseline MAE | Improved |")
        lines.append("| --- | --- | --- | --- | --- |")
        for season in document[name].get("seasons", []):
            lines.append(
                f"| {season['season']} | {season['n']} | {_cell(season['mae'])} | "
                f"{_cell(season['baseline_mae'])} | {season['improved']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _cell(value: object) -> str:
    """Render a metric cell. Missing is a blank."""
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
