"""Rolling-origin comparison: feature sets, baselines, the report, and the reference fit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from endzone.config import configured_tracking_uri
from endzone.features.pregame import (
    FEATURE_SET_CURRENT,
    FEATURE_SET_H2_FULL,
    PKG_CORE_INJ_FEATURE_SETS,
    build_pregame_features,
    feature_columns_for,
    prepare_training_data,
)
from endzone.modeling.cv import RollingOriginConfig, make_rolling_origin_config
from endzone.modeling.dataset import (
    DEFAULT_MAX_NULL_FRACTION,
    cv_dataset_frame,
    load_data,
    load_injury_inputs,
    load_qb_inputs,
    validate_feature_columns,
)
from endzone.modeling.ensembles import (
    DERIVED_CV_MODELS,
    MODEL_ENS_MEAN,
    MODEL_ENS_MEAN_ELO,
    MODEL_STACK_ELO,
    elo_mean_result,
    ensemble_mean_result,
    stack_elo_result,
)
from endzone.modeling.evaluation import (
    MIN_LOG_LOSS_IMPROVEMENT,
    acceptance_decisions,
    attach_reference_comparison,
    calibration_table_for_results,
    label_market_benchmark,
    market_information_table_for_results,
    qb_change_subset_comparison,
)
from endzone.modeling.hypotheses import (
    MARKET_BENCHMARK_PROFILES,
    CvModelSpec,
    ExperimentSpec,
    experiment_for,
    phase_tracking_settings,
    resolve_reference_spec,
)
from endzone.modeling.models import MODEL_LOGISTIC
from endzone.modeling.reference import compare_reference_metrics, load_reference_metrics
from endzone.modeling.report import comparison_table, render_phase1_report, write_phase1_outputs
from endzone.modeling.scoring import (
    CvScoreSettings,
    _requested_feature_sets,
    _score_baseline,
    _score_elo,
    _score_market,
    _score_model_spec,
    _shared_params,
)
from endzone.modeling.tracking import log_phase1_to_mlflow
from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_phase1_baselines(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    config: RollingOriginConfig | None = None,
    lookback_games: int = 10,
    logistic_c: float = 1.0,
    logistic_max_iter: int = 1000,
    random_state: int = 42,
    logistic_feature_sets: tuple[str, ...] | None = None,
    reference_model: str | None = None,
    reference_feature_set: str | None = None,
    phase: int = 1,
    hypothesis: str | None = None,
    max_null_fraction: float = DEFAULT_MAX_NULL_FRACTION,
    expected_starters: pd.DataFrame | None = None,
    qb_dropbacks: pd.DataFrame | None = None,
    depth_chart_starters: pd.DataFrame | None = None,
    injury_reports: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score logistic feature sets, the home prior, Elo, and the market.

    The market is skipped when the moneyline columns are absent or every
    line is missing. 2024 and 2025 are not scored. The home prior, Elo, and
    the market are always scored. ``logistic_feature_sets`` replaces the
    experiment file's model rows with fixed-C logistic on those sets.
    The phase 1 call with no extra arguments is unchanged.

    Args:
        games: Completed games, one row per game.
        team_games: Team-game EPA used to build the logistic matrices.
        config: Rolling-origin window. Defaults to 2016-2023.
        lookback_games: Rolling window passed to ``build_pregame_features``.
        logistic_c: Inverse regularization strength.
        logistic_max_iter: Logistic solver iteration cap.
        random_state: Seed for the logistic fit and the stored config.
        logistic_feature_sets: Logistic sets to score. None uses the experiment file.
        reference_model: Reference model name. None uses the experiment file.
        reference_feature_set: Reference feature set. None uses the experiment file.
        phase: ``1`` keeps the phase 1 report. ``2`` is a phase 2 hypothesis.
        hypothesis: Short or full hypothesis name. None on phase 2 keeps the default file.
        max_null_fraction: Fail a requested column at or above this null share.
        expected_starters: Depth-chart QB1 table. None leaves QB columns missing.
        qb_dropbacks: Passer dropbacks. None leaves QB columns missing.
        depth_chart_starters: Depth 1 chart rows for the absence counts.
            None leaves those columns missing when reports are also None.
        injury_reports: Injury rows, including timestamps. None leaves the
            absence columns missing when starters are also None.

    Returns:
        Dict with ``results``, ``comparison_table``, ``markdown``, ``config``,
        ``reference_model``, and ``reference_feature_set``.

    Raises:
        ValueError: The phase, a feature set, or the reference cannot be used.
    """
    resolved: RollingOriginConfig = make_rolling_origin_config(
        **(config._asdict() if config is not None else {})
    )
    reference_spec: dict[str, str] = resolve_reference_spec(
        phase,
        reference_model=reference_model,
        reference_feature_set=reference_feature_set,
        hypothesis=hypothesis,
    )
    experiment: ExperimentSpec = experiment_for(phase, hypothesis)
    if logistic_feature_sets is None:
        model_specs: tuple[CvModelSpec, ...] = experiment.model_specs
    else:
        model_specs = tuple(
            CvModelSpec(MODEL_LOGISTIC, feature_set_name)
            for feature_set_name in logistic_feature_sets
        )
    for spec in model_specs:
        feature_columns_for(spec.feature_set)
    tracking: dict[str, Any] = phase_tracking_settings(phase, hypothesis=hypothesis)
    features: pd.DataFrame = build_pregame_features(
        games,
        team_games,
        lookback_games=lookback_games,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
        depth_chart_starters=depth_chart_starters,
        injury_reports=injury_reports,
    )
    validate_feature_columns(
        features,
        _requested_feature_sets(model_specs, reference_spec["feature_set"]),
        max_null_fraction=max_null_fraction,
    )
    settings = CvScoreSettings(
        resolved=resolved,
        lookback_games=lookback_games,
        logistic_c=logistic_c,
        logistic_max_iter=logistic_max_iter,
        random_state=random_state,
        reference_model=reference_spec["model"],
        reference_feature_set=reference_spec["feature_set"],
        log_shrinkage=experiment.log_shrinkage,
        log_qb=experiment.log_qb,
    )
    shared: dict[str, Any] = _shared_params(settings)
    current_matrix: pd.DataFrame
    current_target: pd.Series
    current_matrix, current_target = prepare_training_data(
        games,
        features,
        feature_set=FEATURE_SET_CURRENT,
    )
    results: list[dict[str, Any]] = [
        _score_baseline(current_matrix, current_target, settings, shared)
    ]
    derived_specs: tuple[CvModelSpec, ...] = tuple(
        spec for spec in model_specs if spec.model in DERIVED_CV_MODELS
    )
    derived_counts: dict[str, int] = {}
    for spec in derived_specs:
        derived_counts[spec.model] = derived_counts.get(spec.model, 0) + 1
    repeated: list[str] = [name for name, count in derived_counts.items() if count > 1]
    if repeated:
        joined: str = ", ".join(repeated)
        raise ValueError(f"derived model scored more than once: {joined}")
    for spec in model_specs:
        if spec.model in DERIVED_CV_MODELS:
            continue
        results.append(_score_model_spec(games, features, spec, settings, shared))
    needs_elo: bool = any(
        spec.model in {MODEL_ENS_MEAN_ELO, MODEL_STACK_ELO} for spec in derived_specs
    )
    if needs_elo:
        results.append(_score_elo(games, current_matrix, current_target, settings, shared))
    for spec in derived_specs:
        if spec.model == MODEL_ENS_MEAN:
            results.append(ensemble_mean_result(results, spec, shared))
        elif spec.model == MODEL_ENS_MEAN_ELO:
            results.append(elo_mean_result(results, spec, shared))
        elif spec.model == MODEL_STACK_ELO:
            results.append(
                stack_elo_result(
                    games,
                    features,
                    results,
                    spec,
                    shared,
                    cv_start_season=settings.resolved.cv_start_season,
                    cv_end_season=settings.resolved.cv_end_season,
                    logistic_max_iter=settings.logistic_max_iter,
                    random_state=settings.random_state,
                )
            )
        else:
            raise ValueError(f"Unknown derived model {spec.model!r}.")
    if not needs_elo:
        results.append(_score_elo(games, current_matrix, current_target, settings, shared))
    results.append(_score_market(games, settings, shared))
    attach_reference_comparison(
        results,
        reference_spec["model"],
        reference_spec["feature_set"],
        n_bootstrap=resolved.n_bootstrap,
        seed=resolved.bootstrap_seed,
    )
    table: pd.DataFrame = comparison_table(results)
    apply_acceptance: bool = phase == 2
    qb_subset: dict[str, Any] | None = None
    if experiment.qb_subset and experiment.qb_feature_set is not None:
        qb_subset = qb_change_subset_comparison(
            features,
            results,
            reference_spec["model"],
            reference_spec["feature_set"],
            experiment.qb_feature_set,
            n_bootstrap=resolved.n_bootstrap,
            seed=resolved.bootstrap_seed,
        )
    report: dict[str, Any] = {
        "results": results,
        "comparison_table": table,
        "config": resolved._asdict(),
        "lookback_games": lookback_games,
        "reference_model": reference_spec["model"],
        "reference_feature_set": reference_spec["feature_set"],
        "phase": phase,
        "hypothesis": tracking["hypothesis"],
        "title": tracking["title"],
        "apply_acceptance": apply_acceptance,
        "min_log_loss_improvement": MIN_LOG_LOSS_IMPROVEMENT,
        "markdown": "",
        "dataset_frame": cv_dataset_frame(games, features, resolved.cv_end_season),
        "dataset_source": (
            "fct_games joined to pregame features, "
            f"seasons through {resolved.cv_end_season}, lookback {lookback_games}"
        ),
        "qb_change_subset": qb_subset,
    }
    if apply_acceptance:
        report["acceptance"] = acceptance_decisions(
            results,
            reference_spec["model"],
            reference_spec["feature_set"],
        )
        if experiment.report_profile in MARKET_BENCHMARK_PROFILES:
            report["acceptance"] = label_market_benchmark(report["acceptance"])
    if experiment.report_profile in MARKET_BENCHMARK_PROFILES:
        report["calibration"] = calibration_table_for_results(results)
        report["market_information"] = market_information_table_for_results(
            results,
            n_bootstrap=resolved.n_bootstrap,
            seed=resolved.bootstrap_seed,
        )
    stacker_frames: list[pd.DataFrame] = [
        result["stacker_coefficients"]
        for result in results
        if isinstance(result.get("stacker_coefficients"), pd.DataFrame)
    ]
    if stacker_frames:
        report["stacker_coefficients"] = pd.concat(stacker_frames, ignore_index=True)
    report["markdown"] = render_phase1_report(report)
    return report


def run_cv_cli(args: argparse.Namespace) -> None:
    """Fit a rolling-origin CV phase, write the report, and log MLflow."""
    config: RollingOriginConfig = make_rolling_origin_config(
        cv_start_season=args.cv_start_season,
        cv_end_season=args.cv_end_season,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    settings: dict[str, Any] = phase_tracking_settings(args.phase, hypothesis=args.hypothesis)
    experiment: ExperimentSpec = experiment_for(args.phase, args.hypothesis)
    output_dir: str = args.output_dir
    if args.phase != 1 and args.output_dir == "reports/phase1":
        output_dir = str(settings["output_dir"])
    logger.info(
        "Running phase %s rolling-origin CV for seasons %s through %s",
        args.phase,
        config.cv_start_season,
        config.cv_end_season,
    )
    fixture_path: str | None = args.fixture_path if args.fixture_path else None
    games, team_games = load_data(fixture_path=fixture_path)
    expected_starters: pd.DataFrame | None = None
    qb_dropbacks: pd.DataFrame | None = None
    if experiment.log_qb or experiment.qb_subset:
        expected_starters, qb_dropbacks = load_qb_inputs(fixture_path)
    depth_chart_starters: pd.DataFrame | None = None
    injury_reports: pd.DataFrame | None = None
    scored_sets: set[str] = {spec.feature_set for spec in experiment.model_specs}
    if scored_sets.intersection(PKG_CORE_INJ_FEATURE_SETS):
        depth_chart_starters, injury_reports = load_injury_inputs(fixture_path)
    report: dict[str, Any] = run_phase1_baselines(
        games,
        team_games,
        config=config,
        lookback_games=args.lookback,
        logistic_c=args.C,
        logistic_max_iter=args.max_iter,
        random_state=args.seed,
        reference_model=args.reference_model,
        reference_feature_set=args.reference_feature_set,
        phase=args.phase,
        hypothesis=args.hypothesis,
        max_null_fraction=args.max_null_fraction,
        expected_starters=expected_starters,
        qb_dropbacks=qb_dropbacks,
        depth_chart_starters=depth_chart_starters,
        injury_reports=injury_reports,
    )
    paths: dict[str, Any] = write_phase1_outputs(report, Path(output_dir))
    tracking_uri: str = configured_tracking_uri()
    logged: dict[str, Any] = log_phase1_to_mlflow(
        report,
        paths,
        tracking_uri,
        experiment_name=str(settings["experiment_name"]),
        phase_tag=str(settings["phase_tag"]),
        hypothesis=settings["hypothesis"],
        note=str(settings["note"]),
    )
    logger.info("Wrote %s and %s", paths["markdown_path"], paths["csv_path"])
    run_list: str = ", ".join(item["run_id"] for item in logged["run_ids"])
    logger.info("MLflow runs tagged phase=%s: %s", settings["phase_tag"], run_list)


def run_reference_check(
    games: pd.DataFrame | None = None,
    team_games: pd.DataFrame | None = None,
    fixture_path: str | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Refit logistic ``h2_full`` and compare it to the committed reference.

    The fit is ``run_phase1_baselines`` on the reference window: logistic
    ``h2_full``, the home prior, Elo, and the market. Pass both frames to
    skip the warehouse read.

    Args:
        games: Completed games. None loads them with ``team_games``.
        team_games: Team-game EPA. None loads it with ``games``.
        fixture_path: Fixture directory for ``load_data`` when frames are omitted.
        reference_path: Reference JSON. None uses the committed file.

    Returns:
        The dict from ``compare_reference_metrics``.
    """
    reference: dict[str, Any] = load_reference_metrics(reference_path)
    if games is None or team_games is None:
        games, team_games = load_data(fixture_path=fixture_path)
    config: RollingOriginConfig = make_rolling_origin_config(
        cv_start_season=int(reference["cv_start_season"]),
        cv_end_season=int(reference["cv_end_season"]),
        n_bootstrap=int(reference["n_bootstrap"]),
        bootstrap_seed=int(reference["bootstrap_seed"]),
    )
    report: dict[str, Any] = run_phase1_baselines(
        games,
        team_games,
        config=config,
        logistic_feature_sets=(FEATURE_SET_H2_FULL,),
        reference_model=MODEL_LOGISTIC,
        reference_feature_set=FEATURE_SET_H2_FULL,
        phase=2,
        hypothesis="h3",
    )
    return compare_reference_metrics(report, reference)
