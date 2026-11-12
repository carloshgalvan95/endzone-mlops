"""Score one rolling-origin row: home prior, logistic, boosting, Elo, or the market."""

from __future__ import annotations

from typing import Any, NamedTuple

import pandas as pd

from endzone.features.pregame import (
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_MEAN_RATING,
    ELO_SCALE,
    ELO_SEASON_REVERSION,
    FEATURE_SET_COLUMNS,
    QB_PRIOR_DROPBACKS,
    REPLACEMENT_EPA_PER_DROPBACK,
    SEASON_BLEND_K,
    SEASON_SHRINK_FACTOR,
    feature_columns_for,
    market_home_probabilities,
    prepare_training_data,
)
from endzone.features.qbelo import (
    QBELO_HOME_ADVANTAGE,
    QBELO_SCALE,
    rating_win_probability,
)
from endzone.modeling.cv import (
    RollingOriginConfig,
    _baseline_fold_predictor,
    _elo_fold_predictor,
    _logistic_fold_predictor,
    _logistic_l2_fold_predictor,
    _lookup_fold_predictor,
    cv_result_record,
    rolling_origin_cv,
)
from endzone.modeling.ensembles import (
    DERIVED_CV_MODELS,
    MODEL_GBM_PKG_CORE_INJ,
    hist_gbm_fold_predictor,
    hist_gradient_boosting_param_fields,
)
from endzone.modeling.hypotheses import CvModelSpec
from endzone.modeling.models import (
    DEFAULT_L2_C_GRID,
    MODEL_BASELINE_HOME,
    MODEL_LOGISTIC,
    MODEL_LOGISTIC_L2,
    _format_regularization_c,
    chosen_c_param_fields,
)


class CvScoreSettings(NamedTuple):
    """Shared knobs for one rolling-origin comparison. The public call stays the same."""

    resolved: RollingOriginConfig
    lookback_games: int
    logistic_c: float
    logistic_max_iter: int
    random_state: int
    reference_model: str
    reference_feature_set: str
    log_shrinkage: bool
    log_qb: bool


def _shared_params(settings: CvScoreSettings) -> dict[str, Any]:
    """Params logged on every model in one comparison."""
    params: dict[str, Any] = {
        "lookback_games": settings.lookback_games,
        "cv_start_season": settings.resolved.cv_start_season,
        "cv_end_season": settings.resolved.cv_end_season,
        "n_bootstrap": settings.resolved.n_bootstrap,
        "bootstrap_seed": settings.resolved.bootstrap_seed,
        "random_state": settings.random_state,
        "reference_model": settings.reference_model,
        "reference_feature_set": settings.reference_feature_set,
    }
    if settings.log_shrinkage:
        params["shrink_factor"] = SEASON_SHRINK_FACTOR
        params["blend_k"] = SEASON_BLEND_K
    if settings.log_qb:
        params["qb_prior_dropbacks"] = QB_PRIOR_DROPBACKS
        params["replacement_epa_per_dropback"] = REPLACEMENT_EPA_PER_DROPBACK
    return params


def _requested_feature_sets(
    model_specs: tuple[CvModelSpec, ...],
    reference_feature_set: str,
) -> tuple[str, ...]:
    """Feature sets whose columns must exist before any fold is fit."""
    requested: list[str] = []
    for spec in model_specs:
        if spec.feature_set in FEATURE_SET_COLUMNS and spec.feature_set not in requested:
            requested.append(spec.feature_set)
    if reference_feature_set in FEATURE_SET_COLUMNS and reference_feature_set not in requested:
        requested.append(reference_feature_set)
    return tuple(requested)


def _score_baseline(
    matrix: pd.DataFrame,
    target: pd.Series,
    settings: CvScoreSettings,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Score the constant home prior on the current matrix."""
    baseline_cv: dict[str, Any] = rolling_origin_cv(
        matrix,
        target,
        _baseline_fold_predictor(),
        settings.resolved,
    )
    reason: str | None = None if not baseline_cv["oof"].empty else "no_scored_folds"
    return cv_result_record(
        MODEL_BASELINE_HOME,
        "none",
        "ok" if reason is None else "skipped",
        reason,
        {**shared, "model_type": MODEL_BASELINE_HOME, "features": "none"},
        baseline_cv,
    )


def logistic_c_for_spec(spec: CvModelSpec, fallback_c: float) -> float:
    """Return the C for one logistic row.

    A spec with ``fixed_c`` uses that value. Every other row uses
    ``fallback_c``, which is the command's ``--C``.

    Args:
        spec: Model row from the experiment file.
        fallback_c: C from the command. Used when the row does not lock C.

    Returns:
        The C passed to logistic regression for this row.
    """
    if spec.fixed_c is None:
        return float(fallback_c)
    return float(spec.fixed_c)


def _score_model_spec(
    games: pd.DataFrame,
    features: pd.DataFrame,
    spec: CvModelSpec,
    settings: CvScoreSettings,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Score one logistic, fold-tuned L2, or locked booster row.

    A mean or a stacker is not scored here. Those rows use probabilities
    that this function has already produced.
    """
    if spec.model in DERIVED_CV_MODELS:
        raise ValueError(f"{spec.model} is not a fitted model")
    matrix: pd.DataFrame
    target: pd.Series
    matrix, target = prepare_training_data(games, features, feature_set=spec.feature_set)
    columns: str = ",".join(feature_columns_for(spec.feature_set))
    if spec.model == MODEL_GBM_PKG_CORE_INJ:
        predictor = hist_gbm_fold_predictor()
        model_params: dict[str, Any] = {
            **shared,
            "model_type": spec.model,
            "feature_set": spec.feature_set,
            "features": columns,
            **hist_gradient_boosting_param_fields(),
        }
    elif spec.model == MODEL_LOGISTIC_L2:
        predictor = _logistic_l2_fold_predictor(
            DEFAULT_L2_C_GRID,
            settings.logistic_c,
            settings.logistic_max_iter,
            settings.random_state,
        )
        model_params = {
            **shared,
            "model_type": spec.model,
            "feature_set": spec.feature_set,
            "features": columns,
            "max_iter": settings.logistic_max_iter,
            "regularization": "l2",
        }
    elif spec.model == MODEL_LOGISTIC:
        predictor = _logistic_fold_predictor(
            logistic_c_for_spec(spec, settings.logistic_c),
            settings.logistic_max_iter,
            settings.random_state,
        )
        model_params = {
            **shared,
            "model_type": spec.model,
            "feature_set": spec.feature_set,
            "features": columns,
            "max_iter": settings.logistic_max_iter,
            "regularization": "l2",
            "C": logistic_c_for_spec(spec, settings.logistic_c),
        }
    else:
        raise ValueError(f"Unknown CV model {spec.model!r}.")
    scored: dict[str, Any] = rolling_origin_cv(matrix, target, predictor, settings.resolved)
    if spec.model == MODEL_LOGISTIC_L2:
        model_params["c_grid"] = ",".join(
            _format_regularization_c(value) for value in DEFAULT_L2_C_GRID
        )
        model_params["c_fallback"] = settings.logistic_c
        model_params.update(chosen_c_param_fields(scored["folds"]))
    reason: str | None = None if not scored["oof"].empty else "no_scored_folds"
    return cv_result_record(
        spec.model,
        spec.feature_set,
        "ok" if reason is None else "skipped",
        reason,
        model_params,
        scored,
    )


def _score_elo(
    games: pd.DataFrame,
    matrix: pd.DataFrame,
    target: pd.Series,
    settings: CvScoreSettings,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Score Elo on the same rows as the home prior."""
    elo_cv: dict[str, Any] = rolling_origin_cv(
        matrix,
        target,
        _elo_fold_predictor(games),
        settings.resolved,
    )
    reason: str | None = None if not elo_cv["oof"].empty else "no_scored_folds"
    return cv_result_record(
        "elo",
        "elo",
        "ok" if reason is None else "skipped",
        reason,
        {
            **shared,
            "model_type": "elo",
            "feature_set": "elo",
            "features": "elo_rating",
            "elo_k": ELO_K,
            "elo_home_advantage": ELO_HOME_ADVANTAGE,
            "elo_mean_rating": ELO_MEAN_RATING,
            "elo_season_reversion": ELO_SEASON_REVERSION,
            "elo_scale": ELO_SCALE,
        },
        elo_cv,
    )


def _score_rating_gap(
    features: pd.DataFrame,
    matrix: pd.DataFrame,
    target: pd.Series,
    settings: CvScoreSettings,
    shared: dict[str, Any],
    column: str,
    model_name: str,
) -> dict[str, Any]:
    """Score a stored Elo gap plus home advantage 65 on the 400 scale.

    The gap is already leakage-safe. The fold predictor only looks up the
    precomputed probability for the evaluation game ids.
    """
    indexed: pd.Series = features.drop_duplicates("game_id").set_index("game_id")[column]
    gaps = indexed.loc[matrix["game_id"].to_numpy()].to_numpy(dtype=float)
    probabilities: pd.Series = pd.Series(
        rating_win_probability(gaps),
        index=matrix["game_id"].to_numpy(),
    )
    scored: dict[str, Any] = rolling_origin_cv(
        matrix,
        target,
        _lookup_fold_predictor(probabilities),
        settings.resolved,
    )
    reason: str | None = None if not scored["oof"].empty else "no_scored_folds"
    return cv_result_record(
        model_name,
        model_name,
        "ok" if reason is None else "skipped",
        reason,
        {
            **shared,
            "model_type": model_name,
            "feature_set": model_name,
            "features": column,
            "elo_home_advantage": QBELO_HOME_ADVANTAGE,
            "elo_scale": QBELO_SCALE,
        },
        scored,
    )


def _score_market(
    games: pd.DataFrame,
    settings: CvScoreSettings,
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Score the no-vig closing line, or record why it was skipped."""
    market: dict[str, Any] = market_home_probabilities(games)
    market_params: dict[str, Any] = {
        **shared,
        "model_type": "market",
        "feature_set": "market",
        "features": "no_vig_closing_moneyline",
    }
    if market["status"] != "ok":
        return cv_result_record(
            "market",
            "market",
            "skipped",
            str(market["reason"]),
            market_params,
            None,
        )
    market_probability: pd.Series = market["probabilities"]
    kept: pd.Series = market_probability.notna()
    market_frame: pd.DataFrame = pd.DataFrame(
        {
            "game_id": games.loc[kept, "game_id"].to_numpy(),
            "gameday": games.loc[kept, "gameday"].to_numpy(),
            "season": games.loc[kept, "season"].to_numpy(),
        }
    )
    market_target: pd.Series = pd.Series(games.loc[kept, "home_won"].to_numpy())
    by_game: pd.Series = pd.Series(
        market_probability.loc[kept].to_numpy(dtype=float),
        index=market_frame["game_id"].to_numpy(),
    )
    market_cv: dict[str, Any] = rolling_origin_cv(
        market_frame,
        market_target,
        _lookup_fold_predictor(by_game),
        settings.resolved,
    )
    reason: str | None = None if not market_cv["oof"].empty else "no_scored_folds"
    return cv_result_record(
        "market",
        "market",
        "ok" if reason is None else "skipped",
        reason,
        {**market_params, "n_missing_lines": market["n_missing_lines"]},
        market_cv,
    )
