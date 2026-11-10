"""Score one week with the reference logistic model.

Canonical command: ``python -m endzone.modeling.score``.
Loads the registered model at alias ``prod`` when that alias can be read.
If the alias is missing or the registry cannot be reached, fits ``h2_full``
on completed games strictly before the scored week and registers that fit
in the tracking store only. The ``prod`` alias is never set here. A week
file that is already on disk is kept unless ``--replace-published`` rebuilds
every scheduled game in that week. A write that would drop a published
``game_id`` fails. Margin and total estimates are included only when both
pass ADR-007.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from endzone.config import configured_registry_uri, configured_tracking_uri
from endzone.features.pregame import (
    build_pregame_features,
    feature_columns_for,
    market_home_probabilities,
    prepare_training_data,
)
from endzone.modeling.dataset import load_data, load_upcoming_games, validate_feature_columns
from endzone.modeling.models import fit_logistic_pipeline
from endzone.modeling.points import attach_score_estimates
from endzone.modeling.registry import (
    PROD_ALIAS,
    ProdAliasError,
    ProdModel,
    default_model_name,
    format_scored_version,
    load_prod_model,
    pinned_registry,
)
from endzone.modeling.tracking import (
    RegisteredModelLog,
    activate_tracking_experiment,
    get_git_sha,
    log_registered_sklearn_model,
    log_sklearn_model,
)
from endzone.modeling.weekly import (
    DEFAULT_OUTPUT_DIR,
    REGISTERED_MODEL_NAME,
    WEEKLY_EXPERIMENT_NAME,
    WEEKLY_FEATURE_SET,
    build_prediction_document,
    games_before_week,
    kept_published_week,
    load_team_catalog,
    parse_optional_int,
    refuse_dropped_game_ids,
    restore_published_predictions,
    select_scheduled_week,
    select_slate,
    slate_for_features,
    team_catalog_document,
    team_games_before_week,
    write_prediction_bundle,
)
from endzone.utils.logging import setup_logger
from endzone.utils.step_status import status_directory, write_step_status

logger = setup_logger(__name__)

# A local sqlite store that cannot register still publishes the week file.
LOCAL_MODEL_VERSION: str = "local"


class WeeklyScoreSettings(NamedTuple):
    """Knobs for one weekly fit. Defaults match the reference logistic model."""

    season: int | None = None
    week: int | None = None
    lookback_games: int = 10
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    random_state: int = 42
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tracking_uri: str | None = None
    registry_uri: str | None = None
    model_name: str = REGISTERED_MODEL_NAME
    generated_at: datetime | None = None
    replace_published: bool = False


def score_upcoming_week(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    upcoming: pd.DataFrame,
    team_catalog: pd.DataFrame,
    settings: WeeklyScoreSettings,
) -> dict[str, Any]:
    """Score one week, loading alias ``prod`` when the registry has it.

    The default slate is the next unplayed week in ``fct_upcoming_games``.
    When that week file is already on disk, it is kept and nothing is fit.
    A readable ``prod`` alias scores the slate. The version on the file is
    ``<version>@prod``. A missing alias or an unreachable registry logs the
    failure and fits ``h2_full`` on games before the week, which is the
    previous behavior. That fit is registered in the tracking store and does
    not receive the alias. ``settings.replace_published`` scores every
    scheduled game in the requested week, played and unplayed. Same-week
    finals are not training rows and are not features. A Sunday rebuild does
    not learn from a game that already finished that week. Pregame features
    still use only rows with ``gameday`` strictly before the game being scored.

    Returns:
        Dict with ``document``, ``paths``, ``model_version``, ``model_alias``,
        ``model_source``, ``run_id``, ``kept_published``, ``fallback_reason``,
        and ``data_digest``. ``fallback_reason`` is None when alias ``prod``
        scored the week. It is a short code, never a host or a token.

    Raises:
        ValueError: The slate is empty, the fit has one class, or the new
            file would drop a published ``game_id``.
    """
    if settings.replace_published and (settings.season is None or settings.week is None):
        raise ValueError(
            "replace-published needs both season and week so the full slate is explicit."
        )
    if settings.replace_published:
        chosen: dict[str, Any] = select_scheduled_week(
            games,
            upcoming,
            int(settings.season),
            int(settings.week),
        )
    else:
        chosen = select_slate(upcoming, settings.season, settings.week)
        kept: dict[str, Any] | None = kept_published_week(
            Path(settings.output_dir),
            int(chosen["season"]),
            int(chosen["week"]),
        )
        if kept is not None:
            kept["model_source"] = "kept_published"
            kept["fallback_reason"] = None
            document_kept: dict[str, Any] = kept["document"]
            kept["model_alias"] = document_kept.get("model_alias")
            kept["data_digest"] = _text_or_none(document_kept.get("data_digest"))
            _emit_score_status(kept)
            return kept
    target_season: int = int(chosen["season"])
    target_week: int = int(chosen["week"])
    incoming_ids: set[str] = {str(game_id) for game_id in chosen["games"]["game_id"].tolist()}
    refuse_dropped_game_ids(Path(settings.output_dir), target_season, target_week, incoming_ids)
    # Same-week finals stay out of the fit. Thursday's result is not a
    # feature for Sunday, and it is not a training label either.
    history: pd.DataFrame = games_before_week(games, target_season, target_week)
    history_ids: set[str] = set(history["game_id"].astype(str))
    prior_team_games: pd.DataFrame = team_games_before_week(
        team_games,
        target_season,
        target_week,
        history_ids,
    )
    slate: pd.DataFrame = slate_for_features(chosen["games"])
    features: pd.DataFrame = build_pregame_features(
        pd.concat([history, slate], ignore_index=True),
        prior_team_games,
        lookback_games=settings.lookback_games,
    )
    history_features: pd.DataFrame = features.loc[
        features["game_id"].astype(str).isin(history_ids)
    ].copy()
    validate_feature_columns(history_features, (WEEKLY_FEATURE_SET,))
    columns: list[str] = list(feature_columns_for(WEEKLY_FEATURE_SET))
    score_frame: pd.DataFrame = _score_frame(slate, features, columns)
    score_frame[columns] = score_frame[columns].astype(float)
    tracking_uri: str = settings.tracking_uri or configured_tracking_uri()
    registry_uri: str = _registry_uri_for_score(settings)
    loaded_attempt: dict[str, Any] = _try_load_prod(registry_uri)
    loaded: ProdModel | None = loaded_attempt["model"]
    fallback_reason: str | None = loaded_attempt["fallback_reason"]
    probabilities: pd.Series | None = None
    if loaded is not None:
        try:
            probabilities = _home_win_probabilities(loaded.model, score_frame, columns)
        except Exception as exc:
            logger.warning(
                "Loaded %s@%s could not score this slate (%s: %s). "
                "Fitting logistic h2_full on games before this week.",
                loaded.model_name,
                loaded.alias,
                type(exc).__name__,
                exc,
            )
            loaded = None
            probabilities = None
            fallback_reason = "loaded_model_could_not_score"
    train_x: pd.DataFrame | None = None
    example: pd.DataFrame | None = None
    signature: Any = None
    fitted_pipeline: Pipeline | None = None
    if loaded is None or probabilities is None:
        fitted: dict[str, Any] = _fit_week_model(
            history,
            features,
            score_frame,
            columns,
            settings,
        )
        fitted_pipeline = fitted["pipeline"]
        train_x = fitted["train_x"]
        probabilities = fitted["probabilities"]
        example = fitted["example"]
        signature = fitted["signature"]
    # Sprint 2 opens its own run. Pin the registry to the tracking store so a
    # Unity Catalog URI used to load prod does not receive a weekly version.
    with pinned_registry(tracking_uri):
        score_attachment: dict[str, Any] = attach_score_estimates(
            history,
            features,
            slate,
            tracking_uri=tracking_uri,
        )
        activate_tracking_experiment(WEEKLY_EXPERIMENT_NAME, tracking_uri)
        generated_at: str = _generated_at(settings.generated_at)
        model_name: str = settings.model_name
        model_alias: str | None = None
        version_label: str = ""
        run_id: str = ""
        source: str = "refit"
        with mlflow.start_run(run_name=f"score-{chosen['season']}-w{chosen['week']}"):
            if fitted_pipeline is not None and example is not None:
                source = "refit"
                _log_score_params(
                    settings,
                    chosen,
                    train_x,
                    slate,
                    columns,
                    model_name=settings.model_name,
                    model_source=source,
                )
                mlflow.set_tag("git_commit", get_git_sha())
                mlflow.set_tag("feature_set", WEEKLY_FEATURE_SET)
                mlflow.set_tag("stage", "weekly")
                mlflow.set_tag("model_source", source)
                mlflow.set_tag("model_alias", "none")
                logged: RegisteredModelLog = _register_weekly_model(
                    fitted_pipeline,
                    signature=signature,
                    input_example=example,
                    model_name=settings.model_name,
                    tracking_uri=tracking_uri,
                )
                version_label = logged.model_version
                run_id = logged.run_id
                mlflow.log_param("model_version", version_label)
            else:
                source = "registry"
                fallback_reason = None
                assert loaded is not None
                model_name = loaded.model_name
                model_alias = loaded.alias
                version_label = format_scored_version(loaded.version, loaded.alias)
                _log_score_params(
                    settings,
                    chosen,
                    None,
                    slate,
                    columns,
                    model_name=model_name,
                    model_source=source,
                )
                mlflow.set_tag("git_commit", get_git_sha())
                mlflow.set_tag("feature_set", WEEKLY_FEATURE_SET)
                mlflow.set_tag("stage", "weekly")
                mlflow.set_tag("model_source", source)
                mlflow.set_tag("model_alias", loaded.alias)
                mlflow.log_param("model_version", version_label)
                mlflow.log_param("registered_model", loaded.model_name)
                active = mlflow.active_run()
                run_id = "" if active is None else str(active.info.run_id)
    document: dict[str, Any] = build_prediction_document(
        season=chosen["season"],
        week=chosen["week"],
        games=slate,
        model_probabilities=probabilities,
        market_probabilities=_market_by_game(slate),
        model_name=model_name,
        model_version=version_label,
        feature_set=WEEKLY_FEATURE_SET,
        generated_at=generated_at,
        score_lines=score_attachment["lines"],
        score_model=score_attachment["score_model"],
        model_alias=model_alias,
    )
    teams: dict[str, Any] = team_catalog_document(team_catalog)
    paths: dict[str, str] = write_prediction_bundle(document, teams, Path(settings.output_dir))
    if model_alias:
        logger.info(
            "Scored %s week %s with %s version %s (%s games).",
            chosen["season"],
            chosen["week"],
            model_name,
            version_label,
            len(document["games"]),
        )
    else:
        logger.info(
            "Prod alias was not used. Registered %s version %s for %s week %s (%s games).",
            model_name,
            version_label,
            chosen["season"],
            chosen["week"],
            len(document["games"]),
        )
    if source == "refit" and fallback_reason is None:
        fallback_reason = "weekly_fit"
    scored: dict[str, Any] = {
        "document": document,
        "paths": paths,
        "model_version": version_label,
        "model_alias": model_alias,
        "model_source": source,
        "run_id": run_id,
        "kept_published": False,
        "fallback_reason": fallback_reason,
        "data_digest": _feature_digest(score_frame),
    }
    _emit_score_status(scored)
    return scored


def _fit_week_model(
    history: pd.DataFrame,
    features: pd.DataFrame,
    score_frame: pd.DataFrame,
    columns: list[str],
    settings: WeeklyScoreSettings,
) -> dict[str, Any]:
    """Fit logistic ``h2_full`` on ``history`` and score ``score_frame``.

    Returns:
        Dict with ``pipeline``, ``train_x``, ``probabilities``, ``example``,
        and ``signature``.

    Raises:
        ValueError: Completed games have only one class.
    """
    train_x, train_y = prepare_training_data(
        history,
        features,
        feature_set=WEEKLY_FEATURE_SET,
    )
    labeled: pd.Series = train_y.notna()
    train_x = train_x.loc[labeled].reset_index(drop=True)
    train_y = train_y.loc[labeled].astype(int).reset_index(drop=True)
    if train_y.nunique() < 2:
        raise ValueError("Completed games need both home wins and home losses to fit h2_full.")
    # Float columns keep a missing value representable in the model signature.
    train_x[columns] = train_x[columns].astype(float)
    pipeline: Pipeline = fit_logistic_pipeline(
        train_x,
        train_y,
        C=settings.logistic_c,
        max_iter=settings.logistic_max_iter,
        random_state=settings.random_state,
    )
    example: pd.DataFrame = train_x[columns].head(1)
    signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
    return {
        "pipeline": pipeline,
        "train_x": train_x,
        "probabilities": _home_win_probabilities(pipeline, score_frame, columns),
        "example": example,
        "signature": signature,
    }


def _home_win_probabilities(
    model: object,
    score_frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """Home-win probability for each slate row, using class 1 when it is named."""
    predict = getattr(model, "predict_proba", None)
    if not callable(predict):
        raise TypeError("The scoring model has no predict_proba.")
    raw = predict(score_frame[columns])
    column: int = 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        labels: list[int] = [int(label) for label in list(classes)]
        if 1 not in labels:
            raise ValueError("The scoring model has no home-win class.")
        column = labels.index(1)
    home: list[float] = [float(row[column]) for row in raw]
    return pd.Series(home, index=score_frame["game_id"].astype(str))


def _registry_uri_for_score(settings: WeeklyScoreSettings) -> str:
    """Registry used to load ``prod``.

    An explicit registry URI wins. A caller that passed a tracking URI, which
    tests do, uses that same store. Otherwise the configured registry URI is
    used, so Pages can point ``MLFLOW_REGISTRY_URI`` at Unity Catalog while
    tracking stays on sqlite.
    """
    if settings.registry_uri is not None and settings.registry_uri.strip() != "":
        return settings.registry_uri.strip()
    if settings.tracking_uri is not None and settings.tracking_uri.strip() != "":
        return settings.tracking_uri.strip()
    return configured_registry_uri()


def _try_load_prod(registry_uri: str) -> dict[str, Any]:
    """Load alias ``prod``, or a fallback code that names no host.

    Any failure is a fallback, including a missing alias and a registry that
    cannot be reached. The week file is still written. The returned reason
    is a short code. The exception text stays in the log and is not stored,
    because it can carry a registry address.

    Returns:
        ``model`` (a ``ProdModel`` or None) and ``fallback_reason``
        (None, ``alias_missing``, or ``registry_unreadable``).
    """
    model_name: str = default_model_name(registry_uri)
    try:
        loaded: ProdModel = load_prod_model(model_name, registry_uri)
    except ProdAliasError as exc:
        logger.warning(
            "Alias %s is missing on %s (%s). Fitting logistic h2_full on games before this week.",
            PROD_ALIAS,
            model_name,
            exc,
        )
        return {"model": None, "fallback_reason": "alias_missing"}
    except Exception as exc:
        logger.warning(
            "Could not load %s@%s from registry %s (%s: %s). "
            "Fitting logistic h2_full on games before this week.",
            model_name,
            PROD_ALIAS,
            registry_uri,
            type(exc).__name__,
            exc,
        )
        return {"model": None, "fallback_reason": "registry_unreadable"}
    logger.info(
        "Loaded %s version %s alias %s.",
        loaded.model_name,
        loaded.version,
        loaded.alias,
    )
    return {"model": loaded, "fallback_reason": None}


def _score_frame(
    slate: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return feature rows for the slate, in slate order."""
    ordered_ids: list[str] = [str(game_id) for game_id in slate["game_id"].tolist()]
    feature_rows: pd.DataFrame = features.copy()
    feature_rows["game_id"] = feature_rows["game_id"].astype(str)
    found: set[str] = set(feature_rows["game_id"])
    missing: list[str] = [game_id for game_id in ordered_ids if game_id not in found]
    if missing:
        missing_list: str = ", ".join(missing)
        raise ValueError(f"Feature build did not return every upcoming game: {missing_list}.")
    merged: pd.DataFrame = pd.DataFrame({"game_id": ordered_ids}).merge(
        feature_rows,
        on="game_id",
        how="left",
    )
    return merged.loc[:, ["game_id", *columns]].reset_index(drop=True)


def _market_by_game(slate: pd.DataFrame) -> dict[str, float | None]:
    """No-vig home probability for each slate game, or None when the line is absent.

    A null or non-finite price on one game stays null. It does not blank a
    sibling game that has both moneylines.
    """
    market: dict[str, Any] = market_home_probabilities(slate)
    logger.info(
        "Market lines on the slate: status=%s scored=%s missing=%s",
        market["status"],
        market["n_scored"],
        market["n_missing_lines"],
    )
    probabilities: pd.Series | None = market["probabilities"]
    mapped: dict[str, float | None] = {}
    game_ids: list[str] = [str(game_id) for game_id in slate["game_id"].tolist()]
    if probabilities is None:
        return {game_id: None for game_id in game_ids}
    # Position order matches game_id order. The series index is not the id.
    for game_id, value in zip(game_ids, probabilities.tolist(), strict=True):
        mapped[game_id] = _finite_probability(value)
    return mapped


def _finite_probability(value: object) -> float | None:
    """Return a finite probability. Missing and non-finite cells are None."""
    if isinstance(value, list | tuple | dict):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        number: float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _local_registry(tracking_uri: str) -> bool:
    """True for a sqlite or file store. A Databricks URI still has to register."""
    text: str = tracking_uri.strip()
    return text.startswith("sqlite:") or text.startswith("file:")


def _register_weekly_model(
    pipeline: Pipeline,
    *,
    signature: Any,
    input_example: pd.DataFrame,
    model_name: str,
    tracking_uri: str,
) -> RegisteredModelLog:
    """Register the fitted pipeline. A local registry failure still returns a version.

    Databricks tracking re-raises. A sqlite or file store that cannot register
    logs the artifact when it can and uses ``LOCAL_MODEL_VERSION`` so the
    week file is still written.
    """
    try:
        with pinned_registry(tracking_uri):
            return log_registered_sklearn_model(
                pipeline,
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name,
            )
    except (RuntimeError, mlflow.MlflowException) as exc:
        if not _local_registry(tracking_uri):
            raise
        logger.warning(
            "Local MLflow did not register %s (%s). The week file uses version %s.",
            model_name,
            exc,
            LOCAL_MODEL_VERSION,
        )
        try:
            log_sklearn_model(pipeline)
        except mlflow.MlflowException as log_exc:
            logger.warning("Local MLflow did not store the model artifact: %s", log_exc)
        active = mlflow.active_run()
        run_id: str = "" if active is None else str(active.info.run_id)
        return RegisteredModelLog(
            trusted_types=[],
            model_version=LOCAL_MODEL_VERSION,
            model_uri="",
            run_id=run_id,
        )


def _generated_at(clock: datetime | None) -> str:
    """UTC timestamp with a ``Z`` suffix."""
    moment: datetime = clock if clock is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_score_params(
    settings: WeeklyScoreSettings,
    chosen: dict[str, Any],
    train_x: pd.DataFrame | None,
    slate: pd.DataFrame,
    columns: list[str],
    *,
    model_name: str,
    model_source: str,
) -> None:
    """Log the fit settings on the active run. No held-out test metric is read.

    A registry load has no training frame. Those params say ``loaded``.
    """
    mlflow.log_param("model_type", "logistic")
    mlflow.log_param("feature_set", WEEKLY_FEATURE_SET)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("model_source", model_source)
    mlflow.log_param("C", settings.logistic_c)
    mlflow.log_param("max_iter", settings.logistic_max_iter)
    mlflow.log_param("random_state", settings.random_state)
    mlflow.log_param("lookback_games", settings.lookback_games)
    mlflow.log_param("season", chosen["season"])
    mlflow.log_param("week", chosen["week"])
    mlflow.log_param("n_scored", len(slate))
    mlflow.log_param("features", ",".join(columns))
    if train_x is None:
        mlflow.log_param("n_train", "loaded")
        mlflow.log_param("train_season_min", "loaded")
        mlflow.log_param("train_season_max", "loaded")
        return
    mlflow.log_param("n_train", len(train_x))
    mlflow.log_param("train_season_min", int(train_x["season"].min()))
    mlflow.log_param("train_season_max", int(train_x["season"].max()))


def build_parser() -> argparse.ArgumentParser:
    """Build the weekly scoring parser."""
    parser = argparse.ArgumentParser(
        description="Score the next unplayed week with logistic h2_full."
    )
    parser.add_argument(
        "--season",
        type=str,
        default="auto",
        help="Season to score. auto uses the earliest season that still has an unplayed game.",
    )
    parser.add_argument(
        "--week",
        type=str,
        default="auto",
        help="Week to score. auto uses the earliest unplayed week in that season.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for latest.json, teams.json, and predictions/<season>-w<week>.json.",
    )
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixture-path",
        type=str,
        default=None,
        help="Parquet fixture directory. Omit it to read the warehouse.",
    )
    parser.add_argument(
        "--teams-path",
        type=str,
        default=None,
        help="Local teams_colors_logos CSV or teams.json. Omit it to download nflverse.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. The default is the configured sqlite or Databricks URI.",
    )
    parser.add_argument(
        "--registry-uri",
        type=str,
        default=None,
        help=(
            "MLflow registry URI used to load alias prod. "
            "The default is MLFLOW_REGISTRY_URI, or the tracking URI when that is unset. "
            "A missing alias falls back to a fit on this week's games."
        ),
    )
    parser.add_argument(
        "--restore-published",
        type=str,
        default=None,
        help=(
            "Base URL of the live site. Copy non-example week files into the output "
            "directory before scoring. A failed copy does not stop the score."
        ),
    )
    parser.add_argument(
        "--replace-published",
        action="store_true",
        help=(
            "Rebuild every scheduled game for --season and --week, including games "
            "already played. Fits only on completed games before that week. "
            "Refuses to drop a game_id that is already in the week file."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Score one week and return the written document summary."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    try:
        return _score_from_args(args)
    except Exception:
        try:
            _emit_score_status({"model_source": "failed", "fallback_reason": "score_failed"})
        except OSError as emit_error:
            logger.warning("Score status was not written: %s", type(emit_error).__name__)
        raise


def _score_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Load inputs and score one week. The caller records a failed status."""
    if args.replace_published and (
        parse_optional_int(args.season) is None or parse_optional_int(args.week) is None
    ):
        raise ValueError(
            "replace-published needs both --season and --week so the full slate is explicit."
        )
    if args.restore_published:
        restore_published_predictions(Path(args.output_dir), args.restore_published)
    fixture_path: str | None = args.fixture_path if args.fixture_path else None
    games, team_games = load_data(fixture_path)
    upcoming: pd.DataFrame = load_upcoming_games(fixture_path)
    teams_path: Path | None = Path(args.teams_path) if args.teams_path else None
    catalog: pd.DataFrame = load_team_catalog(teams_path)
    settings = WeeklyScoreSettings(
        season=parse_optional_int(args.season),
        week=parse_optional_int(args.week),
        lookback_games=args.lookback,
        logistic_c=args.C,
        logistic_max_iter=args.max_iter,
        random_state=args.seed,
        output_dir=Path(args.output_dir),
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
        replace_published=bool(args.replace_published),
    )
    return score_upcoming_week(games, team_games, upcoming, catalog, settings)


def _emit_score_status(result: Mapping[str, Any]) -> None:
    """Write the score step file when the workflow asked for one.

    The file carries the version, the alias, the fallback code, and the
    feature digest. It does not carry a registry address.
    """
    directory: Path | None = status_directory()
    if directory is None:
        return
    try:
        _write_score_status(directory, result)
    except OSError as exc:
        logger.warning("Score status was not written: %s", type(exc).__name__)


def _write_score_status(directory: Path, result: Mapping[str, Any]) -> None:
    """Write score.txt. The caller handles a file error."""
    document: object = result.get("document")
    season: object = ""
    week: object = ""
    games: object = ""
    if isinstance(document, dict):
        season = document.get("season", "")
        week = document.get("week", "")
        game_rows: object = document.get("games")
        if isinstance(game_rows, list):
            games = len(game_rows)
    write_step_status(
        directory,
        "score",
        {
            "season": season,
            "week": week,
            "games": games,
            "model_version": result.get("model_version"),
            "model_alias": result.get("model_alias"),
            "model_source": result.get("model_source"),
            "fallback_reason": result.get("fallback_reason"),
            "data_digest": result.get("data_digest"),
            "kept_published": bool(result.get("kept_published")),
        },
    )


def _feature_digest(frame: pd.DataFrame) -> str:
    """Sha256 of the scored feature rows, in game id order."""
    ordered: pd.DataFrame = frame.sort_values("game_id").reset_index(drop=True)
    columns: list[str] = [str(column) for column in ordered.columns]
    lines: list[str] = ["|".join(columns)]
    for row in ordered.itertuples(index=False, name=None):
        cells: list[str] = ["" if value is None else str(value) for value in row]
        lines.append("|".join(cells))
    payload: str = "\n".join(lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_or_none(value: object) -> str | None:
    """Return stripped text, or None when the cell is not text."""
    if not isinstance(value, str):
        return None
    text: str = value.strip()
    if text == "":
        return None
    return text


if __name__ == "__main__":
    main()
