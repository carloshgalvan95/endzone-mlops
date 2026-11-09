"""Score one week with the reference logistic model.

Canonical command: ``python -m endzone.modeling.score``.
Fits ``h2_full`` on completed games strictly before the scored week, logs the
pipeline with a signature, registers ``endzone-winprob``, and writes the week
JSON under ``site/data``. A week file that is already on disk is kept unless
``--replace-published`` rebuilds every scheduled game in that week. A write
that would drop a published ``game_id`` fails. Margin and total estimates are
included only when both pass ADR-007.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from endzone.config import configured_tracking_uri
from endzone.features.pregame import (
    build_pregame_features,
    feature_columns_for,
    market_home_probabilities,
    prepare_training_data,
)
from endzone.modeling.dataset import load_data, load_upcoming_games, validate_feature_columns
from endzone.modeling.models import fit_logistic_pipeline
from endzone.modeling.points import attach_score_estimates
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
    """Fit ``h2_full`` on games before the slate week and score that week.

    The default slate is the next unplayed week in ``fct_upcoming_games``.
    When that week file is already on disk, it is kept and nothing is fit.
    ``settings.replace_published`` scores every scheduled game in the requested
    week, played and unplayed. Same-week finals are not training rows and are
    not features. A Sunday rebuild does not learn from a game that already
    finished that week. Pregame features still use only rows with ``gameday``
    strictly before the game being scored.

    Returns:
        Dict with ``document``, ``paths``, ``model_version``, ``run_id``,
        and ``kept_published``.

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

    columns: list[str] = list(feature_columns_for(WEEKLY_FEATURE_SET))
    # Float columns keep a missing value representable in the model signature.
    train_x[columns] = train_x[columns].astype(float)
    pipeline: Pipeline = fit_logistic_pipeline(
        train_x,
        train_y,
        C=settings.logistic_c,
        max_iter=settings.logistic_max_iter,
        random_state=settings.random_state,
    )
    score_frame: pd.DataFrame = _score_frame(slate, features, columns)
    score_frame[columns] = score_frame[columns].astype(float)
    probabilities: pd.Series = pd.Series(
        pipeline.predict_proba(score_frame[columns])[:, 1],
        index=score_frame["game_id"].astype(str),
    )
    example: pd.DataFrame = train_x[columns].head(1)
    signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
    tracking_uri: str = settings.tracking_uri or configured_tracking_uri()
    # Sprint 2 opens its own run. It has to finish before the win-prob run.
    score_attachment: dict[str, Any] = attach_score_estimates(
        history,
        features,
        slate,
        tracking_uri=tracking_uri,
    )
    activate_tracking_experiment(WEEKLY_EXPERIMENT_NAME, tracking_uri)
    generated_at: str = _generated_at(settings.generated_at)
    with mlflow.start_run(run_name=f"score-{chosen['season']}-w{chosen['week']}"):
        _log_score_params(settings, chosen, train_x, slate, columns)
        mlflow.set_tag("git_commit", get_git_sha())
        mlflow.set_tag("feature_set", WEEKLY_FEATURE_SET)
        mlflow.set_tag("stage", "weekly")
        logged: RegisteredModelLog = _register_weekly_model(
            pipeline,
            signature=signature,
            input_example=example,
            model_name=settings.model_name,
            tracking_uri=tracking_uri,
        )
    document: dict[str, Any] = build_prediction_document(
        season=chosen["season"],
        week=chosen["week"],
        games=slate,
        model_probabilities=probabilities,
        market_probabilities=_market_by_game(slate),
        model_name=settings.model_name,
        model_version=logged.model_version,
        feature_set=WEEKLY_FEATURE_SET,
        generated_at=generated_at,
        score_lines=score_attachment["lines"],
        score_model=score_attachment["score_model"],
    )
    teams: dict[str, Any] = team_catalog_document(team_catalog)
    paths: dict[str, str] = write_prediction_bundle(document, teams, Path(settings.output_dir))
    logger.info(
        "Registered %s version %s for %s week %s (%s games).",
        settings.model_name,
        logged.model_version,
        chosen["season"],
        chosen["week"],
        len(document["games"]),
    )
    return {
        "document": document,
        "paths": paths,
        "model_version": logged.model_version,
        "run_id": logged.run_id,
        "kept_published": False,
    }


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
    train_x: pd.DataFrame,
    slate: pd.DataFrame,
    columns: list[str],
) -> None:
    """Log the fit settings on the active run. No held-out test metric is read."""
    mlflow.log_param("model_type", "logistic")
    mlflow.log_param("feature_set", WEEKLY_FEATURE_SET)
    mlflow.log_param("model_name", settings.model_name)
    mlflow.log_param("C", settings.logistic_c)
    mlflow.log_param("max_iter", settings.logistic_max_iter)
    mlflow.log_param("random_state", settings.random_state)
    mlflow.log_param("lookback_games", settings.lookback_games)
    mlflow.log_param("season", chosen["season"])
    mlflow.log_param("week", chosen["week"])
    mlflow.log_param("n_train", len(train_x))
    mlflow.log_param("n_scored", len(slate))
    mlflow.log_param("train_season_min", int(train_x["season"].min()))
    mlflow.log_param("train_season_max", int(train_x["season"].max()))
    mlflow.log_param("features", ",".join(columns))


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
        replace_published=bool(args.replace_published),
    )
    return score_upcoming_week(games, team_games, upcoming, catalog, settings)


if __name__ == "__main__":
    main()
